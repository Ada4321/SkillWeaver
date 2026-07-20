#!/bin/bash
#SBATCH --job-name=mcts_simpler_carrot_on_plate_vlm
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --exclude=babel-u5-32,babel-v5-16,babel-v5-20,babel-v5-24,babel-v5-28,babel-v5-32,babel-y9-08,babel-y9-12,babel-y9-16,babel-z5-28,babel-z5-20,babel-m5-32,babel-n9-32,babel-p9-28,babel-n5-20
set -euxo pipefail
#SBATCH --output=slurm/output/output_%j.log
#SBATCH --error=slurm/error/error_%j.log

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

# skills/gemini.py reads GEMINI_API_KEY (or GOOGLE_API_KEY) before its hardcoded fallback.
export GEMINI_API_KEY="your_gemini_api_key"

USER=hez2

cd /home/${USER}/code/Grounded_MCTS_IsaacLab
source ~/miniconda3/etc/profile.d/conda.sh

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load cuda-12.9
conda activate isaaclab

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

resolve_python_bin() {
  local py_bin
  hash -r
  py_bin="$(command -v python || true)"
  if [[ -z "${py_bin}" || ! -x "${py_bin}" ]]; then
    echo "Python executable is unavailable; re-activating conda env '${CONDA_DEFAULT_ENV:-isaaclab}'..." >&2
    conda activate isaaclab
    hash -r
    py_bin="$(command -v python || true)"
  fi
  if [[ -z "${py_bin}" || ! -x "${py_bin}" ]]; then
    echo "Failed to resolve an executable python after conda activation." >&2
    return 1
  fi
  printf '%s\n' "${py_bin}"
}

# =========================
# Task selection (VLM planning version)
# =========================
# simpler / BridgeData WidowX table task: put the carrot on the plate.
# VLM-planned data collection: the VLM actor plans pick -> place itself and MCTS
# searches over its proposals. The scene json 'tasks' field points at tasks.json, whose
# single task type 'place' has ONE task, so
# task_ids=[0].
TASK_NAME="carrot_on_plate"
TASK_KEY="${TASK_KEY:-tasks}"
TASK_ID=0
IGNORE_COLLISION="[]"

# WidowX has a SINGLE general pick policy (IsaacLabEnvs WidowX-Pick-v0); the
# skills=simpler_widowx preset empties the upright/convex/handle dispatch lists
# so every pick routes to this ckpt with the full object point cloud.
PICK_CKPT="${PICK_CKPT:-/home/suli/last_gripper_ep_9800_rew_2142.01.pth}"
if [[ ! -f "${PICK_CKPT}" ]]; then
  echo "Missing WidowX pick ckpt: ${PICK_CKPT}" >&2
  exit 1
fi

# =========================
# Batch scene selection
# =========================
SCENE_ROOT="${SCENE_ROOT:-/data/group_data/katefgroup-ssd/sim_scene_gen/scenes/simpler/${TASK_NAME}}"
# Supported formats: "1", "1,3,7", "10-15", "1,3,10-12"
SCENE_IDS="${SCENE_IDS:-1}"

# Positional override for scene ids (optional):
#   bash data_collect_carrot_on_plate_vlm.sh 1,3,10-12
if [[ "$#" -ge 1 ]] && [[ "$1" != *=* ]]; then
  SCENE_IDS="$1"
  shift
fi

build_scene_list() {
  local ids_spec="$1"
  local -a out=()
  local -A seen=()

  IFS=',' read -ra chunks <<< "$ids_spec"
  for raw in "${chunks[@]}"; do
    local chunk="${raw//[[:space:]]/}"
    [[ -z "$chunk" ]] && continue

    if [[ "$chunk" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local a="${BASH_REMATCH[1]}"
      local b="${BASH_REMATCH[2]}"
      if (( a > b )); then
        echo "Invalid range in SCENE_IDS: $chunk" >&2
        return 1
      fi
      for ((i=a; i<=b; i++)); do
        local name
        printf -v name "scene_%04d" "$i"
        if [[ -z "${seen[$name]:-}" ]]; then
          out+=("$name")
          seen[$name]=1
        fi
      done
    elif [[ "$chunk" =~ ^[0-9]+$ ]]; then
      local name
      printf -v name "scene_%04d" "$chunk"
      if [[ -z "${seen[$name]:-}" ]]; then
        out+=("$name")
        seen[$name]=1
      fi
    else
      echo "Invalid SCENE_IDS token: '$chunk'" >&2
      return 1
    fi
  done

  if [[ "${#out[@]}" -eq 0 ]]; then
    echo "No scene ids resolved from SCENE_IDS='$ids_spec'" >&2
    return 1
  fi

  printf '%s\n' "${out[@]}"
}

mapfile -t SCENE_NAMES < <(build_scene_list "$SCENE_IDS")

declare -a SCENE_FILES=()
for sname in "${SCENE_NAMES[@]}"; do
  sfile="${SCENE_ROOT}/${sname}/${sname}.json"
  if [[ ! -f "$sfile" ]]; then
    echo "Missing scene file: $sfile" >&2
    exit 1
  fi
  SCENE_FILES+=("$sfile")
done

echo "TASK_NAME=${TASK_NAME} TASK_KEY=${TASK_KEY} (VLM planning)"
echo "Resolved scenes (${#SCENE_FILES[@]}):"
printf '  - %s\n' "${SCENE_FILES[@]}"

# =========================
# simpler VLM-planning settings
# =========================
SAVE_ROOT="${SAVE_ROOT:-/data/group_data/katefgroup/datasets/vla_data_sim/simpler/${TASK_NAME}_vlm}"
VIEW_NAMES="['front']"
ACTOR_MODEL="gemini-2.5-pro"

RL_PORT=8888
# 2-step plan (pick -> place). Depth is a search budget
# rather than a required plan length.
DEPTH=2
N_SIMULATIONS=6
C_PUCT=0.5
EVAL_NUM=1
PROCESS_NUM=1
# place_with_drop vertical clearance (simulator.isaaclab.place_offset; conf
# default 0.2, libero scripts 0.02). Bridge-scale objects drop from lower.
PLACE_OFFSET=0.01

# NOTE (env overrides): unlike the libero scripts, do NOT pass dome_light /
# floor_z / wall overrides here — conf/env/simpler_table.yaml already carries
# the values calibrated against the SimplerEnv bridge setup.
for scene_file in "${SCENE_FILES[@]}"; do
  scene_name="$(basename "$(dirname "$scene_file")")"
  scene_out_root="${SAVE_ROOT}/${scene_name}"

  echo "Running scene: ${scene_name} (${scene_file})"
  PYTHON_BIN="$(resolve_python_bin)"

  "${PYTHON_BIN}" -u main.py \
      run.save_rollouts_dir="${scene_out_root}" \
      run.runner.mode=isaaclab_eval \
      run.runner.num_processes="${PROCESS_NUM}" \
      run.use_timestamp_seed=true \
      run.save_node_videos=false \
      run.eval.max_task_per_scene=1 \
      run.eval.task_subsample_strategy=balanced_scene_mod \
      run.eval.success_exp_to_advance=1 \
      run.eval.max_exp_per_task=2 \
      run.eval.only_search_unsuccessful_tasks=true \
      skills=simpler_widowx \
      skills.vlm.model="${ACTOR_MODEL}" \
      skills.server.rl_policy.port=${RL_PORT} \
      skills.rl.backend="local" \
      skills.rl.pick_ckpt="${PICK_CKPT}" \
      skills.curobo.align_empty_gripper_wrist_to_world_z=false \
      skills.curobo.planner_execute_mode1_replan=false \
      skills.curobo.planner_replan_every_k_steps=30 \
      skills.curobo.planner_max_replans=10 \
      skills.curobo.planner_skip_points_after_first_plan=12 \
      skills.curobo.planner_goal_pos_tolerance_m=0.03 \
      skills.curobo.planner_goal_rot_tolerance_rad=0.15 \
      skills.curobo.ignore_collision_assets="${IGNORE_COLLISION}" \
      mcts.tasks.task_ids="[${TASK_ID}]" \
      mcts.max_depth=${DEPTH} \
      mcts.rollout_max_steps=0 \
      mcts.rollout_mode=judge_rollout \
      mcts.n_rollouts_per_node=1 \
      mcts.n_simulations=${N_SIMULATIONS} \
      mcts.c_puct=${C_PUCT} \
      mcts.num_children_per_expand=2 \
      mcts.max_num_try_per_expansion=1 \
      mcts.max_dead_end_count=2 \
      mcts.backprop_dead_end=true \
      mcts.use_history=true \
      mcts.prompts.actor_system.use_3d_range=true \
      mcts.prompts.actor_place.use_3d_range=true \
      mcts.prompts.actor_place_with_drop.use_3d_range=true \
      mcts.prompts.actor_place_with_orientation.use_3d_range=true \
      mcts.prompts.judge.use_3d_range=true \
      mcts.data_collection_mode=true \
      mcts.save_restore_augmented_traj=true \
      mcts.save_original_also_when_augmented=false \
      mcts.mode="stop_if_any_success" \
      mcts.target_success_count=2 \
      mcts.avoid_terminal_in_selection=true \
      mcts.tasks.task_key="${TASK_KEY}" \
      mcts.save_tree_observations=true \
      mcts.save_tree_pcd=false \
      simulator=simpler \
      simulator.view_names=${VIEW_NAMES} \
      simulator.isaaclab.task="GroundedBase-v0" \
      simulator.isaaclab.samples_per_pixel_per_frame=2 \
      simulator.isaaclab.eval_num=${EVAL_NUM} \
      simulator.isaaclab.enable_pick=true \
      simulator.isaaclab.enable_place=false \
      simulator.isaaclab.place_offset=${PLACE_OFFSET} \
      "simulator.isaaclab.pregrasp_noise_std_xyz=[0.022,0.022,0.01]" \
      simulator.isaaclab.pregrasp_fallback_enable=true \
      "simulator.isaaclab.place_noise_std_xyz=[0.01,0.01,0.002]" \
      simulator.isaaclab.collect_depth=true \
      simulator.isaaclab.data_collect_depth_storage=uint16 \
      simulator.isaaclab.data_collect_depth_scale=1000.0 \
      simulator.isaaclab.collect_segmentation=false \
      simulator.isaaclab.collect_object_poses=true \
      simulator.app.headless=true \
      simulator.app.enable_cameras=true \
      simulator.app.rendering_mode="quality" \
      simulator.app.kit_args="--no-window" \
      env="simpler_table" \
      env.scene_desc_file="${scene_file}" \
      env.front_camera_count=4 \
      env.layout_sample_weights=[1,1,1,1] \
      "$@"
done
