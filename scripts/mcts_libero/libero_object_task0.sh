#!/bin/bash
#SBATCH --job-name=mcts_libero_object_batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --exclude=babel-u5-32,babel-v5-16,babel-v5-20,babel-v5-24,babel-v5-28,babel-v5-32,babel-y9-08,babel-y9-12,babel-y9-16,babel-z5-28,babel-z5-20,babel-m5-32,babel-n9-32,babel-p9-28,babel-n5-20
set -euxo pipefail
unset PYTHONPATH
export LD_LIBRARY_PATH=
#SBATCH --output=slurm/output/output_%j.log
#SBATCH --error=slurm/error/error_%j.log

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

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
# Batch scene selection
# =========================
SCENE_ROOT="${SCENE_ROOT:-/data/group_data/katefgroup-ssd/sim_scene_gen/scenes/libero/libero_object/task_0}"
# Supported formats: "1", "1,3,7", "10-15", "1,3,10-12"
SCENE_IDS="${SCENE_IDS:-1}"

# Positional override for scene ids (optional):
#   bash libero_10_task1_batch.sh 1,3,10-12
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

echo "Resolved scenes (${#SCENE_FILES[@]}):"
printf '  - %s\n' "${SCENE_FILES[@]}"

# =========================
# Original libero_10 settings
# =========================
SAVE_ROOT="${SAVE_ROOT:-output/libero_object/libero_object_task0_new}"
VIEW_NAMES="['front']"
ACTOR_MODEL="gemini-2.5-pro"
TASK_KEY="${TASK_KEY:-tasks}"

RL_PORT=8888
DEPTH=2
N_SIMULATIONS=3
C_PUCT=0.5
EVAL_NUM=1
PROCESS_NUM=1

PICK_CKPT="/home/hez2/code/IsaacLabEnvs/logs/rl_games/gripper/2026-02-19_06-40-46_pick_sam3d_general_8gpu_tablez0.25/nn/last_gripper_ep_9600_rew_1706.0063.pth"
PICK_UPRIGHT_CKPT="/home/hez2/code/IsaacLabEnvs/logs/rl_games/gripper/2026-02-19_19-30-54_pick_sam3d_upright_gt10_4gpu/nn/last_gripper_ep_8400_rew_1541.1537.pth"

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
      run.eval.task_subsample_seed=0 \
      run.eval.success_exp_to_advance=1 \
      run.eval.max_exp_per_task=3 \
      run.eval.only_search_unsuccessful_tasks=true \
      skills.vlm.model="${ACTOR_MODEL}" \
      skills.server.rl_policy.port=${RL_PORT} \
      skills.rl.backend="local" \
      skills.rl.pick_ckpt="${PICK_CKPT}" \
      skills.rl.pick_upright_ckpt="${PICK_UPRIGHT_CKPT}" \
      skills.curobo.align_empty_gripper_wrist_to_world_z=false \
      skills.curobo.planner_execute_mode1_replan=false \
      skills.curobo.planner_replan_every_k_steps=30 \
      skills.curobo.planner_max_replans=10 \
      skills.curobo.planner_skip_points_after_first_plan=12 \
      skills.curobo.planner_goal_pos_tolerance_m=0.03 \
      skills.curobo.planner_goal_rot_tolerance_rad=0.15 \
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
      mcts.mode="" \
      mcts.target_success_count=3 \
      mcts.avoid_terminal_in_selection=true \
      mcts.tasks.task_key="${TASK_KEY}" \
      mcts.save_tree_observations=false \
      mcts.save_tree_pcd=false \
      simulator=isaaclab \
      simulator.view_names=${VIEW_NAMES} \
      simulator.isaaclab.task="GroundedBase-v0" \
      simulator.isaaclab.samples_per_pixel_per_frame=2 \
      simulator.isaaclab.eval_num=${EVAL_NUM} \
      simulator.isaaclab.enable_pick=true \
      simulator.isaaclab.place_offset=0.06 \
      "simulator.isaaclab.pregrasp_noise_std_xyz=[0.022,0.022,0.01]" \
      "simulator.isaaclab.place_noise_std_xyz=[0.01,0.01,0.01]" \
      simulator.isaaclab.collect_depth=true \
      simulator.isaaclab.data_collect_depth_storage=uint16 \
      simulator.isaaclab.data_collect_depth_scale=1000.0 \
      simulator.isaaclab.collect_segmentation=false \
      simulator.isaaclab.collect_object_poses=true \
      simulator.app.headless=true \
      simulator.app.enable_cameras=true \
      simulator.app.rendering_mode="quality" \
      simulator.app.kit_args="--no-window" \
      env.scene_desc_file="${scene_file}" \
      "env.dome_light_intensity_range=[1500, 4000]" \
      env.front_camera_aug_azimuth_ranges_deg="-20.0:20.0" \
      "env.floor_z_range=[-1.0, -0.7]" \
      env.dome_light_texture_aug_prob=0.5 \
      env.front_camera_count=4 \
      env.enable_scene_visual_floor=true \
      env.enable_scene_visual_walls=true \
      env.hide_moveable_ground_visual=true \
      "$@"
done
