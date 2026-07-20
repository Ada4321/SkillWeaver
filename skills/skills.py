import logging
import os
import time

import numpy as np


def _dbg_release_state(simulator, task_env, tag):
    """Gated debug (MCTS_DEBUG_RELEASE=1): print grasped objs + EE z + finger joints, to see whether
    the place release actually opened the gripper / lifted the EE before the next skill runs."""
    if not os.environ.get("MCTS_DEBUG_RELEASE"):
        return
    import sys
    try:
        grasped, _ = simulator.get_grasped_and_collided_objs(task_env)
    except Exception as e:
        grasped = f"err:{e}"
    try:
        ee = simulator.get_ee_pose(task_env); ee_z = round(float(ee[2]), 4)
    except Exception as e:
        ee_z = f"err:{e}"
    try:
        fg = simulator.get_gripper_positions(task_env)
        fg = [round(float(x), 4) for x in (fg.tolist() if hasattr(fg, "tolist") else fg)]
    except Exception as e:
        fg = f"err:{e}"
    print(f"[DBG_RELEASE] {tag}: grasped={grasped} ee_z={ee_z} fingers={fg}", file=sys.stderr, flush=True)

# from utils.geometry_utils import get_target_loc_from_vlm3d
from utils.image_utils import _decode_base64
from utils.random_control import timestamp_seed_scope
from utils.geometry_utils import (
    q2R_wxyz, R2q_wxyz, quat_world_from_ee_axes,
    _quat_normalize_wxyz, _quat_angle_distance_rad_wxyz,
)
from utils.data_collect_utils import (
    _init_step_data_buffer,
    _collect_pre_step_data,
    _append_step_data,
    _extend_step_data,
)
from skills._constants import (
    ACTION_MODE_JOINT_ANGLES,
    ACTION_MODE_DELTA_EEPOSE,
    CONVEX_UPRIGHT_COS_THRESHOLD,
)
from skills._planner_exec import (
    _log_timing,
    _init_planner_stats,
    _record_planner_chunks,
    _write_planner_stats_to_node,
    _get_mode1_replan_cfg,
    _get_current_joint_targets,
    _get_ee_goal_errors,
    _goal_pose_reached,
    _skip_plan_prefix_points,
    _step_joint_action_with_mode1_substeps,
    _execute_planner_joint_states_mode1,
    _get_current_motion_planning_target,
    _get_current_release_joint_target,
    _step_release_at_current_pose,
    _step_joint_action_with_substeps,
)
from skills._holds import (
    _get_grasp_hold_cfg,
    _get_place_hold_cfg,
    _extract_root_velocity_norms,
    _execute_place_hold,
    _execute_grasp_hold,
)
from skills._targets import (
    _get_object_height,
    _resolve_place_target_base,
    get_object_z_max,
    _maybe_override_xy_for_straighten,
    get_surface_target_loc_from_vlm3d,
    get_container_opening_center_from_vlm3d,
    _compute_pre_grasp_quat_for_convex,
    get_3d_target_grasp,
    get_3d_target_place,
    _compute_cup_bottom_xy_offset_world,
    _apply_place_noise,
    _resolve_target_quat_candidates,
    _place_part_target_or_none,
)
from skills.place_plan import (
    build_place_plan,
    ORIENT_FREE,
    ORIENT_OBJECT,
    ORIENT_GRIPPER,
    RELEASE_RL,
    APPROACH_FROM_ALIGNMENT,
)

# ACTION_MODE_* and CONVEX_UPRIGHT_COS_THRESHOLD moved to skills._constants (imported above).


def _object_category_candidates(simulator, task_env, object_name: str) -> list[str]:
    candidates: list[str] = []
    get_category = getattr(simulator, "get_object_category", None)
    if callable(get_category):
        try:
            category = get_category(task_env, object_name)
        except Exception:
            category = None
        if isinstance(category, str) and category:
            candidates.append(category)
    if isinstance(object_name, str) and object_name:
        candidates.append(object_name)
        base, sep, suffix = object_name.rpartition("_")
        if sep and suffix.isdigit():
            candidates.append(base)
    return list(dict.fromkeys(candidates))


def _object_matches_category_set(simulator, task_env, object_name: str, categories: set[str]) -> bool:
    return any(candidate in categories for candidate in _object_category_candidates(simulator, task_env, object_name))


# Per-step data-collection helpers moved to utils.data_collect_utils (imported at top).


def _resolve_pick_policy_agent_name(object_name, simulator, task_env, rlpolicy):
    if rlpolicy is None:
        return "pick"
    resolve_fn = getattr(rlpolicy, "resolve_pick_agent_name", None)
    if callable(resolve_fn):
        return str(resolve_fn(object_name=object_name, simulator=simulator, task_env=task_env))
    return "pick"


def _resolve_place_policy_agent_name(held_object_name, simulator, task_env, rlpolicy):
    if rlpolicy is None or not held_object_name:
        return "place"
    resolve_fn = getattr(rlpolicy, "resolve_place_agent_name", None)
    if callable(resolve_fn):
        return str(resolve_fn(held_object_name=held_object_name, simulator=simulator, task_env=task_env))
    return "place"


def return_failure(reason: str, trajectory=None):
    terminate = True
    return trajectory, terminate, reason


def return_success(trajectory):
    terminate = False
    term_reason = "Success!"
    return trajectory, terminate, term_reason


def execute_rl_policy(
    task_name,
    object_name,
    simulator,
    task_env,
    rlpolicy,
    data_collection_mode=False,
    policy_agent_name=None,
    success_out=None,
    **kwargs
):
    agent_name = str(policy_agent_name) if policy_agent_name else task_name
    use_top_pcd_for_obs = bool(getattr(rlpolicy, "uses_top_pcd", lambda _n: False)(agent_name))
    use_handle_pcd_for_obs = bool(getattr(rlpolicy, "uses_handle_pcd", lambda _n: False)(agent_name))
    use_half_pcd_for_obs = simulator.uses_half_pcd_for_object(task_env, object_name)
    policy_obs, _ = simulator.reset(
        env=task_env,
        reset_level="task",
        task_name=task_name,
        object_name=object_name,
        use_top_pcd_for_obs=use_top_pcd_for_obs,
        use_handle_pcd_for_obs=use_handle_pcd_for_obs,
        use_half_pcd_for_obs=use_half_pcd_for_obs,
        **kwargs
    )
    if isinstance(policy_obs, dict):
        policy_obs = policy_obs["obs"]
    rlpolicy.reset_agent_new_episode(policy_obs, agent_name=agent_name)

    trajectory = _init_step_data_buffer(data_collection_mode)

    _rl_infer_times = []
    _rl_step_times = []
    while True:
        _t0 = time.perf_counter()
        action = rlpolicy.get_action(policy_obs, agent_name=agent_name)
        _rl_infer_times.append(time.perf_counter() - _t0)
        pre_step_data = _collect_pre_step_data(
            simulator,
            task_env,
            action,
            action_mode=ACTION_MODE_DELTA_EEPOSE,
            data_collection_mode=data_collection_mode,
        )
        _t0 = time.perf_counter()
        policy_obs, _, dones, _ = simulator.step(task_env, action, action_mode=ACTION_MODE_DELTA_EEPOSE)
        _rl_step_times.append(time.perf_counter() - _t0)
        rlpolicy.post_action_operation(dones, agent_name=agent_name)
        _append_step_data(
            trajectory,
            simulator,
            task_env,
            action,
            data_collection_mode,
            pre_step_data=pre_step_data,
        )
        if dones.sum() > 0:
            break
    if _rl_infer_times:
        _n = len(_rl_infer_times)
        _log_timing(
            f"[timing] rl_policy n={_n}"
            f" infer_mean_s={sum(_rl_infer_times)/_n:.4f} infer_total_s={sum(_rl_infer_times):.3f}"
            f" step_mean_s={sum(_rl_step_times)/_n:.4f} step_total_s={sum(_rl_step_times):.3f}"
        )
    # Capture task.success BEFORE simulator.unset(), which clears base_env.task.
    if success_out is not None:
        captured = None
        try:
            base_env = simulator._unwrap_env(task_env)
            task_obj = getattr(base_env, "task", None)
            success_tensor = getattr(task_obj, "success", None) if task_obj is not None else None
            if success_tensor is not None and len(success_tensor) > 0:
                element = success_tensor[0]
                captured = bool(element.item() if hasattr(element, "item") else element)
        except Exception:
            logging.exception("execute_rl_policy: failed to read task.success; treating as None.")
            captured = None
        success_out["success"] = captured
    simulator.unset(task_env)

    return return_success(trajectory)


def execute_drop(
    node,
    child_node,
    simulator,
    task_env,
    curobo,
    rlpolicy=None,
    data_collection_mode=False,
    planner_stats=None,
    retract_along_local_z: bool = False,
    disable_downward_yaw_fallback: bool = False,
    retract_dir_world=None,
):
    local_stats_created = planner_stats is None
    if local_stats_created:
        planner_stats = _init_planner_stats()

    trajectory = _init_step_data_buffer(data_collection_mode)
    current_q = simulator.get_joint_positions(task_env)
    if retract_along_local_z:
        # place_upright path: skip cuRobo release planning + retract motion.
        # Just open the gripper in place via a single is_release=True step.
        n_chunks = _step_release_at_current_pose(
            simulator,
            task_env,
            trajectory,
            data_collection_mode,
        )
        _record_planner_chunks(planner_stats, n_chunks)
        if local_stats_created:
            _write_planner_stats_to_node(child_node, planner_stats)
        return return_success(trajectory)
    res_curobo = curobo.release(payload={
        "current_q": current_q,
        "retract_along_local_z": retract_along_local_z,
        "disable_downward_yaw_fallback": disable_downward_yaw_fallback,
        "retract_dir_world": retract_dir_world,
    })
    joint_states = res_curobo["joint_states"]
    if joint_states is None:
        if local_stats_created:
            _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure("No valid trajectory found by cuRobo in release!")

    mode1_cfg = _get_mode1_replan_cfg(curobo)
    if not mode1_cfg["enabled"]:
        joint_states = np.array(joint_states)
        for action in joint_states:
            n_chunks = _step_joint_action_with_substeps(
                simulator,
                task_env,
                action,
                trajectory,
                data_collection_mode,
                is_release=True,
            )
            _record_planner_chunks(planner_stats, n_chunks)
    else:
        fixed_goal_pose = res_curobo.get("goal_pose", None)
        n_replans = 0
        while True:
            reached, need_replan, pos_err, rot_err = _execute_planner_joint_states_mode1(
                simulator=simulator,
                task_env=task_env,
                joint_states=joint_states,
                trajectory=trajectory,
                data_collection_mode=data_collection_mode,
                planner_stats=planner_stats,
                goal_pose=fixed_goal_pose,
                k_steps=mode1_cfg["k_steps"],
                pos_tol_m=mode1_cfg["pos_tol_m"],
                rot_tol_rad=mode1_cfg["rot_tol_rad"],
            )
            if reached:
                break
            if (not need_replan) or (n_replans >= int(mode1_cfg["max_replans"])):
                if local_stats_created:
                    _write_planner_stats_to_node(child_node, planner_stats)
                return return_failure(
                    "Failed to reach planner goal pose in release "
                    f"(replans={n_replans}, pos_err={pos_err}, rot_err={rot_err}).",
                    trajectory=trajectory,
                )

            n_replans += 1
            current_q = simulator.get_joint_positions(task_env)
            res_curobo = curobo.release(
                payload={
                    "current_q": current_q,
                    "goal_pose_override": fixed_goal_pose,
                    "prepend_start_buffer": False,
                    "retract_along_local_z": retract_along_local_z,
                    "disable_downward_yaw_fallback": disable_downward_yaw_fallback,
                    "retract_dir_world": retract_dir_world,
                }
            )
            joint_states = res_curobo["joint_states"]
            if joint_states is None:
                if local_stats_created:
                    _write_planner_stats_to_node(child_node, planner_stats)
                return return_failure("No valid trajectory found by cuRobo in release replan!", trajectory=trajectory)
            if fixed_goal_pose is None:
                fixed_goal_pose = res_curobo.get("goal_pose", None)
            joint_states = _skip_plan_prefix_points(
                joint_states,
                mode1_cfg.get("skip_points_after_first_plan", 0),
            )

    if local_stats_created:
        _write_planner_stats_to_node(child_node, planner_stats)
    return return_success(trajectory)


def _append_curobo_failure_record(child_node, record):
    """Accumulate cuRobo planning failure info on the node; serialized into the exp
    tree json as `curobo_failures` (a node can plan several times: hover move,
    arti Stage-A attempts, ...)."""
    try:
        lst = getattr(child_node, "curobo_failures", None)
        if lst is None:
            lst = []
            child_node.curobo_failures = lst
        lst.append(record)
    except Exception:
        logging.exception("failed to record curobo failure info on node")


def execute_move_held_object(
    node,
    child_node,
    simulator,
    task_env,
    curobo,
    rlpolicy=None,
    data_collection_mode=False,
    predefined_target_loc_3d=None,
    predefined_target_quat=None,
    rotation_only=False,
    preserve_object_position_on_fallback=False,
    planner_stats=None,
    disable_downward_yaw_fallback: bool = False,
    allow_object_yaw_fallback: bool = False,
    thin_collision_spheres: bool = False,
    narrow_fingers: bool = False,
    candidate_align_meta=None,
    failure_log_stage: str = "move",
):
    local_stats_created = planner_stats is None
    if local_stats_created:
        planner_stats = _init_planner_stats()

    if child_node.action_params["point_3d"] is None and predefined_target_loc_3d is None and not rotation_only:
        if local_stats_created:
            _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure("Action_params incomplete!")

    trajectory = _init_step_data_buffer(data_collection_mode)
    target_loc_3d = None
    if not rotation_only:
        target_loc_3d = predefined_target_loc_3d if predefined_target_loc_3d is not None else child_node.action_params["point_3d"]
    current_q = simulator.get_joint_positions(task_env)
    current_gripper = simulator.get_gripper_positions(task_env)

    quat_candidates = _resolve_target_quat_candidates(predefined_target_quat)

    move_payload = {
        "target_loc": target_loc_3d,
        "target_quat": quat_candidates[0],
        "current_q": current_q,
        "gripper_state": current_gripper,
    }
    move_payload["preserve_object_position_on_fallback"] = bool(preserve_object_position_on_fallback)
    move_payload["disable_downward_yaw_fallback"] = bool(disable_downward_yaw_fallback)
    move_payload["allow_object_yaw_fallback"] = bool(allow_object_yaw_fallback)
    # Drawer/door close-push drive: plan with the thin-sphere planner (collision_sphere_buffer=0) so the
    # gripper fingers fit around the recessed middle/bottom handle. The cabinet stays an obstacle for
    # ARM avoidance; only the 4mm gripper-sphere inflation (which RL's reset-IK doesn't have) is removed.
    if thin_collision_spheres:
        move_payload["thin_collision_spheres"] = True
    # Handle-grasp drive: plan with fingers locked at curobo.narrow_finger_width (full-open
    # fingertips sweep the next handle down on tight cabinets) and command the jaws at that
    # width during the drive; see cuRobo._get_narrow_motion_gen.
    if narrow_fingers:
        move_payload["narrow_fingers"] = True

    res_curobo = None
    joint_states = None
    cand_won_idx = -1
    cand_fail_records = []
    first_fail_goal = None
    for cand_idx, cand_quat in enumerate(quat_candidates):
        move_payload["target_quat"] = cand_quat
        _t0 = time.perf_counter()
        res_curobo = curobo.move(payload=move_payload)
        _log_timing(
            f"[timing] curobo_plan_s={time.perf_counter()-_t0:.3f} cand_idx={cand_idx}"
        )
        joint_states = res_curobo["joint_states"]
        if joint_states is not None:
            cand_won_idx = cand_idx
            break
        status = res_curobo.get("status")
        rec = {"cand": cand_idx, "status": status}
        if candidate_align_meta and cand_idx < len(candidate_align_meta):
            rec["relax_deg"] = candidate_align_meta[cand_idx].get("relax_deg")
            rec["sweep_deg"] = candidate_align_meta[cand_idx].get("sweep_deg")
        cand_fail_records.append(rec)
        if first_fail_goal is None:
            first_fail_goal = res_curobo.get("goal_pose_0")
        logging.warning(
            "[move_cand] %s cand %d/%d failed: status=%s%s",
            failure_log_stage, cand_idx + 1, len(quat_candidates), status,
            (f" relax={rec.get('relax_deg')} sweep={rec.get('sweep_deg')}"
             if "relax_deg" in rec else ""),
        )
    if joint_states is None:
        culprits = []
        try:
            if first_fail_goal is not None and hasattr(curobo, "log_goal_collision_culprits"):
                culprits = curobo.log_goal_collision_culprits(
                    first_fail_goal, tag=failure_log_stage,
                    narrow_width=(getattr(curobo, "narrow_finger_width", None) if narrow_fingers else None),
                )
        except Exception:
            logging.exception("[collide_probe] culprit probe raised; continuing.")
        _append_curobo_failure_record(child_node, {
            "stage": failure_log_stage,
            "n_candidates": len(quat_candidates),
            "won": None,
            "candidates": cand_fail_records,
            "culprits": culprits,
        })
        if candidate_align_meta:
            logging.warning(
                "[place_align] all %d candidates failed (max relax tier tried: %.1f deg)",
                len(quat_candidates),
                float(candidate_align_meta[-1]["relax_deg"]) if len(candidate_align_meta) == len(quat_candidates) else -1.0,
            )
        if local_stats_created:
            _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure("No valid trajectory found by cuRobo!")
    if cand_fail_records:
        # Won after some failed candidates: keep the partial record for post-hoc analysis.
        _append_curobo_failure_record(child_node, {
            "stage": failure_log_stage,
            "n_candidates": len(quat_candidates),
            "won": cand_won_idx,
            "candidates": cand_fail_records,
            "culprits": None,
        })
    if candidate_align_meta and cand_won_idx < len(candidate_align_meta):
        m = candidate_align_meta[cand_won_idx]
        logging.warning(
            "[place_align] cand %d/%d won: relax_tier=%.1f deg, actual_residual=%.2f deg, free_axis_sweep=%.0f deg",
            cand_won_idx, len(quat_candidates),
            m["relax_deg"], m["residual_deg"], m["sweep_deg"],
        )
    if len(quat_candidates) > 1:
        _log_timing(
            f"[timing] cand_won_idx={cand_won_idx} total={len(quat_candidates)}"
        )

    mode1_cfg = _get_mode1_replan_cfg(curobo)
    force_gripper_delta = None
    get_grasped_fn = getattr(simulator, "get_grasped_and_collided_objs", None)
    if callable(get_grasped_fn):
        try:
            grasped_names, _ = get_grasped_fn(task_env)
            if len(grasped_names) > 0:
                force_gripper_delta = -1.0
        except Exception:
            force_gripper_delta = None
    if not mode1_cfg["enabled"]:
        joint_states = np.array(joint_states)
        for action in joint_states:
            n_chunks = _step_joint_action_with_substeps(
                simulator,
                task_env,
                action,
                trajectory,
                data_collection_mode,
                is_release=False,
            )
            _record_planner_chunks(planner_stats, n_chunks)
    else:
        fixed_goal_pose = res_curobo.get("goal_pose", None)
        n_replans = 0
        while True:
            if callable(get_grasped_fn):
                try:
                    grasped_names, _ = get_grasped_fn(task_env)
                    force_gripper_delta = -1.0 if len(grasped_names) > 0 else None
                except Exception:
                    force_gripper_delta = None
            reached, need_replan, pos_err, rot_err = _execute_planner_joint_states_mode1(
                simulator=simulator,
                task_env=task_env,
                joint_states=joint_states,
                trajectory=trajectory,
                data_collection_mode=data_collection_mode,
                planner_stats=planner_stats,
                goal_pose=fixed_goal_pose,
                k_steps=mode1_cfg["k_steps"],
                pos_tol_m=mode1_cfg["pos_tol_m"],
                rot_tol_rad=mode1_cfg["rot_tol_rad"],
                force_gripper_delta=force_gripper_delta,
            )
            if reached:
                break
            if (not need_replan) or (n_replans >= int(mode1_cfg["max_replans"])):
                if local_stats_created:
                    _write_planner_stats_to_node(child_node, planner_stats)
                return return_failure(
                    "Failed to reach planner goal pose in move "
                    f"(replans={n_replans}, pos_err={pos_err}, rot_err={rot_err}).",
                    trajectory=trajectory,
                )

            n_replans += 1
            current_q = simulator.get_joint_positions(task_env)
            current_gripper = simulator.get_gripper_positions(task_env)
            replan_payload = dict(move_payload)
            replan_payload["current_q"] = current_q
            replan_payload["gripper_state"] = current_gripper
            replan_payload["goal_pose_override"] = fixed_goal_pose
            replan_payload["prepend_start_buffer"] = False
            res_curobo = curobo.move(payload=replan_payload)
            joint_states = res_curobo["joint_states"]
            if joint_states is None:
                if local_stats_created:
                    _write_planner_stats_to_node(child_node, planner_stats)
                return return_failure("No valid trajectory found by cuRobo in move replan!", trajectory=trajectory)
            if fixed_goal_pose is None:
                fixed_goal_pose = res_curobo.get("goal_pose", None)
            joint_states = _skip_plan_prefix_points(
                joint_states,
                mode1_cfg.get("skip_points_after_first_plan", 0),
            )

    if local_stats_created:
        _write_planner_stats_to_node(child_node, planner_stats)
    return return_success(trajectory)


def execute_grasp_rlpolicy(
    node,
    child_node,
    simulator,
    task_env,
    curobo,
    rlpolicy=None,
    data_collection_mode=False,
):
    planner_stats = _init_planner_stats()

    ## Compute pre-grasp location
    pre_grasp_location_3d = get_3d_target_grasp(node, child_node, simulator, task_env)
    if pre_grasp_location_3d is None:
        _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure("Action_params incomplete!")

    ## For convex+upright objects, override pre-grasp wrist orientation so that
    ## the two-finger axis is perpendicular to the object's handle axis.
    object_name = child_node.action_params["asset"]
    convex_objects = getattr(rlpolicy, "convex_objects", set()) if rlpolicy is not None else set()
    pre_grasp_quat = None
    if object_name in convex_objects:
        pre_grasp_quat = _compute_pre_grasp_quat_for_convex(simulator, task_env, object_name)

    ## Move gripper to pre-grasp location. Optional planner-failure fallback: when the
    ## move fails (with replan disabled that is always "cuRobo could not plan", zero
    ## motion executed), retry with the goal shifted in xy around the noised base --
    ## growing radii, each split into num_directions headings (from +x, CCW).
    candidate_offsets = [np.zeros(3, dtype=np.float32)]
    if bool(getattr(simulator, "pregrasp_fallback_enable", False)):
        n_dirs = int(getattr(simulator, "pregrasp_fallback_num_directions", 8))
        for radius in np.asarray(getattr(simulator, "pregrasp_fallback_radii", []), dtype=np.float32).reshape(-1):
            for k in range(n_dirs):
                theta = 2.0 * np.pi * k / n_dirs
                candidate_offsets.append(
                    np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0], dtype=np.float32)
                )
    trajectory = _init_step_data_buffer(data_collection_mode)
    terminate, term_reason = True, "pregrasp: no candidate tried"
    for attempt_idx, offset in enumerate(candidate_offsets):
        trajectory_move, terminate, term_reason = execute_move_held_object(
            node,
            child_node,
            simulator,
            task_env,
            curobo,
            rlpolicy,
            data_collection_mode=data_collection_mode,
            predefined_target_loc_3d=pre_grasp_location_3d + offset,
            predefined_target_quat=pre_grasp_quat,
            planner_stats=planner_stats,
        )
        _extend_step_data(trajectory, trajectory_move, data_collection_mode)
        if not terminate:
            if attempt_idx > 0:
                logging.info(
                    f"pregrasp fallback: goal offset {offset[:2].tolist()} reached "
                    f"after {attempt_idx + 1} attempts ({object_name})."
                )
            break
        if len(candidate_offsets) > 1:
            logging.info(
                f"pregrasp fallback: attempt {attempt_idx + 1}/{len(candidate_offsets)} failed "
                f"({term_reason})"
            )
    child_node.pregrasp_fallback_attempts = len(candidate_offsets) if terminate else attempt_idx + 1
    child_node.pregrasp_fallback_offset = (
        offset[:2].tolist() if (not terminate and attempt_idx > 0) else None
    )
    if terminate:
        _write_planner_stats_to_node(child_node, planner_stats)
        return trajectory, terminate, term_reason

    ## Execute grasp with RL policy
    task_name = "pick"
    policy_agent_name = _resolve_pick_policy_agent_name(object_name, simulator, task_env, rlpolicy)
    pick_success_out = {}
    trajectory_rl, terminate, term_reason = execute_rl_policy(
        task_name,
        object_name,
        simulator,
        task_env,
        rlpolicy,
        data_collection_mode=data_collection_mode,
        policy_agent_name=policy_agent_name,
        success_out=pick_success_out,
    )
    child_node.rl_pick_success = pick_success_out.get("success")
    ## Concatenate trajectories and return
    _extend_step_data(trajectory, trajectory_rl, data_collection_mode)
    if terminate:
        _write_planner_stats_to_node(child_node, planner_stats)
        return trajectory, terminate, term_reason
    trajectory_hold = _execute_grasp_hold(
        simulator=simulator,
        task_env=task_env,
        curobo=curobo,
        object_name=object_name,
        data_collection_mode=data_collection_mode,
    )
    _extend_step_data(trajectory, trajectory_hold, data_collection_mode)
    _write_planner_stats_to_node(child_node, planner_stats)
    return trajectory, terminate, term_reason


def _establish_drawer_grasp(simulator, task_env, n_steps, data_collection_mode):
    """Stage A.5 (open only): close the jaws onto the handle for n_steps while RIGIDLY PINNING the
    arm joints to the post-cuRobo-drive config (action_mode=0 = direct joint targets). Mirrors
    OpenDrawerGraspedEnv forced-close, which overwrites robot_dof_targets[:, :n_arm] = grasp_arm_q
    every step so the closing reaction can't drift the gripper off the handle (the zero-EE-delta IK
    "hold" follows that reaction and drifts -- the original bug). Gripper target = current finger
    - 0.01 per control step (RL gripper_action=-1 * 0.01), clamped to the finger lower limit."""
    import torch as _torch
    base_env = simulator._unwrap_env(task_env)
    n_arm = len(base_env.arm_joint_indices)
    arm_q = base_env.robot.data.joint_pos[:, :n_arm].clone()           # freeze cuRobo arm config
    finger_lower = float(base_env.robot_dof_lower_limits[-1])
    trajectory = _init_step_data_buffer(data_collection_mode)
    for _ in range(int(n_steps)):
        cur_finger = base_env.robot.data.joint_pos[:, -2:]
        gripper_t = (cur_finger - 0.01).clamp_min(finger_lower)
        joint_targets = _torch.cat((arm_q, gripper_t), dim=-1)[0].detach().cpu().numpy()  # (n_arm+2,)
        pre = _collect_pre_step_data(
            simulator, task_env, joint_targets,
            action_mode=ACTION_MODE_JOINT_ANGLES, data_collection_mode=data_collection_mode,
        )
        simulator.step(task_env, joint_targets, action_mode=ACTION_MODE_JOINT_ANGLES)
        _append_step_data(trajectory, simulator, task_env, joint_targets, data_collection_mode, pre_step_data=pre)
    return trajectory


def _execute_arti_skill(node, child_node, simulator, task_env, curobo, rlpolicy, data_collection_mode,
                    task_name, sample_kind, do_grasp):
    """Shared articulation open/close execution (drawer/door). task_name = the RL task / agent
    (open_drawer | close_drawer | close_door); sample_kind = "open" (handle_grasps) | "close"
    (closepush_init) for the Stage-A init pose; do_grasp = Stage-A.5 forced jaw-close (open only).
    The selected part arrives fully resolved in action_params: asset = the part key,
    asset_object = the owning object (the VLM parse emits this schema). The part key flows
    through the rest of the pipeline (Stage-A sampling + Stage-B
    reset -> task.prepare); no borrowed int indices. See docs/arti/skill_plumbing.md §5."""
    planner_stats = _init_planner_stats()
    part_key = child_node.action_params.get("asset")
    cabinet = child_node.action_params.get("asset_object")
    if part_key is None or cabinet is None:
        _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure(f"{task_name}: action_params incomplete (need asset part key + asset_object)!")
    _dbg_release_state(simulator, task_env, f"{task_name}_start(after recover)")
    grasped, _ = simulator.get_grasped_and_collided_objs(task_env)
    if len(grasped) > 0:
        _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure(f"{task_name}: gripper not empty (holding {grasped}).")

    # Stage A: sample up to part_init_max_samples init gripper poses (q0->world;
    # prismatic/revolute) and cuRobo-drive to the first reachable one. A single random
    # sample wastes the other collision-free entries in the npz when it happens to be
    # unreachable (e.g. microwave door open at 120 deg puts many press poses in awkward
    # spots near the base).
    # NOTE: the init quat is the grasp_site pose; cuRobo's target frame must match (validate).
    max_tries = max(1, int(getattr(curobo, "part_init_max_samples", 5)))
    trajectory, terminate, term_reason = None, True, "Stage-A init: no attempt ran"
    for attempt in range(max_tries):
        pos_w, quat_w, _finger, _open = simulator.sample_part_init_pose(task_env, cabinet, part_key, kind=sample_kind)
        traj_attempt, terminate, term_reason = execute_move_held_object(
            node, child_node, simulator, task_env, curobo, rlpolicy,
            data_collection_mode=data_collection_mode,
            predefined_target_loc_3d=pos_w, predefined_target_quat=quat_w, planner_stats=planner_stats,
            # Handle-grasp drive (open): narrow-finger planner + narrow jaw command, else the
            # full-open fingertips sweep the next handle down. Close-push keeps the jaws-open
            # arrival the RL close policies were trained on (thin planner only).
            thin_collision_spheres=not do_grasp,
            narrow_fingers=do_grasp,
            failure_log_stage=f"{task_name}_init",
        )
        if trajectory is None:
            trajectory = traj_attempt
        else:
            _extend_step_data(trajectory, traj_attempt, data_collection_mode)
        if not terminate:
            if attempt > 0:
                logging.warning("%s: Stage-A init succeeded on attempt %d/%d.", task_name, attempt + 1, max_tries)
            break
        logging.warning(
            "%s: Stage-A init attempt %d/%d failed (%s); resampling.",
            task_name, attempt + 1, max_tries, term_reason,
        )
    if terminate:
        _write_planner_stats_to_node(child_node, planner_stats)
        return trajectory, terminate, term_reason

    # Stage A.5 (grasped open only): close jaws onto the handle.
    if do_grasp:
        base_env = simulator._unwrap_env(task_env)
        n_forced = int(getattr(base_env.cfg, "drawer_grasp_forced_steps", 10))
        traj_grasp = _establish_drawer_grasp(simulator, task_env, n_forced, data_collection_mode)
        _extend_step_data(trajectory, traj_grasp, data_collection_mode)

    # Stage B: run the RL policy. The part key flows to simulator.reset -> task.prepare.
    success_out = {}
    trajectory_rl, terminate, term_reason = execute_rl_policy(
        task_name, cabinet, simulator, task_env, rlpolicy,
        data_collection_mode=data_collection_mode, policy_agent_name=task_name,
        success_out=success_out, part=part_key,
    )
    setattr(child_node, f"rl_{task_name}_success", success_out.get("success"))
    _extend_step_data(trajectory, trajectory_rl, data_collection_mode)

    # Stage C (grasped open only): disengage. The RL pull ends with the fingers still
    # around the handle, which the NEXT skill's planner (full-open locked fingers + 4mm
    # buffer) judges as INVALID_START_STATE_WORLD_COLLISION. Unified with the place
    # release: open the jaws in place, then curobo.release-retract along the drawer's
    # slide axis (two-stage world handling + in-place-open ultimate fallback). A Stage-C
    # failure terminates the skill like any other stage failure.
    if do_grasp and not terminate:
        n_chunks = _step_release_at_current_pose(
            simulator, task_env, trajectory, data_collection_mode, num_steps=2,
        )
        _record_planner_chunks(planner_stats, n_chunks)
        pull_dir = None
        try:
            # part_key is already resolved at the top of this function (action_params asset).
            pj = simulator._part_joint(task_env, cabinet, part_key)
            if pj.get("joint_type") == "prismatic":
                art = simulator._articulation(simulator._unwrap_env(task_env), cabinet)
                root_quat = art.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
                v = q2R_wxyz(root_quat) @ np.asarray(pj["axis_obj"], dtype=np.float64)
                norm = float(np.linalg.norm(v))
                if norm > 1e-9:
                    pull_dir = (v / norm).tolist()
        except Exception:
            logging.exception(
                "%s: Stage-C failed to derive slide-axis pull dir; using vertical lift.", task_name,
            )
        traj_disengage, term_c, reason_c = execute_drop(
            node, child_node, simulator, task_env, curobo, rlpolicy,
            data_collection_mode=data_collection_mode, planner_stats=planner_stats,
            retract_dir_world=pull_dir,
        )
        _extend_step_data(trajectory, traj_disengage, data_collection_mode)
        if term_c:
            _write_planner_stats_to_node(child_node, planner_stats)
            return trajectory, term_c, f"{task_name} Stage-C disengage failed: {reason_c}"

    _write_planner_stats_to_node(child_node, planner_stats)
    return trajectory, terminate, term_reason


def execute_open_drawer(node, child_node, simulator, task_env, curobo, rlpolicy=None, data_collection_mode=False):
    return _execute_arti_skill(node, child_node, simulator, task_env, curobo, rlpolicy, data_collection_mode,
                           task_name="open_drawer", sample_kind="open", do_grasp=True)


def execute_close_drawer(node, child_node, simulator, task_env, curobo, rlpolicy=None, data_collection_mode=False):
    return _execute_arti_skill(node, child_node, simulator, task_env, curobo, rlpolicy, data_collection_mode,
                           task_name="close_drawer", sample_kind="close", do_grasp=False)


def execute_close_door(node, child_node, simulator, task_env, curobo, rlpolicy=None, data_collection_mode=False):
    # Revolute door, push-to-close. The door part key arrives in action_params like any part.
    return _execute_arti_skill(node, child_node, simulator, task_env, curobo, rlpolicy, data_collection_mode,
                           task_name="close_door", sample_kind="close", do_grasp=False)


def _half_height_at(simulator, task_env, object_name, quat_override=None):
    """Half of the held object's z-extent, optionally under a goal orientation; None on failure."""
    try:
        rng = simulator.get_object_3d_range(task_env, object_name, quat_override=quat_override)
        return 0.5 * float(rng[-1] - rng[-2])
    except Exception:
        return None


def _hover_above(move_target_loc_3d, clearance, simulator):
    """Hover point: target + clearance on z, with the place xy/z noise applied."""
    hover = move_target_loc_3d.copy()
    hover[2] += clearance
    return _apply_place_noise(hover, simulator)


def _dbg_align_dump(simulator, task_env, object_name, target_asset, alignment, target_quat_candidates):
    """MCTS_DEBUG_ALIGN=1 stderr dump of the current grasp/object/target frames and every
    goal-quat candidate (object-axis alignment debugging)."""
    if not os.environ.get("MCTS_DEBUG_ALIGN"):
        return
    try:
        import sys as _sys
        def _qconj(q): w,x,y,z=q; return np.array([w,-x,-y,-z],dtype=np.float64)
        def _qmul(a,b):
            aw,ax,ay,az=a; bw,bx,by,bz=b
            return np.array([aw*bw-ax*bx-ay*by-az*bz,
                             aw*bx+ax*bw+ay*bz-az*by,
                             aw*by-ax*bz+ay*bw+az*bx,
                             aw*bz+ax*by-ay*bx+az*bw],dtype=np.float64)
        def _R(q):
            w,x,y,z=q/np.linalg.norm(q)
            return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                             [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                             [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        _fmt=lambda v:'['+','.join(f'{c:+.2f}' for c in v)+']'
        ee_now=simulator.get_ee_pose(task_env)[3:].astype(np.float64)
        obj_now=simulator.get_object_pose(task_env,object_name)[3:].astype(np.float64)
        tgt_now=simulator.get_object_pose(task_env,target_asset)[3:].astype(np.float64)
        q_rel=_qmul(_qconj(ee_now),obj_now)
        Ree=_R(ee_now); Robj=_R(obj_now)
        print(f"[DBG_ALIGN] held={object_name} alignment={alignment}",file=_sys.stderr)
        print(f"[DBG_ALIGN] GRASP NOW: palm_x(world)={_fmt(Ree@[1,0,0])} approach_z={_fmt(Ree@[0,0,1])} finger_y={_fmt(Ree@[0,1,0])}",file=_sys.stderr)
        print(f"[DBG_ALIGN]   bottle local-z(long?) world now={_fmt(Robj@[0,0,1])} local-x={_fmt(Robj@[1,0,0])} local-y={_fmt(Robj@[0,1,0])}",file=_sys.stderr)
        print(f"[DBG_ALIGN]   cabinet local-x world={_fmt(_R(tgt_now)@[1,0,0])} local-y={_fmt(_R(tgt_now)@[0,1,0])} local-z={_fmt(_R(tgt_now)@[0,0,1])}",file=_sys.stderr)
        print(f"[DBG_ALIGN]   q_rel(obj-in-ee, wxyz)={[round(float(c),3) for c in q_rel]}",file=_sys.stderr)
        for i,cand in enumerate(target_quat_candidates or []):
            cand=np.asarray(cand,dtype=np.float64)
            Rdelta=_R(_qmul(cand,_qconj(obj_now)))
            palm_goal=Rdelta@(Ree@np.array([1.0,0,0]))
            bottle_goal=_R(cand)@np.array([0,0,1.0])
            print(f"[DBG_ALIGN]   cand[{i}]: bottle_long_world={_fmt(bottle_goal)} -> palm_goal_x(world)={_fmt(palm_goal)}",file=_sys.stderr)
        _sys.stderr.flush()
    except Exception as _e:
        print(f"[DBG_ALIGN] error: {_e}",file=__import__('sys').stderr,flush=True)


def _resolve_alignment_candidates(plan, object_name, simulator, task_env, curobo):
    """Alignment intent -> object-goal quaternion candidates + relax/sweep meta (shared by
    the drop and RL finish families). (None, []) for ORIENT_FREE. Consumes plan.asset /
    plan.alignment (the single source of intent)."""
    align_meta = []
    if plan.orientation_mode == ORIENT_OBJECT:
        solver = simulator.get_rotation_candidates_from_alignment
    elif plan.orientation_mode == ORIENT_GRIPPER:
        solver = simulator.get_gripper_rotation_candidates_from_alignment
    else:
        return None, align_meta
    candidates = solver(
        task_env, object_name, plan.target_object, plan.alignment,
        sweep_angles_deg=getattr(curobo, "alignment_free_axis_sweep_deg", None),
        relax_ladder_deg=getattr(curobo, "place_align_relax_ladder_deg", None),
        meta_out=align_meta,
        part=plan.place_into_part,
    )
    if plan.orientation_mode == ORIENT_OBJECT:
        _dbg_align_dump(simulator, task_env, object_name, plan.target_object, plan.alignment, candidates)
    return candidates, align_meta


def _place_drop_finish(
    plan,
    node,
    child_node,
    simulator,
    task_env,
    curobo,
    rlpolicy=None,
    data_collection_mode=False,
):
    """Unified drop-family finish (vertical, n_app = -z): hover above the resting pose by
    the branch clearance knob, move the held object there with the (optional) target
    orientation, then release.

    Collapses the three former bodies, parameterized by the PlacePlan:
      - place_with_drop              -> orientation=free,    clearance=place_offset
      - place_with_orientation_drop  -> orientation=object,  clearance=place_offset_orientation_drop
      - place_with_gripper_orientation_drop -> orientation=gripper, clearance=place_offset_gripper_orientation

    Behavior-preserving differences kept exactly: the held-object lookup + straighten
    override + downward-yaw-fallback disable + object-axis height re-prediction all apply
    only to the orientation branches; plain drop skips them.
    """
    planner_stats = _init_planner_stats()
    orient = plan.orientation_mode

    ## Held object is needed for alignment + the straighten xy-override (orientation only).
    object_name = None
    if orient != ORIENT_FREE:
        grasped_names, _ = simulator.get_grasped_and_collided_objs(task_env)
        if len(grasped_names) == 0:
            _write_planner_stats_to_node(child_node, planner_stats)
            return return_failure("No object is being held!")
        object_name = grasped_names[0]

    ## Target base: part interior floor + half height (shared with RL) else container opening top
    ## + half CURRENT height (get_3d_target_place); straighten xy override applied inside.
    move_target_loc_3d, _is_part_target = _resolve_place_target_base(
        node, child_node, simulator, task_env, mode="container", held_object_name=object_name,
    )
    if move_target_loc_3d is None:
        _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure("Action_params incomplete!")

    ## Orientation alignment -> object-goal quaternion candidates (None for free).
    target_quat_candidates, align_meta = _resolve_alignment_candidates(
        plan, object_name, simulator, task_env, curobo,
    )

    ## Re-predict the half height under the target orientation and correct the CURRENT-pose
    ## term baked in above. Applies to any alignment that reorients the held object
    ## (object-axis AND gripper-axis -- gripper-align rotates the object rigidly, so its
    ## z-extent changes too); plain drop has no orientation change. For the non-vertical
    ## regime this correction is later overwritten by move_target[2] = point_3d.z, so it only
    ## matters for a VERTICAL gripper/object-align place (rest exactly on the surface/container top).
    if orient in (ORIENT_OBJECT, ORIENT_GRIPPER) and target_quat_candidates:
        half_pred = _half_height_at(simulator, task_env, object_name, quat_override=target_quat_candidates[0])
        half_cur = _half_height_at(simulator, task_env, object_name)
        if half_pred is not None and half_cur is not None:
            move_target_loc_3d[2] += half_pred - half_cur

    ## Approach axis (motion geometry, from the gripper alignment): the insert direction and
    ## the retract (-n_app) for the non-vertical regime. NOT the regime judge — an oblique
    ## insert (e.g. the 120-deg-door microwave) legitimately enters at an angle, so the
    ## approach axis says HOW to move, never WHETHER the target is side-opening.
    gravity_dir = np.array([0.0, 0.0, -1.0])
    n_app = gravity_dir
    if plan.approach_source == APPROACH_FROM_ALIGNMENT and object_name is not None:
        axis = simulator.get_gripper_approach_axis_world(
            task_env, object_name, plan.target_object, plan.alignment,
            sweep_angles_deg=getattr(curobo, "alignment_free_axis_sweep_deg", None),
            part=plan.place_into_part,
        )
        if axis is not None:
            n_app = np.asarray(axis, dtype=np.float64)

    ## Regime (RFC §5): decided by the TARGET's own opening direction (unified meta store),
    ## not by the gripper axis — the z-rule must not flip with the VLM's alignment choice.
    ## Vertical (opening up: table/surface/bowl/drawer) keeps the gravity-finish (hover by
    ## the branch clearance + drop, retract +z). Non-vertical (side-opening cavity, e.g. the
    ## microwave) uses the small settle clearance and retracts along -n_app.
    ## plan.asset=None is a dead path for the drop family (get_3d_target_place would have
    ## KeyError'd already) — defensive vertical default. Part places query the PART's own
    ## opening dir (object_opening_dirs is [name][part]; a mixed drawer+door asset's base
    ## follows the door rule, which would misjudge a drawer place as non-vertical).
    if plan.target_object is not None:
        opening_dir_world = simulator.get_opening_dir_world(
            task_env, plan.target_object, part=plan.place_into_part or "base"
        )
    else:
        opening_dir_world = np.array([0.0, 0.0, 1.0])
    is_vertical = float(opening_dir_world[2]) >= float(
        np.cos(np.radians(getattr(simulator, "approach_vertical_angle_deg", 20.0)))
    )

    # Non-vertical (e.g. horizontal microwave insert): the container-top z_max baked in by
    # get_3d_target_place is a "drop from the rim" assumption that doesn't hold for a sideways
    # insert. Use the VLM point's own z (its intended object-center height inside the cavity);
    # this also overrides the +0.5*height get_3d_target_place added, so the object center lands
    # exactly at the point. Part places (place_into_part set) keep their AABB-top target.
    if not is_vertical and plan.place_into_part is None:
        move_target_loc_3d[2] = float(plan.point_3d[2])

    # Quantity B (clearance): vertical -> branch knob; non-vertical -> small settle gap.
    clearance = (
        getattr(simulator, plan.clearance_key)
        if is_vertical
        else float(getattr(simulator, "place_offset_nonvertical", 0.01))
    )
    # Quantity C (retract): along -n_app; None means the vertical +z lift in curobo.release.
    retract_dir_world = None if is_vertical else (-n_app).tolist()

    hover_target_loc_3d = _hover_above(move_target_loc_3d, clearance, simulator)

    ## Move held object to the hover loc with the (optional) target orientation.
    trajectory, terminate, term_reason = execute_move_held_object(
        node,
        child_node,
        simulator,
        task_env,
        curobo,
        rlpolicy,
        data_collection_mode=data_collection_mode,
        predefined_target_loc_3d=hover_target_loc_3d,
        predefined_target_quat=target_quat_candidates,
        preserve_object_position_on_fallback=getattr(curobo, "place_preserve_object_position_on_fallback", True),
        planner_stats=planner_stats,
        disable_downward_yaw_fallback=(orient != ORIENT_FREE),
        candidate_align_meta=align_meta if target_quat_candidates else None,
    )
    if terminate:
        _write_planner_stats_to_node(child_node, planner_stats)
        return trajectory, terminate, term_reason

    ## Release the object above the target.
    trajectory_drop, terminate, term_reason = execute_drop(
        node,
        child_node,
        simulator,
        task_env,
        curobo,
        rlpolicy,
        data_collection_mode=data_collection_mode,
        planner_stats=planner_stats,
        retract_dir_world=retract_dir_world,
    )

    _extend_step_data(trajectory, trajectory_drop, data_collection_mode)
    _write_planner_stats_to_node(child_node, planner_stats)
    return trajectory, terminate, term_reason


def execute_move_without_place(
    node,
    child_node,
    simulator,
    task_env,
    curobo,
    rlpolicy=None,
    data_collection_mode=False,
):
    planner_stats = _init_planner_stats()

    if (
        child_node.action_params.get("point_3d") is None
        or child_node.action_params.get("asset_id") is None
        or child_node.action_params.get("asset") is None
        or child_node.action_params.get("alignment") is None
    ):
        _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure("Action_params incomplete!")

    grasped_names, _ = simulator.get_grasped_and_collided_objs(task_env)
    if len(grasped_names) == 0:
        _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure("No object is being held!")
    object_name = grasped_names[0]

    target_quat_candidates = simulator.get_rotation_candidates_from_alignment(
        task_env,
        object_name,
        child_node.action_params["asset"],
        child_node.action_params["alignment"],
        sweep_angles_deg=getattr(curobo, "alignment_free_axis_sweep_deg", None),
    )

    move_target_loc_3d = np.asarray(child_node.action_params["point_3d"], dtype=float).copy()

    trajectory, terminate, term_reason = execute_move_held_object(
        node,
        child_node,
        simulator,
        task_env,
        curobo,
        rlpolicy,
        data_collection_mode=data_collection_mode,
        predefined_target_loc_3d=move_target_loc_3d,
        predefined_target_quat=target_quat_candidates,
        preserve_object_position_on_fallback=getattr(curobo, "place_preserve_object_position_on_fallback", True),
        planner_stats=planner_stats,
        disable_downward_yaw_fallback=True,
    )
    _write_planner_stats_to_node(child_node, planner_stats)
    if terminate:
        return trajectory, terminate, term_reason
    return return_success(trajectory)


def _place_rl_finish(
    plan,
    node,
    child_node,
    simulator,
    task_env,
    curobo,
    rlpolicy=None,
    data_collection_mode=False,
):
    """Unified RL-policy place finish: place (orientation=free) and place_with_orientation
    (orientation=object_align). Move the held object to a hover above the surface target,
    hand off to the RL place policy, then release.

    Merges the two former RL bodies. Orientation-specific bits (alignment candidates, the
    target-orientation z-extent re-prediction, the MCTS_DEBUG_ALIGN dump, the held-object
    quat in the move, and object_original_height passed to the RL reset) are gated on
    `orient`; the geometric place_upright pre-check is plain-place only. Everything else is
    shared. IMPORTANT: place_with_orientation formerly inlined its own RL rollout (which
    omitted the top/half pcd obs flags) and its own release; both now go through the shared
    execute_rl_policy / execute_drop helpers, plus it now gets the settle-hold + cup-bottom
    alignment. These are behavior changes for place_with_orientation -- regress an RL
    place_with_orientation npz.
    """
    planner_stats = _init_planner_stats()
    orient_mode = plan.orientation_mode
    # has_orient: this place carries an orientation constraint (object-axis OR gripper-axis align),
    # so it produces goal-quat candidates and re-predicts the target-orientation z-extent. ORIENT_FREE
    # (plain place) has neither. gripper-align uses the gripper-axis candidate solver instead.
    has_orient = orient_mode in (ORIENT_OBJECT, ORIENT_GRIPPER)

    grasped_names, _ = simulator.get_grasped_and_collided_objs(task_env)
    if len(grasped_names) == 0:
        _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure("No object is being held!")
    object_name = grasped_names[0]

    ## Target base: part interior floor + half height (shared with drop) else the surface z
    ## (no half height yet -- lifted below once the goal orientation is known);
    ## straighten xy override applied inside.
    move_target_loc_3d, is_part_target = _resolve_place_target_base(
        node, child_node, simulator, task_env, mode="surface",
        held_object_name=(object_name if has_orient else None),
    )
    if move_target_loc_3d is None:
        _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure("Action_params incomplete!")

    ## Orientation alignment -> object-goal quaternion candidates. object_align aligns a held-object
    ## axis; gripper_align aligns a gripper axis (mapped to the held-object goal via the rigid grasp).
    target_quat_candidates, align_meta = _resolve_alignment_candidates(
        plan, object_name, simulator, task_env, curobo,
    )

    ## Lift the target by half the held object's height so the object center rests on the
    ## surface. object/gripper-align re-predicts the half height under the target
    ## orientation (it changes the footprint); plain place uses the current height.
    ## Skipped for a part target -- the part base already includes floor + half height.
    if not is_part_target:
        half = None
        if has_orient and target_quat_candidates:
            half = _half_height_at(simulator, task_env, object_name, quat_override=target_quat_candidates[0])
        if half is None:
            object_height = _get_object_height(object_name, simulator, task_env)
            half = 0.5 * object_height if object_height is not None else None
        if half is not None:
            move_target_loc_3d[2] += half

    ## place_upright pre-detection. Both require the held object to be in the upright
    ## category set; plain place additionally checks the current object/gripper geometry.
    will_dispatch_place_upright = False
    if rlpolicy is not None:
        try:
            place_upright_set = getattr(rlpolicy, "place_upright_objects", set())
            if _object_matches_category_set(simulator, task_env, object_name, place_upright_set):
                if has_orient:
                    will_dispatch_place_upright = True
                else:
                    held_pose = simulator.get_object_pose(task_env, object_name)
                    R_held = q2R_wxyz(np.asarray(held_pose[3:], dtype=np.float64))
                    ee_pose_cur = simulator.get_ee_pose(task_env)
                    R_ee_cur = q2R_wxyz(np.asarray(ee_pose_cur[3:7], dtype=np.float64))
                    tgt_thr = getattr(rlpolicy, "place_upright_tgt_cos_threshold", 0.7071)
                    grp_thr = getattr(rlpolicy, "place_upright_gripper_cos_threshold", 0.707)
                    if float(R_held[2, 2]) >= tgt_thr and abs(float(R_ee_cur[2, 2])) <= grp_thr:
                        will_dispatch_place_upright = True
        except Exception:
            will_dispatch_place_upright = False

    ## Hover clearance: base knob from the preset table (plan.clearance_key), with the
    ## runtime place_upright override (a geometry decision, not a static preset choice).
    hover_target_loc_3d = _hover_above(
        move_target_loc_3d,
        (simulator.place_offset_rlpolicy_upright
         if will_dispatch_place_upright
         else getattr(simulator, plan.clearance_key)),
        simulator,
    )

    ## Move held object to the hover loc (with the target orientation for object_align).
    move_kwargs = dict(
        data_collection_mode=data_collection_mode,
        predefined_target_loc_3d=hover_target_loc_3d,
        preserve_object_position_on_fallback=getattr(curobo, "place_preserve_object_position_on_fallback", True),
        planner_stats=planner_stats,
    )
    if has_orient:
        move_kwargs["predefined_target_quat"] = target_quat_candidates
        move_kwargs["disable_downward_yaw_fallback"] = True
        move_kwargs["candidate_align_meta"] = align_meta if target_quat_candidates else None
    else:
        move_kwargs["allow_object_yaw_fallback"] = will_dispatch_place_upright
    trajectory, terminate, term_reason = execute_move_held_object(
        node, child_node, simulator, task_env, curobo, rlpolicy, **move_kwargs,
    )
    if terminate:
        _write_planner_stats_to_node(child_node, planner_stats)
        return trajectory, terminate, term_reason

    grasped_names, _ = simulator.get_grasped_and_collided_objs(task_env)
    if len(grasped_names) > 0:
        task_name = "place"
        object_name = grasped_names[0]
        object_original_pose = simulator.get_object_pose(task_env, object_name)
        place_policy_agent_name = _resolve_place_policy_agent_name(
            object_name, simulator, task_env, rlpolicy
        )
        if place_policy_agent_name == "place_upright":
            trajectory_drop, terminate, term_reason = execute_drop(
                node,
                child_node,
                simulator,
                task_env,
                curobo,
                rlpolicy,
                data_collection_mode=data_collection_mode,
                planner_stats=planner_stats,
                retract_along_local_z=True,
                disable_downward_yaw_fallback=True,
            )
            _extend_step_data(trajectory, trajectory_drop, data_collection_mode)
            if terminate:
                _write_planner_stats_to_node(child_node, planner_stats)
                return trajectory, terminate, term_reason
            _write_planner_stats_to_node(child_node, planner_stats)
            return return_success(trajectory)

        ## Hold a few steps so physics settles before RL takes over.
        trajectory_pre_hold = _execute_place_hold(
            simulator=simulator,
            task_env=task_env,
            curobo=curobo,
            object_name=object_name,
            data_collection_mode=data_collection_mode,
        )
        _extend_step_data(trajectory, trajectory_pre_hold, data_collection_mode)

        ## Optional: align cup-body bottom (not AABB center) with the container center,
        ## using the goal orientation (target-aligned quat for object_align, else current).
        rl_goal_position = move_target_loc_3d
        target_asset = plan.target_object
        if (
            getattr(rlpolicy, "place_align_cup_bottom", False)
            and object_name in getattr(rlpolicy, "convex_objects", set())
            and target_asset != "table"
        ):
            goal_quat_for_cup = (
                target_quat_candidates[0]
                if (has_orient and target_quat_candidates)
                else object_original_pose[3:7]
            )
            delta_xy = _compute_cup_bottom_xy_offset_world(
                simulator, task_env, object_name, goal_quat_for_cup
            )
            if delta_xy is not None:
                rl_goal_position = rl_goal_position - delta_xy

        ## Execute place with the RL policy (shared helper for both variants).
        rl_kwargs = dict(
            data_collection_mode=data_collection_mode,
            trajectory=trajectory,
            goal_position=rl_goal_position,
            object_original_pose=object_original_pose,
            policy_agent_name=place_policy_agent_name,
        )
        if has_orient:
            rl_kwargs["object_original_height"] = _get_object_height(object_name, simulator, task_env)
        trajectory_rl, terminate, term_reason = execute_rl_policy(
            task_name, object_name, simulator, task_env, rlpolicy, **rl_kwargs,
        )
        _extend_step_data(trajectory, trajectory_rl, data_collection_mode)

        ## Release the object above the target.
        trajectory_drop, terminate, term_reason = execute_drop(
            node,
            child_node,
            simulator,
            task_env,
            curobo,
            rlpolicy,
            data_collection_mode=data_collection_mode,
            planner_stats=planner_stats,
        )
        _extend_step_data(trajectory, trajectory_drop, data_collection_mode)
        _dbg_release_state(simulator, task_env, f"after_drop(agent={place_policy_agent_name}, terminate={terminate})")
        if terminate:
            _write_planner_stats_to_node(child_node, planner_stats)
            return trajectory, terminate, term_reason

    _write_planner_stats_to_node(child_node, planner_stats)
    return return_success(trajectory)


def execute_rotate_held_object(
    node,
    child_node,
    simulator,
    task_env,
    curobo,
    rlpolicy=None,
    data_collection_mode=False,
    planner_stats=None,
):
    local_stats_created = planner_stats is None
    if local_stats_created:
        planner_stats = _init_planner_stats()

    grasped_names, _ = simulator.get_grasped_and_collided_objs(task_env)
    if len(grasped_names) == 0:
        if local_stats_created:
            _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure("No object is being held!")
    if (
        child_node.action_params["asset_id"] is None
        or child_node.action_params["asset"] is None
        or child_node.action_params["alignment"] is None
    ):
        if local_stats_created:
            _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure("Action_params incomplete!")

    held_object_name = grasped_names[0]
    target_object_name = child_node.action_params["asset"]
    alignment = child_node.action_params["alignment"]
    target_quat = simulator.get_rotation_from_alignment(
        task_env,
        held_object_name,
        target_object_name,
        alignment,
    )

    trajectory, terminate, term_reason = execute_move_held_object(
        node,
        child_node,
        simulator,
        task_env,
        curobo,
        rlpolicy,
        data_collection_mode=data_collection_mode,
        predefined_target_quat=target_quat,
        rotation_only=True,
        preserve_object_position_on_fallback=False,
        planner_stats=planner_stats,
    )
    if local_stats_created:
        _write_planner_stats_to_node(child_node, planner_stats)
    return trajectory, terminate, term_reason


def execute_rotate_held_object_rlpolicy(
    node,
    child_node,
    simulator,
    task_env,
    curobo,
    rlpolicy=None,
    data_collection_mode=False,
    planner_stats=None,
):
    local_stats_created = planner_stats is None
    if local_stats_created:
        planner_stats = _init_planner_stats()

    grasped_names, _ = simulator.get_grasped_and_collided_objs(task_env)
    if len(grasped_names) == 0:
        trajectory = None
        terminate = True
        term_reason = "No object is being held!"
        if local_stats_created:
            _write_planner_stats_to_node(child_node, planner_stats)
        return trajectory, terminate, term_reason

    if (
        child_node.action_params["asset_id"] is None
        or child_node.action_params["asset"] is None
        or child_node.action_params["alignment"] is None
    ):
        trajectory = None
        terminate = True
        term_reason = "Action_params incomplete!"
        if local_stats_created:
            _write_planner_stats_to_node(child_node, planner_stats)
        return trajectory, terminate, term_reason

    task_name = "pose"
    held_object_name = grasped_names[0]
    target_object_name = child_node.action_params["asset"]
    alignment = child_node.action_params["alignment"]
    object_center_3d = simulator.get_object_center_location(task_env, held_object_name)

    target_quat = simulator.get_rotation_from_alignment(
        task_env,
        held_object_name,
        target_object_name,
        alignment,
    )

    policy_obs, _ = simulator.reset(
        env=task_env,
        reset_level="task",
        task_name=task_name,
        object_name=held_object_name,
        goal_pose=np.concatenate([np.asarray(object_center_3d), np.asarray(target_quat)]),
    )
    if isinstance(policy_obs, dict):
        policy_obs = policy_obs["obs"]
    rlpolicy.reset_agent_new_episode(policy_obs, agent_name=task_name)

    trajectory = _init_step_data_buffer(data_collection_mode)
    while True:
        action = rlpolicy.get_action(policy_obs, agent_name=task_name)
        pre_step_data = _collect_pre_step_data(
            simulator,
            task_env,
            action,
            action_mode=ACTION_MODE_DELTA_EEPOSE,
            data_collection_mode=data_collection_mode,
        )
        policy_obs, _, dones, _ = simulator.step(task_env, action, action_mode=ACTION_MODE_DELTA_EEPOSE)
        rlpolicy.post_action_operation(dones, agent_name=task_name)
        _append_step_data(
            trajectory,
            simulator,
            task_env,
            action,
            data_collection_mode,
            pre_step_data=pre_step_data,
        )
        if dones.sum() > 0:
            break
    simulator.unset(task_env)

    terminate = False
    term_reason = "Success!"
    if local_stats_created:
        _write_planner_stats_to_node(child_node, planner_stats)
    return trajectory, terminate, term_reason





# =====================================================================================
# Unified place engine entry point. See docs/place_pipeline_rfc.md and skills/place_plan.py.
# Registry -> execute_place_skill (build the PlacePlan) -> execute_place (dispatch on the
# finish/release mode to the shared family helpers above).
# =====================================================================================

def execute_place(
    plan,
    node,
    child_node,
    simulator,
    task_env,
    curobo,
    rlpolicy=None,
    data_collection_mode=False,
):
    """Unified place engine. Consumes a PlacePlan (skills/place_plan.py).

    Top-level branch is the finish regime:
      - RELEASE_RL      -> _place_rl_finish (the unified RL family; place / place_with_orientation).
      - drop/lower      -> _place_drop_finish (the unified drop family; covers place_with_drop,
                           place_with_orientation_drop, place_with_gripper_orientation_drop).
    """
    if plan.release_mode == RELEASE_RL:
        return _place_rl_finish(
            plan,
            node,
            child_node,
            simulator,
            task_env,
            curobo,
            rlpolicy=rlpolicy,
            data_collection_mode=data_collection_mode,
        )
    return _place_drop_finish(
        plan,
        node,
        child_node,
        simulator,
        task_env,
        curobo,
        rlpolicy=rlpolicy,
        data_collection_mode=data_collection_mode,
    )


def execute_place_skill(
    skill,
    node,
    child_node,
    simulator,
    task_env,
    curobo,
    rlpolicy=None,
    data_collection_mode=False,
):
    """Registry entry for the place family: build the PlacePlan for ``skill`` then run the
    engine. Centralizes the per-skill "Action_params incomplete!" validation in
    build_place_plan(). Bound per skill via functools.partial in the MCTS registry.
    """
    plan, err = build_place_plan(skill, child_node.action_params)
    if err is not None:
        planner_stats = _init_planner_stats()
        _write_planner_stats_to_node(child_node, planner_stats)
        return return_failure(err)
    return execute_place(
        plan,
        node,
        child_node,
        simulator,
        task_env,
        curobo,
        rlpolicy,
        data_collection_mode=data_collection_mode,
    )
