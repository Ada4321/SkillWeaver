"""Motion-planner execution primitives (mode-1 replan + joint stepping) and planner-stats.

Leaf module: imports only utils + skills._constants; never imports skills.skills, so
skills.skills can import these back without a cycle. Simulator/curobo are passed in as args.
"""
import logging
import time
import numpy as np

from skills._constants import ACTION_MODE_JOINT_ANGLES, ACTION_MODE_DELTA_EEPOSE
from utils.data_collect_utils import _init_step_data_buffer, _collect_pre_step_data, _append_step_data
from utils.geometry_utils import _quat_normalize_wxyz, _quat_angle_distance_rad_wxyz


def _log_timing(msg: str):
    pass


def _init_planner_stats():
    return {
        "planner_chunk_count": 0,
        "planner_split_count": 0,
    }


def _record_planner_chunks(planner_stats, chunk_count: int):
    if planner_stats is None:
        return
    n_chunks = int(chunk_count)
    planner_stats["planner_chunk_count"] = int(planner_stats.get("planner_chunk_count", 0)) + n_chunks
    if n_chunks > 1:
        planner_stats["planner_split_count"] = int(planner_stats.get("planner_split_count", 0)) + 1


# _quat_normalize_wxyz / _quat_angle_distance_rad_wxyz moved to utils.geometry_utils (imported above).


def _get_mode1_replan_cfg(curobo):
    k_steps = max(1, int(getattr(curobo, "planner_replan_every_k_steps", 5) or 1))
    return {
        "enabled": bool(getattr(curobo, "planner_execute_mode1_replan", False)),
        "k_steps": k_steps,
        "pos_tol_m": max(0.0, float(getattr(curobo, "planner_goal_pos_tolerance_m", 0.01) or 0.0)),
        "rot_tol_rad": max(0.0, float(getattr(curobo, "planner_goal_rot_tolerance_rad", 0.10) or 0.0)),
        "max_replans": max(0, int(getattr(curobo, "planner_max_replans", 5) or 0)),
        "skip_points_after_first_plan": max(
            0, int(getattr(curobo, "planner_skip_points_after_first_plan", 0) or 0)
        ),
    }


def _get_current_joint_targets(simulator, task_env):
    arm_q = np.asarray(simulator.get_joint_positions(task_env), dtype=np.float32).reshape(-1)
    gripper_q = np.asarray(simulator.get_gripper_positions(task_env), dtype=np.float32).reshape(-1)
    return np.concatenate([arm_q, gripper_q], axis=0)


def _get_ee_goal_errors(simulator, task_env, goal_pose):
    if goal_pose is None:
        return None, None
    get_ee_pose_fn = getattr(simulator, "get_ee_pose", None)
    if not callable(get_ee_pose_fn):
        raise RuntimeError("Simulator must provide get_ee_pose for mode=1 planner replan execution.")

    ee_pose = np.asarray(get_ee_pose_fn(task_env), dtype=np.float64).reshape(-1)
    goal = np.asarray(goal_pose, dtype=np.float64).reshape(-1)
    if ee_pose.size < 7 or goal.size < 7:
        raise RuntimeError(f"Invalid ee pose dimensions for goal check: ee={ee_pose.size}, goal={goal.size}")

    pos_err = float(np.linalg.norm(ee_pose[:3] - goal[:3]))
    rot_err = _quat_angle_distance_rad_wxyz(ee_pose[3:7], goal[3:7])
    return pos_err, rot_err


def _goal_pose_reached(pos_err, rot_err, pos_tol_m: float, rot_tol_rad: float) -> bool:
    if pos_err is None or rot_err is None:
        return False
    return (float(pos_err) <= float(pos_tol_m)) and (float(rot_err) <= float(rot_tol_rad))


def _skip_plan_prefix_points(joint_states, n_skip: int):
    arr = np.asarray(joint_states, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] <= 0:
        return arr
    n_skip = max(0, int(n_skip))
    if n_skip <= 0:
        return arr
    if n_skip >= arr.shape[0]:
        return arr[-1:, :]
    return arr[n_skip:, :]


def _step_joint_action_with_mode1_substeps(
    simulator,
    task_env,
    action,
    trajectory,
    data_collection_mode: bool,
    joint_prev,
    max_steps: int | None = None,
    force_gripper_delta: float | None = None,
):
    sub_actions = [np.asarray(action, dtype=np.float32)]
    split_fn = getattr(simulator, "split_motion_planning_action", None)
    if not callable(split_fn):
        raise RuntimeError("Simulator must provide split_motion_planning_action for planner execution.")
    split_actions = split_fn(task_env, action)
    if split_actions:
        sub_actions = [np.asarray(a, dtype=np.float32) for a in split_actions]

    convert_fn = getattr(simulator, "joint_target_to_mode1_action", None)
    if not callable(convert_fn):
        raise RuntimeError("Simulator must provide joint_target_to_mode1_action for mode=1 planner execution.")

    joint_prev = np.asarray(joint_prev, dtype=np.float32)
    executed_chunks = 0
    _curobo_step_times = []
    for sub_action in sub_actions:
        if max_steps is not None and executed_chunks >= int(max_steps):
            break
        mode1_action = np.asarray(
            convert_fn(task_env, joint_start=joint_prev, joint_target=sub_action, clamp=True),
            dtype=np.float32,
        )
        if force_gripper_delta is not None and mode1_action.size >= 7:
            mode1_action = mode1_action.copy()
            mode1_action[-1] = float(force_gripper_delta)
        pre_step_data = _collect_pre_step_data(
            simulator,
            task_env,
            mode1_action,
            action_mode=ACTION_MODE_DELTA_EEPOSE,
            data_collection_mode=data_collection_mode,
            disable_action_ema=True,
        )
        _t0 = time.perf_counter()
        simulator.step(task_env, mode1_action, action_mode=ACTION_MODE_DELTA_EEPOSE, disable_action_ema=True)
        _curobo_step_times.append(time.perf_counter() - _t0)
        _append_step_data(
            trajectory,
            simulator,
            task_env,
            mode1_action,
            data_collection_mode,
            pre_step_data=pre_step_data,
        )
        # Recompute from simulator state: mode=1 execution may not exactly hit the planned sub_action.
        joint_prev = _get_current_joint_targets(simulator, task_env)
        executed_chunks += 1

    completed = executed_chunks == len(sub_actions)
    if _curobo_step_times:
        _n = len(_curobo_step_times)
        _log_timing(f"[timing] curobo_step n={_n} mean_s={sum(_curobo_step_times)/_n:.4f} total_s={sum(_curobo_step_times):.3f}")
    return executed_chunks, joint_prev, completed


def _execute_planner_joint_states_mode1(
    simulator,
    task_env,
    joint_states,
    trajectory,
    data_collection_mode: bool,
    planner_stats,
    goal_pose,
    k_steps: int,
    pos_tol_m: float,
    rot_tol_rad: float,
    force_gripper_delta: float | None = None,
):
    # If planner goal is unavailable, execute the whole plan in mode=1 once (no error-gated replan).
    goal_check_enabled = goal_pose is not None
    joint_prev = _get_current_joint_targets(simulator, task_env)
    steps_since_last_check = 0
    last_pos_err = None
    last_rot_err = None

    joint_states = np.asarray(joint_states, dtype=np.float32)
    for action in joint_states:
        remaining = None
        if goal_check_enabled:
            remaining = max(0, int(k_steps) - int(steps_since_last_check))
            if remaining == 0:
                last_pos_err, last_rot_err = _get_ee_goal_errors(simulator, task_env, goal_pose)
                if _goal_pose_reached(last_pos_err, last_rot_err, pos_tol_m=pos_tol_m, rot_tol_rad=rot_tol_rad):
                    return True, False, last_pos_err, last_rot_err
                return False, True, last_pos_err, last_rot_err

        n_chunks, joint_prev, _ = _step_joint_action_with_mode1_substeps(
            simulator=simulator,
            task_env=task_env,
            action=action,
            trajectory=trajectory,
            data_collection_mode=data_collection_mode,
            joint_prev=joint_prev,
            max_steps=remaining,
            force_gripper_delta=force_gripper_delta,
        )
        _record_planner_chunks(planner_stats, n_chunks)
        steps_since_last_check += int(n_chunks)

        if goal_check_enabled and steps_since_last_check >= int(k_steps):
            last_pos_err, last_rot_err = _get_ee_goal_errors(simulator, task_env, goal_pose)
            if _goal_pose_reached(last_pos_err, last_rot_err, pos_tol_m=pos_tol_m, rot_tol_rad=rot_tol_rad):
                return True, False, last_pos_err, last_rot_err
            return False, True, last_pos_err, last_rot_err

    if not goal_check_enabled:
        return True, False, None, None

    last_pos_err, last_rot_err = _get_ee_goal_errors(simulator, task_env, goal_pose)
    if _goal_pose_reached(last_pos_err, last_rot_err, pos_tol_m=pos_tol_m, rot_tol_rad=rot_tol_rad):
        return True, False, last_pos_err, last_rot_err
    return False, True, last_pos_err, last_rot_err


def _write_planner_stats_to_node(child_node, planner_stats):
    if child_node is None:
        return
    if planner_stats is None:
        child_node.planner_chunk_count = 0
        child_node.planner_split_count = 0
        return
    child_node.planner_chunk_count = int(planner_stats.get("planner_chunk_count", 0))
    child_node.planner_split_count = int(planner_stats.get("planner_split_count", 0))


def _get_current_motion_planning_target(simulator, task_env) -> np.ndarray:
    current_q = np.asarray(simulator.get_joint_positions(task_env), dtype=np.float32).reshape(-1)
    get_gripper_fn = getattr(simulator, "get_gripper_positions", None)
    if not callable(get_gripper_fn):
        return current_q
    current_gripper = np.asarray(get_gripper_fn(task_env), dtype=np.float32).reshape(-1)
    return np.concatenate([current_q, current_gripper], axis=0)


def _get_current_release_joint_target(simulator, task_env) -> np.ndarray:
    joint_target = _get_current_motion_planning_target(simulator, task_env)
    if joint_target.shape[0] < 2:
        return joint_target
    max_opening = None
    get_max_fn = getattr(simulator, "get_max_gripper_opening", None)
    if callable(get_max_fn):
        max_opening = get_max_fn(task_env)
    if max_opening is None:
        max_opening = 0.04
    joint_target[-2:] = float(max_opening)
    return joint_target.astype(np.float32, copy=False)


def _step_release_at_current_pose(
    simulator,
    task_env,
    trajectory,
    data_collection_mode: bool,
    num_steps: int = 3,
) -> int:
    num_steps = max(1, int(num_steps))
    try:
        base_env = simulator._unwrap_env(task_env)
        exec_mode = "teleop" if bool(getattr(base_env.cfg, "release_teleop_joint_state", False)) else "PD"
    except Exception:
        exec_mode = "?"
    logging.warning(
        "[RELEASE_IN_PLACE] %s open gripper at current pose for %d steps",
        exec_mode,
        num_steps,
    )
    for step_idx in range(num_steps):
        action = _get_current_release_joint_target(simulator, task_env)
        logging.warning(
            "[RELEASE_IN_PLACE] step=%d/%d gripper_target=%s",
            step_idx + 1,
            num_steps,
            np.asarray(action[-2:], dtype=float).tolist() if action.shape[0] >= 2 else None,
        )
        pre_step_data = _collect_pre_step_data(
            simulator,
            task_env,
            action,
            action_mode=ACTION_MODE_JOINT_ANGLES,
            data_collection_mode=data_collection_mode,
        )
        simulator.step(task_env, action, action_mode=ACTION_MODE_JOINT_ANGLES, is_release=True)
        _append_step_data(
            trajectory,
            simulator,
            task_env,
            action,
            data_collection_mode,
            pre_step_data=pre_step_data,
        )
    return num_steps


def _step_joint_action_with_substeps(
    simulator,
    task_env,
    action,
    trajectory,
    data_collection_mode: bool,
    is_release: bool = False,
):
    sub_actions = [np.asarray(action, dtype=np.float32)]
    split_fn = getattr(simulator, "split_motion_planning_action", None)
    if not callable(split_fn):
        raise RuntimeError("Simulator must provide split_motion_planning_action for mode=0 sub-step execution.")
    split_actions = split_fn(task_env, action)
    if split_actions:
        sub_actions = [np.asarray(a, dtype=np.float32) for a in split_actions]

    for sub_action in sub_actions:
        pre_step_data = _collect_pre_step_data(
            simulator,
            task_env,
            sub_action,
            action_mode=ACTION_MODE_JOINT_ANGLES,
            data_collection_mode=data_collection_mode,
        )
        simulator.step(task_env, sub_action, action_mode=ACTION_MODE_JOINT_ANGLES, is_release=is_release)
        _append_step_data(
            trajectory,
            simulator,
            task_env,
            sub_action,
            data_collection_mode,
            pre_step_data=pre_step_data,
        )
    return len(sub_actions)

