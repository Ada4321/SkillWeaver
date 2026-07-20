"""Grasp/place "hold" execution: settle the object for a few steps before RL takes over.

Leaf module (imports utils + skills._constants only; never skills.skills)."""
import numpy as np

from skills._constants import ACTION_MODE_DELTA_EEPOSE
from utils.data_collect_utils import _init_step_data_buffer, _collect_pre_step_data, _append_step_data


def _get_grasp_hold_cfg(curobo):
    min_steps = max(0, int(getattr(curobo, "min_grasp_hold_steps", 0) or 0))
    max_steps_raw = getattr(curobo, "max_grasp_hold_steps", min_steps)
    max_steps = max(min_steps, int(max_steps_raw if max_steps_raw is not None else min_steps))
    lin_vel_thresh = max(0.0, float(getattr(curobo, "grasp_hold_lin_vel_thresh", 0.0) or 0.0))
    ang_vel_thresh = max(0.0, float(getattr(curobo, "grasp_hold_ang_vel_thresh", 0.0) or 0.0))
    disable_ema = bool(getattr(curobo, "grasp_hold_disable_ema", True))
    return {
        "min_steps": min_steps,
        "max_steps": max_steps,
        "lin_vel_thresh": lin_vel_thresh,
        "ang_vel_thresh": ang_vel_thresh,
        "disable_ema": disable_ema,
    }


def _get_place_hold_cfg(curobo):
    min_steps = max(0, int(getattr(curobo, "min_place_hold_steps", 0) or 0))
    max_steps_raw = getattr(curobo, "max_place_hold_steps", min_steps)
    max_steps = max(min_steps, int(max_steps_raw if max_steps_raw is not None else min_steps))
    lin_vel_thresh = max(0.0, float(getattr(curobo, "place_hold_lin_vel_thresh", 0.0) or 0.0))
    ang_vel_thresh = max(0.0, float(getattr(curobo, "place_hold_ang_vel_thresh", 0.0) or 0.0))
    disable_ema = bool(getattr(curobo, "place_hold_disable_ema", True))
    return {
        "min_steps": min_steps,
        "max_steps": max_steps,
        "lin_vel_thresh": lin_vel_thresh,
        "ang_vel_thresh": ang_vel_thresh,
        "disable_ema": disable_ema,
    }


def _extract_root_velocity_norms(root_velocity):
    linear = None
    angular = None
    if isinstance(root_velocity, dict):
        linear = root_velocity.get("linear", None)
        angular = root_velocity.get("angular", None)
    elif isinstance(root_velocity, (tuple, list)) and len(root_velocity) >= 2:
        linear = root_velocity[0]
        angular = root_velocity[1]
    else:
        vec = np.asarray(root_velocity, dtype=np.float64).reshape(-1)
        if vec.size >= 6:
            linear = vec[:3]
            angular = vec[3:6]
    if linear is None or angular is None:
        return None, None
    linear = np.asarray(linear, dtype=np.float64).reshape(-1)
    angular = np.asarray(angular, dtype=np.float64).reshape(-1)
    if linear.size < 3 or angular.size < 3:
        return None, None
    lin_norm = float(np.linalg.norm(linear[:3]))
    ang_norm = float(np.linalg.norm(angular[:3]))
    return lin_norm, ang_norm


def _execute_place_hold(
    simulator,
    task_env,
    curobo,
    object_name: str,
    data_collection_mode: bool,
):
    cfg = _get_place_hold_cfg(curobo)
    min_steps = int(cfg["min_steps"])
    max_steps = int(cfg["max_steps"])
    if max_steps <= 0:
        return _init_step_data_buffer(data_collection_mode)

    hold_action = np.zeros((7,), dtype=np.float32)
    hold_action[-1] = -1.0
    trajectory_hold = _init_step_data_buffer(data_collection_mode)

    get_obj_vel_fn = getattr(simulator, "get_object_root_velocity", None)
    can_check_velocity = callable(get_obj_vel_fn) and object_name is not None

    for step_idx in range(max_steps):
        pre_step_data = _collect_pre_step_data(
            simulator,
            task_env,
            hold_action,
            action_mode=ACTION_MODE_DELTA_EEPOSE,
            data_collection_mode=data_collection_mode,
            disable_action_ema=bool(cfg["disable_ema"]),
        )
        simulator.step(
            task_env,
            hold_action,
            action_mode=ACTION_MODE_DELTA_EEPOSE,
            disable_action_ema=bool(cfg["disable_ema"]),
        )
        _append_step_data(
            trajectory_hold,
            simulator,
            task_env,
            hold_action,
            data_collection_mode,
            pre_step_data=pre_step_data,
        )

        steps_done = step_idx + 1
        if steps_done < min_steps:
            continue

        if not can_check_velocity:
            break

        root_velocity = get_obj_vel_fn(task_env, object_name)
        lin_norm, ang_norm = _extract_root_velocity_norms(root_velocity)
        if lin_norm is None or ang_norm is None:
            break
        if lin_norm <= float(cfg["lin_vel_thresh"]) and ang_norm <= float(cfg["ang_vel_thresh"]):
            break

    return trajectory_hold


def _execute_grasp_hold(
    simulator,
    task_env,
    curobo,
    object_name: str,
    data_collection_mode: bool,
):
    cfg = _get_grasp_hold_cfg(curobo)
    min_steps = int(cfg["min_steps"])
    max_steps = int(cfg["max_steps"])
    if max_steps <= 0:
        return _init_step_data_buffer(data_collection_mode)

    hold_action = np.zeros((7,), dtype=np.float32)
    hold_action[-1] = -1.0
    trajectory_hold = _init_step_data_buffer(data_collection_mode)

    get_obj_vel_fn = getattr(simulator, "get_object_root_velocity", None)
    can_check_velocity = callable(get_obj_vel_fn) and object_name is not None

    for step_idx in range(max_steps):
        pre_step_data = _collect_pre_step_data(
            simulator,
            task_env,
            hold_action,
            action_mode=ACTION_MODE_DELTA_EEPOSE,
            data_collection_mode=data_collection_mode,
            disable_action_ema=bool(cfg["disable_ema"]),
        )
        simulator.step(
            task_env,
            hold_action,
            action_mode=ACTION_MODE_DELTA_EEPOSE,
            disable_action_ema=bool(cfg["disable_ema"]),
        )
        _append_step_data(
            trajectory_hold,
            simulator,
            task_env,
            hold_action,
            data_collection_mode,
            pre_step_data=pre_step_data,
        )

        steps_done = step_idx + 1
        if steps_done < min_steps:
            continue

        if not can_check_velocity:
            break

        root_velocity = get_obj_vel_fn(task_env, object_name)
        lin_norm, ang_norm = _extract_root_velocity_norms(root_velocity)
        if lin_norm is None or ang_norm is None:
            break
        if lin_norm <= float(cfg["lin_vel_thresh"]) and ang_norm <= float(cfg["ang_vel_thresh"]):
            break

    return trajectory_hold


