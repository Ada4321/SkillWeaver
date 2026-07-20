"""Data-collection helpers (per-step data buffer shaping for VLA trajectory recording).

Quaternion algebra that used to live here moved to utils.geometry_utils.
The per-step data helpers are added below (moved from skills/skills.py).
"""
def _init_step_data_buffer(data_collection_mode: bool):
    return [] if data_collection_mode else None


def _collect_pre_step_data(
    simulator,
    task_env,
    action,
    action_mode: int,
    data_collection_mode: bool,
    disable_action_ema: bool = False,
):
    if not data_collection_mode:
        return None
    collect_pre_fn = getattr(simulator, "collect_step_data_pre_action", None)
    if not callable(collect_pre_fn):
        raise RuntimeError("Simulator must provide collect_step_data_pre_action for data collection.")
    return collect_pre_fn(
        task_env,
        action,
        action_mode=action_mode,
        disable_action_ema=bool(disable_action_ema),
    )


def _append_step_data(step_data_buffer, simulator, task_env, action, data_collection_mode: bool, pre_step_data=None):
    if data_collection_mode and step_data_buffer is not None:
        if pre_step_data is None:
            raise RuntimeError("pre_step_data is required in data_collection_mode.")
        step_data = pre_step_data
        collect_post_fn = getattr(simulator, "collect_step_data_post_action", None)
        if callable(collect_post_fn):
            post_step_data = collect_post_fn(task_env, pre_step_data=pre_step_data)
            if isinstance(post_step_data, dict):
                step_data = dict(pre_step_data)
                for key, value in post_step_data.items():
                    if key in step_data and isinstance(step_data[key], dict) and isinstance(value, dict):
                        merged = dict(step_data[key])
                        merged.update(value)
                        step_data[key] = merged
                    else:
                        step_data[key] = value
        step_data_buffer.append(step_data)


def _extend_step_data(dst, src, data_collection_mode: bool):
    if not data_collection_mode or dst is None or src is None:
        return
    dst.extend(src)
