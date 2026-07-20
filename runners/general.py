import os
import gc
import time
import numpy as np
import json
import logging
import random
from PIL import Image
import multiprocessing as mp
import sys
from pathlib import Path
# import shutil

from skills.gemini import GeminiVLMServer
from tree_search.mcts import MonteCarloTreeSearch
from tree_search.tasks.task_templates import (
    BaseTaskTemplate,
)
from simulators import init_simulator


def setup_main_logging(queue: mp.Queue, level=logging.ERROR):
    fmt = logging.Formatter("[%(asctime)s %(processName)s %(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    listener = logging.handlers.QueueListener(queue, sh, respect_handler_level=True)
    listener.start()
    return listener


def setup_worker_logging(queue: mp.Queue, level=logging.ERROR):
    qh = logging.handlers.QueueHandler(queue)
    qh.setLevel(level)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(qh)
    root.setLevel(level)
    logging.captureWarnings(False)


def writer_loop(io_queue: mp.Queue):
    import imageio.v2 as iio
    logger = logging.getLogger("writer")

    try:
        while True:
            task = io_queue.get()
            if task is None:
                break

            t = task.get("type")
            if t == "write_json":
                path = task["path"]; data = task["data"]
                # os.makedirs(os.path.dirname(path), exist_ok=True)
                try:
                    tmp = path + ".tmp"
                    with open(tmp, "w") as fp:
                        json.dump(data, fp, indent=4)
                    os.replace(tmp, path)
                except Exception:
                    logger.exception(f"write_json failed: {path}")

            elif t == "write_gif":
                path = task["path"]
                frames = task["frames"]        # List[np.ndarray] (H,W,3) uint8
                duration = int(task.get("duration", 100))
                loop = int(task.get("loop", 0))
                # os.makedirs(os.path.dirname(path), exist_ok=True)
                try:
                    if not frames:
                        continue
                    images = [Image.fromarray(f) for f in frames]
                    images[0].save(
                        path,
                        save_all=True,
                        append_images=images[1:],
                        duration=duration,
                        loop=loop
                    )
                except Exception:
                    logger.exception(f"write_gif failed: {path}")

            elif t == "write_video":
                # mp4
                path   = task["path"]
                frames = task["frames"]
                fps    = int(task.get("fps", 20))
                codec  = task.get("codec", "libx264")
                crf    = task.get("crf", 23)
                preset = task.get("preset", "veryfast")
                pix_fmt= task.get("pix_fmt", "yuv420p")

                try:
                    writer = iio.get_writer(
                        path,
                        format="FFMPEG",
                        mode="I",
                        fps=fps,
                        codec=codec,
                    )
                    for fr in frames:
                        writer.append_data(fr)
                    writer.close()
                except Exception:
                    logger.exception(f"write_video failed: {path}")

            else:
                logger.warning(f"unknown io task: {task}")
    finally:
        pass


def _build_tree_searcher_in_child(cfg):
    sim_cfg = cfg["simulator"]
    simulator = init_simulator(sim_cfg, env_cfg=cfg.get("env"))
    skills_cfg = cfg["skills"]
    rl_cfg = skills_cfg["rl"]
    rl_backend = str(rl_cfg.get("backend", "local")).lower()

    if rl_backend == "server":
        from skills.rl_policy_client import RLPolicyClient

        server_cfg = skills_cfg.get("server", {}).get("rl_policy", {})
        host = str(server_cfg.get("host", "127.0.0.1"))
        port = int(server_cfg.get("port", 9080))
        timeout_s = float(server_cfg.get("timeout_s", 60.0))
        rlpolicy = RLPolicyClient(
            {
            "url": f"http://{host}:{port}",
            "timeout_s": timeout_s,
            "rl_cfg": rl_cfg,
            }
        )
    elif rl_backend == "local":
        # Import RLPolicy after simulator initialization so Isaac Lab's AppLauncher has run
        from skills.rl_policy import RLPolicy

        agent_cfg_file = simulator.agent_cfg_file
        policy_args = {
            "num_envs": getattr(simulator.args, "num_envs", 1),
            "agent_cfg_file": agent_cfg_file,
        }
        sim_device = getattr(simulator.args, "device", None)
        if sim_device:
            policy_args["device"] = sim_device
        if rl_cfg.get("pick_ckpt"):
            policy_args["pick_ckpt"] = rl_cfg["pick_ckpt"]
        if rl_cfg.get("pick_upright_ckpt"):
            policy_args["pick_upright_ckpt"] = rl_cfg["pick_upright_ckpt"]
        if rl_cfg.get("pick_convex_ckpt"):
            policy_args["pick_convex_ckpt"] = rl_cfg["pick_convex_ckpt"]
        if rl_cfg.get("pick_convex_lying_ckpt"):
            policy_args["pick_convex_lying_ckpt"] = rl_cfg["pick_convex_lying_ckpt"]
        if rl_cfg.get("pick_handle_ckpt"):
            policy_args["pick_handle_ckpt"] = rl_cfg["pick_handle_ckpt"]
        if rl_cfg.get("upright_list"):
            policy_args["upright_list"] = rl_cfg["upright_list"]
        if rl_cfg.get("convex_list"):
            policy_args["convex_list"] = rl_cfg["convex_list"]
        if rl_cfg.get("handle_list"):
            policy_args["handle_list"] = rl_cfg["handle_list"]
        if rl_cfg.get("handle_pcd_agents") is not None:
            policy_args["handle_pcd_agents"] = rl_cfg["handle_pcd_agents"]
        policy_args["place_align_cup_bottom"] = bool(rl_cfg.get("place_align_cup_bottom", False))
        if rl_cfg.get("place_ckpt"):
            policy_args["place_ckpt"] = rl_cfg["place_ckpt"]
        if rl_cfg.get("place_upright_ckpt"):
            policy_args["place_upright_ckpt"] = rl_cfg["place_upright_ckpt"]
        if rl_cfg.get("place_upright_list"):
            policy_args["place_upright_list"] = rl_cfg["place_upright_list"]
        policy_args["place_upright_tgt_cos_threshold"] = float(
            rl_cfg.get("place_upright_tgt_cos_threshold", 0.866)
        )
        policy_args["place_upright_gripper_cos_threshold"] = float(
            rl_cfg.get("place_upright_gripper_cos_threshold", 0.707)
        )
        if rl_cfg.get("pose_ckpt"):
            policy_args["pose_ckpt"] = rl_cfg["pose_ckpt"]
        if rl_cfg.get("open_drawer_ckpt"):
            policy_args["open_drawer_ckpt"] = rl_cfg["open_drawer_ckpt"]
        if rl_cfg.get("close_drawer_ckpt"):
            policy_args["close_drawer_ckpt"] = rl_cfg["close_drawer_ckpt"]
        if rl_cfg.get("close_door_ckpt"):
            policy_args["close_door_ckpt"] = rl_cfg["close_door_ckpt"]
        rlpolicy = RLPolicy(policy_args)
    else:
        raise ValueError(f"Unsupported skills.rl.backend: {rl_backend}")

    vlm_provider = skills_cfg["vlm"].get("provider", "gemini")
    if vlm_provider == "claude_bridge":
        from skills.claude_bridge import ClaudeAgentBridgeVLM

        vlm = ClaudeAgentBridgeVLM(skills_cfg["vlm"])
    elif vlm_provider == "gemini":
        vlm = GeminiVLMServer(skills_cfg["vlm"])
    else:
        raise ValueError(f"Unsupported skills.vlm.provider: {vlm_provider}")
    from skills.cuRobo import cuRoboServer

    curobo_server = cuRoboServer(simulator=simulator, args=skills_cfg["curobo"])

    logging.warning("Init Searcher!")
    mcts_target = cfg.get("mcts", {}).get("_target_", None)
    if mcts_target:
        import importlib
        module_path, cls_name = mcts_target.rsplit(".", 1)
        searcher_cls = getattr(importlib.import_module(module_path), cls_name)
        logging.warning("Using searcher class from mcts._target_: %s", mcts_target)
    else:
        searcher_cls = MonteCarloTreeSearch
    ts = searcher_cls(
        cfg=cfg,
        # Skills
        vlm=vlm,
        curobo_server=curobo_server,
        rlpolicy=rlpolicy,
        # Simulator
        simulator=simulator
    )
    return ts


def _load_surface_tgt_list(surface_tgt_list_path: str | None, scene_dir: Path) -> list[str]:
    path = surface_tgt_list_path
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def iter_scene_tasks(
    scene_desc_path: str,
    task_filter=None,
    eval_num: int = 1,
    surface_tgt_list_path: str | None = None,
    force_upright: bool = False,
    upright_random_prob: float = 0.5,
    include_task_entry: bool = False,
    task_key: str = "tasks",
):
    """Return flattened task list.

    Default tuple: (task_type, task_idx, task_variations, layout_idx)
    If include_task_entry=True: (task_type, task_idx, task_variations, raw_task_entry, layout_idx)

    layout_idx is None for the legacy `dict[task_type, list]` format. For the new
    `{"tasks_by_layout": [task_map_0, task_map_1, ...]}` format, each task entry
    is emitted once per layout, tagged with that layout's index.
    """
    scene_desc_path = Path(scene_desc_path)
    with open(scene_desc_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    tasks_path = scene_data.get(task_key)
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # Normalize into list of (layout_idx_or_None, task_map) so the loop below
    # is shared between legacy and new formats.
    if isinstance(tasks, dict) and "tasks_by_layout" in tasks:
        layouts = tasks["tasks_by_layout"]
        if not isinstance(layouts, list):
            raise ValueError(
                f"{tasks_path}: 'tasks_by_layout' must be a list of task maps."
            )
        per_layout = [(i, layouts[i]) for i in range(len(layouts))]
    else:
        per_layout = [(None, tasks)]

    if task_filter:
        task_filter = set(task_filter)

    sample_count = max(1, int(eval_num))
    surface_tgt_list = _load_surface_tgt_list(surface_tgt_list_path, scene_desc_path.parent)
    task_sampler = BaseTaskTemplate(
        surface_tgt_list,
        force_upright=force_upright,
        upright_random_prob=upright_random_prob,
    )

    out = []
    for layout_idx, task_map in per_layout:
        for task_type, task_list in task_map.items():
            if task_filter and task_type not in task_filter:
                continue
            for idx, task_entry in enumerate(task_list):
                variations = [task_sampler.sample(task_entry) for _ in range(sample_count)]
                if include_task_entry:
                    out.append((task_type, idx, variations, task_entry, layout_idx))
                else:
                    out.append((task_type, idx, variations, layout_idx))
    return out


def _cleanup_after_episode(tree_searcher):
    # Clear transient MCTS caches so per-episode memory does not accumulate.
    try:
        if hasattr(tree_searcher, "partial_plan_cache"):
            cache = getattr(tree_searcher, "partial_plan_cache")
            if hasattr(cache, "clear"):
                cache.clear()
    except Exception:
        logging.exception("Episode cleanup: failed to clear partial_plan_cache.")
    try:
        if hasattr(tree_searcher, "root"):
            tree_searcher.root = None
    except Exception:
        logging.exception("Episode cleanup: failed to clear tree_searcher.root.")

    # Explicitly reclaim Python and CUDA cache memory between episodes.
    try:
        gc.collect()
    except Exception:
        logging.exception("Episode cleanup: gc.collect failed.")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        logging.exception("Episode cleanup: torch CUDA cleanup failed.")


def run_and_save_one_episode(
    tree_searcher,
    task_env,
    out_dir,
    save_tag,
    use_mcts=False,
    io_queue=None,
    wid=0,
    layout_idx: int | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    # Pin the env's layout for this episode when the task carries one. Set it on the
    # underlying BaseEnv (which `_sample_layout_idx` reads); the IsaacLab simulator
    # interface object does not own this attribute.
    layout_pin_target = (
        task_env.unwrapped if (task_env is not None and hasattr(task_env, "unwrapped")) else task_env
    )
    prev_force = getattr(layout_pin_target, "force_layout_idx", None) if layout_pin_target is not None else None
    if layout_pin_target is not None:
        layout_pin_target.force_layout_idx = layout_idx
    run_cfg = (getattr(tree_searcher, "cfg", {}) or {}).get("run", {})
    save_node_videos = bool(run_cfg.get("save_node_videos", True))
    mcts_cfg = getattr(tree_searcher, "mcts_cfg", {}) or {}
    save_restore_augmented_traj = bool(mcts_cfg.get("save_restore_augmented_traj", False))
    save_original_also_when_augmented = bool(mcts_cfg.get("save_original_also_when_augmented", False))
    if not save_restore_augmented_traj:
        save_original_also_when_augmented = False

    def _stack_over_time(seq):
        """Recursively stack a list of step dicts/arrays over time axis."""
        first = seq[0]
        if isinstance(first, dict):
            out = {}
            for k in first.keys():
                out[k] = _stack_over_time([step[k] for step in seq if k in step])
            return out
        else:
            return np.stack(seq, axis=0)

    def _flatten_for_npz(d, prefix=""):
        flat = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix == "" else f"{prefix}/{k}"
            if isinstance(v, dict):
                flat.update(_flatten_for_npz(v, key))
            else:
                flat[key] = v
        return flat

    def _promote_seg_label_mapping(flat):
        """Move sensor/seg_label_mapping (stacked T-length string array) to meta/seg_id_to_label."""
        key = "sensor/seg_label_mapping"
        if key not in flat:
            return
        arr = flat.pop(key)
        # arr is shape (T,) of identical JSON strings; take the first
        label_json = arr[0] if hasattr(arr, "__len__") else arr
        if isinstance(label_json, (bytes, np.bytes_)):
            label_json = label_json.decode("utf-8")
        flat["meta/seg_id_to_label"] = np.asarray(str(label_json), dtype=np.str_)

    def _trajectory_to_video_frames(traj):
        frames_by_view = {}
        if not isinstance(traj, (list, tuple)) or len(traj) == 0:
            return frames_by_view

        for step in traj:
            if not isinstance(step, dict):
                continue
            sensor = step.get("sensor")
            if not isinstance(sensor, dict):
                continue
            rgbs = sensor.get("rgbs")
            if not isinstance(rgbs, dict):
                continue
            for view_name, frame in rgbs.items():
                if frame is None:
                    continue
                arr = np.asarray(frame)
                if arr.ndim != 3 or arr.shape[2] < 3:
                    continue
                arr = arr[..., :3]
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                frames_by_view.setdefault(view_name, []).append(arr)
        return frames_by_view

    def _resolve_traj_variants(result_dict):
        success_trajs = result_dict.get("success_trajectories") or []
        success_node_end_indices = result_dict.get("success_trajectory_node_end_indices") or []
        success_node_end_grasped_names = result_dict.get("success_trajectory_node_end_grasped_names") or []
        success_node_subgoals = result_dict.get("success_trajectory_node_subgoals") or []
        success_node_ids = result_dict.get("success_trajectory_node_ids") or []
        if not save_restore_augmented_traj:
            return [
                (
                    "",
                    success_trajs,
                    success_node_end_indices,
                    success_node_end_grasped_names,
                    success_node_subgoals,
                    success_node_ids,
                )
            ]

        success_trajs_restore2step = result_dict.get("success_trajectories_restore2step") or []
        success_node_end_indices_restore2step = result_dict.get("success_trajectory_node_end_indices_restore2step") or []
        success_node_end_grasped_names_restore2step = (
            result_dict.get("success_trajectory_node_end_grasped_names_restore2step")
            or success_node_end_grasped_names
        )
        success_node_subgoals_restore2step = (
            result_dict.get("success_trajectory_node_subgoals_restore2step")
            or success_node_subgoals
        )
        success_node_ids_restore2step = (
            result_dict.get("success_trajectory_node_ids_restore2step")
            or success_node_ids
        )
        variants = []
        if save_original_also_when_augmented:
            variants.append(
                (
                    "",
                    success_trajs,
                    success_node_end_indices,
                    success_node_end_grasped_names,
                    success_node_subgoals,
                    success_node_ids,
                )
            )
        variants.append(
            (
                "_restore2step",
                success_trajs_restore2step,
                success_node_end_indices_restore2step,
                success_node_end_grasped_names_restore2step,
                success_node_subgoals_restore2step,
                success_node_ids_restore2step,
            )
        )
        return variants

    def _inject_node_boundary_meta(
        flat,
        node_end_indices,
        node_end_grasped_names=None,
        node_subgoals=None,
        node_ids=None,
    ):
        if node_end_indices is None:
            return
        node_ends = np.asarray(node_end_indices, dtype=np.int64).reshape(-1)
        if node_ends.size == 0:
            return
        node_starts = np.zeros_like(node_ends)
        node_starts[1:] = node_ends[:-1]
        node_counts = node_ends - node_starts
        flat["meta/node_end_indices_exclusive"] = node_ends
        flat["meta/node_start_indices_inclusive"] = node_starts
        flat["meta/node_step_counts"] = node_counts

        if node_end_grasped_names is not None:
            values = list(node_end_grasped_names)
            if len(values) < node_ends.size:
                values.extend([""] * (node_ends.size - len(values)))
            elif len(values) > node_ends.size:
                values = values[: node_ends.size]
            names = [v if isinstance(v, str) else ("" if v is None else str(v)) for v in values]
            flat["meta/node_end_grasped_name"] = np.asarray(names, dtype=np.str_)

        if node_subgoals is not None:
            values = list(node_subgoals)
            if len(values) < node_ends.size:
                values.extend([""] * (node_ends.size - len(values)))
            elif len(values) > node_ends.size:
                values = values[: node_ends.size]
            subgoals = [v if isinstance(v, str) else ("" if v is None else str(v)) for v in values]
            flat["meta/node_subgoal"] = np.asarray(subgoals, dtype=np.str_)

        if node_ids is not None:
            values = list(node_ids)
            if len(values) < node_ends.size:
                values.extend([-1] * (node_ends.size - len(values)))
            elif len(values) > node_ends.size:
                values = values[: node_ends.size]
            flat["meta/node_ids"] = np.asarray(values, dtype=np.int64)

    try:
        _search_wall_t0 = time.perf_counter()
        search_outputs, node_trajectories, all_scores = \
            tree_searcher.search(task_env=task_env, worker_id=wid)
        _search_wall_s = time.perf_counter() - _search_wall_t0

        result_dict = search_outputs[0]
        if isinstance(result_dict, dict):
            result_dict["search_wall_s"] = _search_wall_s
        node_trajectories = node_trajectories[0]
        traj_variants = _resolve_traj_variants(result_dict)
        final_score = sum(all_scores)
        if final_score > 0 and not any(result_dict.get("success_trajectories") or []):
            final_score = 0

        json_path = os.path.join(out_dir, f"{save_tag}.json")
        traj_dir = os.path.join(out_dir, "traj")
        os.makedirs(traj_dir, exist_ok=True)

        # Strip raw trajectories from JSON; they are saved separately as npz.
        result_for_json = dict(result_dict)
        result_for_json.pop("success_trajectories", None)
        result_for_json.pop("success_trajectories_restore2step", None)

        if io_queue is not None:
            try:
                io_queue.put({"type": "write_json", "path": json_path, "data": result_for_json}, timeout=0.1)
            except Exception:
                io_queue.put({"type": "write_json", "path": json_path, "data": result_for_json})
            for suffix, success_trajs, node_end_indices_list, node_end_grasped_names_list, node_subgoals_list, node_ids_list in traj_variants:
                for idx, traj in enumerate(success_trajs):
                    try:
                        npz_path = os.path.join(traj_dir, f"{save_tag}_traj{idx}{suffix}.npz")
                        # Stack along time dimension per key
                        if traj:
                            stacked = _stack_over_time(traj)
                            flat = _flatten_for_npz(stacked)
                            node_end_indices = node_end_indices_list[idx] if idx < len(node_end_indices_list) else None
                            node_end_grasped_names = (
                                node_end_grasped_names_list[idx]
                                if idx < len(node_end_grasped_names_list)
                                else None
                            )
                            node_subgoals = (
                                node_subgoals_list[idx]
                                if idx < len(node_subgoals_list)
                                else None
                            )
                            node_ids = node_ids_list[idx] if idx < len(node_ids_list) else None
                            _inject_node_boundary_meta(
                                flat,
                                node_end_indices,
                                node_end_grasped_names=node_end_grasped_names,
                                node_subgoals=node_subgoals,
                                node_ids=node_ids,
                            )
                            _promote_seg_label_mapping(flat)
                            np.savez_compressed(npz_path, **flat)
                    except Exception:
                        pass

            if save_node_videos:
                for node_id, traj in node_trajectories.items():
                    frames_by_view = _trajectory_to_video_frames(traj)
                    if not frames_by_view:
                        continue
                    for view_name, view_frames in frames_by_view.items():
                        if not view_frames:
                            continue
                        view_dir = os.path.join(out_dir, f"{view_name}_videos")
                        os.makedirs(view_dir, exist_ok=True)
                        mp4_path = os.path.join(view_dir, f"{save_tag}_video_{node_id}.mp4")
                        task = {
                            "type": "write_video",
                            "path": mp4_path,
                            "frames": view_frames,
                            "fps": 20,
                            "codec": "libx264",
                            "crf": 23,
                            "preset": "veryfast",
                            "pix_fmt": "yuv420p"
                        }
                        try:
                            io_queue.put(task, timeout=0.1)
                        except Exception:
                            io_queue.put(task)

        else:
            with open(json_path, "w") as f:
                json.dump(result_for_json, f, indent=4)
            for suffix, success_trajs, node_end_indices_list, node_end_grasped_names_list, node_subgoals_list, node_ids_list in traj_variants:
                for idx, traj in enumerate(success_trajs):
                    npz_path = os.path.join(traj_dir, f"{save_tag}_traj{idx}{suffix}.npz")
                    try:
                        if traj:
                            stacked = _stack_over_time(traj)
                            flat = _flatten_for_npz(stacked)
                            node_end_indices = node_end_indices_list[idx] if idx < len(node_end_indices_list) else None
                            node_end_grasped_names = (
                                node_end_grasped_names_list[idx]
                                if idx < len(node_end_grasped_names_list)
                                else None
                            )
                            node_subgoals = (
                                node_subgoals_list[idx]
                                if idx < len(node_subgoals_list)
                                else None
                            )
                            node_ids = node_ids_list[idx] if idx < len(node_ids_list) else None
                            _inject_node_boundary_meta(
                                flat,
                                node_end_indices,
                                node_end_grasped_names=node_end_grasped_names,
                                node_subgoals=node_subgoals,
                                node_ids=node_ids,
                            )
                            _promote_seg_label_mapping(flat)
                            np.savez_compressed(npz_path, **flat)
                    except Exception:
                        pass

            if save_node_videos:
                import imageio.v2 as iio
                for node_id, traj in node_trajectories.items():
                    frames_by_view = _trajectory_to_video_frames(traj)
                    if not frames_by_view:
                        continue
                    for view_name, view_frames in frames_by_view.items():
                        if not view_frames:
                            continue
                        view_dir = os.path.join(out_dir, f"{view_name}_videos")
                        os.makedirs(view_dir, exist_ok=True)
                        mp4_path = os.path.join(view_dir, f"{save_tag}_video_{node_id}.mp4")
                        with iio.get_writer(
                            mp4_path,
                            format="FFMPEG",
                            mode="I",
                            fps=20,
                            codec="libx264",
                        ) as writer:
                            for fr in view_frames:
                                writer.append_data(fr)

        return final_score
    finally:
        if layout_pin_target is not None:
            layout_pin_target.force_layout_idx = prev_force
        _cleanup_after_episode(tree_searcher)
