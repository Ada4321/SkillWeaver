import os
import atexit
import json
import logging, logging.handlers
import multiprocessing as mp
from multiprocessing import current_process
import traceback
import time
import random
from pathlib import Path

from .general import (
    _build_tree_searcher_in_child, 
    run_and_save_one_episode, 
    setup_main_logging, 
    setup_worker_logging,
    writer_loop,
    iter_scene_tasks,
)
from utils.random_control import configure_timestamp_seed_control

_CTX = {}
_LOG_QUEUE = None
_LOG_LISTENER = None
_IO_QUEUE = None
_IO_WRITER = None


def _resolve_eval_num_from_cfg(cfg: dict) -> int:
    """Read eval_num from config without initializing IsaacSim/AppLauncher in parent process."""
    sim_cfg = (cfg or {}).get("simulator", {})
    isaac_cfg = sim_cfg.get("isaaclab", sim_cfg)
    try:
        return int(isaac_cfg.get("eval_num", 20))
    except Exception:
        return 20


def _configure_worker_runtime_dirs(out_dir: str, wid: int, pid: int) -> None:
    """Assign per-worker Kit/Omniverse cache+log dirs to avoid cross-process locking."""
    worker_root = os.path.join(out_dir, "_worker_runtime", f"worker_{wid}_pid_{pid}")
    omni_user = os.path.join(worker_root, "omni_user")
    omni_cache = os.path.join(worker_root, "omni_cache")
    omni_log = os.path.join(worker_root, "omni_log")
    omni_data = os.path.join(worker_root, "omni_data")
    xdg_cache = os.path.join(worker_root, "xdg_cache")
    xdg_config = os.path.join(worker_root, "xdg_config")
    xdg_data = os.path.join(worker_root, "xdg_data")

    for d in (omni_user, omni_cache, omni_log, omni_data, xdg_cache, xdg_config, xdg_data):
        os.makedirs(d, exist_ok=True)

    os.environ["OMNI_USER_DIR"] = omni_user
    os.environ["OMNI_CACHE_DIR"] = omni_cache
    os.environ["OMNI_LOG_DIR"] = omni_log
    os.environ["OMNI_DATA_DIR"] = omni_data
    os.environ["XDG_CACHE_HOME"] = xdg_cache
    os.environ["XDG_CONFIG_HOME"] = xdg_config
    os.environ["XDG_DATA_HOME"] = xdg_data

    logging.warning(
        "Worker %s(pid=%s) runtime dirs set: OMNI_USER_DIR=%s OMNI_CACHE_DIR=%s OMNI_LOG_DIR=%s",
        wid,
        pid,
        omni_user,
        omni_cache,
        omni_log,
    )
    return worker_root


def _append_worker_event(out_dir: str, wid: int, pid: int, msg: str) -> None:
    """Best-effort local event log outside logging queue (useful during worker init crashes)."""
    try:
        worker_root = os.path.join(out_dir, "_worker_runtime", f"worker_{wid}_pid_{pid}")
        os.makedirs(worker_root, exist_ok=True)
        path = os.path.join(worker_root, "worker_events.log")
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def _cleanup_worker_env():
    env = _CTX.get("task_env")
    ts = _CTX.get("tree_searcher")
    if env is None or ts is None:
        return
    try:
        ts.simulator.shutdown_env(env)
    except Exception:
        logging.exception("Worker cleanup: shutdown_env failed.")
    try:
        ts.simulator.shutdown_app()
    except Exception:
        logging.exception("Worker cleanup: shutdown_app failed.")
    try:
        out_dir = _CTX.get("out_dir")
        wid = _CTX.get("wid", -1)
        pid = os.getpid()
        if out_dir is not None:
            _append_worker_event(out_dir, wid, pid, "atexit cleanup finished")
    except Exception:
        pass


def _init_worker(cfg, out_dir, k, use_mcts, log_queue, io_queue, task_key: str):
    setup_worker_logging(log_queue)
    configure_timestamp_seed_control(bool(cfg.get("run", {}).get("use_timestamp_seed", False)))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    proc = current_process()
    wid = (proc._identity or [0])[0]
    pid = os.getpid()
    worker_root = _configure_worker_runtime_dirs(out_dir, wid=wid, pid=pid)
    _append_worker_event(out_dir, wid, pid, f"init_worker start pid={pid} ppid={os.getppid()} name={proc.name}")
    logging.warning("Worker init start: wid=%s pid=%s ppid=%s root=%s", wid, pid, os.getppid(), worker_root)

    try:
        t0 = time.perf_counter()
        ts = _build_tree_searcher_in_child(cfg)
        logging.warning(
            "Worker init: _build_tree_searcher_in_child done wid=%s pid=%s elapsed=%.3fs",
            wid,
            pid,
            time.perf_counter() - t0,
        )
        _append_worker_event(out_dir, wid, pid, "build_tree_searcher done")

        t1 = time.perf_counter()
        task_env = ts.simulator.make_env()
        logging.warning(
            "Worker init: make_env done wid=%s pid=%s elapsed=%.3fs",
            wid,
            pid,
            time.perf_counter() - t1,
        )
        _append_worker_event(out_dir, wid, pid, "make_env done")
    except BaseException:
        tb = traceback.format_exc()
        logging.error("Worker init failed: wid=%s pid=%s\n%s", wid, pid, tb)
        _append_worker_event(out_dir, wid, pid, f"init_worker failed\n{tb}")
        raise

    global _CTX
    _CTX = dict(
        tree_searcher=ts,
        out_dir=out_dir,
        k=k,
        use_mcts=use_mcts,
        io_queue=io_queue,
        wid=wid,
        task_env=task_env,
        task_key=task_key,
    )
    _append_worker_event(out_dir, wid, pid, "init_worker success")
    atexit.register(_cleanup_worker_env)


def _format_task_tag(
    scene_name: str,
    task_type: str,
    task_idx: int,
    task_key: str | None = None,
    layout_idx: int | None = None,
) -> str:
    key = task_key or "tasks"
    base = f"{scene_name}+{key}+{task_type}+task{task_idx}"
    if layout_idx is None:
        return base
    return f"{base}+L{int(layout_idx)}"


def _normalize_layout_ids(raw) -> set[int] | None:
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        items = [raw]
    else:
        items = list(raw)
    if not items:
        return None
    out: set[int] = set()
    for x in items:
        out.add(int(x))
    if any(v < 0 for v in out):
        raise ValueError(f"mcts.tasks.layout_ids contains negative id(s): {sorted(out)}")
    return out


def _normalize_task_ids(raw) -> set[tuple[str | None, int]] | None:
    """Normalize mcts.tasks.task_ids into a set of (task_type_or_None, task_idx).

    Each item is either:
      - int: matches any task_type with this idx -> (None, idx)
      - "idx": same as int
      - "type:idx": matches only this (type, idx)
    """
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        items = [raw]
    else:
        items = list(raw)
    if not items:
        return None
    out: set[tuple[str | None, int]] = set()
    for x in items:
        if isinstance(x, int):
            if x < 0:
                raise ValueError(f"mcts.tasks.task_ids contains negative idx: {x}")
            out.add((None, int(x)))
            continue
        s = str(x).strip()
        if ":" in s:
            type_part, idx_part = s.split(":", 1)
            type_part = type_part.strip()
            idx_part = idx_part.strip()
            if not type_part:
                raise ValueError(f"mcts.tasks.task_ids entry has empty task_type: {x!r}")
            idx = int(idx_part)
            if idx < 0:
                raise ValueError(f"mcts.tasks.task_ids contains negative idx: {x!r}")
            out.add((type_part, idx))
        else:
            idx = int(s)
            if idx < 0:
                raise ValueError(f"mcts.tasks.task_ids contains negative idx: {x!r}")
            out.add((None, idx))
    return out


def _dedup_tasks_pick_random_layout(tasks, seed: int):
    """Collapse cross-layout duplicates of the same task signature.

    For each unique (task_type, raw_task_entry) keep one entry whose
    layout_idx is chosen uniformly at random across the duplicates' layouts.
    Tasks whose serialized entries differ (different objects / coordinates /
    sub-steps) are NOT merged. No-op when there are no duplicates.
    """
    rng = random.Random(seed)
    groups: dict = {}
    order: list = []
    for entry in tasks:
        task_type, _idx, _variations, raw_task_entry, _layout = entry
        try:
            payload = json.dumps(raw_task_entry, sort_keys=True, default=str)
        except Exception:
            payload = repr(raw_task_entry)
        sig = (task_type, payload)
        if sig not in groups:
            order.append(sig)
            groups[sig] = []
        groups[sig].append(entry)
    return [rng.choice(groups[sig]) for sig in order]


def _choose_balanced_task_index(
    scene_name: str,
    total_task_groups: int,
    offset_seed: int = 0,
) -> int:
    """Pick a deterministic single task index from scene id for cross-scene balancing."""
    if total_task_groups <= 0:
        raise ValueError(f"total_task_groups must be > 0, got {total_task_groups}")
    scene_digits = "".join(ch for ch in scene_name if ch.isdigit())
    if scene_digits:
        scene_id = int(scene_digits)
    else:
        # Stable fallback for non-standard scene names.
        scene_id = sum((i + 1) * ord(c) for i, c in enumerate(scene_name))
    return int((scene_id + int(offset_seed)) % total_task_groups)


def _write_task_success(out_dir: str, scores: list[int]) -> None:
    eval_num = len(scores)
    task_success = sum(scores)
    task_rate = task_success / eval_num if eval_num > 0 else 0
    task_rate_str = "Final state log:\n"
    for i, score in enumerate(scores):
        task_rate_str += f"Exp {i}: {score}\n"
    task_rate_str += f"\nAvg Success Rate: {task_rate}"
    with open(os.path.join(out_dir, "success_rate.txt"), "w") as f:
        f.write(task_rate_str)


def _is_success_score(score) -> bool:
    try:
        return float(score) > 0.0
    except Exception:
        return bool(score)


def _count_task_existing_successes(save_rollouts_dir: str, task_tag: str) -> int:
    """Count successful exps for a specific task_tag across all timestamp subdirs under save_rollouts_dir.

    Structure: save_rollouts_dir / <timestamp> / <task_tag> / exp_* / traj / *.npz
    """
    root = Path(save_rollouts_dir)
    if not root.exists():
        return 0
    successful_exps: set[str] = set()
    for npz_path in root.glob(f"*/{task_tag}/exp_*/traj/*.npz"):
        exp_dir = npz_path.parent.parent
        if exp_dir.name.startswith("exp_"):
            successful_exps.add(exp_dir.name)
    return len(successful_exps)


def _collect_successful_exp_counts(save_rollouts_dir: str) -> dict[str, int]:
    """Count, per task tag, how many distinct exp_* dirs contain at least one success npz.

    Backward compatibility:
    - old format: scene+task_type+task_idx
    - new format: scene+task_key+task_type+task_idx

    If old format is detected, we additionally synthesize a +tasks+ tag with the same count.
    """
    root = Path(save_rollouts_dir)
    if not root.exists():
        return {}

    def _to_tasks_key_tag(task_tag: str) -> str | None:
        parts = task_tag.split("+")
        if len(parts) != 3:
            return None
        scene_name, task_type, task_idx = parts
        if not scene_name or not task_type or not task_idx.startswith("task"):
            return None
        return f"{scene_name}+tasks+{task_type}+{task_idx}"

    # task_tag -> set of exp_dir names that have at least one success npz
    task_successful_exps: dict[str, set[str]] = {}
    for npz_path in root.glob("**/exp_*/traj/*.npz"):
        try:
            if npz_path.parent.name != "traj":
                continue
            exp_dir = npz_path.parent.parent
            if not exp_dir.name.startswith("exp_"):
                continue
            task_tag = exp_dir.parent.name
            if not task_tag:
                continue
            task_successful_exps.setdefault(task_tag, set()).add(exp_dir.name)
            compat_tag = _to_tasks_key_tag(task_tag)
            if compat_tag:
                task_successful_exps.setdefault(compat_tag, set()).add(exp_dir.name)
        except Exception:
            continue
    return {tag: len(exps) for tag, exps in task_successful_exps.items()}



def _run_task_evals(
    tree_searcher,
    task_env,
    out_dir: str,
    task_tag: str,
    raw_task_entry,
    task_variations: list[str],
    k: int,
    use_mcts: bool,
    success_exp_to_advance: int = 0,
    save_rollouts_dir: str = "",
    layout_idx: int | None = None,
):
    task_out_dir = os.path.join(out_dir, task_tag)
    os.makedirs(task_out_dir, exist_ok=True)

    task_scores = []
    if success_exp_to_advance > 0 and save_rollouts_dir:
        success_exp_count = _count_task_existing_successes(save_rollouts_dir, task_tag)
        if success_exp_count > 0:
            logging.info(
                "Task %s: found %d pre-existing success exp(s) on disk, counting toward success_exp_to_advance=%d.",
                task_tag, success_exp_count, success_exp_to_advance,
            )
    else:
        success_exp_count = 0

    if success_exp_to_advance > 0 and success_exp_count >= success_exp_to_advance:
        logging.warning(
            "Task early-stop (pre-existing): task=%s already has %d success exp(s) >= success_exp_to_advance=%d, skipping.",
            task_tag, success_exp_count, success_exp_to_advance,
        )
        _write_task_success(task_out_dir, task_scores)
        return task_scores

    for ep_id, task_entry in enumerate(task_variations):
        tree_searcher.simulator.args.raw_task_entry = raw_task_entry
        tree_searcher.simulator.args.instruction = task_entry
        score = evaluate_isaaclab(
            ep_id,
            tree_searcher,
            os.path.join(task_out_dir, f"exp_{ep_id}"),
            k=k,
            use_mcts=use_mcts,
            task_env=task_env,
            save_tag_base=task_tag,
            layout_idx=layout_idx,
        )
        task_scores.append(score)
        if _is_success_score(score):
            success_exp_count += 1
        if success_exp_to_advance > 0 and success_exp_count >= success_exp_to_advance:
            logging.warning(
                "Task early-stop: task=%s reached success_exp_to_advance=%d after %d exp(s).",
                task_tag,
                success_exp_to_advance,
                len(task_scores),
            )
            break

    _write_task_success(task_out_dir, task_scores)
    return task_scores



def _map_episode_func(task_payload):
    task_type, task_idx, ep_id, task_entry, raw_task_entry, scene_name, layout_idx = task_payload
    task_tag = _format_task_tag(
        scene_name, task_type, task_idx,
        task_key=_CTX.get("task_key"), layout_idx=layout_idx,
    )
    wid = _CTX.get("wid", -1)
    pid = os.getpid()
    out_dir = _CTX.get("out_dir")
    try:
        _append_worker_event(out_dir, wid, pid, f"episode start task={task_tag} ep={ep_id}")
        _CTX["tree_searcher"].simulator.args.instruction = task_entry
        _CTX["tree_searcher"].simulator.args.task_type = task_type
        _CTX["tree_searcher"].simulator.args.raw_task_entry = raw_task_entry
        score = evaluate_isaaclab(
            ep_id,
            _CTX["tree_searcher"],
            os.path.join(_CTX["out_dir"], task_tag, f"exp_{ep_id}"),
            k=_CTX["k"],
            use_mcts=_CTX["use_mcts"],
            task_env=_CTX.get("task_env"),
            save_tag_base=task_tag,
            layout_idx=layout_idx,
        )
        _append_worker_event(out_dir, wid, pid, f"episode done task={task_tag} ep={ep_id} score={score}")
        return {"task_tag": task_tag, "ep_id": ep_id, "score": score}
    except BaseException:
        tb = traceback.format_exc()
        logging.error("Worker episode failed: wid=%s pid=%s task=%s ep=%s\n%s", wid, pid, task_tag, ep_id, tb)
        _append_worker_event(out_dir, wid, pid, f"episode failed task={task_tag} ep={ep_id}\n{tb}")
        raise


def evaluate_isaaclab(
    exp_count,
    tree_searcher,
    out_dir,
    k: int = 1,
    use_mcts=False,
    task_env=None,
    save_tag_base: str | None = None,
    layout_idx: int | None = None,
):
    # make simulation env
    created_env = task_env is None
    if created_env:
        task_env = tree_searcher.simulator.make_env()
    try:
        if save_tag_base:
            save_tag = save_tag_base
        else:
            save_tag = f"exp{exp_count}"
        score = run_and_save_one_episode(
            tree_searcher, task_env, out_dir, save_tag,
            use_mcts=use_mcts, io_queue=_CTX.get("io_queue"), wid=_CTX.get("wid", 0),
            layout_idx=layout_idx)
    finally:
        if created_env:
            tree_searcher.simulator.shutdown_env(task_env)
    return score


def evaluate_isaaclab_mp(cfg, out_dir, k: int = 1, use_mcts=False, num_processes: int = 1):
    configure_timestamp_seed_control(bool(cfg.get("run", {}).get("use_timestamp_seed", False)))
    ctx = mp.get_context("spawn")

    global _LOG_QUEUE, _LOG_LISTENER, _IO_QUEUE, _IO_WRITER
    _LOG_QUEUE = ctx.Queue(-1)
    _LOG_LISTENER = setup_main_logging(_LOG_QUEUE)
    _IO_QUEUE = None
    _IO_WRITER = None
    
    try:
        env_cfg = cfg.get("env")
        eval_num = _resolve_eval_num_from_cfg(cfg)
        eval_cfg = (cfg.get("run", {}).get("eval", {}) or {})

        max_exp_per_task_raw = eval_cfg.get("max_exp_per_task")
        if max_exp_per_task_raw is None:
            max_exp_per_task = int(eval_num)
        else:
            max_exp_per_task = int(max_exp_per_task_raw)
            if max_exp_per_task <= 0:
                raise ValueError("run.eval.max_exp_per_task must be > 0 when provided.")

        max_task_per_scene_raw = eval_cfg.get("max_task_per_scene")
        if max_task_per_scene_raw is None:
            max_task_per_scene = None
        else:
            max_task_per_scene = int(max_task_per_scene_raw)
            if max_task_per_scene <= 0:
                raise ValueError("run.eval.max_task_per_scene must be > 0 when provided.")
        task_subsample_strategy = str(eval_cfg.get("task_subsample_strategy", "random")).strip().lower()
        if task_subsample_strategy not in ("random", "balanced_scene_mod"):
            raise ValueError(
                "run.eval.task_subsample_strategy must be one of {'random', 'balanced_scene_mod'}, "
                f"got {task_subsample_strategy!r}"
            )
        task_subsample_seed = int(eval_cfg.get("task_subsample_seed", 0))

        success_exp_to_advance_raw = eval_cfg.get("success_exp_to_advance")
        if success_exp_to_advance_raw is None:
            success_exp_to_advance = 0
        else:
            success_exp_to_advance = int(success_exp_to_advance_raw)
            if success_exp_to_advance < 0:
                raise ValueError("run.eval.success_exp_to_advance must be >= 0.")
        only_search_unsuccessful_tasks = bool(eval_cfg.get("only_search_unsuccessful_tasks", False))

        if success_exp_to_advance > max_exp_per_task:
            logging.warning(
                "success_exp_to_advance (%d) > max_exp_per_task (%d); early-stop will never trigger.",
                success_exp_to_advance,
                max_exp_per_task,
            )

        if num_processes > 1 and success_exp_to_advance > 0:
            logging.warning(
                "success_exp_to_advance with num_processes>1 is forced to single-process to preserve strict early-stop semantics."
            )
            num_processes = 1

        scene_file = (env_cfg or {}).get("scene_desc_file")
        scene_name = os.path.splitext(os.path.basename(scene_file))[0] if scene_file else "scene"
        task_filter = cfg.get("run", {}).get("task_types")
        tasks_cfg = cfg.get("mcts", {}).get("tasks", {})
        surface_tgt_list_path = tasks_cfg.get("surface_tgt_list")
        force_upright = tasks_cfg.get("force_upright", False)
        upright_random_prob = tasks_cfg.get("upright_random_prob", 0.5)
        task_key = tasks_cfg.get("task_key", "tasks")
        tasks = iter_scene_tasks(
            scene_file,
            task_filter,
            eval_num=max_exp_per_task,
            surface_tgt_list_path=surface_tgt_list_path,
            force_upright=force_upright,
            upright_random_prob=upright_random_prob,
            include_task_entry=True,
            task_key=task_key,
        )
        layout_id_whitelist = _normalize_layout_ids(tasks_cfg.get("layout_ids"))
        task_id_whitelist = _normalize_task_ids(tasks_cfg.get("task_ids"))
        if layout_id_whitelist is not None or task_id_whitelist is not None:
            before_whitelist = len(tasks)
            if layout_id_whitelist is not None and tasks and tasks[0][4] is None:
                raise ValueError(
                    f"mcts.tasks.layout_ids={sorted(layout_id_whitelist)} given, but "
                    f"scene_file={scene_file} uses legacy non-layout task format. "
                    f"Layout selection only applies to scenes with 'tasks_by_layout'."
                )
            filtered_tasks = []
            for entry in tasks:
                task_type, task_idx, _variations, _raw, layout_idx = entry
                if layout_id_whitelist is not None and int(layout_idx) not in layout_id_whitelist:
                    continue
                if task_id_whitelist is not None:
                    if (None, int(task_idx)) not in task_id_whitelist and (
                        str(task_type),
                        int(task_idx),
                    ) not in task_id_whitelist:
                        continue
                filtered_tasks.append(entry)
            tasks = filtered_tasks
            logging.warning(
                "[runner] task whitelist applied: scene_file=%s layout_ids=%s task_ids=%s before=%d after=%d",
                scene_file,
                sorted(layout_id_whitelist) if layout_id_whitelist is not None else None,
                sorted(task_id_whitelist) if task_id_whitelist is not None else None,
                before_whitelist,
                len(tasks),
            )
        total_task_groups = len(tasks)
        skipped_successful_tasks = 0
        if only_search_unsuccessful_tasks:
            success_threshold = max(1, success_exp_to_advance)
            successful_exp_counts = _collect_successful_exp_counts(cfg["run"]["save_rollouts_dir"])
            filtered_tasks = []
            for task_type, task_idx, task_variations, raw_task_entry, layout_idx in tasks:
                task_tag = _format_task_tag(
                    scene_name, task_type, task_idx, task_key=task_key, layout_idx=layout_idx
                )
                if successful_exp_counts.get(task_tag, 0) >= success_threshold:
                    skipped_successful_tasks += 1
                    continue
                filtered_tasks.append((task_type, task_idx, task_variations, raw_task_entry, layout_idx))
            tasks = filtered_tasks
            logging.warning(
                "only_search_unsuccessful_tasks enabled: scene_file=%s kept=%d/%d skipped_successful=%d success_threshold=%d",
                scene_file,
                len(tasks),
                total_task_groups,
                skipped_successful_tasks,
                success_threshold,
            )
        total_task_groups_after_success_filter = len(tasks)
        # Collapse cross-layout duplicates so subsampling doesn't draw the
        # same logical task twice (e.g. in_the_wild scenes replicate the
        # same pick_up task across every layout). Pick a random layout per
        # unique task signature; tasks with differing entries are kept
        # separately. No-op for libero-style scenes where each layout has a
        # distinct task list.
        dedup_seed = int(time.time_ns()) ^ int(os.getpid())
        before_dedup = len(tasks)
        if layout_id_whitelist is None:
            tasks = _dedup_tasks_pick_random_layout(tasks, dedup_seed)
            if len(tasks) < before_dedup:
                logging.warning(
                    "Task dedup collapsed cross-layout duplicates: scene_file=%s before=%d after=%d dedup_seed=%d",
                    scene_file, before_dedup, len(tasks), dedup_seed,
                )
        else:
            logging.warning(
                "Task dedup skipped: explicit mcts.tasks.layout_ids=%s preserves all selected layouts.",
                sorted(layout_id_whitelist),
            )
        total_task_groups_after_success_filter = len(tasks)
        task_sample_seed = None
        if (
            task_id_whitelist is None
            and max_task_per_scene is not None
            and total_task_groups_after_success_filter > max_task_per_scene
        ):
            if task_subsample_strategy == "balanced_scene_mod" and max_task_per_scene == 1:
                selected_idx = _choose_balanced_task_index(
                    scene_name=scene_name,
                    total_task_groups=total_task_groups_after_success_filter,
                    offset_seed=task_subsample_seed,
                )
                tasks = [tasks[selected_idx]]
                logging.warning(
                    "Balanced task subsampling enabled: scene_file=%s selected_idx=%d sampled=%d/%d strategy=%s seed=%d",
                    scene_file,
                    selected_idx,
                    len(tasks),
                    total_task_groups_after_success_filter,
                    task_subsample_strategy,
                    task_subsample_seed,
                )
            else:
                task_sample_seed = int(time.time_ns()) ^ int(os.getpid())
                rng = random.Random(task_sample_seed)
                selected_indices = sorted(rng.sample(range(total_task_groups_after_success_filter), k=max_task_per_scene))
                tasks = [tasks[i] for i in selected_indices]
                logging.warning(
                    "Random task subsampling enabled: scene_file=%s sampled=%d/%d timestamp_seed=%d strategy=%s",
                    scene_file,
                    len(tasks),
                    total_task_groups_after_success_filter,
                    task_sample_seed,
                    task_subsample_strategy,
                )

        print(
            f"[runner] Loaded {len(tasks)} task groups from scene_file={scene_file} "
            f"task_filter={task_filter} max_exp_per_task={max_exp_per_task} "
            f"success_exp_to_advance={success_exp_to_advance} "
            f"only_search_unsuccessful_tasks={only_search_unsuccessful_tasks} "
            f"skipped_successful_tasks={skipped_successful_tasks} "
            f"max_task_per_scene={max_task_per_scene} "
            f"total_before_subsample={total_task_groups_after_success_filter} "
            f"total_before_success_filter={total_task_groups} "
            f"task_sample_seed={task_sample_seed} "
            f"task_subsample_strategy={task_subsample_strategy} "
            f"task_subsample_seed={task_subsample_seed}",
            flush=True,
        )
        if len(tasks) == 0:
            if only_search_unsuccessful_tasks:
                success_rate_str = "Final state log:\n\nAvg Success Rate: 0"
                with open(f"{os.path.join(out_dir, 'success_rate.txt')}", "w") as f:
                    f.write(success_rate_str)
                logging.warning(
                    "No pending tasks remain after filtering successful tasks. scene_file=%s",
                    scene_file,
                )
                return
            raise RuntimeError(
                f"No tasks loaded from scene_file={scene_file}. "
                f"Check run.task_types/task JSON/task sampler inputs."
            )

        if num_processes == 1:
            global _CTX
            _CTX.pop("io_queue", None)

            scene_timing: dict = {}
            exp_count = 0
            success_count = 0
            scores = []
            print("[runner] Building tree_searcher...", flush=True)
            _build_t0 = time.perf_counter()
            tree_searcher = _build_tree_searcher_in_child(cfg)
            scene_timing["build_tree_searcher_s"] = time.perf_counter() - _build_t0
            print("[runner] tree_searcher built. Creating task_env...", flush=True)
            try:
                _mkenv_t0 = time.perf_counter()
                task_env = tree_searcher.simulator.make_env()
                scene_timing["env_make_s"] = time.perf_counter() - _mkenv_t0
            except BaseException:
                logging.exception("task_env creation failed.")
                raise
            print("[runner] task_env created. Entering task loop...", flush=True)
            _scene_run_t0 = time.perf_counter()
            try:
                for task_type, task_idx, task_variations, raw_task_entry, layout_idx in tasks:
                    print(
                        f"[runner] Start task_type={task_type} task_idx={task_idx} "
                        f"layout_idx={layout_idx} num_variations={len(task_variations)}",
                        flush=True,
                    )
                    task_tag = _format_task_tag(
                        scene_name, task_type, task_idx,
                        task_key=task_key, layout_idx=layout_idx,
                    )
                    tree_searcher.simulator.args.task_type = task_type
                    tree_searcher.simulator.args.instruction = raw_task_entry
                    try:
                        task_scores = _run_task_evals(
                            tree_searcher,
                            task_env,
                            out_dir,
                            task_tag,
                            raw_task_entry,
                            task_variations,
                            k=k,
                            use_mcts=use_mcts,
                            success_exp_to_advance=success_exp_to_advance,
                            save_rollouts_dir=cfg["run"]["save_rollouts_dir"],
                            layout_idx=layout_idx,
                        )
                    except BaseException:
                        logging.exception(
                            "Task evaluation failed for scene=%s task_type=%s task_idx=%s",
                            scene_name,
                            task_type,
                            task_idx,
                        )
                        raise
                    exp_count += len(task_scores)
                    success_count += sum(task_scores)
                    scores.extend(task_scores)
                    print(
                        f"[runner] Finished task_type={task_type} task_idx={task_idx} "
                        f"task_scores={task_scores}",
                        flush=True,
                    )
                    logging.warning(
                        "All done! scene=%s task=%s idx=%d success_rate=%.3f",
                        scene_name, task_type, task_idx,
                        (sum(task_scores) / len(task_scores)) if task_scores else 0,
                    )
            finally:
                scene_timing["scene_run_s"] = time.perf_counter() - _scene_run_t0
                _shut_env_t0 = time.perf_counter()
                tree_searcher.simulator.shutdown_env(task_env)
                scene_timing["env_shutdown_s"] = time.perf_counter() - _shut_env_t0
                # Write scene_timing.json BEFORE shutdown_app, since the latter
                # often kills the process abruptly via IsaacLab/Omniverse and
                # the finally suffix never runs (resource_tracker semaphore
                # leak warning at end-of-log is the tell).
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    with open(os.path.join(out_dir, "scene_timing.json"), "w") as _f:
                        json.dump(scene_timing, _f, indent=2)
                except Exception:
                    logging.exception("Failed to write scene_timing.json")
                _shut_app_t0 = time.perf_counter()
                tree_searcher.simulator.shutdown_app()
                scene_timing["app_shutdown_s"] = time.perf_counter() - _shut_app_t0
                # Best-effort second write (may not survive if shutdown_app
                # exits the process). The first write above is the authoritative
                # one for env_wall_s computation.
                try:
                    with open(os.path.join(out_dir, "scene_timing.json"), "w") as _f:
                        json.dump(scene_timing, _f, indent=2)
                except Exception:
                    pass
            success_rate = success_count / exp_count if exp_count > 0 else 0
        else:
            _IO_QUEUE = ctx.Queue(maxsize=64)
            _IO_WRITER = ctx.Process(target=writer_loop, args=(_IO_QUEUE,), daemon=True)
            _IO_WRITER.start()

            episode_jobs = []
            ordered_task_tags = []
            task_ep_counts = {}
            for task_type, task_idx, task_variations, raw_task_entry, layout_idx in tasks:
                task_tag = _format_task_tag(
                    scene_name, task_type, task_idx,
                    task_key=task_key, layout_idx=layout_idx,
                )
                ordered_task_tags.append(task_tag)
                task_ep_counts[task_tag] = len(task_variations)
                for ep_id, task_entry in enumerate(task_variations):
                    episode_jobs.append(
                        (task_type, task_idx, ep_id, task_entry, raw_task_entry, scene_name, layout_idx)
                    )

            with mp.Pool(
                processes=num_processes,
                initializer=_init_worker,
                initargs=(cfg, out_dir, k, use_mcts, _LOG_QUEUE, _IO_QUEUE, task_key),
            ) as pool:
                it = pool.imap_unordered(
                    _map_episode_func,
                    episode_jobs,
                    chunksize=1,
                )
                episode_results = list(it)

            scores_by_task = {
                task_tag: [0 for _ in range(task_ep_counts.get(task_tag, 0))]
                for task_tag in ordered_task_tags
            }
            for result in episode_results:
                task_tag = result["task_tag"]
                ep_id = result["ep_id"]
                if task_tag in scores_by_task and 0 <= ep_id < len(scores_by_task[task_tag]):
                    scores_by_task[task_tag][ep_id] = result["score"]

            scores = []
            success_count = 0
            exp_count = 0
            for task_tag in ordered_task_tags:
                task_scores = scores_by_task.get(task_tag, [])
                _write_task_success(os.path.join(out_dir, task_tag), task_scores)
                scores.extend(task_scores)
                success_count += sum(task_scores)
                exp_count += len(task_scores)
            success_rate = success_count / exp_count if exp_count > 0 else 0

        success_rate_str = "Final state log:\n"
        for i, score in enumerate(scores):
            success_rate_str += f"Exp {i}: {score}\n"
        success_rate_str += f"\nAvg Success Rate: {success_rate}"

        with open(f"{os.path.join(out_dir, 'success_rate.txt')}", "w") as f:
            f.write(success_rate_str)

        logging.warning(f"All done! Success rate: {success_rate}")

    finally:
        try:
            from simulators.isaaclab import shutdown_sim_app
            shutdown_sim_app()
        except Exception:
            logging.exception("Final cleanup: shutdown_sim_app failed.")
        if _IO_QUEUE is not None:
            try:
                _IO_QUEUE.put(None)
            except Exception:
                logging.exception("Final cleanup: failed to enqueue IO shutdown sentinel.")
            try:
                if _IO_WRITER is not None:
                    _IO_WRITER.join(timeout=300)
                    if _IO_WRITER.is_alive():
                        _IO_WRITER.terminate()
                        _IO_WRITER.join(timeout=5)
            except Exception:
                logging.exception("Final cleanup: failed to join/terminate IO writer.")
            try:
                _IO_QUEUE.close()
                _IO_QUEUE.join_thread()
            except Exception:
                logging.exception("Final cleanup: failed to close IO queue.")

        if _LOG_LISTENER is not None:
            _LOG_LISTENER.stop()
        if _LOG_QUEUE is not None:
            _LOG_QUEUE.close()
            _LOG_QUEUE.join_thread()
