"""
Parallel MCTS searcher using IsaacLab env pool for leaf parallelization.

Each MCTS round selects k = pool_size // num_children_per_expand leaf nodes
simultaneously (sampling without replacement via in-flight exclusion). Their
children are executed in parallel across the env pool using batch physics
stepping from skills/execution_parallel.py.

Planning (curobo) is sequential; execution is parallel. The speedup is
roughly proportional to pool_size / num_children_per_expand.

Usage (Hydra):
    python main.py mcts=parallel simulator.isaaclab.pool_size=4
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import List, Optional, Tuple  # noqa: F401

import torch
from tqdm import tqdm

from tree_search.mcts import MonteCarloTreeSearch, TreeNode

logger = logging.getLogger(__name__)


def _snap_num_envs(snap: dict) -> int:
    """Return the batch dimension of a snapshot dict (first tensor's dim-0)."""
    for v in snap.values():
        if isinstance(v, dict):
            result = _snap_num_envs(v)
            if result is not None:
                return result
        elif hasattr(v, "shape") and len(v.shape) >= 1:
            return v.shape[0]
    return 1


class MCTSSearcherParallel(MonteCarloTreeSearch):
    """
    Drop-in replacement for MonteCarloTreeSearch that parallelizes skill
    execution across multiple IsaacLab env slots.

    Overrides search() to:
      1. Select k leaf nodes per round (sampling without replacement).
      2. Plan phases for each child sequentially (curobo is shared).
      3. Execute all children in parallel via run_parallel_slots().
      4. Post-process, judge, and backprop.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_multiple(
        self,
        root: TreeNode,
        k: int,
    ) -> List[Tuple[List[TreeNode], TreeNode]]:
        """Select up to k leaf nodes with in-flight exclusion.

        Returns list of (path, leaf) pairs (may be fewer than k if the
        tree has fewer distinct selectable leaves).
        """
        in_flight: set = set()
        results = []
        for _ in range(k):
            path = self._select(root, exclude=in_flight)
            leaf = path[-1]
            if id(leaf) in in_flight:
                break  # no more distinct leaves available
            in_flight.add(id(leaf))
            results.append((path, leaf))
        return results

    @staticmethod
    def _extract_slot_snapshot(full_snap: dict, slot_idx: int) -> dict:
        """Extract a single-env snapshot from a multi-env snapshot.

        full_snap tensors have shape (num_envs, ...).  We index into
        slot_idx and unsqueeze to produce shape (1, ...) so that the
        normal restore(snap, env_id=0) path works correctly.
        """
        single = {}
        for k, v in full_snap.items():
            if isinstance(v, dict):
                inner = {}
                for n, d in v.items():
                    if isinstance(d, dict):
                        inner[n] = {
                            kk: vv[slot_idx: slot_idx + 1].clone()
                            for kk, vv in d.items()
                        }
                    else:
                        inner[n] = d
                single[k] = inner
            else:
                single[k] = v
        return single

    def _snapshot_slot_to_node(
        self, node: TreeNode, full_snap: dict, slot_idx: int
    ) -> None:
        """Store a per-slot snapshot in node.snapshot["state_token"].

        Extracts slot_idx from the full multi-env snapshot so that later
        restore(state_token, env_id=0) reconstructs this slot's state at
        env slot 0.
        """
        node.snapshot["state_token"] = self._extract_slot_snapshot(
            full_snap, slot_idx
        )

    # ------------------------------------------------------------------
    # Core parallel expansion
    # ------------------------------------------------------------------

    def _expand_leaves_parallel(
        self,
        leaf_paths: List[Tuple[List[TreeNode], TreeNode]],
        pool_size: int,
        num_children: int,
    ) -> List[Tuple[List[TreeNode], TreeNode, List[TreeNode]]]:
        """Plan + execute all children across leaf_paths in parallel.

        Returns list of (path, leaf, new_children_list) — one entry per
        input leaf.  new_children_list may be empty on planning failure.
        """
        from skills.execution_parallel import (
            SlotState,
            plan_phases_for_skill,
            run_parallel_slots,
        )

        # Track new children per leaf for later post-processing.
        per_leaf_new: dict = {id(leaf): [] for _, leaf in leaf_paths}
        per_leaf_dead_end: dict = {id(leaf): False for _, leaf in leaf_paths}

        slots: List[SlotState] = []
        # env_id -> (path, leaf, child_node)
        slot_meta: dict = {}

        env_id_counter = 0
        max_plan_retries = max(
            1,
            int(self.mcts_cfg.get(
                "max_plan_retries_per_child",
                self.mcts_cfg.get("max_num_try_per_expansion", 1),
            )),
        )

        for path, leaf in leaf_paths:
            if leaf.is_terminal or env_id_counter >= pool_size:
                continue

            # Set curobo scene to this leaf's world state (sequential).
            self._recover_curobo_scene(leaf)

            children_created = 0
            for _ci in range(num_children):
                if env_id_counter >= pool_size:
                    break

                env_id = env_id_counter

                # Restore leaf's sim state into this env slot.
                # Single-env snapshots (extracted from parallel execution) are
                # stored at src_env_id=0; pass src_env_id=0 so restore() reads
                # the correct row when writing to an arbitrary env_id.
                state_token = leaf.snapshot["state_token"]
                snap_size = _snap_num_envs(state_token)
                src_env_id = 0 if snap_size == 1 else env_id
                t0 = time.perf_counter()
                self.simulator.restore(
                    self.task_env,
                    state_token,
                    env_id=env_id,
                    src_env_id=src_env_id,
                    close_gripper_on_restore=bool(leaf.snapshot.get("grasped_names")),
                )
                recover_dur = time.perf_counter() - t0

                # Pop a planned child from cache.
                child_node = None
                for _attempt in range(max_plan_retries):
                    child_node = self._pop_cached_plan(leaf)
                    if child_node is None:
                        child_node = TreeNode(
                            term_reason="No cached plan available for expansion."
                        )
                        child_node.is_terminal = True
                        child_node.is_invalid = True
                        child_node.value = -1
                        child_node.visit_count = 10000
                    if not child_node.is_invalid:
                        break

                if child_node is None or child_node.is_invalid:
                    child_node.parent = leaf
                    child_node.depth = self._get_depth(leaf) + 1
                    leaf.add_child(child_node)
                    continue

                self._add_timing(child_node, "recover_simulator_s", recover_dur)

                # Plan execution phases for this skill + env slot.
                skill_id = child_node.planning_params.get("skill")
                if skill_id not in ("pick", "place_with_drop", "rotate_held_object"):
                    child_node.is_terminal = True
                    child_node.is_invalid = True
                    child_node.term_reason = (
                        f"parallel expansion: unsupported skill {skill_id!r}"
                    )
                    child_node.parent = leaf
                    child_node.depth = self._get_depth(leaf) + 1
                    leaf.add_child(child_node)
                    continue

                t0 = time.perf_counter()
                phases = plan_phases_for_skill(
                    skill_id,
                    node=leaf,
                    child_node=child_node,
                    simulator=self.simulator,
                    env=self.task_env,
                    curobo=self.curobo,
                    rlpolicy=self.rlpolicy,
                    env_id=env_id,
                )
                self._add_timing(
                    child_node, "curobo_plan_s", time.perf_counter() - t0
                )

                if phases is None:
                    child_node.is_terminal = True
                    child_node.is_invalid = True
                    child_node.term_reason = "curobo planning failed in parallel expansion"
                    child_node.parent = leaf
                    child_node.depth = self._get_depth(leaf) + 1
                    leaf.add_child(child_node)
                    continue

                slot = SlotState(
                    env_id=env_id,
                    child_node=child_node,
                    phases=phases,
                )
                slots.append(slot)
                slot_meta[env_id] = (path, leaf, child_node)
                env_id_counter += 1
                children_created += 1

            if children_created == 0:
                per_leaf_dead_end[id(leaf)] = True
                leaf.dead_end_count += 1
            self.partial_plan_cache.pop(leaf, None)

        if not slots:
            return [
                (path, leaf, list(per_leaf_new[id(leaf)]))
                for path, leaf in leaf_paths
            ]

        # -----------------------------------------------------------------
        # Execute all slots in parallel (batch physics stepping).
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        slots = run_parallel_slots(
            slots,
            simulator=self.simulator,
            env=self.task_env,
            curobo=self.curobo,
            rlpolicy=self.rlpolicy,
        )
        execute_dur = time.perf_counter() - t0
        logger.debug(
            f"[parallel] executed {len(slots)} slots in {execute_dur:.2f}s"
        )

        # -----------------------------------------------------------------
        # Post-execution: snapshot all envs at once, then collect per-slot.
        # -----------------------------------------------------------------
        full_snap = self.simulator.snapshot(self.task_env)

        for slot in slots:
            env_id = slot.env_id
            path, leaf, child_node = slot_meta[env_id]
            depth = self._get_depth(leaf) + 1

            if slot.terminated:
                child_node.is_terminal = True
                child_node.is_invalid = True
                child_node.term_reason = slot.terminate_reason
                child_node.parent = leaf
                child_node.depth = depth
                leaf.add_child(child_node)
                continue

            # Collect scene state from this env slot.
            t0 = time.perf_counter()
            (
                next_rgbs, next_pcds, next_scene_mesh, next_object_meshes,
                _, _, _, _,
                object_center_3d, object_range_3d, object_range_2d,
                object_axis_annotation, grasped_names, collided_names, _,
            ) = self.simulator.collect_scene_state(
                self.task_env,
                views=self.view_names,
                env_id=env_id,
                need_object_range_2d=self.need_object_range_2d,
            )
            self._add_timing(
                child_node, "collect_scene_state_s", time.perf_counter() - t0
            )

            t0 = time.perf_counter()
            child_node.observations = {
                "image_all_views": self._encode_rgbs(next_rgbs),
                "pcd_all_views": self._encode_pcds(next_pcds),
            }
            self._add_timing(
                child_node, "encode_observations_s", time.perf_counter() - t0
            )

            child_node.parent = leaf
            child_node.snapshot = {
                "scene_mesh": next_scene_mesh,
                "object_meshes": next_object_meshes,
                "grasped_names": grasped_names[:1],
                "collided_names": collided_names,
                "object_center_3d": object_center_3d,
                "object_range_3d": object_range_3d,
                "object_range_2d": object_range_2d,
                "object_axis_annotation": object_axis_annotation,
            }
            child_node.depth = depth

            # Store a single-env snapshot for this slot so that future
            # restore(state_token, env_id=0) works correctly.
            t0 = time.perf_counter()
            self._snapshot_slot_to_node(child_node, full_snap, env_id)
            self._add_timing(
                child_node, "snapshot_simulator_s", time.perf_counter() - t0
            )

            leaf.add_child(child_node)
            per_leaf_new[id(leaf)].append(child_node)

        return [
            (path, leaf, list(per_leaf_new[id(leaf)]))
            for path, leaf in leaf_paths
        ]

    def _recover_curobo_scene(self, node: TreeNode) -> None:
        """Update curobo's obstacle map to match node's snapshot."""
        snap = node.snapshot
        q = self.simulator.get_joint_positions(self.task_env)
        self.set_scene(
            snap["scene_mesh"],
            snap["object_meshes"],
            q,
            snap["grasped_names"],
            snap.get("collided_names", []),
        )

    # ------------------------------------------------------------------
    # Overridden search loop
    # ------------------------------------------------------------------

    def search(
        self,
        task_env,
        worker_id: int = 0,
    ):
        logging.warning("Search begins! (parallel mode)")

        global_start_time = datetime.datetime.now()

        self.task_env = task_env
        self._init_rlpolicy_runtime()

        pool_size: int = getattr(task_env, "num_envs", 1)
        num_children: int = self.mcts_cfg["num_children_per_expand"]
        k = max(1, pool_size // num_children)

        obs, instruction, self.replay_reset_params = self._reset_env_until_valid_initial_state()
        self.object_names = self.simulator.get_object_names(self.task_env)

        (
            init_rgbs, init_pcds, scene_mesh, object_meshes,
            hs, ws, ixts, exts,
            object_center_3d, object_range_3d, object_range_2d,
            object_axis_annotation, grasped_names, collided_names, q,
        ) = self.simulator.collect_scene_state(
            self.task_env,
            views=self.view_names,
            need_object_range_2d=self.need_object_range_2d,
        )

        self.instruction = instruction
        self.hs = hs
        self.ws = ws
        self.ixts = ixts
        self.exts = exts

        self.set_scene(
            scene_mesh, object_meshes, q, grasped_names, collided_names,
            new_episode=True,
        )

        init_images_base64 = self._encode_rgbs(init_rgbs)
        init_pcds_base64 = self._encode_pcds(init_pcds)

        self.simulator.prepare_snapshot(self.task_env)

        root = TreeNode(
            observations={
                "image_all_views": init_images_base64,
                "pcd_all_views": init_pcds_base64,
            },
            parent=None,
            snapshot={
                "scene_mesh": scene_mesh,
                "object_meshes": object_meshes,
                "grasped_names": grasped_names,
                "collided_names": collided_names,
                "object_center_3d": object_center_3d,
                "object_range_3d": object_range_3d,
                "object_range_2d": object_range_2d,
                "object_axis_annotation": object_axis_annotation,
            },
            history_info={
                "history_traj": [],
                "reflection": None,
                "progress": None,
            },
        )
        self._snapshot_simulator(root)
        self.root = root

        final_value = 0
        num_sim_round = 0
        num_sim_attempts = 0
        success_paths: list = []

        for sim_idx in tqdm(
            range(self.mcts_cfg["n_simulations"]),
            desc=f"MCTS Parallel Simulations worker {worker_id}",
            leave=False,
        ):
            num_sim_attempts += 1
            start_time = datetime.datetime.now()
            logger.debug(
                f"=== MCTS Parallel Simulation {sim_idx+1}"
                f"/{self.mcts_cfg['n_simulations']} (k={k}) ==="
            )

            # Precompute actor plans for all current leaf nodes.
            self._populate_partial_plan_cache(root)

            # -----------------------------------------------------------------
            # 1) Select k leaves (sampling without replacement).
            # -----------------------------------------------------------------
            leaf_paths = self._select_multiple(root, k)

            # Handle terminal leaves immediately; collect non-terminal.
            non_terminal_leaf_paths = []
            for path, leaf in leaf_paths:
                if leaf.is_terminal:
                    self._backprop(path, leaf.value)
                    if self.mcts_cfg["num_children_per_expand"] == 1:
                        break
                else:
                    non_terminal_leaf_paths.append((path, leaf))

            if not non_terminal_leaf_paths:
                continue

            # -----------------------------------------------------------------
            # 2) Expand all leaves in parallel.
            # -----------------------------------------------------------------
            expand_results = self._expand_leaves_parallel(
                non_terminal_leaf_paths,
                pool_size=pool_size,
                num_children=num_children,
            )

            # -----------------------------------------------------------------
            # 3) Judge + backprop for each leaf's new children.
            # -----------------------------------------------------------------
            for path, leaf, new_children in expand_results:
                if leaf.dead_end_count > 0:
                    leaf.is_terminal = True
                    leaf.value = -1
                    leaf.visit_count = 10000
                    if self.mcts_cfg["backprop_dead_end"]:
                        self._backprop(path, -1)
                    if self.mcts_cfg["num_children_per_expand"] == 1:
                        break
                    continue

                if not new_children:
                    continue

                if self.mcts_cfg["rollout_mode"] == "judge_rollout":
                    pending = [c for c in new_children if self._needs_judge(c)]
                    judge_results = self._judge_nodes_parallel(pending)
                    for child, reward in judge_results:
                        child.judge_score = reward
                        child.reward_assigned = True
                        if reward >= 1.0:
                            child.is_terminal = True
                            if self.mcts_cfg["use_depth_reward"]:
                                depth_reward = (
                                    (self.mcts_cfg["max_depth"] - child.depth)
                                    / self.mcts_cfg["max_depth"]
                                )
                                reward += depth_reward
                        elif child.depth >= self.mcts_cfg["max_depth"]:
                            child.is_terminal = True
                        for _ in range(self.mcts_cfg["n_rollouts_per_node"]):
                            self._backprop(path + [child], reward)
                else:
                    for child in new_children:
                        reward = child.value
                        if reward >= 1.0:
                            child.is_terminal = True
                            if self.mcts_cfg["use_depth_reward"]:
                                depth_reward = (
                                    (self.mcts_cfg["max_depth"] - child.depth)
                                    / self.mcts_cfg["max_depth"]
                                )
                                reward += depth_reward
                        elif child.depth >= self.mcts_cfg["max_depth"]:
                            child.is_terminal = True
                        for _ in range(self.mcts_cfg["n_rollouts_per_node"]):
                            self._backprop(path + [child], reward)

                successful = [
                    c for c in new_children
                    if c.is_terminal and c.value >= 1.0
                ]
                if successful:
                    final_value = 1
                    for c in successful:
                        success_paths.append(path + [c])
                    target = self.mcts_cfg.get("target_success_count", 1)
                    if self.mcts_cfg["mode"] == "stop_if_any_success" and len(
                        success_paths
                    ) >= target:
                        break

            num_sim_round += 1
            end_time = datetime.datetime.now()
            logger.debug(
                f"MCTS Parallel Simulation {sim_idx+1} time: "
                f"{end_time - start_time}"
            )

            # Stop early if any success found (single-branch mode or stop_if_any).
            if (
                final_value == 1
                and (
                    self.mcts_cfg["num_children_per_expand"] == 1
                    or (
                        self.mcts_cfg["mode"] == "stop_if_any_success"
                        and len(success_paths) >= self.mcts_cfg.get(
                            "target_success_count", 1
                        )
                    )
                )
            ):
                break

        global_end_time = datetime.datetime.now()
        global_search_time = (global_end_time - global_start_time).total_seconds()
        logger.debug(f"Total MCTS parallel search time: {global_search_time}")

        search_output, node_trajectories = self._format_search_output(
            num_sim_round=num_sim_round,
            num_sim_attempts=num_sim_attempts,
            root=root,
            global_search_time=global_search_time,
            success_paths=success_paths,
        )

        return [search_output], [node_trajectories], [final_value]
