from __future__ import annotations
from typing import Optional, List
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)
import requests
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import math
from copy import deepcopy
import json
from imageio import imwrite
import random
from itertools import count

logger = logging.getLogger(__name__)

from tree_search.memory import MemoryRetriever

from tree_search.tree_node import (
    TreeNode, serialize_tree, deserialize_tree,
    GUIDANCE_HEADER, _format_guidance_block, _join_guidance_values,
)
from tree_search.mcts_judge import JudgeMixin



from functools import partial
from skills.place_plan import build_place_plan
from skills.skills import (
    # execute_grasp,
    execute_grasp_rlpolicy,
    # execute_grasp_rlpolicy_tmp,
    execute_place_skill,
    execute_move_held_object,
    execute_rotate_held_object,
    execute_rotate_held_object_rlpolicy,
    execute_move_without_place,
    execute_open_drawer,
    execute_close_drawer,
    execute_close_door,
    # execute_place_orientation_aware,
    # execute_place_orientation_aware_rlpolicy,
)

from tree_search.prompts.system_prompt import SYSTEM_PROMPT
from tree_search.prompts.action_prompt import GRASP_PROMPT, PLACE_PROMPT,PLACE_WITH_DROP_PROMPT, MOVE_HELD_OBJECT_PROMPT, ROTATE_HELD_OBJECT_PROMPT, ORIENTATION_AWARE_PLACE_PROMPT, PLACE_WITH_ORIENTATION_DROP_PROMPT, MOVE_WITHOUT_PLACE_PROMPT, GRIPPER_ORIENTATION_PLACE_PROMPT_PANDA, GRIPPER_ORIENTATION_PLACE_PROMPT_WIDOWX, OPEN_DRAWER_PROMPT, CLOSE_DRAWER_PROMPT, CLOSE_DOOR_PROMPT, COORDINATE_SYSTEM_TEXT
from tree_search.prompts.action_prompt import GRASP_DESC, PLACE_DESC, PLACE_WITH_DROP_DESC, MOVE_HELD_OBJECT_DESC, ROTATE_HELD_OBJECT_DESC, ORIENTATION_AWARE_PLACE_DESC, PLACE_WITH_ORIENTATION_DROP_DESC, MOVE_WITHOUT_PLACE_DESC, GRIPPER_ORIENTATION_PLACE_DESC, OPEN_DRAWER_DESC, CLOSE_DRAWER_DESC, CLOSE_DOOR_DESC
from tree_search.prompts.judge_prompt import (
    SUBGOAL_EVAL_PROMPT,
    SUBGOAL_EVAL_PROMPT_NO_SAFETY,
    PROGRESS_EVAL_PROMPT,
    PROGRESS_EVAL_PROMPT_NO_SAFETY,
)

from utils.image_utils import _encode_image, _encode_pcd, _decode_base64, take_even_n_frames
from utils.vis_utils import vis_all
# from utils.pcd_utils import get_3d_location_at_pointcloud_top_center, get_poke_locations
from utils.pcd_utils import get_3d_location_at_pointcloud_top_center, get_poke_locations
from utils.geometry_utils import q2R_wxyz
from utils.prompt_utils import make_view_list, make_visual_observation_placeholder, make_object_info, make_arti_parts_info, make_contact_info, make_skill_primitives_list, make_history_reflection, make_coordinate_system_info




class MonteCarloTreeSearch(JudgeMixin):
    """
    A class that implements basic MCTS with:
     - Selection (UCB-based)
     - Expansion (creating a single new child from the LLM if not terminal)
     - Rollout (simulate until final with ephemeral steps)
     - Backprop (propagate final reward up)
    
    Only the final reward from the LLM judge is used for backprop.
    """

    def __init__(
        self,
        cfg,
        vlm,
        curobo_server,
        # judge,
        rlpolicy=None,
        simulator=None,
    ):
        self.cfg = cfg
        self.mcts_cfg = cfg["mcts"]
        self.skill_cfg = cfg["skills"]

        selection_mode = self.mcts_cfg.get("selection_mode", "ucb")
        if selection_mode not in ("ucb", "random"):
            raise ValueError(
                f"Unknown selection_mode={selection_mode!r}. "
                f"Must be one of: ucb, random."
            )
        if self.mcts_cfg["rollout_mode"] == "no_rollout" and selection_mode != "random":
            raise ValueError(
                f"rollout_mode=no_rollout requires selection_mode=random "
                f"(got selection_mode={selection_mode!r}). UCB selection over "
                f"all-zero rewards is uninformative; set selection_mode=random "
                f"or change rollout_mode."
            )
        self.mcts_cfg["selection_mode"] = selection_mode

        ## Skills
        self.vlm = vlm
        self.curobo = curobo_server
        # self.judge = judge
        self.rlpolicy = rlpolicy
        self.actor_concurrency = max(1, int(self.mcts_cfg.get("actor_concurrency", 1)))
        self.judge_concurrency = max(1, int(self.mcts_cfg.get("judge_concurrency", 1)))
        self.partial_plan_cache = {}

        # self._call_tools_map = dict()
        # self.skill_list = ["pick", "place", "move_held_object", "rotate_held_object", "orientation_aware_place"]
        # self.skill_list = ["pick", "place", "orientation_aware_place"]
        # NOTE: "move_without_place" is opt-in via mcts.use_move_without_place=true.
        # Implementation lives in skills.py (execute_move_without_place),
        # action_prompt.py (MOVE_WITHOUT_PLACE_PROMPT/DESC), conf/mcts/default.yaml
        # (prompts.actor_move_without_place block), and the registrations below;
        # only skill_list membership controls VLM Actor Phase 1 visibility.
        self.skill_list = ["pick", "place_with_drop", "place", "place_with_orientation", "place_with_orientation_drop", "place_with_gripper_orientation_drop"]
        if self.mcts_cfg.get("use_move_without_place", False):
            self.skill_list.append("move_without_place")
        # open/close drawer are opt-in via mcts.use_open_drawer / use_close_drawer (default off).
        if self.mcts_cfg.get("use_open_drawer", False):
            self.skill_list.append("open_drawer")
        if self.mcts_cfg.get("use_close_drawer", False):
            self.skill_list.append("close_drawer")
        if self.mcts_cfg.get("use_close_door", False):
            self.skill_list.append("close_door")
        # self.skill_list = ["pick", "place_with_drop", "rotate_held_object"]
        self._call_tools_map = {skill: None for skill in self.skill_list}
        assert self.skill_cfg["rl"]["use_grasp"], "Currently, we require using RL policy for grasping to ensure stable grasping performance. Please set skills.rl.use_grasp to True in the config."
        self._call_tools_map["pick"] = execute_grasp_rlpolicy  # grasp objects with RL policy
        # Place family now routes through the unified engine: registry -> execute_place_skill
        # (build_place_plan + validation) -> execute_place (dispatch). See skills/place_plan.py
        # and docs/place_pipeline_rfc.md. Behavior is unchanged (engine dispatches to the
        # existing per-skill bodies); the deduplication happens in later migration steps.
        self._call_tools_map["place_with_drop"] = partial(execute_place_skill, "place_with_drop")  # drop directly above target
        self._call_tools_map["place"] = partial(execute_place_skill, "place")  # place objects with RL policy
        self._call_tools_map["place_with_orientation"] = partial(execute_place_skill, "place_with_orientation")  # specific orientation via motion planner
        self._call_tools_map["place_with_orientation_drop"] = partial(execute_place_skill, "place_with_orientation_drop")  # align orientation then directly drop
        self._call_tools_map["place_with_gripper_orientation_drop"] = partial(execute_place_skill, "place_with_gripper_orientation_drop")  # align GRIPPER axis then directly drop
        self._call_tools_map["move_without_place"] = execute_move_without_place  # move held object to target with desired orientation, no release
        if "open_drawer" in self.skill_list:
            self._call_tools_map["open_drawer"] = execute_open_drawer  # grasp handle + RL pull
        if "close_drawer" in self.skill_list:
            self._call_tools_map["close_drawer"] = execute_close_drawer  # push front + RL close
        if "close_door" in self.skill_list:
            self._call_tools_map["close_door"] = execute_close_door  # push door + RL close (revolute)
        # self._call_tools_map["move_held_object"] = execute_move_held_object  # move held object
        # if self.skill_cfg["rl"]["use_pose"]:
        #    self._call_tools_map["rotate_held_object"] = execute_rotate_held_object_rlpolicy  # rotate held object with RL policy
            # self._call_tools_map["orientation_aware_place"] = execute_place_orientation_aware_rlpolicy  # orientation-aware place with RL policy
        # else:
        #    self._call_tools_map["rotate_held_object"] = execute_rotate_held_object  # rotate held object with motion planner
            # self._call_tools_map["orientation_aware_place"] = execute_place_orientation_aware  # orientation-aware place with motion planner
        self.skill_prompts = {
            "pick": GRASP_PROMPT,
            "place_with_drop": PLACE_WITH_DROP_PROMPT,
            "place": PLACE_PROMPT,
            #"move_held_object": MOVE_HELD_OBJECT_PROMPT,
            #"rotate_held_object": ROTATE_HELD_OBJECT_PROMPT,
            "place_with_orientation": ORIENTATION_AWARE_PLACE_PROMPT,
            "place_with_orientation_drop": PLACE_WITH_ORIENTATION_DROP_PROMPT,
            "move_without_place": MOVE_WITHOUT_PLACE_PROMPT,
        }
        # place_with_gripper_orientation_drop: the gripper-axis meaning differs per robot, so
        # pick the matching prompt from the simulator's robot_name (default panda).
        self.skill_prompts["place_with_gripper_orientation_drop"] = (
            GRIPPER_ORIENTATION_PLACE_PROMPT_WIDOWX
            if "widowx" in (getattr(simulator, "robot_name", "") or "").lower()
            else GRIPPER_ORIENTATION_PLACE_PROMPT_PANDA
        )
        self.skill_desc = {
            "pick": GRASP_DESC,
            "place_with_drop": PLACE_WITH_DROP_DESC,
            "place": PLACE_DESC,
            #"move_held_object": MOVE_HELD_OBJECT_DESC,
            #"rotate_held_object": ROTATE_HELD_OBJECT_DESC,
            "place_with_orientation": ORIENTATION_AWARE_PLACE_DESC,
            "place_with_orientation_drop": PLACE_WITH_ORIENTATION_DROP_DESC,
            "move_without_place": MOVE_WITHOUT_PLACE_DESC,
            "place_with_gripper_orientation_drop": GRIPPER_ORIENTATION_PLACE_DESC,
        }
        if "open_drawer" in self.skill_list:
            self.skill_prompts["open_drawer"] = OPEN_DRAWER_PROMPT
            self.skill_desc["open_drawer"] = OPEN_DRAWER_DESC
        if "close_drawer" in self.skill_list:
            self.skill_prompts["close_drawer"] = CLOSE_DRAWER_PROMPT
            self.skill_desc["close_drawer"] = CLOSE_DRAWER_DESC
        if "close_door" in self.skill_list:
            self.skill_prompts["close_door"] = CLOSE_DOOR_PROMPT
            self.skill_desc["close_door"] = CLOSE_DOOR_DESC

        ## Simulator
        self.simulator = simulator
        # View names for multi-view observations
        self.view_names = self.cfg["simulator"].get("view_names", ["front", "top"])
        # Views exposed to VLM: keep only the first front camera ("front"), hide additional front_* views.
        self.vlm_view_names = [view for view in self.view_names if not str(view).startswith("front_")]
        if not self.vlm_view_names:
            self.vlm_view_names = list(self.view_names)
        self.need_object_range_2d = self._should_collect_object_range_2d()
        # Object names discovered from the simulator (table excluded)
        self.object_names: List[str] = []

        ## Initialize key-value memory retriever
        self.memory: Optional[MemoryRetriever] = None
        self.memory_inject = {"actor_phase_1": True, "actor_phase_2": True, "judge": True}
        memory_cfg = self.cfg.get("memory", None)
        if memory_cfg is not None and bool(memory_cfg.get("use_memory", True)):
            inject_cfg = memory_cfg.get("inject", {}) or {}
            for key in self.memory_inject:
                if key in inject_cfg:
                    self.memory_inject[key] = bool(inject_cfg[key])

            def _as_path_list(raw):
                raw = raw or []
                try:
                    return [str(p) for p in list(raw)]
                except TypeError:
                    return [str(raw)] if raw else []

            # memory_source picks the library: general (memory_paths, generalizable
            # rules), task (task_memory_paths, task-specific strategies), or both.
            memory_source = str(memory_cfg.get("memory_source", "general")).lower()
            if memory_source not in ("general", "task", "both"):
                raise ValueError(
                    f"memory.memory_source must be one of general|task|both, got {memory_source!r}"
                )
            memory_paths = []
            if memory_source in ("general", "both"):
                memory_paths += _as_path_list(memory_cfg.get("memory_paths", []))
            if memory_source in ("task", "both"):
                memory_paths += _as_path_list(memory_cfg.get("task_memory_paths", []))

            if not memory_paths:
                logger.info("Memory disabled: no paths for memory_source=%s.", memory_source)
            else:
                try:
                    retriever = MemoryRetriever(
                        memory_paths=memory_paths,
                        llm_client=self.vlm,
                        default_key=memory_cfg.get("default_key", None),
                        max_output_tokens=int(memory_cfg.get("max_output_tokens", 256)),
                        top_k=int(memory_cfg.get("top_k", 2)),
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize memory retriever: {e}")
                    retriever = None
                if retriever is not None and retriever.memory:
                    self.memory = retriever
                    logger.info(
                        "✓ Memory retriever initialized with %d entries from %d file(s)",
                        len(retriever.memory),
                        len(memory_paths),
                    )
                else:
                    logger.info("Memory disabled: no usable entries loaded.")

    def _should_collect_object_range_2d(self) -> bool:
        prompts_cfg = self.mcts_cfg.get("prompts", {})
        if prompts_cfg is None:
            return False
        try:
            prompt_values = prompts_cfg.values()
        except Exception:
            return False
        for prompt_cfg in prompt_values:
            try:
                use_2d = bool(prompt_cfg.get("use_2d_range", False))
            except Exception:
                use_2d = False
            if use_2d:
                return True
        return False


    def _maybe_append_round_b_guidance(self, child_node, tag: str, text: str) -> str:
        """Append the Round-B Guidance block to ``text`` when ``tag`` was applied."""
        info = getattr(child_node, "memory_info", None)
        if not isinstance(info, dict):
            return text
        rb = info.get("round_b")
        if not isinstance(rb, dict):
            return text
        values = rb.get(f"values_for_{tag}") or []
        if not values:
            return text
        return text + _format_guidance_block(_join_guidance_values(values))


    def _snapshot_simulator(
        self,
        node: TreeNode,
    ):
        grasped_names = node.snapshot["grasped_names"]
        state_token = self.simulator.snapshot(self.task_env)
        node.snapshot.update({
            "state_token": state_token
        })

    def _collect_part_info(self) -> dict:
        """Unified per-part snapshot data, LIVE from the sim at snapshot time:
        {object_name: {part_key: {"kind": "drawer"|"door", "state": open/closed/partially open,
        "center": [3], "aabb": 6-tuple like object_range_3d}}} for every movable part.
        Geometry reflects the part's CURRENT opening (the object pcd doesn't move with the
        joint, so an open drawer's contents would land outside the static footprint); state ==
        skill success criterion (get_part_state reuses the RL goal_joint_pos_norm cutoffs).
        Single source for the actor/judge prompt part lines, the place-into-part target/judge
        AABB and the final-state re-check. Part keys only."""
        out: dict = {}
        try:
            names = self.simulator.get_articulated_object_names(self.task_env)
        except Exception:
            return out
        for name in names:
            try:
                part_keys = self.simulator.get_movable_part_keys(self.task_env, name)
            except Exception:
                continue
            per: dict = {}
            for part in part_keys:
                try:
                    kind = ("door" if self.simulator.get_part_joint_type(self.task_env, name, part) == "revolute"
                            else "drawer")
                    state = self.simulator.get_part_state(self.task_env, name, part)
                    rng = tuple(
                        float(v) for v in self.simulator.get_object_3d_range(self.task_env, name, part=part)
                    )
                    center = [0.5 * (rng[0] + rng[1]), 0.5 * (rng[2] + rng[3]), 0.5 * (rng[4] + rng[5])]
                    per[part] = {"kind": kind, "state": state, "center": center, "aabb": rng}
                except Exception:
                    continue
            if per:
                out[name] = per
        return out

    def _add_timing(self, node: TreeNode | None, key: str, delta_s: float) -> None:
        if node is None:
            return
        if not hasattr(node, "timing") or node.timing is None:
            node.timing = {}
        node.timing[key] = float(node.timing.get(key, 0.0)) + float(delta_s)

    def _set_timing(self, node: TreeNode | None, key: str, value_s: float) -> None:
        if node is None:
            return
        if not hasattr(node, "timing") or node.timing is None:
            node.timing = {}
        node.timing[key] = float(value_s)

    @staticmethod
    def _series_distribution_stats(values: list[float]) -> dict | None:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return None
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        return {
            "n_steps": int(arr.shape[0]),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(q50),
            "p25": float(q25),
            "p75": float(q75),
        }

    def _summarize_execution_error_stats(self, trajectory) -> dict | None:
        if not isinstance(trajectory, (list, tuple)) or len(trajectory) == 0:
            return None

        buckets = {
            "all": {"pos": [], "rot": []},
            "mode0": {"pos": [], "rot": []},
            "mode1": {"pos": [], "rot": []},
        }

        def _append_metric(metric_name: str, value, mode):
            if value is None:
                return
            try:
                fval = float(value)
            except (TypeError, ValueError):
                return
            if not np.isfinite(fval):
                return
            buckets["all"][metric_name].append(fval)
            if mode == 0:
                buckets["mode0"][metric_name].append(fval)
            elif mode == 1:
                buckets["mode1"][metric_name].append(fval)

        for step in trajectory:
            if not isinstance(step, dict):
                continue
            tracking = step.get("tracking")
            if not isinstance(tracking, dict):
                continue

            action_mode = None
            action = step.get("action")
            if isinstance(action, dict):
                action_mode = action.get("action_mode", None)
            if action_mode is None:
                action_mode = tracking.get("action_mode", None)
            try:
                action_mode = int(action_mode) if action_mode is not None else None
            except (TypeError, ValueError):
                action_mode = None

            _append_metric("pos", tracking.get("eef_pos_err_m"), action_mode)
            _append_metric("rot", tracking.get("eef_rot_err_rad"), action_mode)

        out = {}
        has_any = False
        for bucket_name, vals in buckets.items():
            pos_stats = self._series_distribution_stats(vals["pos"])
            rot_stats = self._series_distribution_stats(vals["rot"])
            out[bucket_name] = {
                "eef_pos_m": pos_stats,
                "eef_rot_rad": rot_stats,
            }
            if pos_stats is not None or rot_stats is not None:
                has_any = True
        return out if has_any else None

    def _summarize_timing(self, root: TreeNode) -> dict:
        agg: dict[str, dict[str, float]] = {}
        stack = [root]
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            timing = getattr(node, "timing", None)
            if not isinstance(timing, dict):
                continue
            for key, val in timing.items():
                if not isinstance(val, (int, float)):
                    continue
                stat = agg.setdefault(key, {"count": 0, "sum_s": 0.0, "max_s": 0.0})
                stat["count"] += 1
                stat["sum_s"] += float(val)
                stat["max_s"] = max(float(stat["max_s"]), float(val))

        out = {}
        for key, stat in agg.items():
            count = int(stat["count"])
            sum_s = float(stat["sum_s"])
            out[key] = {
                "count": count,
                "sum_s": sum_s,
                "mean_s": (sum_s / count) if count > 0 else 0.0,
                "max_s": float(stat["max_s"]),
            }
        return out


    def _recover_simulator(
        self,
        node: TreeNode,
    ):  
        snapshot = node.snapshot
        state_token = snapshot["state_token"]
        next_scene_mesh = snapshot["scene_mesh"]
        next_object_meshes = snapshot["object_meshes"]
        grasped_names = snapshot["grasped_names"]
        collided_names = snapshot["collided_names"]
        
        self.simulator.restore(
            self.task_env,
            state_token,
            close_gripper_on_restore=bool(grasped_names),
        )

        next_q = self.simulator.get_joint_positions(self.task_env)
        self.set_scene(
            next_scene_mesh, 
            next_object_meshes, 
            next_q, 
            grasped_names,
            collided_names
        )


    def set_scene(
            self, 
            scene_mesh, 
            obj_meshes, 
            q: list, 
            grasped_object_names, 
            colli_with_grasped_object_names,
            new_episode=False
        ):
        self.curobo.reset(
            payload={
                "scene_mesh": scene_mesh,
                "object_meshes": obj_meshes,
                "q": q,
                "grasped_object_names": grasped_object_names,
                "colli_with_grasped_object_names": colli_with_grasped_object_names,
                "new_episode": new_episode,
                # "h": self.h,
                # "w": self.w,
                # "ixt": self.ixt.tolist(),
                # "ext": self.ext.tolist()
        })

    def _init_rlpolicy_runtime(self):
        if not hasattr(self.rlpolicy, "init_agents"):
            raise RuntimeError("Unsupported rlpolicy type: missing init_agents interface.")
        self.rlpolicy.init_agents(simulator=self.simulator, task_env=self.task_env)

    def _settle_after_env_reset(self, settle_steps: int = 20) -> None:
        """Advance physics with zero actions so freshly reset rigid bodies settle."""
        steps = max(0, int(settle_steps))
        for _ in range(steps):
            zero_action = torch.zeros((self.task_env.num_envs, 7), device=self.task_env.device)
            self.simulator.step(self.task_env, zero_action, action_mode=1)

    def _validate_reset_object_bounds(
        self,
        reset_params: dict,
        *,
        abs_xy_max: float,
        abs_z_max: float,
    ) -> tuple[bool, str]:
        """Validate object root poses in replay_reset_params.physics.objects."""
        physics = reset_params.get("physics", {}) if isinstance(reset_params, dict) else {}
        objects = physics.get("objects", {}) if isinstance(physics, dict) else {}
        if not isinstance(objects, dict) or len(objects) == 0:
            return False, "reset_params.physics.objects is missing or empty"

        violations: list[str] = []
        for name, state in objects.items():
            pose = state.get("root_pose_local") if isinstance(state, dict) else None
            if pose is None or len(pose) < 3:
                violations.append(f"{name}: missing root_pose_local[:3]")
                continue
            x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
            if (not np.isfinite(x)) or (not np.isfinite(y)) or (not np.isfinite(z)):
                violations.append(f"{name}: non-finite pose xyz=({x}, {y}, {z})")
                continue
            if abs(x) >= abs_xy_max or abs(y) >= abs_xy_max or abs(z) >= abs_z_max:
                violations.append(
                    f"{name}: xyz=({x:.4f}, {y:.4f}, {z:.4f}) violates "
                    f"|x|<{abs_xy_max}, |y|<{abs_xy_max}, |z|<{abs_z_max}"
                )

        if violations:
            preview = "; ".join(violations[:3])
            if len(violations) > 3:
                preview += f"; ... (+{len(violations) - 3} more)"
            return False, preview
        return True, ""

    def _validate_reset_basket_upright(
        self,
        reset_params: dict,
        *,
        tilt_deg_tol: float,
    ) -> tuple[bool, str]:
        """Reject reset states where basket has non-yaw tilt after settle."""
        physics = reset_params.get("physics", {}) if isinstance(reset_params, dict) else {}
        objects = physics.get("objects", {}) if isinstance(physics, dict) else {}
        if not isinstance(objects, dict):
            return False, "reset_params.physics.objects is missing or invalid"

        basket_state = objects.get("basket")
        if basket_state is None:
            return True, ""
        if not isinstance(basket_state, dict):
            return False, "basket state is invalid"

        pose = basket_state.get("root_pose_local")
        if pose is None or len(pose) < 7:
            return False, "basket: missing root_pose_local[:7]"

        quat = np.asarray(pose[3:7], dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(quat)):
            return False, f"basket: non-finite quaternion={quat.tolist()}"

        quat_norm = float(np.linalg.norm(quat))
        if quat_norm <= 1e-12:
            return False, "basket: quaternion norm is zero"
        qw, qx, qy, qz = quat / quat_norm

        # World-frame direction of the basket's local +Z axis.
        z_axis_world = np.array(
            [
                2.0 * (qx * qz + qw * qy),
                2.0 * (qy * qz - qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
            dtype=np.float64,
        )
        z_axis_world = z_axis_world / max(float(np.linalg.norm(z_axis_world)), 1e-12)
        tilt_deg = float(np.degrees(np.arccos(np.clip(z_axis_world[2], -1.0, 1.0))))
        if tilt_deg > tilt_deg_tol:
            return (
                False,
                f"basket tilt {tilt_deg:.2f}deg exceeds tolerance {tilt_deg_tol:.2f}deg",
            )
        return True, ""

    def _reset_env_until_valid_initial_state(self) -> tuple[dict, str, dict]:
        """Reset env repeatedly until object poses satisfy configured bounds."""
        max_attempts = max(1, int(self.mcts_cfg.get("initial_state_max_reset_attempts", 8)))
        settle_steps = int(self.mcts_cfg.get("initial_state_settle_steps", 20))
        abs_xy_max = float(self.mcts_cfg.get("initial_state_abs_xy_max", 1.0))
        abs_z_max = float(self.mcts_cfg.get("initial_state_abs_z_max", 5.0))
        require_upright_basket = bool(self.mcts_cfg.get("initial_state_require_upright_basket", True))
        basket_tilt_deg_tol = float(self.mcts_cfg.get("initial_state_basket_tilt_deg_tol", 3.0))
        if abs_xy_max <= 0.0 or abs_z_max <= 0.0:
            raise ValueError(
                f"Invalid initial-state bounds: initial_state_abs_xy_max={abs_xy_max}, "
                f"initial_state_abs_z_max={abs_z_max}"
            )
        if basket_tilt_deg_tol < 0.0:
            raise ValueError(
                f"Invalid initial_state_basket_tilt_deg_tol={basket_tilt_deg_tol}; must be >= 0."
            )

        last_reason = ""
        for attempt in range(1, max_attempts + 1):
            obs, instruction = self.simulator.reset(self.task_env, reset_level="env")
            self._settle_after_env_reset(settle_steps=settle_steps)
            self.simulator.set_scene_pose(self.task_env)
            reset_params = self.simulator.collect_replay_reset_params(self.task_env)
            ok, reason = self._validate_reset_object_bounds(
                reset_params,
                abs_xy_max=abs_xy_max,
                abs_z_max=abs_z_max,
            )
            if ok and require_upright_basket:
                ok, reason = self._validate_reset_basket_upright(
                    reset_params,
                    tilt_deg_tol=basket_tilt_deg_tol,
                )
            if ok:
                if attempt > 1:
                    logger.warning(
                        "Initial-state guard accepted reset on attempt %d/%d.",
                        attempt,
                        max_attempts,
                    )
                return obs, instruction, reset_params
            last_reason = reason
            logger.warning(
                "Initial-state guard rejected reset attempt %d/%d: %s",
                attempt,
                max_attempts,
                reason,
            )

        raise RuntimeError(
            "Failed to obtain a valid initial state after "
            f"{max_attempts} reset attempts. Last reason: {last_reason}"
        )


    def search(
        self,
        task_env,
        worker_id: int = 0,
    ):
        logging.warning("Search begins!")

        global_start_time = datetime.datetime.now()
        init_phase_t0 = time.perf_counter()

        self.task_env = task_env
        self._init_rlpolicy_runtime()

        obs, instruction, self.replay_reset_params = self._reset_env_until_valid_initial_state()
        self.object_names = self.simulator.get_object_names(self.task_env)
        # Expose per-object initial world poses (from replay_reset_params) on
        # simulator.args so skills can fetch them later. Used by the
        # straighten task to keep place-xy locked to the object's start pose.
        try:
            _init_objs = (self.replay_reset_params or {}).get("physics", {}).get("objects", {}) or {}
            self.simulator.args.init_object_poses = {
                name: list(entry.get("root_pose_local", []))
                for name, entry in _init_objs.items()
                if isinstance(entry, dict)
            }
        except Exception:
            self.simulator.args.init_object_poses = {}

        init_rgbs, init_pcds, scene_mesh, object_meshes, hs, ws, ixts, exts, object_center_3d, object_range_3d, object_range_2d, object_axis_annotation, object_upright_status, grasped_names, collided_names, q = self.simulator.collect_scene_state(
            self.task_env, views=self.view_names, need_object_range_2d=self.need_object_range_2d
        )
        self.instruction = instruction
        self.hs = hs
        self.ws = ws
        self.ixts = ixts
        self.exts = exts  # world2cam

        self.set_scene(scene_mesh, object_meshes, q, grasped_names, collided_names, new_episode=True)

        init_images_base64 = self._encode_rgbs(init_rgbs)
        init_pcds_base64 = self._encode_pcds(init_pcds)
        # init_image = self._get_view_image(init_images_base64)
        # init_pcd = self._get_view_image(init_pcds_base64)

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
                "object_upright_status": object_upright_status,
                # Unified per-part data (state + open-state geometry), see _collect_part_info.
                "part_info_3d": self._collect_part_info(),
            },
            history_info={
                "history_traj": [],
                "reflection": None,
                "progress": None,
            }
        )
        self._snapshot_simulator(root)
        self.root = root

        init_phase_s = time.perf_counter() - init_phase_t0
        mcts_loop_t0 = time.perf_counter()

        final_value = 0
        num_sim_round = 0
        num_sim_attempts = 0
        success_paths: list[list[TreeNode]] = []

        for sim_idx in tqdm(range(self.mcts_cfg["n_simulations"]), desc=f"MCTS Simulations worker {worker_id}", leave=False):
            num_sim_attempts += 1
            start_time = datetime.datetime.now()
            logger.debug(f"=== MCTS Simulation {sim_idx+1}/{self.mcts_cfg['n_simulations']} ===")

            # Precompute partial plans for all current leaf nodes.
            self._populate_partial_plan_cache(root)

            # 1) SELECTION: get a path from root to a leaf or expandable node
            path = self._select(root)
            leaf_node = path[-1]

            # If leaf is already terminal, no expansion needed; just backprop its known value
            if leaf_node.is_terminal:
                # NOTE: A node is marked as terminal node if (1)it results in success in simulation (2)it reach max depth
                # NOTE: Thus, a terminal node is judged right after it's generated.
                # assert leaf_node.visit_count == 1, "A leaf node should have been visited in simulation!"
                # Backprop the existing value
                self._backprop(path, leaf_node.value)
                if self.mcts_cfg["num_children_per_expand"] > 1:
                    continue
                else:  ## for single branch setting, we just stop here
                    break

            # 2) EXPANSION: expand the leaf node if possible
            # this expansion is the most important part of MCTS visual search
            new_children = self._expand(
                leaf_node, 
                num_children=self.mcts_cfg["num_children_per_expand"]
            )
            
            if leaf_node.dead_end_count > 0:
                leaf_node.is_terminal = True
                leaf_node.value = -1
                reward = -1
                leaf_node.visit_count = 10000
                if self.mcts_cfg["backprop_dead_end"]:
                    self._backprop(path, reward)  # always reaches dead end, failing to produce valid child node, stop visiting this node!
                
                if self.mcts_cfg["num_children_per_expand"] == 1:
                    break

            ## Reward for the new children. Run in both single-branch and
            ## multi-branch settings so single-child expansions still receive a
            ## success/fail signal (e.g. max_depth=1, num_children_per_expand=1).
            rollout_mode = self.mcts_cfg["rollout_mode"]
            if rollout_mode == "judge_rollout":
                pending = [child for child in new_children if self._needs_judge(child)]
                judge_results = self._judge_nodes_parallel(pending)
                for child, reward in judge_results:
                    child.judge_score = reward
                    child.reward_assigned = True
                    if reward >= 1.0:  ## Judge successful
                        child.is_terminal = True
                        if self.mcts_cfg["use_depth_reward"]:
                            depth_reward = (self.mcts_cfg["max_depth"] - child.depth) / self.mcts_cfg["max_depth"]
                            reward += depth_reward
                    elif child.depth >= self.mcts_cfg["max_depth"]:  ## Judge Unsuccessful &  Reach max depth
                        child.is_terminal = True
                    for _ in range(self.mcts_cfg["n_rollouts_per_node"]):
                        self._backprop(path + [child], reward)
            elif rollout_mode == "no_rollout":
                # Backprop 0 by default; only call progress-only judge when the
                # newly expanded child has reached max_depth (end-of-traj).
                for child in new_children:
                    if not self._needs_judge(child):
                        reward = child.value
                    elif child.depth >= self.mcts_cfg["max_depth"]:
                        reward = self._dispatch_terminal_judge(child)
                        child.judge_score = reward
                        child.reward_assigned = True
                    else:
                        reward = 0.0
                    if reward >= 1.0:
                        child.is_terminal = True
                        if self.mcts_cfg["use_depth_reward"]:
                            depth_reward = (self.mcts_cfg["max_depth"] - child.depth) / self.mcts_cfg["max_depth"]
                            reward += depth_reward
                    elif child.depth >= self.mcts_cfg["max_depth"]:
                        child.is_terminal = True
                    for _ in range(self.mcts_cfg["n_rollouts_per_node"]):
                        self._backprop(path + [child], reward)
            elif rollout_mode == "standard_rollout":
                # Roll forward up to rollout_max_steps additional steps from
                # each new child (throwaway, not attached to tree); call
                # terminal judge at the rollout terminal; backprop the scalar
                # reward to path + [child]. Rollout reward is ONLY a backprop
                # signal — it cannot mark the in-tree child as terminal-success.
                # Success is only declared when an in-tree node reaches
                # max_depth and its own terminal judge returns reward >= 1.0
                # (handled by the depth-based is_terminal below + the
                # `successful_children` collection in the search loop epilogue).
                for child in new_children:
                    if not self._needs_judge(child):
                        reward = child.value
                    else:
                        reward = self._standard_rollout_from_child(child)
                    if child.depth >= self.mcts_cfg["max_depth"]:
                        child.is_terminal = True
                    for _ in range(self.mcts_cfg["n_rollouts_per_node"]):
                        self._backprop(path + [child], reward)
            else:
                raise ValueError(
                    f"Unknown rollout_mode={rollout_mode!r}. "
                    f"Must be one of: judge_rollout, no_rollout, standard_rollout."
                )
                    
            num_sim_round += 1
            
            successful_children = [c for c in new_children if c.is_terminal and c.value >= 1.0]
            if successful_children:
                final_value = 1
                # Record all successful child paths in this expansion.
                for c in successful_children:
                    success_paths.append(path + [c])
                target = self.mcts_cfg.get("target_success_count", 1)
                if self.mcts_cfg["num_children_per_expand"] == 1 or (
                    self.mcts_cfg["mode"] == "stop_if_any_success" and len(success_paths) >= target
                ):
                    break
                    
            end_time = datetime.datetime.now()
            search_time = end_time - start_time
            logger.debug(f"MCTS Simulation {sim_idx+1}/{self.mcts_cfg['n_simulations']} time: {search_time}")

        mcts_loop_s = time.perf_counter() - mcts_loop_t0
        global_end_time = datetime.datetime.now()
        global_search_time = global_end_time - global_start_time
        global_search_time = global_search_time.total_seconds()
        logger.debug(f"Total MCTS search time: {global_search_time}")

        # Record the search output for this simulation
        search_output, node_trajectories = self._format_search_output(
            # system_prompt=self.system_prompt,
            num_sim_round=num_sim_round,
            num_sim_attempts=num_sim_attempts,
            root=root,
            global_search_time=global_search_time,
            success_paths=success_paths,
            init_phase_s=init_phase_s,
            mcts_loop_s=mcts_loop_s,
        )

        return [search_output], [node_trajectories], [final_value]

    def _get_view_image(self, image, view: Optional[str] = None):
        if view is None:
            view = self.view_names[0]
        return image[view]
    

    def _encode_rgbs(self, rgbs):
        return {name: _encode_image(Image.fromarray(img)) for name, img in rgbs.items()}
    
    def _encode_pcds(self, pcds):
        return {name: _encode_pcd(pcd) for name, pcd in pcds.items()}
    

    def _make_actor_prompt_system(
        self,
        node: TreeNode,
    ):
        prompt_cfg = self.mcts_cfg["prompts"]["actor_system"]

        object_center_3d = node.snapshot["object_center_3d"]
        object_range_3d = node.snapshot["object_range_3d"]
        object_range_2d = node.snapshot["object_range_2d"]
        grasped_names = node.snapshot["grasped_names"]
        # if node.has_attr("history_info") and node.history_info is not None:
        if node.parent is not None and node.history_info is not None:
            history_traj = node.history_info["history_traj"]
            reflection = node.history_info["reflection"]
        else:
            history_traj = None
            reflection = None

        view_list = make_view_list(self.vlm_view_names)
        visual_obs_placeholder = make_visual_observation_placeholder(self.vlm_view_names)
        obj_parts = obj_ids = None
        include_open_dir = prompt_cfg.get("use_open_dir", False)
        if prompt_cfg.get("use_arti_parts", False):
            _, obj_parts, obj_ids = self._build_arti_entity_map(
                node.snapshot, include_objects=True, include_open_dir=include_open_dir)
        object_information = make_object_info(
            obj_names=self.object_names,
            obj_center_3d=object_center_3d if prompt_cfg.get("use_3d_center", False) else None,
            obj_range_3d=object_range_3d if prompt_cfg.get("use_3d_range", False) else None,
            obj_range_2d=object_range_2d if prompt_cfg.get("use_2d_range", False) else None,
            obj_upright_status=node.snapshot["object_upright_status"]
                if prompt_cfg.get("use_upright_status", True) else None,
            obj_parts=obj_parts,
            obj_ids=obj_ids,
            obj_opening_dirs=self._collect_base_opening_dirs(node.snapshot) if include_open_dir else None,
        )
        contact_information = make_contact_info(grasped_names)
        skill_primitives = make_skill_primitives_list(self.skill_list, self.skill_desc)
        history_text, reflection_text = make_history_reflection(history_traj, reflection)

        system_prompt_text = SYSTEM_PROMPT.format(
            instruction=self.instruction,
            view_list=view_list,
            visual_observations=visual_obs_placeholder,
            object_information=object_information,
            contact_information=contact_information,
            history_information=history_text,
            skill_primitives=skill_primitives,
            reflection_instruction=reflection_text,
        )
        system_prompt_images = [
            node.observations["image_all_views"][view_name] for view_name in self.vlm_view_names
        ]

        return system_prompt_text, system_prompt_images
    
    def _build_arti_entity_map(self, snapshot: dict, include_objects: bool, include_axes: bool = False,
                               include_open_dir: bool = False):
        """Single flat `Asset` index space over articulated parts, optionally preceded by the
        top-level objects, built from a node's snapshot["part_info_3d"] (same timestamp as the
        rest of that prompt's object info — see _collect_part_info; the judge feeds its
        before/after snapshots through this too). include_objects=True (prompt use_arti_parts,
        place family + judge display): the VLM picks an object OR a part by one `Asset`
        integer. include_objects=False (prompt use_arti_only, open/close family): the list
        holds parts only. include_axes (prompt use_axis_annotation) additionally attaches the
        part's own-frame axis-annotation text (static, from the simulator store).
        Returns (entities, obj_parts, obj_ids):
          - entities[i] = ("object", name) | ("drawer"|"door", owner_name, part_key), in
            DISPLAY order: each object followed immediately by its own parts, so the rendered
            numbering is contiguous top-to-bottom. The prompt renders each entry with its
            index i; the phase-2 parse resolves the VLM's Asset integer back to the entity
            via this list (no numeric coupling between the flat index and the part key
            suffix). Ids are a prompt-render concept — the snapshot itself is id-free.
          - obj_parts = {owner_name: [part dict with 'id' = its index in `entities`]} for the
            prompt renderers.
          - obj_ids = {object_name: its index in `entities`} for make_object_info (the
            interleaved numbering shifts objects after the first arti asset)."""
        part_info = snapshot.get("part_info_3d") or {}
        entities: list = []
        obj_parts: dict = {}
        obj_ids: dict = {}

        def _add_parts(name, per):
            parts = []
            for part, info in per.items():
                gidx = len(entities)
                axes = None
                if include_axes:
                    try:
                        axes = self.simulator.get_part_axis_annotation(self.task_env, name, part)
                    except Exception:
                        axes = None
                opening = None
                if include_open_dir:
                    try:
                        opening = [float(v) for v in self.simulator.get_opening_dir_world(self.task_env, name, part)]
                    except Exception:
                        opening = None
                rng = info["aabb"]
                entities.append((info["kind"], name, part))
                parts.append({
                    "kind": info["kind"], "id": gidx, "state": info["state"],
                    "center": list(info["center"]),
                    "aabb_min": [rng[0], rng[2], rng[4]],
                    "aabb_max": [rng[1], rng[3], rng[5]],
                    "axes": axes,
                    "opening": opening,
                })
            if parts:
                obj_parts[name] = parts

        if include_objects:
            for name in self.object_names:
                obj_ids[name] = len(entities)
                entities.append(("object", name))
                per = part_info.get(name)
                if per:
                    _add_parts(name, per)
        else:
            for name, per in part_info.items():
                _add_parts(name, per)
        return entities, obj_parts, obj_ids

    def _collect_base_opening_dirs(self, snapshot: dict):
        """{arti object name -> world-frame unit opening dir of its BASE (= cavity/open face
        direction, live from the current root pose)} for prompt display (use_open_dir).
        Arti objects only — a rigid object's +z default opening is prompt noise."""
        out = {}
        for name in (snapshot.get("part_info_3d") or {}):
            try:
                out[name] = [float(v) for v in self.simulator.get_opening_dir_world(self.task_env, name, "base")]
            except Exception:
                continue
        return out or None

    def _make_actor_prompt_skill(
        self,
        node: TreeNode,
        child_node: TreeNode,
    ):
        planning_params = child_node.planning_params
        skill_name = planning_params["skill"]
        subgoal = planning_params["subgoal"]
        selected_view = planning_params["view"]
        grasped_names = node.snapshot["grasped_names"]

        skill_prompt_key = f"actor_{skill_name}"
        prompt_cfg = self.mcts_cfg["prompts"][skill_prompt_key]

        visual_obs_placeholder = make_visual_observation_placeholder([selected_view])
        # Arti-part exposure is flag-driven per skill prompt (conf/mcts prompts.actor_<skill>):
        #   use_arti_parts (place family): objects + parts in one flat Asset index space, parts
        #     hung under their owner in the full object list.
        #   use_arti_only (open/close family): parts only, rendered as a standalone part list.
        # Both store the entity map on the node; the phase-2 parse resolves the VLM's Asset
        # integer uniformly through it.
        include_axes = prompt_cfg.get("use_axis_annotation", False)
        include_open_dir = prompt_cfg.get("use_open_dir", False)
        obj_ids = None
        if prompt_cfg.get("use_arti_parts", False):
            asset_entities, obj_parts, obj_ids = self._build_arti_entity_map(
                node.snapshot, include_objects=True, include_axes=include_axes,
                include_open_dir=include_open_dir)
            child_node.asset_index_entities = asset_entities
        elif prompt_cfg.get("use_arti_only", False):
            asset_entities, obj_parts, _ = self._build_arti_entity_map(
                node.snapshot, include_objects=False, include_axes=include_axes,
                include_open_dir=include_open_dir)
            child_node.asset_index_entities = asset_entities
        else:
            obj_parts = None
        if prompt_cfg.get("use_arti_only", False):
            object_information = make_arti_parts_info(obj_parts)
        else:
            object_information = make_object_info(
                obj_names=self.object_names,
                obj_center_3d=node.snapshot["object_center_3d"] if prompt_cfg.get("use_3d_center", False) else None,
                obj_range_3d=node.snapshot["object_range_3d"] if prompt_cfg.get("use_3d_range", False) else None,
                obj_range_2d=node.snapshot["object_range_2d"] if prompt_cfg.get("use_2d_range", False) else None,
                obj_axis_annotation=node.snapshot["object_axis_annotation"] if prompt_cfg.get("use_axis_annotation", False) else None,
                obj_upright_status=node.snapshot["object_upright_status"]
                    if prompt_cfg.get("use_upright_status", True) else None,
                obj_parts=obj_parts,
                obj_ids=obj_ids,
                obj_opening_dirs=self._collect_base_opening_dirs(node.snapshot) if include_open_dir else None,
            )
        contact_information = make_contact_info(grasped_names)
        coordinate_system_info = make_coordinate_system_info(
            COORDINATE_SYSTEM_TEXT if prompt_cfg.get("use_coordinate_system", False) else None
        )

        skill_prompt_text = self.skill_prompts[skill_name].format(
            instruction=self.instruction,
            subgoal=subgoal,
            visual_observations=visual_obs_placeholder,
            object_information=object_information,
            contact_information=contact_information,
            coordinate_system=coordinate_system_info,
        )
        skill_prompt_text = self._maybe_append_round_b_guidance(child_node, "actor_phase_2", skill_prompt_text)
        skill_prompt_images = [node.observations["image_all_views"][selected_view]]

        return skill_prompt_text, skill_prompt_images
    

    def _make_judge_prompt_subgoal(
        self,
        child_node: TreeNode,
    ):
        node = child_node.parent

        prompt_cfg = self.mcts_cfg["prompts"]["judge"]

        object_center_3d = node.snapshot["object_center_3d"]
        object_range_3d = node.snapshot["object_range_3d"]
        object_range_2d = node.snapshot["object_range_2d"]
        grasped_names = node.snapshot["grasped_names"]

        object_center_3d_child = child_node.snapshot["object_center_3d"]
        object_range_3d_child = child_node.snapshot["object_range_3d"]
        object_range_2d_child = child_node.snapshot["object_range_2d"]
        grasped_names_child = child_node.snapshot["grasped_names"]

        visual_obs_placeholder = make_visual_observation_placeholder(self.vlm_view_names)
        # Judge part display (flag-gated like the actor): parts hung under their owner with the
        # same flat numbering as the place actor prompt; each side renders ITS OWN snapshot so
        # the before/after part states are honest.
        _judge_arti = prompt_cfg.get("use_arti_parts", False)
        _judge_axes = prompt_cfg.get("use_axis_annotation", False)
        _judge_open_dir = prompt_cfg.get("use_open_dir", False)
        obj_parts = obj_parts_child = obj_ids = obj_ids_child = None
        if _judge_arti:
            _, obj_parts, obj_ids = self._build_arti_entity_map(
                node.snapshot, include_objects=True, include_axes=_judge_axes,
                include_open_dir=_judge_open_dir)
            _, obj_parts_child, obj_ids_child = self._build_arti_entity_map(
                child_node.snapshot, include_objects=True, include_axes=_judge_axes,
                include_open_dir=_judge_open_dir)
        _open_dirs = self._collect_base_opening_dirs(node.snapshot) if _judge_open_dir else None
        object_info = make_object_info(
            obj_names=self.object_names,
            obj_center_3d=object_center_3d if prompt_cfg.get("use_3d_center", False) else None,
            obj_range_3d=object_range_3d if prompt_cfg.get("use_3d_range", False) else None,
            obj_range_2d=object_range_2d if prompt_cfg.get("use_2d_range", False) else None,
            obj_upright_status=node.snapshot["object_upright_status"]
                if prompt_cfg.get("use_upright_status", True) else None,
            obj_parts=obj_parts,
            obj_ids=obj_ids,
            obj_opening_dirs=_open_dirs,
        )
        object_info_child = make_object_info(
            obj_names=self.object_names,
            obj_center_3d=object_center_3d_child if prompt_cfg.get("use_3d_center", False) else None,
            obj_range_3d=object_range_3d_child if prompt_cfg.get("use_3d_range", False) else None,
            obj_range_2d=object_range_2d_child if prompt_cfg.get("use_2d_range", False) else None,
            obj_upright_status=child_node.snapshot["object_upright_status"]
                if prompt_cfg.get("use_upright_status", True) else None,
            obj_parts=obj_parts_child,
            obj_ids=obj_ids_child,
            obj_opening_dirs=_open_dirs,
        )
        contact_info = make_contact_info(grasped_names)
        contact_info_child = make_contact_info(grasped_names_child)
        coordinate_system_info = (
            f"***** Coordinate System Information *****\n{COORDINATE_SYSTEM_TEXT}\n\n"
            if prompt_cfg.get("use_coordinate_system", False) else ""
        )

        if int(self.mcts_cfg.get("max_depth", 1)) >= 2:
            _judge_template = SUBGOAL_EVAL_PROMPT
        else:
            _judge_template = SUBGOAL_EVAL_PROMPT_NO_SAFETY
        subgoal_eval_prompt_text = _judge_template.format(
            instruction=self.instruction,
            subgoal=child_node.planning_params["subgoal"],
            init_visual_observations=visual_obs_placeholder,
            init_object_information=object_info,
            init_contact_information=contact_info,
            current_visual_observations=visual_obs_placeholder,
            current_object_information=object_info_child,
            current_contact_information=contact_info_child,
            coordinate_system=coordinate_system_info,
        )
        subgoal_eval_prompt_text = self._maybe_append_round_b_guidance(child_node, "judge", subgoal_eval_prompt_text)
        subgoal_eval_prompt_images = [
            node.observations["image_all_views"][view_name] for view_name in self.vlm_view_names
        ] + [
            child_node.observations["image_all_views"][view_name] for view_name in self.vlm_view_names
        ]

        return subgoal_eval_prompt_text, subgoal_eval_prompt_images

    def _make_judge_prompt_progress(
        self,
        child_node: TreeNode
    ):
        root = self.root

        prompt_cfg = self.mcts_cfg["prompts"]["judge"]

        object_center_3d_root = root.snapshot["object_center_3d"]
        object_range_3d_root = root.snapshot["object_range_3d"]
        object_range_2d_root = root.snapshot["object_range_2d"]
        grasped_names_root = root.snapshot["grasped_names"]

        object_center_3d_child = child_node.snapshot["object_center_3d"]
        object_range_3d_child = child_node.snapshot["object_range_3d"]
        object_range_2d_child = child_node.snapshot["object_range_2d"]
        grasped_names_child = child_node.snapshot["grasped_names"]

        visual_obs_placeholder = make_visual_observation_placeholder(self.vlm_view_names)
        _judge_arti = prompt_cfg.get("use_arti_parts", False)
        _judge_axes = prompt_cfg.get("use_axis_annotation", False)
        _judge_open_dir = prompt_cfg.get("use_open_dir", False)
        obj_parts_root = obj_parts_child = obj_ids_root = obj_ids_child = None
        if _judge_arti:
            _, obj_parts_root, obj_ids_root = self._build_arti_entity_map(
                root.snapshot, include_objects=True, include_axes=_judge_axes,
                include_open_dir=_judge_open_dir)
            _, obj_parts_child, obj_ids_child = self._build_arti_entity_map(
                child_node.snapshot, include_objects=True, include_axes=_judge_axes,
                include_open_dir=_judge_open_dir)
        _open_dirs = self._collect_base_opening_dirs(root.snapshot) if _judge_open_dir else None
        object_info_root = make_object_info(
            obj_names=self.object_names,
            obj_center_3d=object_center_3d_root if prompt_cfg.get("use_3d_center", False) else None,
            obj_range_3d=object_range_3d_root if prompt_cfg.get("use_3d_range", False) else None,
            obj_range_2d=object_range_2d_root if prompt_cfg.get("use_2d_range", False) else None,
            obj_upright_status=root.snapshot["object_upright_status"]
                if prompt_cfg.get("use_upright_status", True) else None,
            obj_parts=obj_parts_root,
            obj_ids=obj_ids_root,
            obj_opening_dirs=_open_dirs,
        )
        object_info_child = make_object_info(
            obj_names=self.object_names,
            obj_center_3d=object_center_3d_child if prompt_cfg.get("use_3d_center", False) else None,
            obj_range_3d=object_range_3d_child if prompt_cfg.get("use_3d_range", False) else None,
            obj_range_2d=object_range_2d_child if prompt_cfg.get("use_2d_range", False) else None,
            obj_upright_status=child_node.snapshot["object_upright_status"]
                if prompt_cfg.get("use_upright_status", True) else None,
            obj_parts=obj_parts_child,
            obj_ids=obj_ids_child,
            obj_opening_dirs=_open_dirs,
        )
        contact_info_root = make_contact_info(grasped_names_root)
        contact_info_child = make_contact_info(grasped_names_child)

        history_traj = child_node.history_info["history_traj"]
        history_information, _ = make_history_reflection(history_traj, None)
        coordinate_system_info = (
            f"***** Coordinate System Information *****\n{COORDINATE_SYSTEM_TEXT}\n\n"
            if prompt_cfg.get("use_coordinate_system", False) else ""
        )

        if int(self.mcts_cfg.get("max_depth", 1)) >= 2:
            _progress_template = PROGRESS_EVAL_PROMPT
        else:
            _progress_template = PROGRESS_EVAL_PROMPT_NO_SAFETY
        progress_eval_prompt_text = _progress_template.format(
            instruction=self.instruction,
            init_visual_observations=visual_obs_placeholder,
            init_object_information=object_info_root,
            init_contact_information=contact_info_root,
            history_information=history_information,
            current_visual_observations=visual_obs_placeholder,
            current_object_information=object_info_child,
            current_contact_information=contact_info_child,
            coordinate_system=coordinate_system_info,
        )
        progress_eval_prompt_text = self._maybe_append_round_b_guidance(child_node, "judge", progress_eval_prompt_text)
        progress_eval_prompt_images = [
            root.observations["image_all_views"][view_name] for view_name in self.vlm_view_names
        ] + [
            child_node.observations["image_all_views"][view_name] for view_name in self.vlm_view_names
        ]

        return progress_eval_prompt_text, progress_eval_prompt_images
    

    def _plan_actor_for_node(
        self,
        current_node,
    ):
        actor_total_start = time.perf_counter()
        child_node = TreeNode()
        child_node.memory_info = {"round_a": None, "round_b": None}

        ## ===== Actor Phase 1: Subgoal Planning with VLM =====
        # make actor system prompt
        t0 = time.perf_counter()
        system_prompt_text, system_prompt_images = self._make_actor_prompt_system(current_node)
        self._add_timing(child_node, "actor_system_prompt_s", time.perf_counter() - t0)

        # ===== Memory Round A: pick a key tagged actor_phase_1 =====
        if self.memory is not None and self.memory_inject.get("actor_phase_1", True):
            previous_reflection = None
            if current_node.parent is not None and isinstance(current_node.history_info, dict):
                previous_reflection = current_node.history_info.get("reflection")
            t0 = time.perf_counter()
            round_a_log = self.memory.select_for_phase(
                phase="actor_phase_1",
                instruction=self.instruction,
                subgoal=None,
                hint_input=previous_reflection,
                candidate_tags=["actor_phase_1"],
                pose_info=current_node.snapshot.get("object_upright_status"),
            )
            self._add_timing(child_node, "memory_select_s", time.perf_counter() - t0)
            round_a_values = [e["value"] for e in round_a_log["selected_entries"]]
            round_a_log["applied_to"] = ["actor_phase_1"] if round_a_values else []
            child_node.memory_info["round_a"] = round_a_log
            if round_a_values:
                system_prompt_text = system_prompt_text + _format_guidance_block(
                    _join_guidance_values(round_a_values)
                )

        ## subgoal planning with VLM
        t0 = time.perf_counter()
        res_vlm = self.vlm.generate_single_thought(
            prompt={
                "text": system_prompt_text,
                "images": system_prompt_images,
            },
            phase="actor_system",
        )
        self._add_timing(child_node, "actor_system_vlm_s", time.perf_counter() - t0)
        terminate = res_vlm["skill_id"] is None or res_vlm["view_id"] is None or res_vlm["subgoal"] is None

        child_node.actor_prompt_1 = system_prompt_text
        child_node.actor_thinking_text_1 = res_vlm["thinking_text"]

        if terminate:
            term_reason = "VLM failed to generate valid skill/view/subgoal in Actor Phase 1."
            child_node.term_reason = term_reason
            self._set_timing(child_node, "actor_total_s", time.perf_counter() - actor_total_start)
            return child_node, terminate

        next_skill_id = res_vlm["skill_id"]
        next_skill = self.skill_list[next_skill_id]
        next_subgoal = res_vlm["subgoal"]
        pred_view_id = res_vlm["view_id"]
        pred_view = self.view_names[pred_view_id]

        child_node.planning_params = {
            "skill_id": next_skill_id,
            "skill": next_skill,
            "subgoal": next_subgoal,
            "view_id": pred_view_id,
            "view": pred_view,
        }

        # ===== Memory Round B: actor_phase_2 / judge =====
        if self.memory is not None:
            enabled_tags: list[str] = []
            if self.memory_inject.get("actor_phase_2", True):
                enabled_tags.append("actor_phase_2")
            if self.memory_inject.get("judge", True):
                enabled_tags.append("judge")
            if enabled_tags:
                t0 = time.perf_counter()
                round_b_log = self.memory.select_for_phase(
                    phase="actor_phase_2_or_judge",
                    instruction=self.instruction,
                    subgoal=next_subgoal,
                    hint_input=None,
                    candidate_tags=enabled_tags,
                    pose_info=current_node.snapshot.get("object_upright_status"),
                )
                self._add_timing(child_node, "memory_select_s", time.perf_counter() - t0)
                phase2_values: list[str] = []
                judge_values: list[str] = []
                if self.memory_inject.get("actor_phase_2", True):
                    phase2_values = [
                        e["value"] for e in round_b_log["selected_entries"]
                        if "actor_phase_2" in e["tags"]
                    ]
                if self.memory_inject.get("judge", True):
                    judge_values = [
                        e["value"] for e in round_b_log["selected_entries"]
                        if "judge" in e["tags"]
                    ]
                applied: list[str] = []
                if phase2_values:
                    applied.append("actor_phase_2")
                if judge_values:
                    applied.append("judge")
                round_b_log["applied_to"] = applied
                round_b_log["values_for_actor_phase_2"] = phase2_values
                round_b_log["values_for_judge"] = judge_values
                child_node.memory_info["round_b"] = round_b_log

        ## ===== Actor Phase 2: Skill Execution with VLM =====
        # make actor skill prompt
        t0 = time.perf_counter()
        skill_prompt_text, skill_prompt_images = self._make_actor_prompt_skill(current_node, child_node)
        self._add_timing(child_node, "actor_action_prompt_s", time.perf_counter() - t0)
        # action grounding with VLM
        t0 = time.perf_counter()
        res_vlm = self.vlm.generate_single_thought(
            prompt={
                "text": skill_prompt_text,
                "images": skill_prompt_images,
            },
            phase="actor_action",
            h=self.hs[pred_view],
            w=self.ws[pred_view],
        )
        self._add_timing(child_node, "actor_action_vlm_s", time.perf_counter() - t0)
        child_node.actor_prompt_2 = skill_prompt_text
        child_node.actor_thinking_text_2 = res_vlm.pop("thinking_text")
        child_node.action_params = res_vlm
        if "asset_id" in res_vlm:
            asset_entities = getattr(child_node, "asset_index_entities", None)
            if asset_entities is not None:
                # Uniform asset resolution over the flat entity index space (objects and/or
                # parts, per the prompt flags). An object resolves to its name; a part resolves
                # to its part key as `asset` plus the owning object as `asset_object`. asset_id
                # keeps the VLM's flat index verbatim (downstream only null-checks / logs it).
                aid = res_vlm.get("asset_id")
                if not isinstance(aid, int) or aid < 0 or aid >= len(asset_entities):
                    terminate = True
                    child_node.term_reason = f"Invalid asset id returned by VLM: {aid}"
                    logger.error(child_node.term_reason)
                    self._set_timing(child_node, "actor_total_s", time.perf_counter() - actor_total_start)
                    return child_node, terminate
                ent = asset_entities[aid]
                if ent[0] == "object":
                    child_node.action_params["asset"] = ent[1]
                else:  # drawer / door part
                    child_node.action_params["asset"] = ent[2]
                    child_node.action_params["asset_object"] = ent[1]
            else:
                try:
                    child_node.action_params["asset"] = self.object_names[res_vlm["asset_id"]]
                except:
                    terminate = True
                    child_node.term_reason = f"Invalid asset id returned by VLM: {res_vlm['asset_id']}"
                    logger.error(child_node.term_reason)
                    self._set_timing(child_node, "actor_total_s", time.perf_counter() - actor_total_start)
                    return child_node, terminate

            if next_skill == "pick" and child_node.action_params["asset"] == "table":
                terminate = True
                child_node.term_reason = "VLM tried to pick 'table'; not a graspable asset."
                logger.error(child_node.term_reason)
                self._set_timing(child_node, "actor_total_s", time.perf_counter() - actor_total_start)
                return child_node, terminate

        terminate = False
        self._set_timing(child_node, "actor_total_s", time.perf_counter() - actor_total_start)
        return child_node, terminate




    def _execute_plan_for_node(
        self,
        current_node,
        child_node: TreeNode,
    ):
        execute_total_start = time.perf_counter()
        ## ===== Call tools to generate next actions =====
        next_skill = child_node.planning_params["skill"]
        try:
            _tool_func = self._call_tools_map[next_skill]
        except KeyError as exc:
            terminate = True
            child_node.term_reason = f"Invalid skill id returned by VLM: {next_skill}"
            logger.error(child_node.term_reason)
            self._set_timing(child_node, "execute_total_s", time.perf_counter() - execute_total_start)
            return child_node, terminate

        t0 = time.perf_counter()
        tool_out = _tool_func(
            current_node,
            child_node,
            self.simulator,
            self.task_env,
            self.curobo,
            # self.ports,
            self.rlpolicy,
            data_collection_mode=self.mcts_cfg.get("data_collection_mode", False),
        )
        self._add_timing(child_node, "execute_tool_s", time.perf_counter() - t0)
        trajectory, terminate, term_reason = tool_out
        child_node.trajectory = trajectory
        child_node.execution_error_stats = self._summarize_execution_error_stats(trajectory)
        child_node.planner_chunk_count = int(getattr(child_node, "planner_chunk_count", 0) or 0)
        child_node.planner_split_count = int(getattr(child_node, "planner_split_count", 0) or 0)

        if terminate:
            child_node.term_reason = term_reason
            self._set_timing(child_node, "execute_total_s", time.perf_counter() - execute_total_start)
            return child_node, terminate

        t0 = time.perf_counter()
        next_rgbs, next_pcds, next_scene_mesh, next_object_meshes, _, _, _, _, object_center_3d, object_range_3d, object_range_2d, object_axis_annotation, object_upright_status, grasped_names, collided_names, _ = self.simulator.collect_scene_state(
            self.task_env, views=self.view_names, need_object_range_2d=self.need_object_range_2d
        )
        self._add_timing(child_node, "collect_scene_state_s", time.perf_counter() - t0)

        t0 = time.perf_counter()
        next_images_base64 = self._encode_rgbs(next_rgbs)
        next_pcds_base64 = self._encode_pcds(next_pcds)
        self._add_timing(child_node, "encode_observations_s", time.perf_counter() - t0)
        child_node.observations = {
            "image_all_views": next_images_base64,
            "pcd_all_views": next_pcds_base64,
        }
        child_node.parent = current_node
        child_node.snapshot = {
            "scene_mesh": next_scene_mesh,
            "object_meshes": next_object_meshes,
            "grasped_names": grasped_names[:1],
            "collided_names": collided_names,
            "object_center_3d": object_center_3d,
            "object_range_3d": object_range_3d,
            "object_range_2d": object_range_2d,
            "object_axis_annotation": object_axis_annotation,
            "object_upright_status": object_upright_status,
            # Unified per-part data (state + open-state geometry); sim is at post-execution here.
            "part_info_3d": self._collect_part_info(),
        }

        terminate = False
        self._set_timing(child_node, "execute_total_s", time.perf_counter() - execute_total_start)
        return child_node, terminate

    def _select(self, root: TreeNode, exclude: set = None) -> List[TreeNode]:
        """
        SELECTION PHASE:
        Traverse the tree from `root` down to a leaf or node with unvisited children
        using the UCB policy. Return the path of nodes from root to that node.

        If any node has an unvisited child, we stop once we reach that node
        (so expansion can happen there).

        Args:
            exclude: optional set of node id()s to skip during selection (used by
                     parallel MCTS to implement sampling-without-replacement across
                     simultaneously selected leaves).
        """
        path = []
        current = root

        while True:
            path.append(current)

            # If terminal, we're done
            # TODO: we set a node as a leaf node if it is generated from an action that leads to success
            # TODO: which corresponse to last action primitive in a successful branch
            if current.is_terminal:
                return path

            # If no children, we can't go deeper
            if not current.children:
                return path

            candidates = [child for child in current.children if not child.is_invalid]
            if exclude:
                candidates = [c for c in candidates if id(c) not in exclude]
            if not candidates:
                return path

            # Check if any child is unvisited (visit_count == 0).
            unvisited = [child for child in candidates if child.visit_count == 0]
            if unvisited:
                # We'll stop here and expand one of the unvisited children
                child = random.choice(unvisited)
                path.append(child)
                return path

            # Otherwise, pick the best child by UCB.
            # If avoid_terminal_in_selection is set, prefer non-terminal nodes
            # to avoid wasting simulations re-visiting already-evaluated terminal branches.
            if self.mcts_cfg.get("avoid_terminal_in_selection", False):
                non_terminal_candidates = [c for c in candidates if not c.is_terminal]
                ucb_pool = non_terminal_candidates if non_terminal_candidates else candidates
            else:
                ucb_pool = candidates
            if self.mcts_cfg["selection_mode"] == "random":
                current = random.choice(ucb_pool)
            else:
                current = max(ucb_pool, key=lambda c: self._ucb_score(current, c))


    def _collect_leaf_nodes(self, root: TreeNode) -> List[TreeNode]:
        leaves = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(node.children)
            else:
                leaves.append(node)
        return leaves

    def _needs_partial_plan(self, node: TreeNode) -> bool:
        return (not node.is_terminal) and (not node.is_invalid) and (len(node.children) == 0)

    def _plan_actor_attempt(self, current_node: TreeNode) -> TreeNode:
        child_node, terminate = self._plan_actor_for_node(current_node)
        if terminate:
            child_node.is_terminal = True
            child_node.is_invalid = True
            child_node.value = -1
            child_node.visit_count = 10000
        return child_node

    def _populate_partial_plan_cache(self, root: TreeNode) -> None:
        leaves = self._collect_leaf_nodes(root)
        leaf_set = set(leaves)
        for node in list(self.partial_plan_cache.keys()):
            if node not in leaf_set or not self._needs_partial_plan(node):
                self.partial_plan_cache.pop(node, None)

        max_plan_retries = max(
            1,
            int(self.mcts_cfg.get(
                "max_plan_retries_per_child",
                self.mcts_cfg.get("max_num_try_per_expansion", 1),
            )),
        )
        target_attempts = self.mcts_cfg["num_children_per_expand"] * max_plan_retries
        tasks = []
        for node in leaves:
            if not self._needs_partial_plan(node):
                continue
            cache_list = self.partial_plan_cache.setdefault(node, [])
            missing = target_attempts - len(cache_list)
            if missing > 0:
                tasks.extend([node] * missing)

        if not tasks:
            return

        if self.actor_concurrency <= 1 or len(tasks) == 1:
            for node in tasks:
                self.partial_plan_cache[node].append(self._plan_actor_attempt(node))
            return

        with ThreadPoolExecutor(max_workers=self.actor_concurrency) as ex:
            futures = {ex.submit(self._plan_actor_attempt, node): node for node in tasks}
            for fut in as_completed(futures):
                node = futures[fut]
                child_node = fut.result()
                self.partial_plan_cache[node].append(child_node)

    def _pop_cached_plan(self, node: TreeNode) -> Optional[TreeNode]:
        cached_plans = self.partial_plan_cache.get(node, [])
        if not cached_plans:
            return None
        valid_indices = [idx for idx, plan in enumerate(cached_plans) if not plan.is_invalid]
        if valid_indices:
            idx = random.choice(valid_indices)
            return cached_plans.pop(idx)
        return cached_plans.pop()

    def _has_valid_cached_plan(self, node: TreeNode) -> bool:
        cached_plans = self.partial_plan_cache.get(node, [])
        return any(not plan.is_invalid for plan in cached_plans)

    def _expand(
        self, 
        node: TreeNode, 
        num_children: int = 3,
    ):
        """
        EXPANSION PHASE:
        Generate multiple new children (num_children) from the LLM for `node` if not terminal.
        Each LLM call can produce a different text if you set some randomness (temperature).
        If the child text contains <final>, mark it terminal.
        Add all children to `node.children` and return them.
        """
        logging.warning(f"Expanding ...")
        depth = self._get_depth(node)

        max_plan_retries = max(
            1,
            int(self.mcts_cfg.get(
                "max_plan_retries_per_child",
                self.mcts_cfg.get("max_num_try_per_expansion", 1),
            )),
        )
        max_execute_retries = max(
            1,
            int(self.mcts_cfg.get("max_execute_retries_per_plan", 1)),
        )

        new_children = []
        for i in range(num_children):
            invalid_children = []
            plan_attempts = 0
            child_created = False
            while plan_attempts < max_plan_retries:
                planned_node = self._pop_cached_plan(node)
                if planned_node is None:
                    planned_node = TreeNode(term_reason="No cached plan available for expansion.")
                    planned_node.is_terminal = True
                    planned_node.is_invalid = True
                    planned_node.value = -1
                    planned_node.visit_count = 10000

                if planned_node.is_invalid:
                    plan_attempts += 1
                    invalid_children.append(planned_node)
                    if not self._has_valid_cached_plan(node):
                        plan_attempts = max_plan_retries
                    continue

                execute_attempts = 0
                while execute_attempts < max_execute_retries:
                    attempt_start = time.perf_counter()
                    # recover simulation states to the moment after the node is generated
                    t0 = time.perf_counter()
                    self._recover_simulator(node)
                    recover_dur = time.perf_counter() - t0
                    attempt_node = deepcopy(planned_node)
                    self._add_timing(attempt_node, "recover_simulator_s", recover_dur)
                    child_node, terminate = self._execute_plan_for_node(
                        current_node=node,
                        child_node=attempt_node,
                    )
                    self._set_timing(child_node, "expand_attempt_total_s", time.perf_counter() - attempt_start)
                    logger.debug(
                        f"EXPANSION child {i+1}/{num_children}: subgoal => "
                        f"{child_node.planning_params['subgoal']}"
                    )
                    
                    if terminate:
                        execute_attempts += 1
                        if execute_attempts >= max_execute_retries:
                            invalid_children.append(child_node)
                            plan_attempts += 1
                        continue
                    
                    # if success:
                    #     child_node.is_terminal = True
                    #     success_reward = 1
                    #     depth_reward = (self.max_depth - (depth + 1)) / self.max_depth
                    #     child_node.value = success_reward + depth_reward
                    # elif depth == self.max_depth - 1:
                    #     child_node.is_terminal = True
                    #     child_node.value = 0
                    # if depth == self.mcts_cfg["max_depth"] - 1:
                    #     child_node.is_terminal = True
                    #     child_node.value = 0

                    # store current states into node
                    t0 = time.perf_counter()
                    self._snapshot_simulator(child_node)
                    self._add_timing(child_node, "snapshot_simulator_s", time.perf_counter() - t0)

                    node.add_child(child_node)
                    child_node.depth = depth + 1
                    new_children.append(child_node)
                    child_created = True
                    break

                if child_created:
                    break

            if not child_created and invalid_children:
                chosen_invalid = random.choice(invalid_children)
                chosen_invalid.is_terminal = True
                chosen_invalid.is_invalid = True
                chosen_invalid.value = -1
                chosen_invalid.visit_count = 10000
                node.add_child(chosen_invalid)

        if len(new_children) == 0:
            node.dead_end_count += 1

        self.partial_plan_cache.pop(node, None)

        return new_children

    def _needs_judge(self, node: TreeNode) -> bool:
        return (
            (not node.reward_assigned)
            and (not node.is_invalid)
            and (not node.is_terminal)
            and (len(node.children) == 0)
        )

    def _judge_nodes_parallel(self, nodes: List[TreeNode]):
        if not nodes:
            return []
        if self.judge_concurrency <= 1 or len(nodes) == 1:
            return [(node, self._get_reward_from_judge(node)) for node in nodes]

        results = []
        with ThreadPoolExecutor(max_workers=self.judge_concurrency) as ex:
            futures = {ex.submit(self._get_reward_from_judge, node): node for node in nodes}
            for fut in as_completed(futures):
                node = futures[fut]
                reward = fut.result()
                results.append((node, reward))
        return results
    










    def _rollout_step(self, current_node: TreeNode) -> Optional[TreeNode]:
        """
        Take ONE rollout step from current_node WITHOUT attaching to the tree.
        Plans synchronously (bypassing partial_plan_cache), executes in the
        simulator, snapshots the resulting state. Returns the new node, or
        None if planning/execution failed.
        """
        planned_node = self._plan_actor_attempt(current_node)
        if planned_node.is_invalid:
            return None

        self._recover_simulator(current_node)
        attempt_node = deepcopy(planned_node)
        child_node, terminate = self._execute_plan_for_node(
            current_node=current_node,
            child_node=attempt_node,
        )
        if terminate:
            return None

        self._snapshot_simulator(child_node)
        child_node.depth = current_node.depth + 1
        return child_node




    def _standard_rollout_from_child(self, expanded_child: TreeNode) -> float:
        """
        STANDARD MCTS ROLLOUT (throwaway):
        From expanded_child, roll forward up to rollout_max_steps additional
        steps (or until depth >= max_depth) without attaching nodes to the
        tree. Call the terminal VLM progress judge at the rollout terminal and
        return the reward. Marks
        expanded_child.judge_score / reward_assigned so it is not re-judged
        later.
        """
        max_depth = self.mcts_cfg["max_depth"]
        rollout_max_steps = int(self.mcts_cfg.get("rollout_max_steps", 0))

        rollout_terminal = expanded_child
        for _ in range(rollout_max_steps):
            if rollout_terminal.depth >= max_depth:
                break
            next_node = self._rollout_step(rollout_terminal)
            if next_node is None:
                break
            rollout_terminal = next_node

        reward = self._dispatch_terminal_judge(rollout_terminal)
        expanded_child.judge_score = reward
        expanded_child.reward_assigned = True
        return reward


    def _backprop(self, path: List[TreeNode], reward: float) -> None:
        """
        BACKPROP PHASE:
        Update all nodes in `path` with the final rollout reward.
        We do an incremental mean update of .value, and increment .visit_count.
        """
        for node in path:
            node.increment_visits()
            node.update_value_mcts(reward)


    def _ucb_score(self, parent: TreeNode, child: TreeNode) -> float:
        """
        Standard UCB formula: Q + c * sqrt( ln(N_parent) / (N_child) )
        where Q = child.value / child.visit_count (if you prefer average),
        but here we directly store child.value as an incremental mean,
        so child.value itself can be used as Q.
        """
        c = self.mcts_cfg["c_puct"]

        # Exploitation term: child.value is already the average
        exploitation = child.value

        # Exploration term
        # (We add 1 to ensure no division by zero.)
        exploration = c * math.sqrt(
            math.log(parent.visit_count + 1) / (child.visit_count + 1)
        )

        return exploitation + exploration
    

    def _get_depth(self, node: TreeNode) -> int:
        """
        Get the depth of `node` by walking upward.
        """
        depth = 0
        current = node
        while current.parent is not None:
            current = current.parent
            depth += 1
        return depth

    @staticmethod
    def _flatten_task_primitives(task_entry) -> list:
        if not isinstance(task_entry, (list, tuple)) or len(task_entry) == 0:
            return []
        task_type = task_entry[0]
        if task_type == "seq":
            if len(task_entry) < 2 or not isinstance(task_entry[1], (list, tuple)):
                return []
            primitives = []
            for step in task_entry[1]:
                primitives.extend(MonteCarloTreeSearch._flatten_task_primitives(step))
            return primitives
        return [task_entry]

    @staticmethod
    def _name_in_categories(name, categories) -> bool:
        if not isinstance(name, str):
            return False
        if name in categories:
            return True
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0] in categories
        return False

    @staticmethod
    def _resolve_pose_status(pose_status: dict, name: str):
        if not isinstance(pose_status, dict) or not isinstance(name, str):
            return None, None
        if name in pose_status:
            return pose_status.get(name), name
        prefix = f"{name}_"
        matches = [key for key in pose_status.keys() if isinstance(key, str) and key.startswith(prefix)]
        if len(matches) == 1:
            return pose_status.get(matches[0]), matches[0]
        return None, None

    @staticmethod
    def _extract_straighten_object(primitive):
        if not isinstance(primitive, (list, tuple)) or len(primitive) < 2:
            return None
        if isinstance(primitive[1], str):
            return primitive[1]
        if len(primitive) >= 3 and isinstance(primitive[2], str):
            return primitive[2]
        return None

    def _validate_success_path_pose_guard(self, path: list[TreeNode]) -> tuple[bool, list[str]]:
        raw_task_entry = getattr(getattr(self.simulator, "args", None), "raw_task_entry", None)
        primitives = self._flatten_task_primitives(raw_task_entry)
        if not primitives:
            return True, []

        final_node = path[-1] if path else None
        final_snapshot = getattr(final_node, "snapshot", None) or {}
        pose_status = final_snapshot.get("object_upright_status") or {}
        container_categories = getattr(self.simulator, "container_categories", set()) or set()
        elongated_categories = getattr(self.simulator, "elongated_categories", set()) or set()
        reasons: list[str] = []

        def _require_upright(name: str, role: str) -> None:
            status, resolved_name = self._resolve_pose_status(pose_status, name)
            if status != "upright":
                resolved = f" (resolved as {resolved_name})" if resolved_name and resolved_name != name else ""
                observed = status if status is not None else "missing"
                reasons.append(
                    f"{role} {name}{resolved} must be upright for final task success, "
                    f"but final pose is {observed}."
                )

        for primitive in primitives:
            if not isinstance(primitive, (list, tuple)) or len(primitive) == 0:
                continue
            primitive_type = primitive[0]
            if primitive_type == "place" and len(primitive) >= 4:
                src = primitive[2]
                tgt = primitive[3]
                if self._name_in_categories(tgt, container_categories):
                    _require_upright(tgt, "place target container")
                    if self._name_in_categories(src, elongated_categories):
                        _require_upright(src, "place source elongated object")
            elif primitive_type == "straighten":
                obj = self._extract_straighten_object(primitive)
                if self._name_in_categories(obj, container_categories):
                    _require_upright(obj, "straighten target container")

        return len(reasons) == 0, reasons
    

    def _format_search_output(
        self,
        # system_prompt: str,
        num_sim_round: int,
        num_sim_attempts: int,
        root: TreeNode,
        global_search_time: float,
        success_paths: list[list[TreeNode]] | None = None,
        init_phase_s: float = 0.0,
        mcts_loop_s: float = 0.0,
    ) -> dict:
        """
        Formats the final rollout (or path) into a dictionary, similar to original code.
        The last node in `path_nodes` may be the newly expanded or final node.
        """
        save_tree_pcd = bool(
            self.mcts_cfg.get(
                "save_tree_pcd",
                self.mcts_cfg.get("save_tree_pcd_only", True),
            )
        )
        tree, node_trajectories = serialize_tree(
            root,
            save_tree_observations=bool(self.mcts_cfg.get("save_tree_observations", True)),
            save_tree_pcd=save_tree_pcd,
        )
        root_image = root.observations["image_all_views"][self.view_names[0]]
        save_restore_augmented = bool(self.mcts_cfg.get("save_restore_augmented_traj", False))

        success_trajectories = []
        success_trajectory_node_end_indices = []
        success_trajectory_node_end_grasped_names = []
        success_trajectory_node_subgoals = []
        success_trajectory_node_ids = []
        success_trajectories_restore2step = [] if save_restore_augmented else None
        success_trajectory_node_end_indices_restore2step = [] if save_restore_augmented else None
        success_trajectory_node_end_grasped_names_restore2step = [] if save_restore_augmented else None
        success_trajectory_node_subgoals_restore2step = [] if save_restore_augmented else None
        success_trajectory_node_ids_restore2step = [] if save_restore_augmented else None
        raw_task_entry = getattr(getattr(self.simulator, "args", None), "raw_task_entry", None)
        success_guard = {
            "checked": bool(success_paths),
            "raw_task_entry": raw_task_entry,
            "passed_count": 0,
            "blocked_count": 0,
            "blocked_paths": [],
        }

        def _make_restore_prefix_step(template_step: dict, close_gripper: bool) -> dict:
            step = deepcopy(template_step)
            action = step.get("action")
            if not isinstance(action, dict):
                return step

            action_raw = action.get("action_raw")
            if action_raw is not None:
                action_raw_arr = np.zeros_like(np.asarray(action_raw), dtype=np.float32)
                if action_raw_arr.shape[-1] >= 1:
                    action_raw_arr[..., -1] = -1.0 if close_gripper else 0.0
                action["action_raw"] = action_raw_arr

            arm_cmd = action.get("arm_action_cmd")
            if arm_cmd is not None:
                action["arm_action_cmd"] = np.zeros_like(np.asarray(arm_cmd), dtype=np.float32)

            gripper_cmd = action.get("gripper_action_cmd")
            if gripper_cmd is not None:
                gripper_raw = -1.0 if close_gripper else 0.0
                gripper_cmd_arr = np.zeros_like(np.asarray(gripper_cmd), dtype=np.float32)
                gripper_cmd_arr[...] = gripper_raw * 0.01
                action["gripper_action_cmd"] = gripper_cmd_arr

            return step

        if success_paths:
            for p in success_paths:
                guard_passed, guard_reasons = self._validate_success_path_pose_guard(p)
                if not guard_passed:
                    success_guard["blocked_count"] += 1
                    success_guard["blocked_paths"].append(
                        {
                            "path_node_ids": [
                                int(getattr(n, "_serialized_node_id", -1)) for n in p
                            ],
                            "final_node_id": int(getattr(p[-1], "_serialized_node_id", -1)) if p else -1,
                            "reasons": guard_reasons,
                        }
                    )
                    continue
                success_guard["passed_count"] += 1
                traj_steps = []
                traj_steps_restore2step = [] if save_restore_augmented else None
                node_end_indices = []
                node_end_grasped_names = []
                node_subgoals = []
                node_ids = []
                node_end_indices_restore2step = [] if save_restore_augmented else None
                node_ids_restore2step = [] if save_restore_augmented else None
                node_idx_with_traj = 0
                prev_node_end_grasped_names = None
                for n in p:
                    traj = getattr(n, "trajectory", None)
                    if traj is None:
                        continue
                    node_steps = list(traj)
                    if len(node_steps) == 0:
                        continue

                    planning_params = getattr(n, "planning_params", None) or {}
                    subgoal = planning_params.get("subgoal", "")
                    if isinstance(subgoal, str):
                        subgoal_text = subgoal
                    elif subgoal is None:
                        subgoal_text = ""
                    else:
                        subgoal_text = str(subgoal)
                    snapshot = getattr(n, "snapshot", None) or {}
                    node_grasped_names = list(snapshot.get("grasped_names") or [])
                    node_end_grasped_name = node_grasped_names[0] if node_grasped_names else ""
                    node_id = int(getattr(n, "_serialized_node_id", -1))

                    traj_steps.extend(node_steps)
                    node_end_indices.append(len(traj_steps))
                    node_end_grasped_names.append(node_end_grasped_name)
                    node_subgoals.append(subgoal_text)
                    node_ids.append(node_id)

                    if save_restore_augmented:
                        # restore2step prefix belongs to the current node, except the first node.
                        if node_idx_with_traj > 0:
                            close_gripper_on_restore = bool(prev_node_end_grasped_names)
                            prefix_step = _make_restore_prefix_step(
                                node_steps[0], close_gripper=close_gripper_on_restore
                            )
                            traj_steps_restore2step.append(deepcopy(prefix_step))
                            traj_steps_restore2step.append(deepcopy(prefix_step))
                        traj_steps_restore2step.extend(node_steps)
                        node_end_indices_restore2step.append(len(traj_steps_restore2step))
                        node_ids_restore2step.append(node_id)

                    prev_node_end_grasped_names = node_grasped_names
                    node_idx_with_traj += 1

                success_trajectories.append(traj_steps)
                success_trajectory_node_end_indices.append(node_end_indices)
                success_trajectory_node_end_grasped_names.append(node_end_grasped_names)
                success_trajectory_node_subgoals.append(node_subgoals)
                success_trajectory_node_ids.append(node_ids)
                if save_restore_augmented:
                    success_trajectories_restore2step.append(traj_steps_restore2step)
                    success_trajectory_node_end_indices_restore2step.append(node_end_indices_restore2step)
                    success_trajectory_node_end_grasped_names_restore2step.append(list(node_end_grasped_names))
                    success_trajectory_node_subgoals_restore2step.append(list(node_subgoals))
                    success_trajectory_node_ids_restore2step.append(list(node_ids))

        return {
            "instruction": self.instruction,
            "replay_reset_params": getattr(self, "replay_reset_params", None),
            "image": root_image,
            "T": self.simulator.T.tolist(),
            "scene_pose_matrix": self.simulator.scene_pose_matrix.tolist(),
            "ixt": {view: self.ixts[view].tolist() for view in self.ixts.keys()},
            "ext": {view: self.exts[view].tolist() for view in self.exts.keys()},
            "H": self.hs,
            "W": self.ws,
            "view_names": self.view_names,
            "vlm_view_names": self.vlm_view_names,
            "object_names": self.object_names,
            "skill_list": self.skill_list,
            "skill_desc": self.skill_desc,
            "tree": tree,
            "success_trajectories": success_trajectories,
            "success_trajectory_node_end_indices": success_trajectory_node_end_indices,
            "success_trajectory_node_end_grasped_names": success_trajectory_node_end_grasped_names,
            "success_trajectory_node_subgoals": success_trajectory_node_subgoals,
            "success_trajectory_node_ids": success_trajectory_node_ids,
            "success_trajectories_restore2step": success_trajectories_restore2step,
            "success_trajectory_node_end_indices_restore2step": success_trajectory_node_end_indices_restore2step,
            "success_trajectory_node_end_grasped_names_restore2step": success_trajectory_node_end_grasped_names_restore2step,
            "success_trajectory_node_subgoals_restore2step": success_trajectory_node_subgoals_restore2step,
            "success_trajectory_node_ids_restore2step": success_trajectory_node_ids_restore2step,
            "success_guard": success_guard,

            # "system_prompt": system_prompt,  # text or dict
            "num_simulation_rounds": num_sim_attempts,
            "num_expansion_rounds": num_sim_round,
            "global_search_time": global_search_time,
            "n_success_trajectories": len(success_trajectories),
            "init_phase_s": init_phase_s,
            "mcts_loop_s": mcts_loop_s,
            "timing_summary": self._summarize_timing(root),
        }, node_trajectories
