from torch._C import device
from isaaclab.app import AppLauncher

import json
import logging
import math
import os
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import argparse
import trimesh
import torch
import yaml
from PIL import Image

from utils.geometry_utils import lift_pixel_to_world, q2R_wxyz, qt2T
from utils.geometry_utils import _quat2axisangle_wxyz, quat_mul, quat_conj, quat_rotate
from utils.geometry_utils import to_tensor, compute_alignment_rotation, compute_alignment_rotation_candidates

_APP_LAUNCHER = None
_SIM_APP = None
_ISAACLAB_IMPORTED = False
_MAKE_ENV_CALL_COUNT = 0


def _load_category_list_file(path: str | None) -> set[str]:
    if not path:
        return set()
    resolved = os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))
    if not os.path.isfile(resolved):
        return set()
    out: set[str] = set()
    with open(resolved, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item and not item.startswith("#"):
                out.add(item)
    return out


def parse_simulator_args(sim_tokens):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument(
        "--renderer", type=str, default="PathTracing", choices=["RayTracedLighting", "PathTracing"], help="Renderer to use."
    )
    parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1, help="Number of samples per pixel per frame.")
    parser.add_argument("--scene_desc_file", type=str, default=None)
    parser.add_argument("--render_mode", type=str, default="rgb_array")

     # supoorted task types
    parser.add_argument("--enable_pick", action="store_true")
    parser.add_argument("--enable_place", action="store_true")
    parser.add_argument("--enable_pose", action="store_true")
    parser.add_argument("--enable_open_drawer", action="store_true")
    parser.add_argument("--enable_close_drawer", action="store_true")
    parser.add_argument("--enable_close_door", action="store_true")

    parser.add_argument("--eval_num", type=int, default=20)
    parser.add_argument("--grasp_force_threshold", type=float, default=2.0)
    parser.add_argument("--robot_name", type=str, default="panda")
    parser.add_argument("--h", type=int, default=512)
    parser.add_argument("--w", type=int, default=512)
    parser.add_argument("--wrist_h", type=int, default=256)
    parser.add_argument("--wrist_w", type=int, default=256)
    parser.add_argument("--camera_names", type=str, default="['front', 'top', 'wrist']")
    parser.add_argument("--data_collect_h", type=int, default=256)
    parser.add_argument("--data_collect_w", type=int, default=256)
    parser.add_argument("--data_collect_camera_names", type=str, default="['front', 'wrist']")
    parser.add_argument("--collect_depth", action="store_true")
    parser.add_argument("--collect_segmentation", action="store_true")
    parser.add_argument("--collect_object_poses", action="store_true")
    parser.add_argument("--data_collect_depth_storage", type=str, default="uint16")
    parser.add_argument("--data_collect_depth_scale", type=float, default=1000.0)
    parser.add_argument("--data_collect_seg_storage", type=str, default="uint16")
    parser.add_argument("--lift_offset", type=float, default=0.25)
    parser.add_argument("--place_offset", type=float, default=0.2)
    parser.add_argument("--place_offset_center", type=float, default=0.05)
    parser.add_argument("--place_offset_rlpolicy", type=float, default=0.15)
    parser.add_argument("--place_offset_rlpolicy_upright", type=float, default=0.05)
    parser.add_argument("--place_offset_orientation_drop", type=float, default=0.05)
    parser.add_argument("--place_offset_gripper_orientation", type=float, default=0.0)
    parser.add_argument("--container_list", type=str, default="")
    parser.add_argument("--elongated_list", type=str, default="")
    parser.add_argument("--container_upright_cos_threshold", type=float, default=0.866)
    parser.add_argument("--place_noise_std_xyz", type=str, default="[0.0, 0.0, 0.0]")
    parser.add_argument("--pregrasp_offset", type=float, default=0.15)
    parser.add_argument("--pregrasp_noise_std_xyz", type=str, default="[0.01, 0.01, 0.01]")
    # Planner-failure fallback for the pre-grasp goal: when cuRobo cannot plan to the
    # (noised) pre-grasp pose, retry with the goal shifted in the xy plane around that
    # base -- radii tried in order, each radius split into num_directions headings
    # (from +x, CCW). A failed plan executes nothing, so retries leave the sim untouched.
    parser.add_argument("--pregrasp_fallback_enable", action="store_true")
    parser.add_argument("--pregrasp_fallback_radii", type=str, default="[0.02, 0.04, 0.06]")
    parser.add_argument("--pregrasp_fallback_num_directions", type=int, default=8)
    parser.add_argument("--release_offset", type=float, default=0.15)
    parser.add_argument("--release_retract_offset", type=float, default=0.07)
    # Non-vertical (e.g. horizontal cavity insert) settle clearance + the angle threshold
    # (deg, between n_app and gravity) under which a place is treated as vertical.
    parser.add_argument("--place_offset_nonvertical", type=float, default=0.01)
    parser.add_argument("--approach_vertical_angle_deg", type=float, default=20.0)
    parser.add_argument(
        "--agent_cfg_file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "agents", "rl_games_ppo_cfg.yaml"),
    )

    # Add Isaac Lab app args after custom args so AppLauncher can validate for conflicts.
    AppLauncher.add_app_launcher_args(parser)

    args, _ = parser.parse_known_args(sim_tokens)

    global _APP_LAUNCHER, _SIM_APP
    if _APP_LAUNCHER is None:
        print(
            f"[simulator.parse_args] AppLauncher init pid={os.getpid()} ppid={os.getppid()} tokens={len(sim_tokens)}",
            flush=True,
        )
        _APP_LAUNCHER = AppLauncher(args)
        _SIM_APP = _APP_LAUNCHER.app
        _ensure_isaaclab_imports()
    
    return args


def _append_sim_arg(tokens, name, value):
    if value is None:
        return
    flag = f"--{name}"
    if isinstance(value, bool):
        if value:
            tokens.append(flag)
        return
    if isinstance(value, str) and value.startswith("-"):
        tokens.append(f"{flag}={value}")
        return
    if isinstance(value, (list, tuple)):
        tokens.extend([flag, repr(list(value))])
        return
    tokens.extend([flag, str(value)])


def _build_sim_tokens_from_cfg(sim_cfg: dict):
    base_cfg = sim_cfg.get("isaaclab", sim_cfg)
    app_cfg = sim_cfg.get("app", {})

    tokens = []
    for key, val in app_cfg.items():
        _append_sim_arg(tokens, key, val)

    for key in (
        "num_envs",
        "task",
        "renderer",
        "samples_per_pixel_per_frame",
        "render_mode",
        "enable_pick",
        "enable_place",
        "enable_pose",
        "enable_open_drawer",
        "enable_close_drawer",
        "enable_close_door",
        "eval_num",
        "grasp_force_threshold",
        "robot_name",
        "h",
        "w",
        "wrist_h",
        "wrist_w",
        "camera_names",
        "data_collect_h",
        "data_collect_w",
        "data_collect_camera_names",
        "collect_depth",
        "collect_segmentation",
        "collect_object_poses",
        "data_collect_depth_storage",
        "data_collect_depth_scale",
        "data_collect_seg_storage",
        "lift_offset",
        "place_offset",
        "place_offset_center",
        "place_offset_rlpolicy",
        "place_offset_rlpolicy_upright",
        "place_offset_orientation_drop",
        "place_offset_gripper_orientation",
        "container_list",
        "elongated_list",
        "container_upright_cos_threshold",
        "place_noise_std_xyz",
        "pregrasp_offset",
        "pregrasp_noise_std_xyz",
        "pregrasp_fallback_enable",
        "pregrasp_fallback_radii",
        "pregrasp_fallback_num_directions",
        "release_offset",
        "release_retract_offset",
        "place_offset_nonvertical",
        "approach_vertical_angle_deg",
        "agent_cfg_file",
    ):
        _append_sim_arg(tokens, key, base_cfg.get(key))
    return tokens


def parse_simulator_args_from_cfg(sim_cfg: dict, env_cfg: dict | None = None):
    sim_tokens = _build_sim_tokens_from_cfg(sim_cfg)
    args = parse_simulator_args(sim_tokens)
    if env_cfg is None:
        raise ValueError("env config is required and must include env.scene_desc_file")
    if not isinstance(env_cfg, dict):
        raise ValueError("env config must be a dict")
    scene_desc_file = env_cfg.get("scene_desc_file")
    if not scene_desc_file:
        raise ValueError("env.scene_desc_file is required")
    args.scene_desc_file = scene_desc_file
    args.env_overrides = env_cfg
    return args


def shutdown_sim_app():
    """Close the Isaac Sim app to avoid hanging processes on exit."""
    global _APP_LAUNCHER, _SIM_APP, _ISAACLAB_IMPORTED
    if _SIM_APP is not None:
        try:
            _SIM_APP.close()
        except Exception:
            pass
    _SIM_APP = None
    _APP_LAUNCHER = None
    _ISAACLAB_IMPORTED = False


def _ensure_isaaclab_imports():
    """Import Isaac Lab modules after AppLauncher initializes the simulator app."""
    global _ISAACLAB_IMPORTED
    global RlGamesVecEnvWrapper, env_configurations, vecenv, RlGamesGpuEnv
    global omni, Usd, UsdGeom, Gf

    if _ISAACLAB_IMPORTED:
        return
    if _APP_LAUNCHER is None:
        raise RuntimeError("AppLauncher must be initialized before importing Isaac Lab modules.")

    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv
    from rl_games.common import env_configurations, vecenv
    import importlib

    omni = importlib.import_module("omni")
    importlib.import_module("omni.usd")
    from pxr import Usd, UsdGeom, Gf

    globals().update(
        RlGamesVecEnvWrapper=RlGamesVecEnvWrapper,
        env_configurations=env_configurations,
        vecenv=vecenv,
        RlGamesGpuEnv=RlGamesGpuEnv,
        omni=omni,
        Usd=Usd,
        UsdGeom=UsdGeom,
        Gf=Gf,
    )
    _ISAACLAB_IMPORTED = True


class IsaacLabInterface:
    def __init__(self, args):
        _ensure_isaaclab_imports()
        from simulators.base_env_cfg import BaseEnvCfg
        self._BaseEnvCfg = BaseEnvCfg
        self.args = args
        self.env_overrides = getattr(args, "env_overrides", None)

        # self.T = np.eye(4)
        self.T = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.robot_name = args.robot_name
        if "widowx" in (self.robot_name or "").lower():
            # WidowX-250 (SimplerEnv bridge tasks). cuRobo loads the vendored config
            # bundle (kinematics + collision spheres) in this repo's assets/; ee_link=
            # gripper_link. robot_active_joint_indices selects the 6 arm joints from
            # cuRobo's 8-joint cspace output (arm first, then fingers).
            self.robot_config = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "assets", "widowx_gripper", "curobo", "widowx.yml"
            )
            self.robot_joint_names = [
                "waist",
                "shoulder",
                "elbow",
                "forearm_roll",
                "wrist_angle",
                "wrist_rotate",
            ]
            self.robot_arm_dof = 6
            self.robot_active_joint_indices = [0, 1, 2, 3, 4, 5]
        else:
            self.robot_config = "franka.yml"
            self.robot_joint_names = [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ]
            self.robot_arm_dof = 7
            self.robot_active_joint_indices = [0, 1, 2, 3, 4, 5, 6]

        # EE axis convention (the only grasp geometry that differs per robot).
        from simulators.robot_conventions import resolve_ee_convention
        _ee_conv = resolve_ee_convention(self.robot_name)
        self.ee_approach_local = np.array(_ee_conv.approach_local, dtype=np.float64)
        self.ee_finger_local = np.array(_ee_conv.finger_local, dtype=np.float64)

        self.h = args.h
        self.w = args.w
        self.wrist_h = args.wrist_h
        self.wrist_w = args.wrist_w

        self.gripper_open_state = 1.0
        self.gripper_close_state = 0.0
        self.max_gripper_opening = None

        self.lift_offset = float(getattr(args, "lift_offset", 0.25))
        self.place_offset = float(getattr(args, "place_offset", 0.2))
        self.place_offset_center = float(getattr(args, "place_offset_center", 0.05))
        self.place_offset_rlpolicy = float(getattr(args, "place_offset_rlpolicy", 0.15))
        self.place_offset_rlpolicy_upright = float(getattr(args, "place_offset_rlpolicy_upright", 0.05))
        self.place_offset_orientation_drop = float(getattr(args, "place_offset_orientation_drop", 0.05))
        # place_with_gripper_orientation_drop: vertical hover offset above the place point before
        # release. Defaults to 0.0 (object center dropped exactly at the VLM-specified point).
        self.place_offset_gripper_orientation = float(getattr(args, "place_offset_gripper_orientation", 0.0))
        # Non-vertical regime: small settle clearance (quantity B) + the vertical/non-vertical
        # threshold on angle(n_app, gravity). See docs/place_pipeline_rfc.md §5-§6.
        self.place_offset_nonvertical = float(getattr(args, "place_offset_nonvertical", 0.01))
        self.approach_vertical_angle_deg = float(getattr(args, "approach_vertical_angle_deg", 20.0))
        self.container_categories: set[str] = _load_category_list_file(getattr(args, "container_list", ""))
        self.elongated_categories: set[str] = _load_category_list_file(getattr(args, "elongated_list", ""))
        self.container_upright_cos_threshold = float(getattr(args, "container_upright_cos_threshold", 0.866))
        raw_place_noise_std_xyz = getattr(args, "place_noise_std_xyz", "[0.0, 0.0, 0.0]")
        if isinstance(raw_place_noise_std_xyz, str):
            place_noise_std_xyz = np.asarray(eval(raw_place_noise_std_xyz), dtype=np.float32).reshape(-1)
        else:
            place_noise_std_xyz = np.asarray(raw_place_noise_std_xyz, dtype=np.float32).reshape(-1)
        if place_noise_std_xyz.size != 3:
            raise ValueError(f"place_noise_std_xyz must contain exactly 3 values, got {place_noise_std_xyz.size}")
        if np.any(place_noise_std_xyz < 0):
            raise ValueError("place_noise_std_xyz must be non-negative for all axes")
        self.place_noise_std_xyz = place_noise_std_xyz
        self.pregrasp_offset = float(getattr(args, "pregrasp_offset", 0.15))
        raw_pregrasp_noise_std_xyz = getattr(args, "pregrasp_noise_std_xyz", "[0.01, 0.01, 0.01]")
        if isinstance(raw_pregrasp_noise_std_xyz, str):
            pregrasp_noise_std_xyz = np.asarray(eval(raw_pregrasp_noise_std_xyz), dtype=np.float32).reshape(-1)
        else:
            pregrasp_noise_std_xyz = np.asarray(raw_pregrasp_noise_std_xyz, dtype=np.float32).reshape(-1)
        if pregrasp_noise_std_xyz.size != 3:
            raise ValueError(
                f"pregrasp_noise_std_xyz must contain exactly 3 values, got {pregrasp_noise_std_xyz.size}"
            )
        if np.any(pregrasp_noise_std_xyz < 0):
            raise ValueError("pregrasp_noise_std_xyz must be non-negative for all axes")
        self.pregrasp_noise_std_xyz = pregrasp_noise_std_xyz
        self.pregrasp_fallback_enable = bool(getattr(args, "pregrasp_fallback_enable", False))
        raw_pregrasp_fallback_radii = getattr(args, "pregrasp_fallback_radii", "[0.02, 0.04, 0.06]")
        if isinstance(raw_pregrasp_fallback_radii, str):
            pregrasp_fallback_radii = np.asarray(eval(raw_pregrasp_fallback_radii), dtype=np.float32).reshape(-1)
        else:
            pregrasp_fallback_radii = np.asarray(raw_pregrasp_fallback_radii, dtype=np.float32).reshape(-1)
        if np.any(pregrasp_fallback_radii <= 0):
            raise ValueError("pregrasp_fallback_radii must be strictly positive")
        self.pregrasp_fallback_radii = pregrasp_fallback_radii
        self.pregrasp_fallback_num_directions = int(getattr(args, "pregrasp_fallback_num_directions", 8))
        if self.pregrasp_fallback_num_directions < 1:
            raise ValueError("pregrasp_fallback_num_directions must be >= 1")
        self.release_offset = float(getattr(args, "release_offset", 0.15))
        self.release_retract_offset = float(getattr(args, "release_retract_offset", 0.07))

        self.camera_names = eval(args.camera_names)
        self._base_camera_names = list(self.camera_names)
        self.agent_cfg_file = getattr(
            args,
            "agent_cfg_file",
            os.path.join(os.path.dirname(__file__), "agents", "rl_games_ppo_cfg.yaml"),
        )
        self._default_agent_cfg = self._load_agent_cfg(self.agent_cfg_file)

        ## data collection settings
        self.data_collect_h = getattr(args, "data_collect_h", 256)
        self.data_collect_w = getattr(args, "data_collect_w", 256)
        raw_data_collect_camera_names = getattr(args, "data_collect_camera_names", ["front", "wrist"])
        if isinstance(raw_data_collect_camera_names, str):
            self.data_collect_camera_names = list(eval(raw_data_collect_camera_names))
        else:
            self.data_collect_camera_names = list(raw_data_collect_camera_names)
        self._base_data_collect_camera_names = list(self.data_collect_camera_names)
        self.collect_depth = getattr(args, "collect_depth", False)
        self.collect_segmentation = getattr(args, "collect_segmentation", False)
        self.collect_object_poses = getattr(args, "collect_object_poses", False)
        self.data_collect_depth_storage = str(getattr(args, "data_collect_depth_storage", "uint16")).lower()
        self.data_collect_depth_scale = float(getattr(args, "data_collect_depth_scale", 1000.0))
        self.data_collect_seg_storage = str(getattr(args, "data_collect_seg_storage", "uint16")).lower()
        self.front_camera_count = 1
        self._refresh_camera_name_lists(front_camera_count=self.front_camera_count)

        # self._register_rlgames_env()
        self._last_step_action_ctx = None
        self._fk_robot_model = None
        self._fk_joint_state_cls = None
        # Set to True by the cuRobo skill when planner_execute_mode1_replan=True.
        # Controls whether delta EE cmd fields are recorded during data collection.
        self._data_collect_store_ee_cmd: bool = False
        # Per-env mesh templates to avoid repeated USD traversal and triangulation.
        self._mesh_cache_by_env = {}

    def _load_agent_cfg(self, cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    @staticmethod
    def _expand_front_camera_aliases(view_names: list[str], front_camera_count: int) -> list[str]:
        count = max(1, int(front_camera_count))
        expanded: list[str] = []
        for view_name in view_names:
            view = str(view_name)
            if view == "front":
                expanded.append("front")
                for idx in range(1, count):
                    expanded.append(f"front_{idx}")
            else:
                expanded.append(view)
        deduped: list[str] = []
        seen = set()
        for view in expanded:
            if view in seen:
                continue
            seen.add(view)
            deduped.append(view)
        return deduped

    def _refresh_camera_name_lists(self, front_camera_count: int) -> None:
        self.front_camera_count = max(1, int(front_camera_count))
        self.camera_names = self._expand_front_camera_aliases(self._base_camera_names, self.front_camera_count)
        self.data_collect_camera_names = self._expand_front_camera_aliases(
            self._base_data_collect_camera_names, self.front_camera_count
        )

    def _register_rlgames_env(self):
        env_name = "rlgpu"
        try:
            already = (
                hasattr(env_configurations, "configurations") and env_name in env_configurations.configurations
            ) or (hasattr(env_configurations, "registrations") and env_name in env_configurations.registrations)
        except Exception:
            already = False

        if not already:
            vecenv.register(
                "IsaacRlgWrapper",
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
            )
            env_configurations.register(
                env_name,
                {
                    "vecenv_type": "IsaacRlgWrapper",
                    "env_creator": lambda **kwargs: self.make_env(),
                },
            )

    def _unwrap_env(self, env):
        if hasattr(env, "unwrapped") and env.unwrapped is not None:
            return env.unwrapped
        if hasattr(env, "env"):
            return env.env
        return env

    def _env_cache_key(self, env):
        base_env = self._unwrap_env(env)
        return id(base_env)

    def _get_or_init_env_mesh_cache(self, env):
        key = self._env_cache_key(env)
        cache = self._mesh_cache_by_env.get(key)
        if cache is None:
            cache = {
                "table_template": None,
                "object_templates": None,
            }
            self._mesh_cache_by_env[key] = cache
        return cache

    def _invalidate_mesh_cache(self, env=None):
        if env is None:
            self._mesh_cache_by_env.clear()
            return
        key = self._env_cache_key(env)
        self._mesh_cache_by_env.pop(key, None)

    def _is_table_active(self, env) -> bool:
        base_env = self._unwrap_env(env)
        is_table_active = getattr(base_env, "is_table_active", None)
        if callable(is_table_active):
            return bool(is_table_active())
        return bool(getattr(base_env, "table", None) is not None)

    def _get_active_object_names(self, env) -> list[str]:
        base_env = self._unwrap_env(env)
        names = list(base_env.objects.keys())
        if not self._is_table_active(base_env):
            names = [name for name in names if name != "table"]
        return names

    def set_scene_pose(self, env):
        self.scene_pose_matrix = np.eye(4)
        self.scene_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def get_translation_from_3d_point(self, R_quat, t1):
        R = q2R_wxyz(R_quat)
        R2 = self.T[:3, :3]
        t2 = self.T[:3, 3][..., None] # (3, 1)
        translation = (R @ np.linalg.inv(R2)) @ t2 + t1[..., None]  # (3, 1)
        return translation.squeeze(-1)  # (3,)
    
    def get_curobo_q_from_simulator(self, q):
        return q

    def get_simulator_q_from_curobo(self, q):
        return q

    ####### Get Observations #######

    def get_camera_by_name(self, env, name: str):
        base_env = self._unwrap_env(env)
        get_by_view_name = getattr(base_env, "get_camera_by_view_name", None)
        if callable(get_by_view_name):
            camera = get_by_view_name(name)
            if camera is not None:
                return camera
        camera = getattr(base_env, f"{name}_camera", None)
        if camera is None:
            raise KeyError(f"Camera view '{name}' not found in current env.")
        return camera

    def _tick_sim_app(self, steps: int = 2, pause_timeline: bool = False):
        if _SIM_APP is None:
            return

        timeline = None
        was_playing = False
        if pause_timeline:
            try:
                import omni.timeline

                timeline = omni.timeline.get_timeline_interface()
                was_playing = bool(timeline.is_playing())
                if was_playing:
                    timeline.stop()
                    timeline.commit()
            except Exception:
                timeline = None
                was_playing = False

        try:
            for _ in range(steps):
                try:
                    _SIM_APP.update()
                except Exception:
                    break
        finally:
            if pause_timeline and timeline is not None and was_playing:
                try:
                    timeline.play()
                    timeline.commit()
                except Exception:
                    pass

    def _reset_replicator(self):
        try:
            import omni.replicator.core as rep
        except Exception:
            return
        try:
            rep.orchestrator.stop()
        except Exception:
            pass
        try:
            orch = getattr(rep.orchestrator, "_orchestrator", None)
            if orch is not None:
                orch.reset()
        except Exception:
            pass

    def _clear_render_prim(self):
        _ensure_isaaclab_imports()
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return
            prim = stage.GetPrimAtPath("/Render")
            if prim and prim.IsValid():
                stage.RemovePrim("/Render")
        except Exception:
            pass

    def warmup_cameras(self, env, views: list[str] | None = None, steps: int = 5):
        """Ensure camera sensors are initialized and render graph is ready."""
        base_env = self._unwrap_env(env)
        if views is None:
            views = list(getattr(self, "camera_names", []))
        for view in views:
            cam = getattr(base_env, f"{view}_camera", None)
            if cam is None:
                continue
            try:
                if not cam.is_initialized:
                    cam._initialize_callback(None)
            except Exception:
                pass
            try:
                cam.reset()
            except Exception:
                pass
        self._tick_sim_app(steps=steps)

    def _get_camera_output(self, camera, key: str, retries: int = 5):
        last_err = None
        for _ in range(retries):
            try:
                return camera.data.output[key]
            except KeyError as err:
                last_err = err
                # Replicator pipeline can be transient after stage reset; wait a few frames then retry.
                self._tick_sim_app(steps=3)
        if last_err is not None:
            raise last_err
        raise KeyError(key)

    def get_rgb(self, env, view: str, env_id: int = 0):
        env = self._unwrap_env(env)
        camera = self.get_camera_by_name(env, view)
        rgb = self._get_camera_output(camera, "rgb")[env_id].cpu().numpy()
        if view == "front" and getattr(env, "_background_image_path", None):
            rgb = self._composite_background(env, camera, rgb, env_id=env_id)
        return rgb

    def _composite_background(self, env, camera, rgb: np.ndarray, env_id: int = 0) -> np.ndarray:
        """SimplerEnv rgb_overlay: paste the rendered foreground over the scene's
        static `background_image`. Foreground = pixels with real geometry (finite
        depth) — keeps the robot arm and every mesh without needing semantic
        labels; the empty void (no floor/walls in overlay scenes) becomes the
        image. Only called when the scene json set `background_image`."""
        depth = self._get_camera_output(camera, "depth")[env_id]
        depth = depth.detach().cpu().numpy()
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        fg_mask = np.isfinite(depth) & (depth > 0.0) & (depth < 1.0e4)
        h, w = rgb.shape[:2]
        bg = getattr(env, "_background_image_cache", None)
        if bg is None or bg.shape[0] != h or bg.shape[1] != w:
            from PIL import Image
            bg = np.asarray(
                Image.open(env._background_image_path).convert("RGB").resize((w, h)),
                dtype=rgb.dtype,
            )
            env._background_image_cache = bg
        return np.where(fg_mask[..., None], rgb[..., :3], bg).astype(rgb.dtype)
    
    def get_rgbs(self, env, views, env_id: int = 0):
        env = self._unwrap_env(env)
        rgbs = {}
        for view in views:
            rgbs[view] = self.get_rgb(env, view, env_id=env_id)
        return rgbs
    
    def get_depth(self, env, view: str, env_id: int = 0):
        env = self._unwrap_env(env)
        camera = self.get_camera_by_name(env, view)
        depth = self._get_camera_output(camera, "depth")
        depth_image = depth[env_id].cpu().numpy()
        if depth_image.ndim == 3:
            depth_image = depth_image[:, :, 0]
        return depth_image
    
    def get_depths(self, env, views):
        env = self._unwrap_env(env)
        depths = {}
        for view in views:
            depths[view] = self.get_depth(env, view)
        return depths
    
    def get_semantic_segmentation(self, env, view: str, env_id: int = 0):
        env = self._unwrap_env(env)
        camera = self.get_camera_by_name(env, view)
        seg = self._get_camera_output(camera, "semantic_segmentation")
        if seg is None:
            raise KeyError("semantic_segmentation not enabled for this camera")
        seg_image = seg[env_id].detach().cpu().numpy()
        if seg_image.ndim == 3 and seg_image.shape[-1] == 1:
            seg_image = seg_image[:, :, 0]
        return seg_image

    def get_semantic_segmentations(self, env, views, env_id: int = 0):
        env = self._unwrap_env(env)
        segs = {}
        for view in views:
            try:
                segs[view] = self.get_semantic_segmentation(env, view, env_id=env_id)
            except KeyError:
                segs[view] = None
        return segs

    def get_seg_label_mappings(self, env, views):
        """Return {view: {id_str: class_name}} from idToLabels for each camera view."""
        env = self._unwrap_env(env)
        result = {}
        for view in views:
            try:
                camera = self.get_camera_by_name(env, view)
                cam_info = camera.data.info
                seg_info = cam_info.get("semantic_segmentation")
                if seg_info is None:
                    result[view] = {}
                    continue
                id_to_labels = seg_info.get("idToLabels", {})
                result[view] = {k: v.get("class", "") for k, v in id_to_labels.items()}
            except Exception:
                result[view] = {}
        return result

    def get_camera_intrinsic(self, env, view: str, env_id: int = 0):
        env = self._unwrap_env(env)
        camera = self.get_camera_by_name(env, view)
        ixt = camera.data.intrinsic_matrices[env_id].cpu().numpy()  # (3, 3)
        return ixt
    
    def get_camera_intrinsics(self, env, views):
        env = self._unwrap_env(env)
        ixts = {}
        for view in views:
            ixts[view] = self.get_camera_intrinsic(env, view)
        return ixts
        
    def get_camera_extrinsic(self, env, view: str, env_id: int = 0):
        env = self._unwrap_env(env)
        camera = self.get_camera_by_name(env, view)
        # camera data is batched over environments; grab the requested env slot
        t_w = camera.data.pos_w[env_id].detach().cpu().numpy()
        q_w = camera.data.quat_w_ros[env_id].detach().cpu().numpy()
        T_w = qt2T(q_w, t_w)
        return np.linalg.inv(T_w)  # w2c, camera is in ros convention
    
    def get_camera_extrinsics(self, env, views):
        env = self._unwrap_env(env)
        exts = {}
        for view in views:
            exts[view] = self.get_camera_extrinsic(env, view)
        return exts
    
    def get_pcd(self, env, view: str, env_id: int = 0):
        env = self._unwrap_env(env)
        depth = self.get_depth(env, view, env_id=env_id)
        ixt = self.get_camera_intrinsic(env, view, env_id=env_id)
        ext = self.get_camera_extrinsic(env, view, env_id=env_id)
        pcd = lift_pixel_to_world(depth, ixt, ext, reshape=True)  # (H, W, 3)
        return pcd

    def get_pcds(self, env, views, env_id: int = 0):
        env = self._unwrap_env(env)
        pcds = {}
        for view in views:
            pcds[view] = self.get_pcd(env, view, env_id=env_id)
        return pcds
    
    @staticmethod
    def _gf_to_np_mat4(m: "Gf.Matrix4d") -> np.ndarray:
        return np.array(m, dtype=np.float64)

    @staticmethod
    def _triangulate_faces(face_indices: np.ndarray, face_counts: np.ndarray) -> np.ndarray:
        faces = []
        cursor = 0
        for c in face_counts:
            poly = face_indices[cursor : cursor + c]
            cursor += c
            if c == 3:
                faces.append(poly)
            elif c > 3:
                for k in range(1, c - 1):
                    faces.append([poly[0], poly[k], poly[k + 1]])
        if not faces:
            return np.empty((0, 3), dtype=np.int64)
        return np.asarray(faces, dtype=np.int64)

    def _extract_collision_mesh_template(self, obj):
        """Extract collision mesh in object-root local frame for caching."""
        xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
        stage = omni.usd.get_context().get_stage()
        assert stage is not None

        prim_path = obj.cfg.prim_path
        obj_root = stage.GetPrimAtPath(prim_path)
        if not obj_root or not obj_root.IsValid():
            raise RuntimeError(f"Cannot find object prim at {prim_path}")
        obj_root.Load()

        collision_meshes = []
        for prim in Usd.PrimRange(obj_root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            p = str(prim.GetPath()).lower()
            if "/collisions/" in p or "/collision/" in p:
                collision_meshes.append(prim)

        if not collision_meshes:
            raise RuntimeError(f"No collision meshes found under {prim_path}")

        T_root_w_static = self._gf_to_np_mat4(xfc.GetLocalToWorldTransform(obj_root))
        T_w_root_static = np.linalg.inv(T_root_w_static)

        vertices_all = []
        faces_all = []
        v_offset = 0
        for mesh_prim in collision_meshes:
            usd_mesh = UsdGeom.Mesh(mesh_prim)
            points = usd_mesh.GetPointsAttr().Get()
            if points is None or len(points) == 0:
                continue
            face_indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
            face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()
            if face_indices is None or face_counts is None:
                continue

            vertices = np.asarray([list(p) for p in points], dtype=np.float64)
            face_indices = np.asarray(face_indices, dtype=np.int64)
            face_counts = np.asarray(face_counts, dtype=np.int64)
            faces = self._triangulate_faces(face_indices, face_counts)
            if faces.size == 0:
                continue

            T_mp_w_static = self._gf_to_np_mat4(xfc.GetLocalToWorldTransform(mesh_prim))
            T_mp_root = T_w_root_static @ T_mp_w_static
            vertices_h = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
            vertices_root = (T_mp_root @ vertices_h.T).T[:, :3]

            vertices_all.append(vertices_root)
            faces_all.append(faces + v_offset)
            v_offset += len(vertices_root)

        if not vertices_all:
            raise RuntimeError("Collision meshes exist but have zero points.")

        vertices_local = np.concatenate(vertices_all, axis=0).astype(np.float32, copy=False)
        faces = np.concatenate(faces_all, axis=0).astype(np.int32, copy=False)
        return {
            "vertices_local": vertices_local,
            "faces": faces,
            "faces_list": faces.tolist(),
        }

    def _extract_articulation_link_mesh_templates(self, obj, art) -> dict:
        """{body_idx: {vertices_local, faces}} — each link's collision mesh in that LINK's local frame
        (cached, geometry is static). At export the per-link mesh is re-posed by the link's CURRENT
        world pose, so the combined cabinet mesh follows the joint opening. This lets cuRobo (rebuilt
        per node from object_meshes) avoid an OPENED drawer / reach into its hollow interior."""
        xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
        stage = omni.usd.get_context().get_stage()
        assert stage is not None
        obj_root = stage.GetPrimAtPath(obj.cfg.prim_path)
        if not obj_root or not obj_root.IsValid():
            raise RuntimeError(f"Cannot find articulation prim at {obj.cfg.prim_path}")
        obj_root.Load()
        body_names = list(art.body_names)
        body_name_set = set(body_names)
        link_inv_T_w: dict = {}
        per_link: dict = {}  # body_idx -> [(verts_link_local, faces), ...]
        for prim in Usd.PrimRange(obj_root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if "/collisions/" not in str(prim.GetPath()).lower() and "/collision/" not in str(prim.GetPath()).lower():
                continue
            # The owning link = nearest ancestor whose name is an articulation body.
            link_idx, link_prim, anc = None, None, prim.GetParent()
            while anc and anc.IsValid():
                if anc.GetName() in body_name_set:
                    link_idx = body_names.index(anc.GetName()); link_prim = anc; break
                if anc == obj_root:
                    break
                anc = anc.GetParent()
            if link_idx is None:
                continue
            usd_mesh = UsdGeom.Mesh(prim)
            points = usd_mesh.GetPointsAttr().Get()
            if points is None or len(points) == 0:
                continue
            face_indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
            face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()
            if face_indices is None or face_counts is None:
                continue
            verts = np.asarray([list(p) for p in points], dtype=np.float64)
            faces = self._triangulate_faces(np.asarray(face_indices, dtype=np.int64), np.asarray(face_counts, dtype=np.int64))
            if faces.size == 0:
                continue
            if link_idx not in link_inv_T_w:
                link_inv_T_w[link_idx] = np.linalg.inv(self._gf_to_np_mat4(xfc.GetLocalToWorldTransform(link_prim)))
            T_mesh_link = link_inv_T_w[link_idx] @ self._gf_to_np_mat4(xfc.GetLocalToWorldTransform(prim))
            vh = np.concatenate([verts, np.ones((len(verts), 1), dtype=np.float64)], axis=1)
            per_link.setdefault(link_idx, []).append(((T_mesh_link @ vh.T).T[:, :3], faces))
        out: dict = {}
        for link_idx, chunks in per_link.items():
            vall, fall, off = [], [], 0
            for v, f in chunks:
                vall.append(v); fall.append(f + off); off += len(v)
            out[link_idx] = {
                "vertices_local": np.concatenate(vall, axis=0).astype(np.float32, copy=False),
                "faces": np.concatenate(fall, axis=0).astype(np.int32, copy=False),
            }
        return out

    def _articulation_world_mesh(self, env, obj_name: str, art, env_id: int = 0):
        """(vertices_world (N,3), faces_list) for an articulation, each link re-posed by its CURRENT
        world pose so the mesh reflects the live joint opening. Per-link templates cached per env."""
        cache = self._get_or_init_env_mesh_cache(env)
        arti_templates = cache.setdefault("arti_link_templates", {})
        per_link = arti_templates.get(obj_name)
        if per_link is None:
            per_link = self._extract_articulation_link_mesh_templates(env.articulated_objects[obj_name], art)
            arti_templates[obj_name] = per_link
        body_pos = art.data.body_pos_w[env_id].detach().cpu().numpy().astype(np.float64)
        body_quat = art.data.body_quat_w[env_id].detach().cpu().numpy().astype(np.float64)
        vall, fall, off = [], [], 0
        for link_idx, tmpl in per_link.items():
            R = q2R_wxyz(body_quat[link_idx]).astype(np.float64)
            vw = tmpl["vertices_local"].astype(np.float64) @ R.T + body_pos[link_idx]
            vall.append(vw); fall.append(tmpl["faces"].astype(np.int64) + off); off += len(vw)
        vertices_world = np.concatenate(vall, axis=0).astype(np.float32, copy=False)
        faces = np.concatenate(fall, axis=0).astype(np.int32, copy=False)
        return vertices_world, faces.tolist()

    def _arti_link_part_labels(self, env, obj_name: str, art) -> dict:
        """{body_idx: part label}. URDF movable-joint children (URDF order) get 'drawer_i'
        (prismatic) / 'door_i' (revolute); links hanging below them through any joints
        (e.g. fixed handle links) inherit the label; everything else is 'base'. Used to
        split the arti cuRobo obstacle per part for collision attribution."""
        cache = self._get_or_init_env_mesh_cache(env)
        store = cache.setdefault("arti_part_labels", {})
        cached = store.get(obj_name)
        if cached is not None:
            return cached
        objects_cfg = env.scene_desc["objects"] if "objects" in env.scene_desc else env.scene_desc
        urdf_path = env._resolve_scene_asset_path(objects_cfg[obj_name]["urdf_path"])
        import xml.etree.ElementTree as ET
        root = ET.parse(urdf_path).getroot()
        parent_of = {}
        movable_children = []
        for j in root.findall("joint"):
            child = j.find("child").get("link")
            parent_of[child] = j.find("parent").get("link")
            if j.get("type") in ("prismatic", "revolute"):
                movable_children.append((child, j.get("type")))
        label_of_link = {
            child: (f"drawer_{i}" if jtype == "prismatic" else f"door_{i}")
            for i, (child, jtype) in enumerate(movable_children)
        }

        def _resolve(link):
            seen = set()
            cur = link
            while cur is not None and cur not in seen:
                if cur in label_of_link:
                    return label_of_link[cur]
                seen.add(cur)
                cur = parent_of.get(cur)
            return "base"

        labels = {idx: _resolve(name) for idx, name in enumerate(art.body_names)}
        store[obj_name] = labels
        return labels

    def _articulation_world_mesh_parts(self, env, obj_name: str, art, env_id: int = 0) -> dict:
        """{part_label: (vertices_world (N,3) float32, faces_list)} — per-part split of
        _articulation_world_mesh (same per-link re-posing), grouped by _arti_link_part_labels."""
        cache = self._get_or_init_env_mesh_cache(env)
        arti_templates = cache.setdefault("arti_link_templates", {})
        per_link = arti_templates.get(obj_name)
        if per_link is None:
            per_link = self._extract_articulation_link_mesh_templates(env.articulated_objects[obj_name], art)
            arti_templates[obj_name] = per_link
        labels = self._arti_link_part_labels(env, obj_name, art)
        body_pos = art.data.body_pos_w[env_id].detach().cpu().numpy().astype(np.float64)
        body_quat = art.data.body_quat_w[env_id].detach().cpu().numpy().astype(np.float64)
        acc = {}  # label -> (vall, fall, off)
        for link_idx, tmpl in per_link.items():
            label = labels.get(link_idx, "base")
            R = q2R_wxyz(body_quat[link_idx]).astype(np.float64)
            vw = tmpl["vertices_local"].astype(np.float64) @ R.T + body_pos[link_idx]
            vall, fall, off = acc.setdefault(label, ([], [], 0))
            vall.append(vw)
            fall.append(tmpl["faces"].astype(np.int64) + off)
            acc[label] = (vall, fall, off + len(vw))
        return {
            label: (
                np.concatenate(vall, axis=0).astype(np.float32, copy=False),
                np.concatenate(fall, axis=0).astype(np.int32, copy=False).tolist(),
            )
            for label, (vall, fall, _off) in acc.items()
        }

    def _object_world_pcd_t(self, env, name: str, env_id: int = 0, device=None):
        """(P,3) world-frame FULL-object pcd (torch) for `name`, from the unified meta store.
        Rigid: root transform of the base pcd. Arti: base pcd ∪ per movable part (part pcd ∪
        its handle points — the gripper touches the HANDLE, so grasp/contact queries need it,
        unlike the box-interior-only part AABB), each joint-pushed to the CURRENT opening
        (same q0 base-cano frame + `q - lower` convention as _part_world_pcd_t /
        sample_part_init_pose), then one root transform."""
        import isaacsim.core.utils.torch as torch_utils
        env = self._unwrap_env(env)
        if name in getattr(env, "articulated_objects", {}):
            from simulators.tasks import _arti_map_points
            art = env.articulated_objects[name]
            dev = device or art.data.root_pos_w.device
            joints = self._ensure_part_indices(env, name)
            part_pcds = env.object_point_clouds[name]
            handle_pcds = env.object_handle_point_clouds.get(name, {})
            chunks = [part_pcds["base"].to(dev)]
            for part, j in joints.items():
                pts = [part_pcds[part].to(dev)]
                hp = handle_pcds.get(part)
                if hp is not None:
                    pts.append(hp.to(dev))
                pts_q0 = torch.cat(pts, dim=0)
                axis = torch.tensor(j["axis_obj"], device=dev, dtype=pts_q0.dtype)
                origin = torch.tensor(j["origin_obj"], device=dev, dtype=pts_q0.dtype)
                lower = art.data.joint_pos_limits[env_id, j["dof_idx"], 0]
                q = art.data.joint_pos[env_id, j["dof_idx"]]
                chunks.append(_arti_map_points(pts_q0, j["joint_type"], axis, origin, (q - lower).view(1, 1)))
            pl = torch.cat(chunks, dim=0)
            root_pos = art.data.root_pos_w[env_id]
            root_quat = art.data.root_quat_w[env_id]
            return torch_utils.tf_apply(root_quat.unsqueeze(0), root_pos.unsqueeze(0), pl.unsqueeze(0))[0]
        obj = env.objects[name]
        pos = obj.data.root_pos_w[env_id]; quat = obj.data.root_quat_w[env_id]
        pl = env.object_point_clouds[name]["base"].to(device or pos.device)
        return torch_utils.tf_apply(quat.unsqueeze(0), pos.unsqueeze(0), pl.unsqueeze(0))[0]

    def _transform_vertices_to_world(self, obj, vertices_local: np.ndarray, env_id: int = 0) -> np.ndarray:
        root_pos = obj.data.root_pos_w[env_id].detach().cpu().numpy().astype(np.float32, copy=False)
        root_quat = obj.data.root_quat_w[env_id].detach().cpu().numpy().astype(np.float32, copy=False)
        rot = q2R_wxyz(root_quat).astype(np.float32, copy=False)
        return vertices_local @ rot.T + root_pos

    def _get_table_mesh_template(self, env):
        cache = self._get_or_init_env_mesh_cache(env)
        template = cache.get("table_template")
        if template is not None:
            return template
        base_env = self._unwrap_env(env)
        size = base_env.table_size.astype(np.float32)
        mesh = trimesh.creation.box(extents=size)
        faces = mesh.faces.astype(np.int32, copy=False)
        template = {
            "vertices_local": mesh.vertices.astype(np.float32, copy=False),
            "faces": faces,
            "faces_list": faces.tolist(),
        }
        cache["table_template"] = template
        return template

    def _get_object_mesh_templates(self, env):
        cache = self._get_or_init_env_mesh_cache(env)
        templates = cache.get("object_templates")
        if templates is not None:
            return templates
        base_env = self._unwrap_env(env)
        arti_names = set(getattr(base_env, "articulated_objects", {}).keys())
        templates = {}
        for obj_name, obj in base_env.objects.items():
            if obj_name == "table" or obj_name in arti_names:
                continue  # articulated objects use per-link templates (_articulation_world_mesh)
            templates[obj_name] = self._extract_collision_mesh_template(obj)
        cache["object_templates"] = templates
        return templates

    def extract_collision_mesh_from_env(self, obj, in_world=True, env_id: int = 0):
        template = self._extract_collision_mesh_template(obj)
        vertices = template["vertices_local"]
        if in_world:
            vertices = self._transform_vertices_to_world(obj, vertices, env_id=env_id)
        return trimesh.Trimesh(vertices=vertices, faces=template["faces"], process=False)
    
    def extract_scene_mesh(self, env, env_id: int = 0):
        env = self._unwrap_env(env)
        template = self._get_table_mesh_template(env)
        if self._is_table_active(env):
            table = env.table
            pose_w = table.data.root_pose_w[env_id].cpu().numpy().astype(np.float32, copy=False)
            pos = pose_w[:3]
            quat = pose_w[3:]  # wxyz
            rot = q2R_wxyz(quat).astype(np.float32, copy=False)
            vertices_world = template["vertices_local"] @ rot.T + pos
        else:
            # No-table episodes should not inject a support plane into cuRobo world collision.
            return None

        return {
            "vertices": vertices_world.tolist(),
            "faces": template["faces_list"],
        }

    def extract_object_meshes(self, env, env_id: int = 0):
        env = self._unwrap_env(env)
        objects = env.objects
        templates = self._get_object_mesh_templates(env)
        arti_names = set(getattr(env, "articulated_objects", {}).keys())
        object_meshes = dict()
        for obj_name, obj in objects.items():
            if obj_name == "table":
                continue
            parts_payload = None
            if obj_name in arti_names:
                # Articulated: build the mesh from per-link templates re-posed by current link world
                # poses, so an OPENED drawer shows up in the cuRobo world (rebuilt per node).
                # Also export the per-part split (drawer_i/door_i/base) so cuRobo can register
                # part-named obstacles for collision attribution.
                try:
                    parts = self._articulation_world_mesh_parts(
                        env, obj_name, env.articulated_objects[obj_name], env_id=env_id)
                    vall, fall, off = [], [], 0
                    parts_payload = {}
                    for label, (vw, faces) in parts.items():
                        vall.append(vw)
                        fall.append(np.asarray(faces, dtype=np.int64) + off)
                        off += len(vw)
                        parts_payload[label] = {"vertices": vw.tolist(), "faces": faces}
                    vertices_world = np.concatenate(vall, axis=0).astype(np.float32, copy=False)
                    faces_list = np.concatenate(fall, axis=0).astype(np.int32, copy=False).tolist()
                except Exception:
                    logging.exception(
                        "extract_object_meshes: per-part split failed for %s; falling back to combined.",
                        obj_name)
                    parts_payload = None
                    vertices_world, faces_list = self._articulation_world_mesh(
                        env, obj_name, env.articulated_objects[obj_name], env_id=env_id)
            else:
                template = templates[obj_name]
                vertices_world = self._transform_vertices_to_world(obj, template["vertices_local"], env_id=env_id)
                faces_list = template["faces_list"]
            # Asset-frame world quat (wxyz) for this obj at extraction time. cuRobo
            # uses this to set place_sphere.pose's rotation to the asset frame
            # (not the OBB principal-axis frame from trimesh PCA, which is
            # symmetry-ambiguous and produces 180-deg flips in place planning).
            asset_quat_wxyz = obj.data.root_quat_w[env_id].cpu().numpy().tolist()
            entry = {
                "vertices": vertices_world.tolist() if not isinstance(vertices_world, list) else vertices_world,
                "faces": faces_list,
                "asset_quat_wxyz": asset_quat_wxyz,
            }
            if parts_payload:
                entry["parts"] = parts_payload
            object_meshes[obj_name] = entry
        return object_meshes
    
    def get_grasped_and_collided_objs(self, env, env_id: int = 0):
        env = self._unwrap_env(env)
        data = env.contact.data

        if data.net_forces_w is None:
            return [], []
        # Lazy import to ensure isaacsim path is available after AppLauncher runs.
        import isaacsim.core.utils.torch as torch_utils

        force = env.contact.data.net_forces_w[env_id]  # (B, 3)
        obs_contact = force.norm(dim=-1) > 0.01

        # Align positions with available contact forces
        num_force_bodies = force.shape[0]
        num_kp = min(len(env.keypoint_indices), num_force_bodies)
        contact_pos = env.robot.data.body_pos_w[env_id][env.keypoint_indices[:num_kp], :]  # (K, 3) where K matches obs_contact slice
        obs_contact = obs_contact[:num_kp]

        grasped_objects = set()
        for name, obj in env.objects.items():
            if name == "table":
                continue
            # joint-aware world pcd (articulated cabinet drawers follow their opening; rigid = root pose)
            pcd_world = self._object_world_pcd_t(env, name, env_id=env_id, device=contact_pos.device)
            dists = torch.cdist(contact_pos.unsqueeze(0), pcd_world.unsqueeze(0)).min(dim=-1).values[0]
            contact_mask = obs_contact & (dists < 0.02)
            if contact_mask.any():
                grasped_objects.add(name)

        collided_objects = []
        self.grasped_objects = list(grasped_objects)
        return list(grasped_objects), collided_objects

    def min_object_pcd_distance(self, env, name_a: str, name_b: str, env_id: int = 0):
        """Min Euclidean distance between two objects' point clouds in world frame.
        Used as a proximity-based contact proxy (e.g. held object vs place target).
        Returns float('inf') if either object's pcd is unavailable."""
        env = self._unwrap_env(env)
        import isaacsim.core.utils.torch as torch_utils

        def _world_pcd(name):
            obj = env.objects.get(name)
            if obj is None or (name not in env.object_point_clouds and name not in getattr(env, "articulated_objects", {})):
                return None
            return self._object_world_pcd_t(env, name, env_id=env_id)

        pcd_a = _world_pcd(name_a)
        pcd_b = _world_pcd(name_b)
        if pcd_a is None or pcd_b is None:
            return float("inf")
        d = torch.cdist(pcd_a.unsqueeze(0), pcd_b.unsqueeze(0)).min()
        return float(d.item())

    ################################
    
    ##### Get Privileged Info #####
    
    def get_object_names(self, env):
        env = self._unwrap_env(env)
        return sorted(self._get_active_object_names(env))
    
    def get_object_init_pose(self, env, object_name: str):
        env = self._unwrap_env(env)
        return env.object_init_poses[object_name]["base"].cpu().numpy()  # (7,)
    
    def get_object_init_poses(self, env):
        env = self._unwrap_env(env)
        return {
            object_name: env.object_init_poses[object_name]["base"].cpu().numpy() for object_name in env.object_init_poses.keys()
        }

    def get_object_init_height(self, env, object_name: str):
        env = self._unwrap_env(env)
        return env.object_init_heights[object_name]["base"]  # float
    
    def get_object_init_heights(self, env):
        env = self._unwrap_env(env)
        return {
            object_name: env.object_init_heights[object_name]["base"] for object_name in env.object_init_heights.keys()
        }
    
    ## center location of objects at current time step
    def get_object_center_location(self, env, object_name: str, env_id: int = 0):
        env = self._unwrap_env(env)
        # Articulated objects: report the STATIC BASE AABB center (excludes drawers/doors), so
        # the reported center does not shift when a drawer is pulled out or a door swings open.
        if object_name in getattr(env, "articulated_objects", {}):
            pw = self._part_world_pcd_t(env, object_name, part="base", env_id=env_id).detach().cpu().numpy()
            center = 0.5 * (pw.min(axis=0) + pw.max(axis=0))
            return np.round(center.astype(float), 3).tolist()
        obj = env.objects[object_name]
        pos = obj.data.root_pos_w[env_id].cpu().numpy()
        return np.round(pos.astype(float), 3).tolist()
    
    def get_object_center_locations(self, env, env_id: int = 0):
        env = self._unwrap_env(env)
        object_center_3d = dict()
        for name in self._get_active_object_names(env):
            object_center_3d[name] = self.get_object_center_location(env, name, env_id=env_id)
        return object_center_3d
    
    ## pose of objects at current time step
    def get_object_pose(self, env, object_name: str, env_id: int = 0):
        env = self._unwrap_env(env)
        obj = env.objects[object_name]
        pose = obj.data.root_pose_w[env_id].cpu().numpy()
        return pose

    def get_object_root_velocity(self, env, object_name: str, env_id: int = 0):
        env = self._unwrap_env(env)
        obj = env.objects[object_name]
        root_vel = obj.data.root_vel_w[env_id].detach().cpu().numpy().astype(np.float32)
        return {
            "linear": root_vel[:3],
            "angular": root_vel[3:6],
        }
    
    def get_object_poses(self, env):
        env = self._unwrap_env(env)
        object_poses = dict()
        for name in self._get_active_object_names(env):
            object_poses[name] = self.get_object_pose(env, name)
        return object_poses

    # ------------------------------------------------------------------ articulation (drawers)
    # See docs/arti/scene_import.md. These operate on objects loaded via the `articulated` scene
    # JSON flag (a multi-DOF drawer cabinet). Joint pos/limits/set/norm are asset-agnostic; the
    # drawer_id -> (joint, movable link) mapping uses the LiberoCabinetFull naming.
    #
    # Parts (drawers / doors) are resolved from the asset URDF (the scene JSON `urdf_path`),
    # in URDF joint order. Drawers (LiberoCabinetFull): 3 prismatic joints joint_1/2/3 ->
    # movable_top/middle/bottom (order = top->bottom). Doors (LiberoMicrowave): 1 revolute joint
    # joint_0 -> movable_link. Each part carries joint type/axis/origin (object frame) so the
    # prismatic/revolute transforms are asset-driven, not hard-coded. See docs/arti/scene_import.md.

    def _articulation(self, env, object_name: str):
        env = self._unwrap_env(env)
        if object_name not in env.articulated_objects:
            raise KeyError(f"{object_name!r} is not an articulated object.")
        return env.articulated_objects[object_name]

    # --- unified [name][part] meta access (built by base_env._build_part_meta; parts:
    # "base", "drawer_<i>", "door_<j>". Legacy int drawer_id == movable-part position in
    # URDF order == get_movable_part_keys(...)[drawer_id]). ---

    def get_part_names(self, env, object_name: str) -> list:
        env = self._unwrap_env(env)
        return list(env.object_part_names[object_name])

    def get_movable_part_keys(self, env, object_name: str) -> list:
        """Movable part keys in URDF order; the list index is the legacy int drawer_id."""
        env = self._unwrap_env(env)
        return list(env.object_part_names[object_name][1:])

    def part_key_from_drawer_id(self, env, object_name: str, drawer_id: int) -> str:
        """Adapter for the external int schema (action_params 'drawer_id')."""
        return self.get_movable_part_keys(env, object_name)[int(drawer_id)]

    def _ensure_part_indices(self, env, object_name: str) -> dict:
        """Backfill dof_idx/movable_link_idx in env.object_joints[object_name]. These are
        positions in the SPAWNED articulation's joint/body arrays, resolvable only after sim
        init (and PhysX need not keep URDF order), hence lazy. Idempotent; strict on misses."""
        env = self._unwrap_env(env)
        joints = env.object_joints.get(object_name) or {}
        if joints and all(j["dof_idx"] is not None for j in joints.values()):
            return joints
        art = self._articulation(env, object_name)
        joint_names = list(art.joint_names)
        body_names = list(art.body_names)
        for part_key, j in joints.items():
            if j["joint"] not in joint_names:
                raise KeyError(f"{object_name!r} part {part_key!r}: URDF joint {j['joint']!r} not on "
                               f"articulation. Available joints: {joint_names}")
            if j["child"] not in body_names:
                raise KeyError(f"{object_name!r} part {part_key!r}: URDF child link {j['child']!r} not on "
                               f"articulation. Available bodies: {body_names}")
            j["dof_idx"] = int(art.find_joints(j["joint"])[0][0])
            j["movable_link_idx"] = int(art.find_bodies(j["child"])[0][0])
        return joints

    def _part_joint(self, env, object_name: str, part: str) -> dict:
        """object_joints entry of a movable part, with runtime indices ensured."""
        joints = self._ensure_part_indices(env, object_name)
        if part not in joints:
            raise KeyError(f"{object_name!r} has no movable part {part!r}; "
                           f"movable parts: {list(joints.keys())}")
        return joints[part]

    def get_part_joint_type(self, env, object_name: str, part: str) -> str:
        """"prismatic" (drawer) or "revolute" (door). Static URDF field — no sim needed."""
        env = self._unwrap_env(env)
        return env.object_joints[object_name][part]["joint_type"]

    def get_part_joint_norm(self, env, object_name: str, part: str, env_id: int = 0):
        """Normalized opening of one movable part: clamp((q - lower) / (upper - lower), 0, 1)."""
        dof_idx = self._part_joint(env, object_name, part)["dof_idx"]
        q = self.get_object_joint_positions(env, object_name, env_id)[dof_idx]
        lower, upper = self.get_object_joint_limits(env, object_name, env_id)[dof_idx]
        denom = max(float(upper - lower), 1e-6)
        return float(np.clip((q - lower) / denom, 0.0, 1.0))

    def get_part_state(self, env, object_name: str, part: str, env_id: int = 0) -> str:
        """3-bucket part state for the VLM prompt. Reuses the RL tasks' goal_joint_pos_norm as
        the cutoffs (single source of truth): open if norm>=open goal, closed if <=close goal,
        else partially open. See docs/arti/vlm_actor.md §4."""
        norm = self.get_part_joint_norm(env, object_name, part, env_id)
        cfg = self._unwrap_env(env).cfg
        open_goal = float(getattr(cfg, "drawer_open_goal_norm", 0.95))
        close_goal = float(getattr(cfg, "drawer_close_goal_norm", 0.0))
        if norm >= open_goal:
            return "open"
        if norm <= close_goal:
            return "closed"
        return "partially open"

    def get_part_axis_annotation(self, env, object_name: str, part: str):
        """{"x","y","z"} axis-annotation text of one part (kind template / scene JSON override;
        see part_meta.AXIS_ANNOTATION_BY_KIND), or None when absent. The text describes the
        part's own (link) frame — the same frame the alignment APIs use for a part target."""
        env = self._unwrap_env(env)
        return (env.object_axis_annotations.get(object_name) or {}).get(part)

    def get_opening_dir_world(self, env, object_name: str, part: str = "base", env_id: int = 0):
        """World-frame opening direction (unit np (3,)) of an asset/part: R_root @ cano opening.
        Cano dirs live in env.object_opening_dirs (drawer/rigid +z, door +x, door-arti base +x;
        see simulators/part_meta.py). NOTE: intentionally root-frame — a door's cavity opening
        sits on the static base, so it does not swing with the door joint."""
        env = self._unwrap_env(env)
        d = np.asarray(env.object_opening_dirs[object_name][part], dtype=np.float64)
        obj = env.objects[object_name]
        quat = obj.data.root_quat_w[env_id].detach().cpu().numpy().astype(np.float64)
        v = q2R_wxyz(quat) @ d
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def _part_world_pcd_t(self, env, object_name: str, part: str = "base", env_id: int = 0,
                          quat_override=None):
        """(P,3) world-frame pcd of one part (torch) — the single geometry path.
        base: root transform of the base part pcd (rigid = whole object; arti = static base
        link only, so an open drawer/door never balloons the base extent). Movable part:
        q0 pcd -> joint push by the CURRENT opening -> root transform. quat_override replaces
        the root orientation (predicted-pose extent queries)."""
        import isaacsim.core.utils.torch as torch_utils
        env = self._unwrap_env(env)
        pl = env.object_point_clouds[object_name][part]
        obj = env.objects[object_name]
        pos = obj.data.root_pos_w[env_id]
        if quat_override is None:
            quat = obj.data.root_quat_w[env_id]
        else:
            quat = torch.as_tensor(np.asarray(quat_override, dtype=np.float32), device=pos.device)
        if part != "base":
            from simulators.tasks import _arti_map_points
            j = self._part_joint(env, object_name, part)
            art = self._articulation(env, object_name)
            axis = torch.tensor(j["axis_obj"], device=pl.device, dtype=pl.dtype)
            origin = torch.tensor(j["origin_obj"], device=pl.device, dtype=pl.dtype)
            lower = art.data.joint_pos_limits[env_id, j["dof_idx"], 0]
            q = art.data.joint_pos[env_id, j["dof_idx"]]
            pl = _arti_map_points(pl, j["joint_type"], axis, origin, (q - lower).view(1, 1))
        return torch_utils.tf_apply(quat.unsqueeze(0), pos.unsqueeze(0), pl.unsqueeze(0))[0]

    def sample_part_init_pose(self, env, object_name: str, part: str, kind: str = "open", env_id: int = 0):
        """Sample one Stage-A init gripper pose for the selected movable part and transform it
        q0->world. kind="open" -> the part's *_handle_grasps.npz (grasp pose); kind="close" ->
        *_closepush_init.npz (front-panel press pose). Two-layer transform (skill_plumbing.md §5):
        q0 -> + axis*joint_delta (arti state) -> root pose (base). Returns (pos_world(3,), quat_world(4,)
        wxyz, finger, open) as numpy. Init-pose libraries live in the unified meta store
        (env.object_part_init_poses, loaded at build from the scene JSON `drawers` entry)."""
        import isaacsim.core.utils.torch as torch_utils
        env = self._unwrap_env(env)
        key = "handle_grasps" if kind == "open" else "closepush_init"
        lib = env.object_part_init_poses.get(object_name, {}).get(part, {}).get(key)
        if lib is None:
            raise KeyError(
                f"{object_name!r} part {part!r}: no {key!r} init-pose library "
                f"(scene JSON drawers entry missing it, or part has no specs)."
            )
        idx = int(np.random.randint(lib["pos"].shape[0]))
        pos_q0 = torch.tensor(lib["pos"][idx], device=env.device, dtype=torch.float32)
        finger = float(lib["finger"][idx]); jaw_open = float(lib["open"][idx])
        from simulators.tasks import _arti_map_points
        art = self._articulation(env, object_name)
        j = self._part_joint(env, object_name, part)
        dof_idx = j["dof_idx"]
        jtype = j["joint_type"]
        axis = torch.tensor(j["axis_obj"], device=env.device, dtype=torch.float32)
        origin = torch.tensor(j["origin_obj"], device=env.device, dtype=torch.float32)
        lower = art.data.joint_pos_limits[env_id, dof_idx, 0]
        q = art.data.joint_pos[env_id, dof_idx]
        joint_delta = (q - lower)
        # q0 -> current position: prismatic translate / revolute rotate about hinge (torch, batched).
        pos_cur_obj = _arti_map_points(pos_q0.unsqueeze(0), jtype, axis, origin, joint_delta.view(1, 1))[0]
        root_quat = art.data.root_quat_w[env_id]
        root_pos = art.data.root_pos_w[env_id]
        pos_world = torch_utils.quat_apply(root_quat.unsqueeze(0), pos_cur_obj.unsqueeze(0))[0] + root_pos
        # q0 -> current orientation: identity (prismatic) / rotate by delta about axis (revolute).
        # quat_mul (data_collect_utils) is numpy + non-batched -> use CPU numpy (4,) quaternions.
        root_quat_np = root_quat.detach().cpu().numpy().astype(np.float64)
        quat_q0_np = np.asarray(lib["quat"][idx], dtype=np.float64)
        if jtype == "prismatic":
            quat_cur_np = quat_q0_np
        else:
            axis_np = np.asarray(j["axis_obj"], dtype=np.float64)
            axis_np = axis_np / max(float(np.linalg.norm(axis_np)), 1e-9)
            half = 0.5 * float(joint_delta)
            qrot_np = np.array([np.cos(half), *(axis_np * np.sin(half))], dtype=np.float64)
            quat_cur_np = quat_mul(qrot_np, quat_q0_np)
        quat_world = quat_mul(root_quat_np, quat_cur_np)
        # grasp_site target -> panda_hand target: the npz pose is the panda_grip_site (TCP), which
        # sits +z*offset from panda_hand in the hand frame; cuRobo plans the panda_hand. Pull the
        # position back along the (unchanged) hand orientation. Mirrors IsaacLabEnvs
        # open_drawer_grasped_env._reset_idx step 3 (grasp_init_grip_site_offset_z=0.1025).
        GRIP_SITE_OFFSET_Z = 0.1025
        quat_world_t = torch.tensor(quat_world, device=env.device, dtype=torch.float32)
        offset_vec = torch.tensor([0.0, 0.0, GRIP_SITE_OFFSET_Z], device=env.device, dtype=torch.float32)
        hand_pos_world = pos_world - torch_utils.quat_apply(quat_world_t.unsqueeze(0), offset_vec.unsqueeze(0))[0]
        return (hand_pos_world.detach().cpu().numpy(), np.asarray(quat_world, dtype=np.float32), finger, jaw_open)

    def get_object_joint_positions(self, env, object_name: str, env_id: int = 0):
        art = self._articulation(env, object_name)
        return art.data.joint_pos[env_id].detach().cpu().numpy()  # (num_drawers,)

    def get_object_joint_limits(self, env, object_name: str, env_id: int = 0):
        art = self._articulation(env, object_name)
        return art.data.joint_pos_limits[env_id].detach().cpu().numpy()  # (num_drawers, 2)

    def set_object_joint_positions(self, env, object_name: str, joint_pos, env_id: int = 0):
        env = self._unwrap_env(env)
        art = self._articulation(env, object_name)
        jp = torch.as_tensor(joint_pos, device=art.device, dtype=torch.float32).reshape(1, -1)
        jv = torch.zeros_like(jp)
        env_ids = torch.tensor([env_id], device=art.device, dtype=torch.long)
        art.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

    def get_articulated_object_names(self, env) -> list:
        env = self._unwrap_env(env)
        return list(env.articulated_objects.keys())

    def get_object_canonical_xy_extents(self, env, object_name: str):
        env = self._unwrap_env(env)
        pcd = env.object_point_clouds[object_name]["base"].detach().cpu().numpy()
        x_extent = float(pcd[:, 0].max() - pcd[:, 0].min())
        y_extent = float(pcd[:, 1].max() - pcd[:, 1].min())
        return x_extent, y_extent

    def get_object_handle_side(self, env, object_name: str):
        env = self._unwrap_env(env)
        return env.object_handle_sides.get(object_name, {}).get("base")

    def get_object_category(self, env, object_name: str):
        env = self._unwrap_env(env)
        return env.object_categories.get(object_name, {}).get("base")

    def get_object_axis_lengths(self, env, object_name: str):
        # Strict (KeyError on unannotated objects), matching the legacy obj_cfg["axis_length_*"].
        env = self._unwrap_env(env)
        return env.object_axis_lengths[object_name]["base"]

    ## 3D axis-aligned bounding box of objects at current time step
    def get_object_3d_range(self, env, object_name: str, env_id: int = 0, quat_override=None,
                            part: str = "base"):
        """(xmin, xmax, ymin, ymax, zmin, zmax) world AABB of an asset/part, from the single
        pcd geometry path (_part_world_pcd_t). part="base": rigid = whole object; arti = the
        STATIC base link only, so an open drawer/door does not balloon the reported range.
        Movable part (e.g. "drawer_1"): part pcd pushed by the current joint opening — the
        former get_object_drawer_aabb. quat_override replaces the root orientation
        (predicted-pose extent queries, e.g. place z-extent re-prediction)."""
        env = self._unwrap_env(env)
        if object_name == "table":
            return env.table_3d_range
        pw = self._part_world_pcd_t(
            env, object_name, part=part, env_id=env_id, quat_override=quat_override
        ).detach().cpu().numpy()
        return (round(float(pw[:, 0].min()), 3), round(float(pw[:, 0].max()), 3),
                round(float(pw[:, 1].min()), 3), round(float(pw[:, 1].max()), 3),
                round(float(pw[:, 2].min()), 3), round(float(pw[:, 2].max()), 3))
    
    def get_object_3d_ranges(self, env, env_id: int = 0):
        env = self._unwrap_env(env)
        object_range_3d = dict()
        for name in self._get_active_object_names(env):
            if name == "table":
                object_range_3d[name] = env.table_3d_range
            else:
                object_range_3d[name] = self.get_object_3d_range(env, name, env_id=env_id)
        return object_range_3d
    
    def get_object_2d_range(self, env, object_name: str, views):
        env = self._unwrap_env(env)
        all_ranges = self.get_object_2d_ranges(env, views)
        return all_ranges.get(object_name, {view: None for view in views})
    
    def get_object_2d_ranges(self, env, views, env_id: int = 0):
        env = self._unwrap_env(env)
        object_names = self._get_active_object_names(env)
        object_range_2d = {
            name: {view: None for view in views}
            for name in object_names
        }
        segs = self.get_semantic_segmentations(env, views, env_id=env_id)

        for view in views:
            seg_image = segs[view]
            if seg_image is None:
                continue
            cam = self.get_camera_by_name(env, view)
            cam_info = cam.data.info
            seg_info = cam_info.get("semantic_segmentation")
            if seg_info is None:
                continue
            id_to_labels = seg_info.get("idToLabels", {})
            object2id = {
                v["class"]: int(k)
                for k, v in id_to_labels.items()
                if v["class"] in object_names
            }
            for name in object_names:
                sem_id = object2id.get(name)
                if sem_id is None:
                    continue
                mask = seg_image == sem_id
                if not np.any(mask):
                    continue
                ys, xs = np.where(mask)
                xmin = int(xs.min())
                xmax = int(xs.max())
                ymin = int(ys.min())
                ymax = int(ys.max())
                object_range_2d[name][view] = (
                    round(float(xmin), 3),
                    round(float(xmax), 3),
                    round(float(ymin), 3),
                    round(float(ymax), 3),
                )
        return object_range_2d

    def collect_scene_state(self, env, views, env_id: int = 0, need_object_range_2d: bool = True):
        env = self._unwrap_env(env)
        rgbs = self.get_rgbs(env, views, env_id=env_id)
        ixts = self.get_camera_intrinsics(env, views)
        exts = self.get_camera_extrinsics(env, views)
        pcds = self.get_pcds(env, views, env_id=env_id)
        h = {view: rgbs[view].shape[0] for view in views}
        w = {view: rgbs[view].shape[1] for view in views}

        scene_mesh = self.extract_scene_mesh(env, env_id=env_id)
        object_meshes = self.extract_object_meshes(env, env_id=env_id)

        object_center_3d = self.get_object_center_locations(env, env_id=env_id)
        object_range_3d = self.get_object_3d_ranges(env, env_id=env_id)
        active_object_names = self._get_active_object_names(env)
        if need_object_range_2d:
            object_range_2d = self.get_object_2d_ranges(env, views, env_id=env_id)
        else:
            object_range_2d = {
                name: {view: None for view in views}
                for name in active_object_names
            }
        active_names = set(active_object_names)
        object_axis_annotation = {
            name: anno["base"] for name, anno in env.object_axis_annotations.items() if name in active_names
        }
        object_upright_status = self.get_object_upright_status(env, env_id=env_id)

        grasped_objects, collided_objects = self.get_grasped_and_collided_objs(env, env_id=env_id)
        q = self.get_joint_positions(env, env_id=env_id)
        return rgbs, pcds, scene_mesh, object_meshes, h, w, ixts, exts, \
            object_center_3d, object_range_3d, object_range_2d, object_axis_annotation, \
            object_upright_status, grasped_objects, collided_objects, q

    def get_object_upright_status(self, env, env_id: int = 0) -> dict:
        """Return {name: "upright"|"lying"} for whitelisted objects.

        Two branches by whitelist:
          - container_categories: canonical +z is the opening direction by
            annotation convention. Signed test ``R[2, 2] >= threshold``
            (opening must face world +z, not down).
          - elongated_categories: long axis = ``argmax(axis_length_{x,y,z})``
            in canonical frame. ``|R[2, longest_idx]| >= threshold`` (long
            axis vertical, either direction counts).

        Non-listed objects are omitted.
        """
        env = self._unwrap_env(env)
        if not self.container_categories and not self.elongated_categories:
            return {}
        thr = self.container_upright_cos_threshold
        out: dict[str, str] = {}
        for name in self._get_active_object_names(env):
            categories = []
            category = self.get_object_category(env, name)
            if isinstance(category, str) and category:
                categories.append(category)
            categories.append(name)
            base, sep, suffix = name.rpartition("_")
            if sep and suffix.isdigit():
                categories.append(base)
            is_container = any(cat in self.container_categories for cat in categories)
            is_elongated = any(cat in self.elongated_categories for cat in categories)
            if not (is_container or is_elongated):
                continue
            pose = self.get_object_pose(env, name, env_id=env_id)
            R = q2R_wxyz(np.asarray(pose[3:7], dtype=np.float64))
            if is_container:
                cos_signed = float(R[2, 2])
                out[name] = "upright" if cos_signed >= thr else "lying"
            else:
                try:
                    Lx, Ly, Lz = self.get_object_axis_lengths(env, name)
                except Exception:
                    continue
                longest_idx = int(np.argmax([Lx, Ly, Lz]))
                cos_signed = float(R[2, longest_idx])
                out[name] = "upright" if abs(cos_signed) >= thr else "lying"
        return out
    
    ################################


    def get_rotation_from_alignment(self, env, held_object_name, target_object_name, alignment, env_id: int = 0):
        env = self._unwrap_env(env)

        held_object = env.objects[held_object_name]
        target_object = env.objects[target_object_name]

        rot_held = held_object.data.root_quat_w[env_id].cpu().numpy()  # wxyz
        rot_target = target_object.data.root_quat_w[env_id].cpu().numpy()  # wxyz

        goal_quat_held = compute_alignment_rotation(rot_held, rot_target, alignment)
        return goal_quat_held

    def _alignment_target_quat_w(self, env, target_object_name, part=None, env_id: int = 0):
        """World orientation (wxyz np) of an alignment target. part=None/"base": the object
        root frame. Movable part: the part LINK's CURRENT frame (an open door's axes swing
        with the joint; a prismatic drawer's link frame follows its URDF joint origin)."""
        env = self._unwrap_env(env)
        if part is not None and part != "base":
            j = self._part_joint(env, target_object_name, part)
            art = self._articulation(env, target_object_name)
            return art.data.body_quat_w[env_id, j["movable_link_idx"]].detach().cpu().numpy()
        return env.objects[target_object_name].data.root_quat_w[env_id].cpu().numpy()

    def get_rotation_candidates_from_alignment(
        self,
        env,
        held_object_name,
        target_object_name,
        alignment,
        sweep_angles_deg=None,
        relax_ladder_deg=None,
        meta_out=None,
        part=None,
        env_id: int = 0,
    ):
        env = self._unwrap_env(env)

        held_object = env.objects[held_object_name]

        rot_held = held_object.data.root_quat_w[env_id].cpu().numpy()  # wxyz
        rot_target = self._alignment_target_quat_w(env, target_object_name, part, env_id)  # wxyz

        return compute_alignment_rotation_candidates(
            rot_held, rot_target, alignment, sweep_angles_deg=sweep_angles_deg,
            relax_ladder_deg=relax_ladder_deg, meta_out=meta_out,
        )

    def get_gripper_rotation_candidates_from_alignment(
        self,
        env,
        held_object_name,
        target_object_name,
        alignment,
        sweep_angles_deg=None,
        relax_ladder_deg=None,
        meta_out=None,
        part=None,
        env_id: int = 0,
    ):
        """Align a chosen GRIPPER (EE) axis to a target-object axis.

        Unlike get_rotation_candidates_from_alignment (which aligns the held object's
        own axis), the alignment src here is the gripper itself:
        ``alignment = (gripper_axis, target_axis, direction)`` where gripper_axis is
        expressed in the EE (hand) local frame returned by get_ee_pose -- e.g. for the
        panda hand: x = palm normal, y = finger open/close, z = approach (wrist->fingertips).

        Returns HELD-OBJECT goal quats (wxyz), drop-in compatible with the object-goal
        motion path: we (1) solve the EE goal quats satisfying the alignment, then (2) map
        each EE goal to the held-object goal via the current rigid grasp
        ``R_rel = inv(ee_now) @ obj_now``, so curobo's get_ee_goal_from_object_goal
        reproduces exactly that EE orientation.
        """
        env_u = self._unwrap_env(env)
        held_object = env_u.objects[held_object_name]

        ee_quat_now = self.get_ee_pose(env, env_id=env_id)[3:].astype(np.float64)  # wxyz
        rot_target = self._alignment_target_quat_w(env_u, target_object_name, part, env_id)  # wxyz
        obj_quat_now = held_object.data.root_quat_w[env_id].cpu().numpy().astype(np.float64)  # wxyz

        # EE goal quats satisfying the gripper-axis -> target-axis alignment.
        ee_goal_candidates = compute_alignment_rotation_candidates(
            ee_quat_now, rot_target, alignment, sweep_angles_deg=sweep_angles_deg,
            relax_ladder_deg=relax_ladder_deg, meta_out=meta_out,
        )

        # Rigid grasp: object orientation expressed in the EE frame (constant during the grasp).
        q_rel = quat_mul(quat_conj(ee_quat_now), obj_quat_now)

        # Object goal quat for each EE goal: q_obj_goal = q_ee_goal (x) q_rel.
        obj_goal_candidates = [
            quat_mul(np.asarray(q_ee_goal, dtype=np.float64), q_rel)
            for q_ee_goal in ee_goal_candidates
        ]
        return obj_goal_candidates

    def get_gripper_approach_axis_world(
        self,
        env,
        held_object_name,
        target_object_name,
        alignment,
        sweep_angles_deg=None,
        part=None,
        env_id: int = 0,
    ):
        """World-frame unit vector the gripper's approach axis points along at the aligned
        EE goal (minimal-rotation candidate 0). This is n_app for gripper-orientation
        placement: it tells the engine whether the approach is vertical (top-down) or
        non-vertical (e.g. a horizontal microwave insert), and which way to retract.
        Returns None if the alignment yields no candidate.
        """
        env_u = self._unwrap_env(env)
        ee_quat_now = self.get_ee_pose(env, env_id=env_id)[3:].astype(np.float64)  # wxyz
        rot_target = self._alignment_target_quat_w(env_u, target_object_name, part, env_id)  # wxyz
        ee_goal_candidates = compute_alignment_rotation_candidates(
            ee_quat_now, rot_target, alignment, sweep_angles_deg=sweep_angles_deg
        )
        if not ee_goal_candidates:
            return None
        R_goal = q2R_wxyz(np.asarray(ee_goal_candidates[0], dtype=np.float64))
        n_app = R_goal @ np.asarray(self.ee_approach_local, dtype=np.float64)
        norm = float(np.linalg.norm(n_app))
        if norm < 1e-9:
            return None
        return (n_app / norm).astype(np.float64)


    def make_env(self):
        ## make config
        ## TODO: should modify config
        global _MAKE_ENV_CALL_COUNT
        _MAKE_ENV_CALL_COUNT += 1
        print(
            f"[simulator.make_env] call={_MAKE_ENV_CALL_COUNT} pid={os.getpid()} ppid={os.getppid()} task={getattr(self.args, 'task', None)}",
            flush=True,
        )
        self._invalidate_mesh_cache()
        scene_desc_file = getattr(self.args, "scene_desc_file", None)
        if not scene_desc_file:
            raise ValueError("scene_desc_file is required before make_env; set env.scene_desc_file in config/overrides.")
        cfg = self._BaseEnvCfg()
        # Swap in the WidowX robot (USD/init_pos/actuators) when robot_name=widowx;
        # BaseEnvCfg.robot stays Franka by default. Done here (not via env yaml)
        # because ImplicitActuatorCfg objects can't be constructed by the yaml
        # override mechanism. env_overrides applied below can still tweak fields.
        if "widowx" in (getattr(self.args, "robot_name", "") or "").lower():
            import copy as _copy
            from simulators.base_env_cfg import WIDOWX_ROBOT_CFG
            from simulators.robot_conventions import resolve_robot_names
            # Deep-copy: per-run mutations (e.g. the sink init-qpos preset below)
            # must not leak into the shared module-level constant.
            cfg.robot = _copy.deepcopy(WIDOWX_ROBOT_CFG)
            # Robot identity for base_env joint/body-name resolution, and the
            # matching contact-sensor prims (the default cfg.contact targets
            # the Franka finger bodies and would not bind on WidowX).
            cfg.robot_name = "widowx"
            cfg.contact.prim_path = (
                f"/World/envs/env_.*/Robot/{resolve_robot_names('widowx').contact_prim_expr}"
            )
        pool_size = getattr(self.args, "pool_size", 1)
        cfg.scene.num_envs = pool_size if pool_size > 1 else (self.args.num_envs if hasattr(self.args, "num_envs") else cfg.scene.num_envs)
        cfg.sim.device = self.args.device if hasattr(self.args, "device") else cfg.sim.device
        
        cfg.enable_pick = self.args.enable_pick
        cfg.enable_place = self.args.enable_place
        cfg.enable_pose = self.args.enable_pose
        cfg.enable_open_drawer = self.args.enable_open_drawer
        cfg.enable_close_drawer = self.args.enable_close_drawer
        cfg.enable_close_door = self.args.enable_close_door
        if self.env_overrides:
            env_overrides = {k: v for k, v in self.env_overrides.items() if k != "actuators"}
            self._apply_env_overrides(cfg, env_overrides)
            actuator_overrides = self.env_overrides.get("actuators") or {}
            for act_name, act_fields in actuator_overrides.items():
                if not isinstance(act_fields, dict) or act_name not in cfg.robot.actuators:
                    continue
                act_cfg = cfg.robot.actuators[act_name]
                for field, val in act_fields.items():
                    if val is not None:
                        setattr(act_cfg, field, val)
        # Per-task init-qpos preset (after overrides so the env yaml can set it).
        _qpos_preset = (getattr(cfg, "robot_init_qpos_preset", "") or "").strip()
        if _qpos_preset == "widowx_sink":
            from simulators.base_env_cfg import WIDOWX_SINK_INIT_QPOS
            cfg.robot.init_state.joint_pos = dict(WIDOWX_SINK_INIT_QPOS)
            print("[make_env] robot_init_qpos_preset=widowx_sink")
        elif _qpos_preset:
            raise ValueError(f"Unknown robot_init_qpos_preset: {_qpos_preset!r}")
        # Keep env-side camera data_types in sync with simulator collection flags.
        cfg.collect_depth = bool(self.collect_depth)
        cfg.collect_segmentation = bool(self.collect_segmentation)
        cfg.sim.render_interval = cfg.decimation
        front_camera_count = int(getattr(cfg, "front_camera_count", 1))
        if front_camera_count < 1:
            raise ValueError(f"env.front_camera_count must be >= 1, got {front_camera_count}")
        self._refresh_camera_name_lists(front_camera_count=front_camera_count)
        # Non-wrist camera spawning follows the simulator camera list; wrist
        # spawning stays governed by env.enable_wrist_camera alone.
        cfg.camera_spawn_names = tuple(self._base_camera_names)
        if "wrist" in self._base_camera_names and not bool(getattr(cfg, "enable_wrist_camera", True)):
            raise ValueError(
                "camera_names includes 'wrist' but env.enable_wrist_camera is false; "
                "drop 'wrist' from simulator camera_names or enable the wrist camera."
            )

        agent_cfg = self._default_agent_cfg
        # wrap around environment for rl-games
        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

        ## make env
        try:
            base_env = gym.make(
                self.args.task,
                cfg=cfg,
                scene_desc_file=scene_desc_file,
                render_mode=self.args.render_mode,
            )

            # wrap around environment for rl-games
            env = RlGamesVecEnvWrapper(base_env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

            # self._register_rlgames_env()
            vecenv.register(
                "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
            )
            env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
        except BaseException:
            # Catch BaseException to surface silent SystemExit from native/launcher stack.
            import traceback
            print("[simulator.make_env] Failed while creating env. Full traceback:", flush=True)
            traceback.print_exc()
            raise

        return env

    def _apply_env_overrides(self, cfg, overrides):
        if not isinstance(overrides, dict):
            raise ValueError("env overrides must be a dict of BaseEnvCfg fields")
        self._apply_overrides_recursive(cfg, overrides)

    def _apply_overrides_recursive(self, cfg_obj, overrides, path=""):
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(cfg_obj, key):
                raise KeyError(f"Unknown env override '{path}{key}'")
            current = getattr(cfg_obj, key)
            if isinstance(value, dict):
                if isinstance(current, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key in current and hasattr(current[sub_key], "__dataclass_fields__"):
                            self._apply_overrides_recursive(current[sub_key], sub_value, path=f"{path}{key}.{sub_key}.")
                        else:
                            current[sub_key] = sub_value
                elif current is None:
                    setattr(cfg_obj, key, value)
                else:
                    self._apply_overrides_recursive(current, value, path=f"{path}{key}.")
            else:
                setattr(cfg_obj, key, value)

    def shutdown_env(self, env):
        self._invalidate_mesh_cache(env)
        env.close()

    def shutdown_app(self):
        shutdown_sim_app()

    def reset_stage(self):
        """Clear USD stage to allow fresh scene creation without restarting Kit."""
        self._invalidate_mesh_cache()
        _ensure_isaaclab_imports()
        try:
            import omni.usd
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            was_playing = timeline.is_playing()
            if was_playing:
                timeline.stop()
                timeline.commit()
            ctx = omni.usd.get_context()
            try:
                ctx.close_stage()
            except Exception:
                pass
            self._reset_replicator()
            ctx.new_stage()
            if was_playing:
                timeline.play()
                timeline.commit()
        except Exception:
            return
        try:
            if _SIM_APP is not None:
                _SIM_APP.update()
        except Exception:
            pass

    def reset(self, env, reset_level="task", **kwargs):
        base_env = self._unwrap_env(env)
        if reset_level == "env":
            use_reset_params = bool(kwargs.get("use_reset_params", False))
            reset_params = kwargs.get("reset_params", None)
            if hasattr(base_env, "set_next_env_reset_params"):
                if use_reset_params:
                    if reset_params is None:
                        raise ValueError("reset_params is required when use_reset_params=True.")
                    base_env.set_next_env_reset_params(
                        reset_params,
                        enabled=True,
                        consume_once=bool(kwargs.get("consume_reset_params_once", True)),
                    )
                else:
                    base_env.set_next_env_reset_params(None, enabled=False, consume_once=True)
            base_env.enable_env_reset()
            obs = env.reset()
            base_env.disable_env_reset()
        elif reset_level == "task":
            assert "task_name" in kwargs, "task reset requires task argument"
            task_name = kwargs["task_name"]
            assert "object_name" in kwargs, "task reset requires object argument"
            object_name = kwargs["object_name"]
            use_top_pcd_for_obs = bool(kwargs.pop("use_top_pcd_for_obs", False))
            use_handle_pcd_for_obs = bool(kwargs.pop("use_handle_pcd_for_obs", False))
            use_half_pcd_for_obs = bool(kwargs.pop("use_half_pcd_for_obs", False))
            if int(use_top_pcd_for_obs) + int(use_handle_pcd_for_obs) + int(use_half_pcd_for_obs) > 1:
                raise ValueError(
                    "At most one of use_top_pcd_for_obs / use_handle_pcd_for_obs / "
                    "use_half_pcd_for_obs can be True."
                )
            base_env.set_object(object_name)
            # Arti tasks: the part key is resolved at the skill boundary and arrives in
            # kwargs["part"]; it flows to set_task -> task.prepare (the standard task-param
            # path; the task wires its own obs from the unified meta store there). Runtime
            # joint/body indices must be ensured before prepare reads them.
            if kwargs.get("part") is not None and task_name in ("open_drawer", "close_drawer", "close_door"):
                self._ensure_part_indices(base_env, object_name)
            base_env.set_task(**kwargs)
            if use_handle_pcd_for_obs:
                obs_pcd_mode = "handle"
            elif use_top_pcd_for_obs:
                obs_pcd_mode = "top"
            elif use_half_pcd_for_obs:
                obs_pcd_mode = "half"
            else:
                obs_pcd_mode = "full"
            base_env.set_obs_pcd_mode(obs_pcd_mode)
            obs = env.reset()

        instruction = getattr(self.args, "instruction", "")
        return obs, instruction
    
    def setup_task_for_slot(self, env, task_name: str, object_name: str, env_id: int, **kwargs):
        """Task-level setup for a single env slot without disturbing other slots.

        Does NOT call env.reset(). Sets task/object state and resets only the
        per-env buffers for env_id by calling task.reset(env_ids=[env_id]) directly.
        Returns full-batch observations (num_envs, obs_dim).
        """
        base_env = self._unwrap_env(env)
        use_top_pcd_for_obs = bool(kwargs.pop("use_top_pcd_for_obs", False))
        use_handle_pcd_for_obs = bool(kwargs.pop("use_handle_pcd_for_obs", False))
        use_half_pcd_for_obs = bool(kwargs.pop("use_half_pcd_for_obs", False))
        if int(use_top_pcd_for_obs) + int(use_handle_pcd_for_obs) + int(use_half_pcd_for_obs) > 1:
            raise ValueError(
                "At most one of use_top_pcd_for_obs / use_handle_pcd_for_obs / "
                "use_half_pcd_for_obs can be True."
            )
        base_env.set_object(object_name)
        if kwargs.get("part") is not None and task_name in ("open_drawer", "close_drawer", "close_door"):
            self._ensure_part_indices(base_env, object_name)
        base_env.set_task(task_name=task_name, object_name=object_name, **kwargs)
        if use_handle_pcd_for_obs:
            obs_pcd_mode = "handle"
        elif use_top_pcd_for_obs:
            obs_pcd_mode = "top"
        elif use_half_pcd_for_obs:
            obs_pcd_mode = "half"
        else:
            obs_pcd_mode = "full"
        base_env.set_obs_pcd_mode(obs_pcd_mode)
        env_ids = torch.tensor([env_id], device=base_env.device, dtype=torch.long)
        base_env.prev_actions[env_ids] = 0.0
        base_env.task.reset(env=base_env, env_ids=env_ids)
        return {"obs": base_env._get_observations()["policy"]}

    def uses_half_pcd_for_object(self, env, object_name: str) -> bool:
        """Whether the half-pcd obs mode should fire for this object.

        Returns True iff a half pcd was loaded for `object_name` at env init
        time, which happens when cfg.use_half_pcd=True and the object's
        category appears in cfg.half_pcd_list. False otherwise (always False
        when cfg.use_half_pcd=False, since the half-pcd dict stays empty).
        """
        base_env = self._unwrap_env(env)
        return object_name in getattr(base_env, "object_half_point_clouds", {})


    def unset(self, env):
        base_env = self._unwrap_env(env)
        base_env.unset_task()
        base_env.unset_object()

    def collect_replay_reset_params(self, env) -> dict:
        base_env = self._unwrap_env(env)
        if not hasattr(base_env, "export_current_reset_params"):
            raise RuntimeError("Base env does not support export_current_reset_params.")
        return base_env.export_current_reset_params(env_id=0)

    def batch_step(self, env, actions_per_slot: dict, action_mode: int = 1, **kwargs):
        """Step all env slots simultaneously.

        Args:
            actions_per_slot: {env_id: action_array} for every active slot.
                Slots absent from the dict receive a zero action.
            action_mode: applied uniformly to all slots (all parallel execution
                uses ACTION_MODE_DELTA_EEPOSE = 1).

        Returns:
            obs_batch  - raw obs tensor (num_envs, obs_dim)
            dones      - bool tensor (num_envs,)
            extras     - dict
        """
        import torch as _torch
        base_env = self._unwrap_env(env)
        disable_action_ema = bool(kwargs.get("disable_action_ema", False))
        base_env.set_control_params(
            mode=action_mode,
            is_release=kwargs.get("is_release", False),
            disable_action_ema=disable_action_ema,
        )
        # Determine action dimension from the first provided action.
        sample_action = next(iter(actions_per_slot.values()))
        if isinstance(sample_action, _torch.Tensor):
            action_dim = int(sample_action.shape[-1])
        else:
            action_dim = int(np.asarray(sample_action).shape[-1])
        num_envs = int(base_env.num_envs)
        batched = _torch.zeros((num_envs, action_dim), dtype=_torch.float32, device=base_env.device)
        for eid, act in actions_per_slot.items():
            if isinstance(act, _torch.Tensor):
                batched[eid] = act.to(device=base_env.device, dtype=_torch.float32)
            else:
                batched[eid] = _torch.tensor(np.asarray(act, dtype=np.float32), device=base_env.device)
        obs_processor = getattr(env, "_process_obs", None)
        result = env.step(batched)
        obs_out, rew, dones, extras = self._standardize_step_output(result, obs_processor)
        return obs_out, dones, extras

    def step(self, env, action, action_mode=1, **kwargs):
        base_env = self._unwrap_env(env)
        disable_action_ema = bool(kwargs.get("disable_action_ema", False))
        base_env.set_control_params(
            mode=action_mode,
            is_release=kwargs.get("is_release", False),
            disable_action_ema=disable_action_ema,
        )
        self._cache_step_action_context(base_env, action, action_mode, disable_action_ema=disable_action_ema)
        # RlGamesVecEnvWrapper exposes _process_obs to ww observations; reuse it when present
        obs_processor = getattr(env, "_process_obs", None)

        # Use unwrapped env for raw joint targets (action_mode=0) to avoid action-space clipping
        if action_mode == 0:
            result = base_env.step(to_tensor(action, device=base_env.device))
        else:
            result = env.step(to_tensor(action, device=base_env.device))
        return self._standardize_step_output(result, obs_processor)

    def _standardize_step_output(self, result, obs_processor):
        """Normalize step return to (obs, rew, dones, extras) for both wrapped and unwrapped envs."""
        # Wrapped VecEnv returns 4-tuple already
        if len(result) == 4:
            obs_out, rew, dones, extras = result
        elif len(result) == 5:
            obs_dict, rew, terminated, truncated, extras = result
            obs_out = obs_processor(obs_dict) if obs_processor is not None else obs_dict
            dones = terminated | truncated
        else:
            raise RuntimeError(f"Unexpected step result length: {len(result)}")

        # Align extras key with RL-Games convention if needed
        if isinstance(extras, dict) and "log" in extras and "episode" not in extras:
            extras = extras.copy()
            extras["episode"] = extras.pop("log")
        return obs_out, rew, dones, extras
    
    def get_joint_positions(self, env, env_id: int = 0):
        base_env = self._unwrap_env(env)
        # find_joints expects a sequence; avoid passing a generator.
        arm_ids, _ = base_env.robot.find_joints(list(self.robot_joint_names))
        q_arm = base_env.robot.data.joint_pos[:, arm_ids]
        return q_arm[env_id].detach().cpu().numpy()

    def get_gripper_position(self, env, env_id: int = 0):
        base_env = self._unwrap_env(env)
        # Use mean of the two finger joints to represent scalar gripper opening.
        gripper_pos = base_env.robot.data.joint_pos[env_id, -2:]
        return float(gripper_pos.mean().detach().cpu().numpy())

    def get_gripper_positions(self, env, env_id: int = 0):
        base_env = self._unwrap_env(env)
        gripper_pos = base_env.robot.data.joint_pos[env_id, -2:]
        if self.max_gripper_opening is None and hasattr(base_env, "max_gripper_opening"):
            self.max_gripper_opening = float(base_env.max_gripper_opening)
        return gripper_pos.detach().cpu().numpy().astype(float)

    def get_ee_pose(self, env, env_id: int = 0):
        base_env = self._unwrap_env(env)
        eef_pos = base_env.robot.data.body_pos_w[:, base_env.hand_link_idx][env_id].detach().cpu().numpy().astype(np.float32)
        eef_quat = base_env.robot.data.body_quat_w[:, base_env.hand_link_idx][env_id].detach().cpu().numpy().astype(np.float32)
        return np.concatenate([eef_pos, eef_quat], axis=0)

    def get_max_gripper_opening(self, env):
        base_env = self._unwrap_env(env)
        if self.max_gripper_opening is None and hasattr(base_env, "max_gripper_opening"):
            self.max_gripper_opening = float(base_env.max_gripper_opening)
        return self.max_gripper_opening

    def _to_joint_target_np(self, base_env, action, env_id: int = 0) -> np.ndarray:
        act = to_tensor(action, base_env.device).to(torch.float32)
        if act.ndim == 2:
            act = act[env_id]
        act = act.detach().cpu().numpy().astype(np.float32, copy=False)
        arm_dim = int(len(base_env.arm_joint_indices))
        expected_dim = arm_dim + 2
        if act.shape[-1] != expected_dim:
            raise ValueError(f"Expected motion-planning action dim={expected_dim}, got {act.shape[-1]}")
        return act

    def _get_current_joint_target_np(self, base_env, env_id: int = 0) -> np.ndarray:
        arm_ids = list(base_env.arm_joint_indices)
        q_arm = base_env.robot.data.joint_pos[env_id, arm_ids].detach().cpu().numpy().astype(np.float32)
        q_gripper = base_env.robot.data.joint_pos[env_id, -2:].detach().cpu().numpy().astype(np.float32)
        return np.concatenate([q_arm, q_gripper], axis=0)

    def _cache_step_action_context(self, base_env, action, action_mode: int, disable_action_ema: bool = False) -> None:
        ctx = {
            "action_mode": int(action_mode),
            "disable_action_ema": bool(disable_action_ema),
        }
        if int(action_mode) == 0:
            ctx["joint_pre"] = self._get_current_joint_target_np(base_env)
            ctx["joint_target"] = self._to_joint_target_np(base_env, action)
        self._last_step_action_ctx = ctx

    def _ensure_fk_model(self):
        if self._fk_robot_model is not None and self._fk_joint_state_cls is not None:
            return
        try:
            from curobo.types.base import TensorDeviceType
            from curobo.types.robot import RobotConfig, JointState
            from curobo.util_file import join_path, load_yaml, get_robot_configs_path
            from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
        except Exception as exc:
            raise RuntimeError("curobo is required to compute FK for action_mode=0 data collection.") from exc

        tensor_args = TensorDeviceType()
        robot_dict = load_yaml(join_path(get_robot_configs_path(), self.robot_config))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_dict, tensor_args)
        self._fk_robot_model = CudaRobotModel(robot_cfg.kinematics)
        self._fk_joint_state_cls = JointState

    def _fk_ee_pose_from_joint(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_fk_model()
        q = self.get_curobo_q_from_simulator(np.asarray(arm_q, dtype=np.float32))
        q_t = torch.tensor(q, dtype=torch.float32, device="cuda")[None]
        joint_state = self._fk_joint_state_cls.from_position(
            q_t[:, self.robot_active_joint_indices],
            joint_names=self.robot_joint_names,
        )
        fk = self._fk_robot_model.compute_kinematics_from_joint_state(joint_state)
        ee_pos = fk.ee_pose.position[0].detach().cpu().numpy().astype(np.float32)
        ee_quat = fk.ee_pose.quaternion[0].detach().cpu().numpy().astype(np.float32)
        return ee_pos, ee_quat

    def _compute_mode0_semantics_from_joint_transition(self, base_env, joint_pre: np.ndarray, joint_target: np.ndarray):
        arm_dim = int(len(base_env.arm_joint_indices))
        pre_arm = np.asarray(joint_pre[:arm_dim], dtype=np.float32)
        tgt_arm = np.asarray(joint_target[:arm_dim], dtype=np.float32)
        pre_gripper = np.asarray(joint_pre[arm_dim:], dtype=np.float32)
        tgt_gripper = np.asarray(joint_target[arm_dim:], dtype=np.float32)

        pre_pos, pre_quat = self._fk_ee_pose_from_joint(pre_arm)
        tgt_pos, tgt_quat = self._fk_ee_pose_from_joint(tgt_arm)
        pre_quat = np.asarray(pre_quat, dtype=np.float64)
        tgt_quat = np.asarray(tgt_quat, dtype=np.float64)
        pre_quat = pre_quat / max(float(np.linalg.norm(pre_quat)), 1e-12)
        tgt_quat = tgt_quat / max(float(np.linalg.norm(tgt_quat)), 1e-12)
        # Enforce same quaternion hemisphere to avoid sign-flip (q and -q) jumps.
        if float(np.dot(pre_quat, tgt_quat)) < 0.0:
            tgt_quat = -tgt_quat
        delta_pos = tgt_pos - pre_pos
        delta_quat = quat_mul(tgt_quat, quat_conj(pre_quat))
        delta_quat = np.asarray(delta_quat, dtype=np.float64)
        delta_quat = delta_quat / max(float(np.linalg.norm(delta_quat)), 1e-12)
        # Use shortest-arc representation for stable axis-angle conversion.
        if float(delta_quat[0]) < 0.0:
            delta_quat = -delta_quat
        delta_rot = _quat2axisangle_wxyz(delta_quat).astype(np.float32)
        arm_cmd = np.concatenate([delta_pos, delta_rot], axis=0)

        gripper_delta = tgt_gripper - pre_gripper
        gripper_cmd = np.array([float(np.mean(gripper_delta))], dtype=np.float32)

        cmd_limit = getattr(base_env, "cmd_limit", 1.0)
        if hasattr(cmd_limit, "detach"):
            cmd_limit = cmd_limit.detach().cpu().numpy()
        cmd_limit = np.asarray(cmd_limit, dtype=np.float32).reshape(-1)
        if cmd_limit.size == 1:
            arm_raw = arm_cmd / float(cmd_limit[0])
        else:
            arm_raw = arm_cmd / cmd_limit[:6]
        gripper_raw = gripper_cmd / 0.01
        action_raw_unclipped = np.concatenate([arm_raw, gripper_raw], axis=0)
        return action_raw_unclipped, arm_cmd, gripper_cmd

    def joint_target_to_mode1_action(self, env, joint_start, joint_target, clamp: bool = True):
        base_env = self._unwrap_env(env)
        arm_dim = int(len(base_env.arm_joint_indices))
        expected_dim = arm_dim + 2
        joint_start = np.asarray(joint_start, dtype=np.float32).reshape(-1)
        joint_target = np.asarray(joint_target, dtype=np.float32).reshape(-1)
        if joint_start.shape[0] != expected_dim or joint_target.shape[0] != expected_dim:
            raise ValueError(
                f"Expected joint_start/joint_target dim={expected_dim}, got "
                f"{joint_start.shape[0]}/{joint_target.shape[0]}"
            )
        action_raw_unclipped, _, _ = self._compute_mode0_semantics_from_joint_transition(
            base_env,
            joint_pre=joint_start,
            joint_target=joint_target,
        )
        if clamp:
            action_raw_unclipped = np.clip(action_raw_unclipped, -1.0, 1.0)
        return np.asarray(action_raw_unclipped, dtype=np.float32)

    def _clamp_mode0_gripper_segment_target(
        self,
        base_env,
        seg_start: np.ndarray,
        seg_goal: np.ndarray,
        max_gripper_delta: float,
    ) -> np.ndarray:
        arm_dim = int(len(base_env.arm_joint_indices))
        seg_start = np.asarray(seg_start, dtype=np.float32)
        seg_goal = np.asarray(seg_goal, dtype=np.float32).copy()
        max_delta = float(max_gripper_delta)

        # Clamp per-segment gripper delta to avoid release-time finger oscillation.
        gripper_delta = seg_goal[arm_dim:] - seg_start[arm_dim:]
        gripper_delta = np.clip(gripper_delta, -max_delta, max_delta)
        seg_goal[arm_dim:] = seg_start[arm_dim:] + gripper_delta

        # Respect simulator joint limits after per-step clamp.
        gripper_lo = base_env.robot_dof_lower_limits[-2:]
        gripper_hi = base_env.robot_dof_upper_limits[-2:]
        if hasattr(gripper_lo, "detach"):
            gripper_lo = gripper_lo.detach().cpu().numpy()
        if hasattr(gripper_hi, "detach"):
            gripper_hi = gripper_hi.detach().cpu().numpy()
        seg_goal[arm_dim:] = np.clip(
            seg_goal[arm_dim:],
            np.asarray(gripper_lo, dtype=np.float32),
            np.asarray(gripper_hi, dtype=np.float32),
        )
        return seg_goal

    def split_motion_planning_action(
        self,
        env,
        action,
        max_action_raw_abs: float = 1.0,
        eps: float = 1e-6,
        max_depth: int = 10,
        max_gripper_delta: float = 0.01,
    ):
        base_env = self._unwrap_env(env)
        joint_start = self._get_current_joint_target_np(base_env)
        joint_goal = self._to_joint_target_np(base_env, action)

        stack = [(joint_start, joint_goal, 0)]
        sub_targets: list[np.ndarray] = []
        while stack:
            seg_start, seg_goal, depth = stack.pop()
            action_raw_unclipped, _, _ = self._compute_mode0_semantics_from_joint_transition(base_env, seg_start, seg_goal)
            arm_max_abs = float(np.max(np.abs(action_raw_unclipped[:6])))
            if arm_max_abs <= float(max_action_raw_abs) + float(eps):
                seg_goal = self._clamp_mode0_gripper_segment_target(
                    base_env,
                    seg_start,
                    seg_goal,
                    max_gripper_delta=max_gripper_delta,
                )
                sub_targets.append(seg_goal)
                continue
            if depth >= int(max_depth):
                gripper_max_abs = float(np.max(np.abs(action_raw_unclipped[6:])))
                raise RuntimeError(
                    "Failed to split motion-planning action within normalized arm threshold; "
                    f"max_depth={max_depth}, arm_max_abs={arm_max_abs:.4f}, gripper_max_abs={gripper_max_abs:.4f}"
                )
            seg_mid = 0.5 * (seg_start + seg_goal)
            stack.append((seg_mid, seg_goal, depth + 1))
            stack.append((seg_start, seg_mid, depth + 1))
        return [t.astype(np.float32, copy=False) for t in sub_targets]
    
    def prepare_snapshot(self, env):
        pass

    def snapshot(self, env):
        base_env = self._unwrap_env(env)
        scene = base_env.scene
        snap = scene.get_state(is_relative=True)
        snap_cpu = {k: {n: {kk: vv.detach().cpu().clone()
                            for kk, vv in d.items()}
                            for n, d in v.items()}
                            for k, v in snap.items()}
        return snap_cpu

    def restore(
        self,
        env,
        snap_cpu,
        env_id: int = 0,
        src_env_id: int = None,
        close_gripper_on_restore: bool = True,
    ):
        """Restore snap_cpu state into env slot env_id.

        snap_cpu tensors have shape (N, ...).  Normally env_id == src_env_id,
        meaning the data at index env_id in the snapshot is written back to
        env slot env_id.

        When snap_cpu was captured from a different slot (e.g. single-env
        snapshots extracted via _extract_slot_snapshot), pass
        src_env_id=0 so the correct source row is used.  The method
        broadcasts the single source row to fill a full (num_envs, ...)
        tensor before calling scene.reset_to with env_ids=[env_id].
        """
        base_env = self._unwrap_env(env)
        scene = base_env.scene
        snap_device = {k: {n: {kk: vv.to(base_env.device)
                            for kk, vv in d.items()}
                            for n, d in v.items()}
                            for k, v in snap_cpu.items()}

        if src_env_id is not None and src_env_id != env_id:
            # Expand: copy src_env_id row into all rows of a full (num_envs, ...)
            # tensor, then let reset_to pick env_id's row.
            num_envs = base_env.num_envs
            snap_device = {k: {n: {kk: vv[src_env_id:src_env_id + 1].expand(num_envs, *vv.shape[1:]).contiguous()
                                for kk, vv in d.items()}
                                for n, d in v.items()}
                                for k, v in snap_device.items()}

        env_ids = torch.tensor([env_id], device=base_env.device, dtype=torch.int32)
        scene.reset_to(snap_device, env_ids=env_ids, is_relative=True)

        joint_pos = base_env.robot.data.joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        base_env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        base_env.robot.set_joint_position_target(joint_pos)
        base_env.robot_dof_targets[:] = joint_pos
        base_env.robot.set_joint_velocity_target(torch.zeros_like(joint_pos))
        base_env.set_control_params(mode=0, is_release=False)

        # prev_obj_flags = self._set_object_body_flags(base_env, self.grasped_objects, disable_gravity=True, kinematic=True)

        # zero_action = torch.zeros((base_env.num_envs, 7), device=base_env.device)
        # for _ in range(20):
        #     self.step(env, zero_action, action_mode=1)

        zero_action = torch.zeros((base_env.num_envs, 7), device=base_env.device)
        if close_gripper_on_restore:
            zero_action[..., -1] = -1.0
        for _ in range(2):
            self.step(env, zero_action, action_mode=1, disable_action_ema=True)

        # self._restore_object_body_flags(base_env, prev_obj_flags)

    def _set_object_body_flags(self, base_env, object_names, disable_gravity=None, kinematic=None):
        """Set per-object rigid body flags and return previous values for restoration."""
        if not object_names:
            return {}
        try:
            from pxr import Sdf
        except Exception as exc:
            raise RuntimeError("USD modules unavailable; cannot toggle object gravity/kinematic.") from exc
        stage = omni.usd.get_context().get_stage() if "omni" in globals() else None
        if stage is None:
            raise RuntimeError("USD stage unavailable; cannot toggle object gravity/kinematic.")

        prev = {}
        for name in object_names:
            prim_path = getattr(base_env, "object_prims", {}).get(name)
            if prim_path is None:
                continue
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                continue
            prev[name] = {}

            def _set_attr(attr_name, value):
                attr = prim.GetAttribute(attr_name)
                if attr and attr.HasAuthoredValueOpinion():
                    prev[name][attr_name] = attr.Get()
                else:
                    prev[name][attr_name] = None
                if value is not None:
                    if not attr:
                        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool)
                    attr.Set(bool(value))

            if disable_gravity is not None:
                _set_attr("physxRigidBody:disableGravity", disable_gravity)
            if kinematic is not None:
                _set_attr("physxRigidBody:kinematicEnabled", kinematic)
        return prev

    def _restore_object_body_flags(self, base_env, prev_flags):
        """Restore object rigid body flags captured by _set_object_body_flags."""
        if not prev_flags:
            return
        try:
            from pxr import Sdf
        except Exception as exc:
            raise RuntimeError("USD modules unavailable; cannot restore object gravity/kinematic.") from exc
        stage = omni.usd.get_context().get_stage() if "omni" in globals() else None
        if stage is None:
            raise RuntimeError("USD stage unavailable; cannot restore object gravity/kinematic.")

        for name, attrs in prev_flags.items():
            prim_path = getattr(base_env, "object_prims", {}).get(name)
            if prim_path is None:
                continue
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                continue
            for attr_name, val in attrs.items():
                attr = prim.GetAttribute(attr_name)
                if val is None:
                    if attr:
                        attr.Clear()
                else:
                    if not attr:
                        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool)
                    attr.Set(bool(val))
    
    
    ###### Data Collection ######
    def _resize_array_for_collection(self, arr: np.ndarray, mode: str) -> np.ndarray:
        target_h = int(self.data_collect_h)
        target_w = int(self.data_collect_w)
        if arr is None:
            return arr
        if target_h <= 0 or target_w <= 0:
            return arr
        if arr.shape[0] == target_h and arr.shape[1] == target_w:
            return arr
        if mode == "rgb":
            pil_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
            return np.asarray(pil_img.resize((target_w, target_h), resample=Image.BILINEAR), dtype=np.uint8)
        if mode == "depth":
            pil_img = Image.fromarray(arr.astype(np.float32), mode="F")
            return np.asarray(pil_img.resize((target_w, target_h), resample=Image.NEAREST), dtype=np.float32)
        if mode == "seg":
            seg32 = arr.astype(np.int32, copy=False)
            pil_img = Image.fromarray(seg32, mode="I")
            return np.asarray(pil_img.resize((target_w, target_h), resample=Image.NEAREST), dtype=seg32.dtype)
        return arr

    def _scale_intrinsic_for_collection(self, ixt: np.ndarray, src_h: int, src_w: int) -> np.ndarray:
        target_h = int(self.data_collect_h)
        target_w = int(self.data_collect_w)
        if target_h <= 0 or target_w <= 0 or (src_h == target_h and src_w == target_w):
            return ixt
        sx = float(target_w) / float(src_w)
        sy = float(target_h) / float(src_h)
        ixt_new = np.array(ixt, dtype=np.float32, copy=True)
        ixt_new[0, 0] *= sx
        ixt_new[1, 1] *= sy
        ixt_new[0, 2] *= sx
        ixt_new[1, 2] *= sy
        return ixt_new

    def _encode_depth_for_collection(self, depth: np.ndarray) -> np.ndarray:
        storage = self.data_collect_depth_storage
        if storage == "float32":
            return depth.astype(np.float32, copy=False)
        if storage == "float16":
            return depth.astype(np.float16, copy=False)
        # Default: uint16 in millimeters; 65535 denotes invalid depth.
        depth_f = depth.astype(np.float32, copy=False)
        invalid = (~np.isfinite(depth_f)) | (depth_f <= 0.0)
        depth_mm = np.rint(depth_f * self.data_collect_depth_scale)
        depth_mm = np.clip(depth_mm, 0.0, 65534.0).astype(np.uint16)
        depth_mm[invalid] = np.uint16(65535)
        return depth_mm

    def _encode_seg_for_collection(self, seg: np.ndarray) -> np.ndarray:
        storage = self.data_collect_seg_storage
        if storage == "int32":
            return seg.astype(np.int32, copy=False)
        if storage == "uint32":
            return seg.astype(np.uint32, copy=False)
        if storage == "uint8":
            return seg.astype(np.uint8, copy=False)
        # Default: uint16 when possible, otherwise uint32 fallback.
        seg64 = seg.astype(np.int64, copy=False)
        if seg64.size == 0:
            return seg64.astype(np.uint16)
        max_id = int(seg64.max())
        if max_id <= np.iinfo(np.uint16).max:
            return seg64.astype(np.uint16, copy=False)
        return seg64.astype(np.uint32, copy=False)

    def _collect_step_sensor_data(self, env):
        # collect rgb, depth, seg, ixt, ext for all cameras specified in cfg and return as a dict
        rgbs_raw = self.get_rgbs(env, self.data_collect_camera_names)
        rgbs = {
            view: self._resize_array_for_collection(rgb, mode="rgb")
            for view, rgb in rgbs_raw.items()
        }
        data = {"rgbs": rgbs}
        if self.collect_depth:
            depths_raw = self.get_depths(env, self.data_collect_camera_names)
            ixts_raw = self.get_camera_intrinsics(env, self.data_collect_camera_names)
            exts = self.get_camera_extrinsics(env, self.data_collect_camera_names)
            depths = {}
            ixts = {}
            for view, depth in depths_raw.items():
                src_h, src_w = int(depth.shape[0]), int(depth.shape[1])
                depth_resized = self._resize_array_for_collection(depth, mode="depth")
                depths[view] = self._encode_depth_for_collection(depth_resized)
                ixts[view] = self._scale_intrinsic_for_collection(ixts_raw[view], src_h=src_h, src_w=src_w)
            data["depths"] = depths
            data["ixts"] = ixts
            data["exts"] = exts
        if self.collect_segmentation:
            segs_raw = self.get_semantic_segmentations(env, self.data_collect_camera_names)
            segs = {}
            for view, seg in segs_raw.items():
                if seg is None:
                    segs[view] = None
                    continue
                seg_resized = self._resize_array_for_collection(seg, mode="seg")
                segs[view] = self._encode_seg_for_collection(seg_resized)
            data["segs"] = segs
            label_mappings = self.get_seg_label_mappings(env, self.data_collect_camera_names)
            data["seg_label_mapping"] = json.dumps(label_mappings)
        # import ipdb; ipdb.set_trace()
        return data

    def _collect_step_state_data(self, env, env_id: int = 0):
        base_env = self._unwrap_env(env)
        eef_pos = base_env.robot.data.body_pos_w[:, base_env.hand_link_idx][env_id].detach().cpu().numpy().astype(np.float32)
        eef_quat = base_env.robot.data.body_quat_w[:, base_env.hand_link_idx][env_id].detach().cpu().numpy().astype(np.float32)
        eef_axis_angle = _quat2axisangle_wxyz(eef_quat).astype(np.float32)
        gripper_pos = base_env.robot.data.joint_pos[env_id, -2:].detach().cpu().numpy().astype(np.float32)
        arm_ids = list(base_env.arm_joint_indices)
        arm_joint_pos = base_env.robot.data.joint_pos[env_id, arm_ids].detach().cpu().numpy().astype(np.float32)
        state = {
            "ee": {
                "pos": eef_pos,
                "quat": eef_quat,
                "axis_angle": eef_axis_angle,
            },
            "arm_joint_pos": arm_joint_pos,
            "gripper_pos": gripper_pos,
        }
        if bool(getattr(self, "collect_object_poses", False)) and hasattr(base_env, "objects"):
            object_poses = {}
            for name in sorted(self._get_active_object_names(base_env)):
                obj = base_env.objects[name]
                pose = obj.data.root_pose_w[env_id].detach().cpu().numpy().astype(np.float32)
                object_poses[name] = {
                    "position": pose[:3],
                    "rotation_wxyz": pose[3:7],
                }
            state["object_poses"] = object_poses
        return state

    @staticmethod
    def _quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(q))
        if n <= 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / n

    @staticmethod
    def _axis_angle_to_quat_wxyz(axis_angle: np.ndarray) -> np.ndarray:
        aa = np.asarray(axis_angle, dtype=np.float64).reshape(3)
        theta = float(np.linalg.norm(aa))
        if theta <= 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        axis = aa / theta
        half = 0.5 * theta
        s = math.sin(half)
        return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)

    def _quat_angle_distance_rad_wxyz(self, q1: np.ndarray, q2: np.ndarray) -> float:
        q1n = self._quat_normalize_wxyz(q1)
        q2n = self._quat_normalize_wxyz(q2)
        dot = float(np.dot(q1n, q2n))
        dot = min(1.0, max(-1.0, abs(dot)))
        return float(2.0 * math.acos(dot))

    def _collect_step_action_data(self, env, action):
        """Collect delta-cmd action fields.

        Only populated when self._data_collect_store_ee_cmd is True, which is set
        by the cuRobo skill when planner_execute_mode1_replan=True.  When the flag
        is False, no delta EE cmd fields are stored for any step (including RL steps).
        """
        if not self._data_collect_store_ee_cmd:
            return {}

        base_env = self._unwrap_env(env)
        action_mode = int(getattr(base_env, "action_mode", 1))
        disable_action_ema = bool(getattr(base_env, "disable_action_ema", False))
        if isinstance(self._last_step_action_ctx, dict):
            action_mode = int(self._last_step_action_ctx.get("action_mode", action_mode))
            disable_action_ema = bool(self._last_step_action_ctx.get("disable_action_ema", disable_action_ema))

        # mode=0 steps within replan mode have no delta EE cmd.
        if action_mode == 0:
            return {}

        # mode=1: reconstruct the EMA-blended delta cmd that was actually executed.
        act = to_tensor(action, base_env.device).to(torch.float32)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        act = act.clamp(-1.0, 1.0)
        alpha = 0.0 if disable_action_ema else getattr(base_env.cfg, "action_ema_alpha", None)
        if alpha is not None and float(alpha) > 0.0:
            prev = getattr(base_env, "prev_actions", None)
            if prev is None:
                raise RuntimeError("Base env missing prev_actions; cannot reconstruct EMA-blended delta cmd.")
            prev = to_tensor(prev, base_env.device).to(torch.float32)
            if prev.ndim == 1:
                prev = prev.unsqueeze(0)
            if prev.shape[0] < act.shape[0]:
                raise RuntimeError(
                    f"prev_actions batch ({prev.shape[0]}) < action batch ({act.shape[0]}), cannot align EMA context."
                )
            prev = prev[: act.shape[0], :]
            # collect_step_data_pre_action is called before env.step(), so
            # base_env.prev_actions is exactly the EMA history used for this step.
            act = float(alpha) * prev + (1.0 - float(alpha)) * act

        cmd_limit = getattr(base_env, "cmd_limit", 1.0)
        arm_cmd = (act[0, :6] * cmd_limit).detach().cpu().numpy().astype(np.float32)
        delta_pos = arm_cmd[:3]
        delta_axis_angle = arm_cmd[3:6]
        delta_quat = self._axis_angle_to_quat_wxyz(delta_axis_angle).astype(np.float32)
        gripper_delta = float((act[0, 6:] * 0.01).detach().cpu().numpy().mean())

        return {
            "ee_cmd": {
                "delta_pos": delta_pos,
                "delta_quat": delta_quat,
                "delta_axis_angle": delta_axis_angle,
            },
            "gripper_cmd_delta": np.float32(gripper_delta),
        }

    def collect_step_data_post_action(self, env, pre_step_data=None):
        """Collect post-action fields: EE targets (FK), EE actual, arm joint targets/actual, gripper targets/actual."""
        base_env = self._unwrap_env(env)
        env_id = 0
        arm_ids = list(base_env.arm_joint_indices)
        arm_dim = len(arm_ids)

        # joint targets sent to actuator this step (set by _pre_physics_step)
        jt = base_env.robot_dof_targets[env_id].detach().cpu().numpy().astype(np.float32)
        arm_joint_targets = jt[:arm_dim]          # (arm_dim,)
        gripper_targets = jt[-2:].copy()          # (2,) raw per-finger targets

        # post-physics actual joint positions
        ja_arm = base_env.robot.data.joint_pos[env_id, arm_ids].detach().cpu().numpy().astype(np.float32)
        gripper_actual = base_env.robot.data.joint_pos[env_id, -2:].detach().cpu().numpy().astype(np.float32)  # (2,)

        # post-physics actual EE pose
        ee_actual_pos = base_env.robot.data.body_pos_w[env_id, base_env.hand_link_idx].detach().cpu().numpy().astype(np.float32)
        ee_actual_quat = base_env.robot.data.body_quat_w[env_id, base_env.hand_link_idx].detach().cpu().numpy().astype(np.float32)
        ee_actual_axis_angle = _quat2axisangle_wxyz(ee_actual_quat).astype(np.float32)

        # EE targets via FK(arm_joint_targets) — raises on failure
        ee_tgt_pos, ee_tgt_quat = self._fk_ee_pose_from_joint(arm_joint_targets)
        ee_tgt_pos = ee_tgt_pos.astype(np.float32)
        ee_tgt_quat = ee_tgt_quat.astype(np.float32)
        ee_tgt_axis_angle = _quat2axisangle_wxyz(ee_tgt_quat).astype(np.float32)

        return {
            "action": {
                "ee_targets": {
                    "pos": ee_tgt_pos,
                    "quat": ee_tgt_quat,
                    "axis_angle": ee_tgt_axis_angle,
                },
                "ee_actual": {
                    "pos": ee_actual_pos,
                    "quat": ee_actual_quat,
                    "axis_angle": ee_actual_axis_angle,
                },
                "arm_joint_targets": arm_joint_targets,
                "arm_joint_actual": ja_arm,
                "gripper_targets": gripper_targets,
                "gripper_actual": gripper_actual,
            }
        }

    def collect_step_data_pre_action(self, env, action, action_mode: int = 1, disable_action_ema: bool = False):
        """Collect (s_t, a_t) semantics before applying the action."""
        base_env = self._unwrap_env(env)
        self._cache_step_action_context(
            base_env,
            action,
            action_mode,
            disable_action_ema=bool(disable_action_ema),
        )
        sensor_data = self._collect_step_sensor_data(env)
        state_data = self._collect_step_state_data(env)
        action_data = self._collect_step_action_data(env, action)
        return {
            "sensor": sensor_data,
            "state": state_data,
            "action": action_data,
        }
    #############################
