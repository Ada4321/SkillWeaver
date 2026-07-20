from __future__ import annotations

import copy
import math
import os
import json
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.semantics as sem_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, TiledCamera

from simulators.base_env_cfg import BaseEnvCfg
from simulators.robot_conventions import resolve_robot_names
from simulators.part_meta import (
    AXIS_ANNOTATION_BY_KIND,
    OPENING_DIR_CANO_BY_KIND,
    OPENING_DIR_CANO_DEFAULT,
    parse_urdf_parts,
    part_pcd_npz_path,
)
from simulators.tasks import PickTask, PlaceTask, PoseTask, OpenDrawerTask, CloseDrawerTask, CloseDoorTask
from utils.geometry_utils import to_tensor
from utils.random_control import timestamp_seed_scope


_HANDLE_SIDE_TO_SURFACE = {
    "+x": "front",
    "-x": "back",
    "+y": "right",
    "-y": "left",
    "+z": "top",
    "-z": "bottom",
}

_HANDLE_SIDE_ALIASES = {
    "+x": "+x",
    "x+": "+x",
    "-x": "-x",
    "x-": "-x",
    "+y": "+y",
    "y+": "+y",
    "-y": "-y",
    "y-": "-y",
    "+z": "+z",
    "z+": "+z",
    "-z": "-z",
    "z-": "-z",
}


def _category_candidates(name: str, category: object = None) -> list[str]:
    candidates: list[str] = []
    for value in (category, name):
        if not isinstance(value, str) or not value:
            continue
        candidates.append(value)
        base, sep, suffix = value.rpartition("_")
        if sep and suffix.isdigit():
            candidates.append(base)
    return list(dict.fromkeys(candidates))


class BaseEnv(DirectRLEnv):
    cfg: BaseEnvCfg

    @staticmethod
    def _normalize_handle_side(value: object) -> str:
        text = str(value).strip().lower().replace(" ", "")
        normalized = _HANDLE_SIDE_ALIASES.get(text)
        if normalized is None:
            raise ValueError(
                f"Invalid handle_side {value!r}. Expected one of {sorted(_HANDLE_SIDE_TO_SURFACE)}."
            )
        return normalized

    @classmethod
    def _resolve_handle_point_cloud_path(cls, asset_dir: str, handle_side: object) -> str:
        normalized = cls._normalize_handle_side(handle_side)
        side_name = _HANDLE_SIDE_TO_SURFACE[normalized]
        matches = []
        for filename in sorted(os.listdir(asset_dir)):
            lower = filename.lower()
            if lower.endswith("_r25_pcd.npz") and f"_{side_name}_" in lower:
                path = os.path.join(asset_dir, filename)
                if os.path.isfile(path):
                    matches.append(path)
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one handle-side point cloud in '{asset_dir}' for "
                f"handle_side={normalized!r} -> side={side_name!r}, got {len(matches)}: {matches}"
            )
        return matches[0]

    def _load_half_pcd_categories(self) -> set[str]:
        """Load the category whitelist for the half-pcd obs mode.

        Returns an empty set if cfg.use_half_pcd is False (so all downstream
        half-pcd code paths become no-ops and the env behaves identically to
        the pre-feature codebase).
        """
        if not getattr(self.cfg, "use_half_pcd", False):
            return set()
        path = getattr(self.cfg, "half_pcd_list", "") or ""
        if not path:
            raise ValueError(
                "cfg.use_half_pcd=True requires cfg.half_pcd_list to point to a "
                "category whitelist file (e.g. data/assets_for_agent/half_pcd.txt)."
            )
        expanded = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isabs(expanded):
            raise ValueError(
                f"cfg.half_pcd_list must be an absolute path, got: {path!r}"
            )
        if not os.path.isfile(expanded):
            raise FileNotFoundError(
                f"cfg.half_pcd_list file does not exist: {expanded}"
            )
        categories: set[str] = set()
        with open(expanded, "r", encoding="utf-8") as f:
            for line in f:
                item = line.split("#", 1)[0].strip()
                if item:
                    categories.add(item)
        return categories

    @staticmethod
    def _quat_mul_wxyz(
        q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Hamilton product for quaternions in (w, x, y, z) order."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def __init__(self, cfg: BaseEnvCfg, scene_desc_file: str, render_mode: str | None = None, **kwargs):
        # set up distributed loading
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        with open(scene_desc_file, 'r') as f:
            self.scene_desc = json.load(f)
        self._scene_asset_root_path = str(getattr(cfg, "scene_asset_root_path", "/data/group_data/katefgroup-ssd/sim_scene_gen"))

        table_desc = self.scene_desc.get("table")
        self._scene_has_table_desc = isinstance(table_desc, dict)
        self._scene_table_usd_path: str | None = None

        virtual_table_size_cfg = getattr(cfg, "no_table_virtual_table_size", (1.0, 1.0, 0.001))
        if not isinstance(virtual_table_size_cfg, (list, tuple)) or len(virtual_table_size_cfg) != 3:
            raise ValueError(
                "no_table_virtual_table_size must be a 3-element list/tuple [sx, sy, sz], "
                f"got {virtual_table_size_cfg}."
            )
        virtual_table_size = (
            float(virtual_table_size_cfg[0]),
            float(virtual_table_size_cfg[1]),
            float(virtual_table_size_cfg[2]),
        )
        if any(v <= 0.0 for v in virtual_table_size):
            raise ValueError(
                f"no_table_virtual_table_size must be positive, got {virtual_table_size}."
            )

        if self._scene_has_table_desc:
            scene_table_usd = table_desc.get("final_usd_path")
            if not scene_table_usd:
                raise ValueError("scene_desc['table']['final_usd_path'] is required when table is provided.")
            self._scene_table_usd_path = self._resolve_scene_asset_path(scene_table_usd)
            scene_table_size = table_desc.get("size")
            if not isinstance(scene_table_size, (list, tuple)) or len(scene_table_size) != 3:
                raise ValueError(
                    "scene_desc['table']['size'] is required and must be a 3-element list/tuple [sx, sy, sz]."
                )
            self._scene_table_size = (
                float(scene_table_size[0]),
                float(scene_table_size[1]),
                float(scene_table_size[2]),
            )
            if any(v <= 0.0 for v in self._scene_table_size):
                raise ValueError(
                    f"scene_desc['table']['size'] must be positive, got {self._scene_table_size}."
                )
        else:
            self._scene_table_size = virtual_table_size

        table_init_pos_cfg = getattr(cfg, "table_init_pos", (0.5, 0.0, 0.05))
        if not isinstance(table_init_pos_cfg, (list, tuple)) or len(table_init_pos_cfg) != 3:
            raise ValueError(
                f"table_init_pos must be a 3-element list/tuple [x, y, z], got {table_init_pos_cfg}."
            )
        table_init_pos = tuple(float(x) for x in table_init_pos_cfg)
        self._table_xy = (table_init_pos[0], table_init_pos[1])
        # Object placement anchor: decoupled from the table prim when
        # cfg.object_xy_anchor is set (simpler uses (0,0) -> objects at true base
        # xy). None -> legacy table-coupled anchor (libero/real_world).
        _obj_anchor = getattr(cfg, "object_xy_anchor", None)
        self._object_xy_anchor = (
            (float(_obj_anchor[0]), float(_obj_anchor[1])) if _obj_anchor is not None else self._table_xy
        )
        table_init_rot_cfg = getattr(cfg, "table_init_rot", (1.0, 0.0, 0.0, 0.0))
        if not isinstance(table_init_rot_cfg, (list, tuple)) or len(table_init_rot_cfg) != 4:
            raise ValueError(
                f"table_init_rot must be a 4-element list/tuple [w, x, y, z], got {table_init_rot_cfg}."
            )
        self._table_rot = tuple(float(x) for x in table_init_rot_cfg)
        max_surface_height = getattr(cfg, "max_table_surface_height", None)
        self._max_table_surface_height = None if max_surface_height is None else float(max_surface_height)
        min_surface_height = getattr(cfg, "min_table_surface_height", None)
        self._min_table_surface_height = None if min_surface_height is None else float(min_surface_height)
        self._table_spawn_pos = (
            table_init_pos[0],
            table_init_pos[1],
            self._clip_table_center_z(table_init_pos[2]),
        )
        z_min, z_max = getattr(cfg, "table_z_range", (table_init_pos[2], table_init_pos[2]))
        self._table_z_range = (float(z_min), float(z_max))
        if self._table_z_range[0] > self._table_z_range[1]:
            raise ValueError(f"Invalid table_z_range: {self._table_z_range}")
        self._table_disable_prob = float(getattr(cfg, "table_disable_prob", 0.0))
        if not (0.0 <= self._table_disable_prob <= 1.0):
            raise ValueError(f"table_disable_prob must be in [0, 1], got {self._table_disable_prob}")
        self._table_enabled_this_episode = True

        # Optional floor/walls from scene_desc.
        self._scene_floor_usd_path: str | None = None
        self._scene_wall_usd_paths: list[str] = []
        floor_desc = self.scene_desc.get("floor")
        if isinstance(floor_desc, dict):
            scene_floor_usd = floor_desc.get("final_usd_path")
            if scene_floor_usd:
                self._scene_floor_usd_path = self._resolve_scene_asset_path(scene_floor_usd)
        walls_desc = self.scene_desc.get("walls")
        if isinstance(walls_desc, dict):
            wall_paths = walls_desc.get("final_usd_paths", [])
            if isinstance(wall_paths, list):
                for wall_usd in wall_paths:
                    if not wall_usd:
                        continue
                    self._scene_wall_usd_paths.append(self._resolve_scene_asset_path(wall_usd))
        self._enable_scene_visual_floor = bool(getattr(cfg, "enable_scene_visual_floor", True))
        self._enable_scene_visual_walls = bool(getattr(cfg, "enable_scene_visual_walls", True))
        self._scene_floor_visual_enabled = bool(self._scene_floor_usd_path) and self._enable_scene_visual_floor
        self._scene_walls_visual_enabled = bool(self._scene_wall_usd_paths) and self._enable_scene_visual_walls
        self._scene_floor_walls_enabled = bool(self._scene_floor_usd_path and self._scene_wall_usd_paths)

        self._floor_xy = tuple(float(x) for x in getattr(cfg, "floor_xy", (0.0, 0.0)))
        floor_z_min, floor_z_max = getattr(cfg, "floor_z_range", (-0.45, -0.45))
        self._floor_z_range = (float(floor_z_min), float(floor_z_max))
        if self._floor_z_range[0] > self._floor_z_range[1]:
            raise ValueError(f"Invalid floor_z_range: {self._floor_z_range}")
        self._ground_plane_z_offset = float(getattr(cfg, "ground_plane_z_offset_below_floor", 0.02))
        self._ground_plane_z = self._floor_z_range[0] - self._ground_plane_z_offset
        self._hide_moveable_ground_visual = bool(getattr(cfg, "hide_moveable_ground_visual", True))
        self._no_table_floor_above_ground = float(getattr(cfg, "no_table_floor_above_ground", 0.001))
        self._debug_floor_material_binding = bool(getattr(cfg, "debug_floor_material_binding", False))
        if self._no_table_floor_above_ground < 0.0:
            raise ValueError(
                f"no_table_floor_above_ground must be >= 0, got {self._no_table_floor_above_ground}"
            )
        self._current_floor_surface_z = float(self._floor_z_range[0])

        self._wall_height = float(getattr(cfg, "wall_height", 4.0))
        _ws = getattr(cfg, "wall_scale", None)
        self._wall_scale = (
            (float(_ws[0]), float(_ws[1]), float(_ws[2])) if _ws is not None else None
        )
        self._wall_slots = ("front", "back", "left", "right")
        self._wall_prim_ids = ("wall_1", "wall_2", "wall_3", "wall_4")
        if len(self._wall_slots) != len(self._wall_prim_ids):
            raise RuntimeError("Wall slot count and wall prim count must match.")
        self._wall_slot_xy: dict[str, tuple[float, float]] = {
            "front": tuple(float(x) for x in getattr(cfg, "wall_front_xy", (2.0, 0.0))),
            "back": tuple(float(x) for x in getattr(cfg, "wall_back_xy", (-2.0, 0.0))),
            "left": tuple(float(x) for x in getattr(cfg, "wall_left_xy", (0.0, 2.0))),
            "right": tuple(float(x) for x in getattr(cfg, "wall_right_xy", (0.0, -2.0))),
        }

        floor_bias = tuple(float(x) for x in getattr(cfg, "floor_rot_bias", (0.70710678, 0.0, 0.0, 0.70710678)))
        wall_front_bias = tuple(float(x) for x in getattr(cfg, "wall_front_rot_bias", (0.70710678, 0.0, 0.0, -0.70710678)))
        wall_back_bias = tuple(float(x) for x in getattr(cfg, "wall_back_rot_bias", (0.70710678, 0.0, 0.0, 0.70710678)))
        wall_left_bias = tuple(float(x) for x in getattr(cfg, "wall_left_rot_bias", (1.0, 0.0, 0.0, 0.0)))
        wall_right_bias = tuple(float(x) for x in getattr(cfg, "wall_right_rot_bias", (0.0, 0.0, 0.0, 1.0)))
        self._floor_rot = self._quat_mul_wxyz(tuple(float(x) for x in getattr(cfg, "floor_rot", (1.0, 0.0, 0.0, 0.0))), floor_bias)
        self._wall_slot_rot: dict[str, tuple[float, float, float, float]] = {
            "front": self._quat_mul_wxyz(
                tuple(float(x) for x in getattr(cfg, "wall_front_rot", (0.70710678, 0.0, -0.70710678, 0.0))),
                wall_front_bias,
            ),
            "back": self._quat_mul_wxyz(
                tuple(float(x) for x in getattr(cfg, "wall_back_rot", (0.70710678, 0.0, 0.70710678, 0.0))),
                wall_back_bias,
            ),
            "left": self._quat_mul_wxyz(
                tuple(float(x) for x in getattr(cfg, "wall_left_rot", (0.70710678, 0.70710678, 0.0, 0.0))),
                wall_left_bias,
            ),
            "right": self._quat_mul_wxyz(
                tuple(float(x) for x in getattr(cfg, "wall_right_rot", (0.70710678, -0.70710678, 0.0, 0.0))),
                wall_right_bias,
            ),
        }
        self.floor: RigidObject | None = None
        self._moveable_ground: RigidObject | None = None
        self.wall_1: RigidObject | None = None
        self.wall_2: RigidObject | None = None
        self.wall_3: RigidObject | None = None
        self.wall_4: RigidObject | None = None
        self._walls: dict[str, RigidObject] = {}
        self._floor_prim_path_env0 = "/World/envs/env_0/floor"
        self._wall_prim_paths_env0: dict[str, str] = {wall_id: f"/World/envs/env_0/{wall_id}" for wall_id in self._wall_prim_ids}
        
        # Background-image overlay mode (SimplerEnv rgb_overlay style): when
        # cfg.enable_background_overlay is on and the scene json sets
        # `background_image`, the front-camera RGB is composited over that static
        # image (foreground = pixels with finite depth; see simulators/isaaclab.py
        # get_rgb). Flag off (default) -> None -> everything unchanged.
        # `foreground_labels` is carried along for tooling/debug; the depth-based
        # composite itself does not need semantic labels.
        self._background_image_path: str | None = None
        self._foreground_labels: list[str] | None = None
        _bg_overlay_enabled = bool(getattr(cfg, "enable_background_overlay", False))
        _bg_image = self.scene_desc.get("background_image")
        _bg_image_valid = isinstance(_bg_image, str) and bool(_bg_image.strip())
        if _bg_overlay_enabled and not _bg_image_valid:
            raise ValueError(
                "enable_background_overlay=true but scene_desc has no 'background_image'."
            )
        if _bg_image_valid and not _bg_overlay_enabled:
            print("[overlay] scene background_image ignored (enable_background_overlay=false)")
        if _bg_overlay_enabled and _bg_image_valid:
            self._background_image_path = self._resolve_scene_asset_path(_bg_image.strip())
            if not os.path.isfile(self._background_image_path):
                raise FileNotFoundError(
                    f"scene_desc['background_image'] not found: {self._background_image_path}"
                )
            _fg = self.scene_desc.get("foreground_labels")
            if isinstance(_fg, list) and _fg:
                self._foreground_labels = [str(x) for x in _fg]
            print(f"[overlay] background_image={self._background_image_path}")

        # Dome light augmentation
        self._scene_lighting_texture_path: str | None = None
        lighting_texture = self.scene_desc.get("lighting_texture")
        if isinstance(lighting_texture, str):
            lighting_texture = lighting_texture.strip()
            if lighting_texture:
                resolved_lighting_texture = self._resolve_scene_asset_path(lighting_texture)
                if resolved_lighting_texture.lower().endswith(".exr"):
                    self._scene_lighting_texture_path = resolved_lighting_texture

        self._dome_light_prim_path = "/World/Light"
        self._enable_dome_light_augmentation = bool(getattr(cfg, "enable_dome_light_augmentation", True))
        intensity_min, intensity_max = getattr(cfg, "dome_light_intensity_range", (800.0, 2000.0))
        self._dome_light_intensity_range = (float(intensity_min), float(intensity_max))
        yaw_min, yaw_max = getattr(cfg, "dome_light_yaw_range", (-np.pi, np.pi))
        self._dome_light_yaw_range = (float(yaw_min), float(yaw_max))
        self._dome_light_texture_path = self._scene_lighting_texture_path
        self._dome_light_intensity_aug_prob = float(getattr(cfg, "dome_light_intensity_aug_prob", 1.0))
        self._dome_light_yaw_aug_prob = float(getattr(cfg, "dome_light_yaw_aug_prob", 1.0))
        self._dome_light_texture_aug_prob = float(getattr(cfg, "dome_light_texture_aug_prob", 1.0))
        self._dome_light_color_aug_prob = float(getattr(cfg, "dome_light_color_aug_prob", 0.0))
        self._dome_light_color_r_range = tuple(getattr(cfg, "dome_light_color_r_range", (0.5, 1.5)))
        self._dome_light_color_g_range = tuple(getattr(cfg, "dome_light_color_g_range", (0.5, 1.5)))
        self._dome_light_color_b_range = tuple(getattr(cfg, "dome_light_color_b_range", (0.5, 1.5)))

        # Front-camera pose augmentation (spherical coordinates in degrees).
        self._enable_front_camera_pose_augmentation = bool(getattr(cfg, "enable_front_camera_pose_augmentation", True))
        self._front_camera_aug_center = tuple(float(x) for x in getattr(cfg, "front_camera_aug_center", (0.5, 0.0, 0.05)))
        self._front_camera_aug_radius_ranges = self._parse_range_segments(
            getattr(cfg, "front_camera_aug_radius_ranges", "1.0:1.0"),
            field_name="front_camera_aug_radius_ranges",
            min_value=1e-6,
            is_angle=False,
        )
        self._front_camera_aug_azimuth_ranges = self._parse_range_segments(
            getattr(cfg, "front_camera_aug_azimuth_ranges_deg", "0.0:0.0"),
            field_name="front_camera_aug_azimuth_ranges_deg",
            min_value=None,
            is_angle=True,
        )
        self._front_camera_aug_elevation_ranges = self._parse_range_segments(
            getattr(cfg, "front_camera_aug_elevation_ranges_deg", "36.86989765:36.86989765"),
            field_name="front_camera_aug_elevation_ranges_deg",
            min_value=-89.9,
            max_value=89.9,
            is_angle=True,
        )
        self._front_camera_aug_sampling_mode = str(
            getattr(cfg, "front_camera_aug_sampling_mode", "uniform")
        ).strip().lower()
        if self._front_camera_aug_sampling_mode not in ("uniform", "triangular"):
            raise ValueError(
                "front_camera_aug_sampling_mode must be one of {'uniform', 'triangular'}, "
                f"got {self._front_camera_aug_sampling_mode!r}"
            )
        self._front_camera_aug_radius_mode = self._parse_optional_scalar(
            getattr(cfg, "front_camera_aug_radius_mode", None),
            field_name="front_camera_aug_radius_mode",
            min_value=1e-6,
            is_angle=False,
        )
        self._front_camera_aug_azimuth_mode = self._parse_optional_scalar(
            getattr(cfg, "front_camera_aug_azimuth_mode_deg", None),
            field_name="front_camera_aug_azimuth_mode_deg",
            min_value=None,
            is_angle=True,
        )
        self._front_camera_aug_elevation_mode = self._parse_optional_scalar(
            getattr(cfg, "front_camera_aug_elevation_mode_deg", None),
            field_name="front_camera_aug_elevation_mode_deg",
            min_value=-89.9,
            max_value=89.9,
            is_angle=True,
        )
        self._front_camera_aug_roll_deg = float(getattr(cfg, "front_camera_aug_roll_deg", 0.0))
        # Lookat-Z augmentation (None/empty -> legacy table-surface lock).
        lookat_z_spec = getattr(cfg, "front_camera_aug_lookat_z_ranges", None)
        self._front_camera_aug_lookat_z_ranges = (
            self._parse_range_segments(
                lookat_z_spec,
                field_name="front_camera_aug_lookat_z_ranges",
                min_value=None,
                is_angle=False,
            )
            if lookat_z_spec
            else None
        )
        self._front_camera_aug_lookat_z_mode = self._parse_optional_scalar(
            getattr(cfg, "front_camera_aug_lookat_z_mode", None),
            field_name="front_camera_aug_lookat_z_mode",
            min_value=None,
            is_angle=False,
        )
        # Wrist-camera mount pose perturbation (sim-to-real DR; see BaseEnvCfg).
        self._enable_wrist_camera_pose_augmentation = bool(
            getattr(cfg, "enable_wrist_camera_pose_augmentation", False)
        )
        _default_wrist_combos = (
            (0.000, 0.0), (0.010, 7.0), (0.015, 10.0),
            (0.020, 14.0), (0.025, 17.0), (0.030, 18.0),
        )
        _combos_raw = getattr(cfg, "wrist_camera_aug_combos", _default_wrist_combos)
        self._wrist_camera_aug_combos = tuple(
            (float(c[0]), float(c[1])) for c in _combos_raw
        )
        if self._enable_wrist_camera_pose_augmentation and not self._wrist_camera_aug_combos:
            raise ValueError(
                "wrist_camera_aug_combos must be non-empty when "
                "enable_wrist_camera_pose_augmentation is true"
            )
        # Session-level wrist-camera DR: pick ONE combo at env construction
        # (= once per scene in the typical MCTS shell loop) and bake it into
        # cfg.wrist_camera.offset BEFORE super().__init__ builds the camera.
        # Avoids the runtime prim-write bug; DR comes from the shell iterating
        # over many scenes, each getting an independent uniform draw.
        # Per-episode randomization is intentionally NOT done.
        self._wrist_camera_session_combo_idx: int | None = None
        if self._enable_wrist_camera_pose_augmentation:
            with timestamp_seed_scope():
                idx = int(np.random.randint(len(self._wrist_camera_aug_combos)))
            back_m, pitch_deg = self._wrist_camera_aug_combos[idx]
            half = math.radians(float(pitch_deg)) / 2.0
            cfg.wrist_camera.offset.pos = (0.0, float(back_m), 0.0)
            cfg.wrist_camera.offset.rot = (
                math.cos(half), math.sin(half), 0.0, 0.0,
            )
            self._wrist_camera_session_combo_idx = idx
            print(
                f"[wrist-aug] session combo idx={idx} "
                f"(back_m={back_m}, pitch_deg={pitch_deg})",
                flush=True,
            )
        # Per-reset gripper opening randomization (sim-to-real DR; see BaseEnvCfg).
        self._enable_gripper_init_randomization = bool(
            getattr(cfg, "enable_gripper_init_randomization", True)
        )
        self._front_camera_count = int(getattr(cfg, "front_camera_count", 1))
        if self._front_camera_count < 1:
            raise ValueError(f"front_camera_count must be >= 1, got {self._front_camera_count}")
        self._front_camera_view_names = ["front"] + [
            f"front_{idx}" for idx in range(1, self._front_camera_count)
        ]
        self._front_camera_prim_names = {
            view_name: ("FrontCamera" if idx == 0 else f"FrontCamera{idx}")
            for idx, view_name in enumerate(self._front_camera_view_names)
        }
        self._front_camera_prim_path_prefix = "/World/envs/env_"
        self._camera_handles_by_view: dict[str, TiledCamera] = {}

        # Optional layout bank loaded from scene_desc["layouts"].
        self._layout_enabled = False
        self._layout_bank: dict[str, np.ndarray] = {}
        self._layout_min_len = 0
        self._current_layout_idx: int | None = None
        # External pin for layout index. When not None, _sample_layout_idx
        # returns this value verbatim and skips train/test split filtering.
        self.force_layout_idx: int | None = None
        self._layout_sample_candidates: np.ndarray | None = None
        self._layout_table_z_bank: np.ndarray | None = None
        self._layout_cam_poses_bank: np.ndarray | None = None
        self._layout_cam_aperture_bank: np.ndarray | None = None

        raw_split = getattr(cfg, "split", "train")
        self._layout_split = str(raw_split).strip().lower()
        if self._layout_split not in ("train", "test"):
            raise ValueError(f"env.split must be 'train' or 'test', got {raw_split!r}")

        raw_test_layout_ids = getattr(cfg, "test_layout_id", None)
        if raw_test_layout_ids is None:
            self._test_layout_ids = []
        elif isinstance(raw_test_layout_ids, (list, tuple, np.ndarray)):
            self._test_layout_ids = [int(x) for x in raw_test_layout_ids]
        else:
            raise TypeError(
                f"env.test_layout_id must be list[int] or null, got {type(raw_test_layout_ids).__name__}"
            )
        if any(idx < 0 for idx in self._test_layout_ids):
            raise ValueError(f"env.test_layout_id contains negative id(s): {self._test_layout_ids}")
        self._test_layout_ids = sorted(set(self._test_layout_ids))

        raw_layout_weights = getattr(cfg, "layout_sample_weights", None)
        if raw_layout_weights is None:
            self._layout_chunk_weights = [4.0, 3.0, 2.0, 1.0]
        else:
            if not isinstance(raw_layout_weights, (list, tuple, np.ndarray)) or len(raw_layout_weights) != 4:
                raise ValueError(
                    f"env.layout_sample_weights must be a length-4 sequence, got {raw_layout_weights!r}"
                )
            weights = [float(w) for w in raw_layout_weights]
            if any(w < 0 for w in weights):
                raise ValueError(f"env.layout_sample_weights must be all >= 0, got {weights}")
            if sum(weights) <= 0.0:
                raise ValueError(f"env.layout_sample_weights must have sum > 0, got {weights}")
            self._layout_chunk_weights = weights

        self._init_layout_bank()

        # Optionally drop semantic segmentation to avoid Replicator graph issues.
        if not getattr(cfg, "collect_segmentation", False):
            for cam_cfg in (
                getattr(cfg, "front_camera", None),
                getattr(cfg, "top_camera", None),
                getattr(cfg, "wrist_camera", None),
            ):
                if cam_cfg is None or not hasattr(cam_cfg, "data_types"):
                    continue
                cam_cfg.data_types = [dt for dt in cam_cfg.data_types if dt != "semantic_segmentation"]

        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # robot dof limits and targets
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # joint indices (per-robot names from robot_conventions.ROBOT_NAMES;
        # panda row resolves to the exact names previously hard-coded here)
        _rn = resolve_robot_names(getattr(self.cfg, "robot_name", "panda"))
        self.arm_joint_indices, _ = self.robot.find_joints(_rn.arm_joints)
        self.left_joint_idx = self.robot.find_joints(_rn.finger_joints[0])[0][0]
        self.right_joint_idx = self.robot.find_joints(_rn.finger_joints[1])[0][0]
        # use joint limits to scale normalized gripper commands
        gripper_upper_limits = self.robot_dof_upper_limits[[self.left_joint_idx, self.right_joint_idx]]
        self.max_gripper_opening = float(torch.max(gripper_upper_limits))
        self._init_arm_pos_files = list(getattr(self.cfg, "init_arm_pos_files", []) or [])
        self._arm_init_pos_banks: list[torch.Tensor] = []
        self._arm_init_pos_bank_sizes: list[int] = []
        self._load_init_arm_pos_banks()

        # link indices
        self.hand_link_idx = self.robot.find_bodies(_rn.hand_link)[0][0]
        self.left_finger_idx = self.robot.find_bodies(_rn.finger_links[0])[0][0]
        self.right_finger_idx = self.robot.find_bodies(_rn.finger_links[1])[0][0]
        self.left_tip_idx = self.robot.find_bodies(_rn.finger_tips[0])[0][0]
        self.right_tip_idx = self.robot.find_bodies(_rn.finger_tips[1])[0][0]
        self.grasp_site_idx = self.robot.find_bodies(_rn.grasp_site)[0][0]

        # Total number of rigid bodies (links)
        num_bodies = self.robot.data.body_pos_w.shape[1]
        all_body_indices = list(range(num_bodies))

        # Fingers and palm are special
        self.left_finger_link_indices = [self.left_finger_idx]
        self.right_finger_link_indices = [self.right_finger_idx]
        self.palm_link_indices = [self.hand_link_idx]

        special = set(self.left_finger_link_indices + self.right_finger_link_indices + self.palm_link_indices)

        # Everything else is considered "arm"
        self.arm_link_indices = [i for i in all_body_indices if i not in special]

        # NOTE: we need to re-order this
        self.keypoint_indices = [
            *[self.robot.find_bodies(name)[0][0] for name in self.contact.body_names],
            self.grasp_site_idx,
            self.hand_link_idx,
        ]

        # IK controller scale
        self.cmd_limit = torch.tensor([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=torch.float, device=self.device)

        # unit
        self.unit_z = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device)
        # Action smoothing (EMA).
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        self.object = None
        self.current_object_name = None
        self.data_point_cloud = None
        self.data_top_point_cloud = None
        self.data_handle_point_cloud = None
        self.data_half_point_cloud = None
        self.data_point_cloud_obs_reward = None

        # action mode
        self.action_mode = 1  # default: delta EE pose
        self.is_release_action = False
        self.disable_action_ema = False

        # init tasks
        self.tasks = dict()
        self.observasion_spaces = dict()
        # The cfg obs-space constants are calibrated for the 9-DOF Franka
        # (7 arm + 2 fingers); joint_pos/joint_vel/joint_pos_target each
        # contribute num_joints terms, so shift by 3 per DOF of difference
        # (WidowX 8 DOF -> -3, e.g. pick 80 -> 77).
        _obs_dof_delta = 3 * (int(self.robot.num_joints) - 9)
        if self.cfg.enable_pick:
            self.tasks["pick"] = PickTask(self.cfg.scene.num_envs, self.cfg, self.device)
            self.observasion_spaces["pick"] = self.cfg.observation_space_pick + _obs_dof_delta
        if self.cfg.enable_place:
            self.tasks["place"] = PlaceTask(self.cfg.scene.num_envs, self.cfg, self.device)
            self.observasion_spaces["place"] = self.cfg.observation_space_place + _obs_dof_delta
        if self.cfg.enable_pose:
            self.tasks["pose"] = PoseTask(self.cfg.scene.num_envs, self.cfg, self.device)
            self.observasion_spaces["pose"] = self.cfg.observation_space_pose + _obs_dof_delta
        if self.cfg.enable_open_drawer:
            self.tasks["open_drawer"] = OpenDrawerTask(self.cfg.scene.num_envs, self.cfg, self.device)
            self.observasion_spaces["open_drawer"] = self.cfg.observation_space_open_drawer + _obs_dof_delta
        if self.cfg.enable_close_drawer:
            self.tasks["close_drawer"] = CloseDrawerTask(self.cfg.scene.num_envs, self.cfg, self.device)
            self.observasion_spaces["close_drawer"] = self.cfg.observation_space_close_drawer + _obs_dof_delta
        if self.cfg.enable_close_door:
            self.tasks["close_door"] = CloseDoorTask(self.cfg.scene.num_envs, self.cfg, self.device)
            self.observasion_spaces["close_door"] = self.cfg.observation_space_close_door + _obs_dof_delta

        self.env_reset_flag = False
        self.task_name = None
        self.task = None
        self._next_env_reset_params: dict | None = None
        self._use_next_env_reset_params: bool = False
        self._consume_next_env_reset_params: bool = True

    @staticmethod
    def _to_float_list(x) -> list[float]:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        return [float(v) for v in arr.tolist()]

    def set_next_env_reset_params(
        self,
        params: dict | None,
        *,
        enabled: bool = True,
        consume_once: bool = True,
    ) -> None:
        self._next_env_reset_params = params
        self._use_next_env_reset_params = bool(enabled and params is not None)
        self._consume_next_env_reset_params = bool(consume_once)

    def _consume_env_reset_params_if_needed(self) -> None:
        if self._consume_next_env_reset_params and self._use_next_env_reset_params:
            self._next_env_reset_params = None
            self._use_next_env_reset_params = False

    @staticmethod
    def _pose_world_to_local(root_pose_world: np.ndarray, env_origin_world: np.ndarray) -> np.ndarray:
        pose = np.asarray(root_pose_world, dtype=np.float32).copy()
        pose[:3] = pose[:3] - np.asarray(env_origin_world, dtype=np.float32)
        return pose

    @staticmethod
    def _pose_local_to_world(root_pose_local: np.ndarray, env_origin_world: np.ndarray) -> np.ndarray:
        pose = np.asarray(root_pose_local, dtype=np.float32).copy()
        pose[:3] = pose[:3] + np.asarray(env_origin_world, dtype=np.float32)
        return pose

    def _pack_rigid_state_local(self, rigid_obj, env_id: int, env_origin_world: np.ndarray) -> dict[str, list[float]]:
        root_pose_w = rigid_obj.data.root_pose_w[int(env_id)].detach().cpu().numpy().astype(np.float32)
        root_vel_w = rigid_obj.data.root_vel_w[int(env_id)].detach().cpu().numpy().astype(np.float32)
        root_pose_local = self._pose_world_to_local(root_pose_w, env_origin_world)
        return {
            "root_pose_local": self._to_float_list(root_pose_local),
            "root_vel": self._to_float_list(root_vel_w),
        }

    def _write_rigid_state_local(
        self,
        rigid_obj,
        env_ids: torch.Tensor,
        root_pose_local: list[float],
        root_vel: list[float] | None,
    ) -> None:
        num_envs = int(env_ids.shape[0])
        env_origins = self.scene.env_origins[env_ids].detach().cpu().numpy().astype(np.float32)
        pose_local = np.asarray(root_pose_local, dtype=np.float32).reshape(7)
        pose_world = np.stack(
            [self._pose_local_to_world(pose_local, env_origins[i]) for i in range(num_envs)],
            axis=0,
        )
        if root_vel is None:
            vel_world = np.zeros((num_envs, 6), dtype=np.float32)
        else:
            vel_vec = np.asarray(root_vel, dtype=np.float32).reshape(6)
            vel_world = np.repeat(vel_vec[None, :], num_envs, axis=0)

        pose_t = torch.tensor(pose_world, device=self.device, dtype=torch.float32)
        vel_t = torch.tensor(vel_world, device=self.device, dtype=torch.float32)
        rigid_obj.write_root_pose_to_sim(pose_t, env_ids=env_ids)
        rigid_obj.write_root_velocity_to_sim(vel_t, env_ids=env_ids)

    def _get_camera_pose_local(
        self, env_id: int, camera_prim_name: str
    ) -> tuple[list[float], list[float]] | None:
        try:
            import omni.usd
            from pxr import UsdGeom
        except Exception:
            return None

        stage = omni.usd.get_context().get_stage()
        prim_path = f"{self._front_camera_prim_path_prefix}{int(env_id)}/{camera_prim_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None

        xformable = UsdGeom.Xformable(prim)
        translate = None
        quat = None
        for op in xformable.GetOrderedXformOps():
            if translate is None and op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                val = op.Get()
                if val is not None:
                    translate = [float(val[0]), float(val[1]), float(val[2])]
            elif quat is None and op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                val = op.Get()
                if val is not None:
                    imag = val.GetImaginary()
                    quat = [float(val.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])]
        if translate is None or quat is None:
            return None
        return translate, quat

    def _set_camera_pose_local(
        self,
        env_id: int,
        camera_prim_name: str,
        pos: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
    ) -> None:
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim_path = f"{self._front_camera_prim_path_prefix}{int(env_id)}/{camera_prim_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return

        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        orient_op = None
        for op in xformable.GetOrderedXformOps():
            if translate_op is None and op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif orient_op is None and op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient_op = op

        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if orient_op is None:
            orient_op = xformable.AddOrientOp()

        translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        orient_op.Set(
            Gf.Quatd(
                float(quat_wxyz[0]),
                Gf.Vec3d(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
            )
        )

    def get_camera_by_view_name(self, view_name: str):
        camera = self._camera_handles_by_view.get(str(view_name))
        if camera is not None:
            return camera
        return getattr(self, f"{view_name}_camera", None)

    def _iter_front_cameras(self):
        for view_name in self._front_camera_view_names:
            camera = getattr(self, f"{view_name}_camera", None)
            if camera is None:
                continue
            prim_name = self._front_camera_prim_names.get(view_name)
            if prim_name is None:
                continue
            yield view_name, prim_name, camera

    def _get_dome_light_state(self) -> dict[str, float | str]:
        try:
            import omni.usd
            from pxr import UsdGeom, UsdLux
        except Exception:
            return {}

        stage = omni.usd.get_context().get_stage()
        dome_light = UsdLux.DomeLight.Get(stage, self._dome_light_prim_path)
        if not dome_light:
            return {}

        intensity_attr = dome_light.GetIntensityAttr()
        intensity = intensity_attr.Get() if intensity_attr is not None else None

        texture_attr = dome_light.GetTextureFileAttr()
        texture_val = texture_attr.Get() if texture_attr is not None else None
        texture_file = ""
        if texture_val is not None:
            try:
                texture_file = str(texture_val.path)
            except Exception:
                texture_file = str(texture_val)

        yaw_deg = 0.0
        xformable = UsdGeom.Xformable(dome_light.GetPrim())
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rot = op.Get()
                if rot is not None:
                    yaw_deg = float(rot[2])
                break

        color_attr = dome_light.GetColorAttr()
        color_val = color_attr.Get() if color_attr is not None else None

        return {
            "intensity": float(intensity) if intensity is not None else 0.0,
            "yaw_deg": float(yaw_deg),
            "texture_file": texture_file,
            "color": [float(color_val[0]), float(color_val[1]), float(color_val[2])] if color_val is not None else [1.0, 1.0, 1.0],
        }

    def _apply_dome_light_state(self, dome_state: dict) -> None:
        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom, UsdLux
        except Exception as exc:
            raise RuntimeError("USD modules unavailable; cannot apply dome light reset params.") from exc

        stage = omni.usd.get_context().get_stage()
        dome_light = UsdLux.DomeLight.Get(stage, self._dome_light_prim_path)
        if not dome_light:
            return

        if "intensity" in dome_state:
            dome_light.CreateIntensityAttr().Set(float(dome_state["intensity"]))

        if "yaw_deg" in dome_state:
            xformable = UsdGeom.Xformable(dome_light.GetPrim())
            rotate_xyz_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                    rotate_xyz_op = op
                    break
            if rotate_xyz_op is None:
                rotate_xyz_op = xformable.AddRotateXYZOp()
            rotate_xyz_op.Set(Gf.Vec3f(0.0, 0.0, float(dome_state["yaw_deg"])))

        if "texture_file" in dome_state:
            dome_light.CreateTextureFileAttr().Set(Sdf.AssetPath(str(dome_state.get("texture_file", ""))))

        if "color" in dome_state:
            c = dome_state["color"]
            dome_light.CreateColorAttr().Set(Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])))

    def export_current_reset_params(self, env_id: int = 0) -> dict:
        env_id = int(env_id)
        env_origin = self.scene.env_origins[env_id].detach().cpu().numpy().astype(np.float32)
        joint_pos = self.robot.data.joint_pos[env_id].detach().cpu().numpy().astype(np.float32)
        joint_vel = self.robot.data.joint_vel[env_id].detach().cpu().numpy().astype(np.float32)
        cameras_out = {}
        for view_name, prim_name, camera in self._iter_front_cameras():
            front_cam_world = camera.data.pos_w[env_id].detach().cpu().numpy().astype(np.float32)
            front_cam_quat = camera.data.quat_w_ros[env_id].detach().cpu().numpy().astype(np.float32)
            front_local = self._get_camera_pose_local(env_id, prim_name)
            cameras_out[view_name] = {
                "local_pose": (
                    self._to_float_list(np.asarray(front_local[0] + front_local[1], dtype=np.float32))
                    if front_local is not None else None
                ),
                "world_pose": self._to_float_list(np.concatenate([front_cam_world, front_cam_quat], axis=0)),
            }
        if hasattr(self, "top_camera"):
            top_cam_world = self.top_camera.data.pos_w[env_id].detach().cpu().numpy().astype(np.float32)
            top_cam_quat = self.top_camera.data.quat_w_ros[env_id].detach().cpu().numpy().astype(np.float32)
            top_local = self._get_camera_pose_local(env_id, "TopCamera")
            cameras_out["top"] = {
                "local_pose": (
                    self._to_float_list(np.asarray(top_local[0] + top_local[1], dtype=np.float32))
                    if top_local is not None else None
                ),
                "world_pose": self._to_float_list(np.concatenate([top_cam_world, top_cam_quat], axis=0)),
            }
        if hasattr(self, "wrist_camera"):
            wrist_cam_world = self.wrist_camera.data.pos_w[env_id].detach().cpu().numpy().astype(np.float32)
            wrist_cam_quat = self.wrist_camera.data.quat_w_ros[env_id].detach().cpu().numpy().astype(np.float32)
            cameras_out["wrist"] = {
                "world_pose": self._to_float_list(np.concatenate([wrist_cam_world, wrist_cam_quat], axis=0)),
            }

        out = {
            "schema_version": 1,
            "physics": {
                "robot": {
                    "joint_pos": self._to_float_list(joint_pos),
                    "joint_vel": self._to_float_list(joint_vel),
                    "gripper_joint_pos": self._to_float_list(joint_pos[[self.left_joint_idx, self.right_joint_idx]]),
                },
                "table": (
                    self._pack_rigid_state_local(self.table, env_id, env_origin)
                    if self.is_table_active() else None
                ),
                "floor": self._pack_rigid_state_local(self.floor, env_id, env_origin) if self.floor is not None else None,
                "walls": {
                    wall_name: self._pack_rigid_state_local(wall_obj, env_id, env_origin)
                    for wall_name, wall_obj in self._walls.items()
                },
                "objects": {
                    name: self._pack_rigid_state_local(obj, env_id, env_origin)
                    for name, obj in self.objects.items()
                    if name != "table"
                },
                # Articulation (drawer/door) JOINT state. self.objects packs the cabinet as a rigid
                # (root pose only), so without this the drawer opening is LOST on restore -> a drawer
                # opened by open_drawer snaps back closed before the next skill (place-into-drawer)
                # runs. Captured per articulated object so _recover_simulator preserves the opening.
                "articulations": {
                    name: {
                        "joint_pos": self._to_float_list(
                            art.data.joint_pos[env_id].detach().cpu().numpy().astype(np.float32)),
                        "joint_vel": self._to_float_list(
                            art.data.joint_vel[env_id].detach().cpu().numpy().astype(np.float32)),
                    }
                    for name, art in getattr(self, "articulated_objects", {}).items()
                },
                "table_enabled": bool(self.is_table_active()),
                "floor_surface_z": float(self._current_floor_surface_z),
            },
            "visual": {
                "cameras": cameras_out,
                "dome_light": self._get_dome_light_state(),
            },
        }
        return out

    def _apply_env_reset_params(self, env_ids: torch.Tensor, reset_params: dict) -> None:
        if not isinstance(reset_params, dict):
            raise TypeError(f"reset_params must be dict, got {type(reset_params)}")

        physics = reset_params.get("physics")
        if not isinstance(physics, dict):
            raise KeyError("reset_params.physics is required")
        robot_state = physics.get("robot")
        if not isinstance(robot_state, dict):
            raise KeyError("reset_params.physics.robot is required")
        joint_pos = np.asarray(robot_state.get("joint_pos"), dtype=np.float32).reshape(-1)
        if joint_pos.shape[0] != int(self.robot.num_joints):
            raise ValueError(
                f"reset_params.physics.robot.joint_pos dim mismatch: expected {int(self.robot.num_joints)}, got {joint_pos.shape[0]}"
            )
        joint_vel = np.asarray(robot_state.get("joint_vel", np.zeros_like(joint_pos)), dtype=np.float32).reshape(-1)
        if joint_vel.shape[0] != int(self.robot.num_joints):
            raise ValueError(
                f"reset_params.physics.robot.joint_vel dim mismatch: expected {int(self.robot.num_joints)}, got {joint_vel.shape[0]}"
            )

        num_envs = int(env_ids.shape[0])
        joint_pos_t = torch.tensor(np.repeat(joint_pos[None, :], num_envs, axis=0), device=self.device, dtype=torch.float32)
        joint_vel_t = torch.tensor(np.repeat(joint_vel[None, :], num_envs, axis=0), device=self.device, dtype=torch.float32)
        self.robot.write_joint_state_to_sim(joint_pos_t, joint_vel_t, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos_t)
        self.robot.set_joint_velocity_target(joint_vel_t)
        self.robot_dof_targets[env_ids] = joint_pos_t

        table_state = physics.get("table")
        table_enabled_state = bool(physics.get("table_enabled", table_state is not None))
        self._table_enabled_this_episode = bool(table_enabled_state and self.table is not None)
        if table_state is not None and self.table is not None:
            self._write_rigid_state_local(
                self.table,
                env_ids,
                table_state["root_pose_local"],
                table_state.get("root_vel"),
            )
            table_pose_local = np.asarray(table_state["root_pose_local"], dtype=np.float32).reshape(7)
            self._table_spawn_pos = (float(table_pose_local[0]), float(table_pose_local[1]), float(table_pose_local[2]))
            self._update_table_3d_range()
        elif self.table is not None and not self._table_enabled_this_episode:
            env_origins = self.scene.env_origins[env_ids]
            self._set_table_pose(env_ids, env_origins, self._ground_plane_z - 10.0)

        floor_state = physics.get("floor")
        if floor_state is not None and self.floor is not None:
            self._write_rigid_state_local(
                self.floor,
                env_ids,
                floor_state["root_pose_local"],
                floor_state.get("root_vel"),
            )
            floor_pose_local = np.asarray(floor_state["root_pose_local"], dtype=np.float32).reshape(7)
            self._current_floor_surface_z = float(floor_pose_local[2])
        elif "floor_surface_z" in physics:
            self._current_floor_surface_z = float(physics["floor_surface_z"])

        walls_state = physics.get("walls", {})
        if isinstance(walls_state, dict):
            for wall_name, wall_obj in self._walls.items():
                w_state = walls_state.get(wall_name)
                if w_state is None:
                    continue
                self._write_rigid_state_local(
                    wall_obj,
                    env_ids,
                    w_state["root_pose_local"],
                    w_state.get("root_vel"),
                )

        # Position the kinematic ground cuboid to match the restored floor surface.
        if not self.is_table_active():
            self._set_ground_plane_z(env_ids, self._current_floor_surface_z - self._no_table_floor_above_ground)
        elif self._scene_floor_walls_enabled:
            self._set_ground_plane_z(env_ids, self._current_floor_surface_z - self._no_table_floor_above_ground)
        else:
            self._set_ground_plane_z(env_ids, self._floor_z_range[0] - self._ground_plane_z_offset)

        objects_state = physics.get("objects", {})
        if isinstance(objects_state, dict):
            for obj_name, obj_state in objects_state.items():
                if obj_name not in self.objects:
                    continue
                self._write_rigid_state_local(
                    self.objects[obj_name],
                    env_ids,
                    obj_state["root_pose_local"],
                    obj_state.get("root_vel"),
                )

        # Restore articulation (drawer/door) joint openings. The cabinet's root pose was restored above
        # (it is in self.objects), but its joints are a separate buffer -- without this the drawer snaps
        # back to its env-reset (closed) opening, breaking open->place. See export_current_reset_params.
        arti_state = physics.get("articulations", {})
        if isinstance(arti_state, dict):
            for art_name, a_state in arti_state.items():
                art = getattr(self, "articulated_objects", {}).get(art_name)
                if art is None or not isinstance(a_state, dict) or a_state.get("joint_pos") is None:
                    continue
                jp = np.asarray(a_state["joint_pos"], dtype=np.float32).reshape(-1)
                jv = np.asarray(a_state.get("joint_vel", np.zeros_like(jp)), dtype=np.float32).reshape(-1)
                jp_t = torch.tensor(np.repeat(jp[None, :], num_envs, axis=0), device=self.device, dtype=torch.float32)
                jv_t = torch.tensor(np.repeat(jv[None, :], num_envs, axis=0), device=self.device, dtype=torch.float32)
                art.write_joint_state_to_sim(jp_t, jv_t, env_ids=env_ids)

        visual = reset_params.get("visual", {})
        cameras = visual.get("cameras", {}) if isinstance(visual, dict) else {}
        if isinstance(cameras, dict):
            for view_name in self._front_camera_view_names:
                front_state = cameras.get(view_name)
                if front_state is None and view_name == "front":
                    # Backward compatibility with legacy single-front reset payloads.
                    front_state = cameras.get("front")
                if not isinstance(front_state, dict) or front_state.get("local_pose") is None:
                    continue
                front_pose = np.asarray(front_state["local_pose"], dtype=np.float32).reshape(7)
                prim_name = self._front_camera_prim_names.get(view_name)
                if prim_name is None:
                    continue
                for env_id in env_ids.tolist():
                    self._set_camera_pose_local(
                        int(env_id),
                        prim_name,
                        tuple(front_pose[:3].tolist()),
                        tuple(front_pose[3:].tolist()),
                    )

        top = cameras.get("top") if isinstance(cameras, dict) else None
        if isinstance(top, dict) and top.get("local_pose") is not None:
            top_pose = np.asarray(top["local_pose"], dtype=np.float32).reshape(7)
            for env_id in env_ids.tolist():
                self._set_camera_pose_local(
                    int(env_id),
                    "TopCamera",
                    tuple(top_pose[:3].tolist()),
                    tuple(top_pose[3:].tolist()),
                )

        for view_name in self._front_camera_view_names:
            camera = getattr(self, f"{view_name}_camera", None)
            if camera is None:
                continue
            try:
                camera.reset()
            except Exception:
                pass
        if hasattr(self, "top_camera"):
            try:
                self.top_camera.reset()
            except Exception:
                pass

        dome_state = visual.get("dome_light", {}) if isinstance(visual, dict) else {}
        if isinstance(dome_state, dict) and dome_state:
            self._apply_dome_light_state(dome_state)

    def _resolve_scene_asset_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self._scene_asset_root_path, path))

    def _ensure_kinematic_rigidbody_api(self, prim_path: str) -> None:
        """Ensure a prim has rigid-body schemas and kinematic flags expected by RigidObject."""
        try:
            from pxr import PhysxSchema, Sdf, UsdPhysics
        except Exception as exc:
            raise RuntimeError("Failed to import pxr physics schemas while patching rigid-body APIs.") from exc

        prim = prim_utils.get_prim_at_path(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found while patching rigid-body APIs: {prim_path}")

        UsdPhysics.RigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

        prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)

    def _init_layout_bank(self) -> None:
        layout_path = self.scene_desc.get("layouts")
        if not layout_path:
            return

        layout_path = self._resolve_scene_asset_path(layout_path)

        objects_cfg = self.scene_desc["objects"] if "objects" in self.scene_desc else self.scene_desc
        if not isinstance(objects_cfg, dict):
            raise ValueError("scene_desc['objects'] must be a dict when loading layouts.")

        layout_npz = np.load(layout_path, allow_pickle=True)
        lengths = []
        try:
            for name in objects_cfg.keys():
                if name not in layout_npz.files:
                    raise KeyError(f"Missing layout for object '{name}' in {layout_path}.")
                arr = np.asarray(layout_npz[name], dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] < 7:
                    raise ValueError(
                        f"Layout for '{name}' in {layout_path} must have shape (N, >=7), got {arr.shape}."
                    )
                self._layout_bank[name] = arr
                lengths.append(arr.shape[0])
        finally:
            try:
                layout_npz.close()
            except Exception:
                pass

        self._layout_min_len = min(lengths) if lengths else 0
        if self._layout_min_len <= 0:
            raise ValueError(f"No layouts available in {layout_path}.")
        self._layout_enabled = True
        self._init_layout_reset_banks()

    def _init_layout_reset_banks(self) -> None:
        table_z_path_raw = self.scene_desc.get("table_z_path")
        cam_poses_path_raw = self.scene_desc.get("cam_poses_path")
        if not table_z_path_raw or not cam_poses_path_raw:
            raise ValueError(
                "Layout mode requires scene_desc['table_z_path'] and scene_desc['cam_poses_path'] "
                "for layout-conditioned table/camera sampling."
            )

        table_z_path = self._resolve_scene_asset_path(str(table_z_path_raw))
        cam_poses_path = self._resolve_scene_asset_path(str(cam_poses_path_raw))

        if not os.path.exists(table_z_path):
            raise FileNotFoundError(f"table_z_path not found: {table_z_path}")
        if not os.path.exists(cam_poses_path):
            raise FileNotFoundError(f"cam_poses_path not found: {cam_poses_path}")

        table_loaded = np.load(table_z_path, allow_pickle=False)
        if isinstance(table_loaded, np.lib.npyio.NpzFile):
            if "table_z" in table_loaded.files:
                table_z = table_loaded["table_z"]
            elif "arr_0" in table_loaded.files:
                table_z = table_loaded["arr_0"]
            elif len(table_loaded.files) == 1:
                table_z = table_loaded[table_loaded.files[0]]
            else:
                raise ValueError(
                    f"table_z npz must contain key 'table_z' or a single array, got keys={table_loaded.files}"
                )
            table_loaded.close()
        else:
            table_z = table_loaded

        cam_loaded = np.load(cam_poses_path, allow_pickle=False)
        if isinstance(cam_loaded, np.lib.npyio.NpzFile):
            if "cam_poses" in cam_loaded.files:
                cam_poses = cam_loaded["cam_poses"]
            elif "arr_0" in cam_loaded.files:
                cam_poses = cam_loaded["arr_0"]
            elif len(cam_loaded.files) == 1:
                cam_poses = cam_loaded[cam_loaded.files[0]]
            else:
                raise ValueError(
                    f"cam_poses npz must contain key 'cam_poses' or a single array, got keys={cam_loaded.files}"
                )
            cam_loaded.close()
        else:
            cam_poses = cam_loaded

        table_z = np.asarray(table_z, dtype=np.float32)
        cam_poses = np.asarray(cam_poses, dtype=np.float32)

        if table_z.ndim != 1:
            raise ValueError(f"table_z must have shape (N,), got {table_z.shape} from {table_z_path}")
        if cam_poses.ndim != 3 or cam_poses.shape[2] < 7:
            raise ValueError(f"cam_poses must have shape (N, K, >=7), got {cam_poses.shape} from {cam_poses_path}")
        if cam_poses.shape[1] <= 0:
            raise ValueError(f"cam_poses must have K>0, got {cam_poses.shape} from {cam_poses_path}")

        if table_z.shape[0] != cam_poses.shape[0]:
            raise ValueError(
                f"table_z and cam_poses length mismatch: table_z={table_z.shape[0]}, cam_poses={cam_poses.shape[0]}"
            )
        if table_z.shape[0] < self._layout_min_len:
            raise ValueError(
                f"Layout reset banks are shorter than layout bank: reset_len={table_z.shape[0]}, "
                f"layout_min_len={self._layout_min_len}"
            )

        self._layout_table_z_bank = table_z
        self._layout_cam_poses_bank = cam_poses[:, :, :7]
        # Optional 8th dim of cam_poses = per-(layout, k) horizontal_aperture
        # in mm (baked by the FoV-augmenting pre-sample script). Runtime
        # writes this into the FrontCamera prim's horizontalAperture attr.
        if cam_poses.shape[2] >= 8:
            self._layout_cam_aperture_bank = cam_poses[:, :, 7].astype(np.float32)
        else:
            self._layout_cam_aperture_bank = None

    def _sample_layout_idx(self) -> int:
        if not self._layout_enabled or self._layout_min_len <= 0:
            raise RuntimeError("Layout sampling requested but layout bank is not initialized.")

        if self.force_layout_idx is not None:
            forced = int(self.force_layout_idx)
            if forced < 0 or forced >= self._layout_min_len:
                raise ValueError(
                    f"force_layout_idx={forced} out of range [0, {self._layout_min_len})"
                )
            return forced

        if self._layout_sample_candidates is not None:
            candidates = self._layout_sample_candidates
        else:
            all_ids = np.arange(self._layout_min_len, dtype=np.int64)
            test_ids = np.asarray(self._test_layout_ids, dtype=np.int64)
            if test_ids.size > 0:
                if int(np.min(test_ids)) < 0 or int(np.max(test_ids)) >= self._layout_min_len:
                    raise ValueError(
                        f"env.test_layout_id has out-of-range ids for layout size {self._layout_min_len}: "
                        f"{self._test_layout_ids}"
                    )
                test_ids = np.unique(test_ids)

            if self._layout_split == "test":
                candidates = test_ids
            else:
                if test_ids.size == 0:
                    candidates = all_ids
                else:
                    candidates = np.setdiff1d(all_ids, test_ids, assume_unique=False)

            if candidates.size == 0:
                raise ValueError(
                    f"No layout ids available for env.split='{self._layout_split}'. "
                    f"layout_count={self._layout_min_len}, test_layout_id={self._test_layout_ids}"
                )
            self._layout_sample_candidates = candidates.astype(np.int64, copy=False)
            candidates = self._layout_sample_candidates

        probs = self._build_front_heavy_layout_probs(int(candidates.shape[0]), self._layout_chunk_weights)
        with timestamp_seed_scope():
            idx = int(np.random.choice(candidates.shape[0], p=probs))
            return int(candidates[idx])

    @staticmethod
    def _build_front_heavy_layout_probs(num_candidates: int, chunk_weights: list[float]) -> np.ndarray:
        """Build 1D probabilities over four front->back quartiles with the given masses.

        Default chunk_weights of [4, 3, 2, 1] yield the front-heavy distribution.
        """
        if num_candidates <= 0:
            raise ValueError(f"num_candidates must be > 0, got {num_candidates}")
        if len(chunk_weights) != 4:
            raise ValueError(f"chunk_weights must be length 4, got {chunk_weights!r}")

        # Split candidate indices into four contiguous chunks from front to back.
        base = num_candidates // 4
        rem = num_candidates % 4
        sizes = [base + (1 if i < rem else 0) for i in range(4)]

        probs = np.zeros((num_candidates,), dtype=np.float64)
        cursor = 0
        total_mass = 0.0
        for size, mass in zip(sizes, chunk_weights):
            if size <= 0:
                continue
            probs[cursor: cursor + size] = mass / float(size)
            total_mass += mass
            cursor += size

        if total_mass <= 0.0:
            raise RuntimeError("Failed to build layout sampling probabilities.")
        probs /= np.sum(probs)
        return probs

    def _load_init_arm_pos_banks(self) -> None:
        self._arm_init_pos_banks = []
        self._arm_init_pos_bank_sizes = []

        if not self._init_arm_pos_files:
            return

        init_arm_paths: list[str] = []
        arm_dof = len(self.arm_joint_indices)
        for raw_path in self._init_arm_pos_files:
            if not raw_path:
                raise ValueError("init_arm_pos_files contains an empty path")

            init_arm_path = str(raw_path)
            if not os.path.exists(init_arm_path):
                raise FileNotFoundError(f"init_arm_pos file not found: {init_arm_path}")

            loaded = np.load(init_arm_path, allow_pickle=False)
            try:
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    if "arm_pos" in loaded.files:
                        arm_pos = loaded["arm_pos"]
                    elif "arr_0" in loaded.files:
                        arm_pos = loaded["arr_0"]
                    elif len(loaded.files) == 1:
                        arm_pos = loaded[loaded.files[0]]
                    else:
                        raise ValueError(
                            f"init_arm_pos npz must contain 'arm_pos' or a single array, got keys={loaded.files}"
                        )
                else:
                    arm_pos = loaded
                arm_pos = np.asarray(arm_pos, dtype=np.float32)
            finally:
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    loaded.close()

            if arm_pos.ndim != 2 or arm_pos.shape[1] != arm_dof:
                raise ValueError(
                    f"init_arm_pos must have shape (N, {arm_dof}), got {arm_pos.shape} from {init_arm_path}"
                )
            if arm_pos.shape[0] == 0:
                raise ValueError(f"init_arm_pos has no candidate rows: {init_arm_path}")
            if not np.isfinite(arm_pos).all():
                raise ValueError(f"init_arm_pos contains non-finite values: {init_arm_path}")

            bank = torch.tensor(arm_pos, device=self.device, dtype=torch.float32)
            self._arm_init_pos_banks.append(bank)
            self._arm_init_pos_bank_sizes.append(int(bank.shape[0]))
            init_arm_paths.append(init_arm_path)

        self._init_arm_pos_files = init_arm_paths

    def _sample_init_arm_joint_pos(self, num_envs: int) -> torch.Tensor | None:
        if not self._arm_init_pos_banks:
            return None
        arm_dof = len(self.arm_joint_indices)
        with timestamp_seed_scope():
            bank_ids = np.random.randint(0, len(self._arm_init_pos_banks), size=(num_envs,))
            sample_ids_by_bank = {
                bank_idx: np.random.randint(0, bank_size, size=(int(np.count_nonzero(bank_ids == bank_idx)),))
                for bank_idx, bank_size in enumerate(self._arm_init_pos_bank_sizes)
                if np.any(bank_ids == bank_idx)
            }

        arm_joint_pos = torch.empty((num_envs, arm_dof), device=self.device, dtype=torch.float32)
        for bank_idx, sample_ids in sample_ids_by_bank.items():
            env_ids = np.nonzero(bank_ids == bank_idx)[0]
            env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            sample_ids_t = torch.tensor(sample_ids, device=self.device, dtype=torch.long)
            selected = self._arm_init_pos_banks[bank_idx].index_select(0, sample_ids_t)
            arm_joint_pos.index_copy_(0, env_ids_t, selected)

        arm_lower = self.robot_dof_lower_limits[self.arm_joint_indices]
        arm_upper = self.robot_dof_upper_limits[self.arm_joint_indices]
        return torch.clamp(arm_joint_pos, arm_lower, arm_upper)

    def _sample_reset_gripper_joint_pos(self, num_envs: int) -> torch.Tensor:
        gripper_lower = self.robot_dof_lower_limits[[self.left_joint_idx, self.right_joint_idx]]
        gripper_upper = self.robot_dof_upper_limits[[self.left_joint_idx, self.right_joint_idx]]
        low = float(torch.max(gripper_lower).item())
        high = float(torch.min(gripper_upper).item())

        if not self._enable_gripper_init_randomization:
            canonical = float(self.max_gripper_opening)
            if not hasattr(self, "_logged_canonical_gripper_init"):
                print(
                    f"[gripper-init] canonical={canonical:.4f} (max opening); randomization disabled.",
                    flush=True,
                )
                self._logged_canonical_gripper_init = True
            return torch.full(
                (num_envs, 2), canonical, device=self.device, dtype=torch.float32
            )

        if high >= low:
            if high == low:
                open_vals = torch.full((num_envs, 1), low, device=self.device, dtype=torch.float32)
            else:
                with timestamp_seed_scope():
                    open_vals_np = np.random.uniform(low, high, size=(num_envs, 1))
                open_vals = torch.tensor(open_vals_np, device=self.device, dtype=torch.float32)
            return open_vals.repeat(1, 2)

        # Fallback to independent per-finger sampling when shared range is invalid.
        with timestamp_seed_scope():
            left_vals = np.random.uniform(
                float(gripper_lower[0].item()), float(gripper_upper[0].item()), size=(num_envs, 1)
            )
            right_vals = np.random.uniform(
                float(gripper_lower[1].item()), float(gripper_upper[1].item()), size=(num_envs, 1)
            )
        return torch.tensor(
            np.concatenate((left_vals, right_vals), axis=1), device=self.device, dtype=torch.float32
        )

    def is_table_active(self) -> bool:
        return bool(self.table is not None and self._table_enabled_this_episode)

    def _sample_table_enabled(self) -> bool:
        if self.table is None:
            return False
        return bool(self._scene_has_table_desc)

    def _get_floor_surface_z(self) -> float:
        return float(self._current_floor_surface_z)

    def _get_support_surface_z(self) -> float:
        if self.is_table_active():
            return float(self._table_spawn_pos[2]) + float(self._get_table_height()) / 2.0
        return self._get_floor_surface_z()

    def _compute_world_init_pos(self, layout_pos: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
        support_z = self._get_support_surface_z()
        # Objects are anchored at object_xy_anchor (decoupled from the table prim
        # for simpler; = table_xy for legacy coupled placement).
        return (
            float(layout_pos[0]) + self._object_xy_anchor[0],
            float(layout_pos[1]) + self._object_xy_anchor[1],
            float(layout_pos[2]) + support_z,
        )

    def _get_table_height(self) -> float:
        if not hasattr(self, "_scene_table_size") or self._scene_table_size is None:
            raise RuntimeError("Table height is unavailable: scene_desc['table']['size'] was not initialized.")
        return float(self._scene_table_size[2])

    def _sample_table_z(self, layout_idx: int | None = None) -> float:
        if layout_idx is not None:
            if self._layout_table_z_bank is None:
                raise RuntimeError("Layout table_z bank is not initialized.")
            if layout_idx < 0 or layout_idx >= int(self._layout_table_z_bank.shape[0]):
                raise IndexError(
                    f"layout_idx {layout_idx} out of range for table_z bank with len={self._layout_table_z_bank.shape[0]}"
                )
            return float(self._layout_table_z_bank[layout_idx])

        z_min, z_max = self._table_z_range
        if z_min == z_max:
            return z_min
        with timestamp_seed_scope():
            return float(np.random.uniform(z_min, z_max))

    def _clip_table_center_z(self, table_center_z: float) -> float:
        z = float(table_center_z)
        if self._max_table_surface_height is None and self._min_table_surface_height is None:
            return z
        table_half_height = self._get_table_height() / 2.0
        if self._max_table_surface_height is not None:
            z = min(z, float(self._max_table_surface_height) - table_half_height)
        if self._min_table_surface_height is not None:
            z = max(z, float(self._min_table_surface_height) - table_half_height)
        return z

    def _update_table_3d_range(self):
        table_center = self._table_spawn_pos
        table_extent = self.table_size / 2.0
        self.table_3d_range = (
            table_center[0] - table_extent[0], table_center[0] + table_extent[0],
            table_center[1] - table_extent[1], table_center[1] + table_extent[1],
            table_center[2] - table_extent[2], table_center[2] + table_extent[2],
        )

    def _set_table_pose(
        self,
        env_ids: torch.Tensor,
        env_origins: torch.Tensor,
        table_center_z: float,
        update_table_spawn_pos: bool = True,
    ) -> None:
        if self.table is None:
            return
        if update_table_spawn_pos:
            self._table_spawn_pos = (self._table_xy[0], self._table_xy[1], float(table_center_z))
            self._update_table_3d_range()
        table_pos = torch.tensor(self._table_spawn_pos, device=self.device, dtype=torch.float32).unsqueeze(0) + env_origins
        table_quat = torch.tensor(self._table_rot, device=self.device, dtype=torch.float32).unsqueeze(0).expand(table_pos.shape[0], -1)
        table_root_pose = torch.cat((table_pos, table_quat), dim=-1)
        table_root_vel = torch.zeros((table_root_pose.shape[0], 6), device=self.device, dtype=torch.float32)
        self.table.write_root_pose_to_sim(table_root_pose, env_ids=env_ids)
        self.table.write_root_velocity_to_sim(table_root_vel, env_ids=env_ids)

    def _set_ground_plane_z(self, env_ids: torch.Tensor, top_surface_z: float) -> None:
        """Move the kinematic ground cuboid so its top surface is at top_surface_z."""
        if self._moveable_ground is None:
            return
        # Cuboid is 0.02 m thick; center must be half-thickness below the desired surface.
        _HALF_THICKNESS = 0.01
        print(f"[ground] top_surface_z={top_surface_z:.4f}  center_z={top_surface_z - _HALF_THICKNESS:.4f}")
        center_z = top_surface_z - _HALF_THICKNESS
        num_envs = int(env_ids.shape[0])
        env_origins = self.scene.env_origins[env_ids]
        pos = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        pos[:, 2] = center_z
        pos += env_origins
        quat = torch.tensor([1., 0., 0., 0.], device=self.device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
        self._moveable_ground.write_root_pose_to_sim(torch.cat((pos, quat), dim=-1), env_ids=env_ids)
        self._moveable_ground.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=self.device), env_ids=env_ids)

    def _sample_floor_z(self) -> float:
        z_min, z_max = self._floor_z_range
        if z_min == z_max:
            return z_min
        with timestamp_seed_scope():
            return float(np.random.uniform(z_min, z_max))

    @staticmethod
    def _parse_range_segments(
        spec: str,
        field_name: str,
        min_value: float | None = None,
        max_value: float | None = None,
        is_angle: bool = False,
    ) -> list[tuple[float, float]]:
        if not isinstance(spec, str):
            raise TypeError(f"{field_name} must be a string like 'a:b,c:d', got {type(spec)}")
        # Accept both ',' and ';' as interval separators in config strings.
        segments: list[tuple[float, float]] = []
        normalized_spec = spec.replace(";", ",")
        for chunk in normalized_spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(f"{field_name} chunk '{chunk}' must use 'min:max' format.")
            lo_s, hi_s = chunk.split(":", 1)
            lo = float(lo_s.strip())
            hi = float(hi_s.strip())
            if lo > hi:
                raise ValueError(f"{field_name} has invalid interval '{chunk}' with min > max.")
            if min_value is not None and lo < min_value:
                raise ValueError(f"{field_name} lower bound {lo} violates min {min_value}.")
            if max_value is not None and hi > max_value:
                raise ValueError(f"{field_name} upper bound {hi} violates max {max_value}.")
            if is_angle:
                segments.append((float(np.deg2rad(lo)), float(np.deg2rad(hi))))
            else:
                segments.append((lo, hi))
        if not segments:
            raise ValueError(f"{field_name} must contain at least one valid interval.")
        return segments

    @staticmethod
    def _parse_optional_scalar(
        value: object,
        field_name: str,
        min_value: float | None = None,
        max_value: float | None = None,
        is_angle: bool = False,
    ) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() == "none":
                return None
            parsed = float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            parsed = float(value)
        else:
            raise TypeError(
                f"{field_name} must be float/int/None (or string convertible to float), got {type(value)}"
            )

        if min_value is not None and parsed < min_value:
            raise ValueError(f"{field_name} value {parsed} violates min {min_value}.")
        if max_value is not None and parsed > max_value:
            raise ValueError(f"{field_name} value {parsed} violates max {max_value}.")
        if is_angle:
            return float(np.deg2rad(parsed))
        return float(parsed)

    @staticmethod
    def _sample_from_segments(
        segments: list[tuple[float, float]],
        sampling_mode: str = "uniform",
        mode_value: float | None = None,
    ) -> float:
        with timestamp_seed_scope():
            seg_idx = int(np.random.randint(len(segments)))
        lo, hi = segments[seg_idx]
        if lo == hi:
            return float(lo)
        if sampling_mode == "uniform":
            with timestamp_seed_scope():
                return float(np.random.uniform(lo, hi))
        if sampling_mode == "triangular":
            tri_mode = (lo + hi) * 0.5 if mode_value is None else float(np.clip(mode_value, lo, hi))
            with timestamp_seed_scope():
                return float(np.random.triangular(lo, tri_mode, hi))
        raise ValueError(
            f"Unsupported sampling_mode={sampling_mode!r}; expected 'uniform' or 'triangular'."
        )

    @staticmethod
    def _quat_from_rotation_matrix_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
        tr = float(np.trace(rot))
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rot[2, 1] - rot[1, 2]) / s
            qy = (rot[0, 2] - rot[2, 0]) / s
            qz = (rot[1, 0] - rot[0, 1]) / s
        else:
            i = int(np.argmax([rot[0, 0], rot[1, 1], rot[2, 2]]))
            if i == 0:
                s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
                qw = (rot[2, 1] - rot[1, 2]) / s
                qx = 0.25 * s
                qy = (rot[0, 1] + rot[1, 0]) / s
                qz = (rot[0, 2] + rot[2, 0]) / s
            elif i == 1:
                s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
                qw = (rot[0, 2] - rot[2, 0]) / s
                qx = (rot[0, 1] + rot[1, 0]) / s
                qy = 0.25 * s
                qz = (rot[1, 2] + rot[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
                qw = (rot[1, 0] - rot[0, 1]) / s
                qx = (rot[0, 2] + rot[2, 0]) / s
                qy = (rot[1, 2] + rot[2, 1]) / s
                qz = 0.25 * s
        quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
        quat /= max(np.linalg.norm(quat), 1e-12)
        return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    def _quat_wxyz_lookat_usd(
        self,
        cam_pos: tuple[float, float, float],
        target: tuple[float, float, float],
        up_world: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> tuple[float, float, float, float]:
        cam_pos_np = np.asarray(cam_pos, dtype=np.float64)
        target_np = np.asarray(target, dtype=np.float64)
        up_np = np.asarray(up_world, dtype=np.float64)

        forward = target_np - cam_pos_np
        forward /= max(np.linalg.norm(forward), 1e-12)

        # USD camera local -Z is forward in world.
        z_axis = -forward

        x_axis = np.cross(up_np, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            fallback_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
            x_axis = np.cross(fallback_up, z_axis)
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-8:
                return (1.0, 0.0, 0.0, 0.0)
        x_axis /= x_norm

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= max(np.linalg.norm(y_axis), 1e-12)

        rot_wc = np.stack([x_axis, y_axis, z_axis], axis=1)
        return self._quat_from_rotation_matrix_wxyz(rot_wc)

    def _get_front_camera_aug_center(self) -> tuple[float, float, float]:
        center_x, center_y, _ = self._front_camera_aug_center
        # Lookat Z priority:
        #   1. front_camera_aug_lookat_z_ranges set -> sample fresh per episode.
        #   2. Else legacy: lock to current table surface Z (or floor when no
        #      active table).
        if self._front_camera_aug_lookat_z_ranges:
            sampled_z = self._sample_from_segments(
                self._front_camera_aug_lookat_z_ranges,
                sampling_mode=self._front_camera_aug_sampling_mode,
                mode_value=self._front_camera_aug_lookat_z_mode,
            )
            return center_x, center_y, float(sampled_z)
        if self.table is None:
            return float(self._table_xy[0]), float(self._table_xy[1]), self._get_floor_surface_z()
        if not self.is_table_active():
            return float(self._table_xy[0]), float(self._table_xy[1]), self._get_floor_surface_z()
        table_surface_z = float(self._table_spawn_pos[2]) + float(self._get_table_height()) / 2.0
        return center_x, center_y, table_surface_z

    def _sample_front_camera_pose_local(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        radius = self._sample_from_segments(
            self._front_camera_aug_radius_ranges,
            sampling_mode=self._front_camera_aug_sampling_mode,
            mode_value=self._front_camera_aug_radius_mode,
        )
        azimuth = self._sample_from_segments(
            self._front_camera_aug_azimuth_ranges,
            sampling_mode=self._front_camera_aug_sampling_mode,
            mode_value=self._front_camera_aug_azimuth_mode,
        )
        elevation = self._sample_from_segments(
            self._front_camera_aug_elevation_ranges,
            sampling_mode=self._front_camera_aug_sampling_mode,
            mode_value=self._front_camera_aug_elevation_mode,
        )

        center = self._get_front_camera_aug_center()
        cos_el = float(np.cos(elevation))
        cam_pos = (
            float(center[0] + radius * cos_el * np.cos(azimuth)),
            float(center[1] + radius * cos_el * np.sin(azimuth)),
            float(center[2] + radius * np.sin(elevation)),
        )
        lookat_quat = self._quat_wxyz_lookat_usd(cam_pos, center)

        roll_half = np.deg2rad(self._front_camera_aug_roll_deg) / 2.0
        q_roll = (float(np.cos(roll_half)), 0.0, 0.0, float(np.sin(roll_half)))
        cam_quat = self._quat_mul_wxyz(lookat_quat, q_roll)
        return cam_pos, cam_quat

    def _set_front_camera_pose_local(
        self,
        env_id: int,
        pos: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
        view_name: str = "front",
    ) -> None:
        prim_name = self._front_camera_prim_names.get(view_name)
        if prim_name is None:
            raise KeyError(f"Unknown front camera view '{view_name}'")
        self._set_camera_pose_local(env_id, prim_name, pos, quat_wxyz)

    def _randomize_front_camera_pose(self, env_ids: torch.Tensor, layout_idx: int | None = None) -> None:
        if layout_idx is not None:
            if self._layout_cam_poses_bank is None:
                raise RuntimeError("Layout camera pose bank is not initialized.")
            if layout_idx < 0 or layout_idx >= int(self._layout_cam_poses_bank.shape[0]):
                raise IndexError(
                    f"layout_idx {layout_idx} out of range for cam_poses bank with len={self._layout_cam_poses_bank.shape[0]}"
                )
            cam_poses = self._layout_cam_poses_bank[layout_idx]
            num_camposes = int(cam_poses.shape[0])
            if num_camposes <= 0:
                raise ValueError(f"cam_poses[{layout_idx}] has no poses.")

            # Bank-driven FoV: when cam_poses.npz has an 8th aperture dim
            # (pre-sample ran with FoV aug on), apply it per layout. Otherwise
            # stay on cfg.front_camera.spawn.horizontal_aperture set at init.
            aperture_per_cam = (
                self._layout_cam_aperture_bank[layout_idx]
                if self._layout_cam_aperture_bank is not None
                else None
            )
            for env_id in env_ids.tolist():
                for cam_idx, view_name in enumerate(self._front_camera_view_names):
                    pose = cam_poses[cam_idx % num_camposes]
                    pos = (float(pose[0]), float(pose[1]), float(pose[2]))
                    quat = (float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
                    self._set_front_camera_pose_local(int(env_id), pos, quat, view_name=view_name)
                    if aperture_per_cam is not None:
                        self._set_front_camera_aperture_local(
                            int(env_id),
                            view_name,
                            float(aperture_per_cam[cam_idx % num_camposes]),
                        )
        else:
            if not self._enable_front_camera_pose_augmentation:
                return
            for env_id in env_ids.tolist():
                for view_name in self._front_camera_view_names:
                    pos, quat = self._sample_front_camera_pose_local()
                    self._set_front_camera_pose_local(int(env_id), pos, quat, view_name=view_name)

        for view_name in self._front_camera_view_names:
            camera = getattr(self, f"{view_name}_camera", None)
            if camera is None:
                continue
            try:
                camera.reset()
            except Exception:
                pass

    def _set_front_camera_aperture_local(
        self,
        env_id: int,
        view_name: str,
        aperture_mm: float,
    ) -> None:
        """Write horizontal_aperture on the FrontCamera{N} prim.

        vertical_aperture is auto-derived by the renderer from the aspect ratio,
        so we only need to write horizontalAperture here.
        """
        prim_name = self._front_camera_prim_names.get(view_name)
        if prim_name is None:
            raise KeyError(f"Unknown front camera view '{view_name}'")
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim_path = f"{self._front_camera_prim_path_prefix}{int(env_id)}/{prim_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        camera = UsdGeom.Camera(prim)
        if not camera:
            return
        camera.GetHorizontalApertureAttr().Set(float(aperture_mm))

    def _sample_wrist_camera_offset(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Pick one combo uniformly from wrist_camera_aug_combos.

        Returns (pos, quat_wxyz) in wrist_camera_link local frame, layered on
        top of the baked URDF (already at -5cm + 15deg). Each combo is
        (back_m_extra, pitch_deg_extra) with mount_offset semantics: pos =
        (0, back_m_extra, 0); rot = R_x(pitch_deg_extra) as quaternion.
        """
        with timestamp_seed_scope():
            idx = int(np.random.randint(len(self._wrist_camera_aug_combos)))
        back_m, pitch_deg = self._wrist_camera_aug_combos[idx]
        half = math.radians(pitch_deg) / 2.0
        return (
            (0.0, float(back_m), 0.0),
            (math.cos(half), math.sin(half), 0.0, 0.0),
        )

    def _set_wrist_camera_pose_local(
        self,
        env_id: int,
        pos: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
    ) -> None:
        """Write WristCamera prim's local xform (relative to wrist_camera_link)."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim_path = (
            f"/World/envs/env_{int(env_id)}/Robot/wrist_camera_link/WristCamera"
        )
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return

        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        orient_op = None
        for op in xformable.GetOrderedXformOps():
            if translate_op is None and op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif orient_op is None and op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient_op = op

        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if orient_op is None:
            orient_op = xformable.AddOrientOp()

        translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        orient_op.Set(
            Gf.Quatd(
                float(quat_wxyz[0]),
                Gf.Vec3d(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
            )
        )

    def _randomize_wrist_camera_pose(self, env_ids: torch.Tensor) -> None:
        if not self._enable_wrist_camera_pose_augmentation:
            return
        for env_id in env_ids.tolist():
            pos, quat = self._sample_wrist_camera_offset()
            self._set_wrist_camera_pose_local(int(env_id), pos, quat)
        camera = getattr(self, "wrist_camera", None)
        if camera is not None:
            try:
                camera.reset()
            except Exception:
                pass

    def _sample_wall_slot_assignment(self) -> dict[str, str]:
        with timestamp_seed_scope():
            slot_perm = np.random.permutation(len(self._wall_slots))
        return {wall_prim_id: self._wall_slots[int(slot_perm[i])] for i, wall_prim_id in enumerate(self._wall_prim_ids)}

    def _randomize_dome_light(self) -> None:
        if not self._enable_dome_light_augmentation:
            return

        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()
        dome_light = UsdLux.DomeLight.Get(stage, self._dome_light_prim_path)
        if not dome_light:
            return
        with timestamp_seed_scope():
            intensity_hit = np.random.uniform(0.0, 1.0) <= self._dome_light_intensity_aug_prob
        if intensity_hit:
            with timestamp_seed_scope():
                intensity = float(np.random.uniform(self._dome_light_intensity_range[0], self._dome_light_intensity_range[1]))
            dome_light.CreateIntensityAttr().Set(intensity)

        with timestamp_seed_scope():
            yaw_hit = np.random.uniform(0.0, 1.0) <= self._dome_light_yaw_aug_prob
        if yaw_hit:
            with timestamp_seed_scope():
                yaw_rad = float(np.random.uniform(self._dome_light_yaw_range[0], self._dome_light_yaw_range[1]))
            yaw_deg = float(np.degrees(yaw_rad))
            xformable = UsdGeom.Xformable(dome_light.GetPrim())
            rotate_xyz_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                    rotate_xyz_op = op
                    break
            if rotate_xyz_op is None:
                rotate_xyz_op = xformable.AddRotateXYZOp()
            rotate_xyz_op.Set(Gf.Vec3f(0.0, 0.0, yaw_deg))

        with timestamp_seed_scope():
            texture_hit = np.random.uniform(0.0, 1.0) <= self._dome_light_texture_aug_prob
        apply_texture_aug = self._dome_light_texture_path is not None and texture_hit
        if apply_texture_aug:
            dome_light.CreateTextureFileAttr().Set(Sdf.AssetPath(self._dome_light_texture_path))
        else:
            # Explicitly clear texture when texture augmentation is not sampled.
            dome_light.CreateTextureFileAttr().Set(Sdf.AssetPath(""))

        if not apply_texture_aug and self._dome_light_color_aug_prob > 0.0:
            with timestamp_seed_scope():
                color_hit = np.random.uniform(0.0, 1.0) <= self._dome_light_color_aug_prob
            if color_hit:
                with timestamp_seed_scope():
                    r = float(np.random.uniform(self._dome_light_color_r_range[0], self._dome_light_color_r_range[1]))
                with timestamp_seed_scope():
                    g = float(np.random.uniform(self._dome_light_color_g_range[0], self._dome_light_color_g_range[1]))
                with timestamp_seed_scope():
                    b = float(np.random.uniform(self._dome_light_color_b_range[0], self._dome_light_color_b_range[1]))
                dome_light.CreateColorAttr().Set(Gf.Vec3f(r, g, b))
            else:
                dome_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    def _reset_floor_and_walls(self, env_ids: torch.Tensor, force_floor_z: float | None = None) -> None:
        num_envs = int(env_ids.shape[0])
        if num_envs == 0:
            return

        env_origins = self.scene.env_origins[env_ids]
        if force_floor_z is None:
            floor_z_vals = torch.tensor(
                [self._sample_floor_z() for _ in range(num_envs)],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            floor_z_vals = torch.full((num_envs,), float(force_floor_z), device=self.device, dtype=torch.float32)
        self._current_floor_surface_z = float(floor_z_vals[0].item())

        if self.floor is not None:
            floor_pos = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
            floor_pos[:, 0] = float(self._floor_xy[0])
            floor_pos[:, 1] = float(self._floor_xy[1])
            floor_pos[:, 2] = floor_z_vals
            floor_pos += env_origins
            floor_quat = torch.tensor(self._floor_rot, device=self.device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
            floor_root_pose = torch.cat((floor_pos, floor_quat), dim=-1)
            floor_root_vel = torch.zeros((num_envs, 6), device=self.device, dtype=torch.float32)
            self.floor.write_root_pose_to_sim(floor_root_pose, env_ids=env_ids)
            self.floor.write_root_velocity_to_sim(floor_root_vel, env_ids=env_ids)

        wall_slot_assignments = [self._sample_wall_slot_assignment() for _ in range(num_envs)]
        wall_center_z_vals = floor_z_vals + float(self._wall_height) / 2.0

        for wall_prim_id, wall_obj in self._walls.items():
            wall_pos = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
            wall_quat = torch.zeros((num_envs, 4), device=self.device, dtype=torch.float32)
            for i in range(num_envs):
                slot = wall_slot_assignments[i][wall_prim_id]
                slot_xy = self._wall_slot_xy[slot]
                wall_pos[i, 0] = float(slot_xy[0])
                wall_pos[i, 1] = float(slot_xy[1])
                wall_pos[i, 2] = wall_center_z_vals[i]
                wall_quat[i] = torch.tensor(self._wall_slot_rot[slot], device=self.device, dtype=torch.float32)

            wall_pos += env_origins
            wall_root_pose = torch.cat((wall_pos, wall_quat), dim=-1)
            wall_root_vel = torch.zeros((num_envs, 6), device=self.device, dtype=torch.float32)
            wall_obj.write_root_pose_to_sim(wall_root_pose, env_ids=env_ids)
            wall_obj.write_root_velocity_to_sim(wall_root_vel, env_ids=env_ids)

    def _debug_report_floor_material_binding(self) -> None:
        if not self._debug_floor_material_binding:
            return
        if self.floor is None:
            print("[floor-debug] floor object is None.")
            return
        try:
            import omni.usd
            from pxr import Sdf, UsdGeom, UsdShade

            stage = omni.usd.get_context().get_stage()
            floor_prim = stage.GetPrimAtPath(self._floor_prim_path_env0)
            if not floor_prim or not floor_prim.IsValid():
                print(f"[floor-debug] floor prim not found: {self._floor_prim_path_env0}")
                return

            print(f"[floor-debug] scene floor usd: {self._scene_floor_usd_path}")
            reported_any = False
            for prim in Usd.PrimRange(floor_prim):
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
                if material is None:
                    continue
                for shader_prim in Usd.PrimRange(material.GetPrim()):
                    shader = UsdShade.Shader(shader_prim)
                    if not shader:
                        continue
                    shader_id = shader.GetIdAttr().Get()
                    if shader_id != "UsdUVTexture":
                        continue
                    file_input = shader.GetInput("file")
                    if not file_input:
                        continue
                    asset = file_input.Get()
                    if isinstance(asset, Sdf.AssetPath):
                        raw_path = asset.path or ""
                        resolved_path = asset.resolvedPath or ""
                    else:
                        raw_path = str(asset) if asset is not None else ""
                        resolved_path = ""

                    guessed_path = (
                        os.path.normpath(os.path.join(os.path.dirname(self._scene_floor_usd_path), raw_path))
                        if raw_path and self._scene_floor_usd_path
                        else ""
                    )
                    print(
                        f"[floor-debug] gprim={prim.GetPath()} material={material.GetPath()} "
                        f"raw='{raw_path}' resolved='{resolved_path}' "
                        f"resolved_exists={bool(resolved_path and os.path.exists(resolved_path))} "
                        f"guess='{guessed_path}' guess_exists={bool(guessed_path and os.path.exists(guessed_path))}"
                    )
                    reported_any = True
            if not reported_any:
                print("[floor-debug] no UsdUVTexture file binding found under floor material bindings.")
        except Exception as exc:
            print(f"[floor-debug] failed to inspect floor material binding: {exc}")

    def _hide_moveable_ground_visuals(self) -> None:
        if not self._hide_moveable_ground_visual:
            return
        try:
            import omni.usd
            from pxr import Usd, UsdGeom
        except Exception as exc:
            print(f"[ground-visual] skip hide (USD modules unavailable): {exc}")
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        for env_id in range(int(self.num_envs)):
            root_path = f"/World/envs/env_{env_id}/moveable_ground"
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim or not root_prim.IsValid():
                continue
            for prim in Usd.PrimRange(root_prim):
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()

    def _setup_scene(self, use_camera: bool = True):
        prim_utils.create_prim("/World/envs/env_0/Objects", "Xform")
        # setup scene
        self.robot = Articulation(self.cfg.robot)
        self.table = None
        if self._scene_table_usd_path is not None:
            table_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/table",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=self._scene_table_usd_path,
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=self._table_spawn_pos,
                    rot=self._table_rot,
                ),
            )
            self.table = RigidObject(table_cfg)
            self._ensure_kinematic_rigidbody_api("/World/envs/env_0/table")

        self._camera_handles_by_view = {}
        if use_camera:
            # camera_spawn_names (injected by make_env from the simulator
            # camera_names list) gates which non-wrist cameras spawn; None =
            # spawn all (legacy / direct env construction). Wrist spawning is
            # governed solely by enable_wrist_camera below.
            _spawn_names = getattr(self.cfg, "camera_spawn_names", None)
            _spawn_front = _spawn_names is None or any(
                n == "front" or n.startswith("front_") for n in _spawn_names
            )
            _spawn_top = _spawn_names is None or "top" in _spawn_names
            if _spawn_front:
                for view_name in self._front_camera_view_names:
                    front_cfg = copy.deepcopy(self.cfg.front_camera)
                    front_cfg.prim_path = f"/World/envs/env_.*/{self._front_camera_prim_names[view_name]}"
                    front_camera = TiledCamera(front_cfg)
                    setattr(self, f"{view_name}_camera", front_camera)
                    self._camera_handles_by_view[view_name] = front_camera
            # Keep legacy attribute for compatibility (None when front not spawned).
            self.front_camera = self._camera_handles_by_view.get("front")
            if _spawn_top:
                self.top_camera = TiledCamera(self.cfg.top_camera)
                self._camera_handles_by_view["top"] = self.top_camera
            # Robots without a wrist_camera_link (e.g. WidowX / bridge) must skip
            # the wrist camera entirely: its prim path anchors on the robot USD.
            # Downstream references use hasattr/getattr and stay None-safe.
            if bool(getattr(self.cfg, "enable_wrist_camera", True)):
                self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
                self._camera_handles_by_view["wrist"] = self.wrist_camera
        self.contact = ContactSensor(self.cfg.contact)

        moveable_ground_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/moveable_ground",
            spawn=sim_utils.CuboidCfg(
                size=(10.0, 10.0, 0.02),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(opacity=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self._ground_plane_z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        self._moveable_ground = RigidObject(moveable_ground_cfg)
        if self._scene_floor_walls_enabled:
            init_floor_z = self._sample_floor_z()
            self._current_floor_surface_z = float(init_floor_z)
            if self._scene_floor_visual_enabled:
                floor_cfg = RigidObjectCfg(
                    prim_path="/World/envs/env_.*/floor",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=self._scene_floor_usd_path,
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                            disable_gravity=True,
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(self._floor_xy[0], self._floor_xy[1], init_floor_z),
                        rot=self._floor_rot,
                    ),
                )
                self.floor = RigidObject(floor_cfg)
                self._ensure_kinematic_rigidbody_api(self._floor_prim_path_env0)
                self._debug_report_floor_material_binding()

            if self._scene_walls_visual_enabled:
                num_walls = len(self._wall_prim_ids)
                if len(self._scene_wall_usd_paths) < num_walls:
                    raise ValueError(
                        f"Need at least {num_walls} wall USD paths, got {len(self._scene_wall_usd_paths)}."
                    )
                init_wall_usd_assignment = {
                    wall_prim_id: self._scene_wall_usd_paths[i]
                    for i, wall_prim_id in enumerate(self._wall_prim_ids)
                }
                init_wall_slot_assignment = self._sample_wall_slot_assignment()
                init_wall_center_z = init_floor_z + self._wall_height / 2.0
                for wall_prim_id in self._wall_prim_ids:
                    slot = init_wall_slot_assignment[wall_prim_id]
                    wall_xy = self._wall_slot_xy[slot]
                    wall_cfg = RigidObjectCfg(
                        prim_path=f"/World/envs/env_.*/{wall_prim_id}",
                        spawn=sim_utils.UsdFileCfg(
                            usd_path=init_wall_usd_assignment[wall_prim_id],
                            scale=self._wall_scale,
                            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                kinematic_enabled=True,
                                disable_gravity=True,
                            ),
                        ),
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=(wall_xy[0], wall_xy[1], init_wall_center_z),
                            rot=self._wall_slot_rot[slot],
                        ),
                    )
                    wall_obj = RigidObject(wall_cfg)
                    self._ensure_kinematic_rigidbody_api(f"/World/envs/env_0/{wall_prim_id}")
                    self._walls[wall_prim_id] = wall_obj
                    if wall_prim_id == "wall_1":
                        self.wall_1 = wall_obj
                    elif wall_prim_id == "wall_2":
                        self.wall_2 = wall_obj
                    elif wall_prim_id == "wall_3":
                        self.wall_3 = wall_obj
                    elif wall_prim_id == "wall_4":
                        self.wall_4 = wall_obj

        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)
        self._hide_moveable_ground_visuals()

        if self.floor is not None:
            floor_prim = prim_utils.get_prim_at_path(self._floor_prim_path_env0)
            sem_utils.add_update_semantics(floor_prim, semantic_label="floor")
        for wall_prim_id in self._walls.keys():
            wall_prim_path = self._wall_prim_paths_env0[wall_prim_id]
            wall_prim = prim_utils.get_prim_at_path(wall_prim_path)
            sem_utils.add_update_semantics(wall_prim, semantic_label="wall")

        # store prim paths
        self.table_prim = "/World/envs/env_0/table" if self.table is not None else None
        self.object_prims = dict()

        # set up objects in the scene
        self.objects: dict[str, RigidObject] = {}  # rigid objects (articulations are ALSO stored here, dual-store)
        # Articulated scene objects (e.g. a multi-DOF drawer cabinet). Dual-stored: also in
        # self.objects so root-pose / pcd / object-list code keeps working. See docs/arti/scene_import.md.
        self.articulated_objects: dict[str, Articulation] = {}
        # --- Unified [name][part] meta framework (see simulators/part_meta.py). Parts: "base",
        # "drawer_<i>", "door_<j>"; keys are SPARSE (a dict simply lacks the key when the
        # property does not apply). Legacy int drawer_id == movable-part position in URDF
        # order == object_part_names[name][drawer_id + 1].
        #
        # The object loop below fills these dicts FLAT ([name] -> value) as staging;
        # _build_part_meta() (end of _setup_scene) rebinds every one of them to the nested
        # [name][part] form in place. Nothing outside _setup_scene ever sees the flat form.
        self.object_point_clouds: dict[str, torch.Tensor] = {}
        self.object_top_point_clouds: dict[str, torch.Tensor] = {}
        self.object_top_point_cloud_paths: dict[str, str] = {}
        self.object_handle_point_clouds: dict[str, torch.Tensor] = {}
        self.object_handle_point_cloud_paths: dict[str, str] = {}
        self.object_half_point_clouds: dict[str, torch.Tensor] = {}
        self.object_half_point_cloud_paths: dict[str, str] = {}
        self._half_pcd_categories: set[str] = self._load_half_pcd_categories()
        self.object_init_poses: dict[str, torch.Tensor] = {}
        self.object_init_heights: dict[str, float] = {}
        self.object_axis_annotations: dict[str, dict[str, str]] = {}
        self._object_default_local_pos: dict[str, tuple[float, float, float]] = {}
        self._object_default_local_quat: dict[str, tuple[float, float, float, float]] = {}
        # part skeleton / joint topology / opening dirs (built directly nested):
        self.object_part_names: dict[str, list[str]] = {}
        # dof_idx/movable_link_idx stay None until lazily resolved post sim-init (_ensure_part_indices).
        self.object_joints: dict[str, dict[str, dict]] = {}
        # cano-frame opening direction per part (place regime source; kind rules in part_meta).
        self.object_opening_dirs: dict[str, dict[str, tuple[float, float, float]]] = {}
        # per-part link_forward point sets (movable parts only; from the scene JSON drawers list).
        self.object_link_forward_point_clouds: dict[str, dict[str, torch.Tensor]] = {}
        # per-part Stage-A init-pose libraries (numpy arrays, q0 frame; sparse — only parts whose
        # scene JSON drawers entry provides them):
        # [name][part]["handle_grasps"|"closepush_init"] = {pos(N,3), quat(N,4) wxyz, finger(N,), open(N,)}
        self.object_part_init_poses: dict[str, dict[str, dict]] = {}
        # scene_desc/asset-json fields materialized at build (previously re-read from JSON per call).
        self.object_categories: dict[str, dict[str, str]] = {}
        self.object_handle_sides: dict[str, dict[str, str]] = {}
        self.object_axis_lengths: dict[str, dict[str, tuple[float, float, float]]] = {}
        objects_cfg = self.scene_desc["objects"] if "objects" in self.scene_desc else self.scene_desc
        # for obj_name, obj_cfg in self.scene_desc.items():
        for obj_name, obj_cfg in objects_cfg.items():
            name = obj_name
            if obj_cfg.get("handle_side") is None:
                mesh_path = obj_cfg.get("mesh")
                if isinstance(mesh_path, str) and mesh_path:
                    asset_dir = os.path.dirname(self._resolve_scene_asset_path(mesh_path))
                    asset_stem = os.path.splitext(os.path.basename(mesh_path))[0]
                    asset_json = os.path.join(asset_dir, f"{asset_stem}.json")
                    if os.path.isfile(asset_json):
                        try:
                            with open(asset_json, "r") as _f:
                                obj_cfg["handle_side"] = json.load(_f).get("handle_side")
                        except Exception:
                            obj_cfg["handle_side"] = None
            # usd_path = obj_cfg["colored_usd_path"]
            usd_path = self._resolve_scene_asset_path(obj_cfg["final_usd_path"])

            prim_path = f"/World/envs/env_0/Objects/{name}/base_link"
            self.object_prims[name] = prim_path

            # orig_pos = tuple(obj_cfg["centerized_pos"])
            if self._layout_enabled and name in self._layout_bank:
                _pose = self._layout_bank[name][0]  # placeholder; reset will overwrite via layout bank
                orig_pos = (float(_pose[0]), float(_pose[1]), float(_pose[2]))
                rot = (float(_pose[3]), float(_pose[4]), float(_pose[5]), float(_pose[6]))
                orig_height = float(_pose[2])
            else:
                orig_pos = tuple(obj_cfg["rotated_pos"])
                rot = tuple(obj_cfg["rotated_rot"])
                orig_height = obj_cfg["rotated_pos"][2]
            self._object_default_local_pos[name] = (float(orig_pos[0]), float(orig_pos[1]), float(orig_pos[2]))
            self._object_default_local_quat[name] = (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
            pos = self._compute_world_init_pos(orig_pos)
            # pos = (pos[0] * 1.1, pos[1] * 1.5, pos[2])  # spread out a bit

            # `"fixed": true` in the scene json marks a static fixture (e.g. the
            # sink + drying basket in the simpler overlay scene): spawn kinematic
            # so gravity/contacts can't move it. Absent -> dynamic (unchanged).
            _obj_fixed = bool(obj_cfg.get("fixed", False))
            spawn_kwargs = {
                "usd_path": usd_path,
                # "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=_obj_fixed,
                    disable_gravity=_obj_fixed,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                ),
            }
            if not self.cfg.use_usd_mass:
                spawn_kwargs["mass_props"] = sim_utils.MassPropertiesCfg(density=self.cfg.object_density)

            if obj_cfg.get("articulated"):
                # Articulated object (multi-DOF cabinet). final_usd_path points at the joint USD.
                # Mirror IsaacLabEnvs' manipulated-cabinet spawn: enable articulation root, passive
                # drawer joints (actuators={}, moved by gripper contact). See docs/arti/scene_import.md.
                spawn_kwargs["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=True
                )
                arti_cfg = ArticulationCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.UsdFileCfg(**spawn_kwargs),
                    init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=rot),
                    actuators={},
                )
                obj_handler = Articulation(arti_cfg)
                self.articulated_objects[name] = obj_handler
                self.objects[name] = obj_handler  # dual-store for root-pose / pcd / object-list code
                # Per-part obs point sets (handle/link_forward, q0 frame) are loaded from the
                # scene JSON `drawers` list in _build_part_meta (see docs/arti/obs_and_task.md §4).
            else:
                rigid_cfg = RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.UsdFileCfg(**spawn_kwargs),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=pos,
                        # rot=tuple(obj_cfg["rot"]),
                        rot=rot
                    ),
                )
                obj_handler = RigidObject(rigid_cfg)
                self.objects[name] = obj_handler

            # load point cloud
            point_cloud_path = self._resolve_scene_asset_path(obj_cfg["point_cloud"])
            obj_pcd = torch.tensor(np.load(point_cloud_path, allow_pickle=True)["arr_0"], device=self.device, dtype=torch.float32)
            self.object_point_clouds[name] = obj_pcd
            top_pcd_path_str = str(point_cloud_path)
            if top_pcd_path_str.endswith("_pcd.npz"):
                top_pcd_path_str = top_pcd_path_str[: -len("_pcd.npz")] + "_top_r15_pcd.npz"
                self.object_top_point_cloud_paths[name] = top_pcd_path_str
                if os.path.isfile(top_pcd_path_str):
                    obj_top_pcd = torch.tensor(
                        np.load(top_pcd_path_str, allow_pickle=True)["arr_0"],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    self.object_top_point_clouds[name] = obj_top_pcd
            if obj_cfg.get("handle_side") is not None:
                handle_pcd_path = self._resolve_handle_point_cloud_path(
                    os.path.dirname(point_cloud_path),
                    obj_cfg["handle_side"],
                )
                self.object_handle_point_cloud_paths[name] = handle_pcd_path
                obj_handle_pcd = torch.tensor(
                    np.load(handle_pcd_path, allow_pickle=True)["arr_0"],
                    device=self.device,
                    dtype=torch.float32,
                )
                self.object_handle_point_clouds[name] = obj_handle_pcd
            half_pcd_candidates = _category_candidates(name, obj_cfg.get("category"))
            if any(candidate in self._half_pcd_categories for candidate in half_pcd_candidates):
                half_pcd_path = str(point_cloud_path)
                if not half_pcd_path.endswith("_pcd.npz"):
                    raise RuntimeError(
                        f"Cannot derive half pcd path: point_cloud entry "
                        f"{half_pcd_path!r} does not end with '_pcd.npz'."
                    )
                half_pcd_path = half_pcd_path[: -len("_pcd.npz")] + "_half_pcd.npz"
                if not os.path.isfile(half_pcd_path):
                    raise RuntimeError(
                        f"cfg.use_half_pcd=True and one of "
                        f"{half_pcd_candidates!r} is listed in "
                        f"{self.cfg.half_pcd_list!r}, but the expected half "
                        f"pcd file does not exist: {half_pcd_path}"
                    )
                self.object_half_point_cloud_paths[name] = half_pcd_path
                obj_half_pcd = torch.tensor(
                    np.load(half_pcd_path, allow_pickle=True)["arr_0"],
                    device=self.device,
                    dtype=torch.float32,
                )
                self.object_half_point_clouds[name] = obj_half_pcd
            # store init pose
            init_pos = torch.tensor(pos, device=self.device, dtype=torch.float32)  # shape: (3,)
            init_rot = torch.tensor(rot, device=self.device, dtype=torch.float32)  # shape: (4,)
            self.object_init_poses[name] = torch.cat((init_pos, init_rot), dim=-1)
            self.object_init_heights[name] = orig_height

            # add semantic labels
            obj_prim = prim_utils.get_prim_at_path(prim_path)
            sem_utils.add_update_semantics(obj_prim, semantic_label=name)

            # add primary axes annotation
            self.object_axis_annotations[name] = {
                "x": obj_cfg.get("x_axis", ""),
                "y": obj_cfg.get("y_axis", ""),
                "z": obj_cfg.get("z_axis", ""),
            }

        # one-line summary of the half-pcd swap (visible when cfg.use_half_pcd=True)
        if getattr(self.cfg, "use_half_pcd", False):
            print(
                f"[base_env] use_half_pcd=True, whitelist categories="
                f"{sorted(self._half_pcd_categories)}, loaded half pcds for "
                f"{len(self.object_half_point_clouds)} objects: "
                f"{sorted(self.object_half_point_clouds.keys())}",
                flush=True,
            )

        # append table to objects
        if self.table is not None:
            self.objects["table"] = self.table
        # table size
        self.table_size = np.asarray(self._scene_table_size, dtype=np.float32)
        self._update_table_3d_range()
        if self.table is not None and self.table_prim is not None:
            # add semantic labels for table
            table_prim = prim_utils.get_prim_at_path(self.table_prim)
            sem_utils.add_update_semantics(table_prim, semantic_label="table")
            # table axis annotations
            self.object_axis_annotations["table"] = {
                "x": "Pointing from back to front, parallel to the shorter side.",
                "y": "Pointing from left to right, parallel to the longer side.",
                "z": "Pointing upwards from bottom to top, perpendicular to the table surface.",
            }

        # unified [name][part] meta store (reads the flat dicts filled above; see __init__ block)
        self._build_part_meta()

        # register to the scene
        self.scene.articulations["robot"] = self.robot
        if self.table is not None:
            self.scene.rigid_objects["table"] = self.table
        if self.floor is not None:
            self.scene.rigid_objects["floor"] = self.floor
        for wall_prim_id, wall_obj in self._walls.items():
            self.scene.rigid_objects[wall_prim_id] = wall_obj
        for name, obj in self.objects.items():
            # Articulations must register under scene.articulations (not rigid_objects), even
            # though they are dual-stored in self.objects. See docs/arti/scene_import.md.
            if name in self.articulated_objects:
                self.scene.articulations[name] = obj
            else:
                self.scene.rigid_objects[name] = obj
        self.scene.sensors["contact"] = self.contact

        if use_camera:
            for view_name in self._front_camera_view_names:
                camera = self._camera_handles_by_view.get(view_name)
                if camera is not None:
                    self.scene.sensors[f"{view_name}_camera"] = camera
            if getattr(self, "top_camera", None) is not None:
                self.scene.sensors["top_camera"] = self.top_camera
            if getattr(self, "wrist_camera", None) is not None:
                self.scene.sensors["wrist_camera"] = self.wrist_camera
        self.controller = DifferentialIKController(self.cfg.controller, self.num_envs, self.device)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._randomize_dome_light()

    def _asset_json_field(self, obj_cfg, field):
        """Read `field` from the asset's own <asset_dir>/<asset_stem>.json (same convention
        as the in-loop handle_side backfill). None when unavailable."""
        mesh_path = obj_cfg.get("mesh")
        if not (isinstance(mesh_path, str) and mesh_path):
            return None
        asset_dir = os.path.dirname(self._resolve_scene_asset_path(mesh_path))
        asset_stem = os.path.splitext(os.path.basename(mesh_path))[0]
        asset_json = os.path.join(asset_dir, f"{asset_stem}.json")
        if not os.path.isfile(asset_json):
            return None
        try:
            with open(asset_json, "r") as f:
                return json.load(f).get(field)
        except Exception:
            return None

    def _build_part_meta(self):
        """Finalize the unified [name][part] sparse meta framework (declared in __init__).

        Runs at the end of _setup_scene, after the object loop staged the flat [name] -> value
        dicts and obj_cfg backfills / articulation membership are complete. Builds the
        part-native dicts (part_names/joints/opening_dirs/part pcds/obs point sets/D-fields)
        and then REBINDS every staged flat dict to its nested [name][part] form in place —
        after this returns, every meta dict on the env is [name][part]. Part pcds are OFFLINE
        artifacts (scripts/gen_arti_part_pcds.py --include-base) — missing files are a hard
        error, no build-time sampling fallback."""
        objects_cfg = self.scene_desc["objects"] if "objects" in self.scene_desc else self.scene_desc
        pcds_by_part: dict[str, dict] = {}
        handles_by_part: dict[str, dict] = {}
        handle_paths_by_part: dict[str, dict] = {}
        axis_ann_parts: dict[str, dict] = {}
        for name, obj_cfg in objects_cfg.items():
            ## Part skeleton: "base" + one key per URDF movable joint (arti only).
            parts = []
            urdf_path = None
            base_link = None
            if name in self.articulated_objects:
                urdf_path = self._resolve_scene_asset_path(obj_cfg["urdf_path"])
                base_link, parts = parse_urdf_parts(urdf_path)
            self.object_part_names[name] = ["base"] + [p["part_key"] for p in parts]

            ## Cano-frame opening dirs. Base: scene JSON / asset json override, else +x when a
            ## door part exists (cavity behind the door on the front face), else +z.
            base_opening = obj_cfg.get("opening_dir") or self._asset_json_field(obj_cfg, "opening_dir")
            if base_opening is None:
                has_door = any(p["kind"] == "door" for p in parts)
                base_opening = (1.0, 0.0, 0.0) if has_door else OPENING_DIR_CANO_DEFAULT
            opening = {"base": tuple(float(v) for v in base_opening)}
            for p in parts:
                opening[p["part_key"]] = OPENING_DIR_CANO_BY_KIND[p["kind"]]
            self.object_opening_dirs[name] = opening

            ## Joint topology (movable parts only). dof/link indices need the spawned
            ## articulation's joint/body names -> resolved lazily after sim init.
            if parts:
                self.object_joints[name] = {
                    p["part_key"]: {
                        "joint": p["joint"],
                        "child": p["child"],
                        "joint_type": p["joint_type"],
                        "axis_obj": p["axis_obj"],
                        "origin_obj": p["origin_obj"],
                        "dof_idx": None,
                        "movable_link_idx": None,
                    }
                    for p in parts
                }

            ## Part pcds — all from offline npz for arti (required, no fallback). Rigid "base"
            ## = the whole object = the object pcd; arti "base" = the base LINK's own npz (static
            ## shell only, so an open drawer/door never balloons the base AABB).
            def _load_part_npz(part_key, child_link):
                npz_path = part_pcd_npz_path(urdf_path, child_link)
                if not os.path.isfile(npz_path):
                    raise FileNotFoundError(
                        f"{name!r} part {part_key!r}: missing offline part pcd {npz_path}; "
                        f"generate it with scripts/gen_arti_part_pcds.py --include-base"
                    )
                pts = np.load(npz_path, allow_pickle=True)["points"]
                return torch.tensor(pts, device=self.device, dtype=torch.float32)

            pcd_by_part = {}
            if parts:
                pcd_by_part["base"] = _load_part_npz("base", base_link)
            elif name in self.object_point_clouds:
                pcd_by_part["base"] = self.object_point_clouds[name]
            for p in parts:
                pcd_by_part[p["part_key"]] = _load_part_npz(p["part_key"], p["child"])
            if pcd_by_part:
                pcds_by_part[name] = pcd_by_part

            ## Per-part obs point sets from the scene JSON `drawers` list (list order == URDF
            ## movable order; docs/arti/obs_and_task.md §4). Rigid handle pcd joins as "base".
            drawer_specs = obj_cfg.get("drawers", [])
            if drawer_specs and len(drawer_specs) != len(parts):
                raise ValueError(
                    f"{name!r}: scene JSON drawers list has {len(drawer_specs)} entries but the "
                    f"URDF has {len(parts)} movable parts; cannot key obs point sets by part."
                )
            handle_by_part = {}
            lf_by_part = {}
            init_poses_by_part = {}
            for p, spec in zip(parts, drawer_specs):
                handle_by_part[p["part_key"]] = torch.tensor(
                    np.load(self._resolve_scene_asset_path(spec["handle_points"]), allow_pickle=True)["points"],
                    device=self.device, dtype=torch.float32,
                )
                if spec.get("link_forward_points"):
                    lf_by_part[p["part_key"]] = torch.tensor(
                        np.load(self._resolve_scene_asset_path(spec["link_forward_points"]), allow_pickle=True)["points"],
                        device=self.device, dtype=torch.float32,
                    )
                # Per-part axis annotations: kind template, scene JSON spec fields override.
                tmpl = AXIS_ANNOTATION_BY_KIND[p["kind"]]
                axis_ann_parts.setdefault(name, {})[p["part_key"]] = {
                    ax: (spec.get(f"{ax}_axis") or tmpl[ax]) for ax in ("x", "y", "z")
                }
                # Stage-A init-pose libraries (sampled by sample_part_init_pose; numpy, q0 frame).
                libs = {}
                for lib_key in ("handle_grasps", "closepush_init"):
                    if spec.get(lib_key):
                        z = np.load(self._resolve_scene_asset_path(spec[lib_key]), allow_pickle=True)
                        libs[lib_key] = {k: np.asarray(z[k]) for k in ("pos", "quat", "finger", "open")}
                if libs:
                    init_poses_by_part[p["part_key"]] = libs
            if init_poses_by_part:
                self.object_part_init_poses[name] = init_poses_by_part
            if name in self.object_handle_point_clouds:
                handle_by_part["base"] = self.object_handle_point_clouds[name]
                handle_paths_by_part[name] = {"base": self.object_handle_point_cloud_paths[name]}
            if handle_by_part:
                handles_by_part[name] = handle_by_part
            if lf_by_part:
                self.object_link_forward_point_clouds[name] = lf_by_part

            ## scene_desc/asset-json fields materialized (base-only, sparse). handle_side reads
            ## obj_cfg AFTER the in-loop asset-json backfill wrote it back.
            if obj_cfg.get("category") is not None:
                self.object_categories[name] = {"base": obj_cfg["category"]}
            if obj_cfg.get("handle_side") is not None:
                self.object_handle_sides[name] = {"base": obj_cfg["handle_side"]}
            if all(k in obj_cfg for k in ("axis_length_x", "axis_length_y", "axis_length_z")):
                self.object_axis_lengths[name] = {"base": (
                    float(obj_cfg["axis_length_x"]),
                    float(obj_cfg["axis_length_y"]),
                    float(obj_cfg["axis_length_z"]),
                )}

        ## The table is a place target too (surface, opening up) — give it an opening dir so
        ## the place regime judge can query every real target uniformly.
        self.object_opening_dirs["table"] = {"base": (0.0, 0.0, 1.0)}

        ## Rebind: staged flat dicts -> nested [name][part], same attribute names (the flat
        ## form never escapes _setup_scene). Iterating the flat dicts (not objects_cfg) also
        ## covers non-object keys like "table" (axis annotations). Runtime writers
        ## (_reset_idx init-pose refresh) write the nested entries directly.
        self.object_point_clouds = pcds_by_part
        self.object_handle_point_clouds = handles_by_part
        self.object_handle_point_cloud_paths = handle_paths_by_part
        for attr in (
            "object_top_point_clouds",
            "object_top_point_cloud_paths",
            "object_half_point_clouds",
            "object_half_point_cloud_paths",
            "object_init_poses",
            "object_init_heights",
            "object_axis_annotations",
            "_object_default_local_pos",
            "_object_default_local_quat",
        ):
            setattr(self, attr, {key: {"base": val} for key, val in getattr(self, attr).items()})
        ## Per-part axis annotations join the (already-rebound) nested dict.
        for name, parts_ann in axis_ann_parts.items():
            self.object_axis_annotations.setdefault(name, {}).update(parts_ann)

    ############################################################
    # pre-physics step calls
    ############################################################
    def set_control_params(self, mode: int, is_release: bool = False, disable_action_ema: bool = False):
        """
        mode=0: execute actions from motion planning(joint q pos)
        mode=1: execute actions from neural polices(EE pose) TODO: abs or delta???
        """
        self.action_mode = mode
        self.is_release_action = bool(is_release)
        self.disable_action_ema = bool(disable_action_ema)

    def _pre_physics_step(self, actions):
        """
        action_mode=0: execute actions from motion planning(joint q pos)
        action_mode=1: execute actions from neural polices(EE pose) TODO: abs or delta???
        """
        # Normalize action dtype/device to torch tensor.
        actions = to_tensor(actions, self.device).to(torch.float32)
        # print(f"[_pre_physics_step] Pre-physics step ({self.common_step_counter})")
        if self.action_mode == 0:
            # motion planner outputs joint position targets (NOT normalized).
            # Format:
            #   actions: (N, dim_arm + 1 or dim_arm + 2) -> arm + gripper
            q_in = actions  # do not clamp here; planner outputs real q
            if q_in.ndim == 1:
                q_in = q_in.unsqueeze(0).expand(self.num_envs, -1)  # (N, dim_arm + 2)

            # if q_in[0, -1] == 0:
            #     import ipdb; ipdb.set_trace()

            # 1) parse planner output
            ## TODO: We need to figure out the order of joints in isaaclab
            assert q_in.shape[-1] == len(self.arm_joint_indices) + 2, f"q_in dim={q_in.shape[-1]}, arm_joint dim={len(self.arm_joint_indices)}"
            arm_targets = q_in[:, :len(self.arm_joint_indices)]
            gripper_targets = q_in[:, -2:].to(torch.float32)  # (N, 2)
            # gripper_targets = gripper_targets * self.max_gripper_opening

            full_targets = torch.concat((arm_targets, gripper_targets), dim=-1)  # (N, dim_arm + 2)

            if self.is_release_action and bool(getattr(self.cfg, "release_teleop_joint_state", False)):
                # Legacy escape hatch: teleport joints during release. Teleporting can place a
                # finger interpenetrating the just-released object between rendered frames, and
                # PhysX resolves that penetration impulsively (object ejected). Default path is
                # pure PD tracking via _apply_action -- bounded contact forces, gentle nudges.
                self.robot.write_joint_state_to_sim(full_targets, torch.zeros_like(full_targets))
            self.robot_dof_targets[:] = full_targets  # do not clamp here; planner outputs real q
        elif self.action_mode == 1:
            if actions.ndim == 1:
                actions = actions.unsqueeze(0).expand(self.num_envs, -1)  # (N, 7)
            raw_actions = actions.clone().clamp(-1.0, 1.0)
            alpha = 0.0 if bool(getattr(self, "disable_action_ema", False)) else self.cfg.action_ema_alpha
            if alpha is not None and alpha > 0.0:
                self.actions = alpha * self.prev_actions + (1.0 - alpha) * raw_actions
            else:
                self.actions = raw_actions
            self.prev_actions = self.actions.clone()
            arm_action, gripper_action = self.actions[:, :6], self.actions[:, 6:]
            # Scale
            arm_action *= self.cmd_limit
            gripper_action *= 0.01
            # We use DifferentialIKController to convert operational-space commands to joint-space commands.
            self.controller.set_command(
                arm_action,
                self.robot.data.body_pos_w[:, self.hand_link_idx],
                self.robot.data.body_quat_w[:, self.hand_link_idx],
            )
            jacobian = self.robot.root_physx_view.get_jacobians()[:, self.hand_link_idx - 1, :, self.arm_joint_indices]
            arm_targets = self.controller.compute(
                self.robot.data.body_pos_w[:, self.hand_link_idx],
                self.robot.data.body_quat_w[:, self.hand_link_idx],
                jacobian,
                self.robot.data.joint_pos[:, :-2],
            )
            gripper_targets = self.robot.data.joint_pos[:, -2:] + gripper_action
            # Set robot DoF targets.
            self.robot_dof_targets[:] = torch.clamp(
                torch.cat((arm_targets, gripper_targets), dim=-1),
                self.robot_dof_lower_limits,
                self.robot_dof_upper_limits,
            )
        else:
            raise NotImplementedError()


    def _apply_action(self):
        # is_release teleports joints in _pre_physics_step, but the PD drive target must
        # still track the same q: otherwise the stale (closed-gripper) target from before
        # the release pulls the fingers back over the decimation substeps.
        self.robot.set_joint_position_target(self.robot_dof_targets)

    ############################################################
    # post-physics step calls
    ############################################################

    def enable_env_reset(self):
        self.env_reset_flag = True

    def disable_env_reset(self):
        self.env_reset_flag = False
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if self.env_reset_flag:  # env level reset
            super()._reset_idx(env_ids)
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            self.prev_actions[env_ids] = 0.0

            # Global dome light randomization per reset call.
            self._randomize_dome_light()

            ## Clean buffered info
            self.task_name = None
            self.task = None
            self.object = None
            self.current_object_name = None
            self.data_point_cloud = None
            self.data_top_point_cloud = None
            self.data_point_cloud_obs_reward = None
            self.reset_buf[env_ids] = 0
            self.episode_length_buf[env_ids] = 0

            use_saved_reset_params = bool(self._use_next_env_reset_params)
            if use_saved_reset_params:
                if self._next_env_reset_params is None:
                    raise RuntimeError("use_next_env_reset_params is enabled but no reset params are set.")
                self._apply_env_reset_params(env_ids, self._next_env_reset_params)
                self.is_release_action = False
                self._consume_env_reset_params_if_needed()
                return

            # Reset robot joint positions for selected envs.
            joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
            sampled_arm_joint_pos = self._sample_init_arm_joint_pos(joint_pos.shape[0])
            if sampled_arm_joint_pos is not None:
                joint_pos[:, self.arm_joint_indices] = sampled_arm_joint_pos
            sampled_gripper_joint_pos = self._sample_reset_gripper_joint_pos(joint_pos.shape[0])
            joint_pos[:, self.left_joint_idx] = sampled_gripper_joint_pos[:, 0]
            joint_pos[:, self.right_joint_idx] = sampled_gripper_joint_pos[:, 1]
            joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
            joint_vel = torch.zeros_like(joint_pos)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            self.robot.set_joint_position_target(joint_pos)
            self.robot_dof_targets[env_ids] = joint_pos
            self.is_release_action = False

            # Reset all objects to their initial poses/velocities for the selected envs.
            env_origins = self.scene.env_origins[env_ids]
            apply_xy_randomize = (not self._layout_enabled) and bool(self.cfg.randomize_xy)
            xy_range = getattr(self.cfg, "randomize_xy_range", 0.0) if apply_xy_randomize else 0.0
            self._table_enabled_this_episode = self._sample_table_enabled()

            if self._layout_enabled:
                self._current_layout_idx = self._sample_layout_idx()

            forced_floor_z = None
            if not self.is_table_active():
                if self._layout_enabled:
                    forced_floor_z = float(self._layout_table_z_bank[self._current_layout_idx])
                else:
                    lo = self._table_z_range[0]
                    hi = self._max_table_surface_height if self._max_table_surface_height is not None else self._table_z_range[1]
                    with timestamp_seed_scope():
                        forced_floor_z = float(np.random.uniform(lo, hi))
                print(f"[reset] no-table  forced_floor_z={forced_floor_z:.4f}  table_active={self.is_table_active()}")
            if self._scene_floor_walls_enabled:
                self._reset_floor_and_walls(env_ids, force_floor_z=forced_floor_z)
            elif forced_floor_z is not None:
                self._current_floor_surface_z = forced_floor_z

            # Move ground plane so its top surface sits just below the active support surface.
            if not self.is_table_active() and forced_floor_z is not None:
                # No-table: top surface just below the visual floor.
                self._set_ground_plane_z(env_ids, forced_floor_z - self._no_table_floor_above_ground)
            elif self._scene_floor_walls_enabled:
                # Has-table with visual floor: follow the actual floor surface of this episode.
                self._set_ground_plane_z(env_ids, self._current_floor_surface_z - self._no_table_floor_above_ground)
            else:
                # Has-table, no visual floor: fixed safety-net well below the table.
                self._set_ground_plane_z(env_ids, self._floor_z_range[0] - self._ground_plane_z_offset)

            # Reset table pose with sampled z and fixed x/y from cfg.
            if self.table is not None:
                if self.is_table_active():
                    sampled_table_z = self._clip_table_center_z(
                        self._sample_table_z(layout_idx=self._current_layout_idx if self._layout_enabled else None)
                    )
                    self._set_table_pose(env_ids, env_origins, sampled_table_z)
                else:
                    self._set_table_pose(env_ids, env_origins, self._ground_plane_z - 10.0)

            # Front-camera augmentation is applied only on env-level reset.
            # Must be after table z sampling so center.z tracks current table top.
            self._randomize_front_camera_pose(
                env_ids,
                layout_idx=self._current_layout_idx if self._layout_enabled else None,
            )
            # Wrist-camera mount DR is now session-level (sampled once at env
            # construction, baked into cfg.wrist_camera.offset before
            # super().__init__). No runtime per-reset prim writes here -- the
            # previous _randomize_wrist_camera_pose path had a known
            # rendering bug (writes didn't propagate to TiledCamera output).

            for name, obj in self.objects.items():
                if name == "table":
                    continue
                if self._layout_enabled:
                    sampled_pose = self._layout_bank[name][self._current_layout_idx]
                    sampled_pos = sampled_pose[:3]
                    sampled_quat = sampled_pose[3:7]
                    init_pos = torch.tensor(
                        self._compute_world_init_pos(sampled_pos),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    init_quat = torch.tensor(sampled_quat, device=self.device, dtype=torch.float32)
                    # Keep latest sampled layout as the "init pose" exposed to downstream modules.
                    self.object_init_poses[name] = {"base": torch.cat((init_pos, init_quat), dim=-1)}
                    self.object_init_heights[name] = {"base": float(sampled_pos[2])}
                else:
                    sampled_pos = self._object_default_local_pos[name]["base"]
                    sampled_quat = self._object_default_local_quat[name]["base"]
                    init_pos = torch.tensor(
                        self._compute_world_init_pos(sampled_pos),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    init_quat = torch.tensor(sampled_quat, device=self.device, dtype=torch.float32)
                    self.object_init_poses[name] = {"base": torch.cat((init_pos, init_quat), dim=-1)}
                    self.object_init_heights[name] = {"base": float(sampled_pos[2])}

                pos = init_pos.unsqueeze(0) + env_origins
                if xy_range > 0:
                    with timestamp_seed_scope():
                        uni = torch.distributions.Uniform(0.0, 1.0)
                        xy_noise = (uni.sample((pos.shape[0], 2)).to(self.device) * 2.0 - 1.0) * xy_range
                    pos[:, :2] += xy_noise
                quat = init_quat.unsqueeze(0).expand(pos.shape[0], -1)
                root_pose = torch.cat((pos, quat), dim=-1)
                root_vel = torch.zeros((root_pose.shape[0], 6), device=self.device)

                obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)
                obj.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

                # Articulation joint init state comes from the layout npz (env reset, not task
                # reset): per-layout row cols [7:7+J] = normalized opening per joint (URDF dof
                # order). Missing joint cols (or no layout) -> all joints default to 0 (closed).
                # See docs/arti/scene_import.md.
                if name in self.articulated_objects:
                    art = obj
                    J = int(art.data.joint_pos.shape[-1])
                    lower = art.data.joint_pos_limits[env_ids, :, 0]
                    upper = art.data.joint_pos_limits[env_ids, :, 1]
                    joint_norm = torch.zeros((len(env_ids), J), device=self.device, dtype=torch.float32)
                    if self._layout_enabled:
                        row = self._layout_bank[name][self._current_layout_idx]
                        if row.shape[0] > 7:
                            extra = row[7:]
                            if extra.shape[0] != J:
                                raise ValueError(
                                    f"Layout for articulated object '{name}' has {extra.shape[0]} joint "
                                    f"cols but the asset has {J} joints (expected shape (N, 7+{J}))."
                                )
                            joint_norm = torch.tensor(extra, device=self.device, dtype=torch.float32).unsqueeze(0).expand(len(env_ids), -1)
                    joint_pos = lower + joint_norm * (upper - lower)
                    art.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

            self._consume_env_reset_params_if_needed()
                
        else:  # task level reset
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            self.prev_actions[env_ids] = 0.0
            self.task.reset(env=self,  env_ids=env_ids)

    ############################################################
    # others
    ############################################################

    def set_task(self, task_name: str, **kwargs):
        # set task
        assert task_name in self.tasks, f"Task {task_name} not found in environment tasks."
        self.task_name = task_name
        self.task = self.tasks[task_name]

        # object_name = kwargs.get("object_name")
        # if object_name is not None:
        #     self.set_object(object_name)

        # set observation space for the selected agent/task and refresh gym spaces if needed
        # assert task_name in self.observasion_spaces, f"Observation space for task {task_name} not found."
        # new_obs_space = self.observasion_spaces[task_name]
        # if self.cfg.observation_space != new_obs_space:
        #     self.cfg.observation_space = new_obs_space
        #     # rebuild spaces so downstream agents see the updated shape
        #     self._configure_gym_env_spaces()
        self.set_task_obs_space(task_name)

        self.task.prepare(self, **kwargs)

    def set_task_obs_space(self, task_name: str):
        """Update observation space to match the task without switching tasks."""
        assert task_name in self.observasion_spaces, f"Observation space for task {task_name} not found."
        new_obs_space = self.observasion_spaces[task_name]
        if self.cfg.observation_space != new_obs_space:
            self.cfg.observation_space = new_obs_space
            self._configure_gym_env_spaces()

    def unset_task(self):
        self.task_name = None
        self.task = None

    def set_object(self, object_name: str):
        # set object
        assert object_name in self.objects, f"Object {object_name} not found in environment objects."
        self.object = self.objects[object_name]
        self.current_object_name = object_name
        self.data_point_cloud = self.object_point_clouds[object_name]["base"]
        self.data_top_point_cloud = self.object_top_point_clouds.get(object_name, {}).get("base")
        self.data_handle_point_cloud = self.object_handle_point_clouds.get(object_name, {}).get("base")
        self.data_half_point_cloud = self.object_half_point_clouds.get(object_name, {}).get("base")
        self.data_point_cloud_obs_reward = self.data_point_cloud  # default to full cloud

    def unset_object(self):
        self.object = None
        self.current_object_name = None
        self.data_point_cloud = None
        self.data_top_point_cloud = None
        self.data_handle_point_cloud = None
        self.data_half_point_cloud = None
        self.data_point_cloud_obs_reward = None

    def set_obs_pcd_mode(self, mode: str):
        """Select which point cloud feeds obs_dist/obs_closest_point in PickTask.

        Hard-fails (no fallback) when the selected cloud is missing, since each
        pick policy was trained against a specific point-cloud observation.
        """
        if mode == "top":
            if self.data_top_point_cloud is None:
                obj_name = self.current_object_name or "<unknown>"
                expected = self.object_top_point_cloud_paths.get(obj_name, {}).get("base", "<unknown>")
                raise RuntimeError(
                    f"[set_obs_pcd_mode] Cannot enable top point cloud mode for "
                    f"object='{obj_name}': data_top_point_cloud is None.\n"
                    f"  Expected file: {expected}\n"
                    f"  Top mode was requested by the caller (e.g. dispatcher resolved "
                    f"agent_name='pick_convex_lying' and rlpolicy.uses_top_pcd(...)=True). "
                    f"This is a hard error by design — no fallback to full cloud, since "
                    f"the convex-lying policy was trained against top PCD obs and would "
                    f"see a different obs distribution if fed full PCD.\n"
                    f"  Fix one of:\n"
                    f"    1) Generate the missing *_top_r15_pcd.npz for this asset.\n"
                    f"    2) Add this asset to asset_invalid_0315.txt so it is never "
                    f"sampled into scenes.\n"
                    f"    3) Drop this agent_name from rlpolicy.top_pcd_agents (only if "
                    f"you're knowingly switching the policy back to full-cloud obs)."
                )
            self.data_point_cloud_obs_reward = self.data_top_point_cloud
        elif mode == "handle":
            if self.data_handle_point_cloud is None:
                obj_name = self.current_object_name or "<unknown>"
                expected = self.object_handle_point_cloud_paths.get(obj_name, {}).get("base", "<unknown>")
                raise RuntimeError(
                    f"[set_obs_pcd_mode] Cannot enable handle point cloud mode for "
                    f"object='{obj_name}': data_handle_point_cloud is None.\n"
                    f"  Expected file: {expected}\n"
                    f"  Handle mode was requested by the caller (e.g. dispatcher resolved "
                    f"agent_name='pick_handle' and rlpolicy.uses_handle_pcd(...)=True). "
                    f"This is a hard error by design — no fallback to full cloud, since "
                    f"the handle policy was trained against handle-side PCD obs and would "
                    f"see a different obs distribution if fed full PCD."
                )
            self.data_point_cloud_obs_reward = self.data_handle_point_cloud
        elif mode == "half":
            if self.data_half_point_cloud is None:
                obj_name = self.current_object_name or "<unknown>"
                expected = self.object_half_point_cloud_paths.get(obj_name, {}).get("base", "<unknown>")
                raise RuntimeError(
                    f"[set_obs_pcd_mode] Cannot enable half point cloud mode for "
                    f"object='{obj_name}': data_half_point_cloud is None.\n"
                    f"  Expected file: {expected}\n"
                    f"  Half mode was requested because the object's category is in "
                    f"`half_pcd_list`, but no half pcd was loaded for this object. "
                    f"Possible causes:\n"
                    f"    1) cfg.use_half_pcd is False (the global switch is off).\n"
                    f"    2) Neither the object's category nor its numeric-suffix-stripped name "
                    f"is listed in {self.cfg.half_pcd_list!r}.\n"
                    f"    3) The expected file {expected} does not exist on disk."
                )
            self.data_point_cloud_obs_reward = self.data_half_point_cloud
        elif mode == "full":
            self.data_point_cloud_obs_reward = self.data_point_cloud
        else:
            raise ValueError(
                f"Unknown obs pcd mode: {mode!r}; expected 'full', 'top', 'handle', or 'half'."
            )
    
    def _get_observations(self) -> dict:
        if not self.task:
            return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}
        else:
            return self.task.get_observations(env=self)

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.num_envs,), device=self.device)

    def _get_dones(self):
        if not self.task:
            terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            truncated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        else:
            terminated, truncated = self.task.get_dones(env=self)
        return terminated, truncated
