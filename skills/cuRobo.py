import os
import json
import copy
import numpy as np
import torch
import logging
import traceback
import argparse
import trimesh
from scipy.spatial.transform import Rotation as R

# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig, JointState
from curobo.types.math import Pose
from curobo.util_file import get_robot_path, join_path, load_yaml, get_robot_configs_path
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig, Mesh, Sphere
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

from utils.image_utils import _decode_base64
from utils.geometry_utils import proj_3d_to_2d, proj_3d_to_2d_batched, rigid_transform, rigid_transform_batched, T2qt, qt2T, q2R_wxyz, R2q_wxyz

logging.getLogger("curobo").setLevel(logging.WARNING)

"""
and then joint 4 is also off by 4 degrees
So when you are passing the start state to CuRobo do this:
q[3] += 4 * np.pi / 180.0
And then when you carry out the actions in RLBench do this:
action[3] -= 4 * np.pi / 180.0
"""


class cuRoboServer:
    def __init__(self, simulator, args) -> None:
        logging.info("Initializing cuRobo...")
        self.simulator = simulator
        self.robot_config = simulator.robot_config
        self.grasped_object_names = []

        ## scene pose
        self.T = simulator.T

        ## gripper
        self.gripper_name = "franka_panda"

        self.ctrl_pts = np.array([
            [0.05268743, 0., 0.105273141],
            [0.05268743, 0., 0.05268743],
            [0., 0., 0.05268743],
            [0., 0., 0.],
            [0., 0., 0.05268743],
            [-0.05268743, 0., 0.05268743],
            [-0.05268743, 0., 0.105273141]
        ])

        self.finger_pts = self.ctrl_pts[[0, -1]]

        ## camera
        # self.ixt = None
        # self.ext = None
        # self.h = None
        # self.w = None

        ## grasp select
        self.grasp_select_mode = args["grasp_select_mode"]
        # Assets dropped from the cuRobo collision world this run (clutter blocking a recessed grasp).
        self.ignore_collision_assets = set(args.get("ignore_collision_assets") or [])
        # Default behavior for empty gripper move: keep wrist extension (panda_hand +Z) pointing down.
        self.align_empty_gripper_wrist_to_world_z = bool(args.get("align_empty_gripper_wrist_to_world_z", False))
        # Optional fallback correction: after rotating wrist in retry, compensate EE translation to keep held-object target center.
        self.preserve_object_position_on_fallback = bool(args.get("preserve_object_position_on_fallback", False))
        self.place_preserve_object_position_on_fallback = bool(
            args.get("place_preserve_object_position_on_fallback", self.preserve_object_position_on_fallback)
        )
        # Place-alignment relaxation ladder (strict first, then widen the axis-angle
        # tolerance by step up to max; the cap itself is always included). max=0 -> strict only.
        relax_max = max(0.0, float(args.get("place_align_relax_max_deg", 0.0)))
        relax_step = float(args.get("place_align_relax_step_deg", 5.0))
        if relax_step <= 0.0:
            relax_step = 5.0
        ladder = [0.0]
        phi = relax_step
        while phi < relax_max - 1e-9:
            ladder.append(phi)
            phi += relax_step
        if relax_max > 1e-9:
            ladder.append(relax_max)
        self.place_align_relax_ladder_deg = ladder
        # Stage-A arti init: how many closepush/handle-grasp samples to try before giving up.
        self.part_init_max_samples = max(1, int(args.get("part_init_max_samples", 5)))
        self.min_grasp_hold_steps = max(0, int(args.get("min_grasp_hold_steps", 4)))
        self.max_grasp_hold_steps = max(
            self.min_grasp_hold_steps,
            int(args.get("max_grasp_hold_steps", max(self.min_grasp_hold_steps, 12))),
        )
        self.grasp_hold_lin_vel_thresh = max(0.0, float(args.get("grasp_hold_lin_vel_thresh", 0.01)))
        self.grasp_hold_ang_vel_thresh = max(0.0, float(args.get("grasp_hold_ang_vel_thresh", 0.10)))
        self.grasp_hold_disable_ema = bool(args.get("grasp_hold_disable_ema", True))
        self.min_place_hold_steps = max(0, int(args.get("min_place_hold_steps", 2)))
        self.max_place_hold_steps = max(
            self.min_place_hold_steps,
            int(args.get("max_place_hold_steps", self.min_place_hold_steps)),
        )
        self.place_hold_lin_vel_thresh = max(0.0, float(args.get("place_hold_lin_vel_thresh", 0.001)))
        self.place_hold_ang_vel_thresh = max(0.0, float(args.get("place_hold_ang_vel_thresh", 0.02)))
        self.place_hold_disable_ema = bool(args.get("place_hold_disable_ema", True))
        # NOTE: release_fallback_ee_pose (canonical-home release fallback) was removed with
        # the release-ladder simplification (RFC §8.2 -> primary then open-in-place); the
        # config key is now ignored.
        self.planner_execute_mode1_replan = bool(args.get("planner_execute_mode1_replan", False))
        # Propagate replan flag to simulator so data collection knows whether to record EE cmd.
        self.simulator._data_collect_store_ee_cmd = self.planner_execute_mode1_replan
        self.planner_replan_every_k_steps = max(1, int(args.get("planner_replan_every_k_steps", 5)))
        self.planner_goal_pos_tolerance_m = max(0.0, float(args.get("planner_goal_pos_tolerance_m", 0.01)))
        self.planner_goal_rot_tolerance_rad = max(0.0, float(args.get("planner_goal_rot_tolerance_rad", 0.10)))
        self.planner_max_replans = max(0, int(args.get("planner_max_replans", 5)))
        self.planner_skip_points_after_first_plan = max(
            0, int(args.get("planner_skip_points_after_first_plan", 0))
        )
        # Handle-grasp drive: fingers locked at this per-finger opening [m] in the narrow
        # planner AND commanded at it during the drive (full-open fingertips sweep the
        # next handle down on tight cabinets; see _get_narrow_motion_gen).
        self.narrow_finger_width = float(args.get("narrow_finger_width", 0.012))

        self.lift_offset = self.simulator.lift_offset
        self.place_offset = self.simulator.place_offset
        self.place_offset_center = self.simulator.place_offset_center
        self.release_offset = self.simulator.release_offset
        self.release_retract_offset = self.simulator.release_retract_offset

        self.save_id = 0

        tensor_args = TensorDeviceType()
        self._motion_gen_robot_cfg = self._resolve_motion_gen_robot_cfg(self.robot_config)
        if isinstance(self._motion_gen_robot_cfg, RobotConfig):
            robot_cfg = self._motion_gen_robot_cfg
        elif isinstance(self._motion_gen_robot_cfg, dict):
            robot_cfg = RobotConfig.from_dict(copy.deepcopy(self._motion_gen_robot_cfg), tensor_args)
        else:
            raise TypeError(
                f"Unsupported robot config type for cuRobo: {type(self._motion_gen_robot_cfg).__name__}"
            )
        self.robot = CudaRobotModel(robot_cfg.kinematics)

        # ik_cfg = IKSolverConfig.load_from_robot_config(
        #     robot_cfg,
        #     num_seeds=64,               # 并行随机初值个数
        #     use_cuda_graph=False,       # 初次调通时先关掉，便于改 batch/seed
        # )
        # ik_cfg.tensor_args = tensor_args
        # self.ik = IKSolver(ik_cfg)

    def _resolve_motion_gen_robot_cfg(self, robot_config):
        cfg = self._resolve_motion_gen_robot_cfg_raw(robot_config)
        # _make_attach_spheres emits up to 32 spheres per held object, but the stock
        # franka.yml only reserves extra_collision_spheres.attached_object=12 -- the
        # attach call overflows that buffer. Bump the reservation at load time.
        if isinstance(cfg, dict):
            cfg = copy.deepcopy(cfg)
            kin = cfg.setdefault("kinematics", {})
            extra = kin.setdefault("extra_collision_spheres", {}) or {}
            if int(extra.get("attached_object") or 0) < 32:
                extra["attached_object"] = 32
            kin["extra_collision_spheres"] = extra
        return cfg

    def _resolve_motion_gen_robot_cfg_raw(self, robot_config):
        if isinstance(robot_config, RobotConfig):
            return robot_config
        if isinstance(robot_config, dict):
            return robot_config.get("robot_cfg", robot_config)
        if not isinstance(robot_config, str):
            raise TypeError(
                f"robot_config must be str/dict/RobotConfig, got {type(robot_config).__name__}"
            )

        candidates = [robot_config, join_path(get_robot_configs_path(), robot_config)]
        for cfg_path in candidates:
            if isinstance(cfg_path, str) and os.path.isfile(cfg_path):
                cfg_yaml = load_yaml(cfg_path)
                if isinstance(cfg_yaml, dict) and "robot_cfg" in cfg_yaml:
                    return cfg_yaml["robot_cfg"]
                if isinstance(cfg_yaml, dict):
                    return cfg_yaml
                raise ValueError(f"Robot config at {cfg_path} did not parse to a dictionary.")

        raise FileNotFoundError(
            "Could not locate cuRobo robot config. Tried: "
            + ", ".join([str(x) for x in candidates])
        )

    def _scale_gripper(self, state: float) -> float:
        max_open = getattr(self.simulator, "max_gripper_opening", None)
        if max_open is None:
            max_open = 1.0
        return state * max_open

    def _rotation_matrix_from_vectors(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-8 or b_norm < 1e-8:
            return np.eye(3, dtype=np.float64)
        a = a / a_norm
        b = b / b_norm
        c = float(np.dot(a, b))
        if c > 1.0 - 1e-6:
            return np.eye(3, dtype=np.float64)
        if c < -1.0 + 1e-6:
            ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = np.cross(a, ortho)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            x, y, z = axis
            return np.array(
                [
                    [2.0 * x * x - 1.0, 2.0 * x * y, 2.0 * x * z],
                    [2.0 * x * y, 2.0 * y * y - 1.0, 2.0 * y * z],
                    [2.0 * x * z, 2.0 * y * z, 2.0 * z * z - 1.0],
                ],
                dtype=np.float64,
            )

        v = np.cross(a, b)
        s = np.linalg.norm(v)
        vx = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))

    def _align_ee_local_axis_to_world_axis(self, current_quat: np.ndarray, ee_local_axis: np.ndarray, world_axis: np.ndarray) -> np.ndarray:
        cur_q = np.asarray(current_quat, dtype=np.float64)
        if cur_q.shape[0] != 4 or not np.isfinite(cur_q).all():
            raise ValueError(f"Invalid current_quat: {current_quat}")
        cur_R = q2R_wxyz(cur_q)
        ee_local_axis = np.asarray(ee_local_axis, dtype=np.float64)
        world_axis = np.asarray(world_axis, dtype=np.float64)
        ee_axis_world = cur_R @ ee_local_axis
        delta_R = self._rotation_matrix_from_vectors(ee_axis_world, world_axis)
        new_R = delta_R @ cur_R
        return R2q_wxyz(new_R).astype(np.float64)

    def _rotate_quat_world_z(self, quat_wxyz: np.ndarray, yaw_rad: float) -> np.ndarray:
        cur_R = q2R_wxyz(np.asarray(quat_wxyz, dtype=float))
        c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
        yaw_R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return R2q_wxyz(yaw_R @ cur_R).astype(np.float64)

    def _get_downward_ee_quat(self, current_ee_pose: list) -> np.ndarray | None:
        try:
            downward_quat = self._align_ee_local_axis_to_world_axis(
                current_quat=np.array(current_ee_pose[3:], dtype=float),
                ee_local_axis=np.asarray(getattr(self.simulator, "ee_approach_local", (0.0, 0.0, 1.0)), dtype=float),
                world_axis=np.array([0.0, 0.0, -1.0], dtype=float),
            )
        except Exception as exc:
            logging.warning("Failed to compute downward fallback quaternion: %s", exc)
            return None
        if downward_quat.shape[0] != 4 or (not np.isfinite(downward_quat).all()):
            logging.warning("Invalid downward fallback quaternion: %s", downward_quat)
            return None
        return downward_quat

    def _plan_single_with_downward_fallback(
        self,
        start_state: JointState,
        goal_pose_list: list,
        current_ee_pose: list,
        allow_fallback: bool,
        fallback_goal_override: list | None = None,
        yaw_sweep_target_obj_center: np.ndarray | list | None = None,
        allow_object_yaw_fallback: bool = False,
    ):
        goal_pose_0 = None if goal_pose_list is None else np.asarray(goal_pose_list, dtype=float).tolist()
        if goal_pose_list is None:
            return None, goal_pose_0, goal_pose_0

        primary_goal = np.asarray(goal_pose_list, dtype=float).tolist()
        result = self.motion_gen.plan_single(
            start_state,
            Pose.from_list(primary_goal),
            MotionGenPlanConfig(max_attempts=5),
        )
        if result.success:
            return result, goal_pose_0, primary_goal

        # Object-yaw fallback: keep current EE wrist orientation, sweep yaw
        # rotations around world Z while preserving held object's target
        # position. Used for pick_upright-grasped objects where the object is
        # upright; rotating the object around its yaw axis often unlocks IK.
        if (
            allow_object_yaw_fallback
            and yaw_sweep_target_obj_center is not None
            and len(self.grasped_objects) > 0
        ):
            target_obj_center_arr = np.asarray(yaw_sweep_target_obj_center, dtype=float).reshape(-1)
            if target_obj_center_arr.shape[0] == 3 and np.isfinite(target_obj_center_arr).all():
                current_quat_seed = np.asarray(current_ee_pose[3:], dtype=float)
                for yaw_deg in (45, 90, 135, 180, 225, 270, 315):
                    yaw_quat = self._rotate_quat_world_z(current_quat_seed, float(np.deg2rad(yaw_deg)))
                    yaw_goal = self._compute_fallback_goal_preserve_object_position(
                        target_obj_center=target_obj_center_arr,
                        object_to_place=self.grasped_objects[0],
                        current_ee_pose=current_ee_pose,
                        fallback_ee_quat=yaw_quat,
                    )
                    if yaw_goal is None:
                        continue
                    obj_yaw_result = self.motion_gen.plan_single(
                        start_state,
                        Pose.from_list(yaw_goal),
                        MotionGenPlanConfig(max_attempts=5),
                    )
                    if obj_yaw_result.success:
                        logging.warning(
                            "Object-yaw fallback succeeded at yaw=%d deg.",
                            yaw_deg,
                        )
                        return obj_yaw_result, goal_pose_0, yaw_goal
                logging.warning("Object-yaw fallback exhausted 7 candidates without success.")

        if not allow_fallback:
            return result, goal_pose_0, primary_goal

        fallback_goal = None
        if fallback_goal_override is not None:
            fallback_goal_arr = np.asarray(fallback_goal_override, dtype=float)
            if fallback_goal_arr.shape[0] == 7 and np.isfinite(fallback_goal_arr).all():
                fallback_goal = fallback_goal_arr.tolist()
            else:
                logging.warning(
                    "Invalid fallback_goal_override shape/values (%s); falling back to downward-quat-only retry.",
                    fallback_goal_arr,
                )
        if fallback_goal is None:
            downward_quat = self._get_downward_ee_quat(current_ee_pose)
            if downward_quat is None:
                return result, goal_pose_0, primary_goal
            fallback_goal = np.concatenate([np.asarray(primary_goal[:3], dtype=float), downward_quat], axis=0).tolist()
            logging.warning("Primary plan failed with held object; retrying once with downward wrist orientation.")
        else:
            logging.warning(
                "Primary plan failed with held object; retrying once with downward wrist + object-position correction."
            )
        fallback_result = self.motion_gen.plan_single(
            start_state,
            Pose.from_list(fallback_goal),
            MotionGenPlanConfig(max_attempts=5),
        )
        if fallback_result.success:
            return fallback_result, goal_pose_0, fallback_goal

        # Yaw-sweep fallback: keep wrist downward, allow held object to rotate
        # freely around world Z. Only meaningful when we still hold the object
        # and have a target object center to maintain.
        if (
            yaw_sweep_target_obj_center is not None
            and len(self.grasped_objects) > 0
        ):
            target_obj_center_arr = np.asarray(yaw_sweep_target_obj_center, dtype=float).reshape(-1)
            if target_obj_center_arr.shape[0] == 3 and np.isfinite(target_obj_center_arr).all():
                downward_seed = self._get_downward_ee_quat(current_ee_pose)
                if downward_seed is not None:
                    for yaw_deg in (45, 90, 135, 180, 225, 270, 315):
                        yaw_quat = self._rotate_quat_world_z(downward_seed, float(np.deg2rad(yaw_deg)))
                        yaw_goal = self._compute_fallback_goal_preserve_object_position(
                            target_obj_center=target_obj_center_arr,
                            object_to_place=self.grasped_objects[0],
                            current_ee_pose=current_ee_pose,
                            fallback_ee_quat=yaw_quat,
                        )
                        if yaw_goal is None:
                            continue
                        yaw_result = self.motion_gen.plan_single(
                            start_state,
                            Pose.from_list(yaw_goal),
                            MotionGenPlanConfig(max_attempts=5),
                        )
                        if yaw_result.success:
                            logging.warning(
                                "Downward-fallback failed; yaw-sweep succeeded at yaw=%d deg.",
                                yaw_deg,
                            )
                            return yaw_result, goal_pose_0, yaw_goal
                    logging.warning("Yaw-sweep fallback exhausted all 7 candidates without success.")

        return fallback_result, goal_pose_0, fallback_goal


    def _get_start_state_from_current_q(self, current_q):
        current_q = self.simulator.get_curobo_q_from_simulator(np.array(current_q))
        current_q = torch.tensor(current_q, dtype=torch.float32).to("cuda")[None]
        return JointState.from_position(
            current_q[:, self.simulator.robot_active_joint_indices],
            joint_names=self.simulator.robot_joint_names,
        )

    def _compute_mesh_half_height_z(self, mesh: Mesh) -> float:
        m = mesh.get_trimesh_mesh(process=False)
        if m.vertices is None or len(m.vertices) == 0:
            return 0.0
        z_vals = m.vertices[:, 2]
        return 0.5 * (float(z_vals.max()) - float(z_vals.min()))

    def _make_attach_spheres(self, mesh: Mesh, obj_name: str) -> list:
        cuboid = mesh.get_cuboid()
        max_dim = float(np.max(cuboid.dims))
        surface_radius = max(0.002, 0.02 * max_dim)
        # Denser fit: 8 coarse spheres over-approximate a mug by 1-2cm, which eats the
        # entire clearance of tight cavity inserts (microwave). Buffer capacity is bumped
        # to 32 in _resolve_motion_gen_robot_cfg.
        n_spheres = 16 if max_dim <= 0.05 else 32
        spheres = mesh.get_bounding_spheres(
            n_spheres=n_spheres,
            surface_sphere_radius=surface_radius,
        )
        for i, sph in enumerate(spheres):
            sph.name = f"{obj_name}_sph_{i}"
        return spheres

    def _make_place_sphere(self, mesh: Mesh, obj_name: str) -> Sphere:
        cuboid = mesh.get_cuboid()
        max_dim = float(np.max(cuboid.dims))
        radius = max(0.001, 0.5 * max_dim)
        sphere = Sphere(name=f"{obj_name}_place", pose=cuboid.pose, radius=radius)
        sphere.half_height_z = self._compute_mesh_half_height_z(mesh)
        return sphere
    
        
    def reset(self, payload):
        scene_mesh_data = payload["scene_mesh"]
        object_meshes_data: dict = payload["object_meshes"]
        q = payload["q"]
        grasped_object_names = payload["grasped_object_names"][:1]
        colli_with_grasped_object_names = payload["colli_with_grasped_object_names"]
        new_episode = payload.get("new_episode", False)
        # self.h = payload["h"]
        # self.w = payload["w"]
        # self.ixt = np.array(payload["ixt"])
        # self.ext = np.array(payload["ext"])   # NOTE: this ext must be w2c!!!

        if new_episode:
            self.grasped_object_names = []
            self.grasped_objects = []
            self.start_state = None
            self.scene_mesh = None
            self.object_meshes = None

        ## Init world
        mesh_obstacles = []
        if scene_mesh_data is not None:
            scene_mesh = Mesh(
                name="base_scene",
                # pose=[0, 0, 10000, 1, 0, 0, 0],   ## put it far away
                pose=self.simulator.scene_pose,
                **scene_mesh_data
            )
            mesh_obstacles.append(scene_mesh)
        object_meshes = dict()
        # Per-object asset-frame world quats (wxyz) captured at extraction time.
        # Used to override place_sphere.pose's quat (which would otherwise be the
        # trimesh OBB principal-axis frame, which is symmetry-ambiguous and not
        # the same as the asset frame the alignment math uses).
        asset_quats_world: dict = {}
        for obj_name, obj_mesh_data in object_meshes_data.items():
            # if "box" in obj_name: continue
            obj_mesh_data = dict(obj_mesh_data)  # shallow copy so we can pop without mutating caller
            asset_quat = obj_mesh_data.pop("asset_quat_wxyz", None)
            part_meshes_data = obj_mesh_data.pop("parts", None)  # arti per-part meshes (isaaclab.extract_object_meshes)
            if asset_quat is not None:
                asset_quats_world[obj_name] = asset_quat
            obj_mesh = Mesh(
                name=obj_name,
                # pose=[0, 0, 10000, 1, 0, 0, 0],   ## put it far away
                pose=self.simulator.scene_pose,
                **obj_mesh_data
            )
            object_meshes[obj_name] = obj_mesh
            # Table etc. has been merged into scene mesh
            if (obj_name in grasped_object_names or obj_name in colli_with_grasped_object_names
                    or obj_name in self.ignore_collision_assets):
                continue
            if part_meshes_data:
                # Articulated: same union geometry, but one obstacle per part so collision
                # attribution (and future per-part toggles) can name drawer_i/door_i/base.
                for part_label, part_data in part_meshes_data.items():
                    mesh_obstacles.append(Mesh(
                        name=f"{obj_name}::{part_label}",
                        pose=self.simulator.scene_pose,
                        **part_data,
                    ))
            else:
                mesh_obstacles.append(obj_mesh)

        world_config = WorldConfig(mesh=mesh_obstacles)

        # CPU copies (curobo frame) of the world obstacles for the collision-culprit
        # probe on planning failure (log_goal_collision_culprits).
        self._world_trimesh = {}
        try:
            for m in mesh_obstacles:
                v = rigid_transform(
                    np.asarray(m.vertices, dtype=np.float64), self.simulator.scene_pose_matrix
                )
                self._world_trimesh[m.name] = trimesh.Trimesh(
                    vertices=v, faces=np.asarray(m.faces), process=False
                )
        except Exception as exc:
            logging.warning("[collide_probe] failed to build world trimesh dict: %s", exc)
            self._world_trimesh = {}

        ## Init joint states
        ## Set the initial state for robot at the beggining of task
        q = self.simulator.get_curobo_q_from_simulator(q)
        q = torch.tensor(q, dtype=torch.float32).to("cuda")[None]  # list of floats
        self.start_state = JointState.from_position(
            q[:, self.simulator.robot_active_joint_indices],
            joint_names=self.simulator.robot_joint_names,
        )

        # tensor_args = TensorDeviceType()
        # robot_dict  = load_yaml(join_path(get_robot_configs_path(), self.robot_config))["robot_cfg"]
        # robot_cfg   = RobotConfig.from_dict(robot_dict, tensor_args)
        # robot       = CudaRobotModel(robot_cfg.kinematics)
        # sph_batched = robot.get_robot_as_spheres(q)[0]
        # WorldConfig(sphere=sph_batched).save_world_as_mesh(f"robot_cokecan.obj")
        fk = self.robot.compute_kinematics_from_joint_state(self.start_state)
        # print(f"EE pose: {fk.ee_pose}")

        self.init_eepose = fk.ee_pose.position[0].tolist() + fk.ee_pose.quaternion[0].tolist()
        print("curobo init: ", self.init_eepose)

        robot_cfg_for_motion_gen = self._motion_gen_robot_cfg
        if isinstance(robot_cfg_for_motion_gen, dict):
            robot_cfg_for_motion_gen = copy.deepcopy(robot_cfg_for_motion_gen)
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg_for_motion_gen,
            world_config,
            interpolation_dt=0.01,
            evaluate_interpolated_trajectory=True
        )
        self.motion_gen = MotionGen(motion_gen_config)
        # Lazily-built thin-sphere planner (collision_sphere_buffer=0) for the drawer/door grasp drive.
        self._thin_motion_gen = None
        # Lazily-built narrow-finger planner (fingers locked at narrow_finger_width instead of
        # full-open 0.04, + buffer=0) for the handle-grasp drive; see _get_narrow_motion_gen.
        self._narrow_motion_gen = None

        ######## Attach Object after Grasp ########
        try:
            self.motion_gen.detach_object_from_robot()
        except Exception:
            pass
        self.grasped_objects = []
        self.grasped_object_names = []
        objects_to_attach = []

        for obj_name in grasped_object_names:
            try:
                obj_to_attach: Mesh = object_meshes[obj_name]
            except:
                obj_to_attach: Mesh = object_meshes[obj_name + "_visual"]
            self.grasped_object_names.append(obj_name)
            obj_attach_spheres = self._make_attach_spheres(obj_to_attach, obj_name)
            objects_to_attach.extend(obj_attach_spheres)
            place_sphere = self._make_place_sphere(obj_to_attach, obj_name)
            # FIX: place_sphere.pose's quat comes from trimesh OBB principal axes,
            # which is symmetry-ambiguous (PCA sign flips with vertex perturbation
            # and causes get_ee_goal_from_object_goal to flip the held object by
            # ~180 deg). Replace with the asset's actual world quat (the same
            # frame the alignment math uses), transformed into curobo frame.
            asset_quat_world = asset_quats_world.get(obj_name) or asset_quats_world.get(obj_name + "_visual")
            if asset_quat_world is not None:
                try:
                    scene_R = self.simulator.scene_pose_matrix[:3, :3]
                    asset_R_world = q2R_wxyz(np.asarray(asset_quat_world, dtype=float))
                    asset_R_curobo = scene_R @ asset_R_world
                    asset_quat_curobo = R2q_wxyz(asset_R_curobo).astype(float).tolist()
                    new_pose = list(place_sphere.pose)
                    new_pose[3:] = asset_quat_curobo
                    place_sphere.pose = new_pose
                except Exception as _exc:
                    logging.warning("Failed to override place_sphere quat for %s: %s", obj_name, _exc)
            self.grasped_objects.append(place_sphere)

        # Record the attach sphere layout in the grasp-time EE frame for the goal-pose
        # collision probe (rigid wrt the EE during the grasp). Mirrors the +1cm z world
        # offset applied by attach_external_objects_to_robot below.
        self._attached_spheres_ee = []
        if objects_to_attach:
            try:
                t_ee = np.asarray(self.init_eepose[:3], dtype=np.float64)
                R_ee = q2R_wxyz(np.asarray(self.init_eepose[3:], dtype=np.float64))
                for sph in objects_to_attach:
                    p_raw = getattr(sph, "position", None)
                    if p_raw is None:
                        p_raw = sph.pose[:3]
                    p_w = np.asarray(p_raw, dtype=np.float64) + np.array([0.0, 0.0, 0.01])
                    self._attached_spheres_ee.append((R_ee.T @ (p_w - t_ee), float(sph.radius)))
            except Exception as exc:
                logging.warning("[collide_probe] failed to record attach spheres: %s", exc)
                self._attached_spheres_ee = []

        if objects_to_attach:
            self.motion_gen.attach_external_objects_to_robot(
                joint_state=self.start_state,
                external_objects=objects_to_attach,
                world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0])
            )
            
        # assert len(self.grasped_objects) <= 1, "We only support grasping one object at one time!"
            
        self.motion_gen.warmup()

        self.save_id += 1
        

    def get_ee_goal_from_object_goal(
        self, 
        target_loc_3d: list,    ## target object center in curobo system
        object_to_place: Sphere,   ## when converting to sphere, already transformed to curobo system
        current_ee_pose: list,  ## EE pose at the grasping moment, shoulb be already transformed to curobo system
        target_quat: list = None,
        offset: float = 0.05
    ) -> list:
        if target_loc_3d is None:
            target_loc_3d = object_to_place.pose[:3]
        target_loc_3d = np.array(target_loc_3d, dtype=float)
        if not np.isfinite(target_loc_3d).all():
            logging.warning("Invalid target_loc_3d for ee goal: %s", target_loc_3d)
            return None
        if target_quat is None:
            target_quat = object_to_place.pose[3:]
        target_quat = np.array(target_quat, dtype=float)
        if not np.isfinite(target_quat).all():
            logging.warning("Invalid target_quat for ee goal: %s", target_quat)
            return None

        T_world_ee = qt2T(current_ee_pose[3:], current_ee_pose[:3], with_wxyz=True)       ## EE pose in world frame
        T_world_obj_current = qt2T(object_to_place.pose[3:], object_to_place.pose[:3], with_wxyz=True)  ## obj pose in world frame
        if not np.isfinite(T_world_ee).all() or not np.isfinite(T_world_obj_current).all():
            logging.warning("Invalid transforms in ee goal computation.")
            return None
        T_obj_ee = np.linalg.inv(T_world_obj_current) @ T_world_ee                           ## relative trans from ee frame to obj frame

        goal_pose_T_obj = qt2T(target_quat, target_loc_3d, with_wxyz=True)
        goal_pose_T_ee = goal_pose_T_obj @ T_obj_ee
        if not np.isfinite(goal_pose_T_ee).all():
            logging.warning("Invalid goal_pose_T_ee for ee goal computation.")
            return None
        try:
            goal_pose_q_ee, goal_pose_t_ee = T2qt(goal_pose_T_ee, with_wxyz=True)
        except Exception as exc:
            logging.warning("Failed to convert ee goal pose to quaternion: %s", exc)
            return None
        goal_pose_ee = np.concatenate([goal_pose_t_ee, goal_pose_q_ee], 0).tolist()

        return goal_pose_ee

    def _compute_fallback_goal_preserve_object_position(
        self,
        target_obj_center: np.ndarray,
        object_to_place: Sphere,
        current_ee_pose: list,
        fallback_ee_quat: np.ndarray,
    ) -> list | None:
        target_obj_center = np.asarray(target_obj_center, dtype=float)
        fallback_ee_quat = np.asarray(fallback_ee_quat, dtype=float)
        if target_obj_center.shape[0] != 3 or not np.isfinite(target_obj_center).all():
            logging.warning("Invalid target_obj_center for fallback correction: %s", target_obj_center)
            return None
        if fallback_ee_quat.shape[0] != 4 or not np.isfinite(fallback_ee_quat).all():
            logging.warning("Invalid fallback_ee_quat for fallback correction: %s", fallback_ee_quat)
            return None
        try:
            T_world_ee = qt2T(current_ee_pose[3:], current_ee_pose[:3], with_wxyz=True)
            T_world_obj_current = qt2T(object_to_place.pose[3:], object_to_place.pose[:3], with_wxyz=True)
            T_ee_obj = np.linalg.inv(T_world_ee) @ T_world_obj_current
            p_ee_obj = T_ee_obj[:3, 3]
            fallback_rot = q2R_wxyz(fallback_ee_quat)
            fallback_ee_pos = target_obj_center - fallback_rot @ p_ee_obj
        except Exception as exc:
            logging.warning("Failed to compute fallback object-position correction: %s", exc)
            return None
        if not np.isfinite(fallback_ee_pos).all():
            logging.warning("Invalid corrected fallback ee pos: %s", fallback_ee_pos)
            return None
        return np.concatenate([fallback_ee_pos, fallback_ee_quat], axis=0).tolist()
        

    def lift(self, payload):
        current_ee_pose: list = payload["goal_pose"]
        gripper_state = np.array(payload["gripper_state"], dtype=float)
        start_state = self._get_start_state_from_current_q(payload["current_q"])

        goal_pose_list = current_ee_pose.copy()
        goal_pose_list[2] += self.lift_offset  # lift up 10cm
        goal_pose = Pose.from_list(goal_pose_list)

        result = self.motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=5))

        if result.success:
            traj = result.get_interpolated_plan()
            joint_states_np = traj.position.cpu().numpy()
            joint_vel_np = traj.velocity.cpu().numpy()
            # # fix RLBench joint issue
            joint_states_np = self.simulator.get_simulator_q_from_curobo(joint_states_np)

            final_q = joint_states_np[-1]

            gripper_states = np.tile(gripper_state[None, :], (joint_states_np.shape[0], 1))
            joint_states_np = np.concatenate([joint_states_np, gripper_states], -1)
            joint_states_np = np.concatenate([joint_states_np, joint_states_np[-1:].repeat(3, 0)], 0)
            gripper_vel = np.zeros((joint_vel_np.shape[0], 2), dtype=joint_vel_np.dtype)
            joint_vel_np = np.concatenate([joint_vel_np, gripper_vel], -1)
            joint_vel_np = np.concatenate([joint_vel_np, joint_vel_np[-1:].repeat(3, 0)], 0)

            return {
                "joint_states": joint_states_np.tolist(),
                "joint_vels": joint_vel_np.tolist(),
                "final_q": final_q.tolist(),
                "goal_pose_0": goal_pose_list,
                "goal_pose": goal_pose_list,
            }
        
        return {
            "joint_states": None,
            "joint_vels": None,
            "final_q": None,
            "goal_pose_0": None,
            "goal_pose": None
        }
    

    # def move_gripper(self, payload):
    def _get_thin_motion_gen(self):
        """Lazily build a SECOND MotionGen with the default 4mm collision_sphere_buffer removed
        (set to 0), sharing the same (static) world as the main planner. Used only for the
        drawer/door grasp drive: the grasp init pose sits ON the asset's handle, and the 4mm sphere
        inflation makes the thin gripper fingers falsely collide with the recessed middle/bottom
        handle (IK_FAIL) even though the real geometry clears (RL's reset-IK has no such margin).
        The cabinet/microwave stays a collision obstacle so the ARM still avoids it (its spheres
        have ~2.7cm clearance; the 4mm shrink only matters for the gripper grazing the handle)."""
        if self._thin_motion_gen is None:
            rcfg = self._resolve_motion_gen_robot_cfg(self.robot_config)
            rcfg = copy.deepcopy(rcfg) if isinstance(rcfg, dict) else rcfg
            try:
                rcfg["kinematics"]["collision_sphere_buffer"] = 0.0
            except Exception as exc:
                logging.warning("[thin_planner] could not set collision_sphere_buffer=0: %s", exc)
            cfg = MotionGenConfig.load_from_robot_config(
                rcfg, self.motion_gen.world_model,
                interpolation_dt=0.01, evaluate_interpolated_trajectory=True,
            )
            mg = MotionGen(cfg)
            mg.warmup()
            self._thin_motion_gen = mg
        return self._thin_motion_gen

    def _get_narrow_motion_gen(self):
        """Lazily build a THIRD MotionGen for the handle-GRASP drive: fingers locked at
        narrow_finger_width (instead of franka.yml's full-open 0.04) plus the thin planner's
        collision_sphere_buffer=0. Rationale: full-open fingertip spheres hang ~4.5cm below the
        grasped bar and sweep the NEXT handle down on tight cabinets (v01/v03/v04 top-drawer
        grasps penetrate it by 0.7-1.3cm -> IK_FAIL on every sample); locked narrow, the
        fingertips sit ~1.2cm off the bar and clear the neighbor handle by ~3cm, while the
        hand/knuckle spheres (which don't move with the finger joints) already have >1.4cm
        margin. The drive itself commands the same width (see _move_impl), so the planning
        model and the executed jaws agree. NOT for the close-push drive: those poses arrive
        jaws-open by design (RL-trained state)."""
        if self._narrow_motion_gen is None:
            w = float(self.narrow_finger_width)
            rcfg = self._resolve_motion_gen_robot_cfg(self.robot_config)
            rcfg = copy.deepcopy(rcfg) if isinstance(rcfg, dict) else rcfg
            try:
                rcfg["kinematics"]["collision_sphere_buffer"] = 0.0
                rcfg["kinematics"]["lock_joints"] = {
                    "panda_finger_joint1": w, "panda_finger_joint2": w,
                }
            except Exception as exc:
                logging.warning("[narrow_planner] could not set lock_joints/buffer: %s", exc)
            cfg = MotionGenConfig.load_from_robot_config(
                rcfg, self.motion_gen.world_model,
                interpolation_dt=0.01, evaluate_interpolated_trajectory=True,
            )
            mg = MotionGen(cfg)
            mg.warmup()
            self._narrow_motion_gen = mg
        return self._narrow_motion_gen

    def move(self, payload):
        # Handle-grasp drive: swap in the narrow-finger planner (fingers locked at
        # narrow_finger_width + buffer=0); the emitted trajectory commands the same width.
        if payload.get("narrow_fingers"):
            saved_mg = self.motion_gen
            try:
                self.motion_gen = self._get_narrow_motion_gen()
                return self._move_impl(payload)
            finally:
                self.motion_gen = saved_mg
        # Drawer/door close-push drive: temporarily swap in the thin-sphere planner (collision_sphere_buffer
        # 0 instead of the default 4mm) so cuRobo's gripper spheres fit around the recessed handle, while
        # the cabinet/microwave stays an obstacle for ARM avoidance. _move_impl uses self.motion_gen, so
        # swapping it is enough. See _get_thin_motion_gen.
        if payload.get("thin_collision_spheres"):
            saved_mg = self.motion_gen
            try:
                self.motion_gen = self._get_thin_motion_gen()
                return self._move_impl(payload)
            finally:
                self.motion_gen = saved_mg
        return self._move_impl(payload)

    def _move_impl(self, payload):
        # logging.warning("MOVE GRIPPER")
        gripper_state = payload["gripper_state"]
        start_state = self._get_start_state_from_current_q(payload["current_q"])
        fk = self.robot.compute_kinematics_from_joint_state(start_state)
        current_ee_pose = fk.ee_pose.position[0].tolist() + fk.ee_pose.quaternion[0].tolist()
        prepend_start_buffer = bool(payload.get("prepend_start_buffer", True))
        preserve_object_position_on_fallback = bool(
            payload.get("preserve_object_position_on_fallback", self.preserve_object_position_on_fallback)
        )
        
        target_loc_3d = payload.get("target_loc", None)
        if target_loc_3d is not None:
            ## get 3d position of the placement point on the object surface
            ## in curobo coord system
            target_loc_3d = np.array(target_loc_3d, dtype=float)  # xyz
            target_loc_3d = rigid_transform(target_loc_3d[None], self.simulator.scene_pose_matrix)[0]  # (3,)

        target_quat = payload.get("target_quat", None)
        if target_quat is not None:
            target_quat = np.array(target_quat, dtype=float)
            if not np.isfinite(target_quat).all():
                logging.warning("Invalid target_quat in payload; ignoring.")
                target_quat = None
            else:
                try:
                    scene_R = self.simulator.scene_pose_matrix[:3, :3]
                    target_R = q2R_wxyz(target_quat)
                    target_quat = R2q_wxyz(scene_R @ target_R)
                except Exception as exc:
                    logging.warning("Failed to transform target_quat; ignoring. Error: %s", exc)
                    target_quat = None

        assert len(self.grasped_objects) == len(self.grasped_object_names), f"objects: {len(self.grasped_objects)}, names: {len(self.grasped_object_names)}"
        if len(self.grasped_objects) > 0:
            logging.warning("MOVE OBJECT")
            logging.warning(f"Grasped objs: {len(self.grasped_objects)} objs, they are: {self.grasped_object_names}")
            goal_pose_list = self.get_ee_goal_from_object_goal(
                target_loc_3d=target_loc_3d,
                object_to_place=self.grasped_objects[0],
                current_ee_pose=current_ee_pose,
                target_quat=target_quat,
                # offset=self.place_offset
            )
        else:
            logging.warning("MOVE GRIPPER")
            align_empty_gripper_wrist_to_world_z = bool(
                payload.get("align_empty_gripper_wrist_to_world_z", self.align_empty_gripper_wrist_to_world_z)
            )
            empty_gripper_target_quat = target_quat
            if empty_gripper_target_quat is None and align_empty_gripper_wrist_to_world_z:
                # EE approach axis (panda_hand +Z / widowx gripper_link +X) aligned to world -Z = "gripper points down".
                empty_gripper_target_quat = self._align_ee_local_axis_to_world_axis(
                    current_quat=np.array(current_ee_pose[3:], dtype=float),
                    ee_local_axis=np.asarray(getattr(self.simulator, "ee_approach_local", (0.0, 0.0, 1.0)), dtype=float),
                    world_axis=np.array([0.0, 0.0, -1.0], dtype=float),
                )
            if target_loc_3d is None:
                if empty_gripper_target_quat is None:
                    goal_pose_list = current_ee_pose.copy()
                else:
                    goal_pose_list = np.concatenate([np.array(current_ee_pose[:3], dtype=float), empty_gripper_target_quat], 0).tolist()
            else:
                quat = np.array(current_ee_pose[3:] if empty_gripper_target_quat is None else empty_gripper_target_quat, dtype=float)
                target_loc_3d = self.simulator.get_translation_from_3d_point(quat, target_loc_3d)
                goal_pose_list = np.concatenate([target_loc_3d, quat], 0).tolist()
        if goal_pose_list is None:
            return {
                "joint_states": None,
                "joint_vels": None,
                "final_q": None,
                "goal_pose_0": None,
                "goal_pose": None,
                "status": "NO_GOAL",
            }

        goal_pose_override = payload.get("goal_pose_override", None)
        if goal_pose_override is not None:
            goal_pose_override = np.asarray(goal_pose_override, dtype=float).reshape(-1)
            if goal_pose_override.shape[0] == 7 and np.isfinite(goal_pose_override).all():
                goal_pose_list = goal_pose_override.tolist()
            else:
                logging.warning("Invalid goal_pose_override in move payload; ignoring: %s", goal_pose_override)

        # fk = self.robot.compute_kinematics_from_joint_state(start_state)
        # current_ee_pose = fk.ee_pose.position[0].tolist() + fk.ee_pose.quaternion[0].tolist()
        # quat = np.array(current_ee_pose[3:])
        # target_loc_3d = self.simulator.get_translation_from_3d_point(quat, target_loc_3d)
        # goal_pose_list = np.concatenate([target_loc_3d, quat], 0).tolist()
        # goal_pose = Pose.from_list(goal_pose_list)

        # tensor_args = TensorDeviceType()
        # robot_dict  = load_yaml(join_path(get_robot_configs_path(), self.robot_config))["robot_cfg"]
        # robot_cfg   = RobotConfig.from_dict(robot_dict, tensor_args)
        # robot       = CudaRobotModel(robot_cfg.kinematics)
        # sph_batched = robot.get_robot_as_spheres(torch.tensor(payload["current_q"]).to("cuda"))[0]
        # WorldConfig(sphere=sph_batched).save_world_as_mesh(f"robot_0.obj")

        has_held_object = len(self.grasped_objects) > 0
        fallback_goal_override = None
        if has_held_object and preserve_object_position_on_fallback and (target_loc_3d is not None):
            downward_quat = self._get_downward_ee_quat(current_ee_pose)
            if downward_quat is not None:
                fallback_goal_override = self._compute_fallback_goal_preserve_object_position(
                    target_obj_center=target_loc_3d,
                    object_to_place=self.grasped_objects[0],
                    current_ee_pose=current_ee_pose,
                    fallback_ee_quat=downward_quat,
                )

        disable_dyf_move = bool(payload.get("disable_downward_yaw_fallback", False))
        allow_obj_yaw_move = bool(payload.get("allow_object_yaw_fallback", False))
        result, goal_pose_0, goal_pose_used = self._plan_single_with_downward_fallback(
            start_state=start_state,
            goal_pose_list=goal_pose_list,
            current_ee_pose=current_ee_pose,
            allow_fallback=(has_held_object and not disable_dyf_move),
            fallback_goal_override=fallback_goal_override,
            yaw_sweep_target_obj_center=target_loc_3d if has_held_object else None,
            allow_object_yaw_fallback=(allow_obj_yaw_move and has_held_object),
        )

        if result.success:
            traj = result.get_interpolated_plan()
            joint_states_np = traj.position.cpu().numpy()
            joint_vel_np = traj.velocity.cpu().numpy()

            # # fix RLBench joint issue
            joint_states_np = self.simulator.get_simulator_q_from_curobo(joint_states_np)
            if prepend_start_buffer and joint_states_np.shape[0] > 0:
                joint_states_np = np.concatenate([joint_states_np[:1].repeat(5, 0), joint_states_np], 0)
                joint_vel_np = np.concatenate([joint_vel_np[:1].repeat(5, 0), joint_vel_np], 0)
            final_q = joint_states_np[-1]

            if payload.get("narrow_fingers"):
                # Handle-grasp drive: command the jaws at the narrow planner's locked width so
                # the executed fingers match the collision model that cleared the plan.
                w = float(self.narrow_finger_width)
                gripper_state = np.array([w, w], dtype=float)
            elif len(self.grasped_objects) == 0 and len(self.grasped_object_names) == 0:
                max_open = self.simulator.max_gripper_opening
                if max_open is None:
                    raise RuntimeError("simulator.max_gripper_opening is None when forcing empty-gripper move to max opening.")
                gripper_state = np.array([max_open, max_open], dtype=float)
            else:
                gripper_state = np.array(gripper_state, dtype=float)
                gripper_state -= 0.003
            gripper_states = np.tile(gripper_state[None, :], (joint_states_np.shape[0], 1))
            joint_states_np = np.concatenate([joint_states_np, gripper_states], -1)
            joint_states_np = np.concatenate([joint_states_np, joint_states_np[-1:].repeat(3, 0)], 0)
            gripper_vel = np.zeros((joint_vel_np.shape[0], 2), dtype=joint_vel_np.dtype)
            joint_vel_np = np.concatenate([joint_vel_np, gripper_vel], -1)
            joint_vel_np = np.concatenate([joint_vel_np, joint_vel_np[-1:].repeat(3, 0)], 0)

            # for i in range(0, len(joint_states_np), max(1, len(joint_states_np)//10)):
            #     sph_batched = robot.get_robot_as_spheres(torch.tensor(joint_states_np[i]).to("cuda"))[0]
            #     WorldConfig(sphere=sph_batched).save_world_as_mesh(f"robot_{i+1}.obj")

            return {
                "joint_states": joint_states_np.tolist(),
                "joint_vels": joint_vel_np.tolist(),
                "final_q": final_q.tolist(),
                "goal_pose_0": goal_pose_0,
                "goal_pose": goal_pose_used,
            }
        
        return {
            "joint_states": None,
            "joint_vels": None,
            "final_q": None,
            "goal_pose_0": goal_pose_0,
            "goal_pose": goal_pose_used,
            "status": str(getattr(result, "status", None)),
        }


    _HAND_PROXY_SPHERES_EE = (
        ((0.0, 0.0, 0.0), 0.045),
        ((0.0, 0.04, 0.058), 0.028), ((0.0, -0.04, 0.058), 0.028),
        ((0.0, 0.045, 0.10), 0.02), ((0.0, -0.045, 0.10), 0.02),
    )

    def log_goal_collision_culprits(self, ee_goal_pose, tag="", narrow_width=None):
        """Attribute a failed (goal-in-collision) plan to world obstacles: place the
        attached-object spheres (the exact planning geometry) plus a coarse 5-sphere
        hand proxy at the candidate EE goal (curobo frame), and report which world
        meshes they penetrate. Part-split arti obstacles (obj::drawer_i/door_i/base)
        give part-level attribution. narrow_width: per-finger opening when the plan ran
        on the narrow-finger planner — the fingertip proxy spheres (nominally at the
        full-open |y|=0.045) move to |y|=narrow_width+0.01 so the attribution matches
        the actual planning model. Returns [{group, obstacle, depth_m}] sorted by
        depth; also logs the top offenders. Failure-path only (CPU, ~100s of ms)."""
        world = getattr(self, "_world_trimesh", None)
        if not world:
            return []
        try:
            goal = np.asarray(ee_goal_pose, dtype=np.float64).reshape(-1)
            if goal.shape[0] < 7 or not np.isfinite(goal).all():
                return []
            t_g, R_g = goal[:3], q2R_wxyz(goal[3:7])
            groups = []
            for p_ee, r in (getattr(self, "_attached_spheres_ee", None) or []):
                groups.append(("held_object", R_g @ np.asarray(p_ee, dtype=np.float64) + t_g, float(r)))
            for p_ee, r in self._HAND_PROXY_SPHERES_EE:
                if narrow_width is not None and abs(p_ee[1]) == 0.045:
                    # Fingertip proxies follow the finger joints; +0.01/r=0.011 = the real
                    # fingertip sphere from franka_mesh.yml (the coarse full-open proxy r=0.02
                    # would falsely penetrate the grasped bar itself once shifted inward).
                    y = (float(narrow_width) + 0.01) * (1.0 if p_ee[1] > 0 else -1.0)
                    p_ee, r = (p_ee[0], y, p_ee[2]), 0.011
                groups.append(("hand(approx)", R_g @ np.asarray(p_ee, dtype=np.float64) + t_g, float(r)))
            centers = np.stack([g[1] for g in groups], axis=0)
            radii = np.asarray([g[2] for g in groups], dtype=np.float64)
            hits = []
            for name, tm in world.items():
                try:
                    sd = trimesh.proximity.signed_distance(tm, centers)  # + inside
                except Exception:
                    try:
                        _, dist, _ = trimesh.proximity.closest_point(tm, centers)
                        sd = -np.asarray(dist)
                    except Exception:
                        continue
                pen = np.asarray(sd, dtype=np.float64) + radii  # sphere overlaps if > 0
                worst = {}
                for gi, p in enumerate(pen):
                    if p <= 0:
                        continue
                    grp = groups[gi][0]
                    if p > worst.get(grp, 0.0):
                        worst[grp] = float(p)
                for grp, depth in worst.items():
                    hits.append({"group": grp, "obstacle": name, "depth_m": round(depth, 4)})
            hits.sort(key=lambda h: -h["depth_m"])
            prefix = f" {tag}" if tag else ""
            if hits:
                desc = "; ".join(
                    f"{h['group']} vs {h['obstacle']} depth~{h['depth_m']:.3f}" for h in hits[:6]
                )
                logging.warning("[collide_probe]%s goal-in-collision culprits: %s", prefix, desc)
            else:
                logging.warning(
                    "[collide_probe]%s no penetration at goal "
                    "(likely reachability/joint-limit, not collision).", prefix,
                )
            return hits
        except Exception as exc:
            logging.warning("[collide_probe] failed: %s", exc)
            return []


    def _disable_all_world_obstacles(self):
        """Zero the collision checker's obstacle enable flags/counts in place.

        Equivalent to motion_gen.clear_world_cache(), which crashes here: its
        _env_mesh_names rebuild indexes self.cache["mesh"], but self.cache stays
        None when the mesh cache was built lazily by load_collision_model.
        Keeping _wp_mesh_cache intact also lets the update_world(saved_world)
        restore reuse the cached warp meshes instead of rebuilding BVHs.
        """
        cc = self.motion_gen.world_coll_checker
        if cc is None:
            return
        mesh_tensors = getattr(cc, "_mesh_tensor_list", None)
        if mesh_tensors is not None:
            mesh_tensors[2][:] = 0
        env_n_mesh = getattr(cc, "_env_n_mesh", None)
        if env_n_mesh is not None:
            env_n_mesh[:] = 0
        cube_tensors = getattr(cc, "_cube_tensor_list", None)
        if cube_tensors is not None:
            cube_tensors[2][:] = 0
        env_n_obbs = getattr(cc, "_env_n_obbs", None)
        if env_n_obbs is not None:
            env_n_obbs[:] = 0

    def release(self, payload):
        # Attempt 1: plan with the world intact so the retract is obstacle-aware.
        res = self._release_impl(payload, inplace_fallback=False, attempt_tag="with world")
        if res["joint_states"] is not None:
            return res

        # Attempt 2: RL-place-induced finger-in-surface penetration triggers
        # INVALID_START_STATE_WORLD_COLLISION; retry with all obstacles disabled.
        # The release motion only retracts the EE, so dropping world checks is safe.
        saved_world = None
        try:
            saved_world = copy.deepcopy(self.motion_gen.world_model)
            # update_world(WorldConfig()) is a no-op for clearing: with 0 meshes/cuboids
            # cuRobo skips the enable-flag reset, so old obstacles stay active on GPU.
            self._disable_all_world_obstacles()
        except Exception as exc:
            logging.warning("[release] failed to clear world: %s", exc)
            saved_world = None

        try:
            return self._release_impl(payload, inplace_fallback=True, attempt_tag="world disabled")
        finally:
            if saved_world is not None:
                try:
                    self.motion_gen.update_world(saved_world)
                except Exception as exc:
                    logging.warning("[release] failed to restore world: %s", exc)


    def _release_impl(self, payload, inplace_fallback: bool = True, attempt_tag: str = ""):
        logging.warning("RELEASE (%s)", attempt_tag or "single attempt")

        ## get start pose
        start_state = self._get_start_state_from_current_q(payload["current_q"])
        fk = self.robot.compute_kinematics_from_joint_state(start_state)
        current_ee_pose = fk.ee_pose.position[0].tolist() + fk.ee_pose.quaternion[0].tolist()
        prepend_start_buffer = bool(payload.get("prepend_start_buffer", True))

        goal_pose_list = current_ee_pose.copy()
        retract_dir_world = payload.get("retract_dir_world", None)
        if retract_dir_world is not None:
            # Non-vertical retract: back off along -n_app (world) by release_retract_offset.
            # n_app comes from the place engine; -n_app is the straight pull-out direction.
            rd = np.asarray(retract_dir_world, dtype=np.float64).reshape(-1)
            try:
                rd = self.simulator.scene_pose_matrix[:3, :3] @ rd
            except Exception:
                pass
            norm = float(np.linalg.norm(rd))
            if norm > 1e-9:
                rd = rd / norm
                retracted = np.asarray(current_ee_pose[:3], dtype=np.float64) + self.release_retract_offset * rd
                goal_pose_list[0] = float(retracted[0])
                goal_pose_list[1] = float(retracted[1])
                goal_pose_list[2] = float(retracted[2])
        else:
            # Vertical retract: lift +z (unchanged). NOTE: the legacy retract_along_local_z
            # "back_dir" branch was removed -- no caller reaches release with it set
            # (execute_drop short-circuits retract_along_local_z=True to an in-place open).
            goal_pose_list[2] += self.release_offset
        goal_pose_override = payload.get("goal_pose_override", None)
        if goal_pose_override is not None:
            goal_pose_override = np.asarray(goal_pose_override, dtype=float).reshape(-1)
            if goal_pose_override.shape[0] == 7 and np.isfinite(goal_pose_override).all():
                goal_pose_list = goal_pose_override.tolist()
            else:
                logging.warning("Invalid goal_pose_override in release payload; ignoring: %s", goal_pose_override)

        # Release ladder (RFC §8.2): primary plan only. On failure we fall through to the
        # in-place open below. The former downward-wrist / world-Z yaw-sweep / canonical-home
        # retries were downward-centric (they rotate the wrist toward straight-down, which is
        # wrong for a non-vertical retract) and are removed; release/approach no longer share
        # _plan_single_with_downward_fallback.
        goal_pose_0 = np.asarray(goal_pose_list, dtype=float).tolist()
        goal_pose_used = goal_pose_0
        result = self.motion_gen.plan_single(
            start_state,
            Pose.from_list(goal_pose_list),
            MotionGenPlanConfig(max_attempts=5),
        )

        if result.success:
            traj = result.get_interpolated_plan()
            joint_states_np = traj.position.cpu().numpy()
            joint_vel_np = traj.velocity.cpu().numpy()
            # # fix RLBench joint issue
            joint_states_np = self.simulator.get_simulator_q_from_curobo(joint_states_np)
            if prepend_start_buffer and joint_states_np.shape[0] > 0:
                joint_states_np = np.concatenate([joint_states_np[:1].repeat(5, 0), joint_states_np], 0)
                joint_vel_np = np.concatenate([joint_vel_np[:1].repeat(5, 0), joint_vel_np], 0)
            final_q = joint_states_np[-1]

            open_val = self._scale_gripper(self.simulator.gripper_open_state)
            gripper_states = np.ones((joint_states_np.shape[0], 2), dtype=joint_states_np.dtype) * open_val
            joint_states_np = np.concatenate([joint_states_np, gripper_states], -1)
            joint_states_np = np.concatenate([joint_states_np, joint_states_np[-1:].repeat(3, 0)], 0)
            gripper_vel = np.zeros((joint_vel_np.shape[0], 2), dtype=joint_vel_np.dtype)
            joint_vel_np = np.concatenate([joint_vel_np, gripper_vel], -1)
            joint_vel_np = np.concatenate([joint_vel_np, joint_vel_np[-1:].repeat(3, 0)], 0)

            return {
                "joint_states": joint_states_np.tolist(),
                "joint_vels": joint_vel_np.tolist(),
                "final_q": final_q.tolist(),
                "goal_pose_0": goal_pose_0,
                "goal_pose": goal_pose_used,
            }
        
        if not inplace_fallback:
            # Attempt 1 (world intact) failed; signal the caller to retry with
            # obstacles disabled instead of opening in place.
            logging.warning(
                "[release] world-aware plan failed (status=%s); retrying with obstacles disabled.",
                getattr(result, "status", None),
            )
            return {
                "joint_states": None,
                "joint_vels": None,
                "final_q": None,
                "goal_pose_0": goal_pose_0,
                "goal_pose": None,
            }

        # Ultimate fallback: keep EE in place, only open the fingers.
        logging.warning(
            "Release ULTIMATE fallback: motion planning exhausted (status=%s); "
            "keeping EE at current pose and opening gripper only.",
            getattr(result, "status", None),
        )
        try:
            arm_curobo_pos = start_state.position[0].detach().cpu().numpy().reshape(1, -1)
            arm_sim_pos = self.simulator.get_simulator_q_from_curobo(arm_curobo_pos)  # (1, arm_dim)
            n_frames = 8
            arm_traj = np.tile(arm_sim_pos, (n_frames, 1))
            open_val = self._scale_gripper(self.simulator.gripper_open_state)
            gripper_traj = np.ones((n_frames, 2), dtype=arm_traj.dtype) * open_val
            joint_states_np = np.concatenate([arm_traj, gripper_traj], axis=-1)
            joint_vel_np = np.zeros_like(joint_states_np)
            return {
                "joint_states": joint_states_np.tolist(),
                "joint_vels": joint_vel_np.tolist(),
                "final_q": joint_states_np[-1].tolist(),
                "goal_pose_0": goal_pose_0,
                "goal_pose": current_ee_pose,
            }
        except Exception as exc:
            logging.warning("Release ULTIMATE fallback failed to construct in-place trajectory: %s", exc)
            return {
                "joint_states": None,
                "joint_vels": None,
                "final_q": None,
                "goal_pose_0": None,
                "goal_pose": None
            }


#     def run(self, host: str = "0.0.0.0", port: int = 9060) -> None:
#         import uvicorn
#         logging.info(f"🚀 Running cuRobo server on http://{host}:{port}")
#         uvicorn.run(self.app, host=host, port=port)

        
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--port", type=int, default=9060)
#     parser.add_argument("--grasp_select_mode", type=str, default="score")
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO)
#     server = cuRoboServer(args)
#     server.run(port=args.port)
