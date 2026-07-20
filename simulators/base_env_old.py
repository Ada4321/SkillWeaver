from __future__ import annotations

import os
import json
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from simulators.base_env_cfg import BaseEnvCfg
from simulators.tasks import PickTask, PlaceTask, PoseTask, OpenTask, CloseTask
from utils.geometry_utils import to_tensor


class BaseEnv(DirectRLEnv):
    cfg: BaseEnvCfg

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

        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # robot dof limits and targets
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # joint indices
        self.arm_joint_indices, _ = self.robot.find_joints("panda_joint.*")
        self.left_joint_idx = self.robot.find_joints("panda_finger_joint1")[0][0]
        self.right_joint_idx = self.robot.find_joints("panda_finger_joint2")[0][0]
        # use joint limits to scale normalized gripper commands
        gripper_upper_limits = self.robot_dof_upper_limits[[self.left_joint_idx, self.right_joint_idx]]
        self.max_gripper_opening = float(torch.max(gripper_upper_limits))

        # link indices
        self.hand_link_idx = self.robot.find_bodies("panda_hand")[0][0]
        self.left_finger_idx = self.robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_idx = self.robot.find_bodies("panda_rightfinger")[0][0]
        self.left_tip_idx = self.robot.find_bodies("panda_leftfinger_tip")[0][0]
        self.right_tip_idx = self.robot.find_bodies("panda_rightfinger_tip")[0][0]
        self.grasp_site_idx = self.robot.find_bodies("panda_grip_site")[0][0]

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
        self.cmd_limit = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float, device=self.device)

        # unit
        self.unit_z = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device)

        self.object = None
        self.data_point_cloud = None

        # action mode
        self.action_mode = 1  # default: delta EE pose
        self.is_release_action = False

        # init tasks
        self.tasks = dict()
        self.observasion_spaces = dict()
        if self.cfg.enable_pick:
            self.tasks["pick"] = PickTask(self.cfg.scene.num_envs, self.cfg, self.device)
            self.observasion_spaces["pick"] = self.cfg.observation_space_pick
        if self.cfg.enable_place:
            self.tasks["place"] = PlaceTask(self.cfg.scene.num_envs, self.cfg, self.device)
            self.observasion_spaces["place"] = self.cfg.observation_space_place
        if self.cfg.enable_pose:
            self.tasks["pose"] = PoseTask(self.cfg.scene.num_envs, self.cfg, self.device)
            self.observasion_spaces["pose"] = self.cfg.observation_space_pose
        if self.cfg.enable_open:
            self.tasks["open"] = OpenTask(self.cfg.scene.num_envs, self.cfg, self.device)
            self.observasion_spaces["open"] = self.cfg.observation_space_open
        if self.cfg.enable_close:
            self.tasks["close"] = CloseTask(self.cfg.scene.num_envs, self.cfg, self.device)
            self.observasion_spaces["close"] = self.cfg.observation_space_close

        self.env_reset_flag = False
        self.task_name = None
        self.task = None


    def _setup_scene(self, use_camera: bool = True):
        prim_utils.create_prim("/World/envs/env_0/Objects", "Xform")
        # setup scene
        self.robot = Articulation(self.cfg.robot)
        self.table = RigidObject(self.cfg.table) if self.cfg.table is not None else None
        table_pos = self.cfg.table.init_state.pos if self.cfg.table is not None else (0.0, 0.0, 0.0)

        if use_camera:
            self.head_camera = TiledCamera(self.cfg.head_camera)
            self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.contact = ContactSensor(self.cfg.contact)

        # if use_camera:
        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)

        # store prim paths
        self.table_prim = "/World/envs/env_.*/table"
        self.object_prims = dict()

        # set up objects in the scene
        self.objects: dict[str, RigidObject] = {}  # NOTE: For now we only support rigid objects
        self.object_point_clouds: dict[str, torch.Tensor] = {}
        self.object_init_poses: dict[str, torch.Tensor] = {}
        for obj_name, obj_cfg in self.scene_desc.items():
            name = obj_name
            usd_path = obj_cfg["colored_usd_path"]

            prim_path = f"/World/envs/env_0/Objects/{name}/base_link"
            self.object_prims[name] = prim_path

            orig_pos = tuple(obj_cfg["centerized_pos"])
            pos = (orig_pos[0] + table_pos[0] - 0.05, orig_pos[1] + table_pos[1], orig_pos[2] + table_pos[2] + 0.05 / 2)
            rot = tuple(obj_cfg["rot"])

            rigid_cfg = RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    mass_props=sim_utils.MassPropertiesCfg(density=500.0),
                    # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=pos,
                    rot=rot,
                ),
            )
            obj_handler = RigidObject(rigid_cfg)
            self.objects[name] = obj_handler

            # load point cloud
            obj_pcd = torch.tensor(np.load(obj_cfg["point_cloud"], allow_pickle=True)["arr_0"], device=self.device, dtype=torch.float32)
            self.object_point_clouds[name] = obj_pcd
            # store init pose
            init_pos = torch.tensor(pos, device=self.device, dtype=torch.float32)  # shape: (3,)
            init_rot = torch.tensor(rot, device=self.device, dtype=torch.float32)  # shape: (4,)
            self.object_init_poses[name] = torch.cat((init_pos, init_rot), dim=-1)

        # register to the scene
        self.scene.articulations["robot"] = self.robot
        if self.table is not None:
            self.scene.rigid_objects["table"] = self.table
        for name, obj in self.objects.items():
            self.scene.rigid_objects[name] = obj
        self.scene.sensors["contact"] = self.contact

        if use_camera:
            self.scene.sensors["head_camera"] = self.head_camera
            self.scene.sensors["wrist_camera"] = self.wrist_camera
        self.controller = DifferentialIKController(self.cfg.controller, self.num_envs, self.device)

        # add lights (TODO: the visuals seem bad; how to improve?)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    ############################################################
    # pre-physics step calls
    ############################################################
    def set_control_params(self, mode: int, is_release: bool = False):
        """
        mode=0: execute actions from motion planning(joint q pos)
        mode=1: execute actions from neural polices(EE pose) TODO: abs or delta???
        """
        self.action_mode = mode
        self.is_release_action = bool(is_release)

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

            if self.is_release_action:
                # Teleoperate release actions directly; skip drive targets in apply_action.
                self.robot.write_joint_state_to_sim(full_targets, torch.zeros_like(full_targets))
            self.robot_dof_targets[:] = full_targets  # do not clamp here; planner outputs real q
        elif self.action_mode == 1:
            self.actions = actions.clone().clamp(-1.0, 1.0)
            arm_action, gripper_action = self.actions[:, :6], self.actions[:, 6:]
            # We use DifferentialIKController to convert operational-space commands to joint-space commands.
            self.controller.set_command(
                arm_action * self.cmd_limit,
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
            gripper_targets = self.robot.data.joint_pos[:, -2:] + gripper_action * 0.01
            # Set robot DoF targets.
            self.robot_dof_targets[:] = torch.clamp(
                torch.cat((arm_targets, gripper_targets), dim=-1),
                self.robot_dof_lower_limits,
                self.robot_dof_upper_limits,
            )
        else:
            raise NotImplementedError()


    def _apply_action(self):
        if self.action_mode == 0 and self.is_release_action:
            return
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
            self.task_name = None
            self.task = None
            self.reset_buf[env_ids] = 0
            self.episode_length_buf[env_ids] = 0

            joint_pos = self.robot.data.default_joint_pos[env_ids]
            joint_vel = torch.zeros_like(joint_pos)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        else:  # task level reset
            self.task.reset(env=self,  env_ids=env_ids)

    ############################################################
    # others
    ############################################################

    def set_task(self, task_name: str, **kwargs):
        # set task
        assert task_name in self.tasks, f"Task {task_name} not found in environment tasks."
        self.task_name = task_name
        self.task = self.tasks[task_name]

        # set observation space for the selected agent/task and refresh gym spaces if needed
        assert task_name in self.observasion_spaces, f"Observation space for task {task_name} not found."
        new_obs_space = self.observasion_spaces[task_name]
        if self.cfg.observation_space != new_obs_space:
            self.cfg.observation_space = new_obs_space
            # rebuild spaces so downstream agents see the updated shape
            self._configure_gym_env_spaces()

        self.task.prepare(**kwargs)

    def unset_task(self):
        self.task_name = None
        self.task = None

    def set_object(self, object_name: str):
        # set object
        assert object_name in self.objects, f"Object {object_name} not found in environment objects."
        self.object = self.objects[object_name]
        self.data_point_cloud = self.object_point_clouds[object_name]

    def unset_object(self):
        self.object = None
        self.data_point_cloud = None
    
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
