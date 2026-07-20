import torch
import logging
import isaacsim.core.utils.torch as torch_utils

from utils.geometry_utils import to_tensor


def _arti_map_points(pts, joint_type, axis, origin, delta):
    """Map points under a joint's motion (object frame). pts (...,3); axis/origin (3,);
    delta broadcastable to pts[...,:1]. prismatic -> translate by axis*delta; revolute -> rotate
    about the hinge (origin, axis) by delta [rodrigues]. Pass -delta to invert (current->q0)."""
    axis = axis.reshape(*([1] * (pts.dim() - 1)), 3)
    if joint_type == "prismatic":
        return pts + axis * delta
    origin = origin.reshape(*([1] * (pts.dim() - 1)), 3)
    k = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v = pts - origin
    kxv = torch.cross(k.expand_as(v), v, dim=-1)
    kdotv = (v * k).sum(dim=-1, keepdim=True)
    cos = torch.cos(delta)
    sin = torch.sin(delta)
    return v * cos + kxv * sin + k * kdotv * (1.0 - cos) + origin


def _arti_rot_vec(vec, joint_type, axis, delta):
    """Rotate DIRECTION vectors under the joint motion. prismatic -> identity (translation doesn't
    rotate directions); revolute -> rodrigues about axis (no origin)."""
    if joint_type == "prismatic":
        return vec
    zero = torch.zeros(3, device=vec.device, dtype=vec.dtype)
    return _arti_map_points(vec, "revolute", axis, zero, delta)


class PickTask:
    def __init__(self, num_envs, cfg, device):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.consecutive_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        logging.info(f"PickTask initialized with {self.num_envs} environments on device {self.device}")

        self.obj_init_z = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
         # buffers (lazy init)
        self.obs_dist = None
        self.obs_closest_point = None
        self.obs_contact = None
        self.robot_joint_pos = None
        self.wrist_state = None

    def prepare(self, env, **kwargs):
        pass
    
    def reset(self, env, env_ids):
        self.consecutive_success[env_ids] = 0
        env.reset_buf[env_ids] = 0
        env.episode_length_buf[env_ids] = 0
        self.obj_init_z[env_ids] = env.object.data.root_pos_w[env_ids, 2].to(torch.float32)
        self._refresh_buffer(env)

    def update_task_progress(self, env):
        self._refresh_buffer(env)
        num_contact = ((self.obs_contact > 0) & (self.obs_dist[:, :4] < 0.02)).sum(dim=-1)
        obj_z = env.object.data.root_pos_w[:, 2].to(torch.float32)
        lifted = ((obj_z - self.obj_init_z) > float(self.cfg.lift_height)) & (num_contact > 0)
        self.consecutive_success[:] = torch.where(lifted, self.consecutive_success + 1, 0)

    def get_dones(self, env): 
        self.update_task_progress(env)
        self.success = self.consecutive_success >= self.cfg.num_consecutive_success
        truncated = env.episode_length_buf >= env.max_episode_length - 1
        return self.success, truncated

    def get_observations(self, env):
        obs = torch.cat(
            (
                self.robot_joint_pos,
                env.robot.data.joint_vel * self.cfg.dof_velocity_scale,
                env.robot.data.joint_pos_target,
                self.wrist_state,
                env.robot.data.body_pos_w[:, env.left_finger_idx] - env.scene.env_origins,
                env.robot.data.body_pos_w[:, env.right_finger_idx] - env.scene.env_origins,
                env.robot.data.body_pos_w[:, env.grasp_site_idx] - env.scene.env_origins,
                env.object.data.root_pos_w - env.scene.env_origins,
                self.obs_dist,
                self.obs_closest_point.reshape(env.num_envs, -1),
                self.obs_contact,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _refresh_buffer(self, env):
        # Distance Feature
        inv_object_quat, inv_object_pos = torch_utils.tf_inverse(
            env.object.data.root_quat_w, env.object.data.root_pos_w
        )
        link_pos_obj_frame = torch_utils.tf_apply(
            inv_object_quat.unsqueeze(1).expand(-1, len(env.keypoint_indices), -1),
            inv_object_pos.unsqueeze(1),
            env.robot.data.body_pos_w[:, env.keypoint_indices, :],
        )

        pcd = getattr(env, "data_point_cloud_obs_reward", None)
        if pcd is None:
            raise RuntimeError(
                "[PickTask._refresh_buffer] env.data_point_cloud_obs_reward is None. "
                "PickTask refuses to silently fall back to full cloud — call "
                "simulator.reset(... use_top_pcd_for_obs=..., use_handle_pcd_for_obs=...) or "
                "simulator.setup_task_for_slot(... use_top_pcd_for_obs=..., use_handle_pcd_for_obs=...) "
                "before stepping so set_obs_pcd_mode() picks the right cloud (full, top, or handle)."
            )
        if pcd.ndim == 2:
            pcd = pcd.unsqueeze(0).expand(env.num_envs, -1, -1)  # (N, P, 3)
        pcd = pcd.to(dtype=link_pos_obj_frame.dtype, device=link_pos_obj_frame.device)

        distance = torch.cdist(link_pos_obj_frame, pcd)  # (num_envs, num_query, point_cloud_size)
        self.obs_dist, min_indices = distance.min(dim=-1)
        closest_direction = (
            torch.gather(pcd, dim=1, index=min_indices.unsqueeze(-1).expand(-1, -1, 3))
            - link_pos_obj_frame
        )  # (num_envs, num_query, 3)
        self.obs_closest_point = torch_utils.quat_apply(
            env.object.data.root_quat_w.unsqueeze(1).expand(-1, self.obs_dist.shape[1], -1),
            closest_direction,
        )

        # Contact Forces
        force = env.contact.data.net_forces_w
        self.obs_contact = force.squeeze(1).norm(dim=-1) > 0.01

        # Others
        self.robot_joint_pos = torch_utils.scale_transform(
            env.robot.data.joint_pos, env.robot_dof_lower_limits, env.robot_dof_upper_limits
        )
        self.wrist_state = env.robot.data.body_state_w[:, env.hand_link_idx]
        self.wrist_state[:, :3] -= env.scene.env_origins


class PlaceTask:
    def __init__(self, num_envs, cfg, device):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.consecutive_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.goal_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.diff_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)

        logging.info(f"PlaceTask initialized with {self.num_envs} environments on device {self.device}")

    def prepare(self, env, **kwargs):
        self.goal_position: torch.Tensor = to_tensor(kwargs["goal_position"], self.device)
        if self.goal_position.ndim == 1:
            self.goal_position = self.goal_position.unsqueeze(0)
        self.goal_position = self.goal_position.expand(self.num_envs, -1).to(self.goal_pose.dtype)

        self.object_original_pose: torch.Tensor = to_tensor(kwargs["object_original_pose"], self.device)
        if self.object_original_pose.ndim == 1:
            self.object_original_pose = self.object_original_pose.unsqueeze(0)
        self.object_original_pose = self.object_original_pose.expand(self.num_envs, -1).to(self.goal_pose.dtype)

    def reset(self, env, env_ids):
        self.goal_pose[env_ids, 3:] = self.object_original_pose[env_ids, 3:]
        self.goal_pose[env_ids, :3] = self.goal_position[env_ids]

        self.consecutive_success[env_ids] = 0
        env.reset_buf[env_ids] = 0
        env.episode_length_buf[env_ids] = 0

        self._refresh_buffer(env)

    def update_task_progress(self, env):
        POS_TOL = 0.03
        ROT_TOL = 0.3
        self._refresh_buffer(env)
        # Goal position
        # X, Y
        diff_object_pos = self.diff_pose[:, :2].norm(dim=-1)
        # Goal orientation
        diff_object_rot = 2 * self.diff_pose[:, -1].abs().clip(0, 1).arccos()
        rot_ok = diff_object_rot < ROT_TOL
        # Reach
        reached = (
            (diff_object_pos < POS_TOL)
            & (self.diff_pose[:, 2] > -0.05)
            & (env.object.data.root_lin_vel_w.norm(dim=-1) < 0.1)
            # & rot_ok
        )
        # Success
        self.consecutive_success[:] = torch.where(reached, self.consecutive_success + 1, 0)

    def get_dones(self, env): 
        self.update_task_progress(env)
        self.success = self.consecutive_success >= self.cfg.num_consecutive_success
        truncated = env.episode_length_buf >= env.max_episode_length - 1
        return self.success, truncated

    def get_observations(self, env):
        obs = torch.cat(
            (
                self.robot_joint_pos,
                env.robot.data.joint_vel * self.cfg.dof_velocity_scale,
                env.robot.data.joint_pos_target,
                self.wrist_state,
                env.robot.data.body_pos_w[:, env.left_finger_idx] - env.scene.env_origins,
                env.robot.data.body_pos_w[:, env.right_finger_idx] - env.scene.env_origins,
                env.robot.data.body_pos_w[:, env.grasp_site_idx] - env.scene.env_origins,
                env.object.data.root_pos_w - env.scene.env_origins,
                self.obs_dist,
                self.obs_closest_point.reshape(self.num_envs, -1),
                self.obs_contact,
                self.goal_pose,
                self.diff_pose,
            ),
            dim=-1,
        )
        return {"policy": obs}
    
    def _refresh_buffer(self, env):
        # Distance Feature
        inv_object_quat, inv_object_pos = torch_utils.tf_inverse(
            env.object.data.root_quat_w, env.object.data.root_pos_w
        )
        link_pos_obj_frame = torch_utils.tf_apply(
            inv_object_quat.unsqueeze(1).expand(-1, len(env.keypoint_indices), -1),
            inv_object_pos.unsqueeze(1),
            env.robot.data.body_pos_w[:, env.keypoint_indices, :],
        )
        pcd = env.data_point_cloud
        if pcd.ndim == 2:
            pcd = pcd.unsqueeze(0).expand(env.num_envs, -1, -1)  # (N, P, 3)
        pcd = pcd.to(dtype=link_pos_obj_frame.dtype, device=link_pos_obj_frame.device)

        distance = torch.cdist(link_pos_obj_frame, pcd)  # (num_envs, num_query, point_cloud_size)
        self.obs_dist, min_indices = distance.min(dim=-1)
        closest_direction = (
            torch.gather(pcd, dim=1, index=min_indices.unsqueeze(-1).expand(-1, -1, 3))
            - link_pos_obj_frame
        )  # (num_envs, num_query, 3)
        self.obs_closest_point: torch.Tensor = torch_utils.quat_apply(
            env.object.data.root_quat_w.unsqueeze(1).expand(-1, self.obs_dist.shape[1], -1),
            closest_direction,
        )

        # Contact Forces
        force = env.contact.data.net_forces_w
        self.obs_contact = force.squeeze(1).norm(dim=-1) > 0.01

        # Others
        self.robot_joint_pos = torch_utils.scale_transform(
            env.robot.data.joint_pos, env.robot_dof_lower_limits, env.robot_dof_upper_limits
        )
        self.diff_pose[:, :3] = env.scene.env_origins + self.goal_pose[:, :3] - env.object.data.root_pos_w
        self.diff_pose[:, 3:] = torch_utils.quat_mul(
            self.goal_pose[:, 3:], torch_utils.quat_conjugate(env.object.data.root_quat_w)
        )
        self.wrist_state = env.robot.data.body_state_w[:, env.hand_link_idx]
        self.wrist_state[:, :3] -= env.scene.env_origins


class PoseTask:
    def __init__(self, num_envs, cfg, device):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.consecutive_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.goal_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.diff_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)

        logging.info(f"PoseTask initialized with {self.num_envs} environments on device {self.device}")

    def prepare(self, env, **kwargs):
        self.goal_pose = to_tensor(kwargs["goal_pose"], self.device)
        if self.goal_pose.ndim == 1:
            self.goal_pose = self.goal_pose.unsqueeze(0)
        self.goal_pose = self.goal_pose.expand(self.num_envs, -1)
    
    def reset(self, env, env_ids):
        self.consecutive_success[env_ids] = 0
        env.reset_buf[env_ids] = 0
        env.episode_length_buf[env_ids] = 0
        self._refresh_buffer(env)

    def update_task_progress(self, env):
        POS_TOL = 0.03
        ROT_TOL = 0.3

        self._refresh_buffer(env)

        # Goal position
        diff_object_pos = self.diff_pose[:, :3].norm(dim=-1)
        # Goal orientation
        diff_object_rot = 2 * self.diff_pose[:, -1].abs().clip(0, 1).arccos()

        reached = (diff_object_pos < POS_TOL) & (diff_object_rot < ROT_TOL)
        self.consecutive_success[:] = torch.where(reached, self.consecutive_success + 1, 0)

    def get_dones(self, env): 
        self.update_task_progress(env)
        self.success = self.consecutive_success >= self.cfg.num_consecutive_success
        truncated = env.episode_length_buf >= self.max_episode_length - 1
        return self.success, truncated

    def get_observations(self, env):
        obs = torch.cat(
            (
                self.robot_joint_pos,
                env.robot.data.joint_vel * self.cfg.dof_velocity_scale,
                env.robot.data.joint_pos_target,
                self.wrist_state,
                env.robot.data.body_pos_w[:, env.left_finger_idx] - env.scene.env_origins,
                env.robot.data.body_pos_w[:, env.right_finger_idx] - env.scene.env_origins,
                env.robot.data.body_pos_w[:, env.grasp_site_idx] - env.scene.env_origins,
                env.object.data.root_pos_w - env.scene.env_origins,
                self.obs_dist,
                self.obs_closest_point.reshape(self.num_envs, -1),
                self.obs_contact,
                self.goal_pose,
                self.diff_pose,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _refresh_buffer(self, env):
        # Distance Feature
        inv_object_quat, inv_object_pos = torch_utils.tf_inverse(
            env.object.data.root_quat_w, env.object.data.root_pos_w
        )
        link_pos_obj_frame = torch_utils.tf_apply(
            inv_object_quat.unsqueeze(1).expand(-1, len(env.keypoint_indices), -1),
            inv_object_pos.unsqueeze(1),
            env.robot.data.body_pos_w[:, env.keypoint_indices, :],
        )
        pcd = env.data_point_cloud
        if pcd.ndim == 2:
            pcd = pcd.unsqueeze(0).expand(env.num_envs, -1, -1)  # (N, P, 3)
        pcd = pcd.to(dtype=link_pos_obj_frame.dtype, device=link_pos_obj_frame.device)

        distance = torch.cdist(link_pos_obj_frame, pcd)  # (num_envs, num_query, point_cloud_size)
        self.obs_dist, min_indices = distance.min(dim=-1)
        closest_direction = (
            torch.gather(pcd, dim=1, index=min_indices.unsqueeze(-1).expand(-1, -1, 3))
            - link_pos_obj_frame
        )  # (num_envs, num_query, 3)
        self.obs_closest_point: torch.Tensor = torch_utils.quat_apply(
            env.object.data.root_quat_w.unsqueeze(1).expand(-1, self.obs_dist.shape[1], -1),
            closest_direction,
        )

        # Contact Forces
        force = env.contact.data.net_forces_w
        self.obs_contact = force.squeeze(1).norm(dim=-1) > 0.01

        # Others
        self.robot_joint_pos = torch_utils.scale_transform(
            env.robot.data.joint_pos, env.robot_dof_lower_limits, env.robot_dof_upper_limits
        )
        self.diff_pose[:, :3] = env.scene.env_origins + self.goal_pose[:, :3] - env.object.data.root_pos_w
        self.diff_pose[:, 3:] = torch_utils.quat_mul(
            self.goal_pose[:, 3:], torch_utils.quat_conjugate(env.object.data.root_quat_w)
        )
        self.wrist_state = env.robot.data.body_state_w[:, env.hand_link_idx]
        self.wrist_state[:, :3] -= env.scene.env_origins


class _ArtiPartTask:
    """Shared open/close drawer/door task. See docs/arti/obs_and_task.md.

    Obs (87-dim) is reproduced bit-for-bit from IsaacLabEnvs (open_base_env/close_env).
    Reward is training-only and NOT used here (no-op). The MCTS path needs get_observations +
    get_dones (success + failure). Operates on the SELECTED movable part of an articulation.

    The selected part is wired in prepare() (the standard set_task -> task.prepare path,
    same as PlaceTask's goal kwargs) from the unified [name][part] meta store:
    joint topology from env.object_joints, obs point sets from
    env.object_handle_point_clouds / env.object_link_forward_point_clouds (q0/object frame).
    Which point set feeds obs is the task class's own knowledge (obs_kind: open -> handle;
    close -> handle U link_forward, center = link_forward mean). env.object (the cabinet
    Articulation) comes from set_object.
    """

    sign = 1.0          # axis-dir sign: open -> +drawer_goal_dir ; close -> -(=push_dir)
    goal_attr = None    # cfg attr name for goal_joint_pos_norm
    goal_default = 0.0
    failure_kind = None      # "grasp" (open) | "detach" (close)
    obs_kind = "open"        # which q0 point set feeds obs: "open" | "close"

    def __init__(self, num_envs, cfg, device):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.consecutive_success = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failure = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # progress / failure bookkeeping
        self.peak_norm = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._engaged_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._lost_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        # obs buffers (lazy)
        self.obs_dist = None
        self.obs_closest_point = None
        self.obs_contact = None
        self.robot_joint_pos = None
        self.wrist_state = None
        self.object_joint_pos_norm = None
        self.object_joint_vel_norm = None
        self.object_joint_error_norm = None
        self.goal_joint_pos_norm = None
        self.center_world = None
        self.axis_dir_world = None
        logging.info(f"{type(self).__name__} initialized with {num_envs} envs on {device}")

    def prepare(self, env, object_name: str = None, part: str = None, **kwargs):
        """Wire the selected movable part's obs inputs from the unified meta store onto the
        TASK (replaces the former out-of-band setup_drawer_obs that poked env.drawer_*
        attributes). dof/link indices are ensured by the simulator reset path before set_task."""
        if object_name is None or part is None:
            raise ValueError(f"{type(self).__name__}.prepare needs object_name and part kwargs.")
        j = env.object_joints[object_name][part]
        if j["dof_idx"] is None:
            raise RuntimeError(
                f"{object_name!r} part {part!r}: dof_idx unresolved before task prepare "
                f"(the reset path must call _ensure_part_indices first)."
            )
        self.dof_idx = int(j["dof_idx"])
        self.joint_type = j["joint_type"]
        # Joint axis/origin in object frame (from the asset URDF) drive the prismatic vs
        # revolute transforms below. origin is the hinge pivot (revolute); unused for prismatic.
        self.joint_axis_obj = torch.tensor(j["axis_obj"], device=env.device, dtype=torch.float32)
        self.joint_origin_obj = torch.tensor(j["origin_obj"], device=env.device, dtype=torch.float32)
        handle = env.object_handle_point_clouds[object_name][part]
        if self.obs_kind == "close":
            lf = env.object_link_forward_point_clouds.get(object_name, {}).get(part)
            if lf is None:
                raise RuntimeError(f"{object_name!r} part {part!r}: close obs needs link_forward_points.")
            self.query_points_q0 = torch.cat([handle, lf], dim=0)
            self.center_q0 = lf.mean(dim=0)
        else:
            self.query_points_q0 = handle
            self.center_q0 = handle.mean(dim=0)

    # --------------------------------------------------------------- config helpers
    def _goal_norm(self):
        return float(getattr(self.cfg, self.goal_attr, self.goal_default)) if self.goal_attr else self.goal_default

    def _part_jstate(self, env):
        """(q, qd, lower, upper) for the selected drawer dof, each (N,)."""
        art = env.object
        dof = int(self.dof_idx)
        q = art.data.joint_pos[:, dof]
        qd = art.data.joint_vel[:, dof]
        lower = art.data.joint_pos_limits[:, dof, 0]
        upper = art.data.joint_pos_limits[:, dof, 1]
        return q, qd, lower, upper

    def _part_keypoints(self, env):
        """Explicit RL keypoint order [left_finger, right_finger, left_tip, right_tip, grasp_site,
        hand] (open_base_env/close_env), cached. NOT env.keypoint_indices (contact.body_names order)."""
        cached = getattr(self, "_kp_indices", None)
        if cached is None:
            cached = [env.left_finger_idx, env.right_finger_idx, env.left_tip_idx,
                      env.right_tip_idx, env.grasp_site_idx, env.hand_link_idx]
            self._kp_indices = cached
        return cached

    def _contact_perm(self, env):
        """Permutation of the regex contact sensor's bodies to [left_finger, right_finger, left_tip,
        right_tip] (RL's explicit obs_contact order), cached."""
        cached = getattr(self, "_contact_perm_cache", None)
        if cached is None:
            names = list(env.contact.body_names)
            want = ["panda_leftfinger", "panda_rightfinger", "panda_leftfinger_tip", "panda_rightfinger_tip"]
            missing = [w for w in want if w not in names]
            if missing:
                raise RuntimeError(f"contact sensor missing bodies {missing}; has {names}")
            cached = [names.index(w) for w in want]
            self._contact_perm_cache = cached
        return cached

    # --------------------------------------------------------------- reset
    def reset(self, env, env_ids):
        # NOTE: the articulation joint INIT STATE is NOT set here — it is applied at ENV reset from
        # the layout npz (base_env._reset_idx; missing cols -> 0/closed). Task reset only resets
        # bookkeeping and snapshots the (already-set) opening as the peak. See docs/arti/scene_import.md.
        self.consecutive_success[env_ids] = 0
        self.failure[env_ids] = False
        self.peak_norm[env_ids] = 0.0
        self._engaged_once[env_ids] = False
        self._lost_steps[env_ids] = 0
        env.reset_buf[env_ids] = 0
        env.episode_length_buf[env_ids] = 0
        self._refresh_buffer(env)
        self.peak_norm[env_ids] = self.object_joint_pos_norm[env_ids]

    # --------------------------------------------------------------- progress / dones
    def update_task_progress(self, env):
        self._refresh_buffer(env)
        norm = self.object_joint_pos_norm
        # peak = the extreme reached so far in the task direction (open: most-open; close: most-closed).
        self.peak_norm = torch.maximum(self.peak_norm, norm) if self.sign > 0 else torch.minimum(self.peak_norm, norm)
        goal = self._goal_norm()
        # success: open -> norm >= goal ; close -> norm <= goal
        reached = (norm >= goal) if self.sign > 0 else (norm <= goal)
        self.consecutive_success[:] = torch.where(reached, self.consecutive_success + 1, 0)

    def _compute_pinch(self):
        """Two-sided pinch flag, mirroring IsaacLabEnvs open_base_env._compute_num_contact_and_pinch.
        obs_contact/obs_dist cols: [0]=left_finger [1]=right_finger [2]=left_tip [3]=right_tip.
        contact_use_tips_only -> use the two fingertip cols only (else finger|tip per side)."""
        c = self.obs_contact
        d = self.obs_dist
        if bool(getattr(self.cfg, "part_contact_use_tips_only", False)):
            left_contact, right_contact = c[:, 2], c[:, 3]
            left_near = d[:, 2] < 0.02
            right_near = d[:, 3] < 0.02
        else:
            left_contact = c[:, 0] | c[:, 2]
            right_contact = c[:, 1] | c[:, 3]
            left_near = torch.minimum(d[:, 0], d[:, 2]) < 0.02
            right_near = torch.minimum(d[:, 1], d[:, 3]) < 0.02
        return left_contact & right_contact & left_near & right_near

    def _update_failure(self, env):
        # regress: progress reversed from peak by > regress_fail_norm. Direction-aware:
        # open -> fell back closed (peak_max - cur); close -> popped back open (cur - peak_min).
        regress_thr = float(getattr(self.cfg, "part_regress_fail_norm", 0.0))
        if regress_thr > 0.0:
            regressed = (self.peak_norm - self.object_joint_pos_norm) if self.sign > 0 \
                else (self.object_joint_pos_norm - self.peak_norm)
            self.failure |= regressed > regress_thr
        if self.failure_kind == "grasp":
            # open: grasp-lost. pinch = two-sided contact + near (mirrors IsaacLabEnvs
            # open_base_env._compute_num_contact_and_pinch).
            fail_steps = int(getattr(self.cfg, "drawer_grasp_lost_fail_steps", 10))
            engaged = self._compute_pinch()
        else:
            # close: detach. engaged = gripper near the drawer (closest keypoint dist <= detach_dist).
            fail_steps = int(getattr(self.cfg, "part_detach_fail_steps", 15))
            detach_dist = float(getattr(self.cfg, "part_detach_dist", 0.05))
            engaged = self.obs_dist.min(dim=-1).values <= detach_dist
        if fail_steps > 0:
            self._engaged_once |= engaged
            lost = self._engaged_once & (~engaged)
            self._lost_steps = torch.where(lost, self._lost_steps + 1, torch.zeros_like(self._lost_steps))
            self.failure |= self._lost_steps >= fail_steps

    def get_dones(self, env):
        self.update_task_progress(env)
        self._update_failure(env)
        self.success = self.consecutive_success >= self.cfg.num_consecutive_success
        truncated = env.episode_length_buf >= env.max_episode_length - 1
        return self.success | self.failure, truncated

    def compute_reward(self, env):
        return None  # training-only; not used in MCTS execution

    # --------------------------------------------------------------- observation
    def get_observations(self, env):
        base = torch.cat(
            (
                self.robot_joint_pos,
                env.robot.data.joint_vel * self.cfg.dof_velocity_scale,
                env.robot.data.joint_pos_target,
                self.wrist_state,
                env.robot.data.body_pos_w[:, env.left_finger_idx] - env.scene.env_origins,
                env.robot.data.body_pos_w[:, env.right_finger_idx] - env.scene.env_origins,
                env.robot.data.body_pos_w[:, env.grasp_site_idx] - env.scene.env_origins,
                self.center_world,                                   # field 8
                self.obs_dist,                                       # 6
                self.obs_closest_point.reshape(env.num_envs, -1),    # 18
                self.obs_contact,                                    # 4
                self.object_joint_pos_norm.unsqueeze(-1),
                self.goal_joint_pos_norm.unsqueeze(-1),
                self.object_joint_error_norm.unsqueeze(-1),
                self.object_joint_vel_norm.unsqueeze(-1),
            ),
            dim=-1,
        )                                                            # 84 base
        obs = torch.cat((base, self._obs_tail(env)), dim=-1)
        return {"policy": obs}

    def _obs_tail(self, env):
        """Trailing obs dims. Drawer (prismatic): the axis dir (open: goal_dir; close: push_dir),
        3 dims -> 87. CloseDoorTask overrides with the revolute door's 9-dim direction block -> 93."""
        return self.axis_dir_world

    def _refresh_buffer(self, env):
        art = env.object
        root_quat = art.data.root_quat_w
        root_pos = art.data.root_pos_w
        q, qd, lower, upper = self._part_jstate(env)
        denom = (upper - lower).clamp_min(1e-6)
        joint_delta = q - lower                                     # (N,) opening (prismatic m / revolute rad)
        jtype = self.joint_type
        axis = self.joint_axis_obj.to(root_quat)                   # (3,) object frame
        origin = self.joint_origin_obj.to(root_quat)               # (3,) hinge pivot (revolute)

        # RL drawer/door use an EXPLICIT keypoint order [left_finger, right_finger, left_tip,
        # right_tip, grasp_site, hand] (open_base_env/close_env), NOT env.keypoint_indices (which
        # is the base contact.body_names order that pick uses). Match RL exactly.
        kp_indices = self._part_keypoints(env)
        inv_q, inv_p = torch_utils.tf_inverse(root_quat, root_pos)
        K = len(kp_indices)
        kp_obj = torch_utils.tf_apply(
            inv_q.unsqueeze(1).expand(-1, K, -1),
            inv_p.unsqueeze(1),
            env.robot.data.body_pos_w[:, kp_indices, :],
        )                                                          # (N,K,3)
        query_q0 = _arti_map_points(kp_obj, jtype, axis, origin, -joint_delta.view(-1, 1, 1))

        qp = self.query_points_q0
        if qp.ndim == 2:
            qp = qp.unsqueeze(0).expand(env.num_envs, -1, -1)
        qp = qp.to(dtype=query_q0.dtype, device=query_q0.device)
        distance = torch.cdist(query_q0, qp)                       # (N,K,P)
        self.obs_dist, min_idx = distance.min(dim=-1)             # (N,K)
        closest_dir_q0 = (
            torch.gather(qp, 1, min_idx.unsqueeze(-1).expand(-1, -1, 3)) - query_q0
        )                                                          # (N,K,3) q0 frame
        # rotate the direction q0 -> current (revolute), then to world
        closest_dir_cur = _arti_rot_vec(closest_dir_q0, jtype, axis, joint_delta.view(-1, 1, 1))
        self.obs_closest_point = torch_utils.quat_apply(
            root_quat.unsqueeze(1).expand(-1, K, -1), closest_dir_cur
        )                                                          # world frame

        # contact: reorder the single regex sensor's bodies to RL's explicit
        # [left_finger, right_finger, left_tip, right_tip] order (matches obs_dist/obs_contact cols).
        perm = self._contact_perm(env)
        force = env.contact.data.net_forces_w                      # (N, B, 3) in contact.body_names order
        self.obs_contact = force[:, perm, :].norm(dim=-1) > 0.01   # (N,4) explicit order

        # field-8 center (on the movable link -> q0->current), -> world (env-local)
        center_cur_obj = _arti_map_points(
            self.center_q0.to(root_quat).view(1, 3), jtype, axis, origin, joint_delta.view(-1, 1)
        )                                                          # (N,3)
        self.center_world = torch_utils.quat_apply(root_quat, center_cur_obj) + root_pos - env.scene.env_origins

        # axis dir in world (drawer tail: open goal_dir / close push_dir = -goal_dir). Door overrides tail.
        goal_dir = torch_utils.quat_apply(root_quat, axis.view(1, 3).expand(env.num_envs, -1))
        goal_dir = goal_dir / goal_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.axis_dir_world = self.sign * goal_dir

        # joint norms
        self.object_joint_pos_norm = ((q - lower) / denom).clamp(0.0, 1.0)
        self.object_joint_vel_norm = qd / denom
        self.goal_joint_pos_norm = torch.full_like(self.object_joint_pos_norm, self._goal_norm())
        self.object_joint_error_norm = self.goal_joint_pos_norm - self.object_joint_pos_norm

        # robot joints + wrist (same as PickTask)
        self.robot_joint_pos = torch_utils.scale_transform(
            env.robot.data.joint_pos, env.robot_dof_lower_limits, env.robot_dof_upper_limits
        )
        self.wrist_state = env.robot.data.body_state_w[:, env.hand_link_idx]
        self.wrist_state[:, :3] -= env.scene.env_origins


class OpenDrawerTask(_ArtiPartTask):
    sign = 1.0
    goal_attr = "drawer_open_goal_norm"
    goal_default = 0.95
    failure_kind = "grasp"         # grasp-lost failure


class CloseDrawerTask(_ArtiPartTask):
    sign = -1.0                     # push_dir = -drawer_goal_dir
    obs_kind = "close"
    goal_attr = "drawer_close_goal_norm"
    goal_default = 0.0
    failure_kind = "detach"        # detach failure


class CloseDoorTask(_ArtiPartTask):
    """Revolute door, push-to-close. Obs = 84 base + 9 door direction terms (axis_w, radial_dir,
    close_tangent_dir) = 93, mirroring IsaacLabEnvs CloseDoorEnv. See docs/arti/obs_and_task.md."""
    sign = -1.0                     # close: success = norm <= goal; peak = most-closed
    obs_kind = "close"
    goal_attr = "door_close_goal_norm"
    goal_default = 0.0
    failure_kind = "detach"        # push-only -> detach failure (no grasp)

    def _obs_tail(self, env):
        # Revolute-door geometry (mirror CloseDoorEnv._compute_direction_terms), all in world frame.
        art = env.object
        root_quat = art.data.root_quat_w
        axis = self.joint_axis_obj.to(root_quat)
        origin = self.joint_origin_obj.to(root_quat)
        axis_w = torch_utils.quat_apply(root_quat, axis.view(1, 3).expand(env.num_envs, -1))
        axis_w = axis_w / axis_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        hinge_local = (
            torch_utils.quat_apply(root_quat, origin.view(1, 3).expand(env.num_envs, -1))
            + art.data.root_pos_w - env.scene.env_origins
        )
        radial = self.center_world - hinge_local                  # center_world already env-local
        radial = radial - (radial * axis_w).sum(dim=-1, keepdim=True) * axis_w
        radial_dir = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        open_tangent = torch.cross(axis_w, radial_dir, dim=-1)
        open_tangent = open_tangent / open_tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        close_tangent_dir = -open_tangent
        return torch.cat((axis_w, radial_dir, close_tangent_dir), dim=-1)  # 9
