import os
from datetime import datetime
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Basic paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSET_ROOT = PROJECT_ROOT / "assets"

# Root of the downloadable SkillWeaver data bundle (scenes / assets / ckpts / misc).
# Set the SKILLWEAVER_DATA_ROOT env var to wherever you extracted it; see README.
SKILLWEAVER_DATA_ROOT = os.environ.get("SKILLWEAVER_DATA_ROOT", "your_path_to_skillweaver_root")
# The prefix scene JSONs were authored with; rewritten to scene_asset_root_path at load.
LEGACY_SCENE_ASSET_ROOT = "/data/group_data/katefgroup-ssd/sim_scene_gen"
# Mean values from IsaacLabEnvs EventCfg (friction 0.6-1.2, mass scale 0.4-2.0).
EVENT_MEAN_STATIC_FRICTION = 1.0
EVENT_MEAN_DYNAMIC_FRICTION = 1.0
EVENT_MEAN_RESTITUTION = 0.0
EVENT_MEAN_MASS_SCALE = 1.0


# SimplerEnv put_eggplant_in_basket (sink) init qpos — measured live from the
# benchmark env (agent.robot.get_qpos() after reset). Differs from the table
# pose below on all 6 arm joints (max 0.338 rad); fingers identical. Selected
# via env yaml `robot_init_qpos_preset: widowx_sink` (see make_env).
WIDOWX_SINK_INIT_QPOS = {
    "waist": -0.26006001,
    "shoulder": -0.12876,
    "elbow": 0.04461,
    "forearm_roll": -0.00653,
    "wrist_angle": 1.70334005,
    "wrist_rotate": -0.26982999,
    "left_finger": 0.037,
    "right_finger": 0.037,
}

# WidowX-250 (SimplerEnv bridge tasks). Used by SELECTING it in make_env when
# robot_name=widowx (see simulators/isaaclab.py); BaseEnvCfg.robot below stays
# Franka by default, so the LIBERO/Franka path is untouched.
# - USD: converted from wx250s.urdf via IsaacLabEnvs urdf_to_usd (Part 0.A).
# - init_state.joint_pos: SimplerEnv widowx default qpos (table-task setup).
# - actuators: PD gains synced to IsaacLabEnvs WidowX RL training
#   (tasks/widowx/base_env_cfg.py) so trained policies roll out under the
#   same dynamics; see the note at the actuators block below.
WIDOWX_ROBOT_CFG: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str((ASSET_ROOT / "widowx_gripper/usd/wx250s.usd").resolve()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "waist": -0.01840777,
            "shoulder": 0.0398835,
            "elbow": 0.22242722,
            "forearm_roll": -0.00460194,
            "wrist_angle": 1.36524296,
            "wrist_rotate": 0.00153398,
            "left_finger": 0.037,
            "right_finger": 0.037,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # Gains synced to the IsaacLabEnvs WidowX RL training cfg
        # (tasks/widowx/base_env_cfg.py) -- the pick policy was trained under
        # these dynamics, so rollout must match. (Old SAPIEN-ported stiffness
        # 730-1273 saturated the effort limits at ~1 deg of error.)
        "widowx_arm": ImplicitActuatorCfg(
            joint_names_expr=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
            effort_limit_sim=20.0,
            stiffness=200.0,
            damping=10.0,
        ),
        "widowx_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger", "right_finger"],
            effort_limit_sim=15.0,
            stiffness=500.0,
            damping=20.0,
        ),
    },
)


@configclass
class BaseEnvCfg(DirectRLEnvCfg):
    # Global root for all asset paths from scene description (relative refs are joined
    # with this; legacy-absolute refs are rewritten onto it at load — see base_env.py).
    scene_asset_root_path: str = os.path.join(SKILLWEAVER_DATA_ROOT, "sim_scene_gen")

    # env
    episode_length_s: float = 10.0
    decimation: int = 25  # Number of simulation steps per control command
    action_space: int = 7
    action_ema_alpha: float = 0.92
    observation_space: int = 80
    observation_space_pick: int = 80
    observation_space_place: int = 94
    observation_space_pose: int = 94
    observation_space_open_drawer: int = 87
    observation_space_close_drawer: int = 87
    observation_space_close_door: int = 93   # 84 base + 9 revolute-door direction terms
    state_space: int = 0
    scene_desc_file: str | None = None
    init_arm_pos_files: list[str] = [
        "/home/hez2/code/Grounded_MCTS_IsaacLab/data/init_arm_qpos_cano_aug.npz",
        "/home/hez2/code/Grounded_MCTS_IsaacLab/data/init_arm_qpos_cano_all_init.npz",
        "/home/hez2/code/Grounded_MCTS_IsaacLab/data/init_arm_qpos_xy0.05.npz",
    ]

    enable_pick = False
    enable_place = False
    enable_pose = False
    enable_open_drawer = False
    enable_close_drawer = False
    enable_close_door = False

    # open/close drawer task params (see docs/arti/obs_and_task.md). Defaults aligned to the
    # IsaacLabEnvs training scripts. Goal is the normalized drawer opening (0=closed,1=open).
    # NOTE: the joint INIT opening is NOT a cfg param — it comes from the layout npz at env reset
    # (base_env._reset_idx; missing cols -> 0/closed).
    drawer_open_goal_norm: float = 0.95     # open success: pos_norm >= this
    drawer_close_goal_norm: float = 0.0     # close success: pos_norm <= this
    part_regress_fail_norm: float = 0.3   # fail if opening regresses > this from peak (0=off)
    drawer_grasp_forced_steps: int = 10     # open: Stage-A.5 forced jaw-close steps onto the handle
    drawer_grasp_lost_fail_steps: int = 10  # open: fail if grasp (pinch) lost this many steps
    part_contact_use_tips_only: bool = False  # pinch: True -> fingertip cols only (RL default False)
    part_detach_fail_steps: int = 15      # close: fail if detached this many steps
    # Release execution: False (default) = pure PD drive tracking (bounded contact forces;
    # grazing the placed object nudges it instead of ejecting it). True = legacy teleport
    # (write_joint_state_to_sim per step) -- the old workaround for the "gripper never
    # opens" bug whose root cause (release drive targets never applied) is now fixed;
    # keep as an escape hatch in case sticky-release resurfaces.
    release_teleop_joint_state: bool = False

    # 0.1 = the close_door/close_drawer training-run value (params/env.yaml). 0.05 falsely
    # flagged the 5-10cm re-approach phase of the 120-deg door push as detached and killed
    # the episode mid-push.
    part_detach_dist: float = 0.1         # close: "engaged" if nearest keypoint within this
    # close_door (revolute, push) reuses part_regress_fail_norm/detach_fail_steps/detach_dist.
    door_close_goal_norm: float = 0.0       # close-door success: pos_norm <= this

    collect_depth: bool = False
    collect_segmentation: bool = False

    # When True, objects whose category appears in `half_pcd_list` swap their
    # pick-policy obs point cloud for the pre-generated `<stem>_half_pcd.npz`
    # (handle half of the asset, extended by 12.5% L into the head side). The
    # full pcd is still loaded into `object_point_clouds` for grasp-judge,
    # 3d range, and other downstream uses.
    use_half_pcd: bool = False
    half_pcd_list: str = ""

    # randomization
    randomize_xy: bool = False
    randomize_xy_range: float = 0.05
    test_layout_id: list[int] | None = None
    split: str = "train"
    # Sampling masses for the 4 front->back layout quartiles. None = default
    # front-heavy [4, 3, 2, 1]. Must be length 4, all >= 0, with sum > 0.
    layout_sample_weights: list[float] | None = None

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.002,  # simulation frequency
        render_interval=decimation,
        render=sim_utils.RenderCfg(
            rendering_mode="balanced",
            antialiasing_mode="DLAA",
            enable_dl_denoiser=True,
            enable_shadows=True,
            enable_ambient_occlusion=True,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="min",
            static_friction=EVENT_MEAN_STATIC_FRICTION,
            dynamic_friction=EVENT_MEAN_DYNAMIC_FRICTION,
            restitution=EVENT_MEAN_RESTITUTION,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=False,
    )

    # robot
    # Robot identity for name resolution (robot_conventions.ROBOT_NAMES: joint/
    # body names, contact prim expr). Stamped by make_env alongside cfg.robot;
    # "panda" default keeps direct env construction on the legacy names.
    robot_name: str = "panda"

    # Per-task robot init-qpos preset applied in make_env after robot selection.
    # "" = the selected robot cfg's own default; "widowx_sink" = SimplerEnv
    # put_eggplant_in_basket arm pose (WIDOWX_SINK_INIT_QPOS above).
    robot_init_qpos_preset: str = ""

    # Spawn the wrist camera (prim anchors on Robot/wrist_camera_link). Robots
    # without that link (WidowX / bridge has no wrist camera) must disable it.
    enable_wrist_camera: bool = True

    # Gate for the scene json `background_image` overlay composite (SimplerEnv
    # rgb_overlay style, front camera only). Off by default: scenes carrying the
    # field are ignored unless the env yaml opts in (e.g. simpler_sink).
    enable_background_overlay: bool = False

    # Which non-wrist cameras to spawn, injected by make_env from the simulator
    # camera_names list ("front" covers all front_camera_count views). None =
    # spawn everything (legacy behavior, keeps direct env construction working).
    # The wrist camera is governed solely by enable_wrist_camera.
    camera_spawn_names: tuple[str, ...] | None = None

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str((ASSET_ROOT / "franka_gripper/usd/franka_gripper_libero_visual_xml_overlay.usda").resolve()),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.3463,
                "panda_joint2": -0.0387,
                "panda_joint3": -0.3453,
                "panda_joint4": -2.3377,
                "panda_joint5": -0.0176,
                "panda_joint6": 2.3012,
                "panda_joint7": 0.7983,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # table init state (runtime table geometry/size still comes from scene_desc["table"]).
    table_init_pos: tuple[float, float, float] = (0.5, 0.0, 0.05)
    table_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # Table z is sampled uniformly from this range on env reset.
    table_z_range: tuple[float, float] = (0.05, 0.05)
    # XY offset added to each object's layout_pos when placing it in the world.
    # None -> fall back to the table anchor (table_init_pos[:2]), keeping the legacy
    # object<->table coupling (libero/real_world). Set (0,0) to DECOUPLE: objects
    # land at their true base-frame layout_pos while the table prim stays at
    # table_init_pos (used by simpler, whose scene rotated_pos = true base xy).
    object_xy_anchor: tuple[float, float] | None = None
    # Optional cap on table top surface height in world frame.
    max_table_surface_height: float | None = None
    # Optional floor on table top surface height in world frame.
    min_table_surface_height: float | None = None
    # Probability of disabling table on each env-level reset.
    table_disable_prob: float = 0.0
    # When table is disabled, floor surface z is set to ground z + this value.
    no_table_floor_above_ground: float = 0.001
    # Fallback virtual table extents [sx, sy, sz] for scenes without scene_desc["table"].
    no_table_virtual_table_size: tuple[float, float, float] = (1.0, 1.0, 0.001)

    # Optional scene static assets (from scene_desc floor/walls entries).
    # floor center xy in world frame; z is sampled from floor_z_range at reset.
    floor_xy: tuple[float, float] = (0.0, 0.0)
    floor_z_range: tuple[float, float] = (-0.75, -0.2)
    floor_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    floor_rot_bias: tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, 0.70710678)

    wall_height: float = 4.0
    # Uniform scale applied to each wall plane USD (native 4x4). >1 makes the walls
    # taller AND wider to block dome-light/background leakage. None -> native size.
    wall_scale: tuple[float, float, float] | None = None
    wall_front_xy: tuple[float, float] = (2.0, 0.0)
    wall_front_rot: tuple[float, float, float, float] = (0.70710678, 0.0, -0.70710678, 0.0)
    wall_front_rot_bias: tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, -0.70710678)
    wall_back_xy: tuple[float, float] = (-2.0, 0.0)
    wall_back_rot: tuple[float, float, float, float] = (0.70710678, 0.0, 0.70710678, 0.0)
    wall_back_rot_bias: tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, 0.70710678)
    wall_left_xy: tuple[float, float] = (0.0, 2.0)
    wall_left_rot: tuple[float, float, float, float] = (0.70710678, 0.70710678, 0.0, 0.0)
    wall_left_rot_bias: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    wall_right_xy: tuple[float, float] = (0.0, -2.0)
    wall_right_rot: tuple[float, float, float, float] = (0.70710678, -0.70710678, 0.0, 0.0)
    wall_right_rot_bias: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    # Ground plane is spawned at floor_z_range[0] - this offset.
    ground_plane_z_offset_below_floor: float = 0.02
    # Hide the collision ground cuboid from rendering while keeping its collision enabled.
    hide_moveable_ground_visual: bool = True
    # Toggle visual spawning of scene floor/walls independently (collision layer unchanged).
    enable_scene_visual_floor: bool = True
    enable_scene_visual_walls: bool = True
    # Print resolved floor material/texture bindings at scene setup for debugging.
    debug_floor_material_binding: bool = False

    # Dome-light augmentation (applied at env reset).
    enable_dome_light_augmentation: bool = True
    dome_light_intensity_range: tuple[float, float] = (800.0, 2000.0)
    dome_light_yaw_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    dome_light_intensity_aug_prob: float = 1.0
    dome_light_yaw_aug_prob: float = 1.0
    dome_light_texture_aug_prob: float = 0.5
    dome_light_color_aug_prob: float = 0.2
    dome_light_color_r_range: tuple[float, float] = (0.6, 1.4)
    dome_light_color_g_range: tuple[float, float] = (0.6, 1.4)
    dome_light_color_b_range: tuple[float, float] = (0.6, 1.4)

    # cameras
    front_camera_count: int = 1
    front_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/FrontCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.9, 0.0, 0.8), rot=(0.4135, -0.5736, -0.5736, 0.4135), convention="ros"),
         data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 20.0)
        ),
        width=512,
        height=512,
        update_latest_camera_pose=True,
        colorize_semantic_segmentation=False,
    )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/TopCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.5, 0.0, 1.2), rot=(0.0, 0.0, -1.0, 0.0), convention="ros"),
         data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 20.0)
        ),
        width=512,
        height=512,
        colorize_semantic_segmentation=False,
    )
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/wrist_camera_link/WristCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),  # wxyz
            convention="ros",
        ),
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            # focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 20.0)
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=36.8317, clipping_range=(0.01, 20.0)
        ),
        width=256,
        height=256,
        update_latest_camera_pose=True,  # essential for wrist camera
        colorize_semantic_segmentation=False,
    )
    data_collect_camera_names = ["front", "wrist"]

    # Front-camera pose augmentation in spherical coordinates around a fixed center.
    # Ranges accept comma-separated segments, e.g. "0:30,150:180".
    enable_front_camera_pose_augmentation: bool = True
    front_camera_aug_center: tuple[float, float, float] = (0.5, 0.0, 0.05)
    front_camera_aug_radius_ranges: str = "0.8:1.4"
    front_camera_aug_azimuth_ranges_deg: str = "-150.0:-120.0,-70.0:70.0,120.0:150.0"
    front_camera_aug_elevation_ranges_deg: str = "15.0:50.0"
    # Sampling mode for radius/azimuth/elevation within each selected segment.
    # - uniform: flat probability in [lo, hi]
    # - triangular: peak at mode (or segment midpoint when mode is null)
    front_camera_aug_sampling_mode: str = "triangular"
    front_camera_aug_radius_mode: float | None = 1.0
    front_camera_aug_azimuth_mode_deg: float | None = 0.0
    front_camera_aug_elevation_mode_deg: float | None = 30.0
    front_camera_aug_roll_deg: float = 0.0
    # Lookat-Z (camera target Z) augmentation. Empty/None = lock to current
    # table surface Z (legacy behavior). Non-empty = sample lookat Z from these
    # comma-separated segments, overriding the table-surface lock (lets the
    # camera tilt above or below the table top for sim-to-real DR).
    front_camera_aug_lookat_z_ranges: str | None = None  # e.g. "0.03:0.23"
    front_camera_aug_lookat_z_mode: float | None = None

    # Front-camera FoV (horizontal_aperture) augmentation.
    # When enabled, the offline pre-sample script (sample_layout_tablez_camposes.py)
    # bakes a per-(layout, k) aperture into cam_poses.npz (extra 8th dim), and the
    # runtime layout loader writes it to the FrontCamera prim's horizontalAperture
    # attribute. Computed per layout: min aperture is the value that just fits
    # every object_to_fit inside the FoV + a margin; final sample is clipped to
    # `front_camera_fov_aperture_range_mm` and sampled toward the configured mode.
    enable_front_camera_fov_augmentation: bool = False
    front_camera_fov_aperture_range_mm: tuple[float, float] = (18.78, 36.83)
    front_camera_fov_aperture_mode_mm: float | None = None  # None = uniform; else triangular peak
    front_camera_fov_dynamic_min_margin_deg: float = 5.0
    # Comma-separated object names to require inside FoV; "all" = every non-table
    # object in the layout npz; otherwise pass e.g. "apple,banana".
    front_camera_fov_objects_to_fit: str = "all"

    # Wrist-camera mount pose perturbation (sim-to-real DR).
    # Off by default; the baked URDF wrist_camera_mount already defines the
    # nominal pose. When enabled, each episode samples a small offset added
    # to the WristCamera prim's local xform (= OffsetCfg-style), simulating
    # bracket assembly tolerance + Franka hand-eye calibration residual.
    # Ranges are half-widths: uniform sample in [-range, +range] per axis.
    enable_wrist_camera_pose_augmentation: bool = False
    # Discrete (back_m_extra, pitch_deg_extra) combos layered on the baked URDF
    # (which is already at -5cm + 15deg). One combo is sampled ONCE per env
    # construction (= per scene in the typical shell loop) and baked into
    # cfg.wrist_camera.offset before the camera is built. DR comes from the
    # shell running many scenes, each drawing independently. Per-episode
    # randomization is NOT done (the runtime prim-write path had a rendering
    # bug; session-level avoids it).
    # Defaults are the 6 visually-verified combos from snapshot ablation
    # (see docs/wrist_camera_mount_workflow.md):
    #   (0.000, 0.0)  -> total -5cm + 15deg (baseline)
    #   (0.010, 7.0)  -> total -6cm + 22deg
    #   (0.015,10.0)  -> total -6.5cm + 25deg
    #   (0.020,14.0)  -> total -7cm + 29deg
    #   (0.025,17.0)  -> total -7.5cm + 32deg
    #   (0.030,18.0)  -> total -8cm + 33deg
    wrist_camera_aug_combos: tuple = (
        (0.000, 0.0),
        (0.010, 7.0),
        (0.015, 10.0),
        (0.020, 14.0),
        (0.025, 17.0),
        (0.030, 18.0),
    )

    # Per-reset gripper opening randomization (sim-to-real DR).
    # True (default): sample finger joint opening uniformly in [lower, upper]
    # each reset (~ Uniform(0, 0.04) for Panda) and apply the same value to
    # both fingers. False: pin both fingers to the canonical max opening
    # (= max(gripper_upper_limits), ~0.04 for Panda) every reset; no DR.
    enable_gripper_init_randomization: bool = True

    # contact forces
    contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_(left|right)finger(_tip|)",
        # filter_prim_paths_expr=["/World/envs/env_.*/Object_.*"],
        update_period=0.0,
        debug_vis=False,
    )

    # OSC controller
    controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    )

    # other scales
    dof_velocity_scale: float = 0.1
    use_usd_mass: bool = True
    object_density: float = 500.0 * EVENT_MEAN_MASS_SCALE

    num_consecutive_success = 3
    lift_height: float = 0.15
