from datetime import datetime
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Basic paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSET_ROOT = PROJECT_ROOT / "assets"


@configclass
class BaseEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s: float = 10.0
    decimation: int = 4  # Number of simulation steps per control command
    action_space: int = 7
    observation_space: int = 80
    observation_space_pick: int = 80
    observation_space_place: int = 94
    observation_space_pose: int = 94
    state_space: int = 0

    enable_pick = False
    enable_place = False
    enable_pose = False
    enable_open = False
    enable_close = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,  # simulation frequency
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=False,
    )

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str((ASSET_ROOT / "franka_gripper/usd/franka_gripper.usd").resolve()),
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
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=5e3,
                damping=1e2,
            ),
        },
    )

    # # objects
    # objects: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Object",
    #     spawn=sim_utils.MultiAssetSpawnerCfg(
    #         assets_cfg=[],  # NOTE: to be filled when setting up the environment.
    #         mass_props=sim_utils.MassPropertiesCfg(density=500.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_linear_velocity=1000.0,
    #             max_angular_velocity=1000.0,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         random_choice=False,
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 2.0, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        #init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # cameras
    head_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/HeadCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.9, 0.0, 0.8), rot=(0.4135, -0.5736, -0.5736, 0.4135), convention="ros"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 20.0)
        ),
        width=512,
        height=512,
    )
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/WristCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.6, 1.5, 1.3), rot=(-0.1692, -0.1887, 0.7320, -0.6324), convention="ros"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=256,
        height=256,
    )

    # contact forces
    contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_(left|right)finger(_tip|)",
        # filter_prim_paths_expr=["/World/envs/env_.*/Object_.*"],
        update_period=0.0,
        debug_vis=True,
    )

    # OSC controller
    controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    )

    # other scales
    dof_velocity_scale: float = 0.1

    # mode
    collection_mode: bool = False

    # record settings
    timestamp: str = datetime.now().strftime(r"%m%d_%H%M%S")
    save_dir: str = f"out/{timestamp}"

    num_consecutive_success = 2
    lift_height: float = 0.1
