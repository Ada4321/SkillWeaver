"""End-effector axis conventions, the only geometry that differs between robots.

The pick-orientation construction (skills.py) and the "point gripper down"
fallbacks (cuRobo.py) are written against an abstract EE frame described by two
local unit axes:

  - approach_local: the EE local axis that "reaches out" / points down to grasp
                    (Franka panda_hand: +Z wrist extension; WidowX gripper_link: +X)
  - finger_local:   the EE local axis the two fingers open/close along (binormal)

Everything else (handle picking, place pose via measured ee<->object transform,
scalar offsets) is already robot-agnostic. To support a new robot, add one row
here -- do NOT branch on robot_name in skills.py / cuRobo.py.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class EEConvention:
    approach_local: tuple  # EE local axis aligned to world -Z when grasping top-down
    finger_local: tuple    # EE local axis the fingers close along (binormal)


# NOTE: ee_link is gripper_link (fixed in the widowx cuRobo config.yml). FK-verified
# against wx250s.urdf:
#   approach_local: ee_gripper_link sits at [0.0936, 0, 0] in gripper_link -> +X reach.
#   finger_local:   left_finger prismatic axis resolves to exactly +Y in gripper_link
#                   (the fixed gripper subchain does not rotate it).
EE_CONVENTIONS = {
    "panda":  EEConvention(approach_local=(0.0, 0.0, 1.0), finger_local=(0.0, 1.0, 0.0)),
    "widowx": EEConvention(approach_local=(1.0, 0.0, 0.0), finger_local=(0.0, 1.0, 0.0)),
}


def resolve_ee_convention(robot_name: str) -> EEConvention:
    """Pick the EE convention for a robot_name (substring match), default panda."""
    name = (robot_name or "").lower()
    for key, conv in EE_CONVENTIONS.items():
        if key in name:
            return conv
    return EE_CONVENTIONS["panda"]


@dataclass(frozen=True)
class RobotNames:
    """Joint/body names base_env resolves at init. Same seam rule as above:
    add a row per robot, do not branch on robot_name at the use sites."""
    arm_joints: object          # str regex or list of names for find_joints
    finger_joints: tuple        # (left, right) prismatic finger joints
    hand_link: str              # EE body the IK controller drives
    finger_links: tuple         # (left, right) finger bodies
    finger_tips: tuple          # (left, right) fingertip bodies
    grasp_site: str             # grasp reference body (TCP-ish site)
    contact_prim_expr: str      # regex for the 4 finger contact bodies under .../Robot/


ROBOT_NAMES = {
    "panda": RobotNames(
        arm_joints="panda_joint.*",
        finger_joints=("panda_finger_joint1", "panda_finger_joint2"),
        hand_link="panda_hand",
        finger_links=("panda_leftfinger", "panda_rightfinger"),
        finger_tips=("panda_leftfinger_tip", "panda_rightfinger_tip"),
        grasp_site="panda_grip_site",
        contact_prim_expr="panda_(left|right)finger(_tip|)",
    ),
    "widowx": RobotNames(
        arm_joints=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
        finger_joints=("left_finger", "right_finger"),
        hand_link="gripper_link",
        finger_links=("left_finger_link", "right_finger_link"),
        finger_tips=("left_finger_tip", "right_finger_tip"),
        grasp_site="grip_site",
        contact_prim_expr="(left|right)_finger_(link|tip)",
    ),
}


def resolve_robot_names(robot_name: str) -> RobotNames:
    """Pick the name table for a robot_name (substring match), default panda."""
    name = (robot_name or "").lower()
    for key, names in ROBOT_NAMES.items():
        if key in name:
            return names
    return ROBOT_NAMES["panda"]
