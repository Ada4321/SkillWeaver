"""Shared skill-layer constants. Kept in a leaf module so the skills submodules
(_planner_exec / _holds / _targets) and skills.py can all import them without a cycle
(skills.py imports the submodules, so the submodules must not import skills.py)."""

# Action-space modes passed to simulator.step / the step-data collectors.
ACTION_MODE_JOINT_ANGLES = 0
ACTION_MODE_DELTA_EEPOSE = 1

# Cos-angle threshold for "object's local +Z aligned with world +Z".
# Below this -> object is "lying" (tilted by >60deg). Used by both the convex
# pre-grasp wrist-orientation helper and the convex pick-policy dispatcher.
CONVEX_UPRIGHT_COS_THRESHOLD = 0.5
