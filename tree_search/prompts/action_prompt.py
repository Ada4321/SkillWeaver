# ===== Skill-specific prompts =====

COORDINATE_SYSTEM_TEXT = "+x is the front of the scene, +y is right, and +z is up"

GRASP_PROMPT = """{coordinate_system}You are an assistant tasked with predicting critical parameters for one step(a subgoal) in a long-horizon robotic manipulation task.
The overall task is '{instruction}', and the current subgoal is '{subgoal}'.

Your task is to determine the target asset to grasp in the scene based on the current state of the scene.
You MUST pick the targetasset from the provided object list only. Output ONLY the integer index from the list.

Asset list:
{object_information}

Please provide your answer in the following format:
<thinking>
reasoning content.
</thinking>
<answer>
Asset: the id of the object to grasp (integer index from the object list).
</answer>

Example answer:
<thinking>
The target asset is the red apple on the table as the subgoal requires picking up the apple.
</thinking>
<answer>
Asset: 3
</answer>"""


# PLACE_PROMPT = """You are an assistant tasked with predicting critical parameters for one step(a subgoal) in a long-horizon robotic manipulation task.
# The overall task is '{instruction}', and the current subgoal is '{subgoal}'.

# Your task is to determine the target object serving as place destination and ground the place location with a point on the target object surface in the input image reflecting the current state of the scene.
# You MUST pick the target asset from the provided object list only. Output ONLY the integer index from the list.

# Current visual observations:
# {visual_observations}

# Asset list:
# {object_information}

# Please provide your answer in the following format:
# <thinking>
# reasoning content.
# </thinking>
# <answer>
# Asset: the index of the object OR drawer as place destination (integer from the list; a door is not a valid choice).
# Point: (y, x) normalized to 0-1000
# </answer>

# Example answer:
# <thinking>
# The target asset is the rectangular box as the subgoal requires placing the pen into the box.
# </thinking>
# <answer>
# Asset: 4
# Point: (500, 500)
# </answer>"""


PLACE_PROMPT = """{coordinate_system}You are an assistant tasked with predicting critical parameters for one step(a subgoal) in a long-horizon robotic manipulation task.
The overall task is '{instruction}', and the current subgoal is '{subgoal}'.

Your task is to determine the place destination, and determine the goal position in 3D space for the center of the held object, which is usually in a local region relative to the target.
You MUST pick the target from the provided list only. The target can be either a rigid object or an articulated part (e.g. a drawer). Output ONLY the integer index of the selected target.

Asset list:
{object_information}

Please provide your answer in the following format:
<thinking>
reasoning content.
</thinking>
<answer>
Asset: the index of the object OR articulated part as place destination (integer from the list).
Point: (x, y, z) in meters in the world frame.
</answer>

Example answer:
<thinking>
The target asset is the rectangular box as the subgoal requires placing the pen into the box.
</thinking>
<answer>
Asset: 4
Point: (0.5, 0.5, 0.5)
</answer>"""

PLACE_WITH_DROP_PROMPT = PLACE_PROMPT


MOVE_HELD_OBJECT_PROMPT = """{coordinate_system}You are an assistant tasked with predicting critical parameters for one step(a subgoal) in a long-horizon robotic manipulation task.
The overall task is '{instruction}', and the current subgoal is '{subgoal}'.

Your task is to determine the 3D location where the currently held object should be moved based on the provided scene information.
You should predict a 3D coordinate (x, y, z) in the world frame representing the desired position for the object center.

Held Asset:
{contact_information}

Asset list:
{object_information}

Please provide your answer in the following format:
<thinking>
reasoning content.
</thinking>
<answer>
Point: (x, y, z) in meters in the world frame.
</answer>

Example answer:
<thinking>
The held object should be moved above the table to facilitate the next placement action.
</thinking>
<answer>
Point: (0.5, 0.0, 0.75)
</answer>"""


ROTATE_HELD_OBJECT_PROMPT = """{coordinate_system}You are an assistant tasked with predicting critical parameters for one step(a subgoal) in a long-horizon robotic manipulation task.
The overall task is '{instruction}', and the current subgoal is '{subgoal}'.

Your task is to determine the desired orientation for the currently held object.
You should first select a reference object from the provided object list in the scene to align with.
Then, align the held object orientation with respect to the selected reference object at the END STATE of this task.

For orientation alignment, we select one primary axis (x, y, or z) in the canonical object frame for both the held asset and the reference asset, and align the selected axes in the same direction.
Thus, you need to specify which axis to use for both objects and the alignment direction (same, represented with '+', or opposite, represented with '-').
e.g. (x, z, +) means aligning the x axis of the held object with the z axis of the reference object in the same direction;
(x, y, -) means aligning the x axis of the held object with the y axis of the reference object in the opposite direction at the END STATE of this task.

The meaning of the axes in the canonical object frame is described in the asset list below.

Held Asset:
{contact_information}

Asset list:
{object_information}

Please provide your answer in the following format:
<thinking>
## Reasoning about reference object selection
## Carefully reasoning about how to determine the axis alignment based on the object shapes and features
</thinking>
<answer>
Asset: the id of the reference object (integer index from the object list).
Alignment: (Held_Axis, Target_Axis, Direction) => (e.g. (x, y, +)/(z, z, -)/(z, x, -)/...).
</answer>

Example answer:
<thinking>
The reference object is the mug as the subgoal requires putting the carrot into the mug, and it has a clear vertical axis that can be aligned with the carrot.
To ensure the carrot fits into the mug, I will align the z axis of the carrot with the z axis of the mug in the same direction.
</thinking>
<answer>
Asset: 2
Alignment: (z, z, +)
</answer>"""


# ORIENTATION_AWARE_PLACE_PROMPT = """You are an assistant tasked with predicting critical parameters for one step(a subgoal) in a long-horizon robotic manipulation task.
# The overall task is '{instruction}', and the current subgoal is '{subgoal}'.

# Your task is to:
# 1. Determine the target object serving as place destination.
# 2. Ground the place location with a point on the target object surface in the input image reflecting the current state of the scene.
# 3. Align the held object orientation with respect to the target object.

# For asset identification, you MUST pick from the provided list only — an object's index, or (to place INTO a drawer) that drawer's index shown under its cabinet as "Articulated parts". Do NOT pick a door. Output ONLY the integer index.

# For orientation alignment, we select one primary axis (x, y, or z) in the canonical object frame for both the held asset and the target asset, and align the selected axes in the same direction.
# Thus, you need to specify which axis to use for both objects and the alignment direction (same, represented with '+', or opposite, represented with '-').
# e.g. (x, z, +) means aligning the x axis of the held object with the z axis of the target object in the same direction;
# (x, y, -) means aligning the x axis of the held object with the y axis of the target object in the opposite direction at the END STATE of this task.
# The meaning of the axes in the canonical object frame is described in the asset list below.

# Current visual observations:
# {visual_observations}

# Held Asset:
# {contact_information}

# Asset list:
# {object_information}

# Please provide your answer in the following format:
# <thinking>
# ## Reasoning about asset identification
# ## Reasoning about point grounding
# ## Carefully reasoning about orientation alignment
# </thinking>
# <answer>
# Asset: the index of the object OR drawer as place destination (integer from the list; a door is not a valid choice).
# Point: (y, x) normalized to 0-1000
# Alignment: (Held_Axis, Target_Axis, Direction)
# </answer>

# Example answer:
# <thinking>
# The target asset is the mug as the subgoal requires putting the carrot into the mug.
# To ensure the carrot fits into the mug, I will align the z axis of the carrot with the z axis of the mug in the same direction.
# </thinking>
# <answer>
# Asset: 4
# Point: (500, 500)
# Alignment: (z, z, +)
# </answer>"""


ORIENTATION_AWARE_PLACE_PROMPT = """{coordinate_system}You are an assistant tasked with predicting critical parameters for one step(a subgoal) in a long-horizon robotic manipulation task.
The overall task is '{instruction}', and the current subgoal is '{subgoal}'.

Your task is to:
1. Determine the target object serving as place destination.
2. Determine the goal position in 3D space for the center of the held object, which is usually in a local region relative to the target object.
3. Align the held object orientation with respect to the target object.

For asset identification, you MUST pick the target from the provided list only. The target can be either a rigid object or an articulated part (e.g. a drawer). Output ONLY the integer index of the selected target.

For orientation alignment, we select one primary axis (x, y, or z) in the canonical object frame for both the held asset and the target asset or part, and align the selected axes in the same direction.
Thus, you need to specify which axis to use for both objects or parts and the alignment direction (same, represented with '+', or opposite, represented with '-').
e.g. (x, z, +) means aligning the x axis of the held object with the z axis of the target in the same direction;
(x, y, -) means aligning the x axis of the held object with the y axis of the target in the opposite direction at the END STATE of this task.
The meaning of the axes in the canonical object frame is described in the asset list below.

Held Asset:
{contact_information}

Asset list:
{object_information}

Please provide your answer in the following format:
<thinking>
## Reasoning about asset identification
## Reasoning about point grounding
## Carefully reasoning about orientation alignment
</thinking>
<answer>
Asset: the index of the object OR articulated part as place destination (integer from the list).
Point: (x, y, z) in meters in the world frame.
Alignment: (Held_Axis, Target_Axis, Direction)
</answer>

Example answer:
<thinking>
The target asset is the mug as the subgoal requires putting the carrot into the mug.
To ensure the carrot fits into the mug, I will align the z axis of the carrot with the z axis of the mug in the same direction.
</thinking>
<answer>
Asset: 4
Point: (0.5, 0.5, 0.1)
Alignment: (z, z, +)
</answer>"""


PLACE_WITH_ORIENTATION_DROP_PROMPT = ORIENTATION_AWARE_PLACE_PROMPT


MOVE_WITHOUT_PLACE_PROMPT = ORIENTATION_AWARE_PLACE_PROMPT


# ===== place_with_gripper_orientation_drop =====
# Same parser/answer schema as the orientation-aware place skills (Asset / Point / Alignment),
# but the Alignment src is the ROBOT GRIPPER itself, not the held object. The gripper axis
# meaning differs per robot, so the {gripper_axes} block is filled per robot type at registry
# time (mcts.py selects PANDA vs WIDOWX from simulator.robot_name).
_PANDA_GRIPPER_AXES = """- x: palm-normal axis, perpendicular to the two flat palm faces of the gripper.
\t\t- y: the axis the two fingers slide along when opening and closing.
\t\t- z: the approach axis, pointing out of the wrist toward the fingertips (the direction the gripper reaches in)."""

_WIDOWX_GRIPPER_AXES = """- x: the approach axis, pointing out of the wrist toward the fingertips (the direction the gripper reaches in).
\t\t- y: the axis the two fingers slide along when opening and closing.
\t\t- z: palm-normal axis, perpendicular to the two flat palm faces of the gripper."""

_GRIPPER_ORIENTATION_PLACE_TEMPLATE = """{coordinate_system}You are an assistant tasked with predicting critical parameters for one step(a subgoal) in a long-horizon robotic manipulation task.
The overall task is '{instruction}', and the current subgoal is '{subgoal}'.

Your task is to:
1. Determine the target object serving as place destination.
2. Determine the goal position in 3D space for the center of the held object, which is usually in a local region relative to the target object.
3. Choose how the ROBOT GRIPPER should be oriented at the placement, by aligning one of the gripper's axes with one of the target's canonical axes.

For asset identification, you MUST pick the target from the provided list only. The target can be either a rigid object or an articulated part (e.g. a drawer). Output ONLY the integer index of the selected target.

For orientation alignment, we select one primary axis (x, y, or z) of the ROBOT GRIPPER and one primary axis (x, y, or z) in the canonical frame of the TARGET object or part, and align the selected axes in the same direction ('+') or the opposite direction ('-').
This directly controls the direction the gripper extends/approaches when it places and releases the object.
The gripper's primary axes are defined as:
\t\t{gripper_axes}
The meaning of the target object's axes is described in the asset list below.
e.g. (z, z, +) means aligning the gripper's z axis with the z axis of the target in the same direction;
(x, y, -) means aligning the gripper's x axis with the y axis of the target in the opposite direction at the END STATE of this task.

Asset list:
{object_information}

Please provide your answer in the following format:
<thinking>
## Reasoning about asset identification
## Reasoning about point grounding
## Carefully reasoning about how the gripper should approach, and which gripper axis to align with which target object/part axis
</thinking>
<answer>
Asset: the index of the object OR articulated part as place destination (integer from the list).
Point: (x, y, z) in meters in the world frame.
Alignment: (Gripper_Axis, Target_Axis, Direction)
</answer>

Example answer:
<thinking>
The target asset is the safe. I want the gripper to approach horizontally, so it can reach into the safe's opening. I will align the gripper's z axis (the approach axis) with the safe's y axis in the same direction, so the gripper approaches from the front.
</thinking>
<answer>
Asset: 4
Point: (0.5, 0.5, 0.1)
Alignment: (z, y, +)
</answer>"""

GRIPPER_ORIENTATION_PLACE_PROMPT_PANDA = _GRIPPER_ORIENTATION_PLACE_TEMPLATE.replace("{gripper_axes}", _PANDA_GRIPPER_AXES)
GRIPPER_ORIENTATION_PLACE_PROMPT_WIDOWX = _GRIPPER_ORIENTATION_PLACE_TEMPLATE.replace("{gripper_axes}", _WIDOWX_GRIPPER_AXES)

OPEN_DRAWER_PROMPT = """{coordinate_system}You are an assistant predicting parameters for one subgoal in a long-horizon manipulation task.
The overall task is '{instruction}', and the current subgoal is '{subgoal}'.

Choose the drawer to open. Pick the drawer from the cabinet's articulated part list below according to their center, range, and current state information. Output integer indices only.

Articulated part list:
{object_information}

Please provide your answer in the following format:
<thinking>
reasoning content.
</thinking>
<answer>
Asset: the drawer index to open.
</answer>

Example answer:
<thinking>
The subgoal is to open the middle drawer; drawer 1 is the middle one and is currently closed.
</thinking>
<answer>
Asset: 1
</answer>"""

CLOSE_DRAWER_PROMPT = """{coordinate_system}You are an assistant predicting parameters for one subgoal in a long-horizon manipulation task.
The overall task is '{instruction}', and the current subgoal is '{subgoal}'.

Choose the drawer to close. Pick the drawer from the cabinet's articulated part list below according to their center, range, and current state information. Output integer indices only.

Articulated part list:
{object_information}

Please provide your answer in the following format:
<thinking>
reasoning content.
</thinking>
<answer>
Asset: the drawer index to close.
</answer>

Example answer:
<thinking>
The subgoal is to close the middle drawer; drawer 1 is the middle one and is currently open.
</thinking>
<answer>
Asset: 1
</answer>"""

CLOSE_DOOR_PROMPT = """{coordinate_system}You are an assistant predicting parameters for one subgoal in a long-horizon manipulation task.
The overall task is '{instruction}', and the current subgoal is '{subgoal}'.

Your task is to determine the door to be closed.
You MUST pick the target door from the provided articulated part list only. Output ONLY the integer index of the selected door.

Articulated part list:
{object_information}

Please provide your answer in the following format:
<thinking>
reasoning content.
</thinking>
<answer>
Asset: the id of the door to close (integer index from the articulated part list).
</answer>

Example answer:
<thinking>
The subgoal is to close the microwave door; part 0 is the microwave door and it is currently open.
</thinking>
<answer>
Asset: 0
</answer>"""

# ===================================


# ===== Skill descriptions =====

GRASP_DESC = """This skill picks up the target object from the scene using the robot gripper."""

PLACE_WITH_DROP_DESC = """This skill places the currently held object at the specified target location.
It *** first moves the held object to be in the local region above the target location ***, and then ** directly releases ** it.
NOTE:
- The placement is not aware of orientation -- the held object will be placed with a random orientation.
- You should make sure the orientation has been correctly adjusted if needed before using this skill, otherwise the placement may fail (e.g thin objects cannot be placed into thin containers with incorrect orientation).
- This skill is suitable for dropping objects into a ** deep container such as a basket, bucket, or vase **, where the gripper is difficult to reach into the container."""

PLACE_DESC = """This skill places the currently held object at the specified target location.
It *** first moves the held object to be in the local region above the target location ***, and then lowers the held object onto the target surface to ** safely release ** it.
NOTE:
- The placement is not aware of orientation -- the held object will be placed with a random orientation.
- You should make sure the orientation has been correctly adjusted if needed before using this skill, otherwise the placement may fail (e.g thin objects cannot be placed into thin containers with incorrect orientation).
- This skill emphasizes safe placement by lowering the object onto the target surface, which is suitable for placing objects on a ** flat surface such as a table or plate **, where dropping from above may cause instability or damage to the object or environment."""

ORIENTATION_AWARE_PLACE_DESC = """This skill places the currently held object at the specified target location with a desired orientation.
It *** first moves the held object to be in the local region above the target location ***, then reasons about the relative orientation between the held and the target object to achieve precise placement.
NOTE: This skill is suitable for situations where precise orientation is critical, where the held object can only be placed correctly in a certain way, such as placing a thin object (e.g. pen) into a thin container (e.g. pen holder), or placing a tall object (e.g. large milk carton) into a flat container (e.g. low-profile cardboard box). But it may take more time than the 'place' skill to complete due to additional orientation adjustments."""

PLACE_WITH_ORIENTATION_DROP_DESC = """This skill places the currently held object at the specified target location with a desired orientation, then ** directly releases ** it.
It *** first moves the held object to be in the local region above the target location while aligning a chosen primary axis of the held object with one of the target object's primary axes ***, and then ** directly releases ** it.
NOTE:
- This skill is suitable for situations where precise orientation is critical, where the held object can only be placed correctly in a certain way, such as placing a thin object (e.g. pen) into a thin container (e.g. pen holder), or placing a tall object (e.g. large milk carton) into a flat container (e.g. low-profile cardboard box). But it may take more time than the 'place' skill to complete due to additional orientation adjustments.
- This skill is suitable for dropping objects with required orientation into a ** deep container such as a basket, or vase **, where the gripper is difficult to reach into the container."""

MOVE_WITHOUT_PLACE_DESC = """This skill moves the currently held object to a specified target location with a desired orientation, and holds it there without releasing."""

GRIPPER_ORIENTATION_PLACE_DESC = """This skill places the currently held object at the specified target location while orienting the ROBOT GRIPPER itself to a desired direction, then ** directly releases ** it.
Instead of aligning the held object's own axis, you align a chosen primary axis of the gripper with one of the target object's primary axes, so you directly control how the gripper extends/approaches at the placement.
NOTE: Use this when the gripper's approach direction is what matters (e.g. the gripper must come in from a specific OPENING direction to place/insert the object)"""

MOVE_HELD_OBJECT_DESC = """This skill moves the currently held object to a specified 3D location without releasing it."""

ROTATE_HELD_OBJECT_DESC = """This skill estimates a desired rotation for the currently held object based on the current scene states and the subgoal, and then executes the rotation."""

OPEN_DESC = """This skill opens a part in an armored object, such as opening a drawer, a door, or a lid."""

CLOSE_DESC = """This skill closes a part in an armored object, such as closing a drawer, a door, or a lid."""

# ===================================

# ===================================
# open_drawer / close_drawer (multi-DOF cabinet). The VLM answers with one Asset integer =
# the part's index in the flat articulated-part list ({object_information}, prompt
# use_arti_only); the parse resolves it to the part key + owning cabinet via the entity map.
OPEN_DRAWER_DESC = """Open a specific drawer of the target cabinet by grasping its handle and pulling it open. Use when the subgoal is to open/pull out a drawer; the gripper starts free."""
CLOSE_DRAWER_DESC = """Close a specific open drawer of the target cabinet by pushing it shut. Use when the subgoal is to close/push in a drawer; the gripper starts free."""


# ===================================
# close_door (revolute door). Same scheme as open/close_drawer: the VLM picks the door part
# from the flat articulated-part list by its Asset integer.
CLOSE_DOOR_DESC = """Close an open door (e.g. a microwave door) by pushing it shut. Use when the subgoal is to close/shut a door; the gripper starts free and presses the door closed."""
