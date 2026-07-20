SUBGOAL_EVAL_PROMPT = """You are an expert roboticist. The robot is performing the task of '{instruction}', and the current subgoal is {subgoal}.
Your task is to determine whether the subgoal is completed safely by analyzing the provided visual observations, object layout, and contact information.

{coordinate_system}***** Input *****
This is the initial state of the scene.
(1) Visual Observations:
{init_visual_observations}
(2) Object Layout:
{init_object_information}
(3) Contact Information:
{init_contact_information}
From the initital state, the robot has made some progress, resulting in the following current state of the scene.
(1) Visual Observations:
{current_visual_observations}
(2) Object Layout:
{current_object_information}
(3) Contact Information:
{current_contact_information}

***** Output *****
## Thinking
1. Propose the criteria for subgoal completion and safety based on the subgoal.
2. Analyze the Initial State: 
(1) Examine the initial visual observations
(2) Examine the initial object layout
(3) Examine the initial gripper-object contact information
to understand the starting conditions of the scene.
3. Analyze the Current State: 
(1) Examine the current visual observations
(2) Examine the current object layout
(3) Examine the current gripper-object contact information
to assess the changes that have occurred since the initial state.
4. Compare States: Identify the differences between the initial and current states by careful comparison, focusing on changes relevant to the subgoal. If the relevant object is missing from view(e.g. occluded by other objects or some container), you should infer its position based on other information.
5. Apply Criteria: Please match the scene transitions with the proposed criteria above one by one before making decisions.
6. Conclusion - Completion: Decide whether this subgoal is completed.
7. Conclusion - Safety: Decide whether the robot's action is safe(i.e. no other objects being knocked over or SEVERELY damaged).

## Summary
Summarize the image content with one sentence.

## Subgoal Completion and Safety
Whether the subgoal is completed and the robot's action is safe.

Make sure you use '## Thinking', '## Summary', and '## Subgoal Completion and Safety' as section headers in your response."""


SUBGOAL_EVAL_PROMPT_NO_SAFETY = """You are an expert roboticist. The robot is performing the task of '{instruction}', and the current subgoal is {subgoal}.
Your task is to determine whether the subgoal is completed by analyzing the provided visual observations, object layout, and contact information.

{coordinate_system}***** Input *****
This is the initial state of the scene.
(1) Visual Observations:
{init_visual_observations}
(2) Object Layout:
{init_object_information}
(3) Contact Information:
{init_contact_information}
From the initital state, the robot has made some progress, resulting in the following current state of the scene.
(1) Visual Observations:
{current_visual_observations}
(2) Object Layout:
{current_object_information}
(3) Contact Information:
{current_contact_information}

***** Output *****
## Thinking
1. Propose the criteria for subgoal completion based on the subgoal.
2. Analyze the Initial State:
(1) Examine the initial visual observations
(2) Examine the initial object layout
(3) Examine the initial gripper-object contact information
to understand the starting conditions of the scene.
3. Analyze the Current State:
(1) Examine the current visual observations
(2) Examine the current object layout
(3) Examine the current gripper-object contact information
to assess the changes that have occurred since the initial state.
4. Compare States: Identify the differences between the initial and current states by careful comparison, focusing on changes relevant to the subgoal. If the relevant object is missing from view(e.g. occluded by other objects or some container), you should infer its position based on other information.
5. Apply Criteria: Please match the scene transitions with the proposed criteria above one by one before making decisions.
6. Conclusion - Completion: Decide whether this subgoal is completed.

## Summary
Summarize the image content with one sentence.

## Subgoal Completion
Whether the subgoal is completed.

Make sure you use '## Thinking', '## Summary', and '## Subgoal Completion' as section headers in your response."""


PROGRESS_EVAL_PROMPT = """You are an expert roboticist. The robot is performing the task of '{instruction}'.
Your task is to evaluate the robot's progress towards completing the overall task by analyzing the provided initial/current scene states and robot action trajectory.

{coordinate_system}***** Input *****
This is the initial state of the scene.
(1) Visual Observations:
{init_visual_observations}
(2) Object Layout:
{init_object_information}
(3) Contact Information:
{init_contact_information}

From the initial state, the robot has made progress represented by the following trajectory:
{history_information}

Such progress results in the current state of the scene.
(1) Visual Observations:
{current_visual_observations}
(2) Object Layout:
{current_object_information}
(3) Contact Information:
{current_contact_information}

***** Output *****
Based on the above information, please do the following:
## Reflection
Evaluate the overall progress of the robot toward completing the task by analizing the trajectory and comparing initial and current scene state.
Are the subgoals reasonable? Is the robot making meaningful progress towards the final goal?

## Progress Score
Predict a task completion percentage between 0 and 100.
    - In the initial state, the task completion percentage is 0.
    - If all necessary steps are completed, the task completion percentage is 100.
    - The score should increase when a **meaningful** subgoal is completed.
    and should decrease if an action **undoes or degrades** a previously completed meaningful subgoal.
    - The score should also decrease if the robot's action causes SEVERE damage to other objects in the environment(Minor disturbance is acceptable and can be ignored).
    - **OVERRIDE**: If the overall task is fully completed at the current state, output 100 even if some unsafe actions occurred along the trajectory. Safety penalties only reduce the score when the task is INCOMPLETE.
    In this part, just provide one score and one concise explaining sentence.

Make sure you use '## Reflection' and '## Progress Score' as section headers in your response.
Please use EXACTLY ONE number between 0 and 100 for the Progress Score."""


PROGRESS_EVAL_PROMPT_NO_SAFETY = """You are an expert roboticist. The robot is performing the task of '{instruction}'.
Your task is to evaluate the robot's progress towards completing the overall task by analyzing the provided initial/current scene states and robot action trajectory.

{coordinate_system}***** Input *****
This is the initial state of the scene.
(1) Visual Observations:
{init_visual_observations}
(2) Object Layout:
{init_object_information}
(3) Contact Information:
{init_contact_information}

From the initial state, the robot has made progress represented by the following trajectory:
{history_information}

Such progress results in the current state of the scene.
(1) Visual Observations:
{current_visual_observations}
(2) Object Layout:
{current_object_information}
(3) Contact Information:
{current_contact_information}

***** Output *****
Based on the above information, please do the following:
## Reflection
Evaluate the overall progress of the robot toward completing the task by analizing the trajectory and comparing initial and current scene state.
Are the subgoals reasonable? Is the robot making meaningful progress towards the final goal?

## Progress Score
Predict a task completion percentage between 0 and 100.
    - In the initial state, the task completion percentage is 0.
    - If all necessary steps are completed, the task completion percentage is 100.
    - The score should increase when a **meaningful** subgoal is completed.
    and should decrease if an action **undoes or degrades** a previously completed meaningful subgoal.
    In this part, just provide one score and one concise explaining sentence.

Make sure you use '## Reflection' and '## Progress Score' as section headers in your response.
Please use EXACTLY ONE number between 0 and 100 for the Progress Score."""