from __future__ import annotations
from typing import Optional, List
from itertools import count
import json

GUIDANCE_HEADER = "***** Guidance *****"


def _format_guidance_block(value: str) -> str:
    return f"\n\n{GUIDANCE_HEADER}\n{value}\n"


def _join_guidance_values(values: list[str]) -> str:
    """Concatenate multiple guidance values into a single block body.

    Single value: returned verbatim. Multiple values: numbered list.
    """
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return "\n".join(f"{i + 1}. {v}" for i, v in enumerate(values))

def serialize_tree(
    root: TreeNode,
    *,
    save_tree_observations: bool = True,
    save_tree_pcd: bool = True,
) -> None:
    """
    Recursively turn a TreeNode (and its descendants) into a nested dictionary,
    then write to a JSON file.

    Args:
        save_tree_observations: Whether to save node-level observations.
        save_tree_pcd: When saving observations, whether to keep
            observations["pcd_all_views"]. If False, pcd_all_views is omitted
            while image observations are preserved.
    """
    node_trajectories = dict()
    id_counter = count(start=0)

    def _serialize_observations(observations):
        if not save_tree_observations:
            return None
        if not isinstance(observations, dict):
            return observations
        obs_out = dict(observations)
        if not save_tree_pcd:
            obs_out.pop("pcd_all_views", None)
        return obs_out

    def _node_to_dict(node: TreeNode) -> dict:
        nid = next(id_counter)
        node._serialized_node_id = nid  # write back so _format_search_output can reference it

        if node.trajectory is not None:
            node_trajectories[nid] = node.trajectory

        return_dict = {
            "node_id": nid,
            "term_reason": node.term_reason,
            "planning_params": node.planning_params,
            "timing": node.timing,
            "execution_error_stats": node.execution_error_stats,
            "curobo_failures": getattr(node, "curobo_failures", None),
            "pregrasp_fallback_attempts": getattr(node, "pregrasp_fallback_attempts", None),
            "pregrasp_fallback_offset": getattr(node, "pregrasp_fallback_offset", None),
            "planner_chunk_count": int(getattr(node, "planner_chunk_count", 0) or 0),
            "planner_split_count": int(getattr(node, "planner_split_count", 0) or 0),
            "history_info": node.history_info,
            "memory_info": getattr(node, "memory_info", None),
            "actor_prompt_1": node.actor_prompt_1,
            "actor_thinking_text_1": node.actor_thinking_text_1,
            "actor_prompt_2": node.actor_prompt_2,
            "actor_thinking_text_2": node.actor_thinking_text_2,
            "action_params": node.action_params,
            "judge_prompt_1": node.judge_prompt_1,
            "judge_prompt_2": node.judge_prompt_2,
            "judge_thinking_text": node.judge_thinking_text,
            "is_terminal": node.is_terminal,
            "is_invalid": node.is_invalid,
            "value": node.value,
            "visit_count": node.visit_count,
            "depth": node.depth,
            "dead_end_count": node.dead_end_count,
            "children": [_node_to_dict(child) for child in node.children],
        }

        serialized_obs = _serialize_observations(node.observations)
        if serialized_obs is not None:
            return_dict["observations"] = serialized_obs

        return return_dict

    tree_dict = _node_to_dict(root)
    return tree_dict, node_trajectories


def deserialize_tree(filename: str) -> TreeNode:
    """
    Read a JSON file and recursively construct a TreeNode structure.
    """
    def _dict_to_node(d: dict, parent: Optional[TreeNode] = None) -> TreeNode:
        planning_params = d.get("planning_params")
        observations = d.get("observations")
        node = TreeNode(
            term_reason=d.get("term_reason", None),
            planning_params=planning_params,
            observations=observations,
            timing=d.get("timing", None),
            execution_error_stats=d.get("execution_error_stats", None),
            history_info=d.get("history_info", None),
            memory_info=d.get("memory_info", None),
            planner_chunk_count=d.get("planner_chunk_count", 0),
            planner_split_count=d.get("planner_split_count", 0),
            actor_prompt_1=d.get("actor_prompt_1", None),
            actor_thinking_text_1=d.get("actor_thinking_text_1", None),
            actor_prompt_2=d.get("actor_prompt_2", None),
            actor_thinking_text_2=d.get("actor_thinking_text_2", None),
            action_params=d.get("action_params", None),
            judge_prompt_1=d.get("judge_prompt_1", None),
            judge_prompt_2=d.get("judge_prompt_2", None),
            judge_thinking_text=d.get("judge_thinking_text", None),
            parent=parent,
        )
        node.curobo_failures = d.get("curobo_failures", None)
        node.is_terminal = d.get("is_terminal", False)
        node.is_invalid = d.get("is_invalid", False)
        node.value = d.get("value", 0.0)
        node.visit_count = d.get("visit_count", 0)
        node.depth = d.get("depth", 0)
        node.dead_end_count = d.get("dead_end_count", 0)
            
        # Recreate children
        for child_dict in d["children"]:
            child_node = _dict_to_node(child_dict, parent=node)
            node.children.append(child_node)
        return node

    with open(filename, "r", encoding="utf-8") as f:
        tree_dict = json.load(f)
    return _dict_to_node(tree_dict["tree"])


class TreeNode:
    """
    Represents a single node in the tree.
    Each node has:
      - a reference to the parent
      - a list of children
      - a 'thought' text
      - a heuristic or computed value
      - a flag if it's terminal (<final>) or not
    """

    def __init__(
        self,
        term_reason: str = None,
        planning_params: Optional[dict] = None,
        observations: Optional[dict] = None,
        trajectory = None,
        timing: Optional[dict] = None,
        execution_error_stats: Optional[dict] = None,
        parent: Optional[TreeNode] = None,
        snapshot: dict = None,
        history_info: Optional[dict] = None,
        memory_info: Optional[dict] = None,
        actor_prompt_1: Optional[str] = None,
        actor_thinking_text_1: Optional[str] = None,
        actor_prompt_2: Optional[str] = None,
        actor_thinking_text_2: Optional[str] = None,
        action_params: Optional[dict] = None,
        judge_prompt_1: Optional[str] = None,
        judge_prompt_2: Optional[str] = None,
        judge_thinking_text: Optional[str] = None,
        is_terminal: bool = False,
        is_invalid: bool = False,
        depth: int = 0,
        visit_count: int = 0,
        value: float = 0.0,
        dead_end_count: int = 0,
        reward_assigned: bool = False,
        judge_score: Optional[float] = None,
        planner_chunk_count: int = 0,
        planner_split_count: int = 0,
    ):
        ## useful outputs

        # texts
        self.term_reason = term_reason
        self.actor_prompt_1 = actor_prompt_1
        self.actor_thinking_text_1 = actor_thinking_text_1
        self.actor_prompt_2 = actor_prompt_2
        self.actor_thinking_text_2 = actor_thinking_text_2
        self.judge_prompt_1 = judge_prompt_1
        self.judge_prompt_2 = judge_prompt_2
        self.judge_thinking_text = judge_thinking_text

        # planning params
        if planning_params is None:
            self.planning_params = {
                "skill_id": None,
                "skill": None,
                "subgoal": None,
                "view_id": None,
                "view": None,
            }
        else:
            self.planning_params = planning_params

        # observations
        if observations is None:
            self.observations = {
                "image_all_views": None,
                "pcd_all_views": None,
            }
        else:
            self.observations = observations
            self.observations.setdefault("image_all_views", None)
            self.observations.setdefault("pcd_all_views", None)
        self.trajectory = trajectory
        self.timing = dict(timing) if isinstance(timing, dict) else {}
        self.execution_error_stats = execution_error_stats if isinstance(execution_error_stats, dict) else None
        # cuRobo planning failure records ({stage, candidates[{cand,status,...}], culprits});
        # appended by skills._append_curobo_failure_record, serialized for post-hoc analysis.
        self.curobo_failures = None

        # actions
        self.action_params = action_params
        self.planner_chunk_count = int(planner_chunk_count)
        self.planner_split_count = int(planner_split_count)

        # history
        if history_info is None:
            self.history_info = {
                "history_traj": [],
                "reflection": None,
                "progress": None,
            }
        else:
            self.history_info = history_info

        # memory retrieval log per child node
        self.memory_info = memory_info if isinstance(memory_info, dict) else None

        ## for MCTS
        self.parent = parent
        self.children: List[TreeNode] = []
        self.snapshot = snapshot
        self.value: float = value   # heuristic or MCTS Q-value
        self.depth: int = depth
        self.visit_count: int = visit_count # for MCTS
        self.dead_end_count: int = dead_end_count
        self.is_invalid: bool = is_invalid
        self.is_terminal: bool = is_terminal
        self.reward_assigned: bool = reward_assigned
        self.judge_score: Optional[float] = judge_score


    def add_child(self, child_node: TreeNode) -> None:
        self.children.append(child_node)


    def update_value_mcts(self, reward: float) -> None:
        """
        Update the node's average value with a new sample (reward).
        A simple way is to do incremental mean:
          new_average = old_average + (reward - old_average) / N
        """
        old_value = self.value
        self.value += (reward - old_value) / (self.visit_count)


    def increment_visits(self) -> None:
        self.visit_count += 1


