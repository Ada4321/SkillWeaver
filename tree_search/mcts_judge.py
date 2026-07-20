import time

from tree_search.tree_node import TreeNode


class JudgeMixin:
    """VLM-judge reward + terminal-state evaluation (mixed into MCTS via self)."""

    def _get_reward_from_judge(self, child_node: TreeNode):
        if child_node.is_terminal:
            return child_node.value

        judge_total_start = time.perf_counter()

        # ===== Judge Phase 1: Subgoal Completion Evaluation with VLM =====
        t0 = time.perf_counter()
        subgoal_prompt_text, subgoal_prompt_images = self._make_judge_prompt_subgoal(child_node)
        self._add_timing(child_node, "judge_subgoal_prompt_s", time.perf_counter() - t0)
        child_node.judge_prompt_1 = subgoal_prompt_text
        t0 = time.perf_counter()
        subgoal_result = self.vlm.generate_single_thought(
            prompt={
                "text": subgoal_prompt_text,
                "images": subgoal_prompt_images,
            },
            phase="judge_subgoal",
        )
        self._add_timing(child_node, "judge_subgoal_vlm_s", time.perf_counter() - t0)
        thinking_text = subgoal_result.get("thinking_text")
        summary_text = subgoal_result.get("summary_text")
        completion_text = subgoal_result.get("completion_text")
        child_node.judge_thinking_text = thinking_text
        child_node.history_info["history_traj"] = child_node.parent.history_info["history_traj"] + [(summary_text or "Not sure what happened") + "\n" + (completion_text or "")]

        t0 = time.perf_counter()
        progress_prompt_text, progress_prompt_images = self._make_judge_prompt_progress(child_node)
        self._add_timing(child_node, "judge_progress_prompt_s", time.perf_counter() - t0)
        child_node.judge_prompt_2 = progress_prompt_text
        t0 = time.perf_counter()
        progress_result = self.vlm.generate_single_thought(
            prompt={
                "text": progress_prompt_text,
                "images": progress_prompt_images,
            },
            phase="judge_progress",
        )
        self._add_timing(child_node, "judge_progress_vlm_s", time.perf_counter() - t0)
        reflection_text = progress_result.get("reflection_text")
        progress_text = progress_result.get("progress_text")

        child_node.history_info["reflection"] = reflection_text
        child_node.history_info["progress"] = progress_text

        progress_score = progress_result.get("progress_score")
        if progress_score is None:
            progress_score = 0
        else:
            progress_score = max(0, min(100, progress_score))

        self._set_timing(child_node, "judge_total_s", time.perf_counter() - judge_total_start)
        return float(progress_score) / 100

    def _progress_only_judge(self, child_node: TreeNode) -> float:
        """
        PROGRESS-ONLY JUDGE:
        Evaluate task progress from root -> child via VLM (skip the subgoal
        eval). Used by no_rollout / standard_rollout modes which judge an
        entire trajectory rather than per-child subgoals. Writes
        history_info["reflection"] and ["progress"]; returns
        progress_score / 100 in [0, 1].
        """
        if child_node.is_terminal:
            return child_node.value

        judge_total_start = time.perf_counter()

        t0 = time.perf_counter()
        progress_prompt_text, progress_prompt_images = self._make_judge_prompt_progress(child_node)
        self._add_timing(child_node, "judge_progress_prompt_s", time.perf_counter() - t0)
        child_node.judge_prompt_2 = progress_prompt_text
        t0 = time.perf_counter()
        progress_result = self.vlm.generate_single_thought(
            prompt={
                "text": progress_prompt_text,
                "images": progress_prompt_images,
            },
            phase="judge_progress",
        )
        self._add_timing(child_node, "judge_progress_vlm_s", time.perf_counter() - t0)
        reflection_text = progress_result.get("reflection_text")
        progress_text = progress_result.get("progress_text")
        child_node.history_info["reflection"] = reflection_text
        child_node.history_info["progress"] = progress_text

        progress_score = progress_result.get("progress_score")
        if progress_score is None:
            progress_score = 0
        else:
            progress_score = max(0, min(100, progress_score))

        self._set_timing(child_node, "judge_total_s", time.perf_counter() - judge_total_start)
        return float(progress_score) / 100

    def _dispatch_terminal_judge(self, node: TreeNode) -> float:
        """Terminal-state judge: VLM progress score over the root->node trajectory."""
        return self._progress_only_judge(node)
