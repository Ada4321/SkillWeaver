"""
Parallel skill execution across multiple IsaacLab env slots.

Each slot runs an independent MCTS child node's skill. All slots share one
IsaacLab instance (num_envs = pool_size) and advance together via batch_step()
at every physics timestep.

Skill structure per slot (example: pick):
    CuroboPhase  → move to pre-grasp  (replay precomputed traj)
    RLPhase      → close-loop grasp   (policy inference each step)
    HoldPhase    → hold/stabilize     (fixed action for N steps)

Skill structure per slot (example: place_with_drop):
    CuroboPhase  → move to place loc  (replay precomputed traj)
    HoldPhase    → drop (open gripper, zero EE velocity)

Between phases, if the next phase requires curobo planning (CuroboPhase), it is
planned synchronously while other slots receive zero-action hold steps. Planning
takes ~0.2 s, making the pause negligible relative to execution time.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Optional

ACTION_MODE_DELTA_EEPOSE = 1  # mirrors skills.py constant


# ---------------------------------------------------------------------------
# Phase descriptors
# ---------------------------------------------------------------------------

@dataclass
class CuroboPhase:
    """Replay a precomputed joint trajectory from the motion planner."""
    joint_states: List[np.ndarray]   # list of mode-1 sub-actions to execute

@dataclass
class RLPhase:
    """Run a closed-loop RL policy until the done signal fires."""
    task_name: str
    object_name: str
    policy_agent_name: str

@dataclass
class HoldPhase:
    """Apply a fixed action for a fixed number of steps."""
    action: np.ndarray
    max_steps: int
    min_steps: int = 0
    check_velocity: bool = False   # if True, exit early when object settles


# ---------------------------------------------------------------------------
# Per-slot execution state
# ---------------------------------------------------------------------------

@dataclass
class SlotState:
    env_id: int
    child_node: Any                          # TreeNode
    phases: Optional[List]                   # None while planning phase 0
    phase_idx: int = 0
    phase_step: int = 0
    done: bool = False
    terminated: bool = False
    terminate_reason: str = ""
    trajectory: list = field(default_factory=list)
    rl_obs: Optional[Any] = None             # current RL policy observation

    @property
    def current_phase(self):
        if self.phases is None or self.phase_idx >= len(self.phases):
            return None
        return self.phases[self.phase_idx]

    def advance_phase(self):
        self.phase_idx += 1
        self.phase_step = 0
        if self.phase_idx >= len(self.phases):
            self.done = True


# ---------------------------------------------------------------------------
# Skill phase planning helpers
# ---------------------------------------------------------------------------

def _plan_curobo_move(simulator, env, curobo, env_id: int,
                      target_loc_3d, target_quat=None,
                      preserve_object_position_on_fallback: bool = False,
                      rotation_only: bool = False) -> Optional[List[np.ndarray]]:
    """Run curobo.move() for a specific env slot and return mode-1 sub-actions.

    Returns None if planning fails (no valid trajectory).
    """
    current_q = simulator.get_joint_positions(env, env_id=env_id)
    current_gripper = simulator.get_gripper_positions(env, env_id=env_id)
    payload = {
        "target_loc": None if rotation_only else target_loc_3d,
        "target_quat": target_quat,
        "current_q": current_q,
        "gripper_state": current_gripper,
        "preserve_object_position_on_fallback": preserve_object_position_on_fallback,
    }
    res = curobo.move(payload=payload)
    joint_states = res.get("joint_states")
    if joint_states is None:
        return None

    # Convert joint-position trajectory to mode-1 sub-actions (delta EE pose).
    split_fn = getattr(simulator, "split_motion_planning_action", None)
    convert_fn = getattr(simulator, "joint_target_to_mode1_action", None)
    if split_fn is None or convert_fn is None:
        raise RuntimeError("Simulator missing split_motion_planning_action / joint_target_to_mode1_action.")

    sub_actions = []
    joint_prev = np.asarray(current_q, dtype=np.float32)
    # Append gripper to joint_prev so convert_fn gets the full state.
    joint_prev_full = np.concatenate([joint_prev, np.asarray(current_gripper, dtype=np.float32)])
    for js in joint_states:
        js_np = np.asarray(js, dtype=np.float32)
        splits = split_fn(env, js_np)
        if splits:
            for s in splits:
                mode1 = np.asarray(
                    convert_fn(env, joint_start=joint_prev_full, joint_target=np.asarray(s, dtype=np.float32), clamp=True),
                    dtype=np.float32,
                )
                sub_actions.append(mode1)
                joint_prev_full = np.asarray(s, dtype=np.float32)
        else:
            mode1 = np.asarray(
                convert_fn(env, joint_start=joint_prev_full, joint_target=js_np, clamp=True),
                dtype=np.float32,
            )
            sub_actions.append(mode1)
            joint_prev_full = js_np
    return sub_actions


def plan_phases_for_skill(skill_id: str, node, child_node, simulator, env,
                          curobo, rlpolicy, env_id: int) -> Optional[List]:
    """Return the ordered list of Phase objects for a skill on a given env slot.

    Returns None if planning fails (e.g., no valid curobo trajectory).
    """
    from skills.skills import (
        get_3d_target_grasp,
        get_3d_target_place,
        _resolve_pick_policy_agent_name,
        _get_grasp_hold_cfg,
    )

    if skill_id == "pick":
        pre_grasp_loc = get_3d_target_grasp(node, child_node, simulator, env)
        if pre_grasp_loc is None:
            return None
        move_actions = _plan_curobo_move(simulator, env, curobo, env_id, pre_grasp_loc)
        if move_actions is None:
            return None
        object_name = child_node.action_params["asset"]
        policy_agent_name = _resolve_pick_policy_agent_name(object_name, simulator, env, rlpolicy)
        hold_cfg = _get_grasp_hold_cfg(curobo)
        hold_action = np.zeros(7, dtype=np.float32)
        hold_action[-1] = -1.0
        return [
            CuroboPhase(joint_states=move_actions),
            RLPhase(task_name="pick", object_name=object_name, policy_agent_name=policy_agent_name),
            HoldPhase(action=hold_action, max_steps=int(hold_cfg["max_steps"]),
                      min_steps=int(hold_cfg["min_steps"]), check_velocity=True),
        ]

    elif skill_id in ("place_with_drop", "place"):
        move_target = get_3d_target_place(node, child_node, simulator, env, mode="container")
        if move_target is None:
            return None
        preserve = getattr(curobo, "place_preserve_object_position_on_fallback", True)
        move_actions = _plan_curobo_move(
            simulator, env, curobo, env_id, move_target,
            preserve_object_position_on_fallback=preserve,
        )
        if move_actions is None:
            return None
        drop_action = np.zeros(7, dtype=np.float32)
        drop_action[-1] = 1.0   # open gripper
        return [
            CuroboPhase(joint_states=move_actions),
            HoldPhase(action=drop_action, max_steps=30, min_steps=5),
        ]

    elif skill_id == "rotate_held_object":
        # rotate_held_object uses curobo only (no RL in default config)
        target_quat = child_node.action_params.get("target_quat")
        rot_actions = _plan_curobo_move(
            simulator, env, curobo, env_id,
            target_loc_3d=None, target_quat=target_quat, rotation_only=True,
        )
        if rot_actions is None:
            return None
        return [CuroboPhase(joint_states=rot_actions)]

    else:
        raise ValueError(f"plan_phases_for_skill: unsupported skill_id={skill_id!r}")


# ---------------------------------------------------------------------------
# Batch step loop
# ---------------------------------------------------------------------------

def _zero_action(action_dim: int = 7) -> np.ndarray:
    return np.zeros(action_dim, dtype=np.float32)


def run_parallel_slots(
    slots: List[SlotState],
    simulator,
    env,
    curobo,
    rlpolicy,
    max_total_steps: int = 2000,
    action_dim: int = 7,
) -> List[SlotState]:
    """Execute all slots in parallel using batch physics stepping.

    Returns the updated list of SlotState objects (with trajectories and
    terminated/done flags set).
    """
    # --- Initialize RL phases for any slot already starting with RLPhase ---
    for slot in slots:
        if slot.done or slot.phases is None:
            continue
        phase = slot.current_phase
        if isinstance(phase, RLPhase):
            obs = simulator.setup_task_for_slot(
                env=env,
                task_name=phase.task_name,
                object_name=phase.object_name,
                env_id=slot.env_id,
                use_top_pcd_for_obs=bool(
                    getattr(rlpolicy, "uses_top_pcd", lambda _n: False)(phase.policy_agent_name)
                ),
                use_handle_pcd_for_obs=bool(
                    getattr(rlpolicy, "uses_handle_pcd", lambda _n: False)(phase.policy_agent_name)
                ),
                use_half_pcd_for_obs=simulator.uses_half_pcd_for_object(env, phase.object_name),
            )
            if isinstance(obs, dict):
                obs = obs["obs"]
            slot.rl_obs = obs
            rlpolicy.reset_agent_new_episode(obs, agent_name=phase.policy_agent_name)

    for _t in range(max_total_steps):
        if all(s.done or s.terminated for s in slots):
            break

        # ------------------------------------------------------------------
        # 1. Compute actions for all active slots
        # ------------------------------------------------------------------

        # Batch RL inference: one forward pass per unique agent name.
        rl_actions: dict = {}
        for slot in slots:
            if slot.done or slot.terminated or slot.phases is None:
                continue
            phase = slot.current_phase
            if isinstance(phase, RLPhase) and slot.rl_obs is not None:
                name = phase.policy_agent_name
                if name not in rl_actions:
                    rl_actions[name] = rlpolicy.get_action(slot.rl_obs, agent_name=name)

        actions: dict = {}
        for slot in slots:
            if slot.done or slot.terminated or slot.phases is None:
                actions[slot.env_id] = _zero_action(action_dim)
                continue
            phase = slot.current_phase
            if phase is None:
                actions[slot.env_id] = _zero_action(action_dim)
                continue

            if isinstance(phase, CuroboPhase):
                if slot.phase_step < len(phase.joint_states):
                    actions[slot.env_id] = phase.joint_states[slot.phase_step]
                else:
                    actions[slot.env_id] = _zero_action(action_dim)

            elif isinstance(phase, RLPhase):
                if phase.policy_agent_name in rl_actions:
                    actions[slot.env_id] = rl_actions[phase.policy_agent_name][slot.env_id]
                else:
                    actions[slot.env_id] = _zero_action(action_dim)

            elif isinstance(phase, HoldPhase):
                actions[slot.env_id] = phase.action

        # ------------------------------------------------------------------
        # 2. Single batched physics step
        # ------------------------------------------------------------------
        obs_batch, dones_batch, _ = simulator.batch_step(
            env, actions, action_mode=ACTION_MODE_DELTA_EEPOSE
        )

        # ------------------------------------------------------------------
        # 3. Update slot states and detect phase transitions
        # ------------------------------------------------------------------
        slots_needing_planning: List[SlotState] = []

        for slot in slots:
            if slot.done or slot.terminated or slot.phases is None:
                continue
            phase = slot.current_phase
            if phase is None:
                slot.done = True
                continue

            if isinstance(phase, CuroboPhase):
                slot.phase_step += 1
                if slot.phase_step >= len(phase.joint_states):
                    slot.advance_phase()
                    next_phase = slot.current_phase
                    if isinstance(next_phase, RLPhase):
                        # Initialize RL at transition time (per-slot, no full env reset)
                        obs = simulator.setup_task_for_slot(
                            env=env,
                            task_name=next_phase.task_name,
                            object_name=next_phase.object_name,
                            env_id=slot.env_id,
                            use_top_pcd_for_obs=bool(
                                getattr(rlpolicy, "uses_top_pcd", lambda _n: False)(
                                    next_phase.policy_agent_name
                                )
                            ),
                            use_handle_pcd_for_obs=bool(
                                getattr(rlpolicy, "uses_handle_pcd", lambda _n: False)(
                                    next_phase.policy_agent_name
                                )
                            ),
                            use_half_pcd_for_obs=simulator.uses_half_pcd_for_object(
                                env, next_phase.object_name
                            ),
                        )
                        if isinstance(obs, dict):
                            obs = obs["obs"]
                        slot.rl_obs = obs
                        rlpolicy.reset_agent_new_episode(
                            obs, agent_name=next_phase.policy_agent_name
                        )
                    elif isinstance(next_phase, CuroboPhase) and not next_phase.joint_states:
                        # Next phase needs planning — schedule it
                        slots_needing_planning.append(slot)

            elif isinstance(phase, RLPhase):
                done_flag = bool(dones_batch[slot.env_id])
                # Update RL obs
                if obs_batch is not None:
                    slot.rl_obs = obs_batch  # full batch; policy reads by env_id internally or via reset
                rlpolicy.post_action_operation(
                    dones_batch[slot.env_id:slot.env_id + 1],
                    agent_name=phase.policy_agent_name,
                )
                if done_flag:
                    slot.advance_phase()
                    next_phase = slot.current_phase
                    if isinstance(next_phase, CuroboPhase) and not next_phase.joint_states:
                        slots_needing_planning.append(slot)

            elif isinstance(phase, HoldPhase):
                slot.phase_step += 1
                if slot.phase_step >= phase.max_steps:
                    slot.advance_phase()

        # ------------------------------------------------------------------
        # 4. Plan any pending CuroboPhase (synchronous, ~0.2 s each)
        #    Other slots receive zero actions during this short pause.
        # ------------------------------------------------------------------
        for slot in slots_needing_planning:
            phase = slot.current_phase
            if not isinstance(phase, CuroboPhase):
                continue
            # The phase was added with empty joint_states as a placeholder.
            # Re-plan now from the current state of this slot's env.
            child_node = slot.child_node
            new_actions = _plan_curobo_move(
                simulator, env, curobo, slot.env_id,
                target_loc_3d=child_node.action_params.get("point_3d"),
            )
            if new_actions is None:
                slot.terminated = True
                slot.terminate_reason = "curobo replanning failed at phase transition"
            else:
                phase.joint_states = new_actions

    # Mark any slot that ran out of steps as terminated
    for slot in slots:
        if not slot.done and not slot.terminated:
            slot.terminated = True
            slot.terminate_reason = "max_total_steps exceeded"

    return slots
