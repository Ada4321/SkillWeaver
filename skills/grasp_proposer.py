"""GraspGen-based grasp proposer wired into cuRobo motion planning.

Provides:
  - T_PANDA2GRASPGEN: 4x4 transform pinned by visual verification on
    libero butter/cream_cheese/chocolate_pudding boxes (2026-05-25).
    panda_hand_world = grasp_world @ T_PANDA2GRASPGEN
    Origin: no translation offset (IsaacLab cuRobo's EEF link sits at the
    GraspGen grasp grip point, matching LIBERO's behavior).
    Rotation: +90° about Z (GraspGen ±X binormal -> panda_hand ±Y).
  - call_graspgen_server: HTTP POST to the FastAPI server at server_graspgen.py.
  - propose_grasps_panda_hand: end-to-end pcd -> sorted panda_hand world poses.
  - plan_first_reachable: try each candidate via cuRobo.move() until one solves.

Self-contained: no IsaacLab dependency. The cuRobo object is duck-typed
(only requires a .move(payload) method matching skills/cuRobo.py:661).
"""

import base64
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Transform from GraspGen grasp frame to IsaacLab panda_hand (EEF) frame.
# Verified visually 2026-05-25; see eval_grasp/viz_graspgen.py and
# eval_grasp/out/<asset>/rot_pos90z_no_offset.png.
# ---------------------------------------------------------------------------
T_PANDA2GRASPGEN = np.array([
    [0, -1, 0, 0],
    [1,  0, 0, 0],
    [0,  0, 1, 0],
    [0,  0, 0, 1],
], dtype=np.float64)


DEFAULT_GRASPGEN_URL = "http://localhost:9050/predict"


# ---------------------------------------------------------------------------
# encoding helpers (mirror utils/image_utils.py in Grounded_MCTS so the
# existing server_graspgen.py decoder works unchanged)
# ---------------------------------------------------------------------------

def _encode_arr(arr: np.ndarray) -> str:
    buf = BytesIO()
    np.savez_compressed(buf, arr=arr)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _mat_to_pos_quat_wxyz(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """4x4 -> (xyz, wxyz quat). Standard Shepperd's method."""
    pos = mat[:3, 3].astype(np.float64)
    R = mat[:3, :3].astype(np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return pos, np.array([w, x, y, z], dtype=np.float64)


# ---------------------------------------------------------------------------
# server call
# ---------------------------------------------------------------------------

def call_graspgen_server(
    obj_pc_world: np.ndarray,
    url: str = DEFAULT_GRASPGEN_URL,
    timeout: float = 300.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """POST object point cloud (already in world frame) to GraspGen.

    Returns (grasps_4x4_world (N, 4, 4), scores (N,)). Empty arrays on no detections.
    """
    payload = {
        "xyz_scene": _encode_arr(obj_pc_world.astype(np.float32)),
        "obj_mask":  _encode_arr(np.ones(len(obj_pc_world), dtype=np.bool_)),
        # our pcds are clean mesh samples; the server's statistical outlier
        # removal (meant for noisy SAM/depth clouds) wrongly nukes 80-94% of
        # points on thin-walled concave objects (cups) -> too few for PTv3 -> 500.
        "skip_outlier_removal": True,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"GraspGen server error: {data['error']}")
    grasps = np.asarray(data["grasps"], dtype=np.float64)
    scores = np.asarray(data["scores"], dtype=np.float64)
    if grasps.ndim == 1 and grasps.size == 0:
        grasps = np.zeros((0, 4, 4), dtype=np.float64)
        scores = np.zeros((0,), dtype=np.float64)
    return grasps, scores


# ---------------------------------------------------------------------------
# high-level: propose + rank
# ---------------------------------------------------------------------------

def propose_grasps_panda_hand(
    obj_pc_world: np.ndarray,
    url: str = DEFAULT_GRASPGEN_URL,
    top_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run GraspGen and return panda_hand world poses ranked by score desc.

    Returns:
        panda_hand_poses_4x4: (N, 4, 4) world-frame target poses for panda_hand
        scores: (N,)
    """
    grasps_gg, scores = call_graspgen_server(obj_pc_world, url=url)
    if len(grasps_gg) == 0:
        return np.zeros((0, 4, 4)), np.zeros((0,))
    panda_hand = np.einsum("nij,jk->nik", grasps_gg, T_PANDA2GRASPGEN)
    order = np.argsort(-scores)
    panda_hand = panda_hand[order]
    scores = scores[order]
    if top_k is not None and top_k > 0:
        panda_hand = panda_hand[:top_k]
        scores = scores[:top_k]
    return panda_hand, scores


# ---------------------------------------------------------------------------
# cuRobo bridge: try each candidate, return the first reachable plan
# ---------------------------------------------------------------------------

def panda_hand_pose_to_curobo_payload(
    panda_hand_4x4: np.ndarray,
    current_q: np.ndarray,
    gripper_state: np.ndarray,
) -> dict:
    """Convert a 4x4 panda_hand pose into the dict cuRobo.move() expects.

    Uses ``goal_pose_override`` so the [x,y,z,qw,qx,qy,qz] is sent verbatim and
    bypasses cuRobo's internal scene_pose_matrix transform on target_loc/
    target_quat. ``target_loc``/``target_quat`` are left None — cuRobo seeds
    the goal with current EE pose first, then the override replaces it.
    """
    pos, quat_wxyz = _mat_to_pos_quat_wxyz(panda_hand_4x4)
    goal_pose_override = np.concatenate([pos, quat_wxyz]).tolist()  # [x,y,z,qw,qx,qy,qz]
    return {
        "current_q": np.asarray(current_q),
        "gripper_state": np.asarray(gripper_state),
        "target_loc": None,
        "target_quat": None,
        "goal_pose_override": goal_pose_override,
    }


def plan_first_reachable(
    curobo,
    panda_hand_poses: np.ndarray,
    current_q: np.ndarray,
    gripper_state: np.ndarray,
    extra_move_kwargs: Optional[dict] = None,
    verbose: bool = True,
):
    """Try each candidate panda_hand pose with curobo.move(); return first hit.

    Returns dict with keys: candidate_idx, panda_hand_pose, joint_states,
    joint_vels, final_q, goal_pose, goal_pose_0. None if none succeed.
    """
    extra_move_kwargs = extra_move_kwargs or {}
    n = len(panda_hand_poses)
    for i, mat in enumerate(panda_hand_poses):
        payload = panda_hand_pose_to_curobo_payload(mat, current_q, gripper_state)
        payload.update(extra_move_kwargs)
        result = curobo.move(payload)
        if verbose:
            pos = mat[:3, 3].tolist()
            n_waypoints = (len(result["joint_states"]) if result and result.get("joint_states") is not None else 0)
            goal_used = result.get("goal_pose") if result else None
            print(
                f"  [plan_first_reachable] cand {i+1}/{n} "
                f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                f"-> n_waypoints={n_waypoints} goal_used={goal_used}",
                flush=True,
            )
        if result is None:
            continue
        if result.get("joint_states") is None:
            continue
        return {
            "candidate_idx": int(i),
            "panda_hand_pose": np.asarray(mat),
            **result,
        }
    return None
