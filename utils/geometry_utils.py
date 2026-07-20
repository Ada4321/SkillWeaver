import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def proj_3d_to_2d(
        pts: np.ndarray, ixt: np.ndarray, ext: np.ndarray,
        max_x: int, max_y: int
    ):
    """
    Projects 3D points to 2D using intrinsic and extrinsic camera matrices.

    Args:
        pts (np.ndarray): Array of 3D points of shape (N, 3).
        ixt (np.ndarray): Intrinsic matrix of shape (3, 3).
        ext (np.ndarray): Extrinsic matrix of shape (4, 4). -- world2cam
        max_x: w - 1
        max_y: h - 1

    Returns:
        np.ndarray: Array of 2D projected points of shape (N, 2).
    """
    # Ensure pts is in homogeneous coordinates (N, 4)
    ones = np.ones((pts.shape[0], 1))
    pts_homogeneous = np.hstack((pts, ones))  # Shape: (N, 4)

    # Apply extrinsic matrix (4x4) to transform 3D points
    pts_camera = ext @ pts_homogeneous.T  # Shape: (4, N)

    # Apply intrinsic matrix (3x3) to project onto image plane
    pts_image = ixt @ pts_camera[:3, :]  # Shape: (3, N)

    # Normalize by the third (z) coordinate to get 2D points
    pts_2d = pts_image[:2, :] / pts_image[2, :]  # Shape: (2, N)

    # Transpose to get shape (N, 2) and convert to integers
    pts_2d = np.round(pts_2d.T).astype(int)  # Shape: (N, 2)

    # Clamp coordinates to ensure they do not exceed max_x and max_y
    pts_2d[:, 0] = np.clip(pts_2d[:, 0], 0, max_x)  # Clamp x-coordinates
    pts_2d[:, 1] = np.clip(pts_2d[:, 1], 0, max_y)  # Clamp y-coordinates
    
    return pts_2d


def proj_3d_to_2d_batched(
        pts: np.ndarray, ixt: np.ndarray, ext: np.ndarray,
        max_x: int, max_y: int
    ):
    """
    Projects batched 3D points to 2D using intrinsic and extrinsic camera matrices.

    Args:
        pts (np.ndarray): Batched array of 3D points of shape (B, N, 3).
        ixt (np.ndarray): Batched intrinsic matrices of shape (B, 3, 3).
        ext (np.ndarray): Batched extrinsic matrices of shape (B, 4, 4). -- world2cam
        max_x: w - 1
        max_y: h - 1

    Returns:
        np.ndarray: Batched array of 2D projected points of shape (B, N, 2).
    """
    # ---- sanitize pts ----
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 2:
        pts = pts[None, ...]           # (1,N,3)
    assert pts.ndim == 3 and pts.shape[-1] == 3, f"pts must be (B,N,3); got {pts.shape}"
    B, N, _ = pts.shape

    # ---- broadcast intrinsics/extrinsics to (B,*,*) ----
    ixt = np.asarray(ixt, dtype=np.float32)
    if ixt.ndim == 2:
        ixt = np.broadcast_to(ixt, (B, 3, 3))
    elif ixt.shape[0] != B:
        ixt = np.broadcast_to(ixt, (B, 3, 3))

    ext = np.asarray(ext, dtype=np.float32)
    if ext.ndim == 2:
        ext = np.broadcast_to(ext, (B, 4, 4))
    elif ext.shape[0] != B:
        ext = np.broadcast_to(ext, (B, 4, 4))

    # ---- projection begins ----
    B, N, _ = pts.shape

    # Ensure pts is in homogeneous coordinates (B, N, 4)
    ones = np.ones((B, N, 1))
    pts_homogeneous = np.concatenate((pts, ones), axis=-1)  # Shape: (B, N, 4)

    # Apply extrinsic matrix (B, 4, 4) to transform 3D points
    pts_camera = np.einsum('bij,bnj->bni', ext, pts_homogeneous)  # Shape: (B, N, 4)

    # Apply intrinsic matrix (B, 3, 3) to project onto image plane
    pts_image = np.einsum('bij,bnj->bni', ixt, pts_camera[:, :, :3])  # Shape: (B, N, 3)

    # Normalize by the third (z) coordinate to get 2D points
    pts_2d = pts_image[:, :, :2] / pts_image[:, :, 2:3]  # Shape: (B, N, 2)

    # Round and convert to integers
    pts_2d = np.round(pts_2d).astype(int)  # Shape: (B, N, 2)

    # Clamp coordinates to ensure they do not exceed max_x and max_y
    pts_2d[:, :, 0] = np.clip(pts_2d[:, :, 0], 0, max_x)  # Clamp x-coordinates
    pts_2d[:, :, 1] = np.clip(pts_2d[:, :, 1], 0, max_y)  # Clamp y-coordinates

    return pts_2d


def rigid_transform(pts: np.ndarray, pose: np.ndarray):
    """
    Args:
        pts (np.ndarray): Array of 3D points of shape (N, 3).
        pose (np.ndarray): Extrinsic matrix of shape (4, 4).
    """
    ones = np.ones((pts.shape[0], 1))
    pts_homogeneous = np.hstack((pts, ones))  # Shape: (N, 4)

    pts_transformed = pose @ pts_homogeneous.T
    pts_transformed = pts_transformed.T[:, :3]
    return pts_transformed


def rigid_transform_batched(pts: np.ndarray, poses: np.ndarray):
    """
    pts:   (N, 3)
    poses: (n_pose, 4, 4)
    return: (n_pose, N, 3)
    """
    N = pts.shape[0]
    ones = np.ones((N, 1), dtype=pts.dtype)
    pts_h = np.hstack([pts, ones]).T           # (4, N)

    # Broadcast the same (4,N) points to all batches of (n_pose,4,4)
    out = np.einsum('nij,jk->nik', poses, pts_h)  # (n_pose, 4, N)
    out = out[:, :3, :].transpose(0, 2, 1)        # (n_pose, N, 3)
    return out


def lift_pixel_to_world(depth, intrinsic, extrinsic, reshape=False):
    if len(depth.shape) > 2:
        depth = depth.squeeze()  # (H, W)
    
    H, W = depth.shape
    y, x = np.indices((H, W), dtype=np.float32)             # y=row, x=col
    coords_2d_h = np.stack([x, y, np.ones_like(x)], axis=-1)  # (H, W, 3) -> [x, y, 1]

    # coords_2d = np.indices((depth.shape[0], depth.shape[1]), dtype=np.float32).transpose(1, 2, 0)
    # coords_2d_h = np.concatenate([coords_2d, np.ones_like(coords_2d[..., :1])], axis=-1)  # (H, W, 3)
    coords_2d_h = coords_2d_h.reshape(-1, 3)  # (H*W, 3)
    depth_flat = depth.reshape(-1)  # (H*W,)

    intrinsic_inv = np.linalg.inv(intrinsic)
    cam_points = (intrinsic_inv @ coords_2d_h.T).T * depth_flat[:, None]  # (H*W, 3)
    cam_points_h = np.concatenate([cam_points, np.ones_like(cam_points[..., :1])], axis=-1)  # (H*W, 4)

    extrinsic_inv = np.linalg.inv(extrinsic)
    world_points_h = (extrinsic_inv @ cam_points_h.T).T  # (H*W, 4)
    world_points = world_points_h[:, :3] / world_points_h[:, 3:]  # (H*W, 3)

    if reshape:
        return world_points.reshape(depth.shape[0], depth.shape[1], 3)  # (H, W, 3)
    else:
        return world_points  # (H*W, 3)
    

def R2q_wxyz(R_mat):
    quat_xyzw = R.from_matrix(R_mat).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return quat_wxyz


def R2q_xyzw(R_mat):
    quat_xyzw = R.from_matrix(R_mat).as_quat()
    return quat_xyzw


def q2R_wxyz(q):
    R_mat = R.from_quat(np.array([q[1], q[2], q[3], q[0]])).as_matrix()
    return R_mat


def q2R_xyzw(q):
    R_mat = R.from_quat(np.array(q)).as_matrix()
    return R_mat


def quat_world_from_ee_axes(approach_world, finger_world, approach_local, finger_local):
    """Build the EE orientation quat (wxyz) that maps the EE's local approach/finger
    axes onto the desired world directions.

    Solves R(world<-ee) with R @ approach_local = approach_world and
    R @ finger_local = finger_world (finger orthogonalized against approach).
    Generalizes the hand-built ``column_stack([x,y,z])`` grasp basis: passing the
    panda convention (approach_local=[0,0,1], finger_local=[0,1,0]) reproduces it.
    """
    def _n(v):
        v = np.asarray(v, dtype=np.float64)
        return v / (np.linalg.norm(v) + 1e-12)

    a_w = _n(approach_world)
    f_w = np.asarray(finger_world, dtype=np.float64)
    f_w = _n(f_w - np.dot(f_w, a_w) * a_w)  # orthogonalize finger against approach
    B_world = np.column_stack([a_w, f_w, np.cross(a_w, f_w)])

    a_l = _n(approach_local)
    f_l = _n(finger_local)
    f_l = _n(f_l - np.dot(f_l, a_l) * a_l)
    B_local = np.column_stack([a_l, f_l, np.cross(a_l, f_l)])

    return R2q_wxyz(B_world @ B_local.T)


def T2qt(T_mat, with_wxyz=True):
    R_mat = T_mat[:3, :3]
    t = T_mat[:3, 3]
    if with_wxyz:
        q = R2q_wxyz(R_mat)
    else:
        q = R2q_xyzw(R_mat)
    return q, t


def qt2T(q, t,  with_wxyz=True):
    T = np.eye(4)
    if with_wxyz:
        R_mat = q2R_wxyz(q)
    else:
        R_mat = q2R_xyzw(q)
    T[:3, :3] = R_mat
    T[:3, 3] = np.array(t)
    return T


def to_tensor(input_data, device):
    if isinstance(input_data, torch.Tensor):
        return input_data.to(device)
    else:
        return torch.tensor(input_data, device=device)
    

def _quat_to_R(q: np.ndarray, quat_order: str) -> np.ndarray:
    if quat_order == "wxyz":
        return q2R_wxyz(q)
    if quat_order == "xyzw":
        return q2R_xyzw(q)
    raise ValueError(f"Unsupported quaternion order: {quat_order}")


def _R_to_quat(R_mat: np.ndarray, quat_order: str) -> np.ndarray:
    if quat_order == "wxyz":
        return R2q_wxyz(R_mat)
    if quat_order == "xyzw":
        return R2q_xyzw(R_mat)
    raise ValueError(f"Unsupported quaternion order: {quat_order}")


def axis_to_vec(axis: str) -> np.ndarray:
    if axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if axis == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    raise ValueError(f"Unknown axis: {axis}")


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis.astype(np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = axis / axis_norm
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-8 or b_norm < 1e-8:
        return np.eye(3, dtype=np.float64)

    a = a / a_norm
    b = b / b_norm
    c = np.dot(a, b)

    if c > 1.0 - 1e-6:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-6:
        # 180 deg rotation: pick any axis orthogonal to a.
        if abs(a[0]) < 0.9:
            ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, ortho)
        return _rotation_from_axis_angle(axis, np.pi)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def compute_alignment_rotation(
    held_quat: np.ndarray,
    target_quat: np.ndarray,
    alignment,
    quat_order: str = "wxyz",
):
    held_axis, target_axis, direction = alignment
    held_axis_vec = axis_to_vec(held_axis)
    target_axis_vec = axis_to_vec(target_axis)
    if direction == "-":
        target_axis_vec = -target_axis_vec

    held_R = _quat_to_R(np.asarray(held_quat, dtype=np.float32), quat_order)
    target_R = _quat_to_R(np.asarray(target_quat, dtype=np.float32), quat_order)

    held_axis_world = held_R @ held_axis_vec
    target_axis_world = target_R @ target_axis_vec

    # Minimal rotation that aligns held_axis_world to target_axis_world.
    delta_R = rotation_matrix_from_vectors(held_axis_world, target_axis_world)
    new_held_R = delta_R @ held_R

    # return goal quaternion of held object
    return _R_to_quat(new_held_R, quat_order)


_DEFAULT_ALIGNMENT_FREE_AXIS_SWEEP_DEG = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)


def _axis_angle_from_R(R_mat: np.ndarray):
    """Decompose a rotation matrix into (unit axis, angle in [0, pi])."""
    R_mat = np.asarray(R_mat, dtype=np.float64)
    cos_angle = float(np.clip((np.trace(R_mat) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0
    if angle > np.pi - 1e-6:
        # Near 180 deg the skew part vanishes; take the axis from R + I.
        M = R_mat + np.eye(3, dtype=np.float64)
        col = M[:, int(np.argmax(np.diag(M)))]
        norm = float(np.linalg.norm(col))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64), angle
        return col / norm, angle
    axis = np.array(
        [
            R_mat[2, 1] - R_mat[1, 2],
            R_mat[0, 2] - R_mat[2, 0],
            R_mat[1, 0] - R_mat[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * np.sin(angle))
    return axis / np.linalg.norm(axis), angle


def compute_alignment_rotation_candidates(
    held_quat: np.ndarray,
    target_quat: np.ndarray,
    alignment,
    sweep_angles_deg=None,
    quat_order: str = "wxyz",
    relax_ladder_deg=None,
    meta_out=None,
):
    """Return held-object goal quats satisfying the alignment constraint.

    Candidate 0 is the minimal rotation (identical to compute_alignment_rotation).
    The remaining candidates apply an extra rotation about the world-frame
    aligned axis, which preserves the alignment constraint and only sweeps the
    free DoF that the constraint leaves unspecified.

    relax_ladder_deg (e.g. [0, 5, 10, 15]) appends relaxed tiers, ordered
    strict-first: tier phi under-rotates along the same geodesic so the aligned
    axis keeps a residual angle of min(phi, full angle) to the target axis, and
    repeats the free-axis sweep at that tilt. Relaxing toward the CURRENT held
    pose minimizes the wrist rotation the goal demands. Tiers where the full
    angle is already within phi degenerate to no-reorientation and the ladder
    is truncated there. None keeps the strict tier only (previous behavior).

    meta_out: optional list; filled with one dict per candidate (same order):
    {"relax_deg": ladder tier, "residual_deg": actual axis angle left =
    min(tier, full angle), "sweep_deg": free-axis sweep angle}.
    """
    held_axis, target_axis, direction = alignment
    held_axis_vec = axis_to_vec(held_axis)
    target_axis_vec = axis_to_vec(target_axis)
    if direction == "-":
        target_axis_vec = -target_axis_vec

    held_R = _quat_to_R(np.asarray(held_quat, dtype=np.float32), quat_order)
    target_R = _quat_to_R(np.asarray(target_quat, dtype=np.float32), quat_order)

    held_axis_world = held_R @ held_axis_vec
    target_axis_world = target_R @ target_axis_vec

    delta_R = rotation_matrix_from_vectors(held_axis_world, target_axis_world)

    if sweep_angles_deg is None:
        sweep_angles_deg = _DEFAULT_ALIGNMENT_FREE_AXIS_SWEEP_DEG

    if relax_ladder_deg is None:
        ladder = [0.0]
    else:
        ladder = sorted({max(0.0, float(x)) for x in relax_ladder_deg})
        if not ladder or ladder[0] > 1e-9:
            ladder = [0.0] + ladder

    delta_axis, delta_angle = _axis_angle_from_R(delta_R)

    candidates = []
    prev_eff_angle = None
    for phi_deg in ladder:
        eff_angle = max(delta_angle - float(np.deg2rad(phi_deg)), 0.0)
        if prev_eff_angle is not None and abs(eff_angle - prev_eff_angle) < 1e-12:
            break  # already within tolerance; further tiers are identical
        prev_eff_angle = eff_angle
        if phi_deg <= 1e-9:
            tier_R = delta_R  # strict tier: keep bitwise-identical to the old output
        else:
            tier_R = _rotation_from_axis_angle(delta_axis, eff_angle)
        new_held_R = tier_R @ held_R

        free_axis_world = new_held_R @ held_axis_vec  # world-frame held axis after tier_R
        norm = float(np.linalg.norm(free_axis_world))
        if norm > 1e-8:
            free_axis_world = free_axis_world / norm

        residual_deg = float(np.degrees(delta_angle - eff_angle))
        for theta_deg in sweep_angles_deg:
            theta = float(np.deg2rad(float(theta_deg)))
            if abs(theta) < 1e-9:
                cand_R = new_held_R
            else:
                R_extra = _rotation_from_axis_angle(free_axis_world, theta)
                cand_R = R_extra @ new_held_R
            candidates.append(_R_to_quat(cand_R, quat_order))
            if meta_out is not None:
                meta_out.append({
                    "relax_deg": float(phi_deg),
                    "residual_deg": residual_deg,
                    "sweep_deg": float(theta_deg),
                })
    return candidates

# --- quaternion algebra (moved from data_collect_utils) ---
def _quat2axisangle_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    Modifed to process (qw, qx, qy, qz) format.
    """

    # Reorder (w,x,y,z) -> (x,y,z,w)
    x, y, z, w = quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]

    # Clip w for numerical stability
    if w > 1.0:
        w = 1.0
    elif w < -1.0:
        w = -1.0

    den = math.sqrt(max(0.0, 1.0 - w * w))  # = sin(theta/2)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)

    angle = 2.0 * math.acos(w)  # theta
    v = np.array([x, y, z], dtype=np.float64)
    return (v * angle) / den


def quat_mul(q1,q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conj(q): w,x,y,z=q; return np.array([w,-x,-y,-z])


def quat_rotate(q,v):
    return quat_mul(quat_mul(q, np.concatenate([[0],v])), quat_conj(q))[1:]

# --- wxyz quaternion normalize / geodesic distance (moved from skills.py) ---
def _quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_angle_distance_rad_wxyz(q1: np.ndarray, q2: np.ndarray) -> float:
    q1n = _quat_normalize_wxyz(q1)
    q2n = _quat_normalize_wxyz(q2)
    dot = float(np.dot(q1n, q2n))
    dot = min(1.0, max(-1.0, abs(dot)))
    return float(2.0 * np.arccos(dot))
