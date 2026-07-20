"""Place/grasp target-location computation (VLM 3D point -> world target loc/quat).

Leaf module (imports utils + skills._constants only; never skills.skills)."""
import logging
import numpy as np

from skills._constants import CONVEX_UPRIGHT_COS_THRESHOLD
from utils.image_utils import _decode_base64
from utils.random_control import timestamp_seed_scope
from utils.geometry_utils import q2R_wxyz, quat_world_from_ee_axes


# --- object-height helpers ---
def _get_object_height(object_name, simulator, task_env):
    object_range = simulator.get_object_3d_range(task_env, object_name)
    return object_range[-1] - object_range[-2]


def get_object_z_max(object_name, simulator, task_env):
    object_range = simulator.get_object_3d_range(task_env, object_name)
    return object_range[-1]



# --- target-location computation ---
def _maybe_override_xy_for_straighten(move_target_loc_3d, simulator, task_env, held_object_name=None):
    """For the 'straighten' task, lock place xy to the held object's initial
    world xy so the bottle is put back where it started, just standing up.

    No-op for any other task type. Returns the (possibly modified) loc.
    """
    if move_target_loc_3d is None:
        return move_target_loc_3d
    try:
        task_type = getattr(getattr(simulator, "args", None), "task_type", None)
        if str(task_type or "").lower() != "straighten":
            return move_target_loc_3d
        if held_object_name is None:
            grasped_names, _ = simulator.get_grasped_and_collided_objs(task_env)
            if not grasped_names:
                return move_target_loc_3d
            held_object_name = grasped_names[0]
        init_poses = getattr(simulator.args, "init_object_poses", None) or {}
        init_pose = init_poses.get(held_object_name)
        if not init_pose or len(init_pose) < 2:
            return move_target_loc_3d
        prev_xy = (float(move_target_loc_3d[0]), float(move_target_loc_3d[1]))
        move_target_loc_3d[0] = float(init_pose[0])
        move_target_loc_3d[1] = float(init_pose[1])
        print(
            f"[STRAIGHTEN_XY_OVERRIDE] held={held_object_name} "
            f"vlm_xy=({prev_xy[0]:.3f},{prev_xy[1]:.3f}) -> "
            f"init_xy=({move_target_loc_3d[0]:.3f},{move_target_loc_3d[1]:.3f})",
            flush=True,
        )
    except Exception as _exc:
        logging.warning("straighten xy override failed: %s", _exc)
    return move_target_loc_3d


def get_surface_target_loc_from_vlm3d(pcd, vlm_3d_point, target_asset_name=None, simulator=None, task_env=None):
    # When target is the table, the PCD xy-nearest-neighbor can land on top of
    # another object sitting on the table (e.g., a jar), inflating surface_z.
    # Use the table's actual top z instead so place offsets are measured from
    # the bare table surface.
    if (
        target_asset_name == "table"
        and simulator is not None
        and task_env is not None
    ):
        target_loc = np.array(vlm_3d_point, dtype=float).reshape(3,)
        target_loc[2] = float(get_object_z_max("table", simulator, task_env))
        return target_loc

    if isinstance(pcd, str):
        pcd = _decode_base64(pcd).reshape(-1, 3)
    pcd = pcd[~np.isnan(pcd).any(axis=1)]  # (N, 3)
    vlm_3d_point = np.array(vlm_3d_point).reshape(1, 3)  # (1, 3)
    # find the nearest point in pcd to vlm_3d_point in xy plane
    dists = np.linalg.norm(pcd[:, :2] - vlm_3d_point[:, :2], axis=1)  # (N,)
    nearest_idx = np.argmin(dists)
    nearest_point = pcd[nearest_idx]  # (3,)

    target_loc = vlm_3d_point.copy().reshape(3,)
    target_loc[2] = nearest_point[2]  # use the z value of the nearest point
    return target_loc


def get_container_opening_center_from_vlm3d(vlm_3d_point, container_name, simulator, task_env):
    vlm_3d_point = np.array(vlm_3d_point).reshape(1, 3)  # (1, 3)
    container_height = get_object_z_max(container_name, simulator, task_env)
    target_loc = vlm_3d_point.copy().reshape(3,)
    target_loc[2] = container_height
    return target_loc


def _compute_pre_grasp_quat_for_convex(simulator, task_env, object_name, upright_cos_threshold=CONVEX_UPRIGHT_COS_THRESHOLD):
    """
    Build pre-grasp ee target quat (wxyz) for a convex object such that:
      - panda_hand +Z aligned with world -Z (gripper points down)
      - panda_hand +Y is perpendicular to the object's handle axis projected
        to the world horizontal plane; handle axis = whichever of object's
        canonical local x or y has the larger pcd extent
      - between the two ±90deg solutions, pick the one closer to the current
        wrist Y direction to minimize yaw rotation

    Returns None when the object is not upright (local +Z misaligned with
    world +Z by more than ~26deg) or when the handle horizontal projection
    is degenerate.
    """
    obj_pose = simulator.get_object_pose(task_env, object_name)
    obj_R = q2R_wxyz(np.asarray(obj_pose[3:], dtype=float))

    z_local_in_world = obj_R @ np.array([0.0, 0.0, 1.0])
    if z_local_in_world[2] < upright_cos_threshold:
        return None

    x_ext, y_ext = simulator.get_object_canonical_xy_extents(task_env, object_name)
    handle_local = np.array([1.0, 0.0, 0.0]) if x_ext >= y_ext else np.array([0.0, 1.0, 0.0])

    handle_world = obj_R @ handle_local
    handle_xy = np.array([handle_world[0], handle_world[1], 0.0])
    n = np.linalg.norm(handle_xy)
    if n < 1e-6:
        return None
    handle_xy /= n

    y_hand_a = np.array([-handle_xy[1], handle_xy[0], 0.0])
    y_hand_b = -y_hand_a

    ee_pose = simulator.get_ee_pose(task_env)
    cur_R = q2R_wxyz(np.asarray(ee_pose[3:], dtype=float))
    cur_finger_world = cur_R @ simulator.ee_finger_local
    y_hand = y_hand_a if np.dot(cur_finger_world, y_hand_a) >= np.dot(cur_finger_world, y_hand_b) else y_hand_b

    # Build the grasp orientation in terms of the robot's EE axis convention:
    # approach axis points down (world -Z), finger axis perpendicular to handle.
    return quat_world_from_ee_axes(
        approach_world=np.array([0.0, 0.0, -1.0]),
        finger_world=y_hand,
        approach_local=simulator.ee_approach_local,
        finger_local=simulator.ee_finger_local,
    )


def get_3d_target_grasp(node, child_node, simulator, task_env):
    if child_node.action_params["asset_id"] is None or child_node.action_params["asset"] is None:
        return None

    asset = child_node.action_params["asset"]
    snapshot = node.snapshot
    object_center_3d = snapshot["object_center_3d"][asset]
    z_max = snapshot["object_range_3d"][asset][-1]

    pre_grasp_location_3d = np.array(object_center_3d, dtype=np.float32)
    pre_grasp_location_3d[2] = z_max + simulator.pregrasp_offset
    pregrasp_noise_std_xyz = np.asarray(
        getattr(simulator, "pregrasp_noise_std_xyz", [0.01, 0.01, 0.01]), dtype=np.float32
    ).reshape(3,)
    with timestamp_seed_scope():
        pre_grasp_location_3d = pre_grasp_location_3d + np.random.normal(
            loc=0.0, scale=pregrasp_noise_std_xyz, size=(3,)
        )
    return pre_grasp_location_3d


def get_3d_target_place(node, child_node, simulator, task_env, mode="surface"):
    if child_node.action_params["point_3d"] is None:
        return None
    
    pcd = node.observations["pcd_all_views"][child_node.planning_params["view"]]
    point_3d = child_node.action_params["point_3d"]

    if mode == "surface":
        move_target_loc_3d = get_surface_target_loc_from_vlm3d(
            pcd, point_3d,
            target_asset_name=child_node.action_params.get("asset"),
            simulator=simulator,
            task_env=task_env,
        )
    elif mode == "container":
        move_target_loc_3d = get_container_opening_center_from_vlm3d(point_3d, container_name=child_node.action_params["asset"], simulator=simulator, task_env=task_env)

    grasped_names = node.snapshot["grasped_names"]
    if grasped_names:
        object_height = _get_object_height(grasped_names[0], simulator, task_env)
        if object_height is not None:
            move_target_loc_3d[2] += 0.5 * object_height
    return move_target_loc_3d


def _compute_cup_bottom_xy_offset_world(simulator, task_env, object_name, goal_quat_wxyz):
    # World-frame xy offset from object root (AABB center) to the cup body's
    # bottom-circle center, given the orientation the object will have at goal.
    # Returns None when the object lacks handle_side or axis-length metadata.
    handle_side = simulator.get_object_handle_side(task_env, object_name)
    if handle_side not in ("+x", "-x", "+y", "-y"):
        return None
    Lx, Ly, Lz = simulator.get_object_axis_lengths(task_env, object_name)
    if handle_side in ("+y", "-y"):
        sign = 1.0 if handle_side == "+y" else -1.0
        local = np.array([0.0, -sign * 0.5 * (Ly - Lx), 0.0], dtype=np.float64)
    else:
        sign = 1.0 if handle_side == "+x" else -1.0
        local = np.array([-sign * 0.5 * (Lx - Ly), 0.0, 0.0], dtype=np.float64)
    R = q2R_wxyz(np.asarray(goal_quat_wxyz, dtype=np.float64))
    world = R @ local
    world[2] = 0.0
    return world


def _apply_place_noise(target_loc_3d, simulator):
    place_noise_std_xyz = np.asarray(
        getattr(simulator, "place_noise_std_xyz", [0.0, 0.0, 0.0]), dtype=np.float32
    ).reshape(3,)
    with timestamp_seed_scope():
        return target_loc_3d + np.random.normal(
            loc=0.0, scale=place_noise_std_xyz, size=(3,)
        )


def _resolve_target_quat_candidates(predefined_target_quat):
    """Normalize predefined_target_quat to a list of candidates for the cuRobo
    planning loop in execute_move_held_object.

    - None  -> [None] (no orientation constraint, single planning attempt)
    - 1D quat (len 4) -> [quat]
    - 2D ndarray (N, 4) or list-of-quats -> list with N candidates, in order
    """
    if predefined_target_quat is None:
        return [None]
    arr = np.asarray(predefined_target_quat)
    if arr.ndim == 1:
        return [predefined_target_quat]
    if arr.ndim == 2 and arr.shape[1] == 4:
        return [np.asarray(arr[i]) for i in range(arr.shape[0])]
    if isinstance(predefined_target_quat, (list, tuple)) and all(
        np.asarray(q).shape == (4,) for q in predefined_target_quat
    ):
        return list(predefined_target_quat)
    raise ValueError(
        f"predefined_target_quat must be None, a length-4 quat, or a list/array of "
        f"length-4 quats; got shape={getattr(arr, 'shape', None)}"
    )



def _resolve_place_target_base(node, child_node, simulator, task_env, mode, held_object_name=None):
    """Shared place-target base resolver (drop family: mode="container"; RL family:
    mode="surface"). Part target (asset = part key + asset_object) wins for both; otherwise the
    mode-specific base — container: opening top z_max + half held-object height
    (get_3d_target_place); surface: nearest-pcd/table surface z, NO half-height (the RL
    finish lifts it once the goal orientation is known). The straighten xy override
    applies in all cases. Returns (loc_or_None, is_part_target)."""
    loc = _place_part_target_or_none(node, child_node, simulator, task_env)
    is_part_target = loc is not None
    if not is_part_target:
        if child_node.action_params.get("asset_object") is not None:
            # Part place whose target resolution failed entirely: hard-fail here. Falling
            # through to the object branch would feed the PART KEY into name-keyed lookups
            # (get_object_z_max(part_key) -> KeyError crash instead of a clean failure).
            return None, False
        if mode == "container":
            loc = get_3d_target_place(node, child_node, simulator, task_env, mode="container")
        else:
            pcd = node.observations["pcd_all_views"][child_node.planning_params["view"]]
            loc = get_surface_target_loc_from_vlm3d(
                pcd,
                child_node.action_params["point_3d"],
                target_asset_name=child_node.action_params.get("asset"),
                simulator=simulator,
                task_env=task_env,
            )
    loc = _maybe_override_xy_for_straighten(loc, simulator, task_env, held_object_name=held_object_name)
    return loc, is_part_target


# --- articulated-part place target (shared drop/RL) ---
def _place_part_target_or_none(node, child_node, simulator, task_env):
    """Shared place-target resolver for the ARTICULATED-PART case (drawer or door part). If
    this place targets a part (child_node.action_params['asset'] is a part key with
    'asset_object' = the owning object -- from a VLM part-index selection), return the
    target loc: xy from the VLM point_3d,
    z = the drawer's INTERIOR FLOOR taken from the depth point cloud, + half the held object's
    height so the object rests on the floor. The drawer IS just a surface place -- its floor z is
    read via the same get_surface_target_loc_from_vlm3d helper at the drawer's center xy (the
    drawer isn't a snapshot object, but its floor is in the pcd when the open drawer is
    camera-visible). This is more accurate than the collision-box AABB, whose min z is the box's
    OUTER bottom (would sink the object) and max z is the top rim (would sit it too high and block
    the drawer closing). Falls back to the geometry AABB floor (min z) if the pcd lookup fails.
    Returns None when this is not a drawer place. drop/RL share this; xy/height/downstream shared."""
    part = child_node.action_params.get("asset")
    owner = child_node.action_params.get("asset_object")
    point_3d = child_node.action_params.get("point_3d")
    if owner is None or part is None or point_3d is None:
        return None
    # Interior floor z from the point cloud (reuse the surface-place helper at the drawer center xy).
    loc = None
    try:
        pcd = node.observations["pcd_all_views"][child_node.planning_params["view"]]
        loc = get_surface_target_loc_from_vlm3d(
            pcd, point_3d, target_asset_name=owner, simulator=simulator, task_env=task_env,
        )
    except Exception:
        loc = None
    if loc is None:
        # Fallback: geometry AABB floor (min z) when the pcd floor lookup is unavailable.
        try:
            rng = simulator.get_object_3d_range(task_env, owner, part=part)
        except Exception:
            return None
        loc = np.array([float(point_3d[0]), float(point_3d[1]), float(rng[4])], dtype=float)
    grasped_names = node.snapshot.get("grasped_names") if isinstance(node.snapshot, dict) else None
    if grasped_names:
        object_height = _get_object_height(grasped_names[0], simulator, task_env)
        if object_height is not None:
            loc[2] += 0.5 * float(object_height)
    return loc

