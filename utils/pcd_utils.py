"""Point-cloud spatial queries used by the VLM place/grasp target computation.

Moved out of geometry_utils to keep that module pure geometry. These take a point
cloud (optionally base64-encoded) + a VLM 3D point / object mask and return 3D locations.
"""
import numpy as np
from utils.image_utils import _decode_base64


def get_3d_location_from_2d(pcd, uv):
    if isinstance(pcd, str):
        pcd = _decode_base64(pcd)
    ## get 3d position of the placement point on the object surface
    ## in curobo coord system
    pos_2d = np.array(uv)  # y, x 
    pos_2d = np.round(pos_2d).astype(np.int32)
    pos_3d = pcd[int(pos_2d[1]), int(pos_2d[0])]  # (3,)
    return pos_3d


def get_target_loc_from_vlm3d(pcd, vlm_3d_point):
    if isinstance(pcd, str):
        pcd = _decode_base64(pcd).reshape(-1, 3)
    pcd = pcd[~np.isnan(pcd).any(axis=1)]  # (N, 3)
    vlm_3d_point = np.array(vlm_3d_point).reshape(1, 3)  # (1, 3)
    # find the nearest point in pcd to vlm_3d_point in xy plane
    dists = np.linalg.norm(pcd[:, :2] - vlm_3d_point[:, :2], axis=1)  # (N,)
    nearest_idx = np.argmin(dists)
    nearest_point = pcd[nearest_idx]  # (3,)

    # import ipdb; ipdb.set_trace()

    target_loc = vlm_3d_point.copy().reshape(3,)
    target_loc[2] = nearest_point[2]  # use the z value of the nearest point
    return target_loc


def get_3d_location_at_pointcloud_top_center(pcd, obj_mask):
    # xmin, ymin, zmin, xmax, ymax, zmax = get_aabb_from_pcd_mask(obj_poicloud, view_mask)
    # top_center_x = float(xmin + xmax) / 2
    # top_center_y = float(ymin + ymax) / 2
    # top_center_z = float(zmax)
    if isinstance(pcd, str):
        pcd = _decode_base64(pcd).reshape(-1, 3)
    if isinstance(obj_mask, str):
        obj_mask = _decode_base64(obj_mask)
    if len(obj_mask.shape) == 3:
        obj_mask = obj_mask[0]
    obj_mask = obj_mask.reshape(-1,).astype(np.bool_)
    obj_pcd = pcd[obj_mask]

    # top_obj_center = get_obb_top_surface_center(obj_pcd)
    top_obj_center = get_aabb_top_surface_center(obj_pcd)
    return top_obj_center


def get_aabb_top_surface_center(obj_pcd):
    """
    Get the center of the top surface of the Axis-Aligned Bounding Box (AABB) of a point cloud.

    Args:
        view_point_cloud (np.ndarray): A 2D numpy array of shape (N, 3) representing the point cloud.

    Returns:
        np.ndarray: A 1D numpy array representing the center of the top surface.
    """
    # Compute AABB
    # min_coords = obj_pcd.min(axis=0)  # [min_x, min_y, min_z]
    # max_coords = obj_pcd.max(axis=0)  # [max_x, max_y, max_z]
    min_coords = np.percentile(obj_pcd, 1, axis=0)  # [min_x, min_y, min_z]
    max_coords = np.percentile(obj_pcd, 99, axis=0)  # [max_x, max_y, max_z]

    # Calculate center of the top surface
    top_center = np.array([
        (min_coords[0] + max_coords[0]) / 2,
        (min_coords[1] + max_coords[1]) / 2,
        max_coords[2]
    ])

    return top_center


def get_obb_top_surface_center(obj_pcd):
    import open3d as o3d
    """
    Get the center of the top surface of the Oriented Bounding Box (OBB) of a point cloud.

    Args:
        view_point_cloud (np.ndarray): A 2D numpy array of shape (N, 3) representing the point cloud.

    Returns:
        np.ndarray: A 1D numpy array representing the center of the top surface.
    """
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_pcd)

    # Compute the Oriented Bounding Box (OBB)
    obb = pcd.get_oriented_bounding_box()

    # Get the vertices of the OBB
    obb_vertices = np.asarray(obb.get_box_points())

    # Identify the top surface (highest z-coordinate in OBB's local frame)
    top_z = np.max(obb_vertices[:, 2])
    top_surface_points = obb_vertices[obb_vertices[:, 2] == top_z]

    # Compute the center of the top surface
    top_center = np.mean(top_surface_points, axis=0)

    return top_center


def get_poke_locations(pcd, obj_mask):
    if isinstance(pcd, str):
        pcd = _decode_base64(pcd).reshape(-1, 3)
    if isinstance(obj_mask, str):
        obj_mask = _decode_base64(obj_mask)
    if len(obj_mask.shape) == 3:
        obj_mask = obj_mask[0]
    obj_mask = obj_mask.reshape(-1,).astype(np.bool_)
    obj_pcd = pcd[obj_mask]

    # Compute AABB
    min_coords = obj_pcd.min(axis=0)  # [min_x, min_y, min_z]
    max_coords = obj_pcd.max(axis=0)  # [max_x, max_y, max_z]

    # Calculate centers of the bounding box faces
    center_front = [max_coords[0], (min_coords[1] + max_coords[1]) / 2, (min_coords[2] + max_coords[2]) / 2]
    center_back = [min_coords[0], (min_coords[1] + max_coords[1]) / 2, (min_coords[2] + max_coords[2]) / 2]
    center_left = [(min_coords[0] + max_coords[0]) / 2, min_coords[1], (min_coords[2] + max_coords[2]) / 2]
    center_right = [(min_coords[0] + max_coords[0]) / 2, max_coords[1], (min_coords[2] + max_coords[2]) / 2]

    return center_front, center_back, center_left, center_right

