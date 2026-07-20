"""Visualize Franka robot point cloud + wrist-camera link frame at a fixed qpos.

Headless on babel: outputs three files next to this script:
  - robot_pcd.ply               fused, per-link colored point cloud (world frame)
  - wrist_camera_pose.txt       4x4 world matrix + xyz + rpy of wrist_camera_link
  - viz_robot_pcd_wrist_cam.png matplotlib 3D scatter w/ wrist-camera + world axes triads

Run:
  /home/hez2/miniconda3/envs/isaaclab/bin/python \
      /home/hez2/code/Grounded_MCTS_IsaacLab/assets/franka_gripper/viz_robot_pcd_wrist_cam.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import yourdfpy

HERE = Path(__file__).resolve().parent
URDF_PATH = HERE / "robots" / "franka_gripper_libero_obj_visual.urdf"
MESH_ROOT = HERE  # `package://meshes/...` resolves under this dir
OUT_PLY = HERE / "robot_pcd.ply"
OUT_POSE = HERE / "wrist_camera_pose.txt"
OUT_PNG = HERE / "viz_robot_pcd_wrist_cam.png"
OUT_PNG_ZOOM = HERE / "viz_robot_pcd_wrist_cam_zoom.png"

JOINT_POS = {
    "panda_joint1": 0.3463,
    "panda_joint2": -0.0387,
    "panda_joint3": -0.3453,
    "panda_joint4": -2.3377,
    "panda_joint5": -0.0176,
    "panda_joint6": 2.3012,
    "panda_joint7": 0.7983,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}
POINTS_PER_LINK = 2000
AXIS_LEN = 0.18  # meters, length of drawn triad axes
AXIS_LW = 5.0  # line width for axis arrows
AXIS_HEAD = 0.04  # arrowhead size (meters)
PCD_ALPHA = 0.35  # point cloud alpha so arrows pop


def _filename_handler(fname: str) -> str:
    # yourdfpy passes raw mesh paths; rewrite `package://...` to absolute.
    if fname.startswith("package://"):
        rel = fname[len("package://") :]
        return str(MESH_ROOT / rel)
    if not os.path.isabs(fname):
        return str(MESH_ROOT / fname)
    return fname


def load_urdf() -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        str(URDF_PATH),
        filename_handler=_filename_handler,
        build_scene_graph=True,
        load_meshes=True,
    )


def sample_link_pcd(geom, n_points: int) -> np.ndarray:
    # geom can be trimesh.Trimesh or trimesh.Scene
    if isinstance(geom, trimesh.Scene):
        geom = trimesh.util.concatenate(
            [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    if not isinstance(geom, trimesh.Trimesh) or geom.faces is None or len(geom.faces) == 0:
        return np.zeros((0, 3))
    pts, _ = trimesh.sample.sample_surface(geom, n_points)
    return np.asarray(pts)


def collect_pcd(urdf: yourdfpy.URDF):
    cmap = plt.get_cmap("tab20")
    all_pts = []
    all_cols = []
    link_names = [ln for ln in urdf.link_map.keys()]
    for idx, link_name in enumerate(link_names):
        link = urdf.link_map[link_name]
        if not link.visuals:
            continue
        T_world_link = urdf.get_transform(link_name)
        link_pts_local = []
        for visual in link.visuals:
            if visual.geometry is None or visual.geometry.mesh is None:
                continue
            # `visual.geometry.mesh.filename` is original path; loaded geom is cached on the urdf scene
            # Re-load via trimesh using resolved path to keep things explicit.
            mesh_path = _filename_handler(visual.geometry.mesh.filename)
            try:
                geom = trimesh.load(mesh_path, force="mesh")
            except Exception as e:
                print(f"[skip] {link_name} <- {mesh_path}: {e}")
                continue
            if visual.geometry.mesh.scale is not None:
                geom.apply_scale(np.asarray(visual.geometry.mesh.scale))
            # visual origin (relative to the link)
            T_link_visual = visual.origin if visual.origin is not None else np.eye(4)
            pts_mesh = sample_link_pcd(geom, max(POINTS_PER_LINK // max(len(link.visuals), 1), 200))
            if pts_mesh.size == 0:
                continue
            pts_link = (T_link_visual[:3, :3] @ pts_mesh.T).T + T_link_visual[:3, 3]
            link_pts_local.append(pts_link)
        if not link_pts_local:
            continue
        pts_link = np.concatenate(link_pts_local, axis=0)
        pts_world = (T_world_link[:3, :3] @ pts_link.T).T + T_world_link[:3, 3]
        color = (np.asarray(cmap(idx % cmap.N)[:3]) * 255).astype(np.uint8)
        all_pts.append(pts_world)
        all_cols.append(np.tile(color, (pts_world.shape[0], 1)))
        print(f"  {link_name:24s}  +{pts_world.shape[0]:5d} pts")
    return np.concatenate(all_pts, 0), np.concatenate(all_cols, 0)


def rotmat_to_rpy(R: np.ndarray) -> np.ndarray:
    # XYZ-fixed (= roll about x, pitch about y, yaw about z) URDF convention
    sy = -R[2, 0]
    cy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if cy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(sy, cy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(sy, cy)
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def write_pose(T: np.ndarray, path: Path) -> None:
    pos = T[:3, 3]
    rpy = rotmat_to_rpy(T[:3, :3])
    with open(path, "w") as f:
        f.write("# Wrist camera link world pose (URDF link frame)\n")
        f.write("# 4x4 matrix (row-major):\n")
        for r in T:
            f.write("  " + " ".join(f"{v: .8f}" for v in r) + "\n")
        f.write(f"\n# xyz (m): {pos[0]: .6f} {pos[1]: .6f} {pos[2]: .6f}\n")
        f.write(f"# rpy (rad, XYZ): {rpy[0]: .6f} {rpy[1]: .6f} {rpy[2]: .6f}\n")
        f.write(
            f"# rpy (deg, XYZ): {np.rad2deg(rpy[0]): .4f} "
            f"{np.rad2deg(rpy[1]): .4f} {np.rad2deg(rpy[2]): .4f}\n"
        )


def save_ply(pts: np.ndarray, cols: np.ndarray, path: Path) -> None:
    # Minimal binary-less ASCII PLY writer; keeps deps small.
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def draw_triad(ax, T: np.ndarray, length: float, label: str | None = None) -> None:
    o = T[:3, 3]
    R = T[:3, :3]
    colors = ["#ff1f1f", "#1fbf1f", "#1f4fff"]
    axis_names = ["X", "Y", "Z"]
    for i in range(3):
        d = R[:, i] * length
        ax.quiver(
            o[0], o[1], o[2],
            d[0], d[1], d[2],
            color=colors[i],
            linewidth=AXIS_LW,
            arrow_length_ratio=AXIS_HEAD / length,
        )
        tip = o + d
        ax.text(
            tip[0], tip[1], tip[2], axis_names[i],
            fontsize=11, color=colors[i], weight="bold",
        )
    if label:
        ax.text(
            o[0], o[1], o[2] - length * 0.25,
            label, fontsize=10, weight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
        )
    # origin marker so the joint of the triad is obvious
    ax.scatter([o[0]], [o[1]], [o[2]], c="k", s=18, depthshade=False)


def _render(pts, cols, T_cam, path, *, zoom=False, elev=22, azim=45):
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    stride = max(len(pts) // 8000, 1)
    sub = pts[::stride]
    sub_c = cols[::stride] / 255.0
    ax.scatter(
        sub[:, 0], sub[:, 1], sub[:, 2],
        c=sub_c, s=0.6, depthshade=False, alpha=PCD_ALPHA,
    )

    ax_len = AXIS_LEN * (0.6 if zoom else 1.0)
    draw_triad(ax, np.eye(4), ax_len, label="world")
    draw_triad(ax, T_cam, ax_len, label="wrist_cam")

    if zoom:
        cam_o = T_cam[:3, 3]
        half = 0.18
        ax.set_xlim(cam_o[0] - half, cam_o[0] + half)
        ax.set_ylim(cam_o[1] - half, cam_o[1] + half)
        ax.set_zlim(cam_o[2] - half, cam_o[2] + half)
    else:
        mins = sub.min(axis=0)
        maxs = sub.max(axis=0)
        span = (maxs - mins).max()
        mid = (maxs + mins) / 2
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    suffix = "  [zoom on wrist cam]" if zoom else ""
    ax.set_title(f"Franka @ qpos + wrist_camera_link frame (R=X, G=Y, B=Z){suffix}")
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_png(pts: np.ndarray, cols: np.ndarray, T_cam: np.ndarray, path: Path) -> None:
    _render(pts, cols, T_cam, path, zoom=False)
    _render(pts, cols, T_cam, OUT_PNG_ZOOM, zoom=True, elev=18, azim=-60)


def main() -> None:
    print(f"Loading URDF: {URDF_PATH}")
    urdf = load_urdf()

    actuated = set(urdf.actuated_joint_names)
    qpos = {k: v for k, v in JOINT_POS.items() if k in actuated}
    missing = actuated - set(qpos.keys())
    if missing:
        print(f"  [warn] joints with no qpos given (will use 0): {sorted(missing)}")
    print(f"Setting cfg for {len(qpos)} joints")
    urdf.update_cfg(qpos)

    print("Sampling point cloud per link:")
    pts, cols = collect_pcd(urdf)
    print(f"Total points: {len(pts)}")

    T_cam = urdf.get_transform("wrist_camera_link")
    print(f"wrist_camera_link world pose:\n{T_cam}")

    save_ply(pts, cols, OUT_PLY)
    print(f"Wrote {OUT_PLY}")
    write_pose(T_cam, OUT_POSE)
    print(f"Wrote {OUT_POSE}")
    save_png(pts, cols, T_cam, OUT_PNG)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
