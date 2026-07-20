import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from matplotlib.collections import LineCollection
import cv2
import io
from PIL import Image

import sys
import os
# Add utils directory to path for imports
utils_dir = os.path.dirname(os.path.abspath(__file__))
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
from image_utils import _decode_base64, smart_new_hw, _encode_image
from geometry_utils import proj_3d_to_2d, qt2T, rigid_transform


PATTERN_POINT = re.compile(r"\(\d+,\s*\d+\)")
PATTERN_BOX = re.compile(r"\(\d+,\s*\d+,\s*\d+,\s*\d+\)")

gripper_ctrl_points = np.array([
    [0.05268743, 0., 0.105273141],
    [0.05268743, 0., 0.05268743],
    [0., 0., 0.05268743],
    [0., 0., 0.],
    [0., 0., 0.05268743],
    [-0.05268743, 0., 0.05268743],
    [-0.05268743, 0., 0.105273141]
])
gripper_lines = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6]]

# T = np.array([
#     [0, -1, 0, 0],
#     [1, 0, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ])
# scene_pose_matrix = np.array([
#     [1, 0, 0, 0.308951-0.040294],
#     [0, 1, 0, -0.000000+ 0.007072],
#     [0, 0, 1, -0.820018+0.072134],
#     [0, 0, 0, 1]
# ])

# H = 512
# W = 512
# ixt = np.array([
#     [-703.3542416031569, 0.0, 256.0],
#     [0.0, -703.3542416031569, 256.0],
#     [0.0, 0.0, 1.0]
# ])
# ext = np.array([
#     [1.1920928955078125e-07, -0.42261794209480286, -0.9063079357147217, 1.349999189376831],
#     [-1.0, -5.960464477539062e-07, 1.4901161193847656e-07, 3.715465624054559e-08],
#     [-5.662441253662109e-07, 0.9063079357147217, -0.42261791229248047, 1.579999327659607],
#     [0.0, 0.0, 0.0, 1.0]
# ])


def draw_points(fig, ax, points: np.ndarray, colors, labels: list[str]):
    assert len(points) == len(colors) or len(colors) == 1
    assert len(points) == len(labels) or len(labels) == 1
    if len(colors) == 1:
        colors = colors * len(points)

    for i, p in enumerate(points):
        if len(p) < 2:
            continue
        # label = labels[i] + f":({p[0]},{p[1]})"
        label = labels[i]
        x, y = p
        ax.plot(x, y, colors[i], markersize=5)  # plot one circle per point
        ax.text(x + 3, y + 3, label, color="white")

    return fig, ax


# def draw_boxes(img: np.ndarray, boxes: np.ndarray, colors, out_path):
def draw_boxes(fig, ax, boxes: np.ndarray, colors, labels: list[str]):
    if None in boxes:
        return 
    
    assert len(boxes) == len(colors) or len(colors) == 1
    if len(colors) == 1:
        colors = colors * len(boxes)

    boxes_xyhw = []
    for box in boxes:
        box_ = (box[0], box[1], box[2] - box[0], box[3] - box[1])
        boxes_xyhw.append(box_)
    
    for i, box in enumerate(boxes_xyhw):
        x, y, w, h = box
        # label = labels[i] + f":({box[0]},{box[1]},{box[2]},{box[2]})"
        label = labels[i]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=colors[i], facecolor='none')
        ax.add_patch(rect)
        ax.text(x - 3, y - 3, label, color="white")
    
    return fig, ax


def get_point_coords(text, h, w):
    points = PATTERN_POINT.findall(text)
    h_bar, w_bar = smart_new_hw(h, w)
    h_scale, w_scale = h / h_bar, w / w_bar

    points_orig = []
    points_renormed = []
    for point in points:
        point = eval(point)
        points_orig.append(point)
        points_renormed.append([point[0] * w_scale, point[1] * h_scale])
    return points_orig, points_renormed


def get_box_coords(text, h, w):
    boxes = PATTERN_BOX.findall(text)
    h_bar, w_bar = smart_new_hw(h, w)
    h_scale, w_scale = h / h_bar, w / w_bar

    boxes_orig = []
    boxes_renormed = []
    for box in boxes:
        box = eval(box)
        boxes_orig.append(box)
        boxes_renormed.append([box[0] * w_scale, box[1] * h_scale, box[2] * w_scale, box[3] * h_scale])
    return boxes_orig, boxes_renormed


def draw_gripper(fig, ax, gripper_pts_2d: np.ndarray, color="green"):
    line_segs = []
    for gripper_line in gripper_lines:
        line_segs.append(gripper_pts_2d[gripper_line])
    line_segs = np.stack(line_segs, 0)

    # draw gripper lines
    lc = LineCollection(line_segs, linewidths=2, colors=color)
    ax.add_collection(lc)

    endpoints = line_segs.reshape(-1,2)  # (2N,2)
    ax.scatter(endpoints[:,0], endpoints[:,1], s=30, c=color)

    return fig, ax


def project_gripper_to_2d(goal_pose, T, scene_pose_matrix, ixt, ext, H, W):
    if isinstance(goal_pose, list):
        goal_pose = qt2T(goal_pose[3:], goal_pose[:3])
    assert isinstance(goal_pose, np.ndarray)

    goal_pose = np.linalg.inv(scene_pose_matrix) @ (goal_pose @ np.linalg.inv(T))
    # goal_pose = goal_pose @ np.linalg.inv(T)

    ## goal_pose: (4, 4)
    ## gripper_ctrl_points (7, 3)
    gripper_ctrl_points_world = rigid_transform(
        #rigid_transform(gripper_ctrl_points, T), 
        gripper_ctrl_points,
        goal_pose
    )
    gripper_ctrl_points_pixel = proj_3d_to_2d(gripper_ctrl_points_world,
                                              ixt,
                                              ext,
                                              max_x=W,
                                              max_y=H)
    
    return gripper_ctrl_points_pixel


def draw_gripper_from_goal_pose(fig, ax, goal_pose, T, scene_pose_matrix, ixt, ext, H, W, color="green"):
    gripper_pts_2d = project_gripper_to_2d(goal_pose, T, scene_pose_matrix, ixt, ext, H, W)
    fig, ax = draw_gripper(fig, ax, gripper_pts_2d, color=color)
    return fig, ax


def draw_masks(image, mask, borders=True):
    image = image[..., ::-1] / 255.
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=1)

    mask_image = np.where(mask[..., None], mask_image[..., :3] * 0.6 + image * 0.4, image)
    mask_image = (mask_image * 255).astype(np.uint8)
    mask_image = np.where(mask[..., None], mask_image, mask_image[..., ::-1])

    return mask_image


def vis_all(
    image, 
    vis_elems: dict,
    T,
    scene_pose_matrix, 
    ixt, 
    ext, 
    H, 
    W,
    vis_point=True, 
    vis_box=True, 
    vis_mask=False, 
    vis_gripper=False,
    vis_gripper_0=False,
):
    is_input_image_str = isinstance(image, str)
    is_input_image_pil = isinstance(image, Image.Image)

    if is_input_image_str:
        image = np.array(_decode_base64(image, to_image=True))
    elif is_input_image_pil:
        image = np.array(image)

    # exit(0)

    if vis_mask:
        mask = vis_elems["mask"]
        if isinstance(mask, str):
            mask = _decode_base64(mask).astype(np.bool_)
        image = draw_masks(image, mask)

    fig, ax = plt.subplots()
    try:
        ax.imshow(image)

        if vis_point:
            if not isinstance(vis_elems["point"][0], list):
                vis_elems["point"] = [vis_elems["point"]]
            points = []
            point_labels = []
            for i, point in enumerate(vis_elems["point"]):
                # label_point = f"point:({int(point[0])},{int(point[1])})"
                label_point = f"{i}"
                points.append(point)
                point_labels.append(label_point)
            points = np.stack(points, 0)
            fig, ax = draw_points(
                fig, ax, points, ["ro"], labels=point_labels
            )
            # point = vis_elems["point"]
            # label_point = f"point:({int(point[0])},{int(point[1])})"
            # fig, ax = draw_points(
            #     fig, ax, np.array(point)[None], ["ro"], labels=[label_point]
            # )

        if vis_box:
            # box = vis_elems["box"]
            # label_box = f"box:({int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])})"
            if not isinstance(vis_elems["box"][0], list):
                vis_elems["box"] = [vis_elems["box"]]
            boxes = []
            box_labels = []
            for box in vis_elems["box"]:
                label_box = f"box:({int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])})"
                boxes.append(box)
                box_labels.append(label_box)
            boxes = np.stack(boxes, 0)
            # fig, ax = draw_boxes(
            #     fig, ax, np.array(box)[None], ["b"], labels=[label_box]
            # )
            fig, ax = draw_boxes(
                fig, ax, boxes, ["b"], labels=box_labels
            )

        if vis_gripper:
            goal_pose = vis_elems["goal_pose"]
            fig, ax = draw_gripper_from_goal_pose(fig, ax, goal_pose, T, scene_pose_matrix, ixt, ext, H, W)

        if vis_gripper_0:
            goal_pose_0 = vis_elems["goal_pose_0"]
            fig, ax = draw_gripper_from_goal_pose(fig, ax, goal_pose_0, T, scene_pose_matrix, ixt, ext, H, W, color="magenta")

        fig.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches="tight")
        buf.seek(0)
        result_image = Image.open(buf)
        result_image.load()

    finally:
        plt.close(fig)
        # if buf is not None:
        #     buf.close()

    if is_input_image_str:
        result_image = _encode_image(result_image)
        
    return result_image
