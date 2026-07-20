"""Helpers for varying the wrist-camera mount offset.

Frames (the URDF rpy="0 0 pi/2" wrist_camera_mount joint determines this):

    panda_hand:        +Z = toward fingers,  +Y = open/close axis,  +X = palm front (palm normal, outward)
    wrist_camera_link: +X = panda_hand +Y    (open/close)
                       +Y = panda_hand -X    (palm back direction)  <-- "move to back of palm" axis
                       +Z = panda_hand +Z    (toward fingers)

OffsetCfg with ``convention="ros"`` interprets ``pos`` in the parent-prim
(wrist_camera_link) frame, and ``rot`` as ROS-camera optical-axes rotation
(+Z forward = lens, +X right, +Y down).

So ``mount_offset(back_m=X)`` returns ``pos=(0, X, 0)`` -- i.e. moves the
camera by X meters in wrist_camera_link's +Y, which is panda_hand's -X, i.e.
toward the back of the palm.

The URDF default mount itself sits at panda_hand (0.05, 0, 0) -- 5cm out the
palm-front side. ``back_m=0.10`` mirrors that to panda_hand (-0.05, 0, 0),
i.e. 5cm out the palm-back side.

Module is import-safe (no Isaac Lab deps).
"""
from __future__ import annotations

import math


def mount_offset(
    back_m: float = 0.0,
    extend_m: float = 0.0,
    side_m: float = 0.0,
    pitch_deg: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return ``(pos, rot_wxyz)`` in the wrist_camera_link / ROS-camera frame.

    - ``back_m``:   along wrist_camera_link +Y  (= panda_hand -X, palm-back direction)
    - ``extend_m``: along wrist_camera_link +Z  (= panda_hand +Z, toward fingers)
    - ``side_m``:   along wrist_camera_link +X  (= panda_hand +Y, open/close axis)
    - ``pitch_deg``: rotation about lens +X axis (= open/close axis); >0 tilts
                    the lens toward palm-back (useful when back-mounted to
                    look forward over the gripper, use a *negative* value).
    """
    pos = (side_m, back_m, extend_m)
    half = math.radians(pitch_deg) / 2.0
    rot = (math.cos(half), math.sin(half), 0.0, 0.0)
    return pos, rot


# name -> (back_m, extend_m, side_m, pitch_deg)
# back_m=0    -> palm-front 5cm (URDF default)
# back_m=0.05 -> at wrist plate (between front and back)
# back_m=0.10 -> palm-back 5cm (mirror of default)
# back_m=0.15 -> palm-back 10cm
PRESETS: dict[str, tuple[float, float, float, float]] = {
    "default":        (0.00, 0.00, 0.00,   0.0),
    "back_5cm":       (0.05, 0.00, 0.00,   0.0),
    "back_10cm":      (0.10, 0.00, 0.00,   0.0),
    "back_15cm":      (0.15, 0.00, 0.00,   0.0),
    "realsense_like": (0.10, 0.04, 0.00, -20.0),  # back-mounted, lens tilted forward
}


def preset_offset(name: str):
    if name not in PRESETS:
        raise KeyError(f"Unknown wrist-camera preset '{name}'. Available: {sorted(PRESETS)}")
    back_m, extend_m, side_m, pitch_deg = PRESETS[name]
    return mount_offset(
        back_m=back_m,
        extend_m=extend_m,
        side_m=side_m,
        pitch_deg=pitch_deg,
    )


# ---------------------------------------------------------------------------
# Intrinsic alignment: real (after center-crop + resize) -> sim aperture
# ---------------------------------------------------------------------------

def final_fx_after_center_crop_and_resize(
    real_fx: float,
    real_native_short_edge: int,
    final_size: int,
) -> float:
    """fx in pixels after center-crop to a short-edge square then equal resize."""
    return real_fx * (final_size / real_native_short_edge)


def aperture_for_fx(target_fx: float, focal_length: float, width: int) -> float:
    """Solve horizontal_aperture (mm) so that Isaac Lab PinholeCameraCfg yields target_fx."""
    return focal_length * width / target_fx

