"""Part-level meta helpers for the unified [name][part] meta framework.

Part keys: "base" plus one key per URDF movable joint in document order, named by
kind — prismatic -> "drawer_<i>", revolute -> "door_<j>" (per-kind counters). The
legacy integer drawer_id used by skills is the movable-part position in
URDF order, i.e. object_part_names[name][drawer_id + 1].

Pure-python / stdlib module (no torch, no isaac imports) so it can be unit-tested
standalone. This is the ONLY URDF part parser in the repo (the legacy
IsaacLabSim._parse_urdf_movable_joints / articulation_drawer_* side tables are gone).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

PART_KIND_BY_JOINT_TYPE = {"prismatic": "drawer", "revolute": "door"}

# Canonical-frame opening direction by part kind (user-confirmed asset convention:
# cabinet/microwave front face is cano +x; drawers/rigid containers open upward).
OPENING_DIR_CANO_BY_KIND = {"drawer": (0.0, 0.0, 1.0), "door": (1.0, 0.0, 0.0)}
OPENING_DIR_CANO_DEFAULT = (0.0, 0.0, 1.0)

# Per-part axis annotations by kind (part geometry lives in the base cano frame, so the
# axes are the object's axes scoped to the part; q0 = closed configuration). A scene JSON
# drawers entry can override per part via optional x_axis/y_axis/z_axis fields.
AXIS_ANNOTATION_BY_KIND = {
    "drawer": {
        "x": "From the back of the drawer box to its front panel with the handle; the drawer slides out along this axis.",
        "y": "From the left side of the drawer box to the right side.",
        "z": "From the drawer's bottom panel up through its open top.",
    },
    "door": {
        "x": "Perpendicular to the closed door panel, pointing outward from its front face.",
        "y": "From the door-hinge edge to the handle edge.",
        "z": "From the bottom edge of the door to its top edge.",
    },
}


def parse_urdf_parts(urdf_path: str) -> tuple[str, list[dict]]:
    """(base_link_name, movable parts). Parts are the movable (non-fixed) joints in
    URDF order, with per-kind part keys:
    {part_key, kind, joint, child, joint_type, axis_obj(3), origin_obj(3)}.

    Assumes every movable joint's parent is the base/root link and joint rpy == 0
    (true for the LIBERO cabinet/microwave assets) so axis/origin are already in the
    object/base cano frame; raises on violation rather than returning wrong frames.
    """
    root = ET.parse(urdf_path).getroot()
    all_links = [ln.get("name") for ln in root.findall("link")]
    children = {jt.find("child").get("link") for jt in root.findall("joint")}
    base_links = [ln for ln in all_links if ln not in children]
    if len(base_links) != 1:
        raise ValueError(f"{urdf_path}: expected exactly 1 root link, got {base_links}")
    base_link = base_links[0]

    parts = []
    kind_counters: dict = {}
    for jt in root.findall("joint"):
        jtype = jt.get("type")
        if jtype == "fixed":
            continue
        kind = PART_KIND_BY_JOINT_TYPE.get(jtype)
        if kind is None:
            raise ValueError(f"{urdf_path}: unsupported movable joint type {jtype!r}")
        parent = jt.find("parent").get("link")
        if parent != base_link:
            raise ValueError(
                f"{urdf_path}: movable joint {jt.get('name')!r} has parent {parent!r} "
                f"(only base-parented movable joints are supported)"
            )
        o = jt.find("origin")
        if o is not None and o.get("rpy") and any(abs(float(v)) > 1e-12 for v in o.get("rpy").split()):
            raise ValueError(
                f"{urdf_path}: movable joint {jt.get('name')!r} has non-zero origin rpy "
                f"(axis/origin would not be in the base cano frame)"
            )
        ax = jt.find("axis")
        axis = [float(v) for v in ((ax.get("xyz") if ax is not None else "1 0 0").split())]
        origin = [float(v) for v in (((o.get("xyz") if (o is not None and o.get("xyz")) else "0 0 0")).split())]
        idx = kind_counters.get(kind, 0)
        kind_counters[kind] = idx + 1
        parts.append({
            "part_key": f"{kind}_{idx}",
            "kind": kind,
            "joint": jt.get("name"),
            "child": jt.find("child").get("link"),
            "joint_type": jtype,
            "axis_obj": axis,
            "origin_obj": origin,
        })
    return base_link, parts


def part_pcd_npz_path(urdf_path: str, child_link: str) -> str:
    """Offline part-pcd artifact convention (scripts/gen_arti_part_pcds.py):
    <urdf_dir>/<urdf_stem>__<child_link>_part_pcd.npz"""
    stem = os.path.splitext(os.path.basename(urdf_path))[0]
    return os.path.join(os.path.dirname(urdf_path), f"{stem}__{child_link}_part_pcd.npz")
