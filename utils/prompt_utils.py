def make_view_list(view_names):
    lines = ["Available view list ==>"]
    for idx, name in enumerate(view_names):
        lines.append(f"\t{idx}: {name}")
    return "\n".join(lines)


def make_visual_observation_placeholder(view_names: list[str]) -> str:
    lines = []
    for name in view_names:
        lines.append(f"\t{name} view: <IMG>")
    return "\n".join(lines)


def make_object_info(
    obj_names: list[str],
    obj_center_3d: dict = None,
    obj_range_3d: dict = None,
    obj_range_2d: dict = None,
    obj_axis_annotation: dict = None,
    obj_upright_status: dict = None,
    obj_parts: dict = None,
    obj_ids: dict = None,
    obj_opening_dirs: dict = None,
) -> str:
    """obj_ids: {name: displayed index} from the arti entity map — the interleaved numbering
    (each object followed by its parts) shifts objects after the first arti asset. None =
    plain enumerate (prompts without parts).
    obj_opening_dirs: {arti object name: world-frame unit opening dir of its base} (prompt
    use_open_dir); rendered only for names present in the dict."""
    lines = ["List of object names in the scene and their spatial properties ==>"]
    for idx, name in enumerate(obj_names):
        if obj_ids is not None:
            idx = obj_ids[name]
        lines.append(f"\t{idx}: {name}")
        if obj_center_3d is not None:
            lines.append(f"\t\t- 3D center location(in meters): {', '.join(map(str, obj_center_3d[name]))}")
        if obj_range_3d is not None:
            lines.append(f"\t\t- 3D axis-aligned bounding range(in meters): x_min={obj_range_3d[name][0]}, x_max={obj_range_3d[name][1]}, y_min={obj_range_3d[name][2]}, y_max={obj_range_3d[name][3]}, z_min={obj_range_3d[name][4]}, z_max={obj_range_3d[name][5]}")
        if obj_range_2d is not None:
            lines.append(f"\t\t- 2D axis-aligned bounding range(in pixels): x_min={obj_range_2d[name][0]}, x_max={obj_range_2d[name][1]}, y_min={obj_range_2d[name][2]}, y_max={obj_range_2d[name][3]}")
        if obj_axis_annotation is not None:
            axes_info = obj_axis_annotation[name]
            axis_lines = []
            for axis in ["x", "y", "z"]:
                axis_lines.append(f"{axis.lower()}: {axes_info[axis]}")
            lines.append(f"\t\t- Primary axes in canonical object frame:\n\t\t\t" + "\n\t\t\t".join(axis_lines))
        if obj_upright_status is not None and name in obj_upright_status:
            lines.append(f"\t\t- **pose: {obj_upright_status[name]}**")
        if obj_opening_dirs is not None and name in obj_opening_dirs:
            v = obj_opening_dirs[name]
            lines.append(
                f"\t\t- Opening direction (world unit vector): ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"
                " — the object's open face / cavity opening points this way"
            )
        # Articulated parts hung under the object; every part (drawer AND door) reports
        # center + world AABB + state. The part id is its index in the same flat Asset
        # index space as the objects above.
        if obj_parts is not None and obj_parts.get(name):
            lines.append("\t\t- Articulated parts:")
            for p in obj_parts[name]:
                lines.append(f"\t\t\t{p['id']}: type={p['kind'].capitalize()}, {_part_geom_str(p)}")
                lines.extend(_part_axes_lines(p, "\t\t\t\t"))
    return "\n".join(lines)


def _part_geom_str(p: dict) -> str:
    c = ", ".join(f"{v:.3f}" for v in p["center"])
    mn = ", ".join(f"{v:.3f}" for v in p["aabb_min"])
    mx = ", ".join(f"{v:.3f}" for v in p["aabb_max"])
    s = f"center=({c}), AABB min=({mn}) max=({mx}), state={p['state']}"
    opening = p.get("opening")
    if opening:
        s += f", opening dir (world)=({opening[0]:.2f}, {opening[1]:.2f}, {opening[2]:.2f})"
    return s


def _part_axes_lines(p: dict, indent: str) -> list:
    """Axis-annotation sub-block of a part line; present only when the builder attached
    axes (prompt use_axis_annotation). The text describes the part's OWN frame."""
    axes = p.get("axes")
    if not axes:
        return []
    lines = [f"{indent}Primary axes in the part's own frame:"]
    for ax in ("x", "y", "z"):
        lines.append(f"{indent}\t{ax}: {axes.get(ax, '')}")
    return lines


def make_arti_parts_info(obj_parts: dict) -> str:
    """Standalone articulated-part list (prompt use_arti_only): the owner object appears
    once in the header; each part line starts with its flat integer Asset index, same
    line format as the object list."""
    lines = []
    for name, parts in (obj_parts or {}).items():
        lines.append(f"List of articulated parts of {name} ==>")
        for p in parts:
            lines.append(f"\t{p['id']}: type={p['kind'].capitalize()}, {_part_geom_str(p)}")
            lines.extend(_part_axes_lines(p, "\t\t"))
    return "\n".join(lines)


def make_contact_info(grasped_objects: list[str]) -> str:
    if len(grasped_objects) == 0:
        return "The robot gripper is currently empty."
    else:
        return f"The robot is currently grasping the following objects: {', '.join(grasped_objects)}."


def make_skill_primitives_list(skill_primitives: list[str], skill_desc: dict[str, str]) -> str:
    lines = ["Available skill primitive list ==>"]
    for idx, skill in enumerate(skill_primitives):
        # Indent the description's continuation lines so the whole block nests under the
        # primitive and nothing in it sits at the same column as the skill indices.
        desc = skill_desc[skill].replace("\n", "\n\t\t  ")
        lines.append(f"\t{idx}: {skill}\n\t\t- {desc}")
    return "\n".join(lines)


def make_coordinate_system_info(text: str | None) -> str:
    if not text:
        return ""
    return f"Coordinate System Information:\n{text}\n\n"


def make_history_reflection(history_traj=None, expert_reflection=None) -> str:
    if history_traj is None:
        history_text = "We're at the initial state of the scene."
        reflection_text = ""
    else:
        history_text = f"Previously, the robot has made the following progress:\n"
        for idx, node_status in enumerate(history_traj):
            history_text += f"Step {idx + 1}: {node_status}\n"
        if expert_reflection is None:
            reflection_text = ""
        else:
            history_text += "An expert verifier has reviewed the previous progress and provided the following feedback:\n"
            history_text += f"{expert_reflection}"
            reflection_text = "## Reflection\nReflect on the previous history and the expert feedback."

    return history_text, reflection_text