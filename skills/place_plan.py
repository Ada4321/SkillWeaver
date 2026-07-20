"""PlacePlan — common schema + per-skill presets for the unified place engine.

See docs/place_pipeline_rfc.md. The VLM-facing skills stay separate (each keeps its
narrow phase-2 prompt). After the phase-2 parse, `build_place_plan()` aggregates a
skill's fixed preset choices with the VLM's narrow `action_params` into one `PlacePlan`
that the single `execute_place()` engine consumes. Adding a place variant = add a prompt
+ one row in `PLACE_PRESETS`, not a new pipeline.

This module is intent-only: it does NOT resolve geometry (resting pose, regime, clearance
magnitude). Those are derived inside `execute_place()` from the simulator state — the
plan just carries the *choices*, the engine does the *how*.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


# --- orientation: which frame's axis is the alignment "src" (RFC §3 stage 2) ---
ORIENT_FREE = "free"            # no orientation constraint
ORIENT_OBJECT = "object_align"  # align a held-object axis to a target-object axis
ORIENT_GRIPPER = "gripper_align"  # align a gripper axis to a target-object axis

# --- release / finish actuation (RFC §5; drop/rl are vertical-only shortcuts) ---
RELEASE_DROP = "drop"    # release from hover, gravity falls
RELEASE_RL = "rl"        # RL policy performs the descent
RELEASE_LOWER = "lower"  # motion-plan lower then open

# --- approach axis source (RFC §3 stage 3) ---
APPROACH_AUTO_Z = "auto_z"                  # straight down (default, vertical)
APPROACH_FROM_ALIGNMENT = "from_alignment"  # derive n_app from the gripper-axis alignment


@dataclass(frozen=True)
class PlacePreset:
    orientation: str          # ORIENT_*
    release: str              # RELEASE_*
    approach: str             # APPROACH_*
    clearance_key: str        # simulator.<knob> giving the vertical-regime clearance (RFC §6, quantity B)
    requires: Tuple[str, ...]  # action_params keys that must be present


# One row per VLM-visible place skill. The clearance_key is the *vertical* knob; the
# engine overrides it with `place_offset_nonvertical` when the regime is non-vertical.
PLACE_PRESETS = {
    "place_with_drop": PlacePreset(
        orientation=ORIENT_FREE, release=RELEASE_DROP, approach=APPROACH_AUTO_Z,
        clearance_key="place_offset", requires=("point_3d",)),
    "place": PlacePreset(
        orientation=ORIENT_FREE, release=RELEASE_RL, approach=APPROACH_AUTO_Z,
        clearance_key="place_offset_rlpolicy", requires=("point_3d",)),
    "place_with_orientation": PlacePreset(
        orientation=ORIENT_OBJECT, release=RELEASE_RL, approach=APPROACH_AUTO_Z,
        clearance_key="place_offset_rlpolicy", requires=("point_3d", "asset_id", "asset", "alignment")),
    "place_with_orientation_drop": PlacePreset(
        orientation=ORIENT_OBJECT, release=RELEASE_DROP, approach=APPROACH_AUTO_Z,
        clearance_key="place_offset_orientation_drop", requires=("point_3d", "asset_id", "asset", "alignment")),
    "place_with_gripper_orientation_drop": PlacePreset(
        orientation=ORIENT_GRIPPER, release=RELEASE_DROP, approach=APPROACH_FROM_ALIGNMENT,
        clearance_key="place_offset_gripper_orientation", requires=("point_3d", "asset_id", "asset", "alignment")),
}


# Derived family sets — the ONLY source of "which skills are place skills". Import these
# instead of hand-writing skill-name literals (a missed copy used to silently break the
# place plan construction).
PLACE_SKILLS = frozenset(PLACE_PRESETS)
ORIENTATION_PLACE_SKILLS = frozenset(
    k for k, p in PLACE_PRESETS.items() if p.orientation != ORIENT_FREE
)


@dataclass
class PlacePlan:
    skill: str
    # fixed choices from the preset
    orientation_mode: str   # ORIENT_*
    release_mode: str       # RELEASE_*
    approach_source: str    # APPROACH_*
    clearance_key: str
    # intent fields from the VLM (action_params)
    asset: Optional[str] = None       # object name, or part key when asset_object is set
    asset_id: Optional[int] = None
    point_3d: Optional[list] = None
    alignment: Optional[Tuple[str, str, str]] = None
    # owning object name when the VLM selected an articulated part; None = rigid-object target
    asset_object: Optional[str] = None

    @property
    def target_object(self) -> Optional[str]:
        """Object-name view of the target: the owner for a part place, else the asset itself.
        Use for simulator APIs keyed by object name (axes, opening dir, z-max)."""
        return self.asset_object if self.asset_object is not None else self.asset

    @property
    def place_into_part(self) -> Optional[str]:
        """The part key when this place targets an articulated part; None otherwise."""
        return self.asset if self.asset_object is not None else None


def build_place_plan(skill, action_params):
    """Aggregate a skill's preset with the VLM's narrow `action_params`.

    Returns ``(plan, error)``. On success ``plan`` is a PlacePlan and ``error`` is None.
    On a missing required field ``plan`` is None and ``error`` is a message the caller
    turns into ``return_failure(...)`` (this centralizes the per-skill
    "Action_params incomplete!" guards that used to live in each execute_* function).
    """
    preset = PLACE_PRESETS.get(skill)
    if preset is None:
        return None, f"build_place_plan: unknown place skill {skill!r}"
    for key in preset.requires:
        if action_params.get(key) is None:
            # Exact string parity with the per-skill guards being centralized here.
            return None, "Action_params incomplete!"
    plan = PlacePlan(
        skill=skill,
        orientation_mode=preset.orientation,
        release_mode=preset.release,
        approach_source=preset.approach,
        clearance_key=preset.clearance_key,
        asset=action_params.get("asset"),
        asset_id=action_params.get("asset_id"),
        point_3d=action_params.get("point_3d"),
        alignment=action_params.get("alignment"),
        asset_object=action_params.get("asset_object"),
    )
    return plan, None
