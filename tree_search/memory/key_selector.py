"""Key-value memory retrieval module driven by an LLM key selector.

Memory is loaded from one or more JSON files. Each file is a flat dictionary
mapping a manipulation-pattern key to a payload of the form
``{"value": str, "tags": list[str]}`` where allowed tags are
``actor_phase_1``, ``actor_phase_2``, and ``judge``.

Key selection is performed by an LLM over the candidate keys for the round:
the LLM is asked to return up to ``k`` most relevant keys, ranked by relevance.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

ALLOWED_TAGS = {"actor_phase_1", "actor_phase_2", "judge"}


SELECTION_PROMPT_TEMPLATE = """You are a memory key selector for a robot manipulation policy.

Given the current task instruction and subgoal, select the most relevant memory keys from the provided key list.

Instruction:
{instruction}

Subgoal:
{subgoal}

Object poses (only whitelisted objects show a pose):
{pose_info}

Available memory keys:
{keys}

Rules:
- Select the {k} most relevant keys from the available memory keys to the current instruction/subgoal.
- Do not invent a new key.
- Output one key per line, no extra text, no numbering, no quotes.
"""


_PREFIX_STRIP_RE = re.compile(r"^[\-\*\d\.\)\s`'\"]+|[`'\"\s]+$")


def _render_pose_info(pose_info: Optional[dict]) -> str:
    if not pose_info:
        return "(none)"
    return "\n".join(f"- {name}: {status}" for name, status in pose_info.items())


def _load_memory(paths: list[str]) -> dict[str, dict]:
    """Load and merge memory entries from a list of JSON file paths.

    - Files are loaded in list order; later files override earlier on key
      conflict via plain ``dict.update`` (no dedup warning).
    - Missing or unreadable files are logged and skipped.
    - Entries with invalid structure or unknown tags are skipped per-entry.
    """
    merged: dict[str, dict] = {}
    for raw_path in paths or []:
        if not raw_path:
            continue
        path = os.path.abspath(os.path.expanduser(str(raw_path)))
        if not os.path.exists(path):
            logger.warning("Memory file not found, skipping: %s", path)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load memory file %s: %s", path, e)
            continue
        if not isinstance(data, dict):
            logger.warning("Memory file %s does not contain a top-level dict; skipping", path)
            continue

        validated: dict[str, dict] = {}
        for key, entry in data.items():
            if not isinstance(entry, dict):
                logger.warning("Memory key %r in %s has non-dict payload; skipping", key, path)
                continue
            value = entry.get("value")
            tags = entry.get("tags", [])
            if not isinstance(value, str):
                logger.warning("Memory key %r in %s has non-string value; skipping", key, path)
                continue
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                logger.warning("Memory key %r in %s has invalid tags; skipping", key, path)
                continue
            unknown = [t for t in tags if t not in ALLOWED_TAGS]
            if unknown:
                logger.warning(
                    "Memory key %r in %s has unknown tags %r; skipping", key, path, unknown
                )
                continue
            validated[key] = {"value": value, "tags": list(tags)}

        merged.update(validated)
    return merged


def _normalize_line(line: str) -> str:
    return _PREFIX_STRIP_RE.sub("", line).strip()


def select_memory_keys(
    instruction: str,
    subgoal: Optional[str],
    candidate_keys: list[str],
    llm_client: Any,
    k: int,
    default_key: Optional[str] = None,
    max_output_tokens: int = 256,
    pose_info: Optional[dict] = None,
) -> list[str]:
    """Ask the LLM to pick up to ``k`` keys from ``candidate_keys``.

    Returns an ordered list (most → least relevant) of validated keys. On total
    LLM failure, returns ``[default_key]`` if ``default_key`` is a candidate,
    otherwise ``[]``. Never raises.
    """
    if not candidate_keys or k <= 0:
        return []

    keys_text = "\n".join(f"- {key}" for key in candidate_keys)
    prompt_text = SELECTION_PROMPT_TEMPLATE.format(
        instruction=instruction or "",
        subgoal=subgoal if subgoal is not None else "",
        pose_info=_render_pose_info(pose_info),
        keys=keys_text,
        k=k,
    )

    try:
        raw = llm_client._generate_raw(
            prompt={"text": prompt_text, "images": []},
            max_output_tokens=max_output_tokens,
        )
    except TypeError:
        raw = llm_client._generate_raw(
            prompt={"text": prompt_text, "images": []},
        )
    except Exception as e:
        logger.warning("Memory key selection LLM call failed: %s", e)
        raw = None

    if not isinstance(raw, str):
        return _fallback(default_key, candidate_keys)

    normalized_map = {key.strip().lower(): key for key in candidate_keys}
    candidate_set = set(candidate_keys)
    picked: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        stripped = _normalize_line(line)
        if not stripped:
            continue
        match: Optional[str] = None
        if stripped in candidate_set:
            match = stripped
        elif stripped.lower() in normalized_map:
            match = normalized_map[stripped.lower()]
        if match is None or match in seen:
            continue
        picked.append(match)
        seen.add(match)
        if len(picked) >= k:
            break

    if picked:
        return picked
    return _fallback(default_key, candidate_keys)


def _fallback(default_key: Optional[str], candidate_keys: Iterable[str]) -> list[str]:
    if default_key is not None and default_key in candidate_keys:
        return [default_key]
    return []


class MemoryRetriever:
    """Loads memory from a list of JSON files and selects keys per phase."""

    def __init__(
        self,
        memory_paths: list[str],
        llm_client: Any,
        default_key: Optional[str] = None,
        max_output_tokens: int = 256,
        top_k: int = 2,
    ):
        self.memory: dict[str, dict] = _load_memory(memory_paths)
        self.llm_client = llm_client
        self.default_key = default_key
        self.max_output_tokens = int(max_output_tokens)
        self.top_k = max(1, int(top_k))

        self._tag_index: dict[str, list[str]] = {tag: [] for tag in ALLOWED_TAGS}
        for key, entry in self.memory.items():
            for tag in entry.get("tags", []):
                if tag in self._tag_index:
                    self._tag_index[tag].append(key)

    def __bool__(self) -> bool:
        return bool(self.memory)

    def keys_for_tags(self, tags: Iterable[str]) -> list[str]:
        seen = set()
        ordered: list[str] = []
        for tag in tags:
            for key in self._tag_index.get(tag, []):
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        return ordered

    def get_value(self, key: Optional[str]) -> Optional[str]:
        if key is None:
            return None
        entry = self.memory.get(key)
        if entry is None:
            return None
        return entry.get("value")

    def get_tags(self, key: Optional[str]) -> list[str]:
        if key is None:
            return []
        entry = self.memory.get(key)
        if entry is None:
            return []
        return list(entry.get("tags", []))

    def select_for_phase(
        self,
        phase: str,
        instruction: str,
        subgoal: Optional[str],
        hint_input: Optional[str],
        candidate_tags: Iterable[str],
        *,
        k: Optional[int] = None,
        pose_info: Optional[dict] = None,
    ) -> dict:
        """Run one selection round and return a log dict.

        Output shape:
            {
                "phase", "instruction", "subgoal", "hint_input",
                "k": int,
                "available_keys": list[str],
                "selected_keys": list[str],          # ordered, validated
                "selected_entries": [                # one per selected_key
                    {"key": str, "tags": list[str], "value": str}
                ],
                "applied_to": list[str],             # filled by caller
            }
        """
        effective_k = self.top_k if k is None else max(1, int(k))
        candidate_keys = self.keys_for_tags(candidate_tags)
        log: dict = {
            "phase": phase,
            "instruction": instruction,
            "subgoal": subgoal,
            "hint_input": hint_input,
            "k": effective_k,
            "available_keys": list(candidate_keys),
            "selected_keys": [],
            "selected_entries": [],
            "applied_to": [],
        }
        if not candidate_keys:
            return log

        if hint_input:
            instr_for_llm = f"{instruction}\n\nPrevious reflection:\n{hint_input}"
        else:
            instr_for_llm = instruction

        selected = select_memory_keys(
            instruction=instr_for_llm,
            subgoal=subgoal,
            candidate_keys=candidate_keys,
            llm_client=self.llm_client,
            k=effective_k,
            default_key=self.default_key,
            max_output_tokens=self.max_output_tokens,
            pose_info=pose_info,
        )
        log["selected_keys"] = list(selected)
        log["selected_entries"] = [
            {"key": key, "tags": self.get_tags(key), "value": self.get_value(key)}
            for key in selected
        ]
        return log
