import logging
import os
import re
from typing import Any

import numpy as np
import requests

from skills.skills import (
    CONVEX_UPRIGHT_COS_THRESHOLD,
    _object_category_candidates,
)
from utils.geometry_utils import q2R_wxyz


logger = logging.getLogger(__name__)


# SceneGen scenes use "<category>_<4-digit asset id>" object names; libero
# scenes use bare category names (no suffix) or "<name>_<1-2 digits>". Strip
# only the 4+ digit suffix so libero names like "black_bowl_1" stay intact.
_ASSET_SUFFIX_RE = re.compile(r"_\d{4,}$")


def _category_of(object_name: str) -> str:
    return _ASSET_SUFFIX_RE.sub("", object_name)


def load_object_list_from_txt(path: str | None, field_name: str) -> set[str]:
    if not path:
        return set()
    if not isinstance(path, str):
        raise TypeError(f"{field_name} must be a string path, got: {type(path)}")
    expanded = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isabs(expanded):
        raise ValueError(f"{field_name} must be an absolute path, got: {path}")
    resolved = os.path.abspath(expanded)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{field_name} file does not exist: {resolved}")

    names = set()
    with open(resolved, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item and not item.startswith("#"):
                names.add(item)
    return names


class RLPolicyClient:
    def __init__(self, args) -> None:
        if not isinstance(args, dict):
            raise TypeError("RLPolicyClient expects args as dict.")
        self.url = str(args["url"]).rstrip("/")
        self.timeout_s = float(args.get("timeout_s", 60.0))
        self.rl_cfg = args.get("rl_cfg", {})
        self.pick_upright_ckpt = self.rl_cfg.get("pick_upright_ckpt")
        self.pick_convex_ckpt = self.rl_cfg.get("pick_convex_ckpt")
        self.pick_convex_lying_ckpt = self.rl_cfg.get("pick_convex_lying_ckpt")
        self.pick_handle_ckpt = self.rl_cfg.get("pick_handle_ckpt")
        self.pick_upright_height_threshold_m = 0.10
        self.upright_objects = load_object_list_from_txt(self.rl_cfg.get("upright_list"), "upright_list")
        self.convex_objects = load_object_list_from_txt(self.rl_cfg.get("convex_list"), "convex_list")
        self.handle_objects = load_object_list_from_txt(self.rl_cfg.get("handle_list"), "handle_list")
        self.place_align_cup_bottom = bool(self.rl_cfg.get("place_align_cup_bottom", False))
        self.top_pcd_agents = set(self.rl_cfg.get("top_pcd_agents", ["pick_convex_lying"]))
        self.handle_pcd_agents = set(self.rl_cfg.get("handle_pcd_agents", ["pick_handle"]))
        self.session = requests.Session()

    def uses_top_pcd(self, agent_name: str) -> bool:
        return agent_name in self.top_pcd_agents

    def uses_handle_pcd(self, agent_name: str) -> bool:
        return agent_name in self.handle_pcd_agents

    def _to_jsonable(self, x: Any):
        if isinstance(x, dict):
            return {k: self._to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [self._to_jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            try:
                return x.numpy().tolist()
            except Exception:
                pass
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        # Final fallback for scalar-like values.
        return np.asarray(x).tolist()

    def _post(self, endpoint: str, payload: dict):
        url = f"{self.url}{endpoint}"
        payload_json = self._to_jsonable(payload)
        try:
            response = self.session.post(url, json=payload_json, timeout=self.timeout_s)
            try:
                data = response.json()
            except Exception:
                data = {"raw_text": response.text}
            if response.status_code >= 400:
                server_err = data.get("error") if isinstance(data, dict) else None
                server_tb = data.get("traceback") if isinstance(data, dict) else None
                detail = server_err if server_err else str(data)
                if server_tb:
                    detail = f"{detail}\nServer traceback:\n{server_tb}"
                raise RuntimeError(f"HTTP {response.status_code} from server: {detail}")
        except Exception as exc:
            raise RuntimeError(f"RLPolicyClient request failed: {url}, error: {exc}") from exc
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"RLPolicyClient server returned error at {url}: {data['error']}")
        return data

    def _get(self, endpoint: str):
        url = f"{self.url}{endpoint}"
        try:
            response = self.session.get(url, timeout=self.timeout_s)
            try:
                data = response.json()
            except Exception:
                data = {"raw_text": response.text}
            if response.status_code >= 400:
                server_err = data.get("error") if isinstance(data, dict) else None
                server_tb = data.get("traceback") if isinstance(data, dict) else None
                detail = server_err if server_err else str(data)
                if server_tb:
                    detail = f"{detail}\nServer traceback:\n{server_tb}"
                raise RuntimeError(f"HTTP {response.status_code} from server: {detail}")
        except Exception as exc:
            raise RuntimeError(f"RLPolicyClient request failed: {url}, error: {exc}") from exc
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"RLPolicyClient server returned error at {url}: {data['error']}")
        return data

    def health_check(self):
        return self._get("/health")

    def init_agents(self, simulator=None, task_env=None):
        if simulator is None or task_env is None:
            raise ValueError("RLPolicyClient.init_agents requires simulator and task_env.")

        health = self.health_check()
        loaded_agents = set(health.get("agents", [])) if isinstance(health, dict) else set()

        base_env = simulator._unwrap_env(task_env)
        required_agents = []

        if self.rl_cfg.get("use_grasp"):
            required_agents.append("pick")
            if self.pick_upright_ckpt:
                required_agents.append("pick_upright")
            if self.pick_convex_ckpt:
                required_agents.append("pick_convex")
            if self.pick_convex_lying_ckpt:
                required_agents.append("pick_convex_lying")
            if self.pick_handle_ckpt:
                required_agents.append("pick_handle")
            base_env.set_task_obs_space("pick")

        if self.rl_cfg.get("use_place"):
            required_agents.append("place")
            base_env.set_task_obs_space("place")
        if self.rl_cfg.get("use_pose"):
            required_agents.append("pose")
            base_env.set_task_obs_space("pose")
        if self.rl_cfg.get("use_open_drawer"):
            required_agents.append("open_drawer")
            base_env.set_task_obs_space("open_drawer")
        if self.rl_cfg.get("use_close_drawer"):
            required_agents.append("close_drawer")
            base_env.set_task_obs_space("close_drawer")
        if self.rl_cfg.get("use_close_door"):
            required_agents.append("close_door")
            base_env.set_task_obs_space("close_door")

        missing = [name for name in required_agents if name not in loaded_agents]
        if missing:
            raise RuntimeError(
                f"RLPolicyClient server missing required agents: {missing}. Loaded: {sorted(loaded_agents)}"
            )

    def _is_object_lying(self, simulator, task_env, object_name: str) -> bool:
        pose = simulator.get_object_pose(task_env, object_name)
        R = q2R_wxyz(np.asarray(pose[3:], dtype=float))
        return float(R[2, 2]) < CONVEX_UPRIGHT_COS_THRESHOLD

    def _get_object_category(self, object_name: str, simulator, task_env) -> str:
        candidates = _object_category_candidates(simulator, task_env, object_name)
        if candidates:
            return candidates[0]
        return _category_of(object_name)

    def resolve_pick_agent_name(self, object_name: str, simulator, task_env) -> str:
        cats = _object_category_candidates(simulator, task_env, object_name)
        if any(cat in self.handle_objects for cat in cats) and self.pick_handle_ckpt:
            return "pick_handle"

        if any(cat in self.upright_objects for cat in cats) and self.pick_upright_ckpt:
            object_range = simulator.get_object_3d_range(task_env, object_name)
            object_height = float(object_range[-1] - object_range[-2])
            if object_height > self.pick_upright_height_threshold_m:
                return "pick_upright"

        if any(cat in self.convex_objects for cat in cats):
            if self.pick_convex_lying_ckpt and self._is_object_lying(simulator, task_env, object_name):
                return "pick_convex_lying"
            if self.pick_convex_ckpt:
                return "pick_convex"

        return "pick"

    def reset_agent_new_episode(self, init_episode_obs, agent_name: str):
        self._post(
            "/reset_agent_new_episode",
            {
                "agent_name": agent_name,
                "init_episode_obs": self._to_jsonable(init_episode_obs),
            },
        )

    def get_action(self, obs, agent_name: str):
        data = self._post(
            "/get_action",
            {
                "agent_name": agent_name,
                "obs": self._to_jsonable(obs),
            },
        )
        if "actions" not in data:
            raise RuntimeError("RLPolicyClient response missing 'actions'.")
        return np.asarray(data["actions"], dtype=np.float32)

    def post_action_operation(self, dones, agent_name: str):
        self._post(
            "/post_action_operation",
            {
                "agent_name": agent_name,
                "dones": self._to_jsonable(dones),
            },
        )
