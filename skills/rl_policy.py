import logging
import os
import re
import numpy as np
import torch
from copy import deepcopy
import yaml

from isaaclab.utils.assets import retrieve_file_path
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from skills.skills import (
    CONVEX_UPRIGHT_COS_THRESHOLD,
    _object_category_candidates,
)
from utils.geometry_utils import q2R_wxyz


# SceneGen scenes use "<category>_<4-digit asset id>" object names; libero
# scenes use bare category names (no suffix) or "<name>_<1-2 digits>". Strip
# only the 4+ digit suffix so libero names like "black_bowl_1" stay intact.
_ASSET_SUFFIX_RE = re.compile(r"_\d{4,}$")


def _category_of(object_name: str) -> str:
    return _ASSET_SUFFIX_RE.sub("", object_name)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


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


class RLPolicy:
    def __init__(self, args) -> None:
        self.num_envs = args["num_envs"]
        agent_cfg_file = args["agent_cfg_file"]
        base_agent_cfg = load_yaml(agent_cfg_file)
        self.base_agent_cfg = base_agent_cfg

        device_override = args.get("device")
        if device_override:
            base_agent_cfg["params"]["config"]["device"] = device_override
            base_agent_cfg["params"]["config"]["device_name"] = device_override
        self.device = base_agent_cfg["params"]["config"]["device"]

        self.agent_ckpts = {}
        self.agents = {}

        if "pick_ckpt" in args:
            pick_ckpt = args["pick_ckpt"]
            self.agent_ckpts["pick"] = pick_ckpt
            # self.agents["pick"] = self.create_agent_from_ckpt(pick_ckpt, base_agent_cfg)

        if "pick_upright_ckpt" in args:
            pick_upright_ckpt = args["pick_upright_ckpt"]
            self.agent_ckpts["pick_upright"] = pick_upright_ckpt

        if "pick_convex_ckpt" in args:
            pick_convex_ckpt = args["pick_convex_ckpt"]
            self.agent_ckpts["pick_convex"] = pick_convex_ckpt

        if args.get("pick_convex_lying_ckpt"):
            self.agent_ckpts["pick_convex_lying"] = args["pick_convex_lying_ckpt"]

        if args.get("pick_handle_ckpt"):
            self.agent_ckpts["pick_handle"] = args["pick_handle_ckpt"]

        if "place_ckpt" in args:
            place_ckpt = args["place_ckpt"]
            self.agent_ckpts["place"] = place_ckpt
            # self.agents["place"] = self.create_agent_from_ckpt(place_ckpt, base_agent_cfg)

        if "place_upright_ckpt" in args:
            place_upright_ckpt = args["place_upright_ckpt"]
            self.agent_ckpts["place_upright"] = place_upright_ckpt

        if "pose_ckpt" in args:
            pose_ckpt = args["pose_ckpt"]
            self.agent_ckpts["pose"] = pose_ckpt
            # self.agents["pose"] = self.create_agent_from_ckpt(pose_ckpt, base_agent_cfg)
        
        if "open_drawer_ckpt" in args:
            self.agent_ckpts["open_drawer"] = args["open_drawer_ckpt"]

        if "close_drawer_ckpt" in args:
            self.agent_ckpts["close_drawer"] = args["close_drawer_ckpt"]

        if "close_door_ckpt" in args:
            self.agent_ckpts["close_door"] = args["close_door_ckpt"]

        self.pick_upright_height_threshold_m = 0.10
        self.upright_objects = load_object_list_from_txt(args.get("upright_list"), "upright_list")
        self.convex_objects = load_object_list_from_txt(args.get("convex_list"), "convex_list")
        self.handle_objects = load_object_list_from_txt(args.get("handle_list"), "handle_list")
        self.place_upright_objects = load_object_list_from_txt(
            args.get("place_upright_list"), "place_upright_list"
        )
        self.place_upright_tgt_cos_threshold = float(
            args.get("place_upright_tgt_cos_threshold", 0.7071)
        )
        self.place_upright_gripper_cos_threshold = float(
            args.get("place_upright_gripper_cos_threshold", 0.707)
        )
        self.place_align_cup_bottom = bool(args.get("place_align_cup_bottom", False))
        self.top_pcd_agents = set(args.get("top_pcd_agents", ["pick_convex_lying"]))
        self.handle_pcd_agents = set(args.get("handle_pcd_agents", ["pick_handle"]))

    def uses_top_pcd(self, agent_name: str) -> bool:
        return agent_name in self.top_pcd_agents

    def uses_handle_pcd(self, agent_name: str) -> bool:
        return agent_name in self.handle_pcd_agents

    def init_agents(self, simulator=None, task_env=None):
        base_env = simulator._unwrap_env(task_env)
        for agent_name, ckpt_path in self.agent_ckpts.items():
            if agent_name in ("pick_upright", "pick_convex", "pick_convex_lying", "pick_handle"):
                base_env.set_task_obs_space("pick")
            elif agent_name == "place_upright":
                base_env.set_task_obs_space("place")
            else:
                base_env.set_task_obs_space(agent_name)
            self.agents[agent_name] = self.create_agent_from_ckpt(ckpt_path, self.base_agent_cfg)
    
    def create_agent_from_ckpt(self, ckpt_path: str, agent_cfg: dict):
        agent_cfg_current = deepcopy(agent_cfg)
        resume_path = retrieve_file_path(ckpt_path)
        # load previously trained model
        agent_cfg_current["params"]["load_checkpoint"] = True
        agent_cfg_current["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg_current['params']['load_path']}")
        
        # set number of actors into agent config
        agent_cfg_current["params"]["config"]["num_actors"] = self.num_envs
        # create runner from rl-games
        runner = Runner()
        runner.load(agent_cfg_current)
        # obtain the agent from the runner
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)

        return agent


    def reset_agent_new_episode(self, init_episode_obs, agent_name: str):
        agent: BasePlayer = self.agents[agent_name]
        agent.reset()
        # required: enables the flag for batched observations
        _ = agent.get_batch_size(init_episode_obs, 1)
        # initialize RNN states if used
        if agent.is_rnn:
            agent.init_rnn()
        
    
    def get_action(self, obs, agent_name: str):
        agent: BasePlayer = self.agents[agent_name]
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
        return actions
    

    def post_action_operation(self, dones, agent_name: str):
        agent: BasePlayer = self.agents[agent_name]
        # perform operations for terminated episodes
        if len(dones) > 0:
            # reset rnn state for terminated episodes
            if agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

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
        if any(cat in self.handle_objects for cat in cats) and "pick_handle" in self.agents:
            return "pick_handle"

        if (
            any(cat in self.upright_objects for cat in cats)
            and "pick_upright" in self.agents
        ):
            object_range = simulator.get_object_3d_range(task_env, object_name)
            object_height = float(object_range[-1] - object_range[-2])
            if object_height > self.pick_upright_height_threshold_m:
                return "pick_upright"

        if any(cat in self.convex_objects for cat in cats):
            if "pick_convex_lying" in self.agents and self._is_object_lying(simulator, task_env, object_name):
                return "pick_convex_lying"
            if "pick_convex" in self.agents:
                return "pick_convex"

        return "pick"

    def resolve_place_agent_name(
        self, held_object_name: str, simulator, task_env
    ) -> str:
        if "place_upright" not in self.agents or not held_object_name:
            print(f"[PLACE_DISPATCH] place (no place_upright agent or held={held_object_name!r})", flush=True)
            return "place"
        cats = _object_category_candidates(simulator, task_env, held_object_name)
        if not any(cat in self.place_upright_objects for cat in cats):
            print(f"[PLACE_DISPATCH] place (held_cats={cats!r} not in place_upright_list)", flush=True)
            return "place"
        held_pose = simulator.get_object_pose(task_env, held_object_name)
        R_held = q2R_wxyz(np.asarray(held_pose[3:], dtype=float))
        if float(R_held[2, 2]) < self.place_upright_tgt_cos_threshold:
            print(f"[PLACE_DISPATCH] place (held={held_object_name} not upright R_held[2,2]={float(R_held[2,2]):.3f} thr={self.place_upright_tgt_cos_threshold})", flush=True)
            return "place"
        ee_pose = simulator.get_ee_pose(task_env)
        R_ee = q2R_wxyz(np.asarray(ee_pose[3:7], dtype=float))
        if abs(float(R_ee[2, 2])) > self.place_upright_gripper_cos_threshold:
            print(f"[PLACE_DISPATCH] place (gripper too vertical |R_ee[2,2]|={abs(float(R_ee[2,2])):.3f} thr={self.place_upright_gripper_cos_threshold})", flush=True)
            return "place"
        print(f"[PLACE_DISPATCH] PLACE_UPRIGHT (held={held_object_name} R_held[2,2]={float(R_held[2,2]):.3f} |R_ee[2,2]|={abs(float(R_ee[2,2])):.3f})", flush=True)
        return "place_upright"
