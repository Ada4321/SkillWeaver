import argparse
import logging
import traceback
from copy import deepcopy
import os

import gym
import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

try:
    # IsaacLab helper (works when Isaac/Omniverse runtime is available).
    from isaaclab.utils.assets import retrieve_file_path as _isaac_retrieve_file_path
except Exception:
    _isaac_retrieve_file_path = None


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def retrieve_file_path(path: str) -> str:
    if _isaac_retrieve_file_path is not None:
        return _isaac_retrieve_file_path(path)

    # Fallback path resolver for non-Isaac environments.
    resolved = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Checkpoint path does not exist: {resolved}")
    return resolved


class RLPolicyServer:
    def __init__(self, args) -> None:
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        self.app = FastAPI()
        self.app.get("/health")(self.health)
        self.app.post("/reset_agent_new_episode")(self.reset_agent_new_episode)
        self.app.post("/get_action")(self.get_action)
        self.app.post("/post_action_operation")(self.post_action_operation)

        self.num_envs = int(args["num_envs"])
        agent_cfg_file = args["agent_cfg_file"]
        base_agent_cfg = load_yaml(agent_cfg_file)

        device_override = args.get("device")
        if device_override:
            base_agent_cfg["params"]["config"]["device"] = device_override
            base_agent_cfg["params"]["config"]["device_name"] = device_override
        self.device = base_agent_cfg["params"]["config"]["device"]
        self.action_dim = int(args.get("action_dim", 7))
        pick_obs_dim = int(args.get("pick_obs_dim", 80))
        self.obs_dim_by_task = {
            "pick": pick_obs_dim,
            "pick_upright": pick_obs_dim,
            "pick_convex": pick_obs_dim,
            "pick_convex_lying": pick_obs_dim,
            "pick_handle": pick_obs_dim,
            "place": int(args.get("place_obs_dim", 94)),
            "pose": int(args.get("pose_obs_dim", 94)),
            "open": int(args.get("open_obs_dim", 80)),
            "close": int(args.get("close_obs_dim", 80)),
        }

        self.agents = {}
        ckpt_by_task = {
            "pick": args.get("pick_ckpt"),
            "pick_upright": args.get("pick_upright_ckpt"),
            "pick_convex": args.get("pick_convex_ckpt"),
            "pick_convex_lying": args.get("pick_convex_lying_ckpt"),
            "pick_handle": args.get("pick_handle_ckpt"),
            "place": args.get("place_ckpt"),
            "pose": args.get("pose_ckpt"),
            "open": args.get("open_ckpt"),
            "close": args.get("close_ckpt"),
        }
        for task_name, ckpt_path in ckpt_by_task.items():
            if ckpt_path:
                self.agents[task_name] = self.create_agent_from_ckpt(task_name, ckpt_path, base_agent_cfg)

    def _error(self, exc: Exception, status_code: int = 500) -> JSONResponse:
        tb = traceback.format_exc()
        logging.error(tb)
        return JSONResponse(
            content={
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
            },
            status_code=status_code,
        )

    def _build_env_info(self, task_name: str) -> dict:
        obs_dim = int(self.obs_dim_by_task[task_name])
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        return {
            "observation_space": obs_space,
            "action_space": action_space,
            "agents": 1,
            "value_size": 1,
        }

    def _as_device_tensor(self, x):
        t = torch.as_tensor(x, device=self.device)
        if t.is_floating_point():
            return t.to(dtype=torch.float32)
        return t

    def _to_torch_data(self, x):
        # Keep nested observation dict/list structure for rl-games obs_to_torch().
        if isinstance(x, dict):
            return {k: self._to_torch_data(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            arr = np.asarray(x)
            if arr.dtype != object:
                return self._as_device_tensor(arr)
            return [self._to_torch_data(v) for v in x]
        if isinstance(x, np.ndarray):
            return self._as_device_tensor(x)
        if isinstance(x, np.generic):
            return x.item()
        if torch.is_tensor(x):
            x = x.to(self.device)
            if x.is_floating_point():
                x = x.to(dtype=torch.float32)
            return x
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return self._as_device_tensor(np.asarray(x))

    def create_agent_from_ckpt(self, task_name: str, ckpt_path: str, agent_cfg: dict):
        agent_cfg_current = deepcopy(agent_cfg)
        resume_path = retrieve_file_path(ckpt_path)
        agent_cfg_current["params"]["load_checkpoint"] = True
        agent_cfg_current["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg_current['params']['load_path']}")

        agent_cfg_current["params"]["config"]["num_actors"] = self.num_envs
        # Enable environment-free inference server: avoid creating rlgpu env in this process.
        agent_cfg_current["params"]["config"]["env_info"] = self._build_env_info(task_name)
        agent_cfg_current["params"]["config"]["vec_env"] = None
        runner = Runner()
        runner.load(agent_cfg_current)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        return agent

    async def health(self):
        return {"ok": True, "device": self.device, "agents": sorted(self.agents.keys())}

    async def reset_agent_new_episode(self, request: Request):
        try:
            payload = await request.json()
            agent_name = payload["agent_name"]
            if agent_name not in self.agents:
                return JSONResponse(
                    content={"error": f"Agent '{agent_name}' is not loaded on server. Supported names: {sorted(self.agents.keys())}"},
                    status_code=400,
                )
            init_episode_obs = self._to_torch_data(payload["init_episode_obs"])

            agent: BasePlayer = self.agents[agent_name]
            agent.reset()
            _ = agent.get_batch_size(init_episode_obs, 1)
            if agent.is_rnn:
                agent.init_rnn()
            return {"ok": True}
        except Exception as exc:
            return self._error(exc)

    async def get_action(self, request: Request):
        try:
            payload = await request.json()
            agent_name = payload["agent_name"]
            if agent_name not in self.agents:
                return JSONResponse(
                    content={"error": f"Agent '{agent_name}' is not loaded on server. Supported names: {sorted(self.agents.keys())}"},
                    status_code=400,
                )
            obs = self._to_torch_data(payload["obs"])

            agent: BasePlayer = self.agents[agent_name]
            with torch.inference_mode():
                obs = agent.obs_to_torch(obs)
                actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            return {"actions": actions.detach().cpu().numpy().tolist()}
        except Exception as exc:
            return self._error(exc)

    async def post_action_operation(self, request: Request):
        try:
            payload = await request.json()
            agent_name = payload["agent_name"]
            if agent_name not in self.agents:
                return JSONResponse(
                    content={"error": f"Agent '{agent_name}' is not loaded on server. Supported names: {sorted(self.agents.keys())}"},
                    status_code=400,
                )
            dones = torch.as_tensor(payload["dones"], device=self.device, dtype=torch.bool)
            if dones.ndim > 1:
                dones = dones.squeeze()

            agent: BasePlayer = self.agents[agent_name]
            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0
            return {"ok": True}
        except Exception as exc:
            return self._error(exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9080)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--agent_cfg_file", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--pick_obs_dim", type=int, default=80)
    parser.add_argument("--place_obs_dim", type=int, default=94)
    parser.add_argument("--pose_obs_dim", type=int, default=94)
    parser.add_argument("--open_obs_dim", type=int, default=80)
    parser.add_argument("--close_obs_dim", type=int, default=80)
    parser.add_argument("--pick_ckpt", type=str, default=None)
    parser.add_argument("--pick_upright_ckpt", type=str, default=None)
    parser.add_argument("--pick_convex_ckpt", type=str, default=None)
    parser.add_argument("--pick_convex_lying_ckpt", type=str, default=None)
    parser.add_argument("--pick_handle_ckpt", type=str, default=None)
    parser.add_argument("--place_ckpt", type=str, default=None)
    parser.add_argument("--pose_ckpt", type=str, default=None)
    parser.add_argument("--open_ckpt", type=str, default=None)
    parser.add_argument("--close_ckpt", type=str, default=None)
    args = parser.parse_args()

    server = RLPolicyServer(args)
    uvicorn.run(server.app, host=args.host, port=args.port)
