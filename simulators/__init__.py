import os

import gymnasium as gym

from simulators import agents


gym.register(
    id="GroundedBase-v0",
    entry_point="simulators.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:BaseEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


def init_simulator(sim_cfg: dict, env_cfg: dict | None = None):
    simulator_name = sim_cfg.get("name")
    if simulator_name == "isaaclab":
        from simulators.isaaclab import IsaacLabInterface, parse_simulator_args_from_cfg
        sim_args = parse_simulator_args_from_cfg(sim_cfg, env_cfg=env_cfg)
        simulator = IsaacLabInterface(sim_args)
        return simulator
    raise NotImplementedError(f"Simulator {simulator_name} not implemented.")
