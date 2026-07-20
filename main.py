from datetime import datetime
import os
import datetime
import multiprocessing as mp

import hydra
from omegaconf import DictConfig, OmegaConf

from runners import run_eval


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    save_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.run.get("save_tag"):
        save_tag = f"{cfg.run.save_tag}_{save_tag}"
    out_dir = os.path.join(cfg.run.save_rollouts_dir, save_tag)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    cfg = OmegaConf.to_container(cfg, resolve=True)

    run_eval(cfg, out_dir, use_mcts=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
