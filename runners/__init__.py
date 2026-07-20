
def build_runner(runner_mode):
    if runner_mode == "isaaclab_eval":
        from runners.runner_isaaclab import evaluate_isaaclab_mp
        return evaluate_isaaclab_mp
    else:
        raise NotImplementedError()


def run_eval(cfg, out_dir, use_mcts=False):
    runner_cfg = cfg["run"]["runner"]
    eval_cfg = cfg["run"]["eval"]
    runner_func = build_runner(runner_cfg["mode"])
    runner_func(
        cfg=cfg,
        out_dir=out_dir,
        k=eval_cfg["pass_k"],
        use_mcts=use_mcts,
        num_processes=runner_cfg["num_processes"]
    )
