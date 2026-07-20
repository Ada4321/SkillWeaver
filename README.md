# SkillWeaver

MCTS-driven skill search + VLA data collection in IsaacLab.

## 1. Data bundle

The scripts under `scripts/` need scenes, assets, and RL checkpoints that live
**outside** this repo. Download the SkillWeaver data bundle and point one env var
at it — nothing in the repo hard-codes an absolute data path.

```bash
export SKILLWEAVER_DATA_ROOT=/path/where/you/extracted/skillweaver_data
```

Bundle layout:

```
skillweaver_data/
├── sim_scene_gen/                 # scene_asset_root_path
│   ├── scenes/{libero/libero_object/task_0, simpler/<task>}/scene_0001/
│   ├── assets/{libero, simpler, Kitchen_Table}/…       # only the referenced assets
│   └── background/{floors, walls, tables, lightings, poster_overlay}/…
├── ckpts/
│   ├── gripper_pick_general.pth   # franka pick (skills.rl.pick_ckpt)
│   ├── gripper_pick_upright.pth   # franka upright pick
│   └── widowx_pick.pth            # widowx pick (simpler tasks)
└── misc/
    ├── surface_list_0305.txt
    └── widowx_rl_games_ppo_cfg.yaml
```

The bundle currently ships `scene_0001` of each supported task and exactly the
assets/backgrounds those scenes reference. Scene JSONs may embed absolute paths
authored under the original data root; they are rewritten onto
`SKILLWEAVER_DATA_ROOT` automatically at load ([simulators/base_env.py](simulators/base_env.py)).

## 2. Config

```bash
cp env.example .env && $EDITOR .env    # set SKILLWEAVER_DATA_ROOT + GEMINI_API_KEY
source .env
```

`GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is read by [skills/gemini.py](skills/gemini.py)
for the VLM actor/judge.

## 3. Run

Scripts `cd` to the repo root themselves (paths inside the repo are repo-relative),
so you can launch them from anywhere once the two env vars are set:

```bash
# libero (franka)
bash scripts/mcts_libero/libero_object_task0.sh          # SCENE_IDS defaults to 1
# simpler (widowx)
bash scripts/mcts_simpler/data_collect_carrot_on_plate_vlm.sh
```

Collected trajectories are written under `SAVE_ROOT` (defaults to `output/…`;
override with `SAVE_ROOT=…`).

## 4. Rebuilding the bundle (maintainers)

`tools/build_data_bundle.py` regenerates the bundle from the source data tree by
parsing each task's `scene_0001` and copying exactly the referenced asset dirs,
backgrounds, ckpts, and misc files:

```bash
python tools/build_data_bundle.py            # dry-run: prints manifest + size
python tools/build_data_bundle.py --execute  # copy into DST_ROOT
```

Edit `TASKS` / `SCENE_IDS` in that file to widen coverage beyond `scene_0001`.
