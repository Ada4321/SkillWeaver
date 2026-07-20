# SkillWeaver

MCTS-driven skill search + VLA data collection in IsaacLab.

## 1. Installation

(1) Clone a stable release of IsaacLab, using

   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git -b v2.3.1
   ```

(2) Then follow the official [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to install IsaacSim and IsaacLab. IsaacSim installed from either pip or pre-built binary is OK. After this step, you should have a conda environment with IsaacLab and other dependencies (e.g. Pytorch) installed.

(3) Install other dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 1. Data bundle

The scripts under `scripts/` need scenes, assets, and RL checkpoints that live
**outside** this repo. Download the SkillWeaver data bundle and point one env var
at it — nothing in the repo hard-codes an absolute data path.

**Download** `skillweaver_data.tar.gz` (~117 MB, `md5 6443cd7001970dfbf8eb8c2abd0931ac`)
from [Google Drive](https://drive.google.com/file/d/1VP6sJGxixXyHOWoXV9lDUs-C3hM6Pl-O/view?usp=sharing):

```bash
# direct download (handles Google Drive's large-file confirm token):
pip install gdown && gdown 1VP6sJGxixXyHOWoXV9lDUs-C3hM6Pl-O
# ...or grab it from the browser link above.

md5sum -c <<< "6443cd7001970dfbf8eb8c2abd0931ac  skillweaver_data.tar.gz"   # verify
tar -xzf skillweaver_data.tar.gz            # -> skillweaver_data/
export SKILLWEAVER_DATA_ROOT="$(pwd)/skillweaver_data"
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
