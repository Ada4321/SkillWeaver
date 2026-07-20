#!/usr/bin/env python3
"""Build the minimal SkillWeaver data bundle for scripts/ to run.

Collects, for each task's scene_0001, exactly the files the code loads:
  - the scene dir (self-contains scene_XXXX.json + layouts/table_z/cam_poses/tasks npz)
  - every referenced asset_XXXX dir (minus raw_images/, *_ORIG.usd, *.bak*)
  - every referenced background file (+ transitive USD deps)
  - the 3 RL pick ckpts
  - external single config files (surface list, widowx rl_games cfg)

into a clean tree under DST_ROOT/, keeping the sim_scene_gen/{scenes,assets,background}
layout so scene JSONs resolve unchanged against scene_asset_root_path.

Usage:
  python tools/build_data_bundle.py            # dry-run: print manifest + size
  python tools/build_data_bundle.py --execute  # actually copy
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

SRC_SG = "/data/group_data/katefgroup-ssd/sim_scene_gen"
DST_ROOT = "/data/user_data/hez2/skillweaver_data"

# task scene-subdir -> which scene ids to include
TASKS = {
    "libero/libero_object/task_0": ["scene_0001"],
    "simpler/carrot_on_plate": ["scene_0001"],
    "simpler/eggplant_in_basket": ["scene_0001"],
    "simpler/spoon_on_towel": ["scene_0001"],
    "simpler/stack_green_on_yellow": ["scene_0001"],
}

CKPTS = {
    "/home/hez2/code/IsaacLabEnvs/logs/rl_games/gripper/2026-02-19_06-40-46_pick_sam3d_general_8gpu_tablez0.25/nn/last_gripper_ep_9600_rew_1706.0063.pth": "ckpts/gripper_pick_general.pth",
    "/home/hez2/code/IsaacLabEnvs/logs/rl_games/gripper/2026-02-19_19-30-54_pick_sam3d_upright_gt10_4gpu/nn/last_gripper_ep_8400_rew_1541.1537.pth": "ckpts/gripper_pick_upright.pth",
    "/home/suli/last_gripper_ep_9800_rew_2142.01.pth": "ckpts/widowx_pick.pth",
}

MISC = {
    "/home/hez2/code/SceneGen_IsaacLab/asset_gen/asset_post_process/surface_list_0305.txt": "misc/surface_list_0305.txt",
    "/home/hez2/code/IsaacLabEnvs/isaaclabenvs/tasks/widowx/agents/rl_games_ppo_cfg.yaml": "misc/widowx_rl_games_ppo_cfg.yaml",
}

ASSET_EXCLUDES = ["raw_images/", "*_ORIG.usd", "*.bak*", ".asset_hash"]

# tokens that look like an asset path inside a USD (crate blobs still hold plaintext paths)
_DEP_RE = re.compile(r"[@\"']?([A-Za-z0-9_./-]+\.(?:usd|usda|usdc|png|jpg|jpeg|obj|mtl))[@\"']?")


def usd_deps(usd_path):
    """Return relative dep paths referenced by a USD (best-effort via strings)."""
    try:
        out = subprocess.run(["strings", usd_path], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    deps = set()
    for m in _DEP_RE.finditer(out):
        p = m.group(1)
        if p.startswith("/") or p.startswith("./") is False and "/" not in p and p == os.path.basename(usd_path):
            continue
        deps.add(p)
    return sorted(deps)


def collect():
    """Return (scene_dirs, asset_dirs, bg_refs) as source paths.

    bg_refs are the raw 'background/...' relative refs from the scene JSONs;
    resolve_background() turns them into the concrete files/dirs to copy.
    """
    scene_dirs, asset_dirs, bg_refs = set(), set(), set()
    for task, scenes in TASKS.items():
        for sid in scenes:
            sdir = os.path.join(SRC_SG, "scenes", task, sid)
            sj = os.path.join(sdir, f"{sid}.json")
            if not os.path.isfile(sj):
                sys.exit(f"MISSING scene json: {sj}")
            scene_dirs.add(sdir)
            refs = re.findall(r'"((?:assets|background)/[^"]+)"', open(sj).read())
            for r in refs:
                if r.startswith("assets/"):
                    m = re.match(r"(assets/[^/]+/[^/]+/asset_[0-9]+)/", r)
                    if m:
                        asset_dirs.add(os.path.join(SRC_SG, m.group(1)))
                else:  # background/...
                    bg_refs.add(r)
    return sorted(scene_dirs), sorted(asset_dirs), sorted(bg_refs)


def resolve_background(bg_refs):
    """Map referenced background/... relative refs to the concrete source paths to copy.

    Per-category rules (the USDs are compressed crate, so we resolve texture deps by
    the dataset's layout conventions instead of parsing the binaries):
      floors/*      -> copy the whole background/floors dir (tiny, 6.7M; self-contained)
      walls/<s>.usd -> copy the usd + textures/<s>_tex.png (verified 1:1 convention)
      tables/*      -> copy the referenced usd + the whole tables/usd/textures dir
                       (the crate usd's texture binding is not statically resolvable)
      everything else (lightings .exr, poster_overlay .png, ...) -> copy the file as-is
    """
    out = set()
    for r in bg_refs:
        parts = r.split("/")
        cat = parts[1] if len(parts) > 1 else ""
        if cat == "floors":
            out.add(os.path.join(SRC_SG, "background/floors"))
        elif cat == "walls" and r.endswith(".usd"):
            out.add(os.path.join(SRC_SG, r))
            stem = os.path.splitext(os.path.basename(r))[0]
            tex = os.path.join(SRC_SG, "background/walls/textures", f"{stem}_tex.png")
            if os.path.isfile(tex):
                out.add(tex)
        elif cat == "tables":
            out.add(os.path.join(SRC_SG, r))
            out.add(os.path.join(SRC_SG, "background/tables/usd/textures"))
        else:
            out.add(os.path.join(SRC_SG, r))
    return sorted(out)


def rsync(src, dst, execute, excludes=()):
    os.makedirs(os.path.dirname(dst.rstrip("/")), exist_ok=True)
    cmd = ["rsync", "-a"]
    if not execute:
        cmd += ["-n", "--stats"]
    for e in excludes:
        cmd += ["--exclude", e]
    cmd += [src, dst]
    return subprocess.run(cmd, capture_output=True, text=True)


def size_of(paths, excludes=None):
    total = 0
    for p in paths:
        if os.path.isfile(p):
            total += os.path.getsize(p)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                if excludes and "raw_images" in os.path.basename(root):
                    dirs[:] = []
                    continue
                for fn in files:
                    if excludes and (fn.endswith("_ORIG.usd") or ".bak" in fn or fn == ".asset_hash"):
                        continue
                    fp = os.path.join(root, fn)
                    if os.path.isfile(fp):
                        total += os.path.getsize(fp)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="actually copy (default: dry-run)")
    args = ap.parse_args()

    scene_dirs, asset_dirs, bg_refs = collect()
    bg_paths = resolve_background(bg_refs)

    print(f"scene dirs      : {len(scene_dirs)}")
    print(f"asset dirs      : {len(asset_dirs)}  (excl {ASSET_EXCLUDES})")
    print(f"background paths: {len(bg_paths)}  (from {len(bg_refs)} refs)")
    print(f"ckpts           : {len(CKPTS)}")
    print(f"misc            : {len(MISC)}")

    gb = 1024 ** 3
    sz = (size_of(scene_dirs) + size_of(asset_dirs, excludes=True)
          + size_of(bg_paths) + size_of(list(CKPTS)) + size_of(list(MISC)))
    print(f"\nestimated bundle size: {sz/gb:.2f} GB\n")

    missing = [c for c in list(CKPTS) + list(MISC) if not os.path.exists(c)]
    if missing:
        print("!! MISSING sources:")
        for m in missing:
            print("   ", m)

    if not args.execute:
        print("dry-run only. re-run with --execute to copy.")
        return

    # scenes
    for sdir in scene_dirs:
        rel = os.path.relpath(sdir, SRC_SG)
        dst = os.path.join(DST_ROOT, "sim_scene_gen", rel) + "/"
        rsync(sdir + "/", dst, True)
    # assets (stripped)
    for adir in asset_dirs:
        rel = os.path.relpath(adir, SRC_SG)
        dst = os.path.join(DST_ROOT, "sim_scene_gen", rel) + "/"
        rsync(adir + "/", dst, True, excludes=ASSET_EXCLUDES)
    # backgrounds (files or whole dirs, per resolve_background rules)
    for p in bg_paths:
        rel = os.path.relpath(p, SRC_SG)
        dst = os.path.join(DST_ROOT, "sim_scene_gen", rel)
        if os.path.isdir(p):
            rsync(p + "/", dst + "/", True)
        else:
            rsync(p, dst, True)
    # ckpts + misc
    for src, rel in {**CKPTS, **MISC}.items():
        if not os.path.exists(src):
            continue
        dst = os.path.join(DST_ROOT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    print(f"\nDONE. bundle at {DST_ROOT}")


if __name__ == "__main__":
    main()
