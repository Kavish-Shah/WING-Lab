"""Relabel a LeRobot dataset with synthetic joint encoder faults to train a Masker policy.

Implements the Self-Adaptive VLA recipe for Week 3-5:
  For each demonstration episode:
    1. Sample a random joint offset delta ~ Uniform(-max_deg, +max_deg)
       for pitch joints (shoulder_lift, elbow_flex, wrist_flex).
    2. Subtract delta from recorded joint states and joint actions:
         q_observed <- q_observed - delta
         a_commanded <- a_commanded - delta
    3. Leave camera video frames completely untouched.

This trains the policy:
  "When camera vision shows the true link position, but motor encoders report
   (true - delta), automatically command (action - delta) to compensate."

The script updates:
  - data/*/*.parquet: rewritten observation.state and action columns
  - meta/stats.json: recomputed global normalization quantiles
  - meta/episodes/*.parquet: recomputed per-episode statistics

Usage:
  python collect/relabel_masker.py --src data/cube500_bin --dst data/cube500_masker_2deg --max-deg 2.0
  python collect/relabel_masker.py --src data/cube500_bin --dst data/cube500_masker_5deg --max-deg 5.0
  python collect/relabel_masker.py --src data/cube500_bin --dst data/cube500_masker_10deg --max-deg 10.0
"""
import argparse
import glob
import json
import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as papq

def update_stats_dict(stat_entry, array_data):
    """Update min, max, mean, std, and quantiles for a feature dictionary."""
    stat_entry["min"] = array_data.min(0).tolist()
    stat_entry["max"] = array_data.max(0).tolist()
    stat_entry["mean"] = array_data.mean(0).tolist()
    stat_entry["std"] = array_data.std(0).tolist()
    for q in ("q01", "q10", "q50", "q90", "q99"):
        stat_entry[q] = np.quantile(array_data, float(q[1:]) / 100.0, axis=0).tolist()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source LeRobot dataset directory (e.g. data/cube500_bin)")
    ap.add_argument("--dst", required=True, help="output directory for relabeled dataset")
    ap.add_argument("--max-deg", type=float, required=True, help="maximum fault offset in degrees (e.g. 2.0, 5.0, 10.0)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducible delta generation")
    ap.add_argument("--joints", nargs="+", default=["shoulder_lift", "elbow_flex", "wrist_flex"],
                    help="joints to inject offsets on (indices 1, 2, 3)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing dst directory")
    args = ap.parse_args()

    SRC = pathlib.Path(args.src).resolve()
    DST = pathlib.Path(args.dst).resolve()

    if not SRC.exists():
        ap.error(f"Source directory {SRC} does not exist")
    if DST.exists():
        if not args.overwrite:
            ap.error(f"Destination {DST} exists; pass --overwrite to replace it")
        shutil.rmtree(DST)

    print(f"Creating masker dataset variant with +/-{args.max_deg} deg offsets...")
    print(f"  Source: {SRC}")
    print(f"  Destination: {DST}")

    # Copy directory structure and files
    # Use copytree; videos will be copied or hardlinked
    DST.mkdir(parents=True, exist_ok=True)
    for item in SRC.iterdir():
        if item.name == "videos":
            # Hardlink video files to save disk space and be instant
            print("  Hardlinking videos to save disk space...")
            dst_videos = DST / "videos"
            shutil.copytree(item, dst_videos, copy_function=os.link)
        elif item.is_dir():
            shutil.copytree(item, DST / item.name)
        else:
            shutil.copy2(item, DST / item.name)

    # Mapping joint names to indices (0: pan, 1: shoulder, 2: elbow, 3: wrist_flex, 4: wrist_roll, 5: gripper)
    joint_indices_map = {
        "shoulder_pan": 0,
        "shoulder_lift": 1,
        "elbow_flex": 2,
        "wrist_flex": 3,
        "wrist_roll": 4
    }
    target_indices = [joint_indices_map[j] for j in args.joints if j in joint_indices_map]
    print(f"  Injecting offsets on joint indices {target_indices} ({args.joints})")

    rng = np.random.RandomState(args.seed)

    # First, collect all unique episode IDs across parquet files
    parquet_files = sorted(glob.glob(f"{DST}/data/*/*.parquet"))
    all_episodes = set()
    for pq in parquet_files:
        df = pd.read_parquet(pq, columns=["episode_index"])
        all_episodes.update(df["episode_index"].unique())

    all_episodes = sorted(list(all_episodes))
    print(f"  Found {len(all_episodes)} episodes.")

    # Assign a fixed delta per episode
    episode_deltas = {}
    for ep_idx in all_episodes:
        delta = np.zeros(6, dtype=np.float32)
        for j_idx in target_indices:
            delta[j_idx] = rng.uniform(-args.max_deg, args.max_deg)
        episode_deltas[ep_idx] = delta

    all_actions = []
    all_states = []

    # Relabel data
    for pq in parquet_files:
        df = pd.read_parquet(pq)
        A = np.stack(df["action"].to_numpy())
        S = np.stack(df["observation.state"].to_numpy())
        ep_arr = df["episode_index"].to_numpy()

        A_new = A.copy()
        S_new = S.copy()

        for ep_idx in np.unique(ep_arr):
            mask = (ep_arr == ep_idx)
            delta = episode_deltas[ep_idx]
            # Subtract delta from state and action
            S_new[mask] -= delta
            A_new[mask] -= delta

        df["action"] = list(A_new.astype(np.float32))
        df["observation.state"] = list(S_new.astype(np.float32))
        df.to_parquet(pq)

        all_actions.append(A_new)
        all_states.append(S_new)

    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)

    # Rewrite meta/stats.json
    stats_path = DST / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.load(open(stats_path))
        if "action" in stats:
            update_stats_dict(stats["action"], all_actions)
        if "observation.state" in stats:
            update_stats_dict(stats["observation.state"], all_states)
        json.dump(stats, open(stats_path, "w"), indent=4)
        print("  Updated meta/stats.json with recomputed action & state normalization.")

    # Rewrite meta/episodes/*.parquet per-episode stats
    ep_files = glob.glob(f"{DST}/meta/episodes/**/*.parquet", recursive=True)
    if ep_files:
        frames = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        by_ep_act = {int(e): np.stack(g["action"].to_numpy()).astype(np.float64)
                     for e, g in frames.groupby("episode_index")}
        by_ep_state = {int(e): np.stack(g["observation.state"].to_numpy()).astype(np.float64)
                       for e, g in frames.groupby("episode_index")}

        STAT = {
            "min": lambda X: X.min(0), "max": lambda X: X.max(0), "mean": lambda X: X.mean(0),
            "std": lambda X: X.std(0), "count": lambda X: [len(X)],
            "q01": lambda X: np.quantile(X, .01, 0), "q10": lambda X: np.quantile(X, .10, 0),
            "q50": lambda X: np.quantile(X, .50, 0), "q90": lambda X: np.quantile(X, .90, 0),
            "q99": lambda X: np.quantile(X, .99, 0)
        }

        for epf in ep_files:
            t = papq.read_table(epf)
            sch = t.schema
            order = t.column("episode_index").to_pylist()
            for s, fn in STAT.items():
                for feat, by_ep in [("action", by_ep_act), ("observation.state", by_ep_state)]:
                    col = f"stats/{feat}/{s}"
                    if col not in sch.names:
                        continue
                    vals = [([len(by_ep[int(e)])] if s == "count"
                             else list(np.asarray(fn(by_ep[int(e)])).tolist())) for e in order]
                    j = sch.get_field_index(col)
                    t = t.set_column(j, sch.field(j), pa.array(vals, type=sch.field(j).type))
            papq.write_table(t, epf)
        print("  Updated meta/episodes stats.")

    # Write a masker metadata file documenting the split
    meta_masker = {
        "max_deg": args.max_deg,
        "seed": args.seed,
        "target_joints": args.joints,
        "n_episodes": len(all_episodes),
        "total_frames": len(all_actions)
    }
    json.dump(meta_masker, open(DST / "meta" / "masker_config.json", "w"), indent=2)

    print(f"SUCCESS: Generated Masker Dataset: {DST}")
    print(f"  Frames: {len(all_actions)}, Range: [-{args.max_deg} deg, +{args.max_deg} deg]")

if __name__ == "__main__":
    main()
