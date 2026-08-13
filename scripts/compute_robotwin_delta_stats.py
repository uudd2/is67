import argparse
import json
import os
from glob import glob

import h5py
import numpy as np

def task_names(data_root, setting, tasks):
    if tasks:
        return tasks
    return [
        name
        for name in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, name, setting))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--setting", default="aloha-agilex_clean_50")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=50)
    parser.add_argument("--delta-clip", type=float, default=5.0)
    parser.add_argument("--std-floor", type=float, default=1e-6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    count = 0
    sum_delta = np.zeros(14, dtype=np.float64)
    sum_sq_delta = np.zeros(14, dtype=np.float64)
    min_delta = np.full(14, np.inf, dtype=np.float64)
    max_delta = np.full(14, -np.inf, dtype=np.float64)
    num_episodes = 0

    for task in task_names(args.data_root, args.setting, args.tasks):
        data_dir = os.path.join(args.data_root, task, args.setting, "data")
        paths = sorted(
            glob(os.path.join(data_dir, "episode*.hdf5")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].replace("episode", "")),
        )
        if args.max_episodes_per_task is not None:
            paths = paths[: args.max_episodes_per_task]
        for path in paths:
            with h5py.File(path, "r") as f:
                qpos = f["joint_action/vector"][()].astype(np.float32)
            if qpos.shape[0] <= 1:
                continue
            delta = qpos[1:] - qpos[:-1]
            sum_delta += delta.sum(axis=0, dtype=np.float64)
            sum_sq_delta += np.square(delta, dtype=np.float64).sum(axis=0)
            min_delta = np.minimum(min_delta, delta.min(axis=0))
            max_delta = np.maximum(max_delta, delta.max(axis=0))
            count += delta.shape[0]
            num_episodes += 1

    if count == 0:
        raise RuntimeError("No deltas found. Check data root, setting, and tasks.")

    mean = sum_delta / count
    var = np.maximum(sum_sq_delta / count - mean * mean, 0.0)
    std = np.maximum(np.sqrt(var), args.std_floor)

    stats = {
        "setting": args.setting,
        "normalization": "raw_joint_delta_standardized",
        "delta_clip": args.delta_clip,
        "std_floor": args.std_floor,
        "num_episodes": num_episodes,
        "num_deltas": int(count),
        "delta_mean": mean.astype(np.float32).tolist(),
        "delta_std": std.astype(np.float32).tolist(),
        "delta_min": min_delta.astype(np.float32).tolist(),
        "delta_max": max_delta.astype(np.float32).tolist(),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {args.output}")
    print(f"episodes={num_episodes} deltas={count}")
    print("mean=", np.round(mean, 6).tolist())
    print("std=", np.round(std, 6).tolist())


if __name__ == "__main__":
    main()
