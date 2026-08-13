import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image


def _to_uint8(array):
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def _save_png(array, path):
    Image.fromarray(_to_uint8(array)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="/media/dm/Elements/VLANeXt_migration/data/LIBERO_modified/libero_10_no_noops/1.0.0",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--success-index", type=int, default=0, help="Index among successful trajectories.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--main-key", default="image")
    parser.add_argument("--wrist-key", default="wrist_image")
    args = parser.parse_args()

    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    os.makedirs(args.output_dir, exist_ok=True)

    builder = tfds.builder_from_directory(builder_dir=args.data_path)
    read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
    ds = builder.as_dataset(split="train", shuffle_files=False, read_config=read_config)

    selected = None
    success_count = 0
    for traj_id, traj_data in enumerate(ds):
        traj_batch = next(iter(traj_data["steps"].batch(5000)))
        reward = traj_batch["reward"][-1].numpy()
        if reward != 1:
            continue
        if success_count == args.success_index:
            selected = (traj_id, traj_batch)
            break
        success_count += 1

    if selected is None:
        raise RuntimeError(f"No successful trajectory found at success-index={args.success_index}")

    traj_id, traj_batch = selected
    obs = traj_batch["observation"]
    images = _to_uint8(obs[args.main_key].numpy())
    if args.wrist_key in obs:
        wrist_images = _to_uint8(obs[args.wrist_key].numpy())
    else:
        wrist_images = images
    traj_len = int(images.shape[0])
    instruction = traj_batch["language_instruction"][0].numpy().decode("utf-8")

    frame_indices = []
    for i in range(args.num_frames):
        frame_idx = int(args.start_frame + i * args.stride)
        if frame_idx >= traj_len:
            break
        frame_indices.append(frame_idx)

    if not frame_indices:
        raise ValueError(
            f"No frames selected: start={args.start_frame}, stride={args.stride}, traj_len={traj_len}"
        )

    metadata = {
        "data_path": args.data_path,
        "traj_id": int(traj_id),
        "success_index": int(args.success_index),
        "traj_len": traj_len,
        "instruction": instruction,
        "start_frame": int(args.start_frame),
        "num_frames_requested": int(args.num_frames),
        "stride": int(args.stride),
        "frame_indices": frame_indices,
        "main_key": args.main_key,
        "wrist_key": args.wrist_key if args.wrist_key in obs else args.main_key,
    }

    with open(os.path.join(args.output_dir, "instruction.txt"), "w") as f:
        f.write(instruction + "\n")
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    for frame_idx in frame_indices:
        frame_dir = os.path.join(args.output_dir, f"frame_{frame_idx}")
        os.makedirs(frame_dir, exist_ok=True)
        _save_png(images[frame_idx], os.path.join(frame_dir, "sample_view0.png"))
        _save_png(wrist_images[frame_idx], os.path.join(frame_dir, "sample_view1.png"))

    print(f"saved {len(frame_indices)} frames to {args.output_dir}")
    print(f"traj_id={traj_id}, success_index={args.success_index}, traj_len={traj_len}")
    print(f"instruction={instruction!r}")


if __name__ == "__main__":
    main()
