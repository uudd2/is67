import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.libero_act import LiberoAct
from src.datasets.av_aloha_multitask_act import AVAlohaMultitaskAct
from src.models.edar_lite import EDARFeatureCache, FrozenDINOFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Cache strict t+8 DINO EDAR features.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset-name", default="libero_10_no_noops")
    parser.add_argument("--dataset-type", choices=("libero", "av_aloha"), default="libero")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--future-len", type=int, default=8)
    parser.add_argument("--action-stride", type=int, default=1)
    parser.add_argument("--main-camera", default="observation.images.zed_cam_left")
    parser.add_argument("--dino-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    extractor = FrozenDINOFeatureExtractor(args.dino_path).to(device)
    cache = EDARFeatureCache(args.cache_dir, extractor.metadata(), create=True)
    if args.dataset_type == "av_aloha":
        if not args.tasks:
            raise ValueError("--tasks is required for --dataset-type av_aloha")
        dataset = AVAlohaMultitaskAct(
            data_root=os.path.expanduser(args.data_path),
            tasks=args.tasks,
            history_len=1,
            future_len=args.future_len,
            action_stride=args.action_stride,
            full_sequence=True,
            input_modality="image",
            view_mode="single",
            load_future_image=True,
            future_image_mode="horizon",
            buffer_size=1,
            main_camera=args.main_camera,
            balance_tasks=False,
        )
    else:
        dataset = LiberoAct(
            data_path=os.path.expanduser(args.data_path),
            dataset_name=args.dataset_name,
            history_len=1,
            future_len=args.future_len,
            full_sequence=True,
            input_modality="image",
            view_mode="single",
            load_future_image=True,
            future_image_mode="horizon",
            strict_future_horizon=True,
            buffer_size=1,
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    storage_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    processed = 0
    progress = tqdm(loader, desc="Caching EDAR DINO features")
    for batch_index, batch in enumerate(progress):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        frame_ids = list(batch["frame_id"]) + list(batch["future_frame_id"])
        images = torch.cat([batch["image"], batch["future_image"]], dim=0).to(device)
        features = extractor(images).to(storage_dtype).cpu()
        for frame_id, frame_features in zip(frame_ids, features):
            path = cache.path_for(frame_id)
            if not path.exists():
                cache.put(frame_id, frame_features)
                processed += 1
        progress.set_postfix(cached=processed)
    print(f"Cached {processed} frames in {os.path.expanduser(args.cache_dir)}")


if __name__ == "__main__":
    main()
