#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import av
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


tf.config.set_visible_devices([], "GPU")


class SequentialVideoReader:
    """Reads episode ranges sequentially while reusing each LeRobot video file."""

    def __init__(self, root, video_key, fps):
        self.root = Path(root)
        self.video_key = video_key
        self.fps = float(fps)
        self.container = None
        self.decoder = None
        self.current_frame = 0
        self.file_identity = None

    def close(self):
        if self.container is not None:
            self.container.close()
            self.container = None
            self.decoder = None

    def _open(self, chunk_index, file_index):
        identity = (int(chunk_index), int(file_index))
        if identity == self.file_identity:
            return
        self.close()
        path = (
            self.root
            / "videos"
            / self.video_key
            / f"chunk-{identity[0]:03d}"
            / f"file-{identity[1]:03d}.mp4"
        )
        self.container = av.open(str(path))
        self.decoder = self.container.decode(video=0)
        self.current_frame = 0
        self.file_identity = identity

    def read_episode(self, metadata, length):
        prefix = f"videos/{self.video_key}"
        self._open(metadata[f"{prefix}/chunk_index"], metadata[f"{prefix}/file_index"])
        target_frame = int(round(float(metadata[f"{prefix}/from_timestamp"]) * self.fps))
        if target_frame < self.current_frame:
            raise RuntimeError(
                f"Non-monotonic video range for {self.video_key}: "
                f"target={target_frame}, current={self.current_frame}"
            )
        for _ in range(target_frame - self.current_frame):
            try:
                next(self.decoder)
            except StopIteration as error:
                raise RuntimeError(f"Failed to skip frame for {self.video_key}") from error
            self.current_frame += 1

        frames = []
        for _ in range(int(length)):
            try:
                frame = next(self.decoder)
            except StopIteration:
                raise RuntimeError(
                    f"Video ended early: key={self.video_key}, file={self.file_identity}, "
                    f"target={target_frame}, requested={length}"
                )
            frames.append(frame.to_ndarray(format="rgb24"))
            self.current_frame += 1
        return frames


class VLABenchRlds(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "VLABench main/wrist views converted from LeRobot v3."}

    def __init__(self, source_root, repo_id, max_episodes=None, **kwargs):
        self.source_root = str(source_root)
        self.repo_id = str(repo_id)
        self.max_episodes = None if max_episodes is None else int(max_episodes)
        super().__init__(**kwargs)

    def _info(self):
        step_features = tfds.features.FeaturesDict(
            {
                "observation": tfds.features.FeaturesDict(
                    {
                        "image": tfds.features.Image(shape=(224, 224, 3), encoding_format="jpeg"),
                        "wrist_image": tfds.features.Image(shape=(224, 224, 3), encoding_format="jpeg"),
                        "state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    }
                ),
                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                "language_instruction": tfds.features.Text(),
                "is_first": np.bool_,
                "is_last": np.bool_,
                "is_terminal": np.bool_,
            }
        )
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "episode_index": np.int64,
                    "steps": tfds.features.Dataset(step_features),
                }
            ),
            homepage="https://huggingface.co/datasets/lerobot/vlabench_unified",
            description="VLABench LeRobot dataset converted to two-view RLDS/TFDS.",
        )

    def _split_generators(self, dl_manager):
        del dl_manager
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            self.repo_id,
            root=self.source_root,
            download_videos=False,
            video_backend="pyav",
        )
        fps = float(dataset.meta.fps)
        main_reader = SequentialVideoReader(self.source_root, "observation.images.image", fps)
        wrist_reader = SequentialVideoReader(self.source_root, "observation.images.wrist_image", fps)
        episode_count = len(dataset.meta.episodes)
        if self.max_episodes is not None:
            episode_count = min(episode_count, self.max_episodes)

        try:
            for output_index in range(episode_count):
                metadata = dataset.meta.episodes[output_index]
                episode_index = int(metadata["episode_index"])
                start = int(metadata["dataset_from_index"])
                end = int(metadata["dataset_to_index"])
                length = end - start
                columns = dataset.hf_dataset[start:end]
                states = np.asarray(columns["observation.state"], dtype=np.float32)
                actions = np.asarray(columns["action"], dtype=np.float32)
                main_frames = main_reader.read_episode(metadata, length)
                wrist_frames = wrist_reader.read_episode(metadata, length)
                instruction = str(metadata["tasks"][0]).strip()

                def steps():
                    for frame_index in range(length):
                        yield {
                            "observation": {
                                "image": main_frames[frame_index],
                                "wrist_image": wrist_frames[frame_index],
                                "state": states[frame_index],
                            },
                            "action": actions[frame_index],
                            "language_instruction": instruction,
                            "is_first": frame_index == 0,
                            "is_last": frame_index == length - 1,
                            "is_terminal": frame_index == length - 1,
                        }

                if output_index % 100 == 0:
                    print(f"[convert] episode {output_index}/{episode_count}", flush=True)
                yield episode_index, {"episode_index": episode_index, "steps": steps()}
        finally:
            main_reader.close()
            wrist_reader.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/home/dm/datasets/vlabench_unified")
    parser.add_argument("--repo-id", default="lerobot/vlabench_unified")
    parser.add_argument(
        "--output-root",
        default="/media/dm/Elements/VLANeXt_migration/data/VLABench_modified",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-examples-per-shard", type=int, default=250)
    return parser.parse_args()


def main():
    args = parse_args()
    builder = VLABenchRlds(
        source_root=args.source_root,
        repo_id=args.repo_id,
        max_episodes=args.max_episodes,
        data_dir=args.output_root,
    )
    download_config = tfds.download.DownloadConfig(
        max_examples_per_split=None,
        override_max_simultaneous_downloads=1,
    )
    builder.download_and_prepare(download_config=download_config)
    print(f"TFDS written to: {builder.data_dir}")
    print(builder.info)


if __name__ == "__main__":
    main()
