"""Episode-safe top-view LeRobot v3 adapter for Piper real-data Stage-A."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .stage_a_episode_utils import (
    ActionStatsAccumulator,
    build_stage_a_indices,
    normalize_actions,
    split_episode_ids_by_batch,
    valid_stage_a_starts,
)


def load_batch_episode_ids(batch_path: str | Path) -> list[int]:
    info_path = Path(batch_path) / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    total_episodes = int(info["total_episodes"])
    if total_episodes < 0:
        raise ValueError(f"Invalid total_episodes in {info_path}: {total_episodes}")
    return list(range(total_episodes))


def make_episode_split_manifest(
    batches: Sequence[Mapping[str, object]],
    val_ratio: float,
    seed: int,
) -> dict[str, object]:
    episodes_by_batch: dict[str, list[int]] = {}
    batch_metadata: dict[str, dict[str, str]] = {}
    for batch in batches:
        name = str(batch["name"])
        if name in episodes_by_batch:
            raise ValueError(f"Duplicate batch name: {name}")
        path = str(batch["path"])
        repo_id = str(batch["repo_id"])
        episodes_by_batch[name] = load_batch_episode_ids(path)
        batch_metadata[name] = {"path": path, "repo_id": repo_id}
    train, val = split_episode_ids_by_batch(
        episodes_by_batch, val_ratio=val_ratio, seed=seed
    )
    return {
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "batches": batch_metadata,
        "train": train,
        "val": val,
    }


def _lerobot_dataset(batch: Mapping[str, object], episode_ids: Sequence[int], video_backend: str):
    # Lazy import keeps environments that do not train on LeRobot usable.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        str(batch["repo_id"]),
        root=str(batch["path"]),
        episodes=[int(value) for value in episode_ids],
        download_videos=False,
        video_backend=str(video_backend),
    )


def compute_action_stats(
    batches: Sequence[Mapping[str, object]],
    episode_ids_by_batch: Mapping[str, Sequence[int]],
    action_dim: int,
    video_backend: str = "pyav",
) -> dict[str, np.ndarray | int]:
    """Compute normalization statistics from the selected episodes only."""
    accumulator = ActionStatsAccumulator(action_dim=action_dim)
    by_name = {str(batch["name"]): batch for batch in batches}
    for batch_name, episode_ids in episode_ids_by_batch.items():
        episode_ids = [int(value) for value in episode_ids]
        if not episode_ids:
            continue
        if batch_name not in by_name:
            raise KeyError(f"Unknown batch in split manifest: {batch_name}")
        dataset = _lerobot_dataset(by_name[batch_name], episode_ids, video_backend)
        actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != int(action_dim):
            raise ValueError(
                f"Expected {action_dim}D actions in {batch_name}, got {tuple(actions.shape)}"
            )
        accumulator.update(actions)
    return accumulator.finalize()


def _tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got {tuple(image.shape)}")
    if image.shape[0] != 3:
        raise ValueError(f"Expected RGB image tensor, got {tuple(image.shape)}")
    array = image.permute(1, 2, 0).numpy()
    if array.dtype == np.uint8:
        return array
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


class PiperLeRobotStageA(IterableDataset):
    """Streams complete top-view Stage-A transitions without crossing episode boundaries."""

    def __init__(
        self,
        batches: Sequence[Mapping[str, object]],
        episode_ids_by_batch: Mapping[str, Sequence[int]],
        action_stats: Mapping[str, np.ndarray],
        main_camera: str,
        action_dim: int = 14,
        action_horizon: int = 8,
        action_stride: int = 1,
        action_normalization: str = "mean_std",
        video_backend: str = "pyav",
        buffer_size: int = 256,
    ):
        super().__init__()
        self.batches = [dict(batch) for batch in batches]
        self.batch_by_name = {str(batch["name"]): batch for batch in self.batches}
        if len(self.batch_by_name) != len(self.batches):
            raise ValueError("Batch names must be unique")
        self.episode_ids_by_batch = {
            str(name): [int(value) for value in ids]
            for name, ids in episode_ids_by_batch.items()
        }
        unknown = set(self.episode_ids_by_batch) - set(self.batch_by_name)
        if unknown:
            raise KeyError(f"Unknown batches in episode split: {sorted(unknown)}")
        self.action_stats = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in action_stats.items()
            if key != "count"
        }
        self.main_camera = str(main_camera)
        if not self.main_camera:
            raise ValueError("main_camera must be configured explicitly")
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.action_stride = int(action_stride)
        self.action_normalization = str(action_normalization)
        self.video_backend = str(video_backend)
        self.buffer_size = int(buffer_size)
        if self.action_dim != 14:
            raise ValueError(f"Piper real Stage-A expects 14D actions, got {self.action_dim}")
        if self.buffer_size < 1:
            raise ValueError("buffer_size must be positive")

    @staticmethod
    def _rank_worker_shard() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        return rank * num_workers + worker_id, world_size * num_workers

    def _assigned_episode_ids(self) -> dict[str, list[int]]:
        shard_id, num_shards = self._rank_worker_shard()
        flat: list[tuple[str, int]] = []
        for batch_name in sorted(self.episode_ids_by_batch):
            flat.extend(
                (batch_name, episode_id)
                for episode_id in self.episode_ids_by_batch[batch_name]
            )
        assigned = flat[shard_id::num_shards]
        result = {name: [] for name in self.episode_ids_by_batch}
        for batch_name, episode_id in assigned:
            result[batch_name].append(episode_id)
        return result

    def _iter_batch(self, batch_name: str, episode_ids: Sequence[int]):
        if not episode_ids:
            return
        dataset = _lerobot_dataset(
            self.batch_by_name[batch_name], episode_ids, self.video_backend
        )
        episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
        start = 0
        while start < len(dataset):
            episode_id = int(episode_indices[start])
            end = start + 1
            while end < len(dataset) and int(episode_indices[end]) == episode_id:
                end += 1
            columns = dataset.hf_dataset[start:end]
            actions = np.asarray(columns["action"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != self.action_dim:
                raise ValueError(
                    f"Expected actions [T,{self.action_dim}] in {batch_name}/{episode_id}, "
                    f"got {tuple(actions.shape)}"
                )
            actions = normalize_actions(
                actions, self.action_stats, self.action_normalization
            )
            episode_length = end - start
            for local_t in valid_stage_a_starts(
                episode_length, self.action_horizon, self.action_stride
            ):
                local_t = int(local_t)
                action_indices, future_t = build_stage_a_indices(
                    local_t, self.action_horizon, self.action_stride
                )
                current_frame = dataset[start + local_t]
                future_frame = dataset[start + future_t]
                current_image = _tensor_image_to_uint8(
                    current_frame[self.main_camera]
                )
                future_image = _tensor_image_to_uint8(
                    future_frame[self.main_camera]
                )
                namespaced_episode = f"{batch_name}/{episode_id}"
                yield {
                    "image": current_image,
                    "future_image": future_image,
                    "future_actions": torch.from_numpy(actions[action_indices]),
                    "frame_id": f"{namespaced_episode}/{local_t}",
                    "future_frame_id": f"{namespaced_episode}/{future_t}",
                    "episode_id": namespaced_episode,
                    "batch_name": batch_name,
                }
            start = end

    def __iter__(self):
        assigned = self._assigned_episode_ids()
        rng = np.random.default_rng(torch.initial_seed() % (2**63 - 1))
        buffer = []
        for batch_name in sorted(assigned):
            for sample in self._iter_batch(batch_name, assigned[batch_name]):
                buffer.append(sample)
                if len(buffer) >= self.buffer_size:
                    index = int(rng.integers(len(buffer)))
                    buffer[index], buffer[-1] = buffer[-1], buffer[index]
                    yield buffer.pop()
        rng.shuffle(buffer)
        yield from buffer
