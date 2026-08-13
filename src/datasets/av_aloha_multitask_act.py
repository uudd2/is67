import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset


class AVAlohaMultitaskAct(IterableDataset):
    """Streams a task-balanced set of disk-backed AV-ALOHA Zarr datasets."""

    def __init__(
        self,
        data_root,
        tasks,
        history_len=8,
        future_len=8,
        action_stride=1,
        full_sequence=True,
        input_modality="image",
        view_mode="multi",
        load_future_image=False,
        frame_ids_only=False,
        future_image_mode="horizon",
        buffer_size=1000,
        main_camera="observation.images.zed_cam_left",
        secondary_camera="observation.images.zed_cam_right",
        action_normalization="min_max",
        state_normalization="identity",
        normalization_clip=5.0,
        balance_tasks=True,
        max_episodes_per_task=None,
        shuffle_seed=2026,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.tasks = tuple(str(task) for task in tasks)
        self.history_len = int(history_len)
        self.future_len = int(future_len)
        self.action_stride = int(action_stride)
        self.full_sequence = bool(full_sequence)
        self.input_modality = str(input_modality)
        self.view_mode = str(view_mode)
        self.load_future_image = bool(load_future_image)
        self.frame_ids_only = bool(frame_ids_only)
        self.future_image_mode = str(future_image_mode)
        self.buffer_size = int(buffer_size)
        self.main_camera = str(main_camera)
        self.secondary_camera = str(secondary_camera)
        self.action_normalization = str(action_normalization).lower()
        self.state_normalization = str(state_normalization).lower()
        self.normalization_clip = float(normalization_clip)
        self.balance_tasks = bool(balance_tasks)
        self.max_episodes_per_task = (
            None if max_episodes_per_task is None else int(max_episodes_per_task)
        )
        self.shuffle_seed = int(shuffle_seed)

        if self.input_modality != "image" or self.view_mode not in {"single", "multi"}:
            raise ValueError(
                "AV-ALOHA V3 training requires image input with single or multi view."
            )
        if not self.tasks:
            raise ValueError("At least one AV-ALOHA task is required.")
        if self.buffer_size <= 0:
            raise ValueError("AV-ALOHA shuffle buffer must be positive.")
        if self.action_stride <= 0:
            raise ValueError("AV-ALOHA action_stride must be positive.")
        valid_normalizations = {"identity", "mean_std", "min_max"}
        if self.action_normalization not in valid_normalizations:
            raise ValueError(f"Invalid action normalization: {self.action_normalization}.")
        if self.state_normalization not in valid_normalizations:
            raise ValueError(f"Invalid state normalization: {self.state_normalization}.")

        self.task_info = []
        for task in self.tasks:
            path = self.data_root / task
            config_path = path / "config.json"
            if not config_path.is_file() or not (path / ".zgroup").is_file():
                raise FileNotFoundError(f"Incomplete AV-ALOHA Zarr dataset: {path}")
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            task_texts = config.get("tasks", {})
            instruction = str(task_texts.get("0", next(iter(task_texts.values()), "")))
            if not instruction:
                raise ValueError(f"AV-ALOHA task {task} has no language instruction.")
            if int(config["fps"]) != 25:
                raise ValueError(f"Expected AV-ALOHA at 25 FPS, got {config['fps']} for {task}.")
            self.task_info.append(
                {
                    "name": task,
                    "path": path,
                    "instruction": instruction,
                    "num_frames": int(config["num_frames"]),
                    "num_episodes": int(config["num_episodes"]),
                    "stats": config["stats"],
                }
            )

        self.stats = {
            "action": self._aggregate_stats("action"),
            "state": self._aggregate_stats("observation.state"),
        }
        action_dim = int(self.stats["action"]["mean"].shape[0])
        state_dim = int(self.stats["state"]["mean"].shape[0])
        if action_dim != 21 or state_dim != 21:
            raise ValueError(
                f"AV-ALOHA joint training expects 21D action/state, got {action_dim}/{state_dim}."
            )

    def _aggregate_stats(self, key):
        count = 0
        mean = None
        second_moment = None
        value_min = None
        value_max = None
        for info in self.task_info:
            stats = info["stats"][key]
            task_count = int(stats["count"][0])
            task_mean = np.asarray(stats["mean"], dtype=np.float64)
            task_std = np.asarray(stats["std"], dtype=np.float64)
            task_min = np.asarray(stats["min"], dtype=np.float64)
            task_max = np.asarray(stats["max"], dtype=np.float64)
            if mean is None:
                count = task_count
                mean = task_mean
                second_moment = np.square(task_std) * task_count
                value_min = task_min
                value_max = task_max
                continue
            total = count + task_count
            delta = task_mean - mean
            second_moment += (
                np.square(task_std) * task_count
                + np.square(delta) * count * task_count / total
            )
            mean += delta * task_count / total
            count = total
            value_min = np.minimum(value_min, task_min)
            value_max = np.maximum(value_max, task_max)
        return {
            "mean": mean.astype(np.float32),
            "std": np.maximum(
                np.sqrt(second_moment / count).astype(np.float32),
                np.float32(1e-6),
            ),
            "min": value_min.astype(np.float32),
            "max": value_max.astype(np.float32),
        }

    @staticmethod
    def _rank_worker_shard():
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

    def _normalize(self, values, name, mode):
        values = np.asarray(values, dtype=np.float32)
        if mode == "identity":
            return values
        stats = self.stats[name]
        if mode == "mean_std":
            normalized = (values - stats["mean"]) / stats["std"]
            return np.clip(
                normalized,
                -self.normalization_clip,
                self.normalization_clip,
            ).astype(np.float32)
        denominator = np.maximum(stats["max"] - stats["min"], np.float32(1e-6))
        normalized = 2.0 * (values - stats["min"]) / denominator - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _read_image(array, index):
        return np.asarray(array[int(index)], dtype=np.uint8).copy()

    def _episode_ids(self, info, shard_id, num_shards, rng):
        count = info["num_episodes"]
        if self.max_episodes_per_task is not None:
            count = min(count, self.max_episodes_per_task)
        episode_ids = np.arange(count, dtype=np.int64)
        rng.shuffle(episode_ids)
        return episode_ids[shard_id::num_shards]

    def _task_samples(self, info, group, shard_id, num_shards, rng):
        episode_ends = np.asarray(group["meta/episode_ends"][:], dtype=np.int64)
        action_array = group["data/action"]
        state_array = group["data/observation.state"]
        main_images = group[f"data/{self.main_camera}"]
        secondary_images = (
            group[f"data/{self.secondary_camera}"]
            if self.view_mode == "multi"
            else None
        )

        while True:
            episode_ids = self._episode_ids(info, shard_id, num_shards, rng)
            if episode_ids.size == 0:
                raise RuntimeError(
                    f"No episodes of {info['name']} assigned to data shard {shard_id}."
                )
            for episode_id in episode_ids:
                episode_id = int(episode_id)
                start = 0 if episode_id == 0 else int(episode_ends[episode_id - 1])
                end = int(episode_ends[episode_id])
                trajectory_length = end - start
                if trajectory_length <= 0:
                    continue

                actions = self._normalize(
                    action_array[start:end], "action", self.action_normalization
                )
                proprio = self._normalize(
                    state_array[start:end], "state", self.state_normalization
                )
                sample_indices = np.arange(trajectory_length, dtype=np.int64)
                if not self.full_sequence:
                    count = max(1, trajectory_length // 75)
                    sample_indices = rng.choice(
                        trajectory_length,
                        size=count,
                        replace=False,
                    )

                for timestep in sample_indices:
                    timestep = int(timestep)
                    history_observation_indices = np.clip(
                        np.arange(timestep - self.history_len + 1, timestep + 1),
                        0,
                        trajectory_length - 1,
                    )
                    history_action_indices = np.arange(
                        timestep - self.history_len,
                        timestep,
                    )
                    future_action_indices = np.arange(
                        timestep,
                        timestep + self.future_len * self.action_stride,
                        self.action_stride,
                    )

                    history_actions = np.zeros(
                        (self.history_len, actions.shape[1]), dtype=np.float32
                    )
                    valid_history = history_action_indices >= 0
                    if np.any(valid_history):
                        history_actions[valid_history] = actions[
                            history_action_indices[valid_history]
                        ]

                    future_actions = np.zeros(
                        (self.future_len, actions.shape[1]), dtype=np.float32
                    )
                    valid_future = future_action_indices < trajectory_length
                    if np.any(valid_future):
                        future_actions[valid_future] = actions[
                            future_action_indices[valid_future]
                        ]

                    global_timestep = start + timestep
                    current_main = None
                    current_secondary = None
                    if not self.frame_ids_only:
                        current_main = self._read_image(main_images, global_timestep)
                        current_secondary = (
                            self._read_image(secondary_images, global_timestep)
                            if secondary_images is not None
                            else None
                        )
                    denominator = max(trajectory_length - 1, 1)
                    future_timestep = min(
                        timestep + self.future_len * self.action_stride,
                        trajectory_length - 1,
                    )
                    sample = {
                        "proprioception": torch.from_numpy(
                            proprio[history_observation_indices].copy()
                        ),
                        "history_actions": torch.from_numpy(history_actions),
                        "future_actions": torch.from_numpy(future_actions),
                        "instruction": info["instruction"],
                        "task_name": info["name"],
                        "frame_id": f"avaloha_{info['name']}_{global_timestep}",
                        "future_frame_id": (
                            f"avaloha_{info['name']}_{start + future_timestep}"
                        ),
                        "progress": torch.tensor(
                            timestep / denominator, dtype=torch.float32
                        ),
                        "action_progress": torch.tensor(
                            (future_timestep - timestep) / denominator,
                            dtype=torch.float32,
                        ),
                        "goal_distance": torch.tensor(
                            np.log1p((trajectory_length - 1) - timestep)
                            / np.log1p(denominator),
                            dtype=torch.float32,
                        ),
                    }
                    if current_main is not None:
                        sample["image"] = current_main
                        sample["images"] = [current_main]
                        sample["anchor_image"] = current_main
                    if current_secondary is not None:
                        sample["image_wrist"] = current_secondary
                        sample["images"].append(current_secondary)
                        sample["anchor_image_wrist"] = current_secondary

                    if self.load_future_image and not self.frame_ids_only:
                        target_timestep = (
                            trajectory_length - 1
                            if self.future_image_mode == "last"
                            else future_timestep
                        )
                        global_target = start + target_timestep
                        future_main = self._read_image(main_images, global_target)
                        sample["future_image"] = future_main
                        sample["future_images"] = [future_main]
                        if secondary_images is not None:
                            future_secondary = self._read_image(
                                secondary_images, global_target
                            )
                            sample["future_image_wrist"] = future_secondary
                            sample["future_images"].append(future_secondary)
                    yield sample

    def __iter__(self):
        try:
            import zarr
        except ImportError as error:
            raise ImportError(
                "AV-ALOHA training requires zarr 2.x and numcodecs."
            ) from error

        shard_id, num_shards = self._rank_worker_shard()
        rng = np.random.default_rng(self.shuffle_seed + 1009 * shard_id)
        groups = [zarr.open_group(str(info["path"]), mode="r") for info in self.task_info]
        task_iterators = [
            self._task_samples(info, group, shard_id, num_shards, rng)
            for info, group in zip(self.task_info, groups)
        ]
        if self.balance_tasks:
            samples_per_task = int(
                np.ceil(max(info["num_frames"] for info in self.task_info) / num_shards)
            )
        else:
            samples_per_task = int(
                np.ceil(sum(info["num_frames"] for info in self.task_info) / num_shards)
            )

        shuffle_buffer = []
        if self.balance_tasks:
            for _ in range(samples_per_task):
                task_order = rng.permutation(len(task_iterators))
                for task_index in task_order:
                    shuffle_buffer.append(next(task_iterators[int(task_index)]))
                    if len(shuffle_buffer) >= self.buffer_size:
                        index = int(rng.integers(len(shuffle_buffer)))
                        shuffle_buffer[index], shuffle_buffer[-1] = (
                            shuffle_buffer[-1],
                            shuffle_buffer[index],
                        )
                        yield shuffle_buffer.pop()
        else:
            for iterator, info in zip(task_iterators, self.task_info):
                count = int(np.ceil(info["num_frames"] / num_shards))
                for _ in range(count):
                    shuffle_buffer.append(next(iterator))
                    if len(shuffle_buffer) >= self.buffer_size:
                        index = int(rng.integers(len(shuffle_buffer)))
                        shuffle_buffer[index], shuffle_buffer[-1] = (
                            shuffle_buffer[-1],
                            shuffle_buffer[index],
                        )
                        yield shuffle_buffer.pop()

        rng.shuffle(shuffle_buffer)
        yield from shuffle_buffer
