import json

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset


tf.config.set_visible_devices([], "GPU")


class VLABenchTFDSAct(IterableDataset):
    """Streams converted two-view VLABench RLDS episodes for VLANeXt."""

    def __init__(
        self,
        data_path,
        action_stats_path,
        history_len=8,
        future_len=8,
        full_sequence=True,
        input_modality="image",
        view_mode="multi",
        load_future_image=False,
        future_image_mode="horizon",
        buffer_size=1000,
        normalize_actions=True,
        max_episodes=None,
    ):
        super().__init__()
        self.data_path = str(data_path)
        self.history_len = int(history_len)
        self.future_len = int(future_len)
        self.full_sequence = bool(full_sequence)
        self.input_modality = str(input_modality)
        self.view_mode = str(view_mode)
        self.load_future_image = bool(load_future_image)
        self.future_image_mode = str(future_image_mode)
        self.buffer_size = int(buffer_size)
        self.normalize_actions = bool(normalize_actions)
        self.max_episodes = None if max_episodes is None else int(max_episodes)

        if self.input_modality != "image":
            raise ValueError("VLABench TFDS currently supports input_modality='image' only")
        if self.view_mode != "multi":
            raise ValueError("VLABench TFDS expects view_mode='multi'")

        with open(action_stats_path, "r", encoding="utf-8") as handle:
            action_stats = json.load(handle)["action"]
        self.action_low = np.asarray(action_stats["q01"], dtype=np.float32)
        self.action_high = np.asarray(action_stats["q99"], dtype=np.float32)

    def _normalize_actions(self, actions):
        actions = actions.astype(np.float32)
        if not self.normalize_actions:
            return actions
        scale = np.maximum(self.action_high - self.action_low, 1e-6)
        return np.clip(2.0 * (actions - self.action_low) / scale - 1.0, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _distributed_shard(dataset):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        return dataset.shard(
            num_shards=world_size * num_workers,
            index=rank * num_workers + worker_id,
        )

    def __iter__(self):
        builder = tfds.builder_from_directory(self.data_path)
        dataset = builder.as_dataset(split="train", shuffle_files=False)
        if self.max_episodes is not None:
            dataset = dataset.take(self.max_episodes)
        dataset = self._distributed_shard(dataset)

        shuffle_buffer = []
        for trajectory_id, trajectory in enumerate(dataset):
            try:
                steps = next(iter(trajectory["steps"].batch(4000)))
                actions = self._normalize_actions(steps["action"].numpy())
                observations = steps["observation"]
                proprio = observations["state"].numpy().astype(np.float32)
                main_images = observations["image"].numpy()
                wrist_images = observations["wrist_image"].numpy()
                language_values = steps["language_instruction"].numpy()
                instruction = next(
                    (value.decode("utf-8").strip() for value in language_values if value),
                    "",
                )
                trajectory_length = int(actions.shape[0])
                if trajectory_length == 0 or not instruction:
                    continue

                sample_indices = np.arange(trajectory_length)
                if not self.full_sequence:
                    count = max(1, trajectory_length // 75)
                    sample_indices = np.random.choice(trajectory_length, size=count, replace=False)

                for timestep in sample_indices:
                    timestep = int(timestep)
                    history_observation_indices = np.clip(
                        np.arange(timestep - self.history_len + 1, timestep + 1),
                        0,
                        trajectory_length - 1,
                    )
                    history_action_indices = np.arange(timestep - self.history_len, timestep)
                    future_action_indices = np.arange(timestep, timestep + self.future_len)

                    history_actions = np.zeros((self.history_len, 7), dtype=np.float32)
                    valid_history = history_action_indices >= 0
                    if np.any(valid_history):
                        history_actions[valid_history] = actions[history_action_indices[valid_history]]

                    future_actions = np.zeros((self.future_len, 7), dtype=np.float32)
                    valid_future = future_action_indices < trajectory_length
                    if np.any(valid_future):
                        future_actions[valid_future] = actions[future_action_indices[valid_future]]

                    denominator = max(trajectory_length - 1, 1)
                    future_timestep = min(timestep + self.future_len, trajectory_length - 1)
                    sample = {
                        "image": main_images[timestep].copy(),
                        "image_wrist": wrist_images[timestep].copy(),
                        "proprioception": torch.from_numpy(proprio[history_observation_indices].copy()),
                        "history_actions": torch.from_numpy(history_actions),
                        "future_actions": torch.from_numpy(future_actions),
                        "instruction": instruction,
                        "progress": torch.tensor(timestep / denominator, dtype=torch.float32),
                        "action_progress": torch.tensor(
                            (future_timestep - timestep) / denominator,
                            dtype=torch.float32,
                        ),
                        "goal_distance": torch.tensor(
                            np.log1p((trajectory_length - 1) - timestep) / np.log1p(denominator),
                            dtype=torch.float32,
                        ),
                    }

                    if self.load_future_image:
                        target = trajectory_length - 1 if self.future_image_mode == "last" else future_timestep
                        sample["future_image"] = main_images[target].copy()
                        sample["future_image_wrist"] = wrist_images[target].copy()
                        sample["future_images"] = [
                            sample["future_image"],
                            sample["future_image_wrist"],
                        ]

                    shuffle_buffer.append(sample)
                    if len(shuffle_buffer) >= self.buffer_size:
                        index = np.random.randint(len(shuffle_buffer))
                        shuffle_buffer[index], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[index]
                        yield shuffle_buffer.pop()
            except Exception as error:
                print(f"[Warn] Skipping VLABench TFDS trajectory {trajectory_id}: {error}")

        np.random.shuffle(shuffle_buffer)
        yield from shuffle_buffer
