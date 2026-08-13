import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset


tf.config.set_visible_devices([], "GPU")


class BridgeAct(IterableDataset):
    """Streams BridgeData V2 RLDS episodes as VLANeXt training samples."""

    def __init__(
        self,
        data_path,
        history_len=8,
        future_len=8,
        full_sequence=True,
        input_modality="image",
        view_mode="single",
        load_future_image=False,
        future_image_mode="horizon",
        buffer_size=1000,
        main_camera="image_0",
        wrist_camera="image_1",
        action_stats_path=None,
        length=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.history_len = int(history_len)
        self.future_len = int(future_len)
        self.full_sequence = bool(full_sequence)
        self.input_modality = str(input_modality)
        self.view_mode = str(view_mode)
        self.load_future_image = bool(load_future_image)
        self.future_image_mode = str(future_image_mode)
        self.buffer_size = int(buffer_size)
        self.main_camera = str(main_camera)
        self.wrist_camera = str(wrist_camera)
        self.length = length

        if self.input_modality not in {"image", "video"}:
            raise ValueError(f"Unsupported Bridge input modality: {self.input_modality}")
        if self.main_camera not in {f"image_{index}" for index in range(4)}:
            raise ValueError(f"Unsupported Bridge main camera: {self.main_camera}")
        if self.wrist_camera not in {f"image_{index}" for index in range(4)}:
            raise ValueError(f"Unsupported Bridge wrist camera: {self.wrist_camera}")

        if action_stats_path is None:
            candidates = sorted(
                name
                for name in os.listdir(self.data_path)
                if name.startswith("action_proprio_stats_") and name.endswith(".json")
            )
            if not candidates:
                raise FileNotFoundError(f"No Bridge action statistics found in {self.data_path}")
            # This official stats variant removes rotation wraparound outliers.
            preferred = next((name for name in candidates if name.startswith("action_proprio_stats_9cca")), None)
            action_stats_path = os.path.join(self.data_path, preferred or candidates[0])

        with open(action_stats_path, "r", encoding="utf-8") as handle:
            stats = json.load(handle)["action"]
        self.action_low = np.asarray(stats["min"][:6], dtype=np.float32)
        self.action_high = np.asarray(stats["max"][:6], dtype=np.float32)
        self.action_stats_path = action_stats_path

    def _normalize_actions(self, raw_actions):
        denominator = np.maximum(self.action_high - self.action_low, 1e-6)
        motion = 2.0 * (raw_actions[:, :6] - self.action_low) / denominator - 1.0
        motion = np.clip(motion, -1.0, 1.0)
        gripper = np.clip(2.0 * raw_actions[:, 6:7] - 1.0, -1.0, 1.0)
        return np.concatenate([motion, gripper], axis=1).astype(np.float32)

    @staticmethod
    def _distributed_shard(dataset):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        return dataset.shard(
            num_shards=world_size * num_workers,
            index=rank * num_workers + worker_id,
        )

    @staticmethod
    def _decode_instruction(language_values):
        for value in language_values:
            instruction = value.decode("utf-8").strip()
            if instruction:
                return instruction
        return ""

    def __iter__(self):
        builder = tfds.builder_from_directory(builder_dir=self.data_path)
        read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
        decoded_cameras = {self.main_camera}
        if self.view_mode == "multi":
            decoded_cameras.add(self.wrist_camera)
        image_decoders = {
            f"image_{index}": tfds.decode.SkipDecoding()
            for index in range(4)
            if f"image_{index}" not in decoded_cameras
        }
        dataset = builder.as_dataset(
            split="train",
            shuffle_files=False,
            read_config=read_config,
            decoders={"steps": {"observation": image_decoders}},
        )
        if self.length is not None:
            dataset = dataset.take(int(self.length))
        dataset = self._distributed_shard(dataset)

        shuffle_buffer = []
        for trajectory_id, trajectory in enumerate(dataset):
            try:
                def select_training_fields(step):
                    selected = {
                        "action": step["action"],
                        "language_instruction": step["language_instruction"],
                        "state": step["observation"]["state"],
                        "main_image": step["observation"][self.main_camera],
                    }
                    if self.view_mode == "multi":
                        selected["wrist_image"] = step["observation"][self.wrist_camera]
                    return selected

                selected_steps = trajectory["steps"].map(
                    select_training_fields,
                    num_parallel_calls=1,
                    deterministic=True,
                )
                steps = next(iter(selected_steps.batch(2000)))
                trajectory_length = int(steps["action"].shape[0])
                if trajectory_length == 0:
                    continue

                main_images = steps["main_image"].numpy()
                wrist_images = (
                    steps["wrist_image"].numpy()
                    if self.view_mode == "multi"
                    else main_images
                )
                proprio = steps["state"].numpy().astype(np.float32)
                actions = self._normalize_actions(steps["action"].numpy().astype(np.float32))
                language_values = steps["language_instruction"].numpy()
                instruction = self._decode_instruction(language_values)
                if not instruction:
                    continue

                if self.full_sequence:
                    sample_indices = np.arange(trajectory_length)
                else:
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
                        history_actions[valid_history] = actions[
                            np.clip(history_action_indices[valid_history], 0, trajectory_length - 1)
                        ]

                    future_actions = np.zeros((self.future_len, 7), dtype=np.float32)
                    valid_future = future_action_indices < trajectory_length
                    if np.any(valid_future):
                        future_actions[valid_future] = actions[future_action_indices[valid_future]]

                    denominator = max(trajectory_length - 1, 1)
                    future_timestep = min(timestep + self.future_len, trajectory_length - 1)
                    sample = {
                        "proprioception": torch.from_numpy(proprio[history_observation_indices]),
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
                        sample["future_image"] = main_images[target]
                        if self.view_mode == "multi":
                            sample["future_image_wrist"] = wrist_images[target]
                            sample["future_images"] = [main_images[target], wrist_images[target]]

                    if self.input_modality == "video":
                        sample["video"] = main_images[history_observation_indices]
                        sample["anchor_video"] = sample["video"]
                        if self.view_mode == "multi":
                            sample["video_wrist"] = wrist_images[history_observation_indices]
                            sample["anchor_video_wrist"] = sample["video_wrist"]
                    else:
                        sample["image"] = main_images[timestep]
                        sample["anchor_image"] = sample["image"]
                        sample["goal_image"] = main_images[-1]
                        if self.view_mode == "multi":
                            sample["image_wrist"] = wrist_images[timestep]
                            sample["anchor_image_wrist"] = sample["image_wrist"]
                            sample["goal_image_wrist"] = wrist_images[-1]
                            sample["goal_images"] = [sample["goal_image"], sample["goal_image_wrist"]]

                    shuffle_buffer.append(sample)
                    if len(shuffle_buffer) >= self.buffer_size:
                        index = np.random.randint(len(shuffle_buffer))
                        shuffle_buffer[index], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[index]
                        yield shuffle_buffer.pop()

            except Exception as error:
                print(f"[Warn] Skipping Bridge trajectory {trajectory_id} due to error: {error}")

        np.random.shuffle(shuffle_buffer)
        yield from shuffle_buffer
