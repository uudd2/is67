import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset


tf.config.set_visible_devices([], "GPU")


class RoboMimicTFDSAct(IterableDataset):
    """Mixes RoboMimic PH image tasks into VLANeXt two-view samples."""

    INSTRUCTIONS = {
        "lift_ph_image": "lift the cube",
        "can_ph_image": "pick up the can and place it in the target bin",
        "square_ph_image": "insert the square nut onto the square peg",
    }

    def __init__(
        self,
        data_root,
        task_configs,
        history_len=8,
        future_len=8,
        full_sequence=True,
        input_modality="image",
        view_mode="multi",
        buffer_size=1000,
    ):
        super().__init__()
        self.data_root = str(data_root)
        self.task_configs = list(task_configs)
        self.history_len = int(history_len)
        self.future_len = int(future_len)
        self.full_sequence = bool(full_sequence)
        self.input_modality = str(input_modality)
        self.view_mode = str(view_mode)
        self.buffer_size = int(buffer_size)
        if self.input_modality != "image" or self.view_mode != "multi":
            raise ValueError("RoboMimic training requires image input with multi-view mode")

    @staticmethod
    def _shard(dataset):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        return dataset.shard(world_size * num_workers, rank * num_workers + worker_id)

    @staticmethod
    def _quat_to_rotvec(quaternion):
        quaternion = quaternion.astype(np.float32)
        quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-6)
        quaternion = np.where(quaternion[:, 3:4] < 0, -quaternion, quaternion)
        xyz = quaternion[:, :3]
        sin_half = np.linalg.norm(xyz, axis=1, keepdims=True)
        angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[:, 3:4], 0.0, 1.0))
        axis = xyz / np.maximum(sin_half, 1e-6)
        return axis * angle

    def _task_iterators(self):
        iterators = []
        for config_name in self.task_configs:
            path = os.path.join(
                self.data_root,
                "robomimic_ph",
                config_name,
                "1.0.1",
            )
            builder = tfds.builder_from_directory(path)
            dataset = builder.as_dataset(split="train", shuffle_files=False)
            iterators.append([config_name, iter(self._shard(dataset))])
        return iterators

    def __iter__(self):
        active = self._task_iterators()
        shuffle_buffer = []
        trajectory_id = 0
        while active:
            next_active = []
            for config_name, iterator in active:
                try:
                    trajectory = next(iterator)
                except StopIteration:
                    continue
                next_active.append([config_name, iterator])
                try:
                    steps = next(iter(trajectory["steps"].batch(4000)))
                    observations = steps["observation"]
                    main_images = observations["agentview_image"].numpy()
                    wrist_images = observations["robot0_eye_in_hand_image"].numpy()
                    actions = np.clip(steps["action"].numpy(), -1.0, 1.0).astype(np.float32)

                    eef_position = observations["robot0_eef_pos"].numpy().astype(np.float32)
                    eef_rotation = self._quat_to_rotvec(
                        observations["robot0_eef_quat"].numpy()
                    )
                    gripper_qpos = observations["robot0_gripper_qpos"].numpy().astype(np.float32)
                    gripper = np.clip(
                        1.0 - np.mean(np.abs(gripper_qpos), axis=1, keepdims=True) / 0.04,
                        0.0,
                        1.0,
                    )
                    proprio = np.concatenate([eef_position, eef_rotation, gripper], axis=1)
                    trajectory_length = int(actions.shape[0])
                    instruction = self.INSTRUCTIONS[config_name]

                    sample_indices = np.arange(trajectory_length)
                    if not self.full_sequence:
                        count = max(1, trajectory_length // 75)
                        sample_indices = np.random.choice(
                            trajectory_length, size=count, replace=False
                        )

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

                        sample = {
                            "image": main_images[timestep].copy(),
                            "image_wrist": wrist_images[timestep].copy(),
                            "proprioception": torch.from_numpy(
                                proprio[history_observation_indices].copy()
                            ),
                            "history_actions": torch.from_numpy(history_actions),
                            "future_actions": torch.from_numpy(future_actions),
                            "instruction": instruction,
                        }
                        shuffle_buffer.append(sample)
                        if len(shuffle_buffer) >= self.buffer_size:
                            index = np.random.randint(len(shuffle_buffer))
                            shuffle_buffer[index], shuffle_buffer[-1] = (
                                shuffle_buffer[-1],
                                shuffle_buffer[index],
                            )
                            yield shuffle_buffer.pop()
                except Exception as error:
                    print(
                        f"[Warn] Skipping RoboMimic trajectory {trajectory_id} "
                        f"({config_name}): {error}"
                    )
                trajectory_id += 1
            active = next_active

        np.random.shuffle(shuffle_buffer)
        yield from shuffle_buffer
