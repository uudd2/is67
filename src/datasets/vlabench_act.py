import json
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset


class VLABenchAct(IterableDataset):
    """Streams local VLABench LeRobot episodes as VLANeXt training samples."""

    def __init__(
        self,
        data_path,
        repo_id="lerobot/vlabench_unified",
        history_len=8,
        future_len=8,
        full_sequence=True,
        input_modality="image",
        view_mode="multi",
        load_future_image=False,
        future_image_mode="horizon",
        buffer_size=1000,
        main_camera="observation.images.image",
        wrist_camera="observation.images.wrist_image",
        normalize_actions=True,
        max_episodes=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.repo_id = repo_id
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
        self.normalize_actions = bool(normalize_actions)
        self.max_episodes = None if max_episodes is None else int(max_episodes)

        if self.input_modality != "image":
            raise ValueError("VLABench currently supports input_modality='image' only")
        if self.view_mode != "multi":
            raise ValueError("This VLABench adapter expects view_mode='multi'")

        info_path = os.path.join(self.data_path, "meta", "info.json")
        with open(info_path, "r", encoding="utf-8") as handle:
            info = json.load(handle)
        self.total_episodes = int(info["total_episodes"])

    @staticmethod
    def _tensor_image_to_uint8(image):
        image = image.detach().cpu()
        if image.ndim != 3:
            raise ValueError(f"Expected CHW image tensor, got {tuple(image.shape)}")
        image = image.permute(1, 2, 0).numpy()
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

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

    def _episode_ids(self):
        count = self.total_episodes
        if self.max_episodes is not None:
            count = min(count, self.max_episodes)
        shard_id, num_shards = self._rank_worker_shard()
        return list(range(shard_id, count, num_shards))

    @staticmethod
    def _action_bounds(stats):
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        return low, high

    def _normalize(self, actions, low, high):
        if not self.normalize_actions:
            return actions.astype(np.float32)
        scale = np.maximum(high - low, 1e-6)
        return np.clip(2.0 * (actions - low) / scale - 1.0, -1.0, 1.0).astype(np.float32)

    def __iter__(self):
        # Lazy import keeps the regular VLANeXt environment usable for non-VLABench runs.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        episode_ids = self._episode_ids()
        if not episode_ids:
            return
        dataset = LeRobotDataset(
            self.repo_id,
            root=self.data_path,
            episodes=episode_ids,
            download_videos=False,
            video_backend="pyav",
        )
        action_low, action_high = self._action_bounds(dataset.meta.stats["action"])

        episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
        shuffle_buffer = []
        start = 0
        while start < len(dataset):
            episode_id = int(episode_indices[start])
            end = start + 1
            while end < len(dataset) and int(episode_indices[end]) == episode_id:
                end += 1

            try:
                columns = dataset.hf_dataset[start:end]
                proprio = np.asarray(columns["observation.state"], dtype=np.float32)
                actions = self._normalize(
                    np.asarray(columns["action"], dtype=np.float32),
                    action_low,
                    action_high,
                )
                trajectory_length = end - start
                if trajectory_length == 0:
                    start = end
                    continue

                first_frame = dataset[start]
                instruction = str(first_frame["task"]).strip()
                if not instruction:
                    start = end
                    continue

                sample_indices = np.arange(trajectory_length)
                if not self.full_sequence:
                    count = max(1, trajectory_length // 75)
                    sample_indices = np.random.choice(trajectory_length, size=count, replace=False)

                for timestep in sample_indices:
                    timestep = int(timestep)
                    frame = first_frame if timestep == 0 else dataset[start + timestep]
                    main_image = self._tensor_image_to_uint8(frame[self.main_camera])
                    wrist_image = self._tensor_image_to_uint8(frame[self.wrist_camera])

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
                        "image": main_image,
                        "image_wrist": wrist_image,
                        "anchor_image": main_image,
                        "anchor_image_wrist": wrist_image,
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
                        future_frame = dataset[start + target]
                        future_main = self._tensor_image_to_uint8(future_frame[self.main_camera])
                        future_wrist = self._tensor_image_to_uint8(future_frame[self.wrist_camera])
                        sample["future_image"] = future_main
                        sample["future_image_wrist"] = future_wrist
                        sample["future_images"] = [future_main, future_wrist]

                    shuffle_buffer.append(sample)
                    if len(shuffle_buffer) >= self.buffer_size:
                        index = np.random.randint(len(shuffle_buffer))
                        shuffle_buffer[index], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[index]
                        yield shuffle_buffer.pop()

            except Exception as error:
                print(f"[Warn] Skipping VLABench episode {episode_id} due to error: {error}")
            start = end

        np.random.shuffle(shuffle_buffer)
        yield from shuffle_buffer
