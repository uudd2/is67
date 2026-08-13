import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset


class CalvinAct(IterableDataset):
    """Streams local CALVIN LeRobot v2.1 episodes as VLANeXt samples."""

    def __init__(
        self,
        data_path,
        history_len=8,
        future_len=8,
        full_sequence=True,
        input_modality="image",
        view_mode="multi",
        load_future_image=False,
        future_image_mode="horizon",
        buffer_size=1000,
        main_camera="observation.images.top",
        wrist_camera="observation.images.wrist",
        state_indices=None,
        strip_task_prefix=True,
        clip_actions=True,
        max_episodes=None,
        shuffle_seed=2026,
    ):
        super().__init__()
        self.data_path = Path(data_path)
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
        self.state_indices = tuple(range(7) if state_indices is None else state_indices)
        self.strip_task_prefix = bool(strip_task_prefix)
        self.clip_actions = bool(clip_actions)
        self.max_episodes = None if max_episodes is None else int(max_episodes)
        self.shuffle_seed = int(shuffle_seed)

        if self.input_modality != "image":
            raise ValueError("CALVIN currently supports input_modality='image' only.")
        if self.view_mode != "multi":
            raise ValueError("CALVIN training expects view_mode='multi'.")
        if self.buffer_size <= 0:
            raise ValueError("CALVIN shuffle buffer must be positive.")
        if len(self.state_indices) != 7:
            raise ValueError(
                "CALVIN must provide exactly 7 state dimensions to the current action expert."
            )

        info_path = self.data_path / "meta" / "info.json"
        tasks_path = self.data_path / "meta" / "tasks.jsonl"
        if not info_path.is_file() or not tasks_path.is_file():
            raise FileNotFoundError(
                f"CALVIN LeRobot metadata is incomplete under {self.data_path}."
            )

        with info_path.open("r", encoding="utf-8") as handle:
            self.info = json.load(handle)
        if str(self.info.get("codebase_version", "")) != "v2.1":
            raise ValueError(
                "CalvinAct currently expects a LeRobot v2.1 dataset, got "
                f"{self.info.get('codebase_version')!r}."
            )
        self.total_episodes = int(self.info["total_episodes"])
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.data_path_template = str(
            self.info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            )
        )
        self.tasks = self._load_tasks(tasks_path)

    def _load_tasks(self, path):
        tasks = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                tasks[int(record["task_index"])] = self._instruction_text(record["task"])
        return tasks

    def _instruction_text(self, task):
        instruction = str(task).strip()
        if self.strip_task_prefix:
            _, separator, text = instruction.partition(":")
            if separator and text.strip():
                instruction = text.strip()
        return instruction

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
        episode_ids = np.arange(count, dtype=np.int64)
        np.random.default_rng(self.shuffle_seed).shuffle(episode_ids)
        shard_id, num_shards = self._rank_worker_shard()
        return episode_ids[shard_id::num_shards]

    def _episode_path(self, episode_id):
        relative_path = self.data_path_template.format(
            episode_chunk=int(episode_id) // self.chunks_size,
            episode_index=int(episode_id),
        )
        return self.data_path / relative_path

    @staticmethod
    def _decode_image(encoded):
        image_bytes = encoded.get("bytes") if isinstance(encoded, dict) else None
        if not image_bytes:
            raise ValueError("CALVIN image entry does not contain embedded bytes.")
        with Image.open(io.BytesIO(image_bytes)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    def _load_episode(self, episode_id):
        path = self._episode_path(episode_id)
        table = pq.read_table(
            path,
            columns=[
                self.main_camera,
                self.wrist_camera,
                "observation.state",
                "action",
                "task_index",
            ],
        )
        if table.num_rows == 0:
            raise ValueError(f"CALVIN episode {episode_id} is empty.")

        main_images = [self._decode_image(value) for value in table[self.main_camera].to_pylist()]
        wrist_images = [
            self._decode_image(value) for value in table[self.wrist_camera].to_pylist()
        ]
        full_state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        proprio = full_state[:, self.state_indices]
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(
                f"CALVIN episode {episode_id} has invalid action shape {actions.shape}."
            )
        if self.clip_actions:
            actions = np.clip(actions, -1.0, 1.0)

        task_index = int(table["task_index"][0].as_py())
        instruction = self.tasks.get(task_index, "")
        if not instruction:
            raise ValueError(
                f"CALVIN episode {episode_id} has no instruction for task {task_index}."
            )
        return main_images, wrist_images, proprio, actions, instruction

    def __iter__(self):
        shard_id, _ = self._rank_worker_shard()
        rng = np.random.default_rng(self.shuffle_seed + 1009 * shard_id)
        shuffle_buffer = []

        for episode_id in self._episode_ids():
            episode_id = int(episode_id)
            try:
                main_images, wrist_images, proprio, actions, instruction = self._load_episode(
                    episode_id
                )
                trajectory_length = int(actions.shape[0])
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
                        timestep + self.future_len,
                    )

                    history_actions = np.zeros((self.history_len, 7), dtype=np.float32)
                    valid_history = history_action_indices >= 0
                    if np.any(valid_history):
                        history_actions[valid_history] = actions[
                            history_action_indices[valid_history]
                        ]

                    future_actions = np.zeros((self.future_len, 7), dtype=np.float32)
                    valid_future = future_action_indices < trajectory_length
                    if np.any(valid_future):
                        future_actions[valid_future] = actions[
                            future_action_indices[valid_future]
                        ]

                    denominator = max(trajectory_length - 1, 1)
                    future_timestep = min(
                        timestep + self.future_len,
                        trajectory_length - 1,
                    )
                    sample = {
                        "image": main_images[timestep],
                        "image_wrist": wrist_images[timestep],
                        "anchor_image": main_images[timestep],
                        "anchor_image_wrist": wrist_images[timestep],
                        "proprioception": torch.from_numpy(
                            proprio[history_observation_indices].copy()
                        ),
                        "history_actions": torch.from_numpy(history_actions),
                        "future_actions": torch.from_numpy(future_actions),
                        "instruction": instruction,
                        "progress": torch.tensor(
                            timestep / denominator,
                            dtype=torch.float32,
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

                    if self.load_future_image:
                        target = (
                            trajectory_length - 1
                            if self.future_image_mode == "last"
                            else future_timestep
                        )
                        sample["future_image"] = main_images[target]
                        sample["future_image_wrist"] = wrist_images[target]
                        sample["future_images"] = [
                            main_images[target],
                            wrist_images[target],
                        ]

                    shuffle_buffer.append(sample)
                    if len(shuffle_buffer) >= self.buffer_size:
                        index = int(rng.integers(len(shuffle_buffer)))
                        shuffle_buffer[index], shuffle_buffer[-1] = (
                            shuffle_buffer[-1],
                            shuffle_buffer[index],
                        )
                        yield shuffle_buffer.pop()
            except Exception as error:
                print(f"[Warn] Skipping CALVIN episode {episode_id}: {error}")

        rng.shuffle(shuffle_buffer)
        yield from shuffle_buffer
