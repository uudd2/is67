import io
import json
import os
from glob import glob

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

try:
    import cv2
except ImportError:
    cv2 = None


ROBOTWIN_ALOHA_ACTION_MIN = np.array(
    [
        -7.340834140777588,
        -0.0003164021181873977,
        -0.1190902441740036,
        -1.9268131256103516,
        -1.4216828346252441,
        -6.232340335845947,
        0.0,
        -6.316576957702637,
        -0.5742834806442261,
        -0.005311839282512665,
        -1.9893786907196045,
        -2.1285502910614014,
        -6.269529342651367,
        0.0,
    ],
    dtype=np.float32,
)

ROBOTWIN_ALOHA_ACTION_MAX = np.array(
    [
        5.681515216827393,
        3.8880207538604736,
        4.500889301300049,
        1.789766788482666,
        1.5632697343826294,
        4.389739513397217,
        1.0,
        1.3892723321914673,
        3.242604970932007,
        3.6588551998138428,
        1.9530924558639526,
        1.397123098373413,
        3.503765106201172,
        1.0,
    ],
    dtype=np.float32,
)


def _resolve_path(path):
    if path is None or path == "":
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(os.getcwd(), path)


def load_robotwin_delta_stats(path):
    resolved = _resolve_path(path)
    if resolved is None:
        return None
    with open(resolved, "r") as f:
        stats = json.load(f)
    mean = np.asarray(stats["delta_mean"], dtype=np.float32)
    std = np.asarray(stats["delta_std"], dtype=np.float32)
    clip = float(stats.get("delta_clip", 5.0))
    std = np.maximum(std, 1e-6)
    return {"mean": mean, "std": std, "clip": clip, "path": resolved}


def normalize_robotwin_delta(delta, stats):
    if stats is None:
        return delta.astype(np.float32)
    x = (delta - stats["mean"]) / stats["std"]
    x = np.clip(x, -stats["clip"], stats["clip"]) / stats["clip"]
    return x.astype(np.float32)


def denormalize_robotwin_delta(delta_norm, stats):
    if stats is None:
        return delta_norm.astype(np.float32)
    x = np.asarray(delta_norm, dtype=np.float32) * stats["clip"]
    return (x * stats["std"] + stats["mean"]).astype(np.float32)


class RoboTwinAct(IterableDataset):
    def __init__(
        self,
        data_root,
        setting="aloha-agilex_clean_50",
        tasks=None,
        max_episodes_per_task=None,
        history_len=8,
        future_len=8,
        full_sequence=True,
        input_modality="image",
        view_mode="multi",
        load_future_image=False,
        future_image_mode="horizon",
        buffer_size=10000,
        main_camera="head_camera",
        wrist_camera="right_camera",
        cameras=None,
        normalize_actions=True,
        action_mode="absolute",
        delta_stats_path=None,
        anchor_refresh_interval=1,
        latent_bridge_sequence_len=1,
    ):
        super().__init__()
        self.data_root = data_root
        self.setting = setting
        self.tasks = tasks
        self.max_episodes_per_task = max_episodes_per_task
        self.history_len = history_len
        self.future_len = future_len
        self.full_sequence = full_sequence
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.load_future_image = load_future_image
        self.future_image_mode = future_image_mode
        self.buffer_size = buffer_size
        self.main_camera = main_camera
        self.wrist_camera = wrist_camera
        if cameras is None:
            cameras = [main_camera] if view_mode != "multi" else [main_camera, wrist_camera]
        self.cameras = list(cameras)
        self.normalize_actions = normalize_actions
        self.anchor_refresh_interval = max(1, int(anchor_refresh_interval))
        self.latent_bridge_sequence_len = max(1, int(latent_bridge_sequence_len))
        if action_mode not in {"absolute", "joint_delta"}:
            raise ValueError(f"Unknown RoboTwin action_mode: {action_mode}")
        self.action_mode = action_mode
        self.delta_stats = load_robotwin_delta_stats(delta_stats_path)

        self.episodes = self._discover_episodes()
        if not self.episodes:
            raise FileNotFoundError(
                f"No RoboTwin episodes found under {data_root} with setting {setting}"
            )

    def _discover_episodes(self):
        task_names = self.tasks
        if task_names is None:
            task_names = [
                name
                for name in sorted(os.listdir(self.data_root))
                if os.path.isdir(os.path.join(self.data_root, name, self.setting))
            ]

        episodes = []
        for task in task_names:
            task_dir = os.path.join(self.data_root, task, self.setting)
            data_dir = os.path.join(task_dir, "data")
            instruction_dir = os.path.join(task_dir, "instructions")
            paths = sorted(
                glob(os.path.join(data_dir, "episode*.hdf5")),
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0].replace("episode", "")),
            )
            if self.max_episodes_per_task is not None:
                paths = paths[: int(self.max_episodes_per_task)]
            for path in paths:
                episode_id = int(os.path.splitext(os.path.basename(path))[0].replace("episode", ""))
                episodes.append(
                    {
                        "task": task,
                        "path": path,
                        "instruction_path": os.path.join(instruction_dir, f"episode{episode_id}.json"),
                        "episode_id": episode_id,
                    }
                )
        return episodes

    @staticmethod
    def _decode_rgb(encoded):
        if isinstance(encoded, np.ndarray):
            encoded = encoded.tobytes()
        elif isinstance(encoded, np.bytes_):
            encoded = bytes(encoded)
        # RoboTwin HDF5 files are JPEG-encoded with cv2.imencode from arrays
        # named "rgb". Decoding with PIL swaps red/blue relative to the raw
        # RoboTwin observation/video stream, while cv2.imdecode recovers the
        # original numeric channel order used by the simulator.
        if cv2 is not None:
            arr = np.frombuffer(encoded, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is not None:
                return image.astype(np.uint8, copy=False)
        image = Image.open(io.BytesIO(encoded)).convert("RGB")
        return np.asarray(image, dtype=np.uint8)[..., ::-1].copy()

    @staticmethod
    def _load_instruction(path, episode_id):
        if not os.path.exists(path):
            return "Complete the robot manipulation task."
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, str):
            return data
        if isinstance(data, list) and data:
            return str(data[episode_id % len(data)])
        if isinstance(data, dict):
            for key in ("seen", "instructions", "instruction", "language_instruction"):
                value = data.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, list) and value:
                    return str(value[episode_id % len(value)])
        return "Complete the robot manipulation task."

    def _normalize_action(self, actions):
        if not self.normalize_actions:
            return actions.astype(np.float32)
        denom = ROBOTWIN_ALOHA_ACTION_MAX - ROBOTWIN_ALOHA_ACTION_MIN
        denom = np.where(denom == 0, 1.0, denom)
        actions = 2.0 * (actions - ROBOTWIN_ALOHA_ACTION_MIN) / denom - 1.0
        return np.clip(actions, -1.0, 1.0).astype(np.float32)

    def _iter_episode_samples(self, item):
        with h5py.File(item["path"], "r") as f:
            actions_raw = f["joint_action/vector"][()].astype(np.float32)
            encoded_by_camera = {}
            fallback_key = f"observation/{self.main_camera}/rgb"
            fallback_encoded = f[fallback_key][()]
            for camera in self.cameras:
                key = f"observation/{camera}/rgb"
                encoded_by_camera[camera] = f[key][()] if key in f else fallback_encoded

        traj_len = actions_raw.shape[0]
        if traj_len <= 1:
            return

        normalized_joint = self._normalize_action(actions_raw)
        proprio_np = normalized_joint[:-1]
        if self.action_mode == "joint_delta":
            actions_np = normalize_robotwin_delta(
                actions_raw[1:] - actions_raw[:-1],
                self.delta_stats,
            )
        else:
            actions_np = normalized_joint[1:]
        usable_len = actions_np.shape[0]
        instruction = self._load_instruction(item["instruction_path"], item["episode_id"])

        sequence_len = self.latent_bridge_sequence_len
        if self.full_sequence:
            step = sequence_len if sequence_len > 1 else 1
            sample_indices = np.arange(0, usable_len, step)
        else:
            num_samples = max(1, int(usable_len / (self.future_len * 5)))
            sample_indices = np.random.choice(usable_len, size=num_samples, replace=False)

        image_cache = {}

        def get_camera(camera, idx):
            key = (camera, int(idx))
            if key not in image_cache:
                image_cache[key] = self._decode_rgb(encoded_by_camera[camera][int(idx)])
            return image_cache[key]

        def get_main(idx):
            return get_camera(self.cameras[0], idx)

        for t in sample_indices:
            max_anchor_lag = min(int(self.anchor_refresh_interval), int(t))
            anchor_t = int(np.random.randint(int(t) - max_anchor_lag, int(t) + 1))
            progress = 0.0 if usable_len <= 1 else float(t) / float(usable_len - 1)
            trajectory_horizon = max(usable_len - 1, 1)
            goal_distance = float(
                np.log1p(max((usable_len - 1) - int(t), 0)) / np.log1p(trajectory_horizon)
            )
            goal_idx = traj_len - 1
            future_progress_t = min(int(t) + self.future_len, usable_len - 1)
            action_progress = (
                0.0
                if usable_len <= 1
                else float(future_progress_t - int(t)) / float(usable_len - 1)
            )

            if sequence_len > 1:
                anchor_t = int(t)
                seq_idx = np.clip(np.arange(t, t + sequence_len), 0, usable_len - 1)
                seq_proprio = []
                seq_hist_actions = []
                seq_future_actions = []
                seq_progress = []
                seq_action_progress = []
                seq_goal_distance = []
                for seq_t in seq_idx:
                    seq_t = int(seq_t)
                    seq_hist_obs = np.clip(np.arange(seq_t - self.history_len + 1, seq_t + 1), 0, usable_len - 1)
                    seq_hist_act_idx = np.arange(seq_t - self.history_len, seq_t)
                    seq_fut_idx = np.arange(seq_t, seq_t + self.future_len)

                    seq_proprio.append(torch.from_numpy(proprio_np[seq_hist_obs]))

                    hist_actions = np.zeros((self.history_len, actions_np.shape[1]), dtype=np.float32)
                    valid_hist = seq_hist_act_idx >= 0
                    if np.any(valid_hist):
                        hist_actions[valid_hist] = actions_np[np.clip(seq_hist_act_idx[valid_hist], 0, usable_len - 1)]
                    seq_hist_actions.append(torch.from_numpy(hist_actions))

                    fut_actions = np.zeros((self.future_len, actions_np.shape[1]), dtype=np.float32)
                    valid_fut = seq_fut_idx < usable_len
                    if np.any(valid_fut):
                        fut_actions[valid_fut] = actions_np[seq_fut_idx[valid_fut]]
                    seq_future_actions.append(torch.from_numpy(fut_actions))

                    seq_progress.append(0.0 if usable_len <= 1 else float(seq_t) / float(usable_len - 1))
                    seq_future_progress_t = min(seq_t + self.future_len, usable_len - 1)
                    seq_action_progress.append(
                        0.0
                        if usable_len <= 1
                        else float(seq_future_progress_t - seq_t) / float(usable_len - 1)
                    )
                    seq_goal_distance.append(
                        float(
                            np.log1p(max((usable_len - 1) - seq_t, 0))
                            / np.log1p(trajectory_horizon)
                        )
                    )

                sample = {
                    "proprioception": torch.stack(seq_proprio, dim=0),
                    "history_actions": torch.stack(seq_hist_actions, dim=0),
                    "future_actions": torch.stack(seq_future_actions, dim=0),
                    "instruction": instruction,
                    "progress": torch.tensor(seq_progress, dtype=torch.float32),
                    "action_progress": torch.tensor(seq_action_progress, dtype=torch.float32),
                    "goal_distance": torch.tensor(seq_goal_distance, dtype=torch.float32),
                }

                if self.input_modality != "image":
                    raise ValueError("RoboTwin latent_bridge_sequence_len > 1 currently supports image modality only.")

                if self.view_mode == "multi":
                    sample["images"] = [get_camera(camera, anchor_t) for camera in self.cameras]
                    sample["anchor_images"] = [get_camera(camera, anchor_t) for camera in self.cameras]
                    sample["goal_images"] = [get_camera(camera, goal_idx) for camera in self.cameras]
                    sample["sequence_images"] = [
                        [get_camera(camera, int(seq_t)) for camera in self.cameras]
                        for seq_t in seq_idx
                    ]
                    sample["image"] = sample["images"][0]
                    if len(sample["images"]) > 1:
                        sample["image_wrist"] = sample["images"][-1]
                        sample["anchor_image_wrist"] = sample["anchor_images"][-1]
                    sample["goal_image"] = sample["goal_images"][0]
                    sample["goal_image_wrist"] = sample["goal_images"][-1]
                else:
                    sample["image"] = get_main(anchor_t)
                    sample["anchor_image"] = get_main(anchor_t)
                    sample["goal_image"] = get_main(goal_idx)
                    sample["sequence_images"] = [[get_main(int(seq_t))] for seq_t in seq_idx]

                yield sample
                continue

            hist_obs = np.clip(np.arange(t - self.history_len + 1, t + 1), 0, usable_len - 1)
            hist_act_idx = np.arange(t - self.history_len, t)
            fut_idx = np.arange(t, t + self.future_len)

            hist_proprio = torch.from_numpy(proprio_np[hist_obs])

            hist_actions = np.zeros((self.history_len, actions_np.shape[1]), dtype=np.float32)
            valid_hist = hist_act_idx >= 0
            if np.any(valid_hist):
                hist_actions[valid_hist] = actions_np[np.clip(hist_act_idx[valid_hist], 0, usable_len - 1)]

            fut_actions = np.zeros((self.future_len, actions_np.shape[1]), dtype=np.float32)
            valid_fut = fut_idx < usable_len
            if np.any(valid_fut):
                fut_actions[valid_fut] = actions_np[fut_idx[valid_fut]]

            sample = {
                "proprioception": hist_proprio,
                "history_actions": torch.from_numpy(hist_actions),
                "future_actions": torch.from_numpy(fut_actions),
                "instruction": instruction,
                "progress": torch.tensor(progress, dtype=torch.float32),
                "action_progress": torch.tensor(action_progress, dtype=torch.float32),
                "goal_distance": torch.tensor(goal_distance, dtype=torch.float32),
            }

            if self.load_future_image:
                target_idx = traj_len - 1 if self.future_image_mode == "last" else min(t + self.future_len, traj_len - 1)
                sample["future_image"] = get_main(target_idx)
                if self.view_mode == "multi":
                    sample["future_images"] = [get_camera(camera, target_idx) for camera in self.cameras]
                    if len(sample["future_images"]) > 1:
                        sample["future_image_wrist"] = sample["future_images"][1]

            if self.input_modality == "video":
                sample["video"] = np.stack([get_main(i) for i in hist_obs], axis=0)
                anchor_hist_obs = np.clip(
                    np.arange(anchor_t - self.history_len + 1, anchor_t + 1),
                    0,
                    usable_len - 1,
                )
                sample["anchor_video"] = np.stack([get_main(i) for i in anchor_hist_obs], axis=0)
                if self.view_mode == "multi":
                    sample["videos"] = [
                        np.stack([get_camera(camera, i) for i in hist_obs], axis=0)
                        for camera in self.cameras
                    ]
                    sample["video_wrist"] = sample["videos"][1] if len(sample["videos"]) > 1 else sample["video"]
                    sample["anchor_videos"] = [
                        np.stack([get_camera(camera, i) for i in anchor_hist_obs], axis=0)
                        for camera in self.cameras
                    ]
                    sample["anchor_video_wrist"] = sample["anchor_videos"][1] if len(sample["anchor_videos"]) > 1 else sample["anchor_video"]
            elif self.input_modality == "image":
                sample["image"] = get_main(t)
                sample["anchor_image"] = get_main(anchor_t)
                sample["goal_image"] = get_main(goal_idx)
                if self.view_mode == "multi":
                    sample["images"] = [get_camera(camera, t) for camera in self.cameras]
                    sample["image_wrist"] = sample["images"][1] if len(sample["images"]) > 1 else sample["image"]
                    sample["anchor_images"] = [get_camera(camera, anchor_t) for camera in self.cameras]
                    sample["anchor_image_wrist"] = sample["anchor_images"][1] if len(sample["anchor_images"]) > 1 else sample["anchor_image"]
                    sample["goal_images"] = [get_camera(camera, goal_idx) for camera in self.cameras]
                    sample["goal_image_wrist"] = sample["goal_images"][1] if len(sample["goal_images"]) > 1 else sample["goal_image"]
            else:
                raise ValueError(f"Unknown input_modality: {self.input_modality}")

            yield sample

    def __iter__(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        total_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id
        episodes = list(self.episodes[shard_index::total_shards])

        # The discovered episodes are sorted by task name. Without shuffling here,
        # a finite shuffle buffer mostly contains a single task for long stretches,
        # which makes early training batches look like one repeated behavior.
        worker_seed = torch.initial_seed() % (2**32)
        rng = np.random.default_rng(worker_seed)
        rng.shuffle(episodes)

        shuffle_buffer = []
        for item in episodes:
            try:
                for sample in self._iter_episode_samples(item):
                    shuffle_buffer.append(sample)
                    if len(shuffle_buffer) >= self.buffer_size:
                        idx = rng.integers(len(shuffle_buffer))
                        shuffle_buffer[idx], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[idx]
                        yield shuffle_buffer.pop()
            except Exception as e:
                print(f"[Warn] Skipping RoboTwin episode {item['path']} due to error: {e}")

        rng.shuffle(shuffle_buffer)
        for sample in shuffle_buffer:
            yield sample
