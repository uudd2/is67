import os
import json
import hashlib
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], 'GPU')

action_min_spatial = [-0.9375, -0.9375, -0.9375, -0.1875, -0.3675000071525574, -0.36000001430511475]
action_max_spatial = [0.9375, 0.9375, 0.9375, 0.1971428543329239, 0.33642858266830444, 0.375]

action_min_object = [-0.8839285969734192, -0.9375, -0.9375, -0.15000000596046448, -0.29035714268684387, -0.32892856001853943]
action_max_object = [0.9375, 0.8919642567634583, 0.9375, 0.17678570747375488, 0.35035714507102966, 0.1810714304447174]

action_min_goal =  [-0.9375, -0.9375, -0.9375, -0.2582142949104309, -0.375, -0.2871428430080414]
action_max_goal = [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375]

action_min_10 = [-0.9375, -0.9375, -0.9375, -0.23642857372760773, -0.3053571283817291, -0.3675000071525574]
action_max_10 = [0.9375, 0.9375, 0.9375, 0.30000001192092896, 0.29357144236564636, 0.375]


def load_libero_mean_std_stats(path):
    if not path:
        raise ValueError("LIBERO mean/std normalization requires normalization_stats_path.")
    with open(os.path.expanduser(path), "r", encoding="utf-8") as handle:
        raw_stats = json.load(handle)

    stats = {}
    for name in ("action", "state"):
        if name not in raw_stats:
            raise ValueError(f"Missing '{name}' statistics in {path}.")
        mean = np.asarray(raw_stats[name]["mean"], dtype=np.float32)
        std = np.asarray(raw_stats[name]["std"], dtype=np.float32)
        if mean.shape != (7,) or std.shape != (7,):
            raise ValueError(
                f"Expected 7D {name} mean/std in {path}, got {mean.shape}/{std.shape}."
            )
        stats[name] = {
            "mean": mean,
            "std": np.maximum(std, np.float32(1e-6)),
        }
    return stats


def strict_future_start_indices(trajectory_length, future_offset, step=1):
    """Return starts whose endpoint frame t + future_offset exists."""
    trajectory_length = int(trajectory_length)
    future_offset = int(future_offset)
    step = int(step)
    if future_offset <= 0:
        raise ValueError("future_offset must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")
    stop = max(trajectory_length - future_offset, 0)
    return np.arange(0, stop, step, dtype=np.int64)


class LiberoAct(IterableDataset):
    def __init__(
        self,
        data_path,
        dataset_name='libero',
        length=None,
        history_len=15,
        future_len=15,
        full_sequence=True,
        input_modality="video",
        view_mode="single",
        load_future_image=False,
        future_image_mode="horizon",
        future_image_offsets=None,
        strict_future_horizon=False,
        frame_ids_only=False,
        buffer_size=10000,
        anchor_refresh_interval=1,
        latent_bridge_sequence_len=1,
        normalization_mode="min_max",
        normalization_stats_path=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.length = length
        self.history_len = history_len
        self.future_len = future_len
        self.full_sequence = full_sequence
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.load_future_image = load_future_image
        self.future_image_mode = future_image_mode
        self.strict_future_horizon = bool(strict_future_horizon)
        self.frame_ids_only = bool(frame_ids_only)
        if (
            self.strict_future_horizon
            and not self.load_future_image
            and not self.frame_ids_only
        ):
            raise ValueError(
                "strict_future_horizon requires load_future_image=True or "
                "frame_ids_only=True."
            )
        if self.strict_future_horizon and self.future_image_mode != "horizon":
            raise ValueError("strict_future_horizon requires future_image_mode='horizon'.")
        self.future_image_offsets = tuple(
            int(offset) for offset in (future_image_offsets or ())
        )
        if self.future_image_offsets:
            if tuple(sorted(set(self.future_image_offsets))) != self.future_image_offsets:
                raise ValueError(
                    "future_image_offsets must be positive, unique, and sorted."
                )
            if self.future_image_offsets[0] <= 0:
                raise ValueError("future_image_offsets must be positive.")
            if self.future_image_offsets[-1] > int(self.future_len):
                raise ValueError(
                    "future_image_offsets cannot exceed future_len."
                )
        self.buffer_size = buffer_size
        self.anchor_refresh_interval = max(1, int(anchor_refresh_interval))
        self.latent_bridge_sequence_len = max(1, int(latent_bridge_sequence_len))
        if self.future_image_offsets and self.latent_bridge_sequence_len > 1:
            raise ValueError(
                "Multi-horizon future images currently require "
                "latent_bridge_sequence_len=1."
            )
        self.normalization_mode = str(normalization_mode).lower()
        if self.normalization_mode not in {"min_max", "mean_std", "identity"}:
            raise ValueError(
                "normalization_mode must be one of: min_max, mean_std, identity."
            )
        self.normalization_stats = (
            load_libero_mean_std_stats(normalization_stats_path)
            if self.normalization_mode == "mean_std"
            else None
        )

        if 'spatial' in dataset_name:
            self.action_min = np.array(action_min_spatial)
            self.action_max = np.array(action_max_spatial)
        elif 'object' in dataset_name:
            self.action_min = np.array(action_min_object)
            self.action_max = np.array(action_max_object)
        elif 'goal' in dataset_name:
            self.action_min = np.array(action_min_goal)
            self.action_max = np.array(action_max_goal)
        elif '10' in dataset_name:
            self.action_min = np.array(action_min_10)
            self.action_max = np.array(action_max_10)
        elif 'mixed' in dataset_name:
            self.action_min = np.array(action_min_mixed)
            self.action_max = np.array(action_max_mixed)
        else:
            print(f"[Warn] Unknown dataset name '{dataset_name}', defaulting to libero_10 stats.")
            self.action_min = np.array(action_min_10)
            self.action_max = np.array(action_max_10)

    def __iter__(self):
        builder = tfds.builder_from_directory(builder_dir=self.data_path)
        
        read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
        ds = builder.as_dataset(split='train', shuffle_files=False, read_config=read_config)
        
        if self.length is not None:
            ds = ds.take(self.length)

        shuffle_buffer = []
        BUFFER_SIZE = self.buffer_size

        main_key = "image"
        wrist_key = "wrist_image"

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id
        ds_iterator = ds.shard(num_shards=total_shards, index=shard_index)

        for traj_id, traj_data in enumerate(ds_iterator):
            try:
                traj_batch = next(iter(traj_data['steps'].batch(2000)))

                if traj_batch['reward'][-1].numpy() != 1:
                    continue

                traj_len = traj_batch['action'].shape[0]

                obs = traj_batch['observation']
                if self.frame_ids_only:
                    first_image = obs[main_key][0].numpy()
                    last_image = obs[main_key][-1].numpy()
                    if first_image.dtype != np.uint8:
                        first_image = (first_image * 255).astype(np.uint8)
                        last_image = (last_image * 255).astype(np.uint8)
                    images_np = None
                else:
                    images_np = obs[main_key].numpy()
                    if images_np.dtype != np.uint8:
                        images_np = (images_np * 255).astype(np.uint8)
                    first_image = images_np[0]
                    last_image = images_np[-1]
                
                wrist_np = None
                if self.view_mode == "multi" and not self.frame_ids_only:
                    if wrist_key in obs:
                        wrist_np = obs[wrist_key].numpy()
                        if wrist_np.dtype != np.uint8:
                            wrist_np = (wrist_np * 255).astype(np.uint8)
                    else:
                        wrist_np = images_np

                # Process Proprioception: 6D Pose + 1D Normalized Gripper
                raw_state = traj_batch['observation']['state'].numpy().astype(np.float32)
                
                # [Proprio Gripper]
                # Raw: 2D gripper fingers width (qpos), approx range [0, 0.04]. 0.04 = Open, 0 = Closed.
                # Processed: 1.0 - (width / 0.04).
                # Result: Range [0, 1]. 0 = Open, 1 = Closed.
                gripper_qpos = raw_state[:, 6:8]
                gripper_state = 1.0 - (np.mean(np.abs(gripper_qpos), axis=1, keepdims=True) / 0.04)
                gripper_state = np.clip(gripper_state, 0.0, 1.0)
                
                proprio_np = np.concatenate([raw_state[:, :6], gripper_state], axis=1)

                # Normalize the same 7D state/action representation consumed by the policy.
                raw_actions = traj_batch['action'].numpy().astype(np.float32)
                raw_actions = raw_actions[:, :7]
                if self.normalization_mode == "mean_std":
                    action_stats = self.normalization_stats["action"]
                    state_stats = self.normalization_stats["state"]
                    actions_np = (
                        raw_actions - action_stats["mean"]
                    ) / action_stats["std"]
                    proprio_np = (
                        proprio_np - state_stats["mean"]
                    ) / state_stats["std"]
                elif self.normalization_mode == "min_max":
                    delta_pose = raw_actions[:, :6]
                    denominator = self.action_max - self.action_min
                    denominator = np.where(denominator == 0, 1.0, denominator)
                    delta_pose = 2.0 * (delta_pose - self.action_min) / denominator - 1.0
                    delta_pose = np.clip(delta_pose, -1.0, 1.0)
                    gripper_action = np.clip(raw_actions[:, 6:7], -1.0, 1.0)
                    actions_np = np.concatenate([delta_pose, gripper_action], axis=1)
                else:
                    actions_np = raw_actions

                actions_np = actions_np.astype(np.float32, copy=False)
                proprio_np = proprio_np.astype(np.float32, copy=False)
                
                instruction = traj_batch['language_instruction'][0].numpy().decode('utf-8')
                trajectory_fingerprint = hashlib.sha1()
                trajectory_fingerprint.update(str(traj_len).encode("ascii"))
                trajectory_fingerprint.update(instruction.encode("utf-8"))
                trajectory_fingerprint.update(first_image.tobytes())
                trajectory_fingerprint.update(last_image.tobytes())
                trajectory_cache_id = trajectory_fingerprint.hexdigest()[:20]

                sequence_len = self.latent_bridge_sequence_len
                if self.strict_future_horizon:
                    if sequence_len > 1:
                        raise ValueError(
                            "strict_future_horizon currently requires "
                            "latent_bridge_sequence_len=1."
                        )
                    valid_indices = strict_future_start_indices(
                        traj_len,
                        self.future_len,
                    )
                    if self.full_sequence:
                        sample_indices = valid_indices
                    else:
                        num_samples = max(1, int(traj_len / (15 * 5)))
                        num_samples = min(num_samples, len(valid_indices))
                        if num_samples == 0:
                            continue
                        sample_indices = np.random.choice(
                            valid_indices,
                            size=num_samples,
                            replace=False,
                        )
                elif self.full_sequence:
                    step = sequence_len if sequence_len > 1 else 1
                    sample_indices = np.arange(0, traj_len, step)
                else:
                    num_samples = max(1, int(traj_len / (15 * 5)))
                    sample_indices = np.random.choice(traj_len, size=num_samples, replace=False)

                for t in sample_indices:
                    max_anchor_lag = min(int(self.anchor_refresh_interval), int(t))
                    anchor_t = int(np.random.randint(int(t) - max_anchor_lag, int(t) + 1))
                    progress = 0.0 if traj_len <= 1 else float(t) / float(traj_len - 1)
                    trajectory_horizon = max(traj_len - 1, 1)
                    goal_distance = float(
                        np.log1p(max((traj_len - 1) - int(t), 0)) / np.log1p(trajectory_horizon)
                    )
                    goal_idx = traj_len - 1
                    future_progress_t = min(int(t) + self.future_len, traj_len - 1)
                    action_progress = (
                        0.0
                        if traj_len <= 1
                        else float(future_progress_t - int(t)) / float(traj_len - 1)
                    )

                    if sequence_len > 1:
                        anchor_t = int(t)
                        seq_idx = np.clip(np.arange(t, t + sequence_len), 0, traj_len - 1)
                        seq_proprio = []
                        seq_hist_actions = []
                        seq_future_actions = []
                        seq_progress = []
                        seq_action_progress = []
                        seq_goal_distance = []
                        for seq_t in seq_idx:
                            seq_t = int(seq_t)
                            seq_hist_obs = np.clip(
                                np.arange(seq_t - self.history_len + 1, seq_t + 1),
                                0,
                                traj_len - 1,
                            )
                            seq_hist_act_idx = np.arange(seq_t - self.history_len, seq_t)
                            seq_fut_idx = np.arange(seq_t, seq_t + self.future_len)

                            seq_proprio.append(torch.from_numpy(proprio_np[seq_hist_obs]))

                            hist_actions = np.zeros((self.history_len, actions_np.shape[1]), dtype=np.float32)
                            valid_hist = seq_hist_act_idx >= 0
                            if np.any(valid_hist):
                                valid_indices = np.clip(seq_hist_act_idx[valid_hist], 0, traj_len - 1)
                                hist_actions[valid_hist] = actions_np[valid_indices]
                            seq_hist_actions.append(torch.from_numpy(hist_actions))

                            fut_actions = np.zeros((self.future_len, actions_np.shape[1]), dtype=np.float32)
                            valid_fut = seq_fut_idx < traj_len
                            if np.any(valid_fut):
                                fut_actions[valid_fut] = actions_np[seq_fut_idx[valid_fut]]
                            seq_future_actions.append(torch.from_numpy(fut_actions))

                            seq_progress.append(0.0 if traj_len <= 1 else float(seq_t) / float(traj_len - 1))
                            seq_future_progress_t = min(seq_t + self.future_len, traj_len - 1)
                            seq_action_progress.append(
                                0.0
                                if traj_len <= 1
                                else float(seq_future_progress_t - seq_t) / float(traj_len - 1)
                            )
                            seq_goal_distance.append(
                                float(
                                    np.log1p(max((traj_len - 1) - seq_t, 0))
                                    / np.log1p(trajectory_horizon)
                                )
                            )

                        sample = {
                            'proprioception': torch.stack(seq_proprio, dim=0),
                            'history_actions': torch.stack(seq_hist_actions, dim=0),
                            'future_actions': torch.stack(seq_future_actions, dim=0),
                            'instruction': instruction,
                            'progress': torch.tensor(seq_progress, dtype=torch.float32),
                            'action_progress': torch.tensor(seq_action_progress, dtype=torch.float32),
                            'goal_distance': torch.tensor(seq_goal_distance, dtype=torch.float32),
                        }

                        if self.input_modality != "image":
                            raise ValueError("Libero latent_bridge_sequence_len > 1 currently supports image modality only.")

                        if self.view_mode == "multi":
                            sample['images'] = [images_np[anchor_t], wrist_np[anchor_t] if wrist_np is not None else images_np[anchor_t]]
                            sample['anchor_images'] = [images_np[anchor_t], wrist_np[anchor_t] if wrist_np is not None else images_np[anchor_t]]
                            sample['goal_images'] = [images_np[goal_idx], wrist_np[goal_idx] if wrist_np is not None else images_np[goal_idx]]
                            sample['sequence_images'] = [
                                [images_np[int(seq_t)], wrist_np[int(seq_t)] if wrist_np is not None else images_np[int(seq_t)]]
                                for seq_t in seq_idx
                            ]
                            sample['image'] = sample['images'][0]
                            sample['image_wrist'] = sample['images'][1]
                            sample['anchor_image'] = sample['anchor_images'][0]
                            sample['anchor_image_wrist'] = sample['anchor_images'][1]
                            sample['goal_image'] = sample['goal_images'][0]
                            sample['goal_image_wrist'] = sample['goal_images'][1]
                        else:
                            sample['image'] = images_np[anchor_t]
                            sample['anchor_image'] = images_np[anchor_t]
                            sample['goal_image'] = images_np[goal_idx]
                            sample['sequence_images'] = [[images_np[int(seq_t)]] for seq_t in seq_idx]

                        shuffle_buffer.append(sample)
                        if len(shuffle_buffer) >= BUFFER_SIZE:
                            idx = np.random.randint(len(shuffle_buffer))
                            shuffle_buffer[idx], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[idx]
                            yield shuffle_buffer.pop()
                        continue

                    start_hist_obs = t - self.history_len + 1
                    hist_indices_obs = np.arange(start_hist_obs, t + 1)
                    hist_indices_obs = np.clip(hist_indices_obs, 0, traj_len - 1)
                    
                    start_hist_act = t - self.history_len
                    hist_indices_act = np.arange(start_hist_act, t)
                    
                    end_fut = t + self.future_len
                    fut_indices = np.arange(t, end_fut)

                    hist_imgs = (
                        None
                        if self.frame_ids_only
                        else images_np[hist_indices_obs]
                    )
                    hist_imgs_wrist = (
                        wrist_np[hist_indices_obs]
                        if wrist_np is not None and not self.frame_ids_only
                        else None
                    )
                    hist_proprio = torch.from_numpy(proprio_np[hist_indices_obs])
                    
                    hist_actions = np.zeros((self.history_len, actions_np.shape[1]), dtype=np.float32)
                    valid_mask = hist_indices_act >= 0
                    if np.any(valid_mask):
                        valid_indices = hist_indices_act[valid_mask]
                        valid_indices = np.clip(valid_indices, 0, traj_len - 1)
                        hist_actions[valid_mask] = actions_np[valid_indices]
                    hist_actions = torch.from_numpy(hist_actions)
                    
                    fut_acts_np = np.zeros((self.future_len, actions_np.shape[1]), dtype=np.float32)
                    valid_mask_fut = fut_indices < traj_len
                    if np.any(valid_mask_fut):
                        valid_indices_fut = fut_indices[valid_mask_fut]
                        fut_acts_np[valid_mask_fut] = actions_np[valid_indices_fut]
                    fut_acts = torch.from_numpy(fut_acts_np)

                    sample = {
                        'proprioception': hist_proprio,
                        'history_actions': hist_actions,
                        'future_actions': fut_acts,
                        'instruction': instruction,
                        'progress': torch.tensor(progress, dtype=torch.float32),
                        'action_progress': torch.tensor(action_progress, dtype=torch.float32),
                        'goal_distance': torch.tensor(goal_distance, dtype=torch.float32),
                        'frame_id': f"{trajectory_cache_id}_frame_{int(t):06d}",
                    }
                    
                    if self.load_future_image or self.frame_ids_only:
                        if self.future_image_mode == "last":
                            target_idx = traj_len - 1
                        else:
                            target_idx = (
                                int(t) + self.future_len
                                if self.strict_future_horizon
                                else min(t + self.future_len, traj_len - 1)
                            )
                        sample['future_frame_id'] = (
                            f"{trajectory_cache_id}_frame_{int(target_idx):06d}"
                        )
                        if self.load_future_image:
                            sample['future_image'] = images_np[target_idx]
                            if self.future_image_offsets:
                                sample['future_horizon_images'] = [
                                    images_np[min(t + offset, traj_len - 1)]
                                    for offset in self.future_image_offsets
                                ]
                            if self.view_mode == "multi":
                                sample['future_image_wrist'] = wrist_np[target_idx] if wrist_np is not None else images_np[target_idx]
                                sample['future_images'] = [sample['future_image'], sample['future_image_wrist']]

                    if self.frame_ids_only:
                        pass
                    elif self.input_modality == "video":
                        sample['video'] = hist_imgs
                        anchor_hist_obs = np.clip(
                            np.arange(anchor_t - self.history_len + 1, anchor_t + 1),
                            0,
                            traj_len - 1,
                        )
                        sample['anchor_video'] = images_np[anchor_hist_obs]
                        if self.view_mode == "multi":
                            sample['video_wrist'] = hist_imgs_wrist if hist_imgs_wrist is not None else hist_imgs
                            sample['anchor_video_wrist'] = wrist_np[anchor_hist_obs] if wrist_np is not None else sample['anchor_video']
                    elif self.input_modality == "image":
                        sample['image'] = images_np[t]
                        sample['anchor_image'] = images_np[anchor_t]
                        sample['goal_image'] = images_np[goal_idx]
                        if self.view_mode == "multi":
                            sample['image_wrist'] = wrist_np[t] if wrist_np is not None else images_np[t]
                            sample['anchor_image_wrist'] = wrist_np[anchor_t] if wrist_np is not None else images_np[anchor_t]
                            sample['goal_image_wrist'] = wrist_np[goal_idx] if wrist_np is not None else images_np[goal_idx]
                            sample['goal_images'] = [sample['goal_image'], sample['goal_image_wrist']]
                    else:
                        raise ValueError(f"Unknown input_modality: {self.input_modality}")

                    shuffle_buffer.append(sample)
                    
                    if len(shuffle_buffer) >= BUFFER_SIZE:
                        idx = np.random.randint(len(shuffle_buffer))
                        shuffle_buffer[idx], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[idx]
                        yield shuffle_buffer.pop()

            except Exception as e:
                print(f"[Warn] Skipping trajectory {traj_id} due to error: {e}")
                continue

        np.random.shuffle(shuffle_buffer)
        for sample in shuffle_buffer:
            yield sample

def collate_fn(batch):
    return batch

if __name__ == "__main__":
    """
    Fast stats: count how many training samples LiberoAct would yield for each suite.
    Also computes min/max statistics for the first 6 dimensions of actions.
    """
    from tqdm import tqdm

    # Configuration
    BASE_DIR = "/mnt/draven/data/LIBERO_modified"
    SUITES = [
        "libero_spatial",
        # "libero_object",
        # "libero_goal",
        # "libero_10",
    ]
    VERSION = "1.0.0"

    print(f"Scanning Libero datasets in {BASE_DIR}...")
    
    for suite_name in SUITES:
        data_path = os.path.join(BASE_DIR, suite_name, VERSION)
        if not os.path.exists(data_path):
            continue

        builder = tfds.builder_from_directory(builder_dir=data_path)
        read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
        ds = builder.as_dataset(split='train', shuffle_files=False, read_config=read_config)
        
        total_files = builder.info.splits['train'].num_examples
        
        total_trajs = 0
        success_trajs = 0
        total_samples = 0
        
        act_min = np.full(6, np.inf)
        act_max = np.full(6, -np.inf)
        
        print(f"\nProcessing {suite_name} ({total_files} trajectories)...")
        pbar = tqdm(enumerate(ds), total=total_files, unit="traj", desc=suite_name)
        
        for traj_id, traj_data in pbar:
            total_trajs += 1
            try:
                traj_batch = next(iter(traj_data['steps'].batch(2000)))
                if traj_batch['reward'][-1].numpy() != 1:
                    continue

                success_trajs += 1
                traj_len = int(traj_batch['action'].shape[0])
                
                # Global Action Stats
                actions = traj_batch['action'].numpy()[:, :6]
                current_min = np.min(actions, axis=0)
                current_max = np.max(actions, axis=0)
                act_min = np.minimum(act_min, current_min)
                act_max = np.maximum(act_max, current_max)

                total_samples += traj_len
                pbar.set_postfix({"Succ": success_trajs, "Samples": total_samples})

            except Exception as e:
                continue
        
        print(f"--- {suite_name} Stats ---")
        print(f"Total Trajectories: {total_trajs}")
        print(f"Successful Trajs:   {success_trajs}")
        print(f"Avg Samples/Succ:   {total_samples / success_trajs:.4f}" if success_trajs > 0 else "")
        print(f"action_min = {act_min.tolist()}")
        print(f"action_max = {act_max.tolist()}")
