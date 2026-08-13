"""Evaluate a VLANeXt checkpoint in the original RoboMimic robosuite tasks."""

import argparse
import json
import os
from pathlib import Path

# robosuite 1.4.1 uses cached numba kernels that fail to import with newer numba.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

import robosuite

from src.evaluation.libero_bench.VLANeXt_utils import get_processor
from src.evaluation.libero_bench.robot_utils import get_action, get_model, set_seed_everywhere


TASKS = {
    "lift": {
        "file_pattern": "*lift*.hdf5",
        "instruction": "lift the cube",
    },
    "can": {
        "file_pattern": "*can*.hdf5",
        "instruction": "pick up the can and place it in the target bin",
    },
    "square": {
        "file_pattern": "*squa*.hdf5",
        "instruction": "insert the square nut onto the square peg",
    },
}


class DictConfig:
    def __init__(self, values):
        for key, value in values.items():
            setattr(self, key, DictConfig(value) if isinstance(value, dict) else value)


def quat_to_rotvec(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float32)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-6)
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = float(np.linalg.norm(xyz))
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], 0.0, 1.0))
    return xyz / max(sin_half, 1e-6) * angle


def proprio_from_obs(obs):
    gripper = np.clip(
        1.0 - np.mean(np.abs(obs["robot0_gripper_qpos"])) / 0.04,
        0.0,
        1.0,
    )
    return np.concatenate(
        [obs["robot0_eef_pos"], quat_to_rotvec(obs["robot0_eef_quat"]), [gripper]]
    ).astype(np.float32)


def image_from_obs(obs, key):
    # RoboMimic's image writer vertically flips raw robosuite camera observations.
    return np.ascontiguousarray(np.flipud(obs[key]))


def discover_dataset(dataset_root, pattern):
    matches = sorted((Path(dataset_root) / "downloads" / "robomimic_ph").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one RoboMimic file for {pattern}, found {matches}")
    return matches[0]


def make_env(env_meta, gpu_id):
    kwargs = dict(env_meta["env_kwargs"])
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        render_gpu_device_id=gpu_id,
        ignore_done=True,
    )
    # robosuite 1.4.1 incorrectly treats a single physical CUDA id (for example
    # CUDA_VISIBLE_DEVICES=1) as an EGL-local index. Torch is already initialized
    # at this point, so expose the local index only while the EGL context is built.
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    egl_device = os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    if visible_devices and visible_devices.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        return robosuite.make(env_name=env_meta["env_name"], **kwargs)
    finally:
        if visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        if egl_device is not None:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = egl_device


def save_video(frames, path, fps):
    if frames:
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def evaluate(cfg, smoke=False):
    set_seed_everywhere(int(cfg.eval.seed))
    model = get_model(cfg)
    processor = get_processor(cfg)

    checkpoint = Path(cfg.eval.finetuned_checkpoint)
    configured_output_dir = str(getattr(cfg.eval, "output_dir", "")).strip()
    if configured_output_dir:
        output_dir = Path(configured_output_dir)
    else:
        output_dir = checkpoint.parent / f"robomimic_checkpoint_{checkpoint.stem.split('_')[-1]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_names = list(getattr(cfg.eval, "tasks", TASKS.keys()))
    trials = int(cfg.eval.num_trials_per_task)
    max_steps = int(cfg.eval.max_steps)
    if smoke:
        task_names = task_names[:1]
        trials = 1
        max_steps = 1
    execute_steps = int(cfg.eval.num_steps_execute)
    save_videos = bool(getattr(cfg.eval, "save_video", True))
    save_video_every = max(1, int(getattr(cfg.eval, "save_video_every", 1)))
    fps = int(getattr(cfg.eval, "video_fps", 20))
    gpu_id = int(getattr(cfg.eval, "render_gpu_device_id", 0))

    log_path = output_dir / "log.txt"
    summary_path = output_dir / "summary.tsv"
    total_successes = 0
    total_trials = 0

    with log_path.open("w", buffering=1) as log_file, summary_path.open("w", buffering=1) as summary:
        summary.write("task\tsuccesses\ttrials\tsuccess_rate\n")
        for task_name in task_names:
            if task_name not in TASKS:
                raise ValueError(f"Unknown task {task_name}; choose from {list(TASKS)}")
            spec = TASKS[task_name]
            dataset_path = discover_dataset(cfg.eval.dataset_root, spec["file_pattern"])

            with h5py.File(dataset_path, "r") as dataset:
                env_meta = json.loads(dataset["data"].attrs["env_args"])
                demo_names = sorted(
                    dataset["data"].keys(), key=lambda name: int(name.split("_")[-1])
                )
                if trials > len(demo_names):
                    raise ValueError(f"{task_name} has only {len(demo_names)} initial states")

                env = make_env(env_meta, gpu_id)
                task_successes = 0
                try:
                    for trial_index, demo_name in enumerate(demo_names[:trials]):
                        initial_state = dataset["data"][demo_name]["states"][0]
                        env.reset()
                        env.sim.set_state_from_flattened(initial_state)
                        env.sim.forward()
                        obs = env._get_observations(force_update=True)

                        state_history = []
                        action_history = []
                        action_buffer = []
                        frames = []
                        success = False

                        for _ in range(max_steps):
                            main_image = image_from_obs(obs, "agentview_image")
                            wrist_image = image_from_obs(obs, "robot0_eye_in_hand_image")
                            frames.append(main_image)
                            state_history.append(proprio_from_obs(obs))

                            if not action_buffer:
                                observation = {
                                    "full_image": main_image,
                                    "full_image_wrist": wrist_image,
                                    "image_history": [main_image],
                                    "image_history_wrist": [wrist_image],
                                    "state_history": state_history,
                                    "action_history": action_history,
                                }
                                action_chunk = np.asarray(
                                    get_action(
                                        cfg,
                                        model,
                                        observation,
                                        spec["instruction"],
                                        processor=processor,
                                    ),
                                    dtype=np.float32,
                                )
                                if action_chunk.ndim == 1:
                                    action_chunk = action_chunk[None]
                                action_buffer = list(action_chunk[:execute_steps])

                            action = np.clip(action_buffer.pop(0), -1.0, 1.0)
                            action_history.append(action.copy())
                            obs, _, _, _ = env.step(action)
                            if env._check_success():
                                success = True
                                break

                        task_successes += int(success)
                        total_successes += int(success)
                        total_trials += 1
                        message = (
                            f"[{task_name}] trial={trial_index + 1}/{trials} "
                            f"success={success} steps={len(frames)}"
                        )
                        print(message, flush=True)
                        log_file.write(message + "\n")

                        if save_videos and (trial_index + 1) % save_video_every == 0:
                            status = "success" if success else "failure"
                            save_video(
                                frames,
                                output_dir / "videos" / f"{task_name}_{trial_index + 1:03d}_{status}.mp4",
                                fps,
                            )
                finally:
                    env.close()

            task_rate = task_successes / trials
            summary.write(f"{task_name}\t{task_successes}\t{trials}\t{task_rate:.6f}\n")
            log_file.write(f"[{task_name}] success_rate={task_rate * 100:.2f}%\n")

        overall_rate = total_successes / max(total_trials, 1)
        summary.write(f"overall\t{total_successes}\t{total_trials}\t{overall_rate:.6f}\n")
        log_file.write(f"[overall] success_rate={overall_rate * 100:.2f}%\n")
        print(f"[overall] {total_successes}/{total_trials} = {overall_rate * 100:.2f}%")
        print(f"Results: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Load the full policy and run one simulation step on the first task.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        evaluate(DictConfig(yaml.safe_load(config_file)), smoke=args.smoke)
