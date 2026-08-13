#!/usr/bin/env python3
"""Evaluate a VLANeXt checkpoint on the six AV-ALOHA simulation tasks."""

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf

import gym_av_aloha  # noqa: F401 - registers the environments

# Keep VITA's compatible MuJoCo/dm_control stack first. Only append the
# VLANeXt environment to obtain transformers and its model dependencies.
VLANEXT_SITE_PACKAGES = Path(
    "/home/dm/miniconda3/envs/VLANeXt/lib/python3.10/site-packages"
)
if str(VLANEXT_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(VLANEXT_SITE_PACKAGES))

from src.datasets.av_aloha_multitask_act import AVAlohaMultitaskAct
from src.evaluation.libero_bench.VLANeXt_utils import (
    get_processor,
    get_vla,
    get_vla_action,
)


TASKS = {
    "cube_transfer": ("av_aloha_sim_cube_transfer", "cube-transfer-v1"),
    "thread_needle": ("av_aloha_sim_thread_needle", "thread-needle-v1"),
    "peg_insertion": ("av_aloha_sim_peg_insertion", "peg-insertion-v1"),
    "pour_test_tube": ("av_aloha_sim_pour_test_tube", "pour-test-tube-v1"),
    "hook_package": ("av_aloha_sim_hook_package", "hook-package-v1"),
    "slot_insertion": ("av_aloha_sim_slot_insertion", "slot-insertion-v1"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="/media/dm/Elements/VLANeXt_migration/data/AV_ALOHA/converted/iantc104")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--exec-horizon", type=int, default=8)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--video-episodes", type=int, default=2)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--camera-width", type=int, default=320)
    return parser.parse_args()


def denormalize_action(action, stats):
    value_min = stats["min"]
    value_max = stats["max"]
    return (np.clip(action, -1.0, 1.0) + 1.0) * 0.5 * (value_max - value_min) + value_min


def policy_observation(raw_obs, histories):
    main = np.asarray(raw_obs["pixels"]["zed_cam_left"], dtype=np.uint8)
    secondary = np.asarray(raw_obs["pixels"]["zed_cam_right"], dtype=np.uint8)
    state = np.asarray(raw_obs["agent_pos"], dtype=np.float32)
    histories["main"].append(main)
    histories["secondary"].append(secondary)
    histories["state"].append(state)
    return {
        "full_image": main,
        "full_image_wrist": secondary,
        "image_history": list(histories["main"]),
        "image_history_wrist": list(histories["secondary"]),
        "state_history": list(histories["state"]),
        "action_history": list(histories["action"]),
    }


def main():
    args = parse_args()
    unknown = sorted(set(args.tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown AV-ALOHA tasks: {unknown}")
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create(
        {
            "eval": {"finetuned_checkpoint": str(Path(args.checkpoint).resolve())},
            "model": {"diffusion_steps": args.diffusion_steps},
        }
    )
    model = get_vla(cfg)
    processor = get_processor(cfg)
    train_data = model.train_config["data"]
    eval_fps = float(
        args.fps
        if args.fps is not None
        else train_data.get("control_fps", train_data.get("fps", 25.0))
    )
    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else train_data.get("eval_max_steps", round(15.0 * eval_fps))
    )
    dataset_tasks = list(train_data.get("av_aloha_tasks", []))
    stats_dataset = AVAlohaMultitaskAct(
        data_root=args.data_root,
        tasks=dataset_tasks,
        history_len=int(train_data.get("history_len", 8)),
        future_len=int(train_data.get("future_len", 8)),
        action_stride=int(train_data.get("action_stride", 1)),
        buffer_size=max(1, int(train_data.get("buffer_size", 256))),
        main_camera=train_data.get("main_camera", "observation.images.zed_cam_left"),
        secondary_camera=train_data.get("secondary_camera", "observation.images.zed_cam_right"),
        action_normalization=train_data.get("action_normalization", "min_max"),
        state_normalization=train_data.get("state_normalization", "identity"),
    )
    action_stats = stats_dataset.stats["action"]
    task_instructions = {
        info["name"]: info["instruction"] for info in stats_dataset.task_info
    }
    history_len = int(train_data.get("history_len", 8))
    rows = []

    for task_index, task_name in enumerate(args.tasks):
        dataset_task, env_name = TASKS[task_name]
        env = gym.make(
            f"gym_av_aloha/{env_name}",
            disable_env_checker=True,
            fps=eval_fps,
            cameras={
                "zed_cam_left": [args.camera_height, args.camera_width],
                "zed_cam_right": [args.camera_height, args.camera_width],
            },
            render_camera="zed_cam_left",
            enable_av=True,
        )
        prompt = task_instructions[dataset_task]
        successes = 0
        max_rewards = []

        for episode in range(args.episodes):
            raw_obs, _ = env.reset(seed=args.seed + task_index * 1000 + episode)
            histories = {
                "main": deque(maxlen=history_len),
                "secondary": deque(maxlen=history_len),
                "state": deque(maxlen=history_len),
                "action": deque(maxlen=history_len),
            }
            frames = []
            episode_success = False
            episode_max_reward = 0.0
            step = 0

            while step < max_steps and not episode_success:
                obs = policy_observation(raw_obs, histories)
                normalized_chunk = get_vla_action(cfg, model, processor, obs, prompt)
                action_chunk = denormalize_action(normalized_chunk, action_stats)
                if episode == 0 and step == 0:
                    print(
                        f"[{task_name}] instruction={prompt!r} "
                        f"normalized_action_range="
                        f"[{normalized_chunk.min():.3f}, {normalized_chunk.max():.3f}] "
                        f"state[:4]={raw_obs['agent_pos'][:4]} "
                        f"action[:4]={action_chunk[0, :4]} "
                        f"camera_state={raw_obs['agent_pos'][14:21]} "
                        f"camera_action={action_chunk[0, 14:21]}",
                        flush=True,
                    )
                for action in action_chunk[: args.exec_horizon]:
                    raw_obs, reward, terminated, truncated, info = env.step(action)
                    histories["action"].append(np.asarray(action, dtype=np.float32))
                    episode_max_reward = max(episode_max_reward, float(reward))
                    episode_success = bool(info.get("is_success", False))
                    if episode < args.video_episodes:
                        frames.append(np.asarray(raw_obs["pixels"]["zed_cam_left"], dtype=np.uint8))
                    step += 1
                    if episode_success or terminated or truncated or step >= max_steps:
                        break

            successes += int(episode_success)
            max_rewards.append(episode_max_reward)
            if frames:
                imageio.mimsave(
                    video_dir / f"{task_name}_episode{episode:02d}_{'success' if episode_success else 'fail'}.mp4",
                    frames,
                    fps=eval_fps,
                    quality=7,
                )
            print(
                f"[{task_name}] episode={episode + 1}/{args.episodes} "
                f"success={int(episode_success)} max_reward={episode_max_reward:.1f}",
                flush=True,
            )

        success_rate = successes / args.episodes
        rows.append(
            {
                "task": task_name,
                "episodes": args.episodes,
                "successes": successes,
                "success_rate": f"{success_rate:.4f}",
                "mean_max_reward": f"{np.mean(max_rewards):.4f}",
            }
        )
        env.close()

    with (output_dir / "summary.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        total_successes = sum(int(row["successes"]) for row in rows)
        total_episodes = sum(int(row["episodes"]) for row in rows)
        writer.writerow(
            {
                "task": "overall",
                "episodes": total_episodes,
                "successes": total_successes,
                "success_rate": f"{total_successes / total_episodes:.4f}",
                "mean_max_reward": "",
            }
        )
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
