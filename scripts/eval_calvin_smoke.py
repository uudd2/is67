#!/usr/bin/env python3
"""Run a small VLANeXt smoke evaluation in the CALVIN ABC->D environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import hydra
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALVIN_ROOT = Path(
    "/media/dm/Elements/VLANeXt_migration/data/calvin/CALVIN_eval/calvin"
)
DEFAULT_ENV_DIR = Path(
    "/media/dm/Elements/VLANeXt_migration/data/calvin/CALVIN_eval/"
    "task_ABC_D/validation"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / (
    "checkpoints/VLANeXt_vita_hiermq54/"
    "actioneffect_v3_auxonly_mainview_q4_cross0_global24_nodct_bs16_"
    "calvin_abc_d/checkpoint_32000.pt"
)
DEFAULT_TASKS = ("open_drawer", "move_slider_left", "turn_on_led")


def _configure_import_paths(calvin_root: Path) -> None:
    for path in (PROJECT_ROOT, calvin_root / "calvin_env"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _build_eval_cfg(checkpoint: Path, diffusion_steps: int):
    return SimpleNamespace(
        eval=SimpleNamespace(finetuned_checkpoint=str(checkpoint)),
        model=SimpleNamespace(diffusion_steps=int(diffusion_steps)),
    )


def _initial_state(task: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a valid deterministic CALVIN-D state for a smoke task."""
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ],
        dtype=np.float64,
    )
    scene_obs = np.zeros(24, dtype=np.float64)

    # Benchmark-compatible starting preconditions for the selected smoke tasks.
    if task == "close_drawer":
        scene_obs[1] = 0.22
    if task == "move_slider_right":
        scene_obs[0] = 0.28
    if task == "turn_off_lightbulb":
        scene_obs[3] = 0.088
        scene_obs[4] = 1.0
    if task == "turn_off_led":
        scene_obs[5] = 1.0

    block_slider_left = np.array([-0.240851662, 0.0924044687, 0.459990009])
    block_slider_right = np.array([0.070341533, 0.0924044687, 0.459990009])
    block_table = np.array([0.0500000896, -0.120000177, 0.459990009])
    scene_obs[6:9] = block_table
    scene_obs[12:15] = block_slider_left
    scene_obs[18:21] = block_slider_right

    rng = np.random.default_rng(seed)
    rotation_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    scene_obs[11] = rng.uniform(*rotation_range)
    scene_obs[17] = rng.uniform(*rotation_range)
    scene_obs[23] = rng.uniform(*rotation_range)
    return robot_obs, scene_obs


def _video_frame(obs: dict, instruction: str, step: int, success: bool) -> np.ndarray:
    static = np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8)
    wrist = np.asarray(obs["rgb_obs"]["rgb_gripper"], dtype=np.uint8)
    wrist = cv2.resize(wrist, (static.shape[1], static.shape[0]), interpolation=cv2.INTER_AREA)
    frame = np.concatenate([static, wrist], axis=1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 48), (0, 0, 0), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    label = f"step {step} | {'SUCCESS' if success else 'running'} | {instruction}"
    cv2.putText(
        frame,
        label,
        (8, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def _make_observation(obs: dict, state_history: list[np.ndarray]) -> dict:
    static = np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8)
    wrist = np.asarray(obs["rgb_obs"]["rgb_gripper"], dtype=np.uint8)
    return {
        "full_image": static,
        "full_image_wrist": wrist,
        "image_history": [static],
        "image_history_wrist": [wrist],
        "state_history": state_history,
        "action_history": [],
    }


def _write_summary(output_dir: Path, records: list[dict]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    lines = ["task\tinstruction\tsuccess\tsteps\tvideo"]
    for record in records:
        lines.append(
            "\t".join(
                [
                    record["task"],
                    record["instruction"],
                    str(int(record["success"])),
                    str(record["steps"]),
                    record["video"],
                ]
            )
        )
    (output_dir / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> Path:
    calvin_root = args.calvin_root.resolve()
    env_dir = args.env_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not (env_dir / ".hydra" / "merged_config.yaml").is_file():
        raise FileNotFoundError(f"CALVIN-D environment config not found: {env_dir}")

    _configure_import_paths(calvin_root)
    from calvin_env.envs.play_table_env import get_env
    from src.evaluation.libero_bench.VLANeXt_utils import get_processor, get_vla
    from src.evaluation.libero_bench.robot_utils import get_action

    output_dir = args.output_dir
    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / f"calvin_abc_d_smoke_ckpt{checkpoint.stem.split('_')[-1]}_{stamp}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_eval_cfg(checkpoint, args.diffusion_steps)
    print(f"[setup] loading checkpoint: {checkpoint}", flush=True)
    model = get_vla(cfg)
    processor = get_processor(cfg)

    task_cfg = OmegaConf.load(
        calvin_root / "calvin_env" / "conf" / "tasks" / "new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    annotations = OmegaConf.load(
        calvin_root
        / "calvin_models"
        / "conf"
        / "annotations"
        / "new_playtable_validation.yaml"
    )
    obs_space = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}
    env = get_env(env_dir, obs_space=obs_space, show_gui=False)

    records = []
    try:
        for task_index, task in enumerate(args.tasks):
            if task not in annotations:
                raise KeyError(f"Unknown CALVIN task: {task}")
            instruction = str(annotations[task][0])
            robot_obs, scene_obs = _initial_state(task, args.seed + task_index)
            obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            start_info = env.get_info()
            state_history: list[np.ndarray] = []
            action_buffer: list[np.ndarray] = []
            frames = []
            success = False

            print(f"[task {task_index + 1}/{len(args.tasks)}] {task}: {instruction}", flush=True)
            for step in range(args.max_steps):
                state_history.append(
                    np.asarray(obs["robot_obs"][:7], dtype=np.float32).copy()
                )
                frames.append(_video_frame(obs, instruction, step, success=False))

                if not action_buffer:
                    model_obs = _make_observation(obs, state_history)
                    action_chunk = np.asarray(
                        get_action(cfg, model, model_obs, instruction, processor=processor),
                        dtype=np.float32,
                    )
                    if action_chunk.ndim == 1:
                        action_chunk = action_chunk[None, :]
                    count = min(args.exec_horizon, len(action_chunk))
                    action_buffer = [action.copy() for action in action_chunk[:count]]

                action = np.asarray(action_buffer.pop(0), dtype=np.float32)
                action[:6] = np.clip(action[:6], -1.0, 1.0)
                action[6] = 1.0 if action[6] > 0 else -1.0
                obs, _, _, current_info = env.step(action)
                achieved = task_oracle.get_task_info_for_set(
                    start_info, current_info, {task}
                )
                if achieved:
                    success = True
                    frames.append(_video_frame(obs, instruction, step + 1, success=True))
                    break

            video_name = f"{task_index:02d}_{task}_{'success' if success else 'fail'}.mp4"
            video_path = output_dir / video_name
            imageio.mimsave(video_path, frames, fps=args.video_fps, macro_block_size=1)
            record = {
                "task": task,
                "instruction": instruction,
                "success": success,
                "steps": step + 1,
                "video": video_name,
            }
            records.append(record)
            _write_summary(output_dir, records)
            print(
                f"[result] {task}: success={success} steps={step + 1} video={video_path}",
                flush=True,
            )
    finally:
        env.close()
        # CALVIN's destructor otherwise disconnects the already closed client again.
        env.ownsPhysicsClient = False
        env.cid = -1

    print(f"[done] results: {output_dir}", flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calvin-root", type=Path, default=DEFAULT_CALVIN_ROOT)
    parser.add_argument("--env-dir", type=Path, default=DEFAULT_ENV_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--exec-horizon", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.exec_horizon <= 0:
        parser.error("--exec-horizon must be positive")
    return args


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)
    evaluate(parse_args())
