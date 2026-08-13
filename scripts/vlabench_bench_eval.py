"""Evaluate a VLANeXt checkpoint with the official VLABench evaluator."""

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VLABENCH_REPO = PROJECT_ROOT / "third_party" / "VLABench"
VLABENCH_ROOT = VLABENCH_REPO / "VLABench"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VLABENCH_REPO))
os.environ.setdefault("VLABENCH_ROOT", str(VLABENCH_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

# These imports register the task and robot classes used by load_env.
import VLABench.robots  # noqa: F401
import VLABench.tasks  # noqa: F401
from VLABench.evaluation.evaluator import Evaluator
from VLABench.utils.utils import quaternion_to_euler

from src.evaluation.libero_bench.VLANeXt_utils import get_processor
from src.evaluation.libero_bench.robot_utils import get_action, get_model, set_seed_everywhere


DEFAULT_STATS = "/home/dm/datasets/vlabench_unified/meta/stats.json"


def _model_config(checkpoint, diffusion_steps):
    cfg = SimpleNamespace(
        eval=SimpleNamespace(
            finetuned_checkpoint=str(checkpoint),
            image_size=224,
        )
    )
    if diffusion_steps is not None:
        cfg.model = SimpleNamespace(diffusion_steps=int(diffusion_steps))
    return cfg


class VLANeXtVLABenchPolicy:
    """Adapts normalized VLANeXt action chunks to VLABench EE control."""

    control_mode = "ee"
    name = "VLANeXt"

    def __init__(
        self,
        checkpoint,
        action_stats_path,
        execute_steps=8,
        diffusion_steps=None,
        main_camera_index=2,
        wrist_camera_index=3,
        instruction_prefix="primitive: ",
    ):
        self.cfg = _model_config(checkpoint, diffusion_steps)
        self.model = get_model(self.cfg)
        self.processor = get_processor(self.cfg)
        self.execute_steps = int(execute_steps)
        self.main_camera_index = int(main_camera_index)
        self.wrist_camera_index = int(wrist_camera_index)
        self.instruction_prefix = instruction_prefix

        with open(action_stats_path, "r", encoding="utf-8") as handle:
            stats = json.load(handle)["action"]
        self.action_low = np.asarray(stats["q01"], dtype=np.float32)
        self.action_high = np.asarray(stats["q99"], dtype=np.float32)
        self.action_scale = np.maximum(self.action_high - self.action_low, 1e-6)
        self.reset()

    def reset(self):
        self.image_history = []
        self.wrist_history = []
        self.state_history = []
        self.action_history = []
        self.action_queue = collections.deque()
        if hasattr(self, "model"):
            self.model._latent_bridge_vlm_cache = None
            self.model._latent_bridge_cache_step = 0

    def _denormalize(self, action_chunk):
        action_chunk = np.clip(np.asarray(action_chunk, dtype=np.float32), -1.0, 1.0)
        return self.action_low + 0.5 * (action_chunk + 1.0) * self.action_scale

    def _instruction(self, instruction):
        instruction = "" if instruction is None else str(instruction).strip()
        if self.instruction_prefix and not instruction.lower().startswith(
            self.instruction_prefix.lower()
        ):
            instruction = f"{self.instruction_prefix}{instruction}"
        return instruction

    @staticmethod
    def _state_from_observation(observation):
        ee_state = np.asarray(observation["ee_state"], dtype=np.float32)
        robot_frame = np.asarray(observation["robot_frame"], dtype=np.float32)
        relative_position = ee_state[:3] - robot_frame
        euler = np.asarray(quaternion_to_euler(ee_state[3:7]), dtype=np.float32)
        gripper = np.asarray([ee_state[-1]], dtype=np.float32)
        return np.concatenate([relative_position, euler, gripper]).astype(np.float32)

    def predict(self, observation, **kwargs):
        del kwargs
        rgb = np.asarray(observation["rgb"])
        main_image = np.ascontiguousarray(rgb[self.main_camera_index].astype(np.uint8))
        wrist_image = np.ascontiguousarray(rgb[self.wrist_camera_index].astype(np.uint8))
        self.image_history.append(main_image)
        self.wrist_history.append(wrist_image)
        self.state_history.append(self._state_from_observation(observation))

        if not self.action_queue:
            model_observation = {
                "full_image": main_image,
                "full_image_wrist": wrist_image,
                "image_history": self.image_history,
                "image_history_wrist": self.wrist_history,
                "state_history": self.state_history,
                "action_history": self.action_history,
            }
            normalized_chunk = np.asarray(
                get_action(
                    self.cfg,
                    self.model,
                    model_observation,
                    self._instruction(observation.get("instruction")),
                    processor=self.processor,
                ),
                dtype=np.float32,
            )
            if normalized_chunk.ndim == 1:
                normalized_chunk = normalized_chunk[None]
            absolute_chunk = self._denormalize(normalized_chunk)
            for normalized, absolute in zip(
                normalized_chunk[: self.execute_steps],
                absolute_chunk[: self.execute_steps],
            ):
                self.action_queue.append((normalized, absolute))

        normalized_action, absolute_action = self.action_queue.popleft()
        self.action_history.append(normalized_action.copy())

        robot_frame = np.asarray(observation["robot_frame"], dtype=np.float32)
        target_position = absolute_action[:3] + robot_frame
        target_euler = absolute_action[3:6]
        gripper_open = bool(absolute_action[6] >= 0.5)
        gripper_state = np.full(2, 0.04 if gripper_open else 0.0, dtype=np.float32)
        return target_position, target_euler, gripper_state


class StepLimitedEvaluator(Evaluator):
    def __init__(self, *args, max_episode_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_episode_steps = max_episode_steps

    def evaluate_single_episode(self, *args, max_episode_length=200, **kwargs):
        if self.max_episode_steps is not None:
            max_episode_length = min(max_episode_length, self.max_episode_steps)
        return super().evaluate_single_episode(
            *args,
            max_episode_length=max_episode_length,
            **kwargs,
        )


def _load_track(track):
    path = VLABENCH_ROOT / "configs" / "evaluation" / "tracks" / f"{track}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown VLABench evaluation track: {path}")
    with path.open("r", encoding="utf-8") as handle:
        episodes = json.load(handle)
    return list(episodes), episodes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--action-stats", default=DEFAULT_STATS)
    parser.add_argument("--eval-track", default="track_1_in_distribution")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--n-episode", type=int, default=1)
    parser.add_argument("--execute-steps", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--main-camera-index", type=int, default=2)
    parser.add_argument("--wrist-camera-index", type=int, default=3)
    parser.add_argument("--instruction-prefix", default="primitive: ")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    tasks, episode_config = _load_track(args.eval_track)
    if args.tasks:
        missing = [task for task in args.tasks if task not in episode_config]
        if missing:
            raise ValueError(f"Tasks are not present in {args.eval_track}: {missing}")
        tasks = args.tasks
    n_episode = args.n_episode
    max_episode_steps = args.max_episode_steps
    if args.smoke:
        tasks = tasks[:1]
        n_episode = 1
        max_episode_steps = 1

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint.parent / f"vlabench_{checkpoint.stem}_{args.eval_track}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed_everywhere(args.seed)

    policy = VLANeXtVLABenchPolicy(
        checkpoint=checkpoint,
        action_stats_path=args.action_stats,
        execute_steps=args.execute_steps,
        diffusion_steps=args.diffusion_steps,
        main_camera_index=args.main_camera_index,
        wrist_camera_index=args.wrist_camera_index,
        instruction_prefix=args.instruction_prefix,
    )
    evaluator = StepLimitedEvaluator(
        tasks=tasks,
        n_episodes=n_episode,
        episode_config=episode_config,
        max_substeps=1,
        save_dir=str(output_dir),
        visulization=args.save_video,
        metrics=["success_rate", "intention_score", "progress_score"],
        max_episode_steps=max_episode_steps,
    )

    print(
        f"[setup] track={args.eval_track} tasks={len(tasks)} episodes={n_episode} "
        f"execute_steps={args.execute_steps} video={args.save_video}",
        flush=True,
    )
    result = evaluator.evaluate(policy)
    with (output_dir / "evaluation_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
