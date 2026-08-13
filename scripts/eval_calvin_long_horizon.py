#!/usr/bin/env python3
"""Evaluate VLANeXt with the official CALVIN ABC->D long-horizon protocol."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from itertools import product
import json
import os
from pathlib import Path
import sys
import time
import types

import hydra
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.eval_calvin_smoke import (
    DEFAULT_CALVIN_ROOT,
    DEFAULT_ENV_DIR,
    PROJECT_ROOT,
    _build_eval_cfg,
    _configure_import_paths,
    _make_observation,
    _video_frame,
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / (
    "checkpoints/VLANeXt_vita_hiermq54/"
    "actioneffect_v3_auxonly_mainview_q4_cross0_global24_nodct_bs16_"
    "calvin_abc_d/checkpoint_64000.pt"
)


@contextmanager
def _temp_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _load_sequence_module(calvin_root: Path):
    """Load CALVIN's canonical sequence generator without its training stack."""
    models_root = str(calvin_root / "calvin_models")
    if models_root not in sys.path:
        sys.path.insert(0, models_root)
    stub = types.ModuleType("calvin_agent.evaluation.utils")
    stub.temp_seed = _temp_seed
    sys.modules[stub.__name__] = stub
    from calvin_agent.evaluation import multistep_sequences

    return multistep_sequences


def _official_sequences(calvin_root: Path, count: int):
    module = _load_sequence_module(calvin_root)
    possible_conditions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
    }
    combinations = product(*possible_conditions.values())
    combinations = filter(
        lambda values: values.count("table") in (1, 2)
        and values.count("slider_right") < 2
        and values.count("slider_left") < 2,
        combinations,
    )
    initial_states = [
        dict(zip(possible_conditions.keys(), values)) for values in combinations
    ]
    per_state = list(map(len, np.array_split(range(count), len(initial_states))))

    results = []
    with _temp_seed(0):
        for index, (state, state_count) in enumerate(zip(initial_states, per_state)):
            sequences = module.get_sequences_for_state2((state, state_count, index))
            results.extend((state, tuple(sequence.tolist())) for sequence in sequences)
        np.random.shuffle(results)
    return results


def _fnv1_32(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value = (value * 16777619) & 0xFFFFFFFF
        value ^= byte
    return value


def _env_state(initial_condition: dict) -> tuple[np.ndarray, np.ndarray]:
    """Exact state construction used by CALVIN's official evaluator."""
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
    block_rotation = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    slider_left = np.array([-0.240851662, 0.0924044687, 0.459990009])
    slider_right = np.array([0.070341533, 0.0924044687, 0.459990009])
    table = [
        np.array([0.0500000896, -0.120000177, 0.459990009]),
        np.array([0.229995412, -0.119995140, 0.459990010]),
    ]
    seed = _fnv1_32(str(initial_condition.values()))
    with _temp_seed(seed):
        np.random.shuffle(table)
        scene_obs = np.zeros(24, dtype=np.float64)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]

        locations = {
            "slider_right": slider_right,
            "slider_left": slider_left,
        }
        scene_obs[6:9] = locations.get(initial_condition["red_block"], table[0])
        scene_obs[11] = np.random.uniform(*block_rotation)
        if initial_condition["blue_block"] in locations:
            scene_obs[12:15] = locations[initial_condition["blue_block"]]
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = table[1]
        else:
            scene_obs[12:15] = table[0]
        scene_obs[17] = np.random.uniform(*block_rotation)
        scene_obs[18:21] = locations.get(initial_condition["pink_block"], table[1])
        scene_obs[23] = np.random.uniform(*block_rotation)
    return robot_obs, scene_obs


def _metrics(records: list[dict]) -> dict:
    if not records:
        return {"num_sequences": 0, "average_successful_subtasks": 0.0}
    counts = np.asarray([record["successful_subtasks"] for record in records])
    metrics = {
        "num_sequences": len(records),
        "average_successful_subtasks": float(counts.mean()),
    }
    for length in range(1, 6):
        metrics[f"success_rate_{length}"] = float(np.mean(counts >= length))
    return metrics


def _save_results(output_dir: Path, records: list[dict]) -> None:
    metrics = _metrics(records)
    payload = {"metrics": metrics, "sequences": records}
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    lines = ["index\tsuccessful_subtasks\ttasks\tvideo"]
    for record in records:
        lines.append(
            f"{record['index']}\t{record['successful_subtasks']}\t"
            f"{'|'.join(record['tasks'])}\t{record['video']}"
        )
    (output_dir / "results.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_header = [
        "num_sequences",
        "average_successful_subtasks",
        "success_rate_1",
        "success_rate_2",
        "success_rate_3",
        "success_rate_4",
        "success_rate_5",
    ]
    summary_values = [
        str(metrics["num_sequences"]),
        f"{metrics['average_successful_subtasks']:.6f}",
        *(f"{metrics.get(f'success_rate_{length}', 0.0):.6f}" for length in range(1, 6)),
    ]
    (output_dir / "summary.tsv").write_text(
        "\t".join(summary_header) + "\n" + "\t".join(summary_values) + "\n",
        encoding="utf-8",
    )


def evaluate(args: argparse.Namespace) -> Path:
    calvin_root = args.calvin_root.resolve()
    env_dir = args.env_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    _configure_import_paths(calvin_root)
    sequences = _official_sequences(calvin_root, args.num_sequences)

    from calvin_env.envs.play_table_env import get_env
    from src.evaluation.libero_bench.VLANeXt_utils import get_processor, get_vla
    from src.evaluation.libero_bench.robot_utils import get_action

    if args.output_dir is None:
        step = checkpoint.stem.split("_")[-1]
        args.output_dir = PROJECT_ROOT / f"calvin_abc_d_long_horizon_ckpt{step}"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_path = output_dir / "results.json"
    records = []
    if existing_path.is_file() and not args.overwrite:
        records = json.loads(existing_path.read_text(encoding="utf-8")).get(
            "sequences", []
        )
    start_index = max(args.start_index, len(records))

    cfg = _build_eval_cfg(checkpoint, args.diffusion_steps)
    print(f"[setup] checkpoint={checkpoint}", flush=True)
    print(
        f"[setup] sequences={args.num_sequences} start={start_index} "
        f"max_steps={args.max_steps} exec_horizon={args.exec_horizon}",
        flush=True,
    )
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
    env = get_env(
        env_dir,
        obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []},
        show_gui=False,
    )

    try:
        for sequence_index in range(start_index, args.num_sequences):
            initial_condition, tasks = sequences[sequence_index]
            robot_obs, scene_obs = _env_state(initial_condition)
            obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            save_video = args.save_video_every > 0 and (
                sequence_index % args.save_video_every == 0
            )
            video_name = f"sequence_{sequence_index:04d}.mp4" if save_video else ""
            writer = None
            if save_video:
                writer = imageio.get_writer(
                    output_dir / video_name,
                    fps=args.video_fps,
                    codec="libx264",
                    macro_block_size=1,
                )

            successful_subtasks = 0
            try:
                for subtask_index, task in enumerate(tasks):
                    instruction = str(annotations[task][0])
                    start_info = env.get_info()
                    state_history: list[np.ndarray] = []
                    action_buffer: list[np.ndarray] = []
                    if hasattr(model, "_latent_bridge_vlm_cache"):
                        model._latent_bridge_vlm_cache = None
                        model._latent_bridge_cache_step = 0

                    success = False
                    for step in range(args.max_steps):
                        state_history.append(
                            np.asarray(obs["robot_obs"][:7], dtype=np.float32).copy()
                        )
                        if writer is not None:
                            writer.append_data(
                                _video_frame(obs, instruction, step, success=False)
                            )
                        if not action_buffer:
                            model_obs = _make_observation(obs, state_history)
                            action_chunk = np.asarray(
                                get_action(
                                    cfg,
                                    model,
                                    model_obs,
                                    instruction,
                                    processor=processor,
                                ),
                                dtype=np.float32,
                            )
                            if action_chunk.ndim == 1:
                                action_chunk = action_chunk[None, :]
                            count = min(args.exec_horizon, len(action_chunk))
                            action_buffer = [
                                action.copy() for action in action_chunk[:count]
                            ]
                        action = np.asarray(action_buffer.pop(0), dtype=np.float32)
                        action[:6] = np.clip(action[:6], -1.0, 1.0)
                        action[6] = 1.0 if action[6] > 0 else -1.0
                        obs, _, _, current_info = env.step(action)
                        achieved = task_oracle.get_task_info_for_set(
                            start_info, current_info, {task}
                        )
                        if achieved:
                            success = True
                            successful_subtasks += 1
                            if writer is not None:
                                writer.append_data(
                                    _video_frame(obs, instruction, step + 1, success=True)
                                )
                            break
                    print(
                        f"[sequence {sequence_index + 1}/{args.num_sequences}] "
                        f"subtask={subtask_index + 1}/5 task={task} success={success}",
                        flush=True,
                    )
                    if not success:
                        break
            finally:
                if writer is not None:
                    writer.close()

            records.append(
                {
                    "index": sequence_index,
                    "initial_condition": initial_condition,
                    "tasks": list(tasks),
                    "successful_subtasks": successful_subtasks,
                    "video": video_name,
                }
            )
            _save_results(output_dir, records)
            print(f"[metrics] {_metrics(records)}", flush=True)
    finally:
        env.close()
        env.ownsPhysicsClient = False
        env.cid = -1

    print(f"[done] results={output_dir}", flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calvin-root", type=Path, default=DEFAULT_CALVIN_ROOT)
    parser.add_argument("--env-dir", type=Path, default=DEFAULT_ENV_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--exec-horizon", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--save-video-every", type=int, default=20)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_sequences <= 0 or args.max_steps <= 0 or args.exec_horizon <= 0:
        parser.error("sequence count, max steps, and exec horizon must be positive")
    return args


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)
    evaluate(parse_args())
