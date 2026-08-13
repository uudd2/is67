#!/usr/bin/env python3
import argparse
import contextlib
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


VLANEXT_ROOT = Path("/home/dm/QWENLA/VLANeXt_migration/VLANeXt")
ROBOTWIN_ROOT = Path("/home/dm/QWENLA/VLANeXt_migration/RoboTwin")


def add_paths():
    for path in (
        ROBOTWIN_ROOT,
        ROBOTWIN_ROOT / "policy",
        ROBOTWIN_ROOT / "description" / "utils",
        VLANEXT_ROOT,
    ):
        sys.path.insert(0, str(path))


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def discover_tasks(setting):
    dataset_root = ROBOTWIN_ROOT / "precollected_dataset" / "dataset"
    tasks = []
    for path in sorted(dataset_root.glob(f"*/{setting}")):
        if path.is_dir():
            tasks.append(path.parent.name)
    if not tasks:
        raise RuntimeError(f"No tasks found for setting={setting} under {dataset_root}")
    return tasks


def get_embodiment_config(robot_file):
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_task_args(task_name, task_config, ckpt_setting, output_dir, force_clear_cache=True):
    from envs import CONFIGS_PATH
    from script.eval_policy import get_camera_config

    with open(ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["policy_name"] = "VLANeXt"
    args["eval_mode"] = True
    if force_clear_cache:
        args["clear_cache_freq"] = 1

    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)

    def get_embodiment_file(embodiment_type):
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file for {embodiment_type}")
        return robot_file

    with open(Path(CONFIGS_PATH) / "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_config_all = yaml.safe_load(f)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config_all[head_camera_type]["h"]
    args["head_camera_w"] = camera_config_all[head_camera_type]["w"]

    embodiment_type = args.get("embodiment")
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    video_size = None
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    if args.get("eval_video_log", False):
        camera_config = get_camera_config(head_camera_type)
        video_size = f"{camera_config['w']}x{camera_config['h']}"
        args["eval_video_save_dir"] = task_dir

    return args, video_size, task_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--robotwin-setting", default="aloha-agilex_clean_50")
    parser.add_argument("--ckpt-setting", default="robotwin_abs_ckpt6000_video1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--exec-horizon", type=int, default=8)
    parser.add_argument("--abs-qpos-step-clip-norm", type=float, default=0.10)
    parser.add_argument("--flip-rgb", action="store_true", help="Swap R/B channels before feeding images to VLANeXt.")
    parser.add_argument("--tasks", nargs="*")
    return parser.parse_args()


def main():
    args_cli = parse_args()
    add_paths()
    os.chdir(ROBOTWIN_ROOT)

    from envs import CONFIGS_PATH
    from script.eval_policy import class_decorator, eval_policy
    from policy.VLANeXt import deploy_policy

    if args_cli.flip_rgb:
        def _flipped_rgb_from_observation(observation, camera):
            rgb = observation["observation"][camera]["rgb"].astype(np.uint8)
            return rgb[..., ::-1].copy()

        deploy_policy._rgb_from_observation = _flipped_rgb_from_observation

    get_model = deploy_policy.get_model

    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = args_cli.tasks or discover_tasks(args_cli.robotwin_setting)

    first_task_args, _, _ = build_task_args(
        tasks[0],
        args_cli.task_config,
        args_cli.ckpt_setting,
        output_dir,
    )
    usr_args = {
        "task_name": tasks[0],
        "task_config": args_cli.task_config,
        "ckpt_setting": args_cli.ckpt_setting,
        "checkpoint_path": args_cli.checkpoint,
        "seed": args_cli.seed,
        "policy_name": "VLANeXt",
        "instruction_type": args_cli.instruction_type,
        "test_num": args_cli.test_num,
        "diffusion_steps": args_cli.diffusion_steps,
        "exec_horizon": args_cli.exec_horizon,
        "abs_qpos_step_clip_norm": args_cli.abs_qpos_step_clip_norm,
        "device": "cuda",
        "left_arm_dim": len(first_task_args["left_embodiment_config"]["arm_joints_name"][0]),
        "right_arm_dim": len(first_task_args["right_embodiment_config"]["arm_joints_name"][1]),
    }

    print(f"[setup] checkpoint={args_cli.checkpoint}")
    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] tasks={len(tasks)} test_num={args_cli.test_num}")
    print(f"[setup] instruction_type={args_cli.instruction_type} diffusion_steps={args_cli.diffusion_steps} exec_horizon={args_cli.exec_horizon}")
    print(f"[setup] abs_qpos_step_clip_norm={args_cli.abs_qpos_step_clip_norm}")
    print(f"[setup] flip_rgb={args_cli.flip_rgb}")

    model = get_model(usr_args)

    summary_path = output_dir / "summary.tsv"
    with open(summary_path, "w", encoding="utf-8") as summary:
        summary.write("task\tsuccess\ttotal\trate_percent\tvideo\tlog\n")

    total_success = 0
    total_trials = 0
    for idx, task in enumerate(tasks, start=1):
        task_log = log_dir / f"{task}.log"
        with open(task_log, "w", encoding="utf-8") as lf, contextlib.redirect_stdout(Tee(sys.stdout, lf)), contextlib.redirect_stderr(Tee(sys.stderr, lf)):
            print(f"\n[task {idx}/{len(tasks)}] {task}")
            task_args, video_size, task_dir = build_task_args(
                task,
                args_cli.task_config,
                args_cli.ckpt_setting,
                output_dir,
            )
            task_env = class_decorator(task)
            st_seed = 100000 * (1 + args_cli.seed)
            _, success = eval_policy(
                task,
                task_env,
                task_args,
                model,
                st_seed,
                test_num=args_cli.test_num,
                video_size=video_size,
                instruction_type=args_cli.instruction_type,
            )
            total = args_cli.test_num
            rate = 100.0 * success / total if total else 0.0
            video = task_dir / "episode0.mp4"
            print(f"[task done] {task}: {success}/{total} {rate:.1f}% video={video if video.exists() else 'NA'}")

        total_success += success
        total_trials += args_cli.test_num
        video = output_dir / task / "episode0.mp4"
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(f"{task}\t{success}\t{args_cli.test_num}\t{100.0 * success / args_cli.test_num:.1f}\t{video if video.exists() else ''}\t{task_log}\n")

    overall = 100.0 * total_success / total_trials if total_trials else 0.0
    report_path = output_dir / "summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RoboTwin One-Episode Eval\n\n")
        f.write(f"- time: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- checkpoint: `{args_cli.checkpoint}`\n")
        f.write(f"- tasks: {len(tasks)}\n")
        f.write(f"- trials/task: {args_cli.test_num}\n")
        f.write(f"- success: {total_success}/{total_trials}\n")
        f.write(f"- success_rate: {overall:.2f}%\n")
        f.write(f"- summary_tsv: `{summary_path}`\n")

    print(f"[done] success={total_success}/{total_trials} rate={overall:.2f}%")
    print(f"[done] summary={summary_path}")
    print(f"[done] videos={output_dir}")


if __name__ == "__main__":
    main()
