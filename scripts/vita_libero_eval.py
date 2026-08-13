import argparse
import os
import random
import sys
from pathlib import Path
import math

import imageio
import numpy as np
import tensorflow as tf
import torch
from hydra import compose, initialize_config_dir


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

VITA_ROOT = Path("/home/dm/QWENLA/VLANeXt_migration/VITA")
VLANEXT_ROOT = Path("/home/dm/QWENLA/VLANeXt_migration/VLANeXt")
if str(VLANEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(VLANEXT_ROOT))
if str(VITA_ROOT) not in sys.path:
    sys.path.insert(0, str(VITA_ROOT))

from flare.factory import get_policy_class  # noqa: E402
from flare.utils.checkpoints import load_model_weights  # noqa: E402
from flare.utils.dataset_utils import create_dataset_stats  # noqa: E402
from flare.utils.libero_tfds_dataset import ACTION_MAX, ACTION_MIN  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


TASK_LANGUAGE = "pick up the orange juice and place it in the basket"


def get_libero_env(task, resolution=256):
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    with tf.device("/CPU:0"):
        img = tf.image.encode_jpeg(img)
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
        img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
        return img.numpy()


def get_libero_image(obs, resize_size, obs_key="agentview_image"):
    img = obs[obs_key]
    img = img[::-1, ::-1]
    return resize_image(img, resize_size)


def save_rollout_video(rollout_images, idx, success, task_description, log_file, save_dir, fps=20):
    os.makedirs(save_dir, exist_ok=True)
    task_name = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:100]
    mp4_path = os.path.join(save_dir, f"episode={idx}--success={success}--task={task_name}.mp4")
    writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        writer.append_data(img)
    writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def patch_torch_load_for_libero():
    orig_load = torch.load

    def load_with_pickle(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = load_with_pickle


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def suite_key(task_suite_name: str) -> str:
    if "spatial" in task_suite_name:
        return "spatial"
    if "object" in task_suite_name:
        return "object"
    if "goal" in task_suite_name:
        return "goal"
    return "10"


def max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    return 400


def find_task_id(task_suite, task_language: str) -> int:
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        if task.language == task_language:
            return task_id

    print("Available tasks:")
    for task_id in range(task_suite.n_tasks):
        print(f"{task_id}: {task_suite.get_task(task_id).language}")
    raise ValueError(f"Task language not found: {task_language}")


def make_cfg(args):
    overrides = [
        "policy=vita",
        "task=libero_object_orange_juice_tfds",
        "session=vita_libero_eval",
        f"device={args.device}",
        "policy.flow_net.name=simple_flow_net",
        "val.num_episodes=0",
        "val.val_offline_freq=0",
        "val.val_online_freq=0",
        "wandb.enable=false",
    ]
    if args.image_keys:
        overrides.append(f"task.image_keys={args.image_keys}")
    elif args.single_image:
        overrides.append("task.image_keys=[observation.images.image]")

    with initialize_config_dir(
        config_dir=str(VITA_ROOT / "flare" / "configs"),
        version_base="1.3",
    ):
        return compose(config_name="default_policy", overrides=overrides)


def obs_to_batch(obs, image_keys, device):
    img = get_libero_image(obs, (256, 256), obs_key="agentview_image")
    gripper_state = np.clip(1 - (np.mean(np.abs(obs["robot0_gripper_qpos"])) / 0.04), 0.0, 1.0)
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()), [gripper_state])
    ).astype(np.float32)

    batch = {
        "observation.images.image": torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0),
        "observation.state": torch.from_numpy(state).float().unsqueeze(0),
    }

    if "observation.images.wrist_image" in image_keys:
        wrist = get_libero_image(obs, (256, 256), obs_key="robot0_eye_in_hand_image")
        batch["observation.images.wrist_image"] = (
            torch.from_numpy(wrist).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        )

    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def denormalize_action(action_norm: np.ndarray, task_suite_name: str) -> np.ndarray:
    key = suite_key(task_suite_name)
    action_min = ACTION_MIN[key]
    action_max = ACTION_MAX[key]
    action = action_norm.copy()
    action[:6] = (action[:6] + 1.0) / 2.0 * (action_max - action_min) + action_min
    action[6] = 1.0 if action[6] > 0 else -1.0
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="VITA checkpoint directory, e.g. step_0000005000")
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--task-language", default=TASK_LANGUAGE)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single-image", action="store_true")
    parser.add_argument(
        "--image-keys",
        default=None,
        help=(
            "Hydra list of image keys, for example "
            "'[observation.images.image,observation.images.wrist_image]'"
        ),
    )
    parser.add_argument("--save-video-every", type=int, default=1)
    args = parser.parse_args()

    patch_torch_load_for_libero()
    set_seed(args.seed)

    cfg = make_cfg(args)
    device = torch.device(args.device)

    _, stats = create_dataset_stats(cfg)
    policy_cls = get_policy_class(cfg.policy.name)
    policy = policy_cls(cfg, stats)
    load_model_weights(policy, args.checkpoint, device)
    policy.to(device)
    policy.eval()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task_id = args.task_id if args.task_id is not None else find_task_id(task_suite, args.task_language)
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, resolution=256)

    checkpoint = Path(args.checkpoint)
    eval_dir = checkpoint.parent / f"vita_libero_{args.task_suite}_task{task_id}_{checkpoint.name}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_path = eval_dir / "log.txt"

    successes = 0
    max_steps = max_steps_for_suite(args.task_suite)
    print(f"Task {task_id}: {task_description}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Saving eval outputs to: {eval_dir}")

    with log_path.open("w") as log_file:
        log_file.write(f"Task {task_id}: {task_description}\n")
        for episode_idx in range(args.num_trials):
            policy.reset()
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
            replay_images = []
            done = False

            for t in range(max_steps + args.num_steps_wait):
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action())
                    continue

                replay_images.append(get_libero_image(obs, (256, 256), obs_key="agentview_image"))
                batch = obs_to_batch(obs, cfg.task.image_keys, device)

                with torch.inference_mode():
                    action_norm = policy.select_action(batch).squeeze(0).detach().cpu().numpy()

                action = denormalize_action(action_norm, args.task_suite)
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    break

            success = bool(done)
            successes += int(success)
            sr = successes / float(episode_idx + 1)
            print(f"episode {episode_idx + 1}/{args.num_trials}: success={success} sr={sr:.3f}")
            log_file.write(f"episode {episode_idx + 1}: success={success} sr={sr:.3f}\n")
            log_file.flush()

            if args.save_video_every > 0 and episode_idx % args.save_video_every == 0:
                save_rollout_video(
                    replay_images,
                    episode_idx + 1,
                    success=success,
                    task_description=task_description,
                    log_file=log_file,
                    save_dir=str(eval_dir),
                    fps=20,
                )

        final_sr = successes / float(args.num_trials) if args.num_trials else 0.0
        print(f"Final success rate: {successes}/{args.num_trials} = {final_sr * 100:.2f}%")
        log_file.write(f"Final success rate: {successes}/{args.num_trials} = {final_sr * 100:.2f}%\n")

    env.close()


if __name__ == "__main__":
    main()
