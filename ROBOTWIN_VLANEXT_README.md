# RoboTwin + VLANeXt Local Notes

This note records the local RoboTwin integration for VLANeXt. It is separate from the original project README.

## Project Layout

VLANeXt side:

- `scripts/train.py`: main training entry. It now supports `dataset_name: "robotwin"`.
- `src/datasets/robotwin_act.py`: RoboTwin HDF5 dataset adapter.
- `config/robotwin_train_no_progress_film_config.yaml`: full RoboTwin training, Progress-FiLM disabled.
- `config/robotwin_train_progress_film_config.yaml`: full RoboTwin training, Progress-FiLM enabled.
- `config/robotwin_smoke_no_progress_film_config.yaml`: short smoke test, Progress-FiLM disabled.
- `config/robotwin_smoke_progress_film_config.yaml`: short smoke test, Progress-FiLM enabled.

RoboTwin side:

- `/home/dm/QWENLA/VLANeXt_migration/RoboTwin/precollected_dataset/dataset`: RoboTwin data root.
- `/home/dm/QWENLA/VLANeXt_migration/RoboTwin/policy/VLANeXt/deploy_policy.py`: VLANeXt policy adapter for RoboTwin eval.
- `/home/dm/QWENLA/VLANeXt_migration/RoboTwin/policy/VLANeXt/deploy_policy.yml`: eval config.
- `/home/dm/QWENLA/VLANeXt_migration/RoboTwin/policy/VLANeXt/eval_socket.sh`: socket eval launcher.

## Data Layout

Expected RoboTwin data path:

```text
/home/dm/QWENLA/VLANeXt_migration/RoboTwin/precollected_dataset/dataset/<task>/aloha-agilex_clean_50/
```

Each task directory should contain data such as:

```text
data/
instructions/
video/
_traj_data/
```

Current adapter assumes:

- robot setting: `aloha-agilex`
- demo split: `clean_50`
- action/state dimension: `14`
- cameras: `head_camera`, `left_camera`, `right_camera`
- observation state: `joint_action/vector[t]`
- action target: `joint_action/vector[t + 1]`

The `t -> t + 1` target is intentional and matches RoboTwin-style policy training where the model observes the current robot state and predicts the next qpos/action.

## Training Configs

| Config | Purpose | Progress Head | Progress-FiLM | Save Interval |
|---|---:|---:|---:|---:|
| `config/robotwin_train_no_progress_film_config.yaml` | full ablation training | off | off | 1000 |
| `config/robotwin_train_progress_film_config.yaml` | full Progress-FiLM training | on | on | 1000 |
| `config/robotwin_smoke_no_progress_film_config.yaml` | 2-step sanity check | off | off | 1000 |
| `config/robotwin_smoke_progress_film_config.yaml` | 2-step sanity check | on | on | 1000 |

The no-progress config is named `VLANeXt_robotwin_ablation_nofilm_aloha_clean50` to avoid W&B displaying a misleading run name containing `progress`.

## Train Commands

No Progress-FiLM:

```bash
cd /home/dm/QWENLA/VLANeXt_migration/VLANeXt
conda activate VLANeXt

CUDA_VISIBLE_DEVICES=1 python -m scripts.train \
  --config config/robotwin_train_no_progress_film_config.yaml
```

Progress-FiLM:

```bash
cd /home/dm/QWENLA/VLANeXt_migration/VLANeXt
conda activate VLANeXt

CUDA_VISIBLE_DEVICES=1 python -m scripts.train \
  --config config/robotwin_train_progress_film_config.yaml
```

Smoke tests:

```bash
cd /home/dm/QWENLA/VLANeXt_migration/VLANeXt
conda activate VLANeXt

CUDA_VISIBLE_DEVICES=1 python -m scripts.train \
  --config config/robotwin_smoke_no_progress_film_config.yaml

CUDA_VISIBLE_DEVICES=1 python -m scripts.train \
  --config config/robotwin_smoke_progress_film_config.yaml
```

## Eval

Use socket eval for RoboTwin. The reason is environment separation:

- `RoboTwin` env has simulator dependencies such as `sapien`, `mplib`, and CuRobo.
- `VLANeXt` env has model dependencies such as `transformers`, Qwen, and training code.

Example:

```bash
cd /home/dm/QWENLA/VLANeXt_migration/RoboTwin
conda activate RoboTwin

bash policy/VLANeXt/eval_socket.sh \
  click_alarmclock \
  smoke_clean \
  robotwin_progress_film_1000 \
  /home/dm/QWENLA/VLANeXt_migration/VLANeXt/checkpoints/<run_name>/<suite>/checkpoint_1000.pt \
  0 \
  1 \
  1 \
  unseen \
  5 \
  8 \
  VLANeXt
```

Arguments:

| Position | Meaning |
|---:|---|
| 1 | RoboTwin task name |
| 2 | RoboTwin task config |
| 3 | ckpt setting / eval output tag |
| 4 | absolute checkpoint path |
| 5 | seed |
| 6 | GPU id |
| 7 | number of eval episodes |
| 8 | instruction type |
| 9 | diffusion steps |
| 10 | exec horizon |
| 11 | server conda env, usually `VLANeXt` |

## Important Notes

- RoboTwin actions are 14D qpos targets, not LIBERO 7D end-effector actions.
- RoboTwin eval denormalizes model output back to qpos and calls RoboTwin with `action_type="qpos"`.
- LIBERO eval scripts cannot directly evaluate RoboTwin checkpoints.
- The Progress-FiLM disabled configs really set:

```yaml
use_progress_head: false
use_progress_film: false
progress_loss_weight: 0.0
```

- If W&B shows progress-related curves, check the run config and run name. Old runs may have misleading names, but the local no-progress summaries only contain action/DCT losses.

## Verified Smoke Results

Both RoboTwin smoke configs have run through 2 training steps successfully:

- no Progress-FiLM: completed 2/2 steps.
- with Progress-FiLM: completed 2/2 steps.

This verifies the data adapter, multi-image collator, model forward path, and checkpoint save path at a basic level.
