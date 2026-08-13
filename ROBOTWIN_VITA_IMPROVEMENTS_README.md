# RoboTwin + VITA Improvement Notes

This note records the local VITA changes for RoboTwin experiments. It is separate from the original VITA and RoboTwin READMEs.

## Goal

The current work explores whether VITA can perform better on RoboTwin by improving multi-view visual fusion and visual regularization.

The main changes are:

- RoboTwin HDF5 dataset support in VITA.
- Frozen DINOv3 visual observer.
- Top-camera guided cross-attention for left/right views.
- Learnable image-conditioned black-ball augmentation.
- Socket-based RoboTwin eval integration.

## Repositories

VITA:

```text
/home/dm/QWENLA/VLANeXt_migration/VITA
```

RoboTwin:

```text
/home/dm/QWENLA/VLANeXt_migration/RoboTwin
```

DINOv3 local repo:

```text
/home/dm/QWENLA/VLANeXt_migration/dinov3
```

DINOv3 local weights:

```text
/home/dm/QWENLA/VLANeXt_migration/dino weight/dino3/
```

Recommended default weight:

```text
dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

## VITA Code Changes

Main changed files:

```text
/home/dm/QWENLA/VLANeXt_migration/VITA/flare/policies/vita/vita_policy.py
/home/dm/QWENLA/VLANeXt_migration/VITA/flare/policies/observers/dinov3_observer.py
/home/dm/QWENLA/VLANeXt_migration/VITA/flare/models/learnable_occlusion.py
/home/dm/QWENLA/VLANeXt_migration/VITA/flare/configs/policy/vita.yaml
```

## Frozen DINOv3 Observer

The original VITA observer used ResNet18 global features:

```text
image -> ResNet18 -> 512-d feature
```

The new DINOv3 observer supports:

```text
image -> frozen DINOv3 -> cls/patch tokens
```

For `dinov3_vits16`, the feature dimension is:

```text
384
```

With 3 views and 14-d robot state:

```text
3 * 384 + 14 = 1166
```

This is projected by VITA:

```text
obs_encoder: Linear(1166 -> 512)
```

The DINOv3 backbone is frozen by default:

```bash
policy.observer.freeze=true
```

## Multi-View Fusion

Two fusion modes are supported.

### Pool Concat

This is the light baseline:

```text
top image   -> DINOv3 -> global feature
left image  -> DINOv3 -> global feature
right image -> DINOv3 -> global feature

concat [top, left, right, robot_state]
```

Use:

```bash
policy.observer.fusion=pool_concat
```

### Top Cross-Attention

This is the current preferred RoboTwin fusion:

```text
top image   -> DINOv3 patch tokens
left image  -> DINOv3 patch tokens
right image -> DINOv3 patch tokens

left tokens  cross-attend to top tokens
right tokens cross-attend to top tokens

pool top / left / right
concat with robot state
```

Rationale:

- top camera provides global layout;
- left/right cameras provide local manipulator views;
- left/right should use top as global context, rather than all views being blindly concatenated.

Use:

```bash
policy.observer.fusion=top_cross
policy.observer.cross_dropout=0.1
policy.observer.token_dropout=0.05
```

## Learnable Black-Ball Augmentation

The black-ball module is optional and disabled by default.

It can generate soft black circular masks over input images. The current advanced version supports:

- image-conditioned ball generation;
- learnable center;
- learnable radius;
- per-view independent MLP heads;
- optional frozen DINOv3 encoder for ball control;
- diversity, border, radius, and area regularization;
- W&B visualization of clean/masked images.

Important: if black-ball augmentation is disabled, no black balls are applied:

```bash
policy.learnable_occlusion.enabled=false
```

The DINOv3 observer can still be used without black balls.

## Training: DINOv3 + Top Cross, No Black Balls

This is the cleanest structural ablation:

```bash
cd /home/dm/QWENLA/VLANeXt_migration/VITA
conda activate vita

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python flare/train.py \
  policy=vita \
  task=robotwin_put_bottles_dustbin \
  session=robotwin_put_bottles_dustbin_vita_3view_dinov3_topcross \
  device=cuda:0 \
  train.steps=100000 \
  train.batch_size=64 \
  train.num_workers=0 \
  train.log_freq=50 \
  train.save_freq=5000 \
  wandb.enable=true \
  val.val_offline_freq=0 \
  val.val_online_freq=0 \
  policy.flow_net.name=simple_flow_net \
  policy.observer.name=dinov3 \
  policy.observer.freeze=true \
  policy.observer.fusion=top_cross \
  policy.observer.cross_dropout=0.1 \
  policy.observer.token_dropout=0.05 \
  policy.observer.dinov3_repo_dir=/home/dm/QWENLA/VLANeXt_migration/dinov3 \
  policy.observer.dinov3_model_name=dinov3_vits16 \
  'policy.observer.dinov3_weights_path=/home/dm/QWENLA/VLANeXt_migration/dino weight/dino3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth' \
  policy.observer.dinov3_input_size=224 \
  policy.learnable_occlusion.enabled=false
```

## Training: DINOv3 Global Feature Baseline

Use this to compare against top-cross fusion:

```bash
policy.observer.fusion=pool_concat
policy.learnable_occlusion.enabled=false
```

Everything else can stay the same.

## Training: Learnable Black Balls

Example with image-conditioned black-ball generation:

```bash
policy.learnable_occlusion.enabled=true
policy.learnable_occlusion.condition_on_image=true
policy.learnable_occlusion.image_encoder_type=dinov3
policy.learnable_occlusion.image_encoder_freeze=true
policy.learnable_occlusion.separate_view_heads=true
policy.learnable_occlusion.num_balls=60
policy.learnable_occlusion.learn_radius=true
policy.learnable_occlusion.radius_min=2
policy.learnable_occlusion.radius_max=64
policy.learnable_occlusion.area_target=0.7
policy.learnable_occlusion.area_weight=0.1
policy.learnable_occlusion.image_log_freq=100
```

This mode should be treated as a separate augmentation experiment, not mixed with the first DINOv3 top-cross ablation unless the baseline already works.

## Eval

Run VITA eval from RoboTwin's VITA policy directory:

```bash
cd /home/dm/QWENLA/VLANeXt_migration/RoboTwin/policy/VITA
conda activate vita

bash eval_socket.sh \
  put_bottles_dustbin \
  demo_clean \
  aloha-agilex_clean_50 \
  /path/to/VITA/checkpoints/step_0000050000 \
  0 \
  0 \
  20 \
  unseen \
  8
```

Arguments:

```text
task              put_bottles_dustbin
demo type         demo_clean
setting           aloha-agilex_clean_50
checkpoint        VITA checkpoint directory
gpu id            0
start seed        0
num episodes      20
split             unseen or seen
exec horizon      8
```

## Experiment Order

Recommended order:

1. ResNet18 VITA baseline.
2. Frozen DINOv3 `pool_concat`, no black balls.
3. Frozen DINOv3 `top_cross`, no black balls.
4. Frozen DINOv3 `top_cross` with dropout tuning.
5. Add learnable black-ball augmentation only after the clean structural baseline is evaluated.

This keeps the ablation readable and avoids mixing architecture improvement with augmentation effects.

### Role-Query Patch Fusion

This is the current structured multi-arm fusion design.

It uses DINOv3 patch tokens instead of only global features:

```text
top_patch_tokens   [B, 196, 384]
left_patch_tokens  [B, 196, 384]
right_patch_tokens [B, 196, 384]
```

The patch tokens are projected to a smaller role-fusion space:

```text
shared Linear 384 -> 128
+ view embedding
+ 2D position embedding
```

Then the source sequence is:

```text
top visual tokens   [B, 196, 128]
left visual tokens  [B, 196, 128]
right visual tokens [B, 196, 128]
state token         [B,   1, 128]

source_tokens       [B, 589, 128]
```

Role queries:

```text
q_left  [4, 128]
q_right [4, 128]
q_inter [2, 128]
```

Cross-attention:

```text
Query = [q_left, q_right, q_inter]
Key   = source_tokens
Value = source_tokens
```

Output:

```text
role_tokens [B, 10, 128]
flatten     [B, 1280]
obs_encoder [B, 1280] -> [B, 512]
```

Rationale:

- left queries ask for left-arm relevant evidence;
- right queries ask for right-arm relevant evidence;
- inter queries ask for two-arm coordination evidence;
- patch tokens preserve spatial information that global cls features lose.

Use:

```bash
policy.observer.fusion=role_query
policy.observer.role_dim=128
policy.observer.role_left_queries=4
policy.observer.role_right_queries=4
policy.observer.role_inter_queries=2
policy.observer.role_dropout=0.1
policy.observer.token_dropout=0.05
```

Smoke result:

```text
Number of parameters: 42.13M | Trainable params: 20.53M
Training completed
```
