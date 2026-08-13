# 单主视角 EDAR-lite：Codex 实现任务书

## 目标

在现有 LIBERO baseline 中，用单主视角 EDAR-lite 替换当前的 `ActionEncoder + ActionDecoder + VisualDeltaDecoder` 表征学习路径，使动作 latent 同时保留可执行动作信息与环境相关的视觉效果信息。

必须保持现有策略接口不变：

- 动作 chunk：`A_t: [B, 8, 7]`
- Flow 使用的动作 latent：`z_act: [B, 1024]`
- Flow、`z_obs`、6-step Euler rollout 和执行 horizon 不在本任务中修改
- 仅 EDAR-lite 视觉分支使用单个主视角；不要加入 wrist view，也不要加入 view embedding
- 推理时不得运行 DINO 或视觉预测分支；最终动作仍由生成的 `z_act` 解码为 `[B, 8, 7]`

这不是简单加深 `VisualDeltaDecoder`。需要让 ActionEncoder 在表征预训练时同时读取动作 chunk 和当前主视角特征，形成真正环境相关的 `z_act`。

## 数据对齐

每个训练样本提供：

```text
actions       A_t       [B, 8, 7]       # t ... t+7
main_image_t  I_t       [B, 3, H, W]
main_image_f  I_{t+8}   [B, 3, H, W]     # 执行动作 chunk 后的端点帧
```

严格检查 `I_{t+8}` 的索引，不能误用 `t+7`。episode 尾部不足 8 步的样本应过滤，不允许跨 episode 取未来帧。动作使用工程现有的 normalization/statistics。

## 视觉特征

实现 `FrozenDINOFeatureExtractor`：

- backbone：DINOv3-Base，完全冻结，始终保持 `eval()`
- 输入：主视角 RGB，确定性 resize/crop 到 `256 x 256`，使用 DINO 官方 normalization
- 获取最后层 patch tokens，不使用 CLS/register token
- DINOv3-B/16 在 `256 x 256` 下得到 `16 x 16` patch map；用 `AdaptiveAvgPool2d((8, 8))` 压为 64 个空间 token
- 输出：`X_t, X_{t+8}: [B, 64, d_v]`，其中 `d_v` 从 backbone 配置动态读取，不在代码中硬编码
- DINO forward 必须位于 `torch.no_grad()`/inference mode；输出必须 detach
- 提供离线缓存脚本，将 64-token 特征以 fp16/bf16 按 frame id 保存；表征预训练默认优先读取缓存
- 缓存必须记录 backbone 名称、权重版本、预处理分辨率和 normalization，metadata 不匹配时直接报错，不静默复用

不要再使用当前 Qwen `Q4 learned-query resampler` 的 4 个 token 作为视觉预测目标。Q4 可以继续服务原有 VLA 条件路径，但不属于新的 EDAR-lite 视觉监督。

## EDAR-lite Encoder

实现 `SingleViewEDARLiteEncoder`：

```text
A_t [B,8,7]
  -> Linear(7,512) + learned temporal pos + action type embedding

X_t [B,64,d_v]
  -> Linear(d_v,512) + 2D sine-cos pos + visual type embedding

R [4,512]
  -> 4 个可学习 register tokens + register type embedding

concat([A_tokens, R, X_t]) -> [B,76,512]
  -> 4 层双向 Transformer Encoder
  -> 取 4 个 register 输出 [B,4,512]
  -> RMSNorm + Linear(512,256)
  -> Z_act [B,4,256]
  -> flatten + LayerNorm
  -> z_act [B,1024]
```

Transformer 配置：

```yaml
hidden_size: 512
num_layers: 4
num_heads: 8
mlp_ratio: 4
norm: RMSNorm
norm_position: pre_norm
activation: GELU
dropout: 0.0
attention_dropout: 0.0
```

Encoder 的全序列采用双向 self-attention。未来图像 `I_{t+8}` 绝不能进入 Encoder。

## 共享注意力双分支 Decoder

实现 `SingleViewEDARLiteDecoder`。它使用同一组 self-attention 参数，同时用专用 FFN 处理动作与视觉 token。

输入 token：

```text
latent tokens:
  reshape(z_act, [B,4,256]) -> Linear(256,512)                 # 4 tokens

action queries:
  8 个 learned query + temporal position                       # 8 tokens

visual queries（仅表征预训练使用）:
  Linear(d_v,512)(X_t) + 2D position                           # 64 tokens
```

完整训练序列为 `[latent(4), action_query(8), visual_query(64)]`，共 76 tokens。使用 4 个 Decoder block，每个 block 包含：

1. 共享的 pre-norm multi-head self-attention；
2. latent 与 action token 使用 `FFN_act`；
3. visual token 使用独立的 `FFN_vis`；
4. 两个 FFN 均为 `512 -> 2048 -> 512`，GELU，dropout 0。

注意力 mask 必须满足：

| Query | 可访问的 Key/Value |
|---|---|
| latent | latent |
| action query | latent + action query |
| visual query | latent + action query + visual query |

因此 action query 不能访问当前视觉 query 或未来视觉目标，防止动作重建绕过 `z_act`。由于 action queries 是 learned queries、没有 teacher-forced action token，本版本不需要 causal action mask。

输出头：

```text
action states [B,8,512]
  -> Linear(512,7)
  -> A_hat [B,8,7]

visual states [B,64,512]
  -> Linear(512,d_v)
  -> delta [B,64,d_v]
  -> X_hat_{t+8} = normalize(X_t + alpha * delta)
```

`alpha` 为可学习标量，初始化为 `0.1`。损失作用于最终 `X_hat_{t+8}`，不能回归 `X_{t+8} - X_t`。

Decoder 还必须提供：

```python
decode_actions(z_act) -> A_hat
```

该路径只构造 latent 和 action query，不构造 visual query。由于 attention mask 中 action 分支本来就不可见 visual token，其输出应与完整 forward 的动作输出在数值误差内一致。这是策略推理使用的唯一 Decoder 路径。

## 损失

表征预训练只使用：

```python
loss_act = mse(A_hat, A_t)  # normalized action space

target = F.normalize(X_t8.detach().float(), dim=-1)
pred   = F.normalize(X_hat_t8.float(), dim=-1)
loss_effect = (1.0 - (pred * target).sum(dim=-1)).mean()

loss_edar = loss_act + 0.2 * loss_effect
```

余弦损失在 fp32 中计算，避免 bf16/fp16 精度问题。初版不要加入像素重建、feature-delta SmoothL1、wrist loss、对比学习或额外 KL loss。

## 两阶段训练

### Stage A：表征预训练

训练 Encoder + Decoder，DINO 冻结：

```yaml
steps: 100000
batch_size: 64                 # 显存不足时用梯度累积保持有效 batch=64
optimizer: AdamW
lr: 1.0e-4 -> 1.0e-5 cosine
betas: [0.9, 0.99]
weight_decay: 0.01
grad_clip_norm: 1.0
precision: bf16
lambda_effect: 0.2
seed: 使用工程现有固定 seeds
```

保存 Encoder、Decoder、动作 normalization stats、DINO/预处理 metadata。旧 `ActionEncoder/ActionDecoder/VisualDeltaDecoder` 权重不能直接迁移到新 latent 空间。

### Stage B：策略/Latent Flow 训练

- 加载 Stage A checkpoint；冻结 EDAR-lite Encoder 和 Decoder
- 训练样本通过冻结 Encoder 得到目标 `z_act:[B,1024]`；可直接读取预计算的 DINO 特征
- 保持现有 `z_obs -> z_act` Flow、6-step Euler 和 `ActionDecoder` 调用位置
- 将推理 Decoder 调用替换为 `edar.decode_actions(z_act_hat)`
- 策略阶段总损失建议为：

```text
1.0 * L_flow
+ 1.0 * L_latent_consistency
+ 0.2 * L_flow_action_recon
```

旧的 `0.2 * L_action_AE_recon` 与 `0.01 * L_visual_effect` 已在 Stage A 完成，不再参与 Stage B 反向传播；可以保留为无梯度 validation 指标。

推理路径必须为：

```text
policy observation -> z_obs -> 6-step latent flow
-> z_act_hat [B,1024]
-> frozen EDAR-lite decode_actions
-> action chunk [B,8,7]
```

推理时不得加载或调用 DINO，不得要求 `I_{t+8}`。

## 配置与兼容性

先检查仓库实际结构，定位现有：

- `ActionEncoder`
- `ActionDecoder`
- `VisualDeltaDecoder`
- 数据集中的主视角字段与 frame index
- `z_act` 进入 Flow/Decoder 的位置
- checkpoint/config 注册方式

按仓库已有风格选择实际文件路径，不要假设固定目录。建议新增配置开关：

```yaml
action_representation:
  type: single_view_edar_lite
  main_view_key: <仓库现有主视角字段>
  action_horizon: 8
  future_offset: 8
  latent_tokens: 4
  latent_token_dim: 256
  model_dim: 512
  encoder_layers: 4
  decoder_layers: 4
  heads: 8
  visual_grid: 8
  lambda_effect: 0.2
  dino_model: <实际可用的 DINOv3-Base checkpoint id>
  freeze_dino: true
```

保留原 baseline 配置路径，方便同 seed 消融。不得静默改变旧 checkpoint 行为。

## 测试与验收

至少补充以下自动化测试：

1. Shape test：
   - `Z_act=[B,4,256]`
   - `z_act=[B,1024]`
   - `A_hat=[B,8,7]`
   - `X_hat=[B,64,d_v]`
2. Future alignment test：构造带 frame id 的假 episode，确认 target 恰为 `t+8` 且不跨 episode。
3. Freeze test：反向后 DINO 所有参数 `grad is None`，且 DINO 始终为 eval mode。
4. Leakage test：修改 `X_t` 时，在固定 `z_act` 下 `decode_actions(z_act)` 输出不变。
5. Decoder consistency test：完整 forward 的 action 输出与 `decode_actions(z_act)` 在 `atol=1e-5, rtol=1e-4` 内一致。
6. Inference test：mock DINO forward；策略推理期间调用次数必须为 0。
7. Smoke overfit：在小批数据上确认 `loss_act` 和 `loss_effect` 都能明显下降，且无 NaN/Inf。
8. Shuffle diagnostic：validation 中随机打乱 batch 内 `z_act`，重新计算视觉预测；记录
   `shuffle_gap = L_effect(shuffled_z) - L_effect(correct_z)`。训练后应为正且明显大于随机波动，否则视觉分支仍在忽略动作 latent。

训练日志至少包含：

```text
loss_act, loss_effect, total_loss,
z_mean, z_std, z_norm,
visual_cosine, shuffle_gap,
grad_norm, lr
```

## 实施顺序

1. 只读检查现有数据和模型调用链，列出将修改的文件与接口。
2. 先实现并测试冻结 DINO 特征抽取/缓存。
3. 实现 Encoder、共享注意力双分支 Decoder 及上述 mask。
4. 实现 Stage A 训练入口和 checkpoint schema。
5. 将冻结 EDAR-lite 接入 Stage B Flow 训练与推理解码。
6. 跑单元测试、一个短 smoke train 和一次无 DINO 的推理测试。
7. 最后汇报修改文件、配置、测试结果、尚未跑完的长训练项目；不要声称成功率提升，除非已完成同 seed 闭环评估。

## 明确禁止

- 不使用 wrist/第二视角
- 不增加 view embedding
- 不把未来帧输入 Encoder
- 不继续用 Q4 token 作为视觉监督目标
- 不预测像素，不训练 DINO
- 不让 action query 访问 visual query
- 不改变 `z_act:[B,1024]`、Flow 结构、Euler 步数或动作 horizon
- 不在推理时调用 DINO/视觉 Predictor
- 不删除旧 baseline；用配置保留可复现实验路径

## 设计依据

该实现是对 EDAR 的单视角、轻量、接口兼容改造。EDAR 原文的核心是：动作、当前视觉和 register tokens 通过共享注意力形成环境相关 latent，同时用动作重建与未来 DINO patch 特征预测联合约束；原文采用冻结 DINOv3-Base、共享注意力 Decoder、专用动作/视觉 FFN，以及两阶段训练。这里保留这些核心，但将多视角改为单主视角、将网络缩为 4+4 层，并通过 `4 x 256 -> 1024` 适配现有 Latent Flow。
