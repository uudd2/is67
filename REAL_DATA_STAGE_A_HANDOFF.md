# Real robot Stage-A handoff (2026-09-19)

Repository: https://github.com/uudd2/is67 (main).

The user has abandoned the pure RGB prediction approach. Do not assume Future-MAE RGB reconstruction or its loss is the next training objective. Discuss the replacement first. Preserve historical experiments.

Read src/models/edar_lite.py, scripts/train_edar_lite.py and scripts/visualize_edar_lite_predictions.py for existing feature-based Stage-A interfaces. Future-MAE files are historical references, not the selected real-data baseline.

Data host: dm@100.106.134.37
Data root: /home/dm/.cache/huggingface/lerobot/local/
User-collected picking and clothes-folding batches, each episode with independent LeRobot v3 metadata.
Inspected pick_20260919_163008: bi_piper_follower, 30 FPS, action/state 14 dimensions (6 joints + gripper per arm), top/left_wrist/right_wrist RGB cameras, 848x480 AV1 and Parquet. Verify other batches and action semantics before adapting.
Example batches (metadata snapshot):
- fold_clothes_20260918_131512: 242 episodes, 265768 frames
- fold_clothes_20260918_213821: 51 episodes, 55521 frames
- pick_20260919_161241: 42 episodes, 9247 frames
- pick_20260919_163008: 29 episodes, 6693 frames

Plan separate real-data adapters/config/checkpoints, support 14D actions, split by episode, prevent cross-episode temporal sampling, verify normalization. Do not reuse LIBERO feature caches for real images. Stage-B is out of scope. Dataset batches, cameras, horizon, objective and training budget remain to be confirmed. Do not start training or alter old experiments automatically.

The main implementation will be handed to GPT web. Read source before proposing patches; do not guess interfaces. Raw data is on the SSH host, not provided by the repository link.