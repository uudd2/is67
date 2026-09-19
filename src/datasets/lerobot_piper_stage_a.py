"""LeRobot v3 helpers for Piper real-data Stage-A."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .stage_a_episode_utils import split_episode_ids_by_batch


def load_batch_episode_ids(batch_path: str | Path) -> list[int]:
    info_path = Path(batch_path) / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    total_episodes = int(info["total_episodes"])
    if total_episodes < 0:
        raise ValueError(f"Invalid total_episodes in {info_path}: {total_episodes}")
    return list(range(total_episodes))


def make_episode_split_manifest(
    batches: Sequence[Mapping[str, object]],
    val_ratio: float,
    seed: int,
) -> dict[str, object]:
    episodes_by_batch: dict[str, list[int]] = {}
    batch_metadata: dict[str, dict[str, str]] = {}
    for batch in batches:
        name = str(batch["name"])
        if name in episodes_by_batch:
            raise ValueError(f"Duplicate batch name: {name}")
        path = str(batch["path"])
        repo_id = str(batch["repo_id"])
        episodes_by_batch[name] = load_batch_episode_ids(path)
        batch_metadata[name] = {"path": path, "repo_id": repo_id}
    train, val = split_episode_ids_by_batch(
        episodes_by_batch, val_ratio=val_ratio, seed=seed
    )
    return {
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "batches": batch_metadata,
        "train": train,
        "val": val,
    }
