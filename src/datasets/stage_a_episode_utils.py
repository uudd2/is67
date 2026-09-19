"""Shared helpers for episode-safe Stage-A sampling and action normalization."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _validate_temporal_args(episode_length: int, action_horizon: int, action_stride: int) -> tuple[int, int, int]:
    episode_length = int(episode_length)
    action_horizon = int(action_horizon)
    action_stride = int(action_stride)
    if episode_length < 0:
        raise ValueError("episode_length must be non-negative")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if action_stride <= 0:
        raise ValueError("action_stride must be positive")
    return episode_length, action_horizon, action_stride


def valid_stage_a_starts(episode_length: int, action_horizon: int, action_stride: int = 1) -> np.ndarray:
    """Return starts whose action chunk and future target are fully inside one episode."""
    episode_length, action_horizon, action_stride = _validate_temporal_args(
        episode_length, action_horizon, action_stride
    )
    last_start_exclusive = episode_length - action_horizon * action_stride
    if last_start_exclusive <= 0:
        return np.empty((0,), dtype=np.int64)
    return np.arange(last_start_exclusive, dtype=np.int64)


def build_stage_a_indices(start: int, action_horizon: int, action_stride: int = 1) -> tuple[np.ndarray, int]:
    """Build action indices t..t+(H-1)s and the future target t+Hs."""
    _, action_horizon, action_stride = _validate_temporal_args(0, action_horizon, action_stride)
    start = int(start)
    if start < 0:
        raise ValueError("start must be non-negative")
    action_indices = start + np.arange(action_horizon, dtype=np.int64) * action_stride
    future_index = start + action_horizon * action_stride
    return action_indices, int(future_index)


def _stable_batch_seed(seed: int, batch_name: str) -> int:
    digest = hashlib.blake2b(f"{int(seed)}\0{batch_name}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def split_episode_ids_by_batch(
    episodes_by_batch: Mapping[str, Sequence[int]],
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Deterministically split episodes within each batch before temporal windowing."""
    val_ratio = float(val_ratio)
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")
    train: dict[str, list[int]] = {}
    val: dict[str, list[int]] = {}
    for batch_name in sorted(episodes_by_batch):
        ids = [int(value) for value in episodes_by_batch[batch_name]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate episode IDs in batch {batch_name!r}")
        ids = sorted(ids)
        if len(ids) <= 1 or val_ratio == 0.0:
            train[batch_name] = ids
            val[batch_name] = []
            continue
        rng = np.random.default_rng(_stable_batch_seed(seed, batch_name))
        shuffled = np.asarray(ids, dtype=np.int64)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(ids) * val_ratio)))
        val_count = min(val_count, len(ids) - 1)
        val_set = set(int(value) for value in shuffled[:val_count])
        train[batch_name] = [value for value in ids if value not in val_set]
        val[batch_name] = [value for value in ids if value in val_set]
    return train, val


@dataclass
class ActionStatsAccumulator:
    """Streaming per-dimension min/max/mean/std accumulator."""

    action_dim: int

    def __post_init__(self) -> None:
        self.action_dim = int(self.action_dim)
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.count = 0
        self.minimum = np.full(self.action_dim, np.inf, dtype=np.float64)
        self.maximum = np.full(self.action_dim, -np.inf, dtype=np.float64)
        self.mean = np.zeros(self.action_dim, dtype=np.float64)
        self.m2 = np.zeros(self.action_dim, dtype=np.float64)

    def update(self, actions: np.ndarray) -> None:
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.action_dim:
            raise ValueError(f"Expected actions [N,{self.action_dim}], got {tuple(values.shape)}")
        if values.shape[0] == 0:
            return
        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = batch_count
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + delta * delta * (self.count * batch_count / total)
        self.count = total

    def finalize(self) -> dict[str, np.ndarray | int]:
        if self.count == 0:
            raise ValueError("Cannot finalize empty action statistics")
        variance = self.m2 / self.count
        return {
            "count": int(self.count),
            "min": self.minimum.astype(np.float32),
            "max": self.maximum.astype(np.float32),
            "mean": self.mean.astype(np.float32),
            "std": np.sqrt(np.maximum(variance, 0.0)).astype(np.float32),
        }


def normalize_actions(actions: np.ndarray, stats: Mapping[str, np.ndarray], mode: str) -> np.ndarray:
    """Apply one shared train-derived action normalization space."""
    values = np.asarray(actions, dtype=np.float32)
    mode = str(mode)
    if mode == "identity":
        return values.copy()
    if mode == "min_max":
        low = np.asarray(stats["min"], dtype=np.float32)
        high = np.asarray(stats["max"], dtype=np.float32)
        scale = np.maximum(high - low, 1e-6)
        return np.clip(2.0 * (values - low) / scale - 1.0, -1.0, 1.0).astype(np.float32)
    if mode == "mean_std":
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.maximum(np.asarray(stats["std"], dtype=np.float32), 1e-6)
        return ((values - mean) / std).astype(np.float32)
    raise ValueError(f"Unsupported action normalization mode: {mode}")
