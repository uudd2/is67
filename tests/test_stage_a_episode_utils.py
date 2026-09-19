import numpy as np

from src.datasets.stage_a_episode_utils import (
    ActionStatsAccumulator,
    build_stage_a_indices,
    normalize_actions,
    split_episode_ids_by_batch,
    valid_stage_a_starts,
)


def test_valid_starts_require_complete_action_horizon_and_future_frame():
    np.testing.assert_array_equal(valid_stage_a_starts(9, action_horizon=8, action_stride=1), [0])
    np.testing.assert_array_equal(valid_stage_a_starts(10, action_horizon=8, action_stride=1), [0, 1])
    np.testing.assert_array_equal(valid_stage_a_starts(17, action_horizon=8, action_stride=2), [0])
    assert valid_stage_a_starts(8, action_horizon=8, action_stride=1).size == 0


def test_stage_a_indices_match_action_chunk_and_future_target():
    action_indices, future_index = build_stage_a_indices(3, action_horizon=4, action_stride=2)
    np.testing.assert_array_equal(action_indices, [3, 5, 7, 9])
    assert future_index == 11


def test_episode_split_is_disjoint_deterministic_and_per_batch():
    episodes = {
        "fold_a": list(range(10)),
        "fold_b": list(range(100, 105)),
        "pick_a": list(range(200, 203)),
    }
    train_a, val_a = split_episode_ids_by_batch(episodes, val_ratio=0.2, seed=2026)
    train_b, val_b = split_episode_ids_by_batch(episodes, val_ratio=0.2, seed=2026)
    assert train_a == train_b
    assert val_a == val_b
    for batch, ids in episodes.items():
        assert set(train_a[batch]).isdisjoint(val_a[batch])
        assert set(train_a[batch]) | set(val_a[batch]) == set(ids)
        assert len(train_a[batch]) >= 1
        assert len(val_a[batch]) >= 1


def test_single_episode_batch_stays_in_train():
    train, val = split_episode_ids_by_batch({"tiny": [7]}, val_ratio=0.2, seed=1)
    assert train == {"tiny": [7]}
    assert val == {"tiny": []}


def test_action_stats_and_normalization_are_dimensionwise():
    acc = ActionStatsAccumulator(action_dim=2)
    acc.update(np.asarray([[0.0, 10.0], [2.0, 14.0]], dtype=np.float32))
    acc.update(np.asarray([[4.0, 18.0]], dtype=np.float32))
    stats = acc.finalize()
    np.testing.assert_allclose(stats["min"], [0.0, 10.0])
    np.testing.assert_allclose(stats["max"], [4.0, 18.0])
    np.testing.assert_allclose(stats["mean"], [2.0, 14.0])
    np.testing.assert_allclose(stats["std"], np.std([[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]], axis=0))

    values = np.asarray([[0.0, 10.0], [4.0, 18.0]], dtype=np.float32)
    mm = normalize_actions(values, stats, mode="min_max")
    np.testing.assert_allclose(mm, [[-1.0, -1.0], [1.0, 1.0]])
    ms = normalize_actions(values, stats, mode="mean_std")
    np.testing.assert_allclose(ms[0], (values[0] - stats["mean"]) / stats["std"])
