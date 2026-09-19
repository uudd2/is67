import json

from src.datasets.lerobot_piper_stage_a import (
    load_batch_episode_ids,
    make_episode_split_manifest,
)


def test_load_batch_episode_ids_reads_total_episodes(tmp_path):
    batch = tmp_path / "pick"
    (batch / "meta").mkdir(parents=True)
    (batch / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 4}), encoding="utf-8"
    )
    assert load_batch_episode_ids(batch) == [0, 1, 2, 3]


def test_manifest_splits_each_batch_and_is_json_serializable(tmp_path):
    for name, count in [("fold", 5), ("pick", 3)]:
        batch = tmp_path / name
        (batch / "meta").mkdir(parents=True)
        (batch / "meta" / "info.json").write_text(
            json.dumps({"total_episodes": count}), encoding="utf-8"
        )
    manifest = make_episode_split_manifest(
        [
            {"name": "fold", "path": str(tmp_path / "fold"), "repo_id": "local/fold"},
            {"name": "pick", "path": str(tmp_path / "pick"), "repo_id": "local/pick"},
        ],
        val_ratio=0.25,
        seed=2026,
    )
    assert set(manifest) >= {"seed", "val_ratio", "train", "val"}
    for name in ("fold", "pick"):
        assert set(manifest["train"][name]).isdisjoint(manifest["val"][name])
    json.dumps(manifest)
