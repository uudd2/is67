from types import SimpleNamespace
import os

import torch
import torch.nn as nn

from src.datasets.libero_act import strict_future_start_indices
from src.models.edar_lite import (
    EDARFeatureCache,
    FrozenDINOFeatureExtractor,
    SingleViewEDARLite,
)
from src.models.vita_latent_flow import VitaLatentActionGenerator


def _small_edar():
    return SingleViewEDARLite(
        visual_dim=32,
        model_dim=64,
        num_heads=8,
        num_layers=2,
        latent_tokens=4,
        latent_token_dim=256,
    )


def test_edar_shapes_and_decoder_consistency():
    torch.manual_seed(3)
    model = _small_edar().eval()
    actions = torch.randn(2, 8, 7)
    current_visual = torch.randn(2, 64, 32)
    with torch.no_grad():
        latent, token_latent = model.encode(
            actions,
            current_visual,
            return_token_latent=True,
        )
        full_actions, future_visual = model.decoder(latent, current_visual)
        action_only = model.decode_actions(latent)
    assert token_latent.shape == (2, 4, 256)
    assert latent.shape == (2, 1024)
    assert full_actions.shape == (2, 8, 7)
    assert future_visual.shape == (2, 64, 32)
    torch.testing.assert_close(full_actions, action_only, atol=1e-5, rtol=1e-4)


def test_change_weighted_effect_matches_uniform_when_scene_is_static():
    torch.manual_seed(17)
    current = torch.nn.functional.normalize(torch.randn(2, 64, 8), dim=-1)
    future = current.clone()
    prediction = torch.nn.functional.normalize(torch.randn(2, 64, 8), dim=-1)

    weighted, _ = SingleViewEDARLite.change_weighted_effect_loss(
        prediction,
        current,
        future,
    )
    uniform = (1.0 - (prediction * future).sum(dim=-1)).mean()
    torch.testing.assert_close(weighted, uniform)


def test_change_weighted_effect_prioritizes_changed_patch():
    current = torch.zeros(1, 64, 2)
    current[..., 0] = 1.0
    future = current.clone()
    future[:, 0] = torch.tensor([0.0, 1.0])

    wrong_changed_patch = future.clone()
    wrong_changed_patch[:, 0] = torch.tensor([1.0, 0.0])
    wrong_static_patch = future.clone()
    wrong_static_patch[:, 1] = torch.tensor([0.0, 1.0])

    changed_loss, _ = SingleViewEDARLite.change_weighted_effect_loss(
        wrong_changed_patch,
        current,
        future,
    )
    static_loss, _ = SingleViewEDARLite.change_weighted_effect_loss(
        wrong_static_patch,
        current,
        future,
    )
    assert changed_loss > static_loss


def test_decode_actions_has_no_visual_leakage():
    model = _small_edar().eval()
    latent = torch.randn(2, 1024)
    with torch.no_grad():
        expected = model.decode_actions(latent)
        model.decoder.visual_projection.weight.normal_()
        model.decoder.visual_head.weight.normal_()
        actual = model.decode_actions(latent)
    torch.testing.assert_close(expected, actual)


def test_strict_future_alignment_is_t_plus_8_without_crossing_episode():
    indices = strict_future_start_indices(12, future_offset=8)
    assert indices.tolist() == [0, 1, 2, 3]
    assert (indices + 8).tolist() == [8, 9, 10, 11]
    assert strict_future_start_indices(8, future_offset=8).size == 0


class _DummyDINOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(
            model_type="dinov3_vit",
            hidden_size=16,
            patch_size=16,
            num_register_tokens=4,
        )

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        sequence = torch.arange(
            batch_size * (1 + 4 + 256) * 16,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).reshape(batch_size, 261, 16)
        return SimpleNamespace(last_hidden_state=sequence * self.scale)


class _DummyDINOv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(
            model_type="dinov2",
            hidden_size=12,
            patch_size=14,
            num_register_tokens=0,
        )

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        patch_grid = pixel_values.shape[-1] // self.config.patch_size
        sequence = torch.arange(
            batch_size * (1 + patch_grid * patch_grid) * self.config.hidden_size,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).reshape(batch_size, 1 + patch_grid * patch_grid, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=sequence * self.scale)


class _DummyEDARObservationEncoder(nn.Module):
    def forward(self, proprioception=None, return_condition_tokens=False, **kwargs):
        batch_size = proprioception.shape[0]
        latent = torch.randn(batch_size, 1024, device=proprioception.device)
        condition = torch.randn(batch_size, 54, 1024, device=proprioception.device)
        return (latent, condition) if return_condition_tokens else latent


def test_dino_extractor_is_frozen_eval_and_returns_64_tokens():
    backbone = _DummyDINOv3()
    extractor = FrozenDINOFeatureExtractor(
        "dummy-dinov3",
        backbone=backbone,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    extractor.train()
    features = extractor(torch.randint(0, 256, (2, 240, 320, 3), dtype=torch.uint8))
    assert features.shape == (2, 64, 16)
    assert not extractor.training
    assert not extractor.backbone.training
    assert not features.requires_grad
    assert all(not parameter.requires_grad for parameter in extractor.parameters())
    assert all(parameter.grad is None for parameter in extractor.parameters())


def test_dinov2_patch14_is_pooled_from_18x18_to_64_tokens():
    extractor = FrozenDINOFeatureExtractor(
        "dummy-dinov2",
        backbone=_DummyDINOv2(),
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    features = extractor(torch.randint(0, 256, (2, 240, 320, 3), dtype=torch.uint8))
    assert features.shape == (2, 64, 12)
    assert extractor.metadata()["patch_size"] == 14


def test_feature_cache_does_not_serialize_parent_batch_storage(tmp_path):
    metadata = {
        "backbone": "dummy",
        "image_size": 256,
        "output_grid": 8,
    }
    cache = EDARFeatureCache(tmp_path, metadata, create=True)
    batch = torch.randn(32, 64, 1024, dtype=torch.float16)
    cache.put("frame_0", batch[0])
    path = cache.path_for("frame_0")
    assert os.path.getsize(path) < 200_000
    loaded = cache.get("frame_0")
    torch.testing.assert_close(loaded, batch[0])

    cache.put("frame_1", batch[1])
    cache.preload(log_interval=0)
    loaded_many = cache.get_many(["frame_1", "frame_0"])
    torch.testing.assert_close(loaded_many[0], batch[1])
    torch.testing.assert_close(loaded_many[1], batch[0])


def test_stage_b_generator_uses_frozen_edar_without_visual_decode(tmp_path):
    edar = _small_edar()
    checkpoint_path = tmp_path / "edar_stage_a.pt"
    torch.save(
        {
            "schema": "single_view_edar_lite_stage_a_v1",
            "encoder_state_dict": edar.encoder.state_dict(),
            "decoder_state_dict": edar.decoder.state_dict(),
            "dino_metadata": {"hidden_size": 32},
            "config": {
                "model": {
                    "action_representation": {
                        "visual_grid": 8,
                        "model_dim": 64,
                        "latent_tokens": 4,
                        "latent_token_dim": 256,
                        "layers": 2,
                        "heads": 8,
                        "mlp_ratio": 4.0,
                    }
                }
            },
        },
        checkpoint_path,
    )
    generator = VitaLatentActionGenerator(
        action_dim=7,
        horizon=8,
        vlm_hidden_dim=64,
        latent_dim=1024,
        hidden_dim=64,
        action_ae_layers=2,
        flow_layers=8,
        proprio_dim=7,
        hidden_pooling="attention",
        hierarchical_query_pooling=True,
        layer_local_queries_per_layer_list=[8, 4, 4, 4, 4, 4],
        layer_local_token_source_modes=["visual", "text", "all", "all", "all", "all"],
        blockwise_hidden_layer_indices=[0, 0, 1, 4, 8, 12],
        cross_layer_queries=0,
        global_queries=24,
        state_token_conditioning=True,
        state_num_tokens=2,
        state_broadcast_to_mq=False,
        action_representation_type="single_view_edar_lite",
        edar_stage_a_checkpoint=str(checkpoint_path),
        edar_token_flow=True,
        action_effect_enabled=True,
        action_effect_visual_queries=64,
        action_effect_visual_tokens_prepooled=True,
    )
    generator.train()
    assert not generator.edar_encoder.training
    assert not generator.edar_decoder.training
    assert all(not parameter.requires_grad for parameter in generator.edar_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in generator.edar_decoder.parameters())
    actions = torch.randn(2, 8, 7)
    current_visual = torch.randn(2, 64, 32)
    latent = generator.encode_action(actions, current_visual)
    decoded = generator.decode(latent)
    assert latent.shape == (2, 4, 256)
    assert decoded.shape == (2, 8, 7)

    generator.observation_encoder = _DummyEDARObservationEncoder()
    observation_latent, condition = generator.encode_observation(
        proprioception=torch.randn(2, 7),
        return_condition_tokens=True,
    )
    assert observation_latent.shape == (2, 4, 256)
    sampled = generator.sample_latent(
        observation_latent,
        num_steps=2,
        condition_tokens=condition,
    )
    assert sampled.shape == (2, 4, 256)
    assert generator.decode(sampled).shape == (2, 8, 7)

    generator.edar_train_decoder = True
    generator.edar_decoder.requires_grad_(True)
    generator.train()
    assert generator.edar_decoder.training
    assert all(parameter.requires_grad for parameter in generator.edar_decoder.parameters())

    legacy_generator = VitaLatentActionGenerator(
        action_dim=7,
        horizon=8,
        vlm_hidden_dim=64,
        latent_dim=1024,
        hidden_dim=64,
        action_ae_layers=2,
        flow_layers=1,
        action_representation_type="single_view_edar_lite",
        edar_stage_a_checkpoint=str(checkpoint_path),
        edar_token_flow=False,
        action_effect_enabled=True,
        action_effect_visual_queries=64,
        action_effect_visual_tokens_prepooled=True,
    )
    legacy_latent = legacy_generator.encode_action(actions, current_visual)
    assert legacy_latent.shape == (2, 1024)
    assert legacy_generator.decode(legacy_latent).shape == (2, 8, 7)


def test_edar_smoke_overfits_action_and_visual_effect():
    torch.manual_seed(11)
    model = SingleViewEDARLite(
        visual_dim=8,
        model_dim=32,
        num_heads=4,
        num_layers=1,
        latent_tokens=4,
        latent_token_dim=256,
    )
    actions = torch.randn(2, 8, 7)
    current_visual = torch.randn(2, 64, 8)
    future_visual = torch.nn.functional.normalize(
        current_visual + 0.5 * torch.randn_like(current_visual),
        dim=-1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    with torch.no_grad():
        initial_outputs = model(actions, current_visual)
        initial_total, initial_metrics = model.representation_loss(
            initial_outputs,
            actions,
            future_visual,
        )
    for _ in range(40):
        outputs = model(actions, current_visual)
        loss, _ = model.representation_loss(outputs, actions, future_visual)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_outputs = model(actions, current_visual)
        final_total, final_metrics = model.representation_loss(
            final_outputs,
            actions,
            future_visual,
        )
    assert torch.isfinite(final_total)
    assert final_total < initial_total * 0.5
    assert final_metrics["loss_action"] < initial_metrics["loss_action"]
    assert final_metrics["loss_effect"] < initial_metrics["loss_effect"]
