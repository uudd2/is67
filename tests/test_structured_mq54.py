import pytest
import torch

from src.models.vita_latent_flow import EDARSourceAwareTokenFlow, StructuredMQ54


def _inputs(hidden_dim=64):
    batch_size = 2
    visual_per_view = 24
    text_tokens = 10
    sequence_length = visual_per_view * 2 + text_tokens
    hidden_states = tuple(
        torch.randn(batch_size, sequence_length, hidden_dim, requires_grad=True)
        for _ in range(13)
    )
    token_types = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    token_types[:, : visual_per_view * 2] = 1
    grids = torch.tensor([[1, 4, 6], [1, 4, 6]] * batch_size)
    state = torch.randn(batch_size, 2, hidden_dim, requires_grad=True)
    return hidden_states, token_types, grids, state


def _forward():
    module = StructuredMQ54(hidden_dim=64, num_heads=8)
    hidden, token_types, grids, state = _inputs()
    output = module(hidden, token_types, grids, state_tokens=state)
    return module, hidden, state, output


def test_structured_mq54_shape_order_metadata_and_parameters():
    module, _, _, output = _forward()
    assert output.shape == (2, 54, 64)
    assert output[:, :52].shape == (2, 52, 64)
    assert output[:, 52:].shape == (2, 2, 64)
    assert [16, 16, 6, 6, 8, 2] == [
        output[:, 0:16].shape[1],
        output[:, 16:32].shape[1],
        output[:, 32:38].shape[1],
        output[:, 38:44].shape[1],
        output[:, 44:52].shape[1],
        output[:, 52:54].shape[1],
    ]
    assert torch.bincount(module.source_ids, minlength=3).tolist() == [32, 12, 8]
    assert torch.bincount(module.camera_ids, minlength=3).tolist() == [22, 22, 8]
    assert not any("global_queries" in name for name, _ in module.named_parameters())
    assert module.semantic_attention is module.semantic_attention
    assert module.last_semantic_kv_lengths == (24, 24)


@pytest.mark.parametrize("layer", [0, 1, 4, 8])
def test_spatial_gradient_reaches_selected_visual_layers(layer):
    _, hidden, _, output = _forward()
    output[:, :32].sum().backward()
    assert hidden[layer].grad[:, :48].abs().sum() > 0


def test_semantic_gradient_reaches_visual_and_text_inputs():
    _, hidden, _, output = _forward()
    output[:, 32:44].sum().backward()
    assert hidden[8].grad[:, :48].abs().sum() > 0
    assert hidden[12].grad[:, :48].abs().sum() > 0
    assert hidden[12].grad[:, 48:].abs().sum() > 0


def test_text_gradient_reaches_h8_and_h12_text_inputs():
    _, hidden, _, output = _forward()
    output[:, 44:52].sum().backward()
    assert hidden[8].grad[:, 48:].abs().sum() > 0
    assert hidden[12].grad[:, 48:].abs().sum() > 0


def test_structured_edar_flow_shape_sources_cameras_mass_and_euler():
    flow = EDARSourceAwareTokenFlow(
        [("spatial", 32), ("semantic", 12), ("text", 8)],
        camera_ids=[0] * 16 + [1] * 16 + [0] * 6 + [1] * 6 + [2] * 8,
    )
    condition = torch.randn(2, 54, 1024)
    latent = torch.randn(2, 4, 256)
    output = flow(latent, torch.rand(2), condition_tokens=condition)
    assert output.shape == (2, 4, 256)
    assert torch.bincount(flow.source_ids, minlength=3).tolist() == [32, 12, 8]
    assert torch.bincount(flow.camera_ids, minlength=3).tolist() == [22, 22, 8]
    first_cross = flow.blocks[1].cross_attention
    assert first_cross.source_bias.out_features == 8 * 3
    assert first_cross.slot_source_bias.shape == (4, 3)
    assert sum(first_cross.last_source_attention_mass.values()).item() == pytest.approx(
        1.0, abs=1e-5
    )
    for step in range(6):
        timestep = torch.full((2,), (step + 0.5) / 6)
        latent = latent + flow(latent, timestep, condition_tokens=condition) / 6
    assert latent.shape == (2, 4, 256)
