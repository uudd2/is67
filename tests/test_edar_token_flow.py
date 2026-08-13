import pytest
import torch

from src.models.vita_latent_flow import EDARSourceAwareTokenFlow


SOURCE_COUNTS = [
    ("h0_visual", 8),
    ("h0_text", 4),
    ("h1", 4),
    ("h4", 4),
    ("h8", 4),
    ("h12", 4),
    ("global", 24),
]


def test_edar_source_aware_token_flow_shape_backward_and_structure():
    torch.manual_seed(7)
    flow = EDARSourceAwareTokenFlow(SOURCE_COUNTS)
    latent = torch.randn(2, 4, 256, requires_grad=True)
    condition = torch.randn(2, 54, 1024)

    output = flow(latent, torch.rand(2), condition_tokens=condition)
    assert output.shape == (2, 4, 256)
    output.square().mean().backward()
    assert latent.grad is not None
    assert flow.source_ids.numel() == 52
    assert [
        index for index, block in enumerate(flow.blocks) if block.cross_attention is not None
    ] == [1, 3, 5, 7]
    assert not any("value_gate" in name for name, _ in flow.named_parameters())

    first_cross = flow.blocks[1].cross_attention
    assert first_cross.last_gate.shape == (2, 4, 1)
    assert first_cross.last_gate.mean().item() == pytest.approx(
        torch.sigmoid(torch.tensor(-4.0)).item(), abs=1e-7
    )
    source_mass = sum(first_cross.last_source_attention_mass.values())
    assert source_mass.item() == pytest.approx(1.0, abs=1e-5)
    assert not flow.can_cache_condition(condition)
    with pytest.raises(ValueError, match="does not accept condition_cache"):
        flow(latent.detach(), torch.rand(2), condition, condition_cache={})
