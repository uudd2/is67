import pytest
import torch
import torch.nn as nn

from src.models.vita_latent_flow import (
    VitaLatentActionGenerator,
    get_flow_latent_reconstruction_weights,
    get_flow_reconstruction_weights,
    resolve_flow_reconstruction_weights,
)


@pytest.mark.parametrize(
    ("progress", "expected"),
    [
        (0.0, (1.0, 0.1)),
        (0.1, (1.0, 0.1)),
        (0.4, (0.6, 0.5)),
        (0.7, (0.35, 1.0)),
        (1.0, (0.35, 1.0)),
    ],
)
def test_schedule_boundaries(progress, expected):
    actual = get_flow_reconstruction_weights(int(progress * 1000), 1000)
    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == pytest.approx(expected[1])


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (250, (0.8, 0.3)),
        (550, (0.475, 0.75)),
    ],
)
def test_schedule_linear_interpolation(step, expected):
    actual = get_flow_reconstruction_weights(step, 1000)
    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == pytest.approx(expected[1])


def test_zero_max_train_steps_does_not_divide_by_zero():
    assert get_flow_reconstruction_weights(0, 0) == pytest.approx((1.0, 0.1))


def test_disabled_schedule_preserves_fixed_weights():
    assert resolve_flow_reconstruction_weights(
        enabled=False,
        global_step=700,
        max_train_steps=1000,
        fixed_flow_weight=1.0,
        fixed_reconstruction_weight=0.2,
    ) == pytest.approx((1.0, 0.2))


def test_enabled_losses_are_weighted_once():
    flow_loss = torch.tensor(2.0)
    reconstruction_loss = torch.tensor(3.0)
    weights = resolve_flow_reconstruction_weights(
        enabled=True,
        global_step=400,
        max_train_steps=1000,
        fixed_flow_weight=1.0,
        fixed_reconstruction_weight=0.2,
    )
    total = weights[0] * flow_loss + weights[1] * reconstruction_loss
    assert total.item() == pytest.approx(2.7)


def test_resumed_global_step_controls_schedule():
    assert get_flow_reconstruction_weights(112000, 160000) == pytest.approx(
        (0.35, 1.0)
    )


@pytest.mark.parametrize(
    ("progress", "expected"),
    [
        (0.0, (1.0, 1.0, 0.2)),
        (0.1, (1.0, 1.0, 0.2)),
        (0.4, (0.6, 0.6, 0.5)),
        (0.7, (0.35, 0.25, 1.0)),
        (1.0, (0.35, 0.25, 1.0)),
    ],
)
def test_three_loss_schedule_boundaries(progress, expected):
    actual = get_flow_latent_reconstruction_weights(
        int(progress * 1000), 1000
    )
    assert actual == pytest.approx(expected)


def test_three_loss_schedule_interpolation():
    assert get_flow_latent_reconstruction_weights(250, 1000) == pytest.approx(
        (0.8, 0.8, 0.35)
    )
    assert get_flow_latent_reconstruction_weights(550, 1000) == pytest.approx(
        (0.475, 0.45, 0.75)
    )


def test_three_loss_schedule_zero_max_steps():
    assert get_flow_latent_reconstruction_weights(0, 0) == pytest.approx(
        (1.0, 1.0, 0.2)
    )


def test_rollout_reconstruction_reaches_flow_with_frozen_decoder():
    torch.manual_seed(7)
    class TinyFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(8, 8, bias=False)

        def can_cache_condition(self, condition_tokens):
            return False

        def forward(
            self,
            latent,
            timestep,
            condition_tokens=None,
            condition_cache=None,
        ):
            return self.projection(latent)

    class TinyGenerator:
        def __init__(self):
            self.flow = TinyFlow()
            self.num_sampling_steps = 3

        sample_latent = VitaLatentActionGenerator.sample_latent

    generator = TinyGenerator()
    decoder = nn.Linear(8, 4)
    decoder.requires_grad_(False).eval()
    latent = generator.sample_latent(torch.randn(2, 8))
    prediction = decoder(latent)
    reconstruction_loss = torch.nn.functional.l1_loss(
        prediction, torch.randn_like(prediction)
    )
    reconstruction_loss.backward()
    assert generator.flow.projection.weight.grad is not None
    assert generator.flow.projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in decoder.parameters())
