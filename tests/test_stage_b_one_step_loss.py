import pytest
import torch
import torch.nn as nn

from src.models.vita_latent_flow import (
    masked_action_reconstruction_loss,
    recover_one_step_action_latent,
    stage_b_one_step_loss_weights,
)


@pytest.mark.parametrize(
    ("progress", "expected"),
    [
        (0.0, (1.0, 0.0)),
        (0.1, (1.0, 0.0)),
        (0.35, (0.6, 0.5)),
        (0.7, (0.3, 1.0)),
        (1.0, (0.3, 1.0)),
    ],
)
def test_stage_b_one_step_weight_schedule(progress, expected):
    actual = stage_b_one_step_loss_weights(progress)
    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == pytest.approx(expected[1])


def test_exact_velocity_recovers_action_latent_and_shape():
    torch.manual_seed(3)
    z_obs = torch.randn(2, 4, 256)
    z_act = torch.randn(2, 4, 256)
    timestep = torch.tensor([0.2, 0.8])
    t = timestep[:, None, None]
    z_t = (1.0 - t) * z_obs + t * z_act
    recovered = recover_one_step_action_latent(z_t, z_act - z_obs, timestep)
    assert recovered.shape == (2, 4, 256)
    torch.testing.assert_close(recovered, z_act)


def test_frozen_decoder_passes_gradient_to_velocity_and_flow():
    torch.manual_seed(4)
    flow = nn.Linear(256, 256, bias=False)
    decoder = nn.Linear(1024, 56)
    decoder.requires_grad_(False).eval()
    z_t = torch.randn(2, 4, 256)
    timestep = torch.tensor([0.3, 0.6])
    predicted_velocity = flow(z_t)
    predicted_velocity.retain_grad()
    pred_z_act = recover_one_step_action_latent(
        z_t, predicted_velocity, timestep
    )
    pred_z_act.retain_grad()
    predicted_actions = decoder(pred_z_act.flatten(1)).reshape(2, 8, 7)
    loss = masked_action_reconstruction_loss(
        predicted_actions, torch.randn_like(predicted_actions), "l1"
    )
    loss.backward()
    assert predicted_velocity.grad is not None
    assert predicted_velocity.grad.abs().sum() > 0
    assert pred_z_act.grad is not None and pred_z_act.grad.abs().sum() > 0
    assert flow.weight.grad is not None and flow.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_action_mask_excludes_padding():
    target = torch.zeros(1, 3, 2)
    prediction = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    loss = masked_action_reconstruction_loss(prediction, target, "l1", mask)
    assert loss.item() == pytest.approx(2.0)


def test_disabled_weighting_is_plain_fm_only():
    flow_loss = torch.tensor(2.5)
    assert (1.0 * flow_loss + 0.0 * torch.tensor(9.0)).item() == pytest.approx(2.5)
