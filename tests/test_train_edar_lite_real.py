import torch

from scripts.train_edar_lite_real import cosine_diagnostics


def test_cosine_diagnostics_reports_prediction_gain_over_copy():
    current = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    future = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    predicted = future.clone()
    metrics = cosine_diagnostics(predicted, current, future)
    assert torch.allclose(metrics["pred_cosine"], torch.tensor(1.0))
    assert torch.allclose(metrics["copy_cosine"], torch.tensor(0.0))
    assert torch.allclose(metrics["cosine_gain"], torch.tensor(1.0))
