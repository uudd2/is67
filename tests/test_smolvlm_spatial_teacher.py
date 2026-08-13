import unittest

import torch
import torch.nn as nn

from src.models.smolvla_vita import _encode_smolvlm_spatial_pair
from src.models.vita_latent_flow import VisualDeltaDecoder


class _FakeVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)


class _FakeVLMModel:
    def __init__(self):
        self.vision_model = _FakeVisionModel()
        self.connector = nn.Identity()


class _FakeBackbone:
    def __init__(self, num_tokens=64, hidden_size=48):
        self.vlm = _FakeVLMModel()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size

    def get_vlm_model(self):
        return self.vlm

    def embed_image(self, pixels):
        value = pixels.mean(dim=(1, 2, 3), keepdim=False)
        return value[:, None, None].expand(
            pixels.shape[0],
            self.num_tokens,
            self.hidden_size,
        )


class SmolVLMSpatialTeacherTest(unittest.TestCase):
    def test_keeps_64_spatial_tokens_and_splits_current_future(self):
        backbone = _FakeBackbone()
        current = torch.zeros(2, 3, 16, 16)
        future = torch.ones(2, 1, 3, 16, 16)

        current_tokens, future_tokens = _encode_smolvlm_spatial_pair(
            backbone,
            current,
            future,
            batch_size=2,
        )

        self.assertEqual(current_tokens.shape, (2, 64, 48))
        self.assertEqual(future_tokens.shape, (2, 64, 48))
        self.assertTrue(torch.equal(current_tokens, torch.zeros_like(current_tokens)))
        self.assertTrue(torch.equal(future_tokens, torch.ones_like(future_tokens)))
        self.assertFalse(current_tokens.requires_grad)
        self.assertFalse(future_tokens.requires_grad)

    def test_rejects_non_8_by_8_teacher_tokens(self):
        backbone = _FakeBackbone(num_tokens=63)
        pixels = torch.zeros(2, 3, 16, 16)

        with self.assertRaisesRegex(RuntimeError, r"\[B,64,D\]"):
            _encode_smolvlm_spatial_pair(
                backbone,
                pixels,
                pixels,
                batch_size=2,
            )

    def test_visual_decoder_maps_action_latent_to_spatial_teacher_width(self):
        decoder = VisualDeltaDecoder(
            latent_dim=32,
            hidden_dim=64,
            num_layers=2,
            visual_dim=48,
        )
        action_latent = torch.randn(2, 32, requires_grad=True)
        current_tokens = torch.randn(2, 64, 48)

        prediction = decoder(action_latent, current_tokens)
        self.assertEqual(prediction.shape, (2, 64, 48))

        prediction.square().mean().backward()
        self.assertIsNotNone(action_latent.grad)
        self.assertTrue(torch.isfinite(action_latent.grad).all())
        self.assertGreater(action_latent.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
