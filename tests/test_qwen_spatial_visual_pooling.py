import unittest

import torch

from src.models.vita_latent_flow import spatial_pool_qwen_visual_tokens


class QwenSpatialVisualPoolingTest(unittest.TestCase):
    def test_preserves_row_major_8_by_8_grid_after_qwen_merge(self):
        tokens = torch.arange(64 * 3, dtype=torch.float32).reshape(64, 3)

        pooled = spatial_pool_qwen_visual_tokens(
            tokens,
            grid_thw=torch.tensor([1, 16, 16]),
            spatial_merge_size=2,
            output_grid_size=8,
        )

        self.assertEqual(pooled.shape, (64, 3))
        self.assertTrue(torch.equal(pooled, tokens))

    def test_adaptive_pooling_aggregates_corresponding_spatial_regions(self):
        tokens = torch.arange(64, dtype=torch.float32).reshape(64, 1)
        tokens.requires_grad_()

        pooled = spatial_pool_qwen_visual_tokens(
            tokens,
            grid_thw=torch.tensor([1, 8, 8]),
            spatial_merge_size=1,
            output_grid_size=2,
        )

        expected = torch.tensor([[13.5], [17.5], [45.5], [49.5]])
        self.assertTrue(torch.allclose(pooled, expected))

        pooled.sum().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_rejects_grid_smaller_than_requested_output(self):
        with self.assertRaisesRegex(ValueError, "smaller than"):
            spatial_pool_qwen_visual_tokens(
                torch.zeros(16, 4),
                grid_thw=torch.tensor([1, 4, 4]),
                spatial_merge_size=1,
                output_grid_size=8,
            )


if __name__ == "__main__":
    unittest.main()
