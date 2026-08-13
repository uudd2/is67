import unittest

import torch
import torch.nn.functional as F

from src.models.vita_latent_flow import action_effect_ranking_loss


class ActionEffectRankingLossTest(unittest.TestCase):
    def test_global_cosine_matches_legacy_distance(self):
        target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        positive = torch.tensor([[[0.8, 0.2], [0.1, 0.9]]])
        negative = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])

        _, metrics = action_effect_ranking_loss(
            positive,
            negative,
            target,
            distance_mode="global_cosine",
        )
        expected = 1.0 - F.cosine_similarity(
            positive.flatten(1),
            target.flatten(1),
        )

        self.assertTrue(
            torch.allclose(metrics["positive_distance"], expected.mean(), atol=1e-6)
        )

    def test_tokenwise_cosine_prevents_large_token_domination(self):
        target = torch.tensor([[[100.0, 0.0], [0.0, 1.0]]])
        positive = torch.tensor([[[100.0, 0.0], [0.0, -1.0]]])
        negative = target.unsqueeze(1)

        _, global_metrics = action_effect_ranking_loss(
            positive,
            negative,
            target,
            distance_mode="global_cosine",
        )
        _, token_metrics = action_effect_ranking_loss(
            positive,
            negative,
            target,
            distance_mode="tokenwise_cosine",
        )

        self.assertLess(global_metrics["positive_distance"].item(), 0.01)
        self.assertGreater(token_metrics["positive_distance"].item(), 0.99)

    def test_mixed_distance_penalizes_incorrect_magnitude(self):
        target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        positive = (0.5 * target).requires_grad_()
        negative = target.unsqueeze(1).clone().requires_grad_()

        loss, metrics = action_effect_ranking_loss(
            positive,
            negative,
            target,
            margin=0.1,
            distance_mode="tokenwise_mixed",
            huber_weight=0.2,
            scale_floor=1e-3,
        )

        self.assertLess(metrics["positive_direction_distance"].item(), 1e-6)
        self.assertGreater(metrics["positive_magnitude_distance"].item(), 0.0)
        self.assertGreater(
            metrics["positive_distance"].item(),
            metrics["positive_direction_distance"].item(),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(positive.grad).all())
        self.assertTrue(torch.isfinite(negative.grad).all())


if __name__ == "__main__":
    unittest.main()
