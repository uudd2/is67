import unittest

import torch

from src.models.vita_latent_flow import ObservationLatentEncoder


def _build_encoder(adaptive, warmup_steps=5000, proprio_dim=None):
    return ObservationLatentEncoder(
        vlm_hidden_dim=16,
        latent_dim=16,
        hidden_dim=32,
        proprio_dim=proprio_dim,
        hidden_pooling="attention",
        pooling_num_heads=4,
        pooling_num_queries=52,
        gated_weighted_pooling=True,
        hierarchical_query_pooling=True,
        layer_local_queries_per_layer_list=[8, 4, 4, 4, 4, 4],
        layer_local_token_source_modes=[
            "visual",
            "text",
            "all",
            "all",
            "all",
            "all",
        ],
        cross_layer_queries=0,
        global_queries=24,
        adaptive_local_mq=adaptive,
        adaptive_mq_router_dim=8,
        adaptive_mq_num_slots=4,
        adaptive_mq_reserve_source_positions=[0, 2, 3, 4, 5],
        adaptive_mq_temperature=1.0,
        adaptive_mq_route_warmup_steps=warmup_steps,
        blockwise_hidden_layer_indices=[0, 0, 1, 2, 3, 4],
        state_token_conditioning=proprio_dim is not None,
        state_num_tokens=2,
    )


def _inputs(batch_size=2):
    torch.manual_seed(11)
    hidden_states = [
        torch.randn(batch_size, 7, 16)
        for _ in range(5)
    ]
    token_types = torch.tensor(
        [[0, 0, 1, 1, 1, 3, 3]] * batch_size,
        dtype=torch.long,
    )
    return hidden_states, token_types


class AdaptiveMQRouterTest(unittest.TestCase):
    def test_alpha_zero_exactly_recovers_old_mq54_order(self):
        torch.manual_seed(7)
        baseline = _build_encoder(adaptive=False, proprio_dim=14)
        adaptive = _build_encoder(adaptive=True, proprio_dim=14)
        missing, unexpected = adaptive.load_state_dict(
            baseline.state_dict(),
            strict=False,
        )
        self.assertFalse(unexpected)
        self.assertTrue(
            all(
                "adaptive_depth_router" in key
                for key in missing
            )
        )

        baseline.train()
        adaptive.train()
        adaptive.set_adaptive_mq_step(0)
        hidden_states, token_types = _inputs()
        proprio = torch.randn(2, 2, 7)
        _, baseline_tokens = baseline(
            hidden_states=hidden_states,
            hidden_token_type_ids=token_types,
            proprioception=proprio,
            return_condition_tokens=True,
        )
        _, adaptive_tokens = adaptive(
            hidden_states=hidden_states,
            hidden_token_type_ids=token_types,
            proprioception=proprio,
            return_condition_tokens=True,
        )

        self.assertEqual(tuple(adaptive_tokens.shape), (2, 54, 16))
        torch.testing.assert_close(
            adaptive_tokens,
            baseline_tokens,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            adaptive.last_adaptive_mq_route_weights.sum(dim=-1),
            torch.ones(2, 4),
        )
        source_mass = torch.stack(
            [
                value
                for name, value in adaptive.last_adaptive_mq_router_metrics.items()
                if name.endswith("_mass")
            ]
        ).sum()
        torch.testing.assert_close(source_mass, torch.tensor(4.0))

        adaptive.set_adaptive_mq_step(5000)
        _, fully_routed_tokens = adaptive(
            hidden_states=hidden_states,
            hidden_token_type_ids=token_types,
            proprioception=proprio,
            return_condition_tokens=True,
        )
        torch.testing.assert_close(
            fully_routed_tokens[:, :4],
            baseline_tokens[:, :4],
        )
        torch.testing.assert_close(
            fully_routed_tokens[:, 8:28],
            baseline_tokens[:, 8:28],
        )

    def test_full_routing_backpropagates_to_router_and_reserve_queries(self):
        torch.manual_seed(17)
        encoder = _build_encoder(adaptive=True)
        encoder.train()
        encoder.set_adaptive_mq_step(5000)
        hidden_states, token_types = _inputs()
        _, condition_tokens = encoder(
            hidden_states=hidden_states,
            hidden_token_type_ids=token_types,
            return_condition_tokens=True,
        )
        condition_tokens.square().mean().backward()

        self.assertEqual(tuple(condition_tokens.shape), (2, 52, 16))
        self.assertIsNotNone(
            encoder.adaptive_depth_router.selector_queries.grad
        )
        self.assertGreater(
            encoder.adaptive_depth_router.selector_queries.grad.abs().sum().item(),
            0,
        )
        self.assertGreater(
            encoder.adaptive_depth_router.key_proj.weight.grad.abs().sum().item(),
            0,
        )
        self.assertGreater(
            encoder.adaptive_depth_router.source_embedding.grad.abs().sum().item(),
            0,
        )
        self.assertIsNotNone(encoder.adaptive_reserve_query.grad)
        reserve_grad = encoder.adaptive_reserve_query.grad.view(4, 4, -1)
        self.assertTrue((reserve_grad.abs().sum(dim=(1, 2)) > 0).all())

    def test_eval_always_uses_full_adaptive_routing(self):
        encoder = _build_encoder(adaptive=True)
        encoder.set_adaptive_mq_step(0)
        encoder.eval()
        hidden_states, token_types = _inputs(batch_size=1)
        encoder(
            hidden_states=hidden_states,
            hidden_token_type_ids=token_types,
            return_condition_tokens=True,
        )
        route_alpha = encoder.last_adaptive_mq_router_metrics[
            "router/route_alpha"
        ]
        self.assertEqual(route_alpha.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
