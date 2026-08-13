import unittest

import torch

from src.models.vita_latent_flow import (
    ObservationLatentEncoder,
    VitaLatentActionGenerator,
    attention_concentration_score,
    select_dynamic_mq,
)


def _build_encoder(dynamic_topk, proprio_dim=None, candidates_per_source=8):
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
        layer_local_queries_per_layer_list=(
            [
                candidates_per_source,
                4,
                candidates_per_source,
                candidates_per_source,
                candidates_per_source,
                candidates_per_source,
            ]
            if dynamic_topk
            else [8, 4, 4, 4, 4, 4]
        ),
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
        dynamic_topk_local_mq=dynamic_topk,
        dynamic_topk_candidate_source_positions=[0, 2, 3, 4, 5],
        dynamic_topk_text_source_position=1,
        dynamic_topk_candidates_per_source=candidates_per_source,
        dynamic_topk_min_keep_per_source=2,
        dynamic_topk_total_keep=24,
        dynamic_topk_train_random_exploration=dynamic_topk,
        blockwise_hidden_layer_indices=[0, 0, 1, 4, 8, 12],
        state_token_conditioning=proprio_dim is not None,
        state_num_tokens=2,
        state_token_dropout=0.0,
    )


def _inputs(batch_size=3):
    torch.manual_seed(19)
    hidden_states = [
        torch.randn(batch_size, 9, 16)
        for _ in range(13)
    ]
    token_types = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 3, -1, -1],
            [0, 0, 0, 1, 1, 1, 1, 3, -1],
            [0, 1, 1, 1, 1, 1, 3, 3, 3],
        ][:batch_size],
        dtype=torch.long,
    )
    return hidden_states, token_types


class DynamicTopKMQTest(unittest.TestCase):
    def test_attention_concentration_excludes_padding(self):
        attention = torch.tensor(
            [
                [
                    [
                        [0.25, 0.25, 0.25, 0.25],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                ],
                [
                    [
                        [0.25, 0.25, 0.25, 0.25],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                ],
            ],
            dtype=torch.float32,
        )
        padding_mask = torch.tensor(
            [
                [False, False, True, True],
                [False, False, False, False],
            ]
        )
        scores = attention_concentration_score(
            attention,
            key_padding_mask=padding_mask,
        )
        torch.testing.assert_close(scores[:, 0], torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(scores[:, 1], torch.ones(2), atol=1e-6, rtol=0)
        self.assertFalse(scores.requires_grad)

    def test_normalized_entropy_matches_different_valid_lengths(self):
        attention = torch.tensor(
            [
                [[
                    [0.5, 0.5, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]],
                [[
                    [0.25, 0.25, 0.25, 0.25],
                    [1.0, 0.0, 0.0, 0.0],
                ]],
            ]
        )
        padding_mask = torch.tensor(
            [
                [False, False, True, True],
                [False, False, False, False],
            ]
        )
        scores = attention_concentration_score(attention, padding_mask)
        torch.testing.assert_close(scores[0], scores[1], atol=1e-6, rtol=0)

    def test_vectorized_selection_has_minimum_coverage_and_stable_order(self):
        batch_size, groups, queries, hidden_dim = 2, 5, 8, 3
        token_ids = torch.arange(groups * queries).view(1, groups, queries, 1)
        candidate_tokens = token_ids.expand(batch_size, -1, -1, hidden_dim).float()
        candidate_scores = torch.tensor(
            [
                [
                    [8, 7, 6, 5, 4, 3, 2, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 1, 8, 2, 7, 3, 6, 5],
                    [2, 8, 1, 7, 3, 6, 4, 5],
                    [5, 4, 3, 2, 1, 6, 7, 8],
                ],
                [
                    [1, 3, 5, 7, 8, 6, 4, 2],
                    [8, 6, 4, 2, 1, 3, 5, 7],
                    [2, 4, 6, 8, 7, 5, 3, 1],
                    [7, 5, 3, 1, 2, 4, 6, 8],
                    [3, 1, 4, 2, 8, 6, 7, 5],
                ],
            ],
            dtype=torch.float32,
        )
        (
            selected,
            counts,
            mask,
            indices,
            score_mask,
            random_mask,
        ) = select_dynamic_mq(
            candidate_tokens,
            candidate_scores,
            min_keep_per_group=2,
            total_keep=24,
        )
        self.assertEqual(tuple(selected.shape), (2, 24, hidden_dim))
        self.assertTrue((counts >= 2).all())
        self.assertTrue((counts.sum(dim=-1) == 24).all())
        self.assertTrue((mask.sum(dim=(1, 2)) == 24).all())
        self.assertTrue((indices[:, 1:] > indices[:, :-1]).all())
        torch.testing.assert_close(selected[:, :, 0], indices.float())
        torch.testing.assert_close(score_mask, mask)
        self.assertFalse(random_mask.any())

    def test_training_uses_top1_plus_one_random_candidate_per_source(self):
        torch.manual_seed(31)
        candidate_tokens = torch.randn(4, 5, 8, 6)
        candidate_scores = torch.arange(8).view(1, 1, 8).expand(4, 5, -1).float()
        (
            _,
            counts,
            selected_mask,
            _,
            score_mask,
            random_mask,
        ) = select_dynamic_mq(
            candidate_tokens,
            candidate_scores,
            min_keep_per_group=2,
            total_keep=24,
            random_exploration=True,
        )
        self.assertTrue((counts >= 2).all())
        self.assertTrue((counts.sum(dim=-1) == 24).all())
        self.assertTrue((random_mask.sum(dim=-1) == 1).all())
        self.assertTrue(selected_mask[:, :, 7].all())
        self.assertTrue(score_mask[:, :, 7].all())
        self.assertFalse(random_mask[:, :, 7].any())
        self.assertTrue((random_mask.sum(dim=(1, 2)) == 5).all())
        self.assertTrue((score_mask.sum(dim=(1, 2)) == 19).all())

    def test_encoder_outputs_mq54_and_is_deterministic(self):
        encoder = _build_encoder(dynamic_topk=True, proprio_dim=14).eval()
        hidden_states, token_types = _inputs()
        proprio = torch.randn(3, 2, 7)
        first_latent, first_tokens = encoder(
            hidden_states=hidden_states,
            hidden_token_type_ids=token_types,
            proprioception=proprio,
            return_condition_tokens=True,
        )
        first_indices = encoder.last_dynamic_topk_selected_indices.clone()
        second_latent, second_tokens = encoder(
            hidden_states=hidden_states,
            hidden_token_type_ids=token_types,
            proprioception=proprio,
            return_condition_tokens=True,
        )
        self.assertEqual(tuple(first_latent.shape), (3, 16))
        self.assertEqual(tuple(first_tokens.shape), (3, 54, 16))
        self.assertTrue(
            (
                encoder.last_dynamic_topk_selected_mask.sum(dim=-1)
                >= 2
            ).all()
        )
        self.assertTrue(
            (
                encoder.last_dynamic_topk_selected_mask.sum(dim=(1, 2))
                == 24
            ).all()
        )
        torch.testing.assert_close(
            first_indices,
            encoder.last_dynamic_topk_selected_indices,
        )
        torch.testing.assert_close(first_latent, second_latent)
        torch.testing.assert_close(first_tokens, second_tokens)

    def test_encoder_training_records_source_selection_counts(self):
        encoder = _build_encoder(dynamic_topk=True).train()
        hidden_states, token_types = _inputs()
        _, condition_tokens = encoder(
            hidden_states=hidden_states,
            hidden_token_type_ids=token_types,
            return_condition_tokens=True,
        )
        self.assertEqual(tuple(condition_tokens.shape), (3, 52, 16))
        self.assertTrue(
            (
                encoder.last_dynamic_topk_random_selected_mask.sum(dim=-1)
                == 1
            ).all()
        )
        self.assertTrue(
            (
                encoder.last_dynamic_topk_score_selected_mask.sum(
                    dim=(1, 2)
                )
                == 19
            ).all()
        )
        self.assertIn(
            "dynamic_select/h1_selected_count_mean",
            encoder.last_dynamic_topk_metrics,
        )
        self.assertFalse(
            any(
                "_query_" in name
                for name in encoder.last_dynamic_topk_metrics
            )
        )
        source_count_keys = [
            name
            for name in encoder.last_dynamic_topk_metrics
            if name.endswith("_selected_count_mean")
        ]
        self.assertEqual(len(source_count_keys), 5)
        torch.testing.assert_close(
            sum(
                encoder.last_dynamic_topk_metrics[name]
                for name in source_count_keys
            ),
            torch.tensor(24.0),
        )

    def test_old_mq54_checkpoint_queries_expand_per_source(self):
        torch.manual_seed(23)
        baseline = _build_encoder(dynamic_topk=False)
        dynamic = _build_encoder(dynamic_topk=True)
        baseline_queries = baseline.layer_local_query.detach().clone()
        missing, unexpected = dynamic.load_state_dict(
            baseline.state_dict(),
            strict=False,
        )
        self.assertFalse(missing)
        self.assertFalse(unexpected)

        expanded = dynamic.layer_local_query.detach()
        old_offsets = [0, 8, 12, 16, 20, 24, 28]
        new_offsets = [0, 8, 12, 20, 28, 36, 44]
        for position in range(6):
            old_group = baseline_queries[
                old_offsets[position] : old_offsets[position + 1]
            ]
            new_group = expanded[
                new_offsets[position] : new_offsets[position + 1]
            ]
            torch.testing.assert_close(
                new_group[: old_group.shape[0]],
                old_group,
            )
            if position >= 2:
                self.assertLess(
                    (
                        new_group[old_group.shape[0] :] - old_group
                    ).abs().max().item(),
                    0.01,
                )

    def test_candidate12_control_loads_old_queries_and_keeps_mq54(self):
        torch.manual_seed(29)
        baseline = _build_encoder(dynamic_topk=False, proprio_dim=14)
        candidate12 = _build_encoder(
            dynamic_topk=True,
            proprio_dim=14,
            candidates_per_source=12,
        )
        baseline_queries = baseline.layer_local_query.detach().clone()
        missing, unexpected = candidate12.load_state_dict(
            baseline.state_dict(),
            strict=False,
        )
        self.assertFalse(missing)
        self.assertFalse(unexpected)
        self.assertEqual(
            tuple(candidate12.layer_local_query.shape),
            (64, 16),
        )

        old_offsets = [0, 8, 12, 16, 20, 24, 28]
        new_offsets = [0, 12, 16, 28, 40, 52, 64]
        for position in range(6):
            old_group = baseline_queries[
                old_offsets[position] : old_offsets[position + 1]
            ]
            new_group = candidate12.layer_local_query.detach()[
                new_offsets[position] : new_offsets[position + 1]
            ]
            torch.testing.assert_close(
                new_group[: old_group.shape[0]],
                old_group,
            )
            if new_group.shape[0] > old_group.shape[0]:
                expected = old_group.repeat(
                    3 if old_group.shape[0] == 4 else 2,
                    1,
                )[: new_group.shape[0] - old_group.shape[0]]
                self.assertLess(
                    (
                        new_group[old_group.shape[0] :] - expected
                    ).abs().max().item(),
                    0.01,
                )

        hidden_states, token_types = _inputs()
        proprio = torch.randn(3, 2, 7)
        _, condition_tokens = candidate12.eval()(
            hidden_states=hidden_states,
            hidden_token_type_ids=token_types,
            proprioception=proprio,
            return_condition_tokens=True,
        )
        self.assertEqual(tuple(condition_tokens.shape), (3, 54, 16))

    def test_flow_type_boundaries_use_selected_local_count(self):
        generator = VitaLatentActionGenerator(
            action_dim=7,
            horizon=8,
            vlm_hidden_dim=16,
            latent_dim=16,
            hidden_dim=16,
            action_ae_layers=1,
            flow_layers=2,
            proprio_dim=14,
            hidden_pooling="attention",
            pooling_num_heads=4,
            hierarchical_query_pooling=True,
            layer_local_queries_per_layer_list=[8, 4, 8, 8, 8, 8],
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
            dynamic_topk_local_mq=True,
            dynamic_topk_candidate_source_positions=[0, 2, 3, 4, 5],
            dynamic_topk_text_source_position=1,
            dynamic_topk_candidates_per_source=8,
            dynamic_topk_min_keep_per_source=2,
            dynamic_topk_total_keep=24,
            dynamic_topk_train_random_exploration=True,
            blockwise_hidden_layer_indices=[0, 0, 1, 4, 8, 12],
            state_token_conditioning=True,
            state_num_tokens=2,
            flow_cross_attention=True,
            flow_cross_attention_heads=4,
            gated_flow_cross_attention=True,
            mq_token_value_gating=True,
            mq_token_gate_lambda=0.8,
            mq_token_gate_use_action_latent=True,
            fixed_flow_cross_attention_scale=0.1,
        )
        self.assertEqual(
            generator.flow.condition_token_type_counts,
            {
                "local": 28,
                "cross": 0,
                "global": 24,
                "state": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
