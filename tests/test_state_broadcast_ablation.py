import torch

from src.models.vita_latent_flow import ObservationLatentEncoder


def _build_encoder(state_broadcast_to_mq=True):
    return ObservationLatentEncoder(
        vlm_hidden_dim=16,
        latent_dim=16,
        hidden_dim=32,
        proprio_dim=14,
        condition_source="hidden_layer",
        hidden_layer_index=12,
        hidden_pooling="attention",
        pooling_num_heads=4,
        pooling_num_queries=52,
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
        blockwise_hidden_layer_indices=[0, 0, 1, 4, 8, 12],
        state_token_conditioning=True,
        state_num_tokens=2,
        state_token_dropout=0.0,
        hierarchical_global_residual_block=True,
        hierarchical_global_residual_scale=0.1,
        hierarchical_global_ffn_ratio=4.0,
        state_broadcast_to_mq=state_broadcast_to_mq,
    ).eval()


def _inputs():
    torch.manual_seed(17)
    hidden_states = tuple(torch.randn(2, 8, 16) for _ in range(13))
    token_types = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]] * 2)
    proprio_a = torch.randn(2, 2, 7)
    proprio_b = torch.randn(2, 2, 7)
    return hidden_states, token_types, proprio_a, proprio_b


@torch.no_grad()
def test_disabling_state_broadcast_keeps_mq_state_independent():
    encoder = _build_encoder(state_broadcast_to_mq=False)
    hidden_states, token_types, proprio_a, proprio_b = _inputs()

    _, tokens_a = encoder(
        hidden_states=hidden_states,
        hidden_token_type_ids=token_types,
        proprioception=proprio_a,
        return_condition_tokens=True,
    )
    _, tokens_b = encoder(
        hidden_states=hidden_states,
        hidden_token_type_ids=token_types,
        proprioception=proprio_b,
        return_condition_tokens=True,
    )

    assert tokens_a.shape == (2, 54, 16)
    torch.testing.assert_close(tokens_a[:, :52], tokens_b[:, :52])
    assert not torch.allclose(tokens_a[:, 52:], tokens_b[:, 52:])
    assert not torch.allclose(
        tokens_a[:, 52:].mean(dim=1),
        tokens_b[:, 52:].mean(dim=1),
    )


@torch.no_grad()
def test_default_state_broadcast_preserves_legacy_behavior():
    encoder = _build_encoder()
    hidden_states, token_types, proprio_a, proprio_b = _inputs()

    _, tokens_a = encoder(
        hidden_states=hidden_states,
        hidden_token_type_ids=token_types,
        proprioception=proprio_a,
        return_condition_tokens=True,
    )
    _, tokens_b = encoder(
        hidden_states=hidden_states,
        hidden_token_type_ids=token_types,
        proprioception=proprio_b,
        return_condition_tokens=True,
    )

    assert encoder.state_broadcast_to_mq is True
    assert not torch.allclose(tokens_a[:, :52], tokens_b[:, :52])
