import copy

import torch

from src.models.vita_latent_flow import (
    FlowBlock,
    GlobalRelativeContextGate,
    LatentFlowNetwork,
    VitaLatentActionGenerator,
)


def _condition_counts():
    return {"local": 4, "cross": 0, "global": 3, "state": 2}


def _source_counts():
    return [("h0_visual", 2), ("h1", 2), ("global", 3), ("state", 2)]


def _cacheable_flow():
    torch.manual_seed(17)
    flow = LatentFlowNetwork(
        latent_dim=16,
        hidden_dim=16,
        num_layers=2,
        cross_attention=True,
        cross_attention_heads=4,
        gated_cross_attention=True,
        token_value_gating=True,
        token_gate_lambda=0.4,
        token_gate_use_action_latent=True,
        competitive_value_gating=True,
        global_relative_context_gating=True,
        global_relative_gate_score_dim=8,
        static_condition_cache=True,
        fixed_cross_attention_scale=0.1,
        condition_token_type_counts=_condition_counts(),
        condition_token_source_counts=_source_counts(),
    )
    for block in flow.blocks:
        torch.nn.init.normal_(block.value_gate_mlp[-1].weight, std=0.05)
        torch.nn.init.normal_(block.value_gate_mlp[-1].bias, std=0.05)
        torch.nn.init.normal_(
            block.global_relative_context_gate.query_proj.weight,
            std=0.05,
        )
    return flow


def test_global_relative_gate_is_identity_at_initialization():
    gate = GlobalRelativeContextGate(
        token_dim=16,
        context_dim=16,
        score_dim=8,
        gate_scale=0.4,
    )
    result = gate(torch.randn(4, 24, 16), torch.randn(4, 16))
    torch.testing.assert_close(
        result["gates"],
        torch.ones_like(result["gates"]),
        atol=1e-6,
        rtol=0,
    )


def test_global_relative_gate_preserves_mean_and_separates_tokens():
    torch.manual_seed(7)
    gate = GlobalRelativeContextGate(
        token_dim=16,
        context_dim=16,
        score_dim=8,
        gate_scale=0.4,
    )
    torch.nn.init.xavier_uniform_(gate.query_proj.weight)
    result = gate(torch.randn(4, 24, 16), torch.randn(4, 16))
    torch.testing.assert_close(
        result["gates"].mean(dim=1),
        torch.ones(4),
        atol=1e-6,
        rtol=0,
    )
    assert torch.all(result["gates"].std(dim=1, unbiased=False) > 0)


def test_global_relative_gate_does_not_split_identical_tokens():
    gate = GlobalRelativeContextGate(
        token_dim=16,
        context_dim=16,
        score_dim=8,
        gate_scale=0.4,
    )
    torch.nn.init.xavier_uniform_(gate.query_proj.weight)
    token = torch.randn(4, 1, 16)
    result = gate(token.expand(-1, 24, -1).clone(), torch.randn(4, 16))
    torch.testing.assert_close(
        result["gates"],
        torch.ones_like(result["gates"]),
        atol=1e-5,
        rtol=0,
    )


def test_global_relative_gate_query_projection_gets_initial_gradient():
    gate = GlobalRelativeContextGate(
        token_dim=16,
        context_dim=16,
        score_dim=8,
        gate_scale=0.4,
    )
    result = gate(torch.randn(4, 24, 16), torch.randn(4, 16))
    loss = (result["gates"] * torch.randn_like(result["gates"])).sum()
    loss.backward()
    assert gate.query_proj.weight.grad is not None
    assert gate.query_proj.weight.grad.abs().sum().item() > 0


def test_competitive_gate_preserves_group_means_and_state():
    block = FlowBlock(
        hidden_dim=16,
        cross_attention=True,
        num_heads=4,
        gated_cross_attention=True,
        token_value_gating=True,
        condition_token_type_counts=_condition_counts(),
        condition_token_source_counts=_source_counts(),
        token_gate_lambda=0.4,
        token_gate_use_action_latent=True,
        competitive_value_gating=True,
        fixed_cross_attention_scale=0.1,
    )
    logits = torch.tensor(
        [
            [-3.0, -1.0, 1.0, 3.0, -2.0, 0.0, 2.0, -4.0, 4.0],
            [3.0, 2.0, -2.0, -3.0, 1.0, 0.0, -1.0, 2.0, -2.0],
        ]
    ).unsqueeze(-1)
    gate = block._competitive_value_gate(logits)

    torch.testing.assert_close(
        gate[:, :4].mean(dim=1),
        torch.ones(2, 1),
    )
    torch.testing.assert_close(
        gate[:, 4:7].mean(dim=1),
        torch.ones(2, 1),
    )
    torch.testing.assert_close(gate[:, 7:], torch.ones(2, 2, 1))
    assert gate.min().item() >= 0.2
    assert gate.max().item() <= 1.8


def test_competitive_gate_is_identity_at_initialization():
    block = FlowBlock(
        hidden_dim=16,
        cross_attention=True,
        num_heads=4,
        gated_cross_attention=True,
        token_value_gating=True,
        condition_token_type_counts=_condition_counts(),
        condition_token_source_counts=_source_counts(),
        token_gate_lambda=0.4,
        token_gate_use_action_latent=True,
        competitive_value_gating=True,
        fixed_cross_attention_scale=0.1,
    )
    block.capture_token_value_gates = True
    x = torch.randn(2, 16)
    context = torch.randn(2, 9, 16)
    time_embedding = torch.randn(2, 16)
    output = block(x, time_embedding, context)

    assert output.shape == x.shape
    assert len(block.captured_token_value_gates) == 1
    torch.testing.assert_close(
        block.captured_token_value_gates[0],
        torch.ones(2, 9, 1),
    )
    assert set(block.last_token_value_gate_metrics) == {
        "gate/h0_visual_mean",
        "gate/h1_mean",
        "gate/global_mean",
        "gate/local_std",
        "gate/global_std",
        "gate/local_logit_std",
        "gate/global_logit_std",
    }


def test_flow_blocks_have_independent_competitive_gate_networks():
    flow = LatentFlowNetwork(
        latent_dim=16,
        hidden_dim=16,
        num_layers=2,
        cross_attention=True,
        cross_attention_heads=4,
        gated_cross_attention=True,
        token_value_gating=True,
        token_gate_lambda=0.4,
        token_gate_use_action_latent=True,
        competitive_value_gating=True,
        fixed_cross_attention_scale=0.1,
        condition_token_type_counts=_condition_counts(),
        condition_token_source_counts=_source_counts(),
    )
    first, second = flow.blocks
    assert (
        first.value_gate_mlp[-1].weight.data_ptr()
        != second.value_gate_mlp[-1].weight.data_ptr()
    )
    assert (
        first.value_gate_source_embedding.weight.data_ptr()
        != second.value_gate_source_embedding.weight.data_ptr()
    )


def test_flow_blocks_have_independent_global_relative_gates():
    flow = LatentFlowNetwork(
        latent_dim=16,
        hidden_dim=16,
        num_layers=2,
        cross_attention=True,
        cross_attention_heads=4,
        gated_cross_attention=True,
        token_value_gating=True,
        token_gate_lambda=0.4,
        token_gate_use_action_latent=True,
        competitive_value_gating=True,
        global_relative_context_gating=True,
        global_relative_gate_score_dim=8,
        fixed_cross_attention_scale=0.1,
        condition_token_type_counts=_condition_counts(),
        condition_token_source_counts=_source_counts(),
    )
    first, second = flow.blocks
    assert (
        first.global_relative_context_gate.query_proj.weight.data_ptr()
        != second.global_relative_context_gate.query_proj.weight.data_ptr()
    )


def test_global_relative_gate_leaves_local_and_state_gates_unchanged():
    common_kwargs = dict(
        hidden_dim=16,
        cross_attention=True,
        num_heads=4,
        gated_cross_attention=True,
        token_value_gating=True,
        condition_token_type_counts=_condition_counts(),
        condition_token_source_counts=_source_counts(),
        token_gate_lambda=0.4,
        token_gate_use_action_latent=True,
        competitive_value_gating=True,
        fixed_cross_attention_scale=0.1,
    )
    baseline = FlowBlock(**common_kwargs)
    relative = FlowBlock(
        **common_kwargs,
        global_relative_context_gating=True,
        global_relative_gate_score_dim=8,
    )
    relative.load_state_dict(baseline.state_dict(), strict=False)
    torch.nn.init.normal_(baseline.value_gate_mlp[-1].weight)
    torch.nn.init.normal_(baseline.value_gate_mlp[-1].bias)
    relative.value_gate_mlp.load_state_dict(baseline.value_gate_mlp.state_dict())
    baseline.capture_token_value_gates = True
    relative.capture_token_value_gates = True
    relative_gate_mlp_token_counts = []
    hook = relative.value_gate_mlp.register_forward_pre_hook(
        lambda _module, inputs: relative_gate_mlp_token_counts.append(
            inputs[0].shape[1]
        )
    )

    x = torch.randn(2, 16)
    context = torch.randn(2, 9, 16)
    time_embedding = torch.randn(2, 16)
    baseline(x, time_embedding, context)
    relative(x, time_embedding, context)
    hook.remove()
    baseline_gates = baseline.captured_token_value_gates[0]
    relative_gates = relative.captured_token_value_gates[0]

    torch.testing.assert_close(relative_gates[:, :4], baseline_gates[:, :4])
    torch.testing.assert_close(relative_gates[:, 7:], baseline_gates[:, 7:])
    torch.testing.assert_close(
        relative_gates[:, 4:7],
        torch.ones_like(relative_gates[:, 4:7]),
    )
    assert relative_gate_mlp_token_counts == [4]


def test_v3_mq54_uses_eight_fine_grained_gate_sources():
    generator = VitaLatentActionGenerator(
        action_dim=7,
        horizon=8,
        vlm_hidden_dim=16,
        latent_dim=16,
        hidden_dim=16,
        action_ae_layers=1,
        flow_layers=2,
        proprio_dim=7,
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
        flow_cross_attention=True,
        flow_cross_attention_heads=4,
        gated_flow_cross_attention=True,
        mq_token_value_gating=True,
        mq_token_gate_lambda=0.4,
        mq_token_gate_use_action_latent=True,
        mq_competitive_value_gating=True,
        mq_global_relative_context_gating=True,
        mq_global_relative_gate_score_dim=8,
        fixed_flow_cross_attention_scale=0.1,
    )
    first_block = generator.flow.blocks[0]
    assert first_block.gate_source_names == (
        "h0_visual",
        "h0_text",
        "h1",
        "h4",
        "h8",
        "h12",
        "global",
        "state",
    )
    assert first_block.value_gate_source_embedding.num_embeddings == 8
    assert first_block.gate_source_ids.numel() == 54
    assert first_block.condition_type_ids.numel() == 54
    assert first_block.global_relative_context_gate.score_dim == 8


def test_global_residual_block_preserves_learned_query_identity():
    generator = VitaLatentActionGenerator(
        action_dim=7,
        horizon=8,
        vlm_hidden_dim=16,
        latent_dim=16,
        hidden_dim=16,
        action_ae_layers=1,
        flow_layers=1,
        proprio_dim=7,
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
        hierarchical_global_residual_block=True,
        hierarchical_global_residual_scale=0.1,
        hierarchical_global_ffn_ratio=2.0,
        flow_cross_attention=True,
        flow_cross_attention_heads=4,
        gated_flow_cross_attention=True,
        mq_token_value_gating=True,
        mq_token_gate_lambda=0.4,
        mq_token_gate_use_action_latent=True,
        mq_competitive_value_gating=True,
        fixed_flow_cross_attention_scale=0.1,
    )
    encoder = generator.observation_encoder
    with torch.no_grad():
        for parameter in encoder.global_attention.parameters():
            parameter.zero_()
        for parameter in encoder.global_residual_ffn.parameters():
            parameter.zero_()

    hidden_states = tuple(torch.randn(2, 8, 16) for _ in range(13))
    token_source_ids = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]] * 2)
    condition_tokens = encoder._hierarchical_pool_project_tokens(
        hidden_states,
        hidden_token_type_ids=token_source_ids,
    )
    global_tokens = condition_tokens[:, 28:52]
    expected_queries = encoder.global_query.expand(2, -1, -1)

    torch.testing.assert_close(global_tokens, expected_queries)
    assert encoder.last_global_token_metrics[
        "global_pool/query_token_std"
    ].item() > 0
    torch.testing.assert_close(
        encoder.last_global_token_metrics[
            "global_pool/after_attention_residual_token_std"
        ],
        encoder.last_global_token_metrics["global_pool/query_token_std"],
    )


def test_static_condition_cache_preserves_forward_output():
    flow = _cacheable_flow().eval()
    latent = torch.randn(3, 16)
    timestep = torch.rand(3)
    condition = torch.randn(3, 9, 16)

    uncached = flow(latent, timestep, condition_tokens=condition)
    cache = flow.prepare_condition_cache(condition)
    cached = flow(
        latent,
        timestep,
        condition_tokens=condition,
        condition_cache=cache,
    )

    torch.testing.assert_close(cached, uncached, atol=0, rtol=0)


def test_static_condition_cache_preserves_repeated_call_gradients():
    uncached_flow = _cacheable_flow().eval()
    cached_flow = copy.deepcopy(uncached_flow).eval()
    uncached_condition = torch.randn(2, 9, 16, requires_grad=True)
    cached_condition = uncached_condition.detach().clone().requires_grad_(True)
    latents = [torch.randn(2, 16) for _ in range(7)]
    timesteps = [torch.rand(2) for _ in range(7)]

    uncached_loss = sum(
        uncached_flow(
            latent,
            timestep,
            condition_tokens=uncached_condition,
        ).square().mean()
        for latent, timestep in zip(latents, timesteps)
    )
    cache = cached_flow.prepare_condition_cache(cached_condition)
    cached_loss = sum(
        cached_flow(
            latent,
            timestep,
            condition_tokens=cached_condition,
            condition_cache=cache,
        ).square().mean()
        for latent, timestep in zip(latents, timesteps)
    )
    uncached_loss.backward()
    cached_loss.backward()

    torch.testing.assert_close(cached_loss, uncached_loss, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        cached_condition.grad,
        uncached_condition.grad,
        atol=2e-6,
        rtol=2e-5,
    )
    cached_parameters = dict(cached_flow.named_parameters())
    for name, uncached_parameter in uncached_flow.named_parameters():
        cached_parameter = cached_parameters[name]
        assert (uncached_parameter.grad is None) == (
            cached_parameter.grad is None
        ), name
        if uncached_parameter.grad is not None:
            torch.testing.assert_close(
                cached_parameter.grad,
                uncached_parameter.grad,
                atol=2e-6,
                rtol=2e-5,
                msg=lambda message, parameter_name=name: (
                    f"{parameter_name}: {message}"
                ),
            )


def test_static_condition_cache_computes_static_projections_once():
    flow = _cacheable_flow().eval()
    condition = torch.randn(2, 9, 16)
    call_counts = {
        "condition_proj": 0,
        "token_mlp": 0,
        "key_proj": 0,
        "value_proj": 0,
        "relative_key_proj": 0,
    }
    hooks = [
        flow.condition_proj.register_forward_hook(
            lambda *_: call_counts.__setitem__(
                "condition_proj",
                call_counts["condition_proj"] + 1,
            )
        )
    ]
    for block in flow.blocks:
        hooks.extend(
            [
                block.value_gate_token_mlp.register_forward_hook(
                    lambda *_: call_counts.__setitem__(
                        "token_mlp",
                        call_counts["token_mlp"] + 1,
                    )
                ),
                block.value_gate_k_proj.register_forward_hook(
                    lambda *_: call_counts.__setitem__(
                        "key_proj",
                        call_counts["key_proj"] + 1,
                    )
                ),
                block.value_gate_v_proj.register_forward_hook(
                    lambda *_: call_counts.__setitem__(
                        "value_proj",
                        call_counts["value_proj"] + 1,
                    )
                ),
                block.global_relative_context_gate.key_proj.register_forward_hook(
                    lambda *_: call_counts.__setitem__(
                        "relative_key_proj",
                        call_counts["relative_key_proj"] + 1,
                    )
                ),
            ]
        )

    cache = flow.prepare_condition_cache(condition)
    for _ in range(7):
        flow(
            torch.randn(2, 16),
            torch.rand(2),
            condition_tokens=condition,
            condition_cache=cache,
        )
    for hook in hooks:
        hook.remove()

    assert call_counts == {
        "condition_proj": 1,
        "token_mlp": 2,
        "key_proj": 2,
        "value_proj": 2,
        "relative_key_proj": 2,
    }
