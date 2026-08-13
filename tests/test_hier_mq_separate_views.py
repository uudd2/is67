import torch

from src.models.vita_latent_flow import LatentFlowNetwork, ObservationLatentEncoder


LOCAL_COUNTS = [8, 4, 4, 4, 4, 4]
SOURCE_MODES = ["visual", "text", "all", "all", "all", "all"]
HIDDEN_LAYERS = [0, 0, 1, 4, 8, 12]


def _encoder(hidden_dim=32, separate_views=False):
    return ObservationLatentEncoder(
        vlm_hidden_dim=hidden_dim,
        latent_dim=hidden_dim,
        hidden_dim=hidden_dim,
        proprio_dim=7,
        condition_source="hidden_layer",
        hidden_layer_index=12,
        hidden_pooling="attention",
        pooling_num_heads=4,
        pooling_num_queries=52,
        hierarchical_query_pooling=True,
        layer_local_queries_per_layer_list=LOCAL_COUNTS,
        layer_local_token_source_modes=SOURCE_MODES,
        hier_mq_separate_views=separate_views,
        cross_layer_queries=0,
        global_queries=24,
        blockwise_hidden_layer_indices=HIDDEN_LAYERS,
        state_token_conditioning=True,
        state_num_tokens=2,
        state_token_dropout=0.0,
    ).eval()


def _inputs(hidden_dim=32, batch_size=2):
    visual_per_view = 4
    text_tokens = 3
    sequence_length = 2 * visual_per_view + text_tokens
    hidden_states = tuple(
        torch.randn(batch_size, sequence_length, hidden_dim)
        for _ in range(13)
    )
    token_types = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    token_types[:, : 2 * visual_per_view] = 1
    grids = torch.tensor([[1, 2, 2], [1, 2, 2]] * batch_size)
    proprioception = torch.randn(batch_size, 7)
    return hidden_states, token_types, grids, proprioception


def _local_and_global(encoder, inputs):
    hidden_states, token_types, grids, _ = inputs
    tokens = encoder._hierarchical_pool_project_tokens(
        hidden_states,
        hidden_token_type_ids=token_types,
        image_grid_thw=grids,
        spatial_merge_size=1,
    )
    return tokens[:, :28], tokens[:, 28:]


def test_switch_disabled_preserves_original_hier_mq_values():
    torch.manual_seed(1)
    default_encoder = _encoder()
    explicit_off_encoder = _encoder(separate_views=False)
    explicit_off_encoder.load_state_dict(default_encoder.state_dict(), strict=True)
    inputs = _inputs()
    with torch.no_grad():
        default_output = _local_and_global(default_encoder, inputs)
        explicit_output = _local_and_global(explicit_off_encoder, inputs)
    assert torch.equal(default_output[0], explicit_output[0])
    assert torch.equal(default_output[1], explicit_output[1])


def test_local_global_state_and_condition_shapes():
    encoder = _encoder(hidden_dim=1024, separate_views=True)
    inputs = _inputs(hidden_dim=1024, batch_size=1)
    hidden_states, token_types, grids, proprioception = inputs
    with torch.no_grad():
        local, global_tokens = _local_and_global(encoder, inputs)
        _, condition = encoder(
            hidden_states=hidden_states,
            hidden_token_type_ids=token_types,
            proprioception=proprioception,
            return_condition_tokens=True,
            image_grid_thw=grids,
            spatial_merge_size=1,
        )
    state = condition[:, 52:]
    assert local.shape == (1, 28, 1024)
    assert global_tokens.shape == (1, 24, 1024)
    assert state.shape == (1, 2, 1024)
    assert condition.shape == (1, 54, 1024)


def test_h0_visual_output_order_is_view1_then_view2():
    encoder = _encoder(separate_views=True)
    assert LOCAL_COUNTS[0] == 8
    assert encoder._make_shared_view_queries(encoder.layer_local_query[:8]).shape[0] == 4


def test_hidden_layer_groups_output_view1_two_then_view2_two():
    encoder = _encoder(separate_views=True)
    offset = LOCAL_COUNTS[0] + LOCAL_COUNTS[1]
    for count in LOCAL_COUNTS[2:]:
        queries = encoder.layer_local_query[offset : offset + count]
        assert encoder._make_shared_view_queries(queries).shape[0] == 2
        offset += count


def test_two_views_share_query_parameters_and_pooler_instance():
    encoder = _encoder(separate_views=True)
    assert len([module for module in encoder.modules() if module is encoder.pool_attention]) == 1
    shared = encoder._make_shared_view_queries(encoder.layer_local_query[:8])
    assert torch.equal(
        shared,
        0.5 * (encoder.layer_local_query[:4] + encoder.layer_local_query[4:8]),
    )


def test_view2_perturbation_does_not_change_view1_or_h0_text_local_mq():
    encoder = _encoder(separate_views=True)
    inputs = _inputs(batch_size=1)
    hidden_states, token_types, grids, proprioception = inputs
    perturbed = tuple(layer.clone() for layer in hidden_states)
    for layer in perturbed:
        layer[:, 4:8] += 100.0 * torch.randn_like(layer[:, 4:8])
    with torch.no_grad():
        original_local, _ = _local_and_global(encoder, inputs)
        changed_local, _ = _local_and_global(
            encoder,
            (perturbed, token_types, grids, proprioception),
        )
    view1_slices = [(0, 4), (12, 14), (16, 18), (20, 22), (24, 26)]
    for start, end in view1_slices:
        assert torch.equal(original_local[:, start:end], changed_local[:, start:end])


def test_h0_text_output_matches_original_hier_mq_path():
    old_encoder = _encoder(separate_views=False)
    encoder = _encoder(separate_views=True)
    encoder.load_state_dict(old_encoder.state_dict(), strict=True)
    inputs = _inputs(batch_size=1)
    with torch.no_grad():
        old_local, _ = _local_and_global(old_encoder, inputs)
        new_local, _ = _local_and_global(encoder, inputs)
    assert torch.equal(old_local[:, 8:12], new_local[:, 8:12])


def test_split_masks_preserve_original_nonvisual_ranges():
    encoder = _encoder(separate_views=True)
    _, token_types, grids, _ = _inputs()
    view_masks = encoder._view_visual_masks(token_types, grids, 1)
    all_visual = view_masks[0] | view_masks[1]
    for source_mode in SOURCE_MODES:
        original = encoder._source_key_padding_mask(token_types, source_mode)
        if source_mode == "text":
            continue
        split_masks = encoder._split_view_context_masks(original, view_masks)
        original_allowed = ~original
        for view_mask, split_mask in zip(view_masks, split_masks):
            expected_allowed = original_allowed & (~all_visual | view_mask)
            assert torch.equal(~split_mask, expected_allowed)
            assert torch.equal(
                (~split_mask) & ~all_visual,
                original_allowed & ~all_visual,
            )


def test_existing_checkpoint_parameters_load_strictly_both_directions():
    old_encoder = _encoder(separate_views=False)
    new_encoder = _encoder(separate_views=True)
    new_encoder.load_state_dict(old_encoder.state_dict(), strict=True)
    old_encoder.load_state_dict(new_encoder.state_dict(), strict=True)
    assert old_encoder.layer_local_query.shape == new_encoder.layer_local_query.shape
    assert old_encoder.global_query.shape == new_encoder.global_query.shape


def test_flow_input_and_output_shape_remain_single_latent_token():
    flow = LatentFlowNetwork(latent_dim=1024, hidden_dim=32, num_layers=1)
    latent = torch.randn(1, 1, 1024)
    assert flow(latent, torch.rand(1)).shape == latent.shape
