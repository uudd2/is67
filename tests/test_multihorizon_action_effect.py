import unittest

import numpy as np
import torch

from scripts.train import DataCollatorForVLANeXt
from src.models.vita_latent_flow import (
    CausalActionTokenEncoder,
    MultiHorizonVisualPredictor,
    VitaLatentActionGenerator,
    motion_weighted_dense_huber_loss,
)


class _FakeQwenProcessor:
    def __init__(self):
        self.image_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "prompt"

    def __call__(self, text, images, padding, return_tensors):
        self.image_calls.append([np.asarray(image).copy() for image in images])
        batch_size = len(text)
        image_count = len(images)
        return {
            "input_ids": torch.zeros(batch_size, 1, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 1, dtype=torch.long),
            "pixel_values": torch.zeros(image_count, 4),
            "image_grid_thw": torch.tensor(
                [[1, 2, 2]] * image_count,
                dtype=torch.long,
            ),
        }


class _FakeTeacherProcessor:
    def __init__(self):
        self.images = None

    def __call__(self, images, return_tensors):
        self.images = [np.asarray(image).copy() for image in images]
        return {
            "pixel_values": torch.zeros(len(images), 3, 8, 8),
        }


class MultiHorizonCollatorTest(unittest.TestCase):
    def test_current_only_teacher_supports_separate_multi_view_fields(self):
        processor = _FakeQwenProcessor()
        teacher_processor = _FakeTeacherProcessor()
        collator = DataCollatorForVLANeXt(
            processor=processor,
            use_proprio_input_vlm=False,
            use_action_input_policy=False,
            input_modality="image",
            view_mode="multi",
            augmentation={"enabled": False},
            include_action_effect_teacher_images=True,
            action_effect_teacher_processor=teacher_processor,
            action_effect_teacher_type="dinov2",
            action_effect_teacher_current_only=True,
            include_proprio=True,
        )
        main = np.zeros((12, 12, 3), dtype=np.uint8)
        wrist = np.ones((12, 12, 3), dtype=np.uint8)
        sample = {
            "instruction": "move",
            "image": main,
            "image_wrist": wrist,
            "future_actions": torch.zeros(8, 7),
            "proprioception": torch.zeros(8, 7),
            "history_actions": torch.zeros(8, 7),
        }

        inputs, *_ = collator([sample])

        self.assertEqual(len(teacher_processor.images), 1)
        self.assertTrue(np.array_equal(teacher_processor.images[0], main))
        self.assertEqual(
            inputs["action_effect_teacher_current_pixel_values"].shape,
            (1, 3, 8, 8),
        )

    def test_keeps_all_horizons_and_shares_main_view_augmentation(self):
        processor = _FakeQwenProcessor()
        collator = DataCollatorForVLANeXt(
            processor=processor,
            use_proprio_input_vlm=False,
            use_action_input_policy=False,
            input_modality="image",
            view_mode="multi",
            augmentation={
                "enabled": True,
                "random_resized_crop": {
                    "scale": [0.8, 1.0],
                    "ratio": [0.9, 1.1],
                },
                "random_brightness": [0.2],
                "augment_order": [
                    "random_resized_crop",
                    "random_brightness",
                ],
            },
            load_future_image=True,
            include_future_vlm=True,
            future_vlm_main_view_only=True,
            action_effect_horizons=(2, 4, 8),
            include_proprio=True,
        )
        main = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
        wrist = np.flip(main, axis=1).copy()
        sample = {
            "instruction": "move",
            "images": [main, wrist],
            "image": main,
            "image_wrist": wrist,
            "future_horizon_images": [main.copy(), main.copy(), main.copy()],
            "future_images": [main.copy(), wrist.copy()],
            "future_image": main.copy(),
            "future_image_wrist": wrist.copy(),
            "future_actions": torch.zeros(8, 7),
            "proprioception": torch.zeros(8, 7),
            "history_actions": torch.zeros(8, 7),
        }

        inputs, *_ = collator([sample])

        self.assertEqual(len(processor.image_calls), 2)
        current_images, future_images = processor.image_calls
        self.assertEqual(len(current_images), 2)
        self.assertEqual(len(future_images), 3)
        for future_image in future_images:
            self.assertTrue(np.array_equal(current_images[0], future_image))
        self.assertEqual(inputs["future_image_grid_thw"].shape[0], 3)


class MotionWeightedDenseHuberLossTest(unittest.TestCase):
    def test_static_target_falls_back_to_uniform_huber(self):
        current = torch.zeros(1, 4, 3)
        target = current[:, None].clone()
        prediction = torch.ones_like(target, requires_grad=True)

        loss, metrics = motion_weighted_dense_huber_loss(
            prediction,
            target,
            current,
            motion_weight_floor=0.25,
        )

        self.assertAlmostEqual(loss.item(), 0.5, places=6)
        self.assertAlmostEqual(metrics["max_patch_weight"].item(), 1.0, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_same_error_is_penalized_more_on_moving_patch(self):
        current = torch.zeros(1, 4, 2)
        target = current[:, None].clone()
        target[:, :, 0] = 2.0

        moving_error = target.clone()
        moving_error[:, :, 0, 0] += 1.0
        static_error = target.clone()
        static_error[:, :, 1, 0] += 1.0

        moving_loss, _ = motion_weighted_dense_huber_loss(
            moving_error,
            target,
            current,
            motion_weight_floor=0.25,
        )
        static_loss, _ = motion_weighted_dense_huber_loss(
            static_error,
            target,
            current,
            motion_weight_floor=0.25,
        )

        self.assertGreater(moving_loss.item(), static_loss.item())


class MultiHorizonVisualPredictorTest(unittest.TestCase):
    def test_causal_action_tokens_do_not_leak_future_actions(self):
        torch.manual_seed(7)
        encoder = CausalActionTokenEncoder(
            action_dim=3,
            horizon=8,
            token_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        ).eval()
        actions = torch.randn(2, 8, 3)
        changed = actions.clone()
        changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 10.0

        original_tokens = encoder(actions)
        changed_tokens = encoder(changed)

        self.assertTrue(
            torch.allclose(
                original_tokens[:, :2],
                changed_tokens[:, :2],
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertFalse(
            torch.allclose(original_tokens[:, -1], changed_tokens[:, -1])
        )

    def test_predicts_one_spatial_grid_per_horizon(self):
        predictor = MultiHorizonVisualPredictor(
            action_dim=3,
            action_horizon=8,
            action_latent_dim=32,
            visual_dim=24,
            visual_tokens=16,
            horizons=(2, 4, 8),
            token_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        actions = torch.randn(2, 8, 3)
        prefix_latents = torch.randn(2, 3, 32)
        current_visual = torch.randn(2, 16, 24)
        state = torch.randn(2, 8, 3)

        prediction = predictor(
            actions,
            prefix_latents,
            current_visual,
            state,
        )

        self.assertEqual(prediction.shape, (2, 3, 16, 24))
        self.assertTrue(torch.equal(prediction, current_visual[:, None].expand_as(prediction)))

    def test_parallel_prefix_latents_match_independent_encoding(self):
        generator = VitaLatentActionGenerator(
            action_dim=3,
            horizon=8,
            vlm_hidden_dim=16,
            num_queries=4,
            latent_dim=32,
            hidden_dim=32,
            action_ae_layers=1,
            flow_layers=1,
            proprio_dim=24,
            action_effect_enabled=True,
            action_effect_condition_action_encoder=False,
            action_effect_visual_queries=16,
            action_effect_visual_tokens_prepooled=True,
            action_effect_num_heads=4,
            action_effect_visual_dim=24,
            action_effect_horizons=(2, 4, 8),
            action_effect_action_token_dim=16,
            action_effect_multihorizon_layers=1,
        )
        actions = torch.randn(2, 8, 3)
        full_latent = generator.encode_action(actions)
        parallel = generator.encode_action_prefix_latents(actions, full_latent)

        expected = []
        for horizon in (2, 4, 8):
            if horizon == 8:
                expected.append(full_latent)
            else:
                prefix = actions.clone()
                prefix[:, horizon:] = 0
                expected.append(generator.encode_action(prefix))
        expected = torch.stack(expected, dim=1)

        self.assertTrue(torch.allclose(parallel, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
