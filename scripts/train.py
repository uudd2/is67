import os
import yaml
import argparse
import wandb
import random
import re
import time
import math
from PIL import Image

import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoImageProcessor, get_scheduler
from tqdm import tqdm
from contextlib import nullcontext
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TVF

from src.models.VLANeXt import VLANeXt
from src.models.rt2_like_baseline import RT2LikeBaseline
from src.models.smolvla_vita import (
    SmolVLAMQ54VitaPolicy,
    SmolVLAVitaLatentPolicy,
)
from src.models.edar_lite import DeterministicDINOImageProcessor
from src.datasets.libero_act import LiberoAct
from src.datasets.droid_act import DroidAct
from src.datasets.robotwin_act import RoboTwinAct
from src.datasets.bridge_act import BridgeAct
from src.datasets.vlabench_act import VLABenchAct
from src.datasets.vlabench_tfds_act import VLABenchTFDSAct
from src.datasets.robomimic_tfds_act import RoboMimicTFDSAct
from src.datasets.calvin_act import CalvinAct
from src.datasets.av_aloha_multitask_act import AVAlohaMultitaskAct


# -----------------------------------------------------------------------------
# -------------------------------- Prequisite ---------------------------------
# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_training_scheduler(config, optimizer):
    train_config = config["train"]
    scheduler_type = train_config.get("lr_scheduler_type", "cosine")
    if scheduler_type != "cosine_decay_with_warmup":
        return get_scheduler(
            scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=train_config["warmup_steps"],
            num_training_steps=config["data"]["max_steps"],
        )

    warmup_steps = int(train_config["warmup_steps"])
    decay_steps = int(train_config["decay_steps"])
    training_steps = int(config["data"]["max_steps"])
    peak_lr = float(train_config["learning_rate"])
    decay_lr = float(train_config["decay_learning_rate"])
    if decay_steps <= 0 or peak_lr <= 0 or decay_lr < 0:
        raise ValueError("Invalid cosine_decay_with_warmup scheduler settings.")

    if training_steps < decay_steps:
        scale = training_steps / decay_steps
        warmup_steps = int(warmup_steps * scale)
        decay_steps = training_steps

    min_lr_ratio = decay_lr / peak_lr

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            start = 1.0 / (warmup_steps + 1)
            if current_step <= 0:
                return start
            fraction = 1.0 - current_step / warmup_steps
            return (start - 1.0) * fraction + 1.0

        step = min(current_step, decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * step / decay_steps))
        return (1.0 - min_lr_ratio) * cosine + min_lr_ratio

    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)


def reshape_legacy_layer_local_query_state(state_dict, model):
    """Compat for old hierarchical-MQ checkpoints saved as [layers, queries, dim]."""
    model_state = model.state_dict()
    keys = (
        "vita_action_generator.observation_encoder.layer_local_query",
        "module.vita_action_generator.observation_encoder.layer_local_query",
    )
    for key in keys:
        if key not in state_dict or key not in model_state:
            continue
        ckpt_tensor = state_dict[key]
        model_tensor = model_state[key]
        if tuple(ckpt_tensor.shape) == tuple(model_tensor.shape):
            return False
        if ckpt_tensor.numel() != model_tensor.numel():
            return False
        state_dict[key] = ckpt_tensor.reshape(tuple(model_tensor.shape))
        return True
    return False


def _is_action_effect_visual_teacher_key(key):
    return key.removeprefix("module.").startswith("action_effect_visual_teacher.")


def checkpoint_model_state_dict(model):
    """Keep the frozen external visual teacher out of policy checkpoints."""
    return {
        key: value
        for key, value in model.state_dict().items()
        if not _is_action_effect_visual_teacher_key(key)
    }


def load_model_state_for_resume(model, state_dict):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key for key in missing if not _is_action_effect_visual_teacher_key(key)
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Resume checkpoint does not match the model: "
            f"missing={invalid_missing}, unexpected={unexpected}"
        )
    return missing


def configure_partial_vlm_trainability(model, model_config):
    """Apply optional Qwen component-level freezing before DDP/optimizer setup."""
    freeze_vision = bool(model_config.get("freeze_vlm_vision_encoder", False))
    freeze_text_embeddings = bool(
        model_config.get("freeze_vlm_text_embeddings", False)
    )
    trainable_hidden_layers = model_config.get("vlm_trainable_hidden_layers", None)
    if not (freeze_vision or freeze_text_embeddings or trainable_hidden_layers is not None):
        return

    lmm = getattr(model, "lmm", None)
    backbone = getattr(lmm, "model", None)
    language_model = getattr(backbone, "language_model", None)
    if lmm is None or backbone is None or language_model is None:
        raise ValueError(
            "Partial VLM freezing requires a multimodal backbone with "
            "model.visual and model.language_model modules."
        )

    if freeze_vision:
        visual = getattr(backbone, "visual", None)
        if visual is None:
            raise ValueError("freeze_vlm_vision_encoder=true but no visual module exists.")
        visual.requires_grad_(False)

    if freeze_text_embeddings:
        embeddings = getattr(language_model, "embed_tokens", None)
        if embeddings is None:
            raise ValueError(
                "freeze_vlm_text_embeddings=true but no embed_tokens module exists."
            )
        embeddings.requires_grad_(False)
        # Qwen ties lm_head.weight to embed_tokens.weight; this is harmless if untied.
        lm_head = getattr(lmm, "lm_head", None)
        if lm_head is not None:
            lm_head.requires_grad_(False)

    if trainable_hidden_layers is not None:
        layers = getattr(language_model, "layers", None)
        if layers is None:
            raise ValueError("vlm_trainable_hidden_layers requires language_model.layers.")
        trainable_hidden_layers = int(trainable_hidden_layers)
        if not 0 < trainable_hidden_layers <= len(layers):
            raise ValueError(
                "vlm_trainable_hidden_layers must be in "
                f"[1, {len(layers)}], got {trainable_hidden_layers}."
            )
        for index, layer in enumerate(layers):
            layer.requires_grad_(index < trainable_hidden_layers)

    trainable_vlm_params = sum(
        parameter.numel() for parameter in lmm.parameters() if parameter.requires_grad
    )
    total_vlm_params = sum(parameter.numel() for parameter in lmm.parameters())
    print(
        "Partial VLM trainability: "
        f"freeze_vision={freeze_vision}, "
        f"freeze_text_embeddings={freeze_text_embeddings}, "
        f"trainable_hidden_layers={trainable_hidden_layers}, "
        f"trainable_params={trainable_vlm_params}/{total_vlm_params}"
    )


class DataCollatorForVLANeXt:
    def __init__(
        self,
        processor,
        use_proprio_input_vlm=True,
        use_action_input_policy=True,
        input_modality="video",
        view_mode="single",
        fps=20.0,
        augmentation=None,
        load_future_image=False,
        include_progress=False,
        include_action_progress=False,
        include_goal_distance=False,
        include_latent_bridge=False,
        include_future_vlm=False,
        include_action_effect_teacher_images=False,
        action_effect_teacher_processor=None,
        action_effect_teacher_type="dinov2",
        action_effect_teacher_current_only=False,
        future_vlm_main_view_only=False,
        action_effect_horizons=None,
        include_proprio=False,
        bridge_image_size=224,
    ):
        self.processor = processor
        self.use_proprio_input_vlm = use_proprio_input_vlm
        self.use_action_input_policy = use_action_input_policy
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.fps = float(fps)
        self.load_future_image = load_future_image
        self.include_progress = include_progress
        self.include_action_progress = include_action_progress
        self.include_goal_distance = include_goal_distance
        self.include_latent_bridge = include_latent_bridge
        self.include_future_vlm = include_future_vlm
        self.include_action_effect_teacher_images = bool(
            include_action_effect_teacher_images
        )
        self.action_effect_teacher_processor = action_effect_teacher_processor
        self.action_effect_teacher_type = str(action_effect_teacher_type).lower()
        self.action_effect_teacher_current_only = bool(
            action_effect_teacher_current_only
        )
        if (
            self.include_action_effect_teacher_images
            and self.action_effect_teacher_processor is None
        ):
            raise ValueError("Action-effect teacher images require an image processor.")
        self.future_vlm_main_view_only = bool(future_vlm_main_view_only)
        self.action_effect_horizons = tuple(
            int(horizon) for horizon in (action_effect_horizons or ())
        )
        if self.action_effect_horizons:
            if tuple(sorted(set(self.action_effect_horizons))) != self.action_effect_horizons:
                raise ValueError(
                    "action_effect_horizons must be positive, unique, and sorted."
                )
            if self.action_effect_horizons[0] <= 0:
                raise ValueError("action_effect_horizons must be positive.")
        self.include_proprio = bool(include_proprio or use_proprio_input_vlm)
        self.bridge_image_size = int(bridge_image_size)
        self.bridge_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.bridge_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.aug = augmentation or {}
        self.aug_enabled = bool(self.aug.get("enabled", False))

        rrc = self.aug.get("random_resized_crop", {}) or {}
        self.rrc_scale = tuple(rrc.get("scale", (0.9, 0.9)))
        self.rrc_ratio = tuple(rrc.get("ratio", (1.0, 1.0)))

        self.rb = self.aug.get("random_brightness", None)
        self.rc = self.aug.get("random_contrast", None)
        self.rs = self.aug.get("random_saturation", None)
        self.rh = self.aug.get("random_hue", None)
        self.augment_order = list(self.aug.get("augment_order", []))

    def _to_pil(self, img_np: np.ndarray) -> Image.Image:
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def _uniform(self, a: float, b: float) -> float:
        return float(np.random.uniform(a, b))

    def _sample_brightness_factor(self) -> float:
        if not self.rb:
            return 1.0
        if len(self.rb) == 1:
            x = float(self.rb[0])
            return self._uniform(1.0 - x, 1.0 + x)
        return self._uniform(float(self.rb[0]), float(self.rb[1]))

    def _sample_contrast_factor(self) -> float:
        if not self.rc:
            return 1.0
        if len(self.rc) == 1:
            x = float(self.rc[0])
            return self._uniform(1.0 - x, 1.0 + x)
        return self._uniform(float(self.rc[0]), float(self.rc[1]))

    def _sample_saturation_factor(self) -> float:
        if not self.rs:
            return 1.0
        if len(self.rs) == 1:
            x = float(self.rs[0])
            return self._uniform(1.0 - x, 1.0 + x)
        return self._uniform(float(self.rs[0]), float(self.rs[1]))

    def _sample_hue_delta(self) -> float:
        if not self.rh:
            return 0.0
        if len(self.rh) == 1:
            x = float(self.rh[0])
            return self._uniform(-x, x)
        return self._uniform(float(self.rh[0]), float(self.rh[1]))

    def _augment_frames_uint8(self, frames: np.ndarray) -> np.ndarray:
        """Apply augmentation to frames (T,H,W,C) or (H,W,C)."""
        if (not self.aug_enabled) or (not self.augment_order):
            return frames

        is_video = (frames.ndim == 4)
        pil_frames = [self._to_pil(f) for f in (frames if is_video else [frames])]
        out_h, out_w = pil_frames[0].height, pil_frames[0].width

        crop_params = None
        if "random_resized_crop" in self.augment_order and self.aug.get("random_resized_crop", None) is not None:
            i, j, h, w = RandomResizedCrop.get_params(pil_frames[0], scale=self.rrc_scale, ratio=self.rrc_ratio)
            crop_params = (i, j, h, w)

        b_fac = self._sample_brightness_factor() if "random_brightness" in self.augment_order else 1.0
        c_fac = self._sample_contrast_factor() if "random_contrast" in self.augment_order else 1.0
        s_fac = self._sample_saturation_factor() if "random_saturation" in self.augment_order else 1.0
        h_del = self._sample_hue_delta() if "random_hue" in self.augment_order else 0.0
        h_del = float(np.clip(h_del, -0.5, 0.5))

        out = []
        for pil in pil_frames:
            for op in self.augment_order:
                if op == "random_resized_crop" and crop_params is not None:
                    i, j, h, w = crop_params
                    pil = TVF.resized_crop(pil, i, j, h, w, size=(out_h, out_w))
                elif op == "random_brightness":
                    pil = TVF.adjust_brightness(pil, b_fac)
                elif op == "random_contrast":
                    pil = TVF.adjust_contrast(pil, c_fac)
                elif op == "random_saturation":
                    pil = TVF.adjust_saturation(pil, s_fac)
                elif op == "random_hue":
                    pil = TVF.adjust_hue(pil, h_del)
                else:
                    pass
            out.append(np.asarray(pil, dtype=np.uint8))

        out = np.stack(out, axis=0)
        return out if is_video else out[0]

    def _bridge_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        pil = self._to_pil(img_np)
        pil = TVF.resize(pil, [self.bridge_image_size, self.bridge_image_size])
        x = torch.from_numpy(np.asarray(pil, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        return (x - self.bridge_mean) / self.bridge_std

    def _augment_image_pair_uint8(self, current: np.ndarray, future: np.ndarray):
        paired = np.stack([current, future], axis=0)
        augmented = self._augment_frames_uint8(paired)
        return augmented[0], augmented[1]

    def _augment_image_sequence_uint8(
        self,
        current: np.ndarray,
        futures,
    ):
        frames = np.stack([current, *list(futures)], axis=0)
        augmented = self._augment_frames_uint8(frames)
        return augmented[0], list(augmented[1:])

    def __call__(self, batch):
        texts = []
        fps = self.fps

        images = []
        smol_image_batches = []
        videos = []
        
        gt_actions_list = []
        proprio_list = []
        hist_actions_list = []
        future_images_list = []
        progress_list = []
        action_progress_list = []
        goal_distance_list = []
        bridge_images_list = []
        future_vlm_texts = []
        future_vlm_images = []
        action_effect_teacher_current_images = []
        action_effect_teacher_future_images = []

        is_paligemma = "PaliGemma" in self.processor.__class__.__name__
        is_qwen = "Qwen" in self.processor.__class__.__name__
        is_smol = "SmolVLM" in self.processor.__class__.__name__
        is_llama = "Llama" in self.processor.__class__.__name__
        is_sequence_bridge = self.include_latent_bridge and len(batch) > 0 and "sequence_images" in batch[0]
        if self.include_action_effect_teacher_images and (
            (not is_qwen and not is_smol) or self.input_modality != "image"
        ):
            raise ValueError(
                "Action-effect visual targets require Qwen or SmolVLM image inputs."
            )

        for sample in batch:
            instruction = sample["instruction"]
            prepared_future_images = None
            action_effect_teacher_current_image = None
            action_effect_teacher_future_image = None
            needs_future_pair = (
                self.include_future_vlm
                or (
                    self.include_action_effect_teacher_images
                    and not self.action_effect_teacher_current_only
                )
            )

            if is_paligemma:
                vlm_img0 = sample.get("anchor_image", sample["image"]) if self.include_latent_bridge else sample["image"]
                im0 = self._augment_frames_uint8(vlm_img0)
                num_imgs = 1
                if self.view_mode == "multi":
                    vlm_img1 = sample.get("anchor_image_wrist", sample["image_wrist"]) if self.include_latent_bridge else sample["image_wrist"]
                    im1 = self._augment_frames_uint8(vlm_img1)
                    images.extend([im0, im1])
                    num_imgs = 2
                    if self.include_latent_bridge:
                        cur0 = self._augment_frames_uint8(sample["image"])
                        cur1 = self._augment_frames_uint8(sample["image_wrist"])
                        bridge_images_list.append(torch.stack([self._bridge_tensor(cur0), self._bridge_tensor(cur1)], dim=0))
                else:
                    images.append(im0)
                    if self.include_latent_bridge:
                        cur0 = self._augment_frames_uint8(sample["image"])
                        bridge_images_list.append(self._bridge_tensor(cur0).unsqueeze(0))
                
                texts.append("<image>" * num_imgs + instruction)

            elif is_llama:
                vlm_img0 = sample.get("anchor_image", sample["image"]) if self.include_latent_bridge else sample["image"]
                im0 = self._augment_frames_uint8(vlm_img0)
                if self.view_mode == "multi":
                    vlm_img1 = sample.get("anchor_image_wrist", sample["image_wrist"]) if self.include_latent_bridge else sample["image_wrist"]
                    im1 = self._augment_frames_uint8(vlm_img1)
                    images.extend([im0, im1])
                    if self.include_latent_bridge:
                        cur0 = self._augment_frames_uint8(sample["image"])
                        cur1 = self._augment_frames_uint8(sample["image_wrist"])
                        bridge_images_list.append(torch.stack([self._bridge_tensor(cur0), self._bridge_tensor(cur1)], dim=0))
                else:
                    images.append(im0)
                    if self.include_latent_bridge:
                        cur0 = self._augment_frames_uint8(sample["image"])
                        bridge_images_list.append(self._bridge_tensor(cur0).unsqueeze(0))
                
                texts.append(instruction)

            elif is_qwen:
                content = []
                if self.input_modality == "video":
                    vlm_v0 = sample.get("anchor_video", sample["video"]) if self.include_latent_bridge else sample["video"]
                    v0 = self._augment_frames_uint8(vlm_v0)
                    if self.view_mode == "multi":
                        vlm_v1 = sample.get("anchor_video_wrist", sample["video_wrist"]) if self.include_latent_bridge else sample["video_wrist"]
                        v1 = self._augment_frames_uint8(vlm_v1)
                        content.extend([{"type": "video", "video": v0}, {"type": "video", "video": v1}])
                        videos.extend([v0, v1]) 
                        if self.include_latent_bridge:
                            cur0 = self._augment_frames_uint8(sample["video"])[-1]
                            cur1 = self._augment_frames_uint8(sample["video_wrist"])[-1]
                            bridge_images_list.append(torch.stack([self._bridge_tensor(cur0), self._bridge_tensor(cur1)], dim=0))
                    else:
                        content.append({"type": "video", "video": v0})
                        videos.append(v0)
                        if self.include_latent_bridge:
                            cur0 = self._augment_frames_uint8(sample["video"])[-1]
                            bridge_images_list.append(self._bridge_tensor(cur0).unsqueeze(0))

                elif self.input_modality == "image":
                    if self.view_mode == "multi" and "images" in sample:
                        vlm_images_np = sample.get("anchor_images", sample["images"]) if self.include_latent_bridge else sample["images"]
                        if self.action_effect_horizons:
                            future_horizon_images = list(
                                sample.get("future_horizon_images", [])
                            )
                            if len(future_horizon_images) != len(
                                self.action_effect_horizons
                            ):
                                raise ValueError(
                                    "The dataset must provide one main-view future "
                                    "image for every action-effect horizon."
                                )
                            current_main, future_horizon_images = (
                                self._augment_image_sequence_uint8(
                                    vlm_images_np[0],
                                    future_horizon_images,
                                )
                            )
                            sample_images = [current_main]
                            sample_images.extend(
                                self._augment_frames_uint8(img)
                                for img in vlm_images_np[1:]
                            )
                            prepared_future_images = future_horizon_images
                            action_effect_teacher_current_image = current_main
                            action_effect_teacher_future_image = (
                                future_horizon_images[-1]
                            )
                        elif needs_future_pair and "future_images" in sample:
                            future_images_np = sample["future_images"]
                            if not future_images_np:
                                raise ValueError("Action-effect training requires at least one future view.")
                            if not self.future_vlm_main_view_only and len(vlm_images_np) != len(future_images_np):
                                raise ValueError("Current/future view counts must match for action-effect training.")
                            if self.future_vlm_main_view_only:
                                current_main, future_main = self._augment_image_pair_uint8(
                                    vlm_images_np[0], future_images_np[0]
                                )
                                action_effect_teacher_current_image = current_main
                                action_effect_teacher_future_image = future_main
                                sample_images = [current_main]
                                sample_images.extend(
                                    self._augment_frames_uint8(img) for img in vlm_images_np[1:]
                                )
                                prepared_future_images = [future_main]
                            else:
                                paired_views = [
                                    self._augment_image_pair_uint8(current, future)
                                    for current, future in zip(vlm_images_np, future_images_np)
                                ]
                                sample_images = [pair[0] for pair in paired_views]
                                prepared_future_images = [pair[1] for pair in paired_views]
                                action_effect_teacher_current_image = paired_views[0][0]
                                action_effect_teacher_future_image = paired_views[0][1]
                        else:
                            sample_images = [self._augment_frames_uint8(img) for img in vlm_images_np]
                        content.extend([{"type": "image", "image": img} for img in sample_images])
                        images.extend(sample_images)
                        if self.include_latent_bridge:
                            if is_sequence_bridge:
                                sequence_bridge = []
                                for frame_images in sample["sequence_images"]:
                                    current_images = [self._augment_frames_uint8(img) for img in frame_images]
                                    sequence_bridge.append(torch.stack([self._bridge_tensor(img) for img in current_images], dim=0))
                                bridge_images_list.append(torch.stack(sequence_bridge, dim=0))
                            else:
                                current_images = [self._augment_frames_uint8(img) for img in sample["images"]]
                                bridge_images_list.append(torch.stack([self._bridge_tensor(img) for img in current_images], dim=0))
                    else:
                        vlm_img0 = sample.get("anchor_image", sample["image"]) if self.include_latent_bridge else sample["image"]
                        if self.action_effect_horizons:
                            future_horizon_images = list(
                                sample.get("future_horizon_images", [])
                            )
                            if len(future_horizon_images) != len(
                                self.action_effect_horizons
                            ):
                                raise ValueError(
                                    "The dataset must provide one main-view future "
                                    "image for every action-effect horizon."
                                )
                            im0, future_horizon_images = (
                                self._augment_image_sequence_uint8(
                                    vlm_img0,
                                    future_horizon_images,
                                )
                            )
                            action_effect_teacher_current_image = im0
                            action_effect_teacher_future_image = (
                                future_horizon_images[-1]
                            )
                            prepared_future_images = future_horizon_images
                        elif needs_future_pair:
                            im0, future_im0 = self._augment_image_pair_uint8(
                                vlm_img0,
                                sample["future_image"],
                            )
                            action_effect_teacher_current_image = im0
                            action_effect_teacher_future_image = future_im0
                            prepared_future_images = [future_im0]
                        else:
                            im0 = self._augment_frames_uint8(vlm_img0)
                            if (
                                self.include_action_effect_teacher_images
                                and self.action_effect_teacher_current_only
                            ):
                                action_effect_teacher_current_image = im0
                        if self.view_mode == "multi":
                            vlm_img1 = sample.get("anchor_image_wrist", sample["image_wrist"]) if self.include_latent_bridge else sample["image_wrist"]
                            if needs_future_pair and not self.future_vlm_main_view_only:
                                im1, future_im1 = self._augment_image_pair_uint8(
                                    vlm_img1,
                                    sample.get("future_image_wrist", sample["future_image"]),
                                )
                                prepared_future_images.append(future_im1)
                            else:
                                im1 = self._augment_frames_uint8(vlm_img1)
                            content.extend([{"type": "image", "image": im0}, {"type": "image", "image": im1}])
                            images.extend([im0, im1])
                            if self.include_latent_bridge:
                                cur0 = self._augment_frames_uint8(sample["image"])
                                cur1 = self._augment_frames_uint8(sample["image_wrist"])
                                bridge_images_list.append(torch.stack([self._bridge_tensor(cur0), self._bridge_tensor(cur1)], dim=0))
                        else:
                            content.append({"type": "image", "image": im0})
                            images.append(im0)
                            if self.include_latent_bridge:
                                if is_sequence_bridge:
                                    sequence_bridge = []
                                    for frame_images in sample["sequence_images"]:
                                        cur0 = self._augment_frames_uint8(frame_images[0])
                                        sequence_bridge.append(self._bridge_tensor(cur0).unsqueeze(0))
                                    bridge_images_list.append(torch.stack(sequence_bridge, dim=0))
                                else:
                                    cur0 = self._augment_frames_uint8(sample["image"])
                                    bridge_images_list.append(self._bridge_tensor(cur0).unsqueeze(0))
                    if self.include_action_effect_teacher_images:
                        if (
                            self.action_effect_teacher_current_only
                            and action_effect_teacher_current_image is None
                        ):
                            action_effect_teacher_current_image = sample_images[0]
                        if (
                            action_effect_teacher_current_image is None
                            or (
                                not self.action_effect_teacher_current_only
                                and action_effect_teacher_future_image is None
                            )
                        ):
                            raise ValueError(
                                "Action-effect training requires a current/future main-view pair."
                            )
                        action_effect_teacher_current_images.append(
                            action_effect_teacher_current_image
                        )
                        if not self.action_effect_teacher_current_only:
                            action_effect_teacher_future_images.append(
                                action_effect_teacher_future_image
                            )
                else:
                    raise ValueError(f"Unknown input_modality: {self.input_modality}")

                content.append({"type": "text", "text": instruction})

                messages = [{"role": "user", "content": content}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(text)

            elif is_smol:
                if self.input_modality != "image":
                    raise ValueError("The SmolVLA latent policy currently supports image inputs only.")

                if self.view_mode == "multi" and "images" in sample:
                    raw_views = list(sample["images"])
                    future_views = list(sample.get("future_images", []))
                else:
                    raw_views = [sample["image"]]
                    if self.view_mode == "multi":
                        raw_views.append(sample["image_wrist"])
                    future_views = []
                    if needs_future_pair:
                        future_views.append(sample["future_image"])
                        if self.view_mode == "multi":
                            future_views.append(
                                sample.get("future_image_wrist", sample["future_image"])
                            )

                if needs_future_pair:
                    if not future_views:
                        raise ValueError("Action-effect training requires a future main view.")
                    current_main, future_main = self._augment_image_pair_uint8(
                        raw_views[0], future_views[0]
                    )
                    sample_images = [current_main]
                    sample_images.extend(
                        self._augment_frames_uint8(image) for image in raw_views[1:]
                    )
                    action_effect_teacher_current_image = current_main
                    action_effect_teacher_future_image = future_main
                else:
                    sample_images = [
                        self._augment_frames_uint8(image) for image in raw_views
                    ]
                    if (
                        self.include_action_effect_teacher_images
                        and self.action_effect_teacher_current_only
                    ):
                        action_effect_teacher_current_image = sample_images[0]

                smol_image_batches.append(sample_images)
                texts.append(instruction.rstrip() + "\n")
                if self.include_action_effect_teacher_images:
                    if action_effect_teacher_current_image is None:
                        raise ValueError(
                            "Action-effect training requires a current main-view image."
                        )
                    action_effect_teacher_current_images.append(
                        action_effect_teacher_current_image
                    )
                    if not self.action_effect_teacher_current_only:
                        if action_effect_teacher_future_image is None:
                            raise ValueError(
                                "Action-effect training requires a future main-view image."
                            )
                        action_effect_teacher_future_images.append(
                            action_effect_teacher_future_image
                        )

            gt_actions_list.append(sample["future_actions"])
            proprio_list.append(sample["proprioception"])
            hist_actions_list.append(sample["history_actions"])
            if self.include_progress:
                progress_list.append(sample.get("progress", torch.tensor(0.0, dtype=torch.float32)))
            if self.include_action_progress:
                action_progress_list.append(
                    sample.get("action_progress", torch.tensor(0.0, dtype=torch.float32))
                )
            if self.include_goal_distance:
                goal_distance_list.append(sample.get("goal_distance", torch.tensor(0.0, dtype=torch.float32)))
            
            if self.load_future_image and "future_image" in sample:
                f_img = sample["future_image"]
                f_img = torch.from_numpy(f_img).permute(2, 0, 1).float() / 127.5 - 1.0
                future_images_list.append(f_img)

            if self.include_future_vlm:
                if self.input_modality != "image":
                    raise ValueError("Future VLM targets currently support image modality only.")
                if "future_images" in sample:
                    sample_future_images = sample["future_images"]
                elif self.view_mode == "multi" and "future_image_wrist" in sample:
                    sample_future_images = [sample["future_image"], sample["future_image_wrist"]]
                else:
                    sample_future_images = [sample["future_image"]]
                sample_future_images = (
                    prepared_future_images
                    if prepared_future_images is not None
                    else [self._augment_frames_uint8(img) for img in sample_future_images]
                )
                if (
                    self.future_vlm_main_view_only
                    and not self.action_effect_horizons
                ):
                    sample_future_images = sample_future_images[:1]
                future_vlm_images.extend(sample_future_images)

                if is_paligemma:
                    future_vlm_texts.append("<image>" * len(sample_future_images) + instruction)
                elif is_llama:
                    future_vlm_texts.append(instruction)
                elif is_qwen:
                    content = [{"type": "image", "image": img} for img in sample_future_images]
                    content.append({"type": "text", "text": instruction})
                    messages = [{"role": "user", "content": content}]
                    future_vlm_texts.append(
                        self.processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )

        if is_paligemma:
            inputs = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        elif is_llama:
            inputs = self.processor.tokenizer(
                texts,
                padding=True,
                return_tensors="pt"
            )
            image_inputs = self.processor.image_processor(
                images,
                return_tensors="pt"
            )
            inputs["pixel_values"] = image_inputs["pixel_values"]
        elif is_qwen:
            if self.input_modality == "video":
                video_metadata = [
                    {"total_num_frames": v.shape[0], "fps": fps, "frames_indices": list(range(v.shape[0]))}
                    for v in videos
                ]
                inputs = self.processor(
                    text=texts,
                    videos=videos,
                    videos_kwargs={"fps": fps, "return_metadata": True, "video_metadata": video_metadata}, 
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
        elif is_smol:
            inputs = self.processor.tokenizer(
                texts,
                padding=True,
                max_length=48,
                truncation=True,
                return_tensors="pt",
            )
            image_inputs = self.processor.image_processor(
                smol_image_batches,
                return_tensors="pt",
            )
            inputs["pixel_values"] = image_inputs["pixel_values"]
            if "pixel_attention_mask" in image_inputs:
                inputs["pixel_attention_mask"] = image_inputs[
                    "pixel_attention_mask"
                ]

        if self.include_goal_distance:
            if self.input_modality != "image":
                raise ValueError("Goal-distance regression currently supports image modality only.")
            goal_texts = []
            goal_images = []
            for sample in batch:
                instruction = sample["instruction"]
                if self.view_mode == "multi":
                    if "goal_images" in sample:
                        sample_goal_images = sample["goal_images"]
                    else:
                        sample_goal_images = [
                            sample["goal_image"],
                            sample.get("goal_image_wrist", sample["goal_image"]),
                        ]
                else:
                    sample_goal_images = [sample["goal_image"]]
                sample_goal_images = [self._augment_frames_uint8(img) for img in sample_goal_images]
                goal_images.extend(sample_goal_images)

                if is_paligemma:
                    goal_texts.append("<image>" * len(sample_goal_images) + instruction)
                elif is_llama:
                    goal_texts.append(instruction)
                elif is_qwen:
                    content = [{"type": "image", "image": img} for img in sample_goal_images]
                    content.append({"type": "text", "text": instruction})
                    messages = [{"role": "user", "content": content}]
                    goal_texts.append(
                        self.processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )

            if is_paligemma:
                goal_inputs = self.processor(
                    text=goal_texts,
                    images=goal_images,
                    padding=True,
                    return_tensors="pt",
                )
            elif is_llama:
                goal_inputs = self.processor.tokenizer(
                    goal_texts,
                    padding=True,
                    return_tensors="pt",
                )
                goal_image_inputs = self.processor.image_processor(
                    goal_images,
                    return_tensors="pt",
                )
                goal_inputs["pixel_values"] = goal_image_inputs["pixel_values"]
            elif is_qwen:
                goal_inputs = self.processor(
                    text=goal_texts,
                    images=goal_images,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                raise ValueError("Unsupported processor for goal-distance regression.")
            for key, value in goal_inputs.items():
                inputs[f"goal_{key}"] = value

        if self.include_future_vlm:
            if is_paligemma:
                future_inputs = self.processor(
                    text=future_vlm_texts,
                    images=future_vlm_images,
                    padding=True,
                    return_tensors="pt",
                )
            elif is_llama:
                future_inputs = self.processor.tokenizer(
                    future_vlm_texts,
                    padding=True,
                    return_tensors="pt",
                )
                future_image_inputs = self.processor.image_processor(
                    future_vlm_images,
                    return_tensors="pt",
                )
                future_inputs["pixel_values"] = future_image_inputs["pixel_values"]
            elif is_qwen:
                future_inputs = self.processor(
                    text=future_vlm_texts,
                    images=future_vlm_images,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                raise ValueError("Unsupported processor for future VLM targets.")
            for key, value in future_inputs.items():
                inputs[f"future_{key}"] = value

        if self.include_action_effect_teacher_images:
            teacher_batch_size = len(action_effect_teacher_current_images)
            teacher_images = (
                action_effect_teacher_current_images
                + (
                    []
                    if self.action_effect_teacher_current_only
                    else action_effect_teacher_future_images
                )
            )
            if self.action_effect_teacher_type == "smolvlm":
                teacher_images = [[image] for image in teacher_images]
            teacher_inputs = self.action_effect_teacher_processor(
                images=teacher_images,
                return_tensors="pt",
            )
            teacher_pixels = teacher_inputs["pixel_values"]
            if self.action_effect_teacher_type == "smolvlm":
                if teacher_pixels.ndim == 5 and teacher_pixels.shape[1] == 1:
                    teacher_pixels = teacher_pixels[:, 0]
                if teacher_pixels.ndim != 4:
                    raise ValueError(
                        "SmolVLM action-effect preprocessing must produce "
                        "[2B,C,H,W] pixels."
                    )
            inputs["action_effect_teacher_current_pixel_values"] = teacher_pixels[
                :teacher_batch_size
            ]
            if not self.action_effect_teacher_current_only:
                inputs["action_effect_teacher_future_pixel_values"] = teacher_pixels[
                    teacher_batch_size:
                ]

        if self.include_latent_bridge:
            inputs["bridge_pixel_values"] = torch.stack(bridge_images_list, dim=0)

        gt_actions = torch.stack(gt_actions_list)
        proprio = torch.stack(proprio_list) if self.include_proprio else None
        hist_actions = torch.stack(hist_actions_list) if self.use_action_input_policy else None
        future_images = torch.stack(future_images_list) if self.load_future_image else None
        progress = torch.stack(progress_list) if self.include_progress else None
        action_progress = torch.stack(action_progress_list) if self.include_action_progress else None
        goal_distance = torch.stack(goal_distance_list) if self.include_goal_distance else None
        
        if self.include_progress or self.include_action_progress or self.include_goal_distance:
            return inputs, gt_actions, proprio, hist_actions, future_images, progress, action_progress, goal_distance
        return inputs, gt_actions, proprio, hist_actions, future_images

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(config):
    # -----------------------------------------------------------------------------
    # ----------------------------------- Setup -----------------------------------
    # -----------------------------------------------------------------------------
    is_distributed = config['train'].get('distributed', False)
    gradient_accumulation_steps = config['train'].get('gradient_accumulation_steps', 1)
    model_type = config['model'].get('model_type', 'vlanext')
    use_smolvla_vita = model_type == 'smolvla_vita_latent'
    use_smolvla_mq54_vita = model_type == 'smolvla_mq54_vita'
    use_smolvla_family = use_smolvla_vita or use_smolvla_mq54_vita
    use_proprio_input_vlm = config['model'].get('use_proprio_input_vlm', True)
    use_action_input_policy = config['model'].get('use_action_input_policy', True)
    future_image_loss_weight = float(config['model'].get('future_image_loss_weight', 0.0))
    use_progress_head = bool(config['model'].get('use_progress_head', False) or config['model'].get('use_progress_film', False))
    use_action_progress_alignment = (
        config['model'].get('action_generation_mode', 'direct_flow') == 'vita_latent_flow'
        and float(config['model'].get('vita_action_progress_loss_weight', 0.0)) > 0
    )
    use_goal_distance_regression = (
        config['model'].get('action_generation_mode', 'direct_flow') == 'vita_latent_flow'
        and float(config['model'].get('vita_goal_distance_loss_weight', 0.0)) > 0
    )
    use_latent_bridge = bool(config['model'].get('use_latent_bridge', False))
    use_vita_proprio = use_smolvla_family or (
        config['model'].get('action_generation_mode', 'direct_flow') == 'vita_latent_flow'
        and config['model'].get('vita_use_proprio', True)
    )
    use_vlanext_dual_visual_flow = (
        config['model'].get('action_generation_mode', 'direct_flow') == 'vlanext_latent_flow'
        and bool(config['model'].get('vlanext_dual_flow', False))
        and float(config['model'].get('vlanext_visual_flow_loss_weight', 0.0)) > 0
    )
    use_vita_action_effect = (
        config['model'].get('action_generation_mode', 'direct_flow') == 'vita_latent_flow'
        and bool(config['model'].get('vita_action_effect_enabled', False))
    )
    action_effect_horizons = tuple(
        int(horizon)
        for horizon in config['model'].get('vita_action_effect_horizons', [])
    )
    future_image_offsets = tuple(
        int(offset)
        for offset in config['data'].get(
            'future_image_offsets',
            action_effect_horizons,
        )
    )
    if action_effect_horizons:
        if not use_vita_action_effect:
            raise ValueError(
                "vita_action_effect_horizons requires VITA action-effect training."
            )
        if future_image_offsets != action_effect_horizons:
            raise ValueError(
                "data.future_image_offsets must exactly match "
                "model.vita_action_effect_horizons."
            )
    use_smolvla_action_effect = (
        use_smolvla_family
        and bool(config['model'].get('vita_action_effect_enabled', True))
    )
    action_effect_visual_teacher = str(
        config['model'].get('vita_action_effect_visual_teacher', 'qwen')
    ).lower()
    action_representation_type = str(
        config['model'].get('vita_action_representation_type', 'legacy')
    ).lower()
    use_edar_lite = action_representation_type == 'single_view_edar_lite'
    if action_effect_visual_teacher in {'smolvlm_spatial', 'policy_smolvlm'}:
        action_effect_visual_teacher = 'smolvlm'
    use_dinov2_action_effect_teacher = (
        (use_vita_action_effect or use_smolvla_action_effect)
        and action_effect_visual_teacher == 'dinov2'
    )
    use_dinov3_action_effect_teacher = (
        use_vita_action_effect
        and action_effect_visual_teacher == 'dinov3'
    )
    use_smolvlm_action_effect_teacher = (
        use_smolvla_action_effect
        and action_effect_visual_teacher in {'smolvlm', 'smolvlm_spatial'}
    )
    use_preprocessed_action_effect_teacher = (
        use_dinov2_action_effect_teacher
        or use_dinov3_action_effect_teacher
        or use_smolvlm_action_effect_teacher
    )
    enable_future_image_loss = (future_image_loss_weight > 0)
    load_future_image = (
        enable_future_image_loss
        or use_vlanext_dual_visual_flow
        or (use_vita_action_effect and not use_edar_lite)
        or (use_smolvla_action_effect and not use_edar_lite)
    )
    future_image_mode = config['model'].get('future_image_mode', 'horizon')
    input_modality = config["data"].get("input_modality", "video")
    view_mode = config["data"].get("view_mode", "single")
    augmentation = config["data"].get("augmentation", {})
    dataset_name = config['data'].get('dataset_name', 'libero')
    if dataset_name == "droid":
        fps = 15.0
    elif dataset_name == "robotwin":
        fps = float(config["data"].get("fps", 20.0))
    elif dataset_name in {"vlabench", "vlabench_tfds"}:
        fps = float(config["data"].get("fps", 10.0))
    elif dataset_name == "calvin":
        fps = float(config["data"].get("fps", 10.0))
    elif dataset_name == "robomimic":
        fps = float(config["data"].get("fps", 20.0))
    elif dataset_name == "av_aloha":
        fps = float(config["data"].get("fps", 25.0))
    elif dataset_name in {"libero", "bridge"}:
        fps = 20.0
    else:
        fps = 20.0
    full_sequence = bool(config['data'].get('full_sequence', False))
    seed = config['train'].get('seed', 42)

    set_seed(seed)

    if is_distributed:
        dist.init_process_group(backend=config['train']['dist_backend'])
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(f"[Rank {global_rank}] Initialized process group")
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device(config['train'].get('device', 'cuda'))
        
    os.makedirs(config['project']['output_dir'], exist_ok=True)
    save_dir = ""
    wandb_project = config['project'].get('wandb_project', 'VLANeXt')
    wandb_name = config['project']['name']
    if global_rank == 0:
        full_name = config['project']['name']
        parts = full_name.split('_')
        if len(parts) > 3:
            parent_dir = '_'.join(parts[:3])
            sub_dir = '_'.join(parts[3:])
            save_dir = os.path.join(config['project']['output_dir'], parent_dir, sub_dir)
            
            wandb_project = parent_dir
            wandb_name = sub_dir
        else:
            save_dir = os.path.join(config['project']['output_dir'], full_name)
        os.makedirs(save_dir, exist_ok=True)
    if config['project'].get('use_wandb', False) and global_rank == 0:
        wandb.init(
            project=wandb_project,
            entity=config['project'].get('wandb_entity', None),
            name=wandb_name,
            config=config
        )

    pretrained_ckpt_path = config['train'].get('pretrained_checkpoint')
    has_pretrained_ckpt = (pretrained_ckpt_path and os.path.exists(pretrained_ckpt_path))
    attn_implementation = config['model'].get('attn_implementation', 'flash_attention_2')
    if model_type == 'rt2_baseline':
        if global_rank == 0:
            print("Initializing RT2LikeBaseline model...")
        model = RT2LikeBaseline(
            lmm_path=config['model']['lmm_path'],
            vision_encoder_path=config['model'].get('vision_encoder_path', "google/siglip2-base-patch16-256"),
            action_dim=config['model']['action_dim'],
            num_actions=config['data']['future_len'],
            num_history=config['data']['history_len'],
            use_proprio_input_vlm=use_proprio_input_vlm,
            use_transformer_projector=config['model'].get('use_transformer_proprio_projector', True),
            projector_depth=config['model']['projector_depth'],
            projector_num_heads=config['model']['projector_num_heads'],
            backbone_mode=config['model'].get('backbone_mode', 'finetune'),
            gradient_checkpointing=config['model'].get('gradient_checkpointing', False),
            num_bins=config['model'].get('num_bins', 256),
            attn_implementation=attn_implementation,
        ).to(device, dtype=torch.bfloat16)
    elif model_type == 'smolvla_mq54_vita':
        model = SmolVLAMQ54VitaPolicy.from_config(
            config['model'],
            config['data'],
            load_action_effect_visual_teacher=use_smolvla_action_effect,
        ).to(device, dtype=torch.bfloat16)
    elif model_type == 'smolvla_vita_latent':
        model = SmolVLAVitaLatentPolicy(
            lmm_path=config['model']['lmm_path'],
            action_dim=config['model']['action_dim'],
            num_actions=config['data']['future_len'],
            num_history=config['data']['history_len'],
            latent_dim=config['model'].get('vita_latent_dim', 1024),
            action_hidden_dim=config['model'].get('vita_hidden_dim', 1024),
            action_ae_layers=config['model'].get('vita_action_ae_layers', 6),
            action_ae_dropout=config['model'].get('vita_dropout', 0.0),
            num_inference_timesteps=config['model'].get('num_inference_timesteps', 6),
            num_vlm_layers=config['model'].get('smolvla_num_vlm_layers', 16),
            num_expert_layers=config['model'].get('smolvla_num_expert_layers', 16),
            expert_width_multiplier=config['model'].get(
                'smolvla_expert_width_multiplier', 0.75
            ),
            self_attn_every_n_layers=config['model'].get(
                'smolvla_self_attn_every_n_layers', 2
            ),
            attention_mode=config['model'].get('smolvla_attention_mode', 'cross_attn'),
            load_vlm_weights=config['model'].get('smolvla_load_vlm_weights', True),
            freeze_vlm=config['model'].get('smolvla_freeze_vlm', True),
            freeze_vision_encoder=config['model'].get(
                'smolvla_freeze_vision_encoder', True
            ),
            freeze_visual_connector=config['model'].get(
                'smolvla_freeze_visual_connector', False
            ),
            state_dim=config['data']['history_len'] * config['model']['action_dim'],
            enc_recon_weight=config['model'].get('vita_enc_recon_weight', 0.2),
            consistency_weight=config['model'].get('vita_consistency_weight', 0.0),
            flow_recon_weight=config['model'].get('vita_flow_recon_weight', 0.0),
            num_sampling_steps=config['model'].get(
                'vita_num_sampling_steps',
                config['model'].get('num_inference_timesteps', 6),
            ),
            rollout_gradient_checkpointing=config['model'].get(
                'vita_rollout_gradient_checkpointing', True
            ),
            action_effect_enabled=use_smolvla_action_effect,
            action_effect_visual_teacher_type=action_effect_visual_teacher,
            action_effect_visual_teacher_path=config['model'].get(
                'vita_action_effect_visual_teacher_path', None
            ),
            action_effect_visual_ae_weight=config['model'].get(
                'vita_action_effect_visual_ae_weight', 0.01
            ),
            action_effect_decoder_layers=config['model'].get(
                'vita_action_effect_decoder_layers', 2
            ),
            action_representation_type=config['model'].get(
                'vita_action_representation_type', 'legacy'
            ),
            edar_stage_a_checkpoint=config['model'].get(
                'vita_edar_stage_a_checkpoint', None
            ),
            edar_train_decoder=config['model'].get(
                'vita_edar_train_decoder', False
            ),
        ).to(device, dtype=torch.bfloat16)
    else:
        model = VLANeXt(
            lmm_path=config['model']['lmm_path'],
            vision_encoder_path=config['model'].get('vision_encoder_path', "google/siglip2-base-patch16-256"),
            action_dim=config['model']['action_dim'],
            num_actions=config['data']['future_len'],
            num_queries=config['model']['num_queries'],
            num_history=config['data']['history_len'],
            loss_type=config['model'].get('loss_type', 'diffusion'),
            future_image_loss_weight=future_image_loss_weight,
            num_train_timesteps=config['model'].get('num_train_timesteps', 1000),
            num_inference_timesteps=config['model'].get('num_inference_timesteps', 10),
            scheduler_type=config['model']['scheduler_type'],
            condition_type=config['model'].get('condition_type', 'loose'),
            policy_hidden_size=config['model']['policy_hidden_size'],
            policy_depth=config['model']['policy_depth'],
            policy_num_heads=config['model']['policy_num_heads'],
            policy_mlp_ratio=config['model']['policy_mlp_ratio'],
            use_proprio_input_vlm=use_proprio_input_vlm,
            use_action_input_policy=use_action_input_policy,
            use_transformer_proprio_projector=config['model']['use_transformer_proprio_projector'],
            projector_depth=config['model']['projector_depth'],
            projector_num_heads=config['model']['projector_num_heads'],
            use_transformer_connector=config['model']['use_transformer_connector'],
            connector_depth=config['model']['connector_depth'],
            connector_num_heads=config['model']['connector_num_heads'],
            backbone_mode=config['model'].get('backbone_mode', 'finetune'),
            gradient_checkpointing=config['model'].get('gradient_checkpointing', False),
            num_bins=config['model'].get('num_bins', 256),
            generator_hidden_size=config['model'].get('generator_hidden_size', 768),
            generator_depth=config['model'].get('generator_depth', 12),
            generator_num_heads=config['model'].get('generator_num_heads', 12),
            generator_mlp_ratio=config['model'].get('generator_mlp_ratio', 4.0),
            action_vqvae=config['model'].get('action_vqvae', None),
            attn_implementation=attn_implementation,

            dct_loss_weight=config['model'].get('dct_loss_weight', 0.1),
            dct_low_freq_weight=config['model'].get('dct_low_freq_weight', 1.0),
            dct_high_freq_weight=config['model'].get('dct_high_freq_weight', 1.0),
            dct_freq_split=config['model'].get('dct_freq_split', 0.125),
            dct_similarity_type=config['model'].get('dct_similarity_type', 'mae'),
            use_progress_head=use_progress_head,
            use_progress_film=config['model'].get('use_progress_film', False),
            progress_loss_weight=config['model'].get('progress_loss_weight', 0.0),
            progress_detach=config['model'].get('progress_detach', True),
            progress_noise=config['model'].get('progress_noise', 0.0),
            use_vlm_layer_attention=config['model'].get('use_vlm_layer_attention', False),
            vlm_layer_attention_alpha=config['model'].get('vlm_layer_attention_alpha', 0.0),
            vlm_layer_indices=config['model'].get('vlm_layer_indices', None),
            vlm_truncate_layers=config['model'].get('vlm_truncate_layers', None),
            use_latent_bridge=use_latent_bridge,
            latent_bridge_num_heads=config['model'].get('latent_bridge_num_heads', 8),
            latent_bridge_max_views=config['model'].get('latent_bridge_max_views', 4),
            latent_bridge_visual_trainable=config['model'].get('latent_bridge_visual_trainable', True),
            latent_bridge_vlm_uses_proprio=config['model'].get('latent_bridge_vlm_uses_proprio', False),
            action_generation_mode=config['model'].get('action_generation_mode', 'direct_flow'),
            vita_latent_dim=config['model'].get('vita_latent_dim', 512),
            vita_hidden_dim=config['model'].get('vita_hidden_dim', 512),
            vita_action_ae_layers=config['model'].get('vita_action_ae_layers', 4),
            vita_action_encoder_type=config['model'].get(
                'vita_action_encoder_type', 'mlp'
            ),
            vita_action_cnn_layers=config['model'].get(
                'vita_action_cnn_layers', 4
            ),
            vita_action_cnn_kernel_size=config['model'].get(
                'vita_action_cnn_kernel_size', 5
            ),
            vita_flow_layers=config['model'].get('vita_flow_layers', 4),
            vita_flow_mlp_ratio=config['model'].get('vita_flow_mlp_ratio', 4.0),
            vita_dropout=config['model'].get('vita_dropout', 0.0),
            vita_num_sampling_steps=config['model'].get('vita_num_sampling_steps', 6),
            vita_enc_recon_weight=config['model'].get('vita_enc_recon_weight', 0.5),
            vita_flow_recon_weight=config['model'].get('vita_flow_recon_weight', 0.5),
            vita_consistency_weight=config['model'].get('vita_consistency_weight', 1.0),
            vita_diversity_loss_weight=config['model'].get('vita_diversity_loss_weight', 0.0),
            vita_log_diversity_loss=config['model'].get('vita_log_diversity_loss', False),
            vita_action_progress_loss_weight=config['model'].get('vita_action_progress_loss_weight', 0.0),
            vita_goal_distance_loss_weight=config['model'].get('vita_goal_distance_loss_weight', 0.0),
            vita_goal_metric_dim=config['model'].get('vita_goal_metric_dim', 256),
            vita_goal_distance_loss_type=config['model'].get('vita_goal_distance_loss_type', 'huber'),
            vita_goal_distance_detach_goal=config['model'].get('vita_goal_distance_detach_goal', False),
            vita_recon_loss_type=config['model'].get('vita_recon_loss_type', 'l1'),
            vita_use_proprio=config['model'].get('vita_use_proprio', True),
            vita_zobs_mode=config['model'].get('vita_zobs_mode', 'normal'),
            vita_zobs_noise_scale=config['model'].get('vita_zobs_noise_scale', 1.0),
            vita_condition_source=config['model'].get('vita_condition_source', 'auto'),
            vita_hidden_layer_index=config['model'].get('vita_hidden_layer_index', -1),
            vita_secondary_hidden_layer_index=config['model'].get('vita_secondary_hidden_layer_index', None),
            vita_extra_hidden_layer_indices=config['model'].get('vita_extra_hidden_layer_indices', None),
            vita_hidden_pooling=config['model'].get('vita_hidden_pooling', 'mean'),
            vita_pooling_num_heads=config['model'].get('vita_pooling_num_heads', 8),
            vita_pooling_num_queries=config['model'].get('vita_pooling_num_queries', 1),
            vita_gated_weighted_pooling=config['model'].get('vita_gated_weighted_pooling', False),
            vita_obs_pool_token_indices=config['model'].get('vita_obs_pool_token_indices', None),
            vita_hierarchical_query_pooling=config['model'].get('vita_hierarchical_query_pooling', False),
            vita_layer_local_queries_per_layer=config['model'].get('vita_layer_local_queries_per_layer', 2),
            vita_layer_local_queries_per_layer_list=config['model'].get('vita_layer_local_queries_per_layer_list', None),
            vita_layer_local_token_source_modes=config['model'].get('vita_layer_local_token_source_modes', None),
            hier_mq_separate_views=config['model'].get('hier_mq_separate_views', False),
            vita_cross_layer_queries=config['model'].get('vita_cross_layer_queries', 18),
            vita_global_queries=config['model'].get('vita_global_queries', 8),
            vita_adaptive_local_mq=config['model'].get('vita_adaptive_local_mq', False),
            vita_adaptive_mq_router_dim=config['model'].get('vita_adaptive_mq_router_dim', 256),
            vita_adaptive_mq_num_slots=config['model'].get('vita_adaptive_mq_num_slots', 4),
            vita_adaptive_mq_reserve_source_positions=config['model'].get(
                'vita_adaptive_mq_reserve_source_positions',
                None,
            ),
            vita_adaptive_mq_temperature=config['model'].get('vita_adaptive_mq_temperature', 1.0),
            vita_adaptive_mq_route_warmup_steps=config['model'].get(
                'vita_adaptive_mq_route_warmup_steps',
                0,
            ),
            vita_dynamic_topk_local_mq=config['model'].get(
                'vita_dynamic_topk_local_mq',
                False,
            ),
            vita_dynamic_topk_candidate_source_positions=config['model'].get(
                'vita_dynamic_topk_candidate_source_positions',
                None,
            ),
            vita_dynamic_topk_text_source_position=config['model'].get(
                'vita_dynamic_topk_text_source_position',
                1,
            ),
            vita_dynamic_topk_candidates_per_source=config['model'].get(
                'vita_dynamic_topk_candidates_per_source',
                8,
            ),
            vita_dynamic_topk_min_keep_per_source=config['model'].get(
                'vita_dynamic_topk_min_keep_per_source',
                2,
            ),
            vita_dynamic_topk_total_keep=config['model'].get(
                'vita_dynamic_topk_total_keep',
                24,
            ),
            vita_dynamic_topk_train_random_exploration=config['model'].get(
                'vita_dynamic_topk_train_random_exploration',
                False,
            ),
            vita_condition_type_embeddings=config['model'].get('vita_condition_type_embeddings', False),
            vita_dynamic_extra_layer_gates=config['model'].get('vita_dynamic_extra_layer_gates', False),
            vita_extra_layer_gate_scales=config['model'].get('vita_extra_layer_gate_scales', None),
            vita_extra_layer_gate_hidden_dim=config['model'].get('vita_extra_layer_gate_hidden_dim', 256),
            vita_extra_layer_gate_dropout=config['model'].get('vita_extra_layer_gate_dropout', 0.0),
            vita_blockwise_layer_conditioning=config['model'].get('vita_blockwise_layer_conditioning', False),
            vita_blockwise_hidden_layer_indices=config['model'].get('vita_blockwise_hidden_layer_indices', None),
            vita_state_token_conditioning=config['model'].get('vita_state_token_conditioning', False),
            vita_state_num_tokens=config['model'].get('vita_state_num_tokens', 4),
            vita_state_token_dropout=config['model'].get('vita_state_token_dropout', 0.0),
            vita_state_broadcast_to_mq=config['model'].get('vita_state_broadcast_to_mq', True),
            vita_mq_local_overlap_loss_weight=config['model'].get('vita_mq_local_overlap_loss_weight', 0.0),
            vita_mq_cross_overlap_loss_weight=config['model'].get('vita_mq_cross_overlap_loss_weight', 0.0),
            vita_mq_global_overlap_loss_weight=config['model'].get('vita_mq_global_overlap_loss_weight', 0.0),
            vita_mq_competitive_local_attention=config['model'].get('vita_mq_competitive_local_attention', False),
            vita_mq_competitive_attention_tau=config['model'].get('vita_mq_competitive_attention_tau', 1.0),
            vita_mq_competitive_attention_gamma=config['model'].get('vita_mq_competitive_attention_gamma', 1.0),
            vita_mq_local_balance_loss_weight=config['model'].get('vita_mq_local_balance_loss_weight', 0.0),
            vita_hierarchical_global_residual_block=config['model'].get('vita_hierarchical_global_residual_block', False),
            vita_hierarchical_global_residual_scale=config['model'].get('vita_hierarchical_global_residual_scale', 0.1),
            vita_hierarchical_global_ffn_ratio=config['model'].get('vita_hierarchical_global_ffn_ratio', 4.0),
            vita_layer_aligned_mq=config['model'].get('vita_layer_aligned_mq', False),
            vita_layer_aligned_hidden_layer_indices=config['model'].get('vita_layer_aligned_hidden_layer_indices', None),
            vita_layer_aligned_queries_per_layer=config['model'].get('vita_layer_aligned_queries_per_layer', 4),
            vita_layer_aligned_token_source_modes=config['model'].get('vita_layer_aligned_token_source_modes', None),
            vita_layer_aligned_global_conditioning=config['model'].get('vita_layer_aligned_global_conditioning', False),
            vita_layer_aligned_obs_pooling=config['model'].get('vita_layer_aligned_obs_pooling', 'last'),
            vita_layer_aligned_encoder_obs_conditioning=config['model'].get('vita_layer_aligned_encoder_obs_conditioning', True),
            vita_layer_aligned_flow_windows=config['model'].get('vita_layer_aligned_flow_windows', None),
            vita_flow_memory_tokens=config['model'].get('vita_flow_memory_tokens', 0),
            vita_flow_memory_update_scale=config['model'].get('vita_flow_memory_update_scale', 0.1),
            vita_flow_memory_mlp_ratio=config['model'].get('vita_flow_memory_mlp_ratio', 4.0),
            vita_flow_memory_update_type=config['model'].get('vita_flow_memory_update_type', 'attention'),
            vita_flow_memory_shared_update=config['model'].get('vita_flow_memory_shared_update', False),
            vita_flow_memory_bottleneck_dim=config['model'].get('vita_flow_memory_bottleneck_dim', 256),
            vita_flow_layer_logits_init=config['model'].get('vita_flow_layer_logits_init', None),
            vita_flow_cross_attention=config['model'].get('vita_flow_cross_attention', False),
            vita_flow_cross_attention_heads=config['model'].get('vita_flow_cross_attention_heads', 8),
            vita_flow_cross_attention_dim=config['model'].get('vita_flow_cross_attention_dim', None),
            vita_gated_flow_cross_attention=config['model'].get('vita_gated_flow_cross_attention', False),
            vita_typed_flow_cross_attention=config['model'].get('vita_typed_flow_cross_attention', False),
            vita_dynamic_typed_flow_gates=config['model'].get('vita_dynamic_typed_flow_gates', False),
            vita_mq_token_value_gating=config['model'].get('vita_mq_token_value_gating', False),
            vita_mq_headwise_token_value_gating=config['model'].get('vita_mq_headwise_token_value_gating', False),
            vita_mq_token_gate_lambda=config['model'].get('vita_mq_token_gate_lambda', 0.1),
            vita_mq_token_gate_use_action_latent=config['model'].get('vita_mq_token_gate_use_action_latent', False),
            vita_mq_token_gate_centered_residual=config['model'].get('vita_mq_token_gate_centered_residual', False),
            vita_mq_competitive_value_gating=config['model'].get('vita_mq_competitive_value_gating', False),
            vita_mq_global_relative_context_gating=config['model'].get(
                'vita_mq_global_relative_context_gating',
                False,
            ),
            vita_mq_global_relative_gate_score_dim=config['model'].get(
                'vita_mq_global_relative_gate_score_dim',
                128,
            ),
            vita_mq_global_relative_gate_temperature=config['model'].get(
                'vita_mq_global_relative_gate_temperature',
                1.0,
            ),
            vita_flow_static_condition_cache=config['model'].get(
                'vita_flow_static_condition_cache',
                False,
            ),
            vita_flow_cross_attention_fixed_scale=config['model'].get('vita_flow_cross_attention_fixed_scale', None),
            vita_action_representation_type=action_representation_type,
            vita_edar_stage_a_checkpoint=config['model'].get(
                'vita_edar_stage_a_checkpoint',
                None,
            ),
            vita_edar_train_decoder=config['model'].get(
                'vita_edar_train_decoder', False
            ),
            vita_edar_token_flow=config['model'].get(
                'vita_edar_token_flow', False
            ),
            vita_mq_type=config['model'].get('vita_mq_type', 'hier_mq54'),
            vita_enable_one_step_action_loss=config['model'].get(
                'enable_one_step_action_loss', False
            ),
            vita_flow_loss_weight_final=config['model'].get(
                'flow_loss_weight_final', 0.3
            ),
            vita_one_step_action_loss_weight_final=config['model'].get(
                'one_step_action_loss_weight_final', 1.0
            ),
            vita_stage_b_max_train_steps=config['data'].get('max_steps', 1),
            dynamic_flow_reconstruction_weight=config['model'].get(
                'dynamic_flow_reconstruction_weight', False
            ),
            dynamic_flow_latent_reconstruction_weight=config['model'].get(
                'dynamic_flow_latent_reconstruction_weight', False
            ),
            vita_action_effect_enabled=use_vita_action_effect,
            vita_action_effect_condition_action_encoder=config['model'].get(
                'vita_action_effect_condition_action_encoder', True
            ),
            vita_action_effect_visual_queries=config['model'].get('vita_action_effect_visual_queries', 4),
            vita_action_effect_visual_teacher=action_effect_visual_teacher,
            vita_action_effect_spatial_grid_size=config['model'].get(
                'vita_action_effect_spatial_grid_size', 8
            ),
            vita_action_effect_visual_teacher_path=config['model'].get(
                'vita_action_effect_visual_teacher_path', None
            ),
            vita_action_effect_load_visual_teacher=True,
            vita_action_effect_num_heads=config['model'].get('vita_action_effect_num_heads', 8),
            vita_action_effect_encoder_layers=config['model'].get('vita_action_effect_encoder_layers', 2),
            vita_action_effect_decoder_layers=config['model'].get('vita_action_effect_decoder_layers', 2),
            vita_action_effect_projection_dim=config['model'].get('vita_action_effect_projection_dim', None),
            vita_action_effect_state_conditioning=config['model'].get(
                'vita_action_effect_state_conditioning', False
            ),
            vita_action_effect_visual_ae_weight=config['model'].get('vita_action_effect_visual_ae_weight', 0.05),
            vita_action_effect_visual_flow_weight=config['model'].get('vita_action_effect_visual_flow_weight', 0.05),
            vita_action_effect_rank_weight=config['model'].get('vita_action_effect_rank_weight', 0.0),
            vita_action_effect_rank_margin=config['model'].get('vita_action_effect_rank_margin', 0.1),
            vita_action_effect_rank_min_delta_norm=config['model'].get(
                'vita_action_effect_rank_min_delta_norm', 0.0
            ),
            vita_action_effect_rank_distance_mode=config['model'].get(
                'vita_action_effect_rank_distance_mode', 'global_cosine'
            ),
            vita_action_effect_rank_huber_weight=config['model'].get(
                'vita_action_effect_rank_huber_weight', 0.0
            ),
            vita_action_effect_rank_scale_floor=config['model'].get(
                'vita_action_effect_rank_scale_floor', 1e-3
            ),
            vita_action_effect_negative_queue_size=config['model'].get(
                'vita_action_effect_negative_queue_size', 0
            ),
            vita_action_effect_num_negatives=config['model'].get(
                'vita_action_effect_num_negatives', 1
            ),
            vita_action_effect_hard_negatives=config['model'].get(
                'vita_action_effect_hard_negatives', 1
            ),
            vita_action_effect_negative_min_action_distance=config['model'].get(
                'vita_action_effect_negative_min_action_distance', 0.0
            ),
            vita_action_effect_main_view_only=config['model'].get('vita_action_effect_main_view_only', False),
            vita_action_effect_horizons=config['model'].get(
                'vita_action_effect_horizons', None
            ),
            vita_action_effect_horizon_loss_weights=config['model'].get(
                'vita_action_effect_horizon_loss_weights', None
            ),
            vita_action_effect_action_token_dim=config['model'].get(
                'vita_action_effect_action_token_dim', 256
            ),
            vita_action_effect_multihorizon_layers=config['model'].get(
                'vita_action_effect_multihorizon_layers', 2
            ),
            vita_action_effect_motion_weight_floor=config['model'].get(
                'vita_action_effect_motion_weight_floor', 0.25
            ),
            vita_action_effect_motion_weight_eps=config['model'].get(
                'vita_action_effect_motion_weight_eps', 1e-6
            ),
            vita_action_effect_detach_visual_teacher=config['model'].get(
                'vita_action_effect_detach_visual_teacher', False
            ),
            vlanext_dual_flow=config['model'].get('vlanext_dual_flow', False),
            vlanext_visual_flow_loss_weight=config['model'].get('vlanext_visual_flow_loss_weight', 0.0),
        ).to(device, dtype=torch.bfloat16)
    if has_pretrained_ckpt:
        if global_rank == 0:
            print(f"Loading pretrained VLA checkpoint: {pretrained_ckpt_path}")
        checkpoint = torch.load(pretrained_ckpt_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        reshaped_legacy_mq = reshape_legacy_layer_local_query_state(state_dict, model)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if global_rank == 0:
            if reshaped_legacy_mq:
                print("Reshaped legacy layer_local_query checkpoint tensor for current model.")
            print(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        if global_rank == 0:
            if pretrained_ckpt_path:
                print(f"Warning: Pretrained checkpoint path '{pretrained_ckpt_path}' does not exist. Training from scratch.")
            else:
                print("No pretrained checkpoint provided. Training from scratch.")

    configure_partial_vlm_trainability(model, config["model"])

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
        model_unwrapped = model.module
    else:
        model_unwrapped = model

    # -----------------------------------------------------------------------------
    # -------------------------------- Data Loader --------------------------------
    # -----------------------------------------------------------------------------
    data_root = config['data']['data_root']
    if dataset_name == "droid":
        droid_path = os.path.join(data_root, "1.0.1")
        if global_rank == 0:
            print(f"Initializing DROID Dataset: {droid_path}")
    elif dataset_name == "robotwin":
        robotwin_path = os.path.join(data_root, "dataset")
        if global_rank == 0:
            print(f"Initializing RoboTwin Dataset: {robotwin_path}")
    elif dataset_name == "bridge":
        bridge_path = os.path.join(data_root, config['data'].get('version', '1.0.0'))
        if global_rank == 0:
            print(f"Initializing BridgeData Dataset: {bridge_path}")
    elif dataset_name == "vlabench":
        vlabench_path = data_root
        if global_rank == 0:
            print(f"Initializing VLABench Dataset: {vlabench_path}")
    elif dataset_name == "vlabench_tfds":
        vlabench_tfds_path = data_root
        if global_rank == 0:
            print(f"Initializing VLABench TFDS Dataset: {vlabench_tfds_path}")
    elif dataset_name == "robomimic":
        robomimic_path = data_root
        if global_rank == 0:
            print(f"Initializing RoboMimic TFDS Dataset: {robomimic_path}")
    elif dataset_name == "calvin":
        calvin_path = data_root
        if global_rank == 0:
            print(f"Initializing CALVIN LeRobot Dataset: {calvin_path}")
    elif dataset_name == "av_aloha":
        av_aloha_path = data_root
        if global_rank == 0:
            print(f"Initializing AV-ALOHA Multitask Dataset: {av_aloha_path}")
    else:
        task_suite = config['data']['task_suite_name']
        libero_path = os.path.join(data_root, task_suite, "1.0.0")
        if global_rank == 0:
            print(f"Initializing Libero Dataset: {libero_path}")
    action_effect_teacher_processor = None
    if use_dinov2_action_effect_teacher or use_dinov3_action_effect_teacher:
        teacher_path = config['model'].get('vita_action_effect_visual_teacher_path', None)
        if not teacher_path:
            raise ValueError("DINO action-effect training requires a local teacher path.")
        if use_edar_lite:
            dino_metadata = (
                model_unwrapped.edar_dino_metadata
                if use_smolvla_vita
                else model_unwrapped.vita_action_generator.edar_dino_metadata
            )
            configured_teacher = os.path.realpath(os.path.expanduser(teacher_path))
            checkpoint_teacher = os.path.realpath(
                os.path.expanduser(dino_metadata["backbone"])
            )
            if configured_teacher != checkpoint_teacher:
                raise ValueError(
                    "Stage B DINO path does not match the Stage A checkpoint metadata."
                )
            action_effect_teacher_processor = DeterministicDINOImageProcessor(
                dino_metadata["image_mean"],
                dino_metadata["image_std"],
                image_size=int(dino_metadata["image_size"]),
            )
        else:
            action_effect_teacher_processor = AutoImageProcessor.from_pretrained(
                os.path.expanduser(teacher_path),
                local_files_only=True,
            )
    elif use_smolvlm_action_effect_teacher:
        action_effect_teacher_processor = model_unwrapped.processor.image_processor

    collator = DataCollatorForVLANeXt(
        processor=model_unwrapped.processor,
        use_proprio_input_vlm=use_proprio_input_vlm,
        use_action_input_policy=use_action_input_policy,
        input_modality=input_modality,
        view_mode=view_mode,
        fps=fps,
        augmentation=augmentation,
        load_future_image=load_future_image,
        include_progress=use_progress_head,
        include_action_progress=use_action_progress_alignment,
        include_goal_distance=use_goal_distance_regression,
        include_latent_bridge=use_latent_bridge,
        include_future_vlm=(
            use_vlanext_dual_visual_flow
            or (
                use_vita_action_effect
                and not use_preprocessed_action_effect_teacher
            )
        ),
        include_action_effect_teacher_images=use_preprocessed_action_effect_teacher,
        action_effect_teacher_processor=action_effect_teacher_processor,
        action_effect_teacher_type=action_effect_visual_teacher,
        action_effect_teacher_current_only=use_edar_lite,
        future_vlm_main_view_only=(
            (use_vita_action_effect or use_smolvla_action_effect)
            and config['model'].get('vita_action_effect_main_view_only', False)
        ),
        action_effect_horizons=action_effect_horizons,
        include_proprio=use_vita_proprio,
        bridge_image_size=config['model'].get('latent_bridge_image_size', 224),
    )
    total_batch_size = config['data']['batch_size']
    per_device_batch_size = total_batch_size // (world_size * gradient_accumulation_steps)
    if global_rank == 0:
        print(f"Total Batch Size: {total_batch_size}, World Size: {world_size}, Grad Acc Steps: {gradient_accumulation_steps}, Per-Device Batch Size: {per_device_batch_size}")

    default_buffer_size = 30000 if dataset_name == "droid" else 10000
    buffer_size = config['data'].get("buffer_size", default_buffer_size)
    if global_rank == 0:
        print(f"Dataset Shuffle Buffer Size: {buffer_size}")

    def create_dataloader(history_len):
        if dataset_name == "droid":
            ds = DroidAct(
                droid_path=droid_path,
                dataset_name="droid",
                history_len=history_len,
                future_len=config['data']['future_len'],
                action_stride=config['data'].get('action_stride', 1),
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
                buffer_size=buffer_size,
            )
        elif dataset_name == "robotwin":
            ds = RoboTwinAct(
                data_root=robotwin_path,
                setting=config['data'].get('robotwin_setting', 'aloha-agilex_clean_50'),
                tasks=config['data'].get('robotwin_tasks', None),
                max_episodes_per_task=config['data'].get('max_episodes_per_task', None),
                history_len=history_len,
                future_len=config['data']['future_len'],
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
                buffer_size=buffer_size,
                main_camera=config['data'].get('main_camera', 'head_camera'),
                wrist_camera=config['data'].get('wrist_camera', 'right_camera'),
                cameras=config['data'].get('robotwin_cameras', None),
                normalize_actions=config['data'].get('normalize_actions', True),
                action_mode=config['data'].get('robotwin_action_mode', 'absolute'),
                delta_stats_path=config['data'].get('robotwin_delta_stats_path', None),
                anchor_refresh_interval=config['model'].get('latent_bridge_refresh_interval', 1) if use_latent_bridge else 1,
                latent_bridge_sequence_len=config['data'].get('latent_bridge_sequence_len', 1) if use_latent_bridge else 1,
            )
        elif dataset_name == "bridge":
            ds = BridgeAct(
                data_path=bridge_path,
                history_len=history_len,
                future_len=config['data']['future_len'],
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
                buffer_size=buffer_size,
                main_camera=config['data'].get('main_camera', 'image_0'),
                wrist_camera=config['data'].get('wrist_camera', 'image_1'),
                action_stats_path=config['data'].get('action_stats_path', None),
                length=config['data'].get('max_episodes', None),
            )
        elif dataset_name == "vlabench":
            ds = VLABenchAct(
                data_path=vlabench_path,
                repo_id=config['data'].get('repo_id', 'lerobot/vlabench_unified'),
                history_len=history_len,
                future_len=config['data']['future_len'],
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
                buffer_size=buffer_size,
                main_camera=config['data'].get('main_camera', 'observation.images.image'),
                wrist_camera=config['data'].get('wrist_camera', 'observation.images.wrist_image'),
                normalize_actions=config['data'].get('normalize_actions', True),
                max_episodes=config['data'].get('max_episodes', None),
            )
        elif dataset_name == "vlabench_tfds":
            ds = VLABenchTFDSAct(
                data_path=vlabench_tfds_path,
                action_stats_path=config['data']['action_stats_path'],
                history_len=history_len,
                future_len=config['data']['future_len'],
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
                buffer_size=buffer_size,
                normalize_actions=config['data'].get('normalize_actions', True),
                max_episodes=config['data'].get('max_episodes', None),
            )
        elif dataset_name == "robomimic":
            ds = RoboMimicTFDSAct(
                data_root=robomimic_path,
                task_configs=config['data'].get(
                    'robomimic_tasks',
                    ['lift_ph_image', 'can_ph_image', 'square_ph_image'],
                ),
                history_len=history_len,
                future_len=config['data']['future_len'],
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                buffer_size=buffer_size,
            )
        elif dataset_name == "calvin":
            ds = CalvinAct(
                data_path=calvin_path,
                history_len=history_len,
                future_len=config['data']['future_len'],
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
                buffer_size=buffer_size,
                main_camera=config['data'].get(
                    'main_camera',
                    'observation.images.top',
                ),
                wrist_camera=config['data'].get(
                    'wrist_camera',
                    'observation.images.wrist',
                ),
                state_indices=config['data'].get(
                    'calvin_state_indices',
                    list(range(7)),
                ),
                strip_task_prefix=config['data'].get(
                    'calvin_strip_task_prefix',
                    True,
                ),
                clip_actions=config['data'].get(
                    'calvin_clip_actions',
                    True,
                ),
                max_episodes=config['data'].get('max_episodes', None),
                shuffle_seed=seed,
            )
        elif dataset_name == "av_aloha":
            ds = AVAlohaMultitaskAct(
                data_root=av_aloha_path,
                tasks=config['data']['av_aloha_tasks'],
                history_len=history_len,
                future_len=config['data']['future_len'],
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
                buffer_size=buffer_size,
                main_camera=config['data'].get(
                    'main_camera',
                    'observation.images.zed_cam_left',
                ),
                secondary_camera=config['data'].get(
                    'secondary_camera',
                    'observation.images.zed_cam_right',
                ),
                action_normalization=config['data'].get(
                    'action_normalization',
                    'min_max',
                ),
                state_normalization=config['data'].get(
                    'state_normalization',
                    'identity',
                ),
                normalization_clip=config['data'].get(
                    'normalization_clip',
                    5.0,
                ),
                balance_tasks=config['data'].get('balance_tasks', True),
                max_episodes_per_task=config['data'].get(
                    'max_episodes_per_task',
                    None,
                ),
                shuffle_seed=seed,
            )
        else:
            ds = LiberoAct(
                data_path=libero_path,
                dataset_name=task_suite,
                history_len=history_len,
                future_len=config['data']['future_len'],
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
                future_image_offsets=future_image_offsets,
                strict_future_horizon=False,
                buffer_size=buffer_size,
                anchor_refresh_interval=config['model'].get('latent_bridge_refresh_interval', 1) if use_latent_bridge else 1,
                latent_bridge_sequence_len=config['data'].get('latent_bridge_sequence_len', 1) if use_latent_bridge else 1,
                normalization_mode=config['data'].get('normalization_mode', 'min_max'),
                normalization_stats_path=config['data'].get('normalization_stats_path', None),
            )
        return DataLoader(
            ds, 
            batch_size=per_device_batch_size,
            num_workers=config['data']['num_workers'],
            collate_fn=collator
        )

    dataloader = create_dataloader(config['data']['history_len'])

    # -----------------------------------------------------------------------------
    # ---------------------- VQ-VAE Training (if applicable) ----------------------
    # -----------------------------------------------------------------------------
    vqvae_config = config['model'].get('action_vqvae', {})
    if (
        vqvae_config.get('enabled', False) 
        and not config['train'].get('resume_path')
    ):
        if global_rank == 0:
            print("\n=== Starting Action VQ-VAE Pre-training ===")
        vqvae_params = list(model_unwrapped.action_vqvae.parameters())
        vqvae_optim = AdamW(
            vqvae_params,
            lr=float(vqvae_config.get('learning_rate', 1e-3)),
            weight_decay=float(config['train']['weight_decay'])
        )
        vqvae_steps = vqvae_config.get('steps', 1000)
        vqvae_pbar = tqdm(total=vqvae_steps, desc="Pre-training VQ-VAE", disable=global_rank != 0)
        vqvae_iter = iter(dataloader)
        model.train()
        
        for i in range(vqvae_steps):
            try:
                batch = next(vqvae_iter)
            except StopIteration:
                vqvae_iter = iter(dataloader)
                batch = next(vqvae_iter)
            _, gt_actions, _, _, *_ = batch
            gt_actions = gt_actions.to(device, dtype=torch.bfloat16)
            vqvae_optim.zero_grad()
            loss = model(actions=gt_actions, task="action_vqvae_pretrain")
            loss.backward()
            vqvae_optim.step()
            if global_rank == 0:
                vqvae_pbar.update(1)
                vqvae_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                if config['project'].get('use_wandb', False) and i % config['project']['log_interval'] == 0:
                    wandb.log({"vqvae_pretrain/loss": loss.item(), "vqvae_pretrain/step": i})
        
        if vqvae_config.get('frozen', True):
            if global_rank == 0: print("Freezing VQ-VAE parameters after pretraining.")
            model_unwrapped.action_vqvae.requires_grad_(False)
            model_unwrapped.action_vqvae.eval()
        else:
            if global_rank == 0: print("Keeping VQ-VAE parameters trainable (finetuning).")
            model_unwrapped.action_vqvae.requires_grad_(True)
            model_unwrapped.action_vqvae.train()
            
        if global_rank == 0:
            vqvae_pbar.close()
            print("=== Action VQ-VAE Pre-training Finished ===\n")
            torch.save(model_unwrapped.action_vqvae.state_dict(), os.path.join(save_dir, "action_vqvae_pretrained.pt"))

    # -----------------------------------------------------------------------------
    # ------------------------ Action Generation Training -------------------------
    # -----------------------------------------------------------------------------
    base_lr = float(config['train']['learning_rate'])
    weight_decay = float(config['train']['weight_decay'])
    optimizer_betas = tuple(
        float(value)
        for value in config['train'].get('optimizer_betas', (0.9, 0.999))
    )
    if len(optimizer_betas) != 2:
        raise ValueError("optimizer_betas must contain exactly two values.")
    optimizer_eps = float(config['train'].get('optimizer_eps', 1e-8))
    vlm_lr_multiplier = float(config['train'].get('vlm_lr_multiplier', 1.0))
    if vlm_lr_multiplier != 1.0 and hasattr(model_unwrapped, "lmm"):
        vlm_param_ids = {id(p) for p in model_unwrapped.lmm.parameters()}
        if hasattr(model_unwrapped, "vision_encoder"):
            vlm_param_ids.update(id(p) for p in model_unwrapped.vision_encoder.parameters())
        vlm_params = []
        action_params = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in vlm_param_ids:
                vlm_params.append(p)
            else:
                action_params.append(p)
        optimizer_groups = []
        if action_params:
            optimizer_groups.append({
                "params": action_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            })
        if vlm_params:
            optimizer_groups.append({
                "params": vlm_params,
                "lr": base_lr * vlm_lr_multiplier,
                "weight_decay": weight_decay,
            })
        optimizer = AdamW(
            optimizer_groups,
            betas=optimizer_betas,
            eps=optimizer_eps,
        )
        if global_rank == 0:
            print(
                f"Optimizer parameter groups: action_lr={base_lr:.3e}, "
                f"vlm_lr={base_lr * vlm_lr_multiplier:.3e}, "
                f"action_params={sum(p.numel() for p in action_params)}, "
                f"vlm_params={sum(p.numel() for p in vlm_params)}"
            )
    else:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr,
            weight_decay=weight_decay,
            betas=optimizer_betas,
            eps=optimizer_eps,
        )
    
    lr_scheduler = build_training_scheduler(config, optimizer)

    start_step = 0
    if config['train'].get('resume_path'):
        resume_path = config['train']['resume_path']
        if os.path.exists(resume_path):
            if global_rank == 0:
                print(f"Resuming training from checkpoint: {resume_path}")
            # Load on CPU so the checkpoint does not temporarily duplicate the
            # full model and optimizer state in GPU memory during resume.
            checkpoint = torch.load(resume_path, map_location="cpu")
            state_dict = checkpoint['model_state_dict']

            if is_distributed and not list(state_dict.keys())[0].startswith('module.'):
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            elif not is_distributed and list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            reshaped_legacy_mq = reshape_legacy_layer_local_query_state(state_dict, model)
            if global_rank == 0 and reshaped_legacy_mq:
                print("Reshaped legacy layer_local_query checkpoint tensor for current model.")
            missing = load_model_state_for_resume(model, state_dict)
            if global_rank == 0 and missing:
                print(
                    "Reloaded frozen DINOv2 teacher from its local pretrained path; "
                    "teacher weights are intentionally absent from the checkpoint."
                )
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_step = checkpoint['step']
            del state_dict
            del checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if global_rank == 0:
                print(f"Resumed at step {start_step}")
        else:
            if global_rank == 0:
                print(f"Warning: Resume path {resume_path} does not exist. Starting from scratch.")
    
    model.train()
    step = start_step
    batch_idx = 0
    profile_timing = bool(config['project'].get('profile_timing', False))
    timing_interval = int(config['project'].get('timing_interval', config['project'].get('log_interval', 20)))
    timing_sync_cuda = bool(config['project'].get('timing_sync_cuda', True))
    wandb_core_metrics = config['project'].get('wandb_core_metrics', None)
    wandb_core_metrics = (
        {str(name) for name in wandb_core_metrics}
        if wandb_core_metrics is not None
        else None
    )
    wandb_diagnostic_metrics = {
        str(name)
        for name in config['project'].get('wandb_diagnostic_metrics', [])
    }
    wandb_diagnostic_interval = int(
        config['project'].get('wandb_diagnostic_interval', 200)
    )
    if wandb_diagnostic_interval <= 0:
        raise ValueError("project.wandb_diagnostic_interval must be positive.")
    timing_acc = {
        "data_s": 0.0,
        "h2d_s": 0.0,
        "forward_s": 0.0,
        "backward_s": 0.0,
        "optim_s": 0.0,
        "log_save_s": 0.0,
        "total_s": 0.0,
        "micro_batches": 0,
    }

    def mark_time():
        if profile_timing and timing_sync_cuda and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    if global_rank == 0:
        progress_bar = tqdm(total=config['data']['max_steps'], initial=start_step, desc="Finetuning")
    data_iter = iter(dataloader)
    optimizer.zero_grad()
    while step < config['data']['max_steps']:
        iter_t0 = mark_time() if profile_timing else None
        data_t0 = mark_time() if profile_timing else None
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        data_s = (mark_time() - data_t0) if profile_timing else 0.0
            
        if use_progress_head or use_action_progress_alignment or use_goal_distance_regression:
            inputs, gt_actions, proprio, hist_actions, future_images, progress_labels, action_progress_labels, goal_distance_labels = batch
        else:
            inputs, gt_actions, proprio, hist_actions, future_images = batch
            progress_labels = None
            action_progress_labels = None
            goal_distance_labels = None
        del batch
        h2d_t0 = mark_time() if profile_timing else None
        model_inputs = {k: v.to(device) for k, v in inputs.items()}
        del inputs
        for k in [
            'pixel_values', 'pixel_values_videos', 'bridge_pixel_values',
            'goal_pixel_values', 'goal_pixel_values_videos',
            'future_pixel_values', 'future_pixel_values_videos',
            'action_effect_teacher_current_pixel_values',
            'action_effect_teacher_future_pixel_values',
        ]:
            if k in model_inputs:
                model_inputs[k] = model_inputs[k].to(dtype=torch.bfloat16)

        gt_actions = gt_actions.to(device, dtype=torch.bfloat16)
        if proprio is not None:
            proprio = proprio.to(device, dtype=torch.bfloat16)
        if hist_actions is not None:
            hist_actions = hist_actions.to(device, dtype=torch.bfloat16)
        if future_images is not None:
            future_images = future_images.to(device, dtype=torch.bfloat16)
        if progress_labels is not None:
            progress_labels = progress_labels.to(device, dtype=torch.float32)
        if action_progress_labels is not None:
            action_progress_labels = action_progress_labels.to(device, dtype=torch.float32)
        if goal_distance_labels is not None:
            goal_distance_labels = goal_distance_labels.to(device, dtype=torch.float32)
        h2d_s = (mark_time() - h2d_t0) if profile_timing else 0.0
        
        valid_keys = {
            "input_ids", "attention_mask", "action_mask", "pixel_values", "pixel_attention_mask",
            "pixel_values_videos",
            "image_grid_thw", "video_grid_thw", "token_type_ids", "mm_token_type_ids",
            "bridge_pixel_values",
            "goal_input_ids", "goal_attention_mask", "goal_pixel_values",
            "goal_pixel_values_videos", "goal_image_grid_thw", "goal_video_grid_thw",
            "goal_token_type_ids", "goal_mm_token_type_ids",
            "future_input_ids", "future_attention_mask", "future_pixel_values",
            "future_pixel_values_videos", "future_image_grid_thw", "future_video_grid_thw",
            "future_token_type_ids", "future_mm_token_type_ids",
            "action_effect_teacher_current_pixel_values",
            "action_effect_teacher_future_pixel_values",
        }
        forward_args = {k: v for k, v in model_inputs.items() if k in valid_keys}
        do_update = (batch_idx + 1) % gradient_accumulation_steps == 0
        route_model = model.module if hasattr(model, "module") else model
        if hasattr(route_model, "set_vita_adaptive_mq_step"):
            route_model.set_vita_adaptive_mq_step(step)
        sync_context = model.no_sync if (is_distributed and not do_update) else nullcontext
        with sync_context():
            forward_t0 = mark_time() if profile_timing else None
            model_out = model(
                actions=gt_actions,
                proprioception=proprio,
                history_actions=hist_actions,
                future_images=future_images,
                progress_labels=progress_labels,
                action_progress_labels=action_progress_labels,
                goal_distance_labels=goal_distance_labels,
                return_loss_dict=config['project'].get('use_wandb', False) and global_rank == 0,
                **forward_args
            )
            if isinstance(model_out, dict):
                loss = model_out["loss"]
                loss_items = {
                    k: v.detach()
                    for k, v in model_out.items()
                    if k != "loss" and torch.is_tensor(v)
                }
            else:
                loss = model_out
                loss_items = {}
            loss = loss / gradient_accumulation_steps
            forward_s = (mark_time() - forward_t0) if profile_timing else 0.0
            backward_t0 = mark_time() if profile_timing else None
            loss.backward()
            backward_s = (mark_time() - backward_t0) if profile_timing else 0.0
        
        optim_s = 0.0
        log_save_s = 0.0
        if do_update:
            optim_t0 = mark_time() if profile_timing else None
            if config['train']['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['max_grad_norm'])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            optim_s = (mark_time() - optim_t0) if profile_timing else 0.0
            step += 1
            log_t0 = mark_time() if profile_timing else None
            if global_rank == 0:
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})
            if step % config['project']['log_interval'] == 0 and global_rank == 0:
                if config['project'].get('use_wandb', False):
                    log_items = {
                        "train/loss": loss.item() * gradient_accumulation_steps,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "step": step
                    }
                    log_diagnostics = step % wandb_diagnostic_interval == 0
                    for name, value in loss_items.items():
                        should_log = (
                            wandb_core_metrics is None
                            or name in wandb_core_metrics
                            or (
                                log_diagnostics
                                and name in wandb_diagnostic_metrics
                            )
                        )
                        if should_log:
                            log_items[f"train/{name}"] = value.float().item()
                    wandb.log(log_items, step=step)
            if step % config['project']['save_interval'] == 0 and global_rank == 0:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"checkpoint_{step}.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': checkpoint_model_state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'config': config
                }, save_path)
                print(f"\nSaved checkpoint to {save_path}")
            log_save_s = (mark_time() - log_t0) if profile_timing else 0.0
                
        if profile_timing:
            timing_acc["data_s"] += data_s
            timing_acc["h2d_s"] += h2d_s
            timing_acc["forward_s"] += forward_s
            timing_acc["backward_s"] += backward_s
            timing_acc["optim_s"] += optim_s
            timing_acc["log_save_s"] += log_save_s
            timing_acc["total_s"] += mark_time() - iter_t0
            timing_acc["micro_batches"] += 1

            if do_update and global_rank == 0 and step % timing_interval == 0:
                mb = max(1, timing_acc["micro_batches"])
                upd = max(1, mb // gradient_accumulation_steps)
                timing_items = {
                    "timing/data_s_per_micro": timing_acc["data_s"] / mb,
                    "timing/h2d_s_per_micro": timing_acc["h2d_s"] / mb,
                    "timing/forward_s_per_micro": timing_acc["forward_s"] / mb,
                    "timing/backward_s_per_micro": timing_acc["backward_s"] / mb,
                    "timing/optim_s_per_update": timing_acc["optim_s"] / upd,
                    "timing/log_save_s_per_update": timing_acc["log_save_s"] / upd,
                    "timing/total_s_per_micro": timing_acc["total_s"] / mb,
                }
                progress_bar.write(
                    "[timing] "
                    + " ".join(
                        f"{k.split('/')[-1]}={v:.3f}"
                        for k, v in timing_items.items()
                    )
                )
                if config['project'].get('use_wandb', False):
                    wandb.log({**timing_items, "step": step}, step=step)
                for key in timing_acc:
                    timing_acc[key] = 0 if key == "micro_batches" else 0.0

        batch_idx += 1

    if global_rank == 0:
        print("Finetuning finished.")
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'step': step,
            'model_state_dict': checkpoint_model_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'config': config
        }, os.path.join(save_dir, "checkpoint_final.pt"))

        if config['project'].get('use_wandb', False):
            wandb.finish()
            
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/libero_train_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config)
