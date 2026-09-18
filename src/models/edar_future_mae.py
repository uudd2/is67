"""Independent action-conditioned Future-MAE; no future DINO supervision."""
import torch
from torch import nn
import torch.nn.functional as F

from src.models.edar_lite import SingleViewEDARLiteEncoder, SingleViewEDARLiteDecoder


def sincos_2d(grid, dimension):
    if dimension % 4:
        raise ValueError('Position embedding dimension must be divisible by four.')
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
    omega = 10000.0 ** (-torch.arange(dimension // 4).float() / (dimension // 4))
    embeddings = []
    for coordinate in (y, x):
        angles = coordinate.flatten().float()[:, None] * omega[None]
        embeddings.extend((angles.sin(), angles.cos()))
    return torch.cat(embeddings, dim=-1).unsqueeze(0)


def prepare_rgb(images, image_size=224):
    """Deterministic shared resize/center-crop geometry; RGB stays in [0,1]."""
    images = torch.as_tensor(images)
    if images.ndim != 4:
        raise ValueError('Expected a batch of RGB images.')
    if images.shape[-1] == 3 and images.shape[1] != 3:
        images = images.permute(0, 3, 1, 2)
    if images.shape[1] != 3:
        raise ValueError('Expected three RGB channels.')
    integer_input = images.dtype == torch.uint8
    images = images.float()
    if integer_input:
        images = images / 255.0
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError('RGB must be uint8 or floating point in [0,1].')
    height, width = images.shape[-2:]
    scale = image_size / min(height, width)
    height, width = max(image_size, round(height * scale)), max(image_size, round(width * scale))
    images = F.interpolate(images, size=(height, width), mode='bilinear', align_corners=False, antialias=True)
    top, left = (height-image_size)//2, (width-image_size)//2
    return images[:, :, top:top+image_size, left:left+image_size].contiguous()


class MAEBlock(nn.Module):
    def __init__(self, dimension, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, eps=1e-6)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True, dropout=0)
        self.norm2 = nn.LayerNorm(dimension, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dimension, int(dimension*mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dimension*mlp_ratio), dimension))

    def forward(self, tokens):
        normalized = self.norm1(tokens)
        tokens = tokens + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return tokens + self.mlp(self.norm2(tokens))


class EDARFutureMAE(nn.Module):
    def __init__(self, visual_dim=1024, action_dim=7, action_horizon=8,
                 edar_model_dim=512, edar_layers=4, edar_heads=8,
                 image_size=224, patch_size=16, encoder_embed_dim=768,
                 encoder_depth=12, encoder_heads=12, decoder_embed_dim=512,
                 decoder_depth=8, decoder_heads=16, mlp_ratio=4.0,
                 mask_ratio=0.75, norm_pix_loss=False):
        super().__init__()
        if (image_size, patch_size, action_horizon, action_dim) != (224, 16, 8, 7):
            raise ValueError('This experiment uses 224px RGB, 16px patches and 8x7 actions.')
        if norm_pix_loss:
            raise ValueError('First Future-MAE experiment supports raw RGB targets only; use norm_pix_loss=false.')
        if not 0 <= mask_ratio < 1:
            raise ValueError('mask_ratio must be in [0,1).')
        self.image_size, self.patch_size, self.num_patches = image_size, patch_size, 196
        self.mask_ratio = float(mask_ratio)
        edar = dict(action_dim=action_dim, action_horizon=action_horizon, visual_dim=visual_dim,
                    visual_grid=8, model_dim=edar_model_dim, latent_tokens=4, latent_token_dim=256,
                    num_layers=edar_layers, num_heads=edar_heads, mlp_ratio=mlp_ratio)
        self.action_encoder = SingleViewEDARLiteEncoder(**edar)
        self.action_decoder = SingleViewEDARLiteDecoder(**edar)
        # Keep the imported class intact; its unused visual-only parameters are frozen.
        for name, parameter in self.action_decoder.named_parameters():
            if name.startswith('visual_') or '.norm_vis.' in name or '.ffn_vis.' in name:
                parameter.requires_grad_(False)
        self.patch_embed = nn.Conv2d(3, encoder_embed_dim, kernel_size=16, stride=16)
        self.register_buffer('encoder_positions', sincos_2d(14, encoder_embed_dim))
        self.encoder_blocks = nn.ModuleList([MAEBlock(encoder_embed_dim, encoder_heads, mlp_ratio)
                                              for _ in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.decoder_projection = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, decoder_embed_dim))
        self.register_buffer('decoder_positions', sincos_2d(14, decoder_embed_dim))
        self.action_condition = nn.Linear(256, decoder_embed_dim)
        self.decoder_blocks = nn.ModuleList([MAEBlock(decoder_embed_dim, decoder_heads, mlp_ratio)
                                              for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.decoder_prediction = nn.Linear(decoder_embed_dim, 16*16*3)
        # Initialize only new MAE modules, preserving imported EDAR initialization.
        for module in (self.encoder_blocks, self.encoder_norm, self.decoder_projection,
                       self.action_condition, self.decoder_blocks, self.decoder_norm, self.decoder_prediction):
            module.apply(self._initialize)
        nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def patchify(self, images):
        if images.shape[1:] != (3, 224, 224):
            raise ValueError('patchify expects [B,3,224,224].')
        return images.reshape(-1, 3, 14, 16, 14, 16).permute(0, 2, 4, 3, 5, 1).reshape(-1, 196, 768)

    def unpatchify(self, patches):
        if patches.shape[1:] != (196, 768):
            raise ValueError('unpatchify expects [B,196,768].')
        return patches.reshape(-1, 14, 14, 16, 16, 3).permute(0, 5, 1, 3, 2, 4).reshape(-1, 3, 224, 224)

    def random_masking(self, tokens, noise=None):
        batch, length, width = tokens.shape
        keep = int(length * (1.0-self.mask_ratio))
        if noise is None:
            noise = torch.rand(batch, length, device=tokens.device)
        if noise.shape != (batch, length):
            raise ValueError('Mask noise must have shape [B,196].')
        order = noise.argsort(dim=1)
        restore = order.argsort(dim=1)
        visible = tokens.gather(1, order[:, :keep, None].expand(-1, -1, width))
        mask = tokens.new_ones(batch, length)
        mask[:, :keep] = 0
        return visible, mask.gather(1, restore), restore

    def predict_future(self, current_rgb, action_latent, mask_noise=None):
        if current_rgb.shape[1:] != (3, 224, 224):
            raise ValueError('current_rgb must have shape [B,3,224,224].')
        tokens = self.patch_embed(current_rgb).flatten(2).transpose(1, 2)
        tokens = tokens + self.encoder_positions.to(tokens.dtype)
        visible, mask, restore = self.random_masking(tokens, mask_noise)
        for block in self.encoder_blocks:
            visible = block(visible)
        visible = self.decoder_projection(self.encoder_norm(visible))
        masked = self.mask_token.to(visible.dtype).expand(visible.shape[0], 196-visible.shape[1], -1)
        restored = torch.cat([visible, masked], dim=1).gather(1, restore[:, :, None].expand(-1, -1, visible.shape[-1]))
        restored = restored + self.decoder_positions.to(restored.dtype)
        conditions = self.action_condition(action_latent.reshape(-1, 4, 256))
        tokens = torch.cat([restored, conditions], dim=1)
        for block in self.decoder_blocks:
            tokens = block(tokens)
        prediction = self.decoder_prediction(self.decoder_norm(tokens[:, :196]))
        return prediction, mask

    def forward(self, actions, current_visual_tokens, current_rgb, mask_noise=None):
        latent = self.action_encoder(actions, current_visual_tokens.detach())
        decoded = self.action_decoder.decode_actions(latent)
        prediction, mask = self.predict_future(current_rgb, latent, mask_noise)
        return {'action_latent': latent, 'decoded_actions': decoded,
                'pred_future_patches': prediction, 'pred_future_rgb': self.unpatchify(prediction),
                'mae_mask': mask}

    def representation_loss(self, outputs, actions, current_rgb, future_rgb, rgb_weight=0.2):
        loss_action = F.mse_loss(outputs['decoded_actions'].float(), actions.float())
        # ALL 196 FUTURE patches, including positions visible in the current image.
        loss_rgb = F.mse_loss(outputs['pred_future_patches'].float(), self.patchify(future_rgb.detach().float()))
        loss = loss_action + float(rgb_weight)*loss_rgb
        with torch.no_grad():
            pred_mse = F.mse_loss(outputs['pred_future_rgb'].float(), future_rgb.float())
            copy_mse = F.mse_loss(current_rgb.float(), future_rgb.float())
            latent = outputs['action_latent'].float()
            metrics = {'loss': loss.detach(), 'loss_action': loss_action.detach(), 'loss_rgb': loss_rgb.detach(),
                       'action_mse': loss_action.detach(), 'psnr': -10*torch.log10(pred_mse.clamp_min(1e-12)),
                       'copy_mse': copy_mse, 'mae_gain': copy_mse-pred_mse,
                       'latent_mean': latent.mean(), 'latent_std': latent.std(unbiased=False),
                       'latent_norm': latent.norm(dim=-1).mean()}
        return loss, metrics
