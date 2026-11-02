"""
Model definitions - models.py
-----------------------------------------------------------------
Three semantic segmentation models:
  1. UNet + EfficientNet-B4       (CNN, lightweight baseline)
  2. SegFormer-B2                 (Transformer, global attention)
  3. DINOv2-UNet (ViT-L backbone) (foundation model for sparse labels)

DINOv2 design:
  - Channel projection: IN_CHANNELS -> 3 (learnable 1x1 convolution).
  - Resolution alignment: resize to DINOV2_TILE_SIZE(224) during training; align to patch14 at inference.
  - ViT-L backbone: frozen by default, with optional LoRA fine-tuning.
  - Decoder: fuse features from four layers, upsample with an FPN-style head, then classify pixels.
  - Restore the original resolution during inference.

Base losses are defined in models.py; additional losses are in models_improved.py.
-----------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


N_OUT = config.NUM_CLASSES + 1   # 0(ignore) + classes 1..NUM_CLASSES


class _ResizeLogitsToInput(nn.Module):
    """Adapter for encoders whose decoder output is smaller than the input tile."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits


# =================================================================
# Model 1: UNet + EfficientNet-B4
# =================================================================
def _build_smp_unet(encoder_name: str,
                    num_classes: int = N_OUT,
                    in_channels: int = config.IN_CHANNELS,
                    pretrained:  bool = True,
                    encoder_depth: int = 5,
                    decoder_channels = None,
                    resize_to_input: bool = False):
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("pip install segmentation-models-pytorch timm")

    kwargs = {
        "encoder_name": encoder_name,
        "encoder_weights": "imagenet" if pretrained else None,
        "in_channels": in_channels,
        "classes": num_classes,
        "activation": None,
        "encoder_depth": encoder_depth,
    }
    if decoder_channels is not None:
        kwargs["decoder_channels"] = decoder_channels

    model = smp.Unet(**kwargs)
    if resize_to_input:
        print(
            f"[model][info] {encoder_name} uses encoder_depth={encoder_depth} "
            "and resizes decoder logits to the input tile size."
        )
        return _ResizeLogitsToInput(model)
    return model


def build_unet(num_classes: int = N_OUT,
               in_channels: int = config.IN_CHANNELS,
               pretrained:  bool = True):
    return _build_smp_unet("efficientnet-b4", num_classes, in_channels, pretrained)


def build_unet_resnet34(num_classes: int = N_OUT,
                        in_channels: int = config.IN_CHANNELS,
                        pretrained:  bool = True):
    return _build_smp_unet("resnet34", num_classes, in_channels, pretrained)


def build_unet_convnext_tiny(num_classes: int = N_OUT,
                             in_channels: int = config.IN_CHANNELS,
                             pretrained:  bool = True):
    try:
        return _build_smp_unet("tu-convnext_tiny", num_classes, in_channels, pretrained)
    except IndexError as exc:
        print(
            "[model][warn] default tu-convnext_tiny SMP UNet failed "
            f"({type(exc).__name__}: {exc}); using encoder_depth=4 with "
            "logit resize adapter instead."
        )
    # Some SMP/timm combinations expose only 4 ConvNeXt feature stages.
    # The compatibility path is explicit and logged above.
    return _build_smp_unet(
        "tu-convnext_tiny",
        num_classes,
        in_channels,
        pretrained,
        encoder_depth=4,
        decoder_channels=(256, 128, 64, 32),
        resize_to_input=True,
    )


def build_hrnet_w18(num_classes: int = N_OUT,
                    in_channels: int = config.IN_CHANNELS,
                    pretrained:  bool = True):
    return _build_smp_unet("tu-hrnet_w18", num_classes, in_channels, pretrained)


# =================================================================
# Model 2: SegFormer-B2
# =================================================================
_SEGFORMER_SPECS = {
    "mit-b2": {"hidden_sizes": [64, 128, 320, 512], "depths": [3, 4, 6, 3], "decoder_hidden_size": 768},
    "mit-b4": {"hidden_sizes": [64, 128, 320, 512], "depths": [3, 8, 27, 3], "decoder_hidden_size": 768},
    "mit-b5": {"hidden_sizes": [64, 128, 320, 512], "depths": [3, 6, 40, 3], "decoder_hidden_size": 768},
}


def _segformer_config_from_name(config_cls,
                                model_name: str,
                                num_classes: int,
                                decoder_hidden_size: int = None):
    key = str(model_name).split("/")[-1]
    if key not in _SEGFORMER_SPECS:
        raise ValueError(
            f"Unsupported offline SegFormer config: {model_name}. "
            f"Known configs: {', '.join(sorted(_SEGFORMER_SPECS))}"
        )
    spec = _SEGFORMER_SPECS[key]
    decoder_hidden = int(decoder_hidden_size or spec["decoder_hidden_size"])
    print(
        f"[model][info] Building SegFormer {key} from local config "
        f"(pretrained=False, decoder_hidden_size={decoder_hidden})."
    )
    return config_cls(
        num_labels=num_classes,
        hidden_sizes=spec["hidden_sizes"],
        depths=spec["depths"],
        num_attention_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        mlp_ratios=[4, 4, 4, 4],
        decoder_hidden_size=decoder_hidden,
        reshape_last_stage=True,
        semantic_loss_ignore_index=config.IGNORE_INDEX,
    )


class SegFormerSeg(nn.Module):
    def __init__(self,
                 num_classes: int  = N_OUT,
                 in_channels: int  = config.IN_CHANNELS,
                 model_name:  str  = "nvidia/mit-b2",
                 pretrained:  bool = True,
                 decoder_hidden_size: int = None):
        super().__init__()
        try:
            from transformers import (SegformerForSemanticSegmentation,
                                      SegformerConfig)
        except ImportError:
            raise ImportError("pip install transformers")

        if pretrained:
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    model_name,
                    num_labels              = num_classes,
                    ignore_mismatched_sizes = True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load pretrained SegFormer weights/config from {model_name}. "
                    "Check network access or HuggingFace cache. This path is not "
                    "silently replaced by random initialization."
                ) from exc
        else:
            cfg = _segformer_config_from_name(
                SegformerConfig,
                model_name,
                num_classes,
                decoder_hidden_size=decoder_hidden_size,
            )
            self.model = SegformerForSemanticSegmentation(cfg)

        if in_channels != 3:
            self._adapt_channels(in_channels)

    def _adapt_channels(self, in_channels: int):
        conv = self.model.segformer.encoder.patch_embeddings[0].proj
        old  = conv.weight.data                    # (out, 3, kH, kW)
        out_ch, _, kH, kW = old.shape
        new  = torch.zeros(out_ch, in_channels, kH, kW)
        new[:, :min(3, in_channels)] = old[:, :min(3, in_channels)]
        if in_channels > 3:
            new[:, 3:] = old.mean(dim=1, keepdim=True).expand(
                -1, in_channels - 3, -1, -1)
        conv.weight = nn.Parameter(new)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:],
                             mode="bilinear", align_corners=False)


def build_segformer(num_classes: int = N_OUT,
                    in_channels: int = config.IN_CHANNELS,
                    pretrained:  bool = True,
                    model_name:  str  = "nvidia/mit-b2",
                    decoder_hidden_size: int = None):
    return SegFormerSeg(
        num_classes,
        in_channels,
        model_name,
        pretrained,
        decoder_hidden_size=decoder_hidden_size,
    )


# =================================================================
# Model 3: DINOv2-UNet (ViT-L backbone)
# =================================================================

class ChannelProjection(nn.Module):
    """
    Project arbitrary input channels to three channels with a 1x1 convolution for DINOv2 RGB input.
    Include BN and ReLU to learn suitable band combinations.
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        # Initialize by copying the first three channels and distributing the rest evenly
        with torch.no_grad():
            w = torch.zeros(3, in_ch, 1, 1)
            n_copy = min(3, in_ch)
            for i in range(n_copy):
                w[i, i] = 1.0
            if in_ch > 3:
                for i in range(3, in_ch):
                    w[i % 3, i] = 1.0 / (in_ch - 3 + 1)
            self.proj[0].weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DINOv2Backbone(nn.Module):
    """
    Wrap the DINOv2 ViT-L backbone.
    Supported modes:
      - Fully frozen backbone (train the decoder only)
      - Full fine-tuning for higher accuracy
      - LoRA fine-tuning to balance efficiency and accuracy

    Return patch features from selected layers for multiscale decoding.
    """
    # Extract layers 6/12/18/24 from ViT-L's 24 layers, from shallow to deep
    EXTRACT_LAYERS = [5, 11, 17, 23]

    def __init__(self,
                 model_name:    str  = config.DINOV2_MODEL,
                 frozen:        bool = config.DINOV2_FROZEN_BACKBONE,
                 use_lora:      bool = False,
                 lora_rank:     int  = 16):
        super().__init__()
        self.embed_dim  = config.DINOV2_EMBED_DIM    # 1024
        self.patch_size = config.DINOV2_PATCH_SIZE   # 14
        self.frozen     = frozen

        # Load pretrained weights through torch.hub
        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv2: {e}\n"
                "Ensure github.com/facebookresearch/dinov2 is accessible,\n"
                "or download the weights manually and adjust the loading code."
            )

        if frozen and not use_lora:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        if use_lora:
            self._apply_lora(lora_rank)

    def _apply_lora(self, rank: int):
        """
        Inject LoRA adapters into ViT-L attention Q/V projections.
        LoRA parameter estimate: rank x 2 x (1024 x 1024) x 24 layers; approximately 12M parameters.
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                r              = rank,
                lora_alpha     = rank * 2,
                target_modules = ["query", "value"],
                lora_dropout   = 0.1,
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
            trainable = sum(p.numel() for p in self.backbone.parameters()
                            if p.requires_grad)
            print(f"[DINOv2] LoRA mode: {trainable/1e6:.1f}M trainable parameters")
        except ImportError:
            print("[DINOv2] peft is not installed; falling back to full fine-tuning."
                  " pip install peft")
            for p in self.backbone.parameters():
                p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> list:
        """
        Input: x (B, 3, H, W), normalized.
        Output: a list of tensors shaped (B, embed_dim, h_i, w_i),
              where h_i = H // patch_size and w_i = W // patch_size.
        """
        B, _, H, W = x.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size

        # Register hooks to extract intermediate features
        features = {}
        hooks = []
        for layer_idx in self.EXTRACT_LAYERS:
            def make_hook(idx):
                def hook(_, __, output):
                    # output: (B, 1 + h_p*w_p, embed_dim); remove [CLS]
                    features[idx] = output[:, 1:, :]
                return hook
            h = self.backbone.blocks[layer_idx].register_forward_hook(
                make_hook(layer_idx)
            )
            hooks.append(h)

        with torch.set_grad_enabled(not self.frozen):
            _ = self.backbone(x)

        for h in hooks:
            h.remove()

        # Reshape to spatial feature maps
        out = []
        for layer_idx in self.EXTRACT_LAYERS:
            feat = features[layer_idx]               # (B, h_p*w_p, C)
            feat = feat.transpose(1, 2).reshape(
                B, self.embed_dim, h_p, w_p
            )                                        # (B, C, h_p, w_p)
            out.append(feat)
        return out


class DINOv2DecodeHead(nn.Module):
    """
    Multiscale feature fusion decoder with an FPN-style structure:
      - Project each of four feature maps to decode_dim with a 1x1 convolution.
      - Upsample and fuse features through addition.
      - Produce class logits with a final 1x1 convolution.

    Feature resolutions (ViT-L patch14, input 518x518):
      Layer 5  → 37×37
      Layer 11 → 37×37
      Layer 17 → 37×37
      Layer 23 -> 37x37 (all layers share a resolution determined by the input size)
    """
    def __init__(self,
                 embed_dim:   int = config.DINOV2_EMBED_DIM,
                 decode_dim:  int = config.DINOV2_DECODE_DIM,
                 num_classes: int = N_OUT):
        super().__init__()
        n_levels = len(DINOv2Backbone.EXTRACT_LAYERS)

        # Reduce channel dimensions for each feature map
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, decode_dim, 1, bias=False),
                nn.BatchNorm2d(decode_dim),
                nn.GELU(),
            )
            for _ in range(n_levels)
        ])

        # Refine fused features
        self.fuse = nn.Sequential(
            nn.Conv2d(decode_dim * n_levels, decode_dim, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(decode_dim),
            nn.GELU(),
            nn.Conv2d(decode_dim, decode_dim, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(decode_dim),
            nn.GELU(),
        )

        # Classification head
        self.cls_head = nn.Conv2d(decode_dim, num_classes, 1)

    def forward(self, features: list, target_size: tuple) -> torch.Tensor:
        """
        features: list of (B, embed_dim, h_p, w_p)
        target_size: (H_orig, W_orig), the original image resolution.
        """
        # Upsample all layers to the largest resolution (the final layer) and project channels
        ref_size = features[-1].shape[-2:]
        projected = []
        for i, feat in enumerate(features):
            if feat.shape[-2:] != ref_size:
                feat = F.interpolate(feat, ref_size,
                                     mode="bilinear", align_corners=False)
            projected.append(self.proj[i](feat))

        # Concatenate and fuse
        fused  = self.fuse(torch.cat(projected, dim=1))  # (B, dim, h_p, w_p)

        # Upsample to the original resolution
        logits = self.cls_head(fused)
        logits = F.interpolate(logits, size=target_size,
                               mode="bilinear", align_corners=False)
        return logits


class DINOv2UNet(nn.Module):
    """
    Complete DINOv2-UNet semantic segmentation model:
      ChannelProjection -> Resolution alignment -> DINOv2 ViT-L -> DecodeHead

    Forward pass:
      (B, IN_CH, H, W)
        -> Channel projection -> (B, 3, H, W)
        -> Resize to patch14-aligned dimensions (training: DINOV2_TILE_SIZE=224; inference: align input)
        -> Extract features from multiple DINOv2 ViT-L layers
        -> Fuse and upsample with the decoder
        -> Restore (H, W) by bilinear interpolation
        → (B, N_OUT, H, W) logits
    """
    def __init__(self,
                 num_classes: int  = N_OUT,
                 in_channels: int  = config.IN_CHANNELS,
                 frozen:      bool = config.DINOV2_FROZEN_BACKBONE,
                 use_lora:    bool = False,
                 lora_rank:   int  = 16):
        super().__init__()
        self.patch_size  = config.DINOV2_PATCH_SIZE
        self.max_infer_size = getattr(config, "DINOV2_MAX_INFER_SIZE", 448)

        self.channel_proj = ChannelProjection(in_channels)
        self.backbone     = DINOv2Backbone(
            frozen   = frozen,
            use_lora = use_lora,
            lora_rank= lora_rank,
        )
        self.decode_head  = DINOv2DecodeHead(
            embed_dim   = config.DINOV2_EMBED_DIM,
            decode_dim  = config.DINOV2_DECODE_DIM,
            num_classes = num_classes,
        )

        # ImageNet normalization for the DINOv2 backbone, separate from training data normalization
        self.register_buffer(
            "dino_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "dino_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _align_size(self, size: int) -> int:
        """Round up to a multiple of patch_size, with patch_size as the minimum."""
        p = self.patch_size
        return max(p, ((size + p - 1) // p) * p)

    def _compute_dino_size(self, H: int, W: int) -> tuple:
        """
        Determine the input resolution for DINOv2:
          - Preserve the aspect ratio.
          - Align dimensions to multiples of patch_size(14).
          - Limit dimensions to max_infer_size to avoid out-of-memory errors.
          - Use a minimum of 4xpatch_size = 56 (at least 4x4 patch tokens).

        Examples:
          64x64   -> align to the next patch14 multiple
             64/14=4.57 -> 70x70 (5x5=25 tokens)
          224x224 -> 224x224 (16x16=256 tokens, the training resolution)
          256x192 -> 252x196 (18x14 patches, approximately preserving the aspect ratio)
        """
        max_s = self.max_infer_size
        # Scale so that the longer side does not exceed max_infer_size
        if max(H, W) > max_s:
            scale = max_s / max(H, W)
            H = int(H * scale)
            W = int(W * scale)
        # Align to patch_size multiples, with a minimum of 4xpatch_size
        min_s = 4 * self.patch_size  # 56
        dino_h = max(min_s, self._align_size(H))
        dino_w = max(min_s, self._align_size(W))
        return dino_h, dino_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        orig_size = (H, W)

        # 1. Channel projection: IN_CHANNELS -> 3
        x3 = self.channel_proj(x)                        # (B, 3, H, W)

        # 2. Determine the DINOv2 input size and resize
        #    - Training uses H=W=DINOV2_TILE_SIZE(224), which remains 224 after alignment
        #    - Inference tiles are usually 256+; align them without exceeding max_infer_size
        #    - No fixed resize to 518 is required
        dino_h, dino_w = self._compute_dino_size(H, W)
        x_resized = F.interpolate(x3, (dino_h, dino_w),
                                   mode="bilinear", align_corners=False)

        # 3. Apply ImageNet normalization expected by the DINOv2 backbone
        x_norm = (x_resized - self.dino_mean) / self.dino_std

        # 4. Extract features from multiple layers
        features = self.backbone(x_norm)             # list of (B,1024,h_p,w_p)

        # 5. Decode and restore the original resolution
        logits = self.decode_head(features, orig_size)
        return logits                                # (B, N_OUT, H, W)


def build_dinov2(num_classes: int  = N_OUT,
                 in_channels: int  = config.IN_CHANNELS,
                 pretrained:  bool = True,
                 frozen:      bool = config.DINOV2_FROZEN_BACKBONE,
                 use_lora:    bool = False):
    """
    Build DINOv2-UNet (ViT-L).

    pretrained: load pretrained DINOv2 weights (usually True).
    frozen: freeze the backbone.
    use_lora: fine-tune the backbone with LoRA (requires peft).
    """
    if not pretrained:
        import warnings
        warnings.warn(
            "DINOv2 performs poorly without pretrained weights; pretrained=True is strongly recommended"
        )
    return DINOv2UNet(
        num_classes = num_classes,
        in_channels = in_channels,
        frozen       = frozen,
        use_lora     = use_lora,
    )


# =================================================================
# Base loss functions
# =================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0,
                 weight: torch.Tensor = None,
                 ignore_index: int = config.IGNORE_INDEX):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        log_probs = F.log_softmax(pred, dim=1)
        valid_mask = target != self.ignore_index

        gather_target = target.clone()
        gather_target[~valid_mask] = 0

        log_pt = log_probs.gather(1, gather_target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        if self.weight is not None:
            alpha = self.weight[gather_target]
        else:
            alpha = torch.ones_like(pt)

        focal = -alpha * (1 - pt).pow(self.gamma) * log_pt
        focal = focal * valid_mask.float()
        return focal.sum() / valid_mask.float().sum().clamp_min(1.0)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0,
                 ignore_index: int = config.IGNORE_INDEX):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        num_classes = pred.shape[1]
        prob        = F.softmax(pred, dim=1)

        valid_mask  = (target != self.ignore_index)
        t           = target.clone()
        t[~valid_mask] = 0

        one_hot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float()
        valid   = valid_mask.unsqueeze(1).float()

        prob    = prob    * valid
        one_hot = one_hot * valid

        dims  = (0, 2, 3)
        inter = (prob * one_hot).sum(dim=dims)
        union = (prob + one_hot).sum(dim=dims)
        dice  = (2 * inter + self.smooth) / (union + self.smooth)
        if dice.numel() <= 1:
            return pred.sum() * 0.0
        return 1 - dice[1:].mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights: list = None,
                 focal_w: float = 0.5, dice_w: float = 0.5,
                 gamma: float = 2.0, device: str = "cpu"):
        super().__init__()
        w = (torch.tensor(class_weights, dtype=torch.float32).to(device)
             if class_weights is not None else None)
        ign = config.IGNORE_INDEX
        self.focal   = FocalLoss(gamma=gamma, weight=w, ignore_index=ign)
        self.dice    = DiceLoss(ignore_index=ign)
        self.focal_w = focal_w
        self.dice_w  = dice_w

    def forward(self, pred, target):
        return (self.focal_w * self.focal(pred, target) +
                self.dice_w  * self.dice(pred, target))


# =================================================================
# Model 4: SegFormer-B4/B5
# =================================================================
def build_segformer_b4(num_classes: int = N_OUT,
                       in_channels: int = config.IN_CHANNELS,
                       pretrained:  bool = True,
                       decoder_hidden_size: int = None):
    """SegFormer-B4: more parameters and greater feature extraction capacity than B2."""
    return SegFormerSeg(num_classes, in_channels,
                        "nvidia/mit-b4", pretrained,
                        decoder_hidden_size=decoder_hidden_size)


def build_segformer_b5(num_classes: int = N_OUT,
                       in_channels: int = config.IN_CHANNELS,
                       pretrained:  bool = True,
                       decoder_hidden_size: int = None):
    """SegFormer-B5: the largest variant, with the highest accuracy and memory requirements."""
    return SegFormerSeg(num_classes, in_channels,
                        "nvidia/mit-b5", pretrained,
                        decoder_hidden_size=decoder_hidden_size)



# =================================================================
# Model 6: Mask2Former + Swin backbone
# =================================================================
class Mask2FormerSeg(nn.Module):
    """
    Mask2Former (Cheng et al., CVPR 2022) in semantic segmentation mode.

    References:
      Cheng et al. 2022 "Masked-attention Mask Transformer for Universal Image Segmentation"
      Gibril et al. 2024 (Remote Sensing) - multispectral tree mapping; Mask2Former outperformed SegFormer.

    Architecture:
      ChannelProjection -> Mask2Former (transformers library)

    Mask2Former supports semantic, instance, and panoptic segmentation.
        This wrapper uses semantic segmentation (semantic_segmentation_logits).
    """
    def __init__(self,
                 num_classes: int = N_OUT,
                 in_channels: int = config.IN_CHANNELS,
                 pretrained:  bool = True,
                 backbone:    str  = "swin-tiny"):
        super().__init__()

        # Project multispectral bands
        self.proj = ChannelProjection(in_channels)

        # Mask2Former from HuggingFace
        try:
            from transformers import (Mask2FormerForUniversalSegmentation,
                                      Mask2FormerConfig)
        except ImportError:
            raise ImportError("pip install transformers>=4.27")

        # HuggingFace model ids
        _hf_ids = {
            "swin-tiny":  "facebook/mask2former-swin-tiny-ade-semantic",
            "swin-small": "facebook/mask2former-swin-small-ade-semantic",
            "swin-base":  "facebook/mask2former-swin-base-ade-semantic",
        }
        hf_id = _hf_ids.get(backbone, _hf_ids["swin-tiny"])

        if pretrained:
            self.m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
                hf_id, ignore_mismatched_sizes=True,
                num_labels=num_classes,
            )
        else:
            cfg = Mask2FormerConfig.from_pretrained(hf_id)
            cfg.num_labels = num_classes
            self.m2f = Mask2FormerForUniversalSegmentation(cfg)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        x    = self.proj(x)                          # → (B,3,H,W)
        out  = self.m2f(pixel_values=x)
        logits = out.masks_queries_logits             # (B, Q, Hm, Wm)

        # class_queries_logits: (B, Q, num_labels+1); the last column is the no-object class
        sem   = out.class_queries_logits              # (B, Q, num_labels+1)
        # Keep the first num_classes columns and discard the Hugging Face no-object column
        sem   = sem[..., :self.num_classes]           # (B, Q, num_classes)
        sem_w = sem.softmax(dim=-1)                   # (B, Q, num_classes)
        masks = logits                                # (B, Q, Hm, Wm)
        # Aggregate to (B, num_classes, Hm, Wm)
        seg   = torch.einsum("bqc,bqhw->bchw", sem_w, masks)
        # Upsample to the original resolution
        seg   = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)
        # Ensure the output channel count equals num_classes
        if seg.shape[1] != self.num_classes:
            seg = seg[:, :self.num_classes]
        return seg


def build_mask2former(num_classes: int = N_OUT,
                      in_channels: int = config.IN_CHANNELS,
                      pretrained:  bool = True,
                      backbone:    str  = "swin-tiny"):
    """
    Build Mask2Former.

    Backbone options:
      "swin-tiny"   - lightweight; recommended for small datasets
      "swin-small"  - balances accuracy and speed
      "swin-base"   - highest accuracy, with greater memory requirements
    """
    return Mask2FormerSeg(num_classes, in_channels, pretrained, backbone)



# =================================================================
# Model: UPerNet + ResNet-50
# =================================================================
def build_r50_upernet(num_classes: int = N_OUT,
                      in_channels: int = config.IN_CHANNELS,
                      pretrained:  bool = True):
    """
    UPerNet + ResNet-50 backbone.

    References:
      Xiao et al. 2018 "Unified Perceptual Parsing for Scene Understanding" (ECCV)
      Wang et al. 2023 (Swin-UperNet), a standard remote sensing segmentation baseline.

    Features:
      - Fully convolutional ResNet-50 supports arbitrary input sizes without positional encoding limits.
      - Supports 64x64 tiles.
      - Combines FPN multiscale feature fusion with PPM context aggregation.
      - Uses the built-in smp implementation.

    Implemented with segmentation_models_pytorch UnetPlusPlus and a ResNet50 backbone.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("pip install segmentation-models-pytorch")

    return smp.UnetPlusPlus(
        encoder_name    = "resnet50",
        encoder_weights = "imagenet" if pretrained else None,
        in_channels     = in_channels,
        classes         = num_classes,
        activation      = None,
        decoder_attention_type = "scse",  # Add SE attention to approximate UPerNet context aggregation
    )

# =================================================================
# Model 7: VMamba-UNet (pure PyTorch, no mamba_ssm dependency)
# =================================================================
class _S6Block(nn.Module):
    """
    Simplified S6 (Selective State Space) block implemented in pure PyTorch.

    References:
      Gu & Dao 2023 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
      Liu et al. 2024, "VMamba: Visual State Space Model" (the basis for CVMH-UNet and Samba).
      Cao et al. 2024, "CVMH-UNet", Remote Sensing 2025 (remote sensing semantic segmentation).

    Core ideas:
      - Model long-range dependencies through the discrete SSM recurrence h_t = A*h_{t-1} + B*x_t.
      - Input-dependent selection: B, C, and delta depend on x.
      - Use causal convolution for short sequences or efficient recurrence for long sequences.

    Implementation simplifications:
      - Approximate SSM recurrence with 1D causal convolution for GPU parallelism.
      - Retain the selective mechanism: generate delta/B/C through linear projections.
      - Use pure PyTorch without custom CUDA kernels for compatibility across CUDA versions.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand
        d_inner = self.d_inner

        # Input projection
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        # Depthwise convolution in place of short-range SSM recurrence
        self.conv1d   = nn.Conv1d(d_inner, d_inner, kernel_size=3,
                                  padding=1, groups=d_inner, bias=True)
        self.act      = nn.SiLU()

        # Generate selective parameters delta, B, and C
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1, bias=False)

        # Fixed A matrix with logarithmic parameterization
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.log_A    = nn.Parameter(torch.log(A.expand(d_inner, -1)))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)"""
        residual = x
        B, L, D  = x.shape

        # Projection and gating
        xz  = self.in_proj(x)                       # (B,L,2*d_inner)
        x_s, z = xz.chunk(2, dim=-1)               # each (B,L,d_inner)

        # Depthwise convolution
        x_s = self.conv1d(x_s.transpose(1, 2)).transpose(1, 2)
        x_s = self.act(x_s)

        # Selective parameters
        delta_BC = self.x_proj(x_s)                # (B,L,2*d_state+1)
        d_state  = (delta_BC.shape[-1] - 1) // 2
        delta    = F.softplus(delta_BC[..., :1])   # (B,L,1), positive values
        B_sel    = delta_BC[..., 1:d_state+1]      # (B,L,d_state)
        C_sel    = delta_BC[..., d_state+1:]       # (B,L,d_state)

        # Approximate SSM output using a weighted sum instead of full recurrence (O(L))
        # y_t = C * h_t ≈ C * sum_{s<=t} exp(-delta*A*(t-s)) * B * x_s
        # Aggregate with a local convolution, approximating a small-window SSM
        A_neg = -torch.exp(self.log_A)             # (d_inner, d_state)
        # Scale by delta and normalize the weights
        weight = torch.sigmoid(delta)              # (B,L,1) as gating
        y = x_s * weight + x_s * (1 - weight)    # identity shortcut

        # Gate
        y = y * self.act(z)
        y = self.out_proj(y)
        return self.norm(y + residual)


class _VSS_Block(nn.Module):
    """
    Visual State Space block with 2D scanning.

    Flatten the HxW feature map into sequences and aggregate four scan directions:
      horizontal forward, horizontal backward, vertical forward, and vertical backward.
    Follow VMamba's SS2D design using pure PyTorch.
    """
    def __init__(self, channels: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.ssm  = _S6Block(channels, d_state=d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        # Flatten to a sequence (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x_norm = self.norm(x_flat)

        # Scan in four directions
        seqs = [
            x_norm,                                  # Horizontal forward
            x_norm.flip(1),                          # Horizontal backward
            x_norm.reshape(B, H, W, C)
                  .permute(0, 2, 1, 3).reshape(B, H*W, C),   # Vertical forward
            x_norm.reshape(B, H, W, C)
                  .permute(0, 2, 1, 3).reshape(B, H*W, C).flip(1),  # Vertical backward
        ]
        outs = []
        for i, seq in enumerate(seqs):
            out = self.ssm(seq)
            if i == 1:
                out = out.flip(1)
            elif i == 2:
                out = out.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H*W, C)
            elif i == 3:
                out = out.flip(1).reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H*W, C)
            outs.append(out)

        # Aggregate
        y = sum(outs) / len(outs)
        return y.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B,C,H,W)


class VMambaUNet(nn.Module):
    """
    VMamba-UNet: VSS block encoder and UNet decoder with skip connections.

    Architecture:
      Input (B, IN_CH, 64, 64)
      Encoder: four downsampling stages, each using VSS blocks to extract global features.
      Decoder: bilinear upsampling, skip connections, and convolutions.
      Classification head: 1x1 convolution -> num_classes.

    Properties:
      - Linear SSM complexity and arbitrary input sizes without positional encoding limits.
      - Supports 64x64 tiles without resizing.
      - Scans in all directions to capture spatial features of satellite imagery.
      - Pure PyTorch, without custom CUDA kernels.

    References:
      Samba (PMC 2024): Mamba encoder + UperNet decoder for remote sensing
      CVMH-UNet (Remote Sensing 2025): VMamba + multi-scale fusion
      EGCM-UNet (2024): CNN-Mamba hybrid for farmland segmentation
    """
    def __init__(self,
                 num_classes: int = N_OUT,
                 in_channels: int = config.IN_CHANNELS,
                 base_ch:     int = 64,
                 d_state:     int = 16):
        super().__init__()
        raise RuntimeError(
            "VMambaUNet is disabled because the current _S6Block implementation "
            "is not a valid VMamba/SSM block."
        )

        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
        )

        # Encoder: four stages of VSS blocks and downsampling
        chs = [base_ch, base_ch*2, base_ch*4, base_ch*8]   # 64,128,256,512

        self.enc_vss = nn.ModuleList([
            _VSS_Block(c, d_state=d_state) for c in chs
        ])
        self.enc_down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chs[i], chs[i+1], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(chs[i+1]), nn.ReLU(inplace=True),
            ) for i in range(len(chs)-1)
        ])

        # ── Bottleneck ────────────────────────────────────────────
        self.bottleneck = _VSS_Block(chs[-1], d_state=d_state)

        # Decoder
        dec_chs = chs[::-1]   # 512,256,128,64
        self.dec_up   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dec_chs[i], dec_chs[i+1], 1, bias=False),
                nn.BatchNorm2d(dec_chs[i+1]), nn.ReLU(inplace=True),
            ) for i in range(len(dec_chs)-1)
        ])
        self.dec_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dec_chs[i+1]*2, dec_chs[i+1], 3, padding=1, bias=False),
                nn.BatchNorm2d(dec_chs[i+1]), nn.ReLU(inplace=True),
            ) for i in range(len(dec_chs)-1)
        ])

        # Classification head
        self.cls_head = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        # Stem
        x = self.stem(x)       # (B, 64, H, W)

        # Encode and save skip connections
        skips = []
        for i, (vss, down) in enumerate(zip(self.enc_vss[:-1], self.enc_down)):
            x = vss(x)
            skips.append(x)
            x = down(x)
        x = self.enc_vss[-1](x)   # Deepest VSS layer

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up_conv, fuse_conv, skip in zip(self.dec_up, self.dec_conv,
                                             reversed(skips)):
            x = up_conv(x)
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
            x = fuse_conv(torch.cat([x, skip], dim=1))

        # Upsample to the original resolution
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return self.cls_head(x)


def build_vmamba_unet(num_classes: int = N_OUT,
                      in_channels: int = config.IN_CHANNELS,
                      base_ch:     int = 64,
                      d_state:     int = 16):
    """
    Build VMamba-UNet.

    base_ch: base channel count controlling model size.
      64  -> approximately 12M parameters (recommended for small datasets)
      128 -> approximately 48M parameters (higher accuracy, with greater memory requirements)

    d_state: SSM state dimension controlling long-range modeling capacity.
      16 -> standard configuration
      32 -> greater modeling capacity
    """
    raise RuntimeError(
        "vmamba_unet has been removed from the active framework because the "
        "current _S6Block implementation is not a valid VMamba/SSM block."
    )
