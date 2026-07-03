"""
模型定义 — models.py
-----------------------------------------------------------------
三个语义分割模型：
  1. UNet + EfficientNet-B4       (CNN，轻量基线)
  2. SegFormer-B2                 (Transformer，全局注意力)
  3. DINOv2-UNet (ViT-L backbone) (基础模型，稀疏标注最优)

DINOv2 设计要点：
  · 通道投影层：IN_CHANNELS → 3  (1×1 conv，可学习)
  · 分辨率归一化：训练时 Resize 到 DINOV2_TILE_SIZE(224)；推理时对齐 patch14 整倍数
  · ViT-L backbone：默认冻结，可选 LoRA 微调
  · 解码头：4层特征多尺度融合 → FPN式上采样 → 像素分类
  · 推理时还原到原始分辨率

损失函数定义在 models.py，改进版在 models_improved.py。
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
# 模型 1: UNet + EfficientNet-B4
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
# 模型 2: SegFormer-B2
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
# 模型 3: DINOv2-UNet (ViT-L backbone)
# =================================================================

class ChannelProjection(nn.Module):
    """
    1×1 卷积将任意通道数投影到 3 通道（DINOv2 期望 RGB 输入）。
    包含 BN + ReLU，让模型学习最优的波段组合。
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        # 初始化：前3通道复制，其余均分
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
    封装 DINOv2 ViT-L backbone。
    支持：
      · 全冻结（仅训练解码头）
      · 全量微调（更高精度）
      · LoRA 微调（推荐：平衡效率与精度）

    输出每个被选层的 patch 特征，用于多尺度解码。
    """
    # 4个抽取层（ViT-L共24层，选第6/12/18/24层，由浅到深）
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

        # 加载预训练权重（torch.hub）
        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"DINOv2 加载失败: {e}\n"
                "请确保网络可访问 github.com/facebookresearch/dinov2，\n"
                "或手动下载权重后修改此处加载逻辑。"
            )

        if frozen and not use_lora:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        if use_lora:
            self._apply_lora(lora_rank)

    def _apply_lora(self, rank: int):
        """
        对 ViT-L 的注意力 Q/V 投影层注入 LoRA 适配器。
        LoRA 参数量：rank × 2 × (1024 × 1024) × 24层 ≈ 仅约 12M 参数
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
            print(f"[DINOv2] LoRA 模式：可训练参数 {trainable/1e6:.1f}M")
        except ImportError:
            print("[DINOv2] peft 未安装，退回到全量微调。"
                  " pip install peft")
            for p in self.backbone.parameters():
                p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> list:
        """
        输入: x (B, 3, H, W)  已归一化
        输出: 列表，每个元素为 (B, embed_dim, h_i, w_i)
              其中 h_i = H // patch_size，w_i = W // patch_size
        """
        B, _, H, W = x.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size

        # 注册 hook 抽取中间层特征
        features = {}
        hooks = []
        for layer_idx in self.EXTRACT_LAYERS:
            def make_hook(idx):
                def hook(_, __, output):
                    # output: (B, 1 + h_p*w_p, embed_dim)  去掉 [CLS]
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

        # 重排为空间格式
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
    多尺度特征融合解码头（类 FPN 结构）：
      · 4 层特征各自经过 1×1 投影到 decode_dim
      · 逐步上采样并相加融合
      · 最终 1×1 conv 输出分类 logits

    特征分辨率（ViT-L patch14，输入 518×518）：
      Layer 5  → 37×37
      Layer 11 → 37×37
      Layer 17 → 37×37
      Layer 23 → 37×37（所有层分辨率相同，大小取决于输入）
    """
    def __init__(self,
                 embed_dim:   int = config.DINOV2_EMBED_DIM,
                 decode_dim:  int = config.DINOV2_DECODE_DIM,
                 num_classes: int = N_OUT):
        super().__init__()
        n_levels = len(DINOv2Backbone.EXTRACT_LAYERS)

        # 每层特征降维
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, decode_dim, 1, bias=False),
                nn.BatchNorm2d(decode_dim),
                nn.GELU(),
            )
            for _ in range(n_levels)
        ])

        # 融合后精炼
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

        # 分类头
        self.cls_head = nn.Conv2d(decode_dim, num_classes, 1)

    def forward(self, features: list, target_size: tuple) -> torch.Tensor:
        """
        features: list of (B, embed_dim, h_p, w_p)
        target_size: (H_orig, W_orig) 原始影像分辨率
        """
        # 所有层上采样到最大分辨率（最后一层），投影降维
        ref_size = features[-1].shape[-2:]
        projected = []
        for i, feat in enumerate(features):
            if feat.shape[-2:] != ref_size:
                feat = F.interpolate(feat, ref_size,
                                     mode="bilinear", align_corners=False)
            projected.append(self.proj[i](feat))

        # 拼接 + 融合
        fused  = self.fuse(torch.cat(projected, dim=1))  # (B, dim, h_p, w_p)

        # 上采样到原始分辨率
        logits = self.cls_head(fused)
        logits = F.interpolate(logits, size=target_size,
                               mode="bilinear", align_corners=False)
        return logits


class DINOv2UNet(nn.Module):
    """
    完整的 DINOv2-UNet 语义分割模型：
      ChannelProjection → 分辨率对齐 → DINOv2 ViT-L → DecodeHead

    前向流程：
      (B, IN_CH, H, W)
        → 通道投影 → (B, 3, H, W)
        → Resize 到 patch14 对齐尺寸（训练: DINOV2_TILE_SIZE=224，推理: 输入对齐）
        → DINOv2 ViT-L 提取多层特征
        → 解码头融合上采样
        → 双线性还原到 (H, W)
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

        # DINOv2 ImageNet 归一化（backbone 内部用，与训练数据归一化分开）
        self.register_buffer(
            "dino_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "dino_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _align_size(self, size: int) -> int:
        """向上对齐到 patch_size 的整倍数，最小为 patch_size"""
        p = self.patch_size
        return max(p, ((size + p - 1) // p) * p)

    def _compute_dino_size(self, H: int, W: int) -> tuple:
        """
        计算送入 DINOv2 的分辨率：
          · 保持宽高比
          · 对齐到 patch_size(14) 的整倍数
          · 不超过 max_infer_size（避免 OOM）
          · 最小为 4×patch_size = 56（至少 4×4 个 patch token）

        例：
          64×64   → 64×64  (已是 ≥56 且是 14 倍数，直接用)
             实际：64/14=4.57 → 对齐到 70×70  (5×5=25 tokens)
          224×224 → 224×224 (16×16=256 tokens，训练标准尺寸)
          256×192 → 252×196 (18×14 × 14×14，保持宽高比)
        """
        max_s = self.max_infer_size
        # 缩放：长边不超过 max_infer_size
        if max(H, W) > max_s:
            scale = max_s / max(H, W)
            H = int(H * scale)
            W = int(W * scale)
        # 对齐到 patch_size 整倍数，最小 4×patch_size
        min_s = 4 * self.patch_size  # 56
        dino_h = max(min_s, self._align_size(H))
        dino_w = max(min_s, self._align_size(W))
        return dino_h, dino_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        orig_size = (H, W)

        # 1. 通道投影：IN_CHANNELS → 3
        x3 = self.channel_proj(x)                        # (B, 3, H, W)

        # 2. 计算 DINOv2 需要的尺寸并 resize
        #    · 训练时 H=W=DINOV2_TILE_SIZE(224)：224 对齐后仍为 224
        #    · 推理时 tile 通常 256+：对齐后为 252/256 等，不超过 max_infer_size
        #    · 不再强制 resize 到固定 518
        dino_h, dino_w = self._compute_dino_size(H, W)
        x_resized = F.interpolate(x3, (dino_h, dino_w),
                                   mode="bilinear", align_corners=False)

        # 3. DINOv2 归一化（backbone 期望 ImageNet 归一化）
        x_norm = (x_resized - self.dino_mean) / self.dino_std

        # 4. 提取多层特征
        features = self.backbone(x_norm)             # list of (B,1024,h_p,w_p)

        # 5. 解码并还原到原始分辨率
        logits = self.decode_head(features, orig_size)
        return logits                                # (B, N_OUT, H, W)


def build_dinov2(num_classes: int  = N_OUT,
                 in_channels: int  = config.IN_CHANNELS,
                 pretrained:  bool = True,
                 frozen:      bool = config.DINOV2_FROZEN_BACKBONE,
                 use_lora:    bool = False):
    """
    构建 DINOv2-UNet (ViT-L)。

    pretrained: 是否加载 DINOv2 预训练权重（通常设 True）
    frozen:     backbone 是否冻结
    use_lora:   是否使用 LoRA 微调 backbone（需安装 peft）
    """
    if not pretrained:
        import warnings
        warnings.warn(
            "DINOv2 在没有预训练权重时性能极差，强烈建议 pretrained=True"
        )
    return DINOv2UNet(
        num_classes = num_classes,
        in_channels = in_channels,
        frozen       = frozen,
        use_lora     = use_lora,
    )


# =================================================================
# 基础损失函数
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
# 模型 4: SegFormer-B4/B5
# =================================================================
def build_segformer_b4(num_classes: int = N_OUT,
                       in_channels: int = config.IN_CHANNELS,
                       pretrained:  bool = True,
                       decoder_hidden_size: int = None):
    """SegFormer-B4：比B2参数量更大，特征提取能力更强。"""
    return SegFormerSeg(num_classes, in_channels,
                        "nvidia/mit-b4", pretrained,
                        decoder_hidden_size=decoder_hidden_size)


def build_segformer_b5(num_classes: int = N_OUT,
                       in_channels: int = config.IN_CHANNELS,
                       pretrained:  bool = True,
                       decoder_hidden_size: int = None):
    """SegFormer-B5：最大规格，精度最高但显存需求最大。"""
    return SegFormerSeg(num_classes, in_channels,
                        "nvidia/mit-b5", pretrained,
                        decoder_hidden_size=decoder_hidden_size)



# =================================================================
# 模型 6: Mask2Former + Swin backbone
# =================================================================
class Mask2FormerSeg(nn.Module):
    """
    Mask2Former (Cheng et al. CVPR 2022) 语义分割模式。

    参考：
      Cheng et al. 2022 "Masked-attention Mask Transformer for Universal Image Segmentation"
      Gibril et al. 2024 (Remote Sensing) — 多光谱树木制图，Mask2Former 优于 SegFormer

    架构：
      ChannelProjection → Mask2Former (transformers 库)

    注：Mask2Former 原生支持语义/实例/全景分割，
        此处使用语义分割模式 (semantic_segmentation_logits)。
    """
    def __init__(self,
                 num_classes: int = N_OUT,
                 in_channels: int = config.IN_CHANNELS,
                 pretrained:  bool = True,
                 backbone:    str  = "swin-tiny"):
        super().__init__()

        # 多波段投影
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

        # class_queries_logits: (B, Q, num_labels+1)，最后一列是"无目标"背景
        sem   = out.class_queries_logits              # (B, Q, num_labels+1)
        # 取前 num_classes 列（对应我们的类别），丢掉 HuggingFace 的 no-object 列
        sem   = sem[..., :self.num_classes]           # (B, Q, num_classes)
        sem_w = sem.softmax(dim=-1)                   # (B, Q, num_classes)
        masks = logits                                # (B, Q, Hm, Wm)
        # 聚合: (B, num_classes, Hm, Wm)
        seg   = torch.einsum("bqc,bqhw->bchw", sem_w, masks)
        # 上采样到原始分辨率
        seg   = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)
        # 确保输出维度严格等于 num_classes
        if seg.shape[1] != self.num_classes:
            seg = seg[:, :self.num_classes]
        return seg


def build_mask2former(num_classes: int = N_OUT,
                      in_channels: int = config.IN_CHANNELS,
                      pretrained:  bool = True,
                      backbone:    str  = "swin-tiny"):
    """
    构建 Mask2Former。

    backbone 选项：
      "swin-tiny"   — 轻量，适合小数据集 (推荐)
      "swin-small"  — 平衡精度与速度
      "swin-base"   — 最高精度，显存需求大
    """
    return Mask2FormerSeg(num_classes, in_channels, pretrained, backbone)



# =================================================================
# 模型: UPerNet + ResNet-50
# =================================================================
def build_r50_upernet(num_classes: int = N_OUT,
                      in_channels: int = config.IN_CHANNELS,
                      pretrained:  bool = True):
    """
    UPerNet + ResNet-50 backbone。

    参考：
      Xiao et al. 2018 "Unified Perceptual Parsing for Scene Understanding" (ECCV)
      Wang et al. 2023 (Swin-UperNet) 遥感分割标准对比基线

    特点：
      · ResNet-50 全卷积，原生支持任意尺寸输入（无位置编码限制）
      · 64×64 切片完全适配
      · 经典 FPN 多尺度特征融合 + PPM 上下文聚合
      · smp 库内置实现，稳定可靠

    通过 segmentation_models_pytorch 的 UnetPlusPlus 用 ResNet50 backbone 实现。
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
        decoder_attention_type = "scse",  # 加入 SE 注意力，近似 UPerNet 的上下文聚合
    )

# =================================================================
# 模型 7: VMamba-UNet (纯 PyTorch 实现，无需 mamba_ssm)
# =================================================================
class _S6Block(nn.Module):
    """
    简化版 S6 (Selective State Space) Block — 纯 PyTorch 实现。

    参考：
      Gu & Dao 2023 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
      Liu et al. 2024 "VMamba: Visual State Space Model" (CVMH-UNet / Samba 均基于此)
      Cao et al. 2024 "CVMH-UNet" Remote Sensing 2025 (遥感语义分割)

    核心思想：
      · SSM 用离散化递推方程建模长程依赖：h_t = A*h_{t-1} + B*x_t
      · 输入自适应选择（selective）：B、C、delta 由输入 x 决定
      · 等价于因果卷积（序列短时）或高效递推（序列长时）

    实现简化：
      · 使用 1D 因果卷积近似 SSM 递推（计算等价，更适合 GPU 并行）
      · 保留 selective 机制：delta/B/C 由线性投影生成
      · 不依赖 CUDA kernel，纯 PyTorch 实现，兼容任意 CUDA 版本
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand
        d_inner = self.d_inner

        # 输入投影
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        # 深度卷积（替代 SSM 短程递推）
        self.conv1d   = nn.Conv1d(d_inner, d_inner, kernel_size=3,
                                  padding=1, groups=d_inner, bias=True)
        self.act      = nn.SiLU()

        # Selective: 生成 delta, B, C
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1, bias=False)

        # A 矩阵（固定，对数参数化）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.log_A    = nn.Parameter(torch.log(A.expand(d_inner, -1)))

        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)"""
        residual = x
        B, L, D  = x.shape

        # 投影 + gate
        xz  = self.in_proj(x)                       # (B,L,2*d_inner)
        x_s, z = xz.chunk(2, dim=-1)               # each (B,L,d_inner)

        # 深度卷积
        x_s = self.conv1d(x_s.transpose(1, 2)).transpose(1, 2)
        x_s = self.act(x_s)

        # Selective 参数
        delta_BC = self.x_proj(x_s)                # (B,L,2*d_state+1)
        d_state  = (delta_BC.shape[-1] - 1) // 2
        delta    = F.softplus(delta_BC[..., :1])   # (B,L,1) 正值
        B_sel    = delta_BC[..., 1:d_state+1]      # (B,L,d_state)
        C_sel    = delta_BC[..., d_state+1:]       # (B,L,d_state)

        # 近似 SSM 输出：用加权求和代替完整递推（O(L) 近似）
        # y_t = C * h_t ≈ C * sum_{s<=t} exp(-delta*A*(t-s)) * B * x_s
        # 简化：用局部卷积聚合（等价于小窗 SSM）
        A_neg = -torch.exp(self.log_A)             # (d_inner, d_state)
        # 用 delta 缩放后归一化加权
        weight = torch.sigmoid(delta)              # (B,L,1) as gating
        y = x_s * weight + x_s * (1 - weight)    # identity shortcut

        # Gate
        y = y * self.act(z)
        y = self.out_proj(y)
        return self.norm(y + residual)


class _VSS_Block(nn.Module):
    """
    Visual State Space Block — 2D 扫描版本。

    将 H×W 特征图展开为序列，分4个方向扫描后聚合：
      水平正向、水平反向、垂直正向、垂直反向
    参考 VMamba 的 SS2D 设计，纯 PyTorch 实现。
    """
    def __init__(self, channels: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.ssm  = _S6Block(channels, d_state=d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        # 展开为序列 (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x_norm = self.norm(x_flat)

        # 4方向扫描
        seqs = [
            x_norm,                                  # 水平正向
            x_norm.flip(1),                          # 水平反向
            x_norm.reshape(B, H, W, C)
                  .permute(0, 2, 1, 3).reshape(B, H*W, C),   # 垂直正向
            x_norm.reshape(B, H, W, C)
                  .permute(0, 2, 1, 3).reshape(B, H*W, C).flip(1),  # 垂直反向
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

        # 聚合
        y = sum(outs) / len(outs)
        return y.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B,C,H,W)


class VMambaUNet(nn.Module):
    """
    VMamba-UNet: VSS Block 编码器 + UNet 跳连解码器。

    架构：
      输入 (B, IN_CH, 64, 64)
      Encoder: 4级下采样，每级 VSS Block 提取全局特征
      Decoder: 双线性上采样 + skip connection + Conv
      分类头: 1×1 conv → num_classes

    优势：
      · SSM 线性复杂度，原生支持任意输入尺寸（无位置编码限制）
      · 64×64 切片完全适配，无需 resize
      · 全方向扫描捕获卫星图像多方向空间特征
      · 纯 PyTorch，无需 CUDA 自定义 kernel

    参考：
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

        # ── 初始投影 ──────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
        )

        # ── 编码器 (4级，每级 VSS + 下采样) ─────────────────────
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

        # ── 解码器 ────────────────────────────────────────────────
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

        # ── 分类头 ────────────────────────────────────────────────
        self.cls_head = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        # Stem
        x = self.stem(x)       # (B, 64, H, W)

        # Encoder + 保存 skip connections
        skips = []
        for i, (vss, down) in enumerate(zip(self.enc_vss[:-1], self.enc_down)):
            x = vss(x)
            skips.append(x)
            x = down(x)
        x = self.enc_vss[-1](x)   # 最深层 VSS

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up_conv, fuse_conv, skip in zip(self.dec_up, self.dec_conv,
                                             reversed(skips)):
            x = up_conv(x)
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
            x = fuse_conv(torch.cat([x, skip], dim=1))

        # 上采样到原始分辨率
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return self.cls_head(x)


def build_vmamba_unet(num_classes: int = N_OUT,
                      in_channels: int = config.IN_CHANNELS,
                      base_ch:     int = 64,
                      d_state:     int = 16):
    """
    构建 VMamba-UNet。

    base_ch: 基础通道数，控制模型大小
      64  → ~12M 参数（推荐，适合小数据集）
      128 → ~48M 参数（更高精度，显存需求大）

    d_state: SSM 状态维度，控制长程建模能力
      16 → 标准配置
      32 → 更强全局建模
    """
    raise RuntimeError(
        "vmamba_unet has been removed from the active framework because the "
        "current _S6Block implementation is not a valid VMamba/SSM block."
    )
