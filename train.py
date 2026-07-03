"""
Training script.

Supported models:
  python train.py --model unet
  python train.py --model unet_resnet34
  python train.py --model unet_convnext_tiny
  python train.py --model hrnet_w18
  python train.py --model segformer
  python train.py --model segformer_b4
  python train.py --model segformer_b5
  python train.py --model r50_upernet
  python train.py --model all
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from amp_compat import autocast_context, make_grad_scaler
from dataset import build_loaders
from models import (
    build_unet,
    build_unet_resnet34,
    build_unet_convnext_tiny,
    build_hrnet_w18,
    build_segformer,
    build_segformer_b4,
    build_segformer_b5,
    build_r50_upernet,
)
from models_improved import ImprovedLoss, compute_dynamic_weights_from_paths


MODEL_BUILDERS = {
    "unet": build_unet,
    "unet_resnet34": build_unet_resnet34,
    "unet_convnext_tiny": build_unet_convnext_tiny,
    "hrnet_w18": build_hrnet_w18,
    "segformer": build_segformer,
    "segformer_b4": build_segformer_b4,
    "segformer_b5": build_segformer_b5,
    "r50_upernet": build_r50_upernet,
}

ALL_MODELS = list(config.DEEP_MODELS)


def compute_metrics(preds: np.ndarray, targets: np.ndarray):
    num_classes = config.NUM_CLASSES + 1
    valid_mask = targets != config.IGNORE_INDEX
    ious = []

    for cls in range(1, num_classes):
        pred_cls = (preds == cls) & valid_mask
        true_cls = (targets == cls) & valid_mask
        inter = (pred_cls & true_cls).sum()
        union = (pred_cls | true_cls).sum()
        ious.append(float(inter / union) if union > 0 else float("nan"))

    valid_ious = [v for v in ious if not np.isnan(v)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    corr = sum(
        ((preds == c) & (targets == c) & valid_mask).sum()
        for c in range(1, num_classes)
    )
    oa = float(corr) / max(int(valid_mask.sum()), 1)
    return oa, miou, ious


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    losses = []

    for images, masks, lci_w, soft, soft_valid in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        lci_w = lci_w.to(device, non_blocking=True)
        soft = soft.to(device, non_blocking=True)
        soft_valid = soft_valid.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast_context(device):
            logits = model(images)
            loss = criterion(
                logits,
                masks,
                lci_weight=lci_w,
                soft_targets=soft,
                soft_valid_mask=soft_valid,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses, all_p, all_t = [], [], []

    for images, masks, lci_w, soft, soft_valid in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        lci_w = lci_w.to(device, non_blocking=True)
        soft = soft.to(device, non_blocking=True)
        soft_valid = soft_valid.to(device, non_blocking=True)
        with autocast_context(device):
            logits = model(images)
            loss = criterion(
                logits,
                masks,
                lci_weight=lci_w,
                soft_targets=soft,
                soft_valid_mask=soft_valid,
            )
        losses.append(loss.item())
        all_p.append(logits[:, 1:].argmax(dim=1).cpu().numpy() + 1)
        all_t.append(masks.cpu().numpy())

    preds = np.concatenate(all_p, axis=0)
    targets = np.concatenate(all_t, axis=0)
    oa, miou, cls_iou = compute_metrics(preds, targets)
    return float(np.mean(losses)), oa, miou, cls_iou


def build_model(model_name: str, num_classes: int, in_channels: int, pretrained: bool = True):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_BUILDERS[model_name](
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
    )


def build_optimizer(model):
    return optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)


def train(
    model_name: str,
    resume: bool = False,
    resume_path: str = None,
    manifest_csv: str = None,
    img_dir: str = None,
    mask_dir: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 65}")
    print(f"  Model: {model_name.upper():<20} Device: {device}")
    print(
        f"  IN_CHANNELS={config.IN_CHANNELS} "
        f"(raw {config.RAW_CHANNELS} + indices {config.IN_CHANNELS - config.RAW_CHANNELS})"
    )
    print(f"{'=' * 65}\n")

    train_loader, val_loader, train_mask_paths = build_loaders(
        img_dir=img_dir or config.TRAIN_IMG_DIR,
        mask_dir=mask_dir or config.TRAIN_MASK_DIR,
        manifest_csv=manifest_csv,
    )
    num_classes = config.NUM_CLASSES + 1
    model = build_model(model_name, num_classes=num_classes, in_channels=config.IN_CHANNELS)
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Params total: {total:.1f}M  trainable: {trainable:.1f}M\n")

    dyn_weights = compute_dynamic_weights_from_paths(
        train_mask_paths, config.NUM_CLASSES, config.IGNORE_INDEX
    )
    criterion = ImprovedLoss(class_weights=[0.0] + dyn_weights, device=str(device))
    optimizer = build_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    scaler = make_grad_scaler(device)

    best_miou = 0.0
    start_epoch = 1
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"best_{model_name}.pth")
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_oa": []}

    load_path = resume_path or ckpt_path
    if resume and os.path.exists(load_path):
        ckpt = torch.load(load_path, map_location=device)
        saved_num_cls = ckpt.get("num_classes", config.NUM_CLASSES)
        if saved_num_cls != config.NUM_CLASSES:
            state = {
                k: v
                for k, v in ckpt["state_dict"].items()
                if not any(
                    head_key in k
                    for head_key in [
                        "cls_head",
                        "segmentation_head",
                        "decode_head.cls_head",
                        "classifier",
                        "linear_c4",
                    ]
                )
            }
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(ckpt["state_dict"])

        best_miou = ckpt.get("miou", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resume from {load_path}")
        print(f"  Completed epoch={start_epoch - 1}, best mIoU={best_miou:.4f}")
    elif resume:
        print(f"  Checkpoint not found: {load_path}. Start from scratch.")

    for epoch in range(start_epoch, start_epoch + config.NUM_EPOCHS):
        t0 = time.time()
        tl = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        vl, oa, miou, cls_iou = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_miou"].append(miou)
        history["val_oa"].append(oa)

        total_ep = start_epoch + config.NUM_EPOCHS - 1
        print(
            f"Epoch {epoch:03d}/{total_ep} | "
            f"TrLoss={tl:.4f} | VaLoss={vl:.4f} | "
            f"mIoU={miou:.4f} | OA={oa:.4f} | {time.time() - t0:.1f}s"
        )

        if epoch % 10 == 0:
            for i, iou in enumerate(cls_iou):
                cls = i + 1
                name = config.CLASS_NAMES.get(cls, f"cls{cls}")
                tag = (
                    f"  [{cls}] {name:<22} IoU={iou:.4f}"
                    if not np.isnan(iou)
                    else f"  [{cls}] {name:<22} IoU=N/A"
                )
                print(tag)

        if miou > best_miou:
            best_miou = miou
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "miou": miou,
                    "in_channels": config.IN_CHANNELS,
                    "raw_channels": config.RAW_CHANNELS,
                    "num_classes": config.NUM_CLASSES,
                    "band_mode": config.BAND_MODE,
                },
                ckpt_path,
            )
            print(f"  Best checkpoint mIoU={miou:.4f} -> {ckpt_path}")

    print(f"\n[Done] {model_name.upper()} best mIoU={best_miou:.4f}")
    save_history(history, model_name)
    return history


def save_history(history: dict, model_name: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        matplotlib.rcParams["axes.unicode_minus"] = False

        ep = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(ep, history["train_loss"], label="Train Loss")
        ax1.plot(ep, history["val_loss"], label="Val Loss")
        ax1.set_title(f"{model_name.upper()} Loss")
        ax1.legend()
        ax1.set_xlabel("Epoch")

        ax2.plot(ep, history["val_miou"], label="mIoU", color="green")
        ax2.plot(ep, history["val_oa"], label="OA", color="blue", linestyle="--")
        ax2.set_title(f"{model_name.upper()} Validation")
        ax2.legend()
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Epoch")

        plt.tight_layout()
        out = os.path.join(config.CHECKPOINT_DIR, f"history-{model_name}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Training curve -> {out}")
    except Exception as exc:
        print(f"  Failed to save training curve: {exc}")


def precheck_models(models_to_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_cls = config.NUM_CLASSES + 1
    dummy = torch.randn(2, config.IN_CHANNELS, config.TILE_SIZE, config.TILE_SIZE).to(device)
    passed = []

    print("\n" + "=" * 55)
    print("  Model precheck (forward pass with dummy input)")
    print("  pretrained=False for interface check; training still uses pretrained=True")
    print("=" * 55)

    for model_name in models_to_train:
        try:
            mdl = build_model(
                model_name,
                num_classes=n_cls,
                in_channels=config.IN_CHANNELS,
                pretrained=False,
            )
            mdl = mdl.to(device).eval()
            with torch.no_grad():
                out = mdl(dummy)
            expected = (2, n_cls, config.TILE_SIZE, config.TILE_SIZE)
            if out.shape != torch.Size(expected):
                print(f"  FAIL {model_name:<20} output {tuple(out.shape)} != {expected}")
            else:
                print(f"  OK   {model_name:<20} output {tuple(out.shape)}")
                passed.append(model_name)
            del mdl
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  ERROR {model_name:<18} {exc}")

    print("=" * 55)
    print(f"  Passed: {passed}")
    skipped = [m for m in models_to_train if m not in passed]
    print(f"  Skipped: {skipped}\n")
    if skipped:
        raise RuntimeError(
            "Model precheck failed; refusing to continue with a partial model set: "
            + ", ".join(skipped)
        )
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ALL_MODELS + ["all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default=None,
        help="Optional sample-level manifest filter for stage-3 training",
    )
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    args = parser.parse_args()

    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
        print(f"[Override] NUM_EPOCHS = {config.NUM_EPOCHS}")

    config.ensure_runtime_dirs()
    config.print_config_summary()

    models_to_train = ALL_MODELS if args.model == "all" else [args.model]
    models_to_train = precheck_models(models_to_train)

    histories = {}
    for model_name in models_to_train:
        histories[model_name] = train(
            model_name,
            resume=args.resume,
            resume_path=args.resume_path,
            manifest_csv=args.manifest_csv,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
        )

    if len(histories) > 1:
        print("\n" + "=" * 55)
        print("  Model comparison summary")
        print("=" * 55)
        for model_name, history in histories.items():
            print(f"  {model_name:<20} best mIoU: {max(history['val_miou']):.4f}")
