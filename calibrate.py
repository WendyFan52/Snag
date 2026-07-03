"""
Temperature Scaling 置信度校准 — calibrate.py
=================================================================
在训练完成后运行一次，学习温度参数 T 使模型输出概率校准。

原理：
  uncalibrated: softmax(logits)     → 概率系统性偏高（模型过于自信）
  calibrated:   softmax(logits / T) → 概率更接近真实频率

  T > 1: 使概率分布更平缓（降低过度自信）
  T < 1: 使概率分布更尖锐（通常不需要）
  T = 1: 无变化

优化目标：最小化验证集上的负对数似然（NLL）
  T* = argmin -sum[ y_i * log(softmax(logits_i / T)) ]

评估指标：Expected Calibration Error (ECE)
  将预测概率分成10个区间，计算每个区间内"平均置信度"vs"实际准确率"的差

用法：
  python calibrate.py --model segformer
  python calibrate.py --model all

输出：
  checkpoints_{band}/temperature_{model}.json
  → inference.py 自动读取并应用

=================================================================
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from amp_compat import autocast_context
from dataset import SatelliteDataset, get_val_transforms
from inference import load_model


ALL_MODELS = list(config.DEEP_MODELS)


# =================================================================
# ECE (Expected Calibration Error)
# =================================================================
def compute_ece(probs: np.ndarray, labels: np.ndarray,
                n_bins: int = 10) -> float:
    """
    probs:  (N,) — 最高类别的预测概率（置信度）
    labels: (N,) — 是否预测正确（0 or 1）

    ECE = sum_b |B_b| / N × |acc(B_b) - conf(B_b)|
    完美校准时 ECE = 0
    """
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    ece        = 0.0
    n_total    = len(probs)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask   = (probs > lo) & (probs <= hi)
        if not mask.any():
            continue
        acc_b  = labels[mask].mean()   # actual accuracy in bin
        conf_b = probs[mask].mean()    # mean confidence in bin
        ece   += mask.sum() / n_total * abs(acc_b - conf_b)

    return float(ece)


# =================================================================
# Temperature Scaling
# =================================================================
class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(
            torch.ones(1) * config.TEMPERATURE_INIT)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple:
    """收集验证集的全部 logits 和 labels"""
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        # Dataset returns (image, mask, lci_weight, softlabel, has_soft)
        images, masks, _, _, _ = batch
        images = images.to(device)
        masks  = masks.to(device)

        with autocast_context(device):
            logits = model(images)   # (B, C, H, W)

        # Flatten spatial dims, only keep valid (non-ignore) pixels
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0,2,3,1).reshape(-1, C)
        labels_flat = masks.reshape(-1)

        valid = labels_flat != config.IGNORE_INDEX
        all_logits.append(logits_flat[valid].float().cpu())
        all_labels.append(labels_flat[valid].cpu())

    return torch.cat(all_logits, 0), torch.cat(all_labels, 0)


def calibrate(model_name: str,
              manifest_csv: str = None,
              img_dir: str = None,
              mask_dir: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Calibrating: {model_name.upper()}")
    print(f"{'='*50}")

    target_manifest = manifest_csv or os.path.join(config.SPLIT_DIR, "scale_analysis_manifest.csv")

    # Load model + calibration data
    model  = load_model(model_name, device)
    ds     = SatelliteDataset(img_dir or config.TRAIN_IMG_DIR,
                              mask_dir or config.TRAIN_MASK_DIR,
                              transform=get_val_transforms(),
                              manifest_csv=target_manifest)
    num_workers = 0 if os.name == "nt" else 2
    if num_workers == 0:
        print("[Calibration][info] Windows detected; using DataLoader num_workers=0 to avoid multiprocessing pipe errors.")
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Collecting calibration logits from: {target_manifest or 'all tiles'}")
    logits, labels = collect_logits(model, loader, device)
    logits = logits.float()
    print(f"  {logits.shape[0]:,} valid pixels")

    # ── Before calibration ──────────────────────────────────
    probs_before = F.softmax(logits, dim=-1).numpy()
    valid_probs_before = probs_before[:, 1:]
    pred_cls     = valid_probs_before.argmax(axis=1) + 1
    max_conf     = valid_probs_before.max(axis=1)
    correct      = (pred_cls == labels.numpy()).astype(float)
    ece_before   = compute_ece(max_conf, correct)
    acc_before   = correct.mean()
    print(f"\n  Before calibration:")
    print(f"    Accuracy     : {acc_before:.4f}")
    print(f"    Mean conf    : {max_conf.mean():.4f}")
    print(f"    ECE          : {ece_before:.4f}  (lower = better calibrated)")

    # ── Optimize temperature ─────────────────────────────────
    scaler   = TemperatureScaler().to(device)
    logits_d = logits.float().to(device)
    labels_d = labels.to(device)
    optimizer = torch.optim.LBFGS(
        [scaler.temperature],
        lr=config.TEMPERATURE_LR,
        max_iter=config.TEMPERATURE_MAX_ITER,
    )

    def eval_fn():
        optimizer.zero_grad()
        cal_logits = scaler(logits_d)
        loss = F.cross_entropy(cal_logits, labels_d)
        loss.backward()
        return loss

    print("\n  Optimizing temperature...")
    optimizer.step(eval_fn)
    T = float(scaler.temperature.item())
    print(f"  Optimal temperature T = {T:.4f}")

    # ── After calibration ────────────────────────────────────
    with torch.no_grad():
        probs_after = F.softmax(scaler(logits_d), dim=-1).cpu().numpy()
    max_conf_after = probs_after[:, 1:].max(axis=1)
    ece_after      = compute_ece(max_conf_after, correct)

    print(f"\n  After calibration:")
    print(f"    Accuracy     : {acc_before:.4f}  (unchanged)")
    print(f"    Mean conf    : {max_conf_after.mean():.4f}  "
          f"({'decreased' if max_conf_after.mean() < max_conf.mean() else 'increased'})")
    print(f"    ECE          : {ece_after:.4f}  "
          f"({'improved' if ece_after < ece_before else 'worse'} "
          f"from {ece_before:.4f})")

    # ── Per-class calibration report ─────────────────────────
    print(f"\n  Per-class calibration (Class1=Dead trees):")
    for cls in range(1, config.NUM_CLASSES + 1):
        name    = config.CLASS_NAMES.get(cls, str(cls))
        cls_msk = labels.numpy() == cls
        if cls_msk.sum() == 0:
            continue
        c_before = max_conf[cls_msk].mean()
        c_after  = max_conf_after[cls_msk].mean()
        print(f"    [{cls}] {name:<22} conf: {c_before:.3f} → {c_after:.3f}")

    # ── Save temperature ─────────────────────────────────────
    out_path = os.path.join(config.CHECKPOINT_DIR,
                             f"temperature-{model_name}.json")
    result = {
        "model_name"   : model_name,
        "temperature"  : T,
        "ece_before"   : ece_before,
        "ece_after"    : ece_after,
        "accuracy"     : acc_before,
        "band_mode"    : config.BAND_MODE,
        "manifest_csv" : target_manifest,
    }
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Temperature saved -> {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ALL_MODELS + ["all"], default="all")
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default=os.path.join(config.SPLIT_DIR, "scale_analysis_manifest.csv"),
        help="Calibration split manifest. Defaults to the analysis split.",
    )
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    args = parser.parse_args()

    config.ensure_runtime_dirs()
    config.print_config_summary()

    models = ALL_MODELS if args.model == "all" else [args.model]

    results = {}
    for m in models:
        results[m] = calibrate(
            m,
            manifest_csv=args.manifest_csv,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
        )

    if len(results) > 1:
        print(f"\n{'='*50}")
        print("  Calibration Summary")
        print(f"{'='*50}")
        print(f"  {'Model':<15} {'T':>8} {'ECE before':>12} {'ECE after':>12}")
        print(f"  {'-'*50}")
        for m, r in results.items():
            print(f"  {m:<15} {r['temperature']:>8.4f} "
                  f"{r['ece_before']:>12.4f} {r['ece_after']:>12.4f}")
