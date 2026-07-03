"""
Evaluation script (stage-3 ready).

Features:
- per-class IoU / F1 / Precision / Recall
- dead-class focused summary for model ranking
- manifest-based dataset filtering
- CSV export for experiment matrix integration
"""

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from amp_compat import autocast_context
from dataset import SatelliteDataset, get_val_transforms
from inference import ALL_MODELS, load_model
from random_forest_utils import (
    is_random_forest_model,
    load_random_forest,
    predict_proba_for_normalized_nchw,
)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


@torch.no_grad()
def evaluate_model(
    model_name: str,
    device: torch.device,
    manifest_csv: str = None,
    img_dir: str = None,
    mask_dir: str = None,
    batch_size: int = 8,
    num_workers: int = 2,
) -> np.ndarray:
    num_classes = config.NUM_CLASSES + 1
    dataset = SatelliteDataset(
        img_dir or config.TRAIN_IMG_DIR,
        mask_dir or config.TRAIN_MASK_DIR,
        transform=get_val_transforms(),
        manifest_csv=manifest_csv,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    if is_random_forest_model(model_name):
        rf_bundle = load_random_forest()
        for images, masks, _, _, _ in loader:
            probs = predict_proba_for_normalized_nchw(rf_bundle, images.cpu().numpy())
            tgts = masks.cpu().numpy()
            preds = probs[:, 1:].argmax(axis=1).astype(np.int64) + 1
            preds[tgts == config.IGNORE_INDEX] = config.IGNORE_INDEX
            preds = preds.ravel()
            tgts = tgts.ravel()

            valid = tgts != config.IGNORE_INDEX
            for tgt, pred in zip(tgts[valid], preds[valid]):
                if 0 <= tgt < num_classes and 0 <= pred < num_classes:
                    conf[tgt, pred] += 1
        return conf

    model = load_model(model_name, device)
    for images, masks, _, _, _ in loader:
        images = images.to(device)
        with autocast_context(device):
            logits = model(images)
        tgts = masks.cpu().numpy()
        preds = logits[:, 1:].argmax(dim=1).cpu().numpy() + 1
        preds[tgts == config.IGNORE_INDEX] = config.IGNORE_INDEX
        preds = preds.ravel()
        tgts = tgts.ravel()

        valid = tgts != config.IGNORE_INDEX
        for tgt, pred in zip(tgts[valid], preds[valid]):
            if 0 <= tgt < num_classes and 0 <= pred < num_classes:
                conf[tgt, pred] += 1
    return conf


def metrics_from_conf(conf: np.ndarray) -> Dict:
    num_classes = conf.shape[0]
    class_metrics: List[Dict] = []

    for cls in range(1, num_classes):
        tp = float(conf[cls, cls])
        fp = float(conf[:, cls].sum() - tp)
        fn = float(conf[cls, :].sum() - tp)
        iou = _safe_div(tp, tp + fp + fn)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
        class_metrics.append(
            {
                "class_id": cls,
                "class_name": config.CLASS_NAMES.get(cls, f"cls{cls}"),
                "iou": iou,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        )

    valid_ious = [m["iou"] for m in class_metrics if not np.isnan(m["iou"])]
    valid_f1s = [m["f1"] for m in class_metrics if not np.isnan(m["f1"])]
    valid_ps = [m["precision"] for m in class_metrics if not np.isnan(m["precision"])]
    valid_rs = [m["recall"] for m in class_metrics if not np.isnan(m["recall"])]

    correct = sum(conf[c, c] for c in range(1, num_classes))
    total = conf[1:, :].sum()
    oa = _safe_div(float(correct), float(total))
    po = oa if not np.isnan(oa) else 0.0
    pe = _safe_div(
        float(sum(conf[i, :].sum() * conf[:, i].sum() for i in range(1, num_classes))),
        float(total**2 if total else 1),
    )
    kappa = _safe_div(po - pe, 1 - pe) if not np.isnan(pe) else float("nan")

    dead = next((m for m in class_metrics if m["class_id"] == 1), None)

    return {
        "OA": float(oa) if not np.isnan(oa) else 0.0,
        "mIoU": float(np.mean(valid_ious)) if valid_ious else 0.0,
        "mF1": float(np.mean(valid_f1s)) if valid_f1s else 0.0,
        "mPrecision": float(np.mean(valid_ps)) if valid_ps else 0.0,
        "mRecall": float(np.mean(valid_rs)) if valid_rs else 0.0,
        "Kappa": float(kappa) if not np.isnan(kappa) else 0.0,
        "dead_iou": float(dead["iou"]) if dead and not np.isnan(dead["iou"]) else 0.0,
        "dead_f1": float(dead["f1"]) if dead and not np.isnan(dead["f1"]) else 0.0,
        "dead_precision": float(dead["precision"]) if dead and not np.isnan(dead["precision"]) else 0.0,
        "dead_recall": float(dead["recall"]) if dead and not np.isnan(dead["recall"]) else 0.0,
        "per_class": class_metrics,
    }


def print_table(results: Dict[str, Dict]):
    models = list(results.keys())
    keys = ["dead_f1", "dead_iou", "dead_precision", "dead_recall", "mIoU", "OA", "Kappa"]

    print("\n" + "=" * 100)
    print(f"{'Metric':<20}" + "".join(f"{m:<16}" for m in models))
    print("-" * 100)
    for key in keys:
        row = f"{key:<20}"
        for model_name in models:
            row += f"{results[model_name].get(key, 0.0):<16.4f}"
        print(row)
    print("=" * 100)


def save_confusion_matrix(conf: np.ndarray, model_name: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sub = conf[1:, 1:]
        labels = [f"[{c}] {config.CLASS_NAMES.get(c, '')}" for c in range(1, config.NUM_CLASSES + 1)]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(sub, cmap="Blues")
        plt.colorbar(im)
        ax.set_xticks(range(config.NUM_CLASSES))
        ax.set_yticks(range(config.NUM_CLASSES))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(f"{model_name.upper()} Confusion Matrix")
        thresh = sub.max() / 2 if sub.size else 0
        for i in range(config.NUM_CLASSES):
            for j in range(config.NUM_CLASSES):
                ax.text(
                    j,
                    i,
                    str(sub[i, j]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if sub[i, j] > thresh else "black",
                )
        plt.tight_layout()
        out = os.path.join(config.OUTPUT_DIR, f"confusion-{model_name}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Confusion matrix -> {out}")
    except Exception as exc:
        print(f"  Failed to save confusion matrix: {exc}")


def save_summary_csv(results: Dict[str, Dict], out_csv: str):
    rows = []
    for model_name, data in results.items():
        row = {
            "model": model_name,
            "dead_f1": data["dead_f1"],
            "dead_iou": data["dead_iou"],
            "dead_precision": data["dead_precision"],
            "dead_recall": data["dead_recall"],
            "mIoU": data["mIoU"],
            "mF1": data["mF1"],
            "mPrecision": data["mPrecision"],
            "mRecall": data["mRecall"],
            "OA": data["OA"],
            "Kappa": data["Kappa"],
        }
        for cls_metric in data["per_class"]:
            cid = cls_metric["class_id"]
            row[f"class{cid}_iou"] = cls_metric["iou"]
            row[f"class{cid}_f1"] = cls_metric["f1"]
            row[f"class{cid}_precision"] = cls_metric["precision"]
            row[f"class{cid}_recall"] = cls_metric["recall"]
        rows.append(row)

    rows.sort(key=lambda r: r["dead_f1"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["model_rank"] = rank

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Summary CSV -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ALL_MODELS + ["all"], default="all")
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default=None,
        help="Optional sample-level manifest filter (e.g., scale_analysis_manifest.csv)",
    )
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory override for confusion figures",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.output_dir:
        config.set_output_dir(args.output_dir)
        print(f"[Eval] OUTPUT_DIR override -> {config.OUTPUT_DIR}")

    if args.out_csv is None:
        args.out_csv = os.path.join(config.OUTPUT_DIR, "model_summary.csv")

    config.ensure_runtime_dirs()
    config.print_config_summary()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_to_eval = ALL_MODELS if args.model == "all" else [args.model]
    all_results: Dict[str, Dict] = {}

    for model_name in models_to_eval:
        print(f"\nEvaluate {model_name.upper()} ...")
        conf = evaluate_model(
            model_name,
            device,
            manifest_csv=args.manifest_csv,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
        )
        save_confusion_matrix(conf, model_name)
        all_results[model_name] = metrics_from_conf(conf)

    if all_results:
        print_table(all_results)
        save_summary_csv(all_results, args.out_csv)
