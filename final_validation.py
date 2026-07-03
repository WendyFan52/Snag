"""
Final dead-class validation using fixed thresholds, typically from analysis split.
"""

import argparse
import csv
import os

import torch

from analysis_utils import (
    available_models,
    binary_metrics,
    iterate_model_batches,
    threshold_lookup_from_csv,
    write_csv,
)


def load_semantic_summary(path: str):
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Semantic summary CSV not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return {row.get("model", ""): row for row in csv.DictReader(fh)}


def validate_model(
    model_name: str,
    device: torch.device,
    decision_threshold: float,
    manifest_csv: str = None,
    img_dir: str = None,
    mask_dir: str = None,
):
    tp = fp = fn = tn = 0
    for batch in iterate_model_batches(
        model_name,
        device,
        manifest_csv=manifest_csv,
        img_dir=img_dir,
        mask_dir=mask_dir,
    ):
        prob_dead = batch["probs"][:, 1]
        masks = batch["masks"]
        valid = masks != 0
        true_dead = masks == 1
        pred_dead = prob_dead >= decision_threshold
        tp += int((pred_dead & true_dead & valid).sum())
        fp += int((pred_dead & (~true_dead) & valid).sum())
        fn += int(((~pred_dead) & true_dead & valid).sum())
        tn += int(((~pred_dead) & (~true_dead) & valid).sum())
    return binary_metrics(tp, fp, fn, tn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=available_models("all") + ["all"], default="all")
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--threshold_summary_csv", type=str, default=None)
    parser.add_argument("--fallback_threshold", type=float, default=0.5)
    parser.add_argument(
        "--allow_fallback_threshold",
        action="store_true",
        help="Allow fallback_threshold when threshold_summary_csv is missing or lacks a model",
    )
    parser.add_argument("--semantic_summary_csv", type=str, default=None)
    args = parser.parse_args()

    if not args.manifest_csv:
        raise RuntimeError("final_validation.py requires --manifest_csv to avoid validating unintended tiles")

    out_dir = args.output_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold_lookup = threshold_lookup_from_csv(
        args.threshold_summary_csv,
        allow_missing=args.allow_fallback_threshold,
    )
    semantic_lookup = load_semantic_summary(args.semantic_summary_csv)

    rows = []
    for model_name in available_models(args.model):
        if model_name in threshold_lookup:
            threshold = threshold_lookup[model_name]
            threshold_source = "analysis_threshold_summary"
        elif args.allow_fallback_threshold:
            threshold = args.fallback_threshold
            threshold_source = "explicit_fallback"
            print(
                f"[FinalValidation][warn] {model_name} missing in threshold summary; "
                f"using explicitly allowed fallback_threshold={args.fallback_threshold}"
            )
        else:
            raise KeyError(
                f"Threshold summary lacks model '{model_name}'. "
                "Run analyze_thresholds.py for this model or pass --allow_fallback_threshold explicitly."
            )
        metrics = validate_model(
            model_name,
            device,
            threshold,
            manifest_csv=args.manifest_csv,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
        )
        if semantic_lookup and model_name not in semantic_lookup:
            raise KeyError(f"Semantic summary lacks model '{model_name}'")
        semantic = semantic_lookup.get(model_name, {})
        rows.append(
            {
                "tag": args.tag,
                "model": model_name,
                "decision_threshold": threshold,
                "threshold_source": threshold_source,
                "dead_binary_precision": metrics["precision"],
                "dead_binary_recall": metrics["recall"],
                "dead_binary_f1": metrics["f1"],
                "dead_binary_iou": metrics["iou"],
                "dead_binary_specificity": metrics["specificity"],
                "dead_binary_accuracy": metrics["accuracy"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
                "semantic_dead_iou": semantic.get("dead_iou", ""),
                "semantic_dead_f1": semantic.get("dead_f1", ""),
                "semantic_dead_precision": semantic.get("dead_precision", ""),
                "semantic_dead_recall": semantic.get("dead_recall", ""),
                "semantic_mIoU": semantic.get("mIoU", ""),
                "semantic_OA": semantic.get("OA", ""),
                "semantic_Kappa": semantic.get("Kappa", ""),
            }
        )

    rows.sort(key=lambda row: row["dead_binary_f1"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["model_rank"] = rank

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = os.path.join(out_dir, f"final_validation{suffix}.csv")
    write_csv(out_csv, rows)
    print(f"[FinalValidation] -> {out_csv}")


if __name__ == "__main__":
    main()
