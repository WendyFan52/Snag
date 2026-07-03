"""
Threshold scan for dead-class probability.
"""

import argparse
import os
from collections import defaultdict

import torch

from analysis_utils import (
    DEFAULT_GROUP_FIELDS,
    available_models,
    binary_metrics,
    build_thresholds,
    iterate_model_batches,
    load_manifest_lookup,
    require_manifest_fields,
    require_manifest_row,
    write_csv,
)


def _empty_counter():
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "pixels": 0}


def update_counter(counter: dict, tp: int, fp: int, fn: int, tn: int, pixels: int):
    counter["tp"] += int(tp)
    counter["fp"] += int(fp)
    counter["fn"] += int(fn)
    counter["tn"] += int(tn)
    counter["pixels"] += int(pixels)


def finalize_rows(model_name: str, tag: str, counters: dict, group_field: str, group_value: str):
    rows = []
    for thr, stats in sorted(counters.items()):
        metrics = binary_metrics(stats["tp"], stats["fp"], stats["fn"], stats["tn"])
        rows.append(
            {
                "tag": tag,
                "model": model_name,
                "group_field": group_field,
                "group_value": group_value,
                "threshold": thr,
                "pixels": stats["pixels"],
                **metrics,
            }
        )
    return rows


def threshold_scan(
    model_name: str,
    device: torch.device,
    manifest_csv: str = None,
    img_dir: str = None,
    mask_dir: str = None,
    thresholds=None,
    group_fields=None,
):
    thresholds = thresholds or build_thresholds()
    group_fields = tuple(group_fields or DEFAULT_GROUP_FIELDS)
    if not manifest_csv:
        raise RuntimeError("analyze_thresholds.py requires --manifest_csv for grouped threshold analysis")
    manifest_lookup = load_manifest_lookup(manifest_csv)

    global_counts = {thr: _empty_counter() for thr in thresholds}
    grouped_counts = defaultdict(lambda: {thr: _empty_counter() for thr in thresholds})

    for batch in iterate_model_batches(
        model_name,
        device,
        manifest_csv=manifest_csv,
        img_dir=img_dir,
        mask_dir=mask_dir,
    ):
        probs = batch["probs"][:, 1]
        preds = batch["preds"]
        masks = batch["masks"]
        sample_ids = batch["sample_ids"]

        for idx, sample_id in enumerate(sample_ids):
            prob = probs[idx]
            pred_cls = preds[idx]
            mask = masks[idx]
            valid = mask != 0
            if not valid.any():
                continue
            true_dead = mask == 1

            sample_groups = []
            row = require_manifest_row(manifest_lookup, sample_id, "threshold scan")
            require_manifest_fields(row, group_fields, sample_id, "threshold scan")
            for field in group_fields:
                value = str(row.get(field, "")).strip()
                sample_groups.append((field, value))

            for thr in thresholds:
                pred_dead = prob >= thr
                tp = int((pred_dead & true_dead & valid).sum())
                fp = int((pred_dead & (~true_dead) & valid).sum())
                fn = int(((~pred_dead) & true_dead & valid).sum())
                tn = int(((~pred_dead) & (~true_dead) & valid).sum())
                pixels = int(valid.sum())
                update_counter(global_counts[thr], tp, fp, fn, tn, pixels)
                for field, value in sample_groups:
                    update_counter(grouped_counts[(field, value)][thr], tp, fp, fn, tn, pixels)

    all_rows = finalize_rows(model_name, "", global_counts, "global", "all")
    for (field, value), counters in sorted(grouped_counts.items()):
        all_rows.extend(finalize_rows(model_name, "", counters, field, value))

    global_rows = [row for row in all_rows if row["group_field"] == "global"]
    if global_rows:
        best = max(global_rows, key=lambda row: (row["f1"], row["recall"], -abs(row["threshold"] - 0.5)))
    else:
        best = None
    return all_rows, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=available_models("all") + ["all"], default="all")
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--thr_start", type=float, default=0.1)
    parser.add_argument("--thr_stop", type=float, default=0.9)
    parser.add_argument("--thr_step", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = args.output_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    thresholds = build_thresholds(args.thr_start, args.thr_stop, args.thr_step)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scan_rows = []
    best_rows = []
    for model_name in available_models(args.model):
        print(f"[Threshold] {model_name}")
        rows, best = threshold_scan(
            model_name,
            device,
            manifest_csv=args.manifest_csv,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            thresholds=thresholds,
        )
        for row in rows:
            row["tag"] = args.tag
        scan_rows.extend(rows)
        if best is not None:
            best_rows.append(
                {
                    "tag": args.tag,
                    "model": model_name,
                    "best_threshold": best["threshold"],
                    "best_f1": best["f1"],
                    "precision": best["precision"],
                    "recall": best["recall"],
                    "iou": best["iou"],
                    "pixels": best["pixels"],
                }
            )

    suffix = f"_{args.tag}" if args.tag else ""
    scan_csv = os.path.join(out_dir, f"threshold_scan{suffix}.csv")
    best_csv = os.path.join(out_dir, f"threshold_summary{suffix}.csv")
    write_csv(scan_csv, scan_rows)
    write_csv(best_csv, best_rows)
    print(f"[Threshold] scan -> {scan_csv}")
    print(f"[Threshold] summary -> {best_csv}")


if __name__ == "__main__":
    main()
