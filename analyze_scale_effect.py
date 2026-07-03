"""
Scale-effect analysis using tile predictions and sample-level summary fields.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from analysis_utils import (
    assign_bin,
    available_models,
    bin_label,
    build_thresholds,
    iterate_model_batches,
    load_manifest_lookup,
    quantile_bin_edges,
    require_manifest_fields,
    require_manifest_row,
    safe_div,
    safe_float,
    threshold_lookup_from_csv,
    write_csv,
)


PIXEL_BIN_EDGES = build_thresholds(0.0, 1.0, 0.1)

REQUIRED_SAMPLE_FIELDS = (
    "background_type",
    "spatial_pattern_type",
    "climate_zone",
    "tree_density_class",
    "dead_density_class",
    "living_density_class",
    "tree_density_value",
    "dead_density_value",
    "living_density_value",
    "dead_cover_fraction",
    "rs_veg_fraction",
    "rs_water_fraction",
    "rs_bare_fraction",
    "patch_size_dominance_class",
    "small_patch_area_fraction",
    "medium_patch_area_fraction",
    "large_patch_area_fraction",
    "dead_area_class",
    "aggregation_class",
    "dead_area_m2",
    "dead_aggregation_index",
    "dead_area_on_labeled_ratio",
)


def make_bin_stats(edges):
    n_bins = max(len(edges) - 1, 1)
    return {
        "edges": list(edges),
        "pixel_count": np.zeros(n_bins, dtype=np.int64),
        "gt_dead_pixels": np.zeros(n_bins, dtype=np.int64),
        "pred_dead_pixels": np.zeros(n_bins, dtype=np.int64),
        "tp_dead_pixels": np.zeros(n_bins, dtype=np.int64),
        "dead_to_living_pixels": np.zeros(n_bins, dtype=np.int64),
        "dead_prob_sum": np.zeros(n_bins, dtype=np.float64),
    }


def update_bin_stats(stats: dict, values: np.ndarray, valid: np.ndarray, true_dead: np.ndarray, pred_dead: np.ndarray, pred_cls: np.ndarray, prob_dead: np.ndarray):
    if not np.any(valid):
        return
    edges = stats["edges"]
    n_bins = len(edges) - 1
    v = np.clip(values[valid], edges[0], edges[-1])
    idx = assign_bin(v, edges)
    idx = np.clip(idx, 0, n_bins - 1)

    gt_dead = true_dead[valid].astype(np.int64)
    pred_dead_v = pred_dead[valid].astype(np.int64)
    tp_dead = (true_dead[valid] & pred_dead[valid]).astype(np.int64)
    dead_to_living = (true_dead[valid] & (pred_cls[valid] == 2)).astype(np.int64)
    prob = prob_dead[valid].astype(np.float64)

    stats["pixel_count"] += np.bincount(idx, minlength=n_bins)
    stats["gt_dead_pixels"] += np.bincount(idx, weights=gt_dead, minlength=n_bins).astype(np.int64)
    stats["pred_dead_pixels"] += np.bincount(idx, weights=pred_dead_v, minlength=n_bins).astype(np.int64)
    stats["tp_dead_pixels"] += np.bincount(idx, weights=tp_dead, minlength=n_bins).astype(np.int64)
    stats["dead_to_living_pixels"] += np.bincount(idx, weights=dead_to_living, minlength=n_bins).astype(np.int64)
    stats["dead_prob_sum"] += np.bincount(idx, weights=prob, minlength=n_bins)


def summarize_pixel_bins(model_name: str, variable_name: str, stats: dict):
    rows = []
    n_bins = len(stats["edges"]) - 1
    for idx in range(n_bins):
        pixel_count = int(stats["pixel_count"][idx])
        gt_dead_pixels = int(stats["gt_dead_pixels"][idx])
        pred_dead_pixels = int(stats["pred_dead_pixels"][idx])
        tp_dead_pixels = int(stats["tp_dead_pixels"][idx])
        dead_to_living_pixels = int(stats["dead_to_living_pixels"][idx])
        rows.append(
            {
                "model": model_name,
                "variable": variable_name,
                "bin_index": idx,
                "bin_label": bin_label(stats["edges"], idx),
                "bin_low": stats["edges"][idx],
                "bin_high": stats["edges"][idx + 1],
                "pixel_count": pixel_count,
                "gt_dead_pixels": gt_dead_pixels,
                "pred_dead_pixels": pred_dead_pixels,
                "tp_dead_pixels": tp_dead_pixels,
                "true_dead_rate": safe_div(gt_dead_pixels, pixel_count),
                "pred_dead_rate": safe_div(pred_dead_pixels, pixel_count),
                "dead_recall_in_bin": safe_div(tp_dead_pixels, gt_dead_pixels),
                "dead_to_living_error": safe_div(dead_to_living_pixels, gt_dead_pixels),
                "mean_dead_prob": safe_div(stats["dead_prob_sum"][idx], pixel_count),
            }
        )
    return rows


def _sample_default():
    return {
        "valid_pixels": 0,
        "gt_dead_pixels": 0,
        "tp_dead_pixels": 0,
        "pred_dead_pixels": 0,
        "fp_dead_pixels": 0,
        "dead_to_living_pixels": 0,
        "dead_prob_sum_gt": 0.0,
        "dead_prob_sum_all": 0.0,
    }


def compute_scale_effect(
    model_name: str,
    device: torch.device,
    manifest_csv: str = None,
    img_dir: str = None,
    mask_dir: str = None,
    decision_threshold: float = 0.5,
):
    if not manifest_csv:
        raise RuntimeError("analyze_scale_effect.py requires --manifest_csv for sample-level scale-effect fields")
    manifest_lookup = load_manifest_lookup(manifest_csv)
    pixel_stats = {
        "dead_fraction": make_bin_stats(PIXEL_BIN_EDGES),
        "living_fraction": make_bin_stats(PIXEL_BIN_EDGES),
        "tree_fraction": make_bin_stats(PIXEL_BIN_EDGES),
        "dead_relative_fraction": make_bin_stats(PIXEL_BIN_EDGES),
        "local_dead_density_3x3": make_bin_stats(PIXEL_BIN_EDGES),
        "local_dead_density_5x5": make_bin_stats(PIXEL_BIN_EDGES),
        "local_tree_density_3x3": make_bin_stats(PIXEL_BIN_EDGES),
        "local_tree_density_5x5": make_bin_stats(PIXEL_BIN_EDGES),
        "local_dead_relative_density_3x3": make_bin_stats(PIXEL_BIN_EDGES),
    }
    sample_stats = defaultdict(_sample_default)

    for batch in iterate_model_batches(
        model_name,
        device,
        manifest_csv=manifest_csv,
        img_dir=img_dir,
        mask_dir=mask_dir,
    ):
        probs = batch["probs"]
        preds = batch["preds"]
        masks = batch["masks"]
        softs = batch["softs"]
        sample_ids = batch["sample_ids"]

        prob_dead = probs[:, 1]
        pred_dead = prob_dead >= decision_threshold
        true_dead = masks == 1
        valid = masks != 0

        dead_fraction = softs[:, 1]
        living_fraction = softs[:, 2] if softs.shape[1] > 2 else np.zeros_like(dead_fraction)
        tree_fraction = np.clip(dead_fraction + living_fraction, 0.0, 1.0)
        dead_relative_fraction = np.where(tree_fraction > 1e-6, dead_fraction / tree_fraction, 0.0)

        dead_t = torch.from_numpy(dead_fraction).unsqueeze(1).float()
        tree_t = torch.from_numpy(tree_fraction).unsqueeze(1).float()
        local_dead_3 = F.avg_pool2d(dead_t, kernel_size=3, stride=1, padding=1).squeeze(1).numpy()
        local_dead_5 = F.avg_pool2d(dead_t, kernel_size=5, stride=1, padding=2).squeeze(1).numpy()
        local_tree_3 = F.avg_pool2d(tree_t, kernel_size=3, stride=1, padding=1).squeeze(1).numpy()
        local_tree_5 = F.avg_pool2d(tree_t, kernel_size=5, stride=1, padding=2).squeeze(1).numpy()
        local_dead_rel_3 = np.where(local_tree_3 > 1e-6, local_dead_3 / local_tree_3, 0.0)

        variables = {
            "dead_fraction": dead_fraction,
            "living_fraction": living_fraction,
            "tree_fraction": tree_fraction,
            "dead_relative_fraction": dead_relative_fraction,
            "local_dead_density_3x3": local_dead_3,
            "local_dead_density_5x5": local_dead_5,
            "local_tree_density_3x3": local_tree_3,
            "local_tree_density_5x5": local_tree_5,
            "local_dead_relative_density_3x3": local_dead_rel_3,
        }

        for name, values in variables.items():
            update_bin_stats(pixel_stats[name], values, valid, true_dead, pred_dead, preds, prob_dead)

        for idx, sample_id in enumerate(sample_ids):
            stats = sample_stats[sample_id]
            valid_i = valid[idx]
            true_dead_i = true_dead[idx]
            pred_dead_i = pred_dead[idx]
            pred_cls_i = preds[idx]
            prob_i = prob_dead[idx]

            stats["valid_pixels"] += int(valid_i.sum())
            stats["gt_dead_pixels"] += int((true_dead_i & valid_i).sum())
            stats["tp_dead_pixels"] += int((true_dead_i & pred_dead_i & valid_i).sum())
            stats["pred_dead_pixels"] += int((pred_dead_i & valid_i).sum())
            stats["fp_dead_pixels"] += int((pred_dead_i & (~true_dead_i) & valid_i).sum())
            stats["dead_to_living_pixels"] += int((true_dead_i & (pred_cls_i == 2) & valid_i).sum())
            stats["dead_prob_sum_gt"] += float(prob_i[true_dead_i & valid_i].sum())
            stats["dead_prob_sum_all"] += float(prob_i[valid_i].sum())

    pixel_rows = []
    for name, stats in pixel_stats.items():
        pixel_rows.extend(summarize_pixel_bins(model_name, name, stats))

    detailed_rows = []
    sample_rows = []
    for sample_id, stats in sorted(sample_stats.items()):
        row = require_manifest_row(manifest_lookup, sample_id, "scale-effect analysis")
        require_manifest_fields(row, REQUIRED_SAMPLE_FIELDS, sample_id, "scale-effect analysis")
        gt_dead_pixels = stats["gt_dead_pixels"]
        valid_pixels = stats["valid_pixels"]
        sample_row = {
            "model": model_name,
            "sample_id": sample_id,
            "background_type": str(row.get("background_type", "")).strip(),
            "spatial_pattern_type": str(row.get("spatial_pattern_type", "")).strip(),
            "climate_zone": str(row.get("climate_zone", "")).strip(),
            "tree_density_class": str(row.get("tree_density_class", "")).strip(),
            "dead_density_class": str(row.get("dead_density_class", "")).strip(),
            "living_density_class": str(row.get("living_density_class", "")).strip(),
            "tree_density_value": safe_float(row.get("tree_density_value", 0.0)),
            "dead_density_value": safe_float(row.get("dead_density_value", 0.0)),
            "living_density_value": safe_float(row.get("living_density_value", 0.0)),
            "dead_cover_fraction": safe_float(row.get("dead_cover_fraction", 0.0)),
            "rs_veg_fraction": safe_float(row.get("rs_veg_fraction", 0.0)),
            "rs_water_fraction": safe_float(row.get("rs_water_fraction", 0.0)),
            "rs_bare_fraction": safe_float(row.get("rs_bare_fraction", 0.0)),
            "patch_size_dominance_class": str(row.get("patch_size_dominance_class", "")).strip(),
            "small_patch_area_fraction": safe_float(row.get("small_patch_area_fraction", 0.0)),
            "medium_patch_area_fraction": safe_float(row.get("medium_patch_area_fraction", 0.0)),
            "large_patch_area_fraction": safe_float(row.get("large_patch_area_fraction", 0.0)),
            "dead_area_class": str(row.get("dead_area_class", "")).strip(),
            "aggregation_class": str(row.get("aggregation_class", "")).strip(),
            "dead_area_m2": safe_float(row.get("dead_area_m2", 0.0)),
            "dead_aggregation_index": safe_float(row.get("dead_aggregation_index", 0.0)),
            "dead_area_on_labeled_ratio": safe_float(row.get("dead_area_on_labeled_ratio", 0.0)),
            "valid_pixels": valid_pixels,
            "gt_dead_pixels": gt_dead_pixels,
            "tp_dead_pixels": stats["tp_dead_pixels"],
            "pred_dead_pixels": stats["pred_dead_pixels"],
            "fp_dead_pixels": stats["fp_dead_pixels"],
            "dead_recall": safe_div(stats["tp_dead_pixels"], gt_dead_pixels),
            "dead_precision": safe_div(stats["tp_dead_pixels"], stats["pred_dead_pixels"]),
            "mean_dead_prob_on_gt_dead": safe_div(stats["dead_prob_sum_gt"], gt_dead_pixels),
            "mean_dead_prob_all": safe_div(stats["dead_prob_sum_all"], valid_pixels),
            "detect_any": int(stats["tp_dead_pixels"] > 0),
            "detect_20pct": int(safe_div(stats["tp_dead_pixels"], gt_dead_pixels) >= 0.2 if gt_dead_pixels else 0),
            "dead_to_living_error": safe_div(stats["dead_to_living_pixels"], gt_dead_pixels),
        }
        detailed_rows.append(sample_row)
        if gt_dead_pixels > 0:
            sample_rows.append(sample_row)

    summary_rows = []
    for variable in ("dead_area_m2", "dead_aggregation_index", "dead_area_on_labeled_ratio"):
        values = [row[variable] for row in sample_rows]
        edges = quantile_bin_edges(values, num_bins=5)
        if len(edges) <= 1:
            continue
        n_bins = max(len(edges) - 1, 1)
        buckets = [list() for _ in range(n_bins)]
        if values:
            idxs = assign_bin(np.asarray(values, dtype=np.float32), edges)
            idxs = np.clip(idxs, 0, n_bins - 1)
            for row, idx in zip(sample_rows, idxs):
                buckets[int(idx)].append(row)

        for idx, bucket in enumerate(buckets):
            if not bucket:
                continue
            summary_rows.append(
                {
                    "model": model_name,
                    "variable": variable,
                    "bin_index": idx,
                    "bin_label": bin_label(edges, idx),
                    "bin_low": edges[idx],
                    "bin_high": edges[idx + 1],
                    "sample_count": len(bucket),
                    "detect_any_rate": safe_div(sum(r["detect_any"] for r in bucket), len(bucket)),
                    "detect_20pct_rate": safe_div(sum(r["detect_20pct"] for r in bucket), len(bucket)),
                    "mean_dead_recall": safe_div(sum(r["dead_recall"] for r in bucket), len(bucket)),
                    "mean_dead_precision": safe_div(sum(r["dead_precision"] for r in bucket), len(bucket)),
                    "mean_dead_prob_on_gt_dead": safe_div(sum(r["mean_dead_prob_on_gt_dead"] for r in bucket), len(bucket)),
                    "mean_dead_to_living_error": safe_div(sum(r["dead_to_living_error"] for r in bucket), len(bucket)),
                }
            )
    return pixel_rows, detailed_rows, summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=available_models("all") + ["all"], default="all")
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--threshold_summary_csv", type=str, default=None)
    parser.add_argument(
        "--allow_fallback_threshold",
        action="store_true",
        help="Allow decision_threshold when threshold_summary_csv is missing or lacks a model",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold_lookup = threshold_lookup_from_csv(
        args.threshold_summary_csv,
        allow_missing=args.allow_fallback_threshold,
    )

    pixel_rows = []
    detailed_rows = []
    summary_rows = []
    for model_name in available_models(args.model):
        if model_name in threshold_lookup:
            decision_threshold = threshold_lookup[model_name]
            print(f"[Scale] {model_name} threshold={decision_threshold:.3f} source=threshold_summary")
        elif args.allow_fallback_threshold:
            decision_threshold = args.decision_threshold
            print(f"[Scale][warn] {model_name} missing threshold summary; using explicitly allowed fallback threshold={decision_threshold:.3f}")
        else:
            raise KeyError(
                f"Threshold summary lacks model '{model_name}'. "
                "Run analyze_thresholds.py for this model or pass --allow_fallback_threshold explicitly."
            )
        px_rows, det_rows, sum_rows = compute_scale_effect(
            model_name,
            device,
            manifest_csv=args.manifest_csv,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            decision_threshold=decision_threshold,
        )
        for row in px_rows:
            row["tag"] = args.tag
            row["decision_threshold"] = decision_threshold
        for row in det_rows:
            row["tag"] = args.tag
            row["decision_threshold"] = decision_threshold
        for row in sum_rows:
            row["tag"] = args.tag
            row["decision_threshold"] = decision_threshold
        pixel_rows.extend(px_rows)
        detailed_rows.extend(det_rows)
        summary_rows.extend(sum_rows)

    suffix = f"_{args.tag}" if args.tag else ""
    pixel_csv = os.path.join(out_dir, f"pixel_scale_metrics{suffix}.csv")
    sample_detail_csv = os.path.join(out_dir, f"sample_scale_detailed{suffix}.csv")
    sample_summary_csv = os.path.join(out_dir, f"sample_scale_metrics{suffix}.csv")
    write_csv(pixel_csv, pixel_rows)
    write_csv(sample_detail_csv, detailed_rows)
    write_csv(sample_summary_csv, summary_rows)
    print(f"[Scale] pixel -> {pixel_csv}")
    print(f"[Scale] sample detail -> {sample_detail_csv}")
    print(f"[Scale] sample summary -> {sample_summary_csv}")


if __name__ == "__main__":
    main()
