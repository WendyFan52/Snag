"""
Stratified semantic evaluation by sample-level analysis fields.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch

import config
from analysis_utils import (
    DEFAULT_GROUP_FIELDS,
    available_models,
    confusion_from_arrays,
    load_manifest_lookup,
    metrics_from_confusion,
    require_manifest_fields,
    require_manifest_row,
    write_csv,
    iterate_model_batches,
)


def stratified_metrics(
    model_name: str,
    device: torch.device,
    manifest_csv: str = None,
    img_dir: str = None,
    mask_dir: str = None,
    group_fields=None,
):
    group_fields = tuple(group_fields or DEFAULT_GROUP_FIELDS)
    if not manifest_csv:
        raise RuntimeError("analyze_stratified_metrics.py requires --manifest_csv for scientific stratified evaluation")
    manifest_lookup = load_manifest_lookup(manifest_csv)
    num_classes = config.NUM_CLASSES + 1

    global_conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    global_samples = set()
    global_tiles = 0
    group_conf = defaultdict(lambda: np.zeros((num_classes, num_classes), dtype=np.int64))
    group_samples = defaultdict(set)
    group_tiles = defaultdict(int)

    for batch in iterate_model_batches(
        model_name,
        device,
        manifest_csv=manifest_csv,
        img_dir=img_dir,
        mask_dir=mask_dir,
    ):
        preds = batch["preds"]
        masks = batch["masks"]
        sample_ids = batch["sample_ids"]

        for idx, sample_id in enumerate(sample_ids):
            pred = preds[idx]
            mask = masks[idx]
            conf = confusion_from_arrays(mask, pred, num_classes)
            global_conf += conf
            global_samples.add(sample_id)
            global_tiles += 1

            row = require_manifest_row(manifest_lookup, sample_id, "stratified metrics")
            require_manifest_fields(row, group_fields, sample_id, "stratified metrics")
            for field in group_fields:
                value = str(row.get(field, "")).strip()
                key = (field, value)
                group_conf[key] += conf
                group_samples[key].add(sample_id)
                group_tiles[key] += 1

    rows = []
    global_metrics = metrics_from_confusion(global_conf)
    rows.append(
        {
            "model": model_name,
            "group_field": "global",
            "group_value": "all",
            "sample_count": len(global_samples),
            "tile_count": global_tiles,
            "pixel_count": int(global_conf[1:, :].sum()),
            **global_metrics,
        }
    )

    for (field, value), conf in sorted(group_conf.items()):
        rows.append(
            {
                "model": model_name,
                "group_field": field,
                "group_value": value,
                "sample_count": len(group_samples[(field, value)]),
                "tile_count": group_tiles[(field, value)],
                "pixel_count": int(conf[1:, :].sum()),
                **metrics_from_confusion(conf),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=available_models("all") + ["all"], default="all")
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    out_dir = args.output_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for model_name in available_models(args.model):
        print(f"[Stratified] {model_name}")
        rows.extend(
            stratified_metrics(
                model_name,
                device,
                manifest_csv=args.manifest_csv,
                img_dir=args.img_dir,
                mask_dir=args.mask_dir,
            )
        )

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = os.path.join(out_dir, f"stratified_metrics{suffix}.csv")
    write_csv(out_csv, rows)
    print(f"[Stratified] -> {out_csv}")


if __name__ == "__main__":
    main()
