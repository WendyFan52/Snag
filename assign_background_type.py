"""
Assign sample-level analysis strata.

Definitions:
- background_type is estimated from PlanetScope NDVI/NDWI summaries.
- tree_density_value is rs_ndvi_mean, a remote-sensing vegetation-density proxy.
- dead_density_value is dead_cover_fraction, because dead trees are fully labeled.
- living_density_value is living_observed_cover_fraction, because live trees are partial labels.
- spatial_pattern_type is derived from dead-patch geometry and patch-size fractions.
"""

import argparse
import csv
import os
from collections import Counter
from typing import Iterable, List

import config


def load_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: str, rows: List[dict]):
    if not rows:
        raise RuntimeError("No rows to write")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Iterable[float], q: float) -> float:
    seq = sorted(float(v) for v in values)
    if not seq:
        return 0.0
    if len(seq) == 1:
        return seq[0]
    pos = (len(seq) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(seq) - 1)
    frac = pos - lo
    return seq[lo] * (1.0 - frac) + seq[hi] * frac


def classify_tertile(value: float, low_thr: float, high_thr: float) -> str:
    if value <= low_thr:
        return "low"
    if value <= high_thr:
        return "medium"
    return "high"


def _to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def infer_patch_size_dominance(small_fraction: float, medium_fraction: float, large_fraction: float) -> str:
    values = {
        "small_patch_dominated": small_fraction,
        "medium_patch_dominated": medium_fraction,
        "large_patch_dominated": large_fraction,
    }
    return max(values, key=values.get)


def infer_spatial_pattern_type(
    dead_area_m2: float,
    dead_patch_count: float,
    dead_aggregation_index: float,
    small_patch_area_fraction: float,
    large_patch_area_fraction: float,
) -> str:
    if dead_area_m2 <= 0:
        return "no_dead"
    if dead_aggregation_index >= 0.50 and large_patch_area_fraction >= 0.50:
        return "compact_cluster"
    if small_patch_area_fraction >= 0.50 or (dead_aggregation_index <= 0.15 and dead_patch_count >= 6):
        return "fragmented_cluster"
    return "mixed_pattern"


def infer_background_type(rs_available: float, rs_veg_fraction: float, rs_water_fraction: float) -> str:
    if rs_available <= 0:
        return "unknown_background"
    if rs_water_fraction >= config.RS_WATER_FRACTION_THRESHOLD:
        return "water_influenced"
    if rs_veg_fraction < config.RS_SPARSE_VEG_FRACTION_THRESHOLD:
        return "bare_or_sparse_background"
    if rs_veg_fraction < config.RS_DENSE_VEG_FRACTION_THRESHOLD:
        return "sparse_vegetation_background"
    return "dense_vegetation_background"


def assign_backgrounds(summary_csv: str, output_csv: str) -> str:
    rows = load_rows(summary_csv)
    if not rows:
        raise RuntimeError("Summary CSV is empty")

    dead_area_values = [_to_float(r, "dead_area_m2") for r in rows]
    positive_dead_area_values = [v for v in dead_area_values if v > 0]
    dead_patch_count_values = [_to_float(r, "dead_patch_count") for r in rows]
    dead_agg_values = [_to_float(r, "dead_aggregation_index") for r in rows]
    tree_density_values = [_to_float(r, "rs_ndvi_mean") for r in rows]
    dead_density_values = [_to_float(r, "dead_cover_fraction") for r in rows]
    living_density_values = [_to_float(r, "living_observed_cover_fraction") for r in rows]

    area_low = percentile(positive_dead_area_values, 1 / 3)
    area_high = percentile(positive_dead_area_values, 2 / 3)
    patch_low = percentile(dead_patch_count_values, 1 / 3)
    patch_high = percentile(dead_patch_count_values, 2 / 3)
    agg_low = percentile(dead_agg_values, 1 / 3)
    agg_high = percentile(dead_agg_values, 2 / 3)
    tree_low = percentile(tree_density_values, 1 / 3)
    tree_high = percentile(tree_density_values, 2 / 3)
    dead_density_low = percentile(dead_density_values, 1 / 3)
    dead_density_high = percentile(dead_density_values, 2 / 3)
    living_low = percentile(living_density_values, 1 / 3)
    living_high = percentile(living_density_values, 2 / 3)

    output_rows = []
    for row in rows:
        dead_area_m2 = _to_float(row, "dead_area_m2")
        dead_patch_count = _to_float(row, "dead_patch_count")
        dead_agg = _to_float(row, "dead_aggregation_index")
        tree_density = _to_float(row, "rs_ndvi_mean")
        dead_density = _to_float(row, "dead_cover_fraction")
        living_density = _to_float(row, "living_observed_cover_fraction")
        rs_veg_fraction = _to_float(row, "rs_veg_fraction")
        water_fraction = _to_float(row, "rs_water_fraction")
        rs_available = _to_float(row, "rs_available")
        small_fraction = _to_float(row, "small_patch_area_fraction")
        medium_fraction = _to_float(row, "medium_patch_area_fraction")
        large_fraction = _to_float(row, "large_patch_area_fraction")

        row = dict(row)
        row["dead_area_class"] = classify_tertile(dead_area_m2, area_low, area_high)
        row["patch_count_class"] = classify_tertile(dead_patch_count, patch_low, patch_high)
        row["aggregation_class"] = classify_tertile(dead_agg, agg_low, agg_high)
        row["tree_density_value"] = round(tree_density, 8)
        row["dead_density_value"] = round(dead_density, 8)
        row["living_density_value"] = round(living_density, 8)
        row["water_fraction"] = round(water_fraction, 8)
        row["tree_density_class"] = classify_tertile(tree_density, tree_low, tree_high)
        row["dead_density_class"] = classify_tertile(dead_density, dead_density_low, dead_density_high)
        row["living_density_class"] = classify_tertile(living_density, living_low, living_high)
        row["density_definition"] = "tree=rs_ndvi_mean;dead=dead_area/valid_area;living=observed_live_area/valid_area"
        row["patch_size_dominance_class"] = infer_patch_size_dominance(small_fraction, medium_fraction, large_fraction)

        # Backward-compatible aliases for existing downstream scripts.
        row["dead_patch_count_class"] = row["patch_count_class"]
        row["dead_aggregation_class"] = row["aggregation_class"]
        row["spatial_pattern_type"] = infer_spatial_pattern_type(
            dead_area_m2,
            dead_patch_count,
            dead_agg,
            small_fraction,
            large_fraction,
        )
        row["background_type"] = infer_background_type(rs_available, rs_veg_fraction, water_fraction)
        row["background_type_source"] = "remote_sensing_indices" if rs_available > 0 else "unknown_no_image"
        output_rows.append(row)

    write_rows(output_csv, output_rows)

    spatial_counter = Counter(r["spatial_pattern_type"] for r in output_rows)
    background_counter = Counter(r["background_type"] for r in output_rows)
    print(f"[background] samples labeled: {len(output_rows)}")
    print(f"[background] dead area tertiles: low<={area_low:.4f}, medium<={area_high:.4f}")
    print(f"[background] patch-count tertiles: low<={patch_low:.4f}, medium<={patch_high:.4f}")
    print(f"[background] aggregation tertiles: low<={agg_low:.6f}, medium<={agg_high:.6f}")
    print(f"[background] tree density tertiles: low<={tree_low:.6f}, medium<={tree_high:.6f}")
    print(f"[background] dead density tertiles: low<={dead_density_low:.6f}, medium<={dead_density_high:.6f}")
    print(f"[background] living observed density tertiles: low<={living_low:.6f}, medium<={living_high:.6f}")
    print(f"[background] background counts: {dict(background_counter)}")
    print(f"[background] spatial pattern counts: {dict(spatial_counter)}")
    print(f"[background] saved -> {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        default=os.path.join(config.SAMPLE_STATS_DIR, "sample_summary.csv"),
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join(config.SAMPLE_STATS_DIR, "sample_summary_labeled.csv"),
    )
    args = parser.parse_args()
    assign_backgrounds(args.summary_csv, args.output_csv)


if __name__ == "__main__":
    main()
