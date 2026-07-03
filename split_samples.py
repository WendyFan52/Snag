"""
Create sample-level splits for research workflows.

The split is balanced across multiple marginal strata instead of one large
cross-product. This is important for small sample counts, where full
climate x background x density x aggregation stratification would fragment
the data too much.
"""

import argparse
import csv
import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import config


BALANCE_FIELDS = (
    "dead_area_class",
    "aggregation_class",
    "climate_zone",
    "latitude_band",
    "background_type",
    "spatial_pattern_type",
    "tree_density_class",
    "dead_density_class",
    "patch_size_dominance_class",
)

CONTINUOUS_BALANCE_FIELDS = (
    "dead_area_m2",
    "dead_cover_fraction",
    "dead_aggregation_index",
    "rs_veg_fraction",
    "rs_water_fraction",
)

SPLIT_NAMES = ("train_dev", "scale_analysis", "holdout_test")


def load_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: str, rows: List[dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(row: dict, key: str) -> float:
    try:
        value = float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        value = 0.0
    return value if math.isfinite(value) else 0.0


def _value(row: dict, field: str) -> str:
    value = str(row.get(field, "")).strip()
    return value if value else "unknown"


def target_counts(total: int, train_ratio: float, scale_ratio: float, test_ratio: float) -> Dict[str, int]:
    raw = {
        "train_dev": total * train_ratio,
        "scale_analysis": total * scale_ratio,
        "holdout_test": total * test_ratio,
    }
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def build_totals(rows: List[dict], balance_fields: Iterable[str]):
    totals = {field: Counter(_value(row, field) for row in rows) for field in balance_fields}
    cont = {}
    for field in CONTINUOUS_BALANCE_FIELDS:
        values = [_safe_float(row, field) for row in rows]
        mean = sum(values) / len(values) if values else 0.0
        var = sum((v - mean) ** 2 for v in values) / len(values) if values else 0.0
        cont[field] = {"mean": mean, "std": math.sqrt(var) if var > 0 else 1.0}
    return totals, cont


def row_rarity(row: dict, totals: Dict[str, Counter], balance_fields: Iterable[str]) -> Tuple[float, str]:
    rarity = 0.0
    for field in balance_fields:
        count = totals[field].get(_value(row, field), 1)
        rarity += 1.0 / max(count, 1)
    return -rarity, str(row.get("sample_id", ""))


def score_assignment(
    rows: List[dict],
    assignments: List[str],
    balance_fields: Iterable[str],
    targets: Dict[str, int],
    ratios: Dict[str, float],
    totals: Dict[str, Counter],
    cont_totals: Dict[str, Dict[str, float]],
) -> float:
    score = 0.0
    split_counts = Counter(assignments)
    for name in SPLIT_NAMES:
        score += 2.0 * abs(split_counts[name] - targets[name]) / max(targets[name], 1)

    for field in balance_fields:
        observed = {name: Counter() for name in SPLIT_NAMES}
        for row, split_name in zip(rows, assignments):
            observed[split_name][_value(row, field)] += 1
        for value, total_count in totals[field].items():
            for name in SPLIT_NAMES:
                expected = total_count * ratios[name]
                score += abs(observed[name].get(value, 0) - expected) / max(total_count, 1)

    for field, stats in cont_totals.items():
        sums = {name: 0.0 for name in SPLIT_NAMES}
        counts = {name: 0 for name in SPLIT_NAMES}
        for row, split_name in zip(rows, assignments):
            sums[split_name] += _safe_float(row, field)
            counts[split_name] += 1
        for name in SPLIT_NAMES:
            mean = sums[name] / counts[name] if counts[name] else 0.0
            score += 0.25 * abs(mean - stats["mean"]) / max(stats["std"], 1e-9)
    return score


def random_assignments(targets: Dict[str, int], rng: random.Random) -> List[str]:
    assignments = []
    for name in SPLIT_NAMES:
        assignments.extend([name] * targets[name])
    rng.shuffle(assignments)
    return assignments


def assignment_score(
    row: dict,
    split_name: str,
    split_counts: Dict[str, int],
    targets: Dict[str, int],
    field_counts: Dict[str, Dict[str, Counter]],
    totals: Dict[str, Counter],
    cont_sums: Dict[str, Dict[str, float]],
    cont_totals: Dict[str, Dict[str, float]],
    balance_fields: Iterable[str],
    ratios: Dict[str, float],
) -> float:
    if split_counts[split_name] >= targets[split_name]:
        return float("inf")

    score = 0.0
    new_split_count = split_counts[split_name] + 1
    capacity_penalty = abs(new_split_count - targets[split_name]) / max(targets[split_name], 1)
    score += capacity_penalty * 0.25

    for field in balance_fields:
        value = _value(row, field)
        new_count = field_counts[split_name][field][value] + 1
        expected = totals[field][value] * ratios[split_name]
        score += abs(new_count - expected) / max(totals[field][value], 1)

    for field, stats in cont_totals.items():
        new_mean = (cont_sums[split_name][field] + _safe_float(row, field)) / max(new_split_count, 1)
        score += 0.25 * abs(new_mean - stats["mean"]) / max(stats["std"], 1e-9)

    return score


def balance_report_rows(
    rows: List[dict],
    balance_fields: Iterable[str],
    targets: Dict[str, int],
) -> List[dict]:
    report = []
    total_n = len(rows)
    ratios = {name: targets[name] / total_n for name in SPLIT_NAMES}
    by_split = {name: [row for row in rows if row.get("split_group") == name] for name in SPLIT_NAMES}

    for field in balance_fields:
        total_counter = Counter(_value(row, field) for row in rows)
        split_counters = {
            name: Counter(_value(row, field) for row in split_rows)
            for name, split_rows in by_split.items()
        }
        for value in sorted(total_counter):
            item = {
                "metric_type": "categorical",
                "field": field,
                "value": value,
                "total_count": total_counter[value],
            }
            for name in SPLIT_NAMES:
                expected = total_counter[value] * ratios[name]
                observed = split_counters[name].get(value, 0)
                item[f"{name}_count"] = observed
                item[f"{name}_expected"] = round(expected, 4)
                item[f"{name}_deviation"] = round(observed - expected, 4)
            report.append(item)

    for field in CONTINUOUS_BALANCE_FIELDS:
        total_values = [_safe_float(row, field) for row in rows]
        total_mean = sum(total_values) / len(total_values) if total_values else 0.0
        item = {
            "metric_type": "continuous",
            "field": field,
            "value": "mean",
            "total_count": len(total_values),
            "total_mean": round(total_mean, 8),
        }
        for name in SPLIT_NAMES:
            values = [_safe_float(row, field) for row in by_split[name]]
            mean = sum(values) / len(values) if values else 0.0
            item[f"{name}_count"] = len(values)
            item[f"{name}_mean"] = round(mean, 8)
            item[f"{name}_deviation"] = round(mean - total_mean, 8)
        report.append(item)

    return report


def build_splits(
    labeled_summary_csv: str,
    output_dir: str,
    train_ratio: float,
    scale_ratio: float,
    test_ratio: float,
    seed: int,
    include_patch_count_class: bool = False,
    balance_report_csv: str = None,
    search_iterations: int = 5000,
) -> Dict[str, str]:
    if abs((train_ratio + scale_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rows = [dict(row) for row in load_rows(labeled_summary_csv)]
    if not rows:
        raise RuntimeError("Labeled summary CSV is empty")

    balance_fields = list(BALANCE_FIELDS)
    if include_patch_count_class and "patch_count_class" not in balance_fields:
        balance_fields.append("patch_count_class")

    targets = target_counts(len(rows), train_ratio, scale_ratio, test_ratio)
    ratios = {
        "train_dev": train_ratio,
        "scale_analysis": scale_ratio,
        "holdout_test": test_ratio,
    }
    totals, cont_totals = build_totals(rows, balance_fields)

    rng = random.Random(seed)
    best_assignments = None
    best_score = float("inf")
    iterations = max(1, int(search_iterations))
    for _ in range(iterations):
        assignments = random_assignments(targets, rng)
        score = score_assignment(
            rows,
            assignments,
            balance_fields,
            targets,
            ratios,
            totals,
            cont_totals,
        )
        if score < best_score:
            best_score = score
            best_assignments = assignments

    assigned_rows = []
    for row, split_name in zip(rows, best_assignments):
        row = dict(row)
        row["split_group"] = split_name
        row["use_for_training"] = 1 if split_name == "train_dev" else 0
        row["use_for_scale_analysis"] = 1 if split_name in ("scale_analysis", "holdout_test") else 0
        assigned_rows.append(row)

    labeled_rows = sorted(assigned_rows, key=lambda r: (r.get("sample_id", ""), r.get("layer_name", "")))
    train_rows = [row for row in labeled_rows if row["split_group"] == "train_dev"]
    scale_rows = [row for row in labeled_rows if row["split_group"] == "scale_analysis"]
    holdout_rows = [row for row in labeled_rows if row["split_group"] == "holdout_test"]

    outputs = {
        "sample_manifest_labeled": os.path.join(output_dir, "sample_manifest_labeled.csv"),
        "train_manifest": os.path.join(output_dir, "train_manifest.csv"),
        "scale_analysis_manifest": os.path.join(output_dir, "scale_analysis_manifest.csv"),
        "holdout_test_manifest": os.path.join(output_dir, "holdout_test_manifest.csv"),
        "split_balance_report": balance_report_csv or os.path.join(output_dir, "split_balance_report.csv"),
    }

    write_rows(outputs["sample_manifest_labeled"], labeled_rows)
    write_rows(outputs["train_manifest"], train_rows)
    write_rows(outputs["scale_analysis_manifest"], scale_rows)
    write_rows(outputs["holdout_test_manifest"], holdout_rows)
    write_rows(outputs["split_balance_report"], balance_report_rows(labeled_rows, balance_fields, targets))

    print(f"[split] total samples: {len(labeled_rows)}")
    print(f"[split] targets: {targets}")
    print(f"[split] balance score: {best_score:.6f} from {iterations} random assignments")
    print(f"[split] train_dev: {len(train_rows)}")
    print(f"[split] scale_analysis: {len(scale_rows)}")
    print(f"[split] holdout_test: {len(holdout_rows)}")
    for key, path in outputs.items():
        print(f"[split] {key} -> {path}")
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labeled_summary_csv",
        default=os.path.join(config.SAMPLE_STATS_DIR, "sample_summary_labeled.csv"),
    )
    parser.add_argument(
        "--output_dir",
        default=config.SPLIT_DIR,
    )
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--scale_ratio", type=float, default=0.25)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--include_patch_count_class",
        action="store_true",
        help="Add patch_count_class as an extra marginal balance field",
    )
    parser.add_argument(
        "--balance_report_csv",
        default=None,
        help="Optional output path for split_balance_report.csv",
    )
    parser.add_argument("--search_iterations", type=int, default=5000)
    args = parser.parse_args()

    build_splits(
        labeled_summary_csv=args.labeled_summary_csv,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        scale_ratio=args.scale_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        include_patch_count_class=args.include_patch_count_class,
        balance_report_csv=args.balance_report_csv,
        search_iterations=args.search_iterations,
    )


if __name__ == "__main__":
    main()
