"""
Visualize sample strata and split composition.

Inputs:
- sample_summary_labeled.csv
- sample_manifest_labeled.csv (from split output)

Outputs:
- dead_area_distribution.png
- dead_patch_count_distribution.png
- dead_aggregation_distribution.png
- dead_cover_fraction_distribution.png
- rs_veg_fraction_distribution.png
- patch_fraction distributions
- background_type_counts.png
- tree_density_class_counts.png
- dead_density_class_counts.png
- living_density_class_counts.png
- spatial_pattern_type_counts.png
- patch_size_dominance_class_counts.png
- split_counts.png
- split_by_spatial_pattern.png
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

import config


def load_rows(path: str):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def spatial_pattern_value(row):
    value = str(row.get("spatial_pattern_type", "")).strip()
    if value:
        return value
    legacy = str(row.get("background_type", "")).strip()
    old_pattern_labels = {"compact_cluster", "fragmented_cluster", "mixed_pattern", "no_dead"}
    return legacy if legacy in old_pattern_labels else "unknown"


def to_float(row, key):
    try:
        return float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _ensure_out(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _compute_tertiles(values):
    vals = sorted(values)
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    q1 = vals[int((n - 1) * (1 / 3))]
    q2 = vals[int((n - 1) * (2 / 3))]
    return q1, q2


def plot_distribution(rows, metric_key, title, out_png):
    plt = _matplotlib()
    values = [to_float(r, metric_key) for r in rows]
    q1, q2 = _compute_tertiles(values)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=30, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.axvline(q1, color="#C44E52", linestyle="--", linewidth=1.8, label=f"Q33={q1:.4f}")
    ax.axvline(q2, color="#55A868", linestyle="--", linewidth=1.8, label=f"Q66={q2:.4f}")
    ax.set_title(title)
    ax.set_xlabel(metric_key)
    ax.set_ylabel("Sample Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_counter(counter, title, out_png, xlabel="Category"):
    plt = _matplotlib()
    labels = list(counter.keys())
    values = [counter[k] for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color="#4C72B0", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_split_by_field(split_rows, field, out_png, title=None):
    plt = _matplotlib()
    table = defaultdict(Counter)
    split_order = ["train_dev", "scale_analysis", "holdout_test"]

    for r in split_rows:
        bg = spatial_pattern_value(r) if field == "spatial_pattern_type" else (r.get(field, "") or "unknown")
        sp = r.get("split_group", "unknown") or "unknown"
        table[bg][sp] += 1

    bg_keys = sorted(table.keys())
    x = range(len(bg_keys))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, sp in enumerate(split_order):
        vals = [table[b].get(sp, 0) for b in bg_keys]
        pos = [i + (idx - 1) * width for i in x]
        ax.bar(pos, vals, width=width, label=sp)

    ax.set_title(title or f"Split Composition by {field}")
    ax.set_xlabel(field)
    ax.set_ylabel("Count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(bg_keys, rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run(summary_labeled_csv: str, split_labeled_csv: str, out_dir: str):
    out_dir = _ensure_out(out_dir)
    summary_rows = load_rows(summary_labeled_csv)
    split_rows = load_rows(split_labeled_csv) if os.path.exists(split_labeled_csv) else []

    plot_distribution(
        summary_rows,
        "dead_area_m2",
        "Dead Area Distribution",
        os.path.join(out_dir, "dead_area_distribution.png"),
    )
    plot_distribution(
        summary_rows,
        "dead_patch_count",
        "Dead Patch Count Distribution",
        os.path.join(out_dir, "dead_patch_count_distribution.png"),
    )
    plot_distribution(
        summary_rows,
        "dead_aggregation_index",
        "Dead Aggregation Index Distribution",
        os.path.join(out_dir, "dead_aggregation_distribution.png"),
    )
    for metric, title in (
        ("dead_cover_fraction", "Dead Cover Fraction Distribution"),
        ("rs_veg_fraction", "Remote-Sensing Vegetation Fraction Distribution"),
        ("rs_water_fraction", "Remote-Sensing Water Fraction Distribution"),
        ("small_patch_area_fraction", "Small Dead Patch Area Fraction Distribution"),
        ("medium_patch_area_fraction", "Medium Dead Patch Area Fraction Distribution"),
        ("large_patch_area_fraction", "Large Dead Patch Area Fraction Distribution"),
    ):
        if any(metric in r for r in summary_rows):
            plot_distribution(
                summary_rows,
                metric,
                title,
                os.path.join(out_dir, f"{metric}_distribution.png"),
            )

    plot_counter(
        Counter(spatial_pattern_value(r) for r in summary_rows),
        "Spatial Pattern Type Counts",
        os.path.join(out_dir, "spatial_pattern_type_counts.png"),
        xlabel="spatial_pattern_type",
    )
    plot_counter(
        Counter((r.get("background_type", "") or "unknown") for r in summary_rows),
        "Background Type Counts",
        os.path.join(out_dir, "background_type_counts.png"),
        xlabel="background_type",
    )
    for field in ("tree_density_class", "dead_density_class", "living_density_class"):
        plot_counter(
            Counter((r.get(field, "") or "unknown") for r in summary_rows),
            f"{field} Counts",
            os.path.join(out_dir, f"{field}_counts.png"),
            xlabel=field,
        )
    for field in ("climate_zone", "latitude_band", "patch_size_dominance_class"):
        plot_counter(
            Counter((r.get(field, "") or "unknown") for r in summary_rows),
            f"{field} Counts",
            os.path.join(out_dir, f"{field}_counts.png"),
            xlabel=field,
        )

    if split_rows:
        plot_counter(
            Counter((r.get("split_group", "") or "unknown") for r in split_rows),
            "Split Counts",
            os.path.join(out_dir, "split_counts.png"),
            xlabel="split_group",
        )
        plot_split_by_field(
            split_rows,
            "spatial_pattern_type",
            os.path.join(out_dir, "split_by_spatial_pattern.png"),
            title="Split Composition by Spatial Pattern Type",
        )
        for field in ("background_type", "climate_zone", "dead_density_class", "tree_density_class"):
            plot_split_by_field(
                split_rows,
                field,
                os.path.join(out_dir, f"split_by_{field}.png"),
                title=f"Split Composition by {field}",
            )

    print(f"[plot] saved figures to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_labeled_csv",
        default=os.path.join(config.SAMPLE_STATS_DIR, "sample_summary_labeled.csv"),
    )
    parser.add_argument(
        "--split_labeled_csv",
        default=os.path.join(config.SPLIT_DIR, "sample_manifest_labeled.csv"),
    )
    parser.add_argument(
        "--out_dir",
        default=config.figure_subdir("dataset", "sample_strata"),
    )
    args = parser.parse_args()
    run(args.summary_labeled_csv, args.split_labeled_csv, args.out_dir)


if __name__ == "__main__":
    main()
