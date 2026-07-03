"""
Generate publication-oriented figures from analysis/holdout CSV outputs.

This script does not replace make_analysis_figures.py. It creates a cleaner
paper figure set by reducing visual density:
    - keep all-model/all-metric plots as supplementary figures;
    - use selected metrics and selected variables for main paper figures;
    - use consistent colors, labels, spacing, and PDF/PNG outputs.
"""

import argparse
import glob
import math
import os
import warnings
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "unet": "UNet",
    "unet_resnet34": "UNet-ResNet34",
    "unet_convnext_tiny": "UNet-ConvNeXt-T",
    "hrnet_w18": "HRNet-W18",
    "segformer": "SegFormer-B2",
    "segformer_b4": "SegFormer-B4",
    "segformer_b5": "SegFormer-B5",
    "r50_upernet": "R50-UPerNet",
    "random_forest": "Random Forest",
}

MODEL_COLORS = {
    "unet": "#8DA0CB",
    "unet_resnet34": "#66C2A5",
    "unet_convnext_tiny": "#D55E00",
    "hrnet_w18": "#CC79A7",
    "segformer": "#0072B2",
    "segformer_b4": "#009E73",
    "segformer_b5": "#E69F00",
    "r50_upernet": "#56B4E9",
    "random_forest": "#4D4D4D",
}

FIELD_LABELS = {
    "aggregation_class": "Aggregation",
    "background_type": "Background",
    "climate_zone": "Climate zone",
    "dead_area_class": "Dead-tree area",
    "dead_density_class": "Dead-tree density",
    "latitude_band": "Latitude band",
    "patch_size_dominance_class": "Patch-size dominance",
    "spatial_pattern_type": "Spatial pattern",
    "tree_density_class": "Tree density",
}

VALUE_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "large_patch_dominated": "Large patches",
    "medium_patch_dominated": "Medium patches",
    "small_patch_dominated": "Small patches",
    "compact_cluster": "Compact",
    "fragmented_cluster": "Fragmented",
    "mixed_pattern": "Mixed",
    "dense_vegetation_background": "Dense vegetation",
    "water_influenced": "Water-influenced",
    "Equatorial": "Equatorial",
    "MidLatitude": "Mid-latitude",
    "HighLatitude": "High-latitude",
}

CATEGORY_ORDER = {
    "dead_area_class": ["low", "medium", "high"],
    "dead_density_class": ["low", "medium", "high"],
    "tree_density_class": ["low", "medium", "high"],
    "aggregation_class": ["low", "medium", "high"],
    "patch_size_dominance_class": [
        "small_patch_dominated",
        "medium_patch_dominated",
        "large_patch_dominated",
    ],
    "spatial_pattern_type": ["compact_cluster", "mixed_pattern", "fragmented_cluster"],
    "latitude_band": ["Equatorial", "MidLatitude", "HighLatitude"],
    "climate_zone": ["Tropical", "Temperate", "Arid", "Boreal"],
}

PRIMARY_STRATIFIED_FIELDS = [
    "dead_area_class",
    "dead_density_class",
    "tree_density_class",
    "aggregation_class",
    "patch_size_dominance_class",
    "spatial_pattern_type",
    "background_type",
    "climate_zone",
    "latitude_band",
]

STRUCTURE_STRATIFIED_FIELDS = [
    "dead_area_class",
    "dead_density_class",
    "tree_density_class",
    "aggregation_class",
    "patch_size_dominance_class",
    "spatial_pattern_type",
]

CONTEXT_STRATIFIED_FIELDS = [
    "background_type",
    "climate_zone",
    "latitude_band",
]

ORDERED_STRATIFIED_FIELDS = {
    "dead_area_class",
    "dead_density_class",
    "tree_density_class",
    "aggregation_class",
    "patch_size_dominance_class",
}

PIXEL_SCALE_VARIABLES = [
    "dead_fraction",
    "dead_relative_fraction",
    "local_dead_density_3x3",
    "local_dead_density_5x5",
]

SAMPLE_SCALE_VARIABLES = [
    "dead_area_m2",
    "dead_area_on_labeled_ratio",
    "dead_aggregation_index",
]


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    warnings.filterwarnings(
        "ignore",
        message="Auto-removal of grids by pcolor.*",
        category=matplotlib.MatplotlibDeprecationWarning,
    )
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#111111",
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.dpi": 320,
        }
    )
    return plt


def read_csv(path: Path, required: bool = False) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise FileNotFoundError(str(path))
        return None
    df = pd.read_csv(path)
    if df.empty:
        if required:
            raise RuntimeError(f"Empty CSV: {path}")
        return None
    return df


def nice_model(value: str) -> str:
    return MODEL_LABELS.get(str(value), str(value))


def nice_value(value: str) -> str:
    value = str(value)
    return VALUE_LABELS.get(value, value.replace("_", " ").title())


def nice_field(value: str) -> str:
    value = str(value)
    return FIELD_LABELS.get(value, value.replace("_", " ").title())


def sanitize_name(value: str) -> str:
    out = str(value).strip().replace(" ", "_")
    for ch in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        out = out.replace(ch, "-")
    return out


def save_figure(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def csv_path(input_dir: Path, stem: str, tag: str) -> Path:
    return input_dir / f"{stem}_{tag}.csv"


def infer_default_out_dir(analysis_dir: Path, tag: str) -> Path:
    analysis_dir = analysis_dir.resolve()
    if len(analysis_dir.parents) >= 2:
        research_root = analysis_dir.parents[1]
    else:
        research_root = analysis_dir.parent
    return research_root / "figures" / "paper" / tag


def sort_models(models: Iterable[str], ranking_df: Optional[pd.DataFrame]) -> List[str]:
    models = list(dict.fromkeys(str(m) for m in models))
    if ranking_df is None or "model" not in ranking_df.columns:
        return sorted(models)
    if "dead_f1" in ranking_df.columns:
        rank_map = ranking_df.set_index("model")["dead_f1"].astype(float).to_dict()
        return sorted(models, key=lambda m: rank_map.get(m, -1.0), reverse=True)
    return models


def choose_primary_model(
    analysis_summary: Optional[pd.DataFrame],
    holdout_summary: Optional[pd.DataFrame],
    requested: Optional[str],
) -> str:
    if requested:
        return requested
    for df in (holdout_summary, analysis_summary):
        if df is not None and {"model", "dead_f1"}.issubset(df.columns):
            row = df.sort_values("dead_f1", ascending=False).iloc[0]
            return str(row["model"])
    raise RuntimeError("Cannot choose primary model because no model summary was found.")


def choose_top_models(
    analysis_summary: Optional[pd.DataFrame],
    holdout_summary: Optional[pd.DataFrame],
    primary_model: str,
    top_n: int,
) -> List[str]:
    source = holdout_summary if holdout_summary is not None else analysis_summary
    if source is None or "model" not in source.columns:
        return [primary_model]
    source = source.sort_values("dead_f1", ascending=False) if "dead_f1" in source.columns else source
    models = [str(m) for m in source["model"].tolist()]
    if primary_model not in models:
        models.insert(0, primary_model)
    selected = list(dict.fromkeys(models))[: max(1, top_n)]
    if primary_model not in selected:
        selected[-1] = primary_model
    return selected


def ordered_values(field: str, values: Iterable[str]) -> List[str]:
    values = [str(v) for v in values]
    preferred = CATEGORY_ORDER.get(field)
    if preferred:
        ordered = [v for v in preferred if v in values]
        ordered += sorted(v for v in values if v not in ordered)
        return ordered
    return sorted(values)


def plot_model_performance(
    analysis_summary: Optional[pd.DataFrame],
    holdout_summary: Optional[pd.DataFrame],
    out_dir: Path,
):
    plt = _matplotlib()
    frames = []
    if analysis_summary is not None:
        tmp = analysis_summary.copy()
        tmp["dataset"] = "Analysis"
        frames.append(tmp)
    if holdout_summary is not None:
        tmp = holdout_summary.copy()
        tmp["dataset"] = "Holdout"
        frames.append(tmp)
    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    ranking_df = holdout_summary if holdout_summary is not None else analysis_summary
    models = sort_models(df["model"].dropna().astype(str).unique(), ranking_df)
    metrics = [m for m in ("dead_f1", "dead_iou", "dead_precision", "dead_recall") if m in df.columns]
    if not metrics:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics) + 2.0, max(4.2, 0.38 * len(models) + 1.5)), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    y = np.arange(len(models))
    dataset_offsets = {"Analysis": -0.13, "Holdout": 0.13}
    dataset_markers = {"Analysis": "o", "Holdout": "s"}
    dataset_colors = {"Analysis": "#4C78A8", "Holdout": "#F58518"}

    for ax, metric in zip(axes, metrics):
        vals_for_xlim = []
        for dataset in ("Analysis", "Holdout"):
            sub = df[df["dataset"] == dataset].set_index("model")
            if sub.empty or metric not in sub.columns:
                continue
            vals = [float(sub.loc[m, metric]) if m in sub.index and pd.notna(sub.loc[m, metric]) else np.nan for m in models]
            vals_for_xlim.extend([v for v in vals if np.isfinite(v)])
            ax.scatter(
                vals,
                y + dataset_offsets.get(dataset, 0),
                s=46,
                marker=dataset_markers[dataset],
                color=dataset_colors[dataset],
                edgecolor="white",
                linewidth=0.6,
                label=dataset,
                zorder=3,
            )
        if vals_for_xlim:
            low = max(0.0, min(vals_for_xlim) - 0.04)
            ax.set_xlim(low, 1.0)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Score")
        ax.set_yticks(y)
        ax.set_yticklabels([nice_model(m) for m in models])
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--")
        ax.grid(axis="y", visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Model performance on dead-tree detection", fontsize=12, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.14, wspace=0.28)
    save_figure(fig, out_dir / "fig01_model_performance")
    plt.close(fig)


def plot_thresholds(
    threshold_scan: Optional[pd.DataFrame],
    threshold_summary: Optional[pd.DataFrame],
    top_models: List[str],
    primary_model: str,
    out_dir: Path,
):
    if threshold_scan is None:
        return
    plt = _matplotlib()
    df = threshold_scan.copy()
    if "group_field" in df.columns:
        df = df[df["group_field"].astype(str) == "global"]
    if "group_value" in df.columns:
        df = df[df["group_value"].astype(str) == "all"]
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for model in top_models:
        sub = df[df["model"].astype(str) == model].sort_values("threshold")
        if sub.empty:
            continue
        color = MODEL_COLORS.get(model, "#555555")
        axes[0].plot(sub["threshold"], sub["f1"], marker="o", linewidth=1.9, color=color, label=nice_model(model))
        if threshold_summary is not None and {"model", "best_threshold", "best_f1"}.issubset(threshold_summary.columns):
            best = threshold_summary[threshold_summary["model"].astype(str) == model]
            if not best.empty:
                axes[0].scatter(
                    best["best_threshold"].iloc[0],
                    best["best_f1"].iloc[0],
                    s=60,
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=5,
                )

    primary = df[df["model"].astype(str) == primary_model].sort_values("threshold")
    if not primary.empty:
        for metric, color in (("precision", "#4C78A8"), ("recall", "#54A24B"), ("f1", "#E45756")):
            if metric in primary.columns:
                axes[1].plot(primary["threshold"], primary[metric], marker="o", linewidth=1.9, color=color, label=metric.title())

    axes[0].set_title("F1-threshold curves for top models")
    axes[0].set_xlabel("Dead probability threshold")
    axes[0].set_ylabel("F1 score")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, loc="lower left")

    axes[1].set_title(f"Precision-recall tradeoff: {nice_model(primary_model)}")
    axes[1].set_xlabel("Dead probability threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, loc="lower left")

    fig.suptitle("Threshold sensitivity", fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig02_threshold_sensitivity")
    plt.close(fig)


def _plot_stratified_field_set(
    df: pd.DataFrame,
    fields: List[str],
    primary_model: str,
    out_path: Path,
    title: str,
):
    plt = _matplotlib()
    fields = [f for f in fields if f in set(df["group_field"].astype(str))]
    if not fields:
        return

    cols = min(3, len(fields))
    rows = math.ceil(len(fields) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.35 * cols + 0.6, 3.35 * rows), sharey=True)
    axes = np.asarray(axes).ravel()

    for ax, field in zip(axes, fields):
        sub = df[df["group_field"].astype(str) == field].copy()
        sub = sub[sub["group_value"].astype(str) != "all"]
        values = ordered_values(field, sub["group_value"].dropna().astype(str).unique())
        if not values:
            ax.axis("off")
            continue
        sub["group_value"] = sub["group_value"].astype(str)
        sub = sub.set_index("group_value").reindex(values).reset_index()
        x = np.arange(len(values))
        f1 = sub["dead_f1"].astype(float)
        recall = sub["dead_recall"].astype(float)
        if field in ORDERED_STRATIFIED_FIELDS:
            ax.plot(x, f1, marker="o", linewidth=2.0, color="#D55E00", label="Dead F1")
            ax.plot(x, recall, marker="s", linewidth=1.8, color="#0072B2", label="Dead recall")
        else:
            ax.scatter(x - 0.06, f1, marker="o", s=58, color="#D55E00", label="Dead F1", zorder=3)
            ax.scatter(x + 0.06, recall, marker="s", s=58, color="#0072B2", label="Dead recall", zorder=3)
        tick_labels = []
        for _, row in sub.iterrows():
            n = row.get("sample_count", np.nan)
            suffix = f"\n(n={int(n)})" if pd.notna(n) else ""
            tick_labels.append(f"{nice_value(row['group_value'])}{suffix}")
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_title(nice_field(field), fontweight="bold")
        ax.grid(axis="y", linestyle="--")
        ax.grid(axis="x", visible=False)

    for ax in axes[len(fields) :]:
        ax.axis("off")

    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.18, hspace=0.62, wspace=0.25)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_stratified_primary(
    stratified: Optional[pd.DataFrame],
    primary_model: str,
    out_dir: Path,
):
    if stratified is None:
        return
    required = {"model", "group_field", "group_value", "dead_f1", "dead_recall", "sample_count"}
    if not required.issubset(stratified.columns):
        return
    df = stratified[stratified["model"].astype(str) == primary_model].copy()
    _plot_stratified_field_set(
        df,
        STRUCTURE_STRATIFIED_FIELDS,
        primary_model,
        out_dir / "fig03a_stratified_structure_primary_model",
        f"Structure-related stratified performance of {nice_model(primary_model)}",
    )
    _plot_stratified_field_set(
        df,
        CONTEXT_STRATIFIED_FIELDS,
        primary_model,
        out_dir / "fig03b_stratified_context_primary_model",
        f"Context-related stratified performance of {nice_model(primary_model)}",
    )


def plot_clean_stratified_heatmaps(
    stratified: Optional[pd.DataFrame],
    ranking_df: Optional[pd.DataFrame],
    out_dir: Path,
):
    if stratified is None:
        return
    required = {"model", "group_field", "group_value", "dead_f1"}
    if not required.issubset(stratified.columns):
        return
    plt = _matplotlib()
    out_subdir = out_dir / "stratified_dead_f1_heatmaps"

    for field in PRIMARY_STRATIFIED_FIELDS:
        sub = stratified[stratified["group_field"].astype(str) == field].copy()
        if sub.empty:
            continue
        values = ordered_values(field, sub["group_value"].dropna().astype(str).unique())
        models = sort_models(sub["model"].dropna().astype(str).unique(), ranking_df)
        pivot = sub.pivot_table(index="model", columns="group_value", values="dead_f1", aggfunc="mean")
        pivot = pivot.reindex(index=models, columns=values)
        matrix = pivot.to_numpy(dtype=float)
        fig_w = max(6.2, 1.25 * len(values) + 3.0)
        fig_h = max(3.8, 0.42 * len(models) + 1.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(matrix, aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels([nice_value(v) for v in values], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels([nice_model(m) for m in models])
        ax.set_title(f"Dead F1 by {nice_field(field).lower()}", fontweight="bold")
        ax.grid(False)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                val = matrix[y, x]
                if np.isfinite(val):
                    color = "white" if val >= 0.62 else "#1A1A1A"
                    ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("Dead F1")
        fig.tight_layout()
        save_figure(fig, out_subdir / f"dead_f1_{sanitize_name(field)}")
        plt.close(fig)


def plot_pixel_scale(
    pixel_scale: Optional[pd.DataFrame],
    top_models: List[str],
    out_dir: Path,
):
    if pixel_scale is None:
        return
    required = {"model", "variable", "bin_low", "bin_high", "dead_recall_in_bin"}
    if not required.issubset(pixel_scale.columns):
        return

    plt = _matplotlib()
    df = pixel_scale.copy()
    fields = [v for v in PIXEL_SCALE_VARIABLES if v in set(df["variable"].astype(str))]
    if not fields:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharey=True)
    axes = axes.ravel()
    for ax, variable in zip(axes, fields):
        sub = df[df["variable"].astype(str) == variable].copy()
        for model in top_models:
            one = sub[sub["model"].astype(str) == model].sort_values("bin_low")
            if one.empty:
                continue
            x = (one["bin_low"].astype(float) + one["bin_high"].astype(float)) / 2.0
            y = one["dead_recall_in_bin"].astype(float)
            ax.plot(x, y, marker="o", linewidth=1.9, color=MODEL_COLORS.get(model, "#555555"), label=nice_model(model))
        ax.set_title(variable.replace("_", " "), fontweight="bold")
        ax.set_xlabel("Bin midpoint")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", linestyle="--")
    axes[0].set_ylabel("Dead recall")
    axes[2].set_ylabel("Dead recall")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 4), frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Pixel-scale detectability", fontsize=12, fontweight="bold", y=1.01)
    fig.subplots_adjust(bottom=0.16, hspace=0.36, wspace=0.18)
    save_figure(fig, out_dir / "fig04_pixel_scale_detectability")
    plt.close(fig)


def _sample_x_values(df: pd.DataFrame, variable: str):
    x = (df["bin_low"].astype(float) + df["bin_high"].astype(float)) / 2.0
    xlabel = "Bin midpoint"
    if variable == "dead_area_m2":
        x = x / 10000.0
        xlabel = "Dead-tree area (ha)"
    elif variable == "dead_area_on_labeled_ratio":
        x = x * 100.0
        xlabel = "Dead-tree area within labelled area (%)"
    elif variable == "dead_aggregation_index":
        xlabel = "Dead-tree aggregation index"
    return x, xlabel


def plot_sample_scale(
    sample_scale: Optional[pd.DataFrame],
    top_models: List[str],
    out_dir: Path,
):
    if sample_scale is None:
        return
    required = {"model", "variable", "bin_low", "bin_high", "detect_20pct_rate"}
    if not required.issubset(sample_scale.columns):
        return

    plt = _matplotlib()
    df = sample_scale.copy()
    fields = [v for v in SAMPLE_SCALE_VARIABLES if v in set(df["variable"].astype(str))]
    if not fields:
        return

    fig, axes = plt.subplots(1, len(fields), figsize=(4.25 * len(fields), 4.2), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for ax, variable in zip(axes, fields):
        sub = df[df["variable"].astype(str) == variable].copy()
        for model in top_models:
            one = sub[sub["model"].astype(str) == model].sort_values("bin_low")
            if one.empty:
                continue
            x, xlabel = _sample_x_values(one, variable)
            y = one["detect_20pct_rate"].astype(float)
            ax.plot(x, y, marker="o", linewidth=1.9, color=MODEL_COLORS.get(model, "#555555"), label=nice_model(model))
        ax.set_title(variable.replace("_", " "), fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", linestyle="--")
    axes[0].set_ylabel("Detection rate (>=20% dead pixels)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 4), frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Sample-scale detectability", fontsize=12, fontweight="bold", y=1.03)
    fig.subplots_adjust(bottom=0.24, wspace=0.22)
    save_figure(fig, out_dir / "fig05_sample_scale_detectability")
    plt.close(fig)


def plot_cluster_summary(
    analysis_dir: Path,
    ranking_df: Optional[pd.DataFrame],
    out_dir: Path,
):
    files = [Path(p) for p in glob.glob(str(analysis_dir / "cluster_stats_*.csv"))]
    files = [p for p in files if p.name != "cluster_stats_all.csv"]
    if not files:
        return
    rows = []
    for path in files:
        df = read_csv(path)
        if df is None or "model" not in df.columns:
            continue
        model = str(df["model"].dropna().iloc[0])
        image_count = int(df["image_stem"].nunique()) if "image_stem" in df.columns else np.nan
        cluster_count = len(df)
        clusters_per_image = cluster_count / image_count if image_count else np.nan
        rows.append(
            {
                "model": model,
                "cluster_count": cluster_count,
                "image_count": image_count,
                "clusters_per_image": clusters_per_image,
                "median_area_m2": pd.to_numeric(df.get("area_m2"), errors="coerce").median(),
                "mean_conf": pd.to_numeric(df.get("mean_conf"), errors="coerce").mean(),
                "detected_rate": pd.to_numeric(df.get("detected"), errors="coerce").mean(),
            }
        )
    if not rows:
        return

    plt = _matplotlib()
    summary = pd.DataFrame(rows)
    models = sort_models(summary["model"], ranking_df)
    summary = summary.set_index("model").reindex(models).reset_index()
    y = np.arange(len(summary))
    metrics = [
        ("clusters_per_image", "Predicted clusters per image", False),
        ("median_area_m2", "Median cluster area (m2)", True),
        ("mean_conf", "Mean cluster confidence", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, max(4.0, 0.38 * len(summary) + 1.4)), sharey=True)
    for ax, (metric, title, logx) in zip(axes, metrics):
        vals = pd.to_numeric(summary[metric], errors="coerce").astype(float)
        colors = [MODEL_COLORS.get(m, "#777777") for m in summary["model"]]
        ax.barh(y, vals, color=colors, alpha=0.88)
        ax.set_title(title, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([nice_model(m) for m in summary["model"]])
        ax.invert_yaxis()
        if logx:
            ax.set_xscale("log")
        ax.grid(axis="x", linestyle="--")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Predicted object statistics", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig06_cluster_summary")
    plt.close(fig)


def build_paper_figures(
    analysis_dir: Path,
    analysis_tag: str,
    holdout_dir: Optional[Path],
    holdout_tag: str,
    out_dir: Path,
    primary_model: Optional[str],
    top_n: int,
):
    analysis_summary = read_csv(csv_path(analysis_dir, "model_summary", analysis_tag))
    holdout_summary = None
    if holdout_dir is not None:
        holdout_summary = read_csv(csv_path(holdout_dir, "model_summary", holdout_tag))

    primary = choose_primary_model(analysis_summary, holdout_summary, primary_model)
    top_models = choose_top_models(analysis_summary, holdout_summary, primary, top_n)
    ranking_df = holdout_summary if holdout_summary is not None else analysis_summary

    threshold_scan = read_csv(csv_path(analysis_dir, "threshold_scan", analysis_tag))
    threshold_summary = read_csv(csv_path(analysis_dir, "threshold_summary", analysis_tag))
    stratified = read_csv(csv_path(analysis_dir, "stratified_metrics", analysis_tag))
    pixel_scale = read_csv(csv_path(analysis_dir, "pixel_scale_metrics", analysis_tag))
    sample_scale = read_csv(csv_path(analysis_dir, "sample_scale_metrics", analysis_tag))

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PaperFigure] primary_model={primary}")
    print(f"[PaperFigure] top_models={', '.join(top_models)}")
    print(f"[PaperFigure] output -> {out_dir}")

    plot_model_performance(analysis_summary, holdout_summary, out_dir)
    plot_thresholds(threshold_scan, threshold_summary, top_models, primary, out_dir)
    plot_stratified_primary(stratified, primary, out_dir)
    plot_clean_stratified_heatmaps(stratified, ranking_df, out_dir)
    plot_pixel_scale(pixel_scale, top_models, out_dir)
    plot_sample_scale(sample_scale, top_models, out_dir)
    plot_cluster_summary(analysis_dir, ranking_df, out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate clean paper figures from TreeDead CSV outputs.")
    parser.add_argument(
        "--analysis_dir",
        default=os.path.join("_local_runtime", "research_outputs", "inference_outputs_4band", "analysis"),
        help="Directory containing analysis CSV outputs.",
    )
    parser.add_argument("--analysis_tag", default="analysis")
    parser.add_argument(
        "--holdout_dir",
        default=os.path.join("_local_runtime", "research_outputs", "inference_outputs_4band", "holdout"),
        help="Optional directory containing holdout CSV outputs.",
    )
    parser.add_argument("--holdout_tag", default="holdout")
    parser.add_argument("--out_dir", default=None, help="Output directory. Defaults to research_outputs/figures/paper/<analysis_tag>.")
    parser.add_argument("--primary_model", default=None, help="Primary model for stratified panels. Defaults to best holdout model if available.")
    parser.add_argument("--top_n", type=int, default=4, help="Number of top models shown in scale/threshold figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    holdout_dir = Path(args.holdout_dir) if args.holdout_dir else None
    if holdout_dir is not None and not holdout_dir.exists():
        holdout_dir = None
    out_dir = Path(args.out_dir) if args.out_dir else infer_default_out_dir(analysis_dir, args.analysis_tag)
    build_paper_figures(
        analysis_dir=analysis_dir,
        analysis_tag=args.analysis_tag,
        holdout_dir=holdout_dir,
        holdout_tag=args.holdout_tag,
        out_dir=out_dir,
        primary_model=args.primary_model,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
