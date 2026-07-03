"""
Generate organized summary figures from stage-3 CSV outputs.

Figure layout:
    figures/<tag>/models/
    figures/<tag>/thresholds/
    figures/<tag>/stratified/
    figures/<tag>/scale/
    figures/<tag>/validation/
    figures/<tag>/clusters/
    figures/<tag>/maps/
"""

import argparse
import glob
import os
import shutil
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import config


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_name(value: str) -> str:
    out = str(value).strip().replace(" ", "_")
    for ch in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        out = out.replace(ch, "-")
    return out


def maybe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[Figure] skip unreadable csv: {path} ({exc})")
        return None
    if df.empty:
        return None
    return df


def copy_matches(src_dir: str, pattern: str, dst_dir: str):
    matches = glob.glob(os.path.join(src_dir, pattern))
    if not matches:
        return
    ensure_dir(dst_dir)
    for src in matches:
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        shutil.copy2(src, dst)


def plot_model_summary(summary_csv: str, out_dir: str):
    df = maybe_read_csv(summary_csv)
    if df is None or "model" not in df.columns:
        return
    ensure_dir(out_dir)
    plt = _matplotlib()

    if "model_rank" in df.columns:
        df = df.sort_values("model_rank")
    else:
        df = df.sort_values("dead_f1", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.65 * len(df) + 1)))
    ax.barh(df["model"], df["dead_f1"], color="#C44E52", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("dead_f1")
    ax.set_title("Model Ranking by Dead F1")
    for idx, val in enumerate(df["dead_f1"]):
        ax.text(float(val) + 0.005, idx, f"{float(val):.4f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_ranking_dead_f1.png"), dpi=160)
    plt.close(fig)

    metric_cols = [col for col in ("dead_f1", "dead_precision", "dead_recall", "dead_iou", "mIoU") if col in df.columns]
    if metric_cols:
        plot_df = df[["model"] + metric_cols].set_index("model")
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(plot_df)), 6))
        x = np.arange(len(plot_df))
        width = 0.14 if len(metric_cols) >= 4 else 0.18
        offsets = np.linspace(-(len(metric_cols) - 1) / 2, (len(metric_cols) - 1) / 2, len(metric_cols)) * width
        colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2", "#CCB974"]
        for idx, col in enumerate(metric_cols):
            ax.bar(x + offsets[idx], plot_df[col].astype(float).values, width=width, label=col, color=colors[idx % len(colors)], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df.index, rotation=25, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Score")
        ax.set_title("Core Model Metrics")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "model_metric_comparison.png"), dpi=160)
        plt.close(fig)


def plot_thresholds(scan_csv: str, summary_csv: str, out_dir: str):
    ensure_dir(out_dir)
    plt = _matplotlib()

    scan_df = maybe_read_csv(scan_csv)
    if scan_df is not None:
        df = scan_df.copy()
        if "group_field" in df.columns:
            df = df[df["group_field"] == "global"]
        if "group_value" in df.columns:
            df = df[df["group_value"] == "all"]
        for model_name in sorted(df["model"].dropna().astype(str).unique()):
            sub = df[df["model"].astype(str) == model_name].sort_values("threshold")
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for col, color in (
                ("precision", "#4C72B0"),
                ("recall", "#55A868"),
                ("f1", "#C44E52"),
                ("iou", "#8172B2"),
            ):
                if col in sub.columns:
                    ax.plot(sub["threshold"], sub[col], marker="o", linewidth=1.8, label=col, color=color)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Dead probability threshold")
            ax.set_ylabel("Score")
            ax.set_title(f"Threshold Scan - {model_name}")
            ax.grid(alpha=0.25, linestyle="--")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"threshold_curve_{sanitize_name(model_name)}.png"), dpi=160)
            plt.close(fig)

    summary_df = maybe_read_csv(summary_csv)
    if summary_df is not None and {"model", "best_threshold", "best_f1"}.issubset(summary_df.columns):
        df = summary_df.sort_values("best_f1", ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.55 * len(df) + 1.2)))

        axes[0].barh(df["model"], df["best_f1"], color="#C44E52", alpha=0.85)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("best_f1")
        axes[0].set_title("Best Dead F1 by Model")

        axes[1].barh(df["model"], df["best_threshold"], color="#4C72B0", alpha=0.85)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 1.0)
        axes[1].set_xlabel("best_threshold")
        axes[1].set_title("Best Threshold by Model")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "threshold_summary.png"), dpi=160)
        plt.close(fig)


def _draw_heatmap(ax, matrix: np.ndarray, row_labels: Iterable[str], col_labels: Iterable[str], title: str):
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(list(col_labels), rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(list(row_labels), fontsize=8)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                text = "NA"
                color = "black"
            else:
                text = f"{val:.2f}"
                color = "white" if val >= 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    return im


def plot_stratified(stratified_csv: str, out_dir: str):
    df = maybe_read_csv(stratified_csv)
    if df is None or "group_field" not in df.columns:
        return
    ensure_dir(out_dir)
    plt = _matplotlib()

    work = df[(df["group_field"].astype(str) != "global") & (df["group_value"].astype(str) != "")]
    metrics = ["dead_f1", "dead_recall", "dead_precision", "mIoU"]
    for field in sorted(work["group_field"].dropna().astype(str).unique()):
        sub = work[work["group_field"].astype(str) == field].copy()
        if sub.empty:
            continue
        models = list(dict.fromkeys(sub["model"].astype(str).tolist()))
        values = list(dict.fromkeys(sub["group_value"].astype(str).tolist()))
        if not models or not values:
            continue

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(max(12, 0.95 * len(values) + 6), max(7, 0.65 * len(models) + 3.5)),
        )
        axes = axes.ravel()
        im = None
        for idx, metric in enumerate(metrics):
            if metric not in sub.columns:
                axes[idx].axis("off")
                axes[idx].set_title(f"{metric} (missing)")
                continue
            pivot = sub.pivot_table(index="model", columns="group_value", values=metric, aggfunc="mean")
            pivot = pivot.reindex(index=models, columns=values)
            matrix = pivot.to_numpy(dtype=float)
            im = _draw_heatmap(axes[idx], matrix, pivot.index, pivot.columns, metric)
        fig.suptitle(f"Stratified Metrics - {field}", fontsize=12, fontweight="bold", y=0.98)
        fig.subplots_adjust(right=0.88, top=0.90, bottom=0.18, wspace=0.42, hspace=0.60)
        if im is not None:
            # Keep the shared score legend outside the heatmap panels.
            cax = fig.add_axes([0.91, 0.22, 0.018, 0.58])
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label("Score", fontsize=9)
            cbar.ax.tick_params(labelsize=8)
        fig.savefig(os.path.join(out_dir, f"stratified_{sanitize_name(field)}.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)

        bar_metrics = [metric for metric in ("dead_f1", "dead_recall") if metric in sub.columns]
        if not bar_metrics:
            continue
        fig, axes = plt.subplots(1, len(bar_metrics), figsize=(7 * len(bar_metrics), 5.2))
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])
        x = np.arange(len(values))
        width = min(0.8 / max(len(models), 1), 0.25)
        offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width
        colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2", "#CCB974", "#64B5CD", "#8C8C8C"]

        for ax, metric in zip(axes, bar_metrics):
            pivot = sub.pivot_table(index="model", columns="group_value", values=metric, aggfunc="mean")
            pivot = pivot.reindex(index=models, columns=values)
            for idx, model_name in enumerate(models):
                vals = pivot.loc[model_name].astype(float).to_numpy()
                ax.bar(x + offsets[idx], vals, width=width, label=model_name, color=colors[idx % len(colors)], alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(values, rotation=25, ha="right")
            ax.set_ylim(0, 1.02)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by {field}")
            ax.grid(axis="y", alpha=0.2, linestyle="--")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=8,
                frameon=False,
            )
        fig.tight_layout(rect=(0, 0, 0.84, 1))
        fig.savefig(
            os.path.join(out_dir, f"stratified_bar_{sanitize_name(field)}.png"),
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)


def _lineplot_by_variable(df: pd.DataFrame, variable: str, x_col: str, metrics: Iterable[tuple], out_png: str, title: str):
    plt = _matplotlib()
    metrics = list(metrics)
    sub = df[df["variable"].astype(str) == variable].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.8))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    models = sorted(sub["model"].dropna().astype(str).unique())
    for ax, (metric, ylabel) in zip(axes, metrics):
        for model_name in models:
            one = sub[sub["model"].astype(str) == model_name].sort_values(x_col)
            if metric not in one.columns:
                continue
            x = one[x_col].astype(float).to_numpy()
            y = one[metric].astype(float).to_numpy()
            ax.plot(x, y, marker="o", linewidth=1.8, label=model_name)
        ax.set_xlabel(x_col)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_title(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            frameon=False,
        )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.84, 0.95))
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scale(pixel_csv: str, sample_csv: str, out_dir: str):
    ensure_dir(out_dir)
    pixel_df = maybe_read_csv(pixel_csv)
    if pixel_df is not None and {"variable", "bin_index"}.issubset(pixel_df.columns):
        metrics = (
            ("dead_recall_in_bin", "Dead recall"),
            ("mean_dead_prob", "Mean dead probability"),
        )
        for variable in sorted(pixel_df["variable"].dropna().astype(str).unique()):
            _lineplot_by_variable(
                pixel_df,
                variable,
                "bin_index",
                metrics,
                os.path.join(out_dir, f"pixel_{sanitize_name(variable)}.png"),
                f"Pixel Scale Effect - {variable}",
            )

    sample_df = maybe_read_csv(sample_csv)
    if sample_df is not None and {"variable", "bin_index"}.issubset(sample_df.columns):
        metrics = (
            ("detect_any_rate", "Detect-any rate"),
            ("detect_20pct_rate", "Detect-20pct rate"),
            ("mean_dead_recall", "Mean dead recall"),
        )
        for variable in sorted(sample_df["variable"].dropna().astype(str).unique()):
            _lineplot_by_variable(
                sample_df,
                variable,
                "bin_index",
                metrics,
                os.path.join(out_dir, f"sample_{sanitize_name(variable)}.png"),
                f"Sample Scale Effect - {variable}",
            )


def plot_final_validation(final_csv: str, out_dir: str):
    df = maybe_read_csv(final_csv)
    if df is None or "model" not in df.columns:
        return
    ensure_dir(out_dir)
    plt = _matplotlib()

    order_col = "model_rank" if "model_rank" in df.columns else "dead_binary_f1"
    ascending = order_col == "model_rank"
    df = df.sort_values(order_col, ascending=ascending)

    metrics = [col for col in ("dead_binary_f1", "dead_binary_precision", "dead_binary_recall", "dead_binary_iou") if col in df.columns]
    if metrics:
        plot_df = df[["model"] + metrics].set_index("model")
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(plot_df)), 6))
        x = np.arange(len(plot_df))
        width = 0.18
        offsets = np.linspace(-(len(metrics) - 1) / 2, (len(metrics) - 1) / 2, len(metrics)) * width
        colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2"]
        for idx, col in enumerate(metrics):
            ax.bar(x + offsets[idx], plot_df[col].astype(float).values, width=width, label=col, color=colors[idx % len(colors)], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df.index, rotation=25, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Score")
        ax.set_title("Final Holdout Validation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "final_validation_metrics.png"), dpi=160)
        plt.close(fig)

    if "decision_threshold" in df.columns:
        fig, ax = plt.subplots(figsize=(10, max(5, 0.55 * len(df) + 1)))
        ax.barh(df["model"], df["decision_threshold"], color="#4C72B0", alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("decision_threshold")
        ax.set_title("Validation Threshold by Model")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "final_validation_thresholds.png"), dpi=160)
        plt.close(fig)


def plot_clusters(cluster_csv_paths: Iterable[str], out_dir: str):
    ensure_dir(out_dir)
    plt = _matplotlib()
    for path in sorted(cluster_csv_paths):
        df = maybe_read_csv(path)
        if df is None or "size_class" not in df.columns:
            continue
        model_name = os.path.basename(path).replace("cluster_stats_", "").replace(".csv", "")
        summary = (
            df.groupby("size_class", dropna=False)
            .agg(
                n_clusters=("size_class", "size"),
                detection_rate=("detected", "mean"),
                mean_conf=("mean_conf", "mean"),
            )
            .reset_index()
        )
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(summary["size_class"], summary["n_clusters"], color="#4C72B0", alpha=0.85)
        axes[0].set_title(f"Cluster Count - {model_name}")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=25)

        axes[1].bar(summary["size_class"], summary["detection_rate"].astype(float), color="#55A868", alpha=0.85)
        axes[1].set_title(f"Cluster Detection Rate - {model_name}")
        axes[1].set_ylabel("Rate")
        axes[1].set_ylim(0, 1.02)
        axes[1].tick_params(axis="x", rotation=25)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"cluster_overview_{sanitize_name(model_name)}.png"), dpi=160)
        plt.close(fig)

        if {"area_m2", "detected", "mean_conf"}.issubset(df.columns):
            work = df.copy()
            work["area_m2"] = pd.to_numeric(work["area_m2"], errors="coerce")
            work["mean_conf"] = pd.to_numeric(work["mean_conf"], errors="coerce")
            work["detected"] = work["detected"].astype(float)
            work = work.dropna(subset=["area_m2", "mean_conf"])
            work = work[work["area_m2"] > 0]
            if not work.empty:
                n_bins = min(8, max(3, work["area_m2"].nunique()))
                try:
                    work["area_bin"] = pd.qcut(work["area_m2"], q=n_bins, duplicates="drop")
                except ValueError:
                    work["area_bin"] = pd.cut(work["area_m2"], bins=n_bins, duplicates="drop")
                curve = (
                    work.groupby("area_bin", observed=False)
                    .agg(
                        mean_area_m2=("area_m2", "mean"),
                        detection_rate=("detected", "mean"),
                        mean_conf=("mean_conf", "mean"),
                        cluster_count=("area_m2", "size"),
                    )
                    .reset_index(drop=True)
                    .sort_values("mean_area_m2")
                )

                fig, axes = plt.subplots(1, 2, figsize=(13, 5))
                axes[0].plot(curve["mean_area_m2"], curve["detection_rate"], marker="o", color="#55A868", linewidth=2)
                axes[0].set_xlabel("Mean area (m²)")
                axes[0].set_ylabel("Detection rate")
                axes[0].set_ylim(0, 1.02)
                axes[0].set_title(f"Area-Detection Curve - {model_name}")
                axes[0].grid(alpha=0.25, linestyle="--")

                axes[1].plot(curve["mean_area_m2"], curve["mean_conf"], marker="o", color="#C44E52", linewidth=2)
                axes[1].set_xlabel("Mean area (m²)")
                axes[1].set_ylabel("Mean confidence")
                axes[1].set_ylim(0, 1.02)
                axes[1].set_title(f"Area-Confidence Curve - {model_name}")
                axes[1].grid(alpha=0.25, linestyle="--")

                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"cluster_area_curve_{sanitize_name(model_name)}.png"), dpi=160)
                plt.close(fig)


def build_figures(input_dir: str, tag: str, figure_root: str):
    suffix = f"_{tag}" if tag else ""
    model_summary_csv = os.path.join(input_dir, f"model_summary{suffix}.csv")
    threshold_scan_csv = os.path.join(input_dir, f"threshold_scan{suffix}.csv")
    threshold_summary_csv = os.path.join(input_dir, f"threshold_summary{suffix}.csv")
    stratified_csv = os.path.join(input_dir, f"stratified_metrics{suffix}.csv")
    pixel_scale_csv = os.path.join(input_dir, f"pixel_scale_metrics{suffix}.csv")
    sample_scale_csv = os.path.join(input_dir, f"sample_scale_metrics{suffix}.csv")
    final_validation_csv = os.path.join(input_dir, f"final_validation{suffix}.csv")
    cluster_csvs = glob.glob(os.path.join(input_dir, "cluster_stats_*.csv"))

    figure_root = ensure_dir(figure_root)
    split_root = ensure_dir(os.path.join(figure_root, sanitize_name(tag)))
    dirs = {
        "models": ensure_dir(os.path.join(split_root, "models")),
        "thresholds": ensure_dir(os.path.join(split_root, "thresholds")),
        "stratified": ensure_dir(os.path.join(split_root, "stratified")),
        "scale": ensure_dir(os.path.join(split_root, "scale")),
        "validation": ensure_dir(os.path.join(split_root, "validation")),
        "clusters": ensure_dir(os.path.join(split_root, "clusters")),
        "maps": ensure_dir(os.path.join(split_root, "maps")),
    }

    if maybe_read_csv(model_summary_csv) is not None:
        plot_model_summary(model_summary_csv, dirs["models"])
    else:
        print(f"[Figure][warn] missing or empty model summary; skip model plots: {model_summary_csv}")

    if maybe_read_csv(threshold_scan_csv) is not None and maybe_read_csv(threshold_summary_csv) is not None:
        plot_thresholds(threshold_scan_csv, threshold_summary_csv, dirs["thresholds"])
    else:
        print("[Figure][warn] threshold CSV files missing or empty; skip threshold plots")

    if maybe_read_csv(stratified_csv) is not None:
        plot_stratified(stratified_csv, dirs["stratified"])
    else:
        print(f"[Figure][warn] missing or empty stratified metrics; skip stratified plots: {stratified_csv}")

    if maybe_read_csv(pixel_scale_csv) is not None and maybe_read_csv(sample_scale_csv) is not None:
        plot_scale(pixel_scale_csv, sample_scale_csv, dirs["scale"])
    else:
        print("[Figure][warn] scale-effect CSV files missing or empty; skip scale plots")

    if maybe_read_csv(final_validation_csv) is not None:
        plot_final_validation(final_validation_csv, dirs["validation"])
    else:
        print(f"[Figure][warn] missing or empty final validation; skip validation plots: {final_validation_csv}")

    if cluster_csvs:
        plot_clusters(cluster_csvs, dirs["clusters"])
    else:
        print(f"[Figure][warn] no cluster_stats_*.csv files found; skip cluster summary plots: {input_dir}")

    copy_matches(input_dir, "confusion-*.png", dirs["models"])
    copy_matches(input_dir, "cluster_summary_*.png", dirs["clusters"])
    copy_matches(input_dir, "*-cluster-map.png", dirs["clusters"])
    copy_matches(input_dir, "*-cluster-contour.png", dirs["clusters"])
    copy_matches(input_dir, "*-clusters.geojson", dirs["clusters"])
    copy_matches(input_dir, "*-pred.png", dirs["maps"])
    copy_matches(input_dir, "*-class1-conf.png", dirs["maps"])
    copy_matches(input_dir, "*-comparison.png", dirs["maps"])
    copy_matches(input_dir, "*-model-comparison.png", dirs["maps"])
    copy_matches(input_dir, "legend.png", dirs["maps"])

    print(f"[Figure] figure root -> {split_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing split CSV outputs")
    parser.add_argument("--tag", type=str, default=None, help="Split tag, e.g. analysis or holdout")
    parser.add_argument("--figure_root", type=str, default=config.FIGURE_DIR)
    args = parser.parse_args()

    tag = args.tag or os.path.basename(os.path.normpath(args.input_dir)) or "default"
    build_figures(args.input_dir, tag, args.figure_root)


if __name__ == "__main__":
    main()
