"""
Cluster-level analysis for dead-tree predictions.

Inputs:
- inference output directory containing `*-all-probs.npz`
- optional manifest CSV for split filtering

Outputs:
- `cluster_stats_<model>.csv`
- `cluster_summary_<model>.png`
- optional `*-cluster-map.png`
- optional `*-cluster-contour.png`
- optional `*-clusters.geojson` (pixel coordinates)

When `--model all` is used, outputs are generated per model, and an
additional aggregated `cluster_stats_all.csv` / `cluster_summary_all.png`
is also written.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config


CLUSTER_BINS = [
    (1, 1, "1px"),
    (2, 3, "2-3px"),
    (4, 10, "4-10px"),
    (11, 15, "11-15px"),
    (16, 20, "16-20px"),
    (21, 30, "21-30px"),
    (31, 40, "31-40px"),
    (41, 60, "41-60px"),
    (61, 100, "61-100px"),
    (101, 200, "101-200px"),
    (201, 500, "201-500px"),
    (501, 9999, "500+px"),
]

COLOR_BINS = [
    "#FFF3E0",
    "#FFE0B2",
    "#FFCC80",
    "#FFB74D",
    "#FFA726",
    "#FB8C00",
    "#E65100",
    "#C62828",
    "#6A1B9A",
    "#283593",
    "#1565C0",
    "#0D47A1",
]


def load_manifest_sample_ids(manifest_csv: Optional[str]) -> Optional[set]:
    if not manifest_csv:
        return None
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    sample_ids = set()
    with open(manifest_csv, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id:
                sample_ids.add(sample_id)
    if not sample_ids:
        raise RuntimeError(f"No sample_id found in manifest: {manifest_csv}")
    return sample_ids


def classify_cluster(area_px: int) -> str:
    for lo, hi, label in CLUSTER_BINS:
        if lo <= area_px <= hi:
            return label
    return "500+px"


def _sorted_model_names() -> List[str]:
    return sorted(config.SUPPORTED_MODELS, key=len, reverse=True)


def extract_model_name_from_npz(npz_path: str) -> str:
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    if not stem.endswith("-all-probs"):
        raise ValueError(f"Unexpected inference output name: {npz_path}")
    base = stem[: -len("-all-probs")]
    for model_name in _sorted_model_names():
        suffix = f"_{model_name}"
        if base.endswith(suffix):
            return model_name
    raise ValueError(f"Failed to parse model name from: {npz_path}")


def base_image_stem(npz_path: str, model_name: Optional[str] = None) -> str:
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    if stem.endswith("-all-probs"):
        stem = stem[: -len("-all-probs")]
    actual_model = model_name or extract_model_name_from_npz(npz_path)
    suffix = f"_{actual_model}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def extract_sample_id_from_npz(npz_path: str, model_name: Optional[str] = None) -> str:
    stem = base_image_stem(npz_path, model_name=model_name)
    return stem.split("_")[0]


def analyze_one(
    npz_path: str,
    conf_thr: float = 0.3,
    min_area: int = 1,
    model_name: Optional[str] = None,
    morph_open: bool = False,
) -> Tuple[List[dict], List[dict]]:
    data = np.load(npz_path)
    prob_map = data["prob_map"]
    dead_prob = prob_map[1]

    pred = prob_map[1:].argmax(axis=0).astype(np.uint8) + 1
    dead_mask = (pred == 1).astype(np.uint8) * 255
    if morph_open:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dead_mask = cv2.morphologyEx(dead_mask, cv2.MORPH_OPEN, kernel)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dead_mask, connectivity=8)
    px_size_m2 = config.GSD ** 2 if hasattr(config, "GSD") else 9.0
    actual_model = model_name or extract_model_name_from_npz(npz_path)
    image_stem = base_image_stem(npz_path, actual_model)

    records: List[dict] = []
    features: List[dict] = []
    for idx in range(1, n):
        area_px = int(stats[idx, cv2.CC_STAT_AREA])
        if area_px < min_area:
            continue

        obj_mask = labels == idx
        conf_vals = dead_prob[obj_mask]
        mean_conf = float(conf_vals.mean())
        max_conf = float(conf_vals.max())
        detected = mean_conf >= conf_thr

        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])

        records.append(
            {
                "model": actual_model,
                "cluster_id": idx,
                "image_stem": image_stem,
                "area_px": area_px,
                "area_m2": round(area_px * px_size_m2, 1),
                "mean_conf": round(mean_conf, 4),
                "max_conf": round(max_conf, 4),
                "size_class": classify_cluster(area_px),
                "detected": bool(detected),
                "cx": float(centroids[idx][0]),
                "cy": float(centroids[idx][1]),
                "bbox_xmin": x,
                "bbox_ymin": y,
                "bbox_w": w,
                "bbox_h": h,
                "source": os.path.basename(npz_path),
            }
        )

        obj_mask_u8 = obj_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(obj_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
            if contour.shape[0] >= 3:
                ring = [[float(px), float(py)] for px, py in contour.tolist()]
                ring.append(ring[0])
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "model": actual_model,
                            "cluster_id": idx,
                            "image_stem": image_stem,
                            "area_px": area_px,
                            "area_m2": round(area_px * px_size_m2, 1),
                            "mean_conf": round(mean_conf, 4),
                            "max_conf": round(max_conf, 4),
                            "detected": bool(detected),
                            "size_class": classify_cluster(area_px),
                            "bbox_xmin": x,
                            "bbox_ymin": y,
                            "bbox_w": w,
                            "bbox_h": h,
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [ring],
                        },
                    }
                )

    return records, features


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    order = [b[2] for b in CLUSTER_BINS]
    rows = []
    for label in order:
        sub = df[df["size_class"] == label]
        if sub.empty:
            continue
        rows.append(
            {
                "size_class": label,
                "n_clusters": len(sub),
                "pct_of_total": round(len(sub) / len(df) * 100, 1),
                "mean_area_px": round(sub["area_px"].mean(), 1),
                "mean_area_m2": round(sub["area_m2"].mean(), 1),
                "mean_conf": round(sub["mean_conf"].mean(), 4),
                "detection_rate": round(sub["detected"].astype(float).mean() * 100, 1),
                "total_area_m2": round(sub["area_m2"].sum(), 0),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(df: pd.DataFrame, summary: pd.DataFrame, out_path: str, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    order = [b[2] for b in CLUSTER_BINS]
    color_map = {b[2]: c for b, c in zip(CLUSTER_BINS, COLOR_BINS)}

    ax = axes[0]
    sub = summary[summary["size_class"].isin(order)].set_index("size_class")
    sub = sub.reindex([o for o in order if o in sub.index])
    bars = ax.bar(range(len(sub)), sub["n_clusters"], color=[color_map[o] for o in sub.index])
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Number of Clusters")
    ax.set_title("Cluster Count by Size Class")
    for bar, val in zip(bars, sub["n_clusters"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(int(val)), ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bars = ax.barh(range(len(sub)), sub["detection_rate"], color=[color_map[o] for o in sub.index])
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub.index, fontsize=8)
    ax.set_xlabel("Detection Rate (%)")
    ax.set_title("Detection Rate by Size Class\n(mean_conf >= threshold)")
    ax.set_xlim(0, 105)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, sub["detection_rate"]):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8)

    ax = axes[2]
    classes_present = [o for o in order if o in df["size_class"].values]
    data_by_class = [df[df["size_class"] == c]["mean_conf"].values for c in classes_present]
    if data_by_class:
        bp = ax.boxplot(data_by_class, patch_artist=True, notch=False)
        for patch, cls in zip(bp["boxes"], classes_present):
            patch.set_facecolor(color_map[cls])
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(classes_present) + 1))
        ax.set_xticklabels(classes_present, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean Confidence")
    ax.set_title("Confidence Distribution by Size Class")
    ax.axhline(0.3, color="orange", linestyle="--", linewidth=0.8, label="thr=0.3")
    ax.axhline(0.7, color="red", linestyle="--", linewidth=0.8, label="thr=0.7")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cluster_map(npz_path: str, records: List[dict], out_path: str, conf_thr: float):
    if not records:
        return
    data = np.load(npz_path)
    dead_prob = data["prob_map"][1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(dead_prob, cmap="Greens", vmin=0, vmax=1, alpha=0.6)

    contour_colors = {b[2]: c for b, c in zip(CLUSTER_BINS, ["#E8A87C", "#E27D60", "#C38D9E", "#41B3A3"] * 3)}
    for record in records:
        color = contour_colors.get(record["size_class"], "#E27D60")
        marker = "o" if record["detected"] else "x"
        ax.scatter(
            record["cx"],
            record["cy"],
            s=record["area_px"] * 2 + 10,
            c=color,
            marker=marker,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )

    patches = [mpatches.Patch(color=contour_colors.get(b[2], "#E27D60"), label=b[2]) for b in CLUSTER_BINS if b[2] in {r["size_class"] for r in records}]
    ax.legend(handles=patches, loc="upper right", fontsize=7)
    ax.set_title(f"{os.path.basename(npz_path)}\no=detected  x=below threshold ({conf_thr})", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_cluster_contours(npz_path: str, features: List[dict], out_path: str, conf_thr: float):
    if not features:
        return
    data = np.load(npz_path)
    dead_prob = data["prob_map"][1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(dead_prob, cmap="Greens", vmin=0, vmax=1, alpha=0.85)

    for feature in features:
        coords = np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float32)
        if coords.shape[0] < 4:
            continue
        detected = bool(feature["properties"]["detected"])
        color = "#E65100" if detected else "#283593"
        linewidth = 1.6 if detected else 1.1
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth, alpha=0.95)

    ax.set_title(f"{os.path.basename(npz_path)}\nCluster contours (mean_conf >= {conf_thr:.2f})", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def save_geojson(features: List[dict], out_path: str):
    if not features:
        return
    payload = {
        "type": "FeatureCollection",
        "name": os.path.splitext(os.path.basename(out_path))[0],
        "crs_note": "Pixel coordinates in image space; not georeferenced.",
        "features": features,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def write_outputs_for_group(
    model_name: str,
    npz_files: Iterable[str],
    out_dir: str,
    conf_thr: float,
    min_area: int,
    save_maps: bool,
    save_contours: bool,
    export_geojson: bool,
    morph_open: bool,
) -> pd.DataFrame:
    all_records: List[dict] = []
    geojson_count = 0
    npz_files = list(npz_files)
    print(f"[Cluster] {model_name}: {len(npz_files)} files")
    if morph_open:
        print("[Cluster][warn] morph_open enabled: 3x3 opening may remove 1px/small fragments")

    for npz_path in npz_files:
        records, features = analyze_one(
            npz_path,
            conf_thr=conf_thr,
            min_area=min_area,
            model_name=model_name,
            morph_open=morph_open,
        )
        all_records.extend(records)
        print(f"  {os.path.basename(npz_path):50s} -> {len(records)} clusters")

        if save_maps and records:
            plot_cluster_map(npz_path, records, npz_path.replace("-all-probs.npz", "-cluster-map.png"), conf_thr)
        if save_contours and features:
            plot_cluster_contours(npz_path, features, npz_path.replace("-all-probs.npz", "-cluster-contour.png"), conf_thr)
        if export_geojson and features:
            save_geojson(features, npz_path.replace("-all-probs.npz", "-clusters.geojson"))
            geojson_count += 1

    if not all_records:
        print(f"[Cluster] {model_name}: no clusters found")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(out_dir, f"cluster_stats_{model_name}.csv")
    df.to_csv(csv_path, index=False)

    summary = summarize(df)
    chart_path = os.path.join(out_dir, f"cluster_summary_{model_name}.png")
    plot_summary(df, summary, chart_path, title=f"Dead Tree Cluster Analysis - {model_name}")

    print(f"  Cluster stats -> {csv_path} ({len(df)} clusters)")
    print(f"  Cluster chart -> {chart_path}")
    if export_geojson:
        print(f"  GeoJSON files -> {geojson_count}")
    return df


def group_npz_files(npz_files: Iterable[str], requested_model: str) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    if requested_model == "all":
        for npz_path in npz_files:
            model_name = extract_model_name_from_npz(npz_path)
            grouped[model_name].append(npz_path)
    else:
        grouped[requested_model] = list(npz_files)
    return dict(grouped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.SUPPORTED_MODELS) + ["all"])
    parser.add_argument("--conf_thr", type=float, default=0.3, help="Confidence threshold for detection")
    parser.add_argument("--min_area", type=int, default=1, help="Minimum cluster area in pixels")
    parser.add_argument("--save_maps", action="store_true", help="Save per-image cluster maps")
    parser.add_argument("--save_contours", action="store_true", help="Save per-image contour overlays")
    parser.add_argument("--export_geojson", action="store_true", help="Export per-image cluster polygons as GeoJSON")
    parser.add_argument("--morph_open", action="store_true", help="Apply 3x3 opening before cluster extraction")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory override")
    parser.add_argument("--manifest_csv", type=str, default=None, help="Optional sample-level manifest filter")
    args = parser.parse_args()

    if not args.manifest_csv:
        raise RuntimeError("analyze_clusters.py requires --manifest_csv to avoid analyzing unintended inference outputs")

    out_dir = args.output_dir or config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if args.model == "all":
        pattern = os.path.join(out_dir, "*-all-probs.npz")
    else:
        pattern = os.path.join(out_dir, f"*_{args.model}-all-probs.npz")

    npz_files = sorted(glob.glob(pattern))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found: {pattern}")

    allowed_sample_ids = load_manifest_sample_ids(args.manifest_csv)
    if allowed_sample_ids is not None:
        filtered = []
        for npz_path in npz_files:
            try:
                model_name = extract_model_name_from_npz(npz_path)
            except ValueError as exc:
                raise RuntimeError(f"Cannot parse model name from npz file: {npz_path}") from exc
            if extract_sample_id_from_npz(npz_path, model_name=model_name) in allowed_sample_ids:
                filtered.append(npz_path)
        print(f"Manifest filter: {args.manifest_csv}")
        print(f"Matched npz files: {len(filtered)} / {len(npz_files)}")
        npz_files = filtered
        if not npz_files:
            raise RuntimeError("No inference outputs matched manifest sample_ids.")

    grouped = group_npz_files(npz_files, args.model)
    aggregate_frames = []
    for model_name in sorted(grouped.keys()):
        df = write_outputs_for_group(
            model_name=model_name,
            npz_files=grouped[model_name],
            out_dir=out_dir,
            conf_thr=args.conf_thr,
            min_area=args.min_area,
            save_maps=args.save_maps,
            save_contours=args.save_contours,
            export_geojson=args.export_geojson,
            morph_open=args.morph_open,
        )
        if not df.empty:
            aggregate_frames.append(df)

    if args.model == "all" and aggregate_frames:
        df_all = pd.concat(aggregate_frames, ignore_index=True)
        df_all.to_csv(os.path.join(out_dir, "cluster_stats_all.csv"), index=False)
        summary_all = summarize(df_all)
        plot_summary(
            df_all,
            summary_all,
            os.path.join(out_dir, "cluster_summary_all.png"),
            title="Dead Tree Cluster Analysis - all",
        )


if __name__ == "__main__":
    main()
