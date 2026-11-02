"""
Remote sensing index separability analysis - analyze_indices.py
=================================================================
Evaluate how well each index separates Dead(1) from Living(2):

Metrics:
  1. Jeffries-Matusita (JM) distance [0,2]
     Values near 2 indicate complete separation; values near 0 indicate complete overlap.
     > 1.8 = excellent, 1.0-1.8 = moderate, < 1.0 = poor

  2. Separability ratio = |mean_dead - mean_live| / (std_dead + std_live)
     Higher is better.

  3. Overlap ratio = overlapping area of the two distributions (0=none, 1=complete).
     Lower is better.

Usage:
  python analyze_indices.py --band 4band   # Analyze four-band imagery
  python analyze_indices.py --band 8band   # Analyze eight-band imagery
  python analyze_indices.py                # Analyze both

Inputs: TIF tiles in tiles_{band}/images/
      PNG labels in tiles_{band}/masks/
=================================================================
"""

import os, glob, argparse
import numpy as np
from PIL import Image
import rasterio
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from spectral_indices import (compute_indices_4band, compute_indices_8band,
                               INDEX_NAMES_4BAND, INDEX_NAMES_8BAND)
from input_preprocessing import assert_prepared_tile_tags


def jm_distance(p1: np.ndarray, p2: np.ndarray,
                n_bins: int = 100) -> float:
    """
    Jeffries-Matusita distance via histogram approximation.
    JM = 2*(1 - exp(-B/8)) where B is Bhattacharyya coefficient.
    Result in [0, 2].
    """
    all_vals = np.concatenate([p1, p2])
    lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    if hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    h1 = np.histogram(p1, bins=bins)[0].astype(float)
    h2 = np.histogram(p2, bins=bins)[0].astype(float)
    h1 /= h1.sum() + 1e-12
    h2 /= h2.sum() + 1e-12
    # Bhattacharyya coefficient
    B_coeff = np.sum(np.sqrt(h1 * h2))
    jm = 2.0 * (1.0 - B_coeff)
    return float(np.clip(jm, 0, 2))


def overlap_ratio(p1: np.ndarray, p2: np.ndarray,
                  n_bins: int = 100) -> float:
    """Distribution overlap area (0=no overlap, 1=complete overlap)."""
    all_vals = np.concatenate([p1, p2])
    lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    if hi <= lo:
        return 1.0
    bins = np.linspace(lo, hi, n_bins + 1)
    h1 = np.histogram(p1, bins=bins)[0].astype(float)
    h2 = np.histogram(p2, bins=bins)[0].astype(float)
    h1 /= h1.sum() + 1e-12
    h2 /= h2.sum() + 1e-12
    return float(np.sum(np.minimum(h1, h2)))


def collect_pixels(img_dir: str, mask_dir: str,
                   raw_ch: int,
                   max_samples: int = 5000) -> dict:
    """
    Randomly sample Dead(1) and Living(2) pixels from tiles.
    Return a dictionary of pixel values for each index.
    """
    tif_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    if not tif_files:
        raise RuntimeError(f"No TIF files in {img_dir}")

    # Compute spectral indices
    if raw_ch == 4:
        compute_fn = compute_indices_4band
        names = INDEX_NAMES_4BAND
    else:
        compute_fn = compute_indices_8band
        names = INDEX_NAMES_8BAND

    # Also include raw bands
    if raw_ch == 4:
        band_names = ["B", "G", "R", "NIR"]
    else:
        band_names = ["CoastalBlue", "Blue", "GreenI", "Green", "Yellow", "Red", "RedEdge", "NIR"]
    all_names = band_names + names

    pixels = {name: {"dead": [], "live": []} for name in all_names}

    rng = np.random.default_rng(42)
    files = rng.choice(tif_files,
                       min(len(tif_files), max_samples // 50),
                       replace=False).tolist()

    for tif_path in tqdm(files, desc="Sampling pixels"):
        stem = os.path.splitext(os.path.basename(tif_path))[0]
        mask_path = os.path.join(mask_dir, stem + ".png")
        if not os.path.exists(mask_path):
            continue

        with rasterio.open(tif_path) as src:
            data = src.read().astype(np.float32)
            assert_prepared_tile_tags(src.tags(), source=tif_path)

        mask = np.array(Image.open(mask_path))

        C = data.shape[0]
        # Handle tiles that already have indices appended
        raw_data = data[:raw_ch] if C >= raw_ch else data

        # Compute indices
        indices = compute_fn(raw_data)   # (N_idx, H, W)

        # All channels: raw + indices
        all_data = np.concatenate([raw_data, indices], axis=0)  # (raw+idx, H, W)

        dead_mask = (mask == 1)
        live_mask = (mask == 2)

        if dead_mask.sum() < 5 or live_mask.sum() < 5:
            continue

        for i, name in enumerate(all_names):
            if i < all_data.shape[0]:
                pixels[name]["dead"].extend(all_data[i][dead_mask].tolist())
                pixels[name]["live"].extend(all_data[i][live_mask].tolist())

    # Subsample to max_samples
    for name in all_names:
        for cls in ["dead", "live"]:
            arr = np.array(pixels[name][cls])
            if len(arr) > max_samples:
                idx = rng.choice(len(arr), max_samples, replace=False)
                pixels[name][cls] = arr[idx]
            else:
                pixels[name][cls] = arr

    return pixels, all_names


def analyze_and_plot(pixels: dict, names: list,
                     band_mode: str, out_dir: str):
    """Compute separability metrics and plot the results."""
    results = []
    for name in names:
        dead = np.array(pixels[name]["dead"])
        live = np.array(pixels[name]["live"])
        if len(dead) < 10 or len(live) < 10:
            continue

        jm   = jm_distance(dead, live)
        sep  = (abs(dead.mean() - live.mean()) /
                max(dead.std() + live.std(), 1e-8))
        ovlp = overlap_ratio(dead, live)

        results.append({
            "name":    name,
            "jm":      jm,
            "sep":     sep,
            "overlap": ovlp,
            "mean_dead": dead.mean(),
            "mean_live": live.mean(),
            "std_dead":  dead.std(),
            "std_live":  live.std(),
        })

    # Sort by JM distance
    results.sort(key=lambda x: -x["jm"])

    # Print table
    print(f"\n{'='*72}")
    print(f"  {band_mode} Index Separability: Dead(1) vs Living(2)")
    print(f"{'='*72}")
    print(f"  {'Index':<12} {'JM dist':>8} {'Sep ratio':>10} "
          f"{'Overlap':>8} {'mean_dead':>10} {'mean_live':>10}")
    print(f"  {'-'*70}")
    for r in results:
        grade = "excellent" if r["jm"] > 1.5 else "good" if r["jm"] > 1.0 else "poor"
        print(f"  {r['name']:<12} {r['jm']:>8.4f} {r['sep']:>10.4f} "
              f"{r['overlap']:>8.4f} {r['mean_dead']:>10.4f} "
              f"{r['mean_live']:>10.4f}  {grade}")
    print(f"{'='*72}")
    print("  JM>1.5: excellent; otherwise JM>1.0: good; otherwise: poor\n")

    # Plot: bar chart of JM distances
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    names_sorted = [r["name"] for r in results]
    jm_vals      = [r["jm"]   for r in results]
    colors = ["#3B6D11" if jm > 1.5 else
              "#EF9F27" if jm > 1.0 else
              "#E24B4A" for jm in jm_vals]

    bars = ax1.barh(names_sorted[::-1], jm_vals[::-1],
                    color=colors[::-1], edgecolor="white")
    ax1.axvline(1.5, color="green", linestyle="--",
                linewidth=1, label="1.5 excellent")
    ax1.axvline(1.0, color="orange", linestyle="--",
                linewidth=1, label="1.0 good")
    ax1.set_xlabel("JM Distance (higher = better separability)")
    ax1.set_title(f"{band_mode}: Dead vs Living JM Distance")
    ax1.set_xlim(0, 2.1)
    ax1.legend(fontsize=8)

    # Add value labels
    for bar, val in zip(bars, jm_vals[::-1]):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=7)

    # Plot: distribution overlay for top 6 indices
    top6 = results[:6]
    fig2, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig2.suptitle(f"{band_mode}: Top 6 Indices — Dead vs Living Distribution",
                  fontsize=12)

    for ax, r in zip(axes.ravel(), top6):
        dead = np.array(pixels[r["name"]]["dead"])
        live = np.array(pixels[r["name"]]["live"])
        lo = min(np.percentile(dead, 1), np.percentile(live, 1))
        hi = max(np.percentile(dead, 99), np.percentile(live, 99))
        bins = np.linspace(lo, hi, 60)
        ax.hist(dead, bins=bins, color="#993C1D", alpha=0.6,
                density=True, label=f"Dead  μ={r['mean_dead']:.3f}")
        ax.hist(live, bins=bins, color="#3B6D11", alpha=0.6,
                density=True, label=f"Live  μ={r['mean_live']:.3f}")
        ax.set_title(f"{r['name']}  JM={r['jm']:.3f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("Index value")

    fig2.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out1 = os.path.join(out_dir, f"index_separability_{band_mode}.png")
    out2 = os.path.join(out_dir, f"index_distributions_{band_mode}.png")
    fig.tight_layout()
    fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)
    fig2.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig2)
    print(f"  Charts saved:")
    print(f"    {out1}")
    print(f"    {out2}")

    return results


def main(band_mode: str):
    # Temporarily override band mode in config
    import importlib
    orig_mode = config.BAND_MODE
    if band_mode != orig_mode:
        # Reload with correct mode — use config attributes directly
        raw_ch  = 4 if band_mode == "4band" else 8
        img_dir = config._TIF_DIRS[band_mode]
        data_root = os.path.join(config.RESEARCH_DIR, f"tiles_{band_mode}")
        img_dir  = os.path.join(data_root, "images")
        mask_dir = os.path.join(data_root, "masks")
        out_dir  = os.path.join(config.RUN_ROOT,
                                f"inference_outputs_{band_mode}", "data_check")
    else:
        raw_ch   = config.RAW_CHANNELS
        img_dir  = config.TRAIN_IMG_DIR
        mask_dir = config.TRAIN_MASK_DIR
        out_dir  = config.CHECK_REPORT_DIR

    print(f"\nAnalyzing {band_mode} indices...")
    print(f"  Images : {img_dir}")
    print(f"  Masks  : {mask_dir}")

    pixels, names = collect_pixels(img_dir, mask_dir, raw_ch)
    results = analyze_and_plot(pixels, names, band_mode, out_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--band",
                        choices=["4band", "8band", "both"],
                        default="both")
    args = parser.parse_args()

    if args.band == "both":
        for b in ["4band", "8band"]:
            try:
                main(b)
            except Exception as e:
                print(f"  {b} failed: {e}")
    else:
        main(args.band)
