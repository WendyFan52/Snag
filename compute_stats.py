"""
Compute per-band training tile means and standard deviations - compute_stats.py
-----------------------------------------------------------------
Run once after prepare_data.py generates the tiles, then copy the results into
the corresponding img_mean / img_std entries in config.py's _BAND_PROFILES.

Tiles already include spectral indices (IN_CHANNELS channels in total).
      Statistics cover all channels, including the indices.

Usage:
  python compute_stats.py
  python compute_stats.py --max_samples 2000
-----------------------------------------------------------------
"""

import os, argparse, glob, json
import numpy as np
import rasterio
from tqdm import tqdm
import config
from input_preprocessing import assert_prepared_tile_tags


def compute_stats(img_dir:     str,
                  in_channels: int,
                  max_samples: int = 1000) -> tuple:
    """
    Welford's online algorithm avoids loading all data into memory.
    Return (mean, std), each containing in_channels float32 values.
    """
    files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    if not files:
        raise RuntimeError(f"No .tif files found: {img_dir}")

    if len(files) > max_samples:
        rng   = np.random.default_rng(42)
        files = rng.choice(files, max_samples, replace=False).tolist()

    print(f"Computing statistics for {len(files)} tiles (out of "
          f"{len(glob.glob(os.path.join(img_dir,'*.tif')))} total)")
    print(f"Channels: {in_channels}\n")

    count = 0
    mean  = np.zeros(in_channels, dtype=np.float64)
    M2    = np.zeros(in_channels, dtype=np.float64)

    for fpath in tqdm(files, desc="Computing statistics"):
        with rasterio.open(fpath) as src:
            data = src.read().astype(np.float64)   # (C, H, W)
            tags = src.tags()
        C = data.shape[0]
        if C != in_channels:
            raise RuntimeError(
                f"Unexpected channel count for stats file {fpath}: got {C}, "
                f"expected {in_channels}. Refuse to pad/truncate statistics input."
            )
        assert_prepared_tile_tags(tags, source=fpath)

        pixels = data.reshape(in_channels, -1).T    # (H*W, C)
        for px in pixels:
            count += 1
            delta  = px - mean
            mean  += delta / count
            M2    += delta * (px - mean)

    std = np.sqrt(M2 / max(count - 1, 1))
    return mean.astype(np.float32), std.astype(np.float32)


def _fmt(arr):
    return "(" + ", ".join(f"{v:.4f}" for v in arr) + ")"


def write_stats_json(mean: np.ndarray, std: np.ndarray, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "band_mode": config.BAND_MODE,
        "raw_channels": config.RAW_CHANNELS,
        "in_channels": config.IN_CHANNELS,
        "input_preprocessing": config.INPUT_PREPROCESSING_NAME,
        "planetscope_sr_scale": config.PLANETSCOPE_SR_SCALE,
        "img_mean": [float(v) for v in mean],
        "img_std": [float(v) for v in std],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=1000)
    args = parser.parse_args()

    mean, std = compute_stats(
        img_dir     = config.TRAIN_IMG_DIR,
        in_channels = config.IN_CHANNELS,
        max_samples = args.max_samples,
    )
    out_json = write_stats_json(mean, std, config.CHANNEL_STATS_JSON)

    print("\n" + "="*65)
    print(f"  BAND_MODE={config.BAND_MODE}  "
          f"RAW={config.RAW_CHANNELS}ch  IN_CHANNELS={config.IN_CHANNELS}ch")
    print("="*65)
    print("\n  Copy the following two lines into the corresponding config.py block:\n")
    print(f'        "img_mean": {_fmt(mean)},')
    print(f'        "img_std" : {_fmt(std)},')
    print(f"\n  stats_json: {out_json}")

    ch_names = []
    if config.RAW_CHANNELS == 4:
        ch_names = ["B", "G", "R", "NIR",
                    "NDVI", "EVI", "NDWI", "RI", "NGRDI"]
    elif config.RAW_CHANNELS == 8:
        ch_names = ["CoastalBlue", "Blue", "GreenI", "Green", "Yellow",
                    "Red", "RedEdge", "NIR",
                    "NDVI", "EVI", "NDWI", "RI", "NGRDI", "NDRE", "ChlRE"]

    print(f"\n  {'Channel':<20} {'Mean':>10} {'Std. dev.':>10}")
    print(f"  {'-'*42}")
    for i, (m, s) in enumerate(zip(mean, std)):
        name = ch_names[i] if i < len(ch_names) else f"ch{i}"
        print(f"  {name:<20} {m:>10.4f}   {s:>10.4f}")
    print()
