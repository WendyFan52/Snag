"""
计算训练集切片的逐波段均值和标准差 — compute_stats.py
-----------------------------------------------------------------
在 prepare_data.py 生成切片后运行一次，把输出结果粘贴到
config.py 的 _BAND_PROFILES 对应 img_mean / img_std 中。

注意：切片已包含遥感指数（共 IN_CHANNELS 个通道），
      统计的是所有通道（包括指数）的均值/标准差。

用法：
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
    Welford 在线算法：无需加载全部数据，内存友好。
    返回 (mean, std) — 各 in_channels 个 float32
    """
    files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    if not files:
        raise RuntimeError(f"未找到 .tif 文件: {img_dir}")

    if len(files) > max_samples:
        rng   = np.random.default_rng(42)
        files = rng.choice(files, max_samples, replace=False).tolist()

    print(f"统计 {len(files)} 张切片 (共 "
          f"{len(glob.glob(os.path.join(img_dir,'*.tif')))} 张)")
    print(f"通道数: {in_channels}\n")

    count = 0
    mean  = np.zeros(in_channels, dtype=np.float64)
    M2    = np.zeros(in_channels, dtype=np.float64)

    for fpath in tqdm(files, desc="计算统计值"):
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
    print("\n  将以下两行粘贴到 config.py 的对应配置块：\n")
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

    print(f"\n  {'通道':<20} {'均值':>10} {'标准差':>10}")
    print(f"  {'-'*42}")
    for i, (m, s) in enumerate(zip(mean, std)):
        name = ch_names[i] if i < len(ch_names) else f"ch{i}"
        print(f"  {name:<20} {m:>10.4f}   {s:>10.4f}")
    print()
