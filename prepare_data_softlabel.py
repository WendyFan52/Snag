"""
软标签切片生成脚本 — prepare_data_softlabel.py
=================================================================
替代 prepare_data.py，生成三种输出：

  images/{tile}.tif     — 影像切片（与 prepare_data.py 相同）
  masks/{tile}.png      — 硬标签 mask（与 prepare_data.py 相同，用于评估）
  softlabels/{tile}.npy — 软标签 (NUM_CLASSES+1, H, W) float32
  lci/{tile}.npy        — LCI 权重图 (H, W) float32

什么是软标签（Soft Label）？
-----------------------------------------------------------------
普通硬标签：每个像元只有一个类别，例如 mask[y,x] = 1（死树）
软标签：每个像元有一组概率，例如 softlabel[:,y,x] = [0,0.65,0.35,0,0,0]
        表示该像元 65% 是死树，35% 是活树

为什么需要软标签？
-----------------------------------------------------------------
PlanetScope 3m 像元比单株树冠大。图斑边界处的像元同时包含多个类别。
  - 硬标签：强制给边界像元分配一个类别 → 引入系统性错误
  - 软标签：按实际面积比例分配 → 训练信号更真实

计算方法：
-----------------------------------------------------------------
对每个像元，用 SOFT_LABEL_SUBGRID × SOFT_LABEL_SUBGRID 的子格网
细分该像元，统计各类别覆盖的子格数，换算为面积比例。

Dead/Living 边界用更细的格网（subgrid=20），因为这对最难区分。
其他边界用标准格网（subgrid=5~8）。

LCI（Label Confidence Index）：
-----------------------------------------------------------------
LCI[pixel] = max(class_coverage_ratio)
  LCI=1.0 → 像元完全属于一个类（纯净）
  LCI=0.65 → 主类占 65%，混合像元
  LCI=0.3  → 高度混合

训练时：loss_weight[pixel] = LCI[pixel]
纯净像元对训练贡献更大，混合像元贡献小但不为零。

损失函数：
-----------------------------------------------------------------
软标签训练使用 KL 散度替代交叉熵：
  soft_CE = -sum(softlabel * log(softmax(logits)))
这等价于用软标签作为目标分布。

用法：
  python prepare_data_softlabel.py
  python prepare_data_softlabel.py --dry_run
=================================================================
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds as tfm_from_bounds
    import geopandas as gpd
    from PIL import Image
    from pyproj import CRS as ProjCRS
except ImportError as e:
    sys.exit(f"Missing: {e}\npip install rasterio geopandas Pillow pyproj")

import config
from input_preprocessing import preprocessing_tags, sr_to_feature_stack

# 软标签输出目录（新增）
SOFTLABEL_DIR = os.path.join(config.DATA_ROOT, "softlabels")
os.makedirs(SOFTLABEL_DIR, exist_ok=True)

VALID_CLASSES = list(range(1, config.NUM_CLASSES + 1))
N_OUT = config.NUM_CLASSES + 1   # 包含 ignore slot 0


# =================================================================
# 工具函数（与 prepare_data.py 共用）
# =================================================================
def find_shp_for_tif(tif_path: str) -> Optional[str]:
    stem      = Path(tif_path).stem
    sample_id = stem.split("_")[0]
    matches   = glob.glob(
        os.path.join(config.RAW_SHP_DIR, f"{sample_id}_*.shp"))
    if matches:
        return matches[0]
    exact = os.path.join(config.RAW_SHP_DIR, sample_id + ".shp")
    return exact if os.path.exists(exact) else None


def align_crs(gdf, tif_crs):
    """用 pyproj 精确比较 CRS，不同则重投影。"""
    if gdf.crs is None:
        raise RuntimeError("SHP CRS is missing; refuse to assume it matches TIF")
    try:
        if not ProjCRS.from_user_input(gdf.crs).equals(
               ProjCRS.from_user_input(tif_crs)):
            return gdf.to_crs(tif_crs)
    except Exception:
        if str(gdf.crs) != str(tif_crs):
            return gdf.to_crs(tif_crs)
    return gdf


def read_and_prepare(tif_path: str) -> np.ndarray:
    """读取 TIF → 固定比例转换为反射率 → 追加遥感指数 → (C, H, W) float32"""
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
    raw_ch = config.RAW_CHANNELS
    C = data.shape[0]
    if C != raw_ch:
        raise RuntimeError(
            f"Unexpected raw channel count for {tif_path}: got {C}, "
            f"expected {raw_ch} for BAND_MODE={config.BAND_MODE}. "
            "Refuse to pad/truncate bands during soft-label tile preparation."
        )
    return sr_to_feature_stack(data, source=tif_path)


def save_tile_tif(data: np.ndarray, out_path: str):
    C, H, W = data.shape
    with rasterio.open(out_path, "w", driver="GTiff",
                       count=C, height=H, width=W,
                       dtype=rasterio.float32, compress="lzw") as dst:
        dst.update_tags(**preprocessing_tags())
        for i in range(C):
            dst.write(data[i], i + 1)


def load_and_prepare_shapes(shp_path: str, tif_path: str):
    """
    加载 SHP，统一 CRS，过滤非法类别，
    按类别降序排列（Dead=1 最后，覆盖其他类别）。
    """
    with rasterio.open(tif_path) as src:
        tif_crs   = src.crs
        transform = src.transform
        H, W      = src.height, src.width

    gdf = gpd.read_file(shp_path)
    field = config.SHP_CLASS_FIELD

    if field not in gdf.columns:
        raise KeyError(f"Class field {field} not found in {shp_path}")

    gdf = gdf[gdf[field].notna() & ~gdf.geometry.is_empty].copy()
    gdf = align_crs(gdf, tif_crs)
    gdf[field] = gdf[field].astype(int)
    gdf = gdf[gdf[field].isin(VALID_CLASSES)]

    shapes = [(geom, int(val))
              for geom, val in zip(gdf.geometry, gdf[field])
              if geom is not None and not geom.is_empty]
    if not shapes:
        raise RuntimeError(f"No rasterizable valid class geometries in SHP: {shp_path}")

    # Dead(1) 最后绘制 → 覆盖其他类别
    shapes = sorted(shapes, key=lambda x: -x[1])
    return shapes, transform, H, W


# =================================================================
# Current Class_Int already uses the 1-5 framework encoding.
# =================================================================


# =================================================================
# 核心：软标签计算
# =================================================================
def compute_softlabel_map(shapes: list,
                          transform,
                          H: int, W: int) -> tuple:
    """
    向量化软标签计算 — 用整图高分辨率栅格化替代逐像元循环。

    原方法：每个像元单独调用 rasterize() → 百万次调用 → 极慢
    新方法：
      1. 将整张影像一次性栅格化到 (H*sg × W*sg) 高分辨率
      2. reshape 成 (H, sg, W, sg)，统计各类别占子格数比例
      3. 整张影像的软标签在 1 次 rasterize() 调用中完成

    速度提升：从数小时 → 每样本数秒

    Dead/Living 边界精度处理：
      对 Dead(1) 和 Living(2) 各自的图斑，用更细的 sg=20 分别做一次
      高分辨率栅格化，修正这两类的覆盖比例（最重要的边界）。
      其他类别用 sg=base（默认5）。

    返回：
      softlabel (N_OUT, H, W) float32
      lci_map   (H, W)        float32
    """
    base_sg = getattr(config, 'SOFT_LABEL_SUBGRID', 5)
    dl_sg   = config.get_local_subgrid(1, 2, base_sg)  # Dead/Living精细格，默认20

    # ── 1. 粗栅格化定位标注区域 ────────────────────────────────
    hard_mask = rasterize(shapes, (H, W), transform=transform,
                          fill=0, dtype=np.int32, all_touched=False)
    soft_mask = rasterize(shapes, (H, W), transform=transform,
                          fill=0, dtype=np.int32, all_touched=True)

    softlabel = np.zeros((N_OUT, H, W), dtype=np.float32)
    lci_map   = np.zeros((H, W),        dtype=np.float32)

    # 未标注像元 → ignore slot
    softlabel[0, soft_mask == 0] = 1.0

    if not shapes:
        return softlabel, lci_map

    # ── 2. 整图高分辨率栅格化（base_sg 倍）──────────────────────
    # 一次调用替代所有逐像元 rasterize()
    from rasterio.transform import Affine

    def make_hires_transform(transform, sg):
        """将 transform 缩放 sg 倍得到高分辨率 transform"""
        return Affine(transform.a / sg, transform.b, transform.c,
                      transform.d, transform.e / sg, transform.f)

    # base_sg 分辨率：覆盖除 Dead/Living 之外的所有类别
    hires_tfm_base = make_hires_transform(transform, base_sg)
    hires_mask_base = rasterize(shapes,
                                (H * base_sg, W * base_sg),
                                transform=hires_tfm_base,
                                fill=0, dtype=np.int32,
                                all_touched=True)

    # Dead/Living 精细分辨率（dl_sg 倍）
    hires_tfm_dl = make_hires_transform(transform, dl_sg)
    hires_mask_dl = rasterize(shapes,
                              (H * dl_sg, W * dl_sg),
                              transform=hires_tfm_dl,
                              fill=0, dtype=np.int32,
                              all_touched=True)

    # ── 3. 聚合：高分辨率 → 原始分辨率的覆盖比例 ──────────────
    # reshape: (H*sg, W*sg) → (H, sg, W, sg) → mean over sg dims
    def coverage(hires: np.ndarray, sg: int, cls: int) -> np.ndarray:
        """返回 (H, W) float32，每个像元中 cls 类的覆盖比例"""
        return ((hires == cls)
                .reshape(H, sg, W, sg)
                .mean(axis=(1, 3))
                .astype(np.float32))

    # 计算各类别覆盖比例
    # Dead(1) 和 Living(2) 用精细格，其他用粗格
    raw_cov = {}
    for cls in VALID_CLASSES:
        if cls in (1, 2):
            raw_cov[cls] = coverage(hires_mask_dl, dl_sg, cls)
        else:
            raw_cov[cls] = coverage(hires_mask_base, base_sg, cls)

    # ── 4. 过滤噪声、归一化、填充 softlabel ──────────────────────
    min_ratio = getattr(config, 'SOFT_LABEL_MIN_RATIO', 0.05)

    # 叠加所有类别的覆盖比例
    total_cov = np.zeros((H, W), dtype=np.float32)
    for cls in VALID_CLASSES:
        cov = raw_cov[cls].copy()
        cov[cov < min_ratio] = 0.0   # 过滤噪声
        if cls > 0:
            softlabel[cls] += cov
        total_cov += cov

    # 归一化（只对有标注的像元）
    labeled = (soft_mask > 0) & (total_cov > 0)
    for c in range(N_OUT):
        softlabel[c, labeled] /= total_cov[labeled]

    # 保底：有标注但所有类别都被过滤掉 → 用硬标签
    no_coverage = (soft_mask > 0) & (total_cov == 0)
    if no_coverage.any():
        for py, px in zip(*np.where(no_coverage)):
            cls = int(hard_mask[py, px]) if hard_mask[py, px] > 0 else int(soft_mask[py, px])
            if cls > 0:
                softlabel[cls, py, px] = 1.0

    # 未标注像元确保 ignore slot = 1
    softlabel[0, soft_mask == 0] = 1.0

    # ── 5. LCI = 最大类别覆盖比例（纯净度）──────────────────────
    # 取 softlabel[1:] 的最大值（排除 ignore slot 0）
    lci_map = softlabel[1:].max(axis=0)

    return softlabel, lci_map




# =================================================================
# 切片保存
# =================================================================
def tile_and_save(img_data:   np.ndarray,
                  hard_mask:  np.ndarray,
                  softlabel:  np.ndarray,
                  lci_map:    np.ndarray,
                  sample_id:  str,
                  dry_run:    bool = False) -> dict:
    ts   = config.TILE_SIZE
    st   = config.TILE_STRIDE
    mrat = config.MIN_LABEL_RATIO
    C, H, W = img_data.shape
    stats = dict(total=0, saved=0, skipped=0)

    lci_dir = getattr(config, "TRAIN_LCI_DIR",
                      os.path.join(config.DATA_ROOT, "lci"))
    if not dry_run:
        os.makedirs(lci_dir, exist_ok=True)

    for y in range(0, H - ts + 1, st):
        for x in range(0, W - ts + 1, st):
            stats["total"] += 1
            tile_mask = hard_mask[y:y+ts, x:x+ts]

            # 过滤低标注率切片
            labeled_ratio = (tile_mask > 0).sum() / (ts * ts)
            if labeled_ratio < mrat:
                stats["skipped"] += 1
                continue

            if dry_run:
                stats["saved"] += 1
                continue

            fname   = f"{sample_id}_y{y:04d}_x{x:04d}"
            img_out  = os.path.join(config.TRAIN_IMG_DIR,  fname + ".tif")
            mask_out = os.path.join(config.TRAIN_MASK_DIR, fname + ".png")
            soft_out = os.path.join(SOFTLABEL_DIR,          fname + ".npy")
            lci_out  = os.path.join(lci_dir,               fname + ".npy")

            save_tile_tif(img_data[:, y:y+ts, x:x+ts], img_out)
            Image.fromarray(tile_mask.astype(np.uint8)).save(mask_out)
            np.save(soft_out, softlabel[:, y:y+ts, x:x+ts])
            np.save(lci_out,  lci_map[y:y+ts, x:x+ts].astype(np.float32))

            stats["saved"] += 1

    return stats


# =================================================================
# 主流程
# =================================================================
def main(dry_run: bool = False):
    if not dry_run:
        config.ensure_runtime_dirs()
        os.makedirs(config.TRAIN_IMG_DIR, exist_ok=True)
        os.makedirs(config.TRAIN_MASK_DIR, exist_ok=True)
        os.makedirs(SOFTLABEL_DIR, exist_ok=True)
        os.makedirs(getattr(config, "TRAIN_LCI_DIR", os.path.join(config.DATA_ROOT, "lci")), exist_ok=True)

    tif_paths = sorted(glob.glob(
        os.path.join(config.RAW_TIF_DIR, "*.tif")))
    if not tif_paths:
        sys.exit(f"No TIF found: {config.RAW_TIF_DIR}")

    print(f"prepare_data_softlabel.py")
    print(f"  BAND_MODE = {config.BAND_MODE}  "
          f"RAW={config.RAW_CHANNELS}ch -> IN={config.IN_CHANNELS}ch")
    print(f"  TILE_SIZE={config.TILE_SIZE}  "
          f"STRIDE={config.TILE_STRIDE}  "
          f"MIN_LABEL_RATIO={config.MIN_LABEL_RATIO:.0%}")
    print(f"  SOFT_LABEL_SUBGRID={getattr(config,'SOFT_LABEL_SUBGRID',5)}  "
          f"(Dead/Living boundary uses subgrid="
          f"{config.get_local_subgrid(1,2)})")
    print(f"  SOFT_LABEL_CLASSES={getattr(config,'SOFT_LABEL_CLASSES',[1,2])}")
    print(f"  Output: {config.DATA_ROOT}")
    if dry_run:
        print("  [DRY RUN] No files will be written\n")
    else:
        print()

    total_saved = total_skipped = 0
    failed = []

    for tif_path in tqdm(tif_paths, desc="Processing"):
        sample_id = Path(tif_path).stem.split("_")[0]
        shp_path  = find_shp_for_tif(tif_path)

        if shp_path is None:
            raise FileNotFoundError(f"No matching SHP found for sample_id={sample_id}, tif={tif_path}")

        try:
            # 1. 加载图斑
            shapes, transform, H, W = load_and_prepare_shapes(
                shp_path, tif_path)
            if not shapes:
                raise RuntimeError(f"No valid shapes for sample_id={sample_id}, shp={shp_path}")

            # 2. 硬标签（用于 mask 和评估）
            hard_mask = rasterize(shapes, (H, W), transform=transform,
                                  fill=0, dtype=np.int32, all_touched=False)
            hard_mask = hard_mask.astype(np.uint8)

            # 3. 软标签 + LCI（核心计算）
            softlabel, lci_map = compute_softlabel_map(
                shapes, transform, H, W)

            # 4. 影像
            img_data = read_and_prepare(tif_path)

            # 5. 切片保存
            s = tile_and_save(img_data, hard_mask, softlabel,
                              lci_map, sample_id, dry_run)

            tqdm.write(f"  {sample_id}: total={s['total']}  "
                       f"saved={s['saved']}  skipped={s['skipped']}")
            total_saved   += s["saved"]
            total_skipped += s["skipped"]

        except Exception as e:
            tqdm.write(f"  ERROR {sample_id}: {e}")
            failed.append(sample_id)
            raise

    print(f"\n{'='*55}")
    print(f"  {'[DRY RUN] ' if dry_run else ''}Done")
    print(f"  Saved: {total_saved}  Skipped: {total_skipped}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Images:     {config.TRAIN_IMG_DIR}")
    print(f"  Masks:      {config.TRAIN_MASK_DIR}")
    print(f"  Softlabels: {SOFTLABEL_DIR}")
    print(f"  LCI maps:   {getattr(config,'TRAIN_LCI_DIR','')}")
    print(f"{'='*55}")
    if total_saved <= 0:
        raise RuntimeError("No training tiles were saved. Check RAW_TIF_DIR, RAW_SHP_DIR, and MIN_LABEL_RATIO.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate soft-label tiles for dead tree detection")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
