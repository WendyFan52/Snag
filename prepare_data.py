"""
数据预处理脚本 — prepare_data.py
-----------------------------------------------------------------
流程:
  1. 读取 TIF + SHP (Class_Int字段)
  2. SHP 栅格化为 Mask（像素值=类别ID，0=未标注）
  3. 将 PlanetScope SR 存储值按固定比例转为反射率，追加遥感指数
  4. 滑窗切片，过滤低标注率切片
  5. 输出 images/*.tif (多通道) + masks/*.png (单通道)

依赖：pip install rasterio geopandas numpy Pillow tqdm

用法：
  python prepare_data.py
  python prepare_data.py --dry_run   # 只统计，不写文件
-----------------------------------------------------------------
"""

import os, sys, glob, argparse, warnings
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

try:
    import rasterio
    from rasterio.features import rasterize
    import geopandas as gpd
    from PIL import Image
except ImportError as e:
    sys.exit(f"[错误] 缺少依赖: {e}\n请运行: pip install rasterio geopandas Pillow tqdm")

import config
from input_preprocessing import preprocessing_tags, sr_to_feature_stack

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# -----------------------------------------------------------------
# SHP 匹配
# -----------------------------------------------------------------
def find_shp_for_tif(tif_path: str, shp_dir: str) -> Optional[str]:
    stem      = Path(tif_path).stem
    sample_id = stem.split("_")[0]
    matches   = glob.glob(os.path.join(shp_dir, f"{sample_id}_*.shp"))
    if matches:
        return matches[0]
    exact = os.path.join(shp_dir, stem + ".shp")
    return exact if os.path.exists(exact) else None


# -----------------------------------------------------------------
# SHP 栅格化
# -----------------------------------------------------------------
def rasterize_shp(shp_path: str, ref_tif_path: str) -> np.ndarray:
    """
    返回 (H, W) int32 mask，0=未覆盖(未标注)

    栅格化优先级：类别值小的图斑优先（最后绘制，覆盖之前的）。
    Dead_trees(1) 最后绘制，覆盖 Living_trees(2) 及其他所有类别。
    这符合"先标注所有树为活树，再在死树位置叠加死树图斑"的标注工作流。

    CRS 处理：
    - 用 pyproj.CRS.equals() 精确比较（避免字符串表示不同导致误判）
    - SHP 无 CRS 定义时假设与 TIF 一致（发出警告）
    """
    with rasterio.open(ref_tif_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        tif_crs   = src.crs

    gdf = gpd.read_file(shp_path)
    field = config.SHP_CLASS_FIELD

    if field not in gdf.columns:
        raise KeyError(f"SHP missing field '{field}'. Available: {list(gdf.columns)}")

    # CRS alignment: use pyproj for precise comparison
    if gdf.crs is None:
        raise RuntimeError(f"SHP CRS is missing; refuse to assume it matches TIF: {shp_path}")
    elif tif_crs is not None:
        try:
            from pyproj import CRS as ProjCRS
            if not ProjCRS.from_user_input(gdf.crs).equals(
                   ProjCRS.from_user_input(tif_crs)):
                gdf = gdf.to_crs(tif_crs)
        except Exception:
            # Fallback to string comparison
            if str(gdf.crs) != str(tif_crs):
                gdf = gdf.to_crs(tif_crs)

    gdf = gdf[gdf[field].notna() & ~gdf.geometry.is_empty]
    if len(gdf) == 0:
        raise RuntimeError(f"No valid labeled geometries in SHP: {shp_path}")

    gdf[field] = gdf[field].astype(int)
    # Filter to valid classes only (exclude class=0 which conflicts with IGNORE_INDEX)
    valid_classes = set(range(1, config.NUM_CLASSES + 1))
    gdf = gdf[gdf[field].isin(valid_classes)]

    shapes = [(geom, int(val))
              for geom, val in zip(gdf.geometry, gdf[field])
              if geom is not None and not geom.is_empty]

    if not shapes:
        raise RuntimeError(f"No rasterizable valid class geometries in SHP: {shp_path}")

    # CRITICAL: Sort so that class 1 (Dead_trees) is rendered LAST.
    # rasterize() draws shapes in list order; later shapes override earlier ones.
    # Dead(1) must override Living(2) since the annotation workflow
    # typically labels all trees as Living first, then adds Dead on top.
    # General rule: sort descending by class value so class 1 comes last.
    shapes = sorted(shapes, key=lambda x: -x[1])

    mask = rasterize(shapes, out_shape=out_shape,
                     transform=transform, fill=0,
                     dtype=np.int32, all_touched=False)

    # SHP Class_Int already uses the current 1-5 encoding.
    return mask


# -----------------------------------------------------------------
# LCI 权重图计算
# -----------------------------------------------------------------
def compute_lci_map(shp_path: str, ref_tif_path: str) -> np.ndarray:
    """
    计算全图每个像元的 LCI_w（标签置信度指数）。

    LCI_w = LCI_raw = max(class_coverage_ratio per pixel)

    方法：对每个被标注的像元，用 SOFT_LABEL_SUBGRID × SOFT_LABEL_SUBGRID
    的子像元格网计算各类别覆盖比例，取最大值作为 LCI_raw。
    LCI_raw 直接作为 LCI_w，不引入 separability 惩罚。

    子像元精度：
      对于 Dead/Living 边界（sep=0.25），使用 get_local_subgrid(1,2)=20
      更精细的格网意味着更准确的覆盖率估计。

    未标注像元（mask=0）的 LCI 设为 0.0（训练时被 IGNORE_INDEX 跳过）。

    返回：(H, W) float32，值域 [0, 1]
    """
    from rasterio.transform import from_bounds as tfm_from_bounds

    with rasterio.open(ref_tif_path) as src:
        H, W      = src.height, src.width
        transform = src.transform
        tif_crs   = src.crs
        res_x     = abs(src.res[0])
        res_y     = abs(src.res[1])

    gdf = gpd.read_file(shp_path)
    field = config.SHP_CLASS_FIELD

    if field not in gdf.columns:
        return np.zeros((H, W), dtype=np.float32)

    if gdf.crs is None:
        raise RuntimeError(f"SHP CRS is missing; refuse to assume it matches TIF: {shp_path}")
    elif tif_crs is not None:
        try:
            from pyproj import CRS as ProjCRS
            if not ProjCRS.from_user_input(gdf.crs).equals(
                   ProjCRS.from_user_input(tif_crs)):
                gdf = gdf.to_crs(tif_crs)
        except Exception:
            if str(gdf.crs) != str(tif_crs):
                gdf = gdf.to_crs(tif_crs)

    gdf = gdf[gdf[field].notna() & ~gdf.geometry.is_empty].copy()
    gdf[field] = gdf[field].astype(int)
    valid_classes = set(range(1, config.NUM_CLASSES + 1))
    gdf = gdf[gdf[field].isin(valid_classes)]

    shapes = [(geom, int(val))
              for geom, val in zip(gdf.geometry, gdf[field])
              if geom is not None and not geom.is_empty]
    shapes = sorted(shapes, key=lambda x: -x[1])  # Dead last

    if not shapes:
        raise RuntimeError(f"No rasterizable valid class geometries for LCI in SHP: {shp_path}")

    # Coarse rasterization to find labeled pixels
    soft_mask = rasterize(shapes, (H, W), transform=transform,
                          fill=0, dtype=np.int32, all_touched=True)

    lci_map = np.zeros((H, W), dtype=np.float32)
    labeled_ys, labeled_xs = np.where(soft_mask > 0)

    base_sg = getattr(config, 'SOFT_LABEL_SUBGRID', 5)

    for py, px in zip(labeled_ys, labeled_xs):
        # Pixel bounds
        left   = transform.c + px * transform.a
        top    = transform.f + py * transform.e
        right  = left + transform.a
        bottom = top  + transform.e
        if right  < left:  left,  right  = right,  left
        if bottom < top:   top,   bottom = bottom, top

        # Dominant class in this pixel (coarse) → pick subgrid size
        dom_cls = int(soft_mask[py, px])

        # For each neighbor class, check which needs finest subgrid
        # Use the finest required by any class pair touching this pixel
        sg = base_sg
        for cls_b in valid_classes:
            if cls_b != dom_cls:
                sg = max(sg, config.get_local_subgrid(dom_cls, cls_b, base_sg))
            # Early break once we hit max useful subgrid
            if sg >= 20:
                break

        sub_tfm  = tfm_from_bounds(left, bottom, right, top, sg, sg)
        sub_mask = rasterize(shapes, (sg, sg), transform=sub_tfm,
                             fill=0, dtype=np.int32, all_touched=True)
        total = sg * sg

        # Coverage ratios
        ratios = {c: float((sub_mask == c).sum() / total)
                  for c in valid_classes}
        lci_map[py, px] = max(ratios.values()) if ratios else 0.0

    return lci_map


# -----------------------------------------------------------------
# 读取 PlanetScope SR + 追加指数
# -----------------------------------------------------------------
def read_and_prepare(tif_path: str) -> np.ndarray:
    """
    读取 TIF → 固定比例转换为反射率 → 追加遥感指数
    返回 (C_total, H, W) float32
    """
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)   # (C, H, W)

    raw_ch = config.RAW_CHANNELS
    C, H, W = data.shape

    if C != raw_ch:
        raise RuntimeError(
            f"Unexpected raw channel count for {tif_path}: got {C}, "
            f"expected {raw_ch} for BAND_MODE={config.BAND_MODE}. "
            "Refuse to pad/truncate bands during tile preparation."
        )

    return sr_to_feature_stack(data, source=tif_path)



# -----------------------------------------------------------------
# 切片保存
# -----------------------------------------------------------------
def save_tile_tif(data: np.ndarray, out_path: str):
    """保存多通道切片为 GeoTIFF (无地理参考，仅数据)"""
    C, H, W = data.shape
    with rasterio.open(out_path, "w",
                       driver="GTiff", count=C,
                       height=H, width=W,
                       dtype=rasterio.float32,
                       compress="lzw") as dst:
        dst.update_tags(**preprocessing_tags())
        for i in range(C):
            dst.write(data[i], i + 1)




def tile_and_save(img_data: np.ndarray, mask: np.ndarray,
                  sample_id: str,
                  lci_map:   np.ndarray = None,
                  dry_run:   bool = False) -> dict:
    """
    lci_map: (H, W) float32 [0,1] — 与 mask 同尺寸的 LCI 权重图。
             None 时所有有效像元权重设为 1.0。
    切片保存到三个目录：
      TRAIN_IMG_DIR  / {fname}.tif   — 影像切片（含遥感指数）
      TRAIN_MASK_DIR / {fname}.png   — 标注 mask（int8）
      TRAIN_LCI_DIR  / {fname}.npy   — LCI 权重图（float32）
    """
    ts   = config.TILE_SIZE
    st   = config.TILE_STRIDE
    mrat = config.MIN_LABEL_RATIO
    C, H, W = img_data.shape
    stats = dict(total=0, saved=0, skipped=0)

    # LCI dir alongside mask dir
    lci_dir = getattr(config, 'TRAIN_LCI_DIR',
                      os.path.join(config.DATA_ROOT, 'lci'))
    if not dry_run:
        os.makedirs(lci_dir, exist_ok=True)

    for y in range(0, H - ts + 1, st):
        for x in range(0, W - ts + 1, st):
            stats["total"] += 1
            tile_img  = img_data[:, y:y+ts, x:x+ts]
            tile_mask = mask[y:y+ts, x:x+ts]

            labeled_ratio = (tile_mask > 0).sum() / (ts * ts)
            if labeled_ratio < mrat:
                stats["skipped"] += 1
                continue

            if dry_run:
                stats["saved"] += 1
                continue

            fname    = f"{sample_id}_y{y:04d}_x{x:04d}"
            img_out  = os.path.join(config.TRAIN_IMG_DIR,  fname + ".tif")
            mask_out = os.path.join(config.TRAIN_MASK_DIR, fname + ".png")
            lci_out  = os.path.join(lci_dir,               fname + ".npy")

            save_tile_tif(tile_img, img_out)
            Image.fromarray(tile_mask.astype(np.uint8)).save(mask_out)

            # LCI weight tile
            if lci_map is not None:
                tile_lci = lci_map[y:y+ts, x:x+ts].astype(np.float32)
            else:
                # Default: 1.0 for labeled pixels, 0.0 for unlabeled
                tile_lci = (tile_mask > 0).astype(np.float32)
            np.save(lci_out, tile_lci)

            stats["saved"] += 1

    return stats


# -----------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------
def main(dry_run: bool = False):
    tif_paths = sorted(glob.glob(os.path.join(config.RAW_TIF_DIR, "*.tif")))
    if not tif_paths:
        sys.exit(f"[错误] 未找到 TIF: {config.RAW_TIF_DIR}")

    print(f"找到 {len(tif_paths)} 张 TIF")
    print(f"BAND_MODE={config.BAND_MODE}  "
          f"RAW={config.RAW_CHANNELS}ch → IN={config.IN_CHANNELS}ch")
    print(f"切片大小={config.TILE_SIZE}, 步长={config.TILE_STRIDE}, "
          f"最低标注比例={config.MIN_LABEL_RATIO:.0%}\n")

    total_saved = total_skipped = 0
    failed = []

    for tif_path in tqdm(tif_paths, desc="处理影像"):
        sample_id = Path(tif_path).stem.split("_")[0]
        shp_path  = find_shp_for_tif(tif_path, config.RAW_SHP_DIR)

        if shp_path is None:
            raise FileNotFoundError(f"No matching SHP found for sample_id={sample_id}, tif={tif_path}")

        try:
            mask     = rasterize_shp(shp_path, tif_path)
            img_data = read_and_prepare(tif_path)
            # Compute LCI weight map (skip in dry_run to save time)
            lci_map  = None
            if not dry_run:
                lci_map = compute_lci_map(shp_path, tif_path)
            s = tile_and_save(img_data, mask, sample_id,
                              lci_map=lci_map, dry_run=dry_run)

            tqdm.write(f"  ✔ {sample_id} | 总={s['total']} "
                       f"保留={s['saved']} 过滤={s['skipped']}")
            total_saved   += s["saved"]
            total_skipped += s["skipped"]

        except Exception as e:
            tqdm.write(f"  ✘ {sample_id} 失败: {e}")
            failed.append(sample_id)
            raise

    print(f"\n{'='*50}")
    print(f"  {'(DRY RUN) ' if dry_run else ''}完成")
    print(f"  保留切片: {total_saved}   过滤切片: {total_skipped}")
    if failed:
        print(f"  失败: {failed}")
    print(f"{'='*50}")
    if total_saved <= 0:
        raise RuntimeError("No training tiles were saved. Check RAW_TIF_DIR, RAW_SHP_DIR, and MIN_LABEL_RATIO.")

    if not dry_run:
        _print_class_stats()


def _print_class_stats(max_samples: int = 500):
    mask_files = sorted(
        glob.glob(os.path.join(config.TRAIN_MASK_DIR, "*.png"))
    )[:max_samples]
    if not mask_files:
        return
    counts = np.zeros(config.NUM_CLASSES + 1, dtype=np.int64)
    for mf in mask_files:
        m = np.array(Image.open(mf))
        for c in range(config.NUM_CLASSES + 1):
            counts[c] += (m == c).sum()

    total_labeled = counts[1:].sum()
    print(f"\n  类别像素分布（抽样 {len(mask_files)} 张）：")
    print(f"  {'类别':<28} {'像素数':>12} {'占比':>8} {'建议权重':>10}")
    print(f"  {'-'*60}")
    print(f"  {'[0] 未标注(IGNORE)':<28} {counts[0]:>12,}   —")
    for cls in range(1, config.NUM_CLASSES + 1):
        name  = config.CLASS_NAMES.get(cls, f"cls{cls}")
        ratio = counts[cls] / max(total_labeled, 1)
        w_sug = round(1.0 / max(ratio, 0.01), 2)
        print(f"  [{cls}] {name:<25} {counts[cls]:>12,} "
              f"{ratio:>7.2%} {w_sug:>10.2f}")
    print("\n  ↑ 将建议权重填入 config.py 的 CLASS_WEIGHTS（适当平滑后使用）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
