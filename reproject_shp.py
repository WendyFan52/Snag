"""
重投影 SHP 文件 — reproject_shp.py
=================================================================
将 CRS 不匹配的 SHP 文件重投影到与对应 TIF 相同的坐标系。

用法：
  python reproject_shp.py             # 正式执行（原文件备份到 _backup/）
  python reproject_shp.py --dry_run   # 只打印，不写文件
=================================================================
"""

import os
import glob
import shutil
import argparse
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

try:
    import geopandas as gpd
    import rasterio
    from pyproj import CRS as ProjCRS
except ImportError as e:
    raise SystemExit(f"Missing: {e}\npip install geopandas rasterio pyproj")

import config

# =================================================================
# 配置：需要重投影的样本 ID → 目标 EPSG
# 从 check_data.py 报告的 CRS MISMATCH 中读取
# =================================================================
TARGETS = {
    "1009": "EPSG:32617",
    "1164": "EPSG:32617",
    "119":  "EPSG:32632",
    "11":   "EPSG:32632",
    "1352": "EPSG:32632",
    "1353": "EPSG:32632",
    "1331": "EPSG:32633",
    "1332": "EPSG:32633",
    "1357": "EPSG:32628",
}

# 如果不想手写，设为 True 则自动从 TIF 读取目标 CRS
AUTO_DETECT = True    # Auto-detect all mismatches by comparing each SHP to its TIF


def find_shp(sid: str, shp_dir: str) -> list:
    """Return ALL SHP files matching this sample ID (some samples have multiple)."""
    matches = glob.glob(os.path.join(shp_dir, f"{sid}_*.shp"))
    matches += glob.glob(os.path.join(shp_dir, f"{sid}.shp"))
    matches = [m for m in matches if "_backup" not in m]
    return sorted(set(matches))


def find_tif(sid: str, tif_dir: str) -> Optional[str]:
    matches = glob.glob(os.path.join(tif_dir, f"{sid}_*.tif"))
    matches += glob.glob(os.path.join(tif_dir, f"{sid}.tif"))
    return matches[0] if matches else None


def get_tif_crs(tif_path: str) -> str:
    with rasterio.open(tif_path) as src:
        crs = src.crs
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else crs.to_wkt()


def reproject_one(shp_path: str, target_crs: str,
                  backup_dir: str, dry_run: bool) -> bool:
    """重投影一个 SHP 文件，原文件备份到 backup_dir。"""
    try:
        gdf = gpd.read_file(shp_path)
        src_crs = str(gdf.crs) if gdf.crs else "None"

        # 检查是否真的需要重投影
        if gdf.crs is not None:
            try:
                if ProjCRS.from_user_input(gdf.crs).equals(
                   ProjCRS.from_user_input(target_crs)):
                    print(f"  SKIP {os.path.basename(shp_path)}: already {target_crs}")
                    return True
            except Exception:
                pass

        print(f"  {os.path.basename(shp_path)}")
        print(f"    {src_crs} -> {target_crs}")

        if dry_run:
            print(f"    [dry_run] would reproject and overwrite")
            return True

        # 备份：复制所有相关文件（.shp .dbf .prj .shx .cpg）
        stem = shp_path[:-4]
        os.makedirs(backup_dir, exist_ok=True)
        for ext in (".shp", ".dbf", ".prj", ".shx", ".cpg"):
            src_file = stem + ext
            if os.path.exists(src_file):
                dst_file = os.path.join(backup_dir,
                                        os.path.basename(src_file))
                shutil.copy2(src_file, dst_file)

        # 重投影并覆盖
        gdf_repr = gdf.to_crs(target_crs)
        gdf_repr.to_file(shp_path, driver="ESRI Shapefile")
        print(f"    OK — saved to {shp_path}")
        return True

    except Exception as e:
        print(f"  ERROR {os.path.basename(shp_path)}: {e}")
        return False


def main(dry_run: bool = False):
    shp_dir = config.RAW_SHP_DIR
    tif_dir = config.RAW_TIF_DIR
    backup_dir = os.path.join(shp_dir, "_backup")

    print(f"SHP dir : {shp_dir}")
    print(f"TIF dir : {tif_dir}")
    print(f"Backup  : {backup_dir}")
    print(f"Mode    : {'DRY RUN' if dry_run else 'LIVE (will overwrite)'}\n")

    if AUTO_DETECT:
        # 自动扫描所有 SHP，找 CRS 不匹配的
        print("Auto-detecting CRS mismatches...")
        all_shp = glob.glob(os.path.join(shp_dir, "*.shp"))
        all_shp = [s for s in all_shp if "_backup" not in s]
        targets = {}
        for shp_path in sorted(all_shp):
            sid = os.path.basename(shp_path).split("_")[0].split(".")[0]
            tif_path = find_tif(sid, tif_dir)
            if not tif_path:
                continue
            try:
                gdf = gpd.read_file(shp_path)
                if gdf.crs is None:
                    continue
                tif_crs = get_tif_crs(tif_path)
                if not ProjCRS.from_user_input(gdf.crs).equals(
                       ProjCRS.from_user_input(tif_crs)):
                    targets[sid] = tif_crs
                    print(f"  {sid}: {gdf.crs} != {tif_crs}")
            except Exception as e:
                print(f"  {sid}: error — {e}")
    else:
        targets = TARGETS

    print(f"\nReprojecting {len(targets)} samples...\n")
    ok = fail = 0
    for sid, target_crs in sorted(targets.items()):
        shp_paths = find_shp(sid, shp_dir)
        if not shp_paths:
            print(f"  NOT FOUND: {sid}")
            fail += 1
            continue
        # Handle multiple SHP files for the same sample ID
        for shp_path in shp_paths:
            if reproject_one(shp_path, target_crs, backup_dir, dry_run):
                ok += 1
            else:
                fail += 1

    print(f"\n{'='*50}")
    print(f"  Done: {ok} OK,  {fail} failed")
    if not dry_run and ok > 0:
        print(f"  Backups: {backup_dir}")
        print(f"  Run check_data.py again to verify CRS mismatch is resolved")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview only, do not write files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
