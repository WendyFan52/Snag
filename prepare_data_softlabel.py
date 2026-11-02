"""
Generate soft-label tiles - prepare_data_softlabel.py
=================================================================
Alternative to prepare_data.py with the following outputs:

  images/{tile}.tif     - image tiles (same as prepare_data.py)
  masks/{tile}.png      - hard-label masks (same as prepare_data.py, for evaluation)
  softlabels/{tile}.npy - soft labels (NUM_CLASSES+1, H, W) float32
  lci/{tile}.npy        - LCI weight maps (H, W) float32

Soft labels:
-----------------------------------------------------------------
Hard labels assign one class per pixel, e.g. mask[y,x] = 1 (dead tree).
Soft labels assign class probabilities, e.g. softlabel[:,y,x] = [0,0.65,0.35,0,0,0].
        This represents 65% dead-tree coverage and 35% living-tree coverage.

Motivation:
-----------------------------------------------------------------
PlanetScope 3 m pixels exceed individual crown sizes. Polygon boundary pixels contain multiple classes.
  - Hard labels force one class onto boundary pixels, introducing systematic errors.
  - Soft labels reflect actual area fractions, providing more representative training targets.

Computation:
-----------------------------------------------------------------
Subdivide each pixel using a SOFT_LABEL_SUBGRID x SOFT_LABEL_SUBGRID grid,
count the subcells covered by each class, and convert the counts to area fractions.

Use a finer grid (subgrid=20) at Dead/Living boundaries, the hardest pair to distinguish.
Use the standard grid (subgrid=5..8) at other boundaries.

LCI (Label Confidence Index):
-----------------------------------------------------------------
LCI[pixel] = max(class_coverage_ratio)
  LCI=1.0 -> the pixel belongs entirely to one class (pure)
  LCI=0.65 -> the dominant class covers 65% (mixed pixel)
  LCI=0.3  -> highly mixed pixel

During training: loss_weight[pixel] = LCI[pixel].
Pure pixels contribute more to training; mixed pixels contribute less but retain nonzero weight.

Loss function:
-----------------------------------------------------------------
Soft-label training uses KL divergence instead of cross-entropy:
  soft_CE = -sum(softlabel * log(softmax(logits)))
This uses soft labels as the target distribution.

Usage:
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

# Soft-label output directory
SOFTLABEL_DIR = os.path.join(config.DATA_ROOT, "softlabels")
os.makedirs(SOFTLABEL_DIR, exist_ok=True)

VALID_CLASSES = list(range(1, config.NUM_CLASSES + 1))
N_OUT = config.NUM_CLASSES + 1   # Include ignore slot 0


# =================================================================
# Utility functions shared with prepare_data.py
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
    """Compare CRSs using pyproj and reproject when they differ."""
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
    """Read TIF -> scale to reflectance -> append spectral indices -> (C, H, W) float32."""
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
    Load SHP labels, align the CRS, filter invalid classes,
    and sort by descending class ID (Dead=1 is drawn last and overwrites other classes).
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

    # Draw Dead(1) last to overwrite other classes
    shapes = sorted(shapes, key=lambda x: -x[1])
    return shapes, transform, H, W


# =================================================================
# Current Class_Int already uses the 1-5 framework encoding.
# =================================================================


# =================================================================
# Soft-label computation
# =================================================================
def compute_softlabel_map(shapes: list,
                          transform,
                          H: int, W: int) -> tuple:
    """
    Vectorized soft labels using full-image high-resolution rasterization instead of per-pixel loops.

    Per-pixel rasterize() calls require millions of operations and are slow.
    Vectorized procedure:
      1. Rasterize the whole image at (H*sg x W*sg) resolution in one call.
      2. Reshape to (H, sg, W, sg) and compute class fractions from subcell counts.
      3. One rasterize() call produces the full image's soft labels at each grid resolution.

    This reduces processing time from hours to seconds per sample.

    Dead/Living boundary precision:
      Rasterize Dead(1) and Living(2) polygons separately on a finer sg=20 grid
      to refine their coverage fractions at these critical boundaries.
      Use sg=base (default 5) for other classes.

    Returns:
      softlabel (N_OUT, H, W) float32
      lci_map   (H, W)        float32
    """
    base_sg = getattr(config, 'SOFT_LABEL_SUBGRID', 5)
    dl_sg   = config.get_local_subgrid(1, 2, base_sg)  # Fine Dead/Living grid, default 20

    # 1. Locate labeled regions through coarse rasterization
    hard_mask = rasterize(shapes, (H, W), transform=transform,
                          fill=0, dtype=np.int32, all_touched=False)
    soft_mask = rasterize(shapes, (H, W), transform=transform,
                          fill=0, dtype=np.int32, all_touched=True)

    softlabel = np.zeros((N_OUT, H, W), dtype=np.float32)
    lci_map   = np.zeros((H, W),        dtype=np.float32)

    # Assign unlabeled pixels to the ignore slot
    softlabel[0, soft_mask == 0] = 1.0

    if not shapes:
        return softlabel, lci_map

    # 2. Rasterize the full image at base_sg times the original resolution
    # Replace per-pixel rasterize() calls with one full-image call
    from rasterio.transform import Affine

    def make_hires_transform(transform, sg):
        """Scale the transform by sg to obtain the high-resolution transform."""
        return Affine(transform.a / sg, transform.b, transform.c,
                      transform.d, transform.e / sg, transform.f)

    # Use base_sg resolution for all classes except Dead/Living
    hires_tfm_base = make_hires_transform(transform, base_sg)
    hires_mask_base = rasterize(shapes,
                                (H * base_sg, W * base_sg),
                                transform=hires_tfm_base,
                                fill=0, dtype=np.int32,
                                all_touched=True)

    # Use finer resolution for Dead/Living (dl_sg times the original resolution)
    hires_tfm_dl = make_hires_transform(transform, dl_sg)
    hires_mask_dl = rasterize(shapes,
                              (H * dl_sg, W * dl_sg),
                              transform=hires_tfm_dl,
                              fill=0, dtype=np.int32,
                              all_touched=True)

    # 3. Aggregate high-resolution labels to original-resolution coverage fractions
    # reshape: (H*sg, W*sg) → (H, sg, W, sg) → mean over sg dims
    def coverage(hires: np.ndarray, sg: int, cls: int) -> np.ndarray:
        """Return (H, W) float32 coverage fractions for class cls in each pixel."""
        return ((hires == cls)
                .reshape(H, sg, W, sg)
                .mean(axis=(1, 3))
                .astype(np.float32))

    # Compute coverage fractions for each class
    # Use the fine grid for Dead(1)/Living(2), and the coarse grid for other classes
    raw_cov = {}
    for cls in VALID_CLASSES:
        if cls in (1, 2):
            raw_cov[cls] = coverage(hires_mask_dl, dl_sg, cls)
        else:
            raw_cov[cls] = coverage(hires_mask_base, base_sg, cls)

    # 4. Filter noise, normalize, and populate softlabel
    min_ratio = getattr(config, 'SOFT_LABEL_MIN_RATIO', 0.05)

    # Sum coverage fractions over all classes
    total_cov = np.zeros((H, W), dtype=np.float32)
    for cls in VALID_CLASSES:
        cov = raw_cov[cls].copy()
        cov[cov < min_ratio] = 0.0   # Filter noise
        if cls > 0:
            softlabel[cls] += cov
        total_cov += cov

    # Normalize labeled pixels only
    labeled = (soft_mask > 0) & (total_cov > 0)
    for c in range(N_OUT):
        softlabel[c, labeled] /= total_cov[labeled]

    # Fall back to hard labels if filtering removes every class from a labeled pixel
    no_coverage = (soft_mask > 0) & (total_cov == 0)
    if no_coverage.any():
        for py, px in zip(*np.where(no_coverage)):
            cls = int(hard_mask[py, px]) if hard_mask[py, px] > 0 else int(soft_mask[py, px])
            if cls > 0:
                softlabel[cls, py, px] = 1.0

    # Ensure ignore slot = 1 for unlabeled pixels
    softlabel[0, soft_mask == 0] = 1.0

    # 5. LCI = maximum class coverage fraction (purity)
    # Take the maximum of softlabel[1:], excluding ignore slot 0
    lci_map = softlabel[1:].max(axis=0)

    return softlabel, lci_map




# =================================================================
# Save tiles
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

            # Filter tiles with insufficient label coverage
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
# Main workflow
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
            # 1. Load polygons
            shapes, transform, H, W = load_and_prepare_shapes(
                shp_path, tif_path)
            if not shapes:
                raise RuntimeError(f"No valid shapes for sample_id={sample_id}, shp={shp_path}")

            # 2. Hard labels for masks and evaluation
            hard_mask = rasterize(shapes, (H, W), transform=transform,
                                  fill=0, dtype=np.int32, all_touched=False)
            hard_mask = hard_mask.astype(np.uint8)

            # 3. Compute soft labels and LCI
            softlabel, lci_map = compute_softlabel_map(
                shapes, transform, H, W)

            # 4. Read imagery
            img_data = read_and_prepare(tif_path)

            # 5. Save tiles
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
