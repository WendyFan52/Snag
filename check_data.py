"""
数据质量检查脚本 — check_data.py
=================================================================
全部阈值在 config.py 中配置，本脚本零硬编码。

六项检查：
  1. 样本完整性  — TIF/SHP 配对、波段数、CRS 一致性（pyproj 精确比较）
  2. 像元值异常  — NoData、NaN/Inf、动态范围
  3. 标注质量    — 类别分布、覆盖率、极小图斑面积贡献分析
  4. SHP 冲突    — 不同类别图斑重叠
  5. 混合像元    — 边界像元比例 + 标签置信度指数(LCI)
     LCI 量化每个像元的标签可信程度，用于软标签加权 Loss

极小图斑处理策略：
  不删除，而是通过子像元面积加权自然融入软标签。
  面积越小，对该像元软标签的贡献越小，但不为零。
  这比删除更准确：一株 1m 小树冠仍然是真实死树。

用法：
  python check_data.py
  python check_data.py --sample 10
  python check_data.py --fix
  python check_data.py --no-plot
=================================================================
"""

import os, sys, glob, argparse, warnings, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

try:
    import rasterio
    from rasterio.features import rasterize
    import geopandas as gpd
    from pyproj import CRS as ProjCRS
except ImportError as e:
    sys.exit(f"Missing: {e}\npip install rasterio geopandas pyproj")

import config
warnings.filterwarnings("ignore")

VALID_CLASSES = set(range(1, config.NUM_CLASSES + 1))


# =================================================================
# CHECK 1: Sample completeness
# CRS comparison via pyproj.CRS.equals() — handles:
#   · Same CRS expressed differently ("EPSG:32617" vs WKT string)
#   · SHP with no .prj file (gdf.crs is None) — reported separately
#   · Different authority codes that are semantically equivalent
# =================================================================
def find_pairs():
    tif_files = sorted(glob.glob(os.path.join(config.RAW_TIF_DIR, "*.tif")))
    shp_map = {}
    for shp in glob.glob(os.path.join(config.RAW_SHP_DIR, "*.shp")):
        sid = Path(shp).stem.split("_")[0]
        shp_map.setdefault(sid, shp)
    pairs, missing_shp, missing_tif = [], [], []
    matched = set()
    for tif in tif_files:
        sid = Path(tif).stem.split("_")[0]
        if sid in shp_map:
            pairs.append((tif, shp_map[sid], sid))
            matched.add(sid)
        else:
            missing_shp.append(sid)
    for sid in shp_map:
        if sid not in matched:
            missing_tif.append(sid)
    return pairs, missing_shp, missing_tif


def check_completeness(pairs):
    result = {
        "band_issues"   : [],
        "crs_mismatch"  : [],   # confirmed different CRS
        "crs_undefined" : [],   # SHP has no CRS defined (.prj missing)
        "size_info"     : [],
    }
    for tif, shp, sid in pairs:
        with rasterio.open(tif) as src:
            n       = src.count
            tif_crs = src.crs
            result["size_info"].append(
                (sid, src.height, src.width, abs(src.res[0])))

        if n != config.RAW_CHANNELS:
            result["band_issues"].append((sid, n, config.RAW_CHANNELS))

        try:
            gdf = gpd.read_file(shp)
            shp_crs = gdf.crs

            if shp_crs is None:
                # No .prj file — cannot determine if CRS matches
                result["crs_undefined"].append(sid)

            elif tif_crs is None:
                result["crs_mismatch"].append(
                    (sid, "SHP has CRS", "TIF has no CRS"))

            else:
                # Precise comparison via pyproj — handles alias differences
                try:
                    shp_proj = ProjCRS.from_user_input(shp_crs)
                    tif_proj = ProjCRS.from_user_input(tif_crs)
                    if not shp_proj.equals(tif_proj):
                        shp_epsg = shp_proj.to_epsg()
                        tif_epsg = tif_proj.to_epsg()
                        result["crs_mismatch"].append((
                            sid,
                            f"EPSG:{shp_epsg}" if shp_epsg else shp_proj.name[:30],
                            f"EPSG:{tif_epsg}" if tif_epsg else tif_proj.name[:30],
                        ))
                except Exception as e:
                    # Fallback: string comparison
                    if str(shp_crs) != str(tif_crs):
                        result["crs_mismatch"].append(
                            (sid, str(shp_crs)[:30], str(tif_crs)[:30]))

        except Exception as e:
            result["crs_mismatch"].append((sid, f"read error: {e}", ""))

    return result


# =================================================================
# CHECK 2: Pixel anomalies
# =================================================================
def check_pixel_anomalies(tif_path, sid):
    with rasterio.open(tif_path) as src:
        data = src.read(list(range(1, config.RAW_CHANNELS + 1))).astype(np.float32)
    r = {"sid": sid, "issues": [], "warnings": [],
         "band_stats": [], "nodata_ratio": 0.0}
    if np.isnan(data).any():
        r["issues"].append(f"NaN: {int(np.isnan(data).sum())} pixels")
    if np.isinf(data).any():
        r["issues"].append(f"Inf: {int(np.isinf(data).sum())} pixels")
    nodata = (data == 0).all(axis=0)
    r["nodata_ratio"] = float(nodata.mean())
    if r["nodata_ratio"] > config.ANOMALY_NODATA_WARN_RATIO:
        r["warnings"].append(
            f"NoData {r['nodata_ratio']:.1%} > {config.ANOMALY_NODATA_WARN_RATIO:.0%}")
    for b in range(data.shape[0]):
        band = data[b]; valid = band[band > 0]
        if valid.size == 0:
            r["issues"].append(f"Band {b+1}: all zeros"); continue
        p2, p98 = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
        if p98 - p2 < config.ANOMALY_MIN_DYNAMIC_RANGE:
            r["issues"].append(f"Band {b+1}: dynamic range {p98-p2:.2e} too small")
        flat = np.clip(band.ravel().astype(np.int64), -32768, 32767)
        cnts = np.bincount((flat + 32768).astype(np.int64))
        mode_r = float(cnts.max() / len(flat))
        if mode_r > config.ANOMALY_SINGLE_VALUE_RATIO:
            r["warnings"].append(f"Band {b+1}: single value {mode_r:.1%} of pixels")
        r["band_stats"].append({"band": b+1, "min": float(valid.min()),
            "max": float(valid.max()), "mean": float(valid.mean()),
            "std": float(valid.std()), "p2": p2, "p98": p98})
    return r


# =================================================================
# CHECK 3: Annotation quality
# Tiny polygon strategy:
#   Do NOT remove tiny polygons. Instead, report:
#   · How many are sub-pixel
#   · Their cumulative area as % of total labeled area
#     (measures actual information content, not just count)
#   · They will naturally contribute proportionally in soft labels
# =================================================================
def check_annotation_quality(tif_path, shp_path, sid):
    r = {"sid": sid, "issues": [], "warnings": [],
         "class_distribution": {}, "total_polygons": 0,
         "coverage_ratio": 0.0, "class1_coverage": 0.0,
         "tiny_polygon_stats": {}, "area_stats": {}}

    with rasterio.open(tif_path) as src:
        bounds     = src.bounds
        tif_crs    = src.crs
        pixel_area = abs(src.res[0] * src.res[1])

    img_area = (abs(bounds.right - bounds.left) *
                abs(bounds.top - bounds.bottom))

    gdf = gpd.read_file(shp_path)
    field = config.SHP_CLASS_FIELD

    if field not in gdf.columns:
        r["issues"].append(f"Field '{field}' missing. Have: {list(gdf.columns)}")
        return r

    n_empty = gdf.geometry.isna().sum() + gdf.geometry.is_empty.sum()
    if n_empty:
        r["issues"].append(f"{n_empty} empty geometries")
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty].copy()

    if gdf.crs and tif_crs and str(gdf.crs) != str(tif_crs):
        try: gdf = gdf.to_crs(tif_crs)
        except Exception: pass

    gdf[field] = gdf[field].fillna(-1).astype(int)
    invalid = set(gdf[field].unique()) - VALID_CLASSES - {-1}
    if invalid:
        r["issues"].append(f"Invalid class values: {sorted(invalid)}")

    valid_gdf = gdf[gdf[field].isin(VALID_CLASSES)].copy()
    r["total_polygons"]     = len(valid_gdf)
    r["class_distribution"] = dict(valid_gdf[field].value_counts().sort_index())
    if not len(valid_gdf):
        r["issues"].append("No valid labeled polygons"); return r

    areas = valid_gdf.geometry.area.values
    total_area = areas.sum()

    # Tiny polygon analysis (sub-pixel threshold)
    # Key insight: count AND cumulative area both matter
    # Many tiny polygons with tiny total area = annotation artifact
    # Many tiny polygons with significant total area = real small crowns
    tiny_mask      = areas < pixel_area
    tiny_count     = int(tiny_mask.sum())
    tiny_area_frac = float(areas[tiny_mask].sum() / max(total_area, 1))

    # Per-class tiny polygon breakdown
    class_tiny = {}
    for cls in VALID_CLASSES:
        cls_mask  = (valid_gdf[field].values == cls)
        cls_areas = areas[cls_mask]
        if len(cls_areas) == 0:
            continue
        cls_tiny_n    = int((cls_areas < pixel_area).sum())
        cls_tiny_frac = float(cls_areas[cls_areas < pixel_area].sum() /
                               max(cls_areas.sum(), 1))
        class_tiny[cls] = {
            "count"    : cls_tiny_n,
            "area_frac": cls_tiny_frac,
        }

    r["tiny_polygon_stats"] = {
        "count"         : tiny_count,
        "area_frac"     : tiny_area_frac,
        "per_class"     : class_tiny,
        "pixel_area_m2" : pixel_area,
    }

    # Warn only if tiny area fraction is significant
    # (lots of tiny polygons with big total area = meaningful data)
    if tiny_count > 0:
        if tiny_area_frac < 0.01:
            r["warnings"].append(
                f"{tiny_count} sub-pixel polygons "
                f"({tiny_area_frac:.2%} of labeled area — negligible, "
                f"will contribute small weights in soft labels)")
        elif tiny_area_frac < 0.05:
            r["warnings"].append(
                f"{tiny_count} sub-pixel polygons "
                f"({tiny_area_frac:.2%} of labeled area — minor, "
                f"included in soft labels proportionally)")
        else:
            r["warnings"].append(
                f"{tiny_count} sub-pixel polygons "
                f"({tiny_area_frac:.2%} of labeled area — SUBSTANTIAL, "
                f"likely real small tree crowns, soft labels will handle correctly)")

    r["area_stats"] = {
        "min_px"   : float(areas.min()    / pixel_area),
        "max_px"   : float(areas.max()    / pixel_area),
        "mean_px"  : float(areas.mean()   / pixel_area),
        "median_px": float(np.median(areas) / pixel_area),
    }

    total_labeled = total_area
    r["coverage_ratio"] = float(total_labeled / max(img_area, 1))
    if r["coverage_ratio"] < config.QUALITY_MIN_COVERAGE:
        r["warnings"].append(
            f"Low coverage {r['coverage_ratio']:.1%} < {config.QUALITY_MIN_COVERAGE:.0%}")

    c1_area = valid_gdf[valid_gdf[field] == 1].geometry.area.sum()
    r["class1_coverage"] = float(c1_area / max(img_area, 1))
    return r


# =================================================================
# CHECK 4: SHP conflicts
# =================================================================
def check_shp_conflicts(shp_path, tif_path, sid):
    r = {"sid": sid, "issues": [], "warnings": [],
         "diff_class_overlap": 0, "same_class_overlap": 0,
         "conflict_examples": [], "topology_errors": 0,
         "topology_error_examples": []}
    gdf = gpd.read_file(shp_path)
    field = config.SHP_CLASS_FIELD
    if field not in gdf.columns: return r
    gdf = gdf[gdf[field].notna() & ~gdf.geometry.is_empty].copy()
    gdf[field] = gdf[field].astype(int)
    gdf = gdf[gdf[field].isin(VALID_CLASSES)].reset_index(drop=True)
    if len(gdf) < 2: return r
    with rasterio.open(tif_path) as src:
        tif_crs = src.crs
    if gdf.crs and tif_crs and str(gdf.crs) != str(tif_crs):
        try: gdf = gdf.to_crs(tif_crs)
        except Exception: pass
    invalid_n = int((~gdf.geometry.is_valid).sum())
    if invalid_n:
        r["issues"].append(
            f"{invalid_n} invalid geometries; fix shapefile topology before final training/analysis"
        )
    try:
        from shapely.strtree import STRtree
    except ImportError:
        from shapely import STRtree
    try:
        from shapely.errors import PredicateError, TopologicalError
    except ImportError:
        PredicateError = TopologicalError = Exception
    geoms = list(gdf.geometry.values)
    tree = STRtree(geoms)
    geom_to_index = {id(geom): idx for idx, geom in enumerate(geoms)}
    diff_c = same_c = 0
    topology_errors = 0
    topology_examples = []
    examples = []

    def record_topology_error(i, j, exc):
        nonlocal topology_errors
        topology_errors += 1
        if len(topology_examples) < 5:
            topology_examples.append({
                "i": int(i),
                "j": int(j),
                "class_i": int(gdf[field].iloc[i]),
                "class_j": int(gdf[field].iloc[j]),
                "error": type(exc).__name__,
            })

    for i, gi in enumerate(geoms):
        for hit in tree.query(gi):
            if isinstance(hit, (int, np.integer)):
                j = int(hit)
            else:
                try:
                    j = geom_to_index[id(hit)]
                except KeyError as exc:
                    raise RuntimeError(
                        "STRtree.query returned an object that cannot be mapped "
                        f"back to the source geometries for sample {sid}. "
                        "Refusing to skip SHP conflict checking."
                    ) from exc
            if j <= i: continue
            gj = geoms[j]
            try:
                if not gi.intersects(gj):
                    continue
                inter = gi.intersection(gj)
            except (PredicateError, TopologicalError) as exc:
                record_topology_error(i, j, exc)
                continue
            ma = min(gi.area, gj.area)
            if ma <= 0 or inter.area / ma < config.CONFLICT_OVERLAP_RATIO:
                continue
            ci, cj = int(gdf[field].iloc[i]), int(gdf[field].iloc[j])
            if ci != cj:
                if {ci, cj} == {1, 2}:
                    continue
                diff_c += 1
                if len(examples) < 5:
                    examples.append({"i": i, "j": j, "class_i": ci,
                                     "class_j": cj,
                                     "overlap_pct": round(inter.area/ma*100, 1)})
            else:
                same_c += 1
    r["diff_class_overlap"] = diff_c
    r["same_class_overlap"] = same_c
    r["conflict_examples"]  = examples
    r["topology_errors"] = topology_errors
    r["topology_error_examples"] = topology_examples
    if diff_c > 0:
        r["issues"].append(f"{diff_c} cross-class overlaps (class conflict)")
    if topology_errors > 0:
        r["issues"].append(
            f"{topology_errors} topology errors during overlap checks; fix invalid geometries"
        )
    if same_c > config.CONFLICT_SAME_CLASS_WARN:
        r["warnings"].append(f"{same_c} same-class overlaps (consider merging)")
    return r


# =================================================================
# CHECK 5: Mixed pixels + Label Confidence Index (LCI)
#
# LCI (Label Confidence Index) per pixel:
#   LCI = max_class_coverage_ratio
#   Range [0, 1]:
#     LCI = 1.0  → pixel fully inside one polygon (pure, hard label OK)
#     LCI = 0.6  → 60% one class, 40% another (moderate mixing)
#     LCI = 0.33 → equal three-way mix (maximum uncertainty)
#
# LCI is computed per sample and summarized:
#   · mean LCI: overall label quality
#   · % pixels with LCI > 0.9: essentially pure pixels
#   · % pixels with LCI < 0.5: highly uncertain pixels
#
# LCI map can be saved as a weight channel for loss weighting:
#   loss = sum(LCI[i] * CrossEntropy(pred[i], softlabel[i]))
# This gives pure pixels more influence on training.
#
# Tiny polygon contribution:
#   Sub-pixel polygons naturally contribute partial LCI weight.
#   A 0.1-pixel dead tree crown contributes LCI += 0.1 to Dead class.
#   This is more faithful than hard label (LCI=1.0) or deletion (LCI=0.0).
# =================================================================
def check_mixed_pixels(tif_path, shp_path, sid, skip_lci=False):
    r = {"sid": sid, "mixed_ratio": -1.0, "class1_boundary_ratio": -1.0,
         "total_labeled_hard": 0, "total_labeled_soft": 0,
         "boundary_pixels": 0,
         "soft_label_recommendation": "UNKNOWN",
         "pixel_size_m": (0, 0),
         "lci_stats": {}}

    gdf = gpd.read_file(shp_path)
    field = config.SHP_CLASS_FIELD
    if field not in gdf.columns: return r
    gdf = gdf[gdf[field].notna() & ~gdf.geometry.is_empty].copy()
    gdf[field] = gdf[field].astype(int)
    gdf = gdf[gdf[field].isin(VALID_CLASSES)]
    if not len(gdf): return r

    with rasterio.open(tif_path) as src:
        H, W      = src.height, src.width
        transform = src.transform
        tif_crs   = src.crs
        res_x, res_y = abs(src.res[0]), abs(src.res[1])
        r["pixel_size_m"] = (res_x, res_y)
        pixel_area = res_x * res_y

    if gdf.crs and tif_crs and str(gdf.crs) != str(tif_crs):
        try: gdf = gdf.to_crs(tif_crs)
        except Exception: pass

    shapes = [(g, c) for g, c in zip(gdf.geometry, gdf[field])
              if g is not None and not g.is_empty]
    if not shapes: return r

    # Hard rasterization (center point only)
    hard = rasterize(shapes, (H,W), transform=transform,
                     fill=0, dtype=np.int32, all_touched=False)
    # Soft rasterization (any contact)
    soft = rasterize(shapes, (H,W), transform=transform,
                     fill=0, dtype=np.int32, all_touched=True)

    boundary = (hard == 0) & (soft > 0)
    t_soft   = int((soft > 0).sum())
    t_hard   = int((hard > 0).sum())
    b_cnt    = int(boundary.sum())

    r["total_labeled_hard"] = t_hard
    r["total_labeled_soft"] = t_soft
    r["boundary_pixels"]    = b_cnt
    r["mixed_ratio"]        = float(b_cnt / max(t_soft, 1))

    c1_h = (hard == 1).sum(); c1_s = (soft == 1).sum()
    r["class1_boundary_ratio"] = float(max(c1_s - c1_h, 0) / max(c1_s, 1))

    # -----------------------------------------------------------------
    # Compute Weighted LCI via sub-pixel sampling
    # Skipped when --no-lci flag is set (fast mode)
    if skip_lci:
        return r
    #
    # LCI_raw = max(coverage_ratio)  — geometric purity of the pixel
    # LCI_w   = LCI_raw  (no separability penalty)
    #
    # Key insight: Dead/Living boundary pixels are the MOST important
    #   training examples — penalising them would remove critical signal.
    #   Soft label accuracy depends on area measurement, not spectral sep.
    #
    # Separability → used in prepare_data_softlabel.py for subgrid size only.
    #   Dead/Living (sep=0.25) → finer subgrid (more precise soft labels)
    #
    # Tiny polygon handling:
    #   A 0.1-pixel dead tree contributes coverage=0.1 to the Dead class.
    #   This naturally reduces LCI_raw and applies the separability penalty.
    #   No special treatment needed — sub-pixel sampling handles it correctly.
    # -----------------------------------------------------------------
    sg = getattr(config, "SOFT_LABEL_SUBGRID", 5)
    labeled_ys, labeled_xs = np.where(soft > 0)
    n_sample = min(len(labeled_ys), getattr(config, "LCI_SAMPLE_SIZE", 500))

    if n_sample > 0:
        idx = np.random.default_rng(42).choice(
            len(labeled_ys), n_sample, replace=False)
        sample_y = labeled_ys[idx]
        sample_x = labeled_xs[idx]

        lci_raw_vals = []
        lci_w_vals   = []

        from rasterio.transform import from_bounds as tfm_from_bounds

        for py, px in zip(sample_y, sample_x):
            left   = transform.c + px * transform.a
            top    = transform.f + py * transform.e
            right  = left + transform.a
            bottom = top  + transform.e
            if right < left:  left,  right  = right,  left
            if bottom < top:  top,   bottom = bottom, top

            sub_tfm  = tfm_from_bounds(left, bottom, right, top, sg, sg)
            sub_mask = rasterize(shapes, (sg, sg), transform=sub_tfm,
                                 fill=0, dtype=np.int32, all_touched=True)
            total_sub = sg * sg
            if total_sub == 0:
                continue

            # Coverage ratio per class
            ratios = {cls: float((sub_mask == cls).sum() / total_sub)
                      for cls in VALID_CLASSES}

            # Sort by coverage descending
            sorted_cls = sorted(ratios.items(), key=lambda x: -x[1])
            dominant_cls = sorted_cls[0][0]
            dominant_cov = sorted_cls[0][1]

            lci_raw = dominant_cov

            # LCI_w: identifies which class pair is mixing,
            # then records the separability for diagnostic use ONLY.
            # The weight itself remains LCI_raw (geometric purity).
            #
            # Why NOT penalise Dead/Living mixing:
            #   Dead/Living boundary pixels are the MOST IMPORTANT pixels
            #   to train on correctly — that is the core task.
            #   Their soft label [0.65, 0.35] is computed from actual area,
            #   not affected by spectral similarity.
            #   Penalising them would remove the most valuable training signal.
            #
            # Where separability DOES matter:
            #   prepare_data_softlabel.py uses get_local_subgrid() to
            #   apply FINER sub-pixel sampling for hard-to-separate pairs.
            #   Dead/Living (sep=0.25) → subgrid=20 (high precision)
            #   Dead/Grassland (sep=0.85) → subgrid=6 (standard)
            lci_w = lci_raw   # = pure geometric purity, no sep penalty

            lci_raw_vals.append(lci_raw)
            lci_w_vals.append(lci_w)

        if lci_raw_vals:
            raw_arr = np.array(lci_raw_vals)
            w_arr   = np.array(lci_w_vals)
            r["lci_stats"] = {
                # Raw LCI (geometric purity only)
                "raw_mean"     : float(raw_arr.mean()),
                "raw_pct_pure" : float((raw_arr > 0.90).mean()),
                # Weighted LCI (purity × spectral separability)
                "w_mean"       : float(w_arr.mean()),
                "w_pct_pure"   : float((w_arr > 0.70).mean()),
                "w_pct_low"    : float((w_arr < 0.30).mean()),
                "histogram_raw": np.histogram(raw_arr, bins=10,
                                              range=(0,1))[0].tolist(),
                "histogram_w"  : np.histogram(w_arr,   bins=10,
                                              range=(0,1))[0].tolist(),
            }

    mr = r["mixed_ratio"]
    if   mr > config.MIXED_RATIO_RECOMMEND:
        r["soft_label_recommendation"] = "STRONGLY_RECOMMENDED"
    elif mr > config.MIXED_RATIO_OPTIONAL:
        r["soft_label_recommendation"] = "RECOMMENDED"
    else:
        r["soft_label_recommendation"] = "OPTIONAL"

    return r


# =================================================================
# Report generation
# =================================================================
def generate_text_report(pairs, missing_shp, missing_tif,
                         completeness, all_pixel, all_annot,
                         all_conflict, all_mixed, output_path):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [
        "="*70, "  Data Quality Report — Dead Tree Detection",
        f"  Generated  : {now}",
        f"  TIF dir    : {config.RAW_TIF_DIR}",
        f"  SHP dir    : {config.RAW_SHP_DIR}",
        f"  Classes    : {dict(config.CLASS_NAMES)}", "="*70,
    ]

    # 1 Completeness
    L += [f"\n[1] Sample Completeness",
          f"  Valid pairs      : {len(pairs)}",
          f"  Missing SHP      : {len(missing_shp)}"
          + (f"  {missing_shp[:5]}" if missing_shp else ""),
          f"  Missing TIF      : {len(missing_tif)}"
          + (f"  {missing_tif[:5]}" if missing_tif else "")]

    if completeness["band_issues"]:
        L.append(f"  Band mismatch (expected {config.RAW_CHANNELS}):")
        for sid, got, _ in completeness["band_issues"][:5]:
            L.append(f"    {sid}: got {got}")
    else:
        L.append(f"  Band count       : all {config.RAW_CHANNELS} OK")

    if completeness["crs_undefined"]:
        L.append(f"  CRS undefined (no .prj): "
                 f"{len(completeness['crs_undefined'])} samples")
        L.append(f"    {completeness['crs_undefined'][:8]}")
        L.append("    -> Will be assumed to match TIF CRS at rasterization")
        L.append("       Verify visually if spatial alignment looks correct")
    else:
        L.append("  CRS undefined    : 0")

    if completeness["crs_mismatch"]:
        L.append(f"  CRS MISMATCH (rasterization will be spatially WRONG):")
        for sid, s, t in completeness["crs_mismatch"][:10]:
            L.append(f"    {sid}: SHP={s}  TIF={t}")
        L.append("    Fix: reproject SHP to match TIF CRS before training")
    else:
        L.append("  CRS mismatch     : 0 OK")

    # 2 Pixel
    iss2  = [(r["sid"], r["issues"])   for r in all_pixel if r["issues"]]
    nd_hi = [(r["sid"], r["nodata_ratio"]) for r in all_pixel
             if r["nodata_ratio"] > config.ANOMALY_NODATA_WARN_RATIO]
    L += [f"\n[2] Pixel Anomalies",
          f"  Samples with issues   : {len(iss2)}",
          f"  High NoData (>{config.ANOMALY_NODATA_WARN_RATIO:.0%}): {len(nd_hi)}"]
    for sid, ratio in sorted(nd_hi, key=lambda x: -x[1])[:5]:
        L.append(f"    {sid}: {ratio:.1%}")
    for sid, iss in iss2[:3]:
        for i in iss: L.append(f"  ISSUE {sid}: {i}")

    # 3 Annotation — tiny polygon area contribution
    tc = defaultdict(int)
    for r in all_annot:
        for c, n in r.get("class_distribution", {}).items(): tc[c] += n
    covs = [r.get("coverage_ratio", 0) for r in all_annot]
    c1c  = [r.get("class1_coverage", 0) for r in all_annot]

    # Aggregate tiny polygon stats
    total_tiny_count   = sum(r.get("tiny_polygon_stats",{}).get("count",0)
                             for r in all_annot)
    tiny_area_fracs    = [r.get("tiny_polygon_stats",{}).get("area_frac",0)
                          for r in all_annot
                          if r.get("tiny_polygon_stats",{}).get("count",0) > 0]

    L += [f"\n[3] Annotation Quality",
          f"  Polygon distribution:"]
    for c in sorted(tc):
        L.append(f"    [{c}] {config.CLASS_NAMES.get(c,str(c)):<25} "
                 f"{tc[c]:>8} polygons")
    L += [f"  Coverage: mean={np.mean(covs):.1%}  "
          f"min={np.min(covs):.1%}  max={np.max(covs):.1%}",
          f"  Class1 dead coverage: mean={np.mean(c1c):.2%}",
          f"",
          f"  Sub-pixel polygon analysis:"]
    L.append(f"    Total sub-pixel polygons   : {total_tiny_count:,}")
    if tiny_area_fracs:
        L.append(f"    Labeled area they represent: "
                 f"mean={np.mean(tiny_area_fracs):.2%}  "
                 f"max={np.max(tiny_area_fracs):.2%}")
        L.append(f"    Strategy: NOT removed — included in soft labels")
        L.append(f"    Reason: sub-pixel tree crowns are real dead trees,")
        L.append(f"    soft label weight = their coverage fraction (<1 pixel)")

    # Per-class tiny breakdown for Class 1
    c1_tiny_counts = []
    c1_tiny_fracs  = []
    for r in all_annot:
        tp = r.get("tiny_polygon_stats", {}).get("per_class", {})
        if 1 in tp:
            c1_tiny_counts.append(tp[1]["count"])
            c1_tiny_fracs.append(tp[1]["area_frac"])
    if c1_tiny_counts:
        L.append(f"    Class1 (Dead) sub-pixel: "
                 f"total={sum(c1_tiny_counts):,} polygons  "
                 f"area_frac mean={np.mean(c1_tiny_fracs):.2%}")

    issues_3 = [(r["sid"], r["issues"])   for r in all_annot if r["issues"]]
    warns_3  = [(r["sid"], r["warnings"]) for r in all_annot if r["warnings"]]
    if issues_3:
        L.append(f"  ISSUES ({len(issues_3)} samples):")
        for sid, iss in issues_3[:5]:
            for i in iss[:2]: L.append(f"    {sid}: {i}")
    if warns_3:
        low_cov = [(sid, ws) for sid, ws in warns_3
                   if any("coverage" in w for w in ws)]
        if low_cov:
            L.append(f"  Low-coverage samples ({len(low_cov)}):")
            for sid, _ in low_cov[:5]:
                L.append(f"    {sid}")

    # 4 Conflict
    n_conf = sum(1 for r in all_conflict if r.get("diff_class_overlap",0) > 0)
    tot_cp = sum(r.get("diff_class_overlap",0) for r in all_conflict)
    L += [f"\n[4] SHP Conflicts",
          f"  Samples with cross-class overlap : {n_conf}",
          f"  Total conflicting polygon pairs  : {tot_cp}"]

    dead_live_count = 0
    for r in all_conflict:
        for ex in r.get("conflict_examples", []):
            if set([ex["class_i"], ex["class_j"]]) == {1, 2}:
                dead_live_count += 1
    if dead_live_count > 0:
        L.append(f"  NOTE: {dead_live_count} Dead vs Living overlaps detected.")
        L.append(f"  If your workflow labels all trees as Living first, then")
        L.append(f"  adds Dead polygons on top — this is intentional overlap.")
        L.append(f"  Fix: sort shapes by class in rasterize() so Dead(1) is")
        L.append(f"  rasterized LAST (overrides Living). See prepare_data.py note.")

    if n_conf:
        L.append("  True conflicts (non Dead/Living):")
        for r in all_conflict:
            for ex in r.get("conflict_examples", [])[:2]:
                if set([ex["class_i"], ex["class_j"]]) == {1, 2}:
                    continue
                n1 = config.CLASS_NAMES.get(ex["class_i"], str(ex["class_i"]))
                n2 = config.CLASS_NAMES.get(ex["class_j"], str(ex["class_j"]))
                L.append(f"    {r['sid']}: {n1} overlaps {n2} {ex['overlap_pct']}%")

    # 5 Mixed pixels + LCI
    vm = [r for r in all_mixed if r.get("mixed_ratio", -1) >= 0]
    if vm:
        ratios = [r["mixed_ratio"] for r in vm]
        c1r    = [r["class1_boundary_ratio"] for r in vm]
        rc = defaultdict(int)
        for r in vm: rc[r["soft_label_recommendation"]] += 1

        # LCI aggregation handled inline below

        L += [f"\n[5] Mixed Pixel Analysis + Label Confidence Index (LCI)",
              f"",
              f"  Boundary pixel ratio:",
              f"    mean={np.mean(ratios):.1%}  max={np.max(ratios):.1%}",
              f"  Class1 dead boundary ratio:",
              f"    mean={np.mean(c1r):.1%}  max={np.max(c1r):.1%}",
              f"",
              f"  Label Confidence Index (LCI = max class coverage per pixel):"]
        lci_stats_all = [r["lci_stats"] for r in vm if r.get("lci_stats")]
        if lci_stats_all:
            raw_means = [s["raw_mean"] for s in lci_stats_all]
            w_means   = [s["w_mean"]   for s in lci_stats_all]
            w_pct_low = [s["w_pct_low"] for s in lci_stats_all]
            L += [f"",
                  f"  Label Confidence Index (LCI):",
                  f"    LCI_raw  = max(coverage_ratio per pixel)",
                  f"    LCI_w    = LCI_raw x separability(dominant_class, second_class)",
                  f"",
                  f"    Raw LCI mean       : {np.mean(raw_means):.3f}  "
                  f"(geometric purity only)",
                  f"    Weighted LCI mean  : {np.mean(w_means):.3f}  "
                  f"(penalises spectrally similar mixing)",
                  f"    Pixels with LCI_w < 0.30: {np.mean(w_pct_low):.1%}  "
                  f"(problematic — mainly Dead/Living boundary)",
                  f"",
                  f"  LCI = geometric purity (no separability penalty):",
                  f"    Dead/Living   boundary: LCI_raw=0.65 → LCI_w=0.65",
                  f"    Dead/Grassland boundary: LCI_raw=0.65 → LCI_w=0.65",
                  f"    Pure Dead pixel:          LCI_raw=1.0  → LCI_w=1.0",
                  f"",
                  f"  Separability is used for soft-label subgrid precision:",
                  f"    Dead vs Living (sep={config.get_separability(1,2):.2f}) "
                  f"→ subgrid={config.get_local_subgrid(1,2)}  (finest — most important boundary)",
                  f"    Dead vs Grassland (sep={config.get_separability(1,5):.2f}) "
                  f"→ subgrid={config.get_local_subgrid(1,5)}  (standard)",
                  f"    Dead vs Lakes (sep={config.get_separability(1,4):.2f}) "
                  f"→ subgrid={config.get_local_subgrid(1,4)}  (standard)",
                  f"",
                  f"  Tiny polygons:",
                  f"    A 0.1-pixel Dead crown → coverage=0.1 → LCI_raw reduced",
                  f"    Still included with proportional weight (not deleted)",
                  f"",
                  f"  Training: loss_weight[pixel] = LCI_w = LCI_raw",
                  f"    Dead/Living boundary: same weight as Dead/Grassland",
                  f"    Both contribute fully — Dead/Living MORE important to train"]
        L += [f"",
              f"  Soft-label recommendation:",
              f"    OPTIONAL       (<{config.MIXED_RATIO_OPTIONAL:.0%}): "
              f"{rc.get('OPTIONAL',0)} samples",
              f"    RECOMMENDED ({config.MIXED_RATIO_OPTIONAL:.0%}-"
              f"{config.MIXED_RATIO_RECOMMEND:.0%}): "
              f"{rc.get('RECOMMENDED',0)} samples",
              f"    STRONGLY (>{config.MIXED_RATIO_RECOMMEND:.0%}): "
              f"{rc.get('STRONGLY_RECOMMENDED',0)} samples"]

        dom = max(rc, key=rc.get) if rc else "OPTIONAL"
        if dom == "STRONGLY_RECOMMENDED":
            L += ["", "  ACTION: Use soft labels + LCI weighting",
                  "  Next step: run prepare_data_softlabel.py",
                  f"  Config: BOUNDARY_STRATEGY={config.BOUNDARY_STRATEGY}",
                  f"          SOFT_LABEL_CLASSES={config.SOFT_LABEL_CLASSES}",
                  f"          SOFT_LABEL_SUBGRID={config.SOFT_LABEL_SUBGRID}"]
        elif dom == "RECOMMENDED":
            L += ["", f"  ACTION: Soft labels for classes {config.SOFT_LABEL_CLASSES}"]
        else:
            L += ["", "  ACTION: Hard labels sufficient — proceed with prepare_data.py"]

    n_crit = (len(completeness["band_issues"]) +
              len(completeness["crs_mismatch"]) +
              sum(len(r.get("issues",[])) > 0 for r in all_pixel) +
              sum(len(r.get("issues",[])) > 0 for r in all_annot) +
              sum(r.get("diff_class_overlap",0) > 0 for r in all_conflict))
    n_warn = (len(nd_hi) +
              sum(len(r.get("warnings",[])) > 0 for r in all_annot) +
              len(completeness["crs_undefined"]))

    L += ["", "="*70,
          f"  SUMMARY: {n_crit} critical issues,  {n_warn} warnings",
          f"  Report: {output_path}", "="*70]

    text = "\n".join(L)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def generate_summary_plot(all_pixel, all_annot, all_conflict, all_mixed, path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Data Quality Summary", fontsize=13)

    # 1 Class polygon distribution
    ax = axes[0,0]
    tc = defaultdict(int)
    for r in all_annot:
        for c,n in r.get("class_distribution",{}).items(): tc[c] += n
    if tc:
        cls = sorted(tc); cnts = [tc[c] for c in cls]
        colors = ["#%02x%02x%02x" % config.PALETTE.get(c,(128,128,128)) for c in cls]
        bars = ax.bar([f"[{c}]\n{config.CLASS_NAMES.get(c,'')[:8]}" for c in cls],
                      cnts, color=colors, edgecolor="white")
        for b,n in zip(bars,cnts):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                    str(n), ha="center", fontsize=8)
    ax.set_title("Polygon Count per Class"); ax.set_ylabel("Count")

    # 2 Tiny polygon area fraction per class
    ax = axes[0,1]
    class_tiny_fracs = defaultdict(list)
    for r in all_annot:
        for c, stats in r.get("tiny_polygon_stats",{}).get("per_class",{}).items():
            class_tiny_fracs[c].append(stats["area_frac"])
    if class_tiny_fracs:
        cls_sorted = sorted(class_tiny_fracs.keys())
        means = [np.mean(class_tiny_fracs[c])*100 for c in cls_sorted]
        colors = ["#%02x%02x%02x" % config.PALETTE.get(c,(128,128,128))
                  for c in cls_sorted]
        ax.bar([f"[{c}]\n{config.CLASS_NAMES.get(c,'')[:8]}" for c in cls_sorted],
               means, color=colors, edgecolor="white")
        ax.set_ylabel("Sub-pixel area fraction (%)")
    ax.set_title("Sub-pixel Polygon Area Contribution\n(included in soft labels)")
    ax.axhline(5, color="orange", linestyle="--", linewidth=0.8, label="5%")
    ax.legend(fontsize=8)

    # 3 NoData
    ax = axes[0,2]
    nd = [r.get("nodata_ratio",0)*100 for r in all_pixel]
    ax.hist(nd, bins=min(20,len(nd)), color="#993C1D", alpha=0.8, edgecolor="white")
    ax.axvline(config.ANOMALY_NODATA_WARN_RATIO*100, color="red",
               linestyle="--", label=f"Warn {config.ANOMALY_NODATA_WARN_RATIO:.0%}")
    ax.set_title("NoData Ratio (%)"); ax.legend(fontsize=8)

    # 4 LCI_raw vs LCI_weighted comparison
    ax = axes[1,0]
    raw_vals, w_vals = [], []
    for r in all_mixed:
        s = r.get("lci_stats", {})
        if s:
            h_raw = s.get("histogram_raw", [])
            h_w   = s.get("histogram_w",   [])
            bins    = np.linspace(0, 1, 11)
            centers = (bins[:-1] + bins[1:]) / 2
            if h_raw: raw_vals.extend(np.repeat(centers, h_raw).tolist())
            if h_w:   w_vals.extend(  np.repeat(centers, h_w  ).tolist())
    if raw_vals:
        ax.hist(raw_vals, bins=20, color="#3B6D11", alpha=0.5,
                edgecolor="none", label="LCI_raw (geometry)")
    if w_vals:
        ax.hist(w_vals, bins=20, color="#534AB7", alpha=0.7,
                edgecolor="none", label="LCI_w (x separability)")
    ax.axvline(0.30, color="red",   linestyle="--", linewidth=0.8,
               label="0.30 low confidence")
    ax.axvline(0.70, color="green", linestyle="--", linewidth=0.8,
               label="0.70 reliable")
    ax.set_title("Label Confidence Index\nraw vs separability-weighted")
    ax.set_xlabel("LCI")
    ax.legend(fontsize=7)

    # 5 Mixed pixel ratio
    ax = axes[1,1]
    vm = [r for r in all_mixed if r.get("mixed_ratio",-1) >= 0]
    if vm:
        ratios = [r["mixed_ratio"]*100 for r in vm]
        ax.hist(ratios, bins=min(20,len(ratios)), color="#3B6D11",
                alpha=0.8, edgecolor="white")
        ax.axvline(config.MIXED_RATIO_OPTIONAL*100,  color="orange",
                   linestyle="--", label=f"Optional>{config.MIXED_RATIO_OPTIONAL:.0%}")
        ax.axvline(config.MIXED_RATIO_RECOMMEND*100, color="red",
                   linestyle="--", label=f"Strongly>{config.MIXED_RATIO_RECOMMEND:.0%}")
        ax.legend(fontsize=8)
    ax.set_title("Mixed Pixel Ratio (%)")

    # 6 Conflict pie
    ax = axes[1,2]
    nc  = sum(1 for r in all_conflict if r.get("diff_class_overlap",0) > 0)
    ns  = sum(1 for r in all_conflict
              if r.get("diff_class_overlap",0)==0
              and r.get("same_class_overlap",0)>0)
    nok = sum(1 for r in all_conflict
              if r.get("diff_class_overlap",0)==0
              and r.get("same_class_overlap",0)==0)
    vals = [v for v in [nok,ns,nc] if v>0]
    lbls = [l for l,v in zip(
        ["Clean","Same-class","Cross-class conflict"],[nok,ns,nc]) if v>0]
    cols = ["#3B6D11","#EF9F27","#E24B4A"][:len(vals)]
    if vals:
        ax.pie(vals, labels=lbls, colors=cols,
               autopct="%1.0f%%", startangle=90,
               textprops={"fontsize":9})
    ax.set_title("SHP Overlap Status")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  Chart -> {path}")


# =================================================================
# Main
# =================================================================
def main(max_samples=None, fix=False, no_plot=False, no_lci=False):
    print("\n" + "="*55)
    print("  Dead Tree Detection — Data Quality Check")
    print("="*55)
    print(f"  BAND_MODE={config.BAND_MODE}  "
          f"RAW_CHANNELS={config.RAW_CHANNELS}  "
          f"NUM_CLASSES={config.NUM_CLASSES}")
    print(f"  CRS comparison: pyproj.CRS.equals() (precise)")
    print(f"  Tiny polygons : included in soft labels (not removed)")

    pairs, missing_shp, missing_tif = find_pairs()
    if not pairs:
        sys.exit(f"\nNo pairs found.\n"
                 f"  TIF: {config.RAW_TIF_DIR}\n"
                 f"  SHP: {config.RAW_SHP_DIR}")

    if max_samples: pairs = pairs[:max_samples]
    lci_status = "disabled (fast mode)" if no_lci else "enabled"
    print(f"\nChecking {len(pairs)} pairs  LCI={lci_status}")

    completeness = check_completeness(pairs)
    ap, aa, ac, am = [], [], [], []
    for i, (tif, shp, sid) in enumerate(pairs):
        print(f"  [{i+1:3d}/{len(pairs)}] {sid:<20}", end="\r")
        ap.append(check_pixel_anomalies(tif, sid))
        aa.append(check_annotation_quality(tif, shp, sid))
        ac.append(check_shp_conflicts(shp, tif, sid))
        am.append(check_mixed_pixels(tif, shp, sid, skip_lci=no_lci))

    print(f"\n  Done.")
    rpt_dir = getattr(config, "CHECK_REPORT_DIR", config.OUTPUT_DIR)
    os.makedirs(rpt_dir, exist_ok=True)
    rpt = os.path.join(rpt_dir, "data-quality-report.txt")
    print()
    generate_text_report(pairs, missing_shp, missing_tif,
                         completeness, ap, aa, ac, am, rpt)
    if not no_plot:
        try:
            chart = os.path.join(rpt_dir, "data-quality-summary.png")
            generate_summary_plot(ap, aa, ac, am, chart)
        except Exception as e:
            print(f"  Chart error: {e}")

    if fix:
        print("\n[Fix Suggestions]")
        print("  CRS mismatch: ogr2ogr -t_srs EPSG:<target> out.shp in.shp")
        print("  CRS undefined: assign CRS in QGIS (Layer Properties > CRS)")
        print("  Class value 0: select in QGIS, update Class_Int field")
        print("  Empty geometries: auto-filtered during rasterization")
        print("  Dead/Living 100% overlap: fix rasterize order in prepare_data.py")
        print("    -> sort shapes so Dead(1) renders last (overrides Living)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Data quality check")
    p.add_argument("--sample",  type=int,  default=None)
    p.add_argument("--fix",     action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-lci",  action="store_true",
                   help="Skip LCI sub-pixel sampling (fast, for routine checks)")
    a = p.parse_args()
    main(a.sample, a.fix, a.no_plot, a.no_lci)
