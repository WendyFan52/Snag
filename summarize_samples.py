"""
Summarize sample shapefiles with dead-area and dead-patch metrics.

This summary avoids tree-count based assumptions and uses geometry-only
metrics that remain valid when tree labels are merged into patches.
"""

import argparse
import csv
import os

import config
from sample_remote_sensing import build_image_lookup, first_image_for_sample, summarize_sample_image


def load_manifest(path: str):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _percentile(values, q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _to_projected(gdf):
    if gdf.crs is None:
        raise RuntimeError("Shapefile CRS is missing; cannot compute area/density metrics reliably")
    if not gdf.crs.is_geographic:
        return gdf
    utm = gdf.estimate_utm_crs()
    if utm is not None:
        return gdf.to_crs(utm)
    return gdf.to_crs(epsg=3857)


def _extract_patch_areas(dead_union):
    if dead_union is None or dead_union.is_empty:
        return []
    geoms = getattr(dead_union, "geoms", None)
    if geoms is None:
        return [float(dead_union.area)]
    return [float(g.area) for g in geoms if g is not None and not g.is_empty and g.area > 0]


def _safe_union(geoms):
    from shapely.ops import unary_union

    try:
        return unary_union(geoms), False
    except Exception:
        repaired = []
        for geom in geoms:
            try:
                repaired.append(geom.buffer(0))
            except Exception:
                continue
        repaired = [g for g in repaired if g is not None and not g.is_empty]
        if not repaired:
            return None, True
        try:
            return unary_union(repaired), True
        except Exception:
            return None, True


def _class_union_area(gdf, field: str, class_values):
    subset = gdf[gdf[field].isin(class_values)].copy()
    if subset.empty:
        return 0.0, False
    geom_union, fallback = _safe_union(subset.geometry)
    if geom_union is not None and not geom_union.is_empty:
        return float(geom_union.area), fallback
    return float(subset.geometry.area.sum()), True


def _patch_size_stats(patch_areas, dead_area_m2: float):
    small_max = config.PATCH_SMALL_MAX_M2
    medium_max = config.PATCH_MEDIUM_MAX_M2
    small = [a for a in patch_areas if a <= small_max]
    medium = [a for a in patch_areas if small_max < a <= medium_max]
    large = [a for a in patch_areas if a > medium_max]

    def area(values):
        return float(sum(values))

    small_area = area(small)
    medium_area = area(medium)
    large_area = area(large)
    denom = float(dead_area_m2) if dead_area_m2 > 0 else 0.0
    return {
        "small_patch_count": len(small),
        "medium_patch_count": len(medium),
        "large_patch_count": len(large),
        "small_patch_area_m2": small_area,
        "medium_patch_area_m2": medium_area,
        "large_patch_area_m2": large_area,
        "small_patch_area_fraction": float(small_area / denom) if denom > 0 else 0.0,
        "medium_patch_area_fraction": float(medium_area / denom) if denom > 0 else 0.0,
        "large_patch_area_fraction": float(large_area / denom) if denom > 0 else 0.0,
    }


def _repair_geometries(gdf):
    try:
        from shapely import make_valid
        gdf["geometry"] = gdf.geometry.apply(lambda geom: make_valid(geom) if geom is not None and not geom.is_empty else geom)
    except Exception:
        # Fallback for environments without shapely.make_valid
        gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    return gdf


def summarize_one_shp(shp_path: str):
    try:
        import geopandas as gpd
        from shapely.ops import unary_union
    except ImportError:
        raise RuntimeError("geopandas and shapely are required to summarize shapefiles")

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        return {
            "total_objects": 0,
            "labeled_area_m2": 0.0,
            "dead_area_m2": 0.0,
            "dead_area_on_labeled_ratio": 0.0,
            "dead_patch_count": 0,
            "largest_dead_patch_area_m2": 0.0,
            "dead_patch_area_p50_m2": 0.0,
            "dead_patch_area_p75_m2": 0.0,
            "dead_patch_area_p90_m2": 0.0,
            "dead_aggregation_index": 0.0,
            "patch_areas_m2": [],
        }

    field = config.SHP_CLASS_FIELD if config.SHP_CLASS_FIELD in gdf.columns else None
    if field is None:
        field = next((c for c in gdf.columns if c.lower() == config.SHP_CLASS_FIELD.lower()), None)
    if field is None:
        raise KeyError(f"Class field not found in {shp_path}")

    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return {
            "total_objects": 0,
            "labeled_area_m2": 0.0,
            "dead_area_m2": 0.0,
            "dead_area_on_labeled_ratio": 0.0,
            "dead_patch_count": 0,
            "largest_dead_patch_area_m2": 0.0,
            "dead_patch_area_p50_m2": 0.0,
            "dead_patch_area_p75_m2": 0.0,
            "dead_patch_area_p90_m2": 0.0,
            "dead_aggregation_index": 0.0,
            "patch_areas_m2": [],
        }

    gdf[field] = gdf[field].fillna(-1).astype(int)
    gdf = _repair_geometries(gdf)
    if gdf.empty:
        return {
            "total_objects": 0,
            "labeled_area_m2": 0.0,
            "dead_area_m2": 0.0,
            "dead_area_on_labeled_ratio": 0.0,
            "dead_patch_count": 0,
            "largest_dead_patch_area_m2": 0.0,
            "dead_patch_area_p50_m2": 0.0,
            "dead_patch_area_p75_m2": 0.0,
            "dead_patch_area_p90_m2": 0.0,
            "dead_aggregation_index": 0.0,
            "patch_areas_m2": [],
        }
    gdf = _to_projected(gdf)

    labeled_union, labeled_union_fallback = _safe_union(gdf.geometry)
    if labeled_union is not None and not labeled_union.is_empty:
        labeled_area_m2 = float(labeled_union.area)
    else:
        labeled_area_m2 = float(gdf.geometry.area.sum())
        labeled_union_fallback = True

    living_area_m2, living_union_fallback = _class_union_area(gdf, field, {2})
    tree_area_m2, tree_union_fallback = _class_union_area(gdf, field, {1, 2})
    water_area_m2, water_union_fallback = _class_union_area(gdf, field, {4})
    non_tree_area_m2 = max(labeled_area_m2 - tree_area_m2, 0.0)

    dead = gdf[gdf[field] == 1].copy()
    if dead.empty:
        return {
            "total_objects": len(gdf),
            "labeled_area_m2": labeled_area_m2,
            "tree_area_m2": tree_area_m2,
            "living_area_m2": living_area_m2,
            "water_area_m2": water_area_m2,
            "non_tree_area_m2": non_tree_area_m2,
            "dead_area_m2": 0.0,
            "dead_area_on_labeled_ratio": 0.0,
            "tree_area_on_labeled_ratio": float(tree_area_m2 / labeled_area_m2) if labeled_area_m2 > 0 else 0.0,
            "living_area_on_labeled_ratio": float(living_area_m2 / labeled_area_m2) if labeled_area_m2 > 0 else 0.0,
            "water_area_on_labeled_ratio": float(water_area_m2 / labeled_area_m2) if labeled_area_m2 > 0 else 0.0,
            "dead_patch_count": 0,
            "largest_dead_patch_area_m2": 0.0,
            "dead_patch_area_p50_m2": 0.0,
            "dead_patch_area_p75_m2": 0.0,
            "dead_patch_area_p90_m2": 0.0,
            "dead_aggregation_index": 0.0,
            "patch_areas_m2": [],
            "union_fallback_used": int(labeled_union_fallback or living_union_fallback or tree_union_fallback or water_union_fallback),
        }

    dead_union, dead_union_fallback = _safe_union(dead.geometry)
    if dead_union is not None and not dead_union.is_empty:
        dead_area_m2 = float(dead_union.area)
        patch_areas = _extract_patch_areas(dead_union)
    else:
        dead_area_m2 = float(dead.geometry.area.sum())
        patch_areas = [float(a) for a in dead.geometry.area.tolist() if a > 0]
        dead_union_fallback = True
    patch_count = len(patch_areas)
    largest_patch = max(patch_areas) if patch_areas else 0.0
    patch_stats = _patch_size_stats(patch_areas, dead_area_m2)

    return {
        "total_objects": len(gdf),
        "labeled_area_m2": labeled_area_m2,
        "tree_area_m2": tree_area_m2,
        "living_area_m2": living_area_m2,
        "water_area_m2": water_area_m2,
        "non_tree_area_m2": non_tree_area_m2,
        "dead_area_m2": dead_area_m2,
        "dead_area_on_labeled_ratio": float(dead_area_m2 / labeled_area_m2) if labeled_area_m2 > 0 else 0.0,
        "tree_area_on_labeled_ratio": float(tree_area_m2 / labeled_area_m2) if labeled_area_m2 > 0 else 0.0,
        "living_area_on_labeled_ratio": float(living_area_m2 / labeled_area_m2) if labeled_area_m2 > 0 else 0.0,
        "water_area_on_labeled_ratio": float(water_area_m2 / labeled_area_m2) if labeled_area_m2 > 0 else 0.0,
        "dead_patch_count": patch_count,
        "largest_dead_patch_area_m2": largest_patch,
        "dead_patch_area_p50_m2": _percentile(patch_areas, 0.50),
        "dead_patch_area_p75_m2": _percentile(patch_areas, 0.75),
        "dead_patch_area_p90_m2": _percentile(patch_areas, 0.90),
        "dead_aggregation_index": float(largest_patch / dead_area_m2) if dead_area_m2 > 0 else 0.0,
        **patch_stats,
        "patch_areas_m2": patch_areas,
        "union_fallback_used": int(
            labeled_union_fallback
            or living_union_fallback
            or tree_union_fallback
            or water_union_fallback
            or dead_union_fallback
        ),
    }


def summarize_manifest(
    manifest_csv: str,
    output_csv: str,
    cluster_csv: str,
    cluster_distance_m: float = 3.0,
    image_dir: str = None,
    require_images: bool = True,
    allow_geometry_fallback: bool = False,
):
    manifest_rows = load_manifest(manifest_csv)
    image_lookup = build_image_lookup(image_dir or getattr(config, "SAMPLE_IMAGE_DIR", config.RAW_TIF_DIR))
    print(f"[summary] image lookup sample_ids: {len(image_lookup)}")
    summary_rows = []
    cluster_rows = []

    for row in manifest_rows:
        shp_path = row.get("truth_shp_path", "")
        if not shp_path or not os.path.exists(shp_path):
            raise FileNotFoundError(
                f"truth_shp_path not found for sample_id={row.get('sample_id', '')}: {shp_path}"
            )
        stats = summarize_one_shp(shp_path)
        sample_id = row.get("sample_id", "")
        ps_image_path = row.get("ps_image_path", "") or first_image_for_sample(sample_id, image_lookup)
        rs_stats = summarize_sample_image(sample_id, ps_image_path, image_lookup=image_lookup)
        if require_images and int(rs_stats.get("rs_available", 0)) <= 0:
            raise RuntimeError(
                f"Remote-sensing image is required but unavailable for sample_id={sample_id}. "
                f"Set --image_dir correctly or pass --allow_missing_images."
            )

        labeled_area_m2 = float(stats.get("labeled_area_m2", 0.0))
        valid_area_m2 = float(rs_stats.get("valid_area_m2", 0.0))
        valid_area_source = str(rs_stats.get("valid_area_source", "missing_remote_sensing"))
        if valid_area_m2 <= 0:
            if require_images:
                raise RuntimeError(
                    f"Remote-sensing valid area is zero for sample_id={sample_id}, image={ps_image_path}"
                )
            valid_area_m2 = labeled_area_m2
            valid_area_source = "labeled_area_fallback"
            print(f"[summary][warn] sample_id={sample_id} uses labeled_area_m2 as valid_area_m2 because --allow_missing_images is active")

        dead_area_m2 = float(stats.get("dead_area_m2", 0.0))
        living_area_m2 = float(stats.get("living_area_m2", 0.0))
        rs_veg_fraction = float(rs_stats.get("rs_veg_fraction", 0.0))
        rs_veg_area_m2 = rs_veg_fraction * valid_area_m2
        dead_cover_fraction = float(dead_area_m2 / valid_area_m2) if valid_area_m2 > 0 else 0.0
        living_observed_cover_fraction = float(living_area_m2 / valid_area_m2) if valid_area_m2 > 0 else 0.0
        dead_relative_to_veg = float(dead_area_m2 / rs_veg_area_m2) if rs_veg_area_m2 > 0 else 0.0

        summary_rows.append(
            {
                **row,
                "ps_image_path": ps_image_path,
                "total_objects": stats["total_objects"],
                "labeled_area_m2": round(stats["labeled_area_m2"], 4),
                "valid_area_m2": round(valid_area_m2, 4),
                "valid_area_source": valid_area_source,
                "rs_available": int(rs_stats.get("rs_available", 0)),
                "rs_image_path": rs_stats.get("rs_image_path", ""),
                "rs_image_paths": rs_stats.get("rs_image_paths", ""),
                "rs_image_count": int(rs_stats.get("rs_image_count", 0)),
                "rs_valid_image_count": int(rs_stats.get("rs_valid_image_count", 0)),
                "rs_aggregation_method": rs_stats.get("rs_aggregation_method", ""),
                "valid_pixel_count": int(rs_stats.get("valid_pixel_count", 0)),
                "rs_ndvi_mean": round(float(rs_stats.get("rs_ndvi_mean", 0.0)), 8),
                "rs_ndvi_p25": round(float(rs_stats.get("rs_ndvi_p25", 0.0)), 8),
                "rs_ndvi_p50": round(float(rs_stats.get("rs_ndvi_p50", 0.0)), 8),
                "rs_ndvi_p75": round(float(rs_stats.get("rs_ndvi_p75", 0.0)), 8),
                "rs_ndwi_mean": round(float(rs_stats.get("rs_ndwi_mean", 0.0)), 8),
                "rs_ndwi_p25": round(float(rs_stats.get("rs_ndwi_p25", 0.0)), 8),
                "rs_ndwi_p50": round(float(rs_stats.get("rs_ndwi_p50", 0.0)), 8),
                "rs_ndwi_p75": round(float(rs_stats.get("rs_ndwi_p75", 0.0)), 8),
                "rs_veg_fraction": round(rs_veg_fraction, 8),
                "rs_water_fraction": round(float(rs_stats.get("rs_water_fraction", 0.0)), 8),
                "rs_bare_fraction": round(float(rs_stats.get("rs_bare_fraction", 0.0)), 8),
                "rs_veg_area_m2": round(rs_veg_area_m2, 4),
                "tree_area_m2": round(stats.get("tree_area_m2", 0.0), 4),
                "living_area_m2": round(stats.get("living_area_m2", 0.0), 4),
                "water_area_m2": round(stats.get("water_area_m2", 0.0), 4),
                "non_tree_area_m2": round(stats.get("non_tree_area_m2", 0.0), 4),
                "dead_area_m2": round(stats["dead_area_m2"], 4),
                "dead_cover_fraction": round(dead_cover_fraction, 8),
                "living_observed_cover_fraction": round(living_observed_cover_fraction, 8),
                "dead_relative_to_veg": round(dead_relative_to_veg, 8),
                "dead_area_on_labeled_ratio": round(stats["dead_area_on_labeled_ratio"], 8),
                "tree_area_on_labeled_ratio": round(stats.get("tree_area_on_labeled_ratio", 0.0), 8),
                "living_area_on_labeled_ratio": round(stats.get("living_area_on_labeled_ratio", 0.0), 8),
                "water_area_on_labeled_ratio": round(stats.get("water_area_on_labeled_ratio", 0.0), 8),
                "dead_patch_count": stats["dead_patch_count"],
                "largest_dead_patch_area_m2": round(stats["largest_dead_patch_area_m2"], 4),
                "dead_patch_area_p50_m2": round(stats["dead_patch_area_p50_m2"], 4),
                "dead_patch_area_p75_m2": round(stats["dead_patch_area_p75_m2"], 4),
                "dead_patch_area_p90_m2": round(stats["dead_patch_area_p90_m2"], 4),
                "dead_aggregation_index": round(stats["dead_aggregation_index"], 8),
                "small_patch_count": int(stats.get("small_patch_count", 0)),
                "medium_patch_count": int(stats.get("medium_patch_count", 0)),
                "large_patch_count": int(stats.get("large_patch_count", 0)),
                "small_patch_area_m2": round(stats.get("small_patch_area_m2", 0.0), 4),
                "medium_patch_area_m2": round(stats.get("medium_patch_area_m2", 0.0), 4),
                "large_patch_area_m2": round(stats.get("large_patch_area_m2", 0.0), 4),
                "small_patch_area_fraction": round(stats.get("small_patch_area_fraction", 0.0), 8),
                "medium_patch_area_fraction": round(stats.get("medium_patch_area_fraction", 0.0), 8),
                "large_patch_area_fraction": round(stats.get("large_patch_area_fraction", 0.0), 8),
                "union_fallback_used": stats.get("union_fallback_used", 0),
            }
        )
        if stats.get("union_fallback_used", 0):
            message = (
                f"sample_id={sample_id} used geometry repair/union fallback; "
                "union_fallback_used=1"
            )
            if not allow_geometry_fallback:
                raise RuntimeError(
                    f"[summary] {message}. Refusing to continue because area/density metrics "
                    "would use fallback geometry. Fix the shapefile geometry or pass "
                    "--allow_geometry_fallback explicitly."
                )
            print(f"[summary][warn] {message}; allowed by --allow_geometry_fallback")

        for idx, area_m2 in enumerate(stats["patch_areas_m2"], start=1):
            cluster_rows.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "layer_name": row.get("layer_name", ""),
                    "dead_patch_id": idx,
                    "dead_patch_area_m2": round(area_m2, 4),
                    "dead_patch_area_ratio_of_total_dead": round(
                        float(area_m2 / stats["dead_area_m2"]) if stats["dead_area_m2"] > 0 else 0.0,
                        8,
                    ),
                }
            )

    if not summary_rows:
        raise RuntimeError("No valid shapefiles summarized")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    if cluster_rows:
        with open(cluster_csv, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(cluster_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cluster_rows)

    print(f"[summary] samples summarized: {len(summary_rows)}")
    print(f"[summary] sample table -> {output_csv}")
    if cluster_rows:
        print(f"[summary] dead patch table -> {cluster_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_csv",
        default=os.path.join(config.MANIFEST_DIR, "sample_manifest.csv"),
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join(config.SAMPLE_STATS_DIR, "sample_summary.csv"),
    )
    parser.add_argument(
        "--cluster_csv",
        default=os.path.join(config.SAMPLE_STATS_DIR, "cluster_summary.csv"),
    )
    parser.add_argument(
        "--image_dir",
        default=getattr(config, "SAMPLE_IMAGE_DIR", config.RAW_TIF_DIR),
    )
    parser.add_argument(
        "--allow_missing_images",
        action="store_true",
        help="Allow labeled-area fallback when sample PlanetScope clips are missing",
    )
    parser.add_argument(
        "--allow_geometry_fallback",
        action="store_true",
        help="Allow fallback geometry area calculation if union/repair fails",
    )
    parser.add_argument(
        "--cluster_distance_m",
        type=float,
        default=3.0,
        help="Unused in patch-based mode; kept for CLI compatibility",
    )
    args = parser.parse_args()
    summarize_manifest(
        args.manifest_csv,
        args.output_csv,
        args.cluster_csv,
        args.cluster_distance_m,
        image_dir=args.image_dir,
        require_images=not args.allow_missing_images,
        allow_geometry_fallback=args.allow_geometry_fallback,
    )


if __name__ == "__main__":
    main()
