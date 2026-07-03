"""
Build a unified sample manifest for research workflows.

Inputs:
- sample center CSV or SHP
- optional climate-zone SHP
- directory of truth SHP files

Outputs:
- research manifest CSV with one row per sample
"""

import argparse
import csv
import glob
import math
import os
import re
from typing import Dict, Iterable, List, Optional

import config
from koppen_climate import KoppenExtractor, transform_xy
from sample_remote_sensing import build_image_lookup, first_image_for_sample, images_for_sample, join_image_paths


DEFAULT_CENTER_SOURCE = config.SAMPLE_CENTER_CSV
DEFAULT_CLIMATE_SHP = getattr(config, "SAMPLE_CLIMATE_SHP", config.SAMPLE_THERMAL_SHP)
DEFAULT_TRUTH_DIR = config.SAMPLE_TRUTH_DIR
DEFAULT_CENTER_CRS = getattr(config, "SAMPLE_CENTER_CRS", "EPSG:4326")
DEFAULT_IMAGE_DIR = getattr(config, "SAMPLE_IMAGE_DIR", config.RAW_TIF_DIR)


def parse_sample_id(layer_name: str) -> str:
    match = re.match(r"^(\d+)", layer_name or "")
    return match.group(1) if match else ""


def infer_hemisphere(lat: float) -> str:
    if not math.isfinite(float(lat)):
        return "unknown"
    return "North" if lat >= 0 else "South"


def infer_latitude_band(lat: float) -> str:
    if not math.isfinite(float(lat)):
        return "unknown"
    abs_lat = abs(lat)
    if abs_lat < 23.5:
        return "Equatorial"
    if abs_lat < 50:
        return "MidLatitude"
    return "HighLatitude"


def load_center_rows(center_source: str) -> List[dict]:
    ext = os.path.splitext(center_source)[1].lower()
    if ext == ".csv":
        with open(center_source, "r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    if ext == ".shp":
        try:
            import geopandas as gpd
        except ImportError:
            raise RuntimeError("geopandas is required when center source is a shapefile")
        gdf = gpd.read_file(center_source)
        if gdf.crs is None:
            raise RuntimeError(f"Center shapefile CRS is missing: {center_source}")
        out = []
        for _, item in gdf.iterrows():
            row = {k: ("" if v is None else v) for k, v in item.drop(labels=["geometry"], errors="ignore").items()}
            geom = item.geometry
            if geom is not None and not geom.is_empty:
                center = geom.centroid
                row.setdefault("center_x", center.x)
                row.setdefault("center_y", center.y)
            row["_center_crs"] = str(gdf.crs)
            out.append(row)
        return out
    raise ValueError(f"Unsupported center source: {center_source}")


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_value(row: dict, names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name, "")
        if str(value).strip():
            return value
    return ""


def resolve_center(row: dict, default_crs: str) -> dict:
    x = _safe_float(_first_value(row, ["center_x", "x", "lon", "longitude"]))
    y = _safe_float(_first_value(row, ["center_y", "y", "lat", "latitude"]))
    source_crs = str(row.get("_center_crs") or row.get("center_crs") or default_crs or "EPSG:4326")
    if not math.isfinite(x) or not math.isfinite(y):
        raise RuntimeError(f"Invalid sample center coordinates: x={x}, y={y}, row={row}")
    lon = lat = float("nan")
    try:
        lon, lat = transform_xy(x, y, source_crs, "EPSG:4326")
    except Exception as exc:
        raise RuntimeError(f"Failed to transform center ({x}, {y}) from {source_crs}") from exc
    if not math.isfinite(lon) or not math.isfinite(lat):
        raise RuntimeError(
            f"Transformed center is invalid: lon={lon}, lat={lat}, source=({x}, {y}, {source_crs})"
        )
    return {
        "center_x": x,
        "center_y": y,
        "center_crs": source_crs,
        "longitude": lon,
        "latitude": lat,
    }


def _safe_field(gdf, candidates: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in gdf.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def load_climate_lookup(climate_shp: str) -> Dict[str, str]:
    if not climate_shp or not os.path.exists(climate_shp):
        return {}

    try:
        import geopandas as gpd
    except ImportError:
        print("[manifest] geopandas not installed; climate-zone join skipped")
        return {}

    gdf = gpd.read_file(climate_shp)
    source_field = _safe_field(gdf, ["source_shp", "source", "truth_shp_path", "layer_name"])
    zone_field = _safe_field(gdf, ["climate_zone", "zone", "climate", "thermal_zone", "therm_zone"])
    if source_field is None or zone_field is None:
        print("[manifest] climate zone fields not found; climate-zone join skipped")
        return {}

    lookup = {}
    for _, row in gdf.iterrows():
        key = str(row[source_field]).strip()
        zone = str(row[zone_field]).strip()
        if key:
            lookup[key] = zone
    return lookup


def build_truth_lookup(truth_dir: str) -> Dict[str, str]:
    lookup = {}
    for shp_path in glob.glob(os.path.join(truth_dir, "*.shp")):
        stem = os.path.splitext(os.path.basename(shp_path))[0]
        lookup[stem] = shp_path
        sample_id = parse_sample_id(stem)
        if sample_id and sample_id not in lookup:
            lookup[sample_id] = shp_path
    return lookup


def truth_sample_ids(truth_dir: str) -> set:
    ids = set()
    for shp_path in glob.glob(os.path.join(truth_dir, "*.shp")):
        sample_id = parse_sample_id(os.path.splitext(os.path.basename(shp_path))[0])
        if sample_id:
            ids.add(sample_id)
    return ids


def build_manifest(
    center_source: str,
    climate_shp: str,
    truth_dir: str,
    output_csv: str,
    center_crs: str = None,
    image_dir: str = None,
    require_koppen: bool = True,
    allow_extra_truth: bool = False,
) -> str:
    climate_lookup = {} if require_koppen else load_climate_lookup(climate_shp)
    truth_lookup = build_truth_lookup(truth_dir)
    image_lookup = build_image_lookup(image_dir or DEFAULT_IMAGE_DIR)
    print(f"[manifest] image lookup sample_ids: {len(image_lookup)} from {image_dir or DEFAULT_IMAGE_DIR}")
    if not image_lookup:
        print("[manifest][warn] no sample images matched; summary will fail unless --allow_missing_images is used")
    default_center_crs = center_crs or DEFAULT_CENTER_CRS
    try:
        koppen = KoppenExtractor.from_config()
    except Exception as exc:
        print(f"[manifest] Koppen raster not available; climate-zone raster join skipped: {exc}")
        koppen = None
    if koppen is None:
        if require_koppen:
            raise RuntimeError(
                "Koppen raster is required for true spatial climate_zone. "
                "Run download_koppen_climate.py or pass --allow_legacy_climate for debugging only."
            )
        print("[manifest] Koppen raster not found; climate_zone will use existing fields or legacy lookup")
    else:
        print(f"[manifest] Koppen kc raster -> {koppen.kc_path}")

    rows = []
    try:
        center_rows = load_center_rows(center_source)
        center_sample_ids = {
            parse_sample_id(str(row.get("layer_name", "")).strip())
            for row in center_rows
            if parse_sample_id(str(row.get("layer_name", "")).strip())
        }
        extra_truth = sorted(truth_sample_ids(truth_dir) - center_sample_ids)
        if extra_truth and not allow_extra_truth:
            raise RuntimeError(
                "Truth shapefiles exist without matching sample-center rows: "
                + ", ".join(extra_truth)
                + ". Add center rows or pass --allow_extra_truth explicitly."
            )
        if extra_truth:
            print(f"[manifest][warn] extra truth shapefiles not in center_source: {extra_truth}")
        for row in center_rows:
            layer_name = row.get("layer_name", "").strip()
            source_shp = row.get("source_shp", "").strip()
            sample_id = parse_sample_id(layer_name)
            center = resolve_center(row, default_center_crs)
            center_x = center["center_x"]
            center_y = center["center_y"]
            center_crs = center["center_crs"]
            longitude = center["longitude"]
            latitude = center["latitude"]

            # Prefer source_shp only when the file really exists.
            # Some center CSV files still store legacy paths.
            if source_shp and os.path.exists(source_shp):
                truth_path = source_shp
            else:
                source_stem = os.path.splitext(os.path.basename(source_shp))[0] if source_shp else ""
                truth_path = (
                    truth_lookup.get(layer_name)
                    or truth_lookup.get(sample_id, "")
                    or truth_lookup.get(source_stem, "")
                )
            truth_exists = int(bool(truth_path and os.path.exists(truth_path)))
            koppen_info = koppen.sample(center_x, center_y, center_crs) if koppen is not None else {}
            climate_zone = (
                koppen_info.get("climate_zone", "")
            )
            if not climate_zone and not require_koppen:
                climate_zone = (
                    str(row.get("climate_zone", "")).strip()
                    or str(row.get("zone", "")).strip()
                    or str(row.get("thermal_zone", "")).strip()
                    or str(row.get("climate", "")).strip()
                    or climate_lookup.get(layer_name, "")
                    or climate_lookup.get(source_shp)
                    or climate_lookup.get(truth_path, "")
                )
            if require_koppen and not climate_zone:
                raise RuntimeError(
                    f"Koppen raster returned no climate_zone for sample_id={sample_id}, "
                    f"center=({center_x}, {center_y}), crs={center_crs}"
                )

            image_paths = images_for_sample(sample_id, image_lookup)
            if len(image_paths) > 1:
                print(f"[manifest][info] sample_id={sample_id} matched {len(image_paths)} images; downstream rs_* will aggregate them")
            elif not image_paths:
                print(f"[manifest][warn] sample_id={sample_id} matched no image in image_dir={image_dir or DEFAULT_IMAGE_DIR}")

            rows.append({
                "sample_id": sample_id,
                "layer_name": layer_name,
                "truth_shp_path": truth_path,
                "truth_exists": truth_exists,
                "center_x": center_x,
                "center_y": center_y,
                "center_crs": center_crs,
                "longitude": longitude,
                "latitude": latitude,
                "hemisphere": infer_hemisphere(latitude),
                "latitude_band": infer_latitude_band(latitude),
                "climate_zone": climate_zone,
                "koppen_code": koppen_info.get("koppen_code", ""),
                "koppen_class": koppen_info.get("koppen_class", ""),
                "koppen_description": koppen_info.get("koppen_description", ""),
                "koppen_confidence": koppen_info.get("koppen_confidence", ""),
                "ps_image_path": first_image_for_sample(sample_id, image_lookup),
                "ps_image_paths": join_image_paths(image_paths),
                "ps_image_count": len(image_paths),
                "split_group": "",
                "background_type": "",
                "spatial_pattern_type": "",
                "tree_density_class": "",
                "dead_density_class": "",
                "living_density_class": "",
                "use_for_training": 1 if truth_exists else 0,
                "use_for_scale_analysis": 1 if truth_exists else 0,
                "notes": "",
            })
    finally:
        if koppen is not None:
            koppen.close()

    rows.sort(key=lambda x: (x["sample_id"], x["layer_name"]))
    if not rows:
        raise RuntimeError(f"No sample centers were loaded from center_source={center_source}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[manifest] samples: {len(rows)}")
    print(f"[manifest] truth linked: {sum(r['truth_exists'] for r in rows)}")
    print(f"[manifest] images linked: {sum(1 for r in rows if r.get('ps_image_path'))}")
    print(f"[manifest] total matched image files: {sum(int(r.get('ps_image_count', 0) or 0) for r in rows)}")
    print(f"[manifest] saved -> {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--center_source", default=DEFAULT_CENTER_SOURCE)
    parser.add_argument("--center_csv", dest="center_source")
    parser.add_argument("--climate_shp", dest="climate_shp", default=DEFAULT_CLIMATE_SHP)
    parser.add_argument("--thermal_shp", dest="climate_shp")
    parser.add_argument("--center_crs", default=DEFAULT_CENTER_CRS)
    parser.add_argument("--koppen_raster", default=None)
    parser.add_argument("--koppen_conf_raster", default=None)
    parser.add_argument(
        "--allow_legacy_climate",
        action="store_true",
        help="Allow fallback to old climate fields if Koppen raster is unavailable",
    )
    parser.add_argument(
        "--allow_extra_truth",
        action="store_true",
        help="Allow truth SHP files that have no matching sample-center row",
    )
    parser.add_argument("--truth_dir", default=DEFAULT_TRUTH_DIR)
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--output_csv",
        default=os.path.join(config.MANIFEST_DIR, "sample_manifest.csv"),
    )
    args = parser.parse_args()
    if args.koppen_raster:
        config.KOPPEN_KC_RASTER = args.koppen_raster
    if args.koppen_conf_raster:
        config.KOPPEN_CONF_RASTER = args.koppen_conf_raster
    build_manifest(
        args.center_source,
        args.climate_shp,
        args.truth_dir,
        args.output_csv,
        center_crs=args.center_crs,
        image_dir=args.image_dir,
        require_koppen=not args.allow_legacy_climate,
        allow_extra_truth=args.allow_extra_truth,
    )


if __name__ == "__main__":
    main()
