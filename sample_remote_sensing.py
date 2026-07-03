"""
Sample-level remote-sensing summaries used for ecological background strata.

The functions here operate on full sample PlanetScope clips, not on training
tiles. They deliberately use simple, explainable NDVI/NDWI rules because these
fields are analysis strata, not supervised labels.
"""

import glob
import math
import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np

import config


RS_FIELD_DEFAULTS = {
    "rs_available": 0,
    "rs_image_path": "",
    "rs_image_paths": "",
    "rs_image_count": 0,
    "rs_valid_image_count": 0,
    "rs_aggregation_method": "missing_remote_sensing",
    "valid_pixel_count": 0,
    "valid_area_m2": 0.0,
    "valid_area_source": "missing_remote_sensing",
    "rs_ndvi_mean": 0.0,
    "rs_ndvi_p25": 0.0,
    "rs_ndvi_p50": 0.0,
    "rs_ndvi_p75": 0.0,
    "rs_ndwi_mean": 0.0,
    "rs_ndwi_p25": 0.0,
    "rs_ndwi_p50": 0.0,
    "rs_ndwi_p75": 0.0,
    "rs_veg_fraction": 0.0,
    "rs_water_fraction": 0.0,
    "rs_bare_fraction": 0.0,
}


def parse_sample_id(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    match = re.match(r"^(\d+)", stem)
    return match.group(1) if match else ""


def _iter_image_dirs(image_dir: str):
    if not image_dir:
        return []
    parts = []
    for chunk in str(image_dir).split(os.pathsep):
        chunk = chunk.strip().strip('"')
        if chunk:
            parts.append(chunk)
    return parts or [image_dir]


def build_image_lookup(image_dir: str) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    for one_dir in _iter_image_dirs(image_dir):
        if not os.path.isdir(one_dir):
            print(f"[rs][warn] image directory not found and ignored: {one_dir}")
            continue
        for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            for path in glob.glob(os.path.join(one_dir, ext)):
                sample_id = parse_sample_id(path)
                if not sample_id:
                    continue
                lookup.setdefault(sample_id, []).append(path)
    for sample_id in list(lookup):
        lookup[sample_id] = sorted(set(lookup[sample_id]))
    return lookup


def first_image_for_sample(sample_id: str, image_lookup: Dict[str, List[str]]) -> str:
    paths = image_lookup.get(str(sample_id).strip(), [])
    return paths[0] if paths else ""


def images_for_sample(sample_id: str, image_lookup: Dict[str, List[str]]) -> List[str]:
    return list(image_lookup.get(str(sample_id).strip(), []))


def join_image_paths(paths: Iterable[str]) -> str:
    return "|".join(str(path) for path in paths if str(path))


def _safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.where(np.abs(b) > eps, a / (b + eps), 0.0).astype(np.float32)


def _read_sampled_raw(path: str, max_pixels: int = 1_000_000):
    import rasterio

    with rasterio.open(path) as src:
        height, width = src.height, src.width
        scale = max(1, int(math.ceil(math.sqrt((height * width) / max_pixels))))
        out_shape = (src.count, max(1, height // scale), max(1, width // scale))
        data = src.read(out_shape=out_shape).astype(np.float32)
        crs = src.crs
        res_x = abs(src.res[0]) * scale if src.res else config.GSD
        res_y = abs(src.res[1]) * scale if src.res else config.GSD

    if data.shape[0] < config.RAW_CHANNELS:
        raise ValueError(f"Image has {data.shape[0]} bands, expected at least {config.RAW_CHANNELS}: {path}")
    return data[: config.RAW_CHANNELS], crs, res_x, res_y


def _pixel_area_m2(crs, res_x: float, res_y: float) -> float:
    if crs is not None and not getattr(crs, "is_geographic", False) and res_x > 0 and res_y > 0:
        return float(res_x * res_y)
    return float(config.GSD * config.GSD)


def _band_indices(raw: np.ndarray):
    if config.RAW_CHANNELS == 4:
        blue, green, red, nir = raw[0], raw[1], raw[2], raw[3]
        return blue, green, red, nir
    if config.RAW_CHANNELS == 8:
        # PlanetScope SuperDove order:
        # Coastal Blue, Blue, Green I, Green, Yellow, Red, Red Edge, NIR
        blue, green, red, nir = raw[1], raw[3], raw[5], raw[7]
        return blue, green, red, nir
    raise ValueError(f"Unsupported raw channel count: {config.RAW_CHANNELS}")


def _valid_mask(raw: np.ndarray) -> np.ndarray:
    finite = np.all(np.isfinite(raw), axis=0)
    positive = np.any(raw > 0, axis=0)
    return finite & positive


def summarize_image(image_path: str) -> dict:
    stats = dict(RS_FIELD_DEFAULTS)
    if not image_path or not os.path.exists(image_path):
        if image_path:
            print(f"[rs][warn] image path not found; remote-sensing summary unavailable: {image_path}")
        return stats

    try:
        raw, crs, res_x, res_y = _read_sampled_raw(image_path)
    except Exception as exc:
        stats["valid_area_source"] = f"remote_sensing_error:{type(exc).__name__}"
        stats["rs_image_path"] = image_path
        print(f"[rs][warn] failed to read image; using unavailable rs_* fields for this image: {image_path} | {exc}")
        return stats

    blue, green, red, nir = _band_indices(raw)
    valid = _valid_mask(raw)
    valid_count = int(valid.sum())
    if valid_count <= 0:
        stats["rs_image_path"] = image_path
        stats["valid_area_source"] = "remote_sensing_no_valid_pixels"
        return stats

    ndvi = np.clip(_safe_ratio(nir - red, nir + red), -1.0, 1.0)
    ndwi = np.clip(_safe_ratio(green - nir, green + nir), -1.0, 1.0)
    ndvi_v = ndvi[valid]
    ndwi_v = ndwi[valid]

    water = ndwi_v >= config.RS_NDWI_WATER_THRESHOLD
    vegetation = (ndvi_v >= config.RS_NDVI_VEG_THRESHOLD) & (~water)
    bare = (ndvi_v < config.RS_NDVI_BARE_THRESHOLD) & (~water)
    pixel_area = _pixel_area_m2(crs, res_x, res_y)

    stats.update(
        {
            "rs_available": 1,
            "rs_image_path": image_path,
            "rs_image_paths": image_path,
            "rs_image_count": 1,
            "rs_valid_image_count": 1,
            "rs_aggregation_method": "single_image",
            "valid_pixel_count": valid_count,
            "valid_area_m2": float(valid_count * pixel_area),
            "valid_area_source": "remote_sensing_valid_pixels",
            "rs_ndvi_mean": float(np.mean(ndvi_v)),
            "rs_ndvi_p25": float(np.percentile(ndvi_v, 25)),
            "rs_ndvi_p50": float(np.percentile(ndvi_v, 50)),
            "rs_ndvi_p75": float(np.percentile(ndvi_v, 75)),
            "rs_ndwi_mean": float(np.mean(ndwi_v)),
            "rs_ndwi_p25": float(np.percentile(ndwi_v, 25)),
            "rs_ndwi_p50": float(np.percentile(ndwi_v, 50)),
            "rs_ndwi_p75": float(np.percentile(ndwi_v, 75)),
            "rs_veg_fraction": float(np.mean(vegetation)),
            "rs_water_fraction": float(np.mean(water)),
            "rs_bare_fraction": float(np.mean(bare)),
            "_ndvi_values": ndvi_v.astype(np.float32, copy=False),
            "_ndwi_values": ndwi_v.astype(np.float32, copy=False),
        }
    )
    return stats


def _weighted_mean(items: List[dict], key: str, weights: List[float]) -> float:
    total = float(sum(weights))
    if total <= 0:
        return 0.0
    return float(sum(float(item.get(key, 0.0)) * w for item, w in zip(items, weights)) / total)


def summarize_images(image_paths: List[str], sample_id: str = "") -> dict:
    stats = dict(RS_FIELD_DEFAULTS)
    paths = sorted(set(path for path in image_paths if path))
    if not paths:
        print(f"[rs][warn] sample_id={sample_id} has no matched image; rs_* fields unavailable")
        return stats

    per_image = [summarize_image(path) for path in paths]
    valid_items = [item for item in per_image if int(item.get("rs_available", 0)) > 0]
    if not valid_items:
        stats["rs_image_path"] = paths[0]
        stats["rs_image_paths"] = join_image_paths(paths)
        stats["rs_image_count"] = len(paths)
        stats["valid_area_source"] = "remote_sensing_no_valid_images"
        print(f"[rs][warn] sample_id={sample_id} matched {len(paths)} image(s), but none had valid pixels")
        return stats

    weights = [float(item.get("valid_pixel_count", 0) or 0) for item in valid_items]
    valid_areas = [float(item.get("valid_area_m2", 0.0) or 0.0) for item in valid_items]
    valid_areas = [area for area in valid_areas if area > 0]

    if len(valid_items) > 1:
        print(
            f"[rs][info] sample_id={sample_id} uses multi-image rs_* aggregation: "
            f"matched={len(paths)}, valid={len(valid_items)}, "
            f"method=pooled_valid_pixels; valid_area=median_image_area"
        )
        aggregation_method = "multi_image_pooled_valid_pixels_median_area"
        valid_area_source = "remote_sensing_valid_pixels_median_area_multi_image"
    else:
        aggregation_method = "single_image"
        valid_area_source = "remote_sensing_valid_pixels"

    stats.update(
        {
            "rs_available": 1,
            "rs_image_path": valid_items[0].get("rs_image_path", paths[0]),
            "rs_image_paths": join_image_paths(paths),
            "rs_image_count": len(paths),
            "rs_valid_image_count": len(valid_items),
            "rs_aggregation_method": aggregation_method,
            "valid_pixel_count": int(sum(weights)),
            "valid_area_m2": float(np.median(valid_areas)) if valid_areas else 0.0,
            "valid_area_source": valid_area_source,
        }
    )
    ndvi_values = [item.get("_ndvi_values") for item in valid_items if item.get("_ndvi_values") is not None]
    ndwi_values = [item.get("_ndwi_values") for item in valid_items if item.get("_ndwi_values") is not None]
    if ndvi_values:
        ndvi_all = np.concatenate(ndvi_values)
        stats["rs_ndvi_mean"] = float(np.mean(ndvi_all))
        stats["rs_ndvi_p25"] = float(np.percentile(ndvi_all, 25))
        stats["rs_ndvi_p50"] = float(np.percentile(ndvi_all, 50))
        stats["rs_ndvi_p75"] = float(np.percentile(ndvi_all, 75))
    if ndwi_values:
        ndwi_all = np.concatenate(ndwi_values)
        stats["rs_ndwi_mean"] = float(np.mean(ndwi_all))
        stats["rs_ndwi_p25"] = float(np.percentile(ndwi_all, 25))
        stats["rs_ndwi_p50"] = float(np.percentile(ndwi_all, 50))
        stats["rs_ndwi_p75"] = float(np.percentile(ndwi_all, 75))

    for key in (
        "rs_veg_fraction",
        "rs_water_fraction",
        "rs_bare_fraction",
    ):
        stats[key] = _weighted_mean(valid_items, key, weights)
    return stats


def summarize_sample_image(sample_id: str, image_path: str = "", image_lookup: Optional[Dict[str, List[str]]] = None) -> dict:
    lookup_paths = images_for_sample(sample_id, image_lookup or {})
    paths = lookup_paths or ([image_path] if image_path else [])
    return summarize_images(paths, sample_id=sample_id)


def merge_rs_defaults(row: dict, fields: Optional[Iterable[str]] = None) -> dict:
    out = dict(row)
    for key in fields or RS_FIELD_DEFAULTS:
        out.setdefault(key, RS_FIELD_DEFAULTS[key])
    return out
