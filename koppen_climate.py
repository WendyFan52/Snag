"""
Koppen-Geiger climate extraction utilities for sample manifests.

The KGClim historical raster stores numeric climate codes. This module maps
those codes to Koppen classes and samples them at true sample-center positions.
"""

import math
import os
from typing import Dict, Optional, Tuple

import config


KOPPEN_CLASSES: Dict[int, Tuple[str, str, str]] = {
    110: ("Af", "Tropical", "Tropical rainforest"),
    120: ("Aw", "Tropical", "Tropical savannah"),
    130: ("Am", "Tropical", "Tropical monsoon"),
    211: ("BWh", "Arid", "Arid desert hot"),
    212: ("BWk", "Arid", "Arid desert cold"),
    221: ("BSh", "Arid", "Arid steppe hot"),
    222: ("BSk", "Arid", "Arid steppe cold"),
    311: ("Csa", "Temperate", "Temperate dry summer hot summer"),
    312: ("Csb", "Temperate", "Temperate dry summer warm summer"),
    313: ("Csc", "Temperate", "Temperate dry summer cold summer"),
    321: ("Cwa", "Temperate", "Temperate dry winter hot summer"),
    322: ("Cwb", "Temperate", "Temperate dry winter warm summer"),
    323: ("Cwc", "Temperate", "Temperate dry winter cold summer"),
    331: ("Cfa", "Temperate", "Temperate without dry season hot summer"),
    332: ("Cfb", "Temperate", "Temperate without dry season warm summer"),
    333: ("Cfc", "Temperate", "Temperate without dry season cold summer"),
    411: ("Dsa", "Boreal", "Boreal dry summer hot summer"),
    412: ("Dsb", "Boreal", "Boreal dry summer warm summer"),
    413: ("Dsc", "Boreal", "Boreal dry summer cold summer"),
    414: ("Dsd", "Boreal", "Boreal dry summer very cold winter"),
    421: ("Dwa", "Boreal", "Boreal dry winter hot summer"),
    422: ("Dwb", "Boreal", "Boreal dry winter warm summer"),
    423: ("Dwc", "Boreal", "Boreal dry winter cold summer"),
    424: ("Dwd", "Boreal", "Boreal dry winter very cold winter"),
    431: ("Dfa", "Boreal", "Boreal without dry season hot summer"),
    432: ("Dfb", "Boreal", "Boreal without dry season warm summer"),
    433: ("Dfc", "Boreal", "Boreal without dry season cold summer"),
    434: ("Dfd", "Boreal", "Boreal without dry season very cold winter"),
    510: ("ET", "Polar", "Polar tundra"),
    520: ("EF", "Polar", "Polar frost"),
}


def code_to_info(code_value) -> Dict[str, str]:
    try:
        code = int(round(float(code_value)))
    except (TypeError, ValueError):
        code = 0
    klass, zone, desc = KOPPEN_CLASSES.get(code, ("", "", ""))
    return {
        "koppen_code": str(code) if code else "",
        "koppen_class": klass,
        "climate_zone": zone,
        "koppen_description": desc,
    }


def finite_xy(x: float, y: float) -> bool:
    return math.isfinite(float(x)) and math.isfinite(float(y))


def transform_xy(x: float, y: float, src_crs, dst_crs) -> Tuple[float, float]:
    if not src_crs or not dst_crs or str(src_crs).lower() == str(dst_crs).lower():
        return float(x), float(y)
    try:
        from rasterio.warp import transform
    except ImportError as exc:
        raise RuntimeError("rasterio is required for CRS transformation") from exc
    xs, ys = transform(src_crs, dst_crs, [float(x)], [float(y)])
    return float(xs[0]), float(ys[0])


def find_koppen_rasters(data_dir: str) -> Tuple[str, str]:
    kc_raster = ""
    conf_raster = ""
    if not data_dir or not os.path.isdir(data_dir):
        return kc_raster, conf_raster

    candidates = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            lower = name.lower()
            if not lower.endswith((".tif", ".tiff")):
                continue
            path = os.path.join(root, name)
            candidates.append(path)
            if "kc_conf" in lower and not conf_raster:
                conf_raster = path

    def is_kc(path: str) -> bool:
        lower = os.path.basename(path).lower()
        if "kc_conf" in lower:
            return False
        stem = os.path.splitext(lower)[0]
        return stem == "kc" or stem.endswith("_kc") or "_kc_" in stem

    kc_matches = [p for p in candidates if is_kc(p)]
    if kc_matches:
        kc_raster = sorted(kc_matches, key=lambda p: (len(os.path.basename(p)), p))[0]
    return kc_raster, conf_raster


class KoppenExtractor:
    def __init__(self, kc_raster: str, conf_raster: str = ""):
        if not kc_raster or not os.path.exists(kc_raster):
            raise FileNotFoundError(f"Koppen kc raster not found: {kc_raster}")
        import rasterio

        self.rasterio = rasterio
        self.kc_path = kc_raster
        self.conf_path = conf_raster if conf_raster and os.path.exists(conf_raster) else ""
        self.kc = rasterio.open(self.kc_path)
        self.conf = rasterio.open(self.conf_path) if self.conf_path else None

    @classmethod
    def from_config(cls) -> Optional["KoppenExtractor"]:
        kc_raster = getattr(config, "KOPPEN_KC_RASTER", "") or ""
        conf_raster = getattr(config, "KOPPEN_CONF_RASTER", "") or ""
        if not kc_raster or not os.path.exists(kc_raster):
            search_roots = [
                getattr(config, "KOPPEN_EXTRACT_DIR", ""),
                getattr(config, "KOPPEN_DATA_DIR", ""),
            ]
            for root in search_roots:
                found_kc, found_conf = find_koppen_rasters(root)
                if found_kc:
                    kc_raster = found_kc
                    conf_raster = conf_raster or found_conf
                    break
        if not kc_raster:
            return None
        return cls(kc_raster, conf_raster)

    def close(self):
        self.kc.close()
        if self.conf is not None:
            self.conf.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def sample(self, x: float, y: float, source_crs: str = "EPSG:4326") -> Dict[str, str]:
        if not finite_xy(x, y):
            return {}
        rx, ry = transform_xy(float(x), float(y), source_crs, self.kc.crs)
        value = next(self.kc.sample([(rx, ry)]))[0]
        if self.kc.nodata is not None and value == self.kc.nodata:
            return {}
        info = code_to_info(value)
        if self.conf is not None:
            cx, cy = transform_xy(float(x), float(y), source_crs, self.conf.crs)
            conf_value = next(self.conf.sample([(cx, cy)]))[0]
            info["koppen_confidence"] = "" if self.conf.nodata is not None and conf_value == self.conf.nodata else str(float(conf_value))
        else:
            info["koppen_confidence"] = ""
        return info
