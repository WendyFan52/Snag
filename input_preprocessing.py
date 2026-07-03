"""
Input preprocessing for PlanetScope surface-reflectance imagery.

AnalyticMS_SR assets store surface reflectance as scaled integer values
where reflectance = stored_value / 10000. Model inputs use this fixed
physical scaling before spectral-index calculation. Per-scene percentile
stretching is intentionally not used for model inputs or framework visual
products.
"""

import os
from typing import Mapping, Optional

import numpy as np

import config
from spectral_indices import append_indices


def sr_to_reflectance(raw: np.ndarray, source: Optional[str] = None) -> np.ndarray:
    """Convert PlanetScope AnalyticMS_SR stored values to reflectance."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 3:
        raise RuntimeError(
            f"Expected image array with shape (C,H,W), got {arr.shape}"
            + (f" for {source}" if source else "")
        )
    if arr.shape[0] != config.RAW_CHANNELS:
        raise RuntimeError(
            f"Unexpected raw channel count"
            + (f" for {source}" if source else "")
            + f": got {arr.shape[0]}, expected {config.RAW_CHANNELS} for BAND_MODE={config.BAND_MODE}. "
            "Refuse to pad/truncate bands."
        )
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(
            "Input image contains NaN/Inf values"
            + (f": {source}" if source else "")
        )
    if config.PLANETSCOPE_SR_SCALE <= 0:
        raise RuntimeError(f"Invalid PLANETSCOPE_SR_SCALE={config.PLANETSCOPE_SR_SCALE}")

    reflectance = arr / np.float32(config.PLANETSCOPE_SR_SCALE)
    if config.REFLECTANCE_CLIP_MAX <= config.REFLECTANCE_CLIP_MIN:
        raise RuntimeError(
            "Invalid reflectance clipping range: "
            f"{config.REFLECTANCE_CLIP_MIN}..{config.REFLECTANCE_CLIP_MAX}"
        )
    return np.clip(
        reflectance,
        config.REFLECTANCE_CLIP_MIN,
        config.REFLECTANCE_CLIP_MAX,
    ).astype(np.float32, copy=False)


def sr_to_feature_stack(raw: np.ndarray, source: Optional[str] = None) -> np.ndarray:
    """Return raw reflectance bands plus spectral indices."""
    reflectance = sr_to_reflectance(raw, source=source)
    features = append_indices(reflectance, config.RAW_CHANNELS)
    if features.shape[0] != config.IN_CHANNELS:
        raise RuntimeError(
            f"Feature stack channel mismatch"
            + (f" for {source}" if source else "")
            + f": got {features.shape[0]}, expected {config.IN_CHANNELS}."
        )
    if not np.all(np.isfinite(features)):
        raise RuntimeError(
            "Feature stack contains NaN/Inf values"
            + (f": {source}" if source else "")
        )
    return features.astype(np.float32, copy=False)


def preprocessing_tags() -> dict:
    return {
        config.PREPROCESSING_TAG_KEY: config.INPUT_PREPROCESSING_NAME,
        config.REFLECTANCE_SCALE_TAG_KEY: f"{config.PLANETSCOPE_SR_SCALE:g}",
        config.BAND_MODE_TAG_KEY: config.BAND_MODE,
    }


def assert_prepared_tile_tags(tags: Mapping[str, str], source: Optional[str] = None):
    actual = tags.get(config.PREPROCESSING_TAG_KEY)
    if actual != config.INPUT_PREPROCESSING_NAME:
        raise RuntimeError(
            "Prepared tile preprocessing mismatch"
            + (f" for {source}" if source else "")
            + f": got {actual!r}, expected {config.INPUT_PREPROCESSING_NAME!r}. "
            "Regenerate tiles with the current prepare_data_softlabel.py; refusing to use stale percentile-stretched tiles."
        )
    band_mode = tags.get(config.BAND_MODE_TAG_KEY)
    if band_mode and band_mode != config.BAND_MODE:
        raise RuntimeError(
            "Prepared tile band-mode mismatch"
            + (f" for {source}" if source else "")
            + f": got {band_mode!r}, expected {config.BAND_MODE!r}."
        )


def describe_preprocessing() -> str:
    return (
        f"{config.INPUT_PREPROCESSING_NAME}: PlanetScope AnalyticMS_SR stored values "
        f"are divided by {config.PLANETSCOPE_SR_SCALE:g}, clipped to "
        f"[{config.REFLECTANCE_CLIP_MIN:g}, {config.REFLECTANCE_CLIP_MAX:g}], "
        "then spectral indices are computed from reflectance bands."
    )


def is_probably_prepared_tile(path: str) -> bool:
    return os.path.splitext(os.path.basename(path))[0].count("_") >= 2
