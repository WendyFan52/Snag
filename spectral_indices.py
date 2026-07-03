"""
Spectral-index calculation for PlanetScope feature stacks.

Inputs to these functions must already be surface reflectance in [0, 1].
Do not pass per-scene percentile-stretched bands here because vegetation and
water indices require physically consistent band ratios.

4-band order:
    Blue, Green, Red, NIR

8-band PlanetScope SuperDove order:
    Coastal Blue, Blue, Green I, Green, Yellow, Red, Red Edge, NIR
"""

import numpy as np


INDEX_NAMES_4BAND = ["NDVI", "EVI", "NDWI", "RI", "NGRDI"]
INDEX_NAMES_8BAND = ["NDVI", "EVI", "NDWI", "RI", "NGRDI", "NDRE", "ChlRE"]


def _safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.where(np.abs(b) > eps, a / (b + eps), 0.0).astype(np.float32)


def compute_indices_4band(img: np.ndarray) -> np.ndarray:
    """Compute indices from 4-band reflectance: Blue, Green, Red, NIR."""
    if img.shape[0] < 4:
        raise ValueError(f"4-band index calculation needs at least 4 channels, got {img.shape[0]}")
    blue, green, red, nir = img[0], img[1], img[2], img[3]

    ndvi = np.clip(_safe_ratio(nir - red, nir + red), -1, 1)
    evi = np.clip(2.5 * _safe_ratio(nir - red, nir + 6 * red - 7.5 * blue + 1.0), -1, 1)
    ndwi = np.clip(_safe_ratio(green - nir, green + nir), -1, 1)
    ri = np.clip(_safe_ratio(red, green + 1e-6), 0, 5)
    ngrdi = np.clip(_safe_ratio(green - red, green + red), -1, 1)

    return np.stack([ndvi, evi, ndwi, ri, ngrdi], axis=0)


def compute_indices_8band(img: np.ndarray) -> np.ndarray:
    """
    Compute indices from 8-band PlanetScope SuperDove reflectance.

    Band order:
        0 Coastal Blue, 1 Blue, 2 Green I, 3 Green,
        4 Yellow, 5 Red, 6 Red Edge, 7 NIR
    """
    if img.shape[0] < 8:
        raise ValueError(f"8-band index calculation needs at least 8 channels, got {img.shape[0]}")
    blue = img[1]
    green = img[3]
    red = img[5]
    red_edge = img[6]
    nir = img[7]

    ndvi = np.clip(_safe_ratio(nir - red, nir + red), -1, 1)
    evi = np.clip(2.5 * _safe_ratio(nir - red, nir + 6 * red - 7.5 * blue + 1.0), -1, 1)
    ndwi = np.clip(_safe_ratio(green - nir, green + nir), -1, 1)
    ri = np.clip(_safe_ratio(red, green + 1e-6), 0, 5)
    ngrdi = np.clip(_safe_ratio(green - red, green + red), -1, 1)
    ndre = np.clip(_safe_ratio(nir - red_edge, nir + red_edge), -1, 1)
    chlre = np.clip(_safe_ratio(nir, red_edge + 1e-8) - 1, 0, 5)

    return np.stack([ndvi, evi, ndwi, ri, ngrdi, ndre, chlre], axis=0)


def append_indices(img: np.ndarray, raw_channels: int) -> np.ndarray:
    """
    Append normalized spectral indices to reflectance bands.

    The raw reflectance bands remain unchanged. Ratio indices with range [-1, 1]
    are mapped to [0, 1]. RI and ChlRE are clipped to [0, 5] and divided by 5.
    """
    if raw_channels == 4:
        indices = compute_indices_4band(img)
        bounded_11 = {0, 1, 2, 4}
        bounded_05 = {3}
    elif raw_channels == 8:
        indices = compute_indices_8band(img)
        bounded_11 = {0, 1, 2, 4, 5}
        bounded_05 = {3, 6}
    else:
        raise ValueError(f"Unsupported raw channel count: {raw_channels}; expected 4 or 8")

    norm = np.zeros_like(indices, dtype=np.float32)
    for i in range(indices.shape[0]):
        if i in bounded_11:
            norm[i] = (indices[i] + 1.0) / 2.0
        elif i in bounded_05:
            norm[i] = indices[i] / 5.0
        else:
            raise RuntimeError(f"Unhandled index channel {i}")

    return np.concatenate([img.astype(np.float32, copy=False), norm], axis=0)
