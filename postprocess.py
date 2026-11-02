"""
Inference postprocessing - postprocess.py
-----------------------------------------------------------------
Postprocessing tasks:
  NoData masking to exclude predictions in black border regions.
  Connected-component analysis and GeoJSON export of pixel predictions as objects.

Usage (called automatically by inference.py or used independently):
  from postprocess import postprocess
  results = postprocess(prob_map, image_norm, image_rgb,
                        ref_tif_path, model_name)
-----------------------------------------------------------------
"""

import os, json
import numpy as np
import cv2
from PIL import Image
import rasterio

import config


# =================================================================
# NoData mask
# =================================================================
def compute_nodata_mask(image_norm: np.ndarray,
                        threshold: float = config.NODATA_THRESHOLD
                        ) -> np.ndarray:
    """
    Detect NoData (black fill) regions in the image.
    Pixels whose summed band values are below threshold are treated as NoData.

    Input: image_norm (H, W, C) float32 [0, 1]
    Return: (H, W) bool, True = valid pixel.
    """
    raw_view = image_norm[..., :config.RAW_CHANNELS]
    band_sum = raw_view.sum(axis=-1)      # (H, W)
    valid    = (band_sum > threshold).astype(np.uint8) * 255

    # Morphological smoothing to remove small noise regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    valid  = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel)
    valid  = cv2.morphologyEx(valid, cv2.MORPH_OPEN,  kernel)
    return valid > 127


def apply_nodata_mask(prob_map: np.ndarray,
                      valid_mask: np.ndarray) -> tuple:
    """
    Set probabilities and predictions to zero in NoData regions.

    Return: (masked_prob, pred_map).
    """
    masked = prob_map.copy()
    masked[:, ~valid_mask] = 0.0
    pred = masked[1:].argmax(axis=0).astype(np.uint8) + 1
    pred[~valid_mask] = 0
    return masked, pred


# =================================================================
# Object-level postprocessing
# =================================================================
def extract_objects(pred_map:    np.ndarray,
                    prob_map:    np.ndarray,
                    target_cls:  int   = 1,
                    min_area_px: int   = config.MIN_OBJ_AREA_PX,
                    max_area_px: int   = 100000) -> list:
    """
    Extract objects of the target class using connected-component analysis.
    Each object includes its area, centroid, confidence, and contour.
    """
    binary = ((pred_map == target_cls) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cls_prob = prob_map[target_cls]   # (H, W)
    objects  = []

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px or area > max_area_px:
            continue

        obj_mask  = (labels == i)
        conf_vals = cls_prob[obj_mask]
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]),
                       int(stats[i, cv2.CC_STAT_TOP]),
                       int(stats[i, cv2.CC_STAT_WIDTH]),
                       int(stats[i, cv2.CC_STAT_HEIGHT]))

        obj_u8 = obj_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(obj_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        objects.append({
            "label":           target_cls,
            "class_name":      config.CLASS_NAMES.get(target_cls, str(target_cls)),
            "area_px":         area,
            "centroid":        (float(centroids[i][0]), float(centroids[i][1])),
            "bbox":            (x, y, w, h),
            "mean_confidence": float(conf_vals.mean()),
            "max_confidence":  float(conf_vals.max()),
            "contour":         contours[0] if contours else None,
        })

    objects.sort(key=lambda o: o["area_px"], reverse=True)
    return objects


def draw_objects(image_rgb: np.ndarray,
                 objects:   list,
                 conf_thr:  float = config.CONF_THRESHOLD) -> np.ndarray:
    """
    Draw detected objects on the image:
      - Orange-red outlines for high confidence; blue-gray outlines for low confidence
      - Display confidence values
    """
    canvas = image_rgb.copy()
    for obj in objects:
        if obj["mean_confidence"] < conf_thr:
            continue
        conf      = obj["mean_confidence"]
        lo  = getattr(config, "CONFIDENCE_LOW_THRESHOLD",  0.40)
        hi  = getattr(config, "CONFIDENCE_HIGH_THRESHOLD", 0.70)
        if conf >= hi:
            color_bgr = (0, 60, 220)       # red (high confidence)
        elif conf >= lo:
            color_bgr = (0, 180, 230)      # orange (moderate)
        else:
            color_bgr = (120, 120, 80)     # grey-blue (low confidence)

        if obj["contour"] is not None:
            cv2.drawContours(canvas, [obj["contour"]], -1, color_bgr, 2)

        cx, cy = int(obj["centroid"][0]), int(obj["centroid"][1])
        cv2.putText(canvas, f"{conf:.2f}", (cx - 12, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)  # confidence is ASCII-safe
    return canvas


def save_geojson(objects:      list,
                 ref_tif_path: str,
                 out_path:     str,
                 conf_thr:     float = config.CONF_THRESHOLD):
    """Save detected objects as GeoJSON for direct loading in QGIS."""
    with rasterio.open(ref_tif_path) as src:
        transform = src.transform
        crs_str   = src.crs.to_wkt() if src.crs else None

    features = []
    for obj in objects:
        if obj["mean_confidence"] < conf_thr or obj["contour"] is None:
            continue
        cnt = obj["contour"].squeeze()
        if cnt.ndim == 1:
            cnt = cnt[np.newaxis, :]
        if len(cnt) < 3:
            continue

        geo_pts = [rasterio.transform.xy(transform, int(r), int(c))
                   for c, r in cnt]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [geo_pts + [geo_pts[0]]]
            },
            "properties": {
                "class":           obj["label"],
                "class_name":      obj["class_name"],
                "area_px":         obj["area_px"],
                "mean_confidence": round(obj["mean_confidence"], 4),
                "max_confidence":  round(obj["max_confidence"],  4),
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "crs":  {"type": "name",
                 "properties": {"name": crs_str}} if crs_str else None,
        "features": features,
    }
    with open(out_path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"  [OK] GeoJSON -> {out_path}  ({len(features)} objects)")


# =================================================================
# Postprocessing entry point
# =================================================================
def postprocess(prob_map:     np.ndarray,
                image_norm:   np.ndarray,
                image_rgb:    np.ndarray,
                ref_tif_path: str,
                model_name:   str,
                target_cls:   int   = 1,
                conf_thr:     float = config.CONF_THRESHOLD,
                min_area_px:  int   = config.MIN_OBJ_AREA_PX) -> dict:
    """
    Complete postprocessing pipeline:
      1. NoData mask to exclude black borders
      2. Connected-component object extraction
      3. Object detection visualization
      4. GeoJSON export

    Return a results dictionary for use by callers.
    """
    stem   = os.path.splitext(os.path.basename(ref_tif_path))[0]
    prefix = os.path.join(config.OUTPUT_DIR, f"{stem}_{model_name}")

    # 1. NoData mask
    valid_mask        = compute_nodata_mask(image_norm)
    masked_prob, pred = apply_nodata_mask(prob_map, valid_mask)

    nodata_pct = (1 - valid_mask.mean()) * 100
    print(f"  NoData area: {nodata_pct:.1f}%  "
          f"valid pixels: {valid_mask.mean()*100:.1f}%")

    # 2. Object extraction
    objects = extract_objects(pred,
                              masked_prob,
                              target_cls=target_cls,
                              min_area_px=min_area_px)
    hi_conf = sum(1 for o in objects if o["mean_confidence"] >= conf_thr)
    print(f"  Class{target_cls}({config.CLASS_NAMES.get(target_cls)}) "
          f"Objects: {len(objects)} total; confidence>={conf_thr:.2f}: {hi_conf}")

    # 3. Object detection visualization
    obj_img = draw_objects(image_rgb, objects, conf_thr)
    obj_out = prefix + f"-class{target_cls}-objects.png"
    Image.fromarray(obj_img).save(obj_out)
    print(f"  [OK] Object visualization -> {obj_out}")

    # 4. GeoJSON
    geo_out = prefix + f"-class{target_cls}-objects.geojson"
    save_geojson(objects, ref_tif_path, geo_out, conf_thr)

    return {
        "masked_prob": masked_prob,
        "pred_map":    pred,
        "valid_mask":  valid_mask,
        "objects":     objects,
    }
