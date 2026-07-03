"""
推理后处理 — postprocess.py
-----------------------------------------------------------------
解决两个核心问题：
  ④ 黑边(NoData)误推理 → NoData 掩膜
  ⑤ 像素预测→对象级输出 → 连通域分析 + GeoJSON

用法（由 inference.py 自动调用，也可单独使用）：
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
# ④ NoData 掩膜
# =================================================================
def compute_nodata_mask(image_norm: np.ndarray,
                        threshold: float = config.NODATA_THRESHOLD
                        ) -> np.ndarray:
    """
    检测影像中的 NoData（黑色填充）区域。
    所有波段之和 < threshold 的像素视为 NoData。

    输入: image_norm (H, W, C) float32 [0,1]
    返回: (H, W) bool，True = 有效像素
    """
    raw_view = image_norm[..., :config.RAW_CHANNELS]
    band_sum = raw_view.sum(axis=-1)      # (H, W)
    valid    = (band_sum > threshold).astype(np.uint8) * 255

    # 形态学平滑：消除小噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    valid  = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel)
    valid  = cv2.morphologyEx(valid, cv2.MORPH_OPEN,  kernel)
    return valid > 127


def apply_nodata_mask(prob_map: np.ndarray,
                      valid_mask: np.ndarray) -> tuple:
    """
    将 NoData 区域的概率和预测结果清零。

    返回: (masked_prob, pred_map)
    """
    masked = prob_map.copy()
    masked[:, ~valid_mask] = 0.0
    pred = masked[1:].argmax(axis=0).astype(np.uint8) + 1
    pred[~valid_mask] = 0
    return masked, pred


# =================================================================
# ⑤ 对象级后处理
# =================================================================
def extract_objects(pred_map:    np.ndarray,
                    prob_map:    np.ndarray,
                    target_cls:  int   = 1,
                    min_area_px: int   = config.MIN_OBJ_AREA_PX,
                    max_area_px: int   = 100000) -> list:
    """
    连通域分析，提取目标类别的对象列表。
    每个对象包含：面积、质心、置信度、轮廓。
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
    在影像上绘制检测对象：
      · 高置信度→橙红轮廓，低置信度→蓝灰
      · 显示置信度数值
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
    """将检测对象保存为 GeoJSON（可在 QGIS 中直接加载）"""
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
    print(f"  ✔ GeoJSON  → {out_path}  ({len(features)} 个对象)")


# =================================================================
# 统一入口
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
    完整后处理：
      1. NoData 掩膜（过滤黑边）
      2. 连通域对象提取
      3. 对象检测图
      4. GeoJSON 输出

    返回结果字典供外部使用。
    """
    stem   = os.path.splitext(os.path.basename(ref_tif_path))[0]
    prefix = os.path.join(config.OUTPUT_DIR, f"{stem}_{model_name}")

    # 1. NoData 掩膜
    valid_mask        = compute_nodata_mask(image_norm)
    masked_prob, pred = apply_nodata_mask(prob_map, valid_mask)

    nodata_pct = (1 - valid_mask.mean()) * 100
    print(f"  NoData 区域: {nodata_pct:.1f}%  "
          f"有效像素: {valid_mask.mean()*100:.1f}%")

    # 2. 对象提取
    objects = extract_objects(pred,
                              masked_prob,
                              target_cls=target_cls,
                              min_area_px=min_area_px)
    hi_conf = sum(1 for o in objects if o["mean_confidence"] >= conf_thr)
    print(f"  Class{target_cls}({config.CLASS_NAMES.get(target_cls)}) "
          f"对象: 共{len(objects)}个  置信度>={conf_thr:.2f}: {hi_conf}个")

    # 3. 对象检测图
    obj_img = draw_objects(image_rgb, objects, conf_thr)
    obj_out = prefix + f"-class{target_cls}-objects.png"
    Image.fromarray(obj_img).save(obj_out)
    print(f"  ✔ 对象图   → {obj_out}")

    # 4. GeoJSON
    geo_out = prefix + f"-class{target_cls}-objects.geojson"
    save_geojson(objects, ref_tif_path, geo_out, conf_thr)

    return {
        "masked_prob": masked_prob,
        "pred_map":    pred,
        "valid_mask":  valid_mask,
        "objects":     objects,
    }
