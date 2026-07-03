"""
真值可视化对比 — visualize_gt.py
-----------------------------------------------------------------
生成四联对比图：
  原图RGB | 模型预测 | SHP真值叠加 | Class1(死树)概率热力图

用法：
  python visualize_gt.py \
      --image /path/to/image.tif \
      --shp   /path/to/labels.shp \
      --model unet
-----------------------------------------------------------------
"""

import os, argparse
import numpy as np
import cv2
from PIL import Image
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

import config
from inference import (load_model, load_temperature,
                       sliding_window_inference,
                       read_for_display, read_full_tif_features,
                       pred_to_color, confidence_to_image)
from postprocess import compute_nodata_mask, apply_nodata_mask
from random_forest_utils import (
    is_random_forest_model,
    load_random_forest,
    predict_proba_for_image,
)


# -----------------------------------------------------------------
# SHP 栅格化
# -----------------------------------------------------------------
def rasterize_shp(shp_path: str, ref_tif_path: str) -> np.ndarray:
    """将 SHP 栅格化为与 TIF 相同尺寸的 mask (H,W)"""
    with rasterio.open(ref_tif_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        crs       = src.crs

    gdf   = gpd.read_file(shp_path)
    field = config.SHP_CLASS_FIELD

    if field not in gdf.columns:
        raise KeyError(f"SHP 缺少字段 '{field}'")

    if gdf.crs and crs and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    gdf = gdf[gdf[field].notna()]
    if len(gdf) == 0:
        return np.zeros(out_shape, dtype=np.int32)

    gdf[field] = gdf[field].astype(int)
    shapes = [(g, int(v))
              for g, v in zip(gdf.geometry, gdf[field])
              if g is not None and not g.is_empty]

    return rasterize(shapes, out_shape=out_shape,
                     transform=transform, fill=0,
                     dtype=np.int32, all_touched=False)


# -----------------------------------------------------------------
# 真值叠加绘制
# -----------------------------------------------------------------
def draw_gt_overlay(image_rgb: np.ndarray,
                    gt_mask:   np.ndarray,
                    alpha:     float = 0.45) -> np.ndarray:
    """
    在 RGB 影像上叠加 SHP 真值：
      · 半透明填充色（各类按调色板）
      · 白色轮廓线
      · 左上角类别图例
    """
    canvas      = image_rgb.copy().astype(np.float32)
    color_layer = np.zeros_like(canvas)

    for cls_id, rgb in config.PALETTE.items():
        if cls_id == 0:
            continue
        mask = (gt_mask == cls_id)
        if not mask.any():
            continue
        color_layer[mask] = rgb

        # 白色轮廓
        mask_u8  = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas.astype(np.uint8), contours, -1,
                         (255, 255, 255), 1)

    labeled = gt_mask > 0
    canvas[labeled] = (canvas[labeled] * (1 - alpha) +
                       color_layer[labeled] * alpha)

    result = canvas.clip(0, 255).astype(np.uint8)
    _draw_legend(result)
    return result


def _draw_legend(img: np.ndarray, fscale: float = 0.40):
    """Draw class legend using cv2 (ASCII only — cv2 does not support Unicode)."""
    y0 = 16
    for cls_id, name in config.CLASS_NAMES.items():
        rgb = config.PALETTE[cls_id]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(img, (8, y0 - 10), (22, y0 + 2), bgr, -1)
        # Encode to ASCII, replace non-ASCII chars with '?'
        label = f"[{cls_id}] {name}".encode("ascii", errors="replace").decode("ascii")
        cv2.putText(img, label,
                    (26, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    fscale, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 18


# -----------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------
def make_comparison(image_path: str, shp_path: str,
                    model_name: str,
                    device) -> str:
    """生成四联对比图，返回保存路径"""
    # 1. 读图 + 推理
    image_rgb  = read_for_display(image_path)
    image_norm = read_full_tif_features(image_path)

    if is_random_forest_model(model_name):
        print("[RF] visualize_gt uses pixel-wise random forest probabilities; no temperature scaling is applied.")
        rf_bundle = load_random_forest()
        prob_map = predict_proba_for_image(rf_bundle, image_norm)
    else:
        model    = load_model(model_name, device)
        T        = load_temperature(model_name)
        prob_map = sliding_window_inference(model, image_norm, device,
                                            temperature=T)

    # 2. NoData 掩膜
    valid_mask        = compute_nodata_mask(image_norm)
    masked_prob, pred = apply_nodata_mask(prob_map, valid_mask)

    pred_color  = pred_to_color(pred)
    class1_conf = confidence_to_image(masked_prob[1])

    # 3. SHP 真值叠加
    gt_mask    = rasterize_shp(shp_path, image_path)
    gt_overlay = draw_gt_overlay(image_rgb, gt_mask)

    # 4. 四联图拼接（加标签行）
    H = image_rgb.shape[0]
    W = image_rgb.shape[1]
    sep  = np.full((H, 4, 3), 180, dtype=np.uint8)

    label_h  = 26
    label_bar = np.zeros((label_h, (W + 4) * 4, 3), dtype=np.uint8)
    for i, lbl in enumerate(["RGB Image", f"{model_name.upper()} Pred",
                              "SHP Ground Truth", "Class1(Dead) Conf"]):
        x = i * (W + 4) + W // 2 - len(lbl) * 4
        cv2.putText(label_bar, lbl, (max(x, 2), 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

    panel = np.concatenate(
        [image_rgb, sep, pred_color, sep, gt_overlay, sep, class1_conf],
        axis=1
    )
    final = np.vstack([label_bar, panel])

    stem   = os.path.splitext(os.path.basename(image_path))[0]
    out    = os.path.join(config.OUTPUT_DIR,
                          f"{stem}-{model_name}-gt-comparison.png")
    Image.fromarray(final).save(out)
    print(f"  ✔ 四联对比图 → {out}")
    return out


if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--shp",    required=True)
    parser.add_argument("--model", default=config.DEFAULT_MODEL,
                        choices=list(config.SUPPORTED_MODELS) + ["all"])
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = list(config.SUPPORTED_MODELS) if args.model == "all" else [args.model]
    for m in models:
        make_comparison(args.image, args.shp, m, device)
