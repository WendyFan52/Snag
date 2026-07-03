"""
Full-image inference script.
"""

import os
import argparse
import json
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import rasterio

import config
from amp_compat import autocast_context
from dataset import get_infer_transforms
from models import (
    build_unet,
    build_unet_resnet34,
    build_unet_convnext_tiny,
    build_hrnet_w18,
    build_segformer,
    build_segformer_b4,
    build_segformer_b5,
    build_r50_upernet,
)
from input_preprocessing import sr_to_feature_stack, sr_to_reflectance
from postprocess import postprocess
from random_forest_utils import (
    is_random_forest_model,
    load_random_forest,
    predict_proba_for_image,
)


MODEL_BUILDERS = {
    "unet": build_unet,
    "unet_resnet34": build_unet_resnet34,
    "unet_convnext_tiny": build_unet_convnext_tiny,
    "hrnet_w18": build_hrnet_w18,
    "segformer": build_segformer,
    "segformer_b4": build_segformer_b4,
    "segformer_b5": build_segformer_b5,
    "r50_upernet": build_r50_upernet,
}

ALL_MODELS = list(config.SUPPORTED_MODELS)


def _infer_segformer_decoder_hidden_size(state_dict: dict):
    key = "model.decode_head.linear_c.0.proj.weight"
    weight = state_dict.get(key)
    if weight is None:
        return None
    return int(weight.shape[0])


def load_model(model_name: str, device: torch.device):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"best_{model_name}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nRun train.py --model {model_name} first."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"]
    saved_in_ch = ckpt.get("in_channels", config.IN_CHANNELS)
    n_cls = ckpt.get("num_classes", config.NUM_CLASSES) + 1

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}")

    builder = MODEL_BUILDERS[model_name]
    build_kwargs = {
        "num_classes": n_cls,
        "in_channels": saved_in_ch,
    }
    # Builders in this registry expose a consistent inference signature.
    if "pretrained" in inspect.signature(builder).parameters:
        build_kwargs["pretrained"] = False
    if model_name.startswith("segformer") and "decoder_hidden_size" in inspect.signature(builder).parameters:
        decoder_hidden_size = _infer_segformer_decoder_hidden_size(state_dict)
        if decoder_hidden_size is None:
            raise RuntimeError(
                f"Cannot infer SegFormer decoder_hidden_size from checkpoint: {ckpt_path}"
            )
        build_kwargs["decoder_hidden_size"] = decoder_hidden_size
        print(f"[Load] {model_name}: inferred decoder_hidden_size={decoder_hidden_size} from checkpoint")
    model = builder(**build_kwargs)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    print(
        f"[Load] {model_name.upper()} | epoch={ckpt.get('epoch', '?')} | "
        f"mIoU={ckpt.get('miou', 0):.4f} | in_channels={saved_in_ch}"
    )
    return model


def load_temperature(model_name: str, allow_uncalibrated: bool = False) -> float:
    t_path = os.path.join(config.CHECKPOINT_DIR, f"temperature-{model_name}.json")
    if os.path.exists(t_path):
        with open(t_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "temperature" not in data:
            raise KeyError(f"Temperature file lacks 'temperature': {t_path}")
        temperature = float(data["temperature"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise RuntimeError(f"Invalid temperature={temperature!r} in {t_path}")
        ece_before = data.get("ece_before")
        ece_after = data.get("ece_after")
        if ece_before is not None and ece_after is not None:
            print(f"  [Calibration] T={temperature:.4f}  ECE: {ece_before:.4f} -> {ece_after:.4f}")
        else:
            print(f"  [Calibration] T={temperature:.4f}")
        return temperature

    if allow_uncalibrated:
        print("  [Calibration][warn] No temperature file; explicitly using uncalibrated T=1.0")
        return 1.0
    raise FileNotFoundError(
        f"Temperature calibration file not found: {t_path}. "
        "Run calibrate.py first, or pass --allow_uncalibrated explicitly."
    )


def read_full_tif_features(tif_path: str) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)

    raw_ch = config.RAW_CHANNELS
    if data.shape[0] != raw_ch:
        raise RuntimeError(
            f"Unexpected raw channel count for {tif_path}: got {data.shape[0]}, "
            f"expected {raw_ch} for BAND_MODE={config.BAND_MODE}. "
            "Refuse to pad/truncate inference bands."
        )

    data = sr_to_feature_stack(data, source=tif_path)
    return data.transpose(1, 2, 0)

def read_for_display(tif_path: str) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)

    reflectance = sr_to_reflectance(data, source=tif_path)
    if config.RAW_CHANNELS == 4:
        # PlanetScope 4-band order: Blue, Green, Red, NIR.
        rgb = np.stack([reflectance[2], reflectance[1], reflectance[0]], axis=-1)
    elif config.RAW_CHANNELS == 8:
        # PlanetScope 8-band order: Coastal Blue, Blue, Green I, Green,
        # Yellow, Red, Red Edge, NIR.
        rgb = np.stack([reflectance[5], reflectance[3], reflectance[1]], axis=-1)
    else:
        raise RuntimeError(f"Unsupported RAW_CHANNELS for RGB display: {config.RAW_CHANNELS}")

    rgb = np.clip(rgb * config.RGB_DISPLAY_GAIN, 0.0, 1.0)
    if config.RGB_DISPLAY_GAMMA <= 0:
        raise RuntimeError(f"RGB_DISPLAY_GAMMA must be positive, got {config.RGB_DISPLAY_GAMMA}")
    if config.RGB_DISPLAY_GAMMA != 1.0:
        rgb = np.power(rgb, 1.0 / config.RGB_DISPLAY_GAMMA)
    return np.rint(rgb * 255.0).astype(np.uint8)


def pred_to_color(pred_map: np.ndarray) -> np.ndarray:
    color = np.zeros((*pred_map.shape, 3), dtype=np.uint8)
    for cls_id, rgb in config.PALETTE.items():
        color[pred_map == cls_id] = rgb
    return color


def confidence_to_image(conf_map: np.ndarray,
                        low_thr: float = config.CONFIDENCE_LOW_THRESHOLD,
                        high_thr: float = config.CONFIDENCE_HIGH_THRESHOLD) -> np.ndarray:
    h, w = conf_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    low = conf_map < low_thr
    mid = (conf_map >= low_thr) & (conf_map < high_thr)
    high = conf_map >= high_thr

    t_low = np.clip(conf_map[low] / max(low_thr, 1e-9), 0, 1)
    rgb[low, 0] = (t_low * 50).astype(np.uint8)
    rgb[low, 1] = (t_low * 80).astype(np.uint8)
    rgb[low, 2] = (150 + t_low * 105).astype(np.uint8)

    t_mid = (conf_map[mid] - low_thr) / max(high_thr - low_thr, 1e-9)
    rgb[mid, 0] = (200 + t_mid * 55).astype(np.uint8)
    rgb[mid, 1] = (200 - t_mid * 80).astype(np.uint8)
    rgb[mid, 2] = (50 - t_mid * 50).astype(np.uint8)

    t_high = (conf_map[high] - high_thr) / max(1.0 - high_thr, 1e-9)
    rgb[high, 0] = (220 + t_high * 35).astype(np.uint8)
    rgb[high, 1] = (t_high * 30).astype(np.uint8)
    return rgb


def valid_mask_to_img(valid_mask: np.ndarray) -> np.ndarray:
    return np.where(
        valid_mask[:, :, None],
        np.array([[[255, 255, 255]]], dtype=np.uint8),
        np.array([[[30, 30, 30]]], dtype=np.uint8),
    ).astype(np.uint8)


def gaussian_window(size: int, sigma_ratio: float = 6.0) -> np.ndarray:
    sigma = size / sigma_ratio
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return (kernel / kernel.max()).astype(np.float32)


@torch.no_grad()
def sliding_window_inference(model: torch.nn.Module,
                             image_np: np.ndarray,
                             device: torch.device,
                             temperature: float = 1.0) -> np.ndarray:
    transform = get_infer_transforms()
    num_classes = config.NUM_CLASSES + 1
    height, width = image_np.shape[:2]
    tile_size = config.INFER_TILE_SIZE
    overlap = config.INFER_OVERLAP
    stride = tile_size - overlap
    batch_size = config.INFER_BATCH

    accum_prob = np.zeros((num_classes, height, width), dtype=np.float64)
    accum_count = np.zeros((height, width), dtype=np.float64)
    gauss_win = gaussian_window(tile_size).astype(np.float64)

    coords = []
    y = 0
    while True:
        x = 0
        while True:
            y1, y2 = y, min(y + tile_size, height)
            x1, x2 = x, min(x + tile_size, width)
            coords.append((y1, y2, x1, x2))
            if x2 >= width:
                break
            x += stride
        if y2 >= height:
            break
        y += stride

    print(f"[Infer] {height}x{width} tile={tile_size} stride={stride} tiles={len(coords)} bs={batch_size}")

    batch_imgs, batch_meta = [], []

    def flush():
        if not batch_imgs:
            return
        tensor = torch.stack(batch_imgs).to(device)
        with autocast_context(device):
            logits = model(tensor)
        if temperature != 1.0:
            logits = logits.float() / temperature
        probs = F.softmax(logits, dim=1).float().cpu().numpy()

        for idx, (oy1, oy2, ox1, ox2) in enumerate(batch_meta):
            h_r, w_r = oy2 - oy1, ox2 - ox1
            p = probs[idx, :, :h_r, :w_r]
            g = gauss_win[:h_r, :w_r]
            accum_prob[:, oy1:oy2, ox1:ox2] += p * g[None]
            accum_count[oy1:oy2, ox1:ox2] += g
        batch_imgs.clear()
        batch_meta.clear()

    for idx, (y1, y2, x1, x2) in enumerate(coords):
        tile = image_np[y1:y2, x1:x2]
        pad_h, pad_w = tile_size - (y2 - y1), tile_size - (x2 - x1)
        if pad_h > 0 or pad_w > 0:
            tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        aug = transform(image=tile.astype(np.float32))
        batch_imgs.append(aug["image"])
        batch_meta.append((y1, y2, x1, x2))

        if len(batch_imgs) == batch_size or idx == len(coords) - 1:
            flush()

    return (accum_prob / np.maximum(accum_count[None], 1e-9)).astype(np.float32)


def save_legend(path: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        patches = [
            mpatches.Patch(color=[v / 255 for v in config.PALETTE[c]], label=f"[{c}] {n}")
            for c, n in config.CLASS_NAMES.items()
        ]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.legend(handles=patches, loc="center", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except Exception as exc:
        print(f"  [warn] Failed to save legend {path}: {exc}")


def infer_one(
    model_name: str,
    image_path: str,
    device: torch.device,
    allow_uncalibrated: bool = False,
):
    image_norm = read_full_tif_features(image_path)
    image_rgb = read_for_display(image_path)

    if is_random_forest_model(model_name):
        print("[RF] Pixel-wise random forest inference; no temperature scaling is applied.")
        rf_bundle = load_random_forest()
        prob_map = predict_proba_for_image(rf_bundle, image_norm)
    else:
        model = load_model(model_name, device)
        temperature = load_temperature(model_name, allow_uncalibrated=allow_uncalibrated)
        prob_map = sliding_window_inference(model, image_norm, device, temperature=temperature)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    prefix = os.path.join(config.OUTPUT_DIR, f"{stem}_{model_name}")

    pp = postprocess(
        prob_map=prob_map,
        image_norm=image_norm,
        image_rgb=image_rgb,
        ref_tif_path=image_path,
        model_name=model_name,
        target_cls=1,
    )
    masked_prob = pp["masked_prob"]
    pred_map = pp["pred_map"]
    valid_mask = pp["valid_mask"]

    pred_color = pred_to_color(pred_map)
    class1_conf = confidence_to_image(masked_prob[1])
    nodata_img = valid_mask_to_img(valid_mask)

    Image.fromarray(pred_color).save(prefix + "-pred.png")
    Image.fromarray(class1_conf).save(prefix + "-class1-conf.png")
    np.savez_compressed(
        prefix + "-all-probs.npz",
        prob_map=masked_prob,
        class_names=list(config.CLASS_NAMES.values()),
    )

    obj_path = prefix + "-class1-objects.png"
    try:
        obj_arr = np.array(Image.open(obj_path))
    except Exception as exc:
        raise RuntimeError(f"Expected object image was not created: {obj_path}") from exc

    sep = np.full((image_rgb.shape[0], 4, 3), 180, dtype=np.uint8)
    panel = np.concatenate(
        [image_rgb, sep, pred_color, sep, class1_conf, sep, nodata_img, sep, obj_arr],
        axis=1,
    )
    Image.fromarray(panel).save(prefix + "-comparison.png")

    print(f"  Prediction map -> {prefix}-pred.png")
    print(f"  Confidence map -> {prefix}-class1-conf.png")
    print(f"  Probability map -> {prefix}-all-probs.npz")
    print(f"  Comparison map -> {prefix}-comparison.png")

    save_legend(os.path.join(config.OUTPUT_DIR, "legend.png"))
    return masked_prob, pred_map


def save_model_comparison(image_path: str, results: dict):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        matplotlib.rcParams["axes.unicode_minus"] = False

        rgb = read_for_display(image_path)
        items = [("RGB Image", rgb)]
        for model_name, (prob_map, pred_map) in results.items():
            items.append((f"{model_name.upper()} Pred", pred_to_color(pred_map)))
            items.append((f"{model_name.upper()} Dead Conf", confidence_to_image(prob_map[1])))

        fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 5))
        for ax, (label, image) in zip(axes, items):
            ax.imshow(image)
            ax.set_title(label, fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out = os.path.join(config.OUTPUT_DIR, f"{stem}-model-comparison.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"\n  Model comparison -> {out}")
    except Exception as exc:
        print(f"  Failed to save comparison plot: {exc}")


def compare_models(
    image_path: str,
    device: torch.device,
    model_list: list = None,
    allow_uncalibrated: bool = False,
):
    if model_list is None:
        model_list = ALL_MODELS

    results = {}
    for model_name in model_list:
        prob_map, pred_map = infer_one(
            model_name,
            image_path,
            device,
            allow_uncalibrated=allow_uncalibrated,
        )
        results[model_name] = (prob_map, pred_map)

    if len(results) >= 2:
        save_model_comparison(image_path, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ALL_MODELS + ["all"], default="all")
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory override for split-based inference runs",
    )
    parser.add_argument(
        "--allow_uncalibrated",
        action="store_true",
        help="Allow inference without temperature-*.json by explicitly using T=1.0",
    )
    args = parser.parse_args()

    if args.output_dir:
        config.set_output_dir(args.output_dir)
        print(f"[Infer] OUTPUT_DIR override -> {config.OUTPUT_DIR}")

    config.ensure_runtime_dirs()
    config.print_config_summary()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "all":
        compare_models(args.image, device, allow_uncalibrated=args.allow_uncalibrated)
    else:
        infer_one(args.model, args.image, device, allow_uncalibrated=args.allow_uncalibrated)
