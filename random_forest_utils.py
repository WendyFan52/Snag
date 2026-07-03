"""
Random forest baseline utilities.

The random forest uses the same prepared input feature stack as the deep
models: PlanetScope reflectance bands plus spectral indices, followed by the
same channel-wise normalization statistics. It is a pixel-wise spectral
baseline and does not model spatial context.
"""

import json
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio

import config
from dataset import SatelliteDataset, load_mask_png
from input_preprocessing import sr_to_feature_stack


MODEL_NAME = "random_forest"


def is_random_forest_model(model_name: str) -> bool:
    return model_name == MODEL_NAME


def random_forest_checkpoint_path() -> str:
    return os.path.join(config.CHECKPOINT_DIR, f"best_{MODEL_NAME}.joblib")


def random_forest_metadata_path() -> str:
    return os.path.join(config.CHECKPOINT_DIR, f"best_{MODEL_NAME}.json")


def _require_sklearn():
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise ImportError(
            "Random forest baseline requires scikit-learn and joblib. "
            "Install them with: pip install scikit-learn joblib"
        ) from exc
    return joblib, RandomForestClassifier


def normalize_feature_stack(image_hwc: np.ndarray) -> np.ndarray:
    config.require_channel_stats()
    image = np.asarray(image_hwc, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != config.IN_CHANNELS:
        raise RuntimeError(
            f"RF input feature shape mismatch: got {image.shape}, "
            f"expected (H, W, {config.IN_CHANNELS})"
        )
    if not np.isfinite(image).all():
        raise RuntimeError("RF input feature stack contains NaN or Inf values")
    mean = np.asarray(config.IMG_MEAN, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(config.IMG_STD, dtype=np.float32).reshape(1, 1, -1)
    if np.any(std <= 0):
        raise RuntimeError("RF normalization std contains non-positive values")
    return (image - mean) / std


def read_full_tif_features(tif_path: str) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
    if data.shape[0] != config.RAW_CHANNELS:
        raise RuntimeError(
            f"Unexpected raw channel count for {tif_path}: got {data.shape[0]}, "
            f"expected {config.RAW_CHANNELS} for BAND_MODE={config.BAND_MODE}"
        )
    features = sr_to_feature_stack(data, source=tif_path)
    return features.transpose(1, 2, 0)


def _load_mask(mask_path: str) -> np.ndarray:
    return load_mask_png(mask_path)


def _sample_pixels_for_class(
    rng: np.random.Generator,
    features_hwc: np.ndarray,
    labels_hw: np.ndarray,
    weights_hw: np.ndarray,
    cls_id: int,
    keep_probability: float,
) -> Tuple[np.ndarray, np.ndarray]:
    cls_mask = labels_hw == cls_id
    n_cls = int(cls_mask.sum())
    if n_cls == 0:
        return np.empty((0, config.IN_CHANNELS), dtype=np.float32), np.empty((0,), dtype=np.float32)
    if keep_probability < 1.0:
        keep = rng.random(n_cls) < keep_probability
    else:
        keep = np.ones(n_cls, dtype=bool)
    if not keep.any():
        return np.empty((0, config.IN_CHANNELS), dtype=np.float32), np.empty((0,), dtype=np.float32)
    x_cls = features_hwc[cls_mask][keep].astype(np.float32, copy=False)
    w_cls = weights_hw[cls_mask][keep].astype(np.float32, copy=False)
    return x_cls, w_cls


def collect_random_forest_training_samples(
    manifest_csv: str,
    img_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    max_pixels_per_class: int = 150000,
    seed: int = config.SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    if max_pixels_per_class <= 0:
        raise ValueError("max_pixels_per_class must be positive")

    dataset = SatelliteDataset(
        img_dir or config.TRAIN_IMG_DIR,
        mask_dir or config.TRAIN_MASK_DIR,
        transform=None,
        manifest_csv=manifest_csv,
    )
    if not getattr(dataset, "strict_lci", False):
        raise RuntimeError(
            "Random forest training requires LCI maps for sample_weight, but LCI is disabled. "
            "Run prepare_data_softlabel.py and ensure DATA_ROOT/lci contains .npy maps."
        )
    print("[RF] LCI sample weights: enabled and required")
    print("[RF] Label source: hard mask labels; soft labels are not used by this pixel-wise baseline")

    class_counts = {cls: 0 for cls in range(1, config.NUM_CLASSES + 1)}
    for _, mask_path in dataset.samples:
        mask = _load_mask(mask_path)
        for cls in class_counts:
            class_counts[cls] += int((mask == cls).sum())

    missing_classes = [cls for cls, count in class_counts.items() if count == 0]
    if missing_classes:
        raise RuntimeError(
            "Random forest training split lacks required class pixels: "
            + ", ".join(str(cls) for cls in missing_classes)
        )

    keep_probability = {
        cls: min(1.0, float(max_pixels_per_class) / float(count))
        for cls, count in class_counts.items()
    }

    rng = np.random.default_rng(seed)
    x_by_class = defaultdict(list)
    w_by_class = defaultdict(list)

    for img_path, mask_path in dataset.samples:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        image = SatelliteDataset.read_tif(img_path)
        image = normalize_feature_stack(image)
        mask = _load_mask(mask_path)
        lci = dataset._load_lci(stem, mask.shape)
        if not np.isfinite(lci).all():
            raise RuntimeError(f"LCI map contains NaN or Inf values: {stem}")
        if np.any((lci < 0) | (lci > 1)):
            raise RuntimeError(f"LCI map has values outside [0, 1]: {stem}")

        for cls in range(1, config.NUM_CLASSES + 1):
            x_cls, w_cls = _sample_pixels_for_class(
                rng,
                image,
                mask,
                lci,
                cls,
                keep_probability[cls],
            )
            if x_cls.size:
                x_by_class[cls].append(x_cls)
                w_by_class[cls].append(w_cls)

    xs = []
    ys = []
    ws = []
    sampled_counts = {}
    for cls in range(1, config.NUM_CLASSES + 1):
        if not x_by_class[cls]:
            raise RuntimeError(f"Random forest sampled no pixels for class {cls}")
        x_cls = np.concatenate(x_by_class[cls], axis=0)
        w_cls = np.concatenate(w_by_class[cls], axis=0)
        if x_cls.shape[0] > max_pixels_per_class:
            idx = rng.choice(x_cls.shape[0], size=max_pixels_per_class, replace=False)
            x_cls = x_cls[idx]
            w_cls = w_cls[idx]
        y_cls = np.full((x_cls.shape[0],), cls, dtype=np.int64)
        if float(w_cls.sum()) <= 0.0:
            raise RuntimeError(
                f"Random forest sampled class {cls} but all LCI sample weights are zero"
            )
        xs.append(x_cls)
        ys.append(y_cls)
        ws.append(w_cls)
        sampled_counts[cls] = int(x_cls.shape[0])

    x = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    sample_weight = np.concatenate(ws, axis=0).astype(np.float32, copy=False)
    order = rng.permutation(y.shape[0])
    metadata = {
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "sampled_counts": {str(k): int(v) for k, v in sampled_counts.items()},
        "manifest_csv": manifest_csv,
        "tile_count": len(dataset.samples),
        "max_pixels_per_class": int(max_pixels_per_class),
        "seed": int(seed),
    }
    return x[order], y[order], sample_weight[order], metadata


def train_random_forest_classifier(
    manifest_csv: str,
    img_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    max_pixels_per_class: int = 150000,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 2,
    seed: int = config.SEED,
    n_jobs: int = -1,
):
    joblib, RandomForestClassifier = _require_sklearn()
    x, y, sample_weight, metadata = collect_random_forest_training_samples(
        manifest_csv=manifest_csv,
        img_dir=img_dir,
        mask_dir=mask_dir,
        max_pixels_per_class=max_pixels_per_class,
        seed=seed,
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=seed,
        verbose=0,
    )
    print(
        f"[RF] training pixels={len(y):,} features={x.shape[1]} "
        f"classes={sorted(np.unique(y).tolist())}"
    )
    clf.fit(x, y, sample_weight=sample_weight)
    required_classes = set(range(1, config.NUM_CLASSES + 1))
    learned_classes = set(int(v) for v in clf.classes_)
    if learned_classes != required_classes:
        raise RuntimeError(
            f"Random forest learned classes {sorted(learned_classes)}, "
            f"expected {sorted(required_classes)}"
        )
    metadata.update(
        {
            "model_name": MODEL_NAME,
            "band_mode": config.BAND_MODE,
            "raw_channels": config.RAW_CHANNELS,
            "in_channels": config.IN_CHANNELS,
            "num_classes": config.NUM_CLASSES,
            "input_preprocessing": config.INPUT_PREPROCESSING_NAME,
            "img_mean": list(config.IMG_MEAN),
            "img_std": list(config.IMG_STD),
            "n_estimators": int(n_estimators),
            "max_depth": max_depth,
            "min_samples_leaf": int(min_samples_leaf),
            "class_weight": "balanced_subsample",
            "uses_lci_sample_weight": True,
        }
    )
    bundle = {"model": clf, "metadata": metadata}
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = random_forest_checkpoint_path()
    joblib.dump(bundle, ckpt_path)
    with open(random_forest_metadata_path(), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"[RF] checkpoint -> {ckpt_path}")
    print(f"[RF] metadata -> {random_forest_metadata_path()}")
    return bundle


def load_random_forest():
    joblib, _ = _require_sklearn()
    ckpt_path = random_forest_checkpoint_path()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Random forest checkpoint not found: {ckpt_path}. "
            "Run train_random_forest.py first."
        )
    bundle = joblib.load(ckpt_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise RuntimeError(f"Invalid random forest checkpoint format: {ckpt_path}")
    metadata = bundle["metadata"]
    checks = {
        "band_mode": config.BAND_MODE,
        "raw_channels": config.RAW_CHANNELS,
        "in_channels": config.IN_CHANNELS,
        "num_classes": config.NUM_CLASSES,
        "input_preprocessing": config.INPUT_PREPROCESSING_NAME,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in checks.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Random forest checkpoint metadata mismatch: {mismatches}")
    clf = bundle["model"]
    required_classes = set(range(1, config.NUM_CLASSES + 1))
    learned_classes = set(int(v) for v in getattr(clf, "classes_", []))
    if learned_classes != required_classes:
        raise RuntimeError(
            f"Random forest checkpoint learned classes {sorted(learned_classes)}, "
            f"expected {sorted(required_classes)}"
        )
    print(
        f"[RF] loaded checkpoint: {ckpt_path} | "
        f"classes={sorted(learned_classes)} | in_channels={metadata.get('in_channels')}"
    )
    return bundle


def predict_proba_from_normalized_features(bundle: dict, x: np.ndarray) -> np.ndarray:
    clf = bundle["model"]
    required_classes = set(range(1, config.NUM_CLASSES + 1))
    learned_classes = set(int(v) for v in getattr(clf, "classes_", []))
    if learned_classes != required_classes:
        raise RuntimeError(
            f"Random forest checkpoint learned classes {sorted(learned_classes)}, "
            f"expected {sorted(required_classes)}"
        )
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != config.IN_CHANNELS:
        raise RuntimeError(
            f"RF feature matrix shape mismatch: got {x.shape}, "
            f"expected (N, {config.IN_CHANNELS})"
        )
    if not np.isfinite(x).all():
        raise RuntimeError("RF feature matrix contains NaN or Inf values")
    proba_raw = clf.predict_proba(x)
    out = np.zeros((x.shape[0], config.NUM_CLASSES + 1), dtype=np.float32)
    for idx, cls_id in enumerate(clf.classes_):
        cls_int = int(cls_id)
        if 1 <= cls_int <= config.NUM_CLASSES:
            out[:, cls_int] = proba_raw[:, idx].astype(np.float32, copy=False)
    return out


def predict_proba_for_normalized_nchw(
    bundle: dict,
    images_nchw: np.ndarray,
    batch_pixels: int = 500000,
) -> np.ndarray:
    images = np.asarray(images_nchw, dtype=np.float32)
    if images.ndim != 4 or images.shape[1] != config.IN_CHANNELS:
        raise RuntimeError(
            f"RF NCHW input shape mismatch: got {images.shape}, "
            f"expected (B, {config.IN_CHANNELS}, H, W)"
        )
    b, c, h, w = images.shape
    x = images.transpose(0, 2, 3, 1).reshape(-1, c)
    probs = np.zeros((x.shape[0], config.NUM_CLASSES + 1), dtype=np.float32)
    for start in range(0, x.shape[0], batch_pixels):
        end = min(start + batch_pixels, x.shape[0])
        probs[start:end] = predict_proba_from_normalized_features(bundle, x[start:end])
    return probs.reshape(b, h, w, config.NUM_CLASSES + 1).transpose(0, 3, 1, 2)


def predict_proba_for_image(
    bundle: dict,
    image_hwc: np.ndarray,
    batch_pixels: int = 500000,
) -> np.ndarray:
    normalized = normalize_feature_stack(image_hwc)
    h, w, c = normalized.shape
    x = normalized.reshape(-1, c)
    probs = np.zeros((x.shape[0], config.NUM_CLASSES + 1), dtype=np.float32)
    for start in range(0, x.shape[0], batch_pixels):
        end = min(start + batch_pixels, x.shape[0])
        probs[start:end] = predict_proba_from_normalized_features(bundle, x[start:end])
    return probs.reshape(h, w, config.NUM_CLASSES + 1).transpose(2, 0, 1)
