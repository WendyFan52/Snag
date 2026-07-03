"""
Shared utilities for stage-3 analysis scripts.
"""

import csv
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config
from amp_compat import autocast_context
from dataset import SatelliteDataset, get_val_transforms, load_mask_png, validate_mask_values
from inference import ALL_MODELS, load_model, load_temperature
from random_forest_utils import (
    is_random_forest_model,
    load_random_forest,
    predict_proba_for_normalized_nchw,
)


DEFAULT_GROUP_FIELDS = (
    "climate_zone",
    "koppen_class",
    "latitude_band",
    "background_type",
    "spatial_pattern_type",
    "dead_area_class",
    "aggregation_class",
    "patch_size_dominance_class",
    "tree_density_class",
    "dead_density_class",
)


def parse_sample_id(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    return stem.split("_")[0]


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def load_manifest_rows(manifest_csv: Optional[str]) -> List[dict]:
    if not manifest_csv:
        return []
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
    with open(manifest_csv, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_manifest_lookup(manifest_csv: Optional[str]) -> Dict[str, dict]:
    rows = load_manifest_rows(manifest_csv)
    if manifest_csv and not rows:
        raise RuntimeError(f"Manifest contains no rows: {manifest_csv}")
    lookup = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        if not str(row.get("climate_zone", "")).strip() and str(row.get("thermal_zone", "")).strip():
            raise RuntimeError(
                f"sample_id={sample_id} has legacy thermal_zone but no true climate_zone. "
                "Rebuild sample management with Koppen raster instead of using legacy fallback."
            )
        old_pattern_labels = {"compact_cluster", "fragmented_cluster", "mixed_pattern", "no_dead"}
        background_type = str(row.get("background_type", "")).strip()
        spatial_pattern_type = str(row.get("spatial_pattern_type", "")).strip()
        if not spatial_pattern_type and background_type in old_pattern_labels:
            raise RuntimeError(
                f"sample_id={sample_id} has legacy pattern label in background_type={background_type}. "
                "Rebuild sample management so background_type and spatial_pattern_type are separate fields."
            )
        lookup[sample_id] = row
    return lookup


def require_manifest_row(lookup: Dict[str, dict], sample_id: str, context: str = "") -> dict:
    row = lookup.get(sample_id)
    if row is None:
        where = f" during {context}" if context else ""
        raise KeyError(f"sample_id={sample_id} is missing from manifest lookup{where}")
    return row


def require_manifest_fields(row: dict, fields: Sequence[str], sample_id: str, context: str = ""):
    missing = [field for field in fields if not str(row.get(field, "")).strip()]
    if missing:
        where = f" during {context}" if context else ""
        raise KeyError(f"sample_id={sample_id} missing required manifest fields{where}: {missing}")


def build_thresholds(start: float = 0.1, stop: float = 0.9, step: float = 0.1) -> List[float]:
    seq = np.arange(start, stop + 1e-9, step, dtype=np.float32)
    return [round(float(v), 4) for v in seq.tolist()]


def threshold_lookup_from_csv(path: Optional[str], allow_missing: bool = False) -> Dict[str, float]:
    if not path or not os.path.exists(path):
        message = (
            f"threshold summary not found: {path}"
            if path
            else "no threshold summary provided"
        )
        if allow_missing:
            print(f"[analysis][warn] {message}; downstream will use explicitly allowed fallback thresholds")
            return {}
        raise FileNotFoundError(
            f"{message}. Run analyze_thresholds.py first or pass --allow_fallback_threshold explicitly."
        )
    lookup: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            model = str(row.get("model", "")).strip()
            thr = safe_float(row.get("best_threshold", None), default=np.nan)
            if model and not np.isnan(thr):
                lookup[model] = float(thr)
    return lookup


def one_hot_from_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    mask = validate_mask_values(mask, source="analysis one-hot fallback mask")
    eye = np.eye(num_classes, dtype=np.float32)
    out = eye[mask.astype(np.int64)]
    return out.transpose(2, 0, 1)


def confusion_from_arrays(targets: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    valid = (targets != config.IGNORE_INDEX)
    if not np.any(valid):
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    tgt = targets[valid].astype(np.int64)
    pred = preds[valid].astype(np.int64)
    flat = np.bincount(tgt * num_classes + pred, minlength=num_classes * num_classes)
    return flat.reshape(num_classes, num_classes)


def metrics_from_confusion(conf: np.ndarray) -> Dict[str, float]:
    num_classes = conf.shape[0]
    per_class = {}
    ious = []
    f1s = []
    ps = []
    rs = []

    for cls in range(1, num_classes):
        tp = float(conf[cls, cls])
        fp = float(conf[:, cls].sum() - tp)
        fn = float(conf[cls, :].sum() - tp)
        iou = safe_div(tp, tp + fp + fn)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * tp, 2 * tp + fp + fn)
        per_class[cls] = {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(conf[cls, :].sum()),
        }
        ious.append(iou)
        f1s.append(f1)
        ps.append(precision)
        rs.append(recall)

    correct = float(sum(conf[c, c] for c in range(1, num_classes)))
    total = float(conf[1:, :].sum())
    oa = safe_div(correct, total)
    pe = safe_div(
        float(sum(conf[i, :].sum() * conf[:, i].sum() for i in range(1, num_classes))),
        total * total if total else 1.0,
    )
    kappa = safe_div(oa - pe, 1 - pe) if total else 0.0

    dead = per_class.get(1, {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
    living = per_class.get(2, {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})

    result = {
        "OA": oa,
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "mF1": float(np.mean(f1s)) if f1s else 0.0,
        "mPrecision": float(np.mean(ps)) if ps else 0.0,
        "mRecall": float(np.mean(rs)) if rs else 0.0,
        "Kappa": kappa,
        "dead_iou": dead["iou"],
        "dead_precision": dead["precision"],
        "dead_recall": dead["recall"],
        "dead_f1": dead["f1"],
        "living_iou": living["iou"],
        "living_precision": living["precision"],
        "living_recall": living["recall"],
        "living_f1": living["f1"],
        "dead_support": dead["support"],
        "living_support": living["support"],
    }
    for cls, metrics in per_class.items():
        result[f"class{cls}_iou"] = metrics["iou"]
        result[f"class{cls}_precision"] = metrics["precision"]
        result[f"class{cls}_recall"] = metrics["recall"]
        result[f"class{cls}_f1"] = metrics["f1"]
        result[f"class{cls}_support"] = metrics["support"]
    return result


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "iou": iou,
    }


def write_csv(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write("")
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def unique_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def quantile_bin_edges(values: Iterable[float], num_bins: int) -> List[float]:
    seq = sorted(float(v) for v in values if v is not None)
    if not seq:
        return [0.0, 1.0]
    if len(seq) == 1:
        return [seq[0], seq[0]]

    edges = [seq[0]]
    for q in range(1, num_bins):
        pos = (len(seq) - 1) * (q / num_bins)
        lo = int(pos)
        hi = min(lo + 1, len(seq) - 1)
        frac = pos - lo
        value = seq[lo] * (1.0 - frac) + seq[hi] * frac
        edges.append(float(value))
    edges.append(seq[-1])
    return edges


def assign_bin(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    if len(edges) <= 2:
        return np.zeros_like(values, dtype=np.int64)
    return np.digitize(values, edges[1:-1], right=False).astype(np.int64)


def bin_label(edges: Sequence[float], idx: int) -> str:
    lo = edges[idx]
    hi = edges[min(idx + 1, len(edges) - 1)]
    return f"[{lo:.4f}, {hi:.4f}]"


class AnalysisTileDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        manifest_csv: Optional[str] = None,
        softlabel_dir: Optional[str] = None,
    ):
        base = SatelliteDataset(
            img_dir,
            mask_dir,
            softlabel_dir=softlabel_dir,
            manifest_csv=manifest_csv,
            transform=None,
        )
        self.samples = list(base.samples)
        self.softlabel_dir = base.softlabel_dir
        self.strict_softlabels = getattr(base, "strict_softlabels", False)
        self.transform = get_val_transforms()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_softlabel(self, stem: str, mask: np.ndarray) -> np.ndarray:
        if self.softlabel_dir:
            path = os.path.join(self.softlabel_dir, stem + ".npy")
            if os.path.exists(path):
                soft = np.load(path).astype(np.float32)
                expected_shape = (config.NUM_CLASSES + 1, *mask.shape)
                if soft.shape == expected_shape:
                    return soft
                if self.strict_softlabels:
                    raise RuntimeError(
                        f"Softlabel shape mismatch for analysis tile {stem}: "
                        f"{soft.shape} != {expected_shape} ({path})"
                    )
            elif self.strict_softlabels:
                raise FileNotFoundError(
                    f"Softlabel file missing for analysis tile {stem}: {path}. "
                    "SoftLabel is enabled because the directory contains .npy files."
                )
        return one_hot_from_mask(mask, config.NUM_CLASSES + 1)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        sample_id = parse_sample_id(stem)

        image = SatelliteDataset.read_tif(img_path)
        mask = load_mask_png(mask_path)
        soft = self._load_softlabel(stem, mask)

        aug = self.transform(image=image, mask=mask)
        image_t = aug["image"]
        mask_t = aug["mask"].long()
        soft_t = torch.from_numpy(soft).float()
        return image_t, mask_t, soft_t, stem, sample_id


def build_analysis_loader(
    manifest_csv: Optional[str] = None,
    img_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 2,
) -> DataLoader:
    dataset = AnalysisTileDataset(
        img_dir or config.TRAIN_IMG_DIR,
        mask_dir or config.TRAIN_MASK_DIR,
        manifest_csv=manifest_csv,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def iterate_model_batches(
    model_name: str,
    device: torch.device,
    manifest_csv: Optional[str] = None,
    img_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 2,
):
    loader = build_analysis_loader(
        manifest_csv=manifest_csv,
        img_dir=img_dir,
        mask_dir=mask_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if is_random_forest_model(model_name):
        rf_bundle = load_random_forest()
        for images, masks, softs, stems, sample_ids in loader:
            probs = predict_proba_for_normalized_nchw(rf_bundle, images.cpu().numpy())
            preds = probs[:, 1:].argmax(axis=1).astype(np.int64) + 1
            masks_np = masks.cpu().numpy()
            preds[masks_np == config.IGNORE_INDEX] = config.IGNORE_INDEX
            yield {
                "probs": probs,
                "preds": preds,
                "masks": masks_np,
                "softs": softs.cpu().numpy(),
                "stems": list(stems),
                "sample_ids": list(sample_ids),
            }
        return

    model = load_model(model_name, device)
    temperature = load_temperature(model_name)
    for images, masks, softs, stems, sample_ids in loader:
        images = images.to(device)
        with autocast_context(device):
            logits = model(images)
        if temperature != 1.0:
            logits = logits.float() / temperature
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        preds = probs[:, 1:].argmax(axis=1).astype(np.int64) + 1
        masks_np = masks.cpu().numpy()
        preds[masks_np == config.IGNORE_INDEX] = config.IGNORE_INDEX
        yield {
            "probs": probs,
            "preds": preds,
            "masks": masks_np,
            "softs": softs.cpu().numpy(),
            "stems": list(stems),
            "sample_ids": list(sample_ids),
        }


def available_models(requested_model: str) -> List[str]:
    return list(ALL_MODELS) if requested_model == "all" else [requested_model]
