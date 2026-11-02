"""
Dataset loading and augmentation - dataset.py
-----------------------------------------------------------------
Behavior:
  - read_tif() appends spectral indices when reading, consistent with prepare_data.py.
  - Augmentations support arbitrary channel counts; HueSaturationValue is excluded.
  - IGNORE_INDEX=0 is excluded from the loss but remains available during inference.
-----------------------------------------------------------------
"""

import os, glob, csv
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio

import config
from input_preprocessing import assert_prepared_tile_tags, sr_to_feature_stack


def validate_mask_values(mask: np.ndarray, source: str = "mask") -> np.ndarray:
    """Validate semantic mask labels instead of silently clipping them."""
    mask_np = np.asarray(mask, dtype=np.int64)
    valid = ((mask_np >= 0) & (mask_np <= config.NUM_CLASSES)) | (mask_np == config.IGNORE_INDEX)
    if not np.all(valid):
        bad = np.unique(mask_np[~valid])
        preview = ", ".join(str(int(v)) for v in bad[:20])
        raise RuntimeError(
            f"Invalid class value(s) in {source}: {preview}. "
            f"Expected labels 0..{config.NUM_CLASSES} or IGNORE_INDEX={config.IGNORE_INDEX}. "
            "Refusing to clip labels because that would alter the ground truth."
        )
    return mask_np


def load_mask_png(mask_path: str) -> np.ndarray:
    mask = np.array(Image.open(mask_path), dtype=np.int64)
    return validate_mask_values(mask, source=mask_path)


# -----------------------------------------------------------------
# Augmentation pipeline (all operations support arbitrary channel counts)
# -----------------------------------------------------------------
def _image_tensor_transform(extra: list = None) -> A.Compose:
    config.require_channel_stats()
    ops = list(extra or [])
    ops += [
        A.Normalize(
            mean=list(config.IMG_MEAN),
            std=list(config.IMG_STD),
            max_pixel_value=1.0,   # Inputs are already in [0, 1]
        ),
        ToTensorV2(),
    ]
    return A.Compose(ops)


def get_train_transforms():
    return get_train_image_transforms()


def get_train_geom_transforms():
    return A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
    ])


def get_val_geom_transforms():
    return A.ReplayCompose([])


def get_val_transforms():
    return _image_tensor_transform()


def get_infer_transforms():
    """Normalize inference inputs; inference.py handles sliding windows."""
    return _image_tensor_transform()


def get_train_image_transforms():
    return _image_tensor_transform([
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.2),
        A.CoarseDropout(max_holes=4, max_height=8,
                        max_width=8, fill_value=0, p=0.2),
    ])


def _replay_image_only(replay: dict, image: np.ndarray) -> np.ndarray:
    return A.ReplayCompose.replay(replay, image=image)["image"]


def _prepare_sample(image: np.ndarray,
                    mask: np.ndarray,
                    lci_map: np.ndarray,
                    raw_soft: np.ndarray,
                    geom_transform: A.ReplayCompose = None,
                    image_transform: A.Compose = None):
    mask_np = np.asarray(mask, dtype=np.int64)
    lci_np = np.asarray(lci_map, dtype=np.float32)
    soft_np = None if raw_soft is None else np.asarray(raw_soft, dtype=np.float32)
    has_soft = soft_np is not None

    if geom_transform is not None:
        aug = geom_transform(image=image, mask=mask_np)
        image = aug["image"]
        mask_np = np.asarray(aug["mask"], dtype=np.int64)
        replay = aug["replay"]
        lci_np = _replay_image_only(replay, lci_np).astype(np.float32)
        if soft_np is not None:
            soft_hwc = np.moveaxis(soft_np, 0, -1)
            soft_hwc = _replay_image_only(replay, soft_hwc).astype(np.float32)
            soft_np = np.moveaxis(soft_hwc, -1, 0)

    if image_transform is not None:
        image_t = image_transform(image=image)["image"]
    else:
        image_t = torch.from_numpy(np.moveaxis(image, -1, 0).astype(np.float32))

    mask_t = torch.from_numpy(mask_np).long()
    lci_t = torch.from_numpy(lci_np).float()
    if soft_np is not None:
        soft_t = torch.from_numpy(soft_np).float()
    else:
        soft_t = torch.zeros((config.NUM_CLASSES + 1, *mask_np.shape), dtype=torch.float32)
    has_soft_t = torch.tensor(has_soft, dtype=torch.bool)
    return image_t, mask_t, lci_t, soft_t, has_soft_t


# -----------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------
class SatelliteDataset(Dataset):
    """
    Load tile pairs generated by prepare_data.py:
      images/*.tif  → (IN_CHANNELS, H, W) float32
      masks/*.png   -> (H, W) int, 0=unlabeled, 1..6=classes
    """

    def __init__(self, img_dir: str, mask_dir: str,
                 lci_dir:       str = None,
                 softlabel_dir: str = None,
                 manifest_csv:  str = None,
                 geom_transform=None,
                 transform=None):
        self.transform = transform
        self.geom_transform = geom_transform
        self.lci_dir   = lci_dir or getattr(config, 'TRAIN_LCI_DIR', None)
        _default_sl    = os.path.join(config.DATA_ROOT, "softlabels")
        self.softlabel_dir = softlabel_dir or (
            _default_sl if os.path.isdir(_default_sl) else None)
        self.allowed_sample_ids = self._load_allowed_sample_ids(manifest_csv)
        self.samples   = self._load_pairs(img_dir, mask_dir, self.allowed_sample_ids)
        if not self.samples:
            raise RuntimeError(
                f"[Dataset] No tile pairs found.\n"
                f"  images: {img_dir}\n  masks:  {mask_dir}\n"
                f"Run prepare_data.py first."
            )
        # Check if LCI maps exist
        has_lci = False
        npy_count = 0
        if self.lci_dir and os.path.isdir(self.lci_dir):
            npy_count = len([f for f in os.listdir(self.lci_dir)
                             if f.endswith('.npy')])
            has_lci = npy_count > 0
        self.strict_lci = has_lci
        lci_status = f"LCI=on ({npy_count} maps)" if has_lci else "LCI=off (weight=1.0)"
        # Check softlabels
        has_sl = False
        sl_count = 0
        if self.softlabel_dir and os.path.isdir(self.softlabel_dir):
            sl_count = len([f for f in os.listdir(self.softlabel_dir)
                            if f.endswith('.npy')])
            has_sl = sl_count > 0
        self.strict_softlabels = has_sl
        sl_status = f"SoftLabel=on ({sl_count} files)" if has_sl else "SoftLabel=off (one-hot)"
        print(f"[Dataset] {len(self.samples)} tile pairs  "
              f"channels={config.IN_CHANNELS}  {lci_status}  {sl_status}")

    @staticmethod
    def _load_allowed_sample_ids(manifest_csv: str):
        if not manifest_csv:
            return None
        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
        ids = set()
        with open(manifest_csv, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sid = str(row.get("sample_id", "")).strip()
                if sid:
                    ids.add(sid)
        if not ids:
            raise RuntimeError(f"No sample_id found in manifest: {manifest_csv}")
        print(f"[Dataset] manifest filter: {manifest_csv} | sample_ids={len(ids)}")
        return ids

    @staticmethod
    def _load_pairs(img_dir: str, mask_dir: str, allowed_sample_ids=None):
        pairs = []
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.tif"))):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            sample_id = stem.split("_")[0]
            if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
                continue
            mask_path = os.path.join(mask_dir, stem + ".png")
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
        return pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = self.read_tif(img_path)
        mask  = load_mask_png(mask_path)

        # Load LCI weight map
        stem    = os.path.splitext(os.path.basename(img_path))[0]
        lci_map = self._load_lci(stem, mask.shape)
        raw_soft = self._load_softlabel(stem, mask.shape)
        return _prepare_sample(
            image=image,
            mask=mask,
            lci_map=lci_map,
            raw_soft=raw_soft,
            geom_transform=self.geom_transform,
            image_transform=self.transform,
        )

    def _load_softlabel(self, stem: str, spatial_shape: tuple) -> np.ndarray:
        """
        Load soft label (NUM_CLASSES+1, H, W) float32.
        If SoftLabel is enabled, missing or malformed files are fatal.
        """
        if self.softlabel_dir:
            npy_path = os.path.join(self.softlabel_dir, stem + ".npy")
            if os.path.exists(npy_path):
                sl = np.load(npy_path).astype(np.float32)
                expected_shape = (config.NUM_CLASSES + 1, *spatial_shape)
                if sl.shape == expected_shape:
                    return sl
                if self.strict_softlabels:
                    raise RuntimeError(
                        f"Softlabel shape mismatch for tile {stem}: "
                        f"{sl.shape} != {expected_shape} ({npy_path})"
                    )
            elif self.strict_softlabels:
                raise FileNotFoundError(
                    f"Softlabel file missing for tile {stem}: {npy_path}. "
                    "SoftLabel is enabled because the directory contains .npy files."
                )
        # SoftLabel is globally disabled: caller will use hard-label loss.
        # Return None so caller knows to treat this as hard-label-only.
        # (train.py will pass soft_targets=None to loss → Focal path used)
        return None

    def _load_lci(self, stem: str, shape: tuple) -> np.ndarray:
        """Load LCI weight map. If LCI is enabled, missing maps are fatal."""
        if self.lci_dir:
            npy_path = os.path.join(self.lci_dir, stem + ".npy")
            if os.path.exists(npy_path):
                lci = np.load(npy_path).astype(np.float32)
                if lci.shape == shape:
                    return lci
                if self.strict_lci:
                    raise RuntimeError(
                        f"LCI shape mismatch for tile {stem}: "
                        f"{lci.shape} != {shape} ({npy_path})"
                    )
            elif self.strict_lci:
                raise FileNotFoundError(
                    f"LCI file missing for tile {stem}: {npy_path}. "
                    "LCI is enabled because the directory contains .npy files."
                )
        # LCI is globally disabled.
        # Unlabeled pixels are ignored by IGNORE_INDEX anyway.
        return np.ones(shape, dtype=np.float32)

    @staticmethod
    def read_tif(path: str) -> np.ndarray:
        """
        Read a tile TIF:
          - Tiles from prepare_data.py already contain indices and can be read directly.
          - Append indices to raw TIF inputs used for inference.
        Return (H, W, IN_CHANNELS) float32.
        """
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)   # (C, H, W)
            tags = src.tags()

        raw_ch = config.RAW_CHANNELS
        total_ch = config.IN_CHANNELS

        if data.shape[0] == raw_ch:
            data = sr_to_feature_stack(data, source=path)
        elif data.shape[0] != total_ch:
            raise RuntimeError(
                f"Unexpected channel count for {path}: got {data.shape[0]}, "
                f"expected raw={raw_ch} or prepared={total_ch}. "
                "Refuse to pad/truncate channels because that would create virtual input values."
            )
        else:
            assert_prepared_tile_tags(tags, source=path)

        return data.transpose(1, 2, 0)   # (H, W, C)


# -----------------------------------------------------------------
# DataLoader factory
# -----------------------------------------------------------------
class _SubDataset(Dataset):
    """Assign separate transforms to the training and validation splits."""
    def __init__(self, subset, image_transform, geom_transform=None):
        self.subset = subset
        self.image_transform = image_transform
        self.geom_transform = geom_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, mask_path = \
            self.subset.dataset.samples[self.subset.indices[idx]]
        image = SatelliteDataset.read_tif(img_path)
        mask  = load_mask_png(mask_path)
        stem  = os.path.splitext(os.path.basename(img_path))[0]
        lci_map = self.subset.dataset._load_lci(stem, mask.shape)
        raw_soft = self.subset.dataset._load_softlabel(stem, mask.shape)
        return _prepare_sample(
            image=image,
            mask=mask,
            lci_map=lci_map,
            raw_soft=raw_soft,
            geom_transform=self.geom_transform,
            image_transform=self.image_transform,
        )

def build_loaders(img_dir:        str   = config.TRAIN_IMG_DIR,
                  mask_dir:       str   = config.TRAIN_MASK_DIR,
                  lci_dir:        str   = None,
                  softlabel_dir:  str   = None,
                  manifest_csv:   str   = None,
                  val_split: float = config.VAL_SPLIT,
                  seed:      int   = config.SEED):
    """
    lci_dir: path to LCI weight maps (.npy files).
             Defaults to config.TRAIN_LCI_DIR.
             If None or directory not found, LCI defaults to 1.0 (no weighting).
    """
    full_ds = SatelliteDataset(img_dir, mask_dir,
                               lci_dir=lci_dir,
                               softlabel_dir=softlabel_dir,
                               manifest_csv=manifest_csv)
    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val

    g = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(full_ds, [n_train, n_val], generator=g)

    train_set = _SubDataset(
        train_sub,
        image_transform=get_train_image_transforms(),
        geom_transform=get_train_geom_transforms(),
    )
    val_set = _SubDataset(
        val_sub,
        image_transform=get_val_transforms(),
        geom_transform=get_val_geom_transforms(),
    )

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=config.NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True)

    train_mask_paths = [full_ds.samples[i][1] for i in train_sub.indices]

    print(f"[DataLoader] Train={n_train}  Val={n_val}")
    return train_loader, val_loader, train_mask_paths
