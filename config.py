"""
Central configuration for the TreeDead framework.

Switching band mode here updates:
- input channel counts
- normalization stats
- inference tile settings
- output folders
"""

import json
import os


# -----------------------------------------------------------------
# Band mode
# -----------------------------------------------------------------
BAND_MODE = "4band"  # "4band" or "8band"
assert BAND_MODE in ("4band", "8band"), f"Invalid BAND_MODE: {BAND_MODE}"


# -----------------------------------------------------------------
# Band profiles
# -----------------------------------------------------------------
_BAND_PROFILES = {
    "4band": {
        "raw_channels": 4,
        "in_channels": 9,
        "img_mean": (0.0279, 0.0474, 0.0574, 0.2331, 0.7934, 0.6583, 0.1763, 0.2450, 0.4519),
        "img_std" : (0.0183, 0.0231, 0.0283, 0.0807, 0.0909, 0.0665, 0.0806, 0.0631, 0.0533),
        "infer_tile": 256,
        "infer_overlap": 64,
        "infer_batch": 6,
    },
    "8band": {
        "raw_channels": 8,
        "in_channels": 15,
        "img_mean": (0.3197, 0.3326, 0.3555, 0.3551, 0.3433, 0.3333, 0.3814, 0.4210, 0.5330, 0.5485, 0.4791, 0.2032, 0.5127, 0.5462, 0.1028),
        "img_std": (0.2432, 0.2475, 0.2411, 0.2411, 0.2437, 0.2494, 0.2478, 0.2526, 0.1244, 0.1936, 0.1193, 0.1343, 0.1136, 0.1475, 0.1823),
        "infer_tile": 256,
        "infer_overlap": 64,
        "infer_batch": 4,
    },
}

_p = _BAND_PROFILES[BAND_MODE]
RAW_CHANNELS = _p["raw_channels"]
IN_CHANNELS = _p["in_channels"]
IMG_MEAN = _p["img_mean"]
IMG_STD = _p["img_std"]
INFER_TILE_SIZE = _p["infer_tile"]
INFER_OVERLAP = _p["infer_overlap"]
INFER_BATCH = _p["infer_batch"]


# -----------------------------------------------------------------
# PlanetScope SR input preprocessing
# -----------------------------------------------------------------
# PlanetScope AnalyticMS_SR GeoTIFFs store surface reflectance as scaled
# integer values. Model inputs must use a fixed physical scale, not per-scene
# percentile stretching. RGB display also uses fixed reflectance scaling so
# visual products remain comparable across scenes.
INPUT_PREPROCESSING_NAME = "planetscope_sr_reflectance_v1"
PLANETSCOPE_SR_SCALE = float(os.environ.get("TREEDEAD_PLANETSCOPE_SR_SCALE", "10000.0"))
REFLECTANCE_CLIP_MIN = float(os.environ.get("TREEDEAD_REFLECTANCE_CLIP_MIN", "0.0"))
REFLECTANCE_CLIP_MAX = float(os.environ.get("TREEDEAD_REFLECTANCE_CLIP_MAX", "1.0"))
RGB_DISPLAY_GAIN = float(os.environ.get("TREEDEAD_RGB_DISPLAY_GAIN", "3.5"))
RGB_DISPLAY_GAMMA = float(os.environ.get("TREEDEAD_RGB_DISPLAY_GAMMA", "1.0"))
PREPROCESSING_TAG_KEY = "TREEDEAD_PREPROCESSING"
REFLECTANCE_SCALE_TAG_KEY = "TREEDEAD_REFLECTANCE_SCALE"
BAND_MODE_TAG_KEY = "TREEDEAD_BAND_MODE"


# -----------------------------------------------------------------
# Paths
# -----------------------------------------------------------------
_TIF_DIRS = {
    "4band": r"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/code/Wendy_code/TreeDead/AllGT/TIF/4band",
    "8band": r"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/code/Wendy_code/TreeDead/AllGT/TIF/8band",
}

RAW_TIF_DIR = os.environ.get(f"TREEDEAD_TIF_DIR_{BAND_MODE.upper()}", _TIF_DIRS[BAND_MODE])
if BAND_MODE == "4band" and not os.path.exists(RAW_TIF_DIR):
    RAW_TIF_DIR = os.environ.get(
        "TREEDEAD_TIF_DIR_4BAND",
        r"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/code/Wendy_code/TreeDead/AllGT/TIF/4band",
    )
SAMPLE_IMAGE_DIR = os.environ.get("TREEDEAD_SAMPLE_IMAGE_DIR", RAW_TIF_DIR)
RAW_SHP_DIR = r"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/code/Wendy_code/TreeDead/AllGT/allshp"
SHP_CLASS_FIELD = "Class_Int"

SAMPLE_CENTER_CSV = r"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/code/Wendy_code/TreeDead/sample/sample_centers_clean.csv"
SAMPLE_CLIMATE_SHP = r"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/code/Wendy_code/TreeDead/sample/sample_centers_thermal_zones.shp"
SAMPLE_THERMAL_SHP = SAMPLE_CLIMATE_SHP  # legacy alias
SAMPLE_TRUTH_DIR = r"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/code/Wendy_code/TreeDead/AllGT/allshp"
SAMPLE_CENTER_CRS = os.environ.get("TREEDEAD_SAMPLE_CENTER_CRS", "EPSG:4326")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_RUNTIME_ROOT = os.path.join(PROJECT_ROOT, "_local_runtime")
_CLUSTER_BASE = r"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/code/Wendy_code/TreeDead/sample"
_RUNTIME_ROOT = _CLUSTER_BASE
if os.name == "nt":
    _RUNTIME_ROOT = LOCAL_RUNTIME_ROOT

EXTERNAL_DATA_DIR = os.environ.get(
    "TREEDEAD_EXTERNAL_DATA_DIR",
    os.path.join(_RUNTIME_ROOT, "external_data"),
)
KOPPEN_PERIOD = "1984_2013"
KOPPEN_DATA_DIR = os.environ.get(
    "TREEDEAD_KOPPEN_DATA_DIR",
    os.path.join(EXTERNAL_DATA_DIR, "koppen"),
)
KOPPEN_DOWNLOAD_URL = (
    "https://zenodo.org/records/5347837/files/"
    "KGClim_V1_historical_1984_2013.zip?download=1"
)
KOPPEN_KC_URL = "https://glass.hku.hk/archive/KGClim/KGClim_V1_historical/1984_2013/kc_1984_2013.tif"
KOPPEN_CONF_URL = "https://glass.hku.hk/archive/KGClim/KGClim_V1_historical/1984_2013/kc_conf_1984_2013.tif"
KOPPEN_ZIP_PATH = os.path.join(KOPPEN_DATA_DIR, "KGClim_V1_historical_1984_2013.zip")
KOPPEN_EXTRACT_DIR = os.path.join(KOPPEN_DATA_DIR, "KGClim_V1_historical_1984_2013")
KOPPEN_KC_RASTER = os.environ.get(
    "TREEDEAD_KOPPEN_KC_RASTER",
    os.path.join(KOPPEN_DATA_DIR, "kc_1984_2013.tif"),
)
KOPPEN_CONF_RASTER = os.environ.get(
    "TREEDEAD_KOPPEN_CONF_RASTER",
    os.path.join(KOPPEN_DATA_DIR, "kc_conf_1984_2013.tif"),
)

RESEARCH_DIR = os.path.join(_RUNTIME_ROOT, "research_outputs")
EXPERIMENT_TAG = os.environ.get("TREEDEAD_EXP_NAME", "").strip()
RUN_ROOT = os.path.join(RESEARCH_DIR, EXPERIMENT_TAG) if EXPERIMENT_TAG else RESEARCH_DIR
DATA_ROOT = os.path.join(RESEARCH_DIR, f"tiles_{BAND_MODE}")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "images")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "masks")
TRAIN_LCI_DIR = os.path.join(DATA_ROOT, "lci")
OUTPUT_DIR = os.path.join(RUN_ROOT, f"inference_outputs_{BAND_MODE}")
CHECKPOINT_DIR = os.path.join(RUN_ROOT, f"checkpoints_{BAND_MODE}")
MANIFEST_DIR = os.path.join(RESEARCH_DIR, "manifest")
SAMPLE_STATS_DIR = os.path.join(RESEARCH_DIR, "sample_stats")
CHANNEL_STATS_JSON = os.environ.get(
    "TREEDEAD_CHANNEL_STATS_JSON",
    os.path.join(SAMPLE_STATS_DIR, f"channel_stats_{BAND_MODE}_{INPUT_PREPROCESSING_NAME}.json"),
)
SPLIT_DIR = os.path.join(RESEARCH_DIR, "splits")
SCALE_METRICS_DIR = os.path.join(RESEARCH_DIR, "scale_metrics")
FIGURE_DIR = os.path.join(RUN_ROOT, "figures")

RUNTIME_DIRS = (
    DATA_ROOT,
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    TRAIN_LCI_DIR,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    RESEARCH_DIR,
    RUN_ROOT,
    MANIFEST_DIR,
    SAMPLE_STATS_DIR,
    SPLIT_DIR,
    SCALE_METRICS_DIR,
    FIGURE_DIR,
    EXTERNAL_DATA_DIR,
    KOPPEN_DATA_DIR,
)


def _load_channel_stats_json(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        stats = json.load(fh)
    if stats.get("band_mode") != BAND_MODE:
        raise RuntimeError(
            f"Channel stats band_mode mismatch in {path}: "
            f"{stats.get('band_mode')!r} != {BAND_MODE!r}"
        )
    if stats.get("input_preprocessing") != INPUT_PREPROCESSING_NAME:
        raise RuntimeError(
            f"Channel stats preprocessing mismatch in {path}: "
            f"{stats.get('input_preprocessing')!r} != {INPUT_PREPROCESSING_NAME!r}"
        )
    mean = tuple(float(v) for v in stats.get("img_mean", []))
    std = tuple(float(v) for v in stats.get("img_std", []))
    if len(mean) != IN_CHANNELS or len(std) != IN_CHANNELS:
        raise RuntimeError(
            f"Channel stats length mismatch in {path}: "
            f"mean={len(mean)} std={len(std)} expected={IN_CHANNELS}"
        )
    if any(v <= 0 for v in std):
        raise RuntimeError(f"Channel stats contain non-positive std values: {path}")
    return mean, std


_LOADED_CHANNEL_STATS = _load_channel_stats_json(CHANNEL_STATS_JSON)
if _LOADED_CHANNEL_STATS is not None:
    IMG_MEAN, IMG_STD = _LOADED_CHANNEL_STATS


def require_channel_stats():
    if _LOADED_CHANNEL_STATS is None:
        raise RuntimeError(
            "Channel normalization stats for the current preprocessing were not found. "
            f"Expected: {CHANNEL_STATS_JSON}. Run compute_stats.py after regenerating tiles."
        )


def ensure_runtime_dirs(paths=None):
    for path in (paths or RUNTIME_DIRS):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Failed to create runtime directory: {path}") from exc


def set_output_dir(path: str):
    global OUTPUT_DIR, CHECK_REPORT_DIR, RUNTIME_DIRS

    old_output_dir = OUTPUT_DIR
    old_check_report_dir = globals().get("CHECK_REPORT_DIR")
    OUTPUT_DIR = path
    CHECK_REPORT_DIR = os.path.join(OUTPUT_DIR, "data_check")

    runtime_dirs = []
    for entry in RUNTIME_DIRS:
        if entry in (old_output_dir, old_check_report_dir):
            continue
        if entry not in runtime_dirs:
            runtime_dirs.append(entry)
    for entry in (OUTPUT_DIR, CHECK_REPORT_DIR):
        if entry not in runtime_dirs:
            runtime_dirs.append(entry)
    RUNTIME_DIRS = tuple(runtime_dirs)


def print_config_summary():
    print(f"[config] BAND_MODE={BAND_MODE}  RAW={RAW_CHANNELS}ch  IN={IN_CHANNELS}ch")
    print(f"[config] INPUT_PREPROCESSING -> {INPUT_PREPROCESSING_NAME}  SR_SCALE={PLANETSCOPE_SR_SCALE:g}")
    print(f"[config] DATA_ROOT -> {DATA_ROOT}")
    print(f"[config] CHECKPOINT_DIR -> {CHECKPOINT_DIR}")
    print(f"[config] OUTPUT_DIR -> {OUTPUT_DIR}")
    print(f"[config] CHANNEL_STATS_JSON -> {CHANNEL_STATS_JSON}")
    print(f"[config] CHANNEL_STATS_LOADED -> {_LOADED_CHANNEL_STATS is not None}")
    print(f"[config] KOPPEN_DATA_DIR -> {KOPPEN_DATA_DIR}")
    print(f"[config] KOPPEN_KC_RASTER -> {KOPPEN_KC_RASTER}")
    print(f"[config] KOPPEN_CONF_RASTER -> {KOPPEN_CONF_RASTER}")
    print(f"[config] Supported models -> {', '.join(SUPPORTED_MODELS)}")


def ensure_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create directory: {path}") from exc
    return path


def figure_subdir(*parts: str) -> str:
    path = os.path.join(FIGURE_DIR, *[str(part) for part in parts if str(part)])
    return ensure_dir(path)


# -----------------------------------------------------------------
# Classes
# -----------------------------------------------------------------
NUM_CLASSES = 5
IGNORE_INDEX = 0
GSD = 3.0
DEAD_PATCH_MIN_AREA_M2 = 9.0

# Sample-level remote-sensing background estimation.
RS_NDVI_VEG_THRESHOLD = float(os.environ.get("TREEDEAD_RS_NDVI_VEG_THRESHOLD", "0.25"))
RS_NDVI_BARE_THRESHOLD = float(os.environ.get("TREEDEAD_RS_NDVI_BARE_THRESHOLD", "0.15"))
RS_NDWI_WATER_THRESHOLD = float(os.environ.get("TREEDEAD_RS_NDWI_WATER_THRESHOLD", "0.10"))
RS_WATER_FRACTION_THRESHOLD = float(os.environ.get("TREEDEAD_RS_WATER_FRACTION_THRESHOLD", "0.10"))
RS_SPARSE_VEG_FRACTION_THRESHOLD = float(os.environ.get("TREEDEAD_RS_SPARSE_VEG_FRACTION_THRESHOLD", "0.20"))
RS_DENSE_VEG_FRACTION_THRESHOLD = float(os.environ.get("TREEDEAD_RS_DENSE_VEG_FRACTION_THRESHOLD", "0.55"))

# Dead-patch size bins in PlanetScope 3 m pixels.
PATCH_SMALL_MAX_M2 = float(os.environ.get("TREEDEAD_PATCH_SMALL_MAX_M2", str(3 * GSD * GSD)))
PATCH_MEDIUM_MAX_M2 = float(os.environ.get("TREEDEAD_PATCH_MEDIUM_MAX_M2", str(20 * GSD * GSD)))

CLASS_NAMES = {
    1: "Dead_trees",
    2: "Living_trees",
    3: "Roads_Bare_Buildings",
    4: "Lakes_Rivers",
    5: "Grassland",
}

PALETTE = {
    0: (50, 50, 50),
    1: (180, 60, 60),
    2: (34, 139, 34),
    3: (210, 180, 140),
    4: (30, 144, 255),
    5: (154, 205, 50),
}


# -----------------------------------------------------------------
# Tile settings
# -----------------------------------------------------------------
TILE_SIZE = 64
TILE_STRIDE = 32
MIN_LABEL_RATIO = 0.05


# -----------------------------------------------------------------
# Loss weighting
# -----------------------------------------------------------------
CLASS_WEIGHTS = [
    1.0,
    2.0,
    2.5,
    4.0,
    3.0,
]


# -----------------------------------------------------------------
# Post-processing
# -----------------------------------------------------------------
NODATA_THRESHOLD = 0.02
MIN_OBJ_AREA_PX = 3
CONF_THRESHOLD = 0.40


# -----------------------------------------------------------------
# Training hyperparameters
# -----------------------------------------------------------------
BATCH_SIZE = 16
NUM_EPOCHS = 120
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
VAL_SPLIT = 0.15
SEED = 42


# -----------------------------------------------------------------
# Active model registry
# -----------------------------------------------------------------
DEEP_MODELS = (
    "unet",
    "unet_resnet34",
    "unet_convnext_tiny",
    "hrnet_w18",
    "segformer",
    "segformer_b4",
    "segformer_b5",
    "r50_upernet",
)
TRADITIONAL_MODELS = (
    "random_forest",
)
SUPPORTED_MODELS = DEEP_MODELS + TRADITIONAL_MODELS

DEFAULT_MODEL = "segformer_b4"
SMALL_INPUT_PRIORITY = (
    "hrnet_w18",
    "unet_convnext_tiny",
    "unet_resnet34",
    "segformer_b4",
    "segformer",
)


# -----------------------------------------------------------------
# Legacy compatibility
# -----------------------------------------------------------------
# These values are kept only because legacy model definitions still exist
# in models.py, even though those models are no longer part of the active
# framework registry.
DINOV2_MODEL = "dinov2_vitl14"
DINOV2_EMBED_DIM = 1024
DINOV2_PATCH_SIZE = 14
DINOV2_TILE_SIZE = 224
DINOV2_MAX_INFER_SIZE = 448
DINOV2_DECODE_DIM = 256
DINOV2_FROZEN_BACKBONE = True
DINOV2_BACKBONE_LR = 1e-5
DINOV2_HEAD_LR = 1e-4
DINOV2_BATCH_SIZE = 8


# -----------------------------------------------------------------
# Data-quality checks
# -----------------------------------------------------------------
ANOMALY_SINGLE_VALUE_RATIO = 0.80
ANOMALY_NODATA_WARN_RATIO = 0.30
ANOMALY_MIN_DYNAMIC_RANGE = 1e-4

QUALITY_MIN_POLYGON_PIXELS = 1.0
QUALITY_MIN_COVERAGE = 0.03

CONFLICT_OVERLAP_RATIO = 0.01
CONFLICT_SAME_CLASS_WARN = 5

MIXED_RATIO_OPTIONAL = 0.05
MIXED_RATIO_RECOMMEND = 0.20

SOFT_LABEL_SUBGRID = 5
BOUNDARY_STRATEGY = "soft"
SOFT_LABEL_MIN_RATIO = 0.05
SOFT_LABEL_CLASSES = [1, 2]

LCI_SAMPLE_SIZE = 500
CHECK_REPORT_DIR = os.path.join(OUTPUT_DIR, "data_check")
RUNTIME_DIRS = RUNTIME_DIRS + (CHECK_REPORT_DIR,)


# -----------------------------------------------------------------
# Spectral separability
# -----------------------------------------------------------------
SPECTRAL_SEPARABILITY = {
    (1, 2): 0.25,
    (1, 3): 0.70,
    (1, 4): 0.95,
    (1, 5): 0.85,
    (2, 3): 0.75,
    (2, 4): 0.95,
    (2, 5): 0.65,
    (3, 4): 0.85,
    (3, 5): 0.70,
    (4, 5): 0.90,
}


def get_separability(cls_i: int, cls_j: int) -> float:
    if cls_i == cls_j:
        return 1.0
    key = (min(cls_i, cls_j), max(cls_i, cls_j))
    return SPECTRAL_SEPARABILITY.get(key, 0.70)


def get_local_subgrid(cls_i: int, cls_j: int, base: int = None) -> int:
    import importlib

    if base is None:
        cfg = importlib.import_module("config")
        base = getattr(cfg, "SOFT_LABEL_SUBGRID", 5)
    sep = get_separability(cls_i, cls_j)
    return max(base, int(base / max(sep, 0.1)))


# -----------------------------------------------------------------
# Temperature scaling
# -----------------------------------------------------------------
TEMPERATURE_INIT = 1.0
TEMPERATURE_LR = 0.01
TEMPERATURE_MAX_ITER = 50
CONFIDENCE_LOW_THRESHOLD = 0.30
CONFIDENCE_HIGH_THRESHOLD = 0.70
