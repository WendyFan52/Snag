#!/bin/bash
#SBATCH --job-name=sat_seg_4band
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=64:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ygfw39@gmail.com
#SBATCH -o logfiles/4band_train.%j.log
#SBATCH -e logfiles/4band_train_err.%j.log
#SBATCH -D .
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --account=pn39wa-c
#SBATCH --export=NONE

set -Eeuo pipefail

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

trap 'echo "[ERROR] train_4band.sh failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# Usage:
#   sbatch train_4band.sh <exp_name> [train_manifest_csv] [epochs|mode] [resume] [mode]
#
# Modes:
#   all       : train deep models, calibrate them, then train random forest baseline
#   deep_only : train/calibrate deep models only
#   rf_only  : skip deep training/calibration and train only best_random_forest.joblib
#
# RF-only examples:
#   sbatch train_4band.sh exp_stage3 /path/to/train_manifest.csv rf_only
#   TREEDEAD_TRAIN_MODE=rf_only sbatch train_4band.sh exp_stage3 /path/to/train_manifest.csv
#
# Each experiment uses its own code/output directory:
#   code copy      -> run_4band_<exp>/
#   outputs        -> research_outputs/<exp>/...
#   stats marker   -> logfiles/stats_4band_<exp>_<preprocessing>.txt

EXP_NAME="${1:-}"
TRAIN_MANIFEST="${2:-}"
ARG3="${3:-}"
ARG4="${4:-}"
ARG5="${5:-}"
TRAIN_EPOCHS=""
RESUME_MODE=""
TRAIN_MODE="${TREEDEAD_TRAIN_MODE:-all}"

case "$ARG3" in
    all|deep_only|rf_only)
        TRAIN_MODE="$ARG3"
        RESUME_MODE="$ARG4"
        ;;
    "")
        if [ -n "$ARG5" ]; then
            TRAIN_MODE="$ARG5"
        fi
        ;;
    *)
        TRAIN_EPOCHS="$ARG3"
        RESUME_MODE="$ARG4"
        if [ -n "$ARG5" ]; then
            TRAIN_MODE="$ARG5"
        fi
        ;;
esac

case "$TRAIN_MODE" in
    all|deep_only|rf_only)
        ;;
    *)
        fail "Invalid TRAIN_MODE=$TRAIN_MODE. Expected one of: all, deep_only, rf_only"
        ;;
esac

source /dss/dsshome1/06/di35pav/miniconda3/bin/activate Treedead
PYTHON="/dss/dsshome1/06/di35pav/miniconda3/envs/Treedead/bin/python"

export SLURM_EXPORT_ENV=ALL

mkdir -p logfiles

if [ -z "$EXP_NAME" ]; then
    echo "Error: experiment name is required" >&2
    echo "Usage: sbatch train_4band.sh <exp_name>"
    exit 1
fi

CODE="./run_4band_${EXP_NAME}"
mkdir -p "$CODE"
cp ./*.py "$CODE/"

sed -i 's/^BAND_MODE = "8band"/BAND_MODE = "4band"/' "$CODE/config.py"

cd "$CODE"
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export TREEDEAD_EXP_NAME="$EXP_NAME"

echo "========================================"
echo " Job ID    : ${SLURM_JOB_ID:-NA}"
echo " Node      : ${SLURMD_NODENAME:-NA}"
echo " Start     : $(date)"
echo " BAND_MODE : 4band"
echo " EXP_NAME  : $EXP_NAME"
echo " MODE      : $TRAIN_MODE"
echo " Code dir  : $CODE"
$PYTHON - << 'PYEOF'
import config

if config.BAND_MODE != "4band":
    raise RuntimeError(f"Expected BAND_MODE='4band', got {config.BAND_MODE!r}")

print(" Data root   :", config.DATA_ROOT)
print(" Checkpoints :", config.CHECKPOINT_DIR)
PYEOF
echo "========================================"

if [ "$TRAIN_MODE" != "deep_only" ]; then
    echo "[RF] Checking random forest dependencies..."
    $PYTHON - << 'PYEOF'
missing = []
for name in ("joblib", "sklearn"):
    try:
        __import__(name)
    except ImportError:
        missing.append(name)
if missing:
    raise ImportError(
        "Missing random forest dependency/dependencies in the active Treedead environment: "
        + ", ".join(missing)
        + ". Install with: python -m pip install scikit-learn joblib"
    )
print("[RF] dependencies ok: joblib, sklearn")
PYEOF
fi

if [ -z "$TRAIN_MANIFEST" ]; then
    TRAIN_MANIFEST=$($PYTHON -c 'import os,config; print(os.path.join(config.SPLIT_DIR,"train_manifest.csv"))' | tail -n 1)
fi

TRAIN_ARGS=""
if [ -n "$TRAIN_MANIFEST" ] && [ -f "$TRAIN_MANIFEST" ]; then
    echo "[Stage-3] Use manifest: $TRAIN_MANIFEST"
    TRAIN_ARGS="--manifest_csv $TRAIN_MANIFEST"
else
    fail "[Stage-3] train manifest not found: $TRAIN_MANIFEST. Refusing to fall back to full-tile training."
fi
export TRAIN_MANIFEST

CALIB_MANIFEST=""
if [ "$TRAIN_MODE" != "rf_only" ]; then
    CALIB_MANIFEST=$($PYTHON -c 'import os,config; print(os.path.join(config.SPLIT_DIR,"scale_analysis_manifest.csv"))' | tail -n 1)
    if [ ! -f "$CALIB_MANIFEST" ]; then
        fail "[Stage-3] calibration manifest not found: $CALIB_MANIFEST"
    fi
else
    echo "[Stage-3] RF-only mode: calibration manifest is not required"
fi
export CALIB_MANIFEST

if [ -n "$TRAIN_EPOCHS" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --epochs $TRAIN_EPOCHS"
fi
if [ "$RESUME_MODE" = "resume" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --resume"
fi

REPORT_FILE=$($PYTHON -c \
    'import config,os; print(os.path.join(config.CHECK_REPORT_DIR,"data-quality-report.txt"))' | tail -n 1)

if [ -f "$REPORT_FILE" ]; then
    echo "[Step 1] Quality report exists, skipping"
else
    echo "[Step 1] Data quality check..."
    srun $PYTHON check_data.py --fix
fi

CRS_N=$($PYTHON - 2>/dev/null << 'PYEOF'
import os, config
report = os.path.join(config.CHECK_REPORT_DIR, 'data-quality-report.txt')
print(open(report).read().count('SHP=EPSG:4326') if os.path.exists(report) else 0)
PYEOF
)
CRS_N=$(printf '%s\n' "$CRS_N" | awk '/^[0-9]+$/ {v=$0} END {print v}')
if [ "${CRS_N:-0}" -gt 0 ]; then
    echo "[Step 1] $CRS_N CRS mismatches -> fixing"
    srun $PYTHON reproject_shp.py
fi

echo "[Step 2] Checking tiles..."
if ! $PYTHON - << 'PYEOF'
import os, sys, config
from input_preprocessing import assert_prepared_tile_tags
import rasterio

n_img = len([f for f in os.listdir(config.TRAIN_IMG_DIR) if f.endswith('.tif')]) if os.path.isdir(config.TRAIN_IMG_DIR) else 0
soft_d = os.path.join(config.DATA_ROOT, 'softlabels')
lci_d = getattr(config, 'TRAIN_LCI_DIR', os.path.join(config.DATA_ROOT, 'lci'))
n_soft = len([f for f in os.listdir(soft_d) if f.endswith('.npy')]) if os.path.isdir(soft_d) else 0
n_lci = len([f for f in os.listdir(lci_d) if f.endswith('.npy')]) if os.path.isdir(lci_d) else 0
print(f"Images={n_img} Softlabels={n_soft} LCI={n_lci}")
if not (n_img > 0 and n_soft > 0 and n_lci > 0):
    sys.exit(1)
first = sorted(f for f in os.listdir(config.TRAIN_IMG_DIR) if f.endswith(".tif"))[0]
first_path = os.path.join(config.TRAIN_IMG_DIR, first)
with rasterio.open(first_path) as src:
    assert_prepared_tile_tags(src.tags(), source=first_path)
sys.exit(0)
PYEOF
then
    echo "[Step 2] Generating soft-label tiles..."
    srun $PYTHON prepare_data_softlabel.py
fi

PREPROCESSING_ID=$($PYTHON -c 'import config; print(config.INPUT_PREPROCESSING_NAME)' | tail -n 1)
STATS_JSON=$($PYTHON -c 'import config; print(config.CHANNEL_STATS_JSON)' | tail -n 1)
STATS_FILE="../logfiles/stats_4band_${EXP_NAME}_${PREPROCESSING_ID}.txt"
mkdir -p ../logfiles
if [ ! -f "$STATS_FILE" ] || [ ! -f "$STATS_JSON" ]; then
    echo "[Step 3] Computing normalization statistics..."
    srun $PYTHON compute_stats.py --max_samples 2000
    echo "computed $(date)" > "$STATS_FILE"
    echo "[Step 3] Stats JSON saved: $STATS_JSON"
else
    echo "[Step 3] Stats exist, skipping"
fi

echo "[Step 3b] Checking train manifest tile coverage..."
srun $PYTHON - << 'PYEOF'
import csv
import glob
import os

import config

manifest = os.environ["TRAIN_MANIFEST"]
sample_ids = set()
with open(manifest, "r", encoding="utf-8-sig", newline="") as fh:
    for row in csv.DictReader(fh):
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            sample_ids.add(sample_id)

tile_paths = sorted(glob.glob(os.path.join(config.TRAIN_IMG_DIR, "*.tif")))
matched_tiles = 0
for tile_path in tile_paths:
    stem = os.path.splitext(os.path.basename(tile_path))[0]
    sample_id = stem.split("_")[0]
    if sample_id in sample_ids:
        matched_tiles += 1

print(f"[Stage-3] train manifest sample_ids={len(sample_ids)} matched_tiles={matched_tiles}")
if not sample_ids:
    raise RuntimeError(f"No sample_id values found in train manifest: {manifest}")
if matched_tiles <= 0:
    raise RuntimeError(
        "No training tiles matched train manifest sample_id values. "
        f"manifest={manifest} tile_dir={config.TRAIN_IMG_DIR}"
    )
PYEOF

if [ "$TRAIN_MODE" = "rf_only" ]; then
    echo "[Step 4] RF-only mode: skipping deep model training and calibration"
    echo "[Step 4] Training random forest baseline..."
    srun $PYTHON train_random_forest.py --manifest_csv "$TRAIN_MANIFEST"

    echo "========================================"
    echo " Done: $(date)"
    echo "========================================"
    exit 0
fi

echo "[Step 4] Training..."
srun $PYTHON train.py --model all $TRAIN_ARGS

echo "[Step 5] Calibrating..."
srun $PYTHON calibrate.py --model all --manifest_csv "$CALIB_MANIFEST"

if [ "$TRAIN_MODE" = "deep_only" ]; then
    echo "[Step 6] Deep-only mode: skipping random forest baseline"
else
    echo "[Step 6] Training random forest baseline..."
    srun $PYTHON train_random_forest.py --manifest_csv "$TRAIN_MANIFEST"
fi

echo "========================================"
echo " Done: $(date)"
echo "========================================"
