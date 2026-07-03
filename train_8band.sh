#!/bin/bash
#SBATCH --job-name=sat_seg_8band
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ygfw39@gmail.com
#SBATCH -o logfiles/8band_train.%j.log
#SBATCH -e logfiles/8band_train_err.%j.log
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

trap 'echo "[ERROR] train_8band.sh failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# Usage:
#   sbatch train_8band.sh <exp_name> [train_manifest_csv] [epochs] [resume]
#
# Each experiment uses its own code/output directory:
#   code copy      -> run_8band_<exp>/
#   outputs        -> research_outputs/<exp>/...
#   stats marker   -> logfiles/stats_8band_<exp>_<preprocessing>.txt

EXP_NAME="${1:-}"
TRAIN_MANIFEST="${2:-}"
TRAIN_EPOCHS="${3:-}"
RESUME_MODE="${4:-}"

source /dss/dsshome1/06/di35pav/miniconda3/bin/activate Treedead
PYTHON="/dss/dsshome1/06/di35pav/miniconda3/envs/Treedead/bin/python"

mkdir -p logfiles

if [ -z "$EXP_NAME" ]; then
    echo "Error: experiment name is required" >&2
    echo "Usage: sbatch train_8band.sh <exp_name>"
    exit 1
fi

CODE="./run_8band_${EXP_NAME}"
mkdir -p "$CODE"
cp ./*.py "$CODE/"

sed -i 's/^BAND_MODE = "4band"/BAND_MODE = "8band"/' "$CODE/config.py"

cd "$CODE"
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export TREEDEAD_EXP_NAME="$EXP_NAME"
TRAIN_SRUN_EXPORT_ARGS="--export=ALL,TREEDEAD_EXP_NAME=${EXP_NAME}"

echo "========================================"
echo " Job ID    : ${SLURM_JOB_ID:-NA}"
echo " Node      : ${SLURMD_NODENAME:-NA}"
echo " Start     : $(date)"
echo " BAND_MODE : 8band"
echo " EXP_NAME  : $EXP_NAME"
echo " Code dir  : $CODE"
$PYTHON - << 'PYEOF'
import config

if config.BAND_MODE != "8band":
    raise RuntimeError(f"Expected BAND_MODE='8band', got {config.BAND_MODE!r}")

print(" Data root   :", config.DATA_ROOT)
print(" Checkpoints :", config.CHECKPOINT_DIR)
PYEOF
echo "========================================"

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

CALIB_MANIFEST=$($PYTHON -c 'import os,config; print(os.path.join(config.SPLIT_DIR,"scale_analysis_manifest.csv"))' | tail -n 1)
if [ ! -f "$CALIB_MANIFEST" ]; then
    fail "[Stage-3] calibration manifest not found: $CALIB_MANIFEST"
fi
export CALIB_MANIFEST
TRAIN_SRUN_EXPORT_ARGS="${TRAIN_SRUN_EXPORT_ARGS},TRAIN_MANIFEST=${TRAIN_MANIFEST},CALIB_MANIFEST=${CALIB_MANIFEST}"

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
    srun $TRAIN_SRUN_EXPORT_ARGS $PYTHON check_data.py --fix
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
    srun $TRAIN_SRUN_EXPORT_ARGS $PYTHON reproject_shp.py
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
    srun $TRAIN_SRUN_EXPORT_ARGS $PYTHON prepare_data_softlabel.py
fi

PREPROCESSING_ID=$($PYTHON -c 'import config; print(config.INPUT_PREPROCESSING_NAME)' | tail -n 1)
STATS_JSON=$($PYTHON -c 'import config; print(config.CHANNEL_STATS_JSON)' | tail -n 1)
STATS_FILE="../logfiles/stats_8band_${EXP_NAME}_${PREPROCESSING_ID}.txt"
mkdir -p ../logfiles
if [ ! -f "$STATS_FILE" ] || [ ! -f "$STATS_JSON" ]; then
    echo "[Step 3] Computing normalization statistics..."
    srun $TRAIN_SRUN_EXPORT_ARGS $PYTHON compute_stats.py --max_samples 2000
    echo "computed $(date)" > "$STATS_FILE"
    echo "[Step 3] Stats JSON saved: $STATS_JSON"
else
    echo "[Step 3] Stats exist, skipping"
fi

echo "[Step 3b] Checking train manifest tile coverage..."
srun $TRAIN_SRUN_EXPORT_ARGS $PYTHON - << 'PYEOF'
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

echo "[Step 4] Training..."
srun $TRAIN_SRUN_EXPORT_ARGS $PYTHON train.py --model all $TRAIN_ARGS

echo "[Step 5] Calibrating..."
srun $TRAIN_SRUN_EXPORT_ARGS $PYTHON calibrate.py --model all --manifest_csv "$CALIB_MANIFEST"

echo "[Step 6] Training random forest baseline..."
srun $TRAIN_SRUN_EXPORT_ARGS $PYTHON train_random_forest.py --manifest_csv "$TRAIN_MANIFEST"

echo "========================================"
echo " Done: $(date)"
echo "========================================"
