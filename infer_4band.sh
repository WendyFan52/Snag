#!/bin/bash
#SBATCH --job-name=sat_infer_4band
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ygfw39@gmail.com
#SBATCH -o logfiles/4band_infer.%j.log
#SBATCH -e logfiles/4band_infer_err.%j.log
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

require_csv_rows() {
    local csv_path="$1"
    local label="$2"
    if [ ! -s "$csv_path" ]; then
        fail "$label missing or empty: $csv_path"
    fi
    $PYTHON - "$csv_path" "$label" << 'PYEOF'
import csv
import sys

csv_path = sys.argv[1]
label = sys.argv[2]
with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
    reader = csv.DictReader(fh)
    rows = list(reader)
if reader.fieldnames is None:
    raise RuntimeError(f"{label} is not a valid CSV with a header: {csv_path}")
if not rows:
    raise RuntimeError(f"{label} has no data rows: {csv_path}")
print(f"[Stage-3] {label}: rows={len(rows)}")
PYEOF
}

trap 'echo "[ERROR] infer_4band.sh failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# Usage:
#   sbatch infer_4band.sh /path/to/image.tif
#   sbatch infer_4band.sh /path/to/image.tif segformer_b4
#   sbatch infer_4band.sh NONE all /path/to/tif_dir
#   sbatch infer_4band.sh NONE all /path/to/tif_dir analysis /path/to/scale_analysis_manifest.csv
#   sbatch infer_4band.sh NONE all /path/to/tif_dir holdout  /path/to/holdout_test_manifest.csv
#   sbatch infer_4band.sh NONE all /path/to/tif_dir analysis /path/to/scale_analysis_manifest.csv exp_stage3
#
# Available models:
#   unet
#   unet_resnet34
#   unet_convnext_tiny
#   hrnet_w18
#   segformer
#   segformer_b4
#   segformer_b5
#   r50_upernet
#   random_forest
#   all

INFER_IMAGE="${1:-}"
INFER_MODEL="${2:-all}"
INFER_DIR="${3:-}"
SPLIT_TAG="${4:-analysis}"          # analysis | holdout | train | custom
EVAL_MANIFEST="${5:-}"
EXP_NAME="${6:-}"

case "$INFER_IMAGE" in
    NONE|none|-)
        INFER_IMAGE=""
        ;;
esac

if [[ "$SPLIT_TAG" == *"/"* || "$SPLIT_TAG" == *".csv" ]]; then
    fail "Invalid SPLIT_TAG='$SPLIT_TAG'. Did you omit the split tag argument? Use: sbatch infer_4band.sh NONE <model> <tif_dir> holdout <manifest_csv> <exp_name>"
fi

case "$INFER_MODEL" in
    unet|unet_resnet34|unet_convnext_tiny|hrnet_w18|segformer|segformer_b4|segformer_b5|r50_upernet|random_forest|all)
        ;;
    *)
        fail "Unsupported model: $INFER_MODEL"
        ;;
esac

source /dss/dsshome1/06/di35pav/miniconda3/bin/activate Treedead
PYTHON="/dss/dsshome1/06/di35pav/miniconda3/envs/Treedead/bin/python"

mkdir -p logfiles

CODE="./run_4band${EXP_NAME:+_$EXP_NAME}"
mkdir -p "$CODE"
cp ./*.py "$CODE/"
sed -i 's/^BAND_MODE = "8band"/BAND_MODE = "4band"/' "$CODE/config.py"
cd "$CODE"
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
if [ -n "$EXP_NAME" ]; then
    export TREEDEAD_EXP_NAME="$EXP_NAME"
fi

echo "========================================"
echo " Job ID    : ${SLURM_JOB_ID:-NA}"
echo " Node      : ${SLURMD_NODENAME:-NA}"
echo " Start     : $(date)"
echo " Model     : $INFER_MODEL"
echo " BAND_MODE : 4band"
echo " SplitTag  : $SPLIT_TAG"
echo " EXP_NAME  : ${EXP_NAME:-default}"
$PYTHON - << 'PYEOF'
import config

if config.BAND_MODE != "4band":
    raise RuntimeError(f"Expected BAND_MODE='4band', got {config.BAND_MODE!r}")

print(" Output dir:", config.OUTPUT_DIR)
PYEOF
echo "========================================"

if [ -z "$EVAL_MANIFEST" ]; then
    case "$SPLIT_TAG" in
        holdout)
            EVAL_MANIFEST=$($PYTHON -c 'import os,config; print(os.path.join(config.SPLIT_DIR,"holdout_test_manifest.csv"))' | tail -n 1)
            ;;
        train)
            EVAL_MANIFEST=$($PYTHON -c 'import os,config; print(os.path.join(config.SPLIT_DIR,"train_manifest.csv"))' | tail -n 1)
            ;;
        analysis|*)
            EVAL_MANIFEST=$($PYTHON -c 'import os,config; print(os.path.join(config.SPLIT_DIR,"scale_analysis_manifest.csv"))' | tail -n 1)
            ;;
    esac
fi
export EVAL_MANIFEST
export INFER_MODEL
SRUN_EXPORT_ARGS="--export=ALL,INFER_MODEL=${INFER_MODEL},EVAL_MANIFEST=${EVAL_MANIFEST}"
if [ -n "$EXP_NAME" ]; then
    SRUN_EXPORT_ARGS="${SRUN_EXPORT_ARGS},TREEDEAD_EXP_NAME=${EXP_NAME}"
fi

OUTPUT_ROOT=$($PYTHON -c 'import config; print(config.OUTPUT_DIR)' | tail -n 1)
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${SPLIT_TAG}"
mkdir -p "$RUN_OUTPUT_DIR"

EVAL_ARGS=""
if [ -n "$EVAL_MANIFEST" ] && [ -f "$EVAL_MANIFEST" ]; then
    echo "[Stage-3] Evaluation manifest: $EVAL_MANIFEST"
    EVAL_ARGS="--manifest_csv $EVAL_MANIFEST"
else
    fail "[Stage-3] evaluation manifest not found: $EVAL_MANIFEST. Refusing to fall back to full-tile evaluation."
fi

if [ -n "$INFER_IMAGE" ] && [ ! -f "$INFER_IMAGE" ]; then
    fail "Inference image not found: $INFER_IMAGE"
fi
if [ -z "$INFER_IMAGE" ] && [ -n "$INFER_DIR" ] && [ ! -d "$INFER_DIR" ]; then
    fail "Inference directory not found: $INFER_DIR"
fi

echo "[Stage-3] Checking checkpoints..."
srun $SRUN_EXPORT_ARGS $PYTHON - << 'PYEOF'
import os

import config

requested = os.environ["INFER_MODEL"]
models = list(config.SUPPORTED_MODELS) if requested == "all" else [requested]
deep_models = set(config.DEEP_MODELS)
traditional_models = set(config.TRADITIONAL_MODELS)
unsupported = [m for m in models if m not in deep_models and m not in traditional_models]
if unsupported:
    raise RuntimeError(f"Unsupported model(s): {unsupported}")
missing = [
    model_name
    for model_name in models
    if model_name in deep_models
    if not os.path.exists(os.path.join(config.CHECKPOINT_DIR, f"best_{model_name}.pth"))
]
if missing:
    raise FileNotFoundError(
        "Missing checkpoint(s), refusing partial inference: "
        + ", ".join(missing)
        + f" in {config.CHECKPOINT_DIR}"
    )
missing_rf = [
    model_name
    for model_name in models
    if model_name == "random_forest"
    if not os.path.exists(os.path.join(config.CHECKPOINT_DIR, "best_random_forest.joblib"))
]
if missing_rf:
    raise FileNotFoundError(
        "Missing random forest checkpoint, refusing partial inference: "
        + os.path.join(config.CHECKPOINT_DIR, "best_random_forest.joblib")
    )
missing_temperature = [
    model_name
    for model_name in models
    if model_name in deep_models
    if not os.path.exists(os.path.join(config.CHECKPOINT_DIR, f"temperature-{model_name}.json"))
]
if missing_temperature:
    raise FileNotFoundError(
        "Missing temperature calibration file(s), refusing implicit T=1.0 fallback: "
        + ", ".join(missing_temperature)
        + f" in {config.CHECKPOINT_DIR}. Run calibrate.py first."
    )
print(f"[Stage-3] checkpoint/calibration models={len(models)} ok")
PYEOF

if [ -n "$INFER_IMAGE" ]; then
    srun $SRUN_EXPORT_ARGS $PYTHON inference.py --model "$INFER_MODEL" --image "$INFER_IMAGE" --output_dir "$RUN_OUTPUT_DIR"
elif [ -n "$INFER_DIR" ]; then
    TIF_LIST_FILE=""
    if [ -n "$EVAL_MANIFEST" ] && [ -f "$EVAL_MANIFEST" ]; then
        TIF_LIST_FILE=$(mktemp)
        export INFER_DIR EVAL_MANIFEST TIF_LIST_FILE
        $PYTHON - << 'PYEOF'
import csv
import glob
import os

manifest = os.environ["EVAL_MANIFEST"]
infer_dir = os.environ["INFER_DIR"]
out_path = os.environ["TIF_LIST_FILE"]

sample_ids = set()
with open(manifest, "r", encoding="utf-8-sig", newline="") as fh:
    for row in csv.DictReader(fh):
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            sample_ids.add(sample_id)

matches = []
for tif_path in sorted(glob.glob(os.path.join(infer_dir, "*.tif"))):
    stem = os.path.splitext(os.path.basename(tif_path))[0]
    sample_id = stem.split("_")[0]
    if sample_id in sample_ids:
        matches.append(tif_path)

with open(out_path, "w", encoding="utf-8") as fh:
    for tif_path in matches:
        fh.write(tif_path + "\n")

print(f"[Stage-3] manifest sample_ids={len(sample_ids)} matched_tifs={len(matches)}")
PYEOF
        if [ ! -s "$TIF_LIST_FILE" ]; then
            echo "[Stage-3] No tif matched manifest sample_ids in: $INFER_DIR" >&2
            rm -f "$TIF_LIST_FILE"
            exit 1
        fi
    fi

    if [ -n "$TIF_LIST_FILE" ]; then
        while IFS= read -r tif <&3; do
            echo "-> $tif"
            srun --input=none $SRUN_EXPORT_ARGS $PYTHON inference.py --model "$INFER_MODEL" --image "$tif" --output_dir "$RUN_OUTPUT_DIR"
        done 3< "$TIF_LIST_FILE"
        rm -f "$TIF_LIST_FILE"
    else
        for tif in "$INFER_DIR"/*.tif; do
            if [ ! -f "$tif" ]; then
                fail "No .tif files found in inference directory: $INFER_DIR"
            fi
            echo "-> $tif"
            srun --input=none $SRUN_EXPORT_ARGS $PYTHON inference.py --model "$INFER_MODEL" --image "$tif" --output_dir "$RUN_OUTPUT_DIR"
        done
    fi
else
    echo "Usage: sbatch infer_4band.sh <image.tif> [model]"
    echo "   or: sbatch infer_4band.sh \"\" all <tif_dir>"
    exit 1
fi

if ! compgen -G "${RUN_OUTPUT_DIR}/*-all-probs.npz" > /dev/null; then
    fail "No probability npz generated in: $RUN_OUTPUT_DIR"
fi

echo ""
echo "[Evaluation] confusion matrix + mIoU"
MODEL_SUMMARY_CSV="${RUN_OUTPUT_DIR}/model_summary_${SPLIT_TAG}.csv"
srun $SRUN_EXPORT_ARGS $PYTHON evaluate.py --model "$INFER_MODEL" $EVAL_ARGS --output_dir "$RUN_OUTPUT_DIR" --out_csv "$MODEL_SUMMARY_CSV"
require_csv_rows "$MODEL_SUMMARY_CSV" "model summary"

echo ""
echo "[Dead vs Living analysis]"
srun $SRUN_EXPORT_ARGS $PYTHON - << 'PYEOF'
import os
import torch
import config
from evaluate import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_models = list(config.SUPPORTED_MODELS)
requested = os.environ.get("INFER_MODEL", "all")
target_models = all_models if requested == "all" else [requested]
deep_models = set(config.DEEP_MODELS)
missing = [
    m for m in target_models
    if m in deep_models and not os.path.exists(os.path.join(config.CHECKPOINT_DIR, f"best_{m}.pth"))
]
missing += [
    m for m in target_models
    if m == "random_forest" and not os.path.exists(os.path.join(config.CHECKPOINT_DIR, "best_random_forest.joblib"))
]
if missing:
    raise FileNotFoundError(f"Missing checkpoint(s), refusing partial Dead-vs-Living analysis: {missing}")
models_to_check = target_models

for model_name in models_to_check:
    conf = evaluate_model(
        model_name,
        device,
        manifest_csv=(os.environ.get("EVAL_MANIFEST") or None),
    )
    print(f"\n{'=' * 60}")
    print(f"  {model_name.upper()} - Dead vs Living")
    print(f"{'=' * 60}")
    d2l = conf[1, 2] / max(conf[1, 1:].sum(), 1)
    l2d = conf[2, 1] / max(conf[2, 1:].sum(), 1)
    dead_iou = conf[1, 1] / max(conf[1, 1:].sum() + conf[:, 1][1:].sum() - conf[1, 1], 1)
    living_iou = conf[2, 2] / max(conf[2, 1:].sum() + conf[:, 2][1:].sum() - conf[2, 2], 1)
    print(f"  Dead   -> Living error: {d2l:.2%}")
    print(f"  Living -> Dead   error: {l2d:.2%}")
    print(f"  Class1 IoU: {dead_iou:.4f}   Class2 IoU: {living_iou:.4f}")
PYEOF

echo ""
echo "[Cluster analysis]"
srun $SRUN_EXPORT_ARGS $PYTHON analyze_clusters.py --model "$INFER_MODEL" --conf_thr 0.3 --output_dir "$RUN_OUTPUT_DIR" --manifest_csv "$EVAL_MANIFEST" --save_contours --export_geojson
if [ "$INFER_MODEL" = "all" ]; then
    CLUSTER_STATS_CSV="${RUN_OUTPUT_DIR}/cluster_stats_all.csv"
else
    CLUSTER_STATS_CSV="${RUN_OUTPUT_DIR}/cluster_stats_${INFER_MODEL}.csv"
fi
require_csv_rows "$CLUSTER_STATS_CSV" "cluster stats"

echo ""
echo "[Threshold scan]"
srun $SRUN_EXPORT_ARGS $PYTHON analyze_thresholds.py --model "$INFER_MODEL" $EVAL_ARGS --output_dir "$RUN_OUTPUT_DIR" --tag "$SPLIT_TAG"
require_csv_rows "${RUN_OUTPUT_DIR}/threshold_scan_${SPLIT_TAG}.csv" "threshold scan"
require_csv_rows "${RUN_OUTPUT_DIR}/threshold_summary_${SPLIT_TAG}.csv" "threshold summary"

echo ""
echo "[Stratified metrics]"
srun $SRUN_EXPORT_ARGS $PYTHON analyze_stratified_metrics.py --model "$INFER_MODEL" $EVAL_ARGS --output_dir "$RUN_OUTPUT_DIR" --tag "$SPLIT_TAG"
require_csv_rows "${RUN_OUTPUT_DIR}/stratified_metrics_${SPLIT_TAG}.csv" "stratified metrics"

REFERENCE_THRESHOLD_CSV="${RUN_OUTPUT_DIR}/threshold_summary_${SPLIT_TAG}.csv"
if [ "$SPLIT_TAG" = "holdout" ]; then
    ANALYSIS_THRESHOLD_CSV="${OUTPUT_ROOT}/analysis/threshold_summary_analysis.csv"
    if [ -s "$ANALYSIS_THRESHOLD_CSV" ]; then
        REFERENCE_THRESHOLD_CSV="$ANALYSIS_THRESHOLD_CSV"
    else
        fail "[Stage-3] holdout requires non-empty analysis threshold summary: $ANALYSIS_THRESHOLD_CSV. Refusing to use holdout thresholds for final holdout validation."
    fi
fi
if [ ! -s "$REFERENCE_THRESHOLD_CSV" ]; then
    fail "[Stage-3] threshold summary not found or empty: $REFERENCE_THRESHOLD_CSV"
fi

echo ""
echo "[Scale effect analysis]"
srun $SRUN_EXPORT_ARGS $PYTHON analyze_scale_effect.py --model "$INFER_MODEL" $EVAL_ARGS --output_dir "$RUN_OUTPUT_DIR" --tag "$SPLIT_TAG" --threshold_summary_csv "$REFERENCE_THRESHOLD_CSV"
require_csv_rows "${RUN_OUTPUT_DIR}/pixel_scale_metrics_${SPLIT_TAG}.csv" "pixel scale metrics"
require_csv_rows "${RUN_OUTPUT_DIR}/sample_scale_metrics_${SPLIT_TAG}.csv" "sample scale metrics"

if [ "$SPLIT_TAG" = "holdout" ]; then
    echo ""
    echo "[Final validation]"
    srun $SRUN_EXPORT_ARGS $PYTHON final_validation.py --model "$INFER_MODEL" $EVAL_ARGS --output_dir "$RUN_OUTPUT_DIR" --tag "$SPLIT_TAG" --threshold_summary_csv "$REFERENCE_THRESHOLD_CSV" --semantic_summary_csv "$MODEL_SUMMARY_CSV"
    require_csv_rows "${RUN_OUTPUT_DIR}/final_validation_${SPLIT_TAG}.csv" "final validation"
fi

echo ""
echo "[Figure export]"
srun $SRUN_EXPORT_ARGS $PYTHON make_analysis_figures.py --input_dir "$RUN_OUTPUT_DIR" --tag "$SPLIT_TAG"

echo ""
echo "========================================"
echo " Done: $(date)"
echo " Result dir: $RUN_OUTPUT_DIR"
echo "========================================"
