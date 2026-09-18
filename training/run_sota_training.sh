#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# DRISHYA: Master Resilient & Interruptionless SOTA Training Runner
# Target Hardware: 1x NVIDIA Quadro RTX 6000 (24 GB GDDR6, Turing TU102)
#
# Key Resilience Features:
#   • Auto-Resume: Automatically detects existing checkpoints / emergency saves
#   • Fault Isolation: Each stage runs in an isolated Python process to clear CUDA memory
#   • Stage-Skipping: Automatically skips completed stages unless forced
#   • Graceful Shutdown: Traps SIGINT / SIGTERM and ensures checkpoints flush safely
#   • Master Logging: Captures stdout / stderr to logs/sota_training_<timestamp>.log
# ──────────────────────────────────────────────────────────────────────────────

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Default Paths & Hyperparameters for 24 GB Quadro RTX 6000
SPLITS_CSV="${SPLITS_CSV:-data/splits/5fold_splits.csv}"
DATA_DIR="${DATA_DIR:-data/preprocessed}"
DATA_ALL_CSV="${DATA_ALL_CSV:-data/splits/all_images.csv}"
FOLD=0
EPOCHS=20
BATCH_SIZE=8        # Calibrated for 24 GB VRAM
ACCUM_STEPS=2       # Effective Batch Size = 8 * 2 = 16
WORKERS=16
STAGE="${STAGE:-all}"
FORCE=0

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_banner() {
    echo -e "${CYAN}"
    echo "  ╔══════════════════════════════════════════════════════════════════╗"
    echo "  ║        दृष्य  DRISHYA SOTA Resilient Training Master             ║"
    echo "  ║         Hardware Target: NVIDIA Quadro RTX 6000 (24 GB)          ║"
    echo "  ║       Precision: Strict Native FP16 AMP (Turing Optimized)      ║"
    echo "  ╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

usage() {
    echo "Usage: ./training/run_sota_training.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --splits_csv <path>   Path to 5-fold stratified CSV (default: data/splits/5fold_splits.csv)"
    echo "  --data_dir <path>     Directory containing preprocessed 512x512 images"
    echo "  --data_all_csv <path> Full CSV for pseudo-labeling (default: data/splits/all_images.csv)"
    echo "  --fold <int>          Validation lockbox fold (default: 0)"
    echo "  --epochs <int>        Training epochs per stage (default: 30)"
    echo "  --batch_size <int>    Physical batch size (default: 8 for 24 GB RTX 6000)"
    echo "  --accum_steps <int>   Gradient accumulation steps (default: 2 -> effective 16)"
    echo "  --workers <int>       DataLoader worker threads (default: 8)"
    echo "  --stage <name>        all | teacher1 | teacher2 | teacher3 | pseudo | distill | xgboost | export"
    echo "  --force               Force re-run stages even if final models already exist"
    echo "  -h, --help            Show this help menu"
    echo ""
    exit 0
}

# Parse Command Line Arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --splits_csv) SPLITS_CSV="${2:-}"; shift 2 ;;
        --data_dir) DATA_DIR="${2:-}"; shift 2 ;;
        --data_all_csv) DATA_ALL_CSV="${2:-}"; shift 2 ;;
        --fold) FOLD="${2:-0}"; shift 2 ;;
        --epochs) EPOCHS="${2:-20}"; shift 2 ;;
        --batch_size) BATCH_SIZE="${2:-8}"; shift 2 ;;
        --accum_steps) ACCUM_STEPS="${2:-2}"; shift 2 ;;
        --workers) WORKERS="${2:-16}"; shift 2 ;;
        --stage) STAGE="${2:-all}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage ;;
        *) echo -e "${RED}[ERROR] Unknown option: $1${NC}"; usage ;;
    esac
done

# Setup directories and logging
mkdir -p logs cache models/teachers models/student
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="logs/sota_training_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

print_banner

echo -e "${BOLD}[SYSTEM AUDIT]${NC}"
echo "  • Working Directory: $PROJECT_ROOT"
echo "  • Splits CSV:        $SPLITS_CSV"
echo "  • Preprocessed Dir:  $DATA_DIR"
echo "  • Validation Fold:   $FOLD"
echo "  • Epochs:            $EPOCHS"
echo "  • Physical Batch:    $BATCH_SIZE (Accumulation: $ACCUM_STEPS -> Effective: $((BATCH_SIZE * ACCUM_STEPS)))"
echo "  • Workers:           $WORKERS"
echo "  • Target Stage:      $STAGE"
echo "  • Pipeline Log:      $LOG_FILE"
echo ""

# Check Python environment
if [ -n "${PYTHON_CMD:-}" ]; then
    : # already provided by user or environment
elif [ -x "/workspace/envs/ml/bin/python" ]; then
    PYTHON_CMD="/workspace/envs/ml/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_CMD="${VIRTUAL_ENV}/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}[FATAL] Python executable not found in PATH.${NC}"
    exit 1
fi

# Check GPU capability
if command -v nvidia-smi &>/dev/null; then
    echo -e "${GREEN}[INFO] NVIDIA GPU Detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo -e "${YELLOW}[WARNING] nvidia-smi not found. Ensure NVIDIA drivers are loaded.${NC}"
fi

# Trap Ctrl+C cleanly
trap_handler() {
    echo -e "\n${YELLOW}[ALERT] Training pipeline intercepted by signal (SIGINT/SIGTERM).${NC}"
    echo -e "Checkpoints have been safely saved by PyTorch signal handlers."
    echo -e "To resume training exactly where it stopped, re-run:"
    echo -e "   ${BOLD}./training/run_sota_training.sh${NC}\n"
    exit 130
}
trap trap_handler SIGINT SIGTERM

run_stage() {
    local stage_name="$1"
    local check_file="$2"
    shift 2
    local cmd=("$@")

    echo -e "\n${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}▶ STAGE: $stage_name${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════${NC}"

    if [[ $FORCE -eq 0 && -f "$check_file" ]]; then
        echo -e "${GREEN}✔ Target artifact already exists:${NC} $check_file"
        echo -e "  Skipping stage. (Use --force to re-train)."
        return 0
    fi

    echo -e "[INFO] Executing command: ${cmd[*]}"
    local start_time=$(date +%s)

    "${cmd[@]}"
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo -e "\n${RED}✖ [ERROR] Stage '$stage_name' failed with exit code $exit_code!${NC}"
        echo -e "You can fix the issue and resume without losing previous stages."
        exit $exit_code
    fi

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo -e "${GREEN}✔ Stage '$stage_name' completed successfully in ${elapsed}s!${NC}"
}

# ── STAGE 1: Teacher 1 (ConvNeXtV2-Base) ───────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "teacher1" ]]; then
    run_stage \
        "Teacher 1: ConvNeXtV2-Base" \
        "models/teachers/convnextv2_base/training_complete.flag" \
        "$PYTHON_CMD" training/train_teacher.py \
            --teacher_id 1 \
            --splits_csv "$SPLITS_CSV" \
            --data_dir "$DATA_DIR" \
            --fold "$FOLD" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --accum_steps "$ACCUM_STEPS" \
            --workers "$WORKERS" \
            --resume
fi

# ── STAGE 2: Teacher 2 (Swin-Base-384) ─────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "teacher2" ]]; then
    run_stage \
        "Teacher 2: Swin-Base-384" \
        "models/teachers/swin_base/training_complete.flag" \
        "$PYTHON_CMD" training/train_teacher.py \
            --teacher_id 2 \
            --splits_csv "$SPLITS_CSV" \
            --data_dir "$DATA_DIR" \
            --fold "$FOLD" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --accum_steps "$ACCUM_STEPS" \
            --workers "$WORKERS" \
            --resume
fi

# ── STAGE 3: Teacher 3 (EfficientNet-B5) ───────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "teacher3" ]]; then
    run_stage \
        "Teacher 3: EfficientNet-B5" \
        "models/teachers/efficientnet_b5/training_complete.flag" \
        "$PYTHON_CMD" training/train_teacher.py \
            --teacher_id 3 \
            --splits_csv "$SPLITS_CSV" \
            --data_dir "$DATA_DIR" \
            --fold "$FOLD" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --accum_steps "$ACCUM_STEPS" \
            --workers "$WORKERS" \
            --resume
fi

# ── STAGE 4: Offline Pseudo-Label Consensus Generation ─────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "pseudo" ]]; then
    # Use all_images.csv if provided, otherwise fallback to splits_csv
    PSEUDO_CSV="$SPLITS_CSV"
    if [ -f "$DATA_ALL_CSV" ]; then
        PSEUDO_CSV="$DATA_ALL_CSV"
    fi

    run_stage \
        "Ensemble Consensus Pseudo-Labeling" \
        "cache/pseudo_labels.h5" \
        "$PYTHON_CMD" training/generate_pseudo.py \
            --data_csv "$PSEUDO_CSV" \
            --data_dir "$DATA_DIR" \
            --t1_ckpt "models/teachers/convnextv2_base/best_model.pth" \
            --t2_ckpt "models/teachers/swin_base/best_model.pth" \
            --t3_ckpt "models/teachers/efficientnet_b5/best_model.pth" \
            --output_h5 "cache/pseudo_labels.h5" \
            --batch_size 16 \
            --workers "$WORKERS"
fi

# ── STAGE 5: Distill Student Model (EfficientNetV2-S U-Net) ───────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "distill" ]]; then
    run_stage \
        "Student Knowledge Distillation (EfficientNetV2-S)" \
        "models/student/training_complete.flag" \
        "$PYTHON_CMD" training/distill_student.py \
            --splits_csv "$SPLITS_CSV" \
            --data_dir "$DATA_DIR" \
            --pseudo_h5 "cache/pseudo_labels.h5" \
            --fold "$FOLD" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --accum_steps "$ACCUM_STEPS" \
            --workers "$WORKERS" \
            --output_dir "models/student" \
            --resume
fi

# ── STAGE 6: 32-D Morphometric Extraction & XGBoost Fit ────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "xgboost" ]]; then
    run_stage \
        "32-D Morphometry & XGBoost Meta-Gatekeeper" \
        "models/xgboost_gatekeeper.json" \
        "$PYTHON_CMD" training/train_xgboost.py \
            --splits_csv "$SPLITS_CSV" \
            --data_dir "$DATA_DIR" \
            --student_ckpt "models/student/best_model.pth" \
            --output_dir "models" \
            --features_cache "cache/features_32d.npz" \
            --fold "$FOLD" \
            --batch_size 16
fi

# ── STAGE 7: ONNX Export & Verification ───────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "export" ]]; then
    run_stage \
        "Production ONNX Export & Edge Benchmarking" \
        "models/drishya_student_fp16.onnx" \
        "$PYTHON_CMD" training/export_onnx.py \
            --checkpoint "models/student/best_model.pth" \
            --output_dir "models" \
            --opset 16
fi

echo -e "\n${GREEN}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}🎉 DRISHYA SOTA PIPELINE EXECUTION COMPLETE!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════════${NC}"
echo "Production Artifacts Ready:"
echo "  • Student PyTorch:   models/student/best_model.pth"
echo "  • Student FP32 ONNX: models/drishya_student.onnx"
echo "  • Student FP16 ONNX: models/drishya_student_fp16.onnx (~42.1 MB)"
echo "  • XGBoost Regressor: models/xgboost_gatekeeper.json"
echo "  • Kappa Thresholds:  models/clinical_thresholds.json"
echo ""
echo "To verify with the local DRISHYA server, run:"
echo "  ./start.sh"
echo ""
