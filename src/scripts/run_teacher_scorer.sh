#!/bin/bash
# ============================================================================
# run_teacher_scorer.sh
# Teacher Scorer (Section 9): Cheap -> Expensive -> Top-K Diversity
# * OCR 항은 의도적으로 제외하여 실행합니다.
# ============================================================================

: <<'USAGE'
# 기본 실행 (10K_local 데이터)
bash src/scripts/run_teacher_scorer.sh

# 출력 경로 지정
bash src/scripts/run_teacher_scorer.sh \
  --output_jsonl data/SSTK/10K_local/teacher_scores_ar.jsonl \
  --output_overview_json data/SSTK/10K_local/teacher_scores_overview.json \
  --output_overview_csv data/SSTK/10K_local/teacher_scores_overview_by_ar.csv

# Top-K / Cheap M / Diversity IoU 조정
bash src/scripts/run_teacher_scorer.sh \
  --cheap_top_m 40 --top_k 5 --tau_div 0.75

# 실모델 Expensive 강제 (Aesthetic + cos(E_I,E_T))
bash src/scripts/run_teacher_scorer.sh \
  --use_real_expensive 1 \
  --c1_jsonl data/SSTK/10K_local/feats_c1.jsonl \
  --align_device auto --aesthetic_device auto

# 시각화 설정 조정
bash src/scripts/run_teacher_scorer.sh \
  --num_viz 120 --target_ar 1:1 --decision_filter crop

# 시각화 생략 (스코어/통계만)
bash src/scripts/run_teacher_scorer.sh --run_viz 0

# QA 리포트만 별도 생성
bash src/scripts/run_teacher_scorer.sh --run_viz 0 --run_qa 1

# 풀런
export CUDA_VISIBLE_DEVICES=0

bash src/scripts/run_teacher_scorer.sh \
  --candidates_jsonl data/SSTK/10K_local/candidates_ar.jsonl \
  --features_jsonl data/SSTK/10K_local/feats_c2c3c5_v2_strict_enriched.jsonl \
  --c1_jsonl data/SSTK/10K_local/feats_c1.jsonl \
  --parquet data/SSTK/10K_local/filtered_sstk_100.parquet \
  --tar_dir /sstk/20230916/sstk_100 \
  --output_jsonl data/SSTK/10K_local/teacher_scores_ar_p0_real.jsonl \
  --output_overview_json data/SSTK/10K_local/teacher_scores_overview_p0_real.json \
  --output_overview_csv data/SSTK/10K_local/teacher_scores_overview_by_ar_p0_real.csv \
  --qa_out_json data/SSTK/10K_local/teacher_scores_qa_report_p0_real.json \
  --qa_out_csv data/SSTK/10K_local/teacher_scores_qa_report_by_ar_p0_real.csv \
  --viz_out_dir data/SSTK/10K_local/visualizations/teacher_scorer_p0_real \
  --use_real_expensive 1 \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 12 \
  --run_qa 1 \
  --run_viz 1
USAGE

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

CANDIDATES_JSONL="data/SSTK/10K_local/candidates_ar.jsonl"
FEATURES_JSONL="data/SSTK/10K_local/feats_c2c3c5_v2_strict_enriched.jsonl"
C1_JSONL="data/SSTK/10K_local/feats_c1.jsonl"
PARQUET="data/SSTK/10K_local/filtered_sstk_100.parquet"
TAR_DIR="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100"

OUTPUT_JSONL="data/SSTK/10K_local/teacher_scores_ar.jsonl"
OUTPUT_OVERVIEW_JSON="data/SSTK/10K_local/teacher_scores_overview.json"
OUTPUT_OVERVIEW_CSV="data/SSTK/10K_local/teacher_scores_overview_by_ar.csv"

VIZ_OUT_DIR="data/SSTK/10K_local/visualizations/teacher_scorer_v1"
TARGET_AR="all"
DECISION_FILTER="all"
NUM_VIZ=120
RUN_VIZ=1
RUN_QA=1

CHEAP_TOP_M=30
TOP_K=5
TAU_DIV=0.75
USE_REAL_EXPENSIVE=1
ALIGN_MODEL_NAME=""
ALIGN_PRETRAINED=""
ALIGN_DEVICE="auto"
AESTHETIC_DEVICE="auto"
EXP_BATCH_SIZE=24
AESTHETIC_MLP_PATH="weights/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_MLP_URL="https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/sac+logos+ava1-l14-linearMSE.pth"
MAX_IMAGES=0
SEED=42

QA_OUT_JSON="data/SSTK/10K_local/teacher_scores_qa_report.json"
QA_OUT_CSV="data/SSTK/10K_local/teacher_scores_qa_report_by_ar.csv"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --candidates_jsonl) CANDIDATES_JSONL="$2"; shift 2 ;;
    --features_jsonl) FEATURES_JSONL="$2"; shift 2 ;;
    --c1_jsonl) C1_JSONL="$2"; shift 2 ;;
    --parquet) PARQUET="$2"; shift 2 ;;
    --tar_dir) TAR_DIR="$2"; shift 2 ;;

    --output_jsonl) OUTPUT_JSONL="$2"; shift 2 ;;
    --output_overview_json) OUTPUT_OVERVIEW_JSON="$2"; shift 2 ;;
    --output_overview_csv) OUTPUT_OVERVIEW_CSV="$2"; shift 2 ;;

    --viz_out_dir) VIZ_OUT_DIR="$2"; shift 2 ;;
    --target_ar) TARGET_AR="$2"; shift 2 ;;
    --decision_filter) DECISION_FILTER="$2"; shift 2 ;;
    --num_viz) NUM_VIZ="$2"; shift 2 ;;
    --run_viz) RUN_VIZ="$2"; shift 2 ;;
    --run_qa) RUN_QA="$2"; shift 2 ;;

    --cheap_top_m) CHEAP_TOP_M="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --tau_div) TAU_DIV="$2"; shift 2 ;;
    --use_real_expensive) USE_REAL_EXPENSIVE="$2"; shift 2 ;;
    --align_model_name) ALIGN_MODEL_NAME="$2"; shift 2 ;;
    --align_pretrained) ALIGN_PRETRAINED="$2"; shift 2 ;;
    --align_device) ALIGN_DEVICE="$2"; shift 2 ;;
    --aesthetic_device) AESTHETIC_DEVICE="$2"; shift 2 ;;
    --exp_batch_size) EXP_BATCH_SIZE="$2"; shift 2 ;;
    --aesthetic_mlp_path) AESTHETIC_MLP_PATH="$2"; shift 2 ;;
    --aesthetic_mlp_url) AESTHETIC_MLP_URL="$2"; shift 2 ;;
    --max_images) MAX_IMAGES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --qa_out_json) QA_OUT_JSON="$2"; shift 2 ;;
    --qa_out_csv) QA_OUT_CSV="$2"; shift 2 ;;

    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

VENV="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"
if [ -f "$VENV" ]; then
  # shellcheck disable=SC1090
  source "$VENV"
else
  echo "[warn] venv not found: $VENV (using current python)"
fi

echo "[1/3] Running teacher scorer..."
python src/score_teacher.py \
  --candidates_jsonl "$CANDIDATES_JSONL" \
  --features_jsonl "$FEATURES_JSONL" \
  --c1_jsonl "$C1_JSONL" \
  --parquet "$PARQUET" \
  --tar_dir "$TAR_DIR" \
  --output_jsonl "$OUTPUT_JSONL" \
  --output_overview_json "$OUTPUT_OVERVIEW_JSON" \
  --output_overview_by_ar_csv "$OUTPUT_OVERVIEW_CSV" \
  --cheap_top_m "$CHEAP_TOP_M" \
  --top_k "$TOP_K" \
  --tau_div "$TAU_DIV" \
  --use_real_expensive "$USE_REAL_EXPENSIVE" \
  --align_model_name "$ALIGN_MODEL_NAME" \
  --align_pretrained "$ALIGN_PRETRAINED" \
  --align_device "$ALIGN_DEVICE" \
  --aesthetic_device "$AESTHETIC_DEVICE" \
  --exp_batch_size "$EXP_BATCH_SIZE" \
  --aesthetic_mlp_path "$AESTHETIC_MLP_PATH" \
  --aesthetic_mlp_url "$AESTHETIC_MLP_URL" \
  --max_images "$MAX_IMAGES" \
  --seed "$SEED"

if [ "$RUN_QA" -eq 1 ]; then
  echo "[2/3] Building QA report..."
  python src/scripts/qa_teacher_report.py \
    --teacher_scores_jsonl "$OUTPUT_JSONL" \
    --output_json "$QA_OUT_JSON" \
    --output_by_ar_csv "$QA_OUT_CSV"
else
  echo "[2/3] QA report skipped (--run_qa 0)."
fi

if [ "$RUN_VIZ" -eq 1 ]; then
  echo "[3/3] Rendering teacher scorer visualizations..."
  python src/visualize_teacher_scores.py \
    --teacher_scores_jsonl "$OUTPUT_JSONL" \
    --parquet "$PARQUET" \
    --tar_dir "$TAR_DIR" \
    --out_dir "$VIZ_OUT_DIR" \
    --target_ar "$TARGET_AR" \
    --decision_filter "$DECISION_FILTER" \
    --num_samples "$NUM_VIZ" \
    --seed "$SEED"
else
  echo "[3/3] Visualization skipped (--run_viz 0)."
fi

echo "[done] Teacher scorer outputs:"
echo "  - $OUTPUT_JSONL"
echo "  - $OUTPUT_OVERVIEW_JSON"
echo "  - $OUTPUT_OVERVIEW_CSV"
if [ "$RUN_QA" -eq 1 ]; then
  echo "  - $QA_OUT_JSON"
  echo "  - $QA_OUT_CSV"
fi
if [ "$RUN_VIZ" -eq 1 ]; then
  echo "  - $VIZ_OUT_DIR"
fi
