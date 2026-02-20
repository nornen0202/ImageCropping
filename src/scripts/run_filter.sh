#!/bin/bash

# ==============================================================================
# Filter Shutterstock Dataset 
# ==============================================================================
# Usage: ./run_filter.sh [BUCKET] [OUTPUT_FILE] [POOL_SIZE] [TOP_PERCENTILE] [SERVER_MODE]
# Ex) bash ./src/scripts/run_filter.sh sstk_100 filtered.parquet 1000000 0.5 1
# 
# Arguments:
#   BUCKET         : 처리할 sstk 버킷명 (기본값: sstk_100)
#   OUTPUT_FILE    : 저장될 Parquet 파일 경로 (기본값: filtered_sstk_100.parquet)
#   POOL_SIZE      : 최종 선별할 curated pool 사이즈. 0일 경우 샘플링 안함. (기본값: 1000000)
#   TOP_PERCENTILE : 각 카테고리별 Aesthetic 점수 상위 비율 (e.g. 0.5 = 상위 50%) (기본값: 0.5)
#   SERVER_MODE    : 서버 환경 여부 (1일 경우 로컬 가상환경 비활성화) (기본값: 0)
# ==============================================================================

# 파라미터 기본값 설정
BUCKET=${1:-sstk_100}
OUTPUT_FILE=${2:-filtered_${BUCKET}.parquet}
POOL_SIZE=${3:-1000000}
TOP_PERCENTILE=${4:-0.5}
SERVER_MODE=${5:-0}

# 가상환경 활성화 (서버 모드가 아닐 경우에만)
if [ "$SERVER_MODE" -ne 1 ]; then
    if [ -f "/home/jyju25/Venvs/py310_gcf/bin/activate" ]; then
        source /home/jyju25/Venvs/py310_gcf/bin/activate
    fi
fi

# 스크립트 실행 위치에 구애받지 않도록 프로젝트 루트 절대 경로 탐색
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 데이터셋 디렉토리 경로 (서버 등 다른 환경일 경우 환경변수를 통해 덮어쓸 수 있도록 설정)
## Local
#SDP_DIR=${SDP_DIR:-"/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/sdp-sstk"}
#TRAIN_DIR=${TRAIN_DIR:-"/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/SSTK_train_json/v1.0.1"}
#TAR_DIR=${TAR_DIR:-"/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/tars"}
## SPACE N6
SDP_DIR=${SDP_DIR:-"/sstk/sdp-sstk"}
TRAIN_DIR=${TRAIN_DIR:-"/group-volume/jaden.ju/Dataset/SSTK/SSTK_train_json/v1.0.1"}
TAR_DIR=${TAR_DIR:-"/sstk/20230916/sstk_100"}

echo "========================================="
echo "Starting Filter Pipeline"
echo "Project Root      : $PROJECT_ROOT"
echo "SDP Directory     : $SDP_DIR"
echo "Train Directory   : $TRAIN_DIR"
echo "Tar Directory     : $TAR_DIR"
echo "Bucket            : $BUCKET"
echo "Output Parquet    : $OUTPUT_FILE"
echo "Curated Pool Size : $POOL_SIZE"
# bash bc 활용 (혹은 awk)
PCT=$(awk -v pr="$TOP_PERCENTILE" 'BEGIN {print (pr * 100)}')
echo "Top Percentile    : ${PCT}%"
echo "========================================="

# 필터링 스크립트 실행
python "$PROJECT_ROOT/src/filter_sstk_dataset.py" \
    --sdp_dir "$SDP_DIR" \
    --train_dir "$TRAIN_DIR" \
    --tar_dir "$TAR_DIR" \
    --bucket "$BUCKET" \
    --output "$OUTPUT_FILE" \
    --curated_pool_size "$POOL_SIZE" \
    --top_percentile "$TOP_PERCENTILE"

echo "Filtering completed."

if [ "$SERVER_MODE" -ne 1 ]; then
    if declare -f deactivate > /dev/null; then
        deactivate
    fi
fi
