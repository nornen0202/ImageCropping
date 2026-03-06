#!/bin/bash

# ==============================================================================
# Filter Shutterstock Dataset 
# ==============================================================================
# Usage: ./run_filter.sh [BUCKET] [OUTPUT_FILE] [POOL_SIZE] [TOP_PERCENTILE] [SERVER_MODE] [CURATED_IMAGE_DIR] [CURATED_IMAGE_SKIP_EXISTING]
# Ex) bash ./src/scripts/run_filter.sh sstk_100 data/SSTK/10K_local/filtered_sstk_100.parquet 10000 0.2 0 2>&1 | tee data/SSTK/10K_local/filter_debug.log
# Ex) bash ./src/scripts/run_filter.sh sstk_100 data/SSTK/10K/filtered_sstk_100.parquet 10000 0.2 1 2>&1 | tee data/SSTK/10K/filter_debug.log
# Ex) bash ./src/scripts/run_filter.sh sstk_100 data/SSTK/100K/filtered_sstk_100.parquet 1000000 0.5 1 2>&1 | tee data/SSTK/100K/filter_debug.log
# Ex) bash ./src/scripts/run_filter.sh sstk_100
# 
# Arguments:
#   BUCKET         : 처리할 sstk 버킷명 (기본값: sstk_100)
#   OUTPUT_FILE    : 저장될 Parquet 파일 경로 (기본값: filtered_sstk_100.parquet)
#   POOL_SIZE      : 최종 선별할 curated pool 사이즈. 0일 경우 샘플링 안함. (기본값: 1000000)
#   TOP_PERCENTILE : 각 카테고리별 Aesthetic 점수 상위 비율 (e.g. 0.5 = 상위 50%) (기본값: 0.5)
#   SERVER_MODE    : 서버 환경 여부 (1일 경우 로컬 가상환경 비활성화) (기본값: 1)
#   CURATED_IMAGE_DIR : curated 이미지를 추출 저장할 디렉토리 (기본값: 비활성)
#   CURATED_IMAGE_SKIP_EXISTING : curated 이미지 저장 시 기존 파일 skip 여부 (기본값: 1)
#   2>&1 | tee data/SSTK/filter_debug.log : 로그를 파일로 저장하고 싶을 때 사용
# ==============================================================================

# 파라미터 기본값 설정
BUCKET=${1:-sstk_100}

# 스크립트 실행 위치에 구애받지 않도록 프로젝트 루트 절대 경로 탐색
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 출력 경로 확보
OUTPUT_DIR="$PROJECT_ROOT/data/SSTK"
mkdir -p "$OUTPUT_DIR"

OUTPUT_FILE=${2:-"${OUTPUT_DIR}/filtered_${BUCKET}.parquet"}
POOL_SIZE=${3:-1000000}
TOP_PERCENTILE=${4:-0.5}
SERVER_MODE=${5:-1}
CURATED_IMAGE_DIR=${6:-""}
CURATED_IMAGE_SKIP_EXISTING=${7:-1}
FILTER_REQUIRE_TRAIN_MATCH=${FILTER_REQUIRE_TRAIN_MATCH:-1}
FILTER_TAG_EMBED_MULTI_GPU=${FILTER_TAG_EMBED_MULTI_GPU:--1}
FILTER_TAG_EMBED_GPU_IDS=${FILTER_TAG_EMBED_GPU_IDS:-""}
FILTER_TAG_EMBED_BATCH_SIZE=${FILTER_TAG_EMBED_BATCH_SIZE:-128}
FILTER_TAG_EMBED_CHUNK_SIZE=${FILTER_TAG_EMBED_CHUNK_SIZE:-0}
FILTER_TAG_EMBED_DEVICE=${FILTER_TAG_EMBED_DEVICE:-auto}
FILTER_CATEGORY_MAP_WORKERS=${FILTER_CATEGORY_MAP_WORKERS:-0}
FILTER_CATEGORY_MAP_CHUNK_SIZE=${FILTER_CATEGORY_MAP_CHUNK_SIZE:-4096}
FILTER_SAMPLE_EXTRACT_WORKERS=${FILTER_SAMPLE_EXTRACT_WORKERS:-0}

# 가상환경 활성화 (서버 모드가 아닐 경우에만)
if [ "$SERVER_MODE" -ne 1 ]; then
    if [ -f "/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate" ]; then
        source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate
    fi
fi



# 데이터셋 디렉토리 경로 (서버 등 다른 환경일 경우 환경변수를 통해 덮어쓸 수 있도록 설정)
if [ "$SERVER_MODE" -ne 1 ]; then
    ## Local
    SDP_DIR=${SDP_DIR:-"/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/sdp-sstk"}
    TRAIN_DIR=${TRAIN_DIR:-"/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/SSTK_train_json/v1.0.1"}
    TAR_DIR=${TAR_DIR:-"/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100"}
else
    ## SPACE N6
    SDP_DIR=${SDP_DIR:-"/sstk/sdp-sstk"}
    TRAIN_DIR=${TRAIN_DIR:-"/group-volume/jaden.ju/Dataset/SSTK/SSTK_train_json/v1.0.1"}
    TAR_DIR=${TAR_DIR:-"/sstk/20230916/sstk_100"}
fi

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
echo "Train Match Only  : $FILTER_REQUIRE_TRAIN_MATCH"
echo "TagEmbed MGPU     : $FILTER_TAG_EMBED_MULTI_GPU (gpu_ids=${FILTER_TAG_EMBED_GPU_IDS:-auto})"
echo "TagEmbed BS/Chunk : $FILTER_TAG_EMBED_BATCH_SIZE / $FILTER_TAG_EMBED_CHUNK_SIZE"
echo "TagEmbed Device   : $FILTER_TAG_EMBED_DEVICE"
echo "CatMap Workers    : $FILTER_CATEGORY_MAP_WORKERS (chunk=$FILTER_CATEGORY_MAP_CHUNK_SIZE)"
echo "SampleX Workers   : $FILTER_SAMPLE_EXTRACT_WORKERS"
if [ -n "$CURATED_IMAGE_DIR" ]; then
  echo "Curated Image Dir : $CURATED_IMAGE_DIR"
  echo "Curated Img Skip  : $CURATED_IMAGE_SKIP_EXISTING"
fi
echo "========================================="

EXTRA_ARGS=()
if [ -n "$CURATED_IMAGE_DIR" ]; then
  EXTRA_ARGS+=(
    --save_curated_images_dir "$CURATED_IMAGE_DIR"
    --save_curated_images_skip_existing "$CURATED_IMAGE_SKIP_EXISTING"
  )
fi

# 필터링 스크립트 실행
python3 -u "$PROJECT_ROOT/src/filter_sstk_dataset.py" \
    --sdp_dir "$SDP_DIR" \
    --train_dir "$TRAIN_DIR" \
    --tar_dir "$TAR_DIR" \
    --bucket "$BUCKET" \
    --output "$OUTPUT_FILE" \
    --curated_pool_size "$POOL_SIZE" \
    --top_percentile "$TOP_PERCENTILE" \
    --server_mode "$SERVER_MODE" \
    --require_train_match "$FILTER_REQUIRE_TRAIN_MATCH" \
    --tag_embed_multi_gpu "$FILTER_TAG_EMBED_MULTI_GPU" \
    --tag_embed_gpu_ids "$FILTER_TAG_EMBED_GPU_IDS" \
    --tag_embed_batch_size "$FILTER_TAG_EMBED_BATCH_SIZE" \
    --tag_embed_chunk_size "$FILTER_TAG_EMBED_CHUNK_SIZE" \
    --tag_embed_device "$FILTER_TAG_EMBED_DEVICE" \
    --category_map_workers "$FILTER_CATEGORY_MAP_WORKERS" \
    --category_map_chunk_size "$FILTER_CATEGORY_MAP_CHUNK_SIZE" \
    --sample_extract_workers "$FILTER_SAMPLE_EXTRACT_WORKERS" \
    "${EXTRA_ARGS[@]}"

echo "Filtering completed."

if [ "$SERVER_MODE" -ne 1 ]; then
    if declare -f deactivate > /dev/null; then
        deactivate
    fi
fi
