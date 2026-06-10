#!/bin/bash
# Feature Extraction 실행 스크립트
# Usage: ./run_extract_features.sh <metadata_parquet> <bucket_name> <output_jsonl> [--priority high_efficiency|quality_first] [--tar_dir path/to/tars] [--server_mode 1]
: <<'USAGE'
bash src/scripts/run_extract_features.sh \
  data/SSTK/10K_local/filtered_sstk_100.parquet \
  sstk_100 \
  data/SSTK/10K_local/features_sstk_100.jsonl \
  --priority high_efficiency \
  --server_mode 1 \
  2>&1 | tee src/scripts/log_run_extract_features.log
USAGE


if [ "$#" -lt 3 ]; then
    echo "Usage: ./run_extract_features.sh <metadata_parquet> <bucket_name> <output_jsonl> [--priority high_efficiency|quality_first] [--tar_dir path/to/tars] [--server_mode 1|0]"
    echo "Example: ./run_extract_features.sh filtered_sstk_100.parquet sstk_100 features_sstk_100.jsonl --priority quality_first"
    exit 1
fi

METADATA=$1
BUCKET=$2
OUTPUT=$3
PRIORITY="high_efficiency"
TAR_DIR=""
SERVER_MODE=1

shift 3
while [ "$#" -gt 0 ]; do
    case "$1" in
        --priority)
            PRIORITY="$2"
            shift 2
            ;;
        --tar_dir)
            TAR_DIR="$2"
            shift 2
            ;;
        --server_mode)
            SERVER_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

if [ -z "$TAR_DIR" ]; then
    # 데이터셋 디렉토리 경로 (디폴트)
    if [ "$SERVER_MODE" -ne 1 ]; then
        ## Local
        TAR_DIR="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916"
    else
        ## SPACE N6
        TAR_DIR="/sstk/20230916"
    fi
fi

# 가상환경 활성화 (서버 모드가 아닐 경우에만)
if [ "$SERVER_MODE" -ne 1 ]; then
    if [ -f "/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate" ]; then
        source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate
    else
        echo "Warning: Virtual environment not found at /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate. Using default python."
    fi
fi

# PyTorch 및 Ray 연산 최적화 환경변수 (NVIDIA GPU 최적화)
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export OMP_NUM_THREADS=8
# Ray에서 기본 메모리 부족 에러 방지
export RAY_IGNORE_UNHANDLED_ERRORS=1

# PYTHONPATH 설정하여 내부 모듈 참조 허용
SRC_DIR=$(dirname "$0")/..
export PYTHONPATH="${SRC_DIR}/extract_features:${PYTHONPATH}"

echo "============================================"
echo " Starting Feature Extraction Pipeline "
echo "============================================"
echo " - Metadata: $METADATA"
echo " - Bucket: $BUCKET"
echo " - Output: $OUTPUT"
echo " - Priority: $PRIORITY"
echo " - Tar Dir: $TAR_DIR"
echo " - Server Mode: $SERVER_MODE"
echo "============================================"

# GPU 개수 확인 (기본적으로 가용되는 모든 GPU를 Ray Worker 개수로 사용)
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No GPUs detected. Falling back to 1 'GPU' worker for CPU test."
    NUM_GPUS=1
fi

echo "Detected $NUM_GPUS GPUs. Launching pipeline..."

# 실행
python "$SRC_DIR/extract_features_pipeline.py" \
    --input_parquet "$METADATA" \
    --bucket "$BUCKET" \
    --output_jsonl "$OUTPUT" \
    --priority "$PRIORITY" \
    --tar_dir "$TAR_DIR" \
    --num_workers "$NUM_GPUS" \
    --batch_size 16

echo "============================================"
echo " Feature Extraction Pipeline Completed "
echo "============================================"
