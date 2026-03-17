# SSTK Cropping Data Factory v1.9 실행 가이드 (Phase A ~ 10 VLM Teacher)

이 문서는 `SSTK_Cropping_DataFactory_QwenLabeler_Reorganized_KO_v1_9.md` 기준으로, 현재 코드베이스에서 **Phase A(Filter) → Phase B(Perception/Candidate) → 9) Teacher Scorer → 10) VLM/MLLM Teacher Labeler**까지를 처음부터 재현하는 실전 운영 가이드입니다.

범위:
- 포함: `3) Phase A`, `5) Phase B(C1~C6, C4 OCR 포함)`, `9) Teacher Scorer`, `10) VLM/MLLM Teacher`
- 제외: `v1.7 PICD`, `11) UNIC View Adjustment`

---

## 1. 핵심 리팩터링 요약

기존에는 C2/C3/C5를 분리 실행하는 경로가 중심이었고, 단계가 다수로 쪼개져 운영이 번거로웠습니다. 현재는 아래 방식으로 정리되었습니다.

- 신규 end-to-end 오케스트레이터 추가
  - `src/scripts/run_phaseA_to_teacher_e2e.sh`
  - Filter → Precompute → Candidate → Teacher(+QA/+Viz) → VLM Teacher(옵션)까지 1개 커맨드로 실행
- Teacher final score 구조 업데이트
  - 최종 정렬은 `score_rank`
  - keep/minimal/crop 정책은 `score_policy = score_rank + area prior`
  - `teacher_scores_ar_*.jsonl`에는 `macro_scores`, `macro_components`, `macro_masks`, `score_rank`, `score_policy`, `final_legacy`가 함께 저장됨
  - OpenCLIP align 모델은 `ViT-H-14` GPU OOM 시 즉시 CPU fallback 하지 않고 먼저 `ViT-B-32/openai`로 GPU downgrade 후 real-expensive를 유지
- Precompute 통합 모드(`--precompute_mode unified`) 도입
  - C1/C2/C3/C4/C5/C6를 1-pass로 추출 가능 (`run_c1=1`일 때 C1 포함)
  - `enrich_c3_pose_jsonl.py`로 face/gaze proxy 보강 후 최종 병합 피처 직접 생성
  - 분리 실행 대비 I/O/재로딩 오버헤드 감소
- C4 OCR 품질우선 경로 추가
  - `quality_first`에서 PP-OCRv5 server detector(`PP-OCRv5_server_det`)를 1순위로 사용
  - 추출 결과에 `ocr_text_boxes`/`text_overlay_likely` 신호를 동시 기록해 subject routing(`text_document`)과 직접 연동
- 실행 스크립트 품질 개선
  - 멀티 GPU 기본 경로를 Ray 의존에서 **No-Ray 샤딩 병렬**로 전환
  - Ray는 `--mode ray`로 명시한 경우에만 사용(레거시)
  - `run_extract_component.sh`: `--venv_path` 지원, 10K 가이드를 1-pass 권장 흐름으로 갱신
  - `run_teacher_scorer.sh`: `--server_mode`, `--venv_path` 지원, tar 경로 자동 분기 정리
  - HF 토큰 하드코딩 제거(환경변수 연동 방식)

---

## 2. 입력 데이터/디렉토리 전제

필수 입력:
- 이미지 tar 묶음: 예) `.../sstk_100/*.tar`
- SDP 메타 json: `sdp-sstk`
- train 메타 json: `SSTK_train_json/v1.0.1`

`run_filter.sh` 기본 경로 규칙:
- `server_mode=0`(로컬)
  - `SDP_DIR=/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/sdp-sstk`
  - `TRAIN_DIR=/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/SSTK_train_json/v1.0.1`
  - `TAR_DIR=/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100`
- `server_mode=1`(서버)
  - `SDP_DIR=/sstk/sdp-sstk`
  - `TRAIN_DIR=/group-volume/jaden.ju/Dataset/SSTK/SSTK_train_json/v1.0.1`
  - `TAR_DIR=/sstk/20230916/sstk_100`

경로가 다르면 실행 전에 환경변수로 override:
```bash
export SDP_DIR=/your/sdp
export TRAIN_DIR=/your/train_json
export TAR_DIR=/your/tar_root
```

### 2.1 초기 환경 세팅(품질최우선 전체 파이프라인용)
`3.6 처음부터 끝까지(명시형 풀 옵션 템플릿)` 실행 전, 아래 의존성 설치를 먼저 1회 수행하세요.

1) venv 생성/활성화
```bash
python3 -m venv /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate
```

2) 기본 의존성 설치(권장 스크립트)
```bash
cd /media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping
bash src/scripts/install_features_deps_torch251_cu121_stable.sh
```

스크립트 기본 정책:
- Torch 2.5.1 + cu121, OpenMMLab(mmcv/mmengine/mmdet/mmpose), SAM2, EfficientViT 설치
- C5 quality_first용 ScaleLSD 설치 시도(`ENABLE_SCALELSD=1` 기본)
- C6 quality_first용 Gazelle(Gaze-LLE) 설치 시도(`ENABLE_GAZELLE=1` 기본)
- C4 OCR용 PaddleOCR(PP-OCRv5 런타임) 기본 비활성(`ENABLE_C4_OCR=0` 기본)
- 기본 파이프라인 환경은 `ENABLE_QWEN3_VL=0`(기본)으로 유지되어, C3/mmpretrain과의 버전 충돌을 피함
- Qwen3-VL은 **별도 환경**에서 `QWEN3_RUNTIME_ONLY=1` 모드로 설치/운영 권장

3) 품질최우선 모듈 import 검증
```bash
export PYTHONPATH="$(pwd)/third_party/scalelsd:$(pwd)/third_party/gazelle:${PYTHONPATH}"

python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)

try:
    from scalelsd.ssl.misc.train_utils import load_scalelsd_model  # C5 quality_first
    from scalelsd.ssl.models.detector import ScaleLSD
    print("scalelsd(path used in C5): OK")
except Exception as e:
    print("scalelsd(path used in C5): WARN", e)

try:
    from gazelle.model import get_gazelle_model  # C6 quality_first
    print("gazelle(path used in C6): OK")
except Exception as e:
    print("gazelle(path used in C6): WARN", e)

import paddle
from paddleocr import TextDetection  # C4 quality_first
print("paddle:", paddle.__version__)
print("TextDetection(path used in C4): OK")
PY
```

`ModuleNotFoundError: No module named 'paddle'`가 나오면:
- `paddleocr`만 설치되고 `paddlepaddle-gpu`가 실제로 설치/로딩되지 않은 상태입니다.
- 최신 `install_features_deps_torch251_cu121_stable.sh`는 이 경우 자동 재설치 후 `import paddle`를 강제 검증하고, 실패 시 즉시 종료하도록 보강되어 있습니다.

4) (선택) 설치 정책 토글
```bash
# 기본값은 1. 설치를 스킵하려면 0으로 설정 후 installer 실행.
export ENABLE_C4_OCR=1
export ENABLE_SCALELSD=1
export ENABLE_GAZELLE=1
export ENABLE_QWEN3_VL=0
export QWEN3_TRY_PYPI_LATEST=1
# GitHub 차단 환경이면 0 권장 (PyPI/사내 미러만 사용)
export QWEN3_ALLOW_GITHUB_FALLBACK=0
```

5) (권장) Qwen3-VL 전용 별도 환경 준비 (Stage-10 전용)
```bash
# 5-1) 별도 venv 생성
python3 -m venv /group-volume/jaden.ju/Venvs/qwen3_vlm

# 5-2) Qwen3 runtime-only 설치 (파이프라인 deps는 건너뜀)
QWEN3_RUNTIME_ONLY=1 \
QWEN3_ALLOW_GITHUB_FALLBACK=0 \
PYTHON=/group-volume/jaden.ju/Venvs/qwen3_vlm/bin/python \
bash src/scripts/install_features_deps_torch251_cu121_stable.sh
```

참고: C4를 수동 재설치해야 하는 경우에만 아래를 별도 실행하세요.
```bash
pip install -U "jinja2>=3.1.4"
pip install --no-cache-dir -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ paddlepaddle-gpu==3.3.0
pip install -U "paddleocr>=3.0.0"
```

---

## 3. 전체 파이프라인 (권장 운영 경로)

권장: `run_phaseA_to_teacher_e2e.sh` 하나로 실행

### 3.1 로컬(가상환경 자동 활성화) 기본 실행
```bash
DATANAME=10K_local
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 0 \
  --data_dir data/SSTK/${DATANAME} \
  --bucket sstk_100 \
  --precompute_mode unified \
  --export_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --prefer_curated_images 1 \
  --run_tag v17_local
```

### 3.2 서버 기본 실행
```bash
DATANAME=10K
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/${DATANAME} \
  --bucket sstk_100 \
  --precompute_mode unified \
  --export_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --prefer_curated_images 1 \
  --run_tag v17_server
```

### 3.2.1 서버 멀티 GPU 효율 실행(No-Ray 기본)
필터 결과가 이미 있을 때
```bash
DATANAME=10K
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/${DATANAME} \
  --run_filter 0 \
  --run_candidates 0 \
  --run_c1 1 --run_c2 1 --run_c3 1 --run_c4 1 --run_c3_enrich 1 --run_c5 1 --run_merge 1 \
  --run_teacher 1 \
  --precompute_mode unified \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --prefer_curated_images 1 \
  --extract_mode auto \
  --extract_gpu_ids ${GPU_IDS} \
  --num_workers ${N_WORKERS} \
  --use_real_expensive 1 \
  --teacher_multi_gpu 1 \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS} \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 512 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 1 \
  --public_teacher_download_weights 1 \
  --public_teachers gaic,cacnet,cgs \
  --public_gaic_weight_path weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth \
  --public_teacher_max_images -1 \
  --skip_existing 1 \
  --run_tag e2e_260227_r0 \
   | tee src/scripts/logs/run_phaseA_to_teacher_e2e_260227_r0.log
```


### 3.3 필터 결과가 이미 있을 때(재실행 시간 단축)
```bash
DATANAME=10K
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/${DATANAME} \
  --run_filter 0 \
  --skip_existing 1 \
  --precompute_mode unified \
  --run_tag rerun1
```

Filter만 재실행하면서 태그 임베딩 멀티 GPU를 강제하려면:
```bash
DATANAME=10K
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/${DATANAME} \
  --run_filter 1 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_c6 0 --run_merge 0 \
  --run_candidates 0 --run_teacher 0 --run_vlm_teacher 0 \
  --filter_require_train_match 1 \
  --filter_tag_embed_multi_gpu 1 \
  --filter_tag_embed_gpu_ids ${GPU_IDS} \
  --filter_tag_embed_batch_size 128 \
  --filter_tag_embed_chunk_size 0 \
  --filter_tag_embed_device auto \
  --skip_existing 0 \
  --run_tag filter_only_tag_mgpu
```

주의:
- `filter` 단계 캐시(`tag_cat_probs_cache_*.pkl`, `df_mapped_cache_*.parquet`)가 남아 있으면 태그 임베딩 단계가 skip될 수 있습니다.
- 실제 재임베딩을 강제하려면 `data/SSTK/<DATANAME>/cache/filter/`의 관련 캐시를 정리한 뒤 실행하세요.

`5.5 공개 Teacher 추론/변환(설치 포함)`까지 같은 실행에서 자동 적용하려면:
```bash
DATANAME=10K
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/${DATANAME} \
  --run_filter 0 \
  --precompute_mode unified \
  --prefer_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 1 \
  --public_teacher_download_weights 1 \
  --public_teachers gaic,cacnet,cgs \
  --public_gaic_weight_path weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth \
  --public_infer_multi_gpu 1 \
  --public_infer_gpu_ids ${GPU_IDS} \
  --public_infer_num_workers ${N_WORKERS} \
  --public_teacher_max_images -1 \
  --use_real_expensive 1 \
  --teacher_multi_gpu 1 \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS} \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 128 \
  --skip_existing 0 \
  --run_tag e2e_260227_r0 \
   | tee src/scripts/logs/run_phaseA_to_teacher_e2e_260227_r0.log
```
주의: `--public_infer_multi_gpu`는 `-1|0|1`만 유효하며, GPU 개수는 `--public_infer_gpu_ids`/`--public_infer_num_workers`로 지정합니다.

### 3.4 Real Expensive(미학+crop 임베딩 cosine) 활성화
```bash
DATANAME=10K
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/${DATANAME} \
  --run_filter 0 \
  --precompute_mode unified \
  --use_real_expensive 1 \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 12 \
  --run_tag real_exp
```
참고:
- `run_c1=-1` 기본값이면 `--use_real_expensive 1`에서 C1이 자동 활성화됩니다.
- `precompute_mode=unified`에서는 C1도 동일 1-pass precompute 결과(JSONL)에 함께 저장됩니다.

### Teacher Scorer 단계 재실행
```bash
DATANAME=10K_local
GPU_IDS=0,1,2
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/${DATANAME} \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_candidates 0 \
  --run_teacher 1 \
  --prefer_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --use_real_expensive 1 \
  --teacher_multi_gpu 1 \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS} \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 128 \
  --skip_existing 0 \
  --run_tag rerun1_public_e2e
```

### 3.5 GPU 서버 smoke/full 예시
- smoke (후보/teacher만 500장):
```bash
DATANAME=10K
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/${DATANAME} \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_candidates 1 --run_teacher 1 \
  --use_real_expensive 1 \
  --align_device cuda --aesthetic_device cuda --exp_batch_size 12 \
  --max_images 500 \
  --run_tag real_exp_500
```
- full (전량): `--max_images 0`

### 3.6 처음부터 끝까지(명시형 풀 옵션 템플릿)
선행 조건: `2.1 초기 환경 세팅(품질최우선 전체 파이프라인용)`을 완료한 상태에서 실행하세요.

아래 템플릿은 **현재 구현된 품질 최우선 모듈(C4=PP-OCRv5 server, C5=ScaleLSD 우선, C6=Gaze-LLE/Gazelle 우선)**까지 포함한 end-to-end 실행 템플릿입니다.

중요:
- `--run_filter 0`은 `data/SSTK/${DATANAME}/filtered_sstk_*.parquet`가 이미 있을 때만 사용하세요.
- `DATANAME`이 신규 디렉토리이거나 필터 산출물이 없으면 `--run_filter 1`로 바꿔야 합니다.
- `--run_filter 0`일 때는 `--curated_pool_size`, `--top_percentile`는 참고용으로만 남고 실제 필터링 재실행에는 쓰이지 않습니다.
- `--skip_existing 1`은 새 `RUN_TAG`로 1회 실행할 때 권장입니다. 같은 `RUN_TAG`로 산출물을 다시 검증하거나 덮어쓸 때는 `--skip_existing 0`이 더 안전합니다.
- 현재 `run_phaseA_to_teacher_e2e.sh`는 teacher/QA/detailed report뿐 아니라 학습 데이터 생성기(`pairwise/listwise/decision/checklist/regression`)까지 자동 생성할 수 있습니다.

품질 최우선 백엔드 강제(권장):
```bash
# C4 OCR
export C4_BACKEND=ppocrv5_server
export C4_PPOCR_SERVER_MODEL=PP-OCRv5_server_det
export C4_PPOCR_DEVICE=gpu:0

# C5 Horizon/Leveling
export C5_BACKEND=scalelsd
export C5_SCALELSD_AUTO_DOWNLOAD=1

# C6 Gaze/HeadPose
export C6_BACKEND=gazelle
export C6_GAZELLE_DEVICE=cuda
```

```bash
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

POOL_SIZE=100
DATANAME=Test_${POOL_SIZE}
RUN_TAG=260316_r2

#POOL_SIZE=10000
#DATANAME=Full_${POOL_SIZE}
#RUN_TAG=260316_r1

bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --bucket sstk_100 \
  --data_dir data/SSTK/${DATANAME} \
  --server_mode 1 \
  --tar_dir /sstk/20230916/sstk_100 \
  --run_filter 0 \
  --curated_pool_size ${POOL_SIZE} \
  --top_percentile 0.2 \
  --filter_require_train_match 1 \
  --filter_tag_embed_multi_gpu 1 \
  --filter_tag_embed_gpu_ids ${GPU_IDS} \
  --filter_tag_embed_batch_size 128 \
  --filter_tag_embed_chunk_size 0 \
  --filter_tag_embed_device auto \
  --export_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --prefer_curated_images 1 \
  --skip_existing 1 \
  --precompute_mode unified \
  --run_c1 -1 \
  --run_c2 1 --run_c3 1 --run_c4 1 --run_c5 1 --run_c6 1 --run_c3_enrich 1 --run_merge 1 \
  --run_subject_routing 1 \
  --subject_routing_top_n 5 \
  --subject_routing_union_top_m 3 \
  --subject_routing_allow_det_proxy 1 \
  --extract_mode auto \
  --extract_multi_gpu 1 \
  --extract_gpu_ids ${GPU_IDS} \
  --num_workers ${N_WORKERS} \
  --extract_priority quality_first \
  --c5_priority quality_first \
  --batch_size 512 \
  --c3_person_verify_strict 1 \
  --run_candidates 1 \
  --cand_ar_list FREE,1:1,9:16,16:9,3:4,4:3 \
  --use_actual_image_size 1 \
  --strict_actual_size 1 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 1 \
  --public_teacher_download_weights 1 \
  --public_teachers gaic,cacnet,cgs \
  --public_teacher_max_images -1 \
  --public_teacher_device auto \
  --public_gaic_weight_path weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth \
  --public_infer_multi_gpu 1 \
  --public_infer_gpu_ids ${GPU_IDS} \
  --public_infer_num_workers ${N_WORKERS} \
  --public_infer_skip_on_oom 1 \
  --public_infer_fallback_cpu_on_oom 1 \
  --public_infer_fallback_cpu_max_images 3 \
  --public_infer_skip_if_fallback_failed 1 \
  --run_teacher 1 \
  --use_real_expensive 1 \
  --cheap_top_m 30 \
  --top_k 5 \
  --tau_div 0.75 \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 512 \
  --run_component_viz 1 \
  --component_viz_num_samples 120 \
  --component_viz_out_dir data/SSTK/${DATANAME}/artifacts/precompute/visualizations/components_${RUN_TAG} \
  --run_qa 1 \
  --run_viz 1 \
  --num_viz 120 \
  --run_vlm_teacher 0 \
  --vlm_backend qwen25_vl \
  --vlm_model_id Qwen/Qwen3-VL-4B-Instruct \
  --vlm_device auto \
  --vlm_top_m 12 \
  --vlm_top_k 5 \
  --vlm_fallback_backend heuristic \
  --run_tag ${RUN_TAG} \
  | tee src/scripts/logs/run_phaseA_to_teacher_${DATANAME}_${RUN_TAG}.log
```

처음부터 완전 신규 실행이면 위 명령에서 아래 한 줄만 바꿔서 시작하세요.
```bash
  --run_filter 1 \
```

자동 생성 결과(teacher score 이후 후처리):
- `train_pairwise.jsonl`
- `train_listwise.jsonl`
- `train_decision.jsonl`
- `train_checklist.jsonl`
- `train_regression.jsonl`
- `qa_summary.json`
- `TRAINING_DATA_REPORT_KO.md`

최소 확인 포인트:
- teacher score jsonl: `data/SSTK/${DATANAME}/artifacts/teacher/scores/teacher_scores_ar_${RUN_TAG}.jsonl`
- detailed report: `data/SSTK/${DATANAME}/artifacts/reports/${RUN_TAG}_detailed/REPORT_DRAFT_KO.md`
- training label QA: `data/SSTK/${DATANAME}/artifacts/training_labels/${RUN_TAG}/qa_summary.json`
- training label report: `data/SSTK/${DATANAME}/artifacts/training_labels/${RUN_TAG}/TRAINING_DATA_REPORT_KO.md`

권장 재실행 명령 기존 산출물을 덮어쓰는 가장 짧은 재실행입니다. 영향 범위는 subject routing -> teacher -> QA -> viz만입니다.
```bash
POOL_SIZE=100
DATANAME=Test_${POOL_SIZE}
RUN_TAG=260309_r0
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --bucket sstk_100 \
  --data_dir data/SSTK/${DATANAME} \
  --server_mode 1 \
  --tar_dir /sstk/20230916/sstk_100 \
  --run_filter 0 \
  --skip_existing 0 \
  --precompute_mode unified \
  --run_c1 0 \
  --run_c2 0 --run_c3 0 --run_c4 0 --run_c5 0 --run_c6 0 --run_c3_enrich 0 --run_merge 0 \
  --run_subject_routing 1 \
  --subject_routing_top_n 5 \
  --subject_routing_union_top_m 3 \
  --subject_routing_allow_det_proxy 1 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 1 \
  --public_teacher_download_weights 1 \
  --public_teachers gaic,cacnet,cgs \
  --public_teacher_max_images -1 \
  --public_teacher_device auto \
  --public_gaic_weight_path weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth \
  --public_infer_multi_gpu 1 \
  --public_infer_gpu_ids ${GPU_IDS} \
  --public_infer_num_workers ${N_WORKERS} \
  --public_infer_skip_on_oom 1 \
  --public_infer_fallback_cpu_on_oom 1 \
  --public_infer_fallback_cpu_max_images 3 \
  --public_infer_skip_if_fallback_failed 1 \
  --run_candidates 1 \
  --run_teacher 1 \
  --use_real_expensive 1 \
  --cheap_top_m 30 \
  --top_k 5 \
  --tau_div 0.75 \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 512 \
  --save_public_teacher_ref_eval 1 \
  --run_component_viz 0 \
  --run_qa 1 \
  --run_viz 1 \
  --num_viz 120 \
  --run_vlm_teacher 0 \
  --run_tag ${RUN_TAG} \
  | tee src/scripts/logs/run_phaseA_to_teacher_${DATANAME}_${RUN_TAG}.log
```

Filter 태그 임베딩 멀티 GPU 옵션 설명:
- `--filter_tag_embed_multi_gpu -1|0|1`
  - `-1`: auto (GPU 2개 이상이면 멀티 GPU, 아니면 단일 디바이스)
  - `0`: 단일 디바이스 강제
  - `1`: 멀티 GPU 강제
- `--filter_tag_embed_gpu_ids`: 사용할 GPU 목록(CSV). 예: `0,1,2,3`
- `--filter_tag_embed_batch_size`: 태그 임베딩 배치 크기
- `--filter_tag_embed_chunk_size`: 멀티프로세스 chunk 크기(`0`이면 auto)
- `--filter_tag_embed_device`: 단일 디바이스 모드에서의 모델 로드 디바이스(`auto|cuda|cuda:0|cpu`)

참고:
- 로그에서 아래 문구가 보이면 멀티 GPU 경로가 적용된 상태입니다.
  - `[tag-embed] runtime ... multi_gpu=1 ...`
  - `[tag-embed] Encoding dataset unique tags with multi-GPU ...`
- 멀티 GPU 실행 중 오류가 나면 자동으로 단일 디바이스 encode로 fallback 됩니다.
- `tag_cat_probs_cache_*.pkl`가 이미 유효하면 태그 임베딩 단계 자체가 skip됩니다.

정리:
- 위 `3.6` 템플릿 1회 실행으로 `Filter -> Precompute(C1~C6) -> Subject Routing -> Candidate -> Teacher -> QA -> Detailed Report -> Training Labels`까지 수행됩니다.
- `--run_component_viz 1`을 켜면 precompute 시각화가 자동 생성되며, 출력은 기본적으로 `data/SSTK/<DATANAME>/artifacts/precompute/visualizations/components_<run_tag>/`에 저장됩니다.
- 학습 데이터 생성 단계만 끄고 싶으면 `--run_training_labels 0`을 명시하세요.
- precompute 시각화 산출물은 `original/`, `c2_seg/`, `c3_pose/`, `c4_ocr/`, `c5_geom/`, `c6_gaze/`, `combined_all/` 및 `viz_overview.json`입니다.
- 기본 candidate AR 세트는 `FREE`를 포함합니다. (`--cand_ar_list`로 조정 가능)
- `teacher_scores_jsonl`의 `results_by_ar`에 `FREE` 키가 함께 생성되며, `--vlm_target_ar all`이면 Stage10도 `FREE`를 포함해 라벨을 생성합니다.
- `FREE`는 AR 하드제약 타겟이 아니라 AR 비조건부 후보 모드입니다. (다중 AR 후보를 생성하고 scorer에서 FREE 전용 prior만 약하게 적용)

실행 후 품질 최우선 모듈 적용 여부 점검:
```bash
DATANAME=10K
export DATANAME
python - <<'PY'
import json
import os
from collections import Counter

dname = os.environ.get("DATANAME", "10K")
path = f"data/SSTK/{dname}/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl"
c4 = Counter(); c5 = Counter(); c6 = Counter()
with open(path, "r", encoding="utf-8") as f:
    for ln in f:
        d = json.loads(ln)
        m4 = (d.get("c4_ocr_meta", {}) or {}).get("method", "none")
        m5 = (d.get("c5_geom", {}) or {}).get("horizon_roll", {}).get("method", "none")
        m6 = (d.get("c6_gaze", {}) or {}).get("method", "none")
        c4[m4] += 1
        c5[m5] += 1
        c6[m6] += 1
print("c4 methods:", dict(c4))
print("c5 methods:", dict(c5))
print("c6 methods:", dict(c6))
PY
```

판정 기준:
- C4: `ppocrv5_server_det` 비중이 주류인지 확인
- C5: `scalelsd_*` 또는 `scalelsd_ransac` 계열이 사용되는지 확인
- C6: `gazelle_gaze_lle`가 의미 있게 관측되는지 확인(인물 없는 샘플은 `skipped` 가능)
- fallback(`legacy_fallback`, `houghp_fallback`, `proxy_fallback`)이 과도하면 모델/체크포인트 경로를 점검

FREE 파이프라인 반영 여부 빠른 점검:
```bash
python - <<'PY'
import json
from collections import Counter

cand = "data/SSTK/10K_local/artifacts/candidates/candidates_ar_<run_tag>.jsonl"
teach = "data/SSTK/10K_local/artifacts/teacher/scores/teacher_scores_ar_<run_tag>.jsonl"
vlm = "data/SSTK/10K_local/artifacts/vlm_teacher/labels/crop_label_v1_<run_tag>.jsonl"

with open(cand, "r", encoding="utf-8") as f:
    ck = Counter()
    for ln in f:
        for k in (json.loads(ln).get("candidates_by_ar") or {}).keys():
            ck[k] += 1
print("candidate keys:", dict(ck))

with open(teach, "r", encoding="utf-8") as f:
    tk = Counter()
    for ln in f:
        rb = ((json.loads(ln).get("teacher_scorer") or {}).get("results_by_ar") or {})
        for k in rb.keys():
            tk[k] += 1
print("teacher keys:", dict(tk))

with open(vlm, "r", encoding="utf-8") as f:
    vk = Counter(str(json.loads(ln).get("target_ar", "")) for ln in f)
print("vlm target_ar:", dict(vk))
PY
```
위 스니펫의 `<run_tag>`는 실제 실행 태그(예: `full_260226`)로 바꿔서 사용합니다.

### 3.7 filtered parquet만 있고 images가 없을 때: images만 생성 후 4.2 실행

1) `filtered parquet`에서 curated images만 별도 추출:
Local
```bash
DATANAME=10K
python3  src/scripts/export_curated_images_from_parquet.py \
  --input_parquet data/SSTK/${DATANAME}/filtered_sstk_100.parquet \
  --tar_dir /media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100 \
  --output_dir data/SSTK/${DATANAME}/images \
  --bucket sstk_100 \
  --skip_existing 1 \
  --num_workers 0 \
  --auto_workers_cap 8
```
Server
```bash
DATANAME=10K
python3 src/scripts/export_curated_images_from_parquet.py \
  --input_parquet data/SSTK/${DATANAME}/filtered_sstk_100.parquet \
  --tar_dir /sstk/20230916/sstk_100 \
  --output_dir data/SSTK/${DATANAME}/images \
  --bucket sstk_100 \
  --skip_existing 1 \
  --num_workers 0 \
  --auto_workers_cap 8
```

전체 CPU 코어를 강제로 모두 사용하려면(`I/O 포화 가능`):
```bash
DATANAME=10K
python3 src/scripts/export_curated_images_from_parquet.py \
  --input_parquet data/SSTK/${DATANAME}/filtered_sstk_100.parquet \
  --tar_dir /sstk/20230916/sstk_100 \
  --output_dir data/SSTK/${DATANAME}/images \
  --bucket sstk_100 \
  --skip_existing 0 \
  --num_workers -1
```

옵션 설명:
- `--num_workers 0`: auto-balanced(기본). `min(CPU코어수, auto_workers_cap, tar job 수)`를 사용
- `--num_workers -1`: 모든 CPU 코어 사용
- `--auto_workers_cap`: auto 모드 상한(기본 `8`)


2) 4.2(Perception Precompute)만 실행:
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 0 \
  --data_dir data/SSTK/${DATANAME} \
  --run_filter 0 \
  --precompute_mode unified \
  --prefer_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --run_candidates 0 \
  --run_teacher 0 \
  --skip_existing 0 \
  --run_tag precompute_only
```

### 3.8 Stage 10(VLM Teacher)만 단독 실행

운영 TL;DR:
- 기본 모델: `Qwen/Qwen3-VL-4B-Instruct` (호환성 이슈 시 2.5-VL-3B로 임시 전환)
- 필수 입력: `teacher_scores_ar_<run_tag>.jsonl` + `data/SSTK/<DATANAME>/images`
- 권장 실행: `--run_vlm_teacher 1 --run_teacher 0 --skip_existing 0 --vlm_save_raw_response 1`
- 멀티 GPU: `--vlm_multi_gpu 1 --vlm_gpu_ids ... --vlm_num_workers ...`
- OOM 안전장치: `--vlm_fallback_cpu_on_oom 1 --vlm_fallback_cpu_max_images 3 --vlm_skip_if_fallback_failed 1`
- 품질 게이트 1: `summary_json`에서 `task_written`, `fallback_used`, `task_skipped` 확인
- 품질 게이트 2: labels의 `explanations.short/long` 유니크 개수가 `1`이 아닌지 확인
- 품질 게이트 3: `debug/raw_responses` 타입 분포(`selected_crops` 등) 점검
- 결과 검증 시각화: `3.8.1`의 `run_visualize_vlm_teacher.sh` 실행
- 리포트 사용 전: Stage 10 재실행 산출물 기준으로 `REPORT_DRAFT_KO.md` 갱신
- 상세 원인/검증 스크립트: `3.8.2` 참고
- 단계별 재실행 절차: `3.8.3` 체크리스트 참고

### 3.8.0 Qwen3 전용 환경 래퍼 실행(권장)

`C1~C6/Teacher`와 `Qwen3-VL` 간 `transformers` 버전 충돌을 피하려면, Stage-10은 전용 래퍼를 사용하세요.

신규 래퍼:
- `src/scripts/run_vlm_teacher_labeler_qwen3_env.sh`
- 동작: Qwen3 전용 venv 검사/부트스트랩(`QWEN3_RUNTIME_ONLY`) 후, `run_vlm_teacher_labeler.sh`를 해당 venv python으로 강제 실행

예시:
```bash
DATANAME=10K_local
RUNTAG=rerun1_public_e2e_subject_mode
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

bash src/scripts/run_vlm_teacher_labeler_qwen3_env.sh \
  --qwen3_venv_path /group-volume/jaden.ju/Venvs/qwen3_vlm \
  --qwen3_allow_github_fallback 0 \
  --teacher_scores_jsonl data/SSTK/${DATANAME}/artifacts/teacher/scores/teacher_scores_ar_${RUNTAG}.jsonl \
  --image_dir data/SSTK/${DATANAME}/images \
  --output_jsonl data/SSTK/${DATANAME}/artifacts/vlm_teacher/labels/crop_label_v1_${RUNTAG}.jsonl \
  --output_meta_jsonl data/SSTK/${DATANAME}/artifacts/vlm_teacher/meta/meta_norm_v1_${RUNTAG}.jsonl \
  --summary_json data/SSTK/${DATANAME}/artifacts/vlm_teacher/summary/vlm_teacher_summary_${RUNTAG}.json \
  --backend qwen25_vl \
  --model_id Qwen/Qwen3-VL-4B-Instruct \
  --device auto \
  --top_m 12 \
  --top_k 5 \
  --multi_gpu 1 \
  --gpu_ids ${GPU_IDS} \
  --num_workers ${N_WORKERS}
```

참고:
- 최초 1회 강제 재설치가 필요하면 `--reinstall_qwen3_runtime 1` 추가
- `run_phaseA_to_teacher_e2e.sh`는 `--run_vlm_teacher 0`으로 두고, Stage-10만 래퍼로 분리 실행 권장

Teacher Scorer 결과가 이미 있을 때(재추론 없이 10단계만 실행):
- 기본 권장 모델: `Qwen/Qwen3-VL-4B-Instruct`
- 환경 호환성 이슈 시 임시 대안: `--vlm_model_id Qwen/Qwen2.5-VL-3B-Instruct`
```bash
DATANAME=10K_local
BASE_TAG=rerun1_public_e2e
SM_TAG=${BASE_TAG}_subject_mode
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/${DATANAME} \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_candidates 0 \
  --run_teacher 0 \
  --run_vlm_teacher 1 \
  --prefer_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --vlm_backend qwen25_vl \
  --vlm_model_id Qwen/Qwen3-VL-4B-Instruct \
  --vlm_device auto \
  --vlm_top_m 12 \
  --vlm_top_k 5 \
  --vlm_fallback_backend heuristic \
  --vlm_multi_gpu 1 \
  --vlm_gpu_ids ${GPU_IDS} \
  --vlm_num_workers ${N_WORKERS} \
  --vlm_save_raw_response 1 \
  --vlm_strict_backend_init 1 \
  --skip_existing 0 \
  --run_tag ${SM_TAG} \
   | tee src/scripts/logs/run_phaseA_to_teacherr_10K_local_stage10_${SM_TAG}.log
```
```bash
DATANAME=10K
BASE_TAG=e2e_260227_r0
SM_TAG=${BASE_TAG}_subject_mode
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/${DATANAME} \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_candidates 0 \
  --run_teacher 0 \
  --run_vlm_teacher 1 \
  --prefer_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --vlm_backend qwen25_vl \
  --vlm_model_id Qwen/Qwen3-VL-4B-Instruct \
  --vlm_device auto \
  --vlm_top_m 12 \
  --vlm_top_k 5 \
  --vlm_fallback_backend heuristic \
  --vlm_multi_gpu 1 \
  --vlm_gpu_ids ${GPU_IDS} \
  --vlm_num_workers ${N_WORKERS} \
  --vlm_save_raw_response 1 \
  --vlm_strict_backend_init 1 \
  --skip_existing 0 \
  --run_tag ${SM_TAG} \
   | tee src/scripts/logs/run_phaseA_to_teacher_10K_stage10_${SM_TAG}.log
```

OOM 대응 권장:
- GPU OOM 시 CPU fallback 소량 검증: `--vlm_fallback_cpu_on_oom 1 --vlm_fallback_cpu_max_images 3`
- CPU fallback도 실패하면 skip 지속: `--vlm_skip_if_fallback_failed 1`

### 3.8.1 Stage 10 결과 검증 시각화(Overlay + Analytics)

Stage 10 산출물(`crop_label_v1*.jsonl`)이 생성된 뒤, 아래 명령으로 검증용 시각화/분석을 생성할 수 있습니다.

10K_local (`run_tag=rerun1_public_e2e`) 예시:
```bash
DATANAME=10K_local
RUNTAG=rerun1_public_e2e_subject_mode
bash src/scripts/run_visualize_vlm_teacher.sh \
  data/SSTK/${DATANAME}/artifacts/vlm_teacher/labels/crop_label_v1_${RUNTAG}.jsonl \
  data/SSTK/${DATANAME}/filtered_sstk_100.parquet \
  /sstk/20230916/sstk_100 \
  data/SSTK/${DATANAME}/artifacts/vlm_teacher/visualizations/vlm_teacher_${RUNTAG} \
  --teacher_scores_jsonl data/SSTK/${DATANAME}/artifacts/teacher/scores/teacher_scores_ar_${RUNTAG}.jsonl \
  --image_dir data/SSTK/${DATANAME}/images \
  --image_ids_file data/SSTK/${DATANAME}/artifacts/reports/${RUNTAG}/sample_ids_supercat12.txt \
  --target_ar all \
  --num_samples 0 \
  --analytics_use_full_labels 1 \
  --server_mode 1
```

10K (`run_tag=e2e_260227_r0`) 예시:
```bash
DATANAME=10K
RUNTAG=e2e_260227_r0
bash src/scripts/run_visualize_vlm_teacher.sh \
  data/SSTK/${DATANAME}/artifacts/vlm_teacher/labels/crop_label_v1_${RUNTAG}.jsonl \
  data/SSTK/${DATANAME}/filtered_sstk_100.parquet \
  /sstk/20230916/sstk_100 \
  data/SSTK/${DATANAME}/artifacts/vlm_teacher/visualizations/vlm_teacher_${RUNTAG} \
  --teacher_scores_jsonl data/SSTK/${DATANAME}/artifacts/teacher/scores/teacher_scores_ar_${RUNTAG}.jsonl \
  --image_dir data/SSTK/${DATANAME}/images \
  --image_ids_file data/SSTK/${DATANAME}/artifacts/reports/${RUNTAG}/sample_ids_supercat12.txt \
  --target_ar all \
  --num_samples 0 \
  --analytics_use_full_labels 1 \
  --server_mode 1
```

주요 출력:
- overlay: `data/SSTK/<DATANAME>/artifacts/vlm_teacher/visualizations/vlm_teacher_<run_tag>/by_ar/*`
- analytics: `.../analytics/vlm_analytics_summary.json`, `vlm_decision_type_distribution.png`, `vlm_selected_k_distribution.png` 등
- 요약: `.../viz_overview.json`

### 3.8.2 Stage 10 explanations 고정 문구 이슈 원인/검증 포인트

빠른 운영 체크는 `3.8 Stage 10(VLM Teacher)만 단독 실행`의 `운영 TL;DR`를 먼저 확인하세요.

증상:
- `crop_label_v1*.jsonl`의 `explanations.short/long`가 전 레코드에서 동일 문구로 반복
- 예: `Top-K 후보를 선택하고 체크리스트를 검증했습니다.`

원인(코드/산출물 기준):
- 주 원인은 fallback 자체가 아니라, 구버전 파서/정규화 로직의 스키마 흡수 한계
- 실제 Qwen raw 응답은 `selected_topk` 정식 스키마 대신 `selected_crops`/`selected_candidates`/`crop_label` 변형 포맷이 다수
- 구버전에서 이 케이스를 충분히 매핑하지 못해 정규화 시 설명이 기본 문구로 수렴

보완 반영(현재 코드):
- `src/vlm_teacher_labeler.py`
- `_extract_first_json_object`: 입력 echo보다 실제 출력 객체를 우선 선택하도록 점수화 추출
- `_extract_model_selected_items`: `selected_crops`, `selected_candidates`, `crop_labels`, `crop_label`까지 매핑
- bbox만 있는 경우 IoU 기반 candidate_id 매칭으로 복원
- `_build_auto_explanations`: 모델 설명 누락 시 top1/source/final/delta/tau/baseline 기반 동적 설명 생성
- fallback 경로도 정적 문구 대신 동적 설명 사용

검증 포인트(운영 필수):
- `summary_json`에서 `fallback_used`가 과도하지 않은지 확인
- `debug/raw_responses`에서 응답 타입 분포(`selected_crops` 등) 확인
- 최종 labels에서 `explanations` 유니크 개수가 1이 아닌지 확인

유니크 개수 빠른 점검(10K/10K_local 공통):
```bash
python - <<'PY'
import json
from collections import Counter
paths=[
  "data/SSTK/10K/artifacts/vlm_teacher/labels/crop_label_v1_e2e_260227_r0.jsonl",
  "data/SSTK/10K_local/artifacts/vlm_teacher/labels/crop_label_v1_rerun1_public_e2e.jsonl",
]
for p in paths:
    c1=Counter(); c2=Counter(); n=0
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            o=json.loads(line); exp=o.get("explanations") or {}
            c1[(exp.get("short") or "").strip()] += 1
            c2[(exp.get("long") or "").strip()] += 1
            n += 1
    print(p, "rows=", n, "uniq_short=", len(c1), "uniq_long=", len(c2))
PY
```

raw 응답 타입 분포 점검(예: 10K):
```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from collections import Counter
from vlm_teacher_labeler import _extract_first_json_object, _classify_parsed_object
root=Path("data/SSTK/10K/artifacts/vlm_teacher/debug/vlm_teacher_e2e_260227_r0")
ctr=Counter()
for p in root.glob("shard_*/raw_responses/*.txt"):
    t=p.read_text(encoding="utf-8",errors="ignore").strip()
    if not t:
        ctr["empty"] += 1
        continue
    obj=_extract_first_json_object(t)
    ctr["parse_none" if obj is None else _classify_parsed_object(obj)] += 1
print(dict(ctr))
PY
```

### 3.8.3 Stage 10 재실행 체크리스트(필수)

실행 옵션 요약은 `3.8`의 `운영 TL;DR`, 원인/검증 배경은 `3.8.2`를 함께 참조하세요.

1. 코드 최신화
- `src/vlm_teacher_labeler.py`가 최신인지 확인
- 테스트 수행: `PYTHONPATH=src python -m pytest -q tests/test_vlm_teacher_labeler.py`

2. 입력 준비 확인
- Teacher scores 존재 확인
- curated image cache 존재 확인: `data/SSTK/<DATANAME>/images`
- 기존 Stage 10 출력 덮어쓸지(`--skip_existing 0`) 결정

3. Stage 10 재실행
- `3.8 Stage 10(VLM Teacher)만 단독 실행` 템플릿 사용
- 권장: `--vlm_save_raw_response 1` 유지(디버그/검증용)

4. 재실행 후 품질 게이트
- `summary_json`에서 `task_written == expected task 수` 확인
- `counts.fallback_used`/`counts.task_skipped` 비정상 급증 여부 확인
- `explanations` 유니크 개수 확인(위 유니크 점검 스크립트)
- 필요 시 `3.8.1`로 visualization 재생성 후 샘플 점검

5. 리포트 반영
- Stage 10 결과를 사용하는 리포트(`REPORT_DRAFT_KO.md`)는 재실행 산출물 기준으로 갱신
- 구버전 labels 기반 리포트는 explanations/통계가 왜곡될 수 있으므로 재생성 권장

### 3.8.4 프록시 차단 서버용 오프라인 전송(정상 서버 -> 문제 서버)

`Qwen/Qwen3-VL-4B-Instruct`를 프록시 차단 환경에서 실행할 때는,
- 모델 파일(허깅페이스 스냅샷)
- Qwen3 런타임 심볼(`Qwen3VLForConditionalGeneration`)을 포함한 오프라인 wheelhouse
를 같이 옮겨야 안정적으로 동작합니다.

아래 스크립트를 사용하세요:
- `src/scripts/transfer_qwen3_vl_offline.sh`

1) 정상 동작 서버에서 준비 + 번들 생성(프로젝트 경로 기준)
```bash
cd /media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping

# (선택) venv 활성화
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

bash src/scripts/transfer_qwen3_vl_offline.sh prepare
bash src/scripts/transfer_qwen3_vl_offline.sh pack
```

생성물:
- 번들 파일: `artifacts/offline_qwen3_vl/qwen3_vl_4b_offline_bundle.tgz`

2) 문제 서버로 번들 전송
```bash
cd /media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping
scp artifacts/offline_qwen3_vl/qwen3_vl_4b_offline_bundle.tgz <user>@<problem-server>:/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/artifacts/offline_qwen3_vl/
```

3) 문제 서버에서 오프라인 설치
```bash
cd /media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping

# (선택) venv 활성화
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

bash src/scripts/transfer_qwen3_vl_offline.sh install
```

설치 결과:
- 로컬 모델 경로: `artifacts/models/Qwen3-VL-4B-Instruct`
- 오프라인 pip 설치 후 `has_qwen3=True` 검증 수행

4) Stage 10 실행 시 오프라인 모드 + 로컬 모델 경로 사용
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
```

기존 명령에서 아래만 교체:
```bash
--vlm_model_id artifacts/models/Qwen3-VL-4B-Instruct
```

참고:
- `HEAD https://huggingface.co/... timed out` 에러는 오프라인 모드 + 로컬 모델 경로로 회피합니다.
- `transformers==4.44.2` 계열의 `has_qwen3=False` 에러는 스크립트의 wheelhouse 설치 단계로 해결합니다.

### 3.9 기존 산출물 재활용: Subject-mode 보완 + Visualization + Report 자산 생성

아래 템플릿은 이미 생성된 산출물(`filtered parquet`, `precompute`, `candidates`, `teacher`)을 재활용하여,
- subject-mode 보완(`run_subject_routing`) 반영 재실행
- 컴포넌트/teacher 시각화 재생성
- report용 QA/overview/top-k 분석 자산 재생성
을 수행합니다.

권장: 기존 run_tag를 그대로 덮어쓰기보다, `*_subject_mode`처럼 별도 run_tag를 사용하세요.

공통 사전 준비(서버 예시):
```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate
```

10K_local (`base_tag=rerun1_public_e2e`) 예시:
```bash
DATANAME=10K_local
BASE_TAG=rerun1_public_e2e
SM_TAG=${BASE_TAG}_subject_mode
DATA_DIR=data/SSTK/${DATANAME}
REPORT_DIR=${DATA_DIR}/artifacts/reports/${SM_TAG}
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
# C1 전략: 0=기존 c1 재활용(권장), 1=C1 재생성(HF/OpenCLIP 다운로드 가능 환경)
RUN_C1_REBUILD=0

# 0) (선택) base report의 샘플 id/manifest 재사용
mkdir -p "${REPORT_DIR}"
cp -f "${DATA_DIR}/artifacts/reports/${BASE_TAG}/sample_ids_supercat12.txt" "${REPORT_DIR}/"
cp -f "${DATA_DIR}/artifacts/reports/${BASE_TAG}/sample_manifest_supercat12.csv" "${REPORT_DIR}/"
cp -f "${DATA_DIR}/artifacts/reports/${BASE_TAG}/sample_manifest_supercat12.json" "${REPORT_DIR}/"
cp -f "${DATA_DIR}/artifacts/reports/${BASE_TAG}/sample_summary_table.md" "${REPORT_DIR}/"

# 0.1) real expensive 사용 전제: run_c1=0일 때는 기존 c1 jsonl 필수
if [ "${RUN_C1_REBUILD}" -eq 0 ] && [ ! -f "${DATA_DIR}/artifacts/precompute/feats_c1.jsonl" ]; then
  echo "[error] missing c1 jsonl: ${DATA_DIR}/artifacts/precompute/feats_c1.jsonl"
  echo "        해결: RUN_C1_REBUILD=1로 C1 재생성(네트워크/HF 필요) 또는 --use_real_expensive 0"
  exit 1
fi

# 1) 기존 precompute를 재활용해 subject routing + candidates + teacher 재실행
# 기본값: "재활용" 경로(run_c1=0)
# 선택값: RUN_C1_REBUILD=1이면 C1 재생성 수행(HF/OpenCLIP 다운로드 필요)
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir "${DATA_DIR}" \
  --run_filter 0 \
  --run_c1 "${RUN_C1_REBUILD}" --run_c2 0 --run_c3 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_subject_routing 1 \
  --subject_routing_top_n 5 \
  --subject_routing_union_top_m 3 \
  --subject_routing_allow_det_proxy 1 \
  --run_candidates 1 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 0 \
  --public_teacher_download_weights 0 \
  --public_teachers gaic,cacnet,cgs \
  --public_gaic_weight_path weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth \
  --proposal_injection_gate 1 \
  --proposal_injection_min_rate 0.95 \
  --proposal_injection_gate_strict 1 \
  --run_teacher 1 \
  --use_real_expensive 1 \
  --teacher_multi_gpu 1 \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS} \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 128 \
  --prefer_curated_images 1 \
  --curated_image_dir "${DATA_DIR}/images" \
  --skip_existing 0 \
  --run_tag "${SM_TAG}"

# 1.1) 1단계 성공 가드: teacher score가 생성되지 않았으면 즉시 중단
if [ ! -f "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" ]; then
  echo "[error] missing teacher scores: ${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl"
  echo "        step-1 로그/에러를 먼저 해결한 뒤 step-2~4를 실행하세요."
  exit 1
fi

# 2) precompute component visualization(샘플 고정)
bash src/scripts/run_visualize_components.sh \
  "${DATA_DIR}/filtered_sstk_100.parquet" \
  /sstk/20230916/sstk_100 \
  "${DATA_DIR}/artifacts/precompute/visualizations/components_${SM_TAG}" \
  --merged_jsonl "${DATA_DIR}/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl" \
  --image_dir "${DATA_DIR}/images" \
  --image_ids_file "${REPORT_DIR}/sample_ids_supercat12.txt" \
  --draw_c2 1 --draw_c3 1 --draw_c4 1 --draw_c5 1 --draw_c6 1 --draw_combined 1 \
  --num_samples 120 \
  --server_mode 1

# 3) teacher visualization(샘플 고정)
python src/visualize_teacher_scores.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --parquet "${DATA_DIR}/filtered_sstk_100.parquet" \
  --tar_dir /sstk/20230916/sstk_100 \
  --image_dir "${DATA_DIR}/images" \
  --out_dir "${DATA_DIR}/artifacts/teacher/visualizations/teacher_scorer_${SM_TAG}" \
  --image_ids_file "${REPORT_DIR}/sample_ids_supercat12.txt" \
  --num_samples 120 \
  --target_ar all \
  --decision_filter all

# 4) report용 overview/qa/top-k analytics 재생성
python src/scripts/rebuild_teacher_overview.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --output_json "${DATA_DIR}/artifacts/teacher/overview/teacher_scores_overview_${SM_TAG}.json" \
  --output_by_ar_csv "${DATA_DIR}/artifacts/teacher/overview/teacher_scores_overview_by_ar_${SM_TAG}.csv"

python src/scripts/qa_teacher_report.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --output_json "${DATA_DIR}/artifacts/teacher/qa/teacher_scores_qa_report_${SM_TAG}.json" \
  --output_by_ar_csv "${DATA_DIR}/artifacts/teacher/qa/teacher_scores_qa_report_by_ar_${SM_TAG}.csv"

mkdir -p "${REPORT_DIR}/assets/analytics/teacher_topk"
python src/scripts/build_teacher_topk_analytics.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --output_dir "${REPORT_DIR}/assets/analytics/teacher_topk"
```

#### 3.9.1 Subject-mode 결과 재활용: Stage 10(VLM) + VLM 시각화까지

아래는 `*_subject_mode` 산출물에 Stage-10을 추가하고, VLM 결과 시각화/분석 자산까지 만드는 절차입니다.

```bash
DATANAME=10K_local
BASE_TAG=rerun1_public_e2e
SM_TAG=${BASE_TAG}_subject_mode
DATA_DIR=data/SSTK/${DATANAME}
REPORT_DIR=${DATA_DIR}/artifacts/reports/${SM_TAG}
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

# 0) 선행 산출물 확인
if [ ! -f "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" ]; then
  echo "[error] missing teacher scores for ${SM_TAG}"
  echo "        먼저 3.9의 subject-mode + candidate + teacher 재실행 단계를 완료하세요."
  exit 1
fi

# 1) Stage-10 라벨 생성(기존 teacher score 재활용)
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir "${DATA_DIR}" \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_candidates 0 \
  --run_teacher 0 \
  --run_vlm_teacher 1 \
  --prefer_curated_images 1 \
  --curated_image_dir "${DATA_DIR}/images" \
  --vlm_backend qwen25_vl \
  --vlm_model_id Qwen/Qwen3-VL-4B-Instruct \
  --vlm_device auto \
  --vlm_top_m 12 \
  --vlm_top_k 5 \
  --vlm_fallback_backend heuristic \
  --vlm_multi_gpu 1 \
  --vlm_gpu_ids ${GPU_IDS} \
  --vlm_num_workers ${N_WORKERS} \
  --vlm_save_raw_response 1 \
  --vlm_strict_backend_init 1 \
  --skip_existing 0 \
  --run_tag "${SM_TAG}"

# 2) Stage-10 시각화 + analytics 자산 생성
bash src/scripts/run_visualize_vlm_teacher.sh \
  "${DATA_DIR}/artifacts/vlm_teacher/labels/crop_label_v1_${SM_TAG}.jsonl" \
  "${DATA_DIR}/filtered_sstk_100.parquet" \
  /sstk/20230916/sstk_100 \
  "${DATA_DIR}/artifacts/vlm_teacher/visualizations/vlm_teacher_${SM_TAG}" \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --image_dir "${DATA_DIR}/images" \
  --image_ids_file "${REPORT_DIR}/sample_ids_supercat12.txt" \
  --num_samples 120 \
  --analytics_use_full_labels 1 \
  --server_mode 1
```
### 3.9 + 3.9.1 동시에 실행
10K_local (`base_tag=rerun1_public_e2e`) 예시:
```bash
#DATANAME=10K_local
#BASE_TAG=rerun1_public_e2e
DATANAME=10K
BASE_TAG=e2e_260227_r0

SM_TAG=${BASE_TAG}_260304
DATA_DIR=data/SSTK/${DATANAME}
REPORT_DIR=${DATA_DIR}/artifacts/reports/${SM_TAG}
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
# C1 전략: 0=기존 c1 재활용(권장), 1=C1 재생성(HF/OpenCLIP 다운로드 가능 환경)
RUN_C1_REBUILD=0

# 0) (선택) base report의 샘플 id/manifest 재사용
mkdir -p "${REPORT_DIR}"
cp -f "${DATA_DIR}/artifacts/reports/${BASE_TAG}/sample_ids_supercat12.txt" "${REPORT_DIR}/"
cp -f "${DATA_DIR}/artifacts/reports/${BASE_TAG}/sample_manifest_supercat12.csv" "${REPORT_DIR}/"
cp -f "${DATA_DIR}/artifacts/reports/${BASE_TAG}/sample_manifest_supercat12.json" "${REPORT_DIR}/"
cp -f "${DATA_DIR}/artifacts/reports/${BASE_TAG}/sample_summary_table.md" "${REPORT_DIR}/"

# 0.1) real expensive 사용 전제: run_c1=0일 때는 기존 c1 jsonl 필수
if [ "${RUN_C1_REBUILD}" -eq 0 ] && [ ! -f "${DATA_DIR}/artifacts/precompute/feats_c1.jsonl" ]; then
  echo "[error] missing c1 jsonl: ${DATA_DIR}/artifacts/precompute/feats_c1.jsonl"
  echo "        해결: RUN_C1_REBUILD=1로 C1 재생성(네트워크/HF 필요) 또는 --use_real_expensive 0"
  exit 1
fi

# 1) 기존 precompute를 재활용해 subject routing + candidates + teacher 재실행
# 기본값: "재활용" 경로(run_c1=0)
# 선택값: RUN_C1_REBUILD=1이면 C1 재생성 수행(HF/OpenCLIP 다운로드 필요)
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir "${DATA_DIR}" \
  --run_filter 0 \
  --run_c1 "${RUN_C1_REBUILD}" --run_c2 0 --run_c3 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_subject_routing 1 \
  --subject_routing_top_n 5 \
  --subject_routing_union_top_m 3 \
  --subject_routing_allow_det_proxy 1 \
  --run_candidates 1 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 0 \
  --public_teacher_download_weights 0 \
  --public_teachers gaic,cacnet,cgs \
  --public_gaic_weight_path weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth \
  --proposal_injection_gate 1 \
  --proposal_injection_min_rate 0.95 \
  --proposal_injection_gate_strict 1 \
  --run_teacher 1 \
  --use_real_expensive 1 \
  --teacher_multi_gpu 1 \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS} \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 128 \
  --prefer_curated_images 1 \
  --curated_image_dir "${DATA_DIR}/images" \
  --skip_existing 0 \
  --run_tag "${SM_TAG}"

# 1.1) 1단계 성공 가드: teacher score가 생성되지 않았으면 즉시 중단
if [ ! -f "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" ]; then
  echo "[error] missing teacher scores: ${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl"
  echo "        step-1 로그/에러를 먼저 해결한 뒤 step-2~4를 실행하세요."
  exit 1
fi

# 2) precompute component visualization(샘플 고정)
bash src/scripts/run_visualize_components.sh \
  "${DATA_DIR}/filtered_sstk_100.parquet" \
  /sstk/20230916/sstk_100 \
  "${DATA_DIR}/artifacts/precompute/visualizations/components_${SM_TAG}" \
  --merged_jsonl "${DATA_DIR}/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl" \
  --image_dir "${DATA_DIR}/images" \
  --image_ids_file "${REPORT_DIR}/sample_ids_supercat12.txt" \
  --draw_c2 1 --draw_c3 1 --draw_c4 1 --draw_c5 1 --draw_c6 1 --draw_combined 1 \
  --num_samples 120 \
  --server_mode 1

# 3) teacher visualization(샘플 고정)
python src/visualize_teacher_scores.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --parquet "${DATA_DIR}/filtered_sstk_100.parquet" \
  --tar_dir /sstk/20230916/sstk_100 \
  --image_dir "${DATA_DIR}/images" \
  --out_dir "${DATA_DIR}/artifacts/teacher/visualizations/teacher_scorer_${SM_TAG}" \
  --image_ids_file "${REPORT_DIR}/sample_ids_supercat12.txt" \
  --num_samples 120 \
  --target_ar all \
  --decision_filter all

# 4) report용 overview/qa/top-k analytics 재생성
python src/scripts/rebuild_teacher_overview.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --output_json "${DATA_DIR}/artifacts/teacher/overview/teacher_scores_overview_${SM_TAG}.json" \
  --output_by_ar_csv "${DATA_DIR}/artifacts/teacher/overview/teacher_scores_overview_by_ar_${SM_TAG}.csv"

python src/scripts/qa_teacher_report.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --output_json "${DATA_DIR}/artifacts/teacher/qa/teacher_scores_qa_report_${SM_TAG}.json" \
  --output_by_ar_csv "${DATA_DIR}/artifacts/teacher/qa/teacher_scores_qa_report_by_ar_${SM_TAG}.csv"

mkdir -p "${REPORT_DIR}/assets/analytics/teacher_topk"
python src/scripts/build_teacher_topk_analytics.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --output_dir "${REPORT_DIR}/assets/analytics/teacher_topk"

# 0) 선행 산출물 확인
if [ ! -f "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" ]; then
  echo "[error] missing teacher scores for ${SM_TAG}"
  echo "        먼저 3.9의 subject-mode + candidate + teacher 재실행 단계를 완료하세요."
  exit 1
fi

# 1) Stage-10 라벨 생성(기존 teacher score 재활용)
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir "${DATA_DIR}" \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_candidates 0 \
  --run_teacher 0 \
  --run_vlm_teacher 1 \
  --prefer_curated_images 1 \
  --curated_image_dir "${DATA_DIR}/images" \
  --vlm_backend qwen25_vl \
  --vlm_model_id Qwen/Qwen3-VL-4B-Instruct \
  --vlm_device auto \
  --vlm_top_m 12 \
  --vlm_top_k 5 \
  --vlm_fallback_backend heuristic \
  --vlm_multi_gpu 1 \
  --vlm_gpu_ids ${GPU_IDS} \
  --vlm_num_workers ${N_WORKERS} \
  --vlm_save_raw_response 1 \
  --vlm_strict_backend_init 1 \
  --skip_existing 0 \
  --run_tag "${SM_TAG}"

# 2) Stage-10 시각화 + analytics 자산 생성
bash src/scripts/run_visualize_vlm_teacher.sh \
  "${DATA_DIR}/artifacts/vlm_teacher/labels/crop_label_v1_${SM_TAG}.jsonl" \
  "${DATA_DIR}/filtered_sstk_100.parquet" \
  /sstk/20230916/sstk_100 \
  "${DATA_DIR}/artifacts/vlm_teacher/visualizations/vlm_teacher_${SM_TAG}" \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --image_dir "${DATA_DIR}/images" \
  --image_ids_file "${REPORT_DIR}/sample_ids_supercat12.txt" \
  --num_samples 120 \
  --analytics_use_full_labels 1 \
  --server_mode 1
```

이미 `crop_label_v1_${SM_TAG}.jsonl`이 있으면 1단계는 생략하고 2단계만 수행해도 됩니다.

필요한 선행 산출물이 없는 경우:
- `teacher_scores_ar_${SM_TAG}.jsonl` 없음: 3.9의 subject-mode 재실행(후보+teacher)부터 수행
- `sample_ids_supercat12.txt` 없음: base report(`$BASE_TAG`)에서 복사하거나 `--image_ids_file` 옵션을 제거하고 랜덤 샘플 렌더링

주요 주의사항:
- `--enable_public_teacher_proposals 1`을 켜면 candidate overview의 `proposal_injected_rate`를 기본 gate(`>=0.95`)로 검사합니다.
- 과거 산출물 비교 등으로 의도적으로 proposal 주입을 끄는 경우에만 `--proposal_injection_gate 0`을 사용하세요.

10K (`base_tag=e2e_260227_r0`) 예시:
```bash
DATANAME=10K
BASE_TAG=e2e_260227_r0
SM_TAG=${BASE_TAG}_subject_mode
DATA_DIR=data/SSTK/${DATANAME}
REPORT_DIR=${DATA_DIR}/artifacts/reports/${SM_TAG}
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
# C1 전략: 0=기존 c1 재활용(권장), 1=C1 재생성(HF/OpenCLIP 다운로드 가능 환경)
RUN_C1_REBUILD=0

# 0) 10K에는 feats_c1.jsonl이 없을 수 있음(통합 raw 재사용용 alias)
if [ ! -f "${DATA_DIR}/artifacts/precompute/feats_c1.jsonl" ] && [ -f "${DATA_DIR}/artifacts/precompute/feats_c2c3c5_v2_strict_raw.jsonl" ]; then
  ln -sfn feats_c2c3c5_v2_strict_raw.jsonl "${DATA_DIR}/artifacts/precompute/feats_c1.jsonl"
fi

# 1) base report 샘플 id/manifest 복사
mkdir -p "${REPORT_DIR}"
cp -f "${DATA_DIR}/artifacts/reports/${BASE_TAG}/sample_ids_supercat12.txt" "${REPORT_DIR}/"
cp -f "${DATA_DIR}/artifacts/reports/${BASE_TAG}/sample_manifest_supercat12.csv" "${REPORT_DIR}/"
cp -f "${DATA_DIR}/artifacts/reports/${BASE_TAG}/sample_manifest_supercat12.json" "${REPORT_DIR}/"
cp -f "${DATA_DIR}/artifacts/reports/${BASE_TAG}/sample_summary_table.md" "${REPORT_DIR}/"

# 1.1) real expensive 사용 전제: run_c1=0일 때는 기존 c1 jsonl 필수
if [ "${RUN_C1_REBUILD}" -eq 0 ] && [ ! -f "${DATA_DIR}/artifacts/precompute/feats_c1.jsonl" ]; then
  echo "[error] missing c1 jsonl: ${DATA_DIR}/artifacts/precompute/feats_c1.jsonl"
  echo "        해결: RUN_C1_REBUILD=1로 C1 재생성(네트워크/HF 필요) 또는 --use_real_expensive 0"
  exit 1
fi

# 2) subject routing + candidates + teacher 재실행
# 기본값: "재활용" 경로(run_c1=0)
# 선택값: RUN_C1_REBUILD=1이면 C1 재생성 수행(HF/OpenCLIP 다운로드 필요)
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir "${DATA_DIR}" \
  --run_filter 0 \
  --run_c1 "${RUN_C1_REBUILD}" --run_c2 0 --run_c3 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_subject_routing 1 \
  --subject_routing_top_n 5 \
  --subject_routing_union_top_m 3 \
  --subject_routing_allow_det_proxy 1 \
  --run_candidates 1 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 0 \
  --public_teacher_download_weights 0 \
  --public_teachers gaic,cacnet,cgs \
  --public_gaic_weight_path weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth \
  --proposal_injection_gate 1 \
  --proposal_injection_min_rate 0.95 \
  --proposal_injection_gate_strict 1 \
  --run_teacher 1 \
  --use_real_expensive 1 \
  --teacher_multi_gpu 1 \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS} \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 512 \
  --prefer_curated_images 1 \
  --curated_image_dir "${DATA_DIR}/images" \
  --skip_existing 1 \
  --run_tag "${SM_TAG}"

# 2.1) 2단계 성공 가드: teacher score가 생성되지 않았으면 즉시 중단
if [ ! -f "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" ]; then
  echo "[error] missing teacher scores: ${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl"
  echo "        step-2 로그/에러를 먼저 해결한 뒤 visualization/report 단계를 실행하세요."
  exit 1
fi

# 3) precompute/teacher visualization + report analytics
bash src/scripts/run_visualize_components.sh \
  "${DATA_DIR}/filtered_sstk_100.parquet" \
  /sstk/20230916/sstk_100 \
  "${DATA_DIR}/artifacts/precompute/visualizations/components_${SM_TAG}" \
  --merged_jsonl "${DATA_DIR}/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl" \
  --image_dir "${DATA_DIR}/images" \
  --image_ids_file "${REPORT_DIR}/sample_ids_supercat12.txt" \
  --draw_c2 1 --draw_c3 1 --draw_c4 1 --draw_c5 1 --draw_c6 1 --draw_combined 1 \
  --num_samples 120 \
  --server_mode 1

python src/visualize_teacher_scores.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --parquet "${DATA_DIR}/filtered_sstk_100.parquet" \
  --tar_dir /sstk/20230916/sstk_100 \
  --image_dir "${DATA_DIR}/images" \
  --out_dir "${DATA_DIR}/artifacts/teacher/visualizations/teacher_scorer_${SM_TAG}" \
  --image_ids_file "${REPORT_DIR}/sample_ids_supercat12.txt" \
  --num_samples 120 \
  --target_ar all \
  --decision_filter all

python src/scripts/rebuild_teacher_overview.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --output_json "${DATA_DIR}/artifacts/teacher/overview/teacher_scores_overview_${SM_TAG}.json" \
  --output_by_ar_csv "${DATA_DIR}/artifacts/teacher/overview/teacher_scores_overview_by_ar_${SM_TAG}.csv"

python src/scripts/qa_teacher_report.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --output_json "${DATA_DIR}/artifacts/teacher/qa/teacher_scores_qa_report_${SM_TAG}.json" \
  --output_by_ar_csv "${DATA_DIR}/artifacts/teacher/qa/teacher_scores_qa_report_by_ar_${SM_TAG}.csv"

mkdir -p "${REPORT_DIR}/assets/analytics/teacher_topk"
python src/scripts/build_teacher_topk_analytics.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${SM_TAG}.jsonl" \
  --output_dir "${REPORT_DIR}/assets/analytics/teacher_topk"
```

### 3.10 리뷰 반영 재실험 (짧은 A/B 템플릿)

아래 A/B는 `SSTK_rerun1_subject_mode_report_review_KO_260303.md`의 권고를 바로 검증하기 위한 최소 템플릿입니다.

#### A/B-0: proposal injection on/off 영향
```bash
DATANAME=10K_local
DATA_DIR=data/SSTK/${DATANAME}
BASE_TAG=rerun1_public_e2e

# A: injection ON + gate ON
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir "${DATA_DIR}" \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_subject_routing 1 \
  --run_candidates 1 \
  --run_teacher 1 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 0 \
  --public_teacher_download_weights 0 \
  --public_teachers gaic,cacnet,cgs \
  --proposal_injection_gate 1 \
  --proposal_injection_min_rate 0.95 \
  --run_tag ${BASE_TAG}_ab_on

# B: injection OFF
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir "${DATA_DIR}" \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_subject_routing 1 \
  --run_candidates 1 \
  --run_teacher 1 \
  --enable_public_teacher_proposals 0 \
  --proposal_injection_gate 0 \
  --run_tag ${BASE_TAG}_ab_off
```

#### A/B-1: router guard on/off
```bash
DATANAME=10K_local
DATA_DIR=data/SSTK/${DATANAME}
BASE_TAG=rerun1_public_e2e

# ON: 기본(guard 포함 최신 코드)
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir "${DATA_DIR}" \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_subject_routing 1 \
  --run_candidates 1 \
  --run_teacher 1 \
  --run_tag ${BASE_TAG}_guard_on
```

비교 지표:
- `teacher_scores_qa_report_<tag>.json`의
  - `global.subject_mode_kpi.guard_no_person_for_portrait_rate`
  - `global.subject_mode_kpi.guard_no_text_signal_rate`
  - `global.subject_mode_kpi.guard_low_blank_ratio_copyspace_rate`
  - `global.risk_rates.head_top_cut_rate`
  - `global.proposal_injection.teacher_seed_top1_risk_rate_given_seed_top1`
  - `global.proposal_injection.proposal_teacher_seed_top1_risk_rate_given_proposal`
- `global.proposal_injection.proposal_rescue_rate_proxy`
- `by_subject_mode_shot_ar`에서 문제 모드/샷/AR 집중 여부 확인

#### A/B-2: scene/copyspace/text 정책 파라미터 스윕
현재 구현은 `score_teacher.py`의 `apply_subject_policy_overrides()`에 모드별 lambda/tau/w_area가 하드코딩되어 있습니다.
짧은 스윕은 아래처럼 브랜치 분리 후 상수만 바꿔 `--run_tag`를 달리해 비교하세요.
```bash
# 예: scene_landscape에서 cov 가중치 추가 축소 실험
# src/score_teacher.py 의 apply_subject_policy_overrides()에서
# mode=="scene_landscape" 블록 파라미터를 수정 후 실행

bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/10K_local \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_subject_routing 1 \
  --run_candidates 1 \
  --run_teacher 1 \
  --run_tag rerun1_public_e2e_scene_sweep1
```

#### A/B-3: Stress set 자동 샘플링(릴리즈 고정 비교셋)
```bash
DATANAME=10K_local
TAG=rerun1_public_e2e_subject_mode
DATA_DIR=data/SSTK/${DATANAME}

python src/scripts/build_subject_mode_stress_set.py \
  --teacher_scores_jsonl "${DATA_DIR}/artifacts/teacher/scores/teacher_scores_ar_${TAG}.jsonl" \
  --output_json "${DATA_DIR}/artifacts/reports/${TAG}/subject_mode_stress_set.json" \
  --output_csv "${DATA_DIR}/artifacts/reports/${TAG}/subject_mode_stress_set.csv" \
  --max_per_bucket 120 \
  --seed 42
```

레포트 파일 경로:
- `data/SSTK/10K_local/artifacts/reports/<run_tag>/REPORT_DRAFT_KO.md`
- `data/SSTK/10K/artifacts/reports/<run_tag>/REPORT_DRAFT_KO.md`

주의:
- 현재 저장소에는 `REPORT_DRAFT_KO.md` 본문을 자동 생성/갱신하는 스크립트는 없습니다.
- 위 명령은 레포트 본문에서 참조하는 시각화/통계 자산을 재생성하는 절차입니다.
- 기존 본문을 복사해 새 run_tag용으로 사용하려면:
```bash
# 10K_local
cp -f data/SSTK/10K_local/artifacts/reports/${BASE_TAG}/REPORT_DRAFT_KO.md \
      data/SSTK/10K_local/artifacts/reports/${SM_TAG}/REPORT_DRAFT_KO.md

# 10K
cp -f data/SSTK/10K/artifacts/reports/${BASE_TAG}/REPORT_DRAFT_KO.md \
      data/SSTK/10K/artifacts/reports/${SM_TAG}/REPORT_DRAFT_KO.md
```

---

## 4. 단계별 로직/코드 설명

## 4.1 Phase A: Filter & Curate

핵심 엔트리:
- `src/scripts/run_filter.sh`
- `src/filter_sstk_dataset.py`

주요 로직:
- 기술 필터: 해상도/유형/중복(`image_dedup`) 제거
- 미학 필터: `max(aesthetic_score_center, aesthetic_score_pad)` 기반 하위 제거
- 메타 결합: SDP 태그 + train merged tags 통합
- 카테고리 균형: `super_cat` 단위 percentile cut 후 stratified sampling
- 산출: `filtered_<bucket>.parquet`

핵심 컬럼:
- `image_id`, `width`, `height`, `tags`, `tar_name`, `bucket`(환경에 따라)

선택 최적화(반복 실험용):
- `run_filter.sh`에 `CURATED_IMAGE_DIR`를 추가로 넘기면 curated pool 이미지를 `<data_dir>/images` 등에 추출 저장할 수 있습니다.
- 이후 `run_phaseA_to_teacher_e2e.sh`에서 `--prefer_curated_images 1`이면 4.2~ 단계가 해당 디렉토리를 우선 사용하고, 누락 이미지에만 tar fallback 합니다.
- Filter 캐시(`tmp_parquets_*`, `df_mapped_cache_*`, `tag_cat_probs_cache_*`)는 기본적으로 `data/SSTK/<DATANAME>/cache/filter/`를 사용하며, 기존 루트/`Temp`/`Temp/cleanup_*` 위치도 자동 탐색 후 재사용합니다.

---

## 4.2 Phase B-1: Perception Precompute (C1/C2/C3/C4/C5/C6)

공통 실행기:
- `src/scripts/run_extract_component.sh`
- 내부 실행기: `src/extract_features_single.py`, `src/extract_features_pipeline.py`
- 실제 워커: `src/extract_features/worker_core.py`

멀티 GPU 동작 원칙(현재 기본):
- `--mode auto`에서 GPU가 2개 이상이면 `multi`(No-Ray 샤딩) 선택
- `multi`는 `extract_features_single.py`를 GPU별 shard 프로세스로 병렬 실행
- Ray는 `--mode ray`일 때만 사용(권장하지 않음)

컴포넌트별:
- C1: `src/extract_features/c1_clip.py`
  - OpenCLIP 이미지/텍스트 임베딩 추출(`c1_img_embed`, `c1_txt_embed`)
- C2: `src/extract_features/c2_seg.py`
  - `high_efficiency`: YOLOv8 + EfficientViT-SAM
  - `quality_first`: YOLO prior + SAM2(+AMG fallback)
  - 출력: `c2_seg` + `c2_det`(detector raw box)
- C3: `src/extract_features/c3_pose.py`
  - RTMDet + RTMPose + person verifier(YOLO)
  - strict person verify 및 tag/person hint 게이팅
  - 출력: `c3_pose` (bbox, keypoints, 보강 face/headpose_gaze 포함 가능)
- C5: `src/extract_features/c5_geom.py`
  - horizon/roll, symmetry 계산

C3 enrich:
- `src/scripts/enrich_c3_pose_jsonl.py`
- keypoint 기반 `face`, `headpose_gaze` proxy를 안정적으로 추가
- `--use_actual_image_size 1` 권장 (tar 실제 사이즈 기준 정규화)

Subject-Mode enrich(신규):
- `src/scripts/enrich_subject_mode_jsonl.py`
- merged precompute에 아래를 주입:
  - `routing.subject_mode/policy_id/subject_set`
  - `c2_seg` Top-N + `importance_score/bg_like`
  - `c2_primary_idx`, `c2_union_box_xyxy`, `c2_stats`
- e2e 기본값은 `--run_subject_routing 1`이며, Candidate/Teacher/QA는 routed 파일을 우선 사용

권장 결과물:
- `data/SSTK/<DATANAME>/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl`
  - 파일명은 레거시 네이밍이지만, `run_c4=1/run_c6=1`이면 내부 레코드에 `c4_ocr`, `c6_gaze`도 함께 포함됩니다.

---

## 4.3 Phase B-2: Candidate Generator (AR-조건부)

엔트리:
- `src/scripts/run_generate_candidates.sh`
- `src/generate_candidates.py`

핵심 방법론:
- 입력: filtered parquet + C2/C3
- 라우팅: `routing.subject_mode/policy_id` 기반 subject prior/union 보강
- 주체 prior 구성:
  - C2(det/seg) + C3(pose) 결합으로 subject box 추정
- 후보 생성:
  - Baseline(원본/최대면적/slide/copyspace)
  - Grid anchors + multi-scale area
  - subject-centered templates
  - saliency-guided jitter
  - (선택) thirds/phi priors
- 후보 정리:
  - dedupe, NMS, must-keep 보호, diversity farthest-point 샘플링
- 산출:
  - `artifacts/candidates/candidates_ar*.jsonl`
  - overview json/csv + 시각화(`artifacts/candidates/visualizations/candidates_*`)

중요 포인트:
- `--use_actual_image_size 1` + `--strict_actual_size 1`을 기본 유지
- parquet 원본 크기와 tar 실제 이미지 크기 불일치 문제를 방지

---

## 4.4 9) Teacher Scorer (Cheap -> Expensive -> Top-K)

엔트리:
- `src/scripts/run_teacher_scorer.sh`
- `src/score_teacher.py`
- QA: `src/scripts/qa_teacher_report.py`
- 시각화: `src/visualize_teacher_scores.py`

로직 구성:
- Hard constraints
  - AR/면적 범위/face-cut/joint-cut + portrait `head_top_cut`(머리 상단/헤어 컷) 구조적 실패 필터
- Subject-Mode policy schedule
  - `subject_mode/policy_id` 기반으로 lambda/tau/w_area를 재스케줄
  - scene/copyspace/text 모드는 headroom/lookroom 비중을 낮추고 context/copyspace/text 보존 가중치를 강화
  - portrait 모드인데 `num_person<=0`이면 scorer 단계에서 non-portrait 모드로 강등(추가 sanity guard)
- Cheap score (N -> M)
  - subject coverage, cut penalty, composition prior(3분할/phi/center/horizon)
  - headroom/lookroom, symmetry, context, copy-space 반영
- Expensive score
  - proxy 또는 real 모드 선택
  - real 모드: 미학 점수 + crop/image-text 정렬(cosine)
- Keep-vs-Crop 게이팅
  - baseline 대비 개선량 기반
- Top-K Diversity
  - IoU 제약 기반 다양성 보장

산출:
- `artifacts/teacher/scores/teacher_scores_ar*.jsonl`
- `artifacts/teacher/overview/teacher_scores_overview*.json/.csv`
- `artifacts/teacher/qa/teacher_scores_qa_report*.json/.csv`
- `artifacts/teacher/visualizations/teacher_scorer*/`

주요 QA KPI(신규 포함):
- `global.risk_rates.head_top_cut_rate`
- `global.proposal_injection.teacher_seed_top1_risk_rate_given_seed_top1`
- `global.proposal_injection.proposal_teacher_seed_top1_risk_rate_given_proposal`

---

## 4.5 10) VLM/MLLM Teacher Labeler (Qwen 기본, 플러그인형)

엔트리:
- `src/vlm_teacher_labeler.py`
- `src/scripts/run_vlm_teacher_labeler.sh`

로직 구성:
- 입력: `teacher_scores_ar*.jsonl`의 AR별 `cheap_top_m/selected_topk/hard_negatives`
- Stage-1 메타 정규화: `meta_norm_v1` 생성(룰 기반)
- Stage-2 라벨 생성:
  - backend=`qwen25_vl`일 때 이미지+수치 피처를 기반으로 JSON 생성 시도
  - strict post-validation(candidate_id/bbox 정합성) + retry
  - 실패 시 fallback(`heuristic`)으로 numeric top-k 기반 라벨 보정
- OOM 정책:
  - GPU OOM 시 CPU backend 소량 fallback(기본 3 이미지)
  - fallback 실패 task는 skip 또는 fallback backend로 대체

산출:
- `artifacts/vlm_teacher/labels/crop_label_v1*.jsonl`
- `artifacts/vlm_teacher/meta/meta_norm_v1*.jsonl`
- `artifacts/vlm_teacher/summary/vlm_teacher_summary*.json`
- (옵션) `artifacts/vlm_teacher/debug/vlm_teacher*/raw_responses/*`

---

## 5. 실행 스크립트 옵션 가이드

### 5.1 `run_phaseA_to_teacher_e2e.sh` 핵심 옵션

- `--server_mode 0|1`: 로컬/서버 모드
- `--data_dir`: 출력 루트
- `--run_filter 0|1`: Filter 수행 여부
- `--filter_require_train_match 0|1`: Filter에서 `TRAIN_DIR/<bucket>/<same_file>.json` 매칭 파일만 처리(기본 1)
- `--filter_tag_embed_multi_gpu -1|0|1`: Filter 태그 임베딩 멀티 GPU 제어(`-1` auto, 기본 -1)
- `--filter_tag_embed_gpu_ids`: Filter 태그 임베딩용 GPU 목록 CSV(예: `0,1,2,3`)
- `--filter_tag_embed_batch_size`: Filter 태그 임베딩 배치 크기(기본 128)
- `--filter_tag_embed_chunk_size`: Filter 멀티 GPU encode chunk 크기(기본 0=auto)
- `--filter_tag_embed_device`: Filter 단일 디바이스 모드 장치(`auto|cuda|cuda:0|cpu`)
- `--export_curated_images 0|1`: Filter 후 curated 이미지를 로컬 디렉토리로 추출
- `--curated_image_dir`: curated 이미지 디렉토리 (예: `data/SSTK/10K_local/images`)
- `--curated_image_skip_existing 0|1`: 이미지 추출 시 기존 파일 skip
- `--prefer_curated_images 0|1`: 4.2~ 단계에서 curated image dir 우선 로드
- `--precompute_mode unified|split`:
  - `unified` 권장 (C1/C2/C3/C4/C5/C6를 조건부 1-pass)
  - `run_c1=1`이면 C1 포함, `run_c1=0`이면 C2/C3/C4/C5/C6만 수행
  - `split` 레거시(컴포넌트별 분리)
- `--extract_mode auto|single|multi|ray`:
  - `multi` = No-Ray 멀티 GPU 샤딩
  - `ray` = 명시적 레거시 Ray 모드
- `--extract_multi_gpu -1|0|1`:
  - `1` = precompute를 No-Ray `multi`로 강제
  - `0` = precompute를 `single`로 강제
  - `-1` = `--extract_mode` 설정값 사용(기본)
- `--extract_gpu_ids`: precompute에 사용할 GPU 목록 CSV
- `--num_workers`: precompute shard worker 수 (`multi`에서 보통 GPU 개수와 동일)
- `--run_c1 --run_c2 --run_c3 --run_c4 --run_c5 --run_c6 --run_c3_enrich --run_merge`
- `--run_subject_routing 0|1`: merged precompute에 subject_mode + c2 topN 주입(기본 1)
- `--subject_routing_top_n`: c2 top-N instance 수(기본 5)
- `--subject_routing_union_top_m`: union box 계산용 상위 instance 수(기본 3)
- `--subject_routing_allow_det_proxy 0|1`: c2_det proxy로 topN 보강(기본 1)
- `--run_component_viz 0|1`: precompute(C2/C3/C4/C5/C6) 시각화 자동 생성
- `--component_viz_num_samples`: precompute 시각화 샘플 수
- `--component_viz_out_dir`: precompute 시각화 출력 경로
- `--component_viz_image_ids`: precompute 시각화 대상 image_id CSV
- `--run_training_labels -1|0|1`:
  - `1` = teacher score 이후 `pairwise/listwise/decision/checklist/regression` 생성기 실행
  - `0` = training labels 단계 skip
  - `-1` = auto (`run_tag`가 있으면 on)
- `--training_labels_dir`: training labels 출력 경로 (기본 `artifacts/training_labels/<run_tag>`)
- `--component_viz_image_ids_file`: precompute 시각화 대상 image_id 파일(한 줄 1개)
- `--run_candidates --run_teacher`
- `--cand_ar_list`: candidate target AR CSV (기본: `FREE,1:1,9:16,16:9,3:4,4:3`)
- `--teacher_proposals_jsonl`: Candidate 단계에 수동 proposal jsonl 주입(CSV)
- `--enable_public_teacher_proposals 0|1`: 5.5(공개 Teacher setup+infer+build) 자동 수행 후 Candidate에 자동 주입
- `--proposal_injection_gate 0|1`: 공개 proposal 주입률 QA gate 수행(기본 1)
- `--proposal_injection_min_rate`: `proposal_injected_rate` 하한(기본 0.95)
- `--proposal_injection_gate_strict 0|1`: gate 미달 시 즉시 실패 여부(기본 1)
- `--public_teacher_setup 0|1`: 공개 Teacher 준비 단계 실행 여부
- `--public_teacher_download_weights 0|1`: setup 단계에서 가중치 자동 다운로드 시도
- `--public_teachers`: 추론 teacher 목록 CSV (예: `gaic,cacnet,cgs`)
- `--public_teacher_max_images`: 공개 Teacher 추론 이미지 수 (`-1`이면 `--max_images` 상속)
- `--public_teacher_device auto|cuda|cpu`: 공개 Teacher 추론 디바이스
- `--public_gaic_weight_path`: GAIC 체크포인트 경로
- `--public_teacher_raw_jsonl`: 공개 Teacher raw 결과 저장 경로 (기본: `data/SSTK/<DATANAME>/artifacts/public_teachers/raw/teacher_raw_public_<run_tag>.jsonl`)
- `--public_teacher_proposals_jsonl`: raw->proposal 변환 출력 경로 (기본: `data/SSTK/<DATANAME>/artifacts/public_teachers/proposals/teacher_proposals_public_<run_tag>.jsonl`)
- `--public_infer_multi_gpu -1|0|1`: 공개 Teacher 추론 multi-gpu on/off (`-1`이면 auto)
- `--public_infer_gpu_ids`: 공개 Teacher 추론에 사용할 GPU 목록 CSV
- `--public_infer_num_workers`: 공개 Teacher shard worker 수(기본: 사용 GPU 수)
- `--public_infer_skip_on_oom 0|1`: 공개 Teacher 추론 중 OOM 레코드 skip
- `--public_infer_fallback_cpu_on_oom 0|1`: GPU OOM 시 CPU fallback 실행
- `--public_infer_fallback_cpu_max_images`: CPU fallback 시 최대 샘플 수
- `--public_infer_skip_if_fallback_failed 0|1`: CPU fallback 실패 시 단계 전체 skip 여부
- `--use_real_expensive 0|1`
- `--teacher_multi_gpu -1|0|1`: -1이면 auto(real-expensive + multi-gpu 환경에서 자동 on), `>0` 입력도 on으로 처리
- `--teacher_gpu_ids`: teacher 멀티 GPU 목록 CSV
- `--teacher_num_workers`: teacher shard worker 수
- `--teacher_auto_repair 0|1`: teacher 완료 후 shard 병합/검증 자동 복구 (기본 1)
- `--teacher_auto_repair_strict 0|1`: expected rows 불일치 시 shard promote 금지 (기본 1)
- `--teacher_hard_head_top_rule 0|1`: portrait head-top(hair) 컷 하드 reject (기본 1)
- `--teacher_head_top_face_expand_alpha`: face 기반 head-top 상방 보정 계수 (기본 0.35)
- `--teacher_head_top_kp_expand`: keypoint 기반 head-top 상방 보정 (기본 0.06)
- `--teacher_head_top_min_margin`: head-top 최소 안전 마진 (기본 0.008)
- `--teacher_head_top_face_margin_alpha`: face 높이 기반 안전 마진 계수 (기본 0.20)
- `--max_images`: candidate/teacher 단계 처리 수 제한
- `--skip_existing 0|1`: 산출물 존재 시 skip
- `--run_tag`: 결과 파일 suffix
- `--run_vlm_teacher 0|1`: Section 10 실행 여부
- `--vlm_backend`: `qwen25_vl|heuristic`
- `--vlm_fallback_backend`: `heuristic|none`
- `--vlm_model_id`: HF 모델 ID (기본: `Qwen/Qwen3-VL-4B-Instruct`)
- `--vlm_device`, `--vlm_dtype`, `--vlm_max_new_tokens`, `--vlm_temperature`
- `--vlm_top_m`, `--vlm_top_k`, `--vlm_target_ar`, `--vlm_max_images`, `--vlm_max_retries`
- `--vlm_skip_on_oom`, `--vlm_fallback_cpu_on_oom`, `--vlm_fallback_cpu_max_images`, `--vlm_skip_if_fallback_failed`
- `--vlm_strict_backend_init 0|1`: qwen backend init 실패 시 즉시 종료(기본 1)
- `--vlm_multi_gpu -1|0|1`, `--vlm_gpu_ids`, `--vlm_num_workers`
- `--vlm_output_jsonl`, `--vlm_output_meta_jsonl`, `--vlm_summary_json`, `--vlm_debug_dir`

### 5.2 `run_extract_component.sh` 핵심 옵션

- positional: `<input_parquet> <bucket> <output_jsonl>`
- `--component c1 c2 c3 c5` 또는 `all`
- `--priority high_efficiency|quality_first`
- `--mode auto|single|multi|ray` (`multi`는 No-Ray 샤딩)
- `--num_workers`, `--gpu_ids`
- `--server_mode`, `--venv_path`, `--tar_dir`, `--weights_dir`

### 5.3 `run_generate_candidates.sh` 핵심 옵션

- positional: `<input_parquet> <feats_c2_jsonl> <feats_c3_jsonl> <output_jsonl>`
- `--use_actual_image_size 1`
- `--strict_actual_size 1`
- `--max_images`, `--max_candidates_per_ar`, `--ar_list ...`
- `--ar_list` 기본값은 `FREE`를 포함합니다.

v1.9 proposal 주입 관련:
- `--teacher_proposals_jsonl <jsonl...>`
- `--teacher_nms_iou` (default `0.95`)
- `--teacher_max_seeds_per_teacher` (default `1`)
- `--teacher_prefer_expand` (default `1`)
- `--teacher_jitter_shift_fracs` (default `0.03`)
- `--teacher_jitter_scales` (default `0.92 1.0 1.08`)

### 5.4 `run_teacher_scorer.sh` 핵심 옵션

- `--candidates_jsonl`, `--features_jsonl`, `--c1_jsonl`
- `--cheap_top_m`, `--top_k`, `--tau_div`
- `--use_real_expensive 1` 시
  - `--align_device`, `--aesthetic_device`, `--exp_batch_size`
  - `--aesthetic_mlp_path`, `--aesthetic_mlp_url`
- 멀티 GPU 샤딩:
  - `--multi_gpu 1`
  - `--gpu_ids ${GPU_IDS}`
  - `--num_workers ${N_WORKERS}`
- `--run_qa 1`, `--run_viz 1`, `--num_viz`
- `--viz_image_ids`, `--viz_image_ids_file`: teacher viz 대상 image_id 고정(보고서/디버그용)
- 자동 복구: `src/scripts/repair_teacher_outputs.py`
  - teacher shard 출력과 최종 jsonl 불일치 시 자동 병합(promote)
  - overview/QA를 최종 jsonl 기준으로 재생성
  - `expected rows`(candidates + max_images) 검증으로 stale shard 오인 promote 방지

### 5.5 공개 Teacher 추론/변환(v1.9 8.2.0a)

권장 방식은 `run_phaseA_to_teacher_e2e.sh`에서 자동으로 처리하는 것입니다.

멀티 GPU 설계/구현 포인트:
- 공개 teacher 추론은 `filtered parquet`를 shard로 분할하고, GPU별 독립 프로세스로 병렬 실행 후 JSONL을 병합합니다.
- 기본값 `--public_infer_multi_gpu -1`에서는 GPU가 2개 이상이면 자동으로 multi-gpu를 사용합니다.
- 변환(`run_build_teacher_proposals.sh`)은 GPU 연산이 없어서 single-pass 유지가 기본이며, 전체 시간의 병목은 추론 단계에서 해소합니다.

1) 처음부터 끝까지 실행(설치 포함):
```bash
DATANAME=10K
GPU_IDS=0,1,2,3
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/${DATANAME} \
  --run_filter 1 \
  --precompute_mode unified \
  --prefer_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 1 \
  --public_teacher_download_weights 1 \
  --public_teachers gaic,cacnet,cgs \
  --public_gaic_weight_path weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth \
  --public_infer_multi_gpu 1 \
  --public_infer_gpu_ids ${GPU_IDS} \
  --public_infer_num_workers ${N_WORKERS} \
  --public_infer_skip_on_oom 1 \
  --public_infer_fallback_cpu_on_oom 1 \
  --public_infer_fallback_cpu_max_images 3 \
  --public_infer_skip_if_fallback_failed 1 \
  --run_candidates 1 \
  --run_teacher 1 \
  --run_tag full_v17_public
```

2) 필터 결과가 이미 있을 때(4.2~9 + 5.5 자동 적용):
```bash
DATANAME=10K
GPU_IDS=0,1,2,3
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/${DATANAME} \
  --run_filter 0 \
  --precompute_mode unified \
  --prefer_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 1 \
  --public_teacher_download_weights 1 \
  --public_teachers gaic,cacnet,cgs \
  --public_gaic_weight_path weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth \
  --public_infer_multi_gpu 1 \
  --public_infer_gpu_ids ${GPU_IDS} \
  --public_infer_num_workers ${N_WORKERS} \
  --public_infer_skip_on_oom 1 \
  --public_infer_fallback_cpu_on_oom 1 \
  --public_infer_fallback_cpu_max_images 3 \
  --public_infer_skip_if_fallback_failed 1 \
  --run_candidates 1 \
  --run_teacher 1 \
  --skip_existing 0 \
  --run_tag rerun1_public_e2e
```

### 5.6 `run_vlm_teacher_labeler.sh` 핵심 옵션

- 입력/출력:
  - `--teacher_scores_jsonl`
  - `--output_jsonl`
  - `--output_meta_jsonl`
  - `--summary_json`
- backend:
  - `--backend qwen25_vl|heuristic`
  - `--fallback_backend heuristic|none`
  - `--model_id`, `--device`, `--dtype`
- 실행 범위:
  - `--image_dir`(qwen backend 시 권장)
  - `--target_ar all|CSV`
  - `--top_m`, `--top_k`
  - `--max_images`, `--max_retries`
- 안정성:
  - `--skip_on_oom`
  - `--fallback_cpu_on_oom`
  - `--fallback_cpu_max_images`
  - `--skip_if_fallback_failed`
  - `--strict_backend_init` (기본 1)
- 멀티 GPU 샤딩:
  - `--multi_gpu -1|0|1` (`-1`이면 qwen+cuda+multi-gpu 환경에서 auto on)
  - `--gpu_ids ${GPU_IDS}`
  - `--num_workers ${N_WORKERS}`
- 디버그:
  - `--save_raw_response 1`
  - `--debug_dir <path>`

Qwen3-VL(기본) 필수 런타임:
- `Qwen/Qwen3-VL-4B-Instruct`는 별도 전용 환경으로 분리 운영을 권장합니다.
- 기본 파이프라인 환경은 `ENABLE_QWEN3_VL=0`으로 유지하여 C3/mmpretrain 안정성을 우선합니다.
- 현재 환경에서 `Qwen3VLForConditionalGeneration` 심볼이 없으면 로드가 실패합니다.
- 대표 에러: `current=4.44.2, has_qwen3_vl_class=False`

업그레이드 예시(Qwen3-VL, 1순위: 사내 PyPI 미러/기본 인덱스):
```bash
python -m pip install -U \
  transformers \
  "tokenizers>=0.21.0" \
  "huggingface-hub>=0.26.0" \
  "accelerate>=0.30.0"
```
업그레이드 예시(Qwen3-VL, 2순위 fallback: GitHub 소스):
```bash
python -m pip install -U \
  "git+https://github.com/huggingface/transformers" \
  "tokenizers>=0.21.0" \
  "huggingface-hub>=0.26.0" \
  "accelerate>=0.30.0"
```

서버 즉시 복구 절차(권장, Stage-10 전용 env):
```bash
# 1) 전용 venv 생성
python3 -m venv /group-volume/jaden.ju/Venvs/qwen3_vlm

# 2) Qwen3 runtime-only 설치
QWEN3_RUNTIME_ONLY=1 \
QWEN3_ALLOW_GITHUB_FALLBACK=0 \
PYTHON=/group-volume/jaden.ju/Venvs/qwen3_vlm/bin/python \
bash src/scripts/install_features_deps_torch251_cu121_stable.sh

# 3) 래퍼로 Stage-10 실행
bash src/scripts/run_vlm_teacher_labeler_qwen3_env.sh \
  --qwen3_venv_path /group-volume/jaden.ju/Venvs/qwen3_vlm \
  --teacher_scores_jsonl <teacher_scores.jsonl> \
  --image_dir <images_dir> \
  --output_jsonl <labels.jsonl> \
  --output_meta_jsonl <meta.jsonl> \
  --summary_json <summary.json>
```

참고:
- `run_phaseA_to_teacher_e2e.sh`, `run_vlm_teacher_labeler.sh`는 이제 `server_mode=1`에서도 `--venv_path` 파일이 존재하면 자동 활성화합니다.
- `--venv_path`를 실제 서버 venv로 지정하지 않으면 시스템 python(`/usr/bin/python`)이 사용될 수 있습니다.

호환성 이슈 시 임시 대안:
- 환경 고정으로 `transformers` 업그레이드가 어려우면 `--vlm_model_id Qwen/Qwen2.5-VL-3B-Instruct`로 내려서 실행할 수 있습니다.

`The following generation flags are not valid and may be ignored: ['temperature']` 경고:
- greedy decode(`temperature=0`)에서 `temperature`를 함께 전달할 때 나타나는 HF 경고입니다.
- 현재 코드에서는 `temperature>0`일 때만 전달하도록 수정되어 동일 경고가 발생하지 않습니다.

3) Teacher Scorer 재실행 필요 여부:
- 위 E2E 템플릿처럼 한 번에 실행하면 Candidate와 Teacher가 같은 런에서 갱신되므로 별도 재실행이 필요 없습니다.
- 이미 이전 candidate/teacher 결과를 만든 뒤에 `teacher_proposals_jsonl`만 추가 주입하면 candidate와 teacher를 다시 실행해야 합니다.

4) 수동 분리 실행(필요 시):
4-1. 공개 teacher 레포/가중치 준비:
```bash
bash src/scripts/run_setup_public_cropping_teachers.sh \
  --teacher_root_dir third_party/public_cropping_teachers \
  --weights_dir weights/public_cropping_teachers \
  --download_weights 1
```

4-2. GAIC/CACNet/CGS raw 추론(JSONL):
- `--prefer_curated_images 1` + `--curated_image_dir ...`를 사용하면 curated 이미지 디렉토리를 우선 사용하고, 누락 이미지만 `tar_dir`에서 fallback 로드합니다.
- `--curated_image_dir`를 생략하면 래퍼가 기본값으로 `<input_parquet 디렉토리>/images`를 사용합니다.
- 멀티 GPU를 강제하려면 `GPU_IDS=0,1,2,3`, `N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")`를 선언한 뒤 `--multi_gpu 1 --gpu_ids ${GPU_IDS} --num_workers ${N_WORKERS}`를 추가합니다.
```bash
GPU_IDS=0,1,2,3
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
bash src/scripts/run_infer_public_cropping_teachers.sh \
  data/SSTK/10K_local/filtered_sstk_100.parquet \
  /media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100 \
  data/SSTK/10K_local/artifacts/public_teachers/raw/teacher_raw_public_manual.jsonl \
  --teachers gaic cacnet cgs \
  --prefer_curated_images 1 \
  --curated_image_dir data/SSTK/10K_local/images \
  --multi_gpu 1 \
  --gpu_ids ${GPU_IDS} \
  --num_workers ${N_WORKERS} \
  --gaic_weight_path weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth
```

4-3. raw -> `teacher_proposals_jsonl` 변환:
```bash
bash src/scripts/run_build_teacher_proposals.sh \
  data/SSTK/10K_local/artifacts/public_teachers/proposals/teacher_proposals_public_manual.jsonl \
  data/SSTK/10K_local/artifacts/public_teachers/raw/teacher_raw_public_manual.jsonl
```

4-4. Candidate 단계 주입:
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 0 \
  --data_dir data/SSTK/10K_local \
  --run_filter 0 \
  --teacher_proposals_jsonl data/SSTK/10K_local/artifacts/public_teachers/proposals/teacher_proposals_public_manual.jsonl \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_candidates 1 \
  --run_teacher 0 \
  --skip_existing 0 \
  --run_tag public_seeded
```

4-5. Candidate 주입 후 Teacher Scorer 실행:
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 0 \
  --data_dir data/SSTK/10K_local \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_candidates 0 \
  --run_teacher 1 \
  --skip_existing 0 \
  --run_tag public_seeded
```

`use_real_expensive=1` 사용:
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 0 \
  --data_dir data/SSTK/10K_local \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c4 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_candidates 0 \
  --run_teacher 1 \
  --use_real_expensive 1 \
  --skip_existing 0 \
  --run_tag public_seeded_real
```

---

## 6. End-to-End 결과물 맵

기본(`data_dir=data/SSTK/10K_local`, `run_tag=v17_local`) 예시:

- Filter
  - `data/SSTK/10K_local/filtered_sstk_100.parquet`
- Precompute
  - `data/SSTK/10K_local/artifacts/precompute/feats_c2c3c5_v2_strict_raw.jsonl` (unified 중간)
  - `data/SSTK/10K_local/artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl` (최종)
  - `data/SSTK/10K_local/artifacts/precompute/feats_c1.jsonl` (split 모드 또는 별도 C1 추출 시)
  - `precompute_mode=unified` + `run_c1=1`이면 C1은 위 unified jsonl에 함께 저장됨
  - (선택) `data/SSTK/10K_local/artifacts/precompute/visualizations/components_<run_tag>/` (`--run_component_viz 1`)
- Candidate
  - `data/SSTK/10K_local/artifacts/candidates/candidates_ar_v17_local.jsonl`
  - `data/SSTK/10K_local/artifacts/candidates/candidates_ar_v17_local_overview*.{json,csv}`
  - `data/SSTK/10K_local/artifacts/candidates/visualizations/candidates_ar_v17_local_viz/`
- Public Teacher (v1.9 8.2.0a)
  - `data/SSTK/10K_local/artifacts/public_teachers/raw/teacher_raw_public_v17_local.jsonl`
  - `data/SSTK/10K_local/artifacts/public_teachers/proposals/teacher_proposals_public_v17_local.jsonl`
- Teacher
  - `data/SSTK/10K_local/artifacts/teacher/scores/teacher_scores_ar_v17_local.jsonl`
  - `data/SSTK/10K_local/artifacts/teacher/overview/teacher_scores_overview_v17_local.json`
  - `data/SSTK/10K_local/artifacts/teacher/overview/teacher_scores_overview_by_ar_v17_local.csv`
  - `data/SSTK/10K_local/artifacts/teacher/qa/teacher_scores_qa_report_v17_local.json`
  - `data/SSTK/10K_local/artifacts/teacher/qa/teacher_scores_qa_report_by_ar_v17_local.csv`
  - `data/SSTK/10K_local/artifacts/teacher/visualizations/teacher_scorer_v17_local/`
- Cache
  - `data/SSTK/10K_local/cache/actual_image_size_map.json`
- 로그
  - `data/SSTK/10K_local/logs/e2e_v17_local/*.log`

---

## 7. 운영 권장사항 / 트러블슈팅

- 로컬 실행은 가상환경 활성화 권장
  - 본 스크립트는 `server_mode=0`일 때 자동으로
    `/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate`를 source함
- OOM 발생 시
  - `--exp_batch_size` 축소
  - `--aesthetic_device cpu` 또는 `--align_device cpu`
  - precompute는 `--mode single`/작은 배치로 조정
- 실제 이미지 크기 불일치 이슈
  - Candidate/C3 enrich는 반드시 `--use_actual_image_size 1` 유지
  - `data/SSTK/<DATANAME>/cache/actual_image_size_map.json` 캐시를 재사용하면 속도 개선
- C3 품질 이슈
  - `C3_PERSON_VERIFY_STRICT=1` 유지
  - 가능하면 `unified` 모드(C2 person hint 동시 사용) 권장

### 7.1 C6(Gazelle) 오프라인 서버 대응
1.최신 코드 반영 후 설치 스크립트 재실행:
```bash
PYTHON=python3 ENABLE_C4_OCR=0 STRICT_QUALITY_IMPORTS=1 \
bash src/scripts/install_features_deps_torch251_cu121_stable.sh
```

2.pythonjsonlogger 즉시 보정(필요 시):
```bash
python3 -m pip install -U python-json-logger
```

3. 서버가 오프라인이면 로컬에서 아래 자산 업로드:
- `third_party/gazelle`
- `.cache/torch/hub/facebookresearch_dinov2_main`
- `.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth`
- `.cache/torch/hub/checkpoints/gazelle_dinov2_vitb14_inout.pt`

4. 실행 전 환경 고정:
```bash
export C6_TORCH_HUB_DIR=$PWD/.cache/torch/hub
export C6_GAZELLE_REPO=$PWD/third_party/gazelle
export C6_GAZELLE_CKPT=$PWD/.cache/torch/hub/checkpoints/gazelle_dinov2_vitb14_inout.pt
```

---

## 8. 레거시 분리 실행이 필요한 경우

아래 순서로 수동 실행 가능:

1. Filter: `run_filter.sh`
2. C2/C3/C4/C5 각각 `run_extract_component.sh`
3. C3 enrich: `enrich_c3_pose_jsonl.py`
4. merge: `merge_feature_jsonl.py`
5. Candidate: `run_generate_candidates.sh`
6. Teacher: `run_teacher_scorer.sh`

단, 신규 운영은 `run_phaseA_to_teacher_e2e.sh --precompute_mode unified`를 기본으로 권장합니다.

---

## 8.1 C5/C6만 재실행 후 Teacher/VLM 업데이트 (증분 운영)

아래는 **기존 Candidate를 재사용**하면서 `C5/C6`만 갱신하고, 이후 `Teacher Scorer`와 `VLM Teacher`를 다시 생성하는 절차입니다.

사전 변수:
```bash
DATANAME=10K_local
REFRESH_TAG=c5c6_refresh_260305

PARQUET=data/SSTK/${DATANAME}/filtered_sstk_100.parquet
IMAGE_DIR=data/SSTK/${DATANAME}/images

# 기존 기준 feature (이미 만들어져 있는 파일)
BASE_FEATS=data/SSTK/${DATANAME}/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl

# 이번에 재생성할 파일
FEATS_C5C6=data/SSTK/${DATANAME}/artifacts/precompute/feats_c5c6_${REFRESH_TAG}.jsonl
FEATS_REFRESHED=data/SSTK/${DATANAME}/artifacts/precompute/feats_c2c3c5c6_${REFRESH_TAG}.jsonl

# 고정 venv
VENV_ACT=/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate
```

### 1) C5/C6만 quality_first로 재실행
```bash
bash src/scripts/run_extract_component.sh \
  "$PARQUET" sstk_100 "$FEATS_C5C6" \
  --component c5 c6 \
  --priority quality_first \
  --server_mode 0 \
  --venv_path "$VENV_ACT" \
  --image_dir "$IMAGE_DIR" \
  --tar_dir /media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100 \
  --batch_size 8
```

검증 포인트(예: ScaleLSD/Gazelle 경로):
```bash
python - <<'PY'
import json
import os
from collections import Counter
path = os.environ["FEATS_C5C6"]
c5 = Counter(); c6 = Counter()
with open(path, "r", encoding="utf-8") as f:
    for ln in f:
        d = json.loads(ln)
        c5[d.get("c5_geom", {}).get("horizon_roll", {}).get("method", "none")] += 1
        c6[d.get("c6_gaze", {}).get("method", "none")] += 1
print("c5 methods:", dict(c5))
print("c6 methods:", dict(c6))
PY
```

### 2) 기존 feature와 병합(증분 반영)
```bash
python src/scripts/merge_feature_jsonl.py \
  --input_parquet "$PARQUET" \
  --inputs "$BASE_FEATS" "$FEATS_C5C6" \
  --output_jsonl "$FEATS_REFRESHED"
```

중요:
- 현재 `score_teacher.py`는 `c5_geom`을 직접 참조합니다.
- 현재 `score_teacher.py`의 lookroom/gaze는 `c3_pose[].headpose_gaze`를 기본으로 사용하고, 보조적으로 `c6_gaze.people`를 fallback/override로 참조합니다.
- 즉, 권장 경로는 여전히 **`c3+c6` 동시 재추출**이며, 그 결과가 scorer에 자동 반영됩니다.
- `score_teacher.py`의 `p_text`는 C4 OCR box(`c4_ocr` + alias)를 직접 사용해 계산되므로, C4 재추출/병합 결과가 cheap/expensive score 모두에 반영됩니다.

### 3) Teacher Scorer 재실행 (Candidate 재사용)
```bash
bash src/scripts/run_teacher_scorer.sh \
  --server_mode 0 \
  --venv_path "$VENV_ACT" \
  --features_jsonl "$FEATS_REFRESHED" \
  --candidates_jsonl data/SSTK/${DATANAME}/artifacts/candidates/candidates_ar.jsonl \
  --c1_jsonl data/SSTK/${DATANAME}/artifacts/precompute/feats_c1.jsonl \
  --output_jsonl data/SSTK/${DATANAME}/artifacts/teacher/scores/teacher_scores_ar_${REFRESH_TAG}.jsonl \
  --output_overview_json data/SSTK/${DATANAME}/artifacts/teacher/overview/teacher_scores_overview_${REFRESH_TAG}.json \
  --output_overview_csv data/SSTK/${DATANAME}/artifacts/teacher/overview/teacher_scores_overview_by_ar_${REFRESH_TAG}.csv \
  --qa_out_json data/SSTK/${DATANAME}/artifacts/teacher/qa/teacher_scores_qa_report_${REFRESH_TAG}.json \
  --qa_out_csv data/SSTK/${DATANAME}/artifacts/teacher/qa/teacher_scores_qa_report_by_ar_${REFRESH_TAG}.csv \
  --run_viz 0 \
  --run_qa 1
```

### 4) VLM Teacher 라벨 재생성
```bash
bash src/scripts/run_vlm_teacher_labeler.sh \
  --server_mode 0 \
  --venv_path "$VENV_ACT" \
  --teacher_scores_jsonl data/SSTK/${DATANAME}/artifacts/teacher/scores/teacher_scores_ar_${REFRESH_TAG}.jsonl \
  --image_dir "$IMAGE_DIR" \
  --output_jsonl data/SSTK/${DATANAME}/artifacts/vlm_teacher/labels/crop_label_${REFRESH_TAG}.jsonl \
  --output_meta_jsonl data/SSTK/${DATANAME}/artifacts/vlm_teacher/meta/meta_norm_${REFRESH_TAG}.jsonl \
  --summary_json data/SSTK/${DATANAME}/artifacts/vlm_teacher/summary/vlm_teacher_summary_${REFRESH_TAG}.json \
  --backend qwen25_vl \
  --model_id Qwen/Qwen3-VL-4B-Instruct \
  --device auto \
  --dtype auto \
  --top_m 12 \
  --top_k 5
```

요약:
- `C5/C6` 재실행만으로도 `teacher_scores`/`vlm labels`는 증분 갱신 가능
- Candidate는 재생성하지 않아도 됨(후보 로직 변경이 없다면)
- 단, C6을 scorer의 gaze/lookroom 로직에 직접 반영하려면 `c3` 갱신까지 포함

---

## 8.2 C4 OCR(PP-OCRv5) 재실행 후 Routing/Teacher/VLM 갱신

현재 C4는 아래 정책으로 동작합니다.
- `priority=quality_first`: `PP-OCRv5_server_det` 1순위
- `priority=high_efficiency`: `PP-OCRv5_mobile_det` 1순위
- fallback: legacy PaddleOCR det-only

`worker_core.py`에서 C4 결과를 다음 alias로 자동 기록합니다.
- `c4_ocr`, `c4_ocr_meta`
- `ocr_text_boxes`, `ocr_num_boxes`, `ocr_box_count`, `c4_text_boxes`, `c4_ocr_boxes`
- `text_overlay_likely`, `has_text_overlay`, `ocr_text_overlay_likely`

사전 확인(venv):
```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate
# ImportError: soft_unicode 이슈가 있으면 선행 설치
pip install -U "jinja2>=3.1.4"
python - <<'PY'
import paddle
from paddleocr import TextDetection
print("paddle:", paddle.__version__)
print("TextDetection: OK")
PY
```

선택 환경변수:
```bash
# 백엔드 강제: auto|ppocrv5_server|ppocrv5_mobile|legacy_paddleocr|disabled
export C4_BACKEND=auto
# 기본 모델 override
export C4_PPOCR_SERVER_MODEL=PP-OCRv5_server_det
export C4_PPOCR_MOBILE_MODEL=PP-OCRv5_mobile_det
# 디바이스 override (예: gpu:0, cpu)
export C4_PPOCR_DEVICE=gpu:0
```

### 1) C4만 quality_first로 재실행
```bash
DATANAME=10K_local
REFRESH_TAG=c4_refresh_260305
PARQUET=data/SSTK/${DATANAME}/filtered_sstk_100.parquet
IMAGE_DIR=data/SSTK/${DATANAME}/images
FEATS_C4=data/SSTK/${DATANAME}/artifacts/precompute/feats_c4_${REFRESH_TAG}.jsonl
export FEATS_C4

bash src/scripts/run_extract_component.sh \
  "$PARQUET" sstk_100 "$FEATS_C4" \
  --component c4 \
  --priority quality_first \
  --server_mode 0 \
  --venv_path /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate \
  --image_dir "$IMAGE_DIR" \
  --tar_dir /media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100 \
  --batch_size 8
```

검증(메서드/박스 수):
```bash
python - <<'PY'
import json, os
from collections import Counter
path = os.environ["FEATS_C4"]
m = Counter(); b = 0
with open(path, "r", encoding="utf-8") as f:
    for ln in f:
        d = json.loads(ln)
        meta = d.get("c4_ocr_meta", {}) if isinstance(d.get("c4_ocr_meta"), dict) else {}
        m[meta.get("method", "none")] += 1
        b += int(d.get("ocr_text_boxes", 0) or 0)
print("c4 methods:", dict(m))
print("sum ocr_text_boxes:", b)
PY
```

### 2) 기존 feature와 병합 + subject routing 재생성
```bash
BASE_FEATS=data/SSTK/${DATANAME}/artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl
FEATS_REFRESHED=data/SSTK/${DATANAME}/artifacts/precompute/feats_c2c3c4c5_${REFRESH_TAG}.jsonl
FEATS_ROUTED=data/SSTK/${DATANAME}/artifacts/precompute/feats_c2c3c4c5_${REFRESH_TAG}_routed.jsonl

python src/scripts/merge_feature_jsonl.py \
  --input_parquet "$PARQUET" \
  --inputs "$BASE_FEATS" "$FEATS_C4" \
  --output_jsonl "$FEATS_REFRESHED"

python src/scripts/enrich_subject_mode_jsonl.py \
  --input_feats_jsonl "$FEATS_REFRESHED" \
  --input_filtered_parquet "$PARQUET" \
  --output_jsonl "$FEATS_ROUTED" \
  --c2_top_n 5 \
  --c2_union_top_m 3 \
  --allow_det_proxy 1
```

### 3) Teacher/VLM 재생성
`8.1`의 3)~4)와 동일하게 실행하되 `--features_jsonl`에 `"$FEATS_ROUTED"`를 넣으면 됩니다.

### 4) e2e 스크립트 토글
`run_phaseA_to_teacher_e2e.sh`는 `--run_c4`, `--run_c6`를 지원합니다(기본 1).
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 0 \
  --data_dir data/SSTK/10K_local \
  --run_filter 0 \
  --precompute_mode unified \
  --run_c4 1 \
  --run_c6 1 \
  --extract_priority quality_first
```

---

## 9. 문서/코드 매핑

설계 문서:
- `ImplementPlan_Docs/SSTK_Cropping_DataFactory_QwenLabeler_Reorganized_KO_v1_9.md`
- `ImplementPlan_Docs/SSTK_Cropping_DataFactory_QwenLabeler_Reorganized_EN_v1_9.md`

구현 코드:
- Filter: `src/filter_sstk_dataset.py`, `src/scripts/run_filter.sh`
- Precompute: `src/extract_features/*`, `src/extract_features_single.py`, `src/extract_features_pipeline.py`, `src/scripts/run_extract_component.sh`
- C3 enrich: `src/scripts/enrich_c3_pose_jsonl.py`
- Candidate: `src/generate_candidates.py`, `src/scripts/run_generate_candidates.sh`
- Teacher: `src/score_teacher.py`, `src/scripts/run_teacher_scorer.sh`, `src/scripts/qa_teacher_report.py`, `src/visualize_teacher_scores.py`
- VLM Teacher: `src/vlm_teacher_labeler.py`, `src/scripts/run_vlm_teacher_labeler.sh`, `src/visualize_vlm_teacher_labels.py`, `src/scripts/run_visualize_vlm_teacher.sh`
- End-to-end: `src/scripts/run_phaseA_to_teacher_e2e.sh`
