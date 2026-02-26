# SSTK Cropping Data Factory v1.7 실행 가이드 (Phase A ~ 9 Teacher Scorer)

이 문서는 `SSTK_Cropping_DataFactory_QwenLabeler_Reorganized_KO_v1_7.md` 기준으로, 현재 코드베이스에서 **Phase A(Filter) → Phase B(Perception/Candidate) → 9) Teacher Scorer**까지를 처음부터 재현하는 실전 운영 가이드입니다.

범위:
- 포함: `3) Phase A`, `5) Phase B`, `9) Teacher Scorer`
- 제외: `C4 OCR`, `v1.7 PICD`, `10) Qwen2.5-VL Teacher`

---

## 1. 핵심 리팩터링 요약

기존에는 C2/C3/C5를 분리 실행하는 경로가 중심이었고, 단계가 다수로 쪼개져 운영이 번거로웠습니다. 현재는 아래 방식으로 정리되었습니다.

- 신규 end-to-end 오케스트레이터 추가
  - `src/scripts/run_phaseA_to_teacher_e2e.sh`
  - Filter → Precompute → Candidate → Teacher(+QA/+Viz)까지 1개 커맨드로 실행
- Precompute 통합 모드(`--precompute_mode unified`) 도입
  - C1/C2/C3(+C5)를 1-pass로 추출 가능 (`run_c1=1`일 때 C1 포함)
  - `enrich_c3_pose_jsonl.py`로 face/gaze proxy 보강 후 최종 병합 피처 직접 생성
  - 분리 실행 대비 I/O/재로딩 오버헤드 감소
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

---

## 3. 전체 파이프라인 (권장 운영 경로)

권장: `run_phaseA_to_teacher_e2e.sh` 하나로 실행

### 3.1 로컬(가상환경 자동 활성화) 기본 실행
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 0 \
  --data_dir data/SSTK/10K_local \
  --bucket sstk_100 \
  --precompute_mode unified \
  --run_tag v17_local
```

### 3.2 서버 기본 실행
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/10K \
  --bucket sstk_100 \
  --precompute_mode unified \
  --run_tag v17_server
```

### 3.2.1 서버 멀티 GPU 효율 실행(No-Ray 기본)
# 필터 결과가 이미 있을 때 10K_local w/ 2-gpu
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/10K_local \
  --run_filter 0 \
  --precompute_mode unified \
  --extract_mode auto \
  --extract_gpu_ids 0,1 \
  --num_workers 2 \
  --use_real_expensive 1 \
  --teacher_multi_gpu 1 \
  --teacher_gpu_ids 0,1 \
  --teacher_num_workers 2 \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 12 \
  --skip_existing 0 \
  --run_tag real_exp \
   | tee src/scripts/logs/run_phaseA_to_teacher_e2e.log

```
# 필터 결과가 이미 있을 때 10K w/ 4-gpu
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/10K \
  --run_filter 0 \
  --precompute_mode unified \
  --extract_mode auto \
  --extract_gpu_ids 0,1,2,3 \
  --num_workers 4 \
  --use_real_expensive 1 \
  --teacher_multi_gpu 1 \
  --teacher_gpu_ids 0,1,2,3 \
  --teacher_num_workers 4 \
  --use_real_expensive 1 \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 12 \
  --run_tag real_exp \
   | tee src/scripts/logs/run_phaseA_to_teacher_e2e.log
```

### 3.3 필터 결과가 이미 있을 때(재실행 시간 단축)
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/10K \
  --run_filter 0 \
  --skip_existing 1 \
  --precompute_mode unified \
  --run_tag rerun1
```

### 3.4 Real Expensive(미학+crop 임베딩 cosine) 활성화
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/10K \
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

### 3.5 GPU 서버 smoke/full 예시
- smoke (후보/teacher만 500장):
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/10K \
  --run_filter 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c3_enrich 0 --run_c5 0 --run_merge 0 \
  --run_candidates 1 --run_teacher 1 \
  --use_real_expensive 1 \
  --align_device cuda --aesthetic_device cuda --exp_batch_size 12 \
  --max_images 500 \
  --run_tag real_exp_500
```
- full (전량): `--max_images 0`

### 3.6 처음부터 끝까지(명시형 풀 옵션 템플릿)
```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --bucket sstk_100 \
  --data_dir data/SSTK/10K \
  --server_mode 1 \
  --tar_dir /sstk/20230916/sstk_100 \
  --run_filter 1 \
  --curated_pool_size 10000 \
  --top_percentile 0.2 \
  --skip_existing 1 \
  --precompute_mode unified \
  --run_c1 -1 \
  --run_c2 1 --run_c3 1 --run_c3_enrich 1 --run_c5 1 --run_merge 1 \
  --extract_mode auto \
  --extract_priority quality_first \
  --batch_size 16 \
  --c3_person_verify_strict 1 \
  --run_candidates 1 \
  --use_actual_image_size 1 \
  --strict_actual_size 1 \
  --run_teacher 1 \
  --use_real_expensive 1 \
  --cheap_top_m 30 \
  --top_k 5 \
  --tau_div 0.75 \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 12 \
  --run_qa 1 \
  --run_viz 1 \
  --num_viz 120 \
  --run_tag full_v17
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

---

## 4.2 Phase B-1: Perception Precompute (C1/C2/C3/C5)

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

권장 결과물:
- `feats_c2c3c5_v2_strict_enriched.jsonl`

---

## 4.3 Phase B-2: Candidate Generator (AR-조건부)

엔트리:
- `src/scripts/run_generate_candidates.sh`
- `src/generate_candidates.py`

핵심 방법론:
- 입력: filtered parquet + C2/C3
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
  - `candidates_ar*.jsonl`
  - overview json/csv + 시각화(`visualizations/candidates_*`)

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
  - AR/면적 범위/face-cut/joint-cut 등 구조적 실패 필터
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
- `teacher_scores_ar*.jsonl`
- `teacher_scores_overview*.json/.csv`
- `teacher_scores_qa_report*.json/.csv`
- 시각화 디렉토리

---

## 5. 실행 스크립트 옵션 가이드

### 5.1 `run_phaseA_to_teacher_e2e.sh` 핵심 옵션

- `--server_mode 0|1`: 로컬/서버 모드
- `--data_dir`: 출력 루트
- `--run_filter 0|1`: Filter 수행 여부
- `--precompute_mode unified|split`:
  - `unified` 권장 (C1/C2/C3/C5를 조건부 1-pass)
  - `run_c1=1`이면 C1 포함, `run_c1=0`이면 C2/C3/C5만 수행
  - `split` 레거시(컴포넌트별 분리)
- `--extract_mode auto|single|multi|ray`:
  - `multi` = No-Ray 멀티 GPU 샤딩
  - `ray` = 명시적 레거시 Ray 모드
- `--extract_gpu_ids`: precompute에 사용할 GPU 목록 CSV
- `--run_c1 --run_c2 --run_c3 --run_c3_enrich --run_c5 --run_merge`
- `--run_candidates --run_teacher`
- `--use_real_expensive 0|1`
- `--teacher_multi_gpu -1|0|1`: -1이면 auto(real-expensive + multi-gpu 환경에서 자동 on), `>0` 입력도 on으로 처리
- `--teacher_gpu_ids`: teacher 멀티 GPU 목록 CSV
- `--teacher_num_workers`: teacher shard worker 수
- `--max_images`: candidate/teacher 단계 처리 수 제한
- `--skip_existing 0|1`: 산출물 존재 시 skip
- `--run_tag`: 결과 파일 suffix

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

### 5.4 `run_teacher_scorer.sh` 핵심 옵션

- `--candidates_jsonl`, `--features_jsonl`, `--c1_jsonl`
- `--cheap_top_m`, `--top_k`, `--tau_div`
- `--use_real_expensive 1` 시
  - `--align_device`, `--aesthetic_device`, `--exp_batch_size`
  - `--aesthetic_mlp_path`, `--aesthetic_mlp_url`
- 멀티 GPU 샤딩:
  - `--multi_gpu 1`
  - `--gpu_ids 0,1,2,3`
  - `--num_workers 4`
- `--run_qa 1`, `--run_viz 1`, `--num_viz`

---

## 6. End-to-End 결과물 맵

기본(`data_dir=data/SSTK/10K_local`, `run_tag=v17_local`) 예시:

- Filter
  - `data/SSTK/10K_local/filtered_sstk_100.parquet`
- Precompute
  - `data/SSTK/10K_local/feats_c2c3c5_v2_strict_raw.jsonl` (unified 중간)
  - `data/SSTK/10K_local/feats_c2c3c5_v2_strict_enriched.jsonl` (최종)
  - `data/SSTK/10K_local/feats_c1.jsonl` (split 모드 또는 별도 C1 추출 시)
  - `precompute_mode=unified` + `run_c1=1`이면 C1은 위 unified jsonl에 함께 저장됨
- Candidate
  - `data/SSTK/10K_local/candidates_ar_v17_local.jsonl`
  - overview/viz 파일들
- Teacher
  - `data/SSTK/10K_local/teacher_scores_ar_v17_local.jsonl`
  - `teacher_scores_overview_v17_local.json`
  - `teacher_scores_overview_by_ar_v17_local.csv`
  - `teacher_scores_qa_report_v17_local.json`
  - `teacher_scores_qa_report_by_ar_v17_local.csv`
  - `visualizations/teacher_scorer_v17_local/`
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
  - `actual_image_size_map.json` 캐시를 재사용하면 속도 개선
- C3 품질 이슈
  - `C3_PERSON_VERIFY_STRICT=1` 유지
  - 가능하면 `unified` 모드(C2 person hint 동시 사용) 권장

---

## 8. 레거시 분리 실행이 필요한 경우

아래 순서로 수동 실행 가능:

1. Filter: `run_filter.sh`
2. C2/C3/C5 각각 `run_extract_component.sh`
3. C3 enrich: `enrich_c3_pose_jsonl.py`
4. merge: `merge_feature_jsonl.py`
5. Candidate: `run_generate_candidates.sh`
6. Teacher: `run_teacher_scorer.sh`

단, 신규 운영은 `run_phaseA_to_teacher_e2e.sh --precompute_mode unified`를 기본으로 권장합니다.

---

## 9. 문서/코드 매핑

설계 문서:
- `ImplementPlan_Docs/SSTK_Cropping_DataFactory_QwenLabeler_Reorganized_KO_v1_7.md`
- `ImplementPlan_Docs/SSTK_Cropping_DataFactory_QwenLabeler_Reorganized_EN_v1_7.md`

구현 코드:
- Filter: `src/filter_sstk_dataset.py`, `src/scripts/run_filter.sh`
- Precompute: `src/extract_features/*`, `src/extract_features_single.py`, `src/extract_features_pipeline.py`, `src/scripts/run_extract_component.sh`
- C3 enrich: `src/scripts/enrich_c3_pose_jsonl.py`
- Candidate: `src/generate_candidates.py`, `src/scripts/run_generate_candidates.sh`
- Teacher: `src/score_teacher.py`, `src/scripts/run_teacher_scorer.sh`, `src/scripts/qa_teacher_report.py`, `src/visualize_teacher_scores.py`
- End-to-end: `src/scripts/run_phaseA_to_teacher_e2e.sh`
