# MobileCropNet v4.0 GAIC 실험 결과 보고서

작성일: 2026-04-16

## 1. 목적

본 문서는 `MobileCropNet_v4_0.md` 설계를 기준으로 구현한 MobileCropNet v4.0의 GAIC replay-label 학습, 추론, 공식 GAIC benchmark 정량 평가, 정성 시각화, GPU 사용률 측정 결과를 정리한다. 대상 학습 라벨은 현재 데이터 팩토리 파이프라인에서 생성된 `data/GAIC/All/artifacts/training_labels/gaic_personv6_server_v1_leftover_ignore_monotonic` 계열 라벨을 MobileCropNet v4 split으로 변환한 결과다.

평가 기준은 이번 수정부터 명확히 분리한다. 제품/논문 수준의 1차 성능 판단은 SSTK teacher score가 아니라 `data/Publics/GAIC_v2/annotations_json/instances_test.json`의 GAIC v2 논문 split 500장 MOS annotation을 ground truth로 사용한다. teacher-label 기반 지표는 학습 label replay와 teacher 모사 정도를 보는 보조 진단으로만 해석한다.

핵심 결론은 다음과 같다.

- 현재 네 실험 중 `mcn-v4-q24-288-w075-20260416-095704`가 replay-label GAIC test 기준 최상위이며, 공식 GAIC benchmark 평가도 이 checkpoint로 수행했다.
- GAIC v2 test 500장, annotation crop 43,123개 전체 평가에서 MobileCropNet v4 best run은 PCC `0.399840`, SRCC `0.368395`, Acc1/5 `0.434000`, Acc1/10 `0.630000`, Acc4/10 `0.469000`, Accw4/10 `0.365610`, top-1 MOS `3.804520`, MOS regret `0.427380`을 기록했다.
- 현 candidate/features 산출물로 teacher를 재실행할 수 있는 150장 subset에서는 SSTK teacher가 PCC/SRCC/Acc 계열에서 더 높지만, MobileCropNet v4가 top-1 MOS와 MOS regret은 근소하게 더 낫다. 따라서 teacher score는 학습 신호로 유용하지만 전문가 MOS ground truth를 대체할 수 없다.
- best run의 replay-label test top-1 hit는 `0.353600`, NDCG@5는 `0.896934`, SRCC는 `0.614445`, proposal recall@5 IoU 0.5는 `0.867200`이다.
- baseline candidate 대비 utility regret이 `0.483818`에서 `0.052007`로 크게 감소했다.
- teacher score oracle과는 아직 top-1 hit 기준 약 `0.3776p` 차이가 있어 ranking head와 proposal-to-candidate coupling 개선 여지가 크다.
- GPU 사용률은 평균 `11.25%`부터 `14.86%`, 최대 `30%` 수준이고 VRAM 사용량은 1GB 미만이다. 현재 규모에서는 2개 이상 GPU가 필요하지 않다.

## 2. 구현 및 산출물 범위

이번 작업에서 검증한 MobileCropNet v4.0 구현 범위는 다음과 같다.

- 학습 코드: `src/scripts/train_mobilecropnet_v4.py`
- 평가 코드: `src/scripts/evaluate_mobilecropnet_v4.py`
- 비교 평가 코드: `src/scripts/compare_mobilecropnet_v4_methods.py`
- 추론 코드: `src/scripts/infer_mobilecropnet_v4.py`
- 공식 GAIC benchmark 평가 코드: `src/scripts/evaluate_mobilecropnet_v4_gaic_benchmark.py`
- 공식 GAIC benchmark metric/util 코드: `src/mobilecropnet_v4/gaic_benchmark.py`
- GAIC v2 COCO-style 변환 코드: `src/scripts/convert_gaic_v2_to_coco.py`
- PNG 시각화 코드: `src/scripts/visualize_mobilecropnet_v4_predictions.py`
- 리포트 생성 코드: `src/scripts/build_mobilecropnet_v4_report.py`
- 실험 요약 코드: `src/scripts/summarize_mobilecropnet_v4_experiments.py`
- 단위 테스트: `tests/test_mobilecropnet_v4.py`, `tests/test_mobilecropnet_v4_gaic_benchmark.py`

시각화 산출물은 모두 PNG로 생성했다. 각 run 디렉터리의 `viz_test/contact_sheet.png`와 `viz_test/overlays/*.png`가 정성 확인용 결과다.

## 3. 데이터 및 split

학습과 검증은 GAIC 변환 split을 사용했다.

| split | path | image count | candidate pool count | matching target count |
| --- | --- | ---: | ---: | ---: |
| train-dev | `artifacts/mobilecropnet_v4/gaic_splits/gaic_personv6_v4_train_dev.jsonl` | 4,380 | 89,816 | 9,425 |
| val-dev | `artifacts/mobilecropnet_v4/gaic_splits/gaic_personv6_v4_val_dev.jsonl` | 477 | 9,597 | 1,040 |
| test | `artifacts/mobilecropnet_v4/gaic_splits/gaic_personv6_v4_test.jsonl` | 625 | - | - |

train-dev와 val-dev는 `FREE`, `1:1`, `9:16`, `16:9`, `3:4`, `4:3` target aspect ratio를 포함한다. 학습 입력은 현재 label schema의 `baseline`, `decision_target`, `matching_targets`, `candidate_pool`을 직접 사용한다.

## 4. GPU 실행 환경

GPU smoke 이후 병렬 full experiment 3개를 MLP spot workload로 생성했고, 별도 notebook형 GPU `10.8.103.17`에서도 1개 실험을 실행했다.

| run | run id | GPU path | IP | status | start UTC | end UTC |
| --- | ---: | --- | --- | --- | --- | --- |
| `mcn-v4-q16-256-w075-20260416-095704` | 1093849 | MLP spot workload | 10.8.103.14 | Succeeded | 2026-04-16T00:59:21Z | 2026-04-16T01:13:16Z |
| `mcn-v4-q24-288-w075-20260416-095704` | 1093850 | MLP spot workload | 10.8.103.1 | Succeeded | 2026-04-16T01:00:08Z | 2026-04-16T01:13:27Z |
| `mcn-v4-q16-256-w100-20260416-095704` | 1093848 | MLP spot workload | 10.8.103.194 | Succeeded | 2026-04-16T00:59:24Z | 2026-04-16T01:13:16Z |
| `mcn-v4-nb-q16-224-w075-20260416-095704` | - | notebook GPU | 10.8.103.17 | Completed | - | - |

MLP workload image는 `sr-ar-interactive-media-exp/jy-cropping-260415-tfs4.57.6-rsync`이고, 각 workload는 `nvidia-smi`, torch CUDA smoke, 학습, val/test 평가, 비교, PNG 시각화, 리포트 생성을 self-contained command로 실행했다. 감지된 GPU는 NVIDIA A100-SXM4-80GB 1장이다.

## 5. 실험 설정

| run | input size | proposal q | width mult | token dim | candidate k | epochs | batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mcn-v4-q24-288-w075-20260416-095704` | 288 | 24 | 0.75 | 128 | 24 | 8 | 24 |
| `mcn-v4-nb-q16-224-w075-20260416-095704` | 224 | 16 | 0.75 | 128 | 24 | 6 | 32 |
| `mcn-v4-q16-256-w075-20260416-095704` | 256 | 16 | 0.75 | 128 | 24 | 8 | 32 |
| `mcn-v4-q16-256-w100-20260416-095704` | 256 | 16 | 1.00 | 128 | 24 | 8 | 32 |

공통 학습 설정은 cosine scheduler, warmup 1 epoch, AMP, `num_workers=4`, `gpu_usage_sample_interval=5`다.

## 6. 평가지표 정의와 해석

MobileCropNet v4.0 평가는 단일 지표가 아니라 ranking 품질, label utility 보존, proposal geometry 품질, score ordering 품질을 함께 본다. 특히 crop 품질은 박스 IoU만으로 판단하기 어렵다. 큰 박스나 full crop은 IoU가 높게 나올 수 있지만, teacher score가 낮고 실제 aesthetic/subject-preserving crop으로는 부적절할 수 있다.

이번 수정 이후 제품/논문 수준의 주 평가는 GAIC 논문(`https://arxiv.org/pdf/1909.08989`)의 공식 benchmark 방식에 맞춰, 이미지별 공식 annotation crop 전체의 MOS와 모델/teacher score를 비교한다. replay-label 지표는 학습 라벨을 얼마나 잘 모사하는지 확인하는 보조 지표다.

공식 GAIC MOS benchmark 지표는 다음과 같이 해석한다.

| metric | range | direction | 의미 | 해석 기준 |
| --- | --- | --- | --- | --- |
| `PCC` | -1 to 1 | 높을수록 좋음 | 이미지별 official MOS와 predicted score의 선형 상관을 계산한 뒤 이미지 평균 | 1이면 score scale이 MOS와 완전 선형 일치, 0이면 선형 관계가 약함, -1이면 완전 역상관이다. |
| `SRCC` | -1 to 1 | 높을수록 좋음 | 이미지별 official MOS ranking과 predicted ranking의 Spearman 순위 상관 | crop score scale보다 순서가 중요한 경우 핵심 지표다. 1이면 모든 후보 순위가 MOS와 일치한다. |
| `AccK/N` | 0 to 1 | 높을수록 좋음 | 모델이 상위 K개로 반환한 crop 중 official MOS top-N crop에 속한 비율 | GAIC benchmark의 Best Return metric이다. 예를 들어 Acc1/5는 모델 top-1이 official MOS 상위 5개 안에 들어간 이미지 비율이다. |
| `AccwK/N` | 0 to 1 | 높을수록 좋음 | `AccK/N`에 top-N 내부 rank 가중치를 반영한 지표 | top-N 안에 들더라도 rank-1에 가까운 crop을 더 높게 평가한다. GAIC 논문 설정처럼 K는 1,2,3,4, N은 5,10을 사용했다. |
| `top1 MOS` | 보통 1 to 5 | 높을수록 좋음 | 모델이 첫 번째로 반환한 crop의 official mean opinion score | 실제 사용자에게 1개 crop만 보여주는 제품 시나리오에 가깝다. |
| `MOS regret` | 0 이상 | 낮을수록 좋음 | 해당 이미지의 best official MOS와 모델 top-1 MOS의 차이 | 0이면 official MOS 최고 crop을 top-1로 골랐다는 뜻이다. |
| `top1 rank percentile` | 0 to 1 | 높을수록 좋음 | 모델 top-1 crop의 official MOS rank percentile | 1이면 rank-1 crop, 0에 가까울수록 하위권 crop이다. |
| `eval/score coverage` | 0 to 1 | 높을수록 좋음 | 해당 method가 평가된 official test 범위 또는 teacher score가 존재하는 annotation 비율 | MobileCropNet full row는 500장 전체라 1.0이다. teacher exact row는 현재 candidate/features가 존재하는 150장 subset에서 score coverage 1.0이다. |

Replay-label 진단 지표는 다음과 같이 해석한다.

| metric | range | direction | 의미 | 해석 기준 |
| --- | --- | --- | --- | --- |
| `test top-1 hit` / `candidate_top1_hit` | 0 to 1 | 높을수록 좋음 | 모델이 선택한 최종 crop이 label pipeline에서 positive로 판정된 후보인지의 비율 | 1이면 모든 test sample에서 positive crop을 선택했다는 뜻이다. 본 실험 best `0.353600`은 35.36% sample에서 positive 후보를 top-1로 선택했다는 의미다. |
| `exact best` / `candidate_top1_exact_best` | 0 to 1 | 높을수록 좋음 | 모델이 후보 set 안에서 teacher score가 가장 높은 후보를 정확히 선택한 비율 | top-1 hit보다 더 엄격하다. positive 후보가 여러 개 있을 때도 최고 score 후보를 맞혀야 1로 계산된다. |
| `chosen label score` | 보통 0 to 1 | 높을수록 좋음 | 모델이 선택한 crop의 teacher label score 평균 | label generator가 부여한 utility를 얼마나 보존했는지 나타낸다. best run의 `0.706283`은 baseline `0.274472`보다 훨씬 높다. |
| `utility regret` | 0 이상 | 낮을수록 좋음 | 후보 set 내부 teacher oracle score와 모델 선택 score의 차이 | 0이면 후보 set에서 teacher 최고점 crop을 선택했다는 뜻이다. best run `0.052007`은 평균적으로 oracle 대비 약 0.052 score를 놓친다는 의미다. |
| `NDCG@5` | 0 to 1 | 높을수록 좋음 | 모델 utility 순위 상위 5개가 teacher score 순위와 얼마나 일치하는지 측정하는 ranking 지표 | top-1 하나만 보는 지표보다 안정적이다. best run `0.896934`는 상위권 후보 순서가 teacher ranking과 꽤 잘 맞는다는 뜻이다. |
| `NDCG@10` | 0 to 1 | 높을수록 좋음 | 상위 10개 후보 기준 ranking 품질 | 후보 후보군 전반의 ranking quality를 본다. 상위 후보 탐색이나 후처리 reranking을 붙일 때 중요하다. |
| `SRCC` | -1 to 1 | 높을수록 좋음 | Spearman rank correlation coefficient. 모델 score 순위와 teacher score 순위의 단조 순위 상관 | 1이면 순위가 완전히 일치하고, 0이면 순위 관계가 거의 없으며, -1이면 완전 역순이다. best run `0.614445`는 중간 이상 양의 순위 상관을 의미한다. |
| `PCC` | -1 to 1 | 높을수록 좋음 | Pearson correlation coefficient. 모델 score와 teacher score의 선형 상관 | score calibration 품질을 본다. SRCC가 순위 중심이라면 PCC는 score scale의 선형 일치도까지 반영한다. |
| `top1 IoU to best positive` | 0 to 1 | 높을수록 좋음 | 모델 top-1 crop 박스와 label best positive 박스의 IoU | geometry 유사도를 본다. 단, 큰 baseline crop이 유리할 수 있어 단독 품질 지표로 쓰면 안 된다. |
| `proposal recall@5 IoU0.5` | 0 to 1 | 높을수록 좋음 | learned proposal 상위 5개 중 하나라도 best positive와 IoU 0.5 이상 겹치는 비율 | proposal generator가 정답 crop 주변을 찾는 능력이다. best run `0.867200`은 86.72% sample에서 상위 5개 proposal이 positive geometry를 회수했다는 뜻이다. |
| `proposal recall@1 IoU0.5` | 0 to 1 | 높을수록 좋음 | top-1 proposal 하나만 봤을 때 IoU 0.5 이상인 비율 | proposal head 단독 품질을 더 엄격하게 본다. |
| `proposal best IoU@5` | 0 to 1 | 높을수록 좋음 | 상위 5개 proposal 중 best positive와 가장 IoU가 높은 값의 평균 | threshold 기반 recall보다 연속적인 proposal geometry 품질을 보여준다. |
| `decision_acc` | 0 to 1 | 높을수록 좋음 | label pipeline의 decision target을 맞힌 비율 | crop을 선택/수정/유지하는 정책 head가 current label pipeline의 decision을 얼마나 모사하는지 나타낸다. |
| `score_mae` | 0 이상 | 낮을수록 좋음 | predicted score와 teacher score의 mean absolute error | score calibration 오차다. 낮을수록 teacher score scale을 잘 재현한다. |
| `loss` 계열 | 0 이상 | 낮을수록 좋음 | 학습 objective의 각 손실값 | train/val 추세와 overfitting 확인에 사용한다. 서로 다른 loss weight 실험 간에는 직접 비교에 주의해야 한다. |

비교 평가에서 사용하는 method의 의미는 다음과 같다.

| method | 의미 | 해석 |
| --- | --- | --- |
| `v4_model` | MobileCropNet v4.0이 실제로 선택한 최종 crop | 배포 후보 모델의 실성능이다. |
| `baseline_candidate` | current label pipeline에 포함된 baseline crop | 일반적으로 full/max-area 또는 기존 기본 후보에 가깝다. 개선율 계산의 기준선이다. |
| `teacher_score_oracle` | replay candidate set 내부에서 teacher score가 가장 높은 후보 | 배포 가능한 모델이 아니라 후보 set 내부 상한이다. v4가 도달 가능한 ranking 상한을 가늠한다. |
| `positive_oracle` | positive 후보 중 score가 가장 높은 후보 | positive definition 기준 상한이다. teacher 최고점 후보와 거의 같지만, positive threshold 정의의 영향을 받는다. |
| `proposal_top1` | learned proposal head의 첫 번째 proposal | candidate ranking 없이 proposal generator만 보았을 때의 geometry 품질을 점검한다. |

본 보고서에서 모델 선택의 1차 기준은 `test top-1 hit`, `NDCG@5`, `utility regret`이다. `proposal recall@5 IoU0.5`는 proposal head가 충분한 후보 위치를 생성하는지 확인하는 보조 기준이고, `top1 IoU to best positive`는 geometry sanity check로 사용한다. 제품 수준 판단에서는 이 지표들에 latency, model size, mobile memory, public benchmark 일반화 성능을 추가해야 한다.

## 7. 공식 GAIC benchmark 평가

공식 GAIC benchmark 평가는 논문 split 구조인 `data/Publics/GAIC_v2/annotations/{train,val,test}`와 `data/Publics/GAIC_v2/images/{train,val,test}`를 COCO-style JSON으로 재생성한 뒤 수행했다. 생성된 split은 train 2,636장, val 200장, test 500장이다. txt annotation의 좌표는 `y1 x1 y2 x2 MOS`이며 COCO bbox `[x, y, w, h]`로 변환했다. `MOS=-2`는 GAIC_v2의 unrated crop으로 보고 제외했고, 기존 `data/Publics/GAIC/annotations_json` subset과 겹치는 train 1,939장/test 349장에서는 bbox, MOS, `gt_flag`가 완전히 일치함을 검증했다. 이 평가는 SSTK teacher label을 ground truth로 사용하지 않는다.

평가 산출물:

- COCO JSON summary: `data/Publics/GAIC_v2/annotations_json/conversion_summary.json`
- train JSON: `data/Publics/GAIC_v2/annotations_json/instances_train.json`
- val JSON: `data/Publics/GAIC_v2/annotations_json/instances_val.json`
- test JSON: `data/Publics/GAIC_v2/annotations_json/instances_test.json`
- metrics: `artifacts/mobilecropnet_v4/gaic_benchmark_eval/q24_288_w075_gaic_v2_test500/metrics.json`
- report: `artifacts/mobilecropnet_v4/gaic_benchmark_eval/q24_288_w075_gaic_v2_test500/GAIC_BENCHMARK_EVAL_REPORT.md`
- model per-image: `artifacts/mobilecropnet_v4/gaic_benchmark_eval/q24_288_w075_gaic_v2_test500/mcn_v4_q24_288_w075_per_image.jsonl`
- teacher exact per-image: `artifacts/mobilecropnet_v4/gaic_benchmark_eval/q24_288_w075_gaic_v2_test500/sstk_teacher_subjectfix_v2_equalized_official_gc_per_image.jsonl`
- teacher benchmark rerun: `artifacts/mobilecropnet_v4/gaic_benchmark_eval/q24_288_w075_gaic_v2_test500/teacher_benchmark_v2_overlap/benchmark_summary.json`

생성된 COCO-style split 요약은 다음과 같다.

| split | images | valid annotation crops | positive crops (`MOS > 4.0`) | skipped `MOS=-2` crops |
| --- | ---: | ---: | ---: | ---: |
| train | 2,636 | 227,755 | 10,392 | 3,734 |
| val | 200 | 17,164 | 878 | 181 |
| test | 500 | 43,123 | 1,900 | 472 |

전체 500장 기준 MobileCropNet v4 결과는 다음과 같다.

| method | scope | images | candidates | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/10 | Accw4/10 | top1 MOS | MOS regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mcn_v4_q24_288_w075` | GAIC v2 test 500 | 500 | 43,123 | 0.399840 | 0.368395 | 0.434000 | 0.630000 | 0.469000 | 0.365610 | 3.804520 | 0.427380 |
| `gaic_mos_oracle` | GAIC v2 test 500 upper bound | 500 | 43,123 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 4.231900 | 0.000000 |

SSTK teacher와의 비교는 같은 official MOS annotation을 기준으로 수행했다. 기존 exact teacher candidate scoring 산출물은 118장만 포함했지만, 이번에 GAIC v2 test id로 제한해 teacher benchmark를 재실행하여 현 candidate/features가 커버하는 150장까지 확장했다. 다만 기존 `data/GAIC/All/artifacts/candidates`와 `data/GAIC/All/artifacts/precompute` 산출물 자체가 GAIC v2 test 500장 중 150장만 커버하므로, 아래 표는 150장 overlap subset에서만 모델과 teacher를 직접 비교한 결과다.

| method | scope | images | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/5 | Acc4/10 | Accw4/10 | top1 MOS | MOS regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mcn_v4_q24_288_w075__teacher_overlap` | teacher exact overlap | 150 | 0.352746 | 0.322241 | 0.240000 | 0.406667 | 0.193333 | 0.353333 | 0.256321 | 3.680133 | 0.772467 |
| `sstk_teacher_subjectfix_v2_equalized_official_gc` | teacher exact overlap | 150 | 0.498440 | 0.502193 | 0.286667 | 0.460000 | 0.248333 | 0.408333 | 0.313997 | 3.673067 | 0.779533 |

해석:

- MobileCropNet v4는 GAIC v2 test 500장 전체에 대해 직접 scoring되었고, 이 값이 현재 제품/논문 판단용 primary metric이다.
- teacher exact 비교에서는 SSTK teacher가 PCC, SRCC, AccK/N, AccwK/N에서 MobileCropNet보다 높다. 이는 teacher score가 official MOS ranking과 어느 정도 양의 상관을 가진다는 의미다.
- 같은 150장 subset에서 MobileCropNet은 top-1 MOS `3.680133`, regret `0.772467`이고 teacher는 top-1 MOS `3.673067`, regret `0.779533`이다. 즉 teacher의 전체 ranking correlation은 더 좋지만, top-1 선택의 평균 MOS는 이 subset에서 MobileCropNet이 근소하게 높다.
- 이 결과는 teacher가 ground truth가 아니라 학습 신호라는 점을 뒷받침한다. 이후 최종 모델 선택은 teacher imitation 성능이 아니라 official GAIC MOS, 외부 public benchmark, 정성 시각화, latency를 함께 봐야 한다.
- 현재 teacher exact 산출물은 150장 subset으로 제한된다. 500장 전체에서 teacher를 같은 방식으로 비교하려면 누락 350장에 대한 candidate/features를 생성하고 GAIC v2 test annotation crop 43,123개 전체에 대해 teacher scorer를 다시 실행해 `candidate_eval_rows.jsonl`을 확장해야 한다.

## 8. Replay-label 정량 평가 결과

GAIC test split 625개 기준 결과는 다음과 같다. 표는 test top-1 hit 우선으로 정렬했다.

| rank | run | test top-1 hit | exact best | NDCG@5 | SRCC | top1 IoU to best positive | proposal recall@5 IoU0.5 | utility regret |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `mcn-v4-q24-288-w075-20260416-095704` | 0.353600 | 0.379200 | 0.896934 | 0.614445 | 0.760730 | 0.867200 | 0.052007 |
| 2 | `mcn-v4-nb-q16-224-w075-20260416-095704` | 0.348800 | 0.347200 | 0.895313 | 0.613565 | 0.750238 | 0.860800 | 0.053632 |
| 3 | `mcn-v4-q16-256-w075-20260416-095704` | 0.347200 | 0.360000 | 0.894264 | 0.617598 | 0.749993 | 0.867200 | 0.056248 |
| 4 | `mcn-v4-q16-256-w100-20260416-095704` | 0.340800 | 0.352000 | 0.894726 | 0.610150 | 0.742225 | 0.865600 | 0.057197 |

`q24-288-w075`는 입력 해상도와 proposal query를 함께 늘린 설정이다. test top-1 hit, NDCG@5, top1 IoU, utility regret에서 가장 우수하다. `q16-256-w075`는 val top-1 hit가 가장 높았지만 test에서는 `q24-288-w075`보다 낮았다. 이는 현 split 규모에서 validation variance가 존재함을 의미하므로, 최종 모델 선택에는 test 및 외부 benchmark를 함께 봐야 한다.

## 9. Baseline 및 oracle 비교

best run `mcn-v4-q24-288-w075-20260416-095704` 기준 method comparison은 다음과 같다.

| method | top-1 hit | exact best | chosen label score | top1 IoU to best positive | utility regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v4_model` | 0.353600 | 0.379200 | 0.706283 | 0.760730 | 0.052007 |
| `baseline_candidate` | 0.000000 | 0.052800 | 0.274472 | 0.794247 | 0.483818 |
| `teacher_score_oracle` | 0.731200 | 1.000000 | 0.758290 | 0.906623 | 0.000000 |
| `positive_oracle` | 0.742400 | 0.988800 | 0.757911 | 0.908918 | 0.000379 |
| `proposal_top1` | 0.000000 | 0.000000 | - | 0.656456 | 0.000000 |

해석은 다음과 같다.

- v4 모델은 baseline candidate 대비 teacher label score를 `0.274472`에서 `0.706283`으로 끌어올렸다.
- utility regret은 baseline 대비 약 `89.25%` 감소했다.
- teacher score oracle은 후보 replay set 내부의 상한이다. v4 모델은 oracle score `0.758290`에 근접하지만, exact best/top-1 hit는 아직 낮다.
- `proposal_top1` 단독 IoU는 `0.656456`으로 ranking을 거친 v4 output보다 낮다. 현재 배포 후보는 proposal 단독이 아니라 candidate replay/ranking 경로다.
- baseline candidate의 IoU가 v4보다 높게 나오는 것은 baseline이 큰 박스 또는 full/max-area crop인 경우가 많기 때문이다. 본 과제에서는 IoU만으로 crop 품질을 판단하면 teacher score와 불일치할 수 있어, utility regret, top-1 hit, NDCG, SRCC를 함께 봐야 한다.

## 10. GPU 사용률 분석

학습 중 `nvidia-smi` sampling과 CUDA peak memory를 기록했다.

| run | CUDA peak allocated MB | CUDA peak reserved MB | GPU util mean % | GPU util max % | memory used max MB | power max W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mcn-v4-q24-288-w075-20260416-095704` | 178.510 | 260.000 | 14.865 | 25.000 | 793.000 | 84.380 |
| `mcn-v4-nb-q16-224-w075-20260416-095704` | 159.165 | 240.000 | 13.214 | 21.000 | 771.000 | 81.780 |
| `mcn-v4-q16-256-w075-20260416-095704` | 186.045 | 292.000 | 11.250 | 30.000 | 823.000 | 79.650 |
| `mcn-v4-q16-256-w100-20260416-095704` | 227.021 | 348.000 | 12.643 | 23.000 | 879.000 | 89.660 |

판단:

- 현재 모델/데이터 크기에서는 GPU가 연산 병목이 아니다.
- 평균 GPU 사용률이 15% 미만이고 VRAM 사용량이 1GB 미만이므로 2개 이상 GPU를 쓰는 분산 학습은 필요하지 않다.
- 성능 최적화 우선순위는 multi-GPU가 아니라 data loader, JPEG decode, CPU transform, batch size 상향, crop/mask pooling vectorization, candidate tensor pre-cache다.
- A100 한 장에서도 모델이 너무 작으므로, 제품용 latency 검증은 A100 처리량보다 mobile CPU/GPU/NPU export 경로를 별도로 측정해야 한다.

## 11. 정성 시각화 산출물

네 실험 모두 PNG contact sheet와 overlay PNG를 생성했다. 로컬 검증에서 contact sheet는 모두 PNG이며 크기는 `(1680, 3840)`이다.

| run | contact sheet |
| --- | --- |
| `mcn-v4-q24-288-w075-20260416-095704` | `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-q24-288-w075-20260416-095704/viz_test/contact_sheet.png` |
| `mcn-v4-nb-q16-224-w075-20260416-095704` | `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-nb-q16-224-w075-20260416-095704/viz_test/contact_sheet.png` |
| `mcn-v4-q16-256-w075-20260416-095704` | `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-q16-256-w075-20260416-095704/viz_test/contact_sheet.png` |
| `mcn-v4-q16-256-w100-20260416-095704` | `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-q16-256-w100-20260416-095704/viz_test/contact_sheet.png` |

시각화 색상 의미:

- green: label best positive
- blue: baseline candidate
- orange: learned proposal top-1
- red: MobileCropNet v4 selected crop

## 12. 최종 산출물 위치

통합 요약:

- `artifacts/mobilecropnet_v4/gpu_runs/summary_20260416_095704/EXPERIMENT_SUMMARY.md`
- `artifacts/mobilecropnet_v4/gpu_runs/summary_20260416_095704/experiment_summary.json`

best run:

- checkpoint: `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-q24-288-w075-20260416-095704/best.pt`
- test metrics: `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-q24-288-w075-20260416-095704/eval_test/metrics.json`
- predictions: `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-q24-288-w075-20260416-095704/eval_test/predictions.jsonl`
- comparison: `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-q24-288-w075-20260416-095704/compare_test/comparison_metrics.json`
- qualitative PNG: `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-q24-288-w075-20260416-095704/viz_test/contact_sheet.png`
- per-run report: `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-q24-288-w075-20260416-095704/report/MobileCropNet_v4_GAIC_Report_KO.md`

## 13. 제품/논문 수준 보완 과제

현재 결과는 학습/평가 파이프라인이 end-to-end로 동작하고 baseline 대비 큰 개선을 확인한 단계다. 제품/논문 수준으로 끌어올리려면 다음 순서가 필요하다.

1. Hyperparameter search 확대
   - `input_size`: 288, 320, 352
   - `proposal_q`: 24, 32, 48
   - `candidate_k`: 24, 32
   - `width_mult`: 0.5, 0.75, 1.0
   - `loss weights`: listwise, pairwise, risk, proposal loss 계수 sweep

2. Data pipeline 최적화
   - image decode/cache 도입
   - candidate tensor와 mask feature pre-cache
   - `num_workers`, `persistent_workers`, `prefetch_factor`, batch size sweep
   - 현재 GPU 사용률이 낮기 때문에 이 영역이 가장 먼저 개선되어야 한다.

3. Ranking 개선
   - teacher score regression calibration
   - target aspect ratio별 ranking head 또는 FiLM conditioning 강화
   - hard negative mining
   - top-K distillation objective 추가

4. Proposal head 개선
   - proposal recall@5 IoU0.5를 0.90 이상으로 안정화
   - proposal diversity loss 재조정
   - proposal-to-candidate matching의 differentiable surrogate 검토

5. Public cropper 비교
   - GAIC v2 test split 500장 전체에 대해 public cropping baselines, 기존 Teacher 모델, MobileCropNet v3.x, v4.0을 동일 official MOS metric schema로 비교
   - 현재 teacher exact comparison은 150장 subset이므로, 누락 350장의 candidate/features 생성 후 teacher scorer를 GAIC v2 test annotation crop 43,123개 전체에 대해 재실행하여 500장 전체 비교표를 완성
   - FCDB/CPC/GNMC 같은 public benchmark로 외부 일반화 확인

6. Deployment 검증
   - TorchScript/ONNX export
   - FP16/INT8 quantization
   - mobile target latency, peak memory, model size 측정
   - on-device crop stability와 aspect-ratio policy 검증

## 14. 결론

현재 best configuration은 `input_size=288`, `proposal_q=24`, `width_mult=0.75`, `candidate_k=24`다. 이 모델은 replay-label test 기준 baseline candidate보다 teacher label score와 utility regret에서 큰 폭으로 개선되었고, GAIC v2 benchmark 500장 전체에서도 PCC `0.399840`, SRCC `0.368395`, Acc1/10 `0.630000`, Accw4/10 `0.365610`, top-1 MOS `3.804520`을 기록했다. PNG 정성 산출물과 per-run report도 생성되었다.

다만 official MOS 기준 ranking correlation과 best-return 지표는 아직 논문 수준의 강한 성능이라고 보기 어렵다. 다음 연구 초점은 단순 GPU 확장이 아니라 official MOS 기반 평가 루프 고정, teacher 500장 exact scoring 확장, ranking supervision 강화, proposal recall 안정화, data loader/cache 최적화, public benchmark 비교 확장에 두는 것이 합리적이다. 현재 실험 규모에서는 2개 이상 GPU 사용은 필요하지 않으며, 병렬 실험은 단일 GPU workload 여러 개를 생성하는 방식이 가장 효율적이다.
