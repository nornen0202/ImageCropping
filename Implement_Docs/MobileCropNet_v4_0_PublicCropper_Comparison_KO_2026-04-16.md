# MobileCropNet v4.0 Public Cropper 비교 평가 보고서

## 평가 목적

MobileCropNet v4.0의 GAIC benchmark 성능을 기존 데이터 큐레이션 파이프라인에서 사용했던 공개 cropping 모델인 CACNet, CGS, GAIC와 같은 기준에서 비교한다. 평가는 SSTK teacher label이 아니라 GAIC v2로 재생성한 official benchmark annotation을 기준으로 수행했다.

## 평가 데이터

- Annotation: `data/Publics/GAIC_v2/annotations_json/instances_test.json`
- Test image 수: 500
- Official candidate crop 수: 43,123
- 이미지당 평균 candidate 수: 86.246
- 기준 지표: ranking correlation, best return, rank-weighted best return 계열

## Public Cropper 실행 가능성

| 모델 | 코드 경로 | weight 경로 | 판정 | 비고 |
| --- | --- | --- | --- | --- |
| CACNet | `third_party/public_cropping_teachers/cacnet` | `weights/public_cropping_teachers/cacnet/best-FLMS_iou.pth` | 가능 | 단일 crop 회귀 모델이므로 GAIC candidate-ranking 지표는 원 방식으로 계산할 수 없다. IoU 투영값은 별도 진단값으로만 기록했다. |
| CGS | `third_party/public_cropping_teachers/cgs` | `weights/public_cropping_teachers/cgs/pretrained_model/pretrained_model/*.pth` | 가능 | Py3.10용 RoI/RoD Align extension이 포함되어 있어 CUDA forward가 통과했다. |
| GAIC | `third_party/public_cropping_teachers/gaic` | `weights/public_cropping_teachers/gaic/mobilenet_0.682_..._0.874.pth` | 가능 | 원본 GAIC checkout의 roi_align/rod_align은 Python 2/3.5 pyc 기반이라 직접 import가 실패한다. 동일 API를 갖는 CGS의 Py3.10 빌드 extension을 재사용해 평가했다. |

## 구현 산출물

- 평가 스크립트: `src/scripts/evaluate_public_croppers_gaic_v2.py`
- 전체 평가 산출물: `artifacts/mobilecropnet_v4/public_cropper_eval/gaic_v2_test500`
- Smoke 평가 산출물: `artifacts/mobilecropnet_v4/public_cropper_eval/gaic_v2_test500_smoke`
- Public cropper 리포트: `artifacts/mobilecropnet_v4/public_cropper_eval/gaic_v2_test500/PUBLIC_CROPPER_GAIC_V2_EVAL_REPORT.md`

추가로 Product AR/full label 쪽에는 아래 구현과 산출물이 이미 존재한다.

- Product AR 후보에 public cropper score를 부여하는 스크립트: `src/scripts/score_mobilecropnet_v4_product_ar_candidates_with_public_cropper.py`
- public score Product AR label 생성 스크립트: `src/scripts/run_mobilecropnet_v4_product_ar_public_label_generation.sh`
- Product AR test score 요약: `artifacts/mobilecropnet_v4/product_ar_score_labels_260420/public_direct/scores/test_public_product_ar_scores_summary.json`
- public score Product AR 학습 runs: `artifacts/mobilecropnet_v4/product_ar_score_labels_260420/public_direct/runs/*`

이는 "public cropper를 Product AR candidate set 위에서 scorer teacher로 사용하는 경로"까지는 이미 구현되어 있음을 뜻한다.

## Direct Product AR 비교 가능 범위

Product AR direct 평가는 MobileCropNet의 실제 배포 표면에 가까운 평가다. 따라서 public cropper와의 비교는 완전히 같은 의미가 아니다.

| 비교 항목 | GAIC / CGS | CACNet | 비고 |
| --- | --- | --- | --- |
| Product AR candidate-bank top-1 비교 | 가능 | 제한적 | `GAIC`, `CGS`는 candidate scorer라서 같은 candidate pool 위에 점수를 붙일 수 있다. |
| `final_positive_hit_iou_0_5`, `final_best_iou_to_positive`, `final_risk_hit_iou_0_5`, `final_target_ar_compatible` | 가능 | projection-only | candidate-bank top-1 기준으로 계산 가능하다. |
| `proposal_positive_recall_at_k` | 불가 | 불가 | public cropper는 learned proposal generator가 없다. |
| `route_acc`, `decision_acc` | 불가 | 불가 | public cropper는 route/decision/policy head가 없다. |
| explanation head 비교 | 불가 | 불가 | subject mode/checklist/why tag 출력 계약이 다르다. |

즉 `GAIC`, `CGS`는 Product AR/full candidate bank 위에서 **scorer baseline**으로는 비교 가능하지만, MobileCropNet처럼 `proposal generator + ranker + policy`를 함께 보는 **deployed system full comparison**은 아니다. `CACNet`은 단일 crop 회귀형이라 native candidate score가 없으므로 projection 기반 근사 비교만 할 수 있다.

현재 기준으로는 다음이 이미 완료돼 있다.

1. GAIC official benchmark에서의 public cropper 비교
2. unified public benchmark에서의 public cropper 비교
3. Product AR candidate set에 public cropper score를 부여하고 학습 label로 사용하는 경로

반면 다음은 아직 별도 전용 실험/리포트가 없다.

1. Product AR/full label 위에서 `GAIC`, `CGS`가 직접 top-1 candidate를 고르게 하고
2. MobileCropNet의 `evaluate_mobilecropnet_v4_product_ar_direct.py`와 같은 형식의
3. dedicated direct Product AR comparison report를 만드는 작업

따라서 현재 문서의 정량 비교 표는 **GAIC candidate-ranking 표면**과 **unified public benchmark 표면**을 기준으로 읽어야 하며, Product AR direct 기준 apples-to-apples candidate-bank baseline 비교는 추가 구현 항목으로 남아 있다.

## 지표 해석

- `PCC`, `SRCC`: 이미지별 official GAIC MOS와 모델 예측 score의 상관계수 평균이다. 높을수록 official MOS 순위를 잘 따른다.
- `AccK/N`: 모델이 반환한 상위 K개 crop 중 official MOS 상위 N개에 포함되는 비율이다. 높을수록 좋다.
- `AccwK/N`: `AccK/N`에 official rank와 반환 순서 차이를 반영한 rank-weighted 지표다. 높을수록 좋다.
- `top1 MOS`: 모델 top-1 crop의 official MOS 평균이다. 높을수록 좋다.
- `MOS regret`: 이미지 내 최고 official MOS와 모델 top-1 MOS의 차이다. 낮을수록 좋다.
- `top1 rank pct`: 모델 top-1 crop의 official MOS rank percentile이다. 1.0에 가까울수록 좋다.
- `-`: 모델 출력 형식상 해당 GAIC candidate-ranking 지표를 원 방식으로 계산할 수 없음을 뜻한다.

## GAIC v2 Test 500 비교

| 모델 | 평가 이미지 | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/5 | Accw4/5 | top1 MOS | top1 rank pct | MOS regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileCropNet v4 q24 288 w0.75 | 500 | 0.399840 | 0.368395 | 0.434000 | 0.630000 | 0.298000 | 0.244742 | 3.804520 | 0.889153 | 0.427380 |
| CACNet | 500 | - | - | - | - | - | - | - | - | - |
| CGS | 500 | 0.838836 | 0.799759 | 0.550000 | 0.728000 | 0.462000 | 0.401560 | 3.921020 | 0.898081 | 0.310880 |
| GAIC public model | 500 | 0.872629 | 0.845874 | 0.604000 | 0.804000 | 0.494000 | 0.432030 | 3.985300 | 0.930537 | 0.246600 |
| GAIC MOS oracle | 500 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 4.231900 | 1.000000 | 0.000000 |

## 2026-04-20 Public Score Distillation Update

추가 실험에서는 GAIC public cropper score를 MobileCropNet v4 학습 score teacher로 사용하고, SSTK data factory 결과는 explanation/fatal/safety tier metadata로 결합했다. 이 track은 GAIC public model 자체를 비교 평가하는 위 표와 다르다. 목표는 public model score를 그대로 배포하는 것이 아니라 MobileCropNet student가 해당 candidate-ranking preference를 얼마나 흡수할 수 있는지 확인하는 것이다.

핵심 결과는 `candidate_k=96` 학습이 필요하다는 점이다. 기존 `rank_320`은 학습 시 32개 후보만 보므로 official benchmark의 평균 86개 candidate 전체 scoring에서 top-return이 낮았다. `mcn-pubscore-r320k96-tr06-s18-20260420-142309`는 학습 candidate coverage를 96개로 늘려 이 문제를 완화했다.

| 모델 | 평가 이미지 | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/10 | Accw4/10 | top1 MOS | MOS regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GAIC public model score teacher | 500 | 0.872628 | 0.845896 | 0.606000 | 0.804000 | 0.725500 | 0.605688 | 3.985460 | 0.246440 |
| MobileCropNet public-score `r320k96` student | 500 | 0.697523 | 0.675074 | 0.486000 | 0.666000 | 0.561500 | 0.454218 | 3.846940 | 0.384960 |
| MobileCropNet T6 diagnostic `rank_320` student | 500 | 0.616310 | 0.589892 | 0.456000 | 0.656000 | 0.538500 | 0.430915 | 3.830220 | 0.401680 |
| MobileCropNet SSTK-only `hybrid384` student | 500 | 0.447673 | 0.427922 | 0.444000 | 0.650000 | 0.464500 | 0.360553 | 3.822800 | 0.409100 |

따라서 public score distillation student는 아직 GAIC public model 자체에는 못 미치지만, 기존 T6 diagnostic student와 SSTK-only student보다 모든 primary metric에서 높다. 단 이 결과는 GAIC-trained public model을 teacher로 사용한 것이므로 `SSTK-only Product Track` 결론과 분리해서 해석해야 한다. 상세 설계, label tier, 시각화 산출물은 `Implement_Docs/MobileCropNet_v4_0_GAIC_Experiment_Report_KO_2026-04-16.md`의 7.9절에 기록했다.

## CACNet IoU Projection 진단값

아래 값은 CACNet의 예측 crop과 official candidate 간 IoU로 가장 가까운 candidate를 찾는 방식의 참고 진단값이다. 이는 원 GAIC candidate-ranking 평가가 아니므로 위 primary comparison table에는 포함하지 않았다.

| 평가 이미지 | projected top1 MOS | projected MOS regret | projected top1 rank pct | candidate IoU max | candidate IoU mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 3.734740 | 0.497160 | 0.833031 | 0.896787 | 0.645612 |

## Teacher Overlap 150 비교

SSTK teacher의 exact official candidate score는 현재 기존 candidate/features 산출물과 겹치는 150장에 대해서만 존재한다. 따라서 teacher와 MobileCropNet의 비교는 500장 public cropper 비교와 분리해서 해석해야 한다.

| 모델 | 평가 이미지 | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/5 | Accw4/5 | top1 MOS | top1 rank pct | MOS regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileCropNet v4 q24 288 w0.75, teacher-overlap subset | 150 | 0.352746 | 0.322241 | 0.240000 | 0.406667 | 0.193333 | 0.150113 | 3.680133 | 0.821453 | 0.772467 |
| SSTK teacher subjectfix v2 equalized official Gc | 150 | 0.498440 | 0.502193 | 0.286667 | 0.460000 | 0.248333 | 0.195126 | 3.673067 | 0.771434 | 0.779533 |

## 해석

GAIC public model은 동일 GAIC benchmark 계열의 후보 crop scoring 모델이므로 GAIC v2 test 500에서 가장 높은 성능을 보인다. CGS도 candidate ranker 구조라 official candidate 평가에 직접 대응되며 MobileCropNet v4보다 높은 ranking correlation과 best return 지표를 달성했다.

MobileCropNet v4는 현재 SSTK data factory label 기반으로 학습되었고, official GAIC MOS를 직접 최적화하지 않았다. GAIC/CGS public candidate ranker와 비교하면 ranking correlation, best-return, rank-weighted return 지표에서 큰 격차가 있다. 이는 다음 최적화 단계에서 official GAIC train/val split을 이용한 supervised fine-tuning 또는 distillation objective를 추가해야 함을 시사한다.

CACNet은 단일 crop 회귀 모델이라 candidate별 native score가 없다. 따라서 `PCC`, `SRCC`, `AccK/N`, `AccwK/N`, `top1 MOS`, `MOS regret`, `top1 rank pct`를 GAIC candidate-ranking 원 방식으로 계산할 수 없으며 primary table에서는 모두 `-`로 표시했다. IoU projection 진단값은 CACNet이 낸 crop이 official candidate 집합과 얼마나 가까운지만 확인하기 위한 참고값이다.

## 검증

- CACNet smoke: weight load 및 224 입력 forward 통과
- CGS smoke: extractor, GNN weight load 및 CUDA forward 통과
- GAIC smoke: 원본 stale extension import 실패 확인 후 CGS Py3.10 extension 재사용 방식으로 CUDA forward 통과
- Full evaluation: GAIC v2 test 500장, 43,123 official candidate 전체 평가 완료
- Local GPU sample usage: RTX 3050 OEM, util 약 66-83%, memory 약 4.9-5.1GB
- Script syntax check: `py_compile` 통과
- GAIC benchmark unit test: `tests/test_mobilecropnet_v4_gaic_benchmark.py` 5개 통과
