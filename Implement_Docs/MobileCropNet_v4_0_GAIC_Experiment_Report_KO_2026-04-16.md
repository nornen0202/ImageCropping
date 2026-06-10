# MobileCropNet v4.0 GAIC 실험 결과 보고서

작성일: 2026-04-16

최종 업데이트: 2026-04-20

## 1. 목적

본 문서는 `MobileCropNet_v4_0.md` 설계를 기준으로 구현한 MobileCropNet v4.0의 GAIC replay-label 학습, 추론, 공식 GAIC benchmark 정량 평가, 정성 시각화, GPU 사용률 측정 결과를 정리한다. 초기 실험 대상 학습 라벨은 현재 데이터 팩토리 파이프라인에서 생성된 `data/GAIC/All/artifacts/training_labels/gaic_personv6_server_v1_leftover_ignore_monotonic` 계열 라벨을 MobileCropNet v4 split으로 변환한 결과이며, 2026-04-17 추가 실험에서는 GAIC-v2 official split 전체에 맞춰 새로 생성된 baseline SSTK label 완료본을 사용했다.

평가 기준은 이번 수정부터 명확히 분리한다. 제품/논문 수준의 1차 성능 판단은 SSTK teacher score가 아니라 `data/Publics/GAIC_v2/annotations_json/instances_test.json`의 GAIC v2 논문 split 500장 MOS annotation을 ground truth로 사용한다. teacher-label 기반 지표는 학습 label replay와 teacher 모사 정도를 보는 보조 진단으로만 해석한다.

2026-04-17 추가 실험에서는 `SSTK_Teacher_GAIC_v2_Official_Benchmark_Improvement_Report_KO_2026-04-16.md`의 Train2636 완료 상태와 원격 공유 스토리지 산출물을 확인한 뒤, GAIC-v2 official split 전체(train 2,636장, val 200장, test 500장)에 대해 생성된 baseline SSTK training label을 직접 사용해 세 profile을 다시 학습했다. 새 라벨 경로는 `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/{Train2636,Val200,Test500}/artifacts/training_labels/*_largecap_v2_260416_leftover_ignore_monotonic`이다.

2026-04-17 추가 보완 실험부터는 목표를 `SSTK-only Product Track`으로 재정의했다. GAIC train/val/test MOS annotation은 학습 loss, pseudo-label 생성, hard mining, score calibration, checkpoint 선택, hyperparameter 선택에 사용하지 않는다. GAIC MOS는 제품 후보를 SSTK-only validation으로 선정한 뒤 사후 benchmark 평가와 보고에만 사용한다. 목표도 public GAIC 계열 모델을 모든 지표에서 이기는 것이 아니라, 상용 가능한 SSTK data factory label만으로 SSTK teacher 성능에 근접하거나 top-return 지표에서 teacher를 넘는 product candidate를 만드는 것으로 정리한다.

2026-04-20 재정리 이후 MobileCropNet의 공식 목표는 `Product AR cropper + explanation`으로만 둔다. `GAIC public cropper score` 또는 GAIC MOS-trained scorer를 쓰는 track은 MobileCropNet student를 키우는 연구 목표가 아니라, 학습 데이터 큐레이션과 teacher/scorer 성능을 높이기 위한 별도 diagnostic/curation track으로 분리한다. 따라서 `pubscore`/`pubdistill` 계열 결과는 GAIC benchmark scorer teacher의 품질과 label 생성 전략을 분석하는 참고 결과이며, 제품 AR cropper 후보나 AR별 crop 품질 근거로 사용하지 않는다.

핵심 결론은 다음과 같다.

- `SSTK-only Product Track` 재실험 이후 `rank_320` corrected balanced seed/loss/LR-WD/warmup ablation, 352/384 해상도 확장, q24/turbo balanced-selection 재평가까지 완료했다. 모든 제품 후보 선택은 GAIC MOS 없이 SSTK validation의 top-return, listwise/ranking quality, regret, calibration, risk suppression, proposal recall을 결합한 `sstk_balanced_topreturn`으로 수행했다.
- 2026-04-20에는 GAIC public cropper score를 crop score teacher로 쓰고, SSTK data factory는 subject mode, fatal/safety, explanation tier metadata로 유지하는 별도 diagnostic/distillation track을 추가했다. 이 track은 `SSTK-only Product Track`도 `Product AR cropper`도 아니며, 이제부터는 GAIC benchmark scorer teacher와 학습 라벨 큐레이션 전략을 분석하는 별도 track으로만 유지한다. 참고로 과거 student distillation 결과 중 single-checkpoint best는 `mcn-pubdistill-r320k128-zb035-s20-20260420-172325`이며 GAIC v2 official test 500장에서 PCC `0.733951`, SRCC `0.726926`, Acc1/5 `0.506000`, Acc1/10 `0.710000`, Acc4/10 `0.605000`, Accw4/10 `0.489795`, top-1 MOS `3.897880`, MOS regret `0.334020`을 기록했다. 이 수치는 MobileCropNet 제품 후보 성능이 아니라 public-score teacher를 압축했을 때의 진단값이다.
- 이번 추가 sweep의 top-return best는 `mcn-next-hybrid384-tr06-s17-20260417-191315`와 `mcn-next-r320-tr070-s17-20260417-191315`이다. 둘 다 GAIC v2 official test 500장에서 Acc1/5 `0.444000`, Acc1/10 `0.650000`, top-1 MOS `3.822800`, MOS regret `0.409100`을 기록했다. 다만 `hybrid384`는 PCC/SRCC `0.447673`/`0.427922`로 더 안정적이고, `tr070`은 PCC/SRCC `0.370447`/`0.343425`로 ranking correlation이 낮아 top-return 전용 후보로만 해석한다.
- 현재 균형형 product candidate는 `mcn-next-r320-tr06-s19-20260417-191315`이다. 이 run은 `rank_320`, `top_return_weight=0.6`, seed `20260419`로 학습했고 PCC `0.435216`, SRCC `0.405202`, Acc1/5 `0.444000`, Acc1/10 `0.646000`, Acc4/10 `0.491000`, Accw4/10 `0.383345`, top-1 MOS `3.821640`, MOS regret `0.410260`을 기록했다. 이전 `rank320-bal-s18`보다 top-return은 높고, Acc4/10은 같은 계열 내 최상위다.
- 이전 1차 product candidate였던 `mcn-prod-rank320-bal-s18-20260417-170831`과 `mcn-prod-rank320-bal-tr06-20260417-170831`은 기준선으로 유지한다. 새 sweep은 top-1 MOS/regret을 `3.818520/0.413380`에서 `3.822800/0.409100`까지 끌어올렸지만, PCC/SRCC는 아직 full-option SSTK teacher보다 낮다.
- 같은 official test에서 full-option SSTK teacher official-Gc는 PCC `0.514590`, SRCC `0.497884`, Acc1/5 `0.270000`, Acc1/10 `0.448000`, top-1 MOS `3.598460`, MOS regret `0.633440`이다. 따라서 새 `rank_320` 계열 student는 teacher보다 ranking correlation은 낮거나 근접하지만, 제품 사용자가 체감하는 Acc1/N, top-1 MOS, MOS regret은 teacher보다 크게 좋다.
- Top-return surrogate를 무조건 강하게 걸면 일반화가 깨질 수 있음도 다시 확인했다. q24/turbo를 corrected balanced selection으로 재평가했지만 q24는 Acc1/5 `0.302000`, top-1 MOS `3.676320`, turbo는 Acc1/5 `0.316000`, top-1 MOS `3.624940`에 그쳤다. 따라서 제품 후보는 `rank_320`/384 계열로 유지하고, 다음 단계의 핵심은 GAIC MOS를 선택에 쓰는 것이 아니라 SSTK-only selection composite 안에서 top-return, ranking correlation, calibration, destructive-negative suppression, proposal coverage를 더 정교하게 결합하는 것이다.
- 기존 네 실험 중 `mcn-v4-q24-288-w075-20260416-095704`가 replay-label GAIC test 기준 최상위이며, 공식 GAIC benchmark 평가도 이 checkpoint로 수행했다.
- GAIC v2 test 500장, annotation crop 43,123개 전체 평가에서 MobileCropNet v4 best run은 PCC `0.399840`, SRCC `0.368395`, Acc1/5 `0.434000`, Acc1/10 `0.630000`, Acc4/10 `0.469000`, Accw4/10 `0.365610`, top-1 MOS `3.804520`, MOS regret `0.427380`을 기록했다.
- 설계 문서의 권장 프로파일인 HQ-320(`MobileNetV4-Hybrid-M`, input 320, candidate_k 32, proposal_q 32, token_dim 192, set-transformer ranker depth 2)을 구현하고 원격 A100에서 full training/evaluation/PNG explanation visualization까지 완료했다. 산출물은 `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509`에 저장했다.
- HQ-320 run의 공식 GAIC MOS test 500장 결과는 PCC `0.322124`, SRCC `0.313649`, Acc1/5 `0.254000`, Acc1/10 `0.372000`, Acc4/10 `0.328500`, Accw4/10 `0.248184`, top-1 MOS `3.531740`, MOS regret `0.700160`이다. 따라서 HQ-320 프로파일은 실행 가능성과 explanation 출력은 검증됐지만, 현재 SSTK-label 학습 신호/selection 기준에서는 기존 Balanced-288 best를 대체할 제품/논문 성능 개선으로 보지 않는다.
- explanation 추가가 성능 저하 원인인지 확인하기 위해 Balanced-288도 동일한 explanation 출력 경로로 재학습/평가했다. `mcn-v4-balanced288-explain-20260416-180111`은 official MOS test 500장에서 PCC `0.413289`, SRCC `0.384878`, Acc1/5 `0.436000`, Acc1/10 `0.608000`, Acc4/10 `0.469500`, Accw4/10 `0.364822`, top-1 MOS `3.779800`, MOS regret `0.452100`을 기록했다. 이는 기존 Balanced-288 best와 거의 같은 수준이므로 explanation label/PNG 후처리 자체가 HQ-320 저하의 주원인이라는 가설은 기각한다.
- 이후 얕은 `composition: ok 0.63` 수준의 macro-only explanation을 보완하기 위해 SSTK data factory의 `checklist_labels`, `checklist_scores`, `why_tags`, `reject_tags`, `composition_focus`를 학습 target으로 직접 연결했다. 모델에는 13개 세부 checklist class head, 22개 detail score regression head, 31개 why/reject tag multi-label head를 추가했고, prediction JSONL/PNG에는 `headroom_loose`, `lookroom_excessive`, `rule_of_thirds_strong` 같은 label과 세부 score/tag가 함께 출력된다.
- 새 detailed explanation full run은 updated `AGENTS.md`의 MLP GPU workload 방식으로 세 profile을 병렬 실행했다. `hq_320` run `1094298`, `balanced_288` run `1094299`, `turbo_256` run `1094300` 모두 단일 A100 workload에서 성공했다. Official MOS 기준 top-return은 `turbo_256`이 세 profile 중 가장 좋고, PCC/SRCC ranking correlation은 `balanced_288`이 가장 좋다.
- GAIC-v2 official split baseline label 완료본으로 다시 학습한 2026-04-17 run에서는 세 profile 모두 `backbone_pretrained=true`로 실행했다. 새 run 중 official MOS 기준 top-return은 Turbo-256이 가장 좋다. Turbo-256은 SRCC `0.434566`, PCC `0.446322`, Acc1/5 `0.348000`, Acc1/10 `0.538000`, Acc4/10 `0.418000`, Accw4/10 `0.314088`, top-1 MOS `3.718800`, MOS regret `0.513100`을 기록했다. Balanced-288은 replay/validation top-1이 가장 높고, HQ-320은 이번 label에서도 official MOS top-return이 가장 낮다.
- 이후 fixed-AR proposal을 letterbox content rect와 transform rounding까지 보정하는 AR-constrained parameterization으로 수정하고, object/person mode-aware explanation applicability head/mask를 추가한 `arx2` 재실험을 수행했다. 세 profile 모두 spot MLP single-GPU workload(`--allow-spot`)에서 성공했고, replay test 기준 `proposal_raw_top1_target_ar_compatible=1.0`, `proposal_top1_target_ar_compatible=1.0`, `explain_mode_consistency_violation_rate=0.0`, `explain_person_tag_on_object_rate=0.0`을 기록했다.
- `arx2` 재실험의 official MOS 기준 best는 Turbo-256이다. Turbo-256은 PCC `0.456028`, SRCC `0.446876`, Acc1/5 `0.356000`, Acc1/10 `0.542000`, top-1 MOS `3.737200`, MOS regret `0.494700`을 기록해 이전 GAIC-v2 baseline label run보다 개선됐다. 그래도 기존 제품 후보 best의 top-1 MOS `3.804520`, MOS regret `0.427380`에는 아직 도달하지 못했다.
- 같은 official MOS test 500장에서 full-option SSTK teacher official-Gc는 SRCC `0.497884`, PCC `0.514590`, top-1 MOS `3.598460`, MOS regret `0.633440`이다. 즉 teacher는 ranking correlation이 여전히 가장 높지만, Turbo-256 student는 top-1 MOS와 MOS regret에서 teacher보다 낫다.
- 이번에 GAIC v2 test 500장 전체에 대해 C1, C2, C3, C5, C6 gaze, C7 saliency, BLIP caption, real-expensive align/aesthetic scoring을 포함한 full-option SSTK teacher official-Gc benchmark를 재생성했다. C4 OCR는 현재 미구현 상태라 제외했다. 43,123개 official annotation crop 전체에서 full-option teacher는 PCC/SRCC가 MobileCropNet v4보다 높지만, MobileCropNet v4는 Acc 계열과 top-1 MOS/MOS regret이 더 낫다. 따라서 teacher score는 학습 신호로 유용하지만 전문가 MOS ground truth를 대체할 수 없다.
- best run의 replay-label test top-1 hit는 `0.353600`, NDCG@5는 `0.896934`, SRCC는 `0.614445`, proposal recall@5 IoU 0.5는 `0.867200`이다.
- baseline candidate 대비 utility regret이 `0.483818`에서 `0.052007`로 크게 감소했다.
- teacher score oracle과는 아직 top-1 hit 기준 약 `0.3776p` 차이가 있어 ranking head와 proposal-to-candidate coupling 개선 여지가 크다.
- 경량 Balanced/Turbo 계열 GPU 사용률은 평균 `11.25%`부터 `14.86%`, 최대 `30%` 수준이고 VRAM 사용량은 1GB 미만이었다. HQ-320은 train 기준 평균 GPU utilization `48.18%`, 최대 `81%`, `nvidia-smi` 메모리 최대 `7,986MB`, CUDA peak allocated `1,379MB`를 기록했다. HQ-320도 단일 A100에서 여유가 크므로 현재 규모에서는 2개 이상 GPU가 필요하지 않다.

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
- explanation 출력: `decision_label`, `subject_mode_label`, 후보별 `model_explanation`, `model_checklist`, `model_detailed_checklist`, `model_detail_scores`, `model_why_tags`를 prediction JSONL과 PNG side panel에 기록한다.
- detailed explanation 시각화 보정: top crop이 baseline처럼 teacher checklist가 없는 후보일 때는 같은 image/AR 내 가장 가까운 labeled candidate의 detail을 `nearest labeled candidate ... IoU to top=...` 출처와 함께 표시한다. 모델이 `na` class를 예측한 경우에는 품질 라벨처럼 보이지 않도록 `unlabeled p=...`로 렌더링한다.
- 리포트 생성 코드: `src/scripts/build_mobilecropnet_v4_report.py`
- 실험 요약 코드: `src/scripts/summarize_mobilecropnet_v4_experiments.py`
- 단위 테스트: `tests/test_mobilecropnet_v4.py`, `tests/test_gaic_benchmark_eval.py`

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
| `mcn-v4-hq320-explain-20260416-173509` | 1094101 | same-account interactive GPU via CPU server SSH | 10.8.103.214 | Completed task; run still reusable | 2026-04-16T08:35:10Z | 2026-04-16T08:44:19Z |
| `mcn-v4-balanced288-explain-20260416-180111` | 1094101 | same-account interactive GPU via CPU server SSH | 10.8.103.214 | Completed task; run still reusable | 2026-04-16T09:01:12Z | 2026-04-16T09:08:16Z |

MLP workload image는 `sr-ar-interactive-media-exp/jy-cropping-260415-tfs4.57.6-rsync`이고, 각 workload는 `nvidia-smi`, torch CUDA smoke, 학습, val/test 평가, 비교, PNG 시각화, 리포트 생성을 self-contained command로 실행했다. 감지된 GPU는 NVIDIA A100-SXM4-80GB 1장이다.

HQ-320 run은 updated `AGENTS.md`의 규칙에 따라 먼저 같은 계정 interactive run `1094101/jy-mcn-gpu-spot-1`을 확인했다. 직접 `jumping-host.n6.sr-cloud.com` 비대화식 SSH는 publickey/password 인증 문제로 실패했지만, remote CPU 서버 `10.15.192.126`을 경유한 `ssh jaden.ju@10.8.103.214` 경로는 성공했다. 별도로 생성한 MLP smoke workload `1094223/gpu-hq320-smoke-20260416-173149`는 이후 중지했고, 실제 full training은 interactive A100에서 `artifacts/mobilecropnet_v4/gpu_runs/run_hq320_explain_full.sh`로 실행했다. Balanced-288 explanation control run은 같은 interactive A100에서 `artifacts/mobilecropnet_v4/gpu_runs/run_profile_explain_full.sh`로 실행했다.

## 5. 실험 설정

| run | input size | proposal q | width mult | token dim | candidate k | epochs | batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mcn-v4-q24-288-w075-20260416-095704` | 288 | 24 | 0.75 | 128 | 24 | 8 | 24 |
| `mcn-v4-nb-q16-224-w075-20260416-095704` | 224 | 16 | 0.75 | 128 | 24 | 6 | 32 |
| `mcn-v4-q16-256-w075-20260416-095704` | 256 | 16 | 0.75 | 128 | 24 | 8 | 32 |
| `mcn-v4-q16-256-w100-20260416-095704` | 256 | 16 | 1.00 | 128 | 24 | 8 | 32 |
| `mcn-v4-hq320-explain-20260416-173509` | 320 | 32 | 1.00 | 192 | 32 | 6 | 16 |
| `mcn-v4-balanced288-explain-20260416-180111` | 288 | 24 | 0.75 | 128 | 24 | 6 | 16 |

기존 병렬 실험의 공통 학습 설정은 cosine scheduler, warmup 1 epoch, AMP, `num_workers=4`, `gpu_usage_sample_interval=5`다.

HQ-320 run의 추가 설정은 `backbone_name=mobilenetv4_hybrid_medium`, `backbone_pretrained=false`, `ranker_type=set_transformer`, `ranker_depth=2`, `score_target_mode=crop_utility_prob`, `explicit_pairwise_weight=0.15`, `top1_risk_weight=0.10`, `selection_metric=sstk_composite`, `precompute_sample_tensors=true`, `image_tensor_cache_size=512`, `prefetch_factor=4`, `gpu_usage_sample_interval=10`이다. 이는 `MobileCropNet_v4_0.md`의 권장 HQ-320 프로파일을 현재 구현 가능한 범위에서 가장 가깝게 적용한 설정이다.

Balanced-288 explanation control run은 HQ-320과 같은 loss/label/evaluation/explanation 출력 경로를 사용하되 `model_profile=balanced_288`만 적용했다. 목적은 explanation 후처리 추가가 성능을 낮추는지, 아니면 HQ-320 프로파일 자체의 학습/일반화 문제가 성능 차이를 만드는지 분리해 보기 위함이다.

### 5.1 Pretrained backbone 정렬

현재까지 생성된 MobileCropNet v4 checkpoint는 모두 외부 pretrained backbone 없이 학습됐다. `mcn-v4-hq320-explain-20260416-173509`의 checkpoint/config는 `backbone_name=mobilenetv4_hybrid_medium`, `backbone_pretrained=false`이고, `mcn-v4-balanced288-explain-20260416-180111`은 `backbone_name=custom_depthwise`, `backbone_pretrained=false`다. 더 이전 q16/q24 계열 run도 외부 timm backbone 설정이 없었으므로 custom depthwise random initialization으로 보는 것이 맞다.

이후 실험부터는 모든 profile preset이 pretrained backbone을 기본값으로 사용하도록 정렬했다.

| profile | `backbone_name` | `backbone_pretrained` | pretrained source | 입력 정규화 |
|---|---|---:|---|---|
| HQ-320 | `mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k` | true | ImageNet-12k pretrain + ImageNet-1k finetune | ImageNet mean/std |
| Balanced-288 | `mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k` | true | ImageNet-12k pretrain + ImageNet-1k finetune | ImageNet mean/std |
| Turbo-256 | `mobilenetv4_conv_small.e3600_r256_in1k` | true | ImageNet-1k pretrained | `0.5,0.5,0.5` / `0.5,0.5,0.5` |

RepViT 계열은 timm에 `repvit_m2.dist_in1k`, `repvit_m2_3.dist_300e_in1k`, `repvit_m2_3.dist_450e_in1k` 등이 있어 deployment/latency ablation 후보로 유지한다. 다만 현재 품질 개선 기본 preset은 공개 timm weight 중 ImageNet-12k pretrain 후 ImageNet-1k finetune이 있는 MobileNetV4 계열을 우선한다. 평가/추론 script는 checkpoint의 `train_config.image_mean/std`를 자동 사용하며, checkpoint reload 시에는 저장된 state dict를 로드하므로 외부 pretrained download를 반복하지 않는다.

### 5.2 모델별 핵심 설명

MobileCropNet v4 계열 모델은 모두 “이미지에서 crop 후보를 만들고, 같은 이미지 안의 후보들을 ranking해서 top-1 crop을 선택하는 모델”이다. 차이는 backbone 표현력, 입력 해상도, 후보 수, ranker 구조, 그리고 loss/selection이 top-return을 얼마나 직접적으로 보는지에 있다.

| model/profile | 구조 핵심 | 학습/loss 핵심 | 장점 | 현재 판단 |
| --- | --- | --- | --- | --- |
| `q24_288_w075` 기존 best | 288 입력, 24 candidate/proposal, custom depthwise backbone, relation-lite ranker | score/listwise/pairwise/risk/proposal loss 중심 | top-1 MOS와 regret이 여전히 강함 | 기존 제품 후보 baseline. 단 PCC/SRCC는 낮아 후보 전체 score ordering은 약함 |
| `balanced_288` | 288 입력, MobileNetV4 Conv-M pretrained, relation-lite ranker | explanation/detail head와 SSTK ranking loss를 함께 사용 | pretrained backbone으로 ranking correlation이 개선됨 | explanation 검증과 중간 비교용. top-return best는 아님 |
| `hq_320` | 320 입력, MobileNetV4 Hybrid-M pretrained, set-transformer ranker depth 2 | HQ 프로파일 검증, explanation head 포함 | 설계 문서의 권장 고품질 프로파일을 실행 검증 | 현재 SSTK selection/loss에서는 top-return이 낮아 product 후보 아님 |
| `turbo_256` | 256 입력, MobileNetV4 Conv-S pretrained, relation-lite ranker | 빠른 실험과 경량 추론을 목표로 함 | GPU/latency 부담이 가장 낮음 | SSTK validation top-return은 높을 수 있지만 official MOS 일반화가 불안정한 반례 |
| `rank_320` | 320 입력, MobileNetV4 Conv-M pretrained, 32 candidate/proposal, token 192, set-transformer ranker depth 2 | `top_return_loss`, pairwise/listwise, risk, proposal loss를 결합하고 `sstk_topreturn`/`sstk_balanced_topreturn`으로 선택 | PCC/SRCC와 Acc 계열이 동시에 개선되고 teacher보다 top-return이 좋음 | SSTK-only Product Track 핵심 후보군. 초기 `bal-s18`/`bal-tr06`은 기준선으로 유지 |
| `rank_320` corrected sweep | 320 입력, MobileNetV4 Conv-M pretrained, corrected `sstk_balanced_topreturn`, `top_return_weight=0.55/0.6/0.65/0.7`, LR/WD/warmup ablation | GAIC MOS 없이 SSTK-only validation으로 checkpoint 선택 | top-return과 Acc 계열이 추가 개선됨 | `tr06-s19`가 현재 균형형 product candidate. `tr070`은 top-return tie지만 PCC/SRCC가 낮아 보조 후보 |
| `hybrid384` | 384 입력, 48 candidate/proposal, MobileNetV4 Hybrid-M pretrained, token 256, set-transformer depth 3 | `rank_320`과 같은 loss/selection을 더 큰 입력/후보 수로 확장 | top-1 MOS/regret과 Acc1/10이 현재 최상위 | top-return best. 단 Acc4/10/Accw4/10과 비용을 함께 봐야 하므로 `rank_320 tr06-s19`와 병행 후보 |
| `plus_384` ConvNeXtV2 계획 | 384 입력, 48 candidate/proposal, ConvNeXtV2-Tiny급 backbone, set-transformer depth 3 | heavy profile 비교 계획 | 더 높은 표현력과 후보 coverage 기대 | remote pretrained weight 부재로 `1095153`은 종료. 동일 목적은 `hybrid384`로 대체 검증 |
| `t6safe-r320` diagnostic | `rank_320` 구조를 유지하고 T6 deep ranker score label로 학습 | GAIC MOS-trained T6 teacher score를 crop score supervision으로 사용 | 모델 구조의 가능 상한과 label alignment 효과를 진단 | SSTK-only product가 아니며, 상용 후보 선택에는 사용하지 않는다 |
| `arx2` Product AR cropper | AR-constrained proposal, mode-aware explanation applicability, FREE 및 fixed AR row 학습 | GAIC-v2 SSTK baseline label의 `(image, target_ar)` row, proposal/ranking/explanation loss | target AR proposal 호환성 1.0, object mode에서 person-only explanation 차단 | 현재 MobileCropNet Product AR track의 직접 관련 baseline. official MOS best는 Turbo-256 arx2 |
| GAIC public cropper score teacher | public GAIC/CGS 계열 crop ranker, official candidate crop을 직접 scoring | MobileCropNet 학습 모델이 아니라 benchmark teacher/direct scorer | GAIC official MOS와 가장 강하게 정렬됨 | 데이터 큐레이션/teacher 최적화 track의 기준 scorer다. candidate scorer이므로 임의 이미지에서는 별도 candidate/proposal bank가 필요하고, AR policy/checklist explanation은 없다 |
| `pubscore-r320k96` | 320 입력, MobileNetV4 Conv-M pretrained, 96 candidate, proposal 48 | GAIC public cropper score를 crop score teacher로 distill | k32 대비 candidate coverage mismatch를 크게 줄임 | 과거 public-score diagnostic distillation baseline. Product AR cropper 후보가 아니다 |
| `pubdistill-r320k128-zb035-s20` | 320 입력, MobileNetV4 Conv-M pretrained, 128 candidate, proposal 48, set-transformer ranker depth 2 | rank percentile + raw `ranker_score` z-score blend, teacher soft distribution distillation, top-4 coverage loss, public-score 전용 selection | public-score distillation 진단에서 PCC/SRCC/top-1 MOS/regret best | GAIC scorer teacher 압축 가능성 분석용. FREE-only 학습 label 기반이므로 Product AR cropper 성능 근거로 쓰지 않는다 |
| `pubdistill-r320k128-zb035-s21` | `zb035-s20`과 같은 구조/loss, seed만 변경 | 동일한 k128 full-candidate distillation 설정 | public-score distillation 진단에서 Acc1/10 best | seed variance 확인용. Product AR cropper 후보가 아니다 |
| `pubdistill-r320k128-strongtopk-s22` | k128 구조에 listwise/distill/top-k coverage weight를 더 강하게 적용 | Acc4/10을 의식한 stronger top-k return loss | public-score distillation 진단에서 Acc1/5, Acc4/10, Accw4/10 best | metric별 top-k 반환 diagnostic. Product AR cropper 후보가 아니다 |

`rank_320`은 현재 가장 중요한 모델이다. 입력 이미지를 320 해상도로 보고, ImageNet-12k pretrain 후 ImageNet-1k finetune된 `MobileNetV4 Conv-M` backbone으로 feature를 만든 뒤, 32개 crop 후보를 set-transformer ranker가 함께 비교한다. 즉 후보를 독립적으로 점수화하는 모델이 아니라, 같은 이미지 안에서 어떤 crop이 상대적으로 더 좋은지를 학습하는 구조다.

`rank_320`의 학습 목표는 teacher score 절대값 회귀가 아니라 제품에서 실제 반환되는 top-1 crop 품질을 높이는 것이다. 이를 위해 SSTK label 기준 high-score safe 후보들을 positive bag으로 만들고, 모델 top-1이 이 bag 안에 들어가도록 `top_return_loss`를 추가했다. 여기에 pairwise/listwise ranking loss, unsafe/reject/overflow crop을 억제하는 top-1 risk loss, proposal coverage loss, 낮은 weight의 explanation/detail loss를 함께 사용한다. GAIC MOS annotation은 이 과정에 사용하지 않고, 학습 완료 후 official benchmark 사후 평가에만 사용한다.

현재 `rank_320` 결과는 SSTK teacher와 역할이 다르다. teacher는 후보 전체의 score ordering에서 PCC/SRCC가 더 높지만, 실제 제품이 하나의 crop만 반환하는 top-return 지표에서는 `rank_320`이 teacher보다 좋다. 2026-04-17 corrected sweep 이후에는 `rank_320 tr06-s19`가 균형형 후보이고, `hybrid384`가 top-return best다. 따라서 다음 개선은 teacher imitation을 더 강하게 하는 것이 아니라, SSTK-only validation 안에서 top-return, ranking correlation, score calibration, destructive-negative suppression, proposal coverage를 균형 있게 반영하는 방향으로 진행한다.

`pubdistill-r320k128` 계열은 성격이 다르다. GAIC public cropper score를 teacher로 쓰기 때문에 official GAIC MOS benchmark에는 강하지만, MobileCropNet의 공식 제품 목표인 `Product AR cropper + explanation` 결론을 대체하지 않는다. 특히 이 계열의 학습 label은 실제로 `FREE` candidate scorer distillation 중심이므로 fixed AR crop 품질을 보장하지 않는다. 앞으로 이 track은 "강한 candidate scorer student" 개발이 아니라 GAIC benchmark scorer teacher와 label curation 정책을 분석하는 별도 자료로만 유지한다.

## 6. 평가지표 정의와 해석

MobileCropNet v4.0 평가는 단일 지표가 아니라 ranking 품질, label utility 보존, proposal geometry 품질, score ordering 품질을 함께 본다. 특히 crop 품질은 박스 IoU만으로 판단하기 어렵다. 큰 박스나 full crop은 IoU가 높게 나올 수 있지만, teacher score가 낮고 실제 aesthetic/subject-preserving crop으로는 부적절할 수 있다.

이번 수정 이후 제품/논문 수준의 주 평가는 GAIC 논문(`https://arxiv.org/pdf/1909.08989`)의 공식 benchmark 방식에 맞춰, 이미지별 공식 annotation crop 전체의 MOS와 모델/teacher score를 비교한다. replay-label 지표는 학습 라벨을 얼마나 잘 모사하는지 확인하는 보조 지표다.

공식 MOS 평가의 실제 계산 경로는 다음으로 고정한다.

1. `data/Publics/GAIC_v2/annotations_json/instances_test.json`의 official annotation crop 전체를 이미지별 candidate set으로 로드한다.
2. 각 official crop bbox를 원본 좌표계에서 MobileCropNet 입력 letterbox 좌표계로 변환한다.
3. 변환된 official crop bbox 전체를 `model(image, boxes, valid, target_ar_id, image_ar_log, candidate_is_base, box_meta)`에 넣는다.
4. `torch.sigmoid(outputs["utility_logits"])`를 official crop별 predicted score로 사용한다.
5. 같은 candidate index의 official MOS와 predicted score로 이미지별 PCC/SRCC를 계산하고, predicted score 상위 K개가 MOS top-N에 드는지로 `AccK/N`, `AccwK/N`, top-1 MOS, MOS regret을 계산한다.

따라서 PCC/SRCC/AccK/N 평가는 모델 proposal head가 만든 crop으로 수행하지 않는다. proposal head는 MobileCropNet forward 내부에서 항상 계산되지만, official MOS metric에서는 proposal box/logit을 top-1 선택이나 candidate score 계산에 사용하지 않는다. proposal head 결과는 replay-label 평가, proposal recall, 시각화, product candidate generation의 geometry sanity check로 따로 해석한다. 이 분리가 중요하다. official benchmark의 질문은 "이미 주어진 GAIC annotation crop 후보들의 MOS 순서를 모델 utility head가 얼마나 잘 재현하는가"이고, proposal head 평가는 "모델이 좋은 crop 후보를 스스로 생성할 수 있는가"이기 때문이다.

Public cropper 비교도 같은 원칙을 따른다. CGS와 GAIC public model은 candidate ranker라서 official annotation crop bbox를 모델 scoring head에 직접 넣고 MOS와 score를 비교한다. CACNet은 단일 crop regression 모델이라 official crop 후보별 native score가 없다. 따라서 CACNet의 `PCC`, `SRCC`, `AccK/N`, `AccwK/N`, top-1 MOS, MOS regret은 primary GAIC candidate-ranking 방식으로 계산할 수 없으며 비교 표에서는 `-`로 둔다. CACNet에 대해 별도로 기록한 IoU projection 값은 "CACNet이 낸 한 개 crop과 가장 가까운 official annotation crop이 어느 정도 품질인가"를 보는 참고 진단값일 뿐, GAIC candidate-ranking metric이 아니다. SSTK teacher는 현재 `candidate_eval_rows.jsonl`에서 official crop별 exact teacher score를 읽는 방식을 우선 사용하고, exact row가 없을 때만 nearest-IoU projection을 보조로 사용한다.

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
| `eval/score coverage` | 0 to 1 | 높을수록 좋음 | 해당 method가 평가된 official test 범위 또는 teacher score가 존재하는 annotation 비율 | MobileCropNet full row는 500장 전체라 1.0이다. 이번 teacher exact row도 GAIC v2 test 500장 전체 official crop에 대해 score coverage 1.0이다. |

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

본 보고서에서 제품/논문 수준 모델 선택의 1차 기준은 공식 GAIC MOS benchmark의 `SRCC`, `Acc1/5`, `Acc1/10`, `Acc4/10`, `Accw4/10`, `top1 MOS`, `MOS regret`이다. Replay-label의 `test top-1 hit`, `NDCG@5`, `utility regret`은 학습 신호가 의도대로 모사되는지 보는 보조 진단이고, `proposal recall@5 IoU0.5`와 `top1 IoU to best positive`는 proposal/head geometry sanity check로 사용한다. 최종 제품 판단에서는 official MOS 지표에 latency, model size, mobile memory, public benchmark 일반화 성능을 추가해야 한다.

## 7. 공식 GAIC benchmark 평가

공식 GAIC benchmark 평가는 논문 split 구조인 `data/Publics/GAIC_v2/annotations/{train,val,test}`와 `data/Publics/GAIC_v2/images/{train,val,test}`를 COCO-style JSON으로 재생성한 뒤 수행했다. 생성된 split은 train 2,636장, val 200장, test 500장이다. txt annotation의 좌표는 `y1 x1 y2 x2 MOS`이며 COCO bbox `[x, y, w, h]`로 변환했다. `MOS=-2`는 GAIC_v2의 unrated crop으로 보고 제외했고, 기존 `data/Publics/GAIC/annotations_json` subset과 겹치는 train 1,939장/test 349장에서는 bbox, MOS, `gt_flag`가 완전히 일치함을 검증했다. 이 평가는 SSTK teacher label을 ground truth로 사용하지 않는다.

평가 산출물:

- COCO JSON summary: `data/Publics/GAIC_v2/annotations_json/conversion_summary.json`
- train JSON: `data/Publics/GAIC_v2/annotations_json/instances_train.json`
- val JSON: `data/Publics/GAIC_v2/annotations_json/instances_val.json`
- test JSON: `data/Publics/GAIC_v2/annotations_json/instances_test.json`
- model-only metrics: `artifacts/mobilecropnet_v4/gaic_benchmark_eval/q24_288_w075_gaic_v2_test500/metrics.json`
- model + full-option teacher 500 metrics: `artifacts/mobilecropnet_v4/gaic_benchmark_eval/q24_288_w075_gaic_v2_test500_teacher500_qf_c1c6capexp_c7exp_a100/metrics.json`
- model + full-option teacher 500 report: `artifacts/mobilecropnet_v4/gaic_benchmark_eval/q24_288_w075_gaic_v2_test500_teacher500_qf_c1c6capexp_c7exp_a100/GAIC_BENCHMARK_EVAL_REPORT.md`
- model per-image: `artifacts/mobilecropnet_v4/gaic_benchmark_eval/q24_288_w075_gaic_v2_test500_teacher500_qf_c1c6capexp_c7exp_a100/mcn_v4_q24_288_w075_per_image.jsonl`
- teacher exact per-image: `artifacts/mobilecropnet_v4/gaic_benchmark_eval/q24_288_w075_gaic_v2_test500_teacher500_qf_c1c6capexp_c7exp_a100/sstk_teacher_subjectfix_v2_equalized_qf_c1c6capexp_c7exp_a100_official_gc_per_image.jsonl`
- teacher benchmark full-option rerun: `data/GAIC_v2/Test500_SSTK_QF_C1C6CAPEXP_C7_EXP_A100/artifacts/reports/gaic_benchmark_eval_gaic_v2_test500_subjectfix_v2_equalized_qf_c1c6capexp_c7exp_a100/benchmark_summary.json`
- teacher candidate eval rows: `data/GAIC_v2/Test500_SSTK_QF_C1C6CAPEXP_C7_EXP_A100/artifacts/reports/gaic_benchmark_eval_gaic_v2_test500_subjectfix_v2_equalized_qf_c1c6capexp_c7exp_a100/candidate_eval_rows.jsonl`

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
| `mcn_v4_sstk_conservative` | SSTK-only conservative 사후 평가 | 500 | 43,123 | 0.421379 | 0.394981 | 0.378000 | 0.542000 | 0.466000 | 0.363532 | 3.706440 | 0.525460 |
| `mcn_v4_balanced288_explain` | Balanced-288 explanation control | 500 | 43,123 | 0.413289 | 0.384878 | 0.436000 | 0.608000 | 0.469500 | 0.364822 | 3.779800 | 0.452100 |
| `mcn_v4_hq320_explain` | HQ-320 MobileNetV4-Hybrid-M 사후 평가 | 500 | 43,123 | 0.322124 | 0.313649 | 0.254000 | 0.372000 | 0.328500 | 0.248184 | 3.531740 | 0.700160 |
| `gaic_mos_oracle` | GAIC v2 test 500 upper bound | 500 | 43,123 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 4.231900 | 0.000000 |

SSTK teacher와의 비교는 같은 official MOS annotation을 기준으로 수행했다. 기존 exact teacher candidate scoring 산출물은 118장만 포함했고, 직전 재실행도 기존 `data/GAIC/All/artifacts/candidates`와 `data/GAIC/All/artifacts/precompute` 산출물이 커버하던 150장 subset으로 제한됐다. 이후 cheap C2/C3/C5+C7 재실행으로 500장 coverage는 확보했지만, full-option 판단에는 C1/caption/C6/real-expensive가 빠져 있었다. 이번 full run에서는 GAIC v2 test 500장을 새로 staging하고 C1, C2, C3, C5, C6 gaze, C7 saliency, BLIP caption, subject-mode routing, production candidate generation, real-expensive align/aesthetic scoring을 전체 500장에 대해 생성했다. C4 OCR는 현재 미구현이므로 제외했다.

원격 GPU 실행 정보:

| item | value |
| --- | --- |
| MLP run | `gaic-v2-fullopt-c1c6capexp-20260416-131603` |
| run id | `1093964` |
| GPU | NVIDIA A100-SXM4-80GB 1장 |
| status | `Succeeded` |
| start UTC | `2026-04-16T04:17:16Z` |
| end UTC | `2026-04-16T05:45:13Z` |

full-option 산출물 sanity:

| item | value |
| --- | ---: |
| staged GAIC v2 test images | 500 |
| BLIP captions generated | 500 |
| validation status | `ok` |
| C7 saliency processed | 500 |
| avg production candidates/image | 511.344 |
| teacher score rows | 500 |
| official annotation crops | 43,123 |
| full-expensive cache rows | 84,889 |

새 teacher benchmark coverage는 다음과 같다.

| item | value |
| --- | ---: |
| GAIC v2 test images | 500 |
| official annotation crops | 43,123 |
| `candidate_eval_rows.jsonl` rows | 86,246 |
| Gc evaluated images | 500 |
| Ge evaluated images | 500 |
| teacher score coverage | 1.000000 |

500장 전체 direct 비교 결과는 다음과 같다.

| method | scope | images | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/5 | Acc4/10 | Accw4/10 | top1 MOS | MOS regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mcn_v4_q24_288_w075` | GAIC v2 test 500 | 500 | 0.399840 | 0.368395 | 0.434000 | 0.630000 | 0.298000 | 0.469000 | 0.365610 | 3.804520 | 0.427380 |
| `sstk_teacher_subjectfix_v2_equalized_qf_c1c6capexp_c7exp_a100_official_gc` | full-option teacher exact GAIC v2 test 500 | 500 | 0.515146 | 0.498780 | 0.278000 | 0.442000 | 0.224000 | 0.373500 | 0.282491 | 3.604000 | 0.627900 |

해석:

- MobileCropNet v4는 GAIC v2 test 500장 전체에 대해 직접 scoring되었고, 이 값이 현재 제품/논문 판단용 primary metric이다.
- full-option teacher exact 비교에서는 SSTK teacher가 PCC `0.515146`, SRCC `0.498780`으로 MobileCropNet의 PCC `0.399840`, SRCC `0.368395`보다 높다. C1/C6/caption/real-expensive를 추가해도 teacher score는 official MOS와의 전반적 ranking correlation에서 강점을 유지한다.
- 반대로 MobileCropNet은 Acc1/5 `0.434000`, Acc1/10 `0.630000`, Acc4/10 `0.469000`, Accw4/10 `0.365610`, top-1 MOS `3.804520`, MOS regret `0.427380`으로 full-option teacher보다 높다. 즉 현재 teacher scoring은 전체 ranking signal로는 유용하지만, 단일 top-1 crop 선택에서는 official MOS 최고군을 안정적으로 고르는 목적과 완전히 일치하지 않는다.
- cheap C2/C3/C5+C7 teacher 대비 full-option teacher는 SRCC가 `0.489976`에서 `0.498780`으로 상승했고, Acc1/10도 `0.428000`에서 `0.442000`으로 상승했다. 다만 PCC는 `0.522410`에서 `0.515146`으로 소폭 하락했고, Acc4/10과 Accw4/10도 하락했다. 따라서 full-option 추가는 ranking correlation을 일부 개선했지만 top-return 품질을 일관되게 개선하지는 못했다.
- 이 결과는 teacher가 ground truth가 아니라 학습 신호라는 점을 뒷받침한다. 이후 최종 모델 선택은 teacher imitation 성능이 아니라 official GAIC MOS, 외부 public benchmark, 정성 시각화, latency를 함께 봐야 한다.
- 기존 150장 제한은 `data/GAIC_v2/Test500_SSTK_QF_C1C6CAPEXP_C7_EXP_A100` 산출물로 해소됐고, full-option 기준으로도 500장 coverage를 확보했다. 이 산출물은 GAIC v2 test 500장에 대해 C1/C2/C3/C5/C6 raw precompute 500 row, routed C7 features 500 row, captions 500 row, candidates 500 row, teacher scores 500 row, official Gc/Ge candidate eval 86,246 row를 포함한다.

### 7.1 HQ-320 MobileNetV4-Hybrid-M + explanation full run

`MobileCropNet_v4_0.md`의 권장 HQ-320 프로파일을 기준으로 `MobileNetV4-Hybrid-M` backbone과 set-transformer ranker를 적용한 full run을 수행했다. 실행 runner는 `artifacts/mobilecropnet_v4/gpu_runs/run_hq320_explain_full.sh`이고, 원격 경로와 로컬 동기화 경로는 모두 `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509`다.

HQ-320 run 산출물:

- checkpoint: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/best.pt`
- run summary: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/run_summary.json`
- replay test metrics: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/eval_test/metrics.json`
- official GAIC MOS metrics: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/gaic_official_test/metrics.json`
- official GAIC MOS + teacher metrics: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/gaic_official_test_with_teacher/metrics.json`
- predictions with explanation labels: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/eval_test/predictions.jsonl`
- PNG contact sheet: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/viz_test/contact_sheet.png`
- PNG overlays: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/viz_test/overlays/*.png`

HQ-320 training/validation summary:

| item | value |
| --- | ---: |
| best epoch | 5 |
| SSTK composite selection score | 0.783664 |
| val top-1 hit | 0.742952 |
| val exact-best score | 0.514421 |
| val proposal recall@0.5 | 0.990193 |
| val explicit pairwise accuracy | 0.814768 |
| val risk suppression accuracy | 0.518613 |
| val top-1 risk rate | 0.096314 |

HQ-320 replay test summary:

| metric | value |
| --- | ---: |
| candidate top-1 hit | 0.104000 |
| candidate exact-best | 0.131200 |
| NDCG@5 | 0.778383 |
| NDCG@10 | 0.825378 |
| PCC | 0.457866 |
| SRCC | 0.487805 |
| proposal recall@1 IoU0.5 | 0.820800 |
| proposal recall@5 IoU0.5 | 0.846400 |
| top1 IoU to best positive | 0.807610 |
| utility regret | 0.331880 |

HQ-320 official GAIC MOS 500장 결과와 teacher 비교:

| method | images | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/5 | Accw4/5 | top1 MOS | MOS regret | top1 rank pct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mcn_v4_hq320_explain` | 500 | 0.322124 | 0.313649 | 0.254000 | 0.372000 | 0.207500 | 0.161876 | 3.531740 | 0.700160 | 0.761901 |
| `mcn_v4_q24_288_w075` | 500 | 0.399840 | 0.368395 | 0.434000 | 0.630000 | 0.298000 | 0.244742 | 3.804520 | 0.427380 | 0.889153 |
| `mcn_v4_sstk_conservative` | 500 | 0.421379 | 0.394981 | 0.378000 | 0.542000 | 0.293500 | 0.241489 | 3.706440 | 0.525460 | 0.846648 |
| `sstk_teacher_subjectfix_v2_equalized_qf_c1c6capexp_c7exp_a100_official_gc` | 500 | 0.515146 | 0.498780 | 0.278000 | 0.442000 | 0.224000 | 0.175641 | 3.604000 | 0.627900 | 0.774512 |

해석:

- HQ-320은 `MobileNetV4-Hybrid-M` backbone과 set-transformer ranker가 정상 학습, checkpoint 저장, replay 평가, official MOS 평가, PNG explanation 시각화까지 end-to-end 동작함을 검증했다.
- 그러나 official MOS 기준에서는 기존 Balanced-288 best와 SSTK-only conservative보다 낮다. 특히 Acc1/5 `0.254000`, Acc1/10 `0.372000`, top1 MOS `3.531740`, MOS regret `0.700160`은 top-return 품질이 크게 약화됐음을 의미한다.
- val SSTK composite와 replay validation은 양호하지만 official MOS 사후 성능이 악화됐으므로, 현재 selection score가 official top-return 품질을 충분히 대변하지 못한다. HQ-320을 제품/논문 후보로 쓰려면 GAIC annotation을 학습에 쓰지 않더라도 SSTK-only top-return surrogate, destructive-negative suppression, candidate/proposal coupling을 더 강하게 넣어야 한다.
- explanation 출력은 정성 검증에 유효하다. `predictions.jsonl`의 top candidate에는 `model_explanation.utility/positive/risk`, `model_checklist.aesthetic/subject/composition/technical` label과 score가 포함되고, PNG overlay 오른쪽 side panel에는 `decision`, `subject mode`, top crop utility, positive/risk, checklist label이 표시된다.

### 7.2 Balanced-288 explanation control run

HQ-320 성능 저하 원인 중 하나로 “explanation 출력 추가가 학습 또는 추론 score를 바꿨는가”를 점검하기 위해 Balanced-288도 동일한 explanation 출력 경로로 재학습했다. 현재 구현에서 explanation은 `eval_utils.py`와 PNG 시각화 단계에서 기존 `utility/positive/risk/macro/route/decision` 출력을 label로 변환하는 후처리다. 별도의 explanation loss나 학습 target이 추가된 것은 아니다.

Balanced-288 explanation control 산출물:

- runner: `artifacts/mobilecropnet_v4/gpu_runs/run_profile_explain_full.sh`
- checkpoint: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/best.pt`
- run summary: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/run_summary.json`
- official GAIC MOS metrics: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/gaic_official_test/metrics.json`
- official GAIC MOS + teacher metrics: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/gaic_official_test_with_teacher/metrics.json`
- predictions with explanation labels: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/eval_test/predictions.jsonl`
- PNG contact sheet: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/viz_test/contact_sheet.png`

Matched explanation run 비교:

| run | best epoch | selection | replay top1 hit | replay NDCG@5 | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/10 | Accw4/10 | top1 MOS | MOS regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mcn_v4_balanced288_explain` | 4 | 0.775186 | 0.068800 | 0.769326 | 0.413289 | 0.384878 | 0.436000 | 0.608000 | 0.469500 | 0.364822 | 3.779800 | 0.452100 |
| `mcn_v4_hq320_explain` | 5 | 0.783664 | 0.104000 | 0.778383 | 0.322124 | 0.313649 | 0.254000 | 0.372000 | 0.328500 | 0.248184 | 3.531740 | 0.700160 |

해석:

- Balanced-288에 동일한 explanation 출력 경로를 붙여도 official MOS 성능은 기존 Balanced-288 best와 거의 같은 수준이다. 기존 best 대비 Acc1/5는 `0.434000 -> 0.436000`, Acc4/10은 `0.469000 -> 0.469500`으로 유지됐고, PCC/SRCC/top1 MOS는 소폭 변동 수준이다.
- 따라서 HQ-320 저하는 explanation label/PNG 후처리 때문이 아니라 HQ-320 프로파일과 현재 SSTK-only 학습/selection 신호 사이의 불일치가 주원인이라고 보는 것이 타당하다.
- 중요한 역전 현상도 확인된다. HQ-320은 replay top1 hit `0.104000`과 NDCG@5 `0.778383`이 Balanced-288 control보다 약간 높지만, official MOS Acc1/5와 top1 MOS는 크게 낮다. 즉 replay-label 모사 개선이 official MOS best-return 개선으로 연결되지 않는 selection mismatch가 존재한다.

### 7.3 Detailed checklist explanation heads + three-profile parallel full run

기존 explanation 시각화가 `composition: ok 0.63`처럼 거친 macro label만 보여준 이유는 모델/데이터 adapter가 실제 SSTK detailed checklist를 학습하지 않았기 때문이다. 기존 구현은 `macro_targets`를 `aesthetic`, `subject`, `composition`, `technical` 4개 score로만 받아 `macro_head`를 학습했고, `checklist_labels`, `checklist_scores`, `composition_focus`, `why_tags`, `reject_tags`, `safety_penalty`는 tensor target이나 loss로 연결하지 않았다. 따라서 라벨 시각화 산출물에는 `headroom_loose`, `lookroom_excessive`, `rule_of_thirds_strong`가 존재해도 student output에는 이를 복원할 head가 없었다.

이번 구현에서는 다음 경로를 추가했다.

- `src/mobilecropnet_v4/data.py`: `checklist_labels`를 13개 multi-class group으로 decode하고, `checklist_scores`/`composition_focus`/`safety_penalty`를 22개 normalized detail score target으로 변환한다. `why_tags`와 `reject_tags`는 31개 multi-label target으로 변환한다.
- `src/mobilecropnet_v4/model.py`: `checklist_class_head`, `detail_score_head`, `why_tag_head`를 추가하고 각각 CE, SmoothL1, BCE auxiliary loss로 학습한다.
- `src/mobilecropnet_v4/eval_utils.py`, `src/scripts/infer_mobilecropnet_v4.py`: top candidate와 후보별 prediction에 `model_detailed_checklist`, `model_detail_scores`, `model_why_tags`를 기록한다.
- `src/scripts/visualize_mobilecropnet_v4_predictions.py`: PNG side panel에 model detail, label detail, why tag, bbox legend, bbox detail을 표시한다. 색상은 red=model top crop, green=best positive label, blue=baseline, orange=proposal head top-1이다.
- `artifacts/mobilecropnet_v4/gpu_runs/run_profile_detail_explain_full.sh`: pretrained backbone smoke, training, replay eval, method comparison, official MOS eval, teacher comparison, PNG visualization, `run_summary.json` 생성을 하나의 bounded GPU workload command로 묶었다.

세 profile은 updated `AGENTS.md` 원칙대로 multi-GPU 한 run이 아니라 단일 GPU workload 3개를 병렬 생성했다. 모든 run은 local timm pretrained weight를 사용했고, remote GPU에서 Hugging Face 다운로드를 요구하지 않는다.

| profile | run id | run name | status | start UTC | end UTC | core count |
| --- | ---: | --- | --- | --- | --- | ---: |
| HQ-320 | `1094298` | `mcn-v4-hq320-detail-explain-20260416-185647` | Succeeded | `2026-04-16T09:57:24Z` | `2026-04-16T10:14:48Z` | 1 |
| Balanced-288 | `1094299` | `mcn-v4-balanced288-detail-explain-20260416-185647` | Succeeded | `2026-04-16T09:57:25Z` | `2026-04-16T10:11:42Z` | 1 |
| Turbo-256 | `1094300` | `mcn-v4-turbo256-detail-explain-20260416-185647` | Succeeded | `2026-04-16T09:57:53Z` | `2026-04-16T10:11:49Z` | 1 |

Replay-label test와 explanation auxiliary head 검증 결과는 다음과 같다.

| profile | best epoch | selection | val top1 hit | val checklist acc | val detail MAE | val why recall | test top1 hit | test NDCG@5 | test SRCC | proposal recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HQ-320 | 6 | 0.783646 | 0.740869 | 0.799271 | 0.151265 | 0.705057 | 0.094400 | 0.788335 | 0.505618 | 0.852800 |
| Balanced-288 | 5 | 0.777786 | 0.720036 | 0.767571 | 0.159347 | 0.649367 | 0.070400 | 0.776128 | 0.449871 | 0.862400 |
| Turbo-256 | 4 | 0.779765 | 0.726758 | 0.775601 | 0.158942 | 0.689563 | 0.134400 | 0.789742 | 0.351870 | 0.860800 |

Official GAIC MOS test 500장 평가는 다음과 같다. `Acc1/5`, `Acc1/10`, `Acc4/10`, `Accw4/10`은 각각 JSON metric key `top1_in_top5`, `top1_in_top10`, `acc4_of_top10`, `accw4_of_top10`에 대응한다.

| profile | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/10 | Accw4/10 | top1 MOS | MOS regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HQ-320 | 0.378403 | 0.353322 | 0.310000 | 0.496000 | 0.403500 | 0.300627 | 3.633340 | 0.598560 |
| Balanced-288 | 0.455300 | 0.434693 | 0.366000 | 0.524000 | 0.414000 | 0.314935 | 3.692760 | 0.539140 |
| Turbo-256 | 0.405003 | 0.378975 | 0.370000 | 0.540000 | 0.430500 | 0.329951 | 3.716800 | 0.515100 |
| SSTK teacher official-Gc | 0.515146 | 0.498780 | 0.278000 | 0.442000 | 0.373500 | 0.282491 | 3.604000 | 0.627900 |

해석:

- detailed explanation head는 실제로 학습됐다. 세 profile 모두 validation checklist accuracy `0.767~0.799`, detail score MAE `0.151~0.159`, why-tag recall `0.649~0.705`를 기록했다. 즉 이전처럼 macro label을 후처리만 한 것이 아니라 SSTK detailed label을 복원하는 head가 동작한다.
- Official MOS top-return 기준에서는 Turbo-256이 세 profile 중 가장 높다. Acc1/10 `0.540000`, Acc4/10 `0.430500`, Accw4/10 `0.329951`, top1 MOS `3.716800`, MOS regret `0.515100`으로 세 profile 중 best-return 지표가 가장 좋다.
- Ranking correlation 기준에서는 Balanced-288이 가장 높다. PCC `0.455300`, SRCC `0.434693`로 세 profile 중 가장 강하다. 다만 top1 MOS는 Turbo보다 낮다.
- HQ-320은 validation composite와 detailed explanation 학습 지표는 높지만 official MOS top-return은 가장 낮다. 큰 backbone/set-transformer가 현재 SSTK-only selection metric에서 자동으로 better top-return을 만들지는 않는다.
- Full-option SSTK teacher는 PCC/SRCC가 세 student보다 높지만 Acc/top-return은 낮다. 이는 teacher score가 image-level ranking tendency는 잘 잡지만 top-1 crop 선택은 여전히 official MOS와 불일치할 수 있음을 뜻한다.
- 세 detailed run 모두 기존 제품 후보 best `mcn-v4-q24-288-w075-20260416-095704`의 official top1 MOS `3.804520`, MOS regret `0.427380`, Acc1/10 `0.630000`에는 못 미친다. 따라서 이번 detailed explanation checkpoint는 제품/논문 best 교체가 아니라 explainability head 검증 및 profile 비교 산출물로 해석한다.

GPU 사용률은 다음과 같다.

| profile | train GPU util mean avg | train GPU util max | nvidia-smi mem max MB | CUDA peak allocated MB | power max W | samples/s avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HQ-320 | 15.982 | 40.000 | 2501.000 | 1380.016 | 131.820 | 47.066 |
| Balanced-288 | 10.958 | 39.000 | 1919.000 | 866.285 | 118.550 | 60.065 |
| Turbo-256 | 9.464 | 28.000 | 887.000 | 294.058 | 88.420 | 82.245 |

따라서 현재 규모에서는 2개 이상 GPU가 필요하지 않다. 각 실험은 단일 A100에서 14~17분 안에 smoke, 6 epoch training, replay eval, official MOS eval, teacher comparison, PNG visualization을 완료했다. 여러 실험을 수행할 때는 multi-GPU workload 하나보다 `--core-count=1` workload 여러 개를 병렬 실행하는 방식이 더 효율적이다.

정성 시각화 산출물은 모두 PNG로 검증했다. Contact sheet는 각 run마다 `1680x3840`, 개별 overlay 예시는 `1664x1480`이며, side panel에는 model detail과 label detail이 함께 표시된다.

| profile | prediction JSONL | contact sheet |
| --- | --- | --- |
| HQ-320 | `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-hq320-detail-explain-20260416-185647/eval_test/predictions.jsonl` | `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-hq320-detail-explain-20260416-185647/viz_test/contact_sheet.png` |
| Balanced-288 | `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-balanced288-detail-explain-20260416-185647/eval_test/predictions.jsonl` | `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-balanced288-detail-explain-20260416-185647/viz_test/contact_sheet.png` |
| Turbo-256 | `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-turbo256-detail-explain-20260416-185647/eval_test/predictions.jsonl` | `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-turbo256-detail-explain-20260416-185647/viz_test/contact_sheet.png` |

### 7.4 GAIC-v2 official split baseline label 재학습

2026-04-17에는 GAIC-v2 official split 전체에 대해 생성 완료된 baseline SSTK training label을 사용해 MobileCropNet v4를 다시 학습했다. 목적은 이전 `data/GAIC/All` 기반 v4 split 대신 논문 split 구조와 일치하는 Train2636/Val200/Test500 라벨을 적용하고, 최신 패치된 detailed explanation/data-loader/evaluation/visualization 코드를 같은 조건으로 검증하는 것이다. 이 실험에서도 GAIC official MOS annotation은 학습 또는 checkpoint 선택에 사용하지 않았고, 학습 후 test 500장 official MOS benchmark에서 사후 평가했다.

적용한 라벨 경로는 다음과 같다.

| split | label directory | validation | conditional rows | skipped rows | pairwise rows | listwise rows |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Train2636 | `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/training_labels/gaic_v2_train_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic` | ok, errors 0, warnings 0 | 14,915 | 901 | 56,501 | 14,915 |
| Val200 | `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200/artifacts/training_labels/gaic_v2_val_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic` | ok, errors 0, warnings 0 | 1,136 | 64 | 4,374 | 1,136 |
| Test500 | `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/training_labels/gaic_v2_test_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic` | ok, errors 0, warnings 0 | 2,824 | 176 | 11,155 | 2,824 |

`conditional rows`는 `(image, target_ar)` 학습 row 수다. 일부 image/AR row는 conditional label 생성 단계에서 skip되어 Train2636/Val200/Test500의 image 수와 row image count가 정확히 같지는 않다. 반면 official MOS benchmark는 `data/Publics/GAIC_v2/annotations_json/instances_test.json`의 test 500장, 43,123개 official annotation crop 전체를 그대로 사용한다.

학습 실행은 `artifacts/mobilecropnet_v4/gpu_runs/run_gaic_v2_official_labels_profile.sh`로 수행했다. 세 profile 모두 `backbone_pretrained=true`, `score_target_mode=crop_utility_prob`, explicit pairwise/listwise 연결, top-1 risk loss, pseudo-positive proposal bag, `sstk_composite` checkpoint selection, tensor precompute/cache, PNG visualization을 사용했다.

| profile | backbone | input | candidate k | proposal q | pretrained |
| --- | --- | ---: | ---: | ---: | --- |
| Balanced-288 | `mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k` | 288 | 24 | 24 | ImageNet-12k pretrain + ImageNet-1k finetune |
| HQ-320 | `mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k` | 320 | 32 | 32 | ImageNet-12k pretrain + ImageNet-1k finetune |
| Turbo-256 | `mobilenetv4_conv_small.e3600_r256_in1k` | 256 | 16 | 16 | ImageNet-1k pretrained |

GPU 실행 정보는 다음과 같다. HQ-320은 이미 열려 있던 same-account interactive GPU `1094101`을 사용했고, Balanced-288과 Turbo-256은 단일 A100 MLP workload로 병렬 실행했다. 별도로 생성했던 중복 HQ workload `1094658`은 interactive GPU에서 HQ run을 시작한 뒤 중지했다.

| profile | run id | run name | GPU path | IP | status | start UTC | end UTC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Balanced-288 | `1094659` | `mcn-gaicv2-balanced288-20260417-101528` | MLP spot workload | `10.14.231.39` | Succeeded | `2026-04-17T01:16:16Z` | `2026-04-17T01:35:10Z` |
| HQ-320 | `1094101` | `mcn-gaicv2-hq320-interactive-20260417-101528` | same-account interactive GPU | `10.8.103.214` | Succeeded task | `2026-04-17T01:18:00Z` | `2026-04-17T01:39:43Z` |
| Turbo-256 | `1094660` | `mcn-gaicv2-turbo256-20260417-101528` | MLP spot workload | `10.10.95.130` | Succeeded | `2026-04-17T01:17:40Z` | `2026-04-17T01:43:25Z` |

공식 GAIC MOS test 500장 결과는 다음과 같다. `Acc1/5`, `Acc1/10`, `Acc4/10`, `Accw4/10`은 각각 official MOS top-N return 지표다.

| profile | run | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/10 | Accw4/10 | top1 MOS | MOS regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced-288 | `mcn-gaicv2-balanced288-20260417-101528` | 0.415141 | 0.416445 | 0.262000 | 0.430000 | 0.339000 | 0.247730 | 3.553360 | 0.678540 |
| HQ-320 | `mcn-gaicv2-hq320-interactive-20260417-101528` | 0.272176 | 0.264152 | 0.120000 | 0.188000 | 0.207500 | 0.146377 | 3.184100 | 1.047800 |
| Turbo-256 | `mcn-gaicv2-turbo256-20260417-101528` | 0.446322 | 0.434566 | 0.348000 | 0.538000 | 0.418000 | 0.314088 | 3.718800 | 0.513100 |
| SSTK teacher official-Gc | `sstk_teacher_fullopt_largecap_v2_raw` | 0.514590 | 0.497884 | 0.270000 | 0.448000 | 0.378000 | 0.285575 | 3.598460 | 0.633440 |

Replay-label test 결과는 다음과 같다. 이 표는 label 모사와 학습 안정성을 보는 보조 진단이다.

| profile | best epoch | selection | val top1 hit | replay top1 hit | exact best | NDCG@5 | SRCC | PCC | proposal recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced-288 | 3 | 0.505922 | 0.492958 | 0.493272 | 0.326133 | 0.861513 | 0.321323 | 0.376658 | 0.882436 |
| HQ-320 | 5 | 0.494277 | 0.440141 | 0.436261 | 0.326133 | 0.866287 | 0.344658 | 0.372510 | 0.904391 |
| Turbo-256 | 6 | 0.466966 | 0.403169 | 0.391289 | 0.295680 | 0.851758 | 0.304819 | 0.331197 | 0.938385 |

GPU 사용률은 다음과 같다. 모든 실험은 단일 A100에서 충분히 실행 가능했고, 여전히 multi-GPU 학습이 필요한 규모는 아니다.

| profile | GPU util mean | GPU util max | nvidia-smi mem max MB | CUDA peak allocated MB | samples/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced-288 | 22.38 | 38 | 1,935 | 866.3 | 115.79 |
| HQ-320 | 24.81 | 32 | 2,503 | 1,380.0 | 84.16 |
| Turbo-256 | 14.38 | 25 | 889 | 294.1 | 89.49 |

해석은 다음과 같다.

- 새 GAIC-v2 official split label 기반 run 중 official MOS top-return과 ranking correlation을 함께 보면 Turbo-256이 가장 균형이 좋다. Turbo-256은 teacher보다 PCC/SRCC는 낮지만, top1 MOS `3.718800`과 MOS regret `0.513100`은 teacher row보다 좋다.
- Balanced-288은 validation/replay top-1이 가장 높지만 official top-return에서는 Turbo-256보다 낮다. 따라서 SSTK replay 성능만으로 official MOS top-return을 고르기 어렵다는 점이 다시 확인됐다.
- HQ-320은 pretrained MobileNetV4-Hybrid-M로 실행했음에도 official MOS가 가장 낮다. 큰 backbone과 높은 해상도는 현재 SSTK-only selection/loss에서는 자동 개선을 만들지 못한다.
- 이전 제품 후보 best `mcn-v4-q24-288-w075-20260416-095704`와 비교하면, Turbo-256은 PCC/SRCC가 더 높지만 Acc1/10, top1 MOS, MOS regret은 낮다. 따라서 새 GAIC-v2 label 재학습 checkpoint는 ranking correlation 관점의 후보로 볼 수 있지만, top-return 제품 후보를 즉시 교체할 수준은 아니다.

새 산출물 요약은 `artifacts/mobilecropnet_v4/gaic_v2_official_labels/GAIC_V2_OFFICIAL_LABEL_RUN_SUMMARY_20260417.md`와 `artifacts/mobilecropnet_v4/gaic_v2_official_labels/gaic_v2_official_label_runs_summary_20260417.json`에 저장했다.

### 7.5 AR-constrained proposal + mode-aware explanation patch

2026-04-17 추가 시각화 검토에서 두 가지 문제가 확인됐다. 첫째, `target_ar=3:4` 같은 fixed AR row에서도 proposal head top-1 orange box가 1:1에 가까운 경우가 있었다. 둘째, 사람이 없는 `object_single` 이미지에서 `headroom_loose`, `lookroom_adequate`, `avoid_person_cut` 같은 사람 전용 explanation이 출력될 수 있었다. 두 문제 모두 단순 PNG 렌더링 문제가 아니라 모델 출력 의미 체계와 parameterization의 문제로 판단했다.

이번 패치의 설계 판단은 다음과 같다.

- fixed AR proposal은 구조적으로 target AR을 만족해야 한다. target AR token만 주고 자유 `cxcywh`를 회귀하면 학습 초반 또는 failure case에서 AR mismatch proposal이 정상 출력처럼 보인다.
- `FREE`와 fixed AR은 proposal parameterization을 분기한다. `FREE`는 기존 4자유도 `cxcywh`를 유지하고, fixed AR은 `cx`, `cy`, `scale`만 예측한 뒤 target AR로 `w/h`를 결정한다.
- proposal 평가와 시각화는 raw objectness top-1과 target-AR compatible top-1을 분리한다. prediction JSON에는 `proposal_top1_raw`와 `proposal_top1_target_ar`를 모두 기록한다.
- explanation은 subject mode별 applicability ontology가 필요하다. object mode에서 headroom/lookroom/face/person cut은 known-not-applicable이며, 모델이 높은 class score를 내더라도 사용자-facing 산출물에서는 score를 숨긴다.
- applicability는 후처리 mask만으로 끝내지 않고 학습 target과 head를 추가한다. 그래야 모델이 mode-aware abstention을 학습하고, `not_applicable`과 단순 결측 `unlabeled`를 구분할 수 있다.

구현 변경은 다음과 같다.

| 파일 | 변경 |
| --- | --- |
| `src/mobilecropnet_v4/model.py` | fixed AR proposal을 `cx,cy,scale -> xyxy`로 재parameterization하고, 원본 이미지 content rect와 transform rounding을 반영한 letterbox-space AR 보정 및 `checklist_applicability_head`를 추가했다. |
| `src/mobilecropnet_v4/data.py` | `letterbox_content_box`, `EXPLANATION_APPLICABILITY_BY_MODE`, `checklist_applicability_target`, `why_tag_applicable`를 생성하고 class/detail/why valid mask에 반영했다. |
| `src/mobilecropnet_v4/eval_utils.py` | detailed checklist decode 시 mode-aware `not_applicable`/`unavailable` 표시, why tag mask, raw proposal top-1과 target-AR-selected proposal metric 분리, proposal AR metadata, explanation fidelity metric을 기록한다. |
| `src/scripts/visualize_mobilecropnet_v4_predictions.py` | orange box를 target-AR compatible proposal로 표시하고, model/label detail에서 mode-inapplicable score와 person-only tag를 숨긴다. |
| `src/scripts/compare_mobilecropnet_v4_methods.py` | `proposal_top1`과 `proposal_top1_target_ar` method를 분리했다. |
| `tests/test_mobilecropnet_v4.py` | AR-constrained proposal shape/ratio, letterbox content 내부 fixed-AR 보존, applicability output, object-mode person-term masking, end-to-end smoke를 검증한다. |

로컬 검증 결과는 다음과 같다.

| 검증 | 결과 |
| --- | --- |
| unit/smoke test | `/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m pytest tests/test_mobilecropnet_v4.py -q` 통과, 6 passed |
| real-label CPU smoke | `data/GAIC/All/.../gaic_personv6_server_v1_leftover_ignore_monotonic` 4 train row/2 val row 학습 통과 |
| fixed-AR content sanity | remote CPU에서 `16:9` proposal을 `480x160 -> input 64` rounded letterbox content box 기준으로 변환했을 때 max log error `2.0e-07` 기록 |
| replay eval smoke | GAIC-v2 official label smoke 8 row eval에서 `proposal_raw_top1_target_ar_compatible=1.0`, `proposal_top1_target_ar_compatible=1.0`, `explain_mode_consistency_violation_rate=0.0`, `explain_person_tag_on_object_rate=0.0` 기록 |
| PNG smoke | `artifacts/mobilecropnet_v4/local_smoke_ar_explain/viz/contact_sheet.png`, overlay PNG 생성 완료 |
| remote sync/compile | CPU 공유 스토리지 `/group-volume/users/jaden.ju/Sources/ImageCropping`에 코드 반영, `/usr/local/bin/python3 -m py_compile ...` 통과 |

원격 GPU 검증은 updated `AGENTS.md`의 workload 방식에 따라 모두 spot workload로 생성했다. `--allow-spot`을 사용했고 spot mode 규칙에 맞춰 `--quota-id`, `--group-pool-id`는 지정하지 않았다. Pending이 10분을 넘은 run은 없었다.

| run id | run name | profile | status | start UTC | end UTC | 목적 |
| ---: | --- | --- | --- | --- | --- | --- |
| `1094781` | `mcn-arx2-smoke-20260417-132531` | smoke | Succeeded | `2026-04-17T04:26:14Z` | `2026-04-17T04:28:10Z` | Train2636/Val200 label CUDA smoke, Test500 8-row replay eval, PNG smoke |
| `1094784` | `mcn-arx2-balanced-288-20260417-133107` | Balanced-288 | Succeeded | `2026-04-17T04:31:52Z` | `2026-04-17T04:54:56Z` | full train/eval/compare/PNG/official MOS |
| `1094785` | `mcn-arx2-hq-320-20260417-133107` | HQ-320 | Succeeded | `2026-04-17T04:31:51Z` | `2026-04-17T04:56:45Z` | full train/eval/compare/PNG/official MOS |
| `1094786` | `mcn-arx2-turbo-256-20260417-133107` | Turbo-256 | Succeeded | `2026-04-17T04:32:00Z` | `2026-04-17T04:58:50Z` | full train/eval/compare/PNG/official MOS |

`arx2` full replay/evaluation summary:

| profile | best epoch | val top1 hit | replay PCC | replay SRCC | replay NDCG@5 | raw proposal AR compatible | target-AR proposal compatible | mode violation | person tag on object |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced-288 | 1 | 0.4604 | 0.3532 | 0.3101 | 0.8570 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| HQ-320 | 3 | 0.4445 | 0.3684 | 0.3485 | 0.8636 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Turbo-256 | 6 | 0.4278 | 0.3479 | 0.3093 | 0.8553 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

`arx2` official GAIC MOS test 500장 결과:

| method | PCC | SRCC | Acc1/5 | Acc1/10 | top-1 MOS | MOS regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced-288 arx2 | 0.1855 | 0.2168 | 0.1140 | 0.2440 | 3.2947 | 0.9372 |
| HQ-320 arx2 | 0.3602 | 0.3377 | 0.3260 | 0.4900 | 3.6008 | 0.6311 |
| Turbo-256 arx2 | 0.4560 | 0.4469 | 0.3560 | 0.5420 | 3.7372 | 0.4947 |
| SSTK teacher official-Gc | 0.5146 | 0.4979 | 0.2700 | 0.4480 | 3.5985 | 0.6334 |

해석은 다음과 같다. AR/content-rect 보정은 proposal geometry와 시각화 혼동을 제거했고, object mode에서 person-only explanation이 출력되는 문제도 사용자-facing 산출물 기준으로 제거했다. 성능 측면에서는 Turbo-256이 official MOS top-return과 ranking correlation이 가장 좋고, teacher 대비 PCC/SRCC는 낮지만 Acc1/5, Acc1/10, top-1 MOS, MOS regret은 더 좋다. Balanced-288 arx2는 replay validation은 높지만 official MOS가 크게 낮아 selection mismatch의 반례로 기록한다.

정성 검증에서는 `artifacts/mobilecropnet_v4/gaic_v2_ar_explain/mcn-arx2-turbo-256-20260417-133107/viz_test/overlays/0001_392985_1_1.png`와 `0002_404126_3_4.png`를 확인했다. 두 sample 모두 `subject mode=object_single`이며 `headroom`, `lookroom`, `face_cut`, `joint_cut`은 `not_applicable`로 표시되고 score가 숨겨졌다. Proposal AR log error는 각각 `1.98e-08`, `1.04e-07`로 target AR에 맞는다.

## 7.6 SSTK-only Product Track top-return 재실험

사용자 목표 변경에 맞춰 이번 재실험은 `SSTK-only Product Track`만 고려했다. GAIC MOS annotation은 어떤 학습/선택/튜닝 단계에도 사용하지 않았고, GAIC v2 official test 500장은 선택된 checkpoint의 사후 benchmark 평가에만 사용했다. 목표는 public GAIC 계열 모델 전체를 모든 지표에서 이기는 것이 아니라, 상용 가능한 SSTK data factory label만으로 최소한 SSTK teacher에 근접하고 top-return 지표에서는 teacher보다 나은 product candidate를 찾는 것이다.

구현 변경은 다음과 같다.

- `compute_mobilecropnet_v4_loss`에 SSTK label 기반 `top_return_loss`를 추가했다. 이미지별 후보 중 high-score safe bag을 soft target으로 보고, 모델 top-1이 그 bag 안에 들어가도록 bag CE와 margin loss를 결합한다.
- 학습 metric에 `top_return_hit`, `top_return_score`, `top_return_regret`, `top_return_bag_size`를 추가했다.
- `train_mobilecropnet_v4.py`에 `top_return_weight`, `top_return_score_margin`, `top_return_logit_margin`, `top_return_temperature`와 `selection_metric=sstk_topreturn`을 추가했다.
- 제품 후보 탐색용 profile로 `q24_288_w075`, `rank_320`, `plus_384` preset을 추가했다. `plus_384`는 smoke 단계에서 pretrained initialization과 실행 비용이 커서 이번 full sweep에서는 제외하고, `rank_320` 안정화 이후 재검토 대상으로 남겼다.
- 실행 스크립트 `src/scripts/run_mobilecropnet_v4_sstk_product_experiment.sh`와 요약 스크립트 `src/scripts/summarize_mobilecropnet_v4_sstk_product_runs.py`를 추가했다.

원격 GPU workload는 updated `AGENTS.md` 규칙에 따라 `--allow-spot`, `--core-count=1` 단일 GPU run 여러 개로 생성했다. Pending이 10분을 넘은 초기 q24/turbo run은 종료한 뒤 spot run으로 재생성했다.

| run id | run name | profile | status | 목적 |
| ---: | --- | --- | --- | --- |
| `1094867` | `mcn-prod-q24-repro-20260417-160148` | q24_288_w075 | Succeeded | 기존 q24 계열을 GAIC-v2 SSTK label/현재 코드에서 재현 |
| `1094866` | `mcn-prod-rank320-topreturn-20260417-160148` | rank_320 | Succeeded | pretrained Conv-M + set-transformer + top-return product 후보 |
| `1094874` | `mcn-prod-q24-topret-b32-20260417-161228` | q24_288_w075 | Succeeded | q24 custom backbone에 top-return loss 추가 |
| `1094875` | `mcn-prod-turbo-q24-topret-b32-20260417-161228` | turbo_256 | Succeeded | Turbo backbone + q24 candidate/proposal + top-return loss |

SSTK validation/replay와 GAIC v2 official MOS 사후 평가는 다음과 같다. 표는 official MOS의 균형 성능 기준으로 정렬했다.

| run | profile | best epoch | val top-return hit | replay top1 hit | replay SRCC | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/10 | Accw4/10 | top-1 MOS | regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mcn-prod-rank320-topreturn-20260417-160148` | rank_320 | 8 | 0.410258 | 0.438739 | 0.331123 | 0.489461 | 0.462346 | 0.442000 | 0.632000 | 0.484000 | 0.376992 | 3.802640 | 0.429260 |
| 기존 q24 best `mcn-v4-q24-288-w075-20260416-095704` | q24_288_w075 계열 | 8 | - | 0.353600 | 0.614445 | 0.399840 | 0.368395 | 0.434000 | 0.630000 | 0.469000 | 0.365610 | 3.804520 | 0.427380 |
| SSTK teacher official-Gc | teacher | - | - | - | - | 0.514590 | 0.497884 | 0.270000 | 0.448000 | - | - | 3.598460 | 0.633440 |
| `mcn-prod-q24-repro-20260417-160148` | q24_288_w075 | 7 | 0.362243 | 0.366501 | 0.297057 | 0.464389 | 0.446759 | 0.380000 | 0.562000 | 0.459000 | 0.355726 | 3.733980 | 0.497920 |
| `mcn-prod-q24-topret-b32-20260417-161228` | q24_288_w075 | 7 | 0.391531 | 0.383499 | 0.293886 | 0.428227 | 0.407604 | 0.344000 | 0.546000 | 0.439500 | 0.337595 | 3.716140 | 0.515760 |
| `mcn-prod-turbo-q24-topret-b32-20260417-161228` | turbo_256 | 3 | 0.461468 | 0.500000 | 0.333784 | 0.402290 | 0.414060 | 0.286000 | 0.462000 | 0.396000 | 0.300709 | 3.633640 | 0.598260 |

해석은 다음과 같다.

- `rank_320`은 현재 SSTK-only Product Track의 1차 product candidate다. 기존 q24 best보다 PCC/SRCC, Acc1/5, Acc1/10, Acc4/10, Accw4/10이 모두 높고, top-1 MOS/MOS regret은 기존 best와 사실상 동률 수준이다.
- `rank_320`은 full-option SSTK teacher보다 PCC/SRCC는 낮지만, Acc1/5 `0.442000` vs `0.270000`, Acc1/10 `0.632000` vs `0.448000`, top-1 MOS `3.802640` vs `3.598460`, regret `0.429260` vs `0.633440`으로 top-return 품질은 명확히 더 좋다. Product Track 목표 기준으로는 teacher에 근접한 것이 아니라 teacher보다 나은 top-return student다.
- q24 custom backbone에 top-return loss만 추가한 run은 SSTK validation은 개선됐지만 official MOS는 악화됐다. 즉 top-return loss 자체가 충분 조건이 아니며, pretrained backbone과 후보 간 관계를 학습하는 ranker 구조가 같이 필요하다.
- `turbo_256` q24 top-return run은 SSTK validation/replay top-1이 가장 높지만 official MOS top-return이 낮다. 이는 SSTK pseudo top-return만 강하게 최적화하면 teacher-label shortcut에 과적합할 수 있음을 보여준다. 다음 selection metric은 `top_return_hit` 단독이 아니라 replay SRCC/PCC, utility regret, calibration, risk suppression, proposal pseudo-recall을 함께 보는 균형형이어야 한다.

GPU 사용률은 모두 단일 A100 기준 여유가 컸다. q24 custom b16은 평균 GPU util 약 `11.0%`, 최대 `21%`, memory max `715MB`; q24 custom b32는 평균 `13.0%`, 최대 `31%`, memory max `899MB`; turbo q24 b32는 평균 `9.5%`, 최대 `30%`, memory max `1,237MB`; rank320 b16은 평균 `20.7%`, 최대 `41%`, memory max `2,235MB`였다. 현재 규모에서는 2개 이상 GPU가 필요하지 않으며, 독립 단일 GPU spot workload를 병렬로 여러 개 실행하는 방식이 더 효율적이다.

## 7.7 rank_320 balanced-selection seed/ablation

위 7.6 결과를 바탕으로 `rank_320`을 중심으로 seed sweep, top-return loss weight ablation, batch ablation을 병렬 GPU workload로 수행했다. 목적은 단일 `sstk_topreturn` checkpoint가 우연히 좋은 seed인지 확인하고, SSTK-only selection이 top-return만 보다가 official MOS 일반화가 깨지는 문제를 줄이는 것이다.

구현 변경은 다음과 같다.

- `train_mobilecropnet_v4.py`에 `selection_metric=sstk_balanced_topreturn`을 추가했다. 이 metric은 `top_return_hit`, `top1_exact_best_score`, `top_return_score`, `top_return_regret`, pair/listwise quality, `risk_suppression_acc`, `proposal_recall_0_5`, `score_mae`, `top1_risk_rate`를 결합한다.
- val split에는 train pairwise label과 직접 매칭되는 `explicit_pairwise_count`가 0인 경우가 있어, corrected version에서는 pairwise 품질 항목을 `listwise_loss` 기반 `1/(1+loss)` fallback으로 대체했다. 따라서 GAIC MOS 없이도 validation selection score가 top-return과 listwise/ranking quality를 동시에 보게 된다.
- `tests/test_mobilecropnet_v4.py`에 `sstk_balanced_topreturn` selection score 단위 테스트를 추가했다.

GPU workload는 모두 updated `AGENTS.md` 규칙에 따라 `--allow-spot`, `--core-count=1`로 생성했다. `rank_320` batch16 run은 평균 GPU util `15~24%`, 최대 `34~42%`, memory max 약 `2,235MB`였고, batch32 run도 memory max 약 `3,429MB`였다. 따라서 현재 모델 크기에서는 멀티 GPU가 필요하지 않으며, 독립 단일 GPU workload 여러 개를 병렬 실행하는 방식이 맞다.

| run id | run name | 설정 | status | 해석 |
| ---: | --- | --- | --- | --- |
| `1094933` | `mcn-prod-rank320-bal-s17-20260417-170831` | seed 20260417, `top_return_weight=0.8`, b16 | Succeeded | seed17 baseline |
| `1094934` | `mcn-prod-rank320-bal-s18-20260417-170831` | seed 20260418, `top_return_weight=0.8`, b16 | Succeeded | top-return best |
| `1094936` | `mcn-prod-rank320-bal-s19-20260417-170831` | seed 20260419, `top_return_weight=0.8`, b16 | Succeeded | seed variance 확인 |
| `1094931` | `mcn-prod-rank320-bal-tr04-20260417-170831` | seed 20260417, `top_return_weight=0.4`, b16 | Succeeded | top-return weight down |
| `1094932` | `mcn-prod-rank320-bal-tr06-20260417-170831` | seed 20260417, `top_return_weight=0.6`, b16 | Succeeded | balanced product candidate |
| `1094930` | `mcn-prod-rank320-bal-b32-20260417-170831` | seed 20260417, `top_return_weight=0.8`, b32 | Succeeded | batch size ablation |
| `1094963` | `mcn-prod-rank320-bal2-s17-20260417-172640` | corrected selection, seed17, `top_return_weight=0.8`, b16 | Preempted | spot preemption |
| `1095057` | `mcn-prod-rank320-bal2-s17-r2-20260417-173500` | corrected selection 재시도 | Succeeded | corrected metric smoke/full 재현 |

SSTK validation/replay와 GAIC v2 official MOS 사후 평가 결과는 다음과 같다. 표는 top-1 MOS와 regret 중심으로 정렬했다.

| run | best epoch | sel | val top-return hit | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/10 | Accw4/10 | top-1 MOS | regret | replay top1 | utility regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rank320-bal-s18` | 2 | 0.442638 | 0.440948 | 0.453267 | 0.428847 | 0.442000 | 0.646000 | 0.490500 | 0.384303 | 3.818520 | 0.413380 | 0.478045 | 0.201140 |
| `rank320-bal-tr06` | 6 | 0.443560 | 0.432456 | 0.489133 | 0.469995 | 0.434000 | 0.640000 | 0.484000 | 0.376283 | 3.812800 | 0.419100 | 0.448300 | 0.212436 |
| `rank320-bal-s19` | 5 | 0.446378 | 0.444644 | 0.473473 | 0.447829 | 0.434000 | 0.638000 | 0.488000 | 0.379807 | 3.805240 | 0.426660 | 0.462819 | 0.206065 |
| `rank320-topreturn` | 8 | 0.435391 | 0.410258 | 0.489461 | 0.462346 | 0.442000 | 0.632000 | 0.484000 | 0.376992 | 3.802640 | 0.429260 | 0.438739 | 0.212444 |
| `rank320-bal-s17` | 7 | 0.431781 | 0.410258 | 0.482659 | 0.455338 | 0.442000 | 0.632000 | 0.477000 | 0.370398 | 3.802200 | 0.429700 | 0.437323 | 0.216209 |
| `rank320-bal2-s17-r2` | 7 | 0.500668 | 0.410258 | 0.482659 | 0.455338 | 0.442000 | 0.632000 | 0.477000 | 0.370398 | 3.802200 | 0.429700 | 0.437323 | 0.216209 |
| `rank320-bal-tr04` | 6 | 0.437961 | 0.428392 | 0.498876 | 0.474039 | 0.420000 | 0.612000 | 0.471500 | 0.366638 | 3.777220 | 0.454680 | 0.459278 | 0.207709 |
| `rank320-bal-b32` | 5 | 0.440256 | 0.413710 | 0.539392 | 0.510248 | 0.422000 | 0.582000 | 0.453500 | 0.350436 | 3.732960 | 0.498940 | 0.450779 | 0.214617 |

해석은 다음과 같다.

- seed sweep에서는 `top_return_weight=0.8`, batch16의 세 seed가 모두 기존 q24 best와 동률 또는 그 이상 수준의 top-return을 보였다. 특히 seed18은 top-1 MOS `3.818520`, regret `0.413380`으로 7.7 시점의 top-return best였다.
- `top_return_weight=0.6`은 7.7 시점의 top-return best보다는 약간 낮지만 PCC/SRCC `0.489133`/`0.469995`와 Acc1/10 `0.640000`을 동시에 유지했다. 따라서 7.7 시점에는 단일 product candidate로 `rank320-bal-tr06`이 가장 안전했다.
- `top_return_weight=0.4`는 PCC/SRCC는 높지만 Acc1/N과 top-1 MOS가 낮아 product track 기준에서는 underweight다.
- batch32는 PCC/SRCC `0.539392`/`0.510248`로 teacher correlation을 넘었지만 Acc1/10 `0.582000`, top-1 MOS `3.732960`, regret `0.498940`으로 top-return이 무너졌다. 즉 더 큰 batch가 ranking correlation을 좋게 만들 수는 있지만, 제품에서 하나의 crop을 반환하는 목적에는 맞지 않을 수 있다.
- corrected `sstk_balanced_topreturn` fallback run(`rank320-bal2-s17-r2`)은 seed17에서 초기 balanced run과 같은 checkpoint를 선택했다. selection score scale은 바뀌었지만 output은 동일하므로 regression은 없었다. 다만 최종 결론을 더 견고하게 만들려면 corrected metric으로 `top_return_weight=0.6`과 seed18/19를 한 번 더 반복하는 것이 다음 1순위다.

7.7 시점의 추천은 top-1 MOS/regret 데모 후보 `mcn-prod-rank320-bal-s18-20260417-170831`, 균형형 후보 `mcn-prod-rank320-bal-tr06-20260417-170831`였다. 7.8의 corrected sweep 이후에는 이 둘을 기준선으로 내리고, top-return best와 균형형 후보를 새 결과로 갱신한다.

## 7.8 T6 converter 보정 및 corrected Product Track 확장 sweep

7.7의 결론에서 남겨둔 다음 연구를 실제로 수행했다. 범위는 corrected `sstk_balanced_topreturn` 기준 `top_return_weight=0.6` seed sweep, `0.55/0.65/0.7` 좁은 loss ablation, LR/WD/warmup ablation, 352/384 해상도 확장, q24/turbo balanced-selection 재평가, T6-score label converter 보정 및 diagnostic 학습이다. 모든 SSTK-only product run은 GAIC MOS annotation을 학습/선택에 쓰지 않았고, 선택된 checkpoint만 GAIC v2 official test 500장에서 사후 평가했다.

### 7.8.1 T6-score label converter 보정

T6-score label은 GAIC official MOS로 학습한 deep ranker에서 나온 diagnostic/upper-bound 성격의 label이므로 `SSTK-only Product Track`과 분리한다. 다만 converter 안정성, fatal/hard-negative 처리, fixed-AR 확장 가능성 검증에는 유용하다.

초기 strict 보정안 `corrected_v2`는 `contradiction_flag`까지 hard-unsafe로 처리해 train/val/test 보존 이미지가 `2,195/168/415`까지 줄었다. 이 방식은 label/explanation 불일치를 실제 fatal crop과 동일하게 제거해 데이터 손실이 과도하다고 판단했다. 최종 채택한 `corrected_v2b`는 `fatal_flag=true` 또는 `label_quality_tier=hard_negative`만 hard-unsafe로 보고, contradiction은 낮은 confidence/weight 신호로 유지한다.

구현은 `src/scripts/convert_t6_score_labels_to_mobilecropnet_v4_batch.py`에 있다. 핵심 로직은 다음과 같다.

| 단계 | corrected_v2b 동작 | 이유 |
| --- | --- | --- |
| unsafe 판정 | `fatal_flag=true` 또는 `label_quality_tier=hard_negative`를 hard unsafe로 본다. `contradiction_flag`는 기본값에서 unsafe로 보지 않고, `--unsafe_include_contradiction`을 켰을 때만 hard unsafe에 포함한다. | contradiction은 score와 explanation의 불일치 신호일 수 있지만, 항상 crop 품질 fatal은 아니다. 이를 제거하면 학습 이미지가 과도하게 줄어든다. |
| positive 선택 | `--positive_selection_policy best_safe` 기준으로 unsafe가 아닌 후보 중 adjusted rank score가 가장 높은 crop을 `matching_targets` positive로 둔다. | T6 ranker top-1이 fatal/hard-negative일 때 그대로 positive가 되는 문제를 막는다. |
| score 보정 | 기본 `--score_adjustment_policy safe_rescale`은 safe 후보끼리 image-local rank를 0~1로 다시 매기고, unsafe 후보는 `--unsafe_score_cap 0.05` 이하로 cap한다. | score scale은 유지하되 fatal/hard-negative top crop이 rank target 상단을 차지하지 못하게 한다. |
| candidate weight | 원본 T6 `training_weight`를 `candidate_weight`와 `score_targets.t6_training_weight`에 보존하고, unsafe negative는 hard-negative 학습에 충분히 보이도록 weight floor를 둔다. | low-confidence/contradiction 후보는 약하게, 명확한 unsafe negative는 억제 대상으로 학습한다. |
| metadata 보존 | `candidate.t6_teacher`에 `ranker_score`, raw/adjusted rank pct, `fatal_flag`, `contradiction_flag`, `unsafe_after_converter`, `label_quality_tier`, warnings, checklist를 저장한다. | downstream 학습, 시각화, failure audit에서 원본 T6 판단과 converter 보정 결과를 함께 추적한다. |
| pair/listwise 생성 | adjusted score rank pct와 margin으로 pairwise/listwise row를 만든다. hard-unsafe는 높은 score positive와 pair를 이루면 negative 방향으로 들어간다. | MobileCropNet utility head가 T6의 정렬된 ranking 신호를 학습하되, unsafe top-1 오류를 반복하지 않게 한다. |

이 설계 때문에 `corrected_v2b`는 T6 score-only를 그대로 모사하지 않는다. 정확한 목적은 T6 ranker가 주는 강한 MOS ranking 신호를 유지하면서, explanation/fatal audit이 위험하다고 표시한 top crop을 downstream MobileCropNet positive로 승격하지 않는 것이다.

| split | source images | v2 rows | v2b rows | v2b candidates | pairwise rows | listwise rows | 채택 판단 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Train2636 | 2,636 | 2,195 | 2,509 | 216,716 | 240,864 | 2,509 | v2b 채택 |
| Val200 | 200 | 168 | 189 | 16,240 | 18,144 | 189 | v2b 채택 |
| Test500 | 500 | 415 | 475 | 40,923 | 45,600 | 475 | v2b 채택 |

정성 분석용 fatal/hard-negative top-1 debug visualization도 생성했다. T6 top-1이 fatal/hard-negative인 이미지는 train `138`, val `12`, test `27`이고, `artifacts/mobilecropnet_v4/teacher_improvement/t6_top_conflicts_debug_260417`에 overlay/selected crop/manifest를 저장했다. 전체 `119`개 이미지를 시각화했으며, sample들의 selected top crop은 모두 `selected_tier=hard_negative`, `selected_fatal_flag=true`, `selected_contradiction_flag=true`였다.

### 7.8.2 fixed-AR T6 scoring feasibility

FREE-only/candidate-distribution 문제를 줄이기 위해 T6 ranker를 `1:1`, `3:4`, `4:3`, `16:9`, `9:16` fixed-AR static crop에도 적용할 수 있는 경로를 구현했다.

- `src/scripts/export_gaic_fixed_ar_static_candidates.py`: GAIC annotation 이미지에서 target AR별 static candidate row 생성
- `src/scripts/cache_gaic_candidate_timm_roi_features.py`: candidate row에 대한 timm ROI feature cache 생성
- `src/scripts/export_t6_fixed_ar_candidate_scores.py`: T6 deep ranker bundle로 `(image_id, target_ar)` 그룹별 score/rank 산출
- `src/mobilecropnet/bank.py`: fixed-AR full/max-area candidate가 원본 target AR을 위반하던 geometry bug를 수정

bank-fix smoke는 val 2장, target AR 5개, static_k 4 기준 40 row/10 group으로 수행했다. feature cache shape은 `[40,1920]`이고, T6 scoring summary는 `row_count=40`, `group_count=10`, `group_keys=[image_id,target_ar]`로 정상 생성됐다. 따라서 T6 ranker를 fixed-AR proposal/ranking label 생성에 적용하는 것은 가능하다. 단 product training에는 GAIC-MOS-trained T6 label을 직접 쓰지 않으며, full production label을 만들려면 static bank를 더 다양화하고 Train/Val/Test 전체에 feature cache/scoring을 확장해야 한다.

### 7.8.3 GPU workload 실행

모든 GPU workload는 updated `AGENTS.md` 규칙에 맞춰 `--allow-spot`, `--core-count=1`로 생성했다. `convnextv2_tiny` 기반 `plus384` run `1095153`은 remote local pretrained weight가 없어 중단했고, 같은 목적은 `MobileNetV4-Hybrid-M` 기반 `hybrid384` run `1095171`로 대체했다. 현재 모델 규모에서는 2개 이상 GPU가 필요하지 않다. `rank_320` 계열은 train 평균 GPU utilization이 대체로 `10~26%`, memory max 약 `2.2GB`이고, `hybrid384`도 평균 약 `30%`, memory max 약 `3.0GB` 수준이었다.

| run id | run | 설정 | status |
| ---: | --- | --- | --- |
| `1095142` | `mcn-next-r320-tr06-s17-20260417-191315` | seed17, `top_return_weight=0.6` | Succeeded |
| `1095143` | `mcn-next-r320-tr06-s18-20260417-191315` | seed18, `top_return_weight=0.6` | Succeeded |
| `1095144` | `mcn-next-r320-tr06-s19-20260417-191315` | seed19, `top_return_weight=0.6` | Succeeded |
| `1095145` | `mcn-next-r320-tr055-s17-20260417-191315` | `top_return_weight=0.55` | Succeeded |
| `1095146` | `mcn-next-r320-tr065-s17-20260417-191315` | `top_return_weight=0.65` | Succeeded |
| `1095147` | `mcn-next-r320-tr070-s17-20260417-191315` | `top_return_weight=0.7` | Succeeded |
| `1095148` | `mcn-next-r320-lr1e4-s17-20260417-191315` | LR `1e-4` | Succeeded |
| `1095149` | `mcn-next-r320-lr2e4-s17-20260417-191315` | LR `2e-4` | Succeeded |
| `1095150` | `mcn-next-r320-wd5e4-wu2-s17-20260417-191315` | WD `5e-4`, warmup 2 | Succeeded |
| `1095151` | `mcn-next-r320-wd1e3-wu2-s17-20260417-191315` | WD `1e-3`, warmup 2 | Succeeded |
| `1095152` | `mcn-next-r352-tr06-s17-20260417-191315` | input 352, k/q 40 | Succeeded |
| `1095153` | `mcn-next-plus384-tr06-s17-20260417-191315` | ConvNeXtV2-Tiny 계획 | Terminated, pretrained weight 부재 |
| `1095171` | `mcn-next-hybrid384-tr06-s17-20260417-191315` | input 384, Hybrid-M, k/q 48 | Succeeded |
| `1095175` | `mcn-next-q24bal-tr06-s17-20260417-191315` | q24 corrected balanced 재평가 | Succeeded |
| `1095176` | `mcn-next-turbobal-tr06-s17-20260417-191315` | turbo corrected balanced 재평가 | Succeeded |
| `1095157` | `mcn-t6safe-r320-tr06-s17-r2-20260417-191315` | T6 diagnostic seed17 재시도 | Succeeded |
| `1095155` | `mcn-t6safe-r320-tr06-s18-20260417-191315` | T6 diagnostic seed18 | Succeeded |
| `1095169` | `t6-bundle-fixedar-smoke-20260417-102803` | T6 bundle/fixed-AR smoke | Succeeded |

### 7.8.4 SSTK-only Product Track 결과

아래 표는 GAIC v2 official test 500장 사후 평가 결과다. 표는 top-1 MOS/regret 중심으로 정렬했다.

| run | profile | best ep | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/10 | Accw4/10 | top-1 MOS | regret | 해석 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `r320-tr070-s17` | rank_320 | 7 | 0.370447 | 0.343425 | 0.444000 | 0.650000 | 0.461000 | 0.355970 | 3.822800 | 0.409100 | top-return tie, correlation 낮음 |
| `hybrid384-tr06-s17` | hybrid384 | 2 | 0.447673 | 0.427922 | 0.444000 | 0.650000 | 0.464500 | 0.360553 | 3.822800 | 0.409100 | top-return best, 비용 증가 |
| `r320-tr06-s19` | rank_320 | 5 | 0.435216 | 0.405202 | 0.444000 | 0.646000 | 0.491000 | 0.383345 | 3.821640 | 0.410260 | 균형형 product candidate |
| `r320-wd5e4-wu2-s17` | rank_320 | 7 | 0.411973 | 0.390361 | 0.444000 | 0.648000 | 0.452000 | 0.344572 | 3.820800 | 0.411100 | top-return 양호, Accw 낮음 |
| `r320-tr065-s17` | rank_320 | 6 | 0.396026 | 0.372507 | 0.440000 | 0.646000 | 0.471000 | 0.363160 | 3.819920 | 0.411980 | top-return 양호 |
| `r352-tr06-s17` | 352 확장 | 6 | 0.370613 | 0.352621 | 0.438000 | 0.646000 | 0.443000 | 0.334288 | 3.819120 | 0.412780 | 해상도 확장 단독 이득 제한 |
| `r320-lr1e4-s17` | rank_320 | 7 | 0.485484 | 0.461861 | 0.438000 | 0.642000 | 0.482000 | 0.376490 | 3.818940 | 0.412960 | correlation 가장 안정적인 축 |
| `r320-lr2e4-s17` | rank_320 | 8 | 0.413184 | 0.392426 | 0.440000 | 0.640000 | 0.477000 | 0.369409 | 3.810900 | 0.421000 | LR 상향 이득 제한 |
| `r320-tr055-s17` | rank_320 | 6 | 0.394867 | 0.370374 | 0.430000 | 0.636000 | 0.483500 | 0.376055 | 3.809040 | 0.422860 | top-return underweight |
| `r320-tr06-s18` | rank_320 | 2 | 0.420089 | 0.400224 | 0.426000 | 0.638000 | 0.458500 | 0.353358 | 3.806560 | 0.425340 | SSTK val score 과대평가 사례 |
| `r320-wd1e3-wu2-s17` | rank_320 | 7 | 0.407270 | 0.378565 | 0.434000 | 0.634000 | 0.465000 | 0.359061 | 3.804480 | 0.427420 | WD 과대 |
| `r320-tr06-s17` | rank_320 | 6 | 0.386102 | 0.373247 | 0.434000 | 0.636000 | 0.444500 | 0.340958 | 3.798060 | 0.433840 | seed variance |
| `q24bal-tr06-s17` | q24_288 | 7 | 0.428481 | 0.410941 | 0.302000 | 0.500000 | 0.398500 | 0.301356 | 3.676320 | 0.555580 | product 후보 제외 |
| `turbobal-tr06-s17` | turbo_256 | 2 | 0.325528 | 0.343198 | 0.316000 | 0.476000 | 0.372000 | 0.281936 | 3.624940 | 0.606960 | product 후보 제외 |
| `sstk_teacher_fullopt_largecap_v2_raw` | teacher | - | 0.514590 | 0.497884 | 0.270000 | 0.448000 | 0.378000 | 0.285575 | 3.598460 | 0.633440 | data-curation teacher exact official crop score |

해석은 다음과 같다.

- `hybrid384`와 `rank_320 tr070`은 top-1 MOS/regret, Acc1/10 기준 최고지만, `tr070`은 PCC/SRCC가 낮다. top-return demo 후보는 `hybrid384`, 단말 비용과 모델 크기를 고려한 보조 top-return 후보는 `tr070`이다.
- 균형형 product candidate는 `rank_320 tr06-s19`다. top-1 MOS는 `hybrid384`보다 `0.00116` 낮지만 Acc4/10 `0.491000`, Accw4/10 `0.383345`가 이번 SSTK-only sweep 내 최상위권이고, 기존 `rank320-bal-s18`의 top-1 MOS `3.818520`, regret `0.413380`을 넘어섰다.
- `lr1e4`는 PCC/SRCC `0.485484`/`0.461861`로 가장 안정적인 correlation 축이다. 그러나 top-return은 `tr06-s19`와 `hybrid384`보다 낮으므로 단독 배포 후보보다는 ranking calibration 개선의 참고 설정으로 본다.
- q24/turbo는 corrected balanced selection에서도 official MOS top-return이 낮다. 따라서 q24/turbo의 높은 replay/validation top-return은 이번에도 official MOS 일반화로 이어지지 않았고, product 후보군에서 제외한다.

### 7.8.5 T6 diagnostic track

T6-score label은 GAIC official MOS로 학습된 ranker를 거치므로 제품 후보가 아니라 diagnostic upper-bound로만 해석한다. 그래도 fatal top demotion과 safe-positive selection을 적용하면 MobileCropNet 구조 자체의 상한을 볼 수 있다.

| run | best ep | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/10 | Accw4/10 | top-1 MOS | regret | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `t6_teacher_deep_crop_ranker_score_only` | 2 | 0.735179 | 0.709681 | 0.468000 | 0.638000 | 0.571000 | 0.462737 | 3.842160 | 0.389740 | GAIC MOS-trained data-curation teacher, deployable student 아님 |
| `t6safe-r320-tr06-s18` | 7 | 0.616310 | 0.589892 | 0.456000 | 0.656000 | 0.538500 | 0.430915 | 3.830220 | 0.401680 | diagnostic upper-bound |
| `t6safe-r320-tr06-s17-r2` | 8 | 0.514091 | 0.545536 | 0.446000 | 0.654000 | 0.531500 | 0.419104 | 3.826960 | 0.404940 | diagnostic upper-bound |
| `sstk_teacher_fullopt_largecap_v2_raw` | - | 0.514590 | 0.497884 | 0.270000 | 0.448000 | 0.378000 | 0.285575 | 3.598460 | 0.633440 | SSTK-only data-curation teacher baseline |

T6 diagnostic student는 SSTK-only best보다 PCC/SRCC, Acc4/10, Accw4/10이 크게 높다. 이는 모델 구조의 용량 문제가 아니라 label/selection 신호가 official MOS preference와 얼마나 정렬되는지가 병목임을 시사한다. 다만 이 경로는 GAIC MOS가 teacher ranker 학습에 사용되므로 상용 SSTK-only product 결론에는 포함하지 않는다. 또한 `t6_teacher_deep_crop_ranker_score_only` 자체는 MobileCropNet student가 아니라 GAIC MOS로 학습된 data-curation teacher이므로, 이 row는 downstream student가 아직 teacher score ordering을 완전히 흡수하지 못했다는 상한 진단으로만 사용한다.

### 7.8.6 Primary metric 기준 best 모델 판정

공식 GAIC MOS primary metric을 `PCC`, `SRCC`, `Acc1/5`, `Acc1/10`, `Acc4/10`, `Accw4/10`으로 제한해 보면 top-return-only 결론보다 해석이 더 세분화된다.

| track | metric | best measured MobileCropNet | value | 해석 |
| --- | --- | --- | ---: | --- |
| SSTK-only Product | PCC | `r320-lr1e4-s17` | 0.485484 | teacher PCC `0.514590`에 가장 근접한 product student |
| SSTK-only Product | SRCC | `r320-lr1e4-s17` | 0.461861 | 후보 전체 순서 상관이 가장 안정적 |
| SSTK-only Product | Acc1/5 | `r320-tr070-s17`, `hybrid384-tr06-s17`, `r320-tr06-s19`, `r320-wd5e4-wu2-s17` tie | 0.444000 | top-1이 MOS top-5에 드는 비율은 네 run이 동률 |
| SSTK-only Product | Acc1/10 | `r320-tr070-s17`, `hybrid384-tr06-s17` tie | 0.650000 | `hybrid384`가 동률 중 PCC/SRCC가 높아 더 안전한 top-return demo 후보 |
| SSTK-only Product | Acc4/10 | `r320-tr06-s19` | 0.491000 | top-4 반환 후보군의 coverage가 가장 좋음 |
| SSTK-only Product | Accw4/10 | `r320-tr06-s19` | 0.383345 | top-4 안에서 더 높은 MOS rank를 앞쪽에 두는 능력이 가장 좋음 |
| T6 diagnostic student | PCC | `t6safe-r320-tr06-s18` | 0.616310 | measured MobileCropNet student 중 최고 |
| T6 diagnostic student | SRCC | `t6safe-r320-tr06-s18` | 0.589892 | measured MobileCropNet student 중 최고 |
| T6 diagnostic student | Acc1/5 | `t6safe-r320-tr06-s18` | 0.456000 | measured MobileCropNet student 중 최고 |
| T6 diagnostic student | Acc1/10 | `t6safe-r320-tr06-s18` | 0.656000 | measured MobileCropNet student 중 최고 |
| T6 diagnostic student | Acc4/10 | `t6safe-r320-tr06-s18` | 0.538500 | measured MobileCropNet student 중 최고 |
| T6 diagnostic student | Accw4/10 | `t6safe-r320-tr06-s18` | 0.430915 | measured MobileCropNet student 중 최고 |

따라서 현재 측정된 MobileCropNet run만 놓고 보면 T6 diagnostic track의 `t6safe-r320-tr06-s18`이 여섯 primary metric 모두에서 가장 높은 student다. 단 이 결론은 "SSTK-only product best"가 아니라 "GAIC MOS-trained T6 teacher를 사용했을 때 MobileCropNet 구조가 도달 가능한 진단 상한"이다. 상용 SSTK-only Product Track에서는 `r320-lr1e4-s17`이 correlation 축, `r320-tr06-s19`가 Acc4/10/Accw4/10과 top-return 균형 축, `hybrid384-tr06-s17`이 top-return demo 축이다.

7.8.5에는 T6 `hybrid384` 결과가 없다. 따라서 "T6 label을 사용하면 어떤 MobileCropNet 구조가 최상인가"를 완전히 결론내리려면 `t6safe-hybrid384-tr06` 추가 실험이 필요하다. 다만 이 실험은 GAIC MOS-trained T6 teacher를 쓰는 diagnostic track이므로, SSTK-only product 후보 선정에는 필수 실험이 아니다. 목적이 구조 상한 분석이면 수행하고, 목적이 상용 SSTK-only 모델 선택이면 우선순위는 `rank_320`/`hybrid384` SSTK-only label 개선과 selection composite 개선이다.

### 7.8.7 Top-return 의미와 primary metric 우선 설계

본 보고서에서 top-return은 모델이 가장 높은 score로 선택한 top-1 crop을 실제로 반환한다고 가정했을 때의 official MOS 품질을 뜻한다. 수치로는 `top1 MOS`가 높고 `MOS regret`이 낮을수록 좋다. `Acc1/5`와 `Acc1/10`도 top-return 계열이다. 각각 모델 top-1 crop이 official MOS top-5 또는 top-10 후보 안에 드는 이미지 비율이다.

Top-return은 제품에서 한 장의 crop만 반환하는 상황을 직접 반영하지만, 후보 전체 ordering 품질과는 다를 수 있다. 예를 들어 `r320-tr070-s17`은 top-1 MOS와 Acc1/10이 최고권이지만 PCC/SRCC는 낮다. 반대로 batch32 계열은 PCC/SRCC가 높아졌지만 top-1 MOS가 무너졌다. 따라서 `PCC`, `SRCC`, `Acc1/5`, `Acc1/10`, `Acc4/10`, `Accw4/10`을 최우선으로 삼으려면 top-return surrogate만 키우는 방식은 맞지 않는다.

권장 학습/선택 방식은 다음이다.

| 목표 | 필요한 설계 | 피해야 할 설정 |
| --- | --- | --- |
| PCC/SRCC 개선 | official crop 후보처럼 다양한 candidate 전체에 utility score를 안정적으로 매기는 candidate scorer, confidence-weighted pairwise/listwise loss, score calibration loss, augmentation consistency | top-1 positive bag만 강하게 맞추고 하위/중위 후보 ordering을 방치하는 설정 |
| Acc1/5, Acc1/10 유지 | safe high-score positive bag, destructive negative suppression, moderate top-return loss | `top_return_weight`를 과도하게 키워 correlation이 무너지는 설정 |
| Acc4/10, Accw4/10 개선 | top-4 반환 후보 전체가 MOS top-10 안에 들도록 listwise top-N coverage proxy와 diversity-aware reranking을 학습/selection에 포함 | top-1만 좋아지고 2~4위 후보가 불안정한 설정 |
| proposal/candidate 결합 | proposal head는 official metric에 직접 쓰지 않되, generated proposal이 high-score safe candidate를 충분히 회수하도록 proposal recall과 pseudo-positive bag loss를 유지 | proposal top-1을 official candidate-ranking metric처럼 해석하는 것 |
| checkpoint 선택 | SSTK-only validation composite에 ranking correlation proxy, NDCG, pseudo Acc1/N, pseudo Acc4/N, utility regret, risk violation, latency를 함께 넣음 | validation top-return hit 하나만으로 선택 |

현재 결과 기준으로 여섯 primary metric을 동시에 노리려면 `r320-lr1e4-s17`의 correlation 안정성과 `r320-tr06-s19`의 Acc4/10/Accw4/10 강점을 합치는 방향이 가장 타당하다. 즉, `top_return_weight=0.6` 전후를 유지하되 pairwise/listwise confidence weighting, calibration, top-4 listwise return proxy를 강화하고, selection score는 top-1 MOS proxy보다 `SRCC/PCC proxy + Acc1/N proxy + Acc4/N proxy + regret + risk`의 균형형으로 바꾼다. `hybrid384`는 top-return demo와 capacity 상한 확인에는 유효하지만, Acc4/10/Accw4/10이 `r320-tr06-s19`보다 낮으므로 단순 고해상도 확장만으로 primary metric 전체가 좋아진다고 보기는 어렵다.

### 7.8.8 A100 inference time

제품 후보 및 diagnostic 후보의 서버 GPU latency를 같은 환경에서 재측정했다. 실행은 MLP spot workload `1095938/mcn-v4-latency-a100-20260420-114321`에서 수행했고, 장비는 `NVIDIA A100-SXM4-80GB`, torch `2.5.1+cu121`, AMP fp16, batch size 1, warmup 20회, 측정 80회다. 측정 스크립트는 `src/scripts/benchmark_mobilecropnet_v4_inference.py`이며 결과는 `artifacts/mobilecropnet_v4/inference_latency/a100_mcn_v4_primary_20260420_114321.json` 및 `.md`에 저장했다.

주의할 점은 이 수치가 synthetic tensor 기반 model forward latency라는 것이다. 이미지 decode, resize/letterbox, candidate generation, JSON serialization, visualization, disk I/O는 포함하지 않는다. `auto` 후보 수는 checkpoint의 학습 candidate_k이고, `86` 후보 수는 GAIC v2 test official annotation 평균 후보 수에 가까운 proxy다.

| model | profile | input | params M | auto K | auto mean/p95 ms | K=86 mean/p95 ms | peak mem MB auto/86 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r320-tr070-s17` | rank_320 | 320 | 9.710 | 32 | 19.775 / 25.718 | 18.025 / 18.192 | 179.1 / 197.0 |
| `hybrid384-tr06-s17` | hybrid384 | 384 | 13.256 | 48 | 29.104 / 32.400 | 28.461 / 29.589 | 244.6 / 261.5 |
| `r320-tr06-s19` | rank_320 | 320 | 9.710 | 32 | 18.046 / 18.621 | 18.270 / 19.899 | 179.1 / 197.0 |
| `r320-wd5e4-wu2-s17` | rank_320 | 320 | 9.710 | 32 | 18.467 / 21.007 | 18.218 / 19.154 | 179.3 / 196.6 |
| `r320-tr065-s17` | rank_320 | 320 | 9.710 | 32 | 18.219 / 19.656 | 18.094 / 19.348 | 179.7 / 197.5 |
| `r352-tr06-s17` | rank_352 | 352 | 10.373 | 40 | 18.572 / 19.244 | 18.201 / 18.434 | 194.3 / 212.4 |
| `r320-lr1e4-s17` | rank_320 | 320 | 9.710 | 32 | 18.200 / 19.044 | 18.357 / 19.114 | 179.7 / 197.5 |
| `r320-lr2e4-s17` | rank_320 | 320 | 9.710 | 32 | 18.290 / 19.057 | 17.919 / 18.005 | 179.6 / 196.9 |
| `r320-tr055-s17` | rank_320 | 320 | 9.710 | 32 | 18.764 / 21.852 | 18.336 / 20.422 | 179.8 / 197.2 |
| `r320-tr06-s18` | rank_320 | 320 | 9.710 | 32 | 17.945 / 18.032 | 18.248 / 18.639 | 179.7 / 197.4 |
| `r320-wd1e3-wu2-s17` | rank_320 | 320 | 9.710 | 32 | 17.880 / 18.153 | 17.932 / 18.330 | 179.1 / 196.3 |
| `r320-tr06-s17` | rank_320 | 320 | 9.710 | 32 | 18.145 / 19.447 | 18.307 / 21.215 | 180.3 / 197.6 |
| `q24bal-tr06-s17` | q24_288 | 288 | 0.252 | 24 | 7.195 / 7.304 | 7.233 / 7.417 | 15.5 / 28.6 |
| `turbobal-tr06-s17` | turbo_256 | 256 | 2.256 | 16 | 12.636 / 12.946 | 12.310 / 12.498 | 49.3 / 63.5 |
| `t6safe-r320-tr06-s18` | T6 rank_320 | 320 | 9.710 | 32 | 17.943 / 18.155 | 17.968 / 18.249 | 179.4 / 196.4 |
| `t6safe-r320-tr06-s17-r2` | T6 rank_320 | 320 | 9.710 | 32 | 18.010 / 18.496 | 18.114 / 18.826 | 180.1 / 197.1 |

해석은 다음과 같다.

- q24_288은 약 `7.2ms`, turbo_256은 약 `12.3ms`, rank_320 계열은 대부분 약 `18ms`, hybrid384는 약 `28-29ms`다.
- A100에서는 candidate 수를 `auto`에서 `86`으로 늘려도 latency 증가가 작다. 현재 구조에서는 candidate token scoring보다 backbone/feature extraction이 지배적이기 때문이다.
- `hybrid384`는 top-return demo 후보지만 rank_320 대비 latency가 약 `1.6x`다. 서버 GPU에서는 여전히 가볍지만, mobile/NPU export 목표라면 별도 on-device latency 측정이 필요하다.
- T6 diagnostic rank_320 student도 구조가 동일하므로 latency는 SSTK-only rank_320과 사실상 같다. 성능 차이는 추론 비용이 아니라 label/selection 신호 차이에서 온다.

### 7.9 GAIC public cropper score + SSTK explanation label diagnostic track

사용자 제안에 따라 T6 lightweight set ranker score 대신 GAIC public cropper score를 crop score teacher로 사용하는 별도 track을 구현하고 검증했다. 2026-04-20 재정리 이후 이 track은 `SSTK-only Product Track`도 `Product AR cropper`도 아니다. 목적은 "crop preference score는 GAIC public cropper가 제공하고, subject mode, checklist/explanation, fatal/safety audit은 SSTK data factory가 제공하는 방식"으로 학습 라벨 큐레이션과 benchmark scorer teacher를 얼마나 개선할 수 있는지 분석하는 것이다. 과거 `pubscore`/`pubdistill` student 수치는 teacher score를 MobileCropNet 구조로 압축한 진단값으로만 남긴다.

구현 변경은 다음과 같다.

| 파일 | 변경 |
| --- | --- |
| `src/scripts/export_gaic_public_cropper_score_labels.py` | GAIC/CGS public cropper scorer를 official candidate crop 전체에 적용하고, 같은 crop에 대해 SSTK full-option candidate eval row에서 explanation score, fatal flag, warning, subject/fatal metadata를 결합한 label JSONL을 생성한다. |
| `src/scripts/convert_t6_score_labels_to_mobilecropnet_v4_batch.py` | T6 전용 이름을 제거할 수 있도록 `label_source_name`, `candidate_source`, `score_semantics_name` 인자를 추가했다. public score label도 동일한 MobileCropNet v4 batch, pairwise, listwise schema로 변환한다. |
| `src/scripts/visualize_public_score_label_tiers.py` | label tier별 정성 분석용 PNG contact sheet를 생성한다. |
| `src/scripts/visualize_mobilecropnet_v4_track_comparison.py` | track별 best checkpoint의 GAIC official candidate scoring 결과와 `data/test_images` proposal 결과를 한 장에서 비교하는 PNG를 생성한다. |
| `src/scripts/visualize_mobilecropnet_v4_public_ar_comparison.py` | 과거 `pubdistill` checkpoint를 대상으로 multi-AR overlay를 생성한 진단 스크립트다. `pubdistill` label이 FREE-only였기 때문에 이 산출물은 Product AR cropper 품질 근거로 사용하지 않고, unsupported diagnostic으로만 보존한다. |
| `src/scripts/run_gaic_public_score_label_export.sh` | Train2636, Val200, Test500 전체 public-score label export, converter, tier visualization을 하나로 묶은 GPU workload entrypoint다. |

label tier는 다음 기준으로 정의했다. 여기서 "public score 높음/낮음"은 GAIC public cropper의 raw 출력값 자체에 전역 threshold를 건 것이 아니라, 같은 이미지 안의 official candidate crop들을 public cropper score로 정렬한 뒤 계산한 `score_rank_pct_by_image`를 기준으로 한다. 이 값의 범위는 `[0, 1]`이며 `1.0`은 해당 이미지에서 public cropper가 가장 높게 평가한 crop, `0.0`은 가장 낮게 평가한 crop이다. raw public cropper 출력은 `ranker_score`로 보존하지만 모델별 logit/score scale이므로 tier 판정에는 직접 사용하지 않는다.

| 기준 | 값 | 의미 |
| --- | ---: | --- |
| high public score | `selected_by_ranker == true` 또는 `score_rank_pct_by_image >= 0.85` | 이미지별 public cropper top-1 또는 상위 15% candidate |
| low public score | `score_rank_pct_by_image <= 0.20` | 이미지별 하위 20% candidate |
| contradiction | `fatal_flag == true` 또는 `explanation_score < 0.35` | SSTK explanation과 public positive 판단이 충돌할 가능성이 높은 후보 |
| explanation good | `fatal_flag == false`, `contradiction == false`, `explanation_confidence >= 0.60`, `explanation_score >= 0.45` | SSTK explanation metadata 관점에서도 양호한 후보 |
| SSTK negative | `fatal_flag == true`, `contradiction == true`, `explanation_score < 0.45`, 또는 `explanation_confidence < 0.45` | low public 후보 중 일반 negative로 쓰기 적절한 후보 |

| tier | 판정 조건 | 학습 처리 |
| --- | --- | --- |
| `main_positive` | high public score + explanation good | high-weight positive 후보 |
| `score_explanation_conflict` | high public score + contradiction | low-weight positive/metadata 후보. hard unsafe로 제거하지 않는다. |
| `gaic_public_positive_sstk_fatal` | high public score + SSTK fatal | review 대상. converter에서는 positive로 승격하지 않고 unsafe cap/demotion을 적용한다. |
| `low_weight_positive` | high public score지만 `main_positive`, `score_explanation_conflict`, `gaic_public_positive_sstk_fatal` 어느 쪽도 아닌 후보 | public cropper preference는 유지하되 weight를 낮춘 positive 후보 |
| `normal_negative` | low public score + SSTK negative | 일반 negative 후보 |
| `hard_negative` | high/low 조건과 무관하게 fatal만 남은 후보 | diagnostic hard-negative slice |
| `candidate` | 위 조건에 속하지 않는 중립 후보 | listwise/pairwise 후보 pool 유지용 |

전체 label export는 MLP spot workload `1096042/mcn-public-labels-20260420-135442`에서 성공했다. smoke run `1096039`도 별도로 통과했다. 산출물은 다음 경로에 있다.

| split | raw rows | images | converted rows | converted candidates | selected tier summary |
| --- | ---: | ---: | ---: | ---: | --- |
| Train2636 | 227,755 | 2,636 | 2,509 | 216,716 | main `774`, conflict `1,126`, low-weight `577`, fatal-review `159` |
| Val200 | 17,164 | 200 | 189 | 16,240 | main `68`, conflict `86`, low-weight `33`, fatal-review `13` |
| Test500 | 43,123 | 500 | 475 | 40,923 | main `132`, conflict `217`, low-weight `120`, fatal-review `31` |

Test500에서 GAIC public cropper score 자체의 official MOS benchmark 성능은 PCC `0.872628`, SRCC `0.845896`, Acc1/5 `0.606000`, Acc1/10 `0.804000`, Acc4/10 `0.725500`, Accw4/10 `0.605688`, top-1 MOS `3.985460`, MOS regret `0.246440`이다. 앞으로 GAIC benchmark scorer track의 목표는 이 teacher/direct scorer 성능과 라벨 큐레이션 품질을 극대화하는 것이며, MobileCropNet 쪽에서 강한 candidate scorer student를 별도로 만드는 것이 아니다.

처음에는 기존 `rank_320`과 SSTK track best profile인 `hybrid384`를 그대로 적용했다. 두 run은 정상 완료됐지만 서로 다른 한계를 보였다. `rank_320`은 replay label top-1 hit가 `0.907369`로 높고 official PCC/SRCC도 `0.580039`/`0.549180`까지 올랐지만 Acc1/10 `0.546000`, top-1 MOS `3.732080`으로 top-return이 낮았다. 반대로 `hybrid384`는 Acc1/10 `0.646000`, top-1 MOS `3.815800`으로 top-return은 좋지만 PCC/SRCC가 `0.367019`/`0.324848`로 낮았다.

원인 분석 결과, 핵심 병목은 label quality가 아니라 candidate set mismatch였다. converter의 safe positive 평균 MOS는 Test500 기준 `3.982421`로 raw public top 평균 MOS `3.985460`과 거의 같다. 그러나 `rank_320`/`hybrid384`는 학습 시 각각 32/48 후보 중심으로 보며, official benchmark는 평균 86개 candidate crop 전체를 scoring한다. 따라서 public score distillation에서는 training candidate coverage를 official candidate set에 맞추는 것이 더 중요하다.

이를 반영해 `mcn-pubscore-r320k96-tr06-s18-20260420-142309`를 추가로 학습했다. 이 profile은 input 320, `MobileNetV4 Conv-M` pretrained backbone, set-transformer ranker depth 2는 유지하되 `candidate_k=96`, `proposal_q=48`, batch size 8, `MAX_PAIRWISE_PAIRS=96`으로 official candidate coverage를 맞춘다. run은 MLP spot workload `1096052`에서 성공했고, train memory max는 약 `2,061MB`, 마지막 epoch GPU util mean/max는 `7.69%`/`34%`, throughput은 약 `17 samples/s`였다.

GAIC v2 official test 500장 결과는 다음과 같다.

| track/run | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/10 | Accw4/10 | top-1 MOS | regret | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GAIC public cropper score teacher | 0.872628 | 0.845896 | 0.606000 | 0.804000 | 0.725500 | 0.605688 | 3.985460 | 0.246440 | public score 상한 |
| `pubdistill-r320k128-zb035-s20` | 0.733951 | 0.726926 | 0.506000 | 0.710000 | 0.605000 | 0.489795 | 3.897880 | 0.334020 | public-score distillation diagnostic에서 PCC/SRCC/top-1 MOS/regret best |
| `pubdistill-r320k128-zb035-s21` | 0.726589 | 0.720069 | 0.514000 | 0.718000 | 0.596500 | 0.486168 | 3.879200 | 0.352700 | public-score distillation diagnostic에서 Acc1/10 best |
| `pubdistill-r320k128-strongtopk-s22` | 0.720026 | 0.715097 | 0.524000 | 0.714000 | 0.607000 | 0.492677 | 3.890700 | 0.341200 | public-score distillation diagnostic에서 Acc1/5, Acc4/10, Accw4/10 best |
| `pubscore-r320k96-tr06-s18` | 0.697523 | 0.675074 | 0.486000 | 0.666000 | 0.561500 | 0.454218 | 3.846940 | 0.384960 | 이전 public-score diagnostic baseline |
| `t6safe-r320-tr06-s18` | 0.616310 | 0.589892 | 0.456000 | 0.656000 | 0.538500 | 0.430915 | 3.830220 | 0.401680 | 기존 T6 diagnostic best |
| `t6safe-hybrid384-tr06-s17` | 0.425912 | 0.408618 | 0.444000 | 0.650000 | 0.481500 | 0.378624 | 3.822800 | 0.409100 | T6 hybrid 보강 run, 기존 T6 rank_320 미달 |
| `sstk-hybrid384-tr06-s17` | 0.447673 | 0.427922 | 0.444000 | 0.650000 | 0.464500 | 0.360553 | 3.822800 | 0.409100 | SSTK-only top-return best |
| `sstk-r320-tr06-s19` | 0.435216 | 0.405202 | 0.444000 | 0.646000 | 0.491000 | 0.383345 | 3.821640 | 0.410260 | SSTK-only balanced candidate |
| `pubscore-hybrid384-tr06-s17` | 0.367019 | 0.324848 | 0.442000 | 0.646000 | 0.476500 | 0.374469 | 3.815800 | 0.416100 | top-return은 양호하나 correlation 낮음 |
| `pubscore-r320-tr06-s18` | 0.580039 | 0.549180 | 0.376000 | 0.546000 | 0.439500 | 0.348844 | 3.732080 | 0.499820 | k=32 coverage 부족 |

`pubdistill-r320k128` 계열은 기존 `pubscore-r320k96`를 모든 primary metric에서 개선했다. 단일 checkpoint 관점에서는 `pubdistill-r320k128-zb035-s20`이 PCC/SRCC/top-1 MOS/regret이 가장 좋고, top-k 반환 지표 관점에서는 `pubdistill-r320k128-strongtopk-s22`가 Acc1/5, Acc4/10, Accw4/10에서 가장 좋다. GAIC public cropper score teacher와의 격차는 아직 남아 있다.

| metric | public score teacher | `pubdistill-r320k128-zb035-s20` diagnostic | diagnostic - teacher |
| --- | ---: | ---: | ---: |
| PCC | 0.872628 | 0.733951 | -0.138677 |
| SRCC | 0.845896 | 0.726926 | -0.118970 |
| Acc1/5 | 0.606000 | 0.506000 | -0.100000 |
| Acc1/10 | 0.804000 | 0.710000 | -0.094000 |
| Acc4/10 | 0.725500 | 0.605000 | -0.120500 |
| Accw4/10 | 0.605688 | 0.489795 | -0.115893 |
| top-1 MOS | 3.985460 | 3.897880 | -0.087580 |
| MOS regret | 0.246440 | 0.334020 | +0.087580 |

성능 격차의 주요 원인은 다음으로 판단한다.

- GAIC public cropper teacher는 GAIC candidate ranking을 직접 목표로 학습된 RoI/RoD 기반 crop ranker다. 반면 MobileCropNet student는 mobile backbone, compact set-transformer ranker, proposal head를 함께 쓰는 압축 모델이므로 후보 crop별 region feature 표현력이 teacher보다 낮다.
- 현재 distillation label은 `ranker_score`와 `score_rank_pct_by_image` 중심의 scalar target이다. teacher의 region feature, pairwise margin, top-k distribution, score calibration 정보를 직접 distill하지 않으므로 fine ordering이 손실된다.
- `candidate_k=96`으로 평균 86개 official candidate coverage는 맞췄지만, 학습 batch/메모리 제약 때문에 모든 이미지의 full candidate set을 항상 동일한 listwise context로 보지는 않는다. Acc4/10과 Accw4/10 격차는 top-1뿐 아니라 상위 여러 candidate ordering이 아직 teacher를 충분히 모사하지 못한다는 신호다.
- tier 설계에서 SSTK fatal/explanation metadata를 review/audit 목적으로 결합했지만, public score 자체와 SSTK explanation은 서로 다른 teacher에서 온 heterogeneous target이다. `score_explanation_conflict`, `gaic_public_positive_sstk_fatal` slice가 많으면 score distillation에는 약한 label noise나 weight attenuation으로 작용할 수 있다.
- selection metric은 아직 기존 SSTK track에서 쓰던 top-return 중심 composite의 영향을 받는다. public-score distillation track 전용으로 validation score를 PCC/SRCC, Acc1/10, Acc4/10, teacher KL/ListNet loss에 맞춰 재설계하지 않았다.
- 현재 run은 단일 seed와 제한된 hyperparameter 조합이다. k=96 확장이 큰 개선을 만든 점을 고려하면 candidate coverage 이후에는 LR, warmup, weight decay, pair/listwise weight, top-k loss weight의 좁은 ablation이 필요하다.
- teacher direct 평가는 public cropper가 모든 official candidate를 직접 scoring한 결과이고, student 평가는 MobileCropNet scorer가 그 함수를 근사한 결과다. 즉 teacher의 benchmark 점수는 "라벨 상한"이 아니라 실제 teacher model inference이며, student가 이를 따라가려면 단순 hard-positive 학습보다 teacher distribution matching이 필요하다.

아래 성능 향상 우선순위는 과거 public-score student distillation을 개선하기 위해 수립했던 기록이다. 현재 정책에서는 이 항목들을 MobileCropNet Product AR roadmap으로 진행하지 않고, 필요 시 GAIC benchmark scorer teacher/curation 연구의 참고 항목으로만 재분류한다.

1. `candidate_k=128` 또는 이미지별 full-candidate listwise 학습을 추가한다. GPU memory는 gradient accumulation과 smaller batch로 처리하고, validation은 official candidate full-set replay와 동일한 조건으로 맞춘다.
2. `score_rank_pct_by_image`만 쓰지 말고 raw `ranker_score`의 이미지별 z-score/temperature-scaled score를 함께 target으로 둔다. rank percentile은 robust하지만 margin 정보를 잃으므로, margin-aware pairwise loss와 score regression/calibration loss를 병행한다.
3. teacher top-k distribution distillation을 추가한다. 이미지별 후보 전체 또는 top-96에 대해 ListNet/KL/ListMLE loss를 적용하고, Acc4/10 개선을 위해 top-4 coverage loss를 별도로 둔다.
4. validation checkpoint selection을 public-score track 전용으로 분리한다. 최소 구성은 `0.30*SRCC + 0.25*PCC + 0.20*Acc1/10 + 0.15*Acc4/10 + 0.10*Accw4/10` 형태이며, SSTK-only composite과 혼용하지 않는다.
5. student over-score hard-negative mining을 반복한다. validation에서 student는 높게, public teacher는 낮게 평가한 후보를 다음 round pairwise/listwise hard negative로 재투입한다.
6. backbone 확장은 candidate coverage와 loss alignment 이후에 수행한다. `rank_320 k96`을 기준으로 seed/loss ablation을 먼저 끝낸 뒤, `Conv-L 352`, `Hybrid-M 384`, 또는 lightweight second-stage reranker를 비교한다.
7. 가능하면 teacher feature/logit distillation을 도입한다. public cropper 내부 feature 접근이 어렵다면 crop patch + global image context를 함께 넣는 second-stage crop scorer를 student 뒤에 붙여 teacher score를 모사한다.
8. SSTK explanation/fatal은 public score를 대체하는 supervision이 아니라 audit metadata로 분리 유지한다. fatal-review slice는 제품 안전 검토에는 유용하지만, GAIC public score benchmark를 최대화하는 학습에서는 score teacher ordering을 최대한 보존해야 한다.

위 우선순위 중 1-4, 8은 2026-04-20 `public_score_distill_v2` 실험으로 구현 및 검증했다.

| 항목 | 구현/검증 내용 |
| --- | --- |
| full-candidate coverage | `candidate_k=128`, `proposal_q=48`로 학습했다. GAIC v2 official candidate set의 실제 max candidate count는 90이므로 k128은 padding을 제외하면 full-candidate replay와 동일한 coverage다. |
| raw score margin target | converter에 `score_target_policy=blend_rank_zscore`를 추가했다. `score_rank_pct_by_image`와 이미지별 raw `ranker_score` z-score sigmoid target을 `raw_score_blend_weight=0.35`로 혼합해 `crop_utility_prob`/`score_prob`에 반영했다. |
| teacher distribution distillation | listwise row에 `teacher_softmax_local`을 저장하고, 학습 batch에는 `teacher_soft_target` tensor를 추가했다. loss에는 teacher KL/ListNet 성격의 `teacher_distill_loss`를 `TEACHER_DISTILL_WEIGHT=0.35` 또는 `0.45`로 적용했다. |
| top-k coverage loss | Acc4/10 개선을 위해 `topk_coverage_loss`를 추가했다. 이번 실험은 `TOPK_COVERAGE_K=4`, weight `0.15` 및 strong-topk `0.25`를 비교했다. |
| public-score 전용 selection | `selection_metric=public_score_distill`을 추가해 SSTK-only composite과 분리했다. training loop 내부에서는 official MOS가 아니라 validation의 pairwise/listwise/top-return/teacher-distill proxy를 사용하고, checkpoint별 official MOS는 후처리 평가로 계산한다. |
| SSTK metadata 분리 | SSTK fatal/explanation은 public score를 덮어쓰지 않고 label tier, safety audit, visualization metadata로 유지했다. `gaic_public_positive_sstk_fatal`은 review slice이며 public score ordering 자체를 직접 negative로 뒤집지 않는다. |
| hard-negative mining 준비 | `mine_mobilecropnet_v4_hard_negative_pairs.py`를 추가했다. best checkpoint의 train/val prediction JSONL에서 student `model_utility`는 높은데 teacher `score_target`은 낮은 후보를 찾아 다음 round pairwise JSONL로 재투입할 수 있다. 이번 full run의 학습에는 아직 round-2 mining pair가 들어가지 않았으므로 priority 5의 "반복 학습 효과"는 다음 실험에서 별도 검증해야 한다. |

실험은 모두 MLP spot workload, A100 단일 GPU, `--core-count=1 --allow-spot`로 실행했다. `1096165`는 10분 이상 Pending으로 중단한 뒤 `1096178`로 재생성했다.

| run | workload | 핵심 설정 | best epoch | train GPU util mean/max | train mem max | throughput | 결과 요약 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `pubdistill-r320k128-zb035-s20` | 1096163 | seed 20260420, listwise 0.8, distill 0.35, topk 0.15 | 8 | 9.38% / 33% | 2,219 MB | 15.34 img/s | PCC/SRCC/top-1 MOS best |
| `pubdistill-r320k128-zb035-s21` | 1096164 | seed 20260421, listwise 0.8, distill 0.35, topk 0.15 | 8 | 11.81% / 35% | 2,217 MB | 15.22 img/s | Acc1/10 best |
| `pubdistill-r320k128-strongtopk-s22` | 1096178 | seed 20260422, listwise 1.0, distill 0.45, topk 0.25 | 8 | 8.31% / 30% | 2,217 MB | 16.69 img/s | Acc1/5, Acc4/10, Accw4/10 best |

정량적으로는 k96 이전 best 대비 다음과 같이 개선됐다.

| metric | previous `pubscore-r320k96` | best after v2 | delta |
| --- | ---: | ---: | ---: |
| PCC | 0.697523 | 0.733951 | +0.036428 |
| SRCC | 0.675074 | 0.726926 | +0.051852 |
| Acc1/5 | 0.486000 | 0.524000 | +0.038000 |
| Acc1/10 | 0.666000 | 0.718000 | +0.052000 |
| Acc4/10 | 0.561500 | 0.607000 | +0.045500 |
| Accw4/10 | 0.454218 | 0.492677 | +0.038459 |
| top-1 MOS | 3.846940 | 3.897880 | +0.050940 |
| MOS regret | 0.384960 | 0.334020 | -0.050940 |

따라서 이번 결과의 핵심 결론은 "candidate coverage만 맞춘 k96"에서 한 단계 더 나아가, raw margin-aware target, full-list teacher distribution, top-4 coverage loss, public-score 전용 selection을 함께 넣을 때 correlation과 top-return이 동시에 개선된다는 것이다. 우선순위 5는 mining utility까지 구현했지만, train/val prediction 생성 후 round-2 학습까지는 이번 결과 이후의 다음 실험으로 남긴다. 우선순위 7은 public cropper 내부 feature 접근이 필요하므로 아직 구현하지 않았고, 대안으로 crop patch + global image context second-stage scorer를 붙여 teacher score/logit을 더 직접 모사하는 방향이 타당하다.

해석은 다음과 같다.

- GAIC public score track에서는 `candidate_k=96` 확장이 결정적이다. backbone을 무조건 키우는 `hybrid384`보다 official candidate coverage를 맞춘 `r320k96`이 모든 primary metric에서 높다.
- `pubscore-r320k96`는 기존 T6 diagnostic student보다 PCC/SRCC, Acc1/5, Acc1/10, Acc4/10, Accw4/10, top-1 MOS, regret이 모두 좋았다. 이 관찰은 public-score teacher ordering과 candidate coverage가 benchmark metric에 강하게 작용한다는 diagnostic 근거로만 사용한다.
- 단 이 결과는 `SSTK-only Product Track` 결론을 대체하지 않는다. GAIC public cropper score가 외부 GAIC-trained teacher이기 때문에, 상용 SSTK-only 목표에서는 여전히 `sstk-r320-tr06-s19`/`sstk-hybrid384`와 `arx2` 계열이 product baseline이다. public-score distillation 계열은 MobileCropNet Product AR 후보가 아니라 GAIC scorer/curation track 자료다.
- SSTK fatal/explanation metadata는 crop score를 직접 대체하지 않고 audit/tier visualization으로 분리하는 것이 성능과 해석 양쪽에서 더 안전하다. fatal-review tier는 별도 review slice로 남기고, 모델 score 학습은 public cropper ranking coverage를 충분히 보존해야 한다.

정성 산출물은 모두 PNG로 저장했다.

| 산출물 | 경로 |
| --- | --- |
| label tier test contact sheets | `artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/tier_viz/test/*/contact_sheet.png` |
| label tier val contact sheets | `artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/tier_viz/val/*/contact_sheet.png` |
| best-track GAIC 비교 contact sheet | `artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/track_comparison/gaic_test_best_tracks/contact_sheet.png` |
| best-track `data/test_images` 비교 contact sheet | `artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/track_comparison/test_images_best_tracks/contact_sheet.png` |
| aggregate summary | `artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/summary/sstk_product_topreturn_summary.md` |

시각화 bbox 색상 의미는 다음과 같다.

| 산출물 | 색상 | 의미 |
| --- | --- | --- |
| label tier overlay | tier color | 현재 시각화 대상인 focus crop. `main_positive`는 green, `score_explanation_conflict`는 orange, `gaic_public_positive_sstk_fatal`은 red, `normal_negative`는 blue, `low_weight_positive`는 purple, `hard_negative`는 dark red, `candidate`는 gray다. |
| label tier overlay | yellow | 같은 이미지에서 GAIC public cropper score가 가장 높은 `public top` crop |
| label tier overlay | green | 같은 이미지에서 `main_positive`로 분류된 대표 crop |
| label tier overlay | cyan | GAIC official MOS annotation 기준 최고 MOS crop |
| track comparison overlay | red | `sstk_hybrid384` best-track model top-1 crop |
| track comparison overlay | green | `t6_rank320` best-track model top-1 crop |
| track comparison overlay | blue | `public_r320k96` best-track model top-1 crop |

2026-04-20 로컬 동기화 점검 결과, 원격 산출물에서 비어 있지 않았던 overlay PNG를 모두 내려받았다. 주요 overlay count는 run별 `viz_test/overlays`가 각 48장, tier visualization은 test/val의 4개 tier가 각각 32장, track comparison은 GAIC test와 `data/test_images`가 각각 32장이다. 따라서 현재 로컬 `artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/**/overlays`는 원격 산출물과 파일 수가 일치한다.

### 7.10 `pubdistill` AR overlay 산출물 재분류

기존 `viz_test/overlays` 산출물은 대부분 `FREE` target AR만 표시했기 때문에, 사용자가 지적한 것처럼 MobileCropNet의 제품 차별점인 종횡비별 crop과 checklist explanation을 한눈에 보기 어렵다. 이를 보완하려는 목적으로 `pubdistill-r320k128-zb035-s20` checkpoint와 GAIC public cropper를 비교하는 multi-AR overlay를 추가 생성했지만, 이후 정밀 검토에서 이 산출물은 Product AR cropper 정성 결과로 쓰면 안 된다고 판단했다.

이유는 명확하다. `pubdistill-r320k128-zb035-s20`은 GAIC official candidate `FREE` scorer distillation을 위해 학습된 모델이고, 해당 run의 `dataset_summary.json`도 train `target_ar_counts={'FREE': 2509}`, val `{'FREE': 189}`로 확인된다. 따라서 이 checkpoint가 fixed AR별 crop을 출력하더라도 이는 학습 분포 밖의 proposal 진단값이며, 1:1/3:4/4:3/16:9/9:16 품질을 입증하지 않는다. 앞으로 이 산출물은 "unsupported diagnostic"으로만 보존하고, Product AR cropper 정성 비교는 `arx2` 또는 이후 AR별 label로 학습한 Product AR track 산출물을 사용한다.

| 데이터셋 | 생성 방식 | overlay 수 | contact sheet | 상세 보고서 |
| --- | --- | ---: | --- | --- |
| GAICv2 official test sample | `pubdistill` checkpoint의 unsupported fixed-AR proposal 진단과 GAIC public cropper official-candidate top crop을 함께 표시 | 32 | `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/ar_public_comparison/gaic_v2_test_best_s20_vs_public/contact_sheet.png` | `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/ar_public_comparison/gaic_v2_test_best_s20_vs_public/BEST_MODEL_PUBLIC_CROPPER_AR_COMPARISON_REPORT.md` |
| `data/TestImages` | official candidate가 없으므로 `pubdistill` unsupported AR proposal과 centered AR baseline으로 proposal bank를 만들고, GAIC public cropper가 이 bank를 scoring해 top crop을 표시 | 32 | `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/ar_public_comparison/test_images_best_s20_vs_public/contact_sheet.png` | `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/ar_public_comparison/test_images_best_s20_vs_public/BEST_MODEL_PUBLIC_CROPPER_AR_COMPARISON_REPORT.md` |

Overlay PNG는 각각 다음 폴더에 있다.

- GAICv2: `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/ar_public_comparison/gaic_v2_test_best_s20_vs_public/overlays`
- `data/TestImages`: `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/ar_public_comparison/test_images_best_s20_vs_public/overlays`

시각화 bbox 색상 의미는 다음과 같다.

| 색상 | 의미 |
| --- | --- |
| red | MobileCropNet `FREE` crop |
| green | MobileCropNet `1:1` crop |
| blue | MobileCropNet `3:4` crop |
| orange | MobileCropNet `4:3` crop |
| purple | MobileCropNet `16:9` crop |
| cyan | MobileCropNet `9:16` crop |
| yellow | GAIC public cropper top crop |
| white | GAIC official annotation 중 최고 MOS crop. GAICv2 비교에서만 표시 |

해석할 때 중요한 차이는 다음이다.

- GAICv2에서는 GAIC public cropper의 yellow box가 official candidate set 안에서 직접 선택된 top crop이므로 public cropper benchmark 결과와 동일한 의미를 가진다.
- `data/TestImages`에는 official annotation candidate가 없으므로 yellow box는 "GAIC public cropper standalone proposal"이 아니다. MobileCropNet AR proposal과 centered AR baseline으로 만든 bank 안에서 public cropper가 가장 높게 평가한 crop이다. manifest의 `public_cropper.source`가 `mcn_ar_proposal` 또는 `center_ar_baseline`으로 이 출처를 기록한다.
- 이 overlay의 colored AR boxes는 `pubdistill` checkpoint가 fixed AR별로 신뢰 가능한 cropper라는 의미가 아니다. 해당 checkpoint는 FREE-only candidate scorer distillation으로 학습됐으므로, subject가 잘리거나 target AR별 crop이 나쁜 경우는 시각화 오류라기보다 학습 목표 밖의 사용에서 발생한 expected failure로 해석한다.
- Product AR cropper의 제품 차별점은 `arx2`처럼 AR별 label row를 실제로 학습하고 AR-constrained proposal, checklist applicability, target-AR별 ranking/proposal metric을 통과한 모델에서만 주장한다.

GPU 실행 기록은 다음과 같다.

| workload | run name | 결과 | 비고 |
| ---: | --- | --- | --- |
| `1096221` | `mcn-ar-public-viz-20260420-191313` | Failed after GAICv2 artifacts created | GAICv2 비교 산출물은 정상 생성됐고, `data/test_images/Thumbs` 비이미지 파일 때문에 TestImages 단계에서 실패했다. 이후 비이미지 skip을 패치했다. |
| `1096224` | `mcn-ar-public-testimgs-20260420-191811` | Succeeded | 패치 후 `data/TestImages` 비교 산출물 32장을 정상 생성했다. |

독립 비교 보고서도 별도로 추가했다.

- `Implement_Docs/MobileCropNet_v4_0_Best_vs_GAIC_Public_AR_Comparison_Report_KO_2026-04-20.md`

### 7.11 Unified public benchmark 재분석 및 Product AR 실험 설계

2026-04-20 통합 평가 문서는 `artifacts/unified_public_benchmark_20260420/report/UNIFIED_PUBLIC_BENCHMARK_REPORT_KO.md`에 생성되어 있고, 현재 `Implement_Docs/UNIFIED_PUBLIC_BENCHMARK_REPORT_KO.md`로 복사되어 있다. 이 문서를 기준으로 MobileCropNet Product AR track의 다음 실험 방향을 다시 정리했다.

가장 중요한 전제는 평가 계약이다. 통합 public benchmark는 `FCDB/CPC/GNMC` public benchmark와 `GAIC v2` official benchmark를 하나의 method registry로 묶지만, FCDB/CPC/GNMC는 `candidate-window scoring protocol`을 primary로 사용한다. 즉 모델이 crop을 직접 생성하는 것이 아니라, 이미 staged된 후보 window 전체에 score를 매긴 뒤 top-1을 고르는 평가다. 최종 산출물 기준 public benchmark 규모는 `62,511` tasks, `22,511` images, `1,149,036` candidate windows이고, 평가 설정은 `mode=S`, `inject_gt=0`, `score_field=score`, `scoring_fallback=none`, `target_ar_filter_mode=hard`, `max_pairwise_per_task=40`이다.

이 계약은 Product AR cropper의 실제 배포 문제와 다르다. Product AR cropper는 candidate bank 없이 `FREE`, `1:1`, `3:4`, `4:3`, `16:9`, `9:16` crop을 직접 생성하고 explanation을 함께 반환해야 한다. 따라서 앞으로 평가는 두 lane으로 분리한다.

| lane | 목적 | 입력 | 핵심 metric | 해석 |
| --- | --- | --- | --- | --- |
| Scorer-only public benchmark | utility/ranking head가 주어진 후보를 잘 재랭킹하는지 확인 | FCDB/CPC/GNMC staged candidate windows | overall IoU, rank@1/5, CPC weighted pairwise, GNMC AR violation | 기존 `UNIFIED_PUBLIC_BENCHMARK_REPORT_KO.md`의 평가 계약이다. 후보 생성력은 직접 보지 않는다. |
| Product AR proposal-free benchmark | 모델이 직접 AR별 crop을 생성하고 explanation을 붙이는지 확인 | 원본 이미지 + target AR | AR compatibility, overflow, duplicate, IoU@top1, recall/best-IoU@K, explanation fidelity | MobileCropNet 제품 목표에 맞는 신규 평가 lane이다. |

통합 보고서의 기존 결과는 public-score label이 GAIC에만 과적합된 것은 아니라는 점을 보여준다. `mcn_pubscore_r320k96`는 static center/area baseline 대비 overall IoU `+0.0994`, CPC weighted pairwise `+0.1529`, FCDB IoU `+0.0978`, CPC IoU `+0.1272`, GNMC IoU `+0.0934`를 얻었다. 기존 product 계열과 비교해도 `mcn_prod_q24_repro` 대비 overall IoU `+0.0196`, CPC weighted pairwise `+0.0850`, `mcn_prod_rank320_bal_tr06` 대비 overall IoU `+0.0306`, CPC weighted pairwise `+0.0287`이다.

그러나 현재 정책상 이 결과는 MobileCropNet Product AR 목표로 이어지지 않는다. `mcn_pubscore_r320k96`는 public-score scorer distillation diagnostic이고, GAIC benchmark scorer/curation track 자료다. Product AR track에서 참고해야 할 점은 성능 수치 자체가 아니라 다음 교훈이다.

1. 후보 coverage와 candidate distribution alignment가 성능에 매우 크다.
2. FCDB/CPC/GNMC에서는 GAIC-style score만으로 모든 축을 이기지 못한다.
3. T6-safe 계열은 cross-dataset 안정성이 아직 약간 우세하다. 통합 보고서 기준 `mcn_pubscore_r320k96 - mcn_t6safe_r320_tr06_s18` delta는 overall IoU `-0.0037`, CPC weighted pairwise `-0.0002`, FCDB IoU `-0.0115`, CPC IoU `-0.0061`, GNMC IoU `-0.0030`이다.
4. 따라서 Product AR 개선은 GAIC public score distillation이 아니라 AR별 후보 생성, subject/safety/explanation consistency, FCDB/CPC/GNMC cross-dataset guard를 함께 개선하는 방향이어야 한다.

Product AR 다음 실험은 다음 순서로 설계한다.

| phase | 작업 | 산출물 | 성공 기준 |
| --- | --- | --- | --- |
| A. Best existing AR baseline 고정 | 기존 학습 모델 중 Product AR 목적에 맞는 `arx2-turbo-256`을 baseline으로 고정 | `artifacts/mobilecropnet_v4/product_ar_best_20260420/arx2_turbo256_*` PNG/manifest/report | 6개 target AR crop과 checklist panel이 정상 렌더링되고, AR-constrained proposal metric이 유지됨 |
| B. Scorer-only unified replay | `arx2-turbo`, `rank_320 tr06-s19`, `hybrid384`를 FCDB/CPC/GNMC staged windows에 재점수화 | unified registry에 Product AR rows 추가 | q24/rank320 기존 product baseline 대비 non-regression, T6-safe와 gap 분석 |
| C. Proposal-free evaluator 구현 | 원본 이미지에서 모델이 직접 생성한 AR별 crop을 FCDB/CPC/GNMC/GAIC label에 맞춰 평가 | `product_ar_proposal_free_eval.json`, dataset별 PNG contact sheet | AR violation 0, overflow 0에 가깝게 유지, FCDB/GNMC IoU 및 rank proxy가 center baseline보다 높음 |
| D. AR label 학습 확장 | 기존 GAIC-v2 AR label row를 유지하고, `rank_320`/`hybrid384` 구조와 top-return/risk/proposal-bag loss를 AR별 row에 이식 | `mcn-product-ar-r320-*`, `mcn-product-ar-hybrid384-*` | `arx2-turbo` 대비 proposal-free IoU/AR/explanation 지표 개선 |
| E. FCDB/CPC/GNMC 활용 | benchmark-honest split을 먼저 고정하고, train/val partition만 학습/selection에 사용 | public split manifest, leakage report | 최종 holdout에서 unified macro와 worst-dataset non-regression 통과 |
| F. Product report pack | GAIC/TestImages/FCDB/CPC/GNMC의 Product AR contact sheet를 동일 색상/패널 규칙으로 생성 | paper/report-ready PNG 및 Markdown | 정성/정량 결과를 같은 모델 기준으로 한눈에 비교 가능 |

FCDB/CPC/GNMC를 학습에 사용하는 경우에는 `benchmark-honest`와 `production-curation`을 분리한다. `benchmark-honest`에서는 최종 test split을 학습/selection에서 제외하고 일반화 수치를 보고한다. `production-curation`에서는 public label 전체를 teacher/labeler 개선에 사용할 수 있지만, 같은 public split 성능을 일반화 주장으로 쓰지 않는다. 이 원칙은 `UNIFIED_PUBLIC_BENCHMARK_REPORT_KO.md`의 T6 Deep Ranker Improvement Plan과도 일치한다.

이번 요청에 맞춰 기존 학습된 Product AR 모델 중 best인 `arx2-turbo-256` checkpoint를 원격 공유 스토리지에서 내려받고, public cropper 없이 Product AR 추론 시각화를 새로 생성했다. CUDA-only GAIC public cropper scoring은 이번 로컬 CPU 시각화에서는 의도적으로 skip했다.

| dataset | contact sheet | overlays | report |
| --- | --- | --- | --- |
| `data/test_images` sample 6장 | `artifacts/mobilecropnet_v4/product_ar_best_20260420/arx2_turbo256_test_images/contact_sheet.png` | `artifacts/mobilecropnet_v4/product_ar_best_20260420/arx2_turbo256_test_images/overlays` | `artifacts/mobilecropnet_v4/product_ar_best_20260420/arx2_turbo256_test_images/BEST_MODEL_PUBLIC_CROPPER_AR_COMPARISON_REPORT.md` |
| GAIC v2 test sample 6장 | `artifacts/mobilecropnet_v4/product_ar_best_20260420/arx2_turbo256_gaic_v2_test/contact_sheet.png` | `artifacts/mobilecropnet_v4/product_ar_best_20260420/arx2_turbo256_gaic_v2_test/overlays` | `artifacts/mobilecropnet_v4/product_ar_best_20260420/arx2_turbo256_gaic_v2_test/BEST_MODEL_PUBLIC_CROPPER_AR_COMPARISON_REPORT.md` |

### 7.12 T6/public score 기반 Product AR/full label track

2026-04-20 추가 작업에서는 기존 `QF_C1C6CAPEXP_C7_EXP` Product AR/full label row를 버리지 않고, crop score supervision만 T6 또는 GAIC public cropper score로 교체하는 label generation path를 구현했다. 이 track의 목적은 MobileCropNet을 다시 GAIC benchmark scorer student로 돌리는 것이 아니라, 실제 제품 목표인 `FREE`, `1:1`, `3:4`, `4:3`, `16:9`, `9:16` target AR별 cropper와 explanation head를 학습할 수 있는 full label을 만드는 것이다.

구현 스크립트는 다음과 같다.

| script | 역할 |
| --- | --- |
| `src/scripts/export_mobilecropnet_v4_product_ar_candidates.py` | 기존 Product AR batch label의 `matching_targets`와 `candidate_pool`을 `(image_id, target_ar, candidate_id)` 단위 candidate row로 flatten한다. `score_group_id=image_id::target_ar`를 부여해 fixed AR별 ranking을 분리한다. |
| `src/scripts/score_mobilecropnet_v4_product_ar_candidates_with_public_cropper.py` | flatten row를 GAIC/CGS public cropper로 scoring한다. raw public score는 `ranker_score`로 보존하고, 학습 target은 같은 `(image, target_ar)` 그룹 내부 rank percentile로 정규화한다. |
| `src/scripts/run_mobilecropnet_v4_product_ar_t6_label_generation.sh` | T6 deep crop ranker bundle을 이용해 Product AR candidate feature cache를 만들고, 같은 `(image, target_ar)` 그룹 기준 T6 score/rank를 산출한다. |
| `src/scripts/build_mobilecropnet_v4_product_ar_score_labels.py` | scored candidate row를 원래 Product AR batch schema로 merge한다. `score_prob`, `crop_utility_prob`, rank target, pairwise/listwise를 external score 기준으로 교체하되, SSTK explanation/fatal/safety metadata는 보존한다. |
| `src/scripts/run_mobilecropnet_v4_product_ar_score_track_matrix.sh` | 생성된 label dir을 모든 모델 profile 학습에 투입하는 bounded GPU workload runner다. |

핵심 정책은 external score와 SSTK safety/explanation을 같은 supervision으로 섞지 않는 것이다. T6/public score가 높더라도 SSTK teacher가 hard reject, fatal, unsafe negative로 표시한 crop은 그대로 positive로 승격하지 않는다. 대신 raw external score는 `external_score_teacher.raw_score_rank_pct`, `raw_external_ranker_score`에 보존하고, 학습에 들어가는 adjusted score는 `unsafe_score_cap=0.05`로 cap한다. 해당 후보는 `external_high_sstk_hard_reject_demoted` bucket에 들어가며, 같은 `(image, target_ar)` 그룹 안에서 안전한 external-best 후보를 positive로 선택한다. 모든 후보가 unsafe인 예외 상황에서만 원래 SSTK positive fallback을 허용한다. 따라서 이 track은 external scorer의 preference를 최대한 쓰되, 제품 안전 관점의 명백한 hard reject를 모델 positive로 학습하지 않도록 설계했다.

생성된 label 경로와 row 수는 다음과 같다.

| score source | split | label dir | row | candidate | unsafe negative | external-high hard reject demoted |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| GAIC public cropper | Train2636 | `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/training_labels_public_score_product_ar_v1` | 14,915 | 88,960 | 2,851 | 381 |
| GAIC public cropper | Val200 | `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200/artifacts/training_labels_public_score_product_ar_v1` | 1,136 | 6,856 | 231 | 45 |
| GAIC public cropper | Test500 | `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/training_labels_public_score_product_ar_v1` | 2,824 | 17,134 | 561 | 69 |
| T6 deep ranker | Train2636 | `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/training_labels_t6_score_product_ar_v1` | 14,915 | 88,960 | 2,851 | 349 |
| T6 deep ranker | Val200 | `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200/artifacts/training_labels_t6_score_product_ar_v1` | 1,136 | 6,856 | 231 | 34 |
| T6 deep ranker | Test500 | `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/training_labels_t6_score_product_ar_v1` | 2,824 | 17,134 | 561 | 69 |

AR row 분포는 두 track 모두 동일하다. Train 기준 `FREE=2,617`, `1:1=2,512`, `3:4=2,491`, `4:3=2,490`, `16:9=2,391`, `9:16=2,414`이며, Val/Test도 같은 Product AR schema를 유지한다. 이 점이 과거 `pubdistill` FREE-only diagnostic과 가장 큰 차이다.

GPU workload는 모두 updated `AGENTS.md`에 맞춰 `--allow-spot`, `--core-count=1`로 생성했다. Label generation은 public `1096304`, T6 `1096302`가 성공했다. 학습 matrix는 q24 profile을 `1096308`/`1096309`에서 먼저 시작했고, 나머지 profile은 `1096313`-`1096322` single-profile spot workload로 병렬 생성했다. 현재 모델 규모와 GPU usage 기준으로 2개 이상 GPU가 필요한 학습은 아니며, 독립 single-GPU workload 여러 개가 더 적합하다.

학습은 baseline 성격의 `q24_288_w075`뿐 아니라 `turbo_256`, `balanced_288`, `hq_320`, `rank_320`, `hybrid_384` 전체 profile에 적용했다. `public` track의 q24/turbo는 먼저 시작된 matrix workload `1096308`에서 완료했고, 중복 profile 진입 직후 해당 matrix는 종료했다. 나머지는 single-profile spot workload로 병렬 수행했다. 최종 산출물은 원격 공유 스토리지에서 로컬 `artifacts/mobilecropnet_v4/product_ar_score_labels_260420` 및 `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/*/artifacts/training_labels_{public,t6}_score_product_ar_v1`로 내려받았다. 대용량 중복 산출물인 `eval_test/predictions.jsonl`, `last.pt`는 제외했고, `best.pt`, metrics, report, contact sheet, overlay PNG는 보존했다.

GAIC v2 Test500 official MOS benchmark 기준 결과는 다음과 같다. `teacher` 행은 같은 test split에서 기존 SSTK fullopt teacher crop을 official MOS annotation으로 평가한 reference다.

| track | profile | run | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/10 | Accw4/10 | top1 MOS | regret | val top-return | GPU mean/max mem |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| teacher | `sstk_fullopt_largecap_v2_raw` | `sstk_teacher_fullopt_largecap_v2_raw` | 0.514590 | 0.497884 | 0.270 | 0.448 | 0.3780 | 0.2856 | 3.5985 | 0.6334 | - | - |
| public | `q24_288_w075` | `mcn-pubprodar-q24-288-w075-20260420-214900` | 0.505681 | 0.485115 | 0.288 | 0.454 | 0.3965 | 0.3000 | 3.5786 | 0.6533 | 0.4549 | 10.1% / 715MB |
| public | `turbo_256` | `mcn-pubprodar-turbo-256-20260420-214900` | 0.615663 | 0.598505 | 0.334 | 0.546 | 0.4640 | 0.3484 | 3.7484 | 0.4835 | 0.5012 | 13.3% / 889MB |
| public | `balanced_288` | `mcn-pubprodar-balanced-288-20260420-220603` | 0.623650 | 0.640141 | 0.366 | 0.552 | 0.4495 | 0.3518 | 3.7358 | 0.4961 | 0.5119 | 20.8% / 1935MB |
| public | `hq_320` | `mcn-pubprodar-hq-320-20260420-220603` | 0.310303 | 0.291250 | 0.092 | 0.196 | 0.1635 | 0.1129 | 3.0076 | 1.2243 | 0.4406 | 23.7% / 2501MB |
| public | `rank_320` | `mcn-pubprodar-rank-320-20260420-220603` | 0.659666 | 0.635614 | 0.322 | 0.530 | 0.4645 | 0.3519 | 3.6905 | 0.5414 | 0.5351 | 20.8% / 2235MB |
| public | `hybrid_384` | `mcn-pubprodar-hybrid-384-20260420-220603` | 0.322030 | 0.305894 | 0.338 | 0.516 | 0.4050 | 0.3213 | 3.6334 | 0.5985 | 0.4537 | 30.6% / 3047MB |
| T6 | `q24_288_w075` | `mcn-t6prodar-q24-288-w075-20260420-214900` | 0.487867 | 0.459514 | 0.370 | 0.576 | 0.4665 | 0.3668 | 3.7242 | 0.5077 | 0.4032 | 10.7% / 715MB |
| T6 | `turbo_256` | `mcn-t6prodar-turbo-256-20260420-220603` | 0.562893 | 0.539559 | 0.434 | 0.624 | 0.4655 | 0.3672 | 3.7953 | 0.4366 | 0.4555 | 11.8% / 889MB |
| T6 | `balanced_288` | `mcn-t6prodar-balanced-288-20260420-220603` | 0.574305 | 0.555146 | 0.382 | 0.574 | 0.4940 | 0.3943 | 3.7379 | 0.4940 | 0.4387 | 19.4% / 1935MB |
| T6 | `hq_320` | `mcn-t6prodar-hq-320-20260420-220603` | 0.262002 | 0.240117 | 0.212 | 0.342 | 0.3765 | 0.2988 | 3.1514 | 1.0805 | 0.3926 | 22.6% / 2501MB |
| T6 | `rank_320` | `mcn-t6prodar-rank-320-20260420-220603` | 0.553528 | 0.544158 | 0.386 | 0.572 | 0.4630 | 0.3627 | 3.7289 | 0.5030 | 0.4412 | 24.7% / 2235MB |
| T6 | `hybrid_384` | `mcn-t6prodar-hybrid-384-20260420-220603` | 0.284406 | 0.260879 | 0.330 | 0.500 | 0.4030 | 0.3221 | 3.5287 | 0.7032 | 0.4117 | 30.6% / 3047MB |

metric별 best는 분명하게 갈린다. PCC는 `public/rank_320`이 0.659666으로 최고이고, SRCC는 `public/balanced_288`이 0.640141로 최고다. 반면 top-return 계열은 `T6/turbo_256`이 Acc1/5 0.434, Acc1/10 0.624, top1 MOS 3.7953으로 가장 좋고, top-4 coverage 계열은 `T6/balanced_288`이 Acc4/10 0.4940, Accw4/10 0.3943으로 가장 좋다. 즉 public score label은 후보 전체 ordering의 상관 구조를 더 잘 전달했고, T6 score label은 top-1 best-return 선택에 더 유리했다.

이번 Product AR/full label track은 기존 SSTK teacher reference 대비 대부분 핵심 지표를 개선했다. 특히 `T6/turbo_256`은 Acc1/5를 0.270에서 0.434, Acc1/10을 0.448에서 0.624, top1 MOS를 3.5985에서 3.7953으로 올렸다. `public/rank_320`은 PCC/SRCC를 teacher 0.514590/0.497884에서 0.659666/0.635614로 올렸다. 다만 `hq_320`과 `hybrid_384`는 더 무거운 backbone과 높은 해상도가 항상 유리하지 않다는 점을 다시 보였다. 현 full-label 규모에서는 heavy/hybrid backbone이 score target보다 explanation/proposal auxiliary target과 optimization noise에 더 민감하며, candidate score ranking에는 `turbo_256`, `balanced_288`, `rank_320`처럼 중간 규모 profile이 안정적이다.

정성 확인용 산출물은 각 run의 `viz_test/contact_sheet.png`와 `viz_test/overlays/*.png`에 저장했다. 최종 summary는 `artifacts/mobilecropnet_v4/product_ar_score_labels_260420/public_direct/summary_final_20260420/sstk_product_topreturn_summary.md`와 `artifacts/mobilecropnet_v4/product_ar_score_labels_260420/t6_direct/summary_final_20260420/sstk_product_topreturn_summary.md`에 있다. Product AR cropper 관점의 기본 후보는 `T6/turbo_256`을 우선 선정하고, ranking correlation이 더 중요한 candidate auditing 또는 score calibration 분석에는 `public/rank_320`과 `public/balanced_288`을 함께 비교 대상으로 유지한다.

2026-04-21 후속 작업에서는 이 Product AR/full label track 위에 `deploy-align` 실험 lane을 추가했다. 핵심은 "candidate-bank replay로 학습된 utility를 실제 배포 proposal 분포에도 직접 정렬"하는 것이며, 관련 설계/실행/결과는 별도 문서 [MobileCropNet_v4_0_Product_AR_DeployAlign_Report_KO_2026-04-21.md](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/Implement_Docs/MobileCropNet_v4_0_Product_AR_DeployAlign_Report_KO_2026-04-21.md)에서 관리한다. 이 문서는 GAIC scorer 진단이 아니라 `Product AR cropper + explanation`의 실제 배포 표면을 primary로 다룬다. 최종 결과 기준으로는 `deploy-align 0.30/0.15` ablation이 모두 `arx2`는 넘었지만, 제품 후보 baseline인 `T6/turbo_256` 자체는 넘지 못했다. 또한 현재 deploy-align lane의 best checkpoint는 exact direct evaluator가 아니라 `deploy_align_topreturn` proxy로 선택되므로, 잔여 gap 해석에는 이 selection mismatch를 함께 고려해야 한다.

## 8. Replay-label 정량 평가 결과

GAIC test split 625개 기준 결과는 다음과 같다. 표는 test top-1 hit 우선으로 정렬했다.

| rank | run | test top-1 hit | exact best | NDCG@5 | SRCC | top1 IoU to best positive | proposal recall@5 IoU0.5 | utility regret |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `mcn-v4-q24-288-w075-20260416-095704` | 0.353600 | 0.379200 | 0.896934 | 0.614445 | 0.760730 | 0.867200 | 0.052007 |
| 2 | `mcn-v4-nb-q16-224-w075-20260416-095704` | 0.348800 | 0.347200 | 0.895313 | 0.613565 | 0.750238 | 0.860800 | 0.053632 |
| 3 | `mcn-v4-q16-256-w075-20260416-095704` | 0.347200 | 0.360000 | 0.894264 | 0.617598 | 0.749993 | 0.867200 | 0.056248 |
| 4 | `mcn-v4-q16-256-w100-20260416-095704` | 0.340800 | 0.352000 | 0.894726 | 0.610150 | 0.742225 | 0.865600 | 0.057197 |
| 5 | `mcn-v4-hq320-explain-20260416-173509` | 0.104000 | 0.131200 | 0.778383 | 0.487805 | 0.807610 | 0.846400 | 0.331880 |
| 6 | `mcn-v4-balanced288-explain-20260416-180111` | 0.068800 | 0.108800 | 0.769326 | 0.433997 | 0.803647 | 0.860800 | 0.359983 |

`q24-288-w075`는 입력 해상도와 proposal query를 함께 늘린 설정이다. test top-1 hit, NDCG@5, top1 IoU, utility regret에서 가장 우수하다. `q16-256-w075`는 val top-1 hit가 가장 높았지만 test에서는 `q24-288-w075`보다 낮았다. 이는 현 split 규모에서 validation variance가 존재함을 의미하므로, 최종 모델 선택에는 test 및 외부 benchmark를 함께 봐야 한다.

HQ-320은 val SSTK composite score는 높았지만 replay test top-1 hit와 official MOS best-return 지표가 모두 낮았다. 이는 모델 용량 자체보다 학습 objective와 checkpoint selection이 official top-return 품질을 충분히 반영하지 못하는 문제가 더 크다는 신호다.

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
| `mcn-v4-hq320-explain-20260416-173509` | 1379.366 | 1646.000 | 48.182 | 81.000 | 7986.000 | 428.530 |
| `mcn-v4-balanced288-explain-20260416-180111` | 124.377 | 180.000 | 20.083 | 90.000 | 9084.000 | 168.160 |

판단:

- Balanced/Turbo 계열은 평균 GPU 사용률이 15% 미만이고 VRAM 사용량이 1GB 미만이라 GPU 연산보다 data pipeline/CPU transform 영향이 컸다.
- HQ-320은 backbone 확장으로 평균 GPU 사용률이 약 48%, 최대 81%, `nvidia-smi` 메모리 최대 7.8GB 수준까지 상승했다. 그래도 A100 80GB 기준 여유가 크고 batch 16 full run이 9분 안팎에 완료되어 2개 이상 GPU를 쓰는 분산 학습은 필요하지 않다.
- 이후 detailed explanation three-profile workload에서는 local pretrained backbone과 bounded MLP run 환경에서 HQ-320 평균 `15.982%`, Balanced-288 평균 `10.958%`, Turbo-256 평균 `9.464%` GPU utilization을 기록했다. 상세 표는 7.3에 정리했다.
- 성능 최적화 우선순위는 multi-GPU가 아니라 official MOS와 더 정렬되는 SSTK-only loss/selection 설계, data loader, JPEG decode, CPU transform, batch size 상향, crop/mask pooling vectorization, candidate tensor pre-cache다.
- 제품용 latency 검증은 A100 처리량보다 mobile CPU/GPU/NPU export 경로를 별도로 측정해야 한다.

## 11. 정성 시각화 산출물

기존 네 실험과 HQ-320 run 모두 PNG contact sheet와 overlay PNG를 생성했다. 로컬 검증에서 contact sheet는 모두 PNG이며, HQ-320 contact sheet는 `(1680, 3840)`, 개별 overlay 예시는 `(1288, 1024)`다.

| run | contact sheet |
| --- | --- |
| `mcn-v4-q24-288-w075-20260416-095704` | `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-q24-288-w075-20260416-095704/viz_test/contact_sheet.png` |
| `mcn-v4-nb-q16-224-w075-20260416-095704` | `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-nb-q16-224-w075-20260416-095704/viz_test/contact_sheet.png` |
| `mcn-v4-q16-256-w075-20260416-095704` | `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-q16-256-w075-20260416-095704/viz_test/contact_sheet.png` |
| `mcn-v4-q16-256-w100-20260416-095704` | `artifacts/mobilecropnet_v4/gpu_runs/mcn-v4-q16-256-w100-20260416-095704/viz_test/contact_sheet.png` |
| `mcn-v4-hq320-explain-20260416-173509` | `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/viz_test/contact_sheet.png` |
| `mcn-v4-balanced288-explain-20260416-180111` | `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/viz_test/contact_sheet.png` |

시각화 색상 의미:

- green: label best positive
- blue: baseline candidate
- orange: learned proposal top-1 filtered by target AR compatibility
- red: MobileCropNet v4 selected crop

HQ-320 explanation 시각화는 기존 box overlay에 오른쪽 side panel을 추가했다. panel에는 `decision` 예측, `subject mode`, top crop utility label/score, positive/risk label/score, `aesthetic`, `subject`, `composition`, `technical` checklist label/score가 표시된다. 검증 결과 contact sheet는 PNG `1680x3840`, 개별 overlay 예시는 PNG `1288x1024`이며, `eval_test/predictions.jsonl`의 top candidate에 `model_explanation`과 `model_checklist`가 포함된다.

Detailed explanation run에서는 side panel을 한 단계 더 확장했다. `model detail`에는 `third_dist`, `phi_dist`, `center_dist`, `headroom`, `lookroom`, `subject_coverage`, `subject_scale`, `face_cut`, `joint_cut`, `context`, detail score, why tag가 표시되고, `label detail`에는 SSTK data factory가 생성한 `teacher_checklist_labels`, `teacher_checklist_scores`, `teacher_why_tags`가 함께 표시된다. 또한 `bbox legend`와 `bbox details`를 추가해 red/model top crop, green/best positive label, blue/baseline candidate, orange/target-AR compatible proposal top-1의 의미와 각 bbox 좌표/점수/AR error를 명확히 표시한다.

2026-04-17 시각화 검토에서 top crop이 baseline으로 선택되는 경우 baseline 자체에는 teacher checklist가 없어 `label detail`이 전부 `na`로 보이는 문제가 확인됐다. 이는 PNG 렌더링 오류가 아니라 label provenance 문제다. 세 detailed run의 top source는 baseline 비중이 높았고, 직접 non-NA teacher detail을 가진 top crop은 Balanced-288 `160/625`, HQ-320 `145/625`, Turbo-256 `190/625`였다. 보정 후 visualizer는 top crop에 직접 detail이 없으면 같은 image/AR의 가장 가까운 labeled candidate를 찾아 `source: nearest labeled candidate <candidate_id> | IoU to top=<value>`를 표시한다. 전체 625 row 중 fallback 후 non-NA label detail을 표시할 수 있는 row는 세 profile 모두 `605/625`이고, 나머지 `20/625`는 해당 image/AR 후보군 자체에 세부 label이 없다. 샘플 48장 시각화에서는 세 profile 모두 `48/48`장이 non-NA label detail을 표시한다.

또한 `model detail`의 `lookroom=na 0.98` 같은 표기는 모델이 `na` class를 높은 확률로 예측했다는 의미였지만, 사람이 보기에는 `na`와 score가 혼재된 품질 판단처럼 보였다. 시각화는 이를 `unlabeled p=0.98`로 변경했고, `headroom/lookroom` 등 수치 score는 별도 `framing/risk scores` 줄에 표시한다. 향후 학습에서는 `checklist_labels` 값이 `na`인 class target은 valid mask에서 제외하도록 `MobileCropNetV4BatchDataset`을 수정했다. 기존 checkpoint는 이미 `na`를 valid class로 학습했기 때문에 일부 field에서 `unlabeled` 예측 비율이 높지만, 새 학습부터는 결측 라벨을 맞히는 방향의 class loss가 제거된다. 재생성된 시각화 audit은 `artifacts/mobilecropnet_v4/detail_explain/detail_explain_visualization_audit_20260417.json`에 저장했다. 세 profile의 contact sheet와 overlay는 모두 PNG로 재생성했으며, 개별 overlay는 `1664x2100` 중심 크기로 검증했다.

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

HQ-320/explanation run:

- runner: `artifacts/mobilecropnet_v4/gpu_runs/run_hq320_explain_full.sh`
- checkpoint: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/best.pt`
- run log: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/run.log`
- summary: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/run_summary.json`
- test metrics: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/eval_test/metrics.json`
- predictions with explanation: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/eval_test/predictions.jsonl`
- official GAIC MOS metrics: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/gaic_official_test/metrics.json`
- official GAIC MOS + teacher metrics: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/gaic_official_test_with_teacher/metrics.json`
- qualitative PNG: `artifacts/mobilecropnet_v4/hq320_explain/mcn-v4-hq320-explain-20260416-173509/viz_test/contact_sheet.png`

Balanced-288/explanation control run:

- runner: `artifacts/mobilecropnet_v4/gpu_runs/run_profile_explain_full.sh`
- checkpoint: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/best.pt`
- run log: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/run.log`
- summary: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/run_summary.json`
- predictions with explanation: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/eval_test/predictions.jsonl`
- official GAIC MOS metrics: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/gaic_official_test/metrics.json`
- official GAIC MOS + teacher metrics: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/gaic_official_test_with_teacher/metrics.json`
- qualitative PNG: `artifacts/mobilecropnet_v4/profile_explain/mcn-v4-balanced288-explain-20260416-180111/viz_test/contact_sheet.png`

Detailed checklist explanation three-profile runs:

- runner: `artifacts/mobilecropnet_v4/gpu_runs/run_profile_detail_explain_full.sh`
- HQ-320 summary: `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-hq320-detail-explain-20260416-185647/run_summary.json`
- HQ-320 checkpoint: `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-hq320-detail-explain-20260416-185647/best.pt`
- HQ-320 predictions: `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-hq320-detail-explain-20260416-185647/eval_test/predictions.jsonl`
- HQ-320 qualitative PNG: `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-hq320-detail-explain-20260416-185647/viz_test/contact_sheet.png`
- Balanced-288 summary: `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-balanced288-detail-explain-20260416-185647/run_summary.json`
- Balanced-288 checkpoint: `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-balanced288-detail-explain-20260416-185647/best.pt`
- Balanced-288 predictions: `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-balanced288-detail-explain-20260416-185647/eval_test/predictions.jsonl`
- Balanced-288 qualitative PNG: `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-balanced288-detail-explain-20260416-185647/viz_test/contact_sheet.png`
- Turbo-256 summary: `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-turbo256-detail-explain-20260416-185647/run_summary.json`
- Turbo-256 checkpoint: `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-turbo256-detail-explain-20260416-185647/best.pt`
- Turbo-256 predictions: `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-turbo256-detail-explain-20260416-185647/eval_test/predictions.jsonl`
- Turbo-256 qualitative PNG: `artifacts/mobilecropnet_v4/detail_explain/mcn-v4-turbo256-detail-explain-20260416-185647/viz_test/contact_sheet.png`

GAIC-v2 official split baseline label 재학습:

- runner: `artifacts/mobilecropnet_v4/gpu_runs/run_gaic_v2_official_labels_profile.sh`
- aggregate summary: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/GAIC_V2_OFFICIAL_LABEL_RUN_SUMMARY_20260417.md`
- aggregate summary JSON: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/gaic_v2_official_label_runs_summary_20260417.json`
- Balanced-288 checkpoint: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-balanced288-20260417-101528/best.pt`
- Balanced-288 predictions: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-balanced288-20260417-101528/eval_test/predictions.jsonl`
- Balanced-288 official GAIC MOS metrics: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-balanced288-20260417-101528/gaic_official_test/metrics.json`
- Balanced-288 qualitative PNG: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-balanced288-20260417-101528/viz_test/contact_sheet.png`
- HQ-320 checkpoint: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-hq320-interactive-20260417-101528/best.pt`
- HQ-320 predictions: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-hq320-interactive-20260417-101528/eval_test/predictions.jsonl`
- HQ-320 official GAIC MOS metrics: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-hq320-interactive-20260417-101528/gaic_official_test/metrics.json`
- HQ-320 qualitative PNG: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-hq320-interactive-20260417-101528/viz_test/contact_sheet.png`
- Turbo-256 checkpoint: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-turbo256-20260417-101528/best.pt`
- Turbo-256 predictions: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-turbo256-20260417-101528/eval_test/predictions.jsonl`
- Turbo-256 official GAIC MOS metrics: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-turbo256-20260417-101528/gaic_official_test/metrics.json`
- Turbo-256 qualitative PNG: `artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-turbo256-20260417-101528/viz_test/contact_sheet.png`

AR-constrained proposal + mode-aware explanation `arx2` 재학습:

- runner: `artifacts/mobilecropnet_v4/gpu_runs/run_gaic_v2_ar_explain_profile.sh`
- aggregate summary: `artifacts/mobilecropnet_v4/gaic_v2_ar_explain/arx2_summary.md`
- aggregate summary JSON: `artifacts/mobilecropnet_v4/gaic_v2_ar_explain/arx2_summary.json`
- Balanced-288 summary: `artifacts/mobilecropnet_v4/gaic_v2_ar_explain/mcn-arx2-balanced-288-20260417-133107/run_summary.json`
- Balanced-288 official GAIC MOS metrics: `artifacts/mobilecropnet_v4/gaic_v2_ar_explain/mcn-arx2-balanced-288-20260417-133107/gaic_official_test/metrics.json`
- Balanced-288 qualitative PNG: `artifacts/mobilecropnet_v4/gaic_v2_ar_explain/mcn-arx2-balanced-288-20260417-133107/viz_test/contact_sheet.png`
- HQ-320 summary: `artifacts/mobilecropnet_v4/gaic_v2_ar_explain/mcn-arx2-hq-320-20260417-133107/run_summary.json`
- HQ-320 official GAIC MOS metrics: `artifacts/mobilecropnet_v4/gaic_v2_ar_explain/mcn-arx2-hq-320-20260417-133107/gaic_official_test/metrics.json`
- HQ-320 qualitative PNG: `artifacts/mobilecropnet_v4/gaic_v2_ar_explain/mcn-arx2-hq-320-20260417-133107/viz_test/contact_sheet.png`
- Turbo-256 summary: `artifacts/mobilecropnet_v4/gaic_v2_ar_explain/mcn-arx2-turbo-256-20260417-133107/run_summary.json`
- Turbo-256 official GAIC MOS metrics: `artifacts/mobilecropnet_v4/gaic_v2_ar_explain/mcn-arx2-turbo-256-20260417-133107/gaic_official_test/metrics.json`
- Turbo-256 qualitative PNG: `artifacts/mobilecropnet_v4/gaic_v2_ar_explain/mcn-arx2-turbo-256-20260417-133107/viz_test/contact_sheet.png`

SSTK-only Product Track top-return 재실험:

- runner: `src/scripts/run_mobilecropnet_v4_sstk_product_experiment.sh`
- aggregate summary: `artifacts/mobilecropnet_v4/sstk_product_topreturn/summary_20260417_161228_local/sstk_product_topreturn_summary.md`
- aggregate summary JSON: `artifacts/mobilecropnet_v4/sstk_product_topreturn/summary_20260417_161228_local/sstk_product_topreturn_summary.json`
- rank_320 product candidate checkpoint: `artifacts/mobilecropnet_v4/sstk_product_topreturn/mcn-prod-rank320-topreturn-20260417-160148/best.pt`
- rank_320 predictions: `artifacts/mobilecropnet_v4/sstk_product_topreturn/mcn-prod-rank320-topreturn-20260417-160148/eval_test/predictions.jsonl`
- rank_320 official GAIC MOS metrics: `artifacts/mobilecropnet_v4/sstk_product_topreturn/mcn-prod-rank320-topreturn-20260417-160148/gaic_official_test/metrics.json`
- rank_320 qualitative PNG: `artifacts/mobilecropnet_v4/sstk_product_topreturn/mcn-prod-rank320-topreturn-20260417-160148/viz_test/contact_sheet.png`
- q24 repro official GAIC MOS metrics: `artifacts/mobilecropnet_v4/sstk_product_topreturn/mcn-prod-q24-repro-20260417-160148/gaic_official_test/metrics.json`
- q24 top-return official GAIC MOS metrics: `artifacts/mobilecropnet_v4/sstk_product_topreturn/mcn-prod-q24-topret-b32-20260417-161228/gaic_official_test/metrics.json`
- turbo q24 top-return official GAIC MOS metrics: `artifacts/mobilecropnet_v4/sstk_product_topreturn/mcn-prod-turbo-q24-topret-b32-20260417-161228/gaic_official_test/metrics.json`
- balanced-selection aggregate summary: `artifacts/mobilecropnet_v4/sstk_product_topreturn/summary_20260417_173500_balanced_local/sstk_product_topreturn_summary.md`
- balanced-selection aggregate summary JSON: `artifacts/mobilecropnet_v4/sstk_product_topreturn/summary_20260417_173500_balanced_local/sstk_product_topreturn_summary.json`
- top-return best rank_320 checkpoint: `artifacts/mobilecropnet_v4/sstk_product_topreturn/mcn-prod-rank320-bal-s18-20260417-170831/best.pt`
- top-return best rank_320 official GAIC MOS metrics: `artifacts/mobilecropnet_v4/sstk_product_topreturn/mcn-prod-rank320-bal-s18-20260417-170831/gaic_official_test/metrics.json`
- balanced product rank_320 checkpoint: `artifacts/mobilecropnet_v4/sstk_product_topreturn/mcn-prod-rank320-bal-tr06-20260417-170831/best.pt`
- balanced product rank_320 official GAIC MOS metrics: `artifacts/mobilecropnet_v4/sstk_product_topreturn/mcn-prod-rank320-bal-tr06-20260417-170831/gaic_official_test/metrics.json`

Corrected Product Track 확장 sweep:

- aggregate summary: `artifacts/mobilecropnet_v4/sstk_product_next_20260417-191315/summary/sstk_product_topreturn_summary.md`
- aggregate summary JSON: `artifacts/mobilecropnet_v4/sstk_product_next_20260417-191315/summary/sstk_product_topreturn_summary.json`
- top-return best hybrid384 checkpoint: `artifacts/mobilecropnet_v4/sstk_product_next_20260417-191315/mcn-next-hybrid384-tr06-s17-20260417-191315/best.pt`
- top-return best hybrid384 official GAIC MOS metrics: `artifacts/mobilecropnet_v4/sstk_product_next_20260417-191315/mcn-next-hybrid384-tr06-s17-20260417-191315/gaic_official_test/metrics.json`
- rank_320 top-return tie checkpoint: `artifacts/mobilecropnet_v4/sstk_product_next_20260417-191315/mcn-next-r320-tr070-s17-20260417-191315/best.pt`
- rank_320 top-return tie official GAIC MOS metrics: `artifacts/mobilecropnet_v4/sstk_product_next_20260417-191315/mcn-next-r320-tr070-s17-20260417-191315/gaic_official_test/metrics.json`
- balanced product rank_320 checkpoint: `artifacts/mobilecropnet_v4/sstk_product_next_20260417-191315/mcn-next-r320-tr06-s19-20260417-191315/best.pt`
- balanced product rank_320 official GAIC MOS metrics: `artifacts/mobilecropnet_v4/sstk_product_next_20260417-191315/mcn-next-r320-tr06-s19-20260417-191315/gaic_official_test/metrics.json`
- q24/turbo balanced 재평가 summaries: `artifacts/mobilecropnet_v4/sstk_product_next_20260417-191315/mcn-next-q24bal-tr06-s17-20260417-191315/run_summary.json`, `artifacts/mobilecropnet_v4/sstk_product_next_20260417-191315/mcn-next-turbobal-tr06-s17-20260417-191315/run_summary.json`

T6 converter/fixed-AR diagnostic:

- corrected T6 v2b train label: `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/training_labels_t6_score_only_corrected_v2b/train_conditional_detr_batch.jsonl`
- corrected T6 v2b val label: `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200/artifacts/training_labels_t6_score_only_corrected_v2b/train_conditional_detr_batch.jsonl`
- corrected T6 v2b test label: `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/training_labels_t6_score_only_corrected_v2b/train_conditional_detr_batch.jsonl`
- fatal top debug summary: `artifacts/mobilecropnet_v4/teacher_improvement/t6_top_conflicts_debug_260417/fatal_top_summary.json`
- T6 deep ranker bundle summary: `artifacts/mobilecropnet_v4/teacher_improvement/t6_deep_crop_ranker_260417/full_bundle_v2/Gc/deep_ranker_summary.json`
- fixed-AR T6 smoke scores summary: `artifacts/mobilecropnet_v4/fixed_ar_t6/smoke_bundle_v2_bankfix/val_fixed_ar_t6_scores_summary.json`
- T6 diagnostic MobileCropNet summary: `artifacts/mobilecropnet_v4/t6_corrected_v2b_product_20260417-191315/summary/sstk_product_topreturn_summary.md`

Public-score distillation v2 및 best-vs-public multi-AR 비교:

- public-score distill v2 root: `artifacts/mobilecropnet_v4/public_score_distill_v2_260420`
- best single-checkpoint run: `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/mcn-pubdistill-r320k128-zb035-s20-20260420-172325`
- metric-wise top-k best run: `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/mcn-pubdistill-r320k128-strongtopk-s22-retry-20260420-173428`
- GAICv2 best-vs-public contact sheet: `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/ar_public_comparison/gaic_v2_test_best_s20_vs_public/contact_sheet.png`
- GAICv2 best-vs-public overlays: `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/ar_public_comparison/gaic_v2_test_best_s20_vs_public/overlays`
- GAICv2 best-vs-public manifest: `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/ar_public_comparison/gaic_v2_test_best_s20_vs_public/comparison_manifest.json`
- `data/TestImages` best-vs-public contact sheet: `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/ar_public_comparison/test_images_best_s20_vs_public/contact_sheet.png`
- `data/TestImages` best-vs-public overlays: `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/ar_public_comparison/test_images_best_s20_vs_public/overlays`
- `data/TestImages` best-vs-public manifest: `artifacts/mobilecropnet_v4/public_score_distill_v2_260420/ar_public_comparison/test_images_best_s20_vs_public/comparison_manifest.json`
- standalone comparison report: `Implement_Docs/MobileCropNet_v4_0_Best_vs_GAIC_Public_AR_Comparison_Report_KO_2026-04-20.md`

## 13. 제품/논문 수준 보완 과제

이 장은 `SSTK-only Product Track`만을 기준으로 재정리한다. GAIC train 2,636장의 official MOS annotation은 학습 loss, pseudo-label 생성, hard negative mining, proposal target, score calibration, checkpoint 선택, hyperparameter 선택에 사용하지 않는다. GAIC val/test official MOS는 외부 benchmark 및 논문 보고용 사후 감사 지표로 유지한다.

제품 목표도 “public GAIC 계열 모델까지 모든 지표에서 top-1”이 아니라 “상용 가능한 SSTK data factory label만으로 SSTK teacher 성능에 근접하고, 특히 Acc1/N, top-1 MOS, MOS regret 같은 top-return 지표에서 teacher보다 나은 student”로 정의한다. 2026-04-17 corrected sweep 이후 `hybrid384`/`rank_320 tr070`은 top-1 MOS `3.822800`, regret `0.409100`으로 top-return best를 갱신했고, `rank_320 tr06-s19`는 Acc4/10 `0.491000`, Accw4/10 `0.383345`를 유지하는 균형형 candidate가 됐다. 다만 PCC/SRCC는 full-option SSTK teacher `0.514590`/`0.497884`보다 낮으므로 다음 보완 과제는 candidate 전체 ordering과 calibration을 높이면서 top-return을 유지하는 방향이다.

1. Official GAIC MOS 평가 루프 고정
   - 모든 실험은 replay-label validation과 별도로 GAIC v2 val 200장 official MOS benchmark를 기록할 수 있지만, SSTK-only 제품 후보에서는 이 값을 checkpoint 선택 또는 hyperparameter 선택에 사용하지 않는다.
   - 최종 test 보고는 GAIC v2 test 500장 official MOS benchmark 한 번으로 고정하고, test set은 hyperparameter 선택에 사용하지 않는다.
   - 제품 후보 모델 선택은 SSTK held-out validation의 composite score로 수행한다. 구성 요소는 teacher ensemble/listwise ranking agreement, perturbation preference accuracy, pseudo best-return, proposal pseudo-recall, augmentation stability, latency/model size다.
   - 논문/보고서의 최종 성능 표에는 official MOS metric을 primary로, SSTK validation/replay/teacher metric을 diagnostic으로 분리한다. Teacher imitation metric이 좋아도 official MOS 지표가 개선되지 않으면 제품/논문 성능 개선으로 해석하지 않는다.

2. Hyperparameter search 확대
   - 1차 실행 완료: `rank_320` seed 3개, `top_return_weight=0.4/0.6/0.8`, batch16/32 ablation을 모두 single-GPU spot workload로 수행했다. 결과적으로 `top_return_weight=0.8` seed18은 top-return best, `top_return_weight=0.6` seed17은 균형형 product candidate가 됐다.
   - `sstk_balanced_topreturn` selection metric을 구현했다. 이 metric은 단일 `top_return_hit`가 아니라 top-return hit/score/regret, exact-best score, listwise quality, risk suppression, proposal recall, score calibration, top-1 risk penalty를 함께 본다. val split에서 explicit pairwise가 비어 있는 경우에는 `listwise_loss` 기반 fallback을 사용하도록 수정했다.
   - batch32 ablation은 PCC/SRCC를 teacher 이상으로 끌어올렸지만 top-return 지표가 크게 악화됐다. 따라서 현재 product track 기본 batch는 16으로 유지하고, batch24는 corrected balanced metric seed sweep 이후에만 제한적으로 시도한다.
   - 2차 실행 완료: corrected `sstk_balanced_topreturn`으로 `top_return_weight=0.6` seed 3개, `0.55/0.65/0.7` 좁은 loss ablation, LR `1e-4/2e-4`, weight decay `5e-4/1e-3` + warmup 2 epoch, 352/384 확장, q24/turbo balanced-selection 재평가를 수행했다.
   - `top_return_weight=0.6` seed sweep에서는 seed19가 가장 좋았다. `rank_320 tr06-s19`는 top-1 MOS `3.821640`, regret `0.410260`, Acc4/10 `0.491000`, Accw4/10 `0.383345`로 현재 균형형 product candidate다.
   - 좁은 loss ablation에서는 `top_return_weight=0.7`이 top-1 MOS `3.822800`, regret `0.409100`, Acc1/10 `0.650000`으로 top-return tie best였지만 PCC/SRCC `0.370447`/`0.343425`가 낮다. 따라서 `0.7`은 top-return 전용 후보이고, 균형형 기본값은 `0.6`을 유지한다.
   - LR/WD/warmup ablation에서는 LR `1e-4`가 PCC/SRCC `0.485484`/`0.461861`로 correlation이 가장 안정적이지만 top-1 MOS는 `3.818940`으로 약간 낮다. WD `5e-4` + warmup 2는 top-return은 좋지만 Accw4/10이 낮고, WD `1e-3`는 전반적으로 악화됐다.
   - 구조 확장은 352보다 384 Hybrid-M이 유효했다. `hybrid384`는 top-1 MOS `3.822800`, regret `0.409100`, Acc1/10 `0.650000`으로 top-return best다. 단 Acc4/10/Accw4/10은 `rank_320 tr06-s19`보다 낮고 GPU util/memory가 증가하므로 제품 후보는 비용/성능 trade-off로 분리한다.
   - q24/turbo 재평가는 corrected balanced selection에서도 실패했다. q24는 top-1 MOS `3.676320`, turbo는 `3.624940`으로 낮아 product 후보에서 제외한다.
   - 각 sweep의 1차 우열은 SSTK held-out validation composite score로 판정한다. GAIC v2 official MOS는 외부 일반화 감사 지표로 별도 기록하며, 이 값을 보고 선택한 모델은 제품 후보가 아니라 `benchmark-tuned` 후보로 분리한다.
   - SSTK-only sweep에서 GAIC official MOS가 지속적으로 악화되는 경우에는 loss 설계나 pseudo-label 품질을 재검토하되, GAIC annotation을 직접 학습 신호로 역주입하지 않는다.

3. Data pipeline 최적화
   - 1차 구현 완료: `precompute_sample_tensors`로 candidate/positive/box target tensor를 dataset 초기화 시 사전 구축하고, `image_tensor_cache_size`로 worker별 LRU image tensor cache를 도입했다.
   - 1차 구현 완료: `num_workers`, `persistent_workers`, `prefetch_factor`, `pin_memory_device`를 학습 CLI에서 제어할 수 있게 했고, epoch metric에 loader data wait/compute time을 기록한다.
   - 검증 산출물: `artifacts/mobilecropnet_v4/data_pipeline_bench/*.json`, 상세 보고서 `Implement_Docs/MobileCropNet_v4_0_DataPipeline_Optimization_KO_2026-04-16.md`
   - 로컬 512-row benchmark에서 worker 4 + persistent + prefetch 기준 baseline `489.624 samples/s`, precompute `613.936 samples/s`, precompute + image cache `2129.654 samples/s`를 기록했다.
   - 원격 MLP GPU workload `data-pipeline-smoke-20260416-135808` / run id `1093988`에서도 A100-SXM4-80GB 1장 기준 smoke가 성공했다. 동일 계열 설정의 remote dataloader benchmark는 `2588.948 samples/s`였고, CUDA train smoke는 loader metric, GPU util, checkpoint 저장까지 정상 기록했다.
   - 다음 full experiment에서는 `--num_workers 4 --persistent_workers --prefetch_factor 4 --precompute_sample_tensors --image_tensor_cache_size 512`를 1차 권장 설정으로 사용한다.
   - 단, throughput 개선은 official MOS benchmark를 더 자주 안정적으로 돌리기 위한 수단이며, 정량 성능 개선으로 보고하려면 동일 official MOS metric schema에서 유의미한 개선이 확인되어야 한다.

4. Ranking 개선
   - Ranking head의 목표는 teacher score 회귀 자체가 아니라 official MOS candidate ranking으로 일반화되는 crop preference를 학습하는 것이다. 단, GAIC train 2,636장 official MOS annotation은 loss, pair mining, score calibration에 사용하지 않는다.
   - 1차 학습 신호는 SSTK data factory의 teacher score를 절대값 회귀로 쓰지 않고, confidence-weighted pairwise/listwise preference로 변환한다. 큰 margin의 teacher-consensus pair만 강하게 학습하고, teacher 간 불일치 또는 낮은 confidence pair는 label smoothing과 낮은 weight를 적용한다.
   - `MOS regret`를 줄이기 위한 SSTK-only surrogate로 top-1 pairwise risk loss를 추가한다. 각 이미지에서 high-confidence top crop은 destructive perturbation crop, subject cut-off crop, excessive background crop, low-technical-quality crop보다 margin 이상 높게 점수화되도록 학습한다.
   - Counterfactual perturbation preference를 핵심 보강 신호로 사용한다. 좋은 SSTK crop에서 shift, scale, aspect-ratio squeeze, subject truncation, face/person edge cut, salient-region removal, 과도한 empty margin을 생성하고 원본 crop이 변형 crop보다 높게 ranking되도록 한다. 이는 전문가 MOS를 직접 쓰지 않아도 `Acc1/N`, `top1 MOS`, `MOS regret`에 대응되는 top-return inductive bias를 제공한다.
   - Augmentation/equivariance consistency loss를 도입한다. resize, color jitter, horizontal flip, mild crop jitter 후 좌표가 변환된 동일 후보의 상대 ranking이 유지되도록 하여 teacher label noise와 dataset-specific shortcut을 줄인다.
   - 이미지별 score normalization을 적용한다. Official benchmark의 핵심은 이미지 내부 candidate 순위이므로, global score scale 회귀보다 per-image ListNet/ListMLE, pairwise margin, NDCG-style weighting을 우선 적용한다.
   - target aspect ratio별 ranking head 또는 FiLM conditioning 강화는 SSTK validation에서 aspect-ratio별 preference accuracy, pseudo best-return, stability가 개선되는지 먼저 본다. 이후 GAIC official MOS에서는 target aspect ratio별 `SRCC`와 `AccK/N`을 사후 진단한다.
   - Public cropper score는 license와 상용 사용성이 명확히 확인되기 전까지 학습 teacher로 사용하지 않는다. 비교 평가는 가능하지만, SSTK-only 제품 후보의 distillation source는 내부 SSTK teacher/feature 산출물로 제한한다.
   - 채택 기준은 SSTK-only validation 개선 후 GAIC official MOS 사후 평가에서 `SRCC`, `Acc1/5`, `Acc1/10`, `Acc4/10`, `Accw4/10`, `top1 MOS`, `MOS regret` 중 최소 3개 이상이 개선되고, 나머지가 통계적으로 큰 폭 악화되지 않는 것이다.

5. Proposal head 개선
   - Proposal head의 최종 목표는 SSTK label의 단일 positive IoU가 아니라, 사람에게 그럴듯한 고품질 crop 후보군을 candidate set 안에 충분히 포함시키는 것이다. GAIC official top-N crop은 사후 benchmark 진단에만 사용하고 proposal 학습 target으로 사용하지 않는다.
   - 2026-04-17 1차 구조 보완으로 fixed target AR proposal을 AR-constrained parameterization으로 변경했다. `FREE`는 기존 자유 box를 유지하지만, `1:1`, `9:16`, `16:9`, `3:4`, `4:3`은 `cx,cy,scale`만 예측하고 target AR로 width/height를 결정한다. 최종 `arx2` 구현에서는 원본 이미지가 차지하는 letterbox content rect와 resize rounding을 함께 반영해 원본 crop AR이 target AR을 보존하도록 보정한다. 이 변경은 official MOS 성능을 직접 보장하는 개선은 아니지만, proposal 후보군이 target AR 제약을 위반해 evaluation/visualization을 혼동시키는 문제를 제거하는 prerequisite이다.
   - Proposal 평가표는 raw objectness top-1과 target-AR-selected top-1을 분리한다. `proposal_raw_top1_target_ar_log_error`, `proposal_raw_top1_target_ar_compatible`, `proposal_top1_target_ar_log_error`, `proposal_top1_target_ar_compatible`, `proposal_target_ar_recall_at_5_iou_0_5`, `proposal_target_ar_best_iou_at_5`를 기록해 fixed AR geometry sanity와 candidate coverage를 동시에 본다.
   - SSTK-only pseudo-positive bag을 구성한다. 입력은 high-confidence teacher top crops, subject/saliency preserving crops, baseline full crop, aspect-ratio policy crop, rule-of-thirds/context-preserving anchor다. 각 box에는 teacher consensus, subject coverage, saliency coverage, crop area sanity, technical-risk penalty를 결합한 soft objectness weight를 부여한다.
   - Proposal loss는 단일 GT box 회귀가 아니라 bag coverage objective로 바꾼다. Q개 query가 pseudo-positive bag의 서로 다른 mode를 덮도록 `max IoU coverage`, high-confidence bag recall, duplicate suppression을 함께 최적화한다.
   - Query specialization을 명시한다. 예를 들어 full/context query, subject-tight query, portrait/person-safe query, rule-of-thirds query, wide-scene query, detail crop query를 두고, query type별 prior와 aspect-ratio conditioning을 부여한다. 이렇게 하면 proposal diversity가 단순 박스 간 거리 증가가 아니라 의미 있는 crop mode coverage로 작동한다.
   - Destructive proposal hard negative를 추가한다. subject/face/person/text를 절단하거나 salient region을 빼거나 지나치게 작은 crop/과도한 empty crop은 proposal objectness와 ranking score가 동시에 낮아지도록 학습한다.
   - Repair pretraining을 도입한다. 좋은 SSTK crop을 인위적으로 흔든 뒤 proposal head가 원래 crop 또는 subject-preserving crop으로 되돌리는 delta를 예측하도록 하여 localization 안정성을 높인다.
   - 기존 `proposal recall@5 IoU0.5 >= 0.90` 목표는 SSTK pseudo-positive geometry sanity check로 유지한다. 추가로 SSTK validation에서 `pseudo_top1_recall@K`, `pseudo_top5_recall@K`, `teacher_oracle_regret@K`, destructive-negative suppression을 기록한다.
   - GAIC v2 val/test official MOS annotation 기준 `official_top1_recall@K`, `official_top5_recall@K`, `best_candidate_MOS@K`, `oracle_MOS_regret@K`는 proposal 사후 진단 지표로만 사용한다. 최종 제품/논문 성능 목표는 `AccK/N`, `AccwK/N`, `top1 MOS`, `MOS regret` 개선으로 판정한다.

6. SSTK-only 성능 개선 실험 순서
   - Phase A: SSTK label builder를 확장해 ranking/proposal 보강용 JSONL을 생성한다. 산출물에는 teacher-consensus pair, perturbation pair, destructive hard negative, pseudo-positive bag, query-type prior, sample confidence를 포함한다.
   - Phase B: Ranking loss ablation을 수행한다. 비교군은 기존 score regression, confidence-weighted pairwise margin, per-image ListMLE/ListNet, NDCG-weighted top-return loss, augmentation consistency 조합이다.
   - Phase C: Proposal head ablation을 수행한다. 비교군은 기존 IoU regression, pseudo-positive bag coverage, query specialization, repair pretraining, destructive-negative objectness 조합이다.
   - Phase D: 모델 선택은 SSTK held-out validation composite score로 고정한다. 이후 선택된 checkpoint만 GAIC v2 val/test official MOS benchmark에 통과시켜 외부 성능을 보고한다.
   - Phase E: 결과 표는 `SSTK-selected`와 `benchmark-tuned`를 분리한다. 상용 후보는 `SSTK-selected` 행만 사용하고, GAIC annotation을 이용한 어떤 학습/선정 단계도 포함하지 않는다.
   - 2026-04-17 실행 결과, Phase B의 `top_return_loss`와 `sstk_topreturn` selection은 `rank_320`에서는 유효했지만 q24/turbo에서는 official MOS 일반화가 약했다. 이어서 `sstk_balanced_topreturn` selection, seed sweep, loss weight ablation, batch ablation을 수행했고, top-return best `rank320-bal-s18`과 균형형 product candidate `rank320-bal-tr06`을 얻었다.
   - 이후 corrected sweep으로 `top_return_weight=0.6` seed sweep, `0.55/0.65/0.7` loss ablation, LR/WD/warmup ablation, 352/384 구조 확장, q24/turbo balanced-selection 재평가를 완료했다. 결과적으로 top-return best는 `hybrid384-tr06-s17`/`r320-tr070-s17`, 균형형 product candidate는 `r320-tr06-s19`로 갱신했다.
   - 다음 GPU 실험 우선순위는 `rank_320 tr06-s19`와 `hybrid384`를 고정 후보로 두고, SSTK-only perturbation/repair pair와 destructive-negative suppression을 더 직접적으로 loss/selection에 반영하는 것이다. 추가 해상도 증가는 384에서 Acc4/10/Accw4/10이 개선될 때만 확대한다.

7. 현재 GAIC-SSTK 라벨 기반 적용 가능성
   - 위 SSTK-only ranking/proposal 개선 방향은 현재 GAIC 학습 이미지와 `data/GAIC/All/artifacts/training_labels/gaic_personv6_server_v1_leftover_ignore_monotonic` 계열 SSTK data factory label로도 구현 및 검증 가능하다. 이때 사용하는 신호는 GAIC official MOS annotation이 아니라 SSTK teacher/replay label, routing, candidate metadata, safety/reject tag다.
   - 현재 v4 split의 train-dev는 783개 이미지, 4,380개 `(image_id, target_ar)` row를 포함한다. train-dev 기준 `matching_targets`는 9,425개, `candidate_pool`은 89,816개, `ignored_candidates`는 13,319개, `overflow_candidates`는 8,483개다. 즉 proposal positive, ranking pool, hard/unsafe negative, ignored audit bucket이 모두 존재한다.
   - 원본 label directory에는 `train_pairwise.jsonl` 44,282 row, `train_listwise.jsonl` 6,820 row, `train_regression.jsonl` 155,051 row가 이미 존재한다. 따라서 explicit pair/list supervision을 v4 학습에 연결하는 작업은 label 재생성 없이도 가능하다.
   - 현재 row에는 `score_targets`, `score_prob`, `score_rank_pct`, `crop_utility_prob`, `crop_utility_rank_pct`, `crop_utility_softmax_local`, `crop_utility_z_local`, `pseudo_mos_1to5`, `score_margin_to_top1`, `crop_utility_margin_to_top1`, `macro_targets`, `why_tags`, `reject_tags`, `safety_penalty`, `monotonic_label_state`, `safe_leftover_policy_state`가 포함된다. 이는 score 회귀보다 confidence-weighted pairwise/listwise preference와 per-image normalized ranking target을 만들기에 더 적합하다.
   - routing 정보도 모든 train-dev row에 존재한다. `subject_mode`, `subject_mode_id`, `policy_id`, `shot_type`, `scene_subtype`, `subject_prior_bbox_norm_xyxy`, `subject_support_overlay`, `subject_set.union_box_xyxy`가 있어 target aspect ratio conditioning, query specialization, subject-preserving pseudo-positive bag 구성에 사용할 수 있다.
   - 현재 라벨만으로 바로 가능한 1차 개선은 `score_target_mode` 선택, confidence-weighted pairwise/listwise loss, hard/unsafe/overflow/reject-tag 기반 top-1 risk loss, matching/high-score-safe 후보 기반 pseudo-positive proposal bag, SSTK-only validation composite score다.
   - 추가 builder가 필요하지만 현재 GAIC 이미지와 SSTK label로 가능한 2차 개선은 perturbation pair 생성, destructive crop 생성, repair pretraining label 생성, query-type prior label 생성이다. 좋은 crop의 shift/scale/aspect squeeze, subject truncation, face/person edge cut, salient-region removal, excessive empty margin 같은 counterfactual crop은 현재 positive crop, routing subject prior, reject/safety tag를 이용해 만들 수 있다.
   - 다만 현재 GAIC 이미지는 public benchmark 이미지이므로, 여기서의 의미는 “GAIC official MOS 없이 SSTK label만으로 개선 신호를 만들 수 있는지 검증”이다. 상용 가능 성능 결론은 동일한 label builder와 학습 프로토콜을 상용 가능한 SSTK full/held-out split에 적용해 다시 확인해야 한다.
   - 구현 우선순위는 다음으로 갱신한다. 첫째, `rank_320`을 현재 product candidate로 고정하고 seed variance를 확인한다. 둘째, `top_return_loss`를 포함하되 pure top-return selection이 아니라 `sstk_balanced_topreturn` composite를 구현한다. 셋째, hard/unsafe/overflow/reject-tag 기반 top-1 risk와 proposal pseudo-positive bag을 candidate coverage metric에 직접 연결한다. 넷째, perturbation pair와 repair label builder를 학습 데이터에 붙인다. 다섯째, 선택된 checkpoint만 GAIC official MOS benchmark에서 사후 평가한다.
   - 1차 구현 및 검증은 `Implement_Docs/MobileCropNet_v4_0_SSTKOnly_RankingProposal_Enhancement_KO_2026-04-16.md`에 기록했다. 구현 범위는 explicit pairwise/listwise 연결, confidence-weighted score/list/pair loss, hard/unsafe/overflow/reject-tag 기반 top-1 risk loss, high-score safe candidate 기반 proposal positive bag, perturbation/repair label builder, `sstk_composite` checkpoint selection이다.
   - GAIC official MOS test 500 사후 평가에서 conservative SSTK-only 설정은 previous best 대비 PCC `0.399840 -> 0.421379`, SRCC `0.368395 -> 0.394981`로 개선했고 Acc4/10 `0.469000 -> 0.466000`, Accw4/10 `0.365610 -> 0.363532`은 거의 유지했다. 반면 Acc1/5 `0.434000 -> 0.378000`, Acc1/10 `0.630000 -> 0.542000`, top1 MOS `3.804520 -> 3.706440`, MOS regret `0.427380 -> 0.525460`은 악화됐다.
   - 2026-04-17 top-return 보강 run에서는 `rank_320`이 PCC `0.489461`, SRCC `0.462346`, Acc1/5 `0.442000`, Acc1/10 `0.632000`, top-1 MOS `3.802640`, MOS regret `0.429260`을 기록해 기존 q24 best와 사실상 동률 이상의 product candidate가 됐다. 이후 balanced ablation에서는 `rank320-bal-s18`이 top-1 MOS `3.818520`, MOS regret `0.413380`을 기록했고, `rank320-bal-tr06`은 PCC/SRCC `0.489133`/`0.469995`와 top-1 MOS `3.812800`을 동시에 유지했다.
   - corrected 확장 sweep에서는 `hybrid384-tr06-s17`과 `r320-tr070-s17`이 top-1 MOS `3.822800`, MOS regret `0.409100`, Acc1/10 `0.650000`으로 top-return best를 갱신했다. 균형형으로는 `r320-tr06-s19`가 top-1 MOS `3.821640`, MOS regret `0.410260`, Acc4/10 `0.491000`, Accw4/10 `0.383345`를 기록했다. 반대로 q24/turbo는 corrected balanced selection에서도 official MOS가 낮아 pseudo-top-return 과적합 반례로 남았다.
   - 따라서 현재 방향성은 유효하지만 “top-return만 강화”가 아니라 “top-return과 ranking/correlation/calibration/risk/proposal coverage의 균형”이 중요하다. 다음 실험은 GAIC annotation을 쓰지 않고 SSTK-only pseudo top-return pair, perturbation top-return pair, destructive-negative suppression, proposal bag recall을 selection composite와 loss에 더 직접적으로 연결하는 방향으로 진행한다.

8. Public cropper 비교
   - GAIC v2 test split 500장 전체에 대해 public cropping baselines, 기존 Teacher 모델, MobileCropNet v3.x, v4.0을 동일 official MOS metric schema로 비교한다.
   - SSTK teacher exact comparison은 이번 작업에서 GAIC v2 test 500장 전체 full-option으로 확장 완료했다. 이후 비교표에서는 `q24_288_w075_gaic_v2_test500_teacher500_qf_c1c6capexp_c7exp_a100` 산출물을 기준 teacher row로 고정한다.
   - CACNet처럼 단일 crop만 출력해 official candidate별 score가 없는 모델은 `PCC/SRCC/AccK/N/AccwK/N`를 primary 표에서 `-`로 표시하고, IoU projection 진단값은 별도 참고값으로 분리한다.
   - FCDB/CPC/GNMC 같은 public benchmark로 외부 일반화 확인
   - 관련 통합 평가 문서는 `Implement_Docs/SSTK_PublicBenchmark_FCDB_CPC_GNMC_AutoValidation_Implementation_KO_2026-04-15.md`와 `Implement_Docs/Post_20260317_Implementation_Report_With_C7_Viz_KO_2026-04-20.md`에서 확인했다. 전자는 FCDB/CPC/GNMC adapter, staging, evaluation metric, smoke/full 결과를 기록하고, 후자는 9장에 GAIC/FCDB/CPC/GNMC 통합 public benchmark suite를 요약한다. 보고된 task 규모는 FCDB `1,714`, CPC `10,797`, GNMC `50,000` AR tasks이며, 핵심 metric은 `iou_top1`, `gt_rank_at_1`, `gt_rank_at_5`, `coverage_at_09`, CPC pairwise accuracy, GNMC AR violation이다.

9. Product AR cropper 재구축 및 proposal-free 평가 계획
   - AR별 label 생성은 이미 구현되어 있고 `arx2` 학습에 실제 사용됐다. 확인된 학습 label 경로는 `data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/training_labels/gaic_v2_train_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl`이며, train row는 `14,915`개다. target AR 분포는 `FREE 2,617`, `1:1 2,512`, `9:16 2,414`, `16:9 2,391`, `3:4 2,491`, `4:3 2,490`이다. val도 `1,136` row로 FREE 및 5개 fixed AR을 포함한다.
   - 따라서 "AR별 라벨이 전혀 없어서 fixed AR proposal이 학습되지 않았다"는 진단은 현재 `arx2`에는 맞지 않는다. 다만 `pubdistill`/public-score track은 FREE-only label로 학습됐으므로 fixed AR 결과를 제품 품질로 해석할 수 없다.
   - Product AR 평가 축은 official GAIC MOS candidate scorer 평가와 분리한다. official MOS 평가는 이미 주어진 annotation crop을 utility head에 넣는 candidate scorer 평가이고, proposal-free Product AR 평가는 모델이 candidate bank 없이 `FREE`, `1:1`, `3:4`, `4:3`, `16:9`, `9:16` crop을 직접 생성해 subject/safety/explanation까지 반환하는지를 본다.
   - Proposal-free metric은 `target_ar_log_error`, `target_ar_compatible_rate`, boundary/overflow rate, duplicate rate, `proposal_recall@K IoU0.5/0.7`, `best_iou@K`, pseudo-positive bag coverage, destructive-negative suppression, selected top-1 teacher utility, explanation applicability/fidelity로 구성한다. fixed AR에서는 target AR을 만족하지 못한 proposal은 ranking score와 무관하게 fail로 처리한다.
   - GAIC에서는 existing AR label row를 이용해 proposal/ranking/explanation smoke와 ablation을 수행한다. GAIC official MOS는 FREE-form candidate ranking 사후 평가로만 쓰고, fixed AR 품질 판단은 SSTK AR label, teacher audit, 시각화, FCDB/CPC/GNMC 보조 benchmark로 분리한다.
   - FCDB는 expert crop 단일 GT가 있으므로 `FREE` 또는 GT AR-conditioned task로 proposal recall/IoU를 본다. CPC는 pairwise preference가 있으므로 모델 top-1과 후보 쌍 ranking consistency를 본다. GNMC는 다중 AR task와 image-aspect 보정이 이미 구현되어 있으므로 fixed AR proposal compatibility, coverage, `gt_rank@K`, AR violation 회귀 guard로 사용한다.
   - 학습 향상 순서는 먼저 `arx2` Turbo-256/HQ-320/Balanced-288 중 best를 기준선으로 고정하고, `rank_320`/`hybrid384`의 top-return loss를 AR별 label row에 이식한다. 다음으로 SSTK pseudo-positive bag coverage, destructive-negative repair pairs, query specialization을 추가하고, FCDB/CPC/GNMC validation을 제품 외부 일반화 guard로 붙인다. 마지막으로 Product AR 전용 contact sheet를 GAIC/TestImages/FCDB/CPC/GNMC에서 같은 색상 규칙과 checklist panel로 생성한다.

10. Deployment 검증
   - TorchScript/ONNX export
   - FP16/INT8 quantization
   - mobile target latency, peak memory, model size 측정
   - on-device crop stability와 aspect-ratio policy 검증
   - 배포 후보는 SSTK-only protocol로 선정한 checkpoint 중 official MOS benchmark 사후 성능이 유지되는 모델로 제한한다. GAIC annotation을 이용해 배포 checkpoint를 직접 고르지 않는다.

## 14. 결론

현재 `SSTK-only Product Track`의 top-return best는 `mcn-next-hybrid384-tr06-s17-20260417-191315`와 `mcn-next-r320-tr070-s17-20260417-191315`이다. 둘 다 GAIC v2 official test 500장에서 Acc1/5 `0.444000`, Acc1/10 `0.650000`, top-1 MOS `3.822800`, MOS regret `0.409100`을 기록했다. 다만 `hybrid384`는 PCC/SRCC `0.447673`/`0.427922`로 더 안정적이고, `r320-tr070`은 PCC/SRCC `0.370447`/`0.343425`로 낮아 top-return 전용 후보로 제한한다. 균형형 product candidate는 `mcn-next-r320-tr06-s19-20260417-191315`이며 PCC `0.435216`, SRCC `0.405202`, Acc1/5 `0.444000`, Acc1/10 `0.646000`, Acc4/10 `0.491000`, Accw4/10 `0.383345`, top-1 MOS `3.821640`, MOS regret `0.410260`을 기록했다.

Full-option SSTK teacher official-Gc는 PCC/SRCC가 `0.514590`/`0.497884`로 대부분의 `rank_320` run보다 높지만, Acc1/5 `0.270000`, Acc1/10 `0.448000`, top-1 MOS `3.598460`, MOS regret `0.633440`으로 top-return 품질은 `rank_320` 계열보다 낮다. 따라서 이번 목표인 “SSTK teacher에 근접하고 top-return은 teacher보다 좋은 student”는 충족했다. 다만 teacher보다 낮은 PCC/SRCC는 후보 전체 score ordering과 calibration을 더 개선해야 함을 의미한다. batch32 run처럼 PCC/SRCC만 높이고 top-return이 무너지는 설정은 제품 후보에서 제외한다.

2026-04-20 추가 실험에서는 GAIC public cropper score를 crop score teacher로 쓰고 SSTK data factory는 explanation/fatal/safety metadata로 유지하는 public-score distillation track을 구현했다. 최초 best는 `mcn-pubscore-r320k96-tr06-s18-20260420-142309`였으나, 이후 `public_score_distill_v2`에서 `candidate_k=128`, raw score z-score blend target, teacher soft distribution distillation, top-4 coverage loss, public-score 전용 validation selection을 반영해 성능을 다시 끌어올렸다. 단일 checkpoint 기준 best는 `mcn-pubdistill-r320k128-zb035-s20-20260420-172325`이며 PCC `0.733951`, SRCC `0.726926`, Acc1/5 `0.506000`, Acc1/10 `0.710000`, Acc4/10 `0.605000`, Accw4/10 `0.489795`, top-1 MOS `3.897880`, MOS regret `0.334020`을 기록했다. 그러나 이 결과는 이제 MobileCropNet Product AR 성능으로 해석하지 않는다. GAIC benchmark scorer track은 학습 데이터 큐레이션 관점에서 teacher/direct scorer 성능을 극대화하는 별도 문제이고, MobileCropNet 공식 목표는 AR별 crop과 explanation을 안정적으로 반환하는 product model이다.

이 public-score 결과의 핵심 교훈은 `candidate_k` coverage와 loss alignment다. GAIC public score teacher 자체의 Test500 top-1 MOS는 `3.985460`이고 converter의 safe positive 평균 MOS도 `3.982421`로 높았지만, 기존 `rank_320` k=32 run은 official benchmark 전체 후보 scoring에서 Acc1/10 `0.546000`, top-1 MOS `3.732080`에 그쳤다. official candidate set 평균 86개에 맞춰 `candidate_k=96`으로 학습하자 PCC/SRCC와 Acc/top-return이 동시에 크게 개선됐고, k128 full-candidate coverage에 raw margin/listwise teacher distribution/top-k coverage를 추가하자 다시 모든 primary metric이 상승했다. 따라서 public/GAIC benchmark distillation에서는 backbone 확장보다 candidate coverage와 public teacher ordering을 직접 맞추는 loss 설계가 먼저다.

다만 이 결과는 `SSTK-only Product Track` 결론을 대체하지 않는다. GAIC public cropper score는 GAIC-trained external teacher이므로 상용 SSTK-only 제품 후보로 직접 간주하지 않는다. 제품 track 기준으로는 `sstk-r320-tr06-s19`, `sstk-hybrid384`, 그리고 AR별 label을 실제 학습한 `arx2` 계열이 baseline이고, public-score `r320k96/k128` 계열은 benchmark scorer teacher/curation 연구의 참고 결과로만 사용한다.

권장 HQ-320(`MobileNetV4-Hybrid-M`) 프로파일은 구현, full training, official MOS 평가, explanation PNG 시각화까지 완료했다. 다만 공식 GAIC MOS 기준 PCC `0.322124`, SRCC `0.313649`, Acc1/10 `0.372000`, top-1 MOS `3.531740`, MOS regret `0.700160`으로 기존 best보다 낮아 현재 checkpoint는 제품/논문 후보로 승격하지 않는다. HQ-320의 가치는 현재 단계에서는 “실행 가능한 고용량 프로파일과 설명 가능 출력 경로 검증”이며, 성능 개선 후보가 되려면 SSTK-only top-return 학습 신호와 selection metric을 먼저 보강해야 한다.

Detailed checklist explanation 학습은 세 profile에서 모두 end-to-end 검증됐다. 이제 모델은 `composition: ok` 같은 macro label뿐 아니라 `headroom_loose`, `lookroom_excessive`, `rule_of_thirds_strong`, detail score, why tag, teacher label detail을 함께 출력할 수 있다. 다만 official MOS 기준 성능은 `turbo_256` detailed run의 top1 MOS `3.716800`, MOS regret `0.515100`이 세 profile 중 가장 좋고, `balanced_288` detailed run의 PCC `0.455300`, SRCC `0.434693`가 세 profile 중 가장 높지만, 기존 제품 후보 best의 top1 MOS `3.804520`, MOS regret `0.427380`에는 미치지 못한다. 따라서 detailed explanation head는 정성 분석과 failure slice 진단에는 채택하되, 제품 후보 checkpoint 교체는 보류한다.

GAIC-v2 official split baseline label 완료본을 사용한 2026-04-17 재학습도 세 profile 모두 성공했다. 이 실험은 Train2636/Val200/Test500 label validation `ok` 상태의 최신 SSTK baseline label을 적용했고, 세 profile 모두 pretrained MobileNetV4 계열 backbone으로 학습했다. 이후 AR-constrained proposal을 letterbox content rect/rounding 보정까지 포함하도록 수정하고, mode-aware explanation applicability를 학습/추론/시각화에 반영한 `arx2` 재실험도 세 profile 모두 성공했다. `arx2` 기준 official MOS best는 Turbo-256이며 PCC `0.456028`, SRCC `0.446876`, Acc1/10 `0.542000`, top-1 MOS `3.737200`, MOS regret `0.494700`이다. 하지만 SSTK-only Product Track 재정렬 후 `rank_320`/`hybrid384`가 Turbo-256 arx2보다 핵심 top-return 지표에서 높으므로, 현재 후보 우선순위는 `rank_320 tr06-s19`와 `hybrid384`로 이동한다.

이번 요청에서 남겨둔 다음 연구도 수행했다. corrected `sstk_balanced_topreturn` 기준 `top_return_weight=0.6` seed sweep, `0.55/0.65/0.7` 좁은 loss ablation, LR/WD/warmup ablation, 352/384 해상도 확장, q24/turbo balanced-selection 재평가를 모두 완료했다. `turbo_256` 실험과 q24 재평가에서 확인했듯 SSTK validation top-return이나 replay top-1 하나만 높이는 방향은 official MOS top-return 일반화가 깨질 수 있다. 다음 초점은 GAIC annotation을 역주입하지 않으면서 SSTK-only validation protocol 자체를 더 견고하게 만들고, perturbation/repair pair, destructive-negative suppression, fixed-AR candidate distribution 보강을 loss/selection에 직접 연결하는 것이다. 현재 실험 규모에서는 2개 이상 GPU 사용은 필요하지 않으며, 병렬 실험은 `--core-count=1 --allow-spot` 단일 GPU workload 여러 개를 생성하는 방식이 가장 효율적이다.
