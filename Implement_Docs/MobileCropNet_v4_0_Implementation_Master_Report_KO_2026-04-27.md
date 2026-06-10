# MobileCropNet v4.0: 외부 prior 없는 온디바이스 이미지 크롭, 학습형 제안, 관계 인식 랭킹, 정책 실행, 피사체 위치 추정, 릴리스 게이트 배포

작성 기준일: `2026-04-27`
최종 정리: `2026-04-30 13:45 KST` 기준 strict release gate 통과 checkpoint는 아직 없으며, spot/bootstrap 종료로 끊긴 target-repair smoke는 안정 실행 경로에서 재개한다.

## 초록

MobileCropNet v4.0은 제품형 이미지 크롭을 단일 bounding-box 회귀 문제가 아니라, 목표 aspect ratio가 주어진 이미지에서 후보 생성, 후보 ranking, 위험 억제, subject-mode routing, policy action, subject localization, checklist 및 why-tag 설명을 동시에 출력해야 하는 온디바이스 의사결정 문제로 정의한다. 학습 시에는 teacher candidate bank, pairwise/listwise preference, route, subject prior, decision, checklist, rationale label을 모두 supervision으로 사용하지만, 배포 시에는 입력을 `image + target aspect ratio`로 제한한다. 이 no-prior runtime 제약 때문에 높은 candidate-bank replay score만으로는 배포 가능성을 보장할 수 없다.

현재 구현은 learned AR-conditioned proposal query, ROI-style candidate tokenization, RelationLite 또는 set-transformer ranker, utility/positive/risk head, route head, policy/decision/delta head, subject-box head, checklist/detail/why-tag head, runtime executor로 구성된다. 평가는 replay, GAIC official MOS, public equal-4 benchmark, Product-AR direct evaluation, subject-box report, runtime no-prior rerun, qualitative review, latency, strict release gate로 분리된다. 최신 산출물 기준 leaderboard는 문서 기준 `40`개 row를 추적하며, 주요 public/GAIC/equal-4 수치 공백 `3`개는 checkpoint source가 없는 invalid row다. strict product release gate 통과 row는 `0`개이고 deploy candidate는 `null`이다. 2026-04-30 13시대 target-repair 후속 `1101851`/`1101852`/`1101854`/`1101855`는 `Session terminated` 패턴으로 train gate 전 끊겼고, unique-tag 복구 `1101862`~`1101865`도 Start 후 bootstrap 단계에서 종료되어 모델 실행 증거가 없다. 따라서 같은 spot workload를 반복하지 않고 standard P3 또는 사용 가능한 interactive GPU에서 bounded smoke를 먼저 통과시키는 방식으로 재개한다.

가장 강한 완결 비교군인 `SSTK public / subjectprior_route_proposal_v1 / balanced_288`는 `equal4_raw_mean=0.760155`, `equal4_zscore=1.338924`, `direct_test_final_positive_hit_iou_0_5=0.956820`으로 crop alignment는 강하지만 `route_collapse_flag=true`, `route_balanced_accuracy=0.361347`, subject-box runtime calibration 미통과로 배포 후보가 아니다. 최신 route recovery smoke의 best proxy도 `best_route_acc=0.492083`, `best_subject_iou=0.475680`으로 release threshold를 넘지 못했다. 결론적으로 MobileCropNet v4.0은 end-to-end 학습/추론/평가 체계는 성숙했지만, no-prior runtime 안에서 route, subject box, generated proposal surface, policy/action, rationale consistency를 동시에 회복해야 제품 배포 후보가 된다.

## 1. 서론

제품형 image cropping은 보기 좋은 사각형 하나를 찾는 문제로 끝나지 않는다. 같은 이미지라도 `1:1`, `9:16`, `16:9`, `3:4`, `4:3`, `FREE` 요청에 따라 crop 후보와 실패 조건이 달라진다. 사람 portrait에서는 headroom, lookroom, face cut, joint cut이 치명적이고, object crop에서는 subject coverage와 scale이 중요하며, scene/copyspace에서는 horizon, context, blank region 보존이 더 중요하다. 따라서 최종 crop만 출력하는 모델은 제품이 실제로 실패하는 이유를 설명하거나, 안전하지 않은 후보를 억제하거나, route별 다른 policy를 적용하기 어렵다.

MobileCropNet v4.0의 핵심 전제는 cropper가 다음 출력을 함께 내야 한다는 것이다. 모델은 후보 proposal을 만들고, 각 후보의 utility와 positive/risk probability를 예측하며, 이미지의 subject mode route를 판단하고, 최종 action을 `keep_full`, `minimal_crop`, `crop` 중에서 선택한다. 또한 배포 환경에서는 teacher가 제공하던 subject prior를 사용할 수 없으므로, 모델 자신이 subject box와 valid confidence를 예측해야 한다. checklist 및 why-tag는 단순 설명 문구가 아니라 route/policy/crop이 같은 의미 체계 위에서 움직이는지 검수하는 release-safety surface다.

v4.0에서 특히 어려운 지점은 no-prior runtime이다. 학습 데이터에는 teacher candidate bank, subject support, Product-AR score, pairwise/listwise preference가 풍부하게 들어 있지만, 실제 온디바이스 추론에서는 teacher bank도, 외부 subject prior도, ground-truth subject box도 없다. 모델은 image와 target AR만 보고 proposal surface를 만들고, 그 위에서 ranking과 policy를 실행해야 한다. 그래서 이 보고서는 구현 상태를 나열하지 않고, 모델 구조, 데이터 계약, loss 설계, runtime executor, evaluation surface, release gate를 하나의 제품기술 시스템으로 설명한다.

이 문서의 기여는 네 가지다. 첫째, MobileCropNet v4.0이 푸는 ML 문제를 self-contained form으로 정식화한다. 둘째, 각 model head의 목적, 입력, 출력, loss, failure mode를 설명한다. 셋째, teacher-supervised training과 deployment-time no-prior inference의 차이를 분리한다. 넷째, 현재 deploy candidate가 없는 이유를 정량/정성 evidence와 release gate 논리로 해석하고 다음 실험 기준을 제시한다.

### 1.1 AI Application Specialist 평가표 대응 요약

본 문서는 `AI Application Specialist` 인증과제 평가표의 구현 중심 항목에 직접 대응한다. 핵심은 모델 구조만 제안한 것이 아니라, 데이터 adapter, model, train/eval/infer script, qualitative review pack, final leaderboard, strict release gate까지 제품형 AI application 개발 흐름을 코드와 artifact로 연결했다는 점이다.

| 평가 항목 | 본 문서의 대응 내용 | 근거 섹션/산출물 |
| --- | --- | --- |
| AI 기술/지식이해, AI 모델링 | static cropper와 public cropper 계열의 구조/학습 방식/한계를 비교하고, learned AR-conditioned proposal, relation ranker, route/policy/subject/explanation head를 결합한 v4 구조를 설계했다. | 1.6장, 2-11장, 14장 |
| 프로그래밍 & 툴활용 | Python/PyTorch로 `MobileCropNetV4BatchDataset`, `MobileCropNetV4`, train/eval/infer/release scripts를 구현했고, Cline SR을 코드 탐색/구현 보조/로그 분석/재현 정리에 활용했다. | 1.2장, 12장, 부록 C-F |
| AI App 개발 및 On-Device 활용 | 배포 입력을 `image + target_ar`로 제한한 no-prior runtime, exact target-AR postprocess, latency profile, lightweight/quality profile matrix를 구현했다. | 4장, 10장, 13장, 17장 |
| AI Governance | teacher prior와 runtime input을 분리하고, public score 오용 방지, safe/reject/risk gate, qualitative catastrophic bucket, durable artifact 기반 release decision을 유지한다. | 1.3장, 13-16장 |
| 과제 정의 | 제품형 cropper가 풀어야 할 target AR, subject preservation, risk, route, policy action, rationale 문제를 정의했다. | 1-2장 |
| 평가 및 성과 | public equal-4, GAIC official, Product-AR direct, head sanity, subject-box, qualitative, latency, release gate를 같은 checkpoint 기준으로 묶었다. | 13-20장, 부록 G |

### 1.2 AI 프로그래밍 및 사내 Tool 활용

본 구현은 Python/PyTorch 기반의 end-to-end AI application 코드로 구성된다. `src/mobilecropnet_v4/data.py`는 target AR vocab, subject mode vocab, decision vocab, checklist/why-tag vocab, letterbox transform, pairwise/listwise sidecar join, subject-box target mode를 tensor contract로 변환한다. `src/mobilecropnet_v4/model.py`는 AR-conditioned proposal query, candidate pooling/tokenization, RelationLite/set-transformer ranker, utility/positive/risk head, route head, policy/decision/delta head, subject-box head, checklist/detail/why-tag head, runtime baseline scoring을 통합한다. `src/mobilecropnet_v4/eval_utils.py`는 full/minimal baseline 생성, exact target-AR repair, no-prior executor, generated proposal selection, subject-valid policy를 담당한다.

학습과 실험 orchestration은 `src/scripts/train_mobilecropnet_v4.py`와 profile별 runner가 수행한다. 이 entrypoint는 `balanced_288`, `rank_320`, `plus_384`, `hybrid_384`, `q24_288_w075`, quality-first profile 등 profile preset을 지원하고, pairwise/listwise supervision, route-balanced sampler, decision/action source loss, subject-box valid target mode, route expert distillation, warm-start/freeze option, long-running checkpoint/status 저장을 포함한다. 학습 run은 `config.json`, `dataset_summary.json`, `train_status.json`, `metrics.json`, `summary.json`, `best.pt`, `latest.pt`를 남기도록 설계되어 있어 terminal session에 의존하지 않고 재개/감사할 수 있다.

평가 도구도 별도 코드로 분리했다. `evaluate_mobilecropnet_v4_product_ar_direct.py`는 no-prior Product-AR direct evaluation을 수행하고, `evaluate_mobilecropnet_v4_gaic_benchmark.py`와 `export_mobilecropnet_v4_public_benchmark_predictions.py`는 GAIC/public benchmark를 채운다. `audit_mobilecropnet_v4_release_heads.py`, `build_mobilecropnet_v4_subject_box_report.py`, `build_mobilecropnet_v4_qualitative_review_pack.py`, `build_mobilecropnet_v4_final_deployment_leaderboard.py`, `build_mobilecropnet_v4_release_gate_manifest.py`는 auxiliary head, subject-box, qualitative failure, leaderboard, release decision을 machine-readable artifact로 만든다.

구현 과정에서는 사내 AI 개발 보조 도구 `Cline SR`을 활용했다. 활용 범위는 대규모 코드베이스 탐색, 관련 script/metric 연결, 오류 로그와 partial artifact 분석, 반복 실험 runner 작성 보조, 테스트 범위 확인, 보고서와 artifact path 정리다. 다만 최종 판단은 Cline SR의 응답이 아니라 실행 가능한 코드, pytest, `train_status.json`, `metrics.json`, `summary.json`, checkpoint, release manifest, qualitative pack에 기반한다. 즉 AI 도구는 개발 생산성과 추적성을 높이는 보조 수단이고, release evidence는 재현 가능한 산출물로만 인정한다.

### 1.3 AI Governance 및 데이터/모델 사용 적합성

MobileCropNet v4.0은 학습 전용 teacher supervision과 배포 runtime input을 엄격히 분리한다. 학습 row에는 teacher candidate bank, subject prior, pairwise/listwise label이 들어오지만, 배포 시 모델은 `image + target_ar`만 입력으로 사용해야 한다. 이 원칙은 offline replay score가 좋아 보여도 teacher artifact에 의존하는 모델을 배포 후보로 오해하는 위험을 줄인다.

| Governance 항목 | 구현/평가상 적용 |
| --- | --- |
| 데이터 source 분리 | SSTK product semantics, GAIC/public benchmark ranking, UCTR/public cropper teacher score를 서로 다른 supervision source로 관리한다. |
| raw score 오용 방지 | public/GAIC score는 제품 정책 truth가 아니므로 safe conversion과 risk/fatal demotion 이후 distillation source로 사용한다. |
| 개인정보/인물 안전 | portrait/headroom/lookroom/face/joint cut, subject-valid mismatch, low subject IoU를 checklist/risk/qualitative bucket으로 추적한다. |
| train/test leakage 방지 | image-level split과 Product-AR task 단위 평가를 분리하고, benchmark split과 train artifact를 manifest로 추적한다. |
| 생성 산출물 관리 | 개념도나 보조 figure는 설명용으로만 쓰고, 성능/좌표/leaderboard/release 판단은 JSON artifact와 실제 crop overlay를 source of truth로 사용한다. |
| 배포 책임성 | strict release gate가 target-AR compatibility, risk, route balance, subject-box, qualitative catastrophic bucket, latency를 함께 확인한다. |

현재 release gate 통과 checkpoint가 없다는 점은 문서화된 blocker다. 이는 실패를 숨기는 것이 아니라 제품 배포 기준을 보수적으로 적용한 결과이며, public score나 direct hit가 높은 모델도 route/subject/action/qualitative gate를 통과하지 못하면 deploy candidate로 선언하지 않는다.

### 1.4 On-Device 제품화 보강 계획

현재 `balanced_288`과 `rank_320`은 latency 관점에서 on-device 후보권에 있으나, primary blocker는 아직 속도나 export가 아니라 route/subject/proposal/policy quality gate다. 따라서 최적화 순서는 quality gate 통과 checkpoint 확보 후 모바일/NPU 배포 검증으로 넘어가는 방식이 안전하다.

| 제품화 항목 | 현재 상태 | 후속 검증 기준 |
| --- | --- | --- |
| no-prior runtime | `boxes=None` 경로, generated proposal, full/minimal baseline, exact AR repair 구현 | teacher prior 없이 Product-AR direct와 qualitative gate 통과 |
| latency | RTX 3050 forward-only 및 A100 latency 측정 존재 | target device 또는 NPU delegate 기준 재측정 |
| operator coverage | 아직 별도 모바일/NPU export 검증 전 | unsupported op, dynamic shape, ROI/pooling 대체 경로 확인 |
| INT8/QAT | 아직 quality blocker 이후 단계 | INT8 drift, calibration set, route/subject head degradation 측정 |
| memory footprint | profile별 parameter/latency 일부 존재 | peak memory, activation size, CPU fallback 비용 측정 |
| pre/postprocess | letterbox, content-relative box, exact AR postprocess 구현 | platform image pipeline과 좌표계 일치 확인 |

이 순서는 성능을 숨기는 최적화보다 제품 실패 원인을 먼저 닫기 위한 것이다. route/subject/action gate가 통과한 단일 checkpoint가 확보되면, 같은 checkpoint/config/inference command를 기준으로 export, quantization, operator coverage, target device latency를 이어서 검증한다.

### 1.5 현업성과와 비즈니스 기여

MobileCropNet v4.0의 현업성과는 단일 crop score 상승만이 아니라 제품형 AI cropper 개발 절차를 구조화했다는 점에 있다. 후보 생성, ranking, subject preservation, route, policy action, risk, explanation을 함께 다루므로, 단순히 보기 좋은 crop을 고르는 모델보다 제품 검수와 fallback 정책에 필요한 정보를 더 많이 제공한다.

| 현업성과 관점 | 구체적 기여 |
| --- | --- |
| 제품 품질 개선 | target AR별 crop action, subject 보존, 인물/텍스트 안전, risk 억제를 하나의 runtime decision으로 묶는다. |
| 제품 차별화 | route, subject-box, checklist, why-tag를 함께 출력해 설명 가능한 crop 결과와 실패 분석을 제공한다. |
| 업무 효율화 | train/eval/direct/GAIC/public/qualitative/latency/release gate가 script와 artifact로 자동화되어 반복 실험 추적 비용을 줄인다. |
| 배포 리스크 감소 | strict release gate가 public score만 높은 모델을 차단하고, route collapse나 subject-valid mismatch 같은 제품 실패를 조기에 드러낸다. |
| 후속 확장성 | profile preset과 no-prior executor가 분리되어 있어 quality-first 탐색과 on-device profile 최적화를 같은 contract 아래에서 비교할 수 있다. |

따라서 현재 결론은 "배포 후보 없음"으로만 읽으면 안 된다. 더 정확한 해석은 end-to-end 학습/추론/평가/release 체계는 구현됐고, strict gate가 남은 제품 blocker를 식별한 상태라는 것이다. 다음 성과 목표는 route/subject/action internalization을 닫은 단일 checkpoint를 확보하고, 그 checkpoint를 모바일/NPU 최적화 단계로 넘기는 것이다.

### 1.6 AI 모델링 설계 채택 근거와 Public Cropper 비교

현재 v4 구조는 public cropper보다 모든 benchmark 숫자가 높기 때문에 선택한 구조가 아니다. 오히려 GAIC/CGS 같은 public cropper는 GAIC candidate-ranking 표면에서 MobileCropNet student보다 강한 경우가 많다. 예를 들어 GAIC v2 test 500 기준 public GAIC model은 `SRCC=0.845874`, top1 MOS `3.985300`이고, CGS도 `SRCC=0.799759`, top1 MOS `3.921020`이다. 초기 MobileCropNet v4 q24 profile은 같은 표면에서 `SRCC=0.368395`, top1 MOS `3.804520` 수준이었다. 이 차이는 public cropper가 해당 benchmark의 candidate scoring 문제를 잘 풀고 있음을 보여 준다.

그럼에도 v4를 별도로 설계한 이유는 제품 요구사항이 GAIC-style "후보 집합 안에서 MOS가 높은 crop을 고르는 문제"보다 넓기 때문이다. 제품 runtime은 `image + target_ar`만 입력으로 받아야 하고, 후보를 직접 생성해야 하며, `keep_full|minimal_crop|crop` action을 선택하고, route, subject-box, risk, checklist, why-tag까지 함께 출력해야 한다. Public cropper는 이 중 일부 축에서 강하지만, 전체 배포 계약을 만족하는 단일 온디바이스 모델로 그대로 사용할 수는 없다.

| 방법/계열 | 대표 구조와 학습 방식 | 강점 | 제품형 MobileCropNet 관점의 한계 | v4 설계에 반영한 부분 |
| --- | --- | --- | --- | --- |
| Static micro-bank cropper | full/center/thirds/phi/corner/wide 같은 고정 후보를 만들고 후보별 CNN feature를 scoring | 빠르고 export가 쉽고 baseline 비교가 명확함 | 후보 bank 밖의 좋은 crop을 절대 선택할 수 없고, target AR/subject-aware proposal을 학습하기 어려움 | candidate-first, full/minimal baseline, executor 개념은 유지하되 learned AR-conditioned proposal로 대체 |
| Saliency/face/pose/OCR heuristic cropper | saliency map, detector box, face/pose/OCR region을 기준으로 crop 후보 또는 hard rule 생성 | 주피사체/인물/텍스트 safety를 설명하기 쉽고 label factory에서 audit 가능 | detector failure에 취약하고 product action/ranking을 end-to-end로 학습하지 못함 | C1-C7 perception, subject-support, reject/checklist/risk label로 teacher supervision에 사용 |
| GAIC public cropper | grid/anchor 후보와 RoI/RoD-style candidate feature를 사용해 GAIC MOS candidate ranking을 학습 | GAIC official MOS ranking, top1 MOS, SRCC가 강함 | official candidate set scoring에는 강하지만 route, decision, subject-box, explanation, no-prior generated proposal을 제공하지 않음 | GAIC/public score distillation, GAIC benchmark, aesthetic/ranking teacher branch로 사용 |
| CGS public cropper | 후보 간 composition/graph-style 관계를 사용해 crop candidate를 scoring | 후보 관계와 composition ranking에 강하고 GAIC/CGS ensemble에서 보완 효과가 있음 | public crop quality signal은 강하지만 제품 safety/action/checklist contract가 없음 | RelationLite/set-transformer ranker 설계와 public cropper ensemble teacher branch에 반영 |
| CACNet 계열 단일 crop 회귀 | image에서 photographer-like crop 또는 anchor/regression output을 직접 예측 | 단일 crop이 빠르고 시각적 composition prior가 강함 | native candidate score가 없어 GAIC candidate-ranking 지표 계산이 어렵고, multiple valid crop/action/route를 표현하기 어려움 | 단일 bbox 회귀가 아니라 candidate/proposal/ranking 문제로 분해해야 한다는 반례로 사용 |
| DETR/Conditional-DETR 계열 proposal decoder | learned query가 spatial feature map에 cross-attention하면서 object/crop box를 직접 생성 | query slot 기반의 다중 후보 생성과 end-to-end matching에 강함 | crop은 object detection과 달리 target-AR hard constraint, baseline action, multiple-valid crop, route/subject/risk/policy 계약이 함께 필요하고, decoder attention 비용과 export 부담도 큼 | learned query slot 개념은 경량화해 차용하되, full transformer decoder는 proposal이 아니라 후보 set ranking 쪽에 선택적으로 배치 |
| A2-RL/S2CNet/UNIC 등 sequential/view-adjustment 계열 | crop window를 단계적으로 조정하거나 unbounded composition/view adjustment를 학습 | 반복 조정 또는 view expansion처럼 유연한 composition 탐색 가능 | runtime determinism, latency, target-AR hard constraint, safety/reject provenance, multi-head product output이 약함 | 후처리나 teacher 후보 생성 아이디어는 참고하되 deploy runtime 구조로는 채택하지 않음 |
| VLM/instruction cropper | 이미지와 자연어 intent를 보고 crop 후보/설명/checklist를 생성 | semantic intent와 rationale 생성에 강함 | 온디바이스 지연시간, 재현성, full-corpus materialization, deterministic candidate contract가 약함 | offline teacher/분석/멀티모드 보조 label에는 활용 가능하나 v4 runtime 입력에서는 제외 |
| UCTR/public hybrid teacher | FCDB/CPC/GNMC/GAIC supervision을 통합한 learned teacher 또는 routed hybrid scorer | equal-4 teacher ceiling이 높고 public/general geometry와 GAIC ranking을 분리해 볼 수 있음 | 단일 lightweight on-device student가 아니며, teacher candidate bank와 public scoring을 runtime에 그대로 둘 수 없음 | teacher distillation과 benchmark ceiling으로 사용하고, student는 no-prior internalization을 학습 |

이 비교에서 v4의 설계 결정은 다음과 같이 귀결된다.

첫째, v4는 단일 bbox regressor가 아니라 candidate/proposal/ranking 시스템이어야 한다. 같은 이미지에도 target AR, subject mode, copy-space, action policy에 따라 여러 정답 crop이 존재하므로, CACNet식 단일 crop 회귀나 static center/thirds 후보만으로는 제품 surface를 충분히 덮기 어렵다. 따라서 learned AR-conditioned proposal query가 runtime 후보를 직접 생성하고, replay 학습에서는 teacher candidate bank를 사용해 proposal recall과 ranking을 동시에 학습한다.

이때 proposal branch는 full Conditional-DETR transformer decoder가 아니다. DETR에서 유용한 "learned query slot으로 여러 후보를 낸다"는 개념은 가져오지만, 각 query가 spatial feature map에 반복 cross-attention하는 decoder 구조는 사용하지 않는다. 대신 image global feature, target AR embedding, image AR log로 만든 condition token을 learned proposal query에 주입하고, MLP box/logit/token head와 AR-constrained decoding으로 후보를 만든다. 이는 crop proposal을 object detection이 아니라 target-AR compatible candidate surface 생성 문제로 본 선택이다. 후보 간 관계와 self-attention capacity는 proposal 생성부가 아니라 ROI-style candidate tokenization 이후 RelationLite 및 optional set-transformer ranker에 배치한다.

둘째, ranking은 후보별 독립 score가 아니라 image-local set ranking이어야 한다. GAIC/CGS가 강한 이유도 crop 후보를 독립 좌표가 아니라 같은 이미지 안의 상대 선호로 본다는 점에 있다. v4는 이 장점을 받아들여 ROI-style candidate tokenization, RelationLite, set-transformer ranker, pairwise/listwise loss를 넣었다. 다만 public cropper의 score를 제품 truth로 그대로 쓰지 않고, risk/fatal demotion과 SSTK product metadata를 거친 safe distillation source로만 사용한다. 이 분해는 연산 배치 측면에서도 중요하다. transformer decoder를 proposal 단계에 두면 query 수, feature map token 수, decoder depth에 비례해 비용이 커지지만, 후보 수가 제한된 ranking 단계의 set interaction은 제품 latency와 export 안정성을 관리하기 쉽다.

셋째, 제품형 cropper에는 route/policy/subject/explanation head가 필요하다. Public cropper top1 crop이 시각적으로 좋아도 `portrait_single`을 `object_single`으로 routing하거나, subject valid를 틀리거나, `keep_full`이 맞는 이미지를 무조건 crop하면 제품 실패다. 그래서 v4는 utility head 외에 route head, policy decision/delta head, subject-box valid/box head, checklist/why-tag head를 둔다. 이 구조는 benchmark score를 높이기 위한 부가 장식이 아니라 release gate에서 failure family를 분리하기 위한 제품 계약이다.

넷째, teacher와 runtime을 분리해야 한다. UCTR stage3, public GAIC/CGS, production hybrid는 teacher ceiling과 distillation source로 유효하지만, 배포 시에는 teacher bank, public cropper, subject prior를 호출할 수 없다. 따라서 v4는 학습 중에는 강한 teacher supervision을 읽되, runtime forward에서는 `boxes=None` 경로로 generated proposal만 사용하도록 설계했다. 이 no-prior 분리는 offline replay 성능 착시를 막는 핵심 모델링 선택이다.

다섯째, v4는 deployability를 위해 profile을 분리한다. `balanced_288`, `rank_320`, `q24_288_w075`는 온디바이스 후보 profile이고, `plus_384`와 quality-first backbone 계열은 원인 분석과 상한 탐색용 profile이다. Public cropper는 benchmark나 teacher로는 강하지만, 제품 runtime에서는 profile별 latency, operator coverage, memory footprint, exact AR postprocess까지 함께 만족해야 한다. 따라서 v4 구조는 품질 탐색과 배포 profile을 같은 데이터/평가 contract 아래에서 비교할 수 있도록 설계됐다.

결론적으로 v4의 채택 동기는 "public cropper를 대체할 더 높은 GAIC scorer"가 아니라, public cropper의 강한 ranking signal을 흡수하면서도 제품 runtime이 요구하는 no-prior proposal, target-AR compatibility, action policy, subject safety, route/rationale consistency, latency profile을 하나의 student contract 안에 묶기 위함이다. 현재 release gate 미통과는 이 설계가 불필요하다는 뜻이 아니라, public cropper score만으로는 보이지 않는 route/subject/action internalization 문제가 아직 남아 있음을 보여 주는 evidence다.

### 1.7 GAIC/CGS 외 Public Cropper 후보 확장 검토와 실행 계획

지금까지 정량 비교는 이미 로컬 포팅이 끝난 `GAIC`, `CGS`, `CACNet` 중심이었다. 다음 단계에서는 `Implement_Docs/MobileCropNet_v4_0_bundle/MobileCropNet_v4_0.md`의 4.2절 대표 문헌 비교표를 기준으로, GTX 3050급 로컬 GPU에서 smoke/eval이 가능하고 온디바이스 후보로도 검토할 수 있는 public cropper를 확장한다. 여기서 4-benchmark는 현재 unified public benchmark 정의와 동일하게 `FCDB`, `CPC`, `GNMC`, `GAIC`를 뜻한다. dataset별 primary metric은 각각 `FCDB IoU@top1`, `CPC weighted pairwise accuracy`, `GNMC IoU@top1`, `GAIC primary = top1 MOS/SRCC/Accw4@10 조합`이다.

중요한 전제는 논문 수치와 본 프로젝트 4-benchmark 수치를 분리해서 읽어야 한다는 점이다. 논문 수치는 주로 `HCDB/FLMS`, `FCDB`, `FAT`, `GAICv1/v2`, `UGCrop5K`, `SACD`처럼 각 논문이 사용한 dataset과 metric 위에서 보고된다. 반면 본 프로젝트의 비교 표는 같은 staged task manifest, 같은 candidate generation/projection 규칙, 같은 metric runner를 거친 값이어야 한다. 따라서 아래의 논문 공개 수치는 후보 선별과 기대 수준을 정하는 참고값이고, release evidence는 전용 adapter로 재평가한 4-benchmark 산출물만 사용한다.

| 후보 | 공개 상태와 로컬 실행성 | 출력 형식 | 4-benchmark 정량 평가 가능성 | 논문/README 공개 수치 활용 | 우선순위 |
| --- | --- | --- | --- | --- | --- |
| `CACNet` | 로컬 코드와 weight가 이미 존재한다. `third_party/public_cropping_teachers/cacnet`, `weights/public_cropping_teachers/cacnet/best-FLMS_iou.pth`로 smoke forward 통과. | 단일 crop 회귀 | `FCDB/GNMC`는 IoU@top1 direct 가능, `GAIC`는 official candidate IoU projection만 가능, `CPC`는 preferred view와의 overlap 기반 근사 비교가 가능하다. native candidate score가 없으므로 SRCC/Accw 계열은 primary로 쓰지 않는다. | README 기준 원 논문 `FCDB IoU=0.718`, `FLMS IoU=0.854`, `KU-PCP Acc=88.2%`; 공개 PyTorch 재현값은 `FCDB IoU=0.702`, `FLMS IoU=0.841`, `KU-PCP Acc=88.4%`. | P0. 기존 GAIC projection에서 4-benchmark projection/equal-4까지 확장한다. |
| `S2CNet` / Spatial-Semantic Collaborative Cropping | 공식 PyTorch repo와 GAICv1/GAICv2 pretrained weight 링크가 공개되어 있다. RoI/RoD Align build가 필요하지만 구조상 GAIC/CGS adapter와 가장 가깝고, 논문도 `3.92M params`, `162.8 FPS`를 보고한다. | candidate scoring / graph-aware ranker | `GAIC` official candidate scoring에 직접 맞는다. `FCDB/CPC/GNMC`도 staged candidate windows를 score하는 방식으로 direct 비교가 가능하다. UGC/multi-object scene에 강하므로 Product-AR qualitative 비교 가치도 높다. | 논문 표 기준 `UGCrop5K SRCC=0.502, Acc5=60.8, Acc10=72.1`, `GAICv1 SRCC=0.793, Acc5=61.0, Acc10=78.1`, `GAICv2 SRCC=0.861, Acc5=64.0, Acc10=82.7`. | P1 최우선. 새 public cropper baseline로 가장 먼저 추가한다. |
| `LVRN` / Listwise View Ranking | 공개 repo와 pre-trained model 안내가 있으나 `PyTorch 0.4.1` 및 custom `roi_crop/roi_align/roi_pooling` build가 필요하다. 현재 Py3.10/Torch 환경과 직접 호환되지는 않는다. | candidate list top-1 probability | 포팅이 되면 candidate scorer로 4-benchmark direct 비교가 가능하다. 다만 legacy CUDA extension 포팅 비용이 S2CNet보다 크다. | 논문은 listwise ranking과 RoIRefine 장점을 보고하지만, 본 프로젝트 표에는 포팅 후 동일 benchmark 재평가값을 우선 사용한다. | P2. S2CNet 이후 legacy extension 포팅 가능성을 판단한다. |
| `Good View Hunting` VPN/VEN | `ViewProposalNet`과 `ViewEvaluationNet` repo 및 pretrained model 링크가 공개되어 있다. VPN은 `TensorFlow 1.3`, VEN은 PyTorch 기반이라 환경이 분리되어 있다. | VPN: proposal generator, VEN: view scorer | VPN 출력 crop은 `FCDB/GNMC` direct, `GAIC` projection 가능. VEN은 candidate scorer로 direct 평가 가능하지만, TF1/PyTorch split 때문에 바로 현재 runner에 넣기 어렵다. | 프로젝트/논문은 CPC 100만+ view pair와 VPN `75+ FPS`를 보고한다. MARS 논문 비교표 기준 VPN은 `75 FPS`, `HCDB IoU=0.837`, `FCDB IoU=0.716`, `FAT IoU=0.708`. | P2. 환경 포팅 또는 별도 compatibility container가 필요하다. |
| `A2-RL` | 공식 TF repo와 `vfn_rl.pk` pretrained 링크가 공개되어 있다. 다만 legacy TensorFlow 의존성이 Py3.10 로컬 환경과 맞지 않을 가능성이 높다. | sequential action 기반 단일 crop | 단일 crop이므로 `CACNet`과 같은 projection/direct 방식만 가능하다. policy/action trace는 흥미롭지만 MobileCropNet의 route/policy head와 apples-to-apples 비교는 아니다. | MARS 논문 비교표 기준 `4 FPS`, `HCDB IoU=0.818`, `FCDB IoU=0.695`, `FAT IoU=0.630`. | P2. 직접 baseline보다는 paper-number 보조 비교와 qualitative 참고가 현실적이다. |
| `UNIC` / Beyond Image Borders | 공개 PyTorch repo와 pretrained model 링크가 있다. Conditional-DETR 계열 encoder-decoder를 쓰고, inward crop보다 camera view adjustment/unbounded composition에 가깝다. | view adjustment + crop recommendation | bounded crop으로 clipping/projection하면 4-benchmark 진단은 가능하나, 원 방법론의 장점인 out-of-border adjustment가 본 benchmark에서 손실된다. | repo는 GAIC 기반 COCO-format annotation과 pretrained model을 제공한다. 논문 수치는 unbounded composition setting과 분리해 표기해야 한다. | P2. local smoke 후 qualitative/hard-case 비교용으로 제한한다. |
| `Human-centric Image Cropping` | 공식 repo가 공개되어 있고 CPC/GAICD/FCDB/FLMS 및 human bbox를 사용한다. README상 pretrained weight보다는 train/eval 코드와 human-centric sample/bbox 제공 중심이다. | human-aware candidate ranking | 전체 4-benchmark보다는 person slice 전용 비교가 적합하다. human bbox 또는 detector가 필요하므로 범용 public cropper baseline으로 바로 쓰기 어렵다. | 논문은 partition-aware feature와 content-preserving feature를 제안한다. 공개 repo만으로 즉시 full benchmark를 돌리는 것은 제한적이다. | P3. person/portrait qualitative slice에서 보조 비교한다. |
| `MARS` | 공개 code/weight는 확인되지 않았다. 다만 논문이 target AR-conditioned cropping을 명시적으로 다루므로 설계 비교 가치는 높다. | AR-conditioned direct crop | code/weight가 없으면 직접 4-benchmark 평가는 불가하다. 논문 수치만 보조 표에 넣는다. | CVPR 2020 논문 표 기준 MobileNetV2 profile `108 FPS`, `HCDB IoU=0.868`, `FCDB IoU=0.735`, `FAT IoU=0.710`. | Paper-only. AR token/condition 설계 근거로 인용한다. |
| `Rethinking Image Cropping` | 공개 code/weight는 확인되지 않았다. set prediction 관점은 v4 proposal 설계와 직접 관련된다. | set prediction crop proposal | 직접 평가 불가. code가 확인되기 전에는 설계 비교와 논문 수치 보조만 가능하다. | 다중 crop, learnable anchors, Hungarian matching, label smoothing을 보고한다. | Paper-only. proposal module 설계 근거로 사용한다. |
| `TransView`, `Spatial-aware Feature and Rank Consistency` | 대표 문헌이지만 공식 code/weight 링크는 확인되지 않았다. S2CNet 논문 표에는 일부 re-implementation 또는 paper-derived 비교값이 포함된다. | relation/candidate ranking | code/weight가 없으면 직접 4-benchmark 평가는 불가하다. S2CNet이 같은 family의 공개 실행 가능한 대체 baseline 역할을 한다. | 논문 공개 수치는 보조 표에만 싣고, 본 프로젝트 수치로 혼합하지 않는다. | Paper-only 또는 S2CNet 이후 재조사. |
| `GenCrop`, `ClipCrop`, `Cropper`, `ProCrop`, `AesCrop` | GenCrop은 code가 공개되어 있으나 outpainting/weak-supervision data generation 성격이 강하다. ClipCrop/Cropper/ProCrop/AesCrop은 VLM, retrieval, VMamba/decoder 등으로 로컬 GTX 3050 및 온디바이스 mainline baseline에는 부담이 크다. | teacher/data generation/HQ cropper | 4-benchmark local public cropper baseline보다는 offline teacher, hard-case adjudicator, pretraining source로 분리하는 편이 안전하다. | 논문 수치가 있더라도 runtime cost와 task definition이 다르므로 local public cropper 표에는 별도 그룹으로 둔다. | 제외 또는 teacher-only. |

이 검토를 기준으로 실제 추가 평가 후보는 두 묶음으로 제한한다. 첫 번째 묶음은 `CACNet` 4-benchmark projection 확장과 `S2CNet` 신규 adapter다. 이 둘은 현재 로컬 GTX 3050에서 실행 가능성이 높고, public cropper comparison 표에 새 정량 row를 추가할 수 있다. 두 번째 묶음은 `LVRN`, `VPN/VEN`, `A2-RL`, `UNIC`이다. 이들은 공개 code/model은 있지만 legacy framework 또는 task mismatch가 있어, 먼저 smoke 가능성만 확인하고 full equal-4 평가는 포팅 비용을 별도 산정한 뒤 진행한다.

구현 계획은 다음 순서로 고정한다.

1. Public cropper inventory manifest를 만든다. `third_party/public_cropping_teachers/{s2cnet,lvrn,vpn,ven,a2rl,unic,hcic}` 아래에 clone 후보를 분리하고, `artifacts/mobilecropnet_v4/public_cropper_extension_20260508/inventory/public_cropper_inventory.json`에 repo URL, commit hash, license, weight URL, framework, native output type, 실행 가능 판정을 기록한다.
2. Adapter contract를 공통화한다. `score_candidates(image, candidates, target_ar)`를 지원하는 candidate scorer와 `predict_boxes(image, target_ar)`를 지원하는 single/top-k crop generator를 분리한다. 산출 JSONL은 `method`, `image_id`, `dataset`, `target_ar`, `candidate_id`, `bbox_xyxy_norm`, `score`, `native_output_type`, `projection_mode`, `latency_ms`, `valid` 필드를 공통으로 가진다.
3. 기존 runner를 확장한다. `src/scripts/evaluate_public_croppers_gaic_v2.py`와 `src/scripts/export_public_cropper_benchmark_predictions.py`의 method registry에 `s2cnet`을 추가한다. `cacnet`은 이미 method choice에 있으므로 4-benchmark export에서 누락 없이 돌리는 쪽을 우선한다.
4. 로컬 smoke를 먼저 실행한다. 각 후보는 10-20장 image, batch size 1, CUDA memory peak, mean/p50 latency, output validity, candidate score finite rate를 `smoke_summary.json`으로 남긴다. S2CNet은 RoI/RoD extension build와 GAICv2 pretrained weight load를 smoke gate로 둔다.
5. 4-benchmark를 같은 manifest로 실행한다. `stage_public_benchmark_inputs.py`가 만든 FCDB/CPC/GNMC task manifest와 GAIC v2 official annotation을 사용하고, `run_public_benchmark_eval.py`, `evaluate_public_croppers_gaic_v2.py`, `build_unified_public_benchmark_report.py`로 `FCDB IoU`, `CPC weighted`, `GNMC IoU`, `GAIC primary`, `equal4_raw_mean`, `equal4_zscore`를 생성한다.
6. 정성 비교를 분리한다. `build_mobilecropnet_v4_public_cropper_qualitative_comparison.py`에 `method_group=public_extension`을 추가하고, subject cut, over-tight crop, background-dominant crop, target-AR mismatch, text/OCR risk, keep-full false crop 사례를 method별 gallery로 묶는다.
7. 최종 표는 두 층으로 작성한다. 하나는 본 프로젝트에서 직접 실행한 `same-run 4-benchmark table`, 다른 하나는 dataset/metric이 다른 `paper-reported auxiliary table`이다. 두 표를 섞어 rank를 매기지 않는다.

실행 명령은 기존 project Python을 사용한다. 신규 adapter가 추가된 뒤의 full run 예시는 아래와 같다.

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/export_public_cropper_benchmark_predictions.py --task_manifest_jsonl artifacts/public_benchmark/staged/task_manifest.jsonl --output_dir artifacts/mobilecropnet_v4/public_cropper_extension_20260508/public_benchmark_predictions --methods cacnet cgs gaic s2cnet --device cuda
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/evaluate_public_croppers_gaic_v2.py --output_dir artifacts/mobilecropnet_v4/public_cropper_extension_20260508/gaic_v2_test500 --methods cacnet cgs gaic s2cnet --device cuda
```

P0/P1 실행 결과 표는 아래 schema와 수치로 고정한다.

| method | status | native output | FCDB IoU | CPC weighted | GNMC IoU | GAIC top1 MOS | GAIC SRCC | GAIC Accw4@10 | GAIC primary | equal4 raw | qualitative note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `public_cropper_s2cnet` | completed_remote_gpu | candidate scorer with relation graph | 0.7097 | 0.8923 | 0.7924 | 3.9649 | 0.8395 | 0.6159 | 0.7827 | 0.794295 | public-weight diagnostic; original Faster-RCNN object graph 대신 deterministic candidate-derived graph node 사용 |
| `public_cropper_cacnet_projection` | completed_remote_gpu | single crop projection | 0.7575 | 0.8611 | 0.7459 | 3.7355 | 0.5245 | 0.3357 | 0.6075 | 0.743003 | projection-only; native GAIC candidate ranking score 없음 |
| `public_cropper_lvrn` | blocked_until_port | candidate scorer | - | - | - | - | - | - | - | - | PyTorch 0.4.1/custom CUDA port 필요 |
| `public_cropper_a2rl` | blocked_until_legacy_tf_runtime | single crop | - | - | - | projection-only | - | - | - | - | TF legacy runtime 필요 |
| `public_cropper_unic` | smoke_only | view adjustment | TBD | TBD | TBD | projection-only | - | - | - | TBD | unbounded composition setting mismatch |

P0/P1 실행은 local GPU를 쓰지 않고 원격 interactive A100 run(`10.14.231.16`)에서 수행했다. 산출물은 `artifacts/mobilecropnet_v4/public_cropper_extension_20260508_remote/gpu_interactive_full_20260508_093532/` 아래에 있으며, `status.json` 기준 `2026-05-08T00:35:34Z` 시작, `2026-05-08T01:24:11Z` 종료, 최종 `succeeded`다. Full grouped export는 `22,511`개 image group, `62,511`개 benchmark task row, `1,109,036`개 grouped candidate를 scoring했다. CACNet은 단일 crop regressor라 GAIC official 후보에 대한 native score가 없으므로 GAIC top1 MOS/SRCC/Accw4와 GAIC primary는 predicted crop에 가장 가까운 official candidate를 IoU projection으로 정렬한 diagnostic 값이다. S2CNet은 public GAICv2 weight를 사용했지만, 본 benchmark manifest에는 원 논문 protocol의 Faster-RCNN object box 5개가 없어서 candidate set에서 deterministic graph node 5개를 만든 adapter 결과로 표기한다.

근거 링크는 다음을 사용한다. GAIC는 `https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch`, CGS는 `https://github.com/bo-zhang-cs/CGS-Pytorch`, CACNet은 `https://github.com/bo-zhang-cs/CACNet-Pytorch`, S2CNet은 `https://github.com/suyukun666/S2CNet`, LVRN은 `https://github.com/luwr1022/listwise-view-ranking`, VPN/VEN은 `https://github.com/zijunwei/ViewProposalNet` 및 `https://github.com/zijunwei/ViewEvaluationNet`, A2-RL은 `https://github.com/wuhuikai/TF-A2RL`, UNIC은 `https://github.com/liuxiaoyu1104/UNIC`, Human-centric Image Cropping은 `https://github.com/bcmi/Human-Centric-Image-Cropping`, GenCrop은 `https://github.com/jhong93/gencrop`이다. MARS, Rethinking, TransView, Spatial-aware Feature and Rank Consistency는 현재 공개 code/weight가 확인되지 않았으므로 논문 링크와 paper-reported 수치만 auxiliary evidence로 유지한다.

## 2. 문제 정의

입력 이미지를 $I$, 목표 aspect ratio를 $r$라고 하자. 학습 시에는 teacher candidate bank $C_T(I,r)$와 positive target set $P(I,r)$가 주어질 수 있지만, 배포 시 모델은 이를 보지 못한다. MobileCropNet v4.0의 배포 모델은 AR-conditioned proposal set $B_q(I,r)$를 직접 생성하고, 각 후보에 대해 utility $u_q$, positive probability $p_q$, risk probability $\rho_q$, checklist/rationale $e_q$를 예측한다. 동시에 이미지 수준 route $m$, policy decision $d$, subject box $\hat{b}_{subj}$, subject valid probability $\hat{v}_{subj}$를 출력한다.

$$ f_\theta(I,r)=\{B_q,\ell_q,u_q,p_q,\rho_q,m,d,\Delta,\hat{b}_{subj},\hat{v}_{subj},e_q,b^\star\} $$

여기서 $B_q\in[0,1]^{Q\times4}$는 generated proposal box, $\ell_q$는 proposal objectness logit, $u_q$는 candidate utility, $p_q$는 positive likelihood, $\rho_q$는 unsafe/risk likelihood, $m$은 subject mode route, $d$는 action label, $\Delta$는 baseline-relative delta score, $e_q$는 checklist/detail/why-tag 묶음, $b^\star$는 runtime executor가 최종 선택한 crop이다.

deployment error는 한 항으로 줄일 수 없다. candidate bank 위에서 ranking이 좋아도 generated proposal이 좋은 crop을 만들지 못하면 배포는 실패한다. route가 틀리면 checklist와 policy가 틀리고, subject box가 틀리면 portrait/object crop에서 safety 실패가 발생한다.

$$ \mathcal{E}_{deploy}\approx\mathcal{E}_{proposal}+\mathcal{E}_{rank}+\mathcal{E}_{route}+\mathcal{E}_{subject}+\mathcal{E}_{policy}+\mathcal{E}_{rationale} $$

proposal recall은 generated proposal이 teacher positive set을 충분히 덮는지를 본다. replay score와 달리 no-prior runtime의 후보 생성 능력을 직접 측정한다.

$$ R_{prop}@K(\tau)=\frac{1}{N}\sum_{n=1}^{N}\mathbf{1}\left[\max_{q\le K,p\in P_n}\operatorname{IoU}(B_{nq},p)\ge\tau\right] $$

rank utility target은 teacher score, positive flag, risk flag를 함께 반영한다. 외부 public cropper score가 높아도 SSTK hard-reject 후보가 positive로 승격되지 않는 이유가 여기에 있다.

$$ u_i^\star=\alpha s_i^{teacher}+\beta y_i^{positive}-\gamma y_i^{risk} $$

runtime policy는 단순히 decision head의 argmax가 아니라 baseline full, minimal target-AR crop, generated proposal winner 사이에서 action을 고르는 실행 문제다.

$$ a^\star=\arg\max_{a\in\{\mathrm{full},\mathrm{minimal},\mathrm{crop}\}}S_a(I,r,B_q,u_q,\rho_q,d) $$

subject-box loss는 valid calibration과 box regression을 함께 포함한다. valid state가 틀리면 subject가 없는 scene/copyspace에서도 존재하지 않는 subject를 찾거나, portrait/object에서 subject box를 누락한다.

$$ \mathcal{L}_{subj}=\lambda_v\operatorname{BCE}(\hat{v}_{subj},v_{subj})+\mathbf{1}[v_{subj}=1](\lambda_1\|\hat{b}_{subj}-b_{subj}\|_1+\lambda_i(1-\operatorname{IoU}(\hat{b}_{subj},b_{subj}))) $$

route, policy, rationale consistency는 제품형 gate의 핵심이다. 예를 들어 `portrait_single`을 `object_single`로 route하면 headroom/lookroom checklist가 적용되지 않거나 잘못 표시되고, decision이 crop quality와 맞지 않으며, why-tag가 실제 crop failure를 설명하지 못한다.

$$ \mathcal{C}_{rpr}=\mathbf{1}[m=\hat{m}]\cdot\mathbf{1}[d=\hat{d}]\cdot\mathbf{1}[\operatorname{applicable}(e_q,m)]\cdot\mathbf{1}[\operatorname{safe}(b^\star,m)] $$

release gate는 위 항들을 제품 배포 기준으로 묶은 hard/soft decision이다. 현재 strict gate는 public equal-4, GAIC official, Product-AR direct, route balance, subject-box calibration, target-AR compatibility, risk, qualitative catastrophic bucket, latency를 함께 본다.

## 3. 기존 MobileCropNet에서 v4.0으로의 전환

legacy MobileCropNet은 fixed static micro-bank 위에서 후보를 scoring하는 구조였다. full, center crop, thirds, phi, corner context, wide context 같은 손으로 설계한 후보를 만들고, 각 후보를 feature map에서 pooling한 뒤 score/risk/route/decision을 예측했다. 이 설계는 빠르고 이해하기 쉬우며 고정 shape export가 쉽다는 장점이 있다.

그러나 static bank는 candidate recall ceiling을 만든다. 좋은 crop이 bank 밖에 있으면 ranker가 아무리 좋아도 선택할 수 없다. Product-AR 환경에서는 target AR이 다양하고, portrait/object/scene/copyspace의 안전 조건이 다르며, teacher가 생성한 positive manifold가 단순 center/thirds 슬롯으로 충분히 덮이지 않는다. legacy 구조는 candidate-first 원칙은 맞지만, 후보 공간이 고정되어 있고 subject-aware proposal을 학습하기 어렵다.

v4.0은 legacy의 장점 중 candidate-first와 baseline-relative policy는 유지한다. 모델은 여전히 후보별 score를 비교하고 full/minimal baseline과 crop 후보를 함께 고려한다. 대신 static bank를 배포 알고리즘의 중심으로 두지 않고, AR-conditioned learned proposal query가 runtime candidate surface를 생성한다. ranking은 후보 간 관계를 보는 RelationLite 또는 set-transformer로 확장되고, route/policy/subject/explanation head가 제품 계약으로 들어온다.

따라서 v4.0의 변화는 "bbox regressor로 단순화"가 아니라 "static bank replay 모델에서 learned proposal + relation ranker + policy executor 시스템으로 이동"이다. 학습 때 teacher bank를 읽는 이유는 teacher topology를 복제하기 위해서가 아니라, proposal/ranking/policy/explanation supervision을 분해해 student runtime에 내재화하기 위해서다.

## 4. 시스템 구조 개요

![MobileCropNet v4 runtime architecture](assets_mobilecropnet_v4_master_20260427/fig01_mobilecropnet_v4_runtime_architecture.png)

*그림 4-1. MobileCropNet v4.0 runtime architecture. 배포 시 입력은 image와 target AR뿐이며, teacher candidate bank와 subject prior는 training supervision으로만 남아야 한다.*

### 4.1 입력: image + target_ar만 사용

배포 입력은 정규화된 image tensor와 target AR token이다. 이미지 원본 비율은 `image_ar_log`로 제공되고, letterbox padding이 있는 경우 content rectangle이 geometry 계산에 사용된다. 학습 배치에는 후보 box와 target이 함께 들어오지만, runtime forward에서 `boxes=None`이면 모델은 generated proposals를 후보로 사용한다.

### 4.2 이미지 인코더와 조건화

encoder는 profile에 따라 custom depthwise backbone, MobileNetV4 계열, ConvNeXtV2 tiny, RepViT 계열을 사용할 수 있다. 출력 feature map은 candidate pooling에 쓰이고, global average pooled feature는 target AR embedding 및 image aspect ratio log와 결합되어 condition token을 만든다. 대표 shape는 image `[B,3,H,H]`, feature `[B,C,H_f,W_f]`, condition token `[B,D]`다.

### 4.3 AR 조건부 제안 query

proposal branch는 learned query `[Q,D]`를 condition token과 더한 뒤 proposal token, proposal logit, raw box를 만든다. raw box는 target AR이 `FREE`가 아닌 경우 content rectangle 안에서 target AR을 만족하도록 parameterization된다. 출력 shape는 proposal boxes `[B,Q,4]`, proposal logits `[B,Q]`다.

이 branch는 문서와 데이터 lineage에 남아 있는 `conditional-DETR batch` 표현과 혼동하면 안 된다. 학습 데이터 쪽의 conditional-DETR batch는 후보/target을 task 단위로 물질화한 label contract에 가깝고, 최종 runtime proposal module이 Conditional-DETR transformer decoder라는 뜻은 아니다. 현재 runtime 구조는 DETR-like learned query slot을 경량화해 사용하지만, query-to-feature cross-attention decoder layer는 없다. spatial feature map은 proposal decoder가 직접 attend하는 대상이 아니라, 이후 후보 box pooling과 candidate tokenization에서 다시 사용된다.

### 4.4 후보 토큰화와 박스 pooling

학습 replay에서는 label row의 candidate bank가 `boxes [B,K,4]`로 들어오고, runtime에서는 proposal boxes가 candidate로 대체된다. 각 box에 대해 inside/outer/context mask pooling을 수행하고, pooled feature, global feature, AR embedding, image AR log, box geometry를 결합해 candidate token `[B,K,D]`를 만든다. box geometry에는 normalized xyxy, cxcywh, area, aspect, baseline flag가 포함된다.

### 4.5 RelationLite / set-transformer 랭커

crop ranking은 후보별 절대 점수만으로 충분하지 않다. 같은 이미지 안에서 "조금 더 넓은 crop", "subject를 살짝 자르는 crop", "baseline full"의 상대 관계가 중요하다. RelationLite는 공개 논문 모델명을 그대로 가져온 것이 아니라, 본 프로젝트에서 후보 간 상대 관계를 가볍게 주입하기 위해 붙인 내부 경량 relation module 이름이다. 공개 cropper인 GAIC/CGS가 후보 집합 안의 상대 ranking을 중요하게 다루는 점은 설계 동기가 되었지만, 구현은 `src/mobilecropnet_v4/model.py`의 `_relation_refine()`에 있는 자체 MLP 기반 token refinement다.

구조는 후보 $i,j$의 token pair와 pair geometry를 relation MLP에 넣고, 후보 $i$로 들어오는 모든 유효 후보 $j$의 message를 평균한 뒤 residual로 더하는 방식이다. pair geometry에는 center offset, width/height scale ratio, IoU, area delta, 비교 대상 후보가 baseline인지 여부가 들어간다. 따라서 RelationLite는 "이 후보가 절대적으로 좋아 보이는가"보다 "같은 이미지의 다른 후보, 특히 full/minimal baseline 및 인접 crop과 비교했을 때 더 나은 선택인가"를 ranker가 보도록 만든다. 일부 profile은 이 경량 relation update 뒤에 set-transformer ranker를 추가해 후보 set 전체의 self-attention interaction을 한 번 더 처리한다.

### 4.6 다중 출력 head

candidate token 위에는 utility, positive, risk, macro, checklist class/applicability, detail score, why-tag head가 붙는다. global/context branch에는 route head가 있고, weighted candidate token과 baseline/best score를 결합한 policy branch에는 decision, delta, optional policy score head가 붙는다. subject-box head가 켜진 profile은 condition token에서 coarse subject box/valid를 예측하고, proposal-conditioned refinement를 통해 최종 subject box를 보정할 수 있다.

### 4.7 런타임 실행기

runtime executor는 모델의 decision과 proposal winner를 그대로 노출하지 않는다. 먼저 full baseline과 minimal target-AR baseline을 내부 생성하고, generated proposal top-M을 utility/risk로 rerank한다. decision이 `keep_full` 또는 `minimal_crop`이면 baseline lane이 선택될 수 있고, `crop`이면 reranked proposal이 선택된다. 마지막에는 exact target-AR repair와 bounds clamp를 수행한다. experimental baseline decision gate는 검증됐지만 direct quality regression이 확인되어 기본값은 `off`다.

### 4.8 학습 전용 teacher supervision과 배포 시 외부 prior 없는 추론

training row에는 teacher subject prior와 candidate bank가 들어오지만, 이는 runtime input이 아니다. subject prior는 route/proposal/policy를 학습시키는 supervision 또는 optional research conditioning으로 쓰이며, 배포 계약에서는 모델이 예측한 subject box만 사용할 수 있다. 이 구분이 무너지면 offline replay 성능은 좋아 보여도 실제 제품 추론은 teacher artifact에 의존하는 모델이 된다.

## 5. 데이터 어댑터와 학습 row 계약

![Training contract and losses](assets_mobilecropnet_v4_master_20260427/fig02_training_contract_and_losses.png)

*그림 5-1. MobileCropNet v4 학습 row는 단일 bbox target이 아니라 후보, ranking, route, decision, subject-box, explanation target을 함께 제공한다.*

MobileCropNet v4의 학습 row는 하나의 `image_id + target_ar` task다. row에는 image path, baseline candidate, matching positive targets, candidate pool, ignored/overflow candidates, route target, decision target, subject prior box, checklist/rationale labels가 들어온다. pairwise JSONL과 listwise JSONL은 같은 `(image_id, target_ar)` key로 join되어 explicit preference와 teacher soft distribution을 제공한다.

candidate role은 모델의 failure mode를 직접 결정한다. `baseline`은 full 또는 minimal crop과 policy를 비교하기 위한 기준 슬롯이다. `matching_targets`는 proposal recall과 positive ranking의 anchor다. `candidate_pool`은 safe positive, soft positive, hard negative, unsafe negative를 함께 담는다. `ignored_candidates`와 `overflow_candidates`는 학습 surface에 포함할 수 있지만 기본 weight로 쓰면 noise가 커질 수 있으므로 옵션으로 제어된다.

| source field | semantic meaning | tensor target | model head | loss | missing/failure mode |
| --- | --- | --- | --- | --- | --- |
| `image_path`, image size | 입력 이미지와 letterbox geometry | `image`, `image_ar_log`, `letterbox_content_box` | encoder, proposal geometry | all losses | AR repair와 box coordinate가 틀어짐 |
| `target_ar` | 요청 aspect ratio | `target_ar_id` | AR embedding, proposal, executor | proposal, direct eval | target-AR incompatible crop 증가 |
| `baseline` | full/minimal 대비 기준 후보 | `candidate_is_base`, baseline slot | utility, policy | action consistency, decision | keep/minimal/crop policy 식별 불가 |
| `matching_targets` | teacher positive crop set | `positive_boxes`, `positive_scores`, positive candidate slots | proposal, utility, positive | proposal recall, score, positive | generated proposal recall과 top positive 학습 약화 |
| `candidate_pool` | ranking 후보와 hard/risk negatives | `boxes`, `score_target`, `risk_target`, `candidate_weight` | utility, positive, risk | score, listwise, pairwise, risk | replay ranker가 안전/위험 구분을 못함 |
| `ignored_candidates` | 낮은 신뢰 또는 보류 후보 | optional candidate slots | utility/risk | optional weighted rank | 무분별 포함 시 label noise 증가 |
| `overflow_candidates` | 후보 bank 밖 overflow/hard negatives | optional candidate slots | risk, utility | risk/top1 risk | unsafe top1 suppression 약화 |
| pairwise sidecar | 후보 쌍 선호 | `pair_i`, `pair_j`, `pair_label`, `pair_weight` | utility | explicit pairwise | local ordering이 teacher preference와 어긋남 |
| listwise sidecar | teacher soft distribution | `teacher_soft_target` | utility | teacher distill, listwise | score scale만 맞고 분포 imitation이 약함 |
| `routing.subject_mode_id` | subject mode route | `route_target` | route | route CE/focal/balanced CE | route collapse와 checklist applicability 오류 |
| `routing.subject_prior_bbox_norm_xyxy` | teacher subject support | `subject_prior_box`, `subject_box_target` | subject box, optional prior encoder | subject-box, proposal-subject | no-prior subject localization 학습 약화 |
| subject reliability flags | subject box 신뢰도 | `subject_prior_reliability`, `subject_box_weight`, `subject_box_valid` | subject box, route/proposal | valid BCE, weighted box loss | subject-present bias 또는 background false positive |
| `decision_target` | keep/minimal/crop policy | `decision_target`, `decision_score_target`, `delta_target` | policy decision, delta, score | decision, delta, policy score | decision accuracy가 collapse label에 속음 |
| checklist labels/scores | crop quality 세부 판단 | `checklist_class_target`, `detail_score_target` | checklist/detail | CE, SmoothL1 | 설명 head가 generic label로 평탄화 |
| why/reject tags | rationale와 unsafe reason | `why_tag_target`, `why_tag_applicable` | why-tag, risk | multilabel BCE, risk | model rationale warning과 risk miss 증가 |

adapter의 중요한 설계는 applicability mask다. 모든 checklist가 모든 route에 적용되지 않는다. portrait에서는 headroom/lookroom/face_cut/joint_cut이 의미 있지만, object/scene/copyspace에서는 person-only tag가 hallucination이 될 수 있다. 따라서 route별 applicability target이 class/detail/why loss에 반영된다.

## 6. 모델 구조 상세

### 6.1 인코더

encoder는 입력 image tensor를 feature map과 global feature로 변환한다. lightweight profile은 MobileNetV4 Conv-M 또는 custom depthwise stack을 쓰고, ranking profile은 set-transformer ranker와 더 큰 token dimension을 사용한다. quality/upper-bound profile은 입력 해상도, 후보 수, token dimension, pretrained backbone을 함께 키워서 no-prior internalization의 상한을 탐색한다. profile은 단순 입력 크기 preset이 아니라 backbone, candidate coverage, proposal query 수, ranker capacity, 배포 목적을 함께 고정하는 실험 단위다.

| profile | backbone / pretrained weight | input | candidate_k | proposal_q | token_dim | ranker | 특징과 용도 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `q24_288_w075` | custom depthwise, random init | 288 | 24 | 24 | 128 | RelationLite | 초기 제품 후보 및 ablation 기준. 외부 pretrained 없이도 빠르게 학습/추론되지만, 표현력과 public benchmark ceiling이 낮다. |
| `turbo_256` | `mobilenetv4_conv_small.e3600_r256_in1k`, ImageNet-1k pretrained | 256 | 16 | 16 | 128 | RelationLite | latency 최우선 profile. 후보 수가 적어 빠르지만 candidate coverage가 작아 missed proposal risk가 커진다. |
| `balanced_288` | `mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k`, ImageNet-12k pretrain + ImageNet-1k finetune | 288 | 24 | 24 | 128 | RelationLite depth 1 | 품질/속도 균형 profile. 대표 SSTK public winner가 이 profile에서 나왔고, 온디바이스 후보군의 기본 비교점이다. |
| `hq_320` | `mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k`, ImageNet-12k pretrain + ImageNet-1k finetune | 320 | 32 | 32 | 192 | set-transformer depth 2 | 설계상 HQ profile. Hybrid backbone과 set ranking을 검증했지만, 현재 label/selection에서는 큰 backbone이 자동 개선을 만들지는 못했다. |
| `rank_320` | `mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k`, ImageNet-12k pretrain + ImageNet-1k finetune | 320 | 32 | 32 | 192 | set-transformer depth 2 | ranking capacity 강화 profile. Balanced보다 후보 coverage와 set interaction이 크고, top-return/selection 실험의 주력 profile이다. |
| `hybrid_384` | `mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k`, ImageNet-12k pretrain + ImageNet-1k finetune | 384 | 48 | 48 | 256 | set-transformer depth 3 | 384 해상도와 Hybrid-M backbone을 쓰는 heavy on-device 후보. ConvNeXtV2-Tiny weight availability 이슈가 있을 때 `plus_384` 대체 profile로 사용했다. |
| `plus_384` | current preset: `convnextv2_tiny.fcmae_ft_in22k_in1k_384`; 일부 대표 run config는 `convnextv2_tiny` alias로 저장 | 384 | 48 | 48 | 256 | set-transformer depth 3 | 해상도와 후보 coverage를 키운 quality/diagnostic profile. subject/proposal 표현력은 커지지만 VRAM, latency, NaN/불안정성 risk가 증가한다. |
| quality-first 계열 | ConvNeXtV2-B/L/H, SwinV2-B/L, EVA02-B/L 등 large pretrained backbone | 384-512 | 64-96 | 64-96 | 384-768 | set-transformer depth 4 | 온디바이스 제약을 완화한 상한 탐색군. route/subject/proposal/action gate를 동시에 닫는지 확인하기 위한 연구용 profile이며, 기본 배포 후보로 보지 않는다. |

이 표에서 중요한 점은 `balanced_288`, `rank_320`, `plus_384`가 단순히 입력 크기만 다른 모델이 아니라는 것이다. `balanced_288`은 같은 MobileNetV4 Conv-M backbone을 쓰면서 RelationLite로 비용을 낮춘 균형점이고, `rank_320`은 동일 backbone에 set-transformer ranker와 더 큰 token dimension을 얹어 후보 간 상대 ranking을 강화한다. `plus_384`는 backbone family까지 ConvNeXtV2 계열로 바꾸고 후보 수를 48개로 늘려 coverage와 표현력을 키우지만, 학습 안정성과 배포 비용을 더 엄격히 봐야 한다.

### 6.1.1 논문 기재용 학습 환경과 자원

대표 학습 실험은 사내 MLP GPU workload에서 단일 NVIDIA A100-SXM4-80GB GPU를 사용해 실행했다. 실험은 `--core-count=1`, spot workload `--allow-spot`을 기본으로 했고, workload image는 `sr-ar-interactive-media-exp/jy-cropping-260415-tfs4.57.6-rsync`를 사용했다. 학습 코드는 PyTorch 기반 `src/scripts/train_mobilecropnet_v4.py`이고, CUDA 환경에서는 AMP fp16을 켰으며, multi-GPU data parallel은 사용하지 않았다. 현재 모델 규모에서는 단일 A100의 VRAM 여유가 충분했고, 병렬 실험은 하나의 multi-GPU job보다 독립적인 single-GPU workload 여러 개를 동시에 띄우는 방식이 더 효율적이었다.

아래 표는 `SSTK public / subjectprior_route_proposal_v1` 계열의 대표 profile run을 기준으로 정리한 학습 자원이다. 세 run 모두 같은 train/val split을 사용했고, train row는 `39,002`, val row는 `4,947`, batch size는 `16`, 계획 epoch은 `8`이다. `train elapsed`는 `train.log`의 마지막 epoch `elapsed_sec` 기준이며, queue 대기, workload bootstrap, post-eval, public benchmark, visualization 생성 시간은 제외한다. `expected runtime`은 MLP workload 생성 시 배정한 최대 실행 예산이며 실제 학습 시간과 동일한 값이 아니다.

| profile | representative run / run id | train elapsed | best epoch | expected runtime | GPU | avg train GPU util | max train GPU util | max nvidia-smi mem | max CUDA allocated | avg train throughput |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `balanced_288` | `mcn-sstk-public-subjasync-r5bal-balanced-288-20260423-034145` / `1097827` | 51.5 min | 6 | 12 h | A100-SXM4-80GB x1 | 20.43% | 33% | 1,949 MB | 866.9 MB | 96.90 samples/s |
| `rank_320` | `mcn-sstk-public-subjasync-r1-rank-320-20260423-025404` / `1097779` | 60.9 min | 8 | 12 h | A100-SXM4-80GB x1 | 22.31% | 37% | 2,235 MB | 1,124.0 MB | 93.88 samples/s |
| `plus_384` | `mcn-sstk-public-subjasync-r5plus-plus-384-20260423-034246` / `1097828` | 46.3 min | 1 | 14 h | A100-SXM4-80GB x1 | 50.62% | 79% | 6,517 MB | 5,682.6 MB | 75.25 samples/s |

`plus_384`는 자원 사용량이 가장 크고 throughput이 가장 낮다. 또한 대표 run에서는 epoch 1이 best였고 이후 epoch에서 NaN이 발생해 profile 자체의 capacity가 품질 향상으로 안정적으로 이어지지 않았다. 반대로 `balanced_288`과 `rank_320`은 VRAM 사용량이 2.3GB 이하로 낮고 학습이 안정적이었다. 따라서 논문/보고서에서는 `balanced_288`을 deployable balanced baseline, `rank_320`을 ranking-capacity profile, `plus_384`를 high-capacity diagnostic profile로 분리해 기술하는 것이 정확하다.

### 6.2 목표 AR 조건화

target AR는 vocab `FREE`, `1:1`, `9:16`, `16:9`, `3:4`, `4:3`로 tokenized된다. AR embedding과 image aspect ratio log는 global feature와 concat되어 condition token을 만든다. 이 token은 proposal branch, subject-box head, route/policy branch의 공통 condition이다.

### 6.3 제안 query branch

proposal branch는 learned query를 condition token과 더해 proposal token을 만들고, box head와 logit head를 통과시킨다. query 수 $Q$는 profile별로 `16`, `24`, `32`, `48`이 쓰인다. proposal logit은 objectness-like signal이며, direct runtime에서는 proposal top-M selection과 rerank 후보 구성에 사용된다.

구현상 핵심 흐름은 다음과 같다.

1. backbone feature를 global average pooling해 image global feature를 만든다.
2. target AR id는 embedding으로 변환하고, 원본 이미지 비율은 `image_ar_log` scalar로 넣는다.
3. image global feature, target AR embedding, image AR log를 concat한 뒤 `cond_proj` MLP로 condition token을 만든다.
4. trainable learned proposal query `[Q,D]`를 batch 차원으로 확장하고 condition token을 더한다.
5. MLP 기반 proposal token head, box head, logit head가 각각 candidate token, raw box, proposal logit을 만든다.
6. fixed target AR에서는 raw box를 그대로 쓰지 않고 AR-constrained decoder가 content rectangle 안의 target-AR compatible box로 변환한다.

따라서 proposal branch를 "MLP-only crop regressor"로 이해하는 것도 충분하지 않고, "Conditional-DETR decoder"로 이해하는 것도 정확하지 않다. 정확한 표현은 "AR-conditioned learned proposal query slots with MLP heads and AR-constrained decoding"이다. query slot은 DETR에서 온 다중 후보 생성 아이디어를 닮았지만, Conditional-DETR처럼 multi-layer transformer decoder가 spatial feature map에 cross-attention하는 구조는 아니다.

### 6.3.1 Proposal 단계에서 full transformer decoder를 쓰지 않은 이유

첫째, crop proposal의 목표는 object detection의 instance localization과 다르다. Object detector는 query가 feature map에서 물체 instance를 찾아야 하지만, MobileCropNet의 proposal은 target AR, content rectangle, baseline action, subject preservation, composition margin을 만족하는 후보 surface를 만들어야 한다. 좋은 crop은 하나의 물체 box와 일치하지 않을 수 있고, 같은 이미지에서도 target AR와 route에 따라 여러 valid crop이 공존한다. 이 문제에서는 proposal head가 모든 판단을 끝내기보다, 안정적인 후보를 만들고 ranker/policy가 후보 집합 안에서 선택하도록 분해하는 편이 label 구조와 맞다.

둘째, target-AR hard constraint가 proposal search space를 크게 줄인다. fixed AR에서는 width와 height를 자유롭게 회귀하는 것이 아니라, 중심과 scale을 content rectangle 안에서 target AR에 맞게 fit하는 문제가 된다. 이때 heavy decoder가 spatial feature map을 반복 attend하지 않아도, backbone의 global/spatial representation과 AR condition만으로 후보 surface를 만들 수 있다. Spatial detail은 proposal 이후 mask pooling과 candidate tokenization에서 후보별로 다시 읽는다.

셋째, transformer capacity는 후보 생성보다 후보 비교에 더 직접적으로 필요하다. Crop 품질은 후보 하나의 절대 feature보다 같은 이미지 안의 후보 간 상대 관계에 크게 좌우된다. 그래서 본 구현은 proposal을 경량화하고, 후보 box pooling 이후 RelationLite와 optional set-transformer ranker로 full/minimal baseline, generated crop, 인접 crop 간 interaction을 처리한다. 즉 transformer를 배제한 것이 아니라, 제품 품질 판단에 더 직접적인 ranking 위치에 배치한 것이다.

넷째, 학습 target의 성격도 full DETR decoder와 완전히 맞지 않는다. 학습 row에는 object detection GT처럼 명확한 단일 instance target이 아니라 teacher candidate bank, positive/matching target, ignored/overflow candidate, pairwise/listwise preference, Product-AR score, route/action/checklist signal이 함께 들어간다. Proposal decoder가 teacher 후보 분포를 강하게 모방해 proposal recall을 올리더라도, final action/source, subject-valid calibration, route/rationale consistency가 함께 좋아진다는 보장은 없다. 실제 gate blocker도 단순 proposal recall보다 route, subject-valid, source/action, qualitative safety 쪽에서 더 많이 발생했다.

다섯째, 모바일 배포 비용과 export 안정성도 중요한 제약이다. Conditional-DETR decoder를 proposal branch에 넣으면 비용은 대략 query 수, feature map token 수, decoder depth에 비례해 증가한다. target AR을 여러 개 평가하거나 candidate 수를 늘리는 제품 환경에서는 latency, memory bandwidth, operator coverage, NPU 변환 안정성이 모두 부담이 된다. 반면 현재 구조는 proposal branch를 `query + condition token + MLP heads`로 유지하고, 필요한 set interaction만 제한된 candidate token 위에서 수행하므로 배포 profile을 관리하기 쉽다.

따라서 full transformer decoder를 쓰지 않은 것은 단순 속도 절감만의 선택이 아니다. 본 설계는 crop 문제를 "spatial object decoding"이 아니라 "AR-compatible proposal surface generation + image-local candidate ranking + product policy execution"으로 분해한 결과이며, 성능을 높일 여지가 있는 transformer 연산은 proposal 생성부보다 후보 ranking/decision surface에 배치했다. 향후 transformer decoder proposal variant를 실험한다면 proposal recall@K뿐 아니라 generated alignment, final positive hit, subject IoU, subject-valid mismatch, crop-action, baseline preserve, qualitative failure, latency를 함께 통과해야 한다.

### 6.4 AR 제약 제안 parameterization

target AR가 fixed이면 proposal raw output은 content rectangle 내부의 center/scale로 해석되고, width/height는 target AR와 image AR를 반영해 fit된다. `FREE`이면 일반 normalized xyxy box로 동작한다. 이 설계는 generated proposal이 처음부터 target-AR compatible surface에 가깝게 나오도록 만든다.

### 6.5 후보 박스 pooling / ROI 토큰화

각 candidate box는 feature map에서 inside, border, wider context 세 영역으로 mask pooling된다. 단순 crop 내부 texture만 보는 것이 아니라 box 주변 context 손실도 볼 수 있게 하기 위해서다. pooled feature는 global feature, AR embedding, image AR, box geometry와 함께 candidate encoder에 들어간다.

### 6.6 기하 특징

box geometry는 `[x1,y1,x2,y2,cx,cy,w,h,area,aspect,is_base]` 형태다. baseline flag는 policy/action consistency에서 중요하다. 같은 box score라도 baseline full/minimal과 generated crop proposal은 제품 실행 의미가 다르기 때문이다.

### 6.7 관계 인식 랭커

RelationLite는 공개 표준 architecture 명칭이 아니라 MobileCropNet v4 구현에서 사용하는 자체 경량 관계 인식 랭커다. 목적은 crop 후보를 독립적으로 scoring하기 전에, 후보 $i$가 같은 이미지의 후보 $j$들과 어떤 상대 관계에 있는지를 candidate token에 반영하는 것이다. pair feature $g_{ij}$는 상대 center offset, width/height scale ratio, IoU, area delta, baseline flag로 구성된다. 구현상으로는 $t_i$, $t_j$, $g_{ij}$를 concat해 relation MLP $\phi$에 넣고, valid 후보의 message 평균을 $t_i$에 residual로 더한 뒤 layer normalization을 적용한다.

$$ t'_i=\operatorname{LN}\left(t_i+\frac{\sum_j \phi(t_i,t_j,g_{ij})v_j}{\sum_j v_j}\right) $$

이 방식은 full self-attention보다 단순하지만, crop ranking에 필요한 핵심 비교 축을 직접 넣는다. 예를 들어 두 후보가 비슷한 위치에 있어도 하나는 subject를 자르고 다른 하나는 보존하는지, generated proposal이 full/minimal baseline보다 실제로 나은지, 너무 tight한 후보보다 넓은 후보가 제품 action에 맞는지를 token 수준에서 비교할 수 있다. set-transformer profile은 RelationLite 이후 후보 set 전체의 interaction을 self-attention으로 다시 처리하는 선택적 확장이다.

### 6.8 Utility / positive / risk head

utility head는 최종 rank utility를, positive head는 teacher positive와의 일치 가능성을, risk head는 hard-negative/unsafe 후보 가능성을 예측한다. 좋은 cropper는 utility가 높은 crop을 고르는 것뿐 아니라, 위험한 crop이 top1으로 올라오는 것을 억제해야 한다.

### 6.9 Route head

route head는 subject mode를 예측한다. 현재 vocab은 `background_texture_copyspace`, `object_multi`, `object_single`, `other_ambiguous`, `portrait_group`, `portrait_single`, `scene_general`이다. route는 auxiliary classification이 아니라 candidate policy, checklist applicability, action decision, qualitative failure slicing의 prior다.

### 6.10 Policy / decision / delta head

policy branch는 global condition, attention-weighted candidate token, baseline score, best candidate score, optional subject prior token을 결합한다. decision head는 `keep_full|minimal_crop|crop`을 예측하고, delta head는 baseline 대비 개선/악화 signal을 회귀한다. optional policy score head는 continuous score를 threshold-like logits로 변환해 decision logits에 더할 수 있다.

### 6.11 Subject-box head

subject-box head는 condition token에서 coarse subject box와 valid logit을 예측한다. patch 이후에는 content-relative coordinate option, proposal-conditioned 2-stage refinement, subject proposal alignment loss, center/size/aspect/CIoU loss가 추가됐다. 목적은 teacher subject prior를 배포 입력에서 제거하고, 모델 내부 subject localization으로 route/proposal/policy를 지탱하는 것이다.

### 6.12 Checklist / detail score / why-tag head

explanation 계열은 단일 방식으로 통일된 head가 아니라 label classification과 score regression이 병렬로 존재하는 hybrid 구조다. checklist class head는 각 checklist group의 categorical state를 예측하고, applicability head는 해당 group이 현재 route/candidate에서 의미 있는지 예측한다. detail score head는 subject coverage, scale, headroom, lookroom, third/phi/center strength, horizon, context, safety penalty 같은 연속 score를 회귀한다. macro head도 aesthetic/subject/composition/technical score를 regression한 뒤 review 단계에서 `good/ok/bad` 같은 품질 label로 후처리할 수 있다. why-tag head는 `avoid_face_cut`, `balanced_crop`, `rule_of_thirds`, `subject_preserved`, `copy_space_kept` 등 multi-label rationale을 낸다.

초기 explain 계열 run에서는 checklist class CE 비중이 상대적으로 컸지만, 이후 subject-box/head 개선 계열에서는 detail score regression weight를 더 크게 두는 실험이 추가됐다. 따라서 현재 구조를 "label head에서 score regression head로 완전히 교체"한 것으로 쓰면 부정확하다. 더 정확한 표현은 class CE head는 유지하되, 제품 failure severity를 다루기 위해 continuous detail/macro score regression을 함께 두고 일부 best 계열에서 그 비중을 키웠다는 것이다.

### 6.13 런타임 출력 schema

runtime output에는 final crop box뿐 아니라 generated proposals, selected proposal index, route probabilities, decision probabilities, subject box, runtime baselines, checklist/detail/why-tag decode, target-AR repair metadata가 포함된다. 이 schema가 있어야 qualitative review와 release gate가 crop geometry뿐 아니라 route/policy/explanation mismatch를 추적할 수 있다.

### 6.14 Head 계약 요약

아래 표는 각 head를 소스코드 없이 검토할 수 있도록 input, output, loss, failure mode 기준으로 요약한다. 일부 loss는 항상 켜지는 기본 loss가 아니라 experiment flag로 활성화되는 보조 loss지만, 해당 head가 어떤 supervision surface와 연결되는지는 이 표가 기준이다.

| head/module | main input | output | training loss/signal | failure mode |
| --- | --- | --- | --- | --- |
| proposal box/logit head | condition token, proposal query, target AR, optional subject prior token | `proposal_boxes [B,Q,4]`, `proposal_logits [B,Q]` | proposal objectness, positive-box L1/IoU, diversity, generated proposal alignment | replay score는 높지만 runtime proposal recall이 낮음 |
| proposal token head | condition token + proposal query | proposal token `[B,Q,D]` | proposal loss, subject proposal alignment when enabled | subject-safe proposal surface가 형성되지 않음 |
| candidate encoder | ROI pooled inside/border/context feature, global feature, AR embedding, box geometry | candidate token `[B,K,D]` | downstream rank/risk/explanation losses | box context와 geometry를 분리하지 못해 ranker가 brittle해짐 |
| RelationLite / set ranker | candidate tokens, pair geometry, valid mask | refined candidate token `[B,K,D]` | ranking, listwise, pairwise, top-return losses | 후보 간 상대 선호가 틀리고 top1 crop이 불안정 |
| utility head | refined candidate token | utility logit `[B,K]` | score BCE/regression, listwise, pairwise, teacher distillation, top-return | candidate-bank replay ranking 실패 또는 runtime rerank 실패 |
| positive head | refined candidate token | positive logit `[B,K]` | positive BCE, generated positive alignment | 좋은 crop manifold를 충분히 찾지 못함 |
| risk head | refined candidate token | risk logit `[B,K]` | risk BCE, top1 risk suppression, generated risk alignment | unsafe/hard-negative crop이 top candidate로 상승 |
| macro head | refined candidate token | aesthetic/subject/composition/technical score logits | sigmoid score regression, review-time `good/ok/bad` post-labeling | high-level quality explanation이 generic하게 평탄화 |
| checklist class head | refined candidate token | grouped checklist class logits | checklist class CE with applicability mask | headroom/lookroom/context 같은 세부 상태가 실제 crop과 불일치 |
| checklist applicability head | refined candidate token, route-conditioned target | checklist applicability logits | applicability BCE | object/scene에서 person-only checklist를 hallucination |
| detail score head | refined candidate token | continuous checklist/detail scores | SmoothL1/detail score loss | class label은 맞아도 severity/ratio가 틀림 |
| why-tag head | refined candidate token, applicability mask | why-tag multi-label logits | multilabel BCE, reject/why tag supervision | rationale warning, crop failure와 무관한 why-tag |
| route head | global condition, optional candidate/subject context | subject mode logits | route CE, route-balanced CE, focal loss, object-single margin | portrait/scene/copyspace가 object_single 등으로 collapse |
| policy decision head | global condition, weighted candidate token, baseline score, best score, optional subject token | `keep_full|minimal_crop|crop` logits | decision CE, crop rebalance, action consistency | decision label collapse 또는 action-source mismatch |
| delta head | policy token | baseline-relative delta scalar | delta regression | baseline 대비 crop 가치 판단이 불안정 |
| policy score head | policy token | continuous policy score | policy score regression, threshold-logit bias | decision confidence는 높지만 utility와 policy가 어긋남 |
| subject-box coarse head | condition token | coarse subject box, valid logit | subject-box valid BCE, L1/IoU/center/size/aspect/CIoU | subject-present bias, low subject IoU, invalid scene false positive |
| subject proposal/refine heads | proposal tokens, proposal boxes, coarse subject box | refined subject box, refined valid logit, subject proposal logits | subject proposal alignment, subject-box refine losses | subject localization은 개선돼도 proposal/action과 정렬되지 않음 |
| runtime executor | decision logits, generated proposals, utility/risk scores, runtime baselines, target AR | final crop, action source, exact-AR repair metadata | evaluated by direct/no-prior metrics, not trained as a neural layer | crop IoU는 높아도 product action 의미가 깨짐 |

## 7. Proposal 생성과 후보 표면

![Candidate-bank replay vs generated proposal runtime](assets_mobilecropnet_v4_master_20260427/fig11_candidate_bank_vs_generated_runtime.png)

*그림 7-1. Candidate-bank replay와 generated-proposal runtime은 서로 다른 평가 표면이다. 이 그림은 개념 설명용 schematic이며 실험 evidence가 아니다.*

candidate-bank replay는 teacher가 만든 후보 집합 안에서 ranker가 좋은 후보를 고르는 문제다. 이 평가는 utility head, pairwise/listwise loss, risk suppression을 보기 좋지만, 배포 시 모델이 후보를 직접 만들 수 있는지는 보장하지 않는다. generated-proposal runtime은 모델이 image와 target AR만 보고 $B_q$를 생성하고, 그 위에서 top-M rerank와 executor를 수행하는 표면이다.

두 표면의 차이가 현재 핵심 blocker다. 일부 상위 row는 Product-AR direct hit@0.5가 `0.95` 수준으로 높지만, route collapse와 subject-box calibration이 gate를 막는다. route smoke에서는 `best_generated_align_top1_hit=0.0`인 row가 반복되어, route head를 올리는 smoke가 generated proposal alignment를 자동으로 고치지 못한다는 점도 확인됐다.

proposal branch가 성공하려면 세 조건을 동시에 만족해야 한다. 첫째, target AR-compatible box를 충분히 생성해야 한다. 둘째, subject-aware safety를 만족하는 positive manifold를 덮어야 한다. 셋째, generated proposal을 다시 scoring했을 때 candidate-bank replay에서 배운 utility/risk ordering이 유지되어야 한다. 이 세 조건 중 하나라도 깨지면 replay leaderboard는 높고 runtime qualitative review는 실패하는 모델이 된다.

## 8. 피사체 위치 추정과 Subject-Box 통합

![Subject box patch status](assets_mobilecropnet_v4_master_20260427/fig07_subject_box_patch_status.png)

*그림 8-1. Subject-box patch는 SSTK UCTR rank_320에서 subject IoU를 0.5 이상으로 올렸지만, release gate 전체를 닫지는 못했다.*

subject box는 no-prior runtime의 중심이다. teacher pipeline은 subject support와 route를 사용해 후보를 만들 수 있지만, MobileCropNet runtime은 그런 prior를 입력으로 받지 않는다. 따라서 모델이 자체적으로 subject 존재 여부와 box를 예측하지 못하면 portrait/object crop에서 subject coverage, scale, headroom, lookroom, face/joint cut을 안정적으로 판단할 수 없다.

초기 subject-box 경로의 문제는 subject-present bias였다. subject prior가 있는 row만 강하게 학습하면 background/copyspace/scene에서도 subject valid를 과대 예측할 수 있고, 반대로 valid calibration을 보수적으로 두면 portrait/object에서 positive recall이 낮아진다. v2 balanced-valid 아이디어는 subject가 없는 row의 negative calibration과 subject가 있는 row의 IoU/positive recall을 함께 보려는 시도다.

2026-04-24 patch는 네 방향을 반영했다. proposal-side subject supervision을 `subject_box_valid` 기준으로 정렬했고, content-relative subject box parameterization을 추가했으며, predicted subject prior warmup을 넣었고, proposal-conditioned 2-stage refinement와 `subject_proposal_align_loss`를 추가했다. 결과적으로 `mcn-sstk-uctr-subjboxpatch-r320-20260424`는 `direct_subject_iou=0.515918`, `direct_negative_acc=0.749626`, `direct_hit_iou_0_5=0.722441`을 기록했다.

그러나 subject-box 개선은 release gate의 단일 해결책이 아니다. GAIC patch row는 direct subject IoU가 `0.33` 수준에 머물렀고, SSTK patch도 public equal-4, route gate, Product-AR direct, qualitative review를 같은 final bundle에서 모두 통과한 것은 아니다. subject localization은 필요한 조건이지만, generated proposal alignment와 route/policy consistency까지 함께 닫혀야 한다.

## 9. Route, Subject Mode, Composition Policy

![Route subject smoke progress](assets_mobilecropnet_v4_master_20260427/fig05_route_subject_smoke_progress.png)

*그림 9-1. Route/subject recovery smoke. 최신 best도 route reference line을 안정적으로 넘지 못했고, subject IoU와 generated proposal alignment가 함께 닫히지 않았다.*

route는 단순 auxiliary label이 아니다. `portrait_single`에서는 headroom/lookroom/face cut이 중요하고, `portrait_group`에서는 group subject coverage와 joint cut이 중요하다. `object_single`과 `object_multi`는 object coverage와 scale이 중심이고, `scene_general`은 horizon/context, `background_texture_copyspace`는 copyspace와 texture preservation이 중요하다. route가 틀리면 어떤 checklist를 적용해야 하는지, 어떤 crop을 risk로 볼지, 어떤 why-tag를 노출할지가 함께 틀어진다.

현재 반복되는 failure는 route collapse다. Product-AR head audit에서는 portrait/scene/object_multi가 `object_single`으로 과도하게 붕괴하는 패턴이 관찰됐다. 최신 final row 중 `SSTK public / balanced_288`는 direct crop alignment가 강하지만 `route_collapse_flag=true`, `route_balanced_accuracy=0.361347`이다. route smoke best인 `mcn-route-predsubjctx-full-r320-20260424-193100-gpu1098428`도 `best_route_acc=0.492083`, `best_subject_iou=0.475680`으로 strict threshold를 넘지 못했다.

| failure type | symptom | downstream effect | likely cause | recovery direction |
| --- | --- | --- | --- | --- |
| portrait to object collapse | portrait_single/group을 object_single로 예측 | headroom/lookroom/face-cut applicability 약화 | global route feature가 subject structure를 충분히 못 봄 | subject-conditioned route input, route-balanced sampler |
| scene/copyspace collapse | background/copyspace를 generic object/scene으로 혼동 | copyspace/context policy 불안정 | blank region과 foreground 구분 signal 부족 | C7/support-map 또는 copyspace feature 증강 |
| route/action mismatch | route는 crop 필요, decision은 keep/minimal 또는 반대 | executor consistency 하락 | decision label collapse, utility-policy gap | policy score-first, action consistency loss |
| route/rationale mismatch | route와 why-tag가 서로 모순 | qualitative model rationale warning | applicability mask/why BCE가 약함 | score-first rationale, route-conditioned decode |
| route/proposal mismatch | route는 portrait인데 proposal이 subject-safe crop을 못 만듦 | low proposal recall, low subject IoU | proposal branch가 subject-aware하지 않음 | proposal-subject align, generated proposal align loss |

route 회복은 route accuracy 하나만 올리는 문제가 아니다. route balanced accuracy, subject IoU, generated proposal alignment, top1 hit, qualitative bucket을 같이 봐야 한다. 다음 smoke의 성공 기준은 strict threshold `0.50`을 간신히 넘는 것이 아니라, 재평가 여유를 위해 route balanced accuracy `>=0.52`와 subject IoU `>=0.50`을 동시에 만족하는 것이다.

## 10. Policy Head와 런타임 실행기

![Runtime no-prior rerun](assets_mobilecropnet_v4_master_20260427/fig06_runtime_no_prior_rerun_matrix.png)

*그림 10-1. Runtime no-prior rerun. direct hit이 높아도 route와 policy calibration이 release gate를 함께 닫지 못한다.*

policy head는 crop ranker의 부속물이 아니다. 제품은 세 가지 action을 구분해야 한다. `keep_full`은 원본을 유지하는 것이고, `minimal_crop`은 target AR을 맞추기 위한 최소 손실 crop이며, `crop`은 모델이 제안한 composition crop이다. 같은 final IoU라도 action source가 잘못되면 사용자 기대와 rationale이 달라진다.

runtime executor는 full baseline과 minimal baseline을 내부 생성한다. generated proposal top-M은 proposal score와 utility score로 rerank되고, decision head의 label에 따라 baseline 또는 crop proposal이 선택된다. 선택된 box는 exact target-AR postprocess를 거쳐 target-AR compatibility를 보장한다. no-prior rerun에서 `target_ar_compatible=1.0`이 나온 것은 이 executor/postprocess 경로가 형식적으로 작동한다는 뜻이다.

그러나 decision accuracy는 조심해서 해석해야 한다. Product-AR label builder가 한때 `decision_type`을 사실상 모두 `crop`으로 덮어써 policy head가 3-class 문제를 배울 수 없는 상태가 있었다. 이런 상황에서 `decision_acc=1.0`은 좋은 policy가 아니라 label collapse를 맞힌 결과일 수 있다. runtime no-prior rerun에서도 GAIC rank320 calibration v1/v2는 `direct_hit_iou_0_5=0.935198`, `route_acc=0.499646`, `decision_acc=1.0`을 보였지만, route/subject gate는 아직 닫히지 않았다.

baseline decision gate ablation도 같은 결론을 준다. confidence가 낮거나 baseline score가 crop보다 충분히 좋지 않을 때 crop으로 override하는 gate는 safety layer로 보일 수 있지만, current smoke에서는 direct quality regression이 확인되어 기본값이 아니다. policy 문제는 inference 후처리만으로 해결하기보다 training-time calibration, decision score target, action consistency loss, utility-policy alignment로 풀어야 한다.

## 11. Loss 설계

MobileCropNet v4 loss는 단일 bbox regression loss가 아니다. 각 loss는 배포 실패의 다른 축을 줄이기 위해 존재한다.

구현상 loss는 항상 모두 같은 비중으로 켜지는 고정 objective가 아니다. `train_mobilecropnet_v4.py`의 profile/run config가 loss weight를 조정하고, 일부 smoke는 특정 blocker를 보기 위해 explanation loss를 0으로 두기도 한다. 로컬 best checkpoint들을 확인하면 대부분 `macro_head`, `checklist_class_head`, `checklist_applicability_head`, `detail_score_head`, `why_tag_head`를 모두 보유하지만, explain 계열 loss weight는 세대별로 달랐다. 예를 들어 초기 detail explain 계열은 checklist class CE를 더 강하게 두었고, subject-box 개선 계열 일부는 detail score regression 비중을 더 크게 두었다. 따라서 아래 설명은 "항상 켜진 단일 loss 목록"이 아니라, 모델이 지원하고 실험에서 선택적으로 조합한 supervision family로 해석해야 한다.

### 11.1 랭킹 loss

score loss는 후보별 utility logit이 teacher score target을 따라가도록 만든다. 이는 replay ranking의 기본 축이지만, score scale만 맞아서는 top 후보 ordering이 보장되지 않는다.

### 11.2 Listwise loss

listwise loss는 후보 집합 전체의 teacher score distribution을 student utility 분포와 맞춘다. crop ranking은 절대 score보다 한 이미지 내부 후보 간 상대 순서가 중요하므로 listwise supervision이 필요하다.

### 11.3 Pairwise loss

pairwise loss는 score margin이 충분히 있는 후보 쌍에 대해 선호 방향을 직접 학습한다. explicit pairwise sidecar는 teacher가 만든 top1/top-k preference를 slot index로 연결해 utility head의 local ordering을 강화한다.

### 11.4 Positive/risk loss

positive loss는 좋은 crop manifold를 찾게 하고, risk loss와 top1 risk loss는 hard negative 또는 unsafe candidate가 top1으로 올라오는 것을 억제한다. 제품형 cropper에서는 "좋은 crop을 선택"하는 것과 "위험한 crop을 선택하지 않음"이 별도 목표다.

### 11.5 Proposal loss

proposal loss는 generated proposal이 positive boxes를 덮도록 objectness BCE, L1/IoU-style box term, diversity term을 준다. subject-aware profile에서는 proposal-subject loss와 subject proposal alignment loss도 추가될 수 있다. 이 loss가 약하면 candidate-bank replay는 좋아도 runtime proposal surface가 빈약하다.

### 11.6 Subject-box loss

subject-box loss는 valid BCE와 positive-valid row의 box regression을 결합한다. balanced BCE, negative scale, center/size/aspect/CIoU term은 subject-present bias와 poor localization을 동시에 줄이기 위한 옵션이다.

### 11.7 Route loss

route loss는 cross-entropy가 기본이며, class imbalance를 줄이기 위해 route-balanced CE, focal factor, object-single margin loss를 사용할 수 있다. route auxiliary head가 켜진 profile에서는 kind/cardinality hierarchy loss를 더하고, 별도 route expert를 사용할 때는 soft/hard distillation loss를 route head에 전달한다. route collapse는 release hard gate로 직접 이어지므로 route loss는 단순 auxiliary weight로 취급하기 어렵다.

### 11.8 Decision/action consistency loss

decision loss는 `keep_full|minimal_crop|crop` class를 맞히고, action consistency loss는 baseline 후보와 crop 후보의 utility gap이 decision target과 일치하도록 한다. 이후 action/source blocker를 좁히기 위해 decision source BCE, decision source margin, source gate supervision, return source margin, action-return joint loss, return explicit pairwise loss 같은 보조 loss가 추가됐다. 이 계열의 목적은 decision class만 맞히는 것이 아니라 final return surface가 full/minimal baseline과 generated crop 후보 중 제품 action target에 맞는 후보를 실제로 선택하도록 만드는 것이다. decision utility alignment loss는 decision crop probability가 utility에서 본 crop-vs-baseline 선호와 너무 멀어지지 않게 한다.

### 11.9 Checklist/rationale loss

macro, checklist class, applicability, detail score, why-tag loss는 explanation surface를 학습한다. 여기서 checklist 관련 supervision은 class label 방식과 score regression 방식이 공존한다. `checklist_class_head`는 group별 categorical label을 CE로 학습하고, `checklist_applicability_head`는 route별 적용 가능성을 BCE로 학습한다. 반면 `macro_head`와 `detail_score_head`는 sigmoid score를 SmoothL1로 회귀하며, review/eval 단계에서는 이 score를 `good/ok/bad` 같은 품질 label로 후처리할 수 있다. `why_tag_head`는 reject/why tag를 multilabel BCE로 학습한다.

따라서 checklist head를 "label classification에서 score regression으로 완전히 교체"했다고 쓰면 실제 checkpoint 구조와 맞지 않는다. 대표 best checkpoint들은 대체로 class/applicability/detail/why head를 모두 가진다. 다만 성능 개선 과정에서 class CE만으로는 failure severity와 score scale을 충분히 설명하기 어려워, 일부 subject-box/head 개선 계열에서는 `detail_score_weight`를 `checklist_class_weight`보다 크게 두는 방식으로 regression signal의 비중을 키웠다. 핵심은 label imitation 자체가 아니라 route별 applicable explanation만 노출하고, risk/checklist/why-tag가 crop failure를 모순 없이 설명하게 하는 것이다.

### 11.10 Generated proposal alignment loss

generated proposal alignment loss는 학습 중 label candidate bank를 scoring하는 동안에도 generated proposals를 별도 scoring surface로 평가하고, teacher candidate/positive/risk와 맞춘다. 이는 replay와 runtime 사이의 surface gap을 줄이는 핵심 loss다.

### 11.11 Teacher distillation loss

teacher distillation loss는 listwise sidecar의 soft target distribution을 student utility logits에 전달한다. UCTR/public cropper score는 ranking signal로 유용하지만, SSTK hard safety semantics와 혼동하지 않도록 safe conversion/demotion policy를 유지해야 한다. 별도 route expert를 사용할 때는 route logits에 대한 soft/hard distillation도 가능하지만, 이 경우에도 최종 runtime 입력 계약은 `image + target_ar`로 유지되어야 한다.

### 11.12 Checkpoint selection score와 배포 selection score

학습 중 `best.pt`를 고르는 selection score는 run 내부 checkpoint filter다. 최종 배포 선정은 별도의 release gate에서 수행되어야 한다. route-only smoke의 best epoch가 route metric을 올려도 public equal-4, Product-AR direct, subject-box calibration, qualitative review를 통과하지 않으면 deploy candidate가 아니다.

$$ \mathcal{L}=\lambda_s\mathcal{L}_{score}+\lambda_l\mathcal{L}_{list}+\lambda_p\mathcal{L}_{pair}+\lambda_r\mathcal{L}_{risk}+\lambda_q\mathcal{L}_{proposal}+\lambda_b\mathcal{L}_{subj}+\lambda_m\mathcal{L}_{route}+\lambda_d\mathcal{L}_{decision}+\lambda_e\mathcal{L}_{explain}+\lambda_g\mathcal{L}_{genalign} $$

## 12. 학습과 Orchestration

표준 학습 entrypoint는 MobileCropNet v4 batch JSONL, optional pairwise/listwise sidecar, train/val split, output directory를 받아 profile별 model config를 구성한다. `balanced_288`, `rank_320`, `plus_384`, `hybrid_384`, `turbo_256`, `q24_288_w075` 같은 profile은 input size, candidate count, proposal query count, backbone, token dimension, ranker depth를 함께 정의한다.

long-running training은 launch/poll/collect pattern으로 운영한다. run directory에는 `config.json`, `dataset_summary.json`, `train_status.json`, `metrics.json`, `best.pt`, `latest.pt`가 남아야 한다. GPU work는 bounded MLP workload 또는 이미 살아 있는 same-account interactive GPU run에서 실행하고, shared storage artifact를 polling한다. CPU-heavy label conversion이나 report build는 multi CPU 서버를 사용할 수 있지만, 단순 로컬 검증은 프로젝트 지정 Python interpreter로 실행한다.

학습 run은 크게 smoke, full run, final bundle로 나뉜다. smoke는 route/subject/proposal/policy 같은 특정 blocker를 작은 시간 안에 확인한다. full run은 profile/variant별 학습을 완료한다. final bundle은 public benchmark, GAIC official, Product-AR direct, head audit, subject-box report, qualitative review, latency, release gate를 같은 checkpoint 기준으로 채운다. 현재 단계에서는 promising row에 대해 곧바로 full final bundle을 돌리기보다, route/subject recovery를 bounded smoke에서 먼저 증명하는 것이 맞다.

## 13. 평가 표면

MobileCropNet v4는 하나의 점수로 평가할 수 없다. 각 surface는 측정하는 것과 측정하지 못하는 것이 다르다.

| 평가 표면 | 측정 항목 | 측정하지 못하는 항목 | 필요한 이유 |
| --- | --- | --- | --- |
| replay eval | label candidate bank 위 utility, positive, risk, route, decision, explanation | generated proposal recall, no-prior executor | 학습 target을 제대로 읽는지 확인 |
| GAIC official benchmark | official candidate set MOS ranking, SRCC, Acc@k | Product-AR route/policy/subject runtime | public MOS 품질과 ranking correlation 확인 |
| public equal-4 | FCDB/CPC/GNMC/GAIC 균형 score | student no-prior consistency | dataset 하나에 과적합한 cropper 배제 |
| Product-AR direct eval | generated proposal + runtime selection hit, best IoU, risk, target AR | qualitative route/rationale semantics 전체 | 배포 crop action 표면을 직접 확인 |
| head sanity/audit | route balance, checklist, why, risk, subject head | final crop visual quality 전체 | auxiliary head collapse 조기 탐지 |
| subject-box report | subject IoU, valid acc, negative/positive calibration | route/policy/action consistency | no-prior subject localization gate |
| runtime no-prior eval | `image + target_ar` only execution | teacher candidate replay score | 배포 입력 계약 위반 방지 |
| latency benchmark | profile별 forward latency, throughput, params | quality/safety readiness | 온디바이스 feasibility 확인 |
| qualitative review | per-sample crop panel과 review-pack catastrophic bucket | exhaustive statistical proof | 수치가 놓치는 route/subject/rationale 실패 탐지 |
| release gate | 제품 배포 가능성 최종 판정 | 원인 진단의 모든 세부 항목 | shipping checkpoint 선언 기준 |

## 14. Teacher/Public Cropper 상한과 Student 배포

![Teacher public cropper ceiling](assets_mobilecropnet_v4_master_20260427/fig10_teacher_public_cropper_ceiling.png)

*그림 14-1. Teacher/public cropper equal-4 ceiling. teacher ceiling은 높지만, student deployment는 no-prior internalization을 별도로 요구한다.*

teacher/public cropper ceiling은 student가 따라가야 할 품질 상한을 제공한다. 최신 equal-4 teacher leaderboard에서 `production_final_hybrid`는 `equal4_raw_mean=0.882696`, `equal4_zscore=1.148158`, `GAIC primary=0.791485`로 1위다. `universal_crop_teacher_h_stage3_gate128`은 `equal4_raw_mean=0.852693`, `equal4_zscore=0.863135`로 learned UCTR teacher 중 강한 기준이고, `public_cropper_ensemble_best`는 `equal4_raw_mean=0.798613`으로 public cropper score source로 유효하다.

하지만 teacher ceiling이 높다는 사실은 deployable student가 있다는 뜻이 아니다. teacher/public cropper는 offline candidate scoring이나 dataset-specific crop ranking에 강할 수 있지만, MobileCropNet student는 learned proposal generator, route head, policy executor, subject-box head, explanation head를 온디바이스 runtime 안에 내재화해야 한다. `production_final_hybrid`는 benchmark-level routed hybrid policy이지 MobileCropNet runtime에 그대로 들어가는 단일 lightweight checkpoint가 아니다.

따라서 현 병목은 "더 강한 teacher score를 찾는 것"보다 "teacher가 제공한 ranking/subject/route/policy semantics를 no-prior student runtime이 internalize하는 것"이다. UCTR/public score branch는 여전히 유효하지만, safe conversion과 SSTK hard safety metadata를 보존한 distillation이어야 한다.

## 15. 현재 Student 리더보드와 Release Gate

![Strict release gate matrix](assets_mobilecropnet_v4_master_20260427/fig03_strict_release_gate_matrix.png)

*그림 15-1. Strict release gate matrix. 현재 gate pass row는 없다.*

![Current equal4 leaderboard](assets_mobilecropnet_v4_master_20260427/fig04_current_equal4_leaderboard.png)

*그림 15-2. 현재 MobileCropNet shortlist의 equal-4 z-score. 상위 score row와 deploy candidate는 분리해서 해석해야 한다.*

최신 final deployment leaderboard는 `37`개 row를 추적한다. 2026-04-27 최신 원격 artifact 기준으로 `final_deployment_leaderboard_refresh_20260427_qfirst`를 재생성했으며 completed entry는 `34 / 37`이다. 남은 `3`개 row는 pending workload가 아니라 checkpoint source가 없는 invalid row로 분리한다. strict release gate pass는 여전히 `0 / 37`이고, release decision은 `BLOCKED`, deploy candidate는 `null`이다.

| track | profile | equal4 raw | equal4 z | GAIC top1 MOS | GAIC SRCC | direct hit@0.5 | route bal | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `SSTK public` | `balanced_288` | `0.760155` | `1.466947` | `3.723580` | `0.646045` | `0.956820` | `0.361347` | fail, route/subject gate |
| `SSTK public routefix` | `balanced_288` | `0.754432` | `1.209013` | `3.697260` | `0.656426` | `0.944571` | `0.390207` | fail, subject head inactive |
| `SSTK public` | `rank_320` | `0.750415` | `1.044961` | `3.645440` | `0.640636` | `0.959726` | `0.360386` | fail, route/subject gate |
| `GAIC UCTR subjectprior` | `plus_384` | `0.752671` | `0.957636` | `3.830240` | `0.519405` | `0.960340` | `0.395415` | fail, route/subject gate |
| `GAIC public` | `balanced_288` | `0.752083` | `1.010321` | `3.723320` | `0.643775` | `0.927408` | `0.337583` | fail, route/subject gate |
| `SSTK UCTR release_gate_v1` | `rank_320` | `0.724264` | `-0.158411` | `3.864900` | `0.432450` | `0.717044` | `0.294785` | fail, direct/route/subject |

이 표의 정량값은 서로 다른 질문에 답한다. `equal4_zscore`는 public benchmark 균형 품질을, `GAIC top1 MOS/SRCC`는 official candidate set ranking을, `direct hit@0.5`는 Product-AR generated/runtime crop alignment를, `route bal`은 product semantics generalization을 본다. 따라서 한 열이 높다는 사실만으로 deployable model을 의미하지 않는다. 특히 `-`가 있는 row는 final bundle이 불완전하거나 같은 audit surface의 수치가 아직 없으므로, score 순위보다 gate completeness를 먼저 확인해야 한다.

아래 표들은 `MobileCropNet_Training_Data_Generation_Master_Report_KO_2026-04-24.md`의 그림 13-5 아래 selected method 표와 같은 열 구성으로, Teacher/Public ceiling과 MobileCropNet v4 profile별 best final row를 나란히 비교할 수 있게 재구성한 것이다. Teacher/Public row는 `artifacts/unified_public_benchmark_20260423_equal4_teacher/equal4_teacher_leaderboard.json`, MobileCropNet row는 후속 quality-first snapshot인 `artifacts/mobilecropnet_v4/unified_recovery_20260424/final_deployment_leaderboard_refresh_20260428_quality_first_snapshot/final_deployment_leaderboard_quality_first_snapshot.json`의 completed row 중 학습 데이터군과 profile별 `equal4 raw` 최고 row를 기준으로 했다. `equal4 raw`, FCDB/CPC/GNMC/GAIC primary는 같은 public benchmark 정의로 비교 가능하지만, `equal4 z`와 `worst z`는 각 leaderboard cohort의 정규화값이므로 raw metric과 함께 읽어야 한다.

원격 shared storage(`/group-volume/users/jaden.ju/Sources/ImageCropping`)도 확인했다. 최신 final leaderboard JSON은 로컬에 있는 `final_deployment_leaderboard_refresh_20260428_quality_first_snapshot`과 같은 세대가 최신이며, `turbo_256`/`hq_320` 계열은 원격에 GAIC official `metrics.json`은 있지만 `public_benchmark/eval/summary.json` 또는 final equal-4 row가 없어 아래 equal-4 matrix에는 완결 수치를 채울 수 없다. 따라서 해당 profile은 표에 누락 사유를 명시하고 `-`로 둔다.

Teacher/Public ceiling:

| Method | 역할 | equal4 raw | equal4 z | worst z | FCDB IoU | CPC weighted | GNMC IoU | GAIC top1 MOS | GAIC SRCC | GAIC Accw4@10 | GAIC primary |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UCTR-H Stage3 딥 랭커(통합 Teacher) | public geometry와 GAIC ranking을 함께 학습한 deep crop ranker | 0.852693 | 0.863135 | -0.118412 | 0.9085 | 0.9135 | 0.9173 | 3.8461 | 0.6294 | 0.4438 | 0.6715 |
| Public Cropper 앙상블(GAIC+CGS) | GAIC MOS ranker와 CGS composition ranker를 결합한 외부 cropper | 0.798613 | 0.020538 | -0.706939 | 0.7351 | 0.8918 | 0.7772 | 3.9841 | 0.8524 | 0.6240 | 0.7903 |
| S2CNet 공개 Cropper | MobileNetV2 + graph-attention relation crop ranker. 원 논문 object graph 대신 candidate-derived graph node를 사용한 public-weight diagnostic | 0.794295 | -0.018356 | -0.999281 | 0.7097 | 0.8923 | 0.7924 | 3.9649 | 0.8395 | 0.6159 | 0.7827 |
| GAIC 공개 Cropper(MOS 순위 랭커) | GAIC candidate MOS 순위에 강한 public crop ranker | 0.792681 | -0.122526 | -0.678149 | 0.7376 | 0.8784 | 0.7692 | 3.9853 | 0.8459 | 0.6062 | 0.7855 |
| CGS 공개 Cropper(구도 순위 랭커) | composition/graph-style crop 선호를 반영하는 public crop ranker | 0.785886 | -0.147538 | -0.726516 | 0.7334 | 0.8851 | 0.7702 | 3.9210 | 0.7998 | 0.5521 | 0.7548 |
| CACNet 공개 Cropper projection | 단일 crop regressor를 benchmark candidate set에 IoU projection한 diagnostic row. native GAIC candidate score는 없음 | 0.743003 | -0.707904 | -0.892927 | 0.7575 | 0.8611 | 0.7459 | 3.7355 | 0.5245 | 0.3357 | 0.6075 |
| Compact Teacher Proxy(경량 Utility 기준선) | 경량 utility feature로 만든 teacher proxy baseline | 0.720220 | -1.097941 | -1.567600 | 0.7204 | 0.8374 | 0.7329 | 3.6731 | 0.5022 | 0.3140 | 0.5902 |

S2CNet/CACNet 추가 row는 `artifacts/mobilecropnet_v4/public_cropper_extension_20260508_remote/gpu_interactive_full_20260508_093532/tables/PUBLIC_CROPPER_EXTENSION_TABLES.md`와 `public_cropper_extension_tables.json`에서 가져왔다. `equal4 z`는 기존 `2026-04-23` teacher/public leaderboard의 metric distribution을 고정 reference로 사용해 계산했으므로 기존 GAIC/CGS/teacher row의 z값을 재정규화하지 않는다. S2CNet은 GAIC primary가 GAIC/ensemble에 근접하지만 FCDB 축이 약하고, CACNet projection은 FCDB IoU는 높지만 GNMC/GAIC projection이 약해 equal-4 raw가 compact teacher proxy와 public ranker 사이에 위치한다.

SSTK 학습 데이터 계열 profile별 best final row:

| Method | 역할 | equal4 raw | equal4 z | worst z | FCDB IoU | CPC weighted | GNMC IoU | GAIC top1 MOS | GAIC SRCC | GAIC Accw4@10 | GAIC primary |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileCropNet v4 `q24_288_w075` | custom depthwise 288/24 초기 경량 baseline best final row (`SSTK UCTR subjectprior` / `subjectprior_route_proposal_v1`) | 0.732686 | 0.360455 | 0.030176 | 0.7447 | 0.8604 | 0.7346 | 3.7701 | 0.4586 | 0.3571 | 0.5911 |
| MobileCropNet v4 `turbo_256` | MobileNetV4 Conv-S 256/16 latency-first profile. 원격 확인 결과 GAIC-only metrics는 있으나 completed final public/GAIC/equal-4 row 없음 | - | - | - | - | - | - | - | - | - | - |
| MobileCropNet v4 `balanced_288` | MobileNetV4 Conv-M 288/24 RelationLite 균형 profile best final row (`SSTK public` / `subjectprior_route_proposal_v1`) | 0.760155 | 1.477973 | 0.502637 | 0.7661 | 0.8557 | 0.7682 | 3.7236 | 0.6460 | 0.3481 | 0.6507 |
| MobileCropNet v4 `hq_320` | MobileNetV4 Hybrid-M 320/32 HQ 설계 profile. 원격 확인 결과 GAIC-only metrics는 있으나 completed final public/GAIC/equal-4 row 없음 | - | - | - | - | - | - | - | - | - | - |
| MobileCropNet v4 `rank_320` | MobileNetV4 Conv-M 320/32 set-transformer ranking profile best final row (`SSTK public` / `subjectprior_route_proposal_v1`) | 0.750415 | 1.058241 | 0.476080 | 0.7490 | 0.8534 | 0.7646 | 3.6454 | 0.6406 | 0.3056 | 0.6346 |
| MobileCropNet v4 `hybrid_384` | MobileNetV4 Hybrid-M 384/48 heavy on-device 후보 best final row (`SSTK T1` / `baseline_current`) | 0.586342 | -2.311634 | -4.381641 | 0.7210 | 0.4438 | 0.7051 | 3.4064 | 0.3163 | 0.1606 | 0.4754 |
| MobileCropNet v4 `plus_384` | ConvNeXtV2-Tiny/384 48-candidate diagnostic profile best final row (`SSTK UCTR subject-box patch` / `subject_box_iou_patch`) | 0.733569 | 0.289183 | -0.429691 | 0.7440 | 0.8674 | 0.7259 | 3.8986 | 0.4357 | 0.3643 | 0.5970 |

GAIC 학습 데이터 계열 profile별 best final row:

| Method | 역할 | equal4 raw | equal4 z | worst z | FCDB IoU | CPC weighted | GNMC IoU | GAIC top1 MOS | GAIC SRCC | GAIC Accw4@10 | GAIC primary |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileCropNet v4 `q24_288_w075` | custom depthwise 288/24 초기 경량 baseline best final row (`GAIC T1` / `baseline_current`) | 0.713755 | 0.061823 | -0.262755 | 0.7414 | 0.7911 | 0.7316 | 3.8131 | 0.4481 | 0.3519 | 0.5909 |
| MobileCropNet v4 `turbo_256` | MobileNetV4 Conv-S 256/16 latency-first profile. 원격 확인 결과 GAIC-only metrics는 있으나 completed final public/GAIC/equal-4 row 없음 | - | - | - | - | - | - | - | - | - | - |
| MobileCropNet v4 `balanced_288` | MobileNetV4 Conv-M 288/24 RelationLite 균형 profile best final row (`GAIC public` / `subjectprior_route_proposal_v1`) | 0.752083 | 1.033298 | 0.269922 | 0.7621 | 0.8559 | 0.7391 | 3.7233 | 0.6438 | 0.3573 | 0.6512 |
| MobileCropNet v4 `hq_320` | MobileNetV4 Hybrid-M 320/32 HQ 설계 profile. 원격 확인 결과 GAIC-only metrics는 있으나 completed final public/GAIC/equal-4 row 없음 | - | - | - | - | - | - | - | - | - | - |
| MobileCropNet v4 `rank_320` | MobileNetV4 Conv-M 320/32 set-transformer ranking profile best final row (`GAIC UCTR` / `baseline_current`) | 0.752401 | 0.867907 | 0.165878 | 0.7547 | 0.8853 | 0.7371 | 3.8262 | 0.5438 | 0.3963 | 0.6324 |
| MobileCropNet v4 `hybrid_384` | MobileNetV4 Hybrid-M 384/48 heavy on-device 후보. completed final public/GAIC/equal-4 row 없음 | - | - | - | - | - | - | - | - | - | - |
| MobileCropNet v4 `plus_384` | ConvNeXtV2-Tiny/384 48-candidate diagnostic profile best final row (`GAIC UCTR subjectprior` / `subjectprior_route_proposal_v1`) | 0.752671 | 0.977617 | 0.579453 | 0.7587 | 0.8835 | 0.7449 | 3.8302 | 0.5194 | 0.3916 | 0.6236 |

이 비교에서 가장 중요한 차이는 MobileCropNet v4 student profile들이 Teacher/Public ceiling보다 public geometry와 GAIC primary 모두 낮다는 점이다. SSTK/GAIC 양쪽 모두 `balanced_288`, `rank_320`, `plus_384`는 compact teacher proxy보다 equal-4 raw가 높거나 비슷하고 일부 GAIC MOS는 teacher 계열에 접근하지만, UCTR-H Stage3나 public cropper ensemble 수준의 FCDB/GNMC/GAIC primary에는 도달하지 못한다. 또한 이 표는 public/GAIC crop quality만 보여 주므로, 앞의 release gate 표에 있는 route balance, subject-box, direct alignment failure와 함께 읽어야 한다.

2026-04-27 원격 final bundle 동기화와 재생성으로 `SSTK public routefix / balanced_288`, `SSTK UCTR routefix / balanced_288`, subject-box v2/patch, release-gate v1/v2/validcal/nopriorroute 계열의 public/GAIC/direct 수치를 보강했다. `SSTK UCTR release_gate_v1 / rank_320`도 FCDB `0.720592`, CPC weighted `0.857545`, GNMC `0.726827`, GAIC top1 MOS `3.864900`, SRCC `0.432450`, Accw4@10 `0.361629`, GAIC primary `0.592092`로 채워졌지만 direct hit `0.717044`, route balanced `0.294785`, subject IoU `0.440933`라 배포 후보에서 제외한다.

strict gate는 crop quality top-set, head sanity, direct alignment, target-AR compatibility, risk hit, proposal target recall, route collapse, route balance, subject-box gate를 함께 본다. current blocked fallback인 `SSTK public / subjectprior_route_proposal_v1 / balanced_288`는 crop/public/direct 계열은 강하지만 `hard_gate_route_collapse=false`, `hard_gate_route_balance=false`, `hard_gate_subject_box=false`다. 따라서 best_selection_score 또는 equal-4 순위만으로 winner를 확정하지 않는다.

release gate thresholds 중 핵심 hard 기준은 route balanced accuracy `>=0.50`, subject-box IoU `>=0.50`, subject-box predicted confidence `>=0.05`, subject-box valid accuracy `>=0.55`, negative accuracy `>=0.60`, positive accuracy `>=0.50`, target-AR compatibility `>=0.999`다. 현재 상위 student row는 이 중 route/subject 축을 반복적으로 통과하지 못한다.

## 16. 정성 분석과 실패 분류

기존 contact-sheet/collage figure는 본문 qualitative evidence에서 제외한다. 축소된 contact sheet는 많은 샘플을 한 번에 보여주지만, crop result, route, decision, checklist/why-tag를 처음 보는 독자가 읽기에는 너무 작고 우측 텍스트도 과밀하다. 대신 Section 16은 두 종류의 큰 패널 gallery로 재구성한다. 첫 번째는 teacher route 문자열을 기준으로 실제 인물 단독/그룹 사진을 포함한 route-stratified audit gallery이고, 두 번째는 기존 모델 추론 시각화의 역할을 유지하되 crop IoU만이 아니라 route/decision/subject-valid head 품질까지 함께 좋은 sample을 고른 joint head-quality inference gallery다.

route 표시는 단순 시각화 후처리가 아니다. label JSONL에는 `routing.subject_mode` 문자열과 `subject_mode_id`가 함께 있고, 현재 코드 vocab에는 `other_ambiguous`가 중간에 들어가 있어 id 3 이후의 표시명이 한 칸 밀릴 수 있다. 따라서 아래 패널은 teacher label의 id-to-string contract인 `0=background_texture_copyspace`, `1=object_multi`, `2=object_single`, `3=portrait_group`, `4=portrait_single`, `5=scene_general`, `6=other_ambiguous`로 model route id를 semantic decode한다. 이 보정 후에도 ribbon/object sample이 portrait route로 가는 경우는 실제 route failure로 남는다.

### 16.1 Teacher Route 기준 계층화 감사 갤러리

이 gallery는 "좋아 보이는 crop"을 고른 것이 아니라 teacher route별 대표 이미지를 고정한 audit slice다. `portrait_single` 2장, `portrait_group` 2장, `object_single` 1장, `object_multi` 1장, `scene_general` 2장, `background_texture_copyspace` 2장으로 구성했고, 실제 단독 인물과 실제 그룹/커플 이미지를 포함한다. crop geometry는 `direct_test_proposal_topk_rerank/predictions.jsonl`의 no-prior selected box이고, route/decision/checklist/why는 같은 checkpoint를 no-prior 조건으로 다시 decode한 head output이다. cyan overlay는 raw detector prior가 아니라 teacher support map의 `latent_support_bbox_norm_xyxy` 또는 `envelope_norm_xyxy`를 표시한다. raw `subject_prior_bbox_norm_xyxy`는 manifest에 audit field로 남겼으며, 기존 subject-box training/eval target이 이 raw prior에 의존했다는 점은 별도 caveat로 해석해야 한다.

핵심 관찰은 세 가지다. 첫째, 단독 인물 sample 01/02는 crop과 route가 함께 맞지만 subject-valid confidence는 여전히 0에 가까워 subject-box gate를 통과하지 못한다. 둘째, 그룹/커플 sample 03/04는 crop IoU가 높아도 route가 `portrait_group`에서 `portrait_single` 또는 background 계열로 collapse한다. 셋째, ribbon sample 06은 teacher route가 `object_multi`인데 model route가 portrait 계열로 가므로, 사용자가 지적한 sample 01 유형의 문제는 route head failure다. 또한 `decision=crop`은 generated proposal을 선택했다는 뜻이며, 그 box가 시각적으로 `minimal_crop`처럼 보일 수 있다.

![Teacher route audit sample 01](assets_mobilecropnet_v4_master_20260427/fig12_route_stratified_crop_sample_01_sstk_image_444218542.png)

*그림 16-1. Teacher-route audit sample 01, actual portrait_single.*

![Teacher route audit sample 02](assets_mobilecropnet_v4_master_20260427/fig12_route_stratified_crop_sample_02_pond5_image_194228138.png)

*그림 16-2. Teacher-route audit sample 02, actual portrait_single.*

![Teacher route audit sample 03](assets_mobilecropnet_v4_master_20260427/fig12_route_stratified_crop_sample_03_sstk_image_1865518057.png)

*그림 16-3. Teacher-route audit sample 03, actual portrait_group with route collapse.*

![Teacher route audit sample 04](assets_mobilecropnet_v4_master_20260427/fig12_route_stratified_crop_sample_04_sstk_image_1486959038.png)

*그림 16-4. Teacher-route audit sample 04, actual portrait_group with route collapse.*

![Teacher route audit sample 05](assets_mobilecropnet_v4_master_20260427/fig12_route_stratified_crop_sample_05_sstk_image_2196549559.png)

*그림 16-5. Teacher-route audit sample 05, object_single route failure despite strong crop IoU.*

![Teacher route audit sample 06](assets_mobilecropnet_v4_master_20260427/fig12_route_stratified_crop_sample_06_sstk_image_693448510.png)

*그림 16-6. Teacher-route audit sample 06, object_multi ribbon/package sample misrouted as portrait.*

![Teacher route audit sample 07](assets_mobilecropnet_v4_master_20260427/fig12_route_stratified_crop_sample_07_sstk_image_372926398.png)

*그림 16-7. Teacher-route audit sample 07, scene_general partial route match.*

![Teacher route audit sample 08](assets_mobilecropnet_v4_master_20260427/fig12_route_stratified_crop_sample_08_bigstock_image_425619878.png)

*그림 16-8. Teacher-route audit sample 08, scene_general route failure.*

![Teacher route audit sample 09](assets_mobilecropnet_v4_master_20260427/fig12_route_stratified_crop_sample_09_sstk_image_255265765.png)

*그림 16-9. Teacher-route audit sample 09, background/copyspace route match.*

![Teacher route audit sample 10](assets_mobilecropnet_v4_master_20260427/fig12_route_stratified_crop_sample_10_sstk_image_1502754071.png)

*그림 16-10. Teacher-route audit sample 10, background/copyspace partial route match.*

route-stratified gallery manifest는 `assets_mobilecropnet_v4_master_20260427/route_stratified_crop_gallery_20260427/fig12_route_stratified_crop_gallery_manifest.json`에 있다. 총 10개 대표 이미지와 60개 crop PNG가 저장되어 있으며, row-level 집계는 route match `26/60`, decision match `60/60`, crop hit@0.5 `60/60`이다. 이는 crop proposal branch가 꽤 강해도 semantic route와 subject-valid head가 제품 gate를 막는다는 결론과 일치한다.

### 16.2 Head 품질 결합 추론 갤러리

기존 모델 추론 시각화의 역할도 유지한다. 다만 이전 best-case gallery처럼 crop IoU만으로 고르지 않고, 현재 checkpoint를 전체 test row에 다시 no-prior decode한 뒤 다음 composite score와 strict filter로 sample을 고른다.

$$ S=0.30\cdot hit@0.5+0.25\cdot IoU+0.20\cdot route\_match+0.15\cdot decision\_match+0.05\cdot targetAR\_compat+0.05\cdot subjectValid\_acc $$

strict filter는 6개 AR row 모두에서 semantic route match, decision match, hit@0.5, target-AR compatibility가 1인 image만 통과시킨다. 이 조건을 만족한 후보는 43개뿐이며 route별로 `background_texture_copyspace=34`, `scene_general=6`, `portrait_single=3`이다. `portrait_group`, `object_single`, `object_multi`는 strict semantic joint-quality 후보가 0개다. 따라서 아래 10장은 모델이 route/head/crop을 함께 가장 잘 맞춘 slice이면서, 동시에 "잘 맞는 slice가 background/scene/portrait_single에 편중된다"는 route coverage 결함을 보여준다.

![Joint head-quality inference sample 01](assets_mobilecropnet_v4_master_20260427/fig13_joint_head_inference_crop_sample_01_bigstock_image_302661208.png)

*그림 16-11. Joint head-quality inference sample 01, portrait_single.*

![Joint head-quality inference sample 02](assets_mobilecropnet_v4_master_20260427/fig13_joint_head_inference_crop_sample_02_pond5_image_152685564.png)

*그림 16-12. Joint head-quality inference sample 02, portrait_single.*

![Joint head-quality inference sample 03](assets_mobilecropnet_v4_master_20260427/fig13_joint_head_inference_crop_sample_03_bigstock_image_140150411.png)

*그림 16-13. Joint head-quality inference sample 03, scene_general.*

![Joint head-quality inference sample 04](assets_mobilecropnet_v4_master_20260427/fig13_joint_head_inference_crop_sample_04_sstk_image_753174793.png)

*그림 16-14. Joint head-quality inference sample 04, scene_general.*

![Joint head-quality inference sample 05](assets_mobilecropnet_v4_master_20260427/fig13_joint_head_inference_crop_sample_05_sstk_image_364586810.png)

*그림 16-15. Joint head-quality inference sample 05, background/copyspace.*

![Joint head-quality inference sample 06](assets_mobilecropnet_v4_master_20260427/fig13_joint_head_inference_crop_sample_06_bigstock_image_418297522.png)

*그림 16-16. Joint head-quality inference sample 06, background/copyspace.*

![Joint head-quality inference sample 07](assets_mobilecropnet_v4_master_20260427/fig13_joint_head_inference_crop_sample_07_sstk_image_2327743739.png)

*그림 16-17. Joint head-quality inference sample 07, background/copyspace.*

![Joint head-quality inference sample 08](assets_mobilecropnet_v4_master_20260427/fig13_joint_head_inference_crop_sample_08_sstk_image_1158315238.png)

*그림 16-18. Joint head-quality inference sample 08, background/copyspace.*

![Joint head-quality inference sample 09](assets_mobilecropnet_v4_master_20260427/fig13_joint_head_inference_crop_sample_09_bigstock_image_385433447.png)

*그림 16-19. Joint head-quality inference sample 09, background/copyspace.*

![Joint head-quality inference sample 10](assets_mobilecropnet_v4_master_20260427/fig13_joint_head_inference_crop_sample_10_pond5_image_146594935.png)

*그림 16-20. Joint head-quality inference sample 10, background/copyspace.*

joint head-quality gallery manifest는 `assets_mobilecropnet_v4_master_20260427/joint_head_inference_gallery_20260427/fig13_joint_head_inference_gallery_manifest.json`에 있다. 총 10개 대표 이미지와 60개 crop PNG가 저장되어 있으며, row-level 집계는 route match `60/60`, decision match `60/60`, crop hit@0.5 `60/60`이다. 이 gallery도 cyan overlay는 teacher support envelope를 표시한다. 단, release-pass evidence는 아니다. strict 후보가 background/scene/portrait_single에 편중되고 subject confidence가 여전히 0에 가까워 subject-box release gate failure를 뒤집지 못한다.

### 16.3 MCN-public cropper 대표 정성 비교

추가로 MobileCropNet과 public cropper를 같은 이미지 행에서 직접 비교하는 패널을 만들었다. 이번 갱신판은 public benchmark row가 아니라 release-gate direct eval 산출물인 `artifacts/mobilecropnet_v4/shortlist_final_eval_20260423/sstk-public/balanced-288/direct_test_proposal_topk_rerank/predictions.jsonl`에서 샘플을 골랐다. Public benchmark row에는 teacher subject label이 없어서 head match를 검증할 수 없기 때문이다. 필터는 AR 6개 모두에서 `route_acc=1`, `decision_acc=1`, `final_positive_hit_iou_0_5=1`, `final_target_ar_compatible=1`이고 이미지 평균 `final_iou_to_best_positive >= 0.75`인 경우만 통과시켰다.

최종 20장은 teacher subject mode 기준으로 `portrait_single=4`, `object_single=4`, `object_multi=4`, `scene_general=4`, `background_texture_copyspace=4`가 되도록 균형을 맞췄다. 각 패널은 다섯 subject mode를 한 장씩 포함하며, 인물 사진은 portrait slice에 반드시 들어가도록 했다. MCN crop box와 head label은 같은 release-gate direct prediction row에서 가져왔기 때문에, 화면의 `MCN head vote`는 teacher head와 일치하는 샘플만 보여준다. MCN crop tile의 기존 `u=*`, `r=*` 표기는 제거했고, 각 AR crop에는 direct prediction의 selected score를 `s=*`, release-gate `final_iou_to_best_positive`를 `iou=*`로 표시했다.

패널 구성은 네 칸이다. 첫 칸은 teacher head, release-gate match metric, MCN head vote를 요약하고, 둘째 칸은 원본 이미지 위 crop box overlay를 보여준다. 셋째 칸은 MCN의 `FREE`, `1:1`, `3:4`, `4:3`, `16:9`, `9:16` crop 결과이며, 넷째 칸은 public cropper `GAIC`/`CGS`의 free-form top crop이다. Public cropper는 target AR별 product crop API가 아니므로, 오른쪽 비교는 MCN AR crop, center crop, teacher label candidate를 합친 candidate bank 위에서 GAIC/CGS ranker가 고른 free-form top crop으로 해석한다. Public cropper의 `s=*`는 GAIC/CGS ranker score이고, `iou=*`는 teacher matching/positive candidate box 대비 best IoU다.

![MCN public cropper qualitative comparison head-good slice 01](assets_mobilecropnet_v4_master_20260427/fig16_mcn_public_cropper_qualitative_headgood_01.png)

*그림 16-21. Head-good release-gate slice 01. MCN은 teacher-matched head와 AR별 IoU-labelled crop을 보여주고, public cropper는 같은 candidate bank에서 고른 GAIC/CGS free-form top crop을 보여준다.*

![MCN public cropper qualitative comparison head-good slice 02](assets_mobilecropnet_v4_master_20260427/fig16_mcn_public_cropper_qualitative_headgood_02.png)

*그림 16-22. Head-good release-gate slice 02. 다섯 subject mode를 한 장씩 포함해 crop 품질과 head match를 동시에 확인한다.*

![MCN public cropper qualitative comparison head-good slice 03](assets_mobilecropnet_v4_master_20260427/fig16_mcn_public_cropper_qualitative_headgood_03.png)

*그림 16-23. Head-good release-gate slice 03. Public GAIC/CGS는 AR별 출력이 아니라 free-form candidate ranker 출력이다.*

![MCN public cropper qualitative comparison head-good slice 04](assets_mobilecropnet_v4_master_20260427/fig16_mcn_public_cropper_qualitative_headgood_04.png)

*그림 16-24. Head-good release-gate slice 04. MCN crop tile은 `s=* iou=*`를 표시하며 utility/risk 값은 표시하지 않는다.*

이 비교 패널의 machine-readable manifest는 `assets_mobilecropnet_v4_master_20260427/public_cropper_qualitative_comparison_20260508/mcn_public_cropper_qualitative_comparison_manifest.json`에 있다. Manifest 기준 public cropper GAIC/CGS score는 로컬 RTX 3050 OEM CUDA 환경에서 다시 계산했으며, `skipped_count=0`이다. 이 패널은 "public cropper보다 MCN이 항상 낫다"가 아니라, MCN이 AR별 product crop/head contract를 제공하는 반면 public cropper는 candidate ranking 기준의 free-form crop을 제공한다는 구조적 차이를 보여 주기 위한 정성 비교다.

### 16.4 Teacher/MCN crop-head 성공 예시 overview

다음 두 장은 처음 보는 독자가 teacher와 MobileCropNet의 crop/head 출력 contract를 한눈에 비교할 수 있도록 만든 overview figure다. Teacher overview는 label JSONL에서 `portrait_single`, `object_single`, `object_multi`, `scene_general`, `background_texture_copyspace`를 2장씩 골라 selected crop, subject/support box, decision, checklist, why-tag를 함께 표시한다. MCN overview는 `balanced_288` release-gate no-prior inference의 joint head-quality sample에서 crop hit, route match, decision match가 모두 좋은 10장을 사용한다. 단, MCN 성공 slice는 strict filter 통과 후보 자체가 `background_texture_copyspace`, `scene_general`, `portrait_single`에 편중되어 있어 이 그림도 release-pass evidence가 아니라 "잘 되는 경우의 출력 형태"와 "잘 되는 mode의 편중"을 동시에 보여주는 자료로 읽어야 한다.

![Teacher crop/head 성공 예시 overview](assets_mobilecropnet_v4_master_20260427/fig16_teacher_crop_head_success_gallery.png)

*그림 16-25. Teacher crop/head success gallery. 각 카드에서 원본 overlay, selected crop, subject mode, decision, checklist, why-tag를 한 행으로 묶어 teacher label contract를 보여준다.*

![MobileCropNet crop/head 성공 예시 overview](assets_mobilecropnet_v4_master_20260427/fig16_mcn_crop_head_success_gallery.png)

*그림 16-26. MobileCropNet crop/head success gallery. 각 카드에서 selected crop, teacher/model route, teacher/model decision, score, IoU, decoded checklist/why-tag를 함께 표시한다.*

qualitative review pack은 수치 leaderboard가 놓치는 배포 실패를 확인하기 위한 장치다. 위 overview와 비교 패널은 crop 결과를 크게 보여주고, review pack은 catastrophic bucket을 넓게 샘플링한다. 이 evidence를 함께 보면 crop box IoU가 높아도 route가 틀리거나, subject localization이 낮거나, rationale이 crop과 모순되는 경우를 확인할 수 있다.

최신 qualitative summary는 `sample_count=120`, `candidate_count=5`, `gate_pass=false`다. catastrophic bucket은 `low_subject_iou=120/120`, `route_mismatch=120/120`, `low_proposal_recall=68/120`, `checklist_disagreement=55/120`, `model_rationale_warning=54/120`, `low_top1_iou=44/120`이다. route mismatch와 low subject IoU가 전 샘플에서 발생한 것은 단순 threshold issue가 아니라 representation/head/generalization 문제로 봐야 한다.

| failure family | visual symptom | likely upstream cause | affected head | recommended fix |
| --- | --- | --- | --- | --- |
| low subject IoU | crop은 그럴듯하지만 subject box가 주요 피사체를 놓침 | subject-box valid/box calibration 부족 | subject-box, route, checklist | content-relative/refine path 유지, valid-balanced smoke 강화 |
| route mismatch | portrait/object/scene label이 overlay 의미와 불일치 | route collapse, subject structure 부족 | route, applicability, why-tag | route-balanced sampler, subject-conditioned route feature |
| low proposal recall | teacher positive와 겹치는 generated proposal이 부족 | proposal branch가 candidate-bank teacher surface를 internalize 못함 | proposal, utility | generated proposal alignment, proposal-subject loss 강화 |
| checklist disagreement | visible crop failure와 checklist label이 충돌 | applicability mask 또는 detail score learning 약함 | checklist/detail | score-first rationale, route-conditioned decode |
| model rationale warning | why-tag가 generic하거나 crop과 무관 | BCE-only rationale와 route mismatch | why-tag, macro | why-tag를 detail/checklist score에서 부분 rule-derived로 보강 |
| low top1 IoU | top utility crop이 positive와 낮게 겹침 | ranker/proposal surface gap | utility, positive, proposal | top-return, top-k coverage, generated top-M rerank 개선 |
| catastrophic portrait failure | face/head/joint가 잘리거나 lookroom/headroom 불량 | portrait route/proposal/checklist 동시 실패 | route, subject, risk, checklist | portrait-focused mini-pack과 hard-negative mining |

정성 실패는 "눈으로 보니 아직 별로"라는 보조 의견이 아니다. release gate에서는 catastrophic bucket이 남아 있으면 shipping candidate를 선언하지 않는다. 특히 route mismatch와 low subject IoU는 explanation과 action consistency까지 함께 무너뜨리므로 제품 신뢰도를 직접 손상한다.

## 17. 지연 시간과 배포 프로파일

![Inference latency profile](assets_mobilecropnet_v4_master_20260427/fig08_inference_latency_profile.png)

*그림 17-1. Inference latency profile. RTX 3050 forward-only와 A100 산출물은 장비/세대가 달라 절대 비교보다 profile trade-off 확인용으로 해석한다.*

latency evidence는 현재 primary blocker가 아니다. RTX 3050 forward-only benchmark에서 `sstk_public_balanced288`은 mean `6.827 ms`, p95 `8.844 ms`, `146.48 img/s`를 보였고, `sstk_public_rank320`은 mean `7.517 ms`, `sstk_public_plus384`는 mean `19.632 ms`다. tiny `sstk_t1_q24`는 mean `3.470 ms`로 매우 빠르지만 품질/release 측면에서 주 후보가 아니다.

아래 표는 profile별 추론 시간과 public cropper scoring 시간을 같은 형식으로 정리한 것이다. MobileCropNet row는 synthetic model forward-only benchmark라 image decode, disk I/O, crop rendering, platform postprocess를 제외한다. Public cropper `GAIC`/`CGS` row는 로컬 `NVIDIA GeForce RTX 3050 OEM`에서 tensor와 RoI 입력을 미리 만든 뒤 candidate ranker forward와 CUDA synchronization만 측정한 값이다. 따라서 public cropper의 `mean ms/output`은 image group 하나에 대해 후보 crop set을 scoring하는 비용이며, 후보 생성과 image decode/resize/preprocess/output JSON write는 제외한다. `S2CNet`/`CACNet` 추가 row는 remote A100 interactive full grouped export wall-clock에서 계산한 값으로 image decode/resize/preprocess와 JSON output 비용을 포함하므로 RTX 3050 forward-only row와 절대 비교하지 않는다. 기존 full grouped export wall-clock 기준은 ensemble 비교용으로 note에 남긴다.

| method | device/artifact | input | candidates | params M | mean ms/output | p50 ms | p95 ms | img/s | peak mem MB | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MobileCropNet v4 `q24_288_w075` | NVIDIA GeForce RTX 3050 OEM | 288 | 24 | 0.252 | 3.470 | 3.137 | 4.915 | 288.18 | 15.5 | `artifacts/mobilecropnet_v4/inference_latency/rtx3050_sstk_shortlist_20260424.json` |
| MobileCropNet v4 `turbo_256` | NVIDIA A100-SXM4-80GB | 256 | 16 | 2.256 | 12.636 | 12.603 | 12.946 | 79.14 | 49.3 | `artifacts/mobilecropnet_v4/inference_latency/a100_mcn_v4_primary_20260420_114321.json`; RTX 3050 completed latency artifact 없음 |
| MobileCropNet v4 `balanced_288` | NVIDIA GeForce RTX 3050 OEM | 288 | 24 | 8.225 | 6.827 | 6.715 | 8.844 | 146.48 | 154.3 | `artifacts/mobilecropnet_v4/inference_latency/rtx3050_sstk_shortlist_20260424.json` |
| MobileCropNet v4 `hq_320` | - | 320 | 32 | - | - | - | - | - | - | 로컬/원격 latency artifact에서 stable completed row 미확인 |
| MobileCropNet v4 `rank_320` | NVIDIA GeForce RTX 3050 OEM | 320 | 32 | 9.768 | 7.517 | 7.195 | 9.493 | 133.04 | 180.4 | `artifacts/mobilecropnet_v4/inference_latency/rtx3050_sstk_shortlist_20260424.json` |
| MobileCropNet v4 `hybrid_384` | NVIDIA GeForce RTX 3050 OEM | 384 | 48 | 13.256 | 10.256 | 10.282 | 11.183 | 97.51 | 244.0 | `artifacts/mobilecropnet_v4/inference_latency/rtx3050_sstk_t1_hybrid384_20260428.json` |
| MobileCropNet v4 `plus_384` | NVIDIA GeForce RTX 3050 OEM | 384 | 48 | 32.341 | 19.632 | 18.613 | 24.864 | 50.94 | 589.1 | `artifacts/mobilecropnet_v4/inference_latency/rtx3050_sstk_shortlist_20260424.json` |
| MobileCropNet v4 `quality-first cnv2b_448` | NVIDIA A100-SXM4-80GB | 448 | 64 | 104.838 | 41.754 | 41.705 | 42.373 | 23.95 | 636.8 | `artifacts/mobilecropnet_v4/quality_first_20260428/final_eval_bundle/mcn-q-cnv2b448-topagree-imgresroute-r1-routeembed-releasefull-rerun2-20260430/latency/latency.json` |
| Public cropper `GAIC` | NVIDIA GeForce RTX 3050 OEM | min-side 256 | 18.5 avg (11-23) | 2.910 | 30.049 | 2.492 | 175.451 | 33.28 | 252.1 | `artifacts/mobilecropnet_v4/inference_latency/public_cropper_rtx3050_20260508/public_cropper_rtx3050_forward_latency.json`; candidate ranker forward-only |
| Public cropper `CGS` | NVIDIA GeForce RTX 3050 OEM | min-side 256 | 18.5 avg (11-23) | 21.254 | 60.553 | 32.981 | 349.714 | 16.51 | 417.4 | `artifacts/mobilecropnet_v4/inference_latency/public_cropper_rtx3050_20260508/public_cropper_rtx3050_forward_latency.json`; candidate ranker forward-only |
| Public cropper `S2CNet` | NVIDIA A100-SXM4-80GB interactive | min-side 256 | 49.27 avg grouped | 5.796 | 62.717 | - | - | 15.94 | - | `artifacts/mobilecropnet_v4/public_cropper_extension_20260508_remote/gpu_interactive_full_20260508_093532/public_benchmark_predictions/public_cropper_benchmark_predictions_summary.json`; full grouped export wall-time, image decode/preprocess/output 포함 |
| Public cropper `CACNet` projection | NVIDIA A100-SXM4-80GB interactive | 224 | 49.27 avg grouped | 19.525 | 52.428 | - | - | 19.07 | - | same full grouped export summary; single crop regression 후 candidate IoU projection, image decode/preprocess/output 포함 |
| Public cropper `GAIC+CGS ensemble` | cuda, GPU model not recorded | - | 1,109,036 scored total | - | 15.591 | - | - | 64.14 | - | full grouped export wall time; `43.295 ms/scored image group`; combines GAIC and CGS scoring |

Public cropper `GAIC`는 `MobileNetV2` backbone을 쓰는 multi-scale crop candidate ranker다. 후보 crop을 RoIAlign/RoDAlign feature로 변환하고 dimension reduction과 convolution scoring head로 MOS rank score를 낸다. 이번 Py3.10/RTX 3050 측정에서는 GAIC weight를 그대로 쓰되 bundled extension 호환성 때문에 CGS 쪽 RoI/RoD align extension을 재사용했다. `CGS`는 `VGG16` multi-level `RegionFeatureExtractor`로 후보 region feature를 만든 뒤 `CroppingGraph` GNN relation ranker로 후보 간 composition 관계를 scoring한다. 두 public cropper 모두 주어진 후보 crop set을 평가하는 ranker라 `image + target_ar`만 받는 MobileCropNet no-prior 제품 forward와 구조적 역할이 다르다.

추가 측정한 `S2CNet`은 `MobileNetV2` backbone, crop RoI/RoD feature, 5개 graph node, graph-attention relation block을 결합한 candidate ranker다. 원 논문 protocol은 object detector에서 나온 semantic box를 graph node로 쓰지만, 이번 full benchmark adapter는 staged manifest에 object box가 없어서 후보 crop set에서 deterministic node를 만들었다. `CACNet`은 `VGG16` backbone, composition CAM, crop regression head를 쓰는 단일 crop regressor이며, public benchmark에서는 emitted crop과 각 candidate의 IoU를 score로 투영했다. 따라서 이 둘의 A100 timing row는 모델 비교 참고값이며, S2CNet은 protocol adaptation, CACNet은 projection-only라는 제한을 함께 읽어야 한다.

![Latency profile and public cropper comparison](assets_mobilecropnet_v4_master_20260427/fig17_latency_profile_public_cropper_comparison.png)

*그림 17-2. Latency profile and public cropper comparison. MobileCropNet row는 synthetic forward-only latency이고, public cropper `GAIC`/`CGS` row는 로컬 RTX 3050에서 precomputed tensor/RoI 입력에 대해 측정한 candidate-ranker forward-only latency다. Public cropper는 후보 crop set scoring 비용이며 no-prior proposal/runtime 전체 비용은 아니다.*

대표 비교만 따로 보면, `balanced_288`과 `rank_320`은 평균 forward latency가 각각 `6.827 ms`, `7.517 ms`이고 `plus_384`는 `19.632 ms`다. 같은 RTX 3050 forward-only 측정에서 public cropper `GAIC`/`CGS`는 각각 `30.049 ms`, `60.553 ms`이며, p95 tail은 후보 수와 RoI/RoD candidate scoring 경로 때문에 훨씬 크다.

![Representative latency subset](assets_mobilecropnet_v4_master_20260427/fig17_latency_subset_mcn_public_cropper.png)

*그림 17-3. Representative latency subset: MCN profiles vs public croppers. Figure 17-2에서 대표 MCN profile `balanced_288`, `rank_320`, `plus_384`와 public cropper `GAIC`, `CGS`만 추려 mean/p95 forward latency를 분리해 표시했다.*

`balanced_288`과 `rank_320`은 latency만 보면 on-device 후보권에 있다. `plus_384`는 품질 탐색 또는 high-quality profile에 가깝다. A100 측정값은 workload/checkpoint 세대가 달라 RTX 값과 같은 축에서 절대 비교하면 안 된다. 현재는 quantization, NPU export, operator coverage보다 route/subject/proposal/policy quality gate가 먼저다.

모바일/NPU 배포로 가려면 이후 별도 단계에서 operator coverage, INT8/QAT drift, memory footprint, CPU fallback, image preprocessing 비용, exact AR postprocess의 platform implementation을 확인해야 한다. 하지만 strict release gate가 quality에서 막힌 상태에서는 latency 최적화가 다음 병목이 아니다.

## 18. 현재 강점

MobileCropNet v4.0의 가장 큰 강점은 학습 계약이 풍부하다는 점이다. 단일 bbox target이 아니라 candidate bank, pairwise/listwise preference, route, decision, subject prior, risk, checklist, why-tag를 모두 읽는다. 이 구조는 crop quality, product safety, explanation을 같은 모델 안에서 학습할 수 있는 기반이다.

두 번째 강점은 no-prior runtime path가 이미 구현되어 있다는 점이다. `boxes=None` 경로에서 generated proposals를 후보로 사용하고, full/minimal runtime baseline을 내부 생성하며, executor와 exact target-AR repair까지 포함한다. 즉 deployment surface를 평가할 수 있는 코드 경로가 존재한다.

세 번째 강점은 평가 체계가 다면적이라는 점이다. public equal-4, GAIC official, Product-AR direct, head audit, subject-box report, qualitative review, latency, release gate가 분리되어 있어, 하나의 metric이 좋아 보이는 착시를 줄인다. strict release gate가 현재 모든 후보를 막고 있다는 사실 자체도 시스템이 제품 실패를 감추지 않고 있다는 강점이다.

마지막으로 artifact durability가 좋다. training/eval/release 산출물이 JSON/PNG/MD로 남아 있어 다음 Cline SR 기반 GPU workflow가 상태를 이어받을 수 있다.

## 19. 현재 차단 요인

현재 blocker는 teacher score 부족이 아니라 student runtime internalization 부족이다. 첫 번째 blocker는 route generalization이다. route collapse가 반복되고, smoke best도 `0.50` threshold를 안정적으로 넘지 못했다.

두 번째 blocker는 subject-box calibration이다. SSTK UCTR patch는 subject IoU `0.515918`까지 올렸지만, GAIC 계열과 full release bundle에서는 아직 route/crop/subject gate가 함께 닫히지 않는다. valid positive/negative calibration도 별도 관리가 필요하다.

세 번째 blocker는 generated proposal alignment다. candidate-bank replay 성능과 generated runtime proposal 성능 사이에 gap이 남아 있다. route smoke에서 generated align top1 hit가 `0.0`인 경우는 이 문제가 단순 ranker 문제가 아님을 보여준다.

네 번째 blocker는 policy/action consistency다. decision label collapse 이력이 있고, decision accuracy `1.0`을 그대로 믿을 수 없는 row가 있다. policy score와 baseline/crop utility gap이 일관되어야 executor가 제품 의미를 가진 action을 낸다.

다섯 번째 blocker는 qualitative catastrophic failure다. `route_mismatch=120/120`, `low_subject_iou=120/120`인 qualitative pack은 정량 상위 후보도 shipping할 수 없다는 강한 evidence다. 추가로 routefix/subject patch/release-gate v2 row 중 promising row 다수는 아직 동일 final bundle이 완결되지 않았다.

## 20. 다음 실험 계획

다음 단계는 full rerun이 아니라 bounded smoke에서 route/subject/proposal/policy 병목을 먼저 깨는 것이다.

| step | experiment | success criterion |
| --- | --- | --- |
| 1 | route + subject joint smoke | route balanced accuracy `>=0.52`, subject IoU `>=0.50`, valid balanced acc `>=0.52` |
| 2 | route-balanced sampling audit | portrait/object/scene/copyspace per-class recall이 한 class collapse 없이 개선 |
| 3 | subject-conditioned proposal alignment | generated proposal recall@5 IoU0.5 `>=0.90`, generated align top1 hit `>0.20` smoke 달성 |
| 4 | score-first rationale heads | checklist disagreement와 model rationale warning mini-pack에서 50% 이상 감소 |
| 5 | policy/action calibration | decision crop probability와 utility crop preference gap 감소, decision collapse 없음 |
| 6 | qualitative mini-pack | low subject IoU와 route mismatch bucket이 smoke pack에서 `<=30%` |
| 7 | final bundle rerun | equal-4, GAIC official, Product-AR direct, head audit, subject-box, qualitative, latency 모두 같은 checkpoint로 채움 |
| 8 | release gate rerun | strict release gate pass count `>=1` |
| 9 | deploy candidate declaration | checkpoint, config, inference command, runtime contract, fallback policy를 manifest에 고정 |

정량 목표는 보수적으로 둔다. strict threshold가 route balanced `0.50`이라면 smoke target은 `0.52` 이상이어야 한다. subject IoU는 `0.50` 이상, subject valid negative acc는 `0.60` 이상, direct hit@0.5는 current top row 수준인 `0.93` 이상을 유지해야 한다. target-AR compatibility는 `0.999` 이상, risk hit@0.5는 current strong row 수준인 `0.05` 이하를 목표로 둔다. latency는 `balanced_288/rank_320` 계열에서 RTX 3050 forward-only mean `10 ms` 이하를 유지하는 것이 1차 기준이다.

### 20.1 품질 우선 백본 확장 계획

이 항목은 원래 마스터 리포트의 기존 계획이 아니라, 2026-04-27 사용자 지시로 새로 추가된 후속 실험 축이다. 현재 MobileNet 계열은 속도 제약에는 유리하지만 route/subject/proposal/policy head의 표현력이 release gate를 반복적으로 막고 있으므로, 다음 smoke부터는 온디바이스 지연 시간 제약을 완화하고 정량 품질과 정성 실패 감소를 먼저 본다.

우선순위는 두 모델군이다. 첫 번째는 `quality_cnv2b_448`로, `ConvNeXtV2 Base / fcmae_ft_in22k_in1k_384` pretrained weight, 입력 `448`, candidate/proposal `64`, token dim `384`, set-transformer depth `4`를 사용한다. 두 번째는 `quality_cnv2l_512`로, `ConvNeXtV2 Large / fcmae_ft_in22k_in1k_384` pretrained weight, 입력 `512`, candidate/proposal `80`, token dim `512`, set-transformer depth `4`를 사용한다. 두 weight는 `weights/hf/timm/<backbone>/model.safetensors`로 고정해 원격 GPU workload가 외부 다운로드 없이 같은 pretrained checkpoint를 로드하도록 한다.

새 selection metric은 `release_gate_joint`다. 이 metric은 route balanced accuracy, subject IoU, subject valid balanced accuracy, generated proposal top1/recall, action consistency를 함께 보며, crop-only score가 좋아도 route 또는 generated proposal이 붕괴한 epoch을 best로 확정하지 않도록 joint floor를 둔다. 이는 v3 smoke에서 epoch 1은 generated alignment가 높지만 route가 낮고, 마지막 epoch은 route가 올랐지만 generated alignment가 `0.0`으로 붕괴한 충돌을 반영한 조치다.

품질 우선 smoke의 통과 기준은 기존 release gate와 동일하거나 더 엄격하다. bounded smoke에서 route balanced `>=0.52`, subject IoU `>=0.50`, subject valid balanced `>=0.52`, generated align top1 hit `>0.20`, generated proposal recall@5 IoU0.5 `>=0.90` 중 최소 4개 이상이 양호해야 full final bundle로 확장한다. 그 뒤에 equal-4, GAIC official, Product-AR direct, subject-box calibration, route balanced accuracy, policy/action alignment, target-AR compatibility, qualitative review pack, latency를 같은 checkpoint로 재평가한다.

현재까지의 품질 우선 smoke 결과는 문제 축을 더 좁혔다. `quality_cnv2l_512` 계열은 pretrained Large/512를 써도 joint gate가 낮아 제외한다. `mcn-quality-b448-uctr-genrank-ft-r1-20260427`은 generated align top1 `0.950893`, proposal recall `0.920048`, route balanced `0.505222`까지 회복했지만 subject IoU `0.392023`, valid balanced `0.376161`로 막혔다. `mcn-quality-b448-public-genrank-r1-20260427`은 route balanced `0.506953`까지 갔지만 proposal/generation이 `0.0`으로 붕괴했다. 따라서 후속 실험은 `UCTR init -> public/UCTR transfer -> spatial subject head -> calibrated valid threshold -> routecal` 순서로 좁혀 진행한다.

실제 코드에는 subject valid threshold를 release gate surface로 올렸다. `evaluate_mobilecropnet_v4.py`, `evaluate_mobilecropnet_v4_product_ar_direct.py`, `audit_mobilecropnet_v4_release_heads.py`, `infer_mobilecropnet_v4.py`는 이제 `--subject_valid_threshold` 또는 `--valid_conf_threshold`를 통해 0.5 고정값 대신 calibrated threshold를 기록하고 적용한다. 진행 중인 spatial/routecal smoke에서 validation sweep best threshold는 대체로 `0.60` 전후이며, best valid balanced accuracy는 `0.526~0.567` 범위로 0.5 고정보다 낫다. 이 개선은 subject valid gate에는 유효하지만, subject bbox IoU 자체가 `0.39~0.40`대라 localization 구조 개선이 남은 blocker다.

이 blocker를 겨냥해 `subject_box_spatial_mix_bias`를 모델 config로 승격했고, `run_mobilecropnet_v4_route_gate_smoke.sh`에서 `SUBJECT_BOX_SPATIAL_AUX_WEIGHT`, `SUBJECT_BOX_SPATIAL_MIX_BIAS`, quality spatial row/step 한도를 실험별로 조정할 수 있게 했다. 후속 run `1099407`(`mcn-quality-b448-public-spatial-strongsubj-r2-20260427`)은 MLP workload로 실행 중이고, `1099408` UCTR Pending run은 중복 방지를 위해 종료한 뒤 interactive GPU `10.2.4.134`에서 `mcn-quality-b448-uctr-spatial-strongsubj-r2-ip134-20260427`로 재실행했다. 추가로 train rows/steps를 늘린 `1099414`(`mcn-quality-b448-uctr-spatial-strongsubj-full-r3-20260427`)를 생성해 validation 일반화 여부를 확인한다.

추가 구조 패치로 `subject_box_spatial_multiscale`도 도입했다. 이 옵션은 ConvNeXtV2 final stride feature 대신 penultimate 고해상도 feature를 spatial subject head에 공급한다. 목적은 448 입력에서 약 `14x14`인 final map으로는 subject center/size localization이 부족할 수 있다는 가설을 검증하는 것이다. multiscale smoke `1099424`(`mcn-quality-b448-uctr-spatial-ms-r1-20260427`)와 `1099425`(`mcn-quality-b448-public-spatial-ms-r1-20260427`)를 생성했으며 Pending 상태가 10분 이상 지속되면 종료 후 interactive/MLP 경로로 재생성한다.

2026-04-27 20:12 KST 기준 추가 결론은 다음과 같다. YOLOv8n/s/m/l을 내부 subject detector prior로 쓰는 축은 subject IoU `0.401749~0.429757` 범위에서 멈췄고, DINOv2-L subject-only와 DINOv2-B frozen/head-only도 각각 epoch2 subject IoU `0.25005`, `0.36964`로 중단했다. 큰 detector나 foundation backbone을 별도 prior로 붙이는 방식은 현재 release contract의 `image + target_ar only` 내부 일관성 문제를 해결하지 못했다.

그래서 후속 실험은 teacher가 이미 제공한 dense evidence를 더 직접적으로 쓰는 방향으로 바꿨다. `data.py`는 UCTR label의 `subject_support_overlay.support_mass_grid_f16_b64`를 `24x24` float mask로 디코딩해 `subject_support_mask`와 `subject_support_mask_valid`를 배치에 넣는다. `model.py`는 이 mask가 있으면 spatial heatmap CE와 mask BCE target을 bbox rectangle/gaussian 대신 support map으로 학습한다. subject target oracle 분석에서 같은 이미지의 target_AR별 subject bbox가 사실상 동일(`image leave-one-out IoU≈0.995`)하므로, 이 축은 AR별 crop 차이를 학습하기보다 이미지 내 subject localization 자체를 강화하는 목적이다.

단, support grid 자체를 bbox oracle로 과대해석하면 안 된다. 로컬 val split `4,947` row 중 valid subject `3,671` row에서 support grid를 threshold bbox로 바꿔 teacher bbox와 비교하면 best mean IoU는 threshold `0.15` 기준 `0.400601`, hit@0.5는 `0.276219`다. 따라서 support-map smoke의 목적은 최종 bbox target 대체가 아니라 spatial attention/heatmap regularization이며, 결과가 낮으면 모델 decoder뿐 아니라 support-map target 품질도 같이 재검토한다. 분석 산출물은 `artifacts/mobilecropnet_v4/quality_first_20260427/support_map_oracle_analysis/SUPPORT_MAP_ORACLE_ANALYSIS_KO.md`에 둔다.

또한 `--subject_box_spatial_output {mix,spatial,coarse}`를 추가했다. 기존 spatial head는 raw spatial IoU가 fused IoU보다 약간 높은 경우가 있었지만 최종 출력이 coarse/spatial mix라 spatial branch 이득이 희석될 수 있었다. support-map smoke `1099541`은 pending 10분 초과로 종료했고, 새 run `1099544`(`mcn-quality-cnv2b448-supportmap-spatial-r4-20260427`)이 2026-04-27 20:22:58 KST에 Running으로 전환됐다. 이 run은 `--subject_box_spatial_output spatial`로 최종 subject bbox를 spatial branch로 강제해 이 가설을 검증한다.

`1099544`의 epoch 1 validation은 subject IoU `0.395795`, subject valid best balanced `0.559722`, spatial mask IoU `0.276556`, spatial mix `1.0`이다. epoch 2/3에서도 subject IoU는 `0.391171`/`0.402788`, route balanced는 `0.093747`/`0.113889`로, localization은 기존 강한 ConvNeXt-B 448 spatial strong-subject run의 `subject_box_iou=0.414415`보다 낮고 route는 붕괴 상태다. 최종 5 epoch도 best epoch 3 subject IoU `0.402788`, 최종 epoch subject IoU `0.405113`, route balanced `0.115741`이라 raw-target support-map 단독 축은 배포 후보에서 제외한다.

epoch 2/3 결과가 gate 미달이라 사용자 지시의 품질 우선/대입력 축을 실제 실행으로 확장했다. 입력 `640`, ConvNeXtV2-Base pretrained, support-map dense supervision, spatial-output 강제를 묶은 `1099554`(`mcn-quality-cnv2b640-supportmap-spatial-r1-20260427`)는 2026-04-27 20:38:44 KST에 Running으로 전환됐고, 4 epoch에서 best subject IoU `0.319199`, route balanced `0.216667`로 종료됐다. 따라서 입력 크기만 키우는 방식은 현재 label/target 불일치 문제를 해결하지 못했다.

추가 분석 결과, 현재 `legacy` subject target인 `routing.subject_prior_bbox_norm_xyxy`/overlay bbox와 `support_spec.latent_support_bbox_norm_xyxy`, `core_bbox_norm_xyxy`, `envelope_norm_xyxy`가 크게 다르다. 로컬 val `4,947` row 기준 raw와 core 평균 IoU는 `0.428982`, raw와 latent 평균 IoU는 `0.494250`, raw와 envelope 평균 IoU는 `0.599176`이다. 반대로 support mass grid threshold bbox는 latent/core/envelope target에는 평균 IoU `0.72~0.98`로 매우 높다. 즉 support-map mask를 쓰면서 bbox regression target은 legacy raw를 쓰는 구성이 서로 충돌할 수 있다.

이를 분리하기 위해 `data.py`에 `--subject_box_target_source {legacy,raw,overlay,support_latent,support_core,support_envelope,support_hybrid}`를 추가했다. `train_mobilecropnet_v4.py`, `evaluate_mobilecropnet_v4.py`, `evaluate_mobilecropnet_v4_product_ar_direct.py`, `audit_mobilecropnet_v4_release_heads.py`는 이 값을 dataset에 전달하고, eval/direct/audit은 checkpoint의 `train_config.subject_box_target_source`를 자동 승계한다. 로컬/원격 py_compile, dataset smoke, `tests/test_mobilecropnet_v4.py -q` 10개 통과 후 원격에 동기화했다. 분석 artifact는 `artifacts/mobilecropnet_v4/quality_first_20260427/subject_target_source_alignment/SUBJECT_TARGET_SOURCE_ALIGNMENT_KO.md`와 `summary.json`에 고정했다. 새 subject-only smoke는 `1099561`(`support_core`)과 `1099562`(`support_latent`)이며, `1099561`은 environment 단계에서 멈춰 종료 후 `1099568` r2로 재생성했다. `1099562`는 best epoch3 subject IoU `0.461767`, subject valid best balanced `0.571429`로 종료됐고, no-mask ablation `1099575`는 best epoch4 subject IoU `0.463982`, valid best balanced `0.567857`로 종료됐다. raw-target support-map best `0.402788`보다 높아 target-source alignment 가설은 유효하지만 아직 gate `0.50`에는 미달이다. `1099568` support-core r2는 best subject IoU `0.403733`으로 낮아 제외한다.

target-source alignment가 product head에도 이어지는지 보기 위해 joint smoke `1099565`(`support_latent + route_image_only`)를 생성했으나 Pending 10분 초과로 종료했고, 동일 목적 r2 `1099573`을 생성했다. `1099573`은 epoch4 기준 subject IoU `0.477894`, spatial IoU `0.481567`, generated align top1 `0.951786`, proposal recall `0.923052`까지 회복했지만 route balanced `0.355028`로 gate 미달이다. public-transfer r2 `1099564`는 best epoch5 subject IoU `0.402462`, route balanced `0.398710`, generated align top1 `0.935714`, proposal recall `0.893815`로 종료되어 배포 후보가 아니다. support-latent target에서 backbone 표현력을 다시 본 DINOv2-B/518 no-mask `1099581`은 best subject IoU `0.355065`로 제외했고, EVA02-B/448 `1099580`도 best subject IoU `0.450588`로 ConvNeXt-B보다 낮아 제외한다.

2026-04-27 21:24 KST 추가 구조 패치로 `--subject_box_spatial_box_mode {regress,mask_moment,mask_moment_regress,heatmap_moment,heatmap_moment_regress}`와 `--subject_box_spatial_extent_scale`을 추가했다. 기존 spatial box head는 heatmap/mask가 target source와 맞아도 width/height를 MLP가 다시 회귀하므로 support mask의 공간 분포를 충분히 쓰지 못할 수 있다. 새 `mask_moment_regress` 모드는 sigmoid mask 질량의 중심과 분산으로 bbox 중심/크기를 계산하고 MLP는 작은 residual만 보정한다. 이 변경은 로컬 `py_compile`, `tests/test_mobilecropnet_v4.py -q` 10개, 원격 py_compile을 통과했고 원격 shared storage의 `src/mobilecropnet_v4/model.py`, 루트 패키지 `mobilecropnet_v4/model.py`, `src/scripts/train_mobilecropnet_v4.py`에 동기화했다. 병렬 smoke는 `1099587`(`support_envelope + route_image_only`), `1099588`(`support_latent + route_image_only`), `1099589`(`support_latent subject-only`), `1099590`(`support_latent + internal predicted subject prior route`)이다. `1099590`은 외부 teacher prior 없이 내부 예측 subject bbox/token을 route/policy/proposal에 재사용하므로 `image + target_ar only` release contract와 호환된다. 추가로 원격에 pretrained `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k` weight가 존재함을 확인하고, SwinV2-B/384 support-latent `mask_moment_regress` subject-only `1099591`도 생성했다.

`1099587` epoch1에서 spatial IoU `0.467846` 대비 final subject IoU가 `0.443995`로 낮아졌다. 이는 proposal-conditioned subject refinement가 moment spatial box를 보정하기보다 망가뜨리는 경우가 있다는 신호다. 따라서 `run_mobilecropnet_v4_route_gate_smoke.sh`의 `quality_cnv2b448_spatial_genrank_lr2e5` variant에 `SUBJECT_BOX_REFINE_WITH_PROPOSALS=0` 제어를 추가했고, no-refine ablation `1099592`(`mcn-q-b448-slat-mmreg-noref-r1-20260427`)을 생성했다. `1099589` subject-only r1은 Pending 10분 초과로 종료하고 동일 목적 r2 `1099594`를 재생성했다. 또한 `1099590` internal-prior route는 epoch2에서 route balanced `0.492590`, spatial IoU `0.487953`, generated align top1 `0.950781`, proposal recall `0.907938`까지 올라왔지만 final subject IoU는 `0.456482`로 낮아져, 내부 예측 subject prior는 유지하되 subject refinement를 끄는 `1099595`도 생성했다. `1099592` no-refine route-image-only epoch1은 subject IoU `0.480478`을 보였지만 route balanced `0.221049`로 낮아, route 해결은 internal-prior 축에 집중한다. route가 내부 subject box geometry만으로 부족할 가능성에 대비해 `--route_use_subject_spatial_token`도 추가했다. 이 옵션은 subject spatial head의 appearance/location token을 route head 입력에 결합하며 외부 prior를 쓰지 않는다. 로컬/원격 검증 후 internal-prior + no-refine + spatial-token route smoke `1099596`을 생성했다.

subject spatial IoU가 `0.49` 근처에서 멈추는 문제에 대해 bbox extent scale도 분리한다. support mask 질량이 중심에 몰린 경우 variance 기반 width/height가 support-latent bbox보다 작아질 수 있으므로, `subject_box_spatial_extent_scale=1.15` subject-only smoke `1099597`을 추가 생성했다. SwinV2-B/384 support-latent `mask_moment_regress` subject-only `1099591`은 best subject IoU `0.470635`로 종료되어 ConvNeXt-B no-refine/moment 계열보다 낮다.

중간 결과상 no-refine은 subject/proposal 측면에서 유효하다. `1099588` support-latent route-image-only moment-regress는 epoch3 final subject IoU `0.481594`, spatial IoU `0.489662`, generated align top1 `0.957031`, proposal recall `0.927042`이나 route balanced `0.332620`이다. `1099592` no-refine route-image-only는 epoch3 subject IoU `0.493457`, generated align top1 `0.955469`, proposal recall `0.955031`까지 올랐지만 route balanced `0.280921`로 낮다. `1099590` internal-prior route는 epoch3 spatial IoU `0.496478`까지 접근했지만 route balanced가 epoch2 `0.492590`에서 epoch3 `0.443026`으로 떨어졌다. 따라서 다음 핵심은 no-refine subject quality와 internal-prior/spatial-token route signal을 같은 checkpoint에서 유지하는 것이다.

`1099595` internal-prior + no-refine epoch1은 subject IoU `0.488749`, valid best balanced `0.613103`, generated align top1 `0.953125`, proposal recall `0.918103`으로 subject/proposal은 유지했지만 route balanced `0.397120`이다. 이 결과만으로는 full gate 확장 대상이 아니며, route 입력에 subject spatial token을 더한 `1099596`이 다음 판단 포인트다.

`1099596`은 Pending 10분 초과로 종료하고 동일 spatial-token route ablation을 `1099599` r2로 재생성했다.

route 쪽에는 `--route_image_only` ablation을 추가했다. 이 옵션은 variant가 `route_use_candidate_context`나 `route_use_subject_*`를 켜더라도 route head 입력을 image global feature로만 제한한다. route head가 target_AR embedding이나 candidate context에 과도하게 의존하면 crop geometry와 class imbalance를 route로 오인할 수 있으므로, support-map epoch 3의 route balanced `0.113889` 확인 직후 `1099557`(`mcn-quality-b448-uctr-routeimg-spatial-r1-20260427`)과 `1099558`(`mcn-quality-b448-public-routeimg-fromuctr-r1-20260427`)을 생성했다. `1099558`은 Pending 10분 초과로 종료하고 `1099564` r2로 재생성했다. 이 run들은 기존 최고 UCTR spatial-strong checkpoint를 초기값으로 쓰고, `--route_image_only --subject_box_spatial_output spatial` 조건에서 UCTR/public transfer를 분리 검증한다.

final leaderboard 공백을 채우기 위해 row 26/29/30/33/34/35/36/37도 병렬 final eval workload로 생성했다. 이 중 row 26/29/30은 manifest가 가리키는 checkpoint가 원격 full-run 디렉터리에 없어 `Missing checkpoint`로 즉시 실패했다. 따라서 해당 row는 GPU 재시도 대상이 아니라 invalid checkpoint manifest row로 분리한다. row 33은 pending 10분 초과 후 `1099553`으로 재생성해 Running으로 전환됐고, row 34/35/36/37도 Running이다.

### 20.2 2026-04-27 22시대 품질 우선 후속 실행 상태

`support_latent + mask_moment_regress + no-refine` 축은 subject/proposal/action을 처음으로 gate 근처까지 끌어올렸지만, route가 여전히 release blocker다. `1099592`는 subject IoU `0.502281`, valid best balanced `0.590603`, generated align top1 `0.957031`, proposal recall `0.958128`, action consistency `0.999219`로 강하지만 route balanced `0.401620`이다. `1099595`는 내부 예측 subject prior를 route/policy/proposal에 재사용하는 compliant 경로에서 subject IoU `0.502356`, valid best balanced `0.596503`, generated align top1 `0.957031`, proposal recall `0.953753`까지 유지했지만 route balanced `0.453749`다. `1099599`는 subject spatial token을 route 입력에 추가해 route balanced `0.464961`까지 올렸고 subject IoU `0.500339`를 유지했지만, smoke target `0.52`에는 못 미친다.

단순 route bias calibration은 blocker를 풀지 못했다. `1099595` best checkpoint 기준 eval route balanced는 `0.422116 -> 0.425842`로만 개선됐다. 따라서 route 문제는 후처리 bias가 아니라 route representation, class-balanced 학습, label taxonomy 정렬 문제로 본다.

라벨 taxonomy hygiene도 바로잡았다. 현재 UCTR train/val 라벨은 `0 background_texture_copyspace`, `1 object_multi`, `2 object_single`, `3 portrait_group`, `4 portrait_single`, `5 scene_general`이며 `other_ambiguous`는 현재 split에 없다. `SUBJECT_MODE_VOCAB`을 이 순서에 맞춰 수정했고, runtime decode/checklist applicability/qualitative taxonomy가 같은 source of truth를 보게 했다. 로컬 `tests/test_mobilecropnet_v4.py -q` 10개와 원격 import 확인을 통과했다.

old-vocab 상태에서 시작된 routecal `1099621`/`1099624`는 margin loss class 해석이 어긋날 수 있고 route도 개선되지 않아 종료했다. corrected-vocab routecal은 `1099631`(margin 유지)과 `1099632`(margin 제거)로 재생성했다. `1099628` route-best subject-head graft는 subject head만 학습하는 구성에서 route 입력 경로가 기존 route-best와 달라 route가 붕괴했으므로 배포 후보가 아니라 ablation으로만 기록한다.

다음 구조 축으로 route MLP를 확장했다. `MobileCropNetV4`와 `train_mobilecropnet_v4.py`에 `--route_head_depth`, `--route_head_hidden_mult`, `--route_head_dropout`을 추가했고 기본값은 기존 route head와 checkpoint shape를 유지한다. corrected-vocab r3가 gate를 넘지 못하면 같은 support-latent/no-refine checkpoint에서 deep-route head smoke를 바로 실행한다.

interactive GPU `10.2.4.134`의 `quality_cnv2l_512` support-latent no-refine subject-only run은 best subject IoU `0.357005`, valid best balanced `0.508333`으로 종료됐다. 따라서 온디바이스 제약을 완화한 대형 백본 축은 계속 검증하되, 현재 evidence상 backbone 크기만 키우는 접근은 배포 후보를 만들지 못한다. 다음 품질 우선 개선은 백본보다 route/subject target 정렬, internal prior 경로, route head 구조, qualitative failure taxonomy에 집중한다.

corrected-vocab head-only routecal도 현재 route `0.456` 부근에서 정체된다. `1099632` no-margin r3는 epoch2 route balanced `0.455669`, epoch4 `0.453914`이고, interactive margin-on r4도 epoch2 `0.456659`이다. deep-route head-only는 epoch2 `0.444984`와 action consistency 하락으로 중단했다. 이에 따라 `route_hierarchy_weight`와 `route_cardinality_weight` 보조 loss를 추가하고, `mcn-q-b448-backbone-hier-route-r1-ip134`를 시작했다. 이 run은 backbone/subject/candidate/route를 낮은 LR로 함께 풀어 route representation 자체를 다시 맞추되, subject/proposal/action loss를 유지해 기존 no-refine checkpoint의 강점을 보존하는지 확인한다.

2026-04-28 01시대 추가 확인에서 full-model preservation rerun `mcn-q-b448-intprior-auxfine-nomargin-r2-ip134-20260427`은 best epoch2 route balanced `0.489994`로 gate를 넘지 못했다. 같은 checkpoint는 subject IoU `0.523299`, subject valid best balanced `0.570295`, generated align top1 `0.955313`, proposal recall@0.5 `0.957991`, action consistency `0.998973`으로 subject/proposal/action은 강하다. 따라서 남은 핵심 blocker는 route taxonomy internalization으로 더 좁혀졌다.

route-head-only 보정도 충분하지 않았다. `1099672` auxfine route-head-only는 epoch4 기준 route balanced `0.465464`가 best이고, `1099676` hierarchical decode는 epoch1 `0.468449`로 시작했다. head만 다시 학습해도 full r2 route best `0.489994`에 못 미치므로, 단순 head capacity보다 image-level route representation 또는 taxonomy noise/label transfer 문제가 더 크다고 본다.

이에 따라 별도 품질 우선 route expert 실험을 추가했다. 새 스크립트 `src/scripts/train_mobilecropnet_v4_route_expert.py`는 SSTK train JSONL을 이미지 단위로 deduplicate하고 majority subject-mode label을 사용해 image-only route classifier를 학습한다. backbone은 pretrained ConvNeXtV2-B/448, SwinV2-B/384, ConvNeXtV2-L/512를 쓰며 class-balanced sampler, class weight, focal loss, horizontal flip을 적용한다. 이 모델은 외부 teacher subject prior를 쓰지 않고 `image`만 입력받으므로 release contract와 충돌하지 않는다. route expert가 gate를 넘으면 두 가지 후속 경로를 비교한다. 첫째, internal composite runtime에서 MobileCropNet 본체의 subject/proposal/action과 route expert 출력을 함께 사용하는 방식이다. 둘째, route expert prediction/logit을 teacher로 삼아 MobileCropNet route head를 distill하는 방식이다. 둘 다 최종 inference 입력은 `image + target_ar`로 유지한다.

실행 상태는 다음과 같다. `1099679`(`mcn-route-expert-swinv2b384-r1`)는 Running으로 전환됐고, `1099678`(`mcn-route-expert-cnv2b448-r1`) 및 `1099677`(`mcn-route-expert-cnv2l512-r1`)은 생성 직후 Pending이다. Pending이 10분을 넘으면 종료 후 재생성한다. 이 route expert 축도 단독 route balanced `>=0.52`만으로 배포 후보가 되지는 않으며, 이후 subject bbox IoU/valid calibration, proposal/action alignment, checklist/why/risk agreement, target-AR compatibility, qualitative failure taxonomy, latency와 같은 release gate를 같은 runtime 구성으로 다시 평가해야 한다.

초기 결과는 route expert 가설을 지지한다. SwinV2-B route expert는 epoch5에서 route balanced `0.568746`, accuracy `0.589212`를 보였고, ConvNeXtV2-B route expert는 epoch2에서 route balanced `0.578753`, accuracy `0.534232`를 보였다. 다만 ConvNeXtV2-B는 `object_single` recall이 `0.187702`로 낮아 class-balanced score만으로는 배포 후보가 아니다. SwinV2-B는 balanced score가 조금 낮지만 `object_single` recall `0.446602`로 더 안정적이라, 현재 composite 1순위는 SwinV2-B route expert다.

이 route expert를 제품 runtime으로 연결하기 위해 `src/mobilecropnet_v4/route_expert.py`를 추가했다. `audit_mobilecropnet_v4_release_heads.py`, `evaluate_mobilecropnet_v4.py`, `evaluate_mobilecropnet_v4_product_ar_direct.py`, `infer_mobilecropnet_v4.py`에는 `--route_expert_checkpoint` 옵션을 추가했다. 이 옵션은 MobileCropNet 본체의 subject bbox, proposal, policy, action, checklist/why/risk 출력을 유지하고 route logits만 내부 route expert 출력으로 교체한다. 따라서 최종 inference 입력은 여전히 `image + target_ar`이며, external teacher subject prior를 쓰지 않는다.

실제 composite audit도 실행했다. full r2 best checkpoint와 SwinV2-B route expert best checkpoint를 묶은 조합은 route balanced `0.571399`, direct final hit `0.959375~0.962500`, subject IoU `0.538614~0.560793`, target-AR compatibility `1.0`까지 회복했다. 그러나 qualitative taxonomy에서 `route_mismatch`, `top1_not_positive`, `subject_valid_mismatch`, low-subject-IoU bucket이 높게 남아 제품 배포 후보로 freeze하지 않는다. 이 결과는 route expert가 필요하지만 충분조건은 아니며, final crop policy/action과 checklist/why/risk agreement까지 같이 distill해야 함을 보여준다.

이에 따라 runtime route override를 최종 해법으로 고정하지 않고, route expert logit을 학습 중 soft target으로 주입하는 distillation 패치를 추가했다. `train_mobilecropnet_v4.py`와 `model.py`는 `--route_expert_distill_checkpoint`, `--route_expert_distill_weight`, `--route_expert_distill_temperature`, `--route_expert_distill_hard_weight`를 받아 `route_fine_logits`가 image-only expert 분포와 KL/hard CE로 정렬되도록 한다. 이 경로는 inference 시 route expert checkpoint를 필요로 하지 않으므로 최종 runtime 입력 계약 `image + target_ar`를 유지한다. interactive GPU `10.2.4.134`에서 2-step code smoke를 완료했고, ConvNeXtV2-L/512 route expert를 teacher로 쓰는 `1099758` ConvNeXtV2-B/448 public-label route-distill은 epoch1 진행 중이다. SwinV2-B/384 public-label route-distill `1099759`는 epoch 결과 없이 종료하고 logit-cache 적용 replacement `1099763`으로 재생성했다. 또한 SSTK public 학생이 UCTR teacher보다 강하게 보이는 전이 손실을 분리하기 위해 같은 interactive GPU에 UCTR-label + SwinV2 route teacher 조합 `mcn-q-cnv2b448-routedistill-swinteacher-uctr-cache-r1-ip134-20260427`을 cache 적용 run으로 실행 중이다.

운영 효율 측면에서는 route expert distillation loop에 image-path별 logit cache를 추가했다. 현재 실행 중인 run은 시작 시점의 코드로 계속 돌지만, 후속 bounded rerun에서는 같은 이미지가 여러 target-AR row로 반복될 때 route expert backbone을 매번 다시 통과시키지 않는다. 이 패치는 최종 runtime 출력 계약을 바꾸지 않고 학습 시간만 줄이는 변경이며, 로컬/원격 `py_compile`를 통과했다.

2026-04-28 07:40 KST 기준 route-distill 첫 epoch gate는 아직 배포 후보 수준이 아니다. `1099758` public ConvNeXtV2-B/448 route-distill epoch1은 selection `0.679985`, generated align top1 `0.932143`, subject IoU `0.477546`이지만 route balanced `0.478469`, subject valid balanced `0.402296`, generated positive recall `0.014565`라 release gate 미달이다. interactive UCTR-label + SwinTeacher cache run도 epoch1 selection `0.705297`, top-return `0.622321`, generated align top1 `0.961607`까지 올랐지만 route balanced `0.490500`, subject IoU `0.476488`, subject valid balanced `0.421671`로 미달이다. 따라서 이 축은 계속 terminal까지 보되, 현 시점에서는 direct/public/GAIC 확장 대상이 아니라 구조 변경 smoke를 병렬로 투입하는 것이 맞다.

구조 변경 smoke를 재현 가능하게 하기 위해 `run_mobilecropnet_v4_route_gate_smoke.sh`의 quality spatial genrank variant에 `SUBJECT_BOX_SPATIAL_OUTPUT`, `SUBJECT_BOX_SPATIAL_BOX_MODE`, `SUBJECT_BOX_SPATIAL_EXTENT_SCALE`, `SUBJECT_BOX_SPATIAL_MASK_WEIGHT`, `ROUTE_HEAD_DEPTH`, `ROUTE_AUX_HEADS`, `ROUTE_DECODE_MODE`, `ROUTE_HIERARCHY_WEIGHT`, `ROUTE_CARDINALITY_WEIGHT` 환경변수 제어를 추가했다. 이 변경은 기존 variant 기본값을 바꾸지 않고, 후속 run에서 `mask_moment_regress + spatial output + route hierarchy/aux head`를 같은 wrapper로 실행하기 위한 운영 패치다. 로컬/원격 `bash -n` 검사를 통과했고 원격 shared root에 동기화했다.

이 패치 위에서 두 개의 bounded smoke를 추가 생성했다. `1099767`(`mcn-q-cnv2b448-momenthier-uctr-r1-20260427`)은 UCTR labels, ConvNeXtV2-B/448, full-r2 init, SwinV2 route expert distillation, spatial-only `mask_moment_regress`, route hierarchy/aux head를 사용한다. `1099768`(`mcn-q-cnv2b448-momenthier-public-r1-20260427`)은 동일 구조에서 public labels와 ConvNeXtV2-L route expert를 사용한다. 두 run은 public/UCTR inversion을 같은 구조에서 비교하기 위한 신규 실험이며 기존 route-distill run과 중복이 아니다. 생성 직후 상태는 Pending이며, MLP 정책에 따라 10분 이상 Pending이면 종료 후 재생성한다.

terminal 결과는 mixed다. `1099767` UCTR moment-hierarchy는 best epoch2 selection `0.711188`, generated positive recall `0.604219`, subject valid best balanced `0.600833`으로 proposal/action 쪽이 크게 개선됐지만 route balanced `0.448740`, subject IoU `0.483811`이라 gate 미달이다. `1099768` public moment-hierarchy는 epoch3에서 route balanced `0.527143`으로 smoke route gate를 처음 넘겼고 generated positive recall `0.567383`, subject IoU `0.484698`, subject valid best balanced `0.585208`을 기록했다. 그러나 같은 checkpoint를 Product-AR direct/audit으로 승격하자 UCTR direct surface에서 route가 다시 무너졌다. Direct eval은 final hit `0.959688`, proposal target-AR recall@5 `0.972188`, subject IoU `0.555100`, target-AR compatibility `1.0`으로 crop/proposal/subject는 강하지만 route_acc `0.380625`다. release-head audit도 route balanced `0.399154`, top1_hit `0.210000`, route_mismatch `0.488438`, top1_not_positive `0.790000`이라 catastrophic qualitative bucket이 크다. 따라서 `1099768`은 배포 후보가 아니라 “public label에서는 route가 올라가도 UCTR/product route surface로 일반화되지 않는다”는 evidence로 기록한다.

약한 route-distill run은 자원 회수를 위해 정리했다. `1099763` SwinV2 public cache run은 epoch2 route balanced `0.374997`, subject valid balanced `0.245536`으로 악화되어 `Terminated` 처리했다. `1099758` public ConvNeXtV2-B no-cache run도 epoch2 route balanced `0.417440`, generated positive recall `0.014732`라 종료했다. UCTR-label + SwinTeacher cache run은 completed지만 best epoch1 route balanced `0.490500`, subject valid balanced `0.421671`, generated positive recall `0.015025`로 full eval 승격 대상이 아니다.

다음 병렬 축으로 `1099771` EVA02-B/448 moment-hierarchy UCTR run과 `1099772` ConvNeXtV2-B/448 UCTR route-boost run을 시작했다. `1099771`은 강한 pretrained EVA02 backbone이 같은 moment/hierarchy 구조에서 route/subject를 개선하는지 확인한다. `1099772`는 `ROUTE_WEIGHT=14`, route sampler max weight `8`, route hierarchy/cardinality weight `0.55/0.30`, route expert distill `1.10`, hard distill `0.35`로 route 병목만 강하게 밀어보는 ablation이다. 두 run 모두 최종 inference 입력 계약은 `image + target_ar` 그대로다.

public/GAIC 품질 보강을 위해 `public_utility_tune` 축도 추가했다. `publicutil_headonly` r1은 direct hit `0.940938`, subject IoU `0.560793`, GAIC top1 MOS `3.791633`, SRCC `0.405634`, Accw4@10 `0.340100`으로 full-r2보다 GAIC는 좋아졌지만 direct와 qualitative taxonomy가 충분하지 않다. `publicutil_ranker`와 `rankerdetail` r1은 direct hit가 각각 `0.740625`, `0.794375`로 낮아 제외했다. SwinV2-B/384 public-utility r2 headonly는 direct final hit `0.962813`, final best IoU `0.877421`, target-AR compatibility `1.0`이지만 subject IoU `0.504648`, subject valid acc `0.642188`, audit top1 hit `0.251250`으로 단독 배포 후보가 아니다. r2 ranker는 best selection `0.514329`, route balanced `0.289098`, subject valid balanced `0.320202`라 full benchmark 확장 대상이 아니다.

public shard는 `publicutil_headonly`와 `swinv2joint` 두 묶음 모두 완료됐다. `swinv2joint`는 FCDB `0.729293`, CPC weighted `0.802862`, GNMC `0.703646`, overall IoU `0.701871`이고, `publicutil_headonly`는 FCDB `0.714605`, CPC weighted `0.771781`, GNMC `0.720837`, overall IoU `0.713213`이다. 두 결과 모두 crop/public 수치만으로는 release gate를 닫지 못하며, route/subject/head qualitative blocker 때문에 배포 후보에서 제외한다.

품질 우선 백본 확장 지시에 맞춰 subject-only 상한 실험도 확인했다. `1099761` DINOv2-B/518 subject-only는 UCTR support-latent target에서 best subject IoU `0.373998`, valid best balanced `0.514286`, proposal recall `0.034966`으로 종료되어 배포 후보와 후속 full body init 후보에서 제외한다. 이 결과는 단순히 더 큰 pretrained backbone을 쓰는 것만으로 subject bbox가 개선되지 않으며, 다음 subject 개선은 target/valid calibration/head 구조를 같이 바꾸는 경우에만 의미가 있음을 보여준다.

2026-04-28 08시대 후속 판단에서 `1099771`과 `1099772`도 배포 후보에서 제외했다. `1099771` EVA02-B/448 moment-hierarchy UCTR는 epoch2 기준 route balanced `0.427775`, route fine balanced `0.464808`, subject IoU `0.347494`, subject valid balanced `0.355556`으로 subject head가 붕괴해 조기 종료했다. `1099772` ConvNeXtV2-B/448 UCTR route-boost는 epoch1 route balanced `0.444161`, subject IoU `0.391547`에서 시작했고 epoch2 route balanced `0.418328`, subject IoU `0.404719`로 악화되어 종료했다. 결론적으로 더 강한 backbone이나 route loss weight만으로는 `route + subject + generated action` 동시 gate를 넘기지 못한다.

사용자가 지적한 `UCTR teacher > public_cropper_ensemble_best`와 학생 모델의 `SSTK public > SSTK UCTR` 역전은 route label 차이가 아니었다. remote shared storage에서 public-label과 UCTR-label의 동일 이미지 majority route를 비교한 결과 train `7,800` image, val `964` image 모두 route agreement가 `1.0`이며 route class distribution도 완전히 같다. 추가 val row 비교에서는 candidate pool Jaccard 평균이 `1.0`이지만 positive id mismatch가 `16.62%`, top candidate id mismatch가 `60.32%`였다. positive/top box IoU 평균은 `0.856977`이라 UCTR는 전혀 다른 후보를 만드는 것이 아니라 같은 후보 집합 안에서 비슷한 crop 간 순위와 정책 타깃을 더 세밀하게 바꾼다. 따라서 학생 모델의 public 우세는 route label 문제가 아니라 ranking/action target transfer와 top candidate alignment 문제로 본다. 산출물은 `artifacts/mobilecropnet_v4/quality_first_20260427/label_source_analysis/public_vs_uctr_route_agreement_20260427.json`, `public_vs_uctr_candidate_target_shift_val_20260427.json`에 저장했다.

이 분석에 따라 새 병렬 축을 `1099768` public moment-hierarchy checkpoint의 강한 crop/proposal/subject 출력을 보존하면서 UCTR route/head만 재정렬하는 방향으로 바꿨다. `1099775` MLP route-only run은 Pending 후 시작 직후 중단하고 중복 없이 interactive GPU `10.2.4.134`에서 `mcn-q-cnv2b448-public2uctr-routeonly-r1-ip134-20260427`로 실행했다. 이 run은 `1099768` best checkpoint를 init으로 쓰고 `route_head`, `route_kind_head`, `route_cardinality_head`만 학습한다. `1099776`은 같은 init에서 route head와 subject valid head만 학습하는 route+valid calibration run이다. 둘 다 외부 teacher subject prior 없이 `image + target_ar` 입력 계약을 유지하며, SwinV2 route expert logit은 학습 중 distillation target으로만 사용한다. 동시에 route expert 상한을 더 올리기 위해 EVA02-L/448 `mim_m38m_ft_in22k_in1k` pretrained route expert `1099774`를 실행 중이다.

### 20.3 2026-04-28 품질 우선 모델군 재정렬

이 항목은 기존 마스터 리포트 초안에 있던 온디바이스 최적화 계획이 아니라, 2026-04-28 사용자 지시로 추가된 품질 우선 후속 실험 축이다. 이제 속도는 1차 gate가 아니며, MobileNet 또는 ConvNeXtV2-Base보다 큰 pretrained backbone과 더 큰 입력을 사용해 정량 품질과 정성 실패를 먼저 줄이는 모델군을 병렬 검증한다.

우선 `train_mobilecropnet_v4.py`에 multi-expert route distillation을 추가했다. 단일 `--route_expert_distill_checkpoint`뿐 아니라 `--route_expert_distill_checkpoints`와 `--route_expert_distill_weights`를 받을 수 있으며, SwinV2-B/384, ConvNeXtV2-L/512, EVA02-L/448 route expert logits를 ensemble soft target으로 사용한다. distillation은 학습 중에만 쓰고, final inference는 route expert checkpoint 없이 MobileCropNet 본체의 내부 route head를 사용해야 제품 후보로 인정한다.

운영 중 발견한 r1 실패도 수정했다. `1099818` contextual multi-expert distill r1은 model quality failure가 아니라 config 저장 시 `Path` 리스트가 JSON 직렬화되지 않는 오류로 실패했다. recursive JSON-safe 변환을 추가해 `route_expert_distill_checkpoints`와 expert config를 안정적으로 저장하도록 패치했고, 같은 실험을 `1099823` r2로 재생성했다.

동시에 route expert override 평가와 내부 route 평가를 분리했다. `run_mobilecropnet_v4_composite_routeaware_gate.sh`에 `ROUTE_EXPERT_MODE=none`을 추가해, 같은 checkpoint를 외부 route expert 상한선과 내부 route head 제품 계약으로 따로 감사할 수 있다. 제품 후보는 반드시 `ROUTE_EXPERT_MODE=none`, `runtime_subject_prior_mode=none`에서 subject bbox, proposal, route, policy, final crop action, target-AR repair, checklist/why/risk를 내부적으로 출력해야 한다.

현재 병렬 실행 중이거나 방금 terminal 판정을 끝낸 품질 우선 실험은 다음 축이다.

| run id | run | 목적 | backbone/input | 현재 판정 기준 |
| ---: | --- | --- | --- | --- |
| `1099828` | `mcn-q-cnv2b448-imgroute-ensdistill-r2-20260428` | `1099821` 정체 종료 후 재생성한 image-only ensemble distill | ConvNeXtV2-B / 448 | internal route balanced, top1_not_positive 감소 |
| `1099823` | `mcn-q-cnv2b448-actionroute-ensdistill-hardrank-r2-20260428` | contextual route/action distill r1 실패 수정 후 재실행 | ConvNeXtV2-B / 448 | route/action alignment, subject valid calibration |
| `1099830` | `mcn-q-cnv2b448-ranker-top1repair-ensdistill-r1-20260428` | top1_not_positive와 policy/action mismatch를 직접 줄이는 ranker repair | ConvNeXtV2-B / 448 | top-return, top1, qualitative catastrophe 감소 |
| `1099822` | `mcn-q-eva02l448-imgroute-ensdistill-r1-20260428` | 온디바이스 제약 완화 품질 우선 full model | EVA02-L `mim_m38m_ft_in22k_in1k` / 448 | route/subject/proposal/action joint gate |
| `1099832` | `mcn-q-swinv2b384-ensdistill-r1-20260428` | Swin route expert 상한을 full MobileCropNet 본체로 전이 | SwinV2-B `ms_in22k_ft_in1k` / 384 | route/subject/proposal/action joint gate |
| `1099855` | `mcn-q-cnv2l512-full-ensdistill-toprepair-r3-20260428` | `1099840` Pending 오류와 `1099842` batch-default 오류를 정정한 Large/512 full-head 품질 우선 모델 | ConvNeXtV2-L `fcmae_ft_in22k_in1k_384` / 512 | route/subject/top-return joint gate |
| `1099861` | `mcn-q-cnv2h512-full-toprepair-r3-20260428` | 사용자 추가 지시에 따라 MobileNet/CNV2-B 제약을 더 완화한 Huge/512 full-head 모델, r1 batch-default 오류와 r2 HF 429 실패 정정 | ConvNeXtV2-H `fcmae_ft_in22k_in1k_512` / 512 | 품질 우선 상한, subject-valid/top-return 보강 |
| `1099862` | `mcn-q-cnv2b448-embedinit-validtop-r1-20260428` | embedded single-checkpoint upper-bound를 init으로 사용해 subject-valid/top-return을 보정 | ConvNeXtV2-B / 448 | single-checkpoint packaging 성공을 내부 head 품질로 전이 |
| `1099820` | `mcn-q-dinov2l518-imgroute-ensdistill-r1-20260428` | 더 큰 ViT 기반 품질 우선 full model | DINOv2-L `lvd142m` / 518 | 완료, route/subject 붕괴로 제외 |

`1099795` CNV2-L single-expert distill은 epoch3 기준 route balanced `0.488443`, subject valid balanced `0.462629`에 머물러 배포 후보에서 제외했다. `1099793` ConvNeXtV2-H/512 route expert는 epoch1 route balanced `0.585252` 뒤 epoch2 `0.575564`로 하락해 기존 CNV2-L/EVA route expert를 넘지 못했고 GPU를 회수했다. `1099820` DINOv2-L/518 full model도 best epoch3 route balanced `0.250000`, subject IoU `0.197411`, valid balanced `0.208333`으로 끝나 제외했다. `1099822` EVA02-L/448 full model은 terminal best epoch4에서 route balanced `0.528571`을 넘겼지만 subject IoU `0.396547`, subject valid balanced `0.342857`, top1 hit `0.114286`이라 direct/GAIC 승격 대상에서 제외한다. `1099828`은 epoch2 route balanced `0.269583`, `1099832`는 epoch1 route balanced `0.182785`라 조기 종료했다. 따라서 다음 release candidate 가능성은 `1099823`, `1099830`, `1099855`, `1099861`, `1099862` 중 internal no-prior gate를 실제로 통과하는 checkpoint가 나오는지에 달려 있다.

Swin route override 상한선도 같은 기준으로 제외했다. interactive GPU `10.2.4.134`의 final-best + Swin route override 직접 평가는 `final_positive_hit_iou_0_5=0.963750`, `final_iou_to_best_positive=0.837730`, `proposal_positive_recall_at_5=0.972188`, `target_ar_compatible=1.0`로 crop/action 수치는 강하다. 그러나 head audit은 route balanced `0.571399`, subject valid IoU `0.555100`, subject valid balanced `0.706891`에도 불구하고 `top1_not_positive=0.693125`, `route_mismatch=0.434688`, `subject_valid_mismatch=0.284375`를 보였다. 이 평가는 external route expert override를 쓰므로 제품 계약을 만족하지 않고, 정성 catastrophic bucket도 높아 deploy candidate가 아니라 internal distillation의 upper-bound evidence로만 유지한다.

제품 계약에 더 가깝게 보기 위해 Swin route expert를 MobileCropNet checkpoint 내부에 embedding하는 경로도 추가했다. 새 checkpoint `artifacts/mobilecropnet_v4/quality_first_20260427/embedded_route_expert_checkpoints/mcn-q-cnv2b448-public2uctr-actionroute-r1-final-swinroute-embedded-20260428.pt`는 inference 시 별도 route expert CLI 인자를 요구하지 않는다. 단, 이 경로는 "파일 하나" 조건을 맞추는 것일 뿐이며 route/policy/action/checklist가 여전히 분리된 상한선에 머물 수 있다. 따라서 `ROUTE_EXPERT_MODE=none` single-checkpoint audit/direct eval에서 qualitative failure taxonomy까지 함께 통과해야만 배포 후보로 승격한다.

2026-04-28 10:28 KST 기준 embedded single-checkpoint audit/direct eval도 완료했으며 배포 후보에서 제외했다. 직접 평가는 final positive hit `0.963750`, final IoU-to-best-positive `0.837730`, proposal recall@5 `0.972188`, target-AR compatibility `1.0`이지만, head audit failure taxonomy가 `top1_not_positive=0.693125`, `route_mismatch=0.434688`, `low_subject_iou=0.285938`, `subject_valid_mismatch=0.284375`라 제품 수준 qualitative gate를 통과하지 못한다. 이 결과는 route expert를 checkpoint 내부에 넣는 기술 경로는 가능하지만, MobileCropNet 본체의 route/policy/action/checklist internalization은 아직 부족하다는 증거로 유지한다.

같은 시각 새 코드 variant `quality_cnv2h512_full_toprepair_lr5e6`도 추가했다. 이 variant는 `quality_cnv2h_512` profile, ConvNeXtV2-Huge pretrained, 입력 `512`, candidate/proposal `96`, token dim `768`을 쓰며, A100 80GB 메모리 리스크를 줄이기 위해 SwinV2-B route expert 하나만 distillation source로 사용한다. 목적은 속도 제약을 크게 완화했을 때 subject-valid calibration, top-return, route hierarchy가 동시에 좋아지는지 확인하는 것이며, 최종 inference 계약은 여전히 `image + target_ar`이다.

초기 실행 점검에서 Large/Huge variant가 전역 기본 `BATCH_SIZE=24`, `MAX_TRAIN_ROWS=12000`, `ROUTE_SAMPLER_MAX_WEIGHT=4.0`에 가려져 의도한 품질 우선 설정을 쓰지 않는 문제가 확인됐다. `1099842`와 `1099846`은 이 때문에 중단했고, runner를 `QUALITY_*` 및 `CNV2H_*` variant-local 기본값으로 패치했다. corrected workload는 `1099855`와 `1099854`이며, 이후 판정은 이 두 run의 durable metrics를 source of truth로 삼는다.

`1099854`는 Hugging Face 429 때문에 pretrained weight 로딩에 실패했다. 로컬 PC에서 `timm/convnextv2_huge.fcmae_ft_in22k_in1k_512` snapshot을 받아 `weights/hf/timm/convnextv2_huge.fcmae_ft_in22k_in1k_512/model.safetensors`로 원격 shared storage에 올렸고, 같은 설정을 `1099861` r3로 재생성했다. 이 조치로 Huge/512 실험도 사용자 지시의 "가장 좋은 pretrained weight 사용" 조건을 만족한다.

`1099823` contextual hard-rank r2는 terminal best epoch4에서 subject/proposal/action은 유지했지만 내부 route/valid gate에는 미달했다. 수치는 selection `0.735450`, route balanced `0.491905`, subject IoU `0.528067`, subject valid balanced `0.462629`, generated top1 `0.959444`, proposal recall `0.949829`, top-return `0.629925`이다. 따라서 이 checkpoint 자체는 배포 후보가 아니지만, subject/proposal/action head가 이전 embedded upper-bound보다 나을 가능성이 있어 Swin route expert를 checkpoint 내부에 embedding한 single-file no-prior gate eval `1099866`을 추가했다.

### 20.4 2026-04-28 14시대 품질 우선 no-prior 기준선과 public repair

`1099866` 계열의 최신 단일 checkpoint 후보는 `mcn-q-cnv2b448-actionroute-hardrank-r2-swinroute-embedded-20260428`로 고정했다. checkpoint는 `artifacts/mobilecropnet_v4/quality_first_20260427/embedded_route_expert_checkpoints/mcn-q-cnv2b448-actionroute-hardrank-r2-swinroute-embedded-20260428.pt`이며, 최종 inference 입력은 `image + target_ar`만 사용한다. SwinV2-B route expert는 외부 CLI 인자가 아니라 checkpoint 내부 payload로 들어가므로 runtime no-prior 계약과 packaging 측면에서는 기존 override보다 제품 계약에 가깝다.

이 후보의 최종 bundle은 `artifacts/mobilecropnet_v4/quality_first_20260427/final_eval_bundle/mcn-q-cnv2b448-actionroute-hardrank-r2-swinroute-embedded-20260428/`에 있다. 핵심 수치는 FCDB IoU `0.716264`, CPC weighted `0.832132`, GNMC IoU `0.702134`, GAIC top1 MOS `3.849255`, GAIC SRCC `0.391839`, GAIC Accw4@10 `0.359335`, GAIC primary `0.575969`, equal4 raw `0.706625`, route balanced `0.541762`, subject IoU `0.596369`, subject valid negative `0.663242`, final hit `0.969375`, risk hit `0.030937`, A100 mean latency `41.391 ms`다. route/subject/direct 일부 gate는 닫았지만 public/equal-4와 GAIC SRCC가 기존 상위권보다 낮고 raw top1/ranker failure가 높아 배포 후보로 확정하지 않는다.

이 결과로 후속 실험 방향을 단순 backbone 확대가 아니라 `public crop utility 복구 + UCTR/Swin no-prior internal route/subject 유지 + ranker/action alignment repair`로 좁혔다. public repair r2 `1099963`/`1099965`는 wrapper가 init checkpoint와 다른 config로 모델을 만들면서 subject/route 일부 head를 shape mismatch로 건너뛰는 문제가 확인되어 2026-04-28 13:48 KST에 중단했다. 이후 wrapper는 `subject_box_spatial_multiscale=1`, `route_decode_mode=hierarchical`, `route_head_depth=3`, `route_head_hidden_mult=0.75`, `route_use_subject_spatial_token=0`을 기본값으로 맞추도록 패치했다.

corrected r3는 두 GPU 축으로 진행했다. `mcn-q-pubrepair-ranker-r3-20260428`은 MLP workload `1099978`에서 candidate encoder, relation MLP, set ranker, utility/positive/risk head를 함께 보정했고, train summary 기준 best epoch2 selection `0.538890`, val top1 hit `0.372813`, top-return hit `0.478446`, route balanced `0.469035`, subject IoU `0.485918`, subject valid best balanced `0.608607`이다. 이 수치는 Swin embedded 기준선의 내부 head gate를 넘지 못하므로 head audit/direct가 완료되더라도 full public/GAIC 승격은 gate 산출물로 다시 제한한다.

`mcn-q-pubrepair-headonly-r3-ip134-20260428`은 실행 후 checkpoint config를 재검사한 결과 `route_head_hidden_mult=0.5`, `route_use_subject_spatial_token=true`가 들어가 init checkpoint의 `0.75/false`와 달라진 invalid run으로 분리한다. 해당 run의 head audit은 route balanced `0.571399`였지만 subject IoU가 `0.386745`로 급락했으므로, direct eval은 GPU 낭비를 막기 위해 중단하고 status를 `excluded`로 갱신했다. 같은 목적의 corrected head-only run은 interactive GPU `10.2.4.134`에서 `mcn-q-pubrepair-headonly-r4-ip134-20260428`로 재시작했다. r4는 `route_head_hidden_mult=0.75`, `route_use_subject_spatial_token=0`, `subject_box_spatial_output=spatial`을 명시해 init checkpoint와 구성 일치를 보장한다.

사용자가 새로 지시한 품질 우선/배포 제약 완화 축은 단순 대형 backbone 반복이 아니라 `public crop utility`와 `UCTR geometry/action`의 target transfer 문제를 직접 겨냥하도록 구체화했다. `artifacts/mobilecropnet_v4/quality_first_20260428/mixed_public_uctr_labels/`에 public/UCTR interleaved label set을 생성했으며 train `78,004` rows, val `9,894` rows다. 이를 사용한 `mcn-q-mixrepair-rankerpolicy-r1-20260428` MLP workload `1100023`은 public-only ranker-policy와 같은 시간 예산으로 비교하기 위해 첫 train `39,002` rows만 사용한다. 이 run은 ranker/policy/action head를 함께 보정하되 final inference 계약은 여전히 `image + target_ar`와 single embedded checkpoint다.

2026-04-28 14:46 KST 기준 public-only ranker-policy `1099988`은 best epoch2 selection `0.537218`, route balanced `0.483205`, subject IoU `0.485918`, top1 hit `0.367813`, top-return hit `0.474446`로 종료되어 배포 후보에서 제외했다. embedding/gate까지 진행해도 Swin embedded 기준선을 넘을 가능성이 낮아 MLP run을 `Terminated` 처리하고 status를 `excluded`로 남겼다. 반대로 mixed label set은 아직 검증이 끝나지 않았으므로 `1100023` mixed ranker-policy와 `1100027` mixed head-only를 Running 상태로 유지한다. 또한 policy head 업데이트가 역효과인지 분리하기 위해 `1100031` mixed ranker-only ablation도 추가 생성했다.

2026-04-28 15:01 KST 기준 corrected public-only head-only r4도 배포 후보에서 제외했다. best epoch2 selection은 `0.505617`이고 val route balanced `0.516700`, subject IoU `0.485918`, top1 hit `0.317500`, top-return hit `0.423685`다. route는 public-only ranker보다 높지만 top-return과 subject gate가 Swin embedded 기준선에 미치지 못하므로, 자동 embedded gate 전에 interactive GPU 프로세스를 종료하고 status를 `excluded`로 고정했다.

2026-04-28 15:27 KST 기준 public/UCTR transfer 축을 다시 재정렬했다. 단순 interleaved mixed label은 같은 이미지/target_AR row를 public/UCTR 두 번 넣기 때문에 candidate pool은 같아도 ranking/action target이 서로 충돌할 수 있다. 실제로 `1100023` mixed ranker-policy는 best top-return `0.546563`, route balanced `0.431239`, subject IoU `0.412261`로 제외했고, `1100027` mixed head-only도 best top-return `0.529479`, route balanced `0.499778`, subject IoU `0.412261`로 제외했다. 이 결과는 UCTR teacher가 equal-4에서 강해도 학생에게 그대로 interleaved distill하면 public quality와 UCTR geometry가 동시에 살아나지 않는다는 증거다.

이를 보완하기 위해 `src/scripts/build_mobilecropnet_v4_blended_public_uctr_labels.py`를 추가했다. 이 빌더는 public/UCTR를 두 row로 중복하지 않고, 같은 row 안에서 candidate_id 기준 score/utility/rank/positive target을 blend한다. 전체 라벨 생성은 streaming 방식으로 처리하며, 산출물은 `artifacts/mobilecropnet_v4/quality_first_20260428/blended_public_uctr_labels/`에 있다. full split 결과는 train `39,002` rows, val `4,947` rows, key mismatch `0`이고, public weight `0.4`, UCTR weight `0.6`, matching top-k `2`, soft positive margin `0.08`을 사용했다. 평균 top source disagreement는 train `0.340285`, val `0.344903`으로 커서 blend 실험 자체는 타당한 후속 축이다.

현재 blended label set으로 세 개의 병렬 실험을 실행 중이다. `mcn-q-blendrepair-rankerpolicy-r1-ip134-20260428`은 interactive GPU `10.2.4.134`에서 ranker/policy/action을 함께 보정한다. `1100057`은 head-only, `1100056`은 ranker-only MLP workload다. 이 세 축은 모두 최종 입력 계약 `image + target_ar`를 유지하며, 성공 시 SwinV2 route expert를 checkpoint 내부에 embedding한 뒤 `ROUTE_EXPERT_MODE=none`으로 head audit, Product-AR direct eval, qualitative pack을 먼저 통과해야 한다. train summary가 기준선을 넘지 못하면 full public/GAIC는 생성하지 않는다.

### 20.5 2026-04-28 16시대 blend 결과와 warm-start 품질 우선 축

2026-04-28 16:25 KST 기준 blended repair 세부 실험은 모두 배포 후보에서 제외했다. UCTR60/public40 `mcn-q-blendrepair-rankerpolicy-r1-ip134-20260428`은 train best epoch2에서 selection `0.598510`, top1 hit `0.575938`, top-return `0.606247`까지 올렸지만 route balanced `0.481330`, subject IoU `0.485918`로 train gate가 닫히지 않았다. embedded single-checkpoint direct eval도 final hit `0.927813`, final IoU-to-best-positive `0.786939`, proposal recall@5 `0.971875`, subject IoU `0.523292`, subject valid acc `0.715625`, target-AR compatibility `1.0`로 기존 no-prior 기준선보다 crop/action이 낮다. qualitative pack은 `subject_valid_mismatch=0.284375`, `subject_iou_low=0.270313`, `final_not_positive=0.072188`라 catastrophic bucket이 남는다.

동일 label의 head-only `1100057`과 ranker-only `1100056`도 제외했다. 두 run 모두 embedded head audit에서 route balanced `0.571399` 상한은 유지했지만, train top-return/subject gate와 audit top1_not_positive를 개선하지 못했다. head-only top1_not_positive는 `0.690313`, ranker-only는 `0.674375`이고 subject_valid_mismatch는 둘 다 `0.284375`다. public60/UCTR40 `1100071`은 best epoch2 selection `0.565838`, top1 hit `0.570938`, top-return `0.534756`, route balanced `0.485475`, subject IoU `0.485918`로 UCTR60 blend보다 ranking transfer가 약했다. detail score까지 학습 범위를 넓힌 `1100073`도 selection `0.597940`, top-return `0.605452`였지만 route balanced `0.485557`, subject IoU `0.485918`, generated align top1 `0.0`으로 train gate 미달이라 direct/audit 전 MLP를 종료했다.

따라서 현재 결론은 “public/UCTR를 row 중복 없이 blend하면 proxy top-return은 오르지만, 기존 강한 subject/proposal/action checkpoint의 제품 direct 품질을 보존하지 못한다”이다. 단순 head/ranker/policy repair는 target conflict를 충분히 풀지 못했고, final bundle/GAIC 중복 workload `1100096`/`1100097` 및 r2 `1100103`/`1100104`는 gate 순서 위반이므로 중단했다. 새 배포 후보가 생기기 전까지 38-row 통합 리더보드는 유지하며, 남은 3개 공백 row는 checkpoint source가 없는 invalid row라 추가 GPU 평가 대상이 아니다.

후속 축은 두 갈래로 재정렬했다. 첫째, scratch 품질 우선 full-joint는 계속 진행한다. `1100091` ConvNeXtV2-L/512, `1100092` EVA02-L/448, `1100079` SwinV2-B/384는 blended label, multi-expert route distillation, support-latent subject target, no-prior runtime gate를 사용한다. 이들은 아직 첫 validation 전이므로 terminal summary를 기다리되, route/subject/proposal/action gate를 넘지 못하면 final bundle 없이 제외한다.

둘째, scratch가 너무 느리고 불안정하다는 점을 반영해 warm-start blend joint를 추가했다. `run_mobilecropnet_v4_quality_relaxed_joint.sh`는 optional `INIT_CHECKPOINT`와 train-stage early gate를 받도록 패치했다. gate threshold는 route balanced `0.52`, subject IoU `0.50`, subject valid balanced `0.55`, proposal recall `0.93`, action top1 `0.96`, top-return `0.58`이며, 이 기준을 못 넘으면 embedded packaging, Product-AR direct eval, qualitative pack을 생략한다. 첫 r1 warm-start는 wrapper 기본 `ROUTE_HEAD_HIDDEN_MULT=0.70`이 init checkpoint의 `0.75` route head와 맞지 않아 route head 21개 key가 재초기화되는 invalid run으로 분리했다. corrected UCTR60/public40 r2는 interactive GPU `10.2.4.134`의 `mcn-q-cnv2b448-blendjoint-warminit-r2-ip134-20260428`이며 shape mismatch 없이 train progress에 진입했다. public60/UCTR40 MLP r2는 duplicate run이 같은 output dir을 잠깐 공유해 log/status 오염 가능성이 생겼으므로 종료했고, clean output의 r3 `1100191` `mcn-q-cnv2b448-blendp60-warminit-r3-20260428`을 source-of-truth로 재생성했다. 두 유효 run 모두 `ROUTE_HEAD_HIDDEN_MULT=0.75`를 명시하며 final inference 계약은 `image + target_ar`이다.

### 20.6 2026-04-28 16:45 KST 리더보드/실행 상태 보정

통합 리더보드의 `hybrid_384` latency 공백은 로컬 RTX 3050 직접 측정으로 닫았다. `mcn-sstk-t1-baseasync-hybrid-384-20260423-023818`의 mean latency는 candidate `48` 기준 `10.256ms`이고, 38-row 품질 우선 snapshot과 최종 배포 상태 문서 모두 이 값을 사용한다. 현재 남은 수치 공백 3개는 평가 누락이 아니라 checkpoint가 local/remote shared storage에 없는 invalid row다.

public60/UCTR40 warm-start는 동일 command가 두 번 생성됐고, `1100183`은 startup 직후 중복으로 종료했다. 같은 output dir을 공유한 `1100158`도 artifact contamination risk 때문에 종료했으며, r3 `1100191`만 유지한다. interactive GPU `10.2.4.134`의 UCTR60/public40 warm-start는 `ROUTE_HEAD_HIDDEN_MULT=0.75`로 init checkpoint key `530/530`을 shape mismatch 없이 로드하고 완료됐지만, best epoch1 기준 route `0.512705 < 0.52`, top-return `0.544754 < 0.58`로 train gate에서 제외됐다. subject IoU `0.509948`, subject valid balanced `0.590119`, proposal recall `0.968238`, action top1 `0.998214`는 통과했으므로 후속 축은 route/top-return만 보정하는 head-calibration으로 전환한다. scratch relaxed full-joint 중 `1100091` ConvNeXtV2-L/512는 epoch1 route `0.475966`, subject IoU `0.436047`, top-return `0.433194`, action top1 `0.859861`로 gate와 멀고 이후 컨테이너가 종료되어 제외한다. `1100079`/`1100092`와 r3 `1100191`은 terminal summary 전이라 full public/equal-4/GAIC 평가는 아직 보류한다.

후속 rerun을 빠르게 전환하기 위해 같은 wrapper에 `TRAINABLE_PREFIXES`도 추가했다. 현재 full-joint warm-start가 subject/proposal/action을 보존하지 못하면, 다음 bounded smoke는 같은 init checkpoint에서 backbone/subject/proposal을 고정하고 `route_head`, `route_kind_head`, `route_cardinality_head`, `set_ranker`, `utility_head`, `positive_head`, `risk_head`, `policy_head`, `decision_head`, `delta_head`, `policy_score_predictor`, checklist/why heads만 업데이트하는 head-calibration 형태로 전환한다.

이 전환은 바로 실행에 반영했다. 회수한 GPU에는 `1100245` `mcn-q-cnv2b448-blendjoint-headcal-r1-20260428`을 생성했다. duplicate로 생긴 `1100248`은 같은 run name/output 목적이라 2026-04-28 17:29 KST에 종료했고, source of truth는 Running 상태의 `1100245`다. 이 run은 `1099823` init에서 backbone/subject box/proposal box를 보존하고 route/ranker/policy/checklist/subject-valid 계열 head만 학습하므로, full-joint warm-start가 품질을 흔드는지 분리하는 실험이다.

동시에 Running 목록에서 `1100200` ConvNeXtV2-Huge/512 blend joint, `1100201` DINOv2-L/518 blend joint, `1100214` public60 head-calibration을 확인했다. 이들은 같은 output을 공유하지 않는 별도 품질 우선 축이므로 유지한다. 다만 이전 DINO/EVA/large backbone 실패에서 이미 "backbone 확대만으로는 부족하다"는 증거가 있으므로, first validation에서 route/subject/proposal/action/top-return train gate와 멀면 바로 종료하고 warm-start/head-calibration 축으로 GPU를 돌린다.

품질 우선 모델군은 기존 MobileNet/속도 제약보다 crop quality와 internal head 품질을 우선한다. 이에 따라 pretrained DINOv2-L/518 full-joint `1100201`과 pretrained ConvNeXtV2-Huge/512 full-joint `1100200`을 추가 생성했다. 두 run은 SwinV2-B/ConvNeXtV2-L/EVA02-L 및 ConvNeXtV2-B warm-start와 다른 backbone/input-size 축이며, 동일 release train gate를 통과하지 못하면 full public/equal-4/GAIC 평가는 수행하지 않는다.

또한 같은 init checkpoint를 쓰는 public60/UCTR40 head-calibration smoke `1100214`를 추가했다. 이 run은 `TRAINABLE_PREFIXES`로 route/ranker/policy/checklist/why 계열만 업데이트해 full-joint가 subject/proposal/action head를 망가뜨리는지와 head-only 보정으로 release gate를 닫을 수 있는지를 분리한다.

2026-04-28 17:36 KST 기준으로 interactive GPU `10.2.4.134`는 UCTR60 warm-start r2 종료 후 비어 있었으므로, 추가 bounded smoke `mcn-q-cnv2b448-blendjoint-headcal-strict-r1-ip134-20260428`을 실행했다. 이 run은 `1100245`보다 더 보수적으로 `subject_valid_head`/`subject_proposal_head`를 고정하고 route/ranker/policy/checklist/why 계열만 학습한다. 설정은 route weight `14.0`, route expert distill `5.0`, hard distill `0.7`, top-return `0.30`, listwise `0.30`, max train rows `10,000`, max val rows `2,400`이다. 목적은 이미 gate를 넘긴 subject/proposal/action을 흔들지 않고 UCTR60 warm-start의 남은 blocker인 route/top-return만 닫을 수 있는지 확인하는 것이다.

EVA02-L/448 scratch full-joint `1100092`는 terminal summary가 생성됐지만 배포 후보가 아니다. best epoch1에서 route `0.620000`, subject IoU `0.512189`는 기준을 넘겼지만 subject-valid balanced `0.421250`, proposal recall `0.915524`, action top1 `0.825000`, top-return `0.326667`이 gate를 크게 밑돈다. 이 run에는 summary 기반 `train_gate_decision.json`을 보강했고, final bundle/GAIC/public 확장은 만들지 않는다. freed GPU를 활용하기 위해 semantic pretrained backbone 축 `1100269` CLIP-ConvNeXt-B/384와 `1100268` SigLIP-B/384 full-joint smoke를 생성했다. 둘 다 기존 DINOv2-L/ConvNeXtV2-Huge/Swin/EVA 축과 중복되지 않으며, 2026-04-28 17:47 KST 기준 Running으로 전환됐다.

public60/UCTR40 계열도 배포 후보로 승격하지 않는다. clean warm-start r3 `1100191`은 best epoch1 route `0.512187`, top-return `0.487698`로 train gate 미달이다. 같은 label의 headcal `1100214`는 subject-valid balanced `0.636107`까지 올렸지만 route `0.510226`, subject IoU `0.493219`, top-return `0.481865`로 실패했다. SwinV2-B/384 relaxed full-joint `1100079`는 route `0.484853`, subject IoU `0.492803`, top-return `0.515106`으로 제외했고, 구버전 command가 embedded/direct gate로 넘어가려는 것을 2026-04-28 17:50 KST에 종료했다. DINOv2-L/518 `1100201`은 summary 없이 MLP가 종료됐지만 train_status 기준 route `0.436000`, subject IoU `0.385304`, valid balanced `0.507000`, proposal `0.913910`, top-return `0.524500`이라 제외한다.

ConvNeXtV2-Huge/512 `1100200`도 품질 우선 대형 backbone 축으로는 실패했다. train_status 기준 route `0.441500`, subject IoU `0.394403`, subject-valid balanced `0.496750`, proposal recall `0.922069`, action top1 `0.832500`, top-return `0.376500`이라 release train gate와 멀어 2026-04-28 17:54 KST에 종료했다. 이 결과는 "속도 제약 완화 + 대형 pretrained backbone"만으로는 teacher의 route/subject/policy 구조가 internalize되지 않으며, loss/label/head-scope alignment를 같이 해결해야 한다는 결론을 강화한다.

이에 따라 UCTR60 warm-start의 가까운 실패를 겨냥해 두 개의 rank-calibration 축을 추가했다. `1100282` `mcn-q-cnv2b448-blendjoint-rankcal-r1-20260428`은 기존 UCTR60/public40 blended label과 `1099823` init checkpoint를 쓰되, subject/proposal/action head를 보존하고 candidate encoder, relation MLP, set ranker, utility/positive/risk, route, policy, checklist/why head만 학습한다. loss는 route `16.0`, route expert distill `5.5`, hard distill `0.75`, top-return `0.45`, listwise `0.42`로 올렸다. 목적은 subject/proposal/action을 흔들지 않고 route/top-return failure만 닫을 수 있는지 보는 bounded smoke다.

동시에 `public20/UCTR80` candidate-id blend를 remote CPU에서 생성 완료했다. source-of-truth output은 `artifacts/mobilecropnet_v4/quality_first_20260428/blended_uctr80_public20_labels/`이며, train `39,002` rows, val `4,947` rows, key mismatch `0`이다. 평균 positive count는 train `1.177811`, val `1.175460`으로 UCTR60 blend보다 더 hard top-candidate 학습에 가깝다. 같은 rank-calibration scope를 쓰는 `1100295` `mcn-q-cnv2b448-uctr80-rankcal-r1-20260428`도 생성했다. label 생성 중 동일 목적의 중복 builder 프로세스가 발견되어 하나만 유지했고, `blended_uctr80_public20_labels` 경로를 source of truth로 고정한다.

`1100295`는 Succeeded 상태로 종료됐지만 train gate decision은 `excluded_train_gate`다. best epoch1에서 top-return `0.606832`, proposal recall `0.942041`, action top1 `0.990019`로 UCTR60 rankcal보다 나아졌고 top-return은 처음으로 기준 `0.58`을 넘었다. 그러나 route `0.488185`, subject IoU `0.493219`가 미달이며 epoch2도 route `0.476605`, top-return `0.596986`으로 개선하지 못했다. 따라서 UCTR80 signal은 ranking/action alignment 개선 evidence로만 유지하고, `1100295` 자체는 embedded/direct/qualitative gate로 승격하지 않는다.

`1100282` UCTR60 rank-calibration은 terminal gate decision 기준 best selection `0.728389`였지만 route `0.486363`, subject IoU `0.493219`, top-return `0.549120`으로 release train gate를 넘지 못했다. subject-valid balanced `0.636107`, proposal recall `0.941572`, action top1 `0.992557`은 통과했으나 세 핵심 blocker가 동시에 남아 embedded/direct/qualitative gate로 승격하지 않는다.

2026-04-28 18:20 KST에는 UCTR teacher가 public teacher보다 strong한데 학생 모델에서는 public track이 강하게 남는 현상을 label conflict 관점에서 한 번 더 좁혔다. `build_mobilecropnet_v4_blended_public_uctr_labels.py`에 기본 동작을 바꾸지 않는 consensus filter 옵션을 추가했고, remote CPU에서 `public20/UCTR80 + source top-id agreement` label set을 생성했다. source-of-truth output은 `artifacts/mobilecropnet_v4/quality_first_20260428/blended_uctr80_public20_topagree_labels/`이며, train `12,366` rows, val `1,601` rows, key mismatch `0`, skipped top-id mismatch train `26,636`, val `3,346`이다. 필터 후 top-id match rate는 `1.0`, blended top score disagreement mean은 train `0.098905`, val `0.095370`으로 내려갔다. 이 축은 UCTR80 rankcal과 중복이 아니라 public/UCTR target conflict가 큰 row를 제거하는 ablation이며, `1100317` `mcn-q-cnv2b448-topagree-rankcal-r1-20260428`로 생성했다. 생성 직후 상태는 Pending이며 10분 규칙에 따라 Running 전환 여부를 관리한다.

`1100245` UCTR60 head-calibration은 MLP 상태가 `Terminated`로 끝났고 wrapper summary를 쓰기 전에 종료됐으므로, durable `train_status.json`의 best row로 `train_gate_decision.json`을 보강했다. best selection은 `0.721171`이나 route `0.488177`, subject IoU `0.493219`, top-return `0.539201`로 train gate 미달이다. subject-valid balanced `0.628599`, proposal recall `0.939853`, action top1 `0.995720`은 통과했지만 핵심 route/top-return/subject가 동시에 닫히지 않아 embedded/direct/qualitative gate로 승격하지 않는다.

strict headcal `mcn-q-cnv2b448-blendjoint-headcal-strict-r1-ip134-20260428`도 best epoch1 기준 selection `0.734321`까지 올랐지만 route `0.511047`, subject IoU `0.493219`, top-return `0.553338`로 train gate에 못 미쳐 `excluded_train_gate`로 종료됐다. SigLIP-B/384 semantic backbone `1100268`은 epoch1에서 route `0.429496`, subject IoU `0.426907`, proposal recall `0.910273`, action top1 `0.832167`, top-return `0.347667`로 gate와 멀어 조기 종료했다. 회수한 MLP slot에는 `1100328` `mcn-q-cnv2b448-topagree-strict-r1-20260428`을 생성했다. 이 run은 `1100317` topagree rankcal과 같은 consensus label을 쓰지만 candidate encoder/relation MLP를 고정하고 route/ranker/policy/checklist/why head만 보정해 subject/proposal/action 보존성을 분리한다.

2026-04-28 19:05 KST에는 단순 label 비율/consensus만으로 부족할 가능성을 반영해 top-k/ranking loss 축을 추가했다. `run_mobilecropnet_v4_quality_relaxed_joint.sh`에 이미 구현되어 있던 `teacher_distill`, `topk_coverage`, `generated_proposal_positive_margin`, `route_object_single_margin` 손실을 환경변수로 노출했으며, 기본값은 기존 동작과 동일하게 유지했다. 잘못된 label/init path로 만든 r1 `1100332`/`1100333`은 Pending 중 즉시 종료했고, path는 맞지만 trainable scope가 기존 rankcal과 달라 utility/positive/set-ranker를 충분히 학습하지 못하는 r2 `1100335`/`1100336`도 조기 종료했다. corrected topagree r3 `1100341`은 Running이고, UCTR80 r3 `1100340`은 10분 이상 Pending이라 종료한 뒤 corrected r4 `1100351` `mcn-q-cnv2b448-uctr80-topkcal-r4-20260428`로 재생성했다. `1100341`/`1100351`은 기존 `1100317`/`1100295`와 같은 rankcal trainable scope를 쓰지만, top1_not_positive/top-return failure를 직접 벌점화하는 loss ablation이므로 중복 workload가 아니다.

2026-04-28 19:19 KST에는 UCTR80 strict head-calibration `1100344`도 Running 목록에 포함해 추적 대상으로 편입했다. 이 run은 `1100295`와 같은 UCTR80 label/init을 쓰되 candidate encoder/relation MLP를 고정하고 route/ranker/policy/checklist/why 계열만 보정하는 strict scope다. 또한 `1100295`가 raw train route gate에서는 제외됐지만 제품 계약상 route expert를 single checkpoint 내부에 embed하는 방식은 external prior가 아니므로, 진단용 no-prior gate `1100359` `mcn-q-cnv2b448-uctr80-routeens-embedgate-r1-20260428`을 생성했다. `1100359`는 새 학습이 아니라 UCTR80 rankcal best checkpoint에 SwinV2/ConvNeXtV2-L/EVA02-L route expert ensemble을 내장한 뒤 head/direct/qualitative surface를 확인하는 bounded eval이며, Pending이 10분을 넘으면 기존 pending rule에 따라 종료/재생성한다. 같은 시점에 runner에는 후속 route/subject 직접 보정 rerun을 위해 `ROUTE_FOCAL_GAMMA`, route-balanced sampler power/min/max, subject box L1/IoU/center/size/aspect/CIoU weights를 env로 추가 노출했다. 기본값은 기존과 동일하므로 진행 중인 run이나 기존 재현성에는 영향을 주지 않는다.

2026-04-28 19:36 KST에는 interactive full-warm `mcn-q-cnv2b448-uctr80-fullwarm-r1-ip134-20260428`가 완료됐다. best epoch1은 top-return `0.602377`, subject IoU `0.509927`, subject-valid balanced `0.593831`, proposal recall `0.967459`, action top1 `0.995119`로 통과했지만 route balanced `0.509455 < 0.52` 하나만 남아 `excluded_train_gate`로 닫았다. 이는 UCTR80 full-warm이 ranking/action/subject/proposal을 동시에 살린 첫 근접 실패이며, remaining blocker가 route calibration으로 좁혀졌다는 evidence다. 이를 반영해 MLP routefocal `1100365`를 만들었지만 Pending 상태였고, interactive GPU가 비면서 중복을 막기 위해 `1100365`는 종료했다. 대신 interactive `10.2.4.134`에서 `mcn-q-cnv2b448-uctr80-fullwarm-routepolish-r2-ip134-20260428`를 실행했다. r1 routepolish는 `ROUTE_HEAD_HIDDEN_MULT=0.75`가 빠져 init shape mismatch 위험이 있어 중단했고, r2는 full-warm best checkpoint의 `530/530` keys를 모두 로드했으며 skipped shape key `0`이다. r2는 route head/kind/cardinality head만 학습해 subject/proposal/action을 보존하면서 route focal `1.2`, route weight `28.0`, route expert distill `9.0`, sampler power `1.5`로 남은 route gap만 보정한다.

2026-04-28 19:58 KST에는 중간 종료된 잡을 source-of-truth artifact 기준으로 정리했다. `1100317` top-agree rankcal은 epoch2 중간에 MLP 세션이 종료됐고 terminal summary는 없지만, best epoch1의 route balanced `0.497316`, action top1 `0.899878`이 train gate와 멀어 `excluded_train_gate_interrupted_workload`로 닫았다. `1100359` route-ensemble embedded diagnostic은 embedded checkpoint 파일은 생성됐으나 head audit `1,644`장 처리 후 세션이 종료되어 direct eval과 qualitative pack이 없다. 이는 checkpoint 품질 탈락이 아니라 평가 불완료로 기록했으며, active 학습 잡이 정리되기 전에는 중복 GPU 재평가를 만들지 않는다.

2026-04-28 20:00 KST에는 `1100344` UCTR80 strict가 Succeeded로 끝났다. best epoch1에서 top-return `0.603475`, proposal recall `0.942041`, action top1 `0.990814`, subject-valid balanced `0.636107`은 통과했지만 route balanced `0.501092`, subject IoU `0.493219`가 미달이라 배포 후보가 아니다. strict scope는 ranker/action 보존에는 유리했지만 subject IoU를 full-warm 수준으로 끌어올리지 못했다. 따라서 더 유망한 full-warm best checkpoint에 route expert ensemble을 내부 embedding하는 single-checkpoint bounded gate eval `1100378`을 추가했다. 이 축은 final input contract를 `image + target_ar`로 유지하면서 route-only blocker를 내부 모델 크기 증가로 해결할 수 있는지 보는 진단이다.

2026-04-28 20:12 KST에는 full-warm best의 route-only blocker를 빠르게 분해하기 위해 `src/scripts/run_mobilecropnet_v4_route_bias_gate.sh`를 추가했다. 이 runner는 route logit bias만 folded checkpoint로 저장하고, held-out eval route balanced가 `0.52` 이상일 때만 no-prior head/direct/qualitative gate를 실행한다. MLP `1100379` `mcn-q-uctr80-fullwarm-routebias-r1-20260428`은 route logit bias key 선택 버그로 실패했고, corrected source-of-truth r2는 `1100383` `mcn-q-uctr80-fullwarm-routebias-r2-20260428`이다. 같은 tag/output을 공유하던 중복 `1100385`는 2026-04-28 20:19 KST에 종료했다. 이는 route-polish 학습과 중복되는 workload가 아니라, full-warm checkpoint가 단순 bias calibration만으로 release route gate를 닫을 수 있는지 확인하는 bounded 진단이다.

2026-04-28 20:19 KST에는 추가 terminal 결과를 정리했다. `1100351` UCTR80 top-k/margin calibration r4는 best epoch1 기준 top-return `0.604946`, proposal `0.942041`, action `0.989943`을 유지했지만 route `0.488929`, subject IoU `0.493219`로 제외했다. `1100328` top-agree strict는 epoch2 후 workload가 종료됐고 best row는 route `0.502747`만 미달이다. action/top-return/subject/proposal을 동시에 통과한 near-miss지만, full-warm route-only 축이 더 유망하므로 즉시 재시작하지 않는다. `1100379` route-bias r1은 route logit bias folding 코드가 class-count bias가 아닌 hidden-layer bias `(288,)`를 잡아 실패했다. status를 `failed_route_bias_shape_mismatch`로 정리했고, corrected route-bias r2 source of truth는 `1100383`이다. 같은 tag/output을 공유하던 중복 `1100385`는 즉시 종료했다. corrected script는 checkpoint의 `route_head.6.bias (7,)`처럼 calibration bias shape와 정확히 일치하는 route logits bias만 선택한다.

2026-04-28 20:28 KST에는 `1100383` full-warm route-bias r2도 terminal 판정으로 닫았다. calibration train split에서는 route balanced가 `0.793968 -> 0.819253`로 올랐지만, 순수 UCTR held-out eval에서는 `0.407166 -> 0.428291`에 그쳐 `0.52` gate를 넘지 못했다. 이는 full-warm route failure가 단순 class-prior bias 문제가 아니라 label/route representation generalization 문제라는 evidence다. 따라서 calibrated checkpoint는 no-prior head/direct/qualitative로 확장하지 않는다. 회수한 GPU 축에는 `1100328` top-agree strict near-miss checkpoint의 route-bias 진단 `1100393` `mcn-q-topagree-strict-routebias-r1-20260428`을 생성했다. 이 진단도 top-agree 내부 validation만 보지 않고 순수 UCTR held-out split을 gate로 사용한다.

2026-04-28 20:31 KST에는 active GPU를 더 효율적으로 쓰기 위해 route-bias와 다른 두 후속 축을 병렬로 추가했다. `1100394` `mcn-q-cnv2b448-fullwarm-headpolish-r1-20260428`은 full-warm best checkpoint에서 subject/proposal/backbone을 보존하고 route, ranker, policy, checklist, why head만 보정한다. 목적은 full-warm의 좋은 subject/proposal/action metric을 흔들지 않으면서 route와 top-return head를 동시에 닫는 것이다. `1100395` `mcn-q-cnv2b448-topagree-strict-routepolish-r1-20260428`은 top-agree strict near-miss checkpoint에서 route head/kind/cardinality만 학습한다. 이 축은 `1100328`이 route 하나만 놓친 결과를 직접 겨냥하며, route-bias `1100393`과 달리 logit bias가 아니라 route representation/head weight를 미세 보정한다. 세 run은 서로 다른 output과 가설을 사용하므로 중복 workload가 아니다.

2026-04-28 20:36 KST에는 top-agree top-k/margin calibration과 route-bias 진단도 terminal로 닫았다. `1100341`은 best epoch2에서 top-return `0.642832`, subject IoU `0.527490`, subject-valid balanced `0.678317`, proposal recall `0.938755`, action top1 `0.982120`을 통과했지만 route balanced `0.493769`로 제외했다. `1100393` top-agree strict route-bias는 held-out route balanced가 `0.455284 -> 0.472744`에 그쳐 제외했다. route가 단순 loss/positive-margin 또는 class-prior bias로는 닫히지 않는다는 근거가 추가됐으므로, 회수 GPU에는 `1100405` `mcn-q-cnv2b448-fullwarm-relpolish-r1-20260428`을 생성했다. `1100405`는 `1100394`보다 한 단계 더 풀어 candidate encoder와 relation MLP까지 낮은 LR로 학습하되, subject/proposal/backbone은 그대로 보존한다.

2026-04-28 20:38 KST에는 `1100378` full-warm route-expert embedded checkpoint의 head audit이 완료됐다. route balanced `0.625318`, subject IoU `0.555747`, subject-valid balanced `0.743157`, proposal recall@5 `0.959375`로 head release gate는 통과했지만, top1_not_positive `0.671875`, route mismatch `0.435000`가 높아 direct/qualitative 전에는 배포 후보로 보지 않는다. 같은 checkpoint의 release artifact를 선제 확보하기 위해 final bundle `1100406`과 GAIC official `1100407`을 별도 workload로 생성했다. `1100406`은 Running이고, `1100407`은 Pending 10분 규칙으로 감시한다.

2026-04-28 20:42 KST에는 interactive full-warm route-polish `mcn-q-cnv2b448-uctr80-fullwarm-routepolish-r2-ip134-20260428`를 제외했다. best epoch2에서 subject IoU `0.523704`, proposal recall `0.967243`, action top1 `0.995057`, top-return `0.602487`은 통과했지만 route `0.486165`, generated align top1 `0.0`이다. `1100407` GAIC official은 Succeeded로 완료됐고, full-warm routeens embedded checkpoint의 GAIC top1 MOS `3.759520`, SRCC `0.411539`, Accw4@10 `0.352713`이다. head route 개선이 GAIC official 품질로는 충분히 전이되지 않았으므로 `1100378`은 direct/qualitative와 final bundle public/equal-4까지 확인해야 한다.

2026-04-28 20:43 KST에는 `1100378`을 종료했다. head audit 산출물은 확보됐고, 같은 checkpoint의 direct/qualitative 및 public/equal-4 평가는 `1100406` final bundle이 수행 중이므로 중복 GPU 점유를 막기 위한 조치다. gate status는 `stopped_after_head_audit_duplicate_finalbundle`로 정리했다. interactive GPU에는 route-only가 반복 실패한 점을 반영해 top-agree strict checkpoint의 route+ranker/policy head-polish `mcn-q-cnv2b448-topagree-strict-headpolish-r1-ip134-20260428`을 실행했다.

2026-04-28 20:49 KST에는 `1100406` 단일 final bundle도 종료했다. public manifest `62,511` row를 단일 GPU에서 export하는 방식이 너무 느려 `2,144` row 처리 시점에 중단했으며, 이는 checkpoint 품질 실패가 아니라 실행 방식 실패다. 따라서 `1100406`은 더 이상 full public/equal-4 source of truth가 아니고, 이미 완료된 `1100407` GAIC official 산출물만 embedded routeens checkpoint의 GAIC evidence로 유지한다.

2026-04-28 20:52 KST에는 같은 embedded routeens checkpoint의 public/equal-4를 `8`개 shard workload `1100416`~`1100423`으로 재구성했다. 각 shard는 `export_mobilecropnet_v4_public_benchmark_predictions.py --task_shard_index i --task_shard_count 8`로 같은 manifest를 분할 처리하고, 출력은 `artifacts/mobilecropnet_v4/quality_first_20260428/public_shards/mcn-q-cnv2b448-fullwarm-routeens-embedgate-r1-20260428/shardXX/`에 쓴다. 완료 후 `merge_mobilecropnet_v4_public_shards.py`로 `62,511` task row를 검증 병합하고 `evaluate_public_task_manifest_predictions.py`로 FCDB IoU, CPC weighted, GNMC IoU, equal-4를 산출한다. 20:52 KST 기준 `1100418`, `1100420`, `1100421`, `1100422`, `1100423`은 Running이고 `1100416`, `1100417`, `1100419`는 Pending이나 아직 10분 초과 전이다.

같은 시각 route-bias가 held-out split으로 전이되지 않는 문제를 겨냥해 image-global residual route head ablation도 병렬 실행했다. `1100424` `mcn-q-cnv2b448-fullwarm-imgresroute-r1-20260428`은 full-warm best checkpoint에서 route/residual route head만 열고, `1100425` `mcn-q-cnv2b448-topagree-imgresroute-r1-20260428`은 top-agree strict best checkpoint에서 같은 residual head를 학습한다. 둘 다 route expert는 학습 중 distillation target으로만 사용하며 최종 runtime 계약은 `image + target_ar`다.

2026-04-28 20:57 KST에는 `1100416`~`1100423` 8-shard r1이 모두 `Session terminated, killing shell` 로그로 Terminated됐다. 일부 shard는 `1,152~1,536` task row의 partial prediction을 남겼지만 `export_summary.json`이 없고 full manifest coverage가 아니므로 평가 source로 쓰지 않는다. 이는 checkpoint 품질 실패가 아니라 MLP workload 실행 안정성/시간 설정 문제다.

2026-04-28 21:00 KST에는 output collision을 피하기 위해 `public_shards_16x_r2` root로 16-shard r2 `1100431`~`1100446`을 생성했다. 각 shard는 `--task_shard_count 16`, `--expected-run-time=35m`로 더 작은 task partition을 처리한다. 이 r2 set이 모두 terminal summary를 내고 `62,511` row merge가 검증될 때만 embedded routeens checkpoint의 FCDB/CPC/GNMC/equal-4를 리더보드에 반영한다.

2026-04-28 21:05 KST에는 `1100406`의 partial public recovery를 검증했다. 해당 `predictions.jsonl`은 정상 full public benchmark의 `62,511` task row가 아니라 `2,368` image-level row라 FCDB/CPC 일부만 우연히 match되고 GNMC는 missing score가 된다. CPU로 만든 복구 summary는 `summary.invalid_image_level_recovery_20260428.json`으로 보존하고 `INVALID_RECOVERY_REASON.json`을 남겼으며, 리더보드 source of truth에서 제외한다.

같은 시각 사용자 추가 지시인 "온디바이스 제약 완화 품질 우선 모델군"을 별도 실험으로 구체화했다. 기존 scratch `EVA02-L/448` full model은 route와 subject IoU 신호는 있었지만 valid calibration, proposal/action, top-return이 무너졌으므로, backbone과 subject geometry를 고정하고 `subject_valid_head`, `proposal_*`, `candidate_encoder`, `relation_*`, `set_ranker`, utility/positive/risk, policy/action, checklist/why head만 여는 headvalid 보정 축을 설계했다. r1은 MLP command here-doc이 한 줄로 접혀 syntax 실패했고, r2는 label filename을 잘못 지정해 실패했다. 실제 source label path를 확인해 r3 `1100449`(`blended_uctr80_public20_labels/train_blended_public_uctr.jsonl`)와 `1100450`(`blended_uctr80_public20_topagree_labels/train_blended_public_uctr.jsonl`)을 재생성했다. 이 두 run은 더 강한 pretrained backbone 자체를 반복하는 것이 아니라, 큰 백본의 route/subject 신호를 제품 head alignment로 전이할 수 있는지 보는 비중복 후속 실험이다.

2026-04-28 21:07 KST에는 routeens embedded checkpoint의 Product-AR direct eval과 qualitative pack이 없음을 확인했다. head audit은 이미 완료됐으므로 중복하지 않고 direct/qual만 별도 workload로 실행한다. r1 `1100451`은 command의 status JSON env export 결함을 Pending 상태에서 종료했고, corrected r2 `1100452` `mcn-routeens-directqual-r2-20260428`을 source of truth로 생성했다.

2026-04-28 21:11 KST에는 public r2 pending shard를 정리했다. shard00 `1100431`과 shard13 `1100444`가 10분 Pending rule을 넘겨 종료됐고, 같은 output root와 shard index를 유지한 r3 `1100455`(shard00)와 `1100456`(shard13)을 생성했다. 이 시점에 shard00 r3는 Running, shard13 r3는 Pending이며, 나머지 14개 shard는 Running으로 유지된다. 진행 중인 shard의 status 합산은 `25,088` task row다.

2026-04-28 21:17 KST에는 shard13 r3 `1100456`도 Running으로 전환되어 public shard `16 / 16`이 모두 처리 중이다. 합산 진행은 `48,704 / 62,511` task row다. EVA02-L headvalid UCTR80 r3 `1100449`는 Pending 10분 초과로 종료했고, 같은 corrected 설정을 r4 `1100462`로 재생성했다. top-agree r3 `1100450`과 direct/qual r2 `1100452`는 Running이다.

2026-04-28 21:22 KST에는 direct/qual r2 `1100452`를 종료하고 r3 `1100465`로 재생성했다. r2는 `400 / 3,200` image 처리 기준 실제 속도가 `expected-run-time=25m`보다 길어 platform 종료 위험이 있었다. r3는 같은 direct/qual 평가를 `expected-run-time=75m`로 수행한다. public shard는 `14 / 16`개가 `Succeeded`와 `export_summary.json`을 확보했고, 남은 source는 shard00 r3 `1100455`와 shard13 r3 `1100456`이다.

2026-04-28 21:45 KST에는 routeens embedded checkpoint의 public/equal-4 source set을 닫았다. 16-shard r2/r3 source set은 `62,511 / 62,511` task row로 병합됐고 `duplicate_count=0`, `complete=true`다. 기존 evaluator는 429MB prediction과 405MB manifest를 동시에 보존하다 single CPU에서 `Killed`됐으므로, `evaluate_public_task_manifest_predictions.py`에 SQLite 디스크 인덱스 모드를 추가했다. `eval_sqlite` 결과는 `prediction_rows_seen=62511`, `prediction_rows_matched=62511`, `missing_prediction_task_count=0`이며, FCDB `0.720297`, CPC weighted `0.828543`, GNMC `0.705682`, overall IoU `0.704759`다. 이 결과는 FCDB/GNMC를 소폭 올렸지만 CPC를 낮췄기 때문에 winner 확정 근거가 아니다. GAIC official은 `1100407`의 top1 MOS `3.759520`, SRCC `0.411539`, Accw4@10 `0.352713`이고, head audit은 route balanced `0.625318`, subject IoU `0.555747`, subject-valid balanced `0.743157`이지만 `top1_not_positive=0.671875`, `route_mismatch=0.435000`가 남는다. 따라서 이 checkpoint는 Product-AR direct/qualitative `1100465`가 끝날 때까지 provisional gate row로만 유지한다. EVA02-L headvalid UCTR80 r4 `1100462`는 pending issue로 종료됐고, 같은 corrected 설정의 r5 `1100471`이 source-of-truth Running run이다.

2026-04-28 22:10 KST에는 routeens embedded checkpoint의 Product-AR direct/qualitative r3 `1100465`도 Succeeded로 닫혔다. direct metric은 final positive hit `0.958438`, final IoU-to-best-positive `0.824520`, proposal recall@5 `0.979375`, target-AR compatibility `1.0`, subject IoU `0.556337`, subject valid acc `0.729375`, route acc `0.586250`, risk hit `0.029688`이다. qualitative pack은 subject_iou_low `27.31%`, subject_valid_mismatch `27.06%`, final_not_positive `4.16%`, risk_hit `2.97%`, proposal_top5_miss `2.06%`를 기록했다. 따라서 이 checkpoint는 internal route head 수치와 FCDB/GNMC 개선에도 불구하고, 기존 no-prior quality-first row보다 direct final hit/IoU가 낮고 subject qualitative blocker가 반복되어 deploy candidate에서 제외한다.

같은 시각 `1100395` top-agree strict route-polish도 terminal `excluded_train_gate`로 정리했다. best epoch2는 subject IoU `0.527490`, subject-valid balanced `0.678317`, proposal recall `0.938755`, action top1 `0.972458`, top-return `0.639018`을 만족했지만 route balanced `0.504352`만 `0.52` gate를 넘지 못했다. `1100450` EVA02-L headvalid top-agree는 route `0.635456`, subject IoU `0.613148`로 강한 pretrained backbone의 장점이 확인됐지만 subject-valid balanced `0.539950`, action top1 `0.900125`, top-return `0.518727`이 낮아 제품 head alignment가 실패했다. `1100486` top-agree route-polish + Swin route embed gate r1은 command env export 결함으로 중단하고, 동일 가설을 corrected script `artifacts/mobilecropnet_v4/quality_first_20260428/gate_runs/mcn-q-topagree-routepolish-swinembed-gate-r2-20260428/run_gate.sh` 및 workload `1100487`로 재생성했다.

2026-04-28 22:18 KST에는 interactive top-agree strict head-polish `mcn-q-cnv2b448-topagree-strict-headpolish-r1-ip134-20260428`도 `excluded_train_gate`로 닫았다. best epoch1은 subject IoU `0.527490`, subject-valid balanced `0.678317`, proposal recall `0.938755`, action top1 `0.986778`, top-return `0.627610`, generated align `0.945274`를 만족했지만 route balanced `0.502273`으로 실패했다. 또한 `1100471`/`1100472` EVA02-L headvalid UCTR80 r5가 같은 `RUN_NAME`과 output path로 겹쳐 실행된 것을 확인해, r5 산출물은 source-of-truth에서 제외하고 살아 있던 duplicate `1100472`도 종료했다. 동일 가설은 고유 output의 `1100493` `mcn-q-eva02l448-headvalid-uctr80-r6-20260428`로 재생성했고, 별도 가설인 `1100479` `mcn-q-eva02l448-topagree-rankaction-r1-20260428`은 유지한다.

2026-04-28 22:25 KST에는 `1100425` top-agree image-residual route가 MLP wrapper gate 작성 전에 Terminated로 끝난 것을 확인했다. `metrics.json` 기반으로 durable `summary.json`과 `train_gate_decision.json`을 보강했으며, best epoch1은 route `0.506946`, subject IoU `0.527490`, subject-valid balanced `0.678317`, proposal recall `0.938755`, action top1 `0.972458`, top-return `0.639018`, generated align `0.0`이다. route gate와 generated alignment가 닫히지 않아 final gate 확장 대상에서 제외한다. 회수한 GPU에는 `1100494` `mcn-q-swinv2b384-topagree-headpolish-r1-20260428`을 생성했다. 이 축은 SwinV2-B/384 `mcn-q-swinv2b384-blendjoint-r2` best에서 top-agree label로 route/ranker/policy/subject-valid/subject-box head를 보정해, Swin route expert 강점을 full MobileCropNet 내부 head로 전이할 수 있는지 보는 비중복 품질 우선 실험이다.

2026-04-28 22:32 KST에는 `1100487` top-agree route-polish + Swin route embed gate r2가 head audit을 통과했다. route balanced `0.571399`, subject IoU `0.523292`, subject-valid balanced `0.706891`, candidate top1 hit `0.309688`, top1 IoU-to-best-positive `0.826425`이며, 같은 workload에서 Product-AR direct/qualitative가 진행 중이다. direct/qual 대기 시간을 줄이기 위해 같은 embedded checkpoint의 GAIC official `1100498` `mcn-q-topagree-swinembed-gaic-r1-20260428`도 병렬 생성했다. `1100394` full-warm head-polish는 Terminated 후 wrapper gate를 쓰지 못해 `metrics.json` 기반으로 durable `summary.json`/`train_gate_decision.json`을 보강했다. best epoch1은 route `0.504652`, subject IoU `0.541445`, subject-valid balanced `0.633464`, proposal recall `0.966108`, action top1 `0.991883`, top-return `0.603230`, generated align `0.955408`이며 route gate 미달로 제외한다.

2026-04-28 22:39 KST에는 `1100405` full-warm relation-polish도 Terminated 후 wrapper gate를 쓰지 못해 durable summary/gate를 보강했다. best epoch1은 route balanced `0.506673`, subject IoU `0.541445`, subject-valid balanced `0.633464`, proposal recall `0.966108`, action top1 `0.992709`, top-return `0.594832`, generated align `0.953793`으로 route gate 하나만 미달이다. 같은 full-warm init의 head/rel/imgres route 축이 모두 route `0.50~0.51`에 머물러 route representation generalization 문제가 다시 확인됐다. `1100487` 후보가 head gate를 통과했으므로, direct/qualitative 결과를 기다리면서 public/equal-4 export shard 0 `1100505` `mcn-q-topagree-swinembed-pub-s00-r1-20260428`도 생성했다. direct/qualitative에서 catastrophic blocker가 확인되면 추가 shard 확장은 중단한다.

2026-04-28 22:45 KST에는 `1100487`도 최종 release gate에서 제외했다. Product-AR direct는 final hit `0.961875`, final IoU-to-best-positive `0.836988`, proposal recall@5 `0.971875`, target-AR compatible `1.0`, subject IoU `0.523292`, subject-valid acc `0.715625`, route acc `0.565313`, risk hit `0.029688`이다. GAIC official은 top1 MOS `3.741880`, SRCC `0.409028`, Accw4@10 `0.349432`, GAIC primary 약 `0.569762`로 낮다. qualitative pack도 subject_valid_mismatch `28.44%`, subject_iou_low `27.03%`, final_not_positive `3.81%`, risk_hit `2.97%`, proposal_top5_miss `2.81%`를 기록했다. route head만 내부 embedding으로 고쳐도 subject/action qualitative blocker가 반복되므로, public shard00 `1100505`는 `576` task partial에서 중단했고 full public/equal-4 확장은 만들지 않는다.

같은 시각 `1100479` EVA02-L topagree rank/action도 terminal `excluded_train_gate`로 닫았다. route `0.612984`, subject IoU `0.613148`, proposal recall `0.939323`, generated align `0.946317`은 강하지만 subject-valid balanced `0.539950`, action top1 `0.920100`, top-return `0.544320`이 gate 미달이다. 이 결과는 EVA02-L의 route/subject 표현은 유효하지만 제품 action/ranker calibration이 약하다는 신호이므로, 후속 `1100507` `mcn-q-eva02l448-topagree-actionfix-r2-20260428`은 같은 checkpoint를 init으로 두고 valid/action/top-return/policy score head만 더 강하게 보정한다. `1100424` full-warm image-residual route도 route `0.500557`, generated align `0.0`으로 제외했다.

2026-04-28 23:05 KST에는 남은 실험축을 release gate blocker별로 재정렬했다. `1100493` EVA02-L headvalid UCTR80 r6는 epoch1에서 route `0.602188`, subject IoU `0.523736`, proposal recall `0.941643`를 보였지만 subject-valid best balanced `0.474063`, action top1 `0.921563`, top-return `0.481875`가 낮아 아직 배포 후보가 아니다. `1100494` SwinV2-B topagree head-polish와 `1100507` EVA02-L actionfix는 계속 실행 중이다. interactive GPU `10.2.4.134`에는 이미 ConvNeXtV2-L/512 topagree head-polish가 돌고 있음을 확인했기 때문에, 같은 GPU에 잘못 겹쳐 올린 Swin vertical repair r1은 중단하고 `abort_reason.json`을 남겼다.

같은 시각 subject-valid calibration과 vertical AR failure를 별도 후속 축으로 분리했다. top-agree label에서 `9:16`, `3:4` row를 1회 추가 복제한 vertical repair label set을 만들었고 train row는 `12,366 -> 16,017`이다. 이 label을 쓰는 SwinV2-B vertical repair는 workload `1100512` `mcn-q-swinv2b384-verticalrepair-r2-20260428`로 재생성했다. EVA02-L 쪽은 `1100479` best의 route/subject/proposal geometry를 보존하면서 subject-valid negative scale `3.2`, subject-valid weight `6.0`, action/top-return/policy loss를 올린 `1100509` `mcn-q-eva02l448-calaction-neg-r1-20260428`을 만들었다. 이 두 run은 speed보다 품질을 우선하는 품질 우선 모델군이며, 최종 inference 입력은 계속 `image + target_ar`로 제한한다.

2026-04-28 23:16 KST에는 `1100493` EVA02-L headvalid UCTR80 r6가 terminal `excluded_train_gate`로 닫혔다. best epoch2는 route balanced `0.608438`, subject IoU `0.523736`, proposal recall `0.956999`, generated align top1 `0.948438`로 route/subject/proposal geometry는 유지했지만, subject-valid best balanced `0.474688`, action top1 `0.956875`, top-return `0.518125`가 각각 gate `0.55`, `0.96`, `0.58`에 미달했다. 이 결과는 UCTR80 headvalid가 geometry를 회복해도 subject-valid calibration과 action/top-return alignment를 같이 닫지 못하면 release candidate가 될 수 없다는 증거다. 따라서 `1100493`은 public/equal-4/GAIC/direct/qual 확장 없이 제외하고, 같은 원인을 겨냥한 `1100507` actionfix와 `1100509` negative-calibration 축의 terminal 결과를 기다린다.

같은 시간대에 `1100512` vertical repair r2는 Pending 10분 rule에 따라 종료했다. 같은 가설의 r3 `1100513`은 start time 없이 `Terminated`됐고 log가 `The job is not executed.`라 산출물이 없다. 플랫폼에 남은 duplicate pending r3 `1100514`는 같은 output 충돌 위험 때문에 종료했다. source-of-truth vertical repair는 r4 `1100515`이며, 2026-04-28 23:18 KST Running으로 전환됐고 IP는 `10.2.4.169`다. 현재 활성 품질 우선 판단 축은 `1100494`, `1100507`, `1100509`, `1100515`, `1100510/1100511` composite gate, interactive GPU `10.2.4.134`의 ConvNeXtV2-L/512 topagree head-polish다.

2026-04-28 23:23 KST에는 `1100507` EVA02-L actionfix도 terminal `excluded_train_gate`로 제외했다. best epoch2는 route balanced `0.639201`, subject IoU `0.613148`, proposal recall `0.939323`, generated align top1 `0.948190`으로 route/subject/proposal은 좋지만, subject-valid best balanced `0.539950`, action top1 `0.917603`, top-return `0.536829`가 gate에 미달했다. 특히 action/top-return loss를 높였는데도 val action이 `0.96` 근처로 가지 못해, top-agree label만으로는 제품 action/policy alignment가 닫히지 않는다는 결론을 추가한다.

이 결과를 반영해 회수된 GPU에는 `1100517` `mcn-q-eva02l448-uctr80-validtop-r7-20260428`을 추가했다. 이 run은 `1100493` best checkpoint를 init으로 사용하고, route/subject/proposal geometry는 freeze에 가깝게 보존하면서 subject-valid, ranker/utility/positive/risk, policy/decision/action 관련 head만 낮은 LR `3e-6`로 보정한다. UCTR80/public20 label을 유지하는 이유는 `1100493`에서 action이 `0.956875`까지 올라 top-agree `1100507`보다 action head와 더 맞았기 때문이다. 이 축도 train gate를 넘기 전에는 public/GAIC/direct/qual 확장을 하지 않는다.

2026-04-28 23:32 KST에는 SwinV2-B vertical repair r4 `1100515`를 종료하고 bounded smoke r5 `1100519`로 바꿨다. r4는 정상 실행됐지만 `quality_swinv2b_384` + vertical oversampling full rows 조합에서 step 속도상 `expected-run-time=90m` 안에 epoch1 validation까지 닿기 어렵다. r5는 같은 init/label 가설을 유지하되 `1 epoch`, train `6,400` rows, val `1,600` rows, batch `8`, trainable head scope 축소로 먼저 gate 신호를 얻는 실험이다. r5가 route/subject/proposal/action/top-return smoke를 넘기면 full-row rerun을 만들고, 못 넘기면 vertical oversampling은 원인 축에서 제외한다.

2026-04-28 23:39 KST에는 `1100510` fullwarm relation-polish + Swin route embedded gate와 `1100511` fullwarm head-polish + Swin route embedded gate를 모두 제외했다. 두 run 모두 Product-AR direct와 qualitative pack까지 완료했지만, relation-polish는 final positive hit `0.959375`, IoU-to-best-positive `0.825277`, subject-valid acc `0.714375`, route acc `0.565313`, risk-hit `0.029688`이고, qualitative failure가 subject_valid_mismatch `28.56%`, subject_iou_low `27.31%`, final_not_positive `4.06%`로 반복됐다. head-polish도 final hit `0.958438`, IoU-to-best `0.824937`, subject-valid acc `0.714375`, route acc `0.565313`, risk-hit `0.029375`로 유사하며 qualitative failure가 동일하게 남았다. 따라서 Swin route embedding은 route surface를 보완하지만 subject validity/box calibration을 제품 수준으로 internalize하지 못한다.

같은 시각 `1100519` vertical bounded smoke r5는 terminal `excluded_train_gate`로 닫혔다. best epoch1은 route balanced `0.494182`만 gate `0.52`에 미달했고, subject IoU `0.557412`, subject-valid best balanced `0.666533`, proposal recall `0.951878`, action top1 `0.988125`, top-return `0.614226`, generated align top1 `0.946250`은 모두 통과했다. 이 결과는 vertical oversampling이 target-AR/action/top-return head에는 유효하지만 route head를 고정하면 validation route distribution에서 무너진다는 뜻이다. 후속은 route head와 route distillation을 함께 여는 `1100522` `mcn-q-swinv2b384-verticalroute-r6-20260428`로 좁혔다.

2026-04-28 23:44 KST에는 pending/duplicate GPU workload를 정리했다. `1100520` pairwise ranker r2는 Pending 상태라 종료했고, 같은 fullwarm pairwise-ranker 가설은 `1100521` r3로 재생성해 Running 전환을 확인했다. 플랫폼에서 같은 run name의 duplicate `1100523`도 생겼으나 output path collision 위험 때문에 후발 duplicate만 종료했다. 현재 활성 품질 우선 판단 축은 `1100494`, `1100509`, `1100517`, `1100521`, `1100522`, interactive GPU `10.2.4.134`의 ConvNeXtV2-L/512 topagree head-polish다.

2026-04-28 23:47 KST에는 `1100509` EVA02-L cal/action negative-calibration도 terminal `excluded_train_gate`로 닫혔다. best epoch2는 route balanced `0.632959`, subject IoU `0.613148`, proposal recall `0.939323`, generated align top1 `0.949438`로 geometry는 유지했지만, subject-valid best balanced `0.539950`, action top1 `0.923221`, top-return `0.551186`이 gate `0.55/0.96/0.58`에 미달했다. 이 결과는 EVA02-L의 표현력 문제가 아니라 subject-valid calibration과 action/policy head alignment가 label transfer 과정에서 남는 문제라는 판단을 강화한다. 현재 활성 품질 우선 판단 축은 `1100494`, `1100517`, `1100521`, `1100522`, interactive GPU `10.2.4.134`의 ConvNeXtV2-L/512 topagree head-polish로 좁힌다.

2026-04-28 23:52 KST에는 EVA02-L action/top-return blocker의 원인이 ranking supervision 부족인지 분리하기 위해 `1100524` `mcn-q-eva02l448-uctr80-pairvalid-r8-20260428`을 추가 생성했고 Running 전환을 확인했다. 이 run은 `1100493` best를 init으로 쓰며, UCTR80/public20 base label에 top-agree teacher explicit pairwise label `33,508 / 4,447` pairs를 추가한다. `1100517`이 pure UCTR valid/top 보정이라면, `1100524`는 같은 geometry init에 pairwise preference를 넣는 비중복 축이다.

2026-04-28 23:57 KST에는 `1100522` verticalroute r6를 시간 예산 실패로 종료했다. route expert distillation을 켠 설정은 step `75 / 800`에 `332s`가 걸려 `expected-run-time=25m` 안에 validation까지 도달할 수 없었다. 같은 vertical route 가설은 route expert distill 없이 route label 손실과 route head만 여는 빠른 smoke `1100525` `mcn-q-swinv2b384-verticalroutefast-r7-20260428`로 재생성했고 Running 전환을 확인했다. `1100522`는 품질 실패가 아니라 실행 예산 오류로 분리하고, source of truth는 r7이다.

2026-04-29 00:04 KST에는 `1100525` verticalroutefast r7도 terminal `excluded_train_gate`로 제외했다. best epoch1은 subject IoU `0.557412`, subject-valid best balanced `0.668324`, proposal recall `0.951878`, action top1 `0.988375`, top-return `0.613911`, generated align top1 `0.943125`로 route 외 head는 모두 gate를 넘겼지만, route balanced가 `0.469950`으로 더 낮아졌다. vertical oversampling은 subject-valid/action/top-return을 안정화하지만 route distribution을 해결하지 못한다. 따라서 vertical route 축은 배포 후보 실험에서 제외하고, 새로 생긴 중복 `1100527` verticalroute r7은 output collision과 GPU 낭비를 막기 위해 즉시 종료했다.

2026-04-29 00:06 KST에는 interactive GPU `10.2.4.134`의 ConvNeXtV2-L/512 topagree head-polish도 epoch1 기준 route `0.508739`, subject IoU `0.502798`, subject-valid best balanced `0.538390`, proposal recall `0.929252`, action top1 `0.905743`, top-return `0.496255`로 train gate 전반과 멀어 중단하고 `abort_reason.json`을 남겼다. 같은 GPU에는 `1100494` epoch1 best를 init으로 하는 `mcn-q-swinv2b384-topagree-actionboost-r1-ip134-20260429`를 올렸다. 이 run은 route/subject/proposal을 보존하고 action/top-return/policy/ranker head만 강하게 보정하는 비중복 smoke이며, 중복 GPU workload가 아니라 기존 interactive GPU 회수 후 재사용이다.

2026-04-29 00:13 KST 기준 `1100494` SwinV2-B topagree head-polish는 train gate를 통과했다. best epoch2는 route balanced `0.536127`, subject IoU `0.542000`, subject-valid best balanced `0.610453`, proposal recall `0.951709`, action top1 `0.986076`, top-return `0.608271`, generated align top1 `0.943890`이다. 따라서 같은 MLP wrapper가 Swin route expert를 단일 checkpoint 내부에 embed한 `mcn-q-swinv2b384-topagree-headpolish-r1-20260428-swinroute-embedded.pt`를 만들고, 현재 no-prior head audit을 실행 중이다. 반대로 `1100521` fullwarm pairwise ranker r3는 explicit pairwise acc `0.800466`, top-return `0.660237`, subject IoU `0.561004`, subject-valid best balanced `0.629463`은 좋아졌지만 route balanced가 `0.126797`로 붕괴했고 action top1도 `0.927653`이라 조기 제외했다. 이 결과는 pairwise ranker 보정만으로는 route internalization을 보존할 수 없다는 evidence로 남긴다.

2026-04-29 00:16 KST에는 interactive `mcn-q-swinv2b384-topagree-actionboost-r1-ip134-20260429`도 terminal 판정을 끝냈다. action/top-return 보정은 action top1 `0.969362`, top-return `0.619983`, subject IoU `0.560081`, subject-valid best balanced `0.669847`, proposal recall `0.951851`로 효과가 있었지만 route balanced가 `0.494602`로 내려가 release train gate를 통과하지 못했다. 따라서 action-only 보정은 배포 후보 축에서 제외하고, 같은 SwinV2 계열에서 route head까지 함께 열어 pairwise/action을 검증하는 MLP `1100528` `mcn-q-swinv2b384-actionpair-r8-20260428`를 별도 병렬 판단 축으로 유지한다.

2026-04-29 00:17 KST에는 같은 interactive GPU를 비워 두지 않고 `mcn-q-swinv2b384-topagree-actionroutecal-r2-ip134-20260429`를 추가 실행했다. 이 run은 `1100494` best checkpoint에서 route head, subject-valid head, policy/decision/action, utility/positive/risk, checklist/why/detail head만 열고 candidate encoder, relation MLP, set ranker는 고정한다. 직전 actionboost는 candidate/relation/ranker 업데이트가 route candidate-context를 흔들어 route balanced가 `0.494602`로 내려간 것으로 보이므로, 이번 축은 action/top-return 보정 이득을 유지하면서 route를 보존할 수 있는지 확인하는 bounded smoke다.

2026-04-29 00:18 KST에는 `1100517` EVA02-L UCTR80 valid/top-return r7도 terminal `excluded_train_gate`로 닫았다. best epoch2는 route balanced `0.618125`, subject IoU `0.523736`, proposal recall `0.956999`, action top1 `0.971250`, generated align top1 `0.958438`로 geometry와 action은 유지했지만, subject-valid best balanced `0.465156`과 top-return `0.524063`이 gate를 넘지 못했다. 따라서 EVA02-L UCTR80 init의 병목은 큰 backbone 표현력이 아니라 subject-valid calibration과 top-return/ranking alignment이며, 같은 원인을 explicit pairwise로 검증하는 `1100524`만 남긴다.

2026-04-29 00:27 KST에는 interactive `actionroutecal-r2`도 실행 예산 실패로 중단했다. route expert distillation을 켠 설정은 step `50`까지 `257s`가 걸려 bounded smoke 안에 validation까지 도달하기 어렵다. 동일 가설은 route expert distill을 끄고 route/action/policy head만 빠르게 확인하는 `mcn-q-swinv2b384-topagree-actionroutecalfast-r3-ip134-20260429`로 재시작했으며, r2는 품질 실패가 아니라 time-budget 실패로 분리한다.

2026-04-29 00:30 KST에는 `1100494`의 first no-prior head audit이 완료됐고, MLP는 direct phase 도중 platform termination으로 내려갔다. head audit 자체는 route balanced `0.571399`, subject IoU `0.552048`, proposal target-AR recall@5 `0.953750`, target-AR compatibility `1.0`으로 강했지만 fixed subject-valid threshold `0.30`에서 negative acc `0.590419`가 gate `0.60`에 조금 못 미쳐 deploy candidate는 아니다. 다만 threshold sweep은 `0.35`부터 release gate를 통과했고 recommended threshold는 `0.40`이다. threshold `0.40`에서는 negative acc `0.694611`, positive acc `0.756871`, balanced acc `0.725741`이므로, 새 학습 없이 calibration/config만 바꾼 `1100530` `mcn-q-swinv2b384-headpolish-swinembed-vthr040-gate-r1-20260428`을 생성해 head/direct/qualitative gate를 복구 실행한다.

같은 시각 `1100528` SwinV2-B actionpair r8은 terminal `excluded_train_gate`다. explicit pairwise acc `0.690621`, subject IoU `0.557844`, subject-valid best balanced `0.668327`, proposal recall `0.952235`, action top1 `0.974417`, top-return `0.619839`는 모두 강했지만 raw route balanced가 `0.497973`으로 gate `0.52`를 닫지 못했다. interactive `mcn-q-swinv2b384-topagree-actionroutecalfast-r3-ip134-20260429`도 action top1 `0.987095`, top-return `0.619780`, subject-valid best balanced `0.671923`로 개선됐지만 raw route balanced `0.501237`에 막혀 train gate에서 제외했다. r3 embedded gate 프로세스가 남아 있었지만 train gate 미달 후보는 direct/qual 확장 대상이 아니므로 종료했다. 대신 interactive GPU에는 `mcn-q-swinv2b384-topagree-subjectcal-r1-ip134-20260429`를 올려 route/ranker/action을 거의 건드리지 않고 subject box/valid blocker만 줄일 수 있는지 확인한다.

2026-04-29 00:38 KST에는 남아 있던 두 보정 축도 terminal exclusion으로 닫았다. `1100524` EVA02-L pairvalid r8은 route balanced `0.605625`, subject IoU `0.523736`, proposal recall `0.956499`, action top1 `0.962813`, generated align `0.955625`를 유지했지만 subject-valid best balanced `0.442656`, top-return `0.537500`이 gate에 미달했다. interactive SwinV2-B subjectcal r1은 subject IoU `0.559649`, subject-valid `0.652775`, proposal `0.952505`, action `0.992063`, top-return `0.616548`까지 올렸지만 route `0.509166`과 generated align `0.0`으로 제외한다. 따라서 현재 source-of-truth로 남은 배포 판단 축은 `1100494` SwinV2-B topagree head-polish의 calibrated threshold `0.40` no-prior gate인 `1100530` 하나다. 중복 gate `1100531`은 output collision과 GPU 낭비를 막기 위해 종료했으며, 품질 판정 대상이 아니다.

2026-04-29 01:04 KST에는 `1100530` calibrated gate가 Succeeded로 닫혔다. final runtime 입력은 여전히 `image + target_ar`뿐이고 route expert는 single checkpoint 내부에 embed되어 있으며 external subject prior는 쓰지 않는다. direct 결과는 final positive hit `0.963125`, final IoU-to-best-positive `0.850533`, proposal recall@5 `0.974688`, target-AR compatible `1.0`, subject IoU `0.552048`, subject-valid acc `0.740625`, route acc `0.565313`, risk-hit `0.029688`이다. 다만 qualitative failure pack은 subject_iou_low `29.34%`, subject_valid_mismatch `25.94%`, final_not_positive `3.69%`, risk_hit `2.97%`, proposal_top5_miss `2.53%`라 subject 계열 failure가 아직 반복된다. 따라서 `1100530`은 "배포 확정"이 아니라 provisional full-eval 후보로만 승격하고, public/equal-4 8-shard `1100544`~`1100552`와 GAIC official `1100554`를 병렬 실행했다. 같은 checkpoint의 중복 final-bundle/GAIC Pending `1100551`, `1100553`, `1100555`, `1100556`은 종료했다. 동시에 subjectcal+embedded-route vthr0.30과 actionroutecalfast+embedded-route vthr0.40 direct/qual gate를 별도 비중복 비교 축으로 유지한다.

2026-04-29 01:42 KST에는 `1100530`의 final 판단을 보수적으로 닫았다. GAIC official `1100554`는 top1 MOS `3.641080`, SRCC `0.356275`, Accw4@10 `0.316708`, GAIC primary `0.536310`으로 기존 embedded 후보보다 낮고, qualitative subject_iou_low/subject_valid_mismatch가 반복된다. 따라서 `1100530`은 배포 후보에서 제외하고 public/equal-4 shard는 중단했다. 외부/자동 프로세스가 다시 만든 중복 GAIC/public run `1100567`~`1100570`도 같은 checkpoint/output을 오염시키므로 즉시 종료했다. 이는 "수치가 좋은 direct만 보고 winner를 확정하지 않는다"는 release gate 원칙을 적용한 것이다.

같은 시각 `1100565` public60 rankcal과 `1100566` topagree pairrank도 terminal exclusion으로 닫았다. `1100565`는 route balanced `0.496673`, top-return `0.480902`, generated align `0.0`이고, `1100566`은 explicit pairwise acc `0.752276`까지 올렸지만 route `0.487719`, generated align `0.0`이다. public/UCTR top-id agreement와 pairwise supervision은 ranking head에는 신호를 주지만, route와 generated runtime action을 동시에 회복하지 못했다.

subjectcal+embedded-route vthr0.30 gate는 audit numeric pass에도 불구하고 qualitative taxonomy가 더 나빠 제외했다. top1_not_positive `69.44%`, route_mismatch `43.47%`, subject_valid_mismatch `39.13%`, low_subject_iou `30.31%`는 배포 후보에서 즉시 제외할 수준이다. actionroutecalfast embedded vthr0.40 gate `1100542`는 direct phase에서 GPU util `0%`, D-state 프로세스, output 미생성으로 stuck 판정해 종료했다. 같은 checkpoint의 `NUM_WORKERS=0` replacement `1100571`도 audit numeric은 route balanced `0.571399`, subject-valid balanced `0.725741`로 통과했지만, direct가 `600 / 3,200`장 처리 후 platform termination으로 끝났고 audit top1_not_positive `69.13%`, route_mismatch `43.47%`, subject_valid_mismatch `25.94%`, low_subject_iou `29.34%`가 반복되어 배포 후보에서 제외한다.

남는 GPU는 단순 반복 학습이 아니라 품질 우선 EVA02-L/448 runtime 재평가에 투입했다. `1100575` calaction은 route/subject geometry가 가장 강한 축, `1100576` pairvalid는 action/proposal 보존 축, `1100574` actionfix는 top-agree route/subject 전이 축이다. vthr0.40 결과는 모두 제외다. `1100574`는 direct final hit `0.961563`, final IoU `0.822973`, subject IoU `0.685507`, route acc `0.614375`였지만 subject-valid negative acc `0.447904`와 qualitative subject_valid_mismatch `24.28%`가 blocker다. `1100575`는 final hit `0.962188`, final IoU `0.823037`, route acc `0.603125`였지만 subject-valid negative acc `0.444311`, qualitative subject_valid_mismatch `24.91%`가 남았다. `1100576`은 audit subject-valid negative acc `0.388024`, top1_not_positive `72.06%`라 direct를 중단했다.

2026-04-29 01:55 KST에는 사용자 요청의 leaderboard 공백 채움과 release candidate 판단을 분리했다. `1100530`은 GAIC official `1100554`와 qualitative subject blocker 때문에 배포 후보에서 제외 상태를 유지하지만, Unified Final Leaderboard의 FCDB/CPC/GNMC/equal-4 공백을 확인하기 위해 public/equal-4 export는 계속한다. MLP shard01~03 r2 `1100577`~`1100579`는 start time 없이 `The job is not executed.`로 종료되어 플랫폼 실행 실패로 분류했고, 같은 실패 workload 반복은 중단했다. interactive GPU `10.2.4.134`의 shard00은 `3,328 / 약 15,628` task row를 생성 중이며, shard00 완료 후 shard01~03을 순차 실행해 merge/eval source-of-truth를 만든다.

2026-04-29 02:24 KST에는 vthr0.40의 실패 원인을 threshold calibration으로 분리하기 위해 `SUBJECT_VALID_THRESHOLD=0.80`, `SUBJECT_VALID_POLICY=route_and_conf`만 바꾼 `1100587` EVA02-L actionfix와 `1100588` EVA02-L calaction no-prior gate를 생성했고 둘 다 Running 전환을 확인했다. 이후 2026-04-29 03:10 KST에 두 run은 모두 Succeeded로 닫혔지만 release gate에서 제외했다. actionfix는 final hit `0.961563`, final IoU `0.822973`, subject valid pos/neg `0.642706/0.766467`, qualitative subject_valid_mismatch `32.56%`이고, calaction은 final hit `0.962188`, final IoU `0.823037`, subject valid pos/neg `0.642706/0.767665`, qualitative subject_valid_mismatch `32.66%`다. threshold-only 보정은 negative acc를 회복하는 대신 positive를 잃어 mismatch를 더 키우므로 해법이 아니다.

2026-04-29 02:40 KST에는 사용자 추가 지시인 "온디바이스 제약 완화 품질 우선 모델군"을 더 구체화해 실제 병렬 학습으로 확장했다. `1100592` `mcn-q-swinv2l384-topagree-pairroute-r1-20260429`는 SwinV2-L/384 pretrained, `1100593` `mcn-q-eva02l448-topagree-pairaction-r1-20260429`는 EVA02-L/448 `mim_m38m_ft_in22k_in1k`, `1100594` `mcn-q-cnv2h512-topagree-routevalid-r1-20260429`는 ConvNeXtV2-Huge/512 `fcmae_ft_in22k_in1k_512` pretrained를 사용한다. 세 run 모두 `blended_uctr80_public20_topagree_labels`와 explicit teacher pairwise label을 함께 쓰며, route distill, subject-valid negative calibration, action/top-return/policy alignment를 동시에 강화한다. 모두 Running 전환을 확인했고, first validation이 route/subject/proposal/action/top-return train gate를 통과하지 못하면 public/GAIC 확장 없이 제외한다.

2026-04-29 02:43 KST에는 위 병렬 축에 더해 `1100595` `mcn-q-swinv2l512-vact-smoke-r1-20260429`도 생성했다. 이는 `quality_swinv2l_512` profile의 bounded smoke였지만, timm SwinV2-Large pretrained alias가 기본 `img_size=384`로 생성되어 `Input height (512) doesn't match model (384)` assertion으로 실패했다. 이 결과는 모델 품질 실패가 아니라 profile 구현 오류다. `MobileCropNetV4Config`와 checkpoint `model_config`에 `input_size`를 추가하고, Swin/Vision Transformer 계열 timm 생성 시 `img_size`를 전달하며 Swin 계열에는 `strict_img_size=False`를 적용했다. 로컬 `py_compile`, `bash -n`, `tests/test_mobilecropnet_v4.py -q`, profile import 검증을 통과하고 원격 shared storage에 동기화했다. corrected smoke는 `1100597`로 재생성해 Running 전환을 확인했다.

2026-04-29 03:10 KST에는 `1100590` actionroutecalfast direct/qual 보강도 Succeeded로 닫혔다. final hit `0.963210`, final IoU `0.850552`, proposal recall@5 `0.973519`, target-AR compatibility `1.0`은 `1100530`과 같은 수준이지만 subject_iou_low `27.96%`, subject_valid_mismatch `27.45%`, risk_hit `3.19%`가 남아 배포 후보에서 제외한다. 동시에 v0.80 threshold-only 실패를 valid head 자체 문제로 분리하기 위해 `1100598` `mcn-q-eva02l448-validonly-cal-r1-20260429`를 생성했다. 이 run은 EVA02-L calaction checkpoint에서 `subject_valid_head`, `subject_spatial_valid_head`, `subject_refine_valid_head`, `subject_spatial_mix_head`만 trainable로 열고 geometry/action/proposal을 고정해 valid calibration만 평가한다.

동시에 `1100530`의 public/equal-4 공백 채움은 release candidate 판단과 분리해 interactive 16-shard source-of-truth로만 진행한다. 2026-04-29 05:12 KST 기준 shard00~shard10은 각각 `3,907` rows로 완료됐고 shard11은 `2,880` rows 이상을 생성 중이다. 이 수치는 Unified Final Leaderboard completeness를 위한 것이며, `1100530`의 GAIC/qualitative blocker를 뒤집는 근거가 아니다.

2026-04-29 03:23 KST에는 corrected SwinV2-L/512 smoke `1100597`을 terminal 판정했다. `quality_swinv2l_512` profile 구현 오류는 해결됐지만, 배포 품질은 통과하지 못했다. head audit은 route balanced `0.541922`, subject IoU `0.538981`를 보였으나 top1_hit가 `0.134766`에 그쳤고, direct eval도 final hit `0.929688`, final IoU-to-best-positive `0.730682`, proposal recall@5 `0.929688`로 기존 후보보다 약하다. qualitative pack은 subject_valid_mismatch `27.54%`, subject_iou_low `24.02%`, final_not_positive `7.03%`, proposal_top5_miss `7.03%`라서 full training으로 확장하지 않는다.

같은 시각 `1100598` valid-head-only calibration은 terminal `excluded_train_gate`로 닫았다. route balanced `0.631456`, subject IoU `0.566836`, proposal recall `0.939521`, action top1 `0.969792`는 유지됐지만 subject-valid best balanced가 `0.545000 < 0.58`이고 top-return `0.490938`, generated align `0.0`이라 valid head 단독 보정은 해법이 아니다. 사용자 신규 지시의 품질 우선 모델군을 더 넓히기 위해 `1100599` DINOv2-L/518 pretrained top-agree route/action smoke를 생성했다. 이전 DINOv2-L full model 실패와 중복되지 않도록 train `2,048` rows, val `512` rows, explicit pairwise/action/top-return/subject-valid 동시 gate를 둔 bounded smoke다. `1100597` 종료로 회수된 GPU에는 `1100600` `mcn-q-eva02l448-validaction-cal-r1-20260429`를 추가했고 Running 전환을 확인했다. 이 run은 EVA02-L calaction checkpoint에서 subject geometry/proposal을 고정하고 subject-valid, ranker, policy/action, checklist/why head만 열어 threshold-only 실패가 valid head 단독 문제인지 action/ranker 공동 calibration 문제인지 분리한다.

2026-04-29 03:48 KST에는 `1100600`과 `1100601`을 terminal gate에서 제외하고, 실패 원인별 후속 축으로 즉시 전환했다. `1100600`은 route `0.613858`, subject IoU `0.664631`, subject-valid balanced `0.609726`, proposal recall `0.937925`로 geometry/valid는 닫았지만 action top1 `0.925603 < 0.955`, top-return `0.563175 < 0.58`이 부족했다. 따라서 회수 GPU에는 `1100603` `mcn-q-eva02l448-actiontop-cal-r2-20260429`를 생성했다. 이 run은 `1100600/best.pt`를 init으로 쓰고 route/subject/proposal을 고정한 채 ranker, utility/positive/risk, policy/action, checklist/why head만 열어 action/top-return alignment만 보정한다.

`1100601`은 SwinV2-B/384 head-polish init에서 subject/action을 joint 보정한 결과 subject IoU `0.559203`, subject-valid balanced `0.663033`, proposal recall `0.952295`, action top1 `0.970458`, top-return `0.619143`은 gate를 넘겼지만 route balanced `0.496454 < 0.52`로 제외했다. 이 실패는 valid/action이 아니라 raw route internalization 문제이므로, 회수 GPU에는 `1100602` `mcn-q-swinv2b384-subjectaction-routecal-r2-20260429`를 생성했다. 이 run은 `1100601/best.pt`에서 route head 및 route image-residual head만 열고 multi-expert distill과 route-balanced sampler를 강화한다. 두 후속 run은 기존 실패를 그대로 반복하지 않는 head-scope 분리 실험이며, train gate 통과 전에는 direct/GAIC/public으로 승격하지 않는다.

같은 시각 `1100599` DINOv2-L/518 top-agree route/action smoke도 내부 train gate에서 제외했다. best epoch1은 route balanced `0.431641`, subject IoU `0.454616`, subject-valid best balanced `0.505859`, proposal recall `0.898270`, action top1 `0.802734`, top-return `0.539063`이다. 이는 이전 DINOv2-L full model의 route/subject 붕괴와 같은 방향이므로, DINOv2-L 대형 backbone 자체를 반복하는 후속 run은 만들지 않는다. 플랫폼 run 상태가 완전히 Succeeded로 내려가면 회수 GPU는 Swin/EVA head-scope 보정 또는 qualitative failure root-cause 평가처럼 다른 blocker를 겨냥하는 비중복 축에만 투입한다.

`1100599`가 MLP상 Succeeded로 내려간 뒤에는 `1100607` `mcn-q-swinv2b384-subjectaction-routecal-nres-r3-20260429`을 생성했고 Running 전환을 확인했다. 이 run은 `1100602`와 같은 `1100601/best.pt`에서 시작하지만 route image-residual head를 추가하지 않고 기존 `route_head`, `route_kind_head`, `route_cardinality_head`만 학습한다. 목적은 `1100602`의 랜덤 초기화 residual head가 route recovery에 실제로 필요한지, 또는 기존 subject/action head를 흔들 위험만 키우는지 분리하는 것이다.

2026-04-29 04:04 KST에는 `1100603` action/top-return-only 보정도 terminal `excluded_train_gate`로 닫았다. route `0.620923`, subject IoU `0.664631`, subject-valid balanced `0.609726`, proposal recall `0.937925`는 유지됐지만 action top1 `0.921446 < 0.955`, top-return `0.567332 < 0.60`, generated_align_top1 `0.0`으로 action-only head 보정은 해법이 아니었다. 원인은 ranker/policy head만의 문제가 아니라 generated proposal/action coupling 쪽으로 좁혀졌으므로, 회수 GPU에는 `1100609` `mcn-q-eva02l448-genprop-actiontop-r3-20260429`를 생성했다. 이 run은 `1100600/best.pt`에서 proposal head와 ranker/policy/action/checklist/why head를 함께 열어 generated proposal scoring과 final crop action 정합성을 동시에 보정한다.

2026-04-29 04:10 KST에는 `1100592` SwinV2-L/384 pairroute full run을 조기 제외했다. epoch1 validation은 route `0.615312`, subject IoU `0.670447`, subject-valid best balanced `0.615960`, generated_align_top1 `0.937656`으로 route/subject internalization은 강했지만, proposal recall `0.913165 < 0.93`, action top1 `0.852660 < 0.96`, top-return `0.444098 < 0.58`이 release gate와 멀었다. epoch2는 같은 pairroute 목적의 GPU 시간을 더 쓰기보다 중단했고, best checkpoint는 `1100610` `mcn-q-swinv2l384-genprop-actiontop-r2-20260429`와 fast ablation `1100611` `mcn-q-swinv2l384-routeanchor-actionprop-r2-20260429`로 이어 넘겼다. 두 run은 SwinV2-L의 강한 route/subject를 유지한 채 proposal/ranker/policy/action head만 보정해 final crop action surface가 회복되는지 검증한다.

2026-04-29 04:18 KST에는 `1100601`의 특수 실패 형태를 별도 gate로 분리했다. `1100601`은 route balanced `0.496454`만 미달했고 subject IoU `0.559203`, subject-valid best balanced `0.663033`, proposal recall `0.952295`, action top1 `0.970458`, top-return `0.619143`, generated_align_top1 `0.943750`은 모두 train gate를 넘겼다. 따라서 새 학습 반복 대신 checkpoint 내부에 image-only SwinV2-B route expert payload를 embed한 single-checkpoint no-prior gate `1100612` `mcn-q-swinv2b384-subjectaction-swinembed-gate-r1-20260429`를 생성했다. 이 축은 `1100602`/`1100607` route-head 재학습과 중복이 아니라, 최종 입력 `image + target_ar`를 유지하면서 route-only blocker를 내부 branch로 닫을 수 있는지 확인하는 bounded evaluation이다. Pending 10분을 넘으면 종료/재생성하고, head audit/Product-AR direct/qualitative pack이 통과하기 전에는 배포 후보로 올리지 않는다.

`1100612`와 병렬로 `1100615` `mcn-q-swinv2b384-subjectaction-routeens-gate-r1-20260429`도 생성했다. 이 run은 같은 `1100601` strong action/proposal checkpoint를 쓰되, SwinV2-B 단일 route expert가 아니라 SwinV2-B/CNV2L/EVA02-L route expert ensemble을 checkpoint 내부 payload로 embed한다. 이는 route expert 단일/ensemble 선택이 qualitative route mismatch와 subject-valid calibration에 미치는 영향을 분리하는 no-prior bounded gate이며, 외부 teacher subject prior를 요구하지 않는다.

같은 시각 `1100593` EVA02-L/448 pairaction도 조기 제외했다. epoch1은 route `0.615936`, subject IoU `0.640777`, subject-valid best balanced `0.594555`였지만 proposal `0.906293`, action `0.710931`, top-return `0.364921`로 action surface가 더 멀어 epoch2를 중단했다. `1100602` route-residual과 `1100607` no-residual route-only는 terminal summary 없이 종료/중단되어 품질 실패가 아니라 실행 예산 실패로 기록했고, 같은 목적은 더 작은 train/val budget의 `1100613` route-residual smoke와 `1100614` no-residual route smoke로 대체했다.

`1100611` `mcn-q-swinv2l384-routeanchor-actionprop-r2` fast ablation은 train gate에서 제외했다. route `0.606016`, subject IoU `0.669981`, subject-valid best balanced `0.616250`, generated_align_top1 `0.935000`은 유지됐지만 proposal recall `0.917218 < 0.93`, action top1 `0.872292 < 0.955`, top-return `0.451250 < 0.58`이 미달했다. 이는 `1100592` SwinV2-L route/subject 신호가 강해도 proposal/action/top-return surface가 자동으로 회복되지 않음을 다시 보여준다. 같은 계열의 larger-budget `1100610` 결과는 아래 terminal 판정에 별도로 반영한다.

2026-04-29 04:28 KST에는 남은 quality-first terminal 결과를 추가 반영했다. `1100594` ConvNeXtV2-Huge/512 routevalid run은 epoch1 후 조기 제외했다. route `0.490637`, subject IoU `0.492708`, subject-valid best balanced `0.534956`, proposal recall `0.917163`, action top1 `0.725968`, top-return `0.436330`으로, 더 큰 backbone과 512 입력 크기만으로는 route/subject/action blocker가 닫히지 않았다. 이 축은 같은 학습 구조로 반복하지 않는다.

`1100610` `mcn-q-swinv2l384-genprop-actiontop-r2`도 terminal train gate에서 제외했다. route `0.606998`, subject IoU `0.670447`, subject-valid best balanced `0.615960`, generated_align_top1 `0.938279`는 `1100592`의 강한 route/subject internalization을 유지했지만, proposal recall `0.919501 < 0.93`, action top1 `0.843932 < 0.955`, top-return `0.475686 < 0.58`이 미달했다. 따라서 SwinV2-L 계열은 "route/subject는 강하지만 generated proposal과 final action surface가 약하다"는 원인으로 닫고, 새 구조 변화 없이 같은 action-proposal 보정 run을 추가하지 않는다.

`1100612` single Swin route-expert embedded gate는 head audit 후 중단했다. checkpoint 내부 payload embed 자체는 성공했지만, route balanced `0.554562`, subject IoU `0.568525`, top1 hit `0.364147`로 release head gate와 멀다. direct/qualitative까지 계속 돌려도 배포 후보가 될 수 없는 상태라 GPU를 회수했고, candidate decision은 `artifacts/mobilecropnet_v4/quality_first_20260428/gate_runs/mcn-q-swinv2b384-subjectaction-cal-r1-swinembed-gate-r1-20260429-excluded-head-audit/candidate_decision.json`에 기록했다.

route ensemble embed gate `1100615`도 head audit에서 제외했다. route balanced accuracy는 `0.582310`으로 single Swin embed보다 개선됐지만 subject IoU `0.568525`, top1 hit `0.364147`, top1_not_positive `63.59%`, route_mismatch `43.85%`, subject_valid_mismatch `46.66%`가 남아 배포 후보 조건을 만족하지 못했다. 최종 입력은 여전히 `image + target_ar`였고 external teacher subject prior는 쓰지 않았으나, route expert payload embed만으로 subject/top1/정성 failure를 닫지 못한다는 근거로 보존한다.

`1100609` EVA02-L generated proposal/action coupling도 train gate에서 제외했다. route `0.625289`, subject IoU `0.664631`, subject-valid best balanced `0.609726`, proposal recall `0.938929`, generated_align_top1 `0.947631`, explicit pairwise `0.804466`은 강하지만 action top1 `0.917290 < 0.955`, top-return `0.560889 < 0.58`이 남았다. 이 결과는 generated proposal head를 함께 열면 `generated_align_top1=0.0` 문제는 해결되지만, final action/policy/top-return head가 여전히 teacher action surface를 따라가지 못한다는 분리 증거다.

`1100616`은 `1100612`와 같은 single Swin embedded checkpoint에 subject-valid threshold만 `0.30`으로 바꾼 rerun이라 중복 GPU workload로 중단했다. 이미 `1100612`에서 route/subject/top1 head audit이 미달했으므로 threshold-only direct/qual rerun은 release gate를 닫을 수 없다. 중단 decision은 `artifacts/mobilecropnet_v4/quality_first_20260428/gate_runs/mcn-q-swinv2b384-subjectaction-cal-r1-swinembed-vthr030-gate-r1-20260429-stopped-duplicate/candidate_decision.json`에 남겼다.

같은 threshold-only retry `1100617`과 재등록 `1100620`도 start time 없이 Terminated되어 usable artifact가 없다. 단일 Swin route embed checkpoint의 vthr0.30 sweep은 `1100612`의 route/subject/top1 head blocker를 해결하지 못하므로 추가 재등록을 보류한다. 반대로 `1100618`/`1100619`는 `1100609` 실패를 바탕으로 한 비중복 action-surface 후속이었지만, route head를 학습하지 않는 구성에서 default route-expert distill을 그대로 켜 큰 expert load/compute를 유발하는 비효율 config가 확인되어 중단했다. corrected r5 `1100622`는 proposal logit/token scoring과 ranker-policy-action head를 열고, `1100621`은 proposal을 고정한 채 top-return, top-k coverage, teacher distill, action consistency 손실을 강화했다. 두 r5 모두 `ROUTE_EXPERT_DISTILL_WEIGHT=0`으로 재생성해 정상 종료했지만, `1100621`은 route `0.633929`, subject IoU `0.664504`, proposal `0.938777`을 통과하고도 action `0.950417`, top-return `0.507708`로 제외됐고, `1100622`도 route `0.629765`, subject IoU `0.664504`, proposal `0.938777`을 통과했지만 action `0.950208`, top-return `0.513750`으로 제외됐다. 따라서 EVA02-L의 남은 병목은 subject/route가 아니라 final action/top-return surface다.

`1100615` route ensemble checkpoint의 threshold-only v0.30 재시도는 중단한다. `1100623`, `1100624`, `1100626`은 모두 같은 routeens checkpoint의 subject-valid threshold만 바꾸는 중복 gate로 생성됐고, `1100615` head audit에서 이미 subject IoU `0.568525`, top1 hit `0.364147`, top1_not_positive `63.59%`, route_mismatch `43.85%`, subject_valid_mismatch `46.66%` blocker가 확인되어 release gate를 닫을 수 없다. 따라서 세 run은 GPU 낭비를 막기 위해 즉시 회수했고, 추가 재등록하지 않는다.

EVA02-L 후속은 `1100621/best.pt`를 초기값으로 쓰는 `1100625` `mcn-q-eva02l448-topreturn-rescue-r6-20260429`로 확장했다. subject/route/proposal geometry는 r5에서 이미 gate를 넘었으므로, r6는 proposal logit/token, set-ranker, utility/positive/risk, policy/action, checklist/why head만 열고 top-return, action consistency, generated proposal positive margin을 강화한다. subject-valid threshold는 r5의 best threshold `0.458996`에 맞춰 `0.46`으로 보정해 direct gate가 임의 threshold 때문에 실패하지 않도록 했다. terminal summary 전에는 배포 후보가 아니다.

추가로 같은 loss-weight 증량 반복을 피하기 위해 구조 패치 `route_condition_candidate_scores`를 추가했다. 이 옵션은 내부 route logits softmax를 zero-init projection으로 candidate token에 더한 뒤 utility/positive/risk/checklist head와 policy를 재계산한다. projection 마지막 linear를 0으로 초기화했기 때문에 기존 checkpoint 로딩 직후의 score surface는 보존되고, 학습 중 route-conditioned residual만 배운다. 이를 검증하는 병렬 run은 `1100627` `eva02l-routecond-rank-r1`, `1100628` `eva02l-routecond-joint-r1`, `1100629` `eva02l-routecond-prop-r1`이다. 각각 rank/policy만, route+rank/policy joint, proposal scoring까지 여는 비중복 축이며, 모두 `1100609/best.pt`에서 시작한다.

2026-04-29 05:20 KST에는 route-conditioned 구조의 학습/runtime 정합성을 추가로 보정했다. 기존 패치는 label candidate bank의 utility surface와 no-prior runtime path에는 route context를 주입했지만, 학습 중 `return_generated_scores=True`로 계산하는 generated proposal alignment surface에는 같은 route context가 적용되지 않았다. 따라서 `src/mobilecropnet_v4/model.py`를 패치해 generated proposal token에도 route logits softmax projection을 더한 뒤 utility/positive/risk head를 재계산하도록 했다. 이 변경은 `generated_align` selection signal과 실제 generated-proposal runtime selection surface의 괴리를 줄이기 위한 구조 보정이며, 로컬 `py_compile`, 원격 `py_compile`, 로컬 `tests/test_mobilecropnet_v4.py -q` 10개를 통과했다. 이미 실행 중인 `1100625`/`1100627`/`1100628`/`1100629`는 terminal artifact를 source of truth로 판정하고, 통과하지 못하면 후속 rerun은 이 패치를 포함한 코드 기준으로만 생성한다.

2026-04-29 05:30 KST에는 final crop action/top-return 병목을 더 직접 겨냥하기 위해 `return_score_head`를 구현했다. 이 head는 candidate utility/checklist/risk score를 바꾸지 않고 final crop 선택용 `return_logits`만 utility logit에 zero-init residual로 더한다. 기존 checkpoint 로딩 직후에는 `return_logits == utility_logits`이므로 이전 모델의 public/GAIC score surface는 보존되고, 학습 gradient는 top-return/action/generated-proposal alignment로만 return head를 보정한다. `src/scripts/infer_mobilecropnet_v4.py`와 `evaluate_mobilecropnet_v4_product_ar_direct.py`는 `return_logits`가 있으면 runtime generated proposal 선택에 이를 사용하도록 패치했다. 로컬/원격 `py_compile`, `bash -n`, 로컬 v4 테스트 10개를 통과했고 shared remote project에 동기화했다. `1100625`/`1100627`/`1100628`/`1100629`의 1epoch val은 route/subject/proposal은 유지했지만 action `0.9329~0.9501`, top-return `0.5133~0.5495`라 gate 미달이므로, 비중복 후속 `1100633` return-head only, `1100634` route-conditioned return-head, `1100635` r5-init return-head를 추가 병렬 GPU workload로 생성했다.

2026-04-29 05:48 KST 기준 `1100633`/`1100634`/`1100635`의 1epoch validation도 action/top-return gate를 닫지 못했다. 각각 crop action `0.929135`/`0.926642`/`0.913342`, top-return `0.552993`/`0.557357`/`0.546966`으로, 선형 return delta를 top-return/action consistency loss만으로 학습하면 baseline/crop action source와 positive/risk surface가 충분히 분리되지 않는다. 이에 따라 추가한 후속 패치는 두 축이다. 첫째, `return_logits` 전용 `return_positive_weight`, `return_risk_suppression_weight`, `return_risk_suppression_margin` loss를 추가해 final crop selection logit만 positive crop과 risk crop 기준으로 직접 보정한다. 둘째, `return_score_head_depth`, `return_score_head_hidden_mult`, `return_score_action_source_bias`를 추가해 depth `2+`에서는 마지막 linear를 zero-init한 MLP residual을 쓰고, crop/base action-source bias도 zero-init residual로 학습한다. 이 구조는 `utility_head`의 public/GAIC score calibration을 건드리지 않고 final-return score만 분리한다. 로컬 `py_compile`, `bash -n`, `tests/test_mobilecropnet_v4.py -q` 10개 및 원격 `py_compile`/`bash -n`을 통과했고 shared remote project에 동기화했다. 비중복 후속은 `1100636` `returnpos-r1`, `1100637` `returnpos-routecond-r1`, `1100638` `returnpos-r6init-r1`, `1100639` `returnab-routecond-r1`, `1100640` `returnab-rank-r1`이며 모두 Running 전환을 확인했다. train gate 통과 전에는 direct/GAIC/public 승격 대상이 아니다.

2026-04-29 06:03 KST에는 `1100633`/`1100634`/`1100635`가 모두 terminal `excluded_train_gate`로 닫혔다. 셋 모두 route/subject/proposal은 보존했지만 action top1 `0.918121~0.929135`, top-return `0.545719~0.557357`이라 linear return-score family는 종료한다. 진행 중인 `1100637`/`1100638`/`1100639`/`1100640` 1epoch 결과도 action `0.913549~0.940150`, top-return `0.545511~0.556525`라 아직 gate와 멀다. 이에 따라 같은 loss 증량이 아니라 decision head의 crop-vs-baseline margin을 final-return score에 동적으로 주입하는 `return_score_decision_source_bias`를 추가했다. 이 scale은 zero-init trainable parameter이므로 checkpoint load 직후 출력은 변하지 않고, 학습 중에만 crop candidate와 baseline candidate에 반대 방향 source margin을 준다. 로컬/원격 `py_compile`, `bash -n`, v4 테스트 10개를 통과했고 shared remote project에 동기화했다. 비중복 후속 workload는 `1100644` `returndec-policy-r1`과 `1100645` `returndec-routecond-r1`이며, 기존 active `1100642` source-bias 상한 run과 `1100643` strong positive/risk run도 같이 추적한다. `1100644`/`1100645`가 Pending 10분을 넘으면 중단 후 재생성한다.

2026-04-29 06:18 KST에는 action-surface 계열을 다시 정리했다. `1100636`/`1100637`/`1100638` return-positive/risk, `1100639`/`1100640` action-source-aware return MLP, `1100642` source-bias 상한 run은 모두 terminal `excluded_train_gate`다. 특히 `1100642`는 crop/base action source는 `0.977348`까지 맞췄지만 top-return이 `0.557564`로 gate `0.58`에 못 미쳤고, generated return surface가 `0.0`이라 단순 source bias가 정확한 returned crop 선택을 보장하지 못함을 확인했다. `1100644`/`1100645` decision-source-bias 계열은 Pending 없이 Running으로 전환됐고 epoch1 기준 action은 각각 `0.991895`/`0.950956`이지만 top-return은 `0.549044`/`0.550083`에 머물렀다. 따라서 다음 구조는 같은 loss 반복이 아니라 candidate token, policy token, route/decision logits, box geometry, utility/positive/risk score를 결합해 final `return_logits`만 보정하는 `return_score_context_head`로 확장한다. 이 head는 기존 코드에 일부 연결되어 있었지만 policy token 차원을 `token_dim`으로 잘못 가정해 테스트에서 `mat1/mat2` shape 오류가 났고, 이를 `token_dim // 2`로 수정했다. 로컬 `py_compile`, wrapper `bash -n`, v4 테스트 10개, 원격 `py_compile`/`bash -n`을 통과했고, context-head 비중복 workload `1100646` detach, `1100647` route-conditioned, `1100648` policy-coupled를 생성해 Running 전환을 확인했다.

2026-04-29 06:22 KST에는 후발 duplicate 정리를 수행했다. `1100649`가 `1100648`과 같은 `RUN_NAME=mcn-q-eva02l448-returnctx-policy-r1-20260429`로 Running 전환되어 같은 output directory를 쓸 위험이 확인됐으므로 `1100649`를 즉시 종료했다. 이미 `1100648` 산출물도 약 2분간 같은 directory에서 오염됐을 가능성이 있어 `1100648`도 source-of-truth에서 제외하고 종료했다. 같은 policy-coupled 가설은 고유 output path의 clean replacement `1100652` `returnctx-policy-r2`로 재생성했다. 추가로 active 목록에 `1100650` `returnctx-routecond-r1`, `1100651` `returnctx-source-r1`가 확인됐고, 두 run은 각각 route-conditioned context head와 source/geometry-only context head라 output 충돌이 없는 비중복 변형으로 유지했다. 이후 `1100651`은 train.log 0 byte 상태에서 shell이 training 시작 전 종료되어 제외했고, 같은 source/context-only 가설은 clean replacement `1100654` `returnctx-source-r2`로 재생성했다. 플랫폼이 같은 이름의 duplicate pending `1100653`도 함께 등록했지만 시작 전 즉시 종료해 output 오염은 없다.

같은 시각 `headpolish-swinembed-vthr040`의 public/equal-4 completeness 평가도 완료했다. interactive GPU `10.2.4.134`의 16개 shard는 shard00~14 각 `3,907` row, shard15 `3,906` row로 총 `62,511` prediction을 생성했고 merge duplicate는 `0`이다. SQLite public benchmark 평가는 FCDB IoU `0.728326`, CPC weighted `0.781468`, GNMC IoU `0.724369`, overall IoU `0.716813`, overall weighted pairwise `0.781468`이다. 이 값은 final leaderboard 공백을 채우기 위한 수치이며, 해당 checkpoint는 GAIC primary `0.536310`과 qualitative subject blocker 때문에 배포 후보에서 계속 제외한다.

같은 시각 final action/top-return 병목을 분리하기 위해 runtime scoring 경로도 패치했다. 기존 `utility_head`는 public/teacher score calibration과 listwise ranking을 동시에 떠안고 있었고, 제품 executor의 final crop action score까지 같은 logit으로 결정했다. 새 `return_score_delta_head`를 실제 후속 실험에서 사용할 수 있도록 Product-AR direct, public export, GAIC official, generic batch prediction 경로를 `return_logits` 우선으로 정렬했다. return head가 없는 checkpoint에서는 `_return_logits()`가 `utility_logits`를 그대로 반환하므로 과거 산출물과 기존 public/eval 수치는 변하지 않는다. 현재 return-head/return-positive/action-source-aware MLP 계열이 train gate를 닫지 못하면 다음 병렬 workload는 같은 loss 반복이 아니라 action decoder 또는 generated-proposal return selection 구조를 별도 head로 확장하는 쪽으로 넘어간다.

`1100613` route-residual smoke와 `1100614` no-residual route smoke는 모두 train gate에서 제외했다. residual은 route `0.455943`, subject IoU `0.556198`, subject-valid best balanced `0.654476`, proposal `0.955893`, action `0.959548`, top-return `0.611393`이고, no-residual은 route `0.472882`, subject IoU `0.556198`, subject-valid best balanced `0.654476`, proposal `0.955893`, action `0.959548`, top-return `0.611393`이다. action/top-return은 기존 `1100601` 강점을 어느 정도 보존했지만 route가 더 나빠졌고 generated_align_top1도 `0.0`이라 route-only calibration은 해법이 아니다. 추가로 생긴 threshold-only retry `1100620`도 start time 없이 Terminated되어 산출물이 없다.

## 21. 결론

MobileCropNet v4.0은 단일 bbox regressor를 넘어, 학습 데이터 계약, learned proposal, relation-aware ranker, route/policy/subject/explanation heads, no-prior runtime executor, multi-surface evaluation, strict release gate를 갖춘 제품형 cropper system으로 구현되었다. 현재 체계는 연구/제품 양쪽에서 실패를 분해해 볼 수 있을 만큼 충분히 구체적이다.

그러나 2026-04-29 09시대 기준 deploy candidate는 아직 확정되지 않았다. 상위 student row는 public/replay/direct crop alignment에서 강한 신호를 보이지만, route collapse, subject-box calibration, generated proposal alignment, policy/action consistency, qualitative catastrophic bucket이 release gate를 막는다. routeens embedded checkpoint, topagree Swin route embedded checkpoint, fullwarm Swin route embedded relation/head gate는 public/GAIC/head/direct/qual 일부 수치를 확보했지만 CPC 또는 GAIC 하락, direct final hit/IoU 열세, subject_iou_low와 subject_valid_mismatch 반복 때문에 제외했다. `1100530`은 public 공백 수치까지 채웠지만 GAIC official과 qualitative subject blocker 때문에 배포 후보에서 제외했고, `1100571`/`1100590`은 actionroutecalfast direct 보강 후에도 subject_iou_low와 subject_valid_mismatch가 반복되어 닫았다. `1100574`/`1100575`/`1100576` EVA02-L vthr0.40 및 `1100587`/`1100588` vthr0.80도 subject-valid calibration 및 qualitative blocker 때문에 제외했다. `1100621` 이후 `1100730`까지의 action-decoder/return-selector 계열은 route/subject/proposal 일부를 닫아도 action/top-return을 동시에 닫지 못했다. 라벨 진단에서 UCTR-heavy train row의 raw score top이 baseline인데 decision target은 crop인 충돌이 확인되어, 현재 판단 축은 `RETURN_TARGET_MODE=decision_source`를 켠 `1100746`/`1100747`/`1100748`/`1100749`로 이동했다. 과학적 병목은 더 강한 offline teacher 자체가 아니라, teacher가 제공한 route/subject/policy/rationale 구조를 no-prior student runtime에 안정적으로 internalize하면서 generated proposal과 final crop action까지 일관되게 묶는 것이다.

다음 milestone은 최종 모델 선언이 아니라, route balanced accuracy, subject IoU, generated proposal alignment, policy calibration, qualitative mini-pack을 동시에 통과하는 bounded smoke와 full public/GAIC 확장 평가다. public/equal-4 completeness 평가는 완료됐으므로 더 이상 `1100530` 평가 공백은 없다. 현재 action decoder 또는 top-k selector 계열이 top-return/action gate를 닫으면 같은 checkpoint/config로 internal no-prior head audit, Product-AR direct eval, qualitative review pack을 먼저 수행하고, 그 뒤 equal-4/GAIC official/latency를 수행한다. 이 계열도 실패하면 다음 후속은 public/UCTR consensus-disagreement curriculum과 subject-valid calibration을 action decoder 학습 batch에 직접 반영하는 구조로 넘어간다. 새 품질 우선 모델군은 속도보다 정량/정성 품질을 먼저 보는 실험으로 진행하되, deploy candidate 선언은 여전히 같은 checkpoint/config가 internal no-prior head audit, Product-AR direct eval, equal-4, GAIC official, qualitative review pack을 모두 통과한 뒤에만 가능하다.

2026-04-29 10:20 KST 기준으로 위 판단을 더 엄격히 갱신한다. decision-source target corrected 축 `1100746`/`1100747`/`1100748`/`1100749`와 label-surface control `1100752`/`1100753`은 모두 제외했고, 기존 wrapper의 pass처럼 보였던 row도 raw top-return과 generated proposal alignment를 release gate에 다시 넣어 correction decision을 남겼다. 현재 판단 축은 `1100785`, `1100787`, interactive raw0.35 setonly, DINOv2-L/518 smoke `1100788`이며, 이들도 같은 checkpoint/config가 no-prior head audit, Product-AR direct, equal-4, GAIC official, latency, qualitative review pack을 모두 통과하기 전에는 배포 후보가 아니다.

2026-04-29 11:12 KST 기준으로 다시 갱신한다. `1100785`/`1100787`/interactive raw0.35와 public60 후속 평가는 배포 후보에서 제외했고, source of truth는 product-consistent label repair 이후의 `mcn-q-eva02l448-pc-gap006-policyreturnfast-ip134-r2-20260429`, `1100850`, `1100872`, `1100873` 네 축이다. 첫 fast validation은 route/subject/proposal/generated align을 보존했지만 baseline-preserve, crop-action, top-return이 아직 gate와 멀다. 따라서 최종 후보 선언은 보류하며, action-source와 return target을 동시에 닫는 checkpoint가 나올 때만 no-prior/direct/qualitative/public/GAIC/latency로 확장한다.

2026-04-29 06:39 KST에는 context-head 첫 terminal 결과와 후속 구조 패치를 반영했다. `1100646` context-detach는 route `0.625289`, subject IoU `0.664631`, subject-valid best balanced `0.609726`, proposal `0.938929`를 유지했지만 action `0.946384 < 0.955`, top-return `0.557149 < 0.58`라 `excluded_train_gate`다. train top-return `0.837538` 대비 val top-return `0.556318`로, 단순 context residual은 exact returned-crop 선택을 일반화하지 못하고 과적합되는 신호가 명확하다. 이를 근거로 `return_score_policy_match_head`를 추가했다. 새 head는 candidate token, policy token projection, elementwise product, absolute-difference, route/decision logits, utility/positive/risk, geometry를 함께 사용해 final return score residual을 직접 학습하며, 마지막 linear zero-init으로 기존 checkpoint 초기 동작을 보존한다. 로컬 `py_compile`, `bash -n`, v4 테스트 10개, 작은 forward smoke와 원격 `py_compile`/`bash -n`을 통과했고, `1100658` policy-match detached, `1100659` policy-match policy-coupled를 새 비중복 workload로 생성했다. 별도 hard top-return loss 축 `1100656`/`1100657`도 active로 유지한다.

이 시점의 판단은 변하지 않는다. best selection score가 높은 row는 계속 나오지만, release gate는 action/top-return, route/subject, generated proposal, qualitative blocker를 동시에 본다. `1100646`처럼 train metric이 좋아도 val exact selection이 낮으면 배포 후보로 승격하지 않는다. policy-match 계열도 train gate를 통과하지 못하면 즉시 제외하고, 통과하는 경우에만 embedded no-prior head audit, Product-AR direct, qualitative pack, equal-4 public, GAIC official, latency 순서로 확장한다.

2026-04-29 06:44 KST에는 context-head 계열의 추가 terminal 결과를 반영했다. `1100647` route context는 action `0.944306`, top-return `0.555486`으로 실패했고, `1100650` route-conditioned context는 action `0.960308`로 action gate를 통과했지만 top-return `0.558603`만 실패했다. 이 결과는 source/action class 회복과 exact returned-crop selection이 분리된 문제임을 다시 확인한다. `1100656` hard-policy r1은 train 초기 session termination이라 제외하고, same hypothesis replacement `1100660`을 유지한다. active gate는 `1100652`, `1100654`, `1100657`, `1100660`, `1100658`, `1100659`다.

2026-04-29 06:48 KST에는 `1100652` policy-context r2도 terminal `excluded_train_gate`로 닫았다. route `0.625289`, subject IoU `0.664631`, subject-valid best balanced `0.609726`, proposal recall `0.938929`, generated align top1 `0.948878`, action top1 `0.990025`까지는 통과했지만 top-return `0.553616 < 0.58` 단독 실패라 배포 후보에서 제외한다. 이는 final crop action source를 맞추는 능력과 exact returned-crop 선택 능력이 여전히 분리되어 있음을 보여준다. `1100657`/`1100658`/`1100659` r1은 platform termination으로 source-of-truth에서 제외했고, 같은 가설은 `1100661` hard-context r2, `1100663` policy-match-detach r2, `1100662` policy-match-policy r2로 재생성했다. 현재 active/pending gate는 `1100654`, `1100660`, `1100661`, `1100662`, `1100663`이며, 하나라도 train gate를 통과하면 같은 checkpoint/config로 embedded no-prior head audit, Product-AR direct, qualitative review pack, equal-4 public, GAIC official, latency를 순서대로 확장한다.

2026-04-29 06:51 KST에는 `1100654` source/context-only r2도 terminal `excluded_train_gate`다. route `0.625289`, subject IoU `0.664631`, subject-valid best balanced `0.609726`, proposal recall `0.938929`, generated align top1 `0.947631`은 유지했지만 action top1 `0.940150`, top-return `0.541771`로 모두 gate 미달이다. source/geometry feature만 여는 경로는 action source와 exact returned-crop selection을 동시에 닫지 못했으므로 종료한다. 남은 active gate는 `1100660`, `1100661`, `1100662`, `1100663` 네 개다.

2026-04-29 06:56 KST에는 GPU 효율을 위해 competition return head 축을 추가 병렬화했다. `return_score_competition_head`는 모델 내부에 이미 구현되어 있었고 wrapper/CLI도 노출되어 있어, 로컬 `py_compile`, wrapper `bash -n`, competition forward smoke, 원격 `py_compile`/`bash -n`으로 기본 동작을 검증했다. 이후 `1100666` competition-detach r1과 `1100665` competition-policy r1을 생성했다. 이 두 run은 후보별 독립 return residual이 아니라 후보 집합 내 soft competition context를 사용해 exact top-return을 보정하는 실험으로, 기존 policy-match/head-hard loss 실험과 output path와 구조 가설이 모두 다르다. 현재 active/pending gate는 `1100660`, `1100661`, `1100662`, `1100663`, `1100665`, `1100666`이다.

2026-04-29 06:59 KST에는 `1100661` hard-context r2를 stopped 처리했다. MLP 상태는 Running이었지만 start 후 10분 이상 `train_status.json`이 없고 `train.log`도 0 byte라 실제 학습이 시작되지 않은 stuck run으로 판단했다. 같은 hard-context 가설은 고유 output path의 `1100667` r3로 재생성했다. 현재 active/pending gate는 `1100660`, `1100662`, `1100663`, `1100665`, `1100666`, `1100667`이다.

2026-04-29 07:01 KST에는 competition duplicate pending도 정리했다. 플랫폼이 `1100668`/`1100669`를 각각 `1100666`/`1100665`와 같은 run name으로 추가 등록했으므로 output collision 방지를 위해 시작 전 중단했다. 첫 validation 기준 `1100662` policy-match-policy는 action `0.967997`, top-return `0.559227`이고, `1100663` policy-match-detach는 action `0.943890`, top-return `0.559019`이다. 아직 top-return gate `0.58`에는 못 미치므로 epoch 2 terminal summary까지 기다린 뒤, 통과하지 못하면 competition 계열 또는 더 직접적인 selector 구조로 넘어간다.

2026-04-29 07:03 KST에는 `1100660` hard-policy r2가 terminal `excluded_train_gate`로 종료됐다. route `0.625289`, subject IoU `0.664631`, subject-valid best balanced `0.609726`, proposal recall `0.938929`, generated align top1 `0.948254`, action top1 `0.979426`은 통과했지만 top-return `0.556525 < 0.58` 단독 실패다. hard top-return loss와 policy/decision 보정으로 action source는 안정적으로 맞지만 exact returned-crop 선택은 여전히 분리되어 있다. 남은 active gate는 `1100662`, `1100663`, `1100665`, `1100666`, `1100667`이다.

2026-04-29 07:05 KST에는 `1100660` 종료로 회수된 GPU를 `1100670` competition-setrank r1에 투입했다. 이 run은 competition return head뿐 아니라 `set_ranker`도 함께 열어 candidate token 집합의 상대 순서 표현 자체를 top-return loss로 조정한다. 기존 competition detached/policy run이 후보별 residual을 보정하는 축이라면, `1100670`은 ranking representation을 직접 움직이는 축이다. exact selector 병목이 계속 반복되므로 같은 loss 반복이 아니라 구조적으로 상대순위 표현을 바꾸는 비중복 실험으로 관리한다.

2026-04-29 07:10 KST에는 `1100663` policy-match-detach r2가 terminal `excluded_train_gate`로 닫혔다. best epoch1 기준 action `0.943890`, top-return `0.559019`라 action/top-return 모두 미달이고, epoch2 last row는 top-return `0.565254`까지 올랐지만 action `0.942228`로 더 낮아 release gate를 닫지 못했다. `1100665` competition-policy r1은 첫 validation에서 action `0.988570`은 통과했지만 top-return `0.557357`라 아직 exact selector 병목이 반복된다. 남은 active gate는 `1100662`, `1100665`, `1100666`, `1100667`, `1100670`이다.

2026-04-29 07:12 KST에는 `1100662` policy-match-policy r2도 terminal `excluded_train_gate`로 닫았다. action top1 `0.982544`까지는 통과했지만 top-return `0.561513`이라 gate `0.58`에는 여전히 못 미친다. policy token과 candidate token product/diff, policy/decision head 공동 보정만으로는 exact returned-crop selection이 닫히지 않는다는 결론이다. 남은 active/pending gate는 `1100665`, `1100666`, `1100667`, `1100670`이다.

2026-04-29 07:14 KST에는 `1100670` setrank competition run이 Running으로 전환됐고, `1100671` `returnranker-policy-r1` pending도 확인했다. `1100671`은 `set_ranker`, utility/positive/risk, policy/decision, return competition head를 함께 열어 score surface와 action selector를 동시에 재보정하는 비중복 축이다. output path가 기존 run과 달라 유지하되 pending 10분 규칙을 적용한다.

2026-04-29 07:17 KST에는 `1100672` `returnranker-refine-r1` Running을 확인했다. 이 run은 `1100662` policy-match-policy best checkpoint에서 출발해 `set_ranker`, utility/positive/risk, return competition head를 열어 policy-match failure를 score/ranker surface에서 후속 보정한다. 현재 active/pending gate는 `1100665`, `1100666`, `1100667`, `1100670`, `1100671`, `1100672`이며, 모두 output path가 다르므로 비중복으로 유지한다.

2026-04-29 07:19 KST에는 `1100665` competition-policy r1이 terminal `excluded_train_gate`로 닫혔다. generated align top1 `0.950125`, action top1 `0.983167`은 통과했지만 top-return `0.557772`로 gate `0.58`을 넘지 못했다. competition context는 generated/action surface를 일부 개선하지만, exact returned-crop selection은 여전히 별도 병목이다.

2026-04-29 07:21 KST에는 `1100666` competition-detach r1도 terminal `excluded_train_gate`다. best epoch1 기준 action `0.939111`, top-return `0.557357`로 둘 다 gate 미달이며, epoch2 last row도 top-return `0.554447`로 개선되지 않았다. detached competition feature만으로는 selector를 닫지 못하므로, 남은 판단은 hard-context `1100667`, setrank `1100670`, returnranker `1100671`/`1100672`로 좁힌다.

2026-04-29 07:24 KST에는 `1100667` hard-context r3도 terminal `excluded_train_gate`로 닫았다. best epoch1 기준 action `0.951995`, top-return `0.554863`으로 둘 다 gate 미달이다. `1100670` setrank r1은 Running 전환 후 10분 이상 `train.log` 0 byte와 `train_status=starting` 상태가 유지되어 stopped 처리했고 abnormal/stuck execution으로 제외한다. 현재 active/pending gate는 정상 학습 중인 `1100671`, `1100672`와 public-heavy target 전이 가설인 `1100673`이다.

2026-04-29 07:25 KST에는 UCTR teacher가 public ensemble보다 강한데도 SSTK public student가 더 잘 나오는 현상을 실험 축으로 분리했다. `1100673` `returnranker-public60-r1`은 `public60/UCTR40` label을 사용하고 `1100662` best checkpoint에서 출발해 `set_ranker`, utility/positive/risk, policy/decision, return competition head를 함께 연다. 목적은 UCTR-heavy target이 같은 candidate pool 안에서 더 큰/느슨한 positive crop을 만들며 exact top-return/product action 정렬을 어렵게 하는지 확인하는 것이다.

2026-04-29 07:30 KST에는 `1100671`~`1100675` 다섯 개가 모두 MLP Running 상태임을 확인했다. `1100671` ranker-policy epoch1은 route `0.616345`, subject IoU `0.664631`, subject-valid balanced `0.609726`, proposal `0.938929`, generated align `0.950748`을 유지했지만 action `0.930798`, top-return `0.556318`로 아직 실패권이다. `1100672` ranker-refine epoch1도 action `0.952203`, top-return `0.557149`라 action/top-return gate를 닫지 못했다. `1100674` detach ranker와 `1100675` setrank r2는 train progress를 기록 중이며, `1100673` public60/UCTR40은 MLP Running이지만 output directory 작성 전 startup 단계라 log를 계속 감시한다.

이 시점부터 후속 실험 설계는 사용자 신규 지시를 기준으로 명시적으로 재정렬한다. 품질 우선 모델군은 MobileNet/온디바이스 제약을 완화한 backbone/input 확장만 뜻하지 않는다. 이미 DINOv2-L/518, ConvNeXtV2-Huge/512, SwinV2-L/384/512, EVA02-L/448에서 backbone 확대만으로는 route/subject/proposal/action/top-return/qualitative gate를 동시에 닫지 못했다. 따라서 다음 구조적 후속은 같은 backbone 반복이 아니라 candidate set 전체를 입력으로 받는 returned-crop selector/action decoder, generated proposal token과 policy/action token의 cross-attention, public/UCTR consensus와 disagreement를 분리한 target curriculum, subject-valid calibration과 final action selector의 공동 gate를 중심으로 설계한다. 단, 최종 inference 계약은 여전히 `image + target_ar` 단일 입력, external teacher subject prior 없음, checkpoint/config 내부 출력 일관성 유지다.

위 설계를 바로 실행 가능한 코드로 좁혀 `return_score_set_refiner_head`를 추가했다. 이 구조는 candidate token, policy token, route/decision logits, utility/positive/risk, geometry를 candidate set 단위 Transformer로 다시 읽은 뒤 `return_logits`만 zero-init residual로 보정한다. zero-init 마지막 linear 때문에 옵션을 켜도 checkpoint 로딩 직후에는 기존 return score와 같고, 옵션이 off인 기존 모델/현재 running workload에는 영향이 없다. 로컬 `py_compile`, wrapper `bash -n`, v4 테스트 10개, 작은 forward smoke를 통과했고, remote shared project에서도 `py_compile`/`bash -n`을 통과했다. `1100671`~`1100675`가 terminal gate를 통과하지 못하면 이 set-level selector를 비중복 후속 GPU workload로 투입한다.

2026-04-29 07:43 KST에는 `1100671` ranker-policy와 `1100672` ranker-refine이 terminal `excluded_train_gate`로 닫혔다. `1100671`은 best epoch1 action `0.930798`, top-return `0.556318`이고 epoch2는 top-return `0.562968`로 약간 올랐지만 action `0.917498`로 더 나빠졌다. `1100672`는 best epoch2 action `0.934954`, top-return `0.555694`다. 둘 다 route/subject/proposal은 유지했지만 final action/top-return 동시 gate가 닫히지 않았다. 회수 GPU에는 set-level selector를 바로 투입했다. 최초 생성된 `1100677` setrefiner-only와 `1100680` setrefiner-public60은 Running으로 전환됐지만 route expert distill default가 켜진 비효율이 확인되어 중단했다. source-of-truth set-refiner 축은 `1100681` setrefiner-detach, `1100682` setrefiner-policy, clean replacement `1100687` setrefiner-only-r2, `1100688` public60-joint-r1이다. 같은 run name으로 생긴 duplicate pending `1100679`/`1100683`/`1100686`은 output collision 방지를 위해 중단했다. 별도 interactive GPU에서는 `1100665/best.pt` embedded checkpoint의 bypass head/direct/qual audit을 진행 중이며, 이 결과는 train top-return proxy와 실제 no-prior runtime output의 차이를 분리하기 위한 참고 gate로만 사용한다.

2026-04-29 07:50 KST에는 남은 ranker/competition 계열도 닫았다. `1100673` public60/UCTR40은 epoch1에서 action `1.000000`, generated_align `0.957917`이었지만 subject IoU `0.556115`, subject-valid balanced `0.545000`, top-return `0.465139`로 멀었고 epoch2 중 platform termination으로 끝났다. `1100674` detach ranker는 best epoch2 action `0.916874`, top-return `0.560889`, `1100675` setrank r2는 best epoch1 action `0.921654`, top-return `0.564422`라 모두 `excluded_train_gate`다. corrected selector-only/public60 replacements 중 route expert distill default가 켜진 `1100689`/`1100690`은 비효율로 중단했고, source-of-truth set-refiner 축은 `1100681`, `1100682`, `1100687`, `1100688` 네 개로 좁혔다.

### 20.7 2026-04-29 08시대 set-refiner 종료와 action-decoder 실행

`1100681`/`1100682`/`1100687`/`1100688` set-level selector 계열은 모두 terminal train gate를 닫지 못했다. `1100687` clean set-refiner-only는 route `0.625289`, subject IoU `0.664631`, valid balanced `0.609726`, proposal recall `0.938929`, action `0.983998`까지 유지했지만 top-return `0.553616 < 0.58`로 제외했다. `1100682` policy-coupled set-refiner도 action `0.955320`은 간신히 통과했으나 top-return `0.565254`로 막혔다. `1100688` public60/UCTR40 joint는 action `1.000000`과 generated align `0.961250`을 만들었지만 subject IoU `0.556115`, valid balanced `0.545000`, top-return `0.466111`로 실패했다. 결론은 public60 target shift가 teacher benchmark 우세를 student final-action 품질로 자동 전이하지 않는다는 것이다.

따라서 현 시점 배포 후보는 여전히 없다. top-return plateau가 반복되므로 같은 loss 증량을 반복하지 않고, 이미 구현된 `return_score_action_decoder_head` 축을 실제 GPU gate로 올렸다. 2026-04-29 08:36 KST 기준 `1100692` preserve-nodistill, `1100693` public60-nodistill, `1100700` top-k distill set-refiner, interactive `mcn-q-eva02l448-setrefiner-topreturn-ip134-r2`는 모두 terminal `excluded_train_gate`다. 각각 best top-return은 `0.500208`, `0.465000`, `0.563383`, `0.554447` 수준으로 gate `0.58`을 넘지 못했고, public60 shift는 action을 올리는 대신 subject IoU/valid와 top-return을 악화했다. `1100665/best.pt` bypass audit도 route balanced `0.571399`, top1 hit `0.274688`, top1_not_positive `0.725313`, subject_valid_mismatch `0.284375`로 제외했으며, 이 결과는 runtime bypass가 제품 후보가 아니라 head-collapse evidence임을 뜻한다.

남은 source-of-truth는 `1100746`/`1100747`/`1100748`/`1100749` decision-source target corrected run과 label-surface control `1100752`/`1100753`이다. `1100724` actiondecoder-setonly-actionguard는 epoch2에서 action `0.957190`까지 넘겼지만 top-return `0.563799`로 낮아져 제외했고, `1100725` setonly-jointguard도 action `0.956775`를 넘겼지만 top-return `0.566708` 단독 실패라 제외했다. `1100726` gap005 jointguard는 best epoch1 action `0.946800`, top-return `0.570241`로 실패했고, `1100729` return-pair gap005는 top-return `0.572943`까지 갔지만 action `0.939111` 및 return explicit pairwise acc `0.531270`이 낮아 제외했다. `1100730` gap006 policy는 action `0.946176`, top-return `0.572319`, interactive setonly-gap006은 action `0.968828`, top-return `0.563591`라 둘 다 raw target 충돌을 넘지 못해 제외했다. interactive `mcn-q-eva02l448-actiondecoder-listwise-ip134-r1`은 best epoch1 기준 action `0.945345`, top-return `0.573982`라 제외했고, interactive `mcn-q-eva02l448-actiondecoder-policy-actionguard-ip134-r1`도 action `0.945761`, top-return `0.568579`로 제외했다. `1100701`은 action `0.965295`를 통과했지만 top-return `0.552785`라 제외했다. `1100702`는 epoch1 top-return `0.575852`로 가장 근접했지만 action `0.937448`이 무너져 제외했고, `1100704`는 action `0.960723`을 통과했지만 top-return `0.565669`에 머물러 제외했다. `1100715` action-guard도 top-return `0.572111`, action `0.945553`이라 제외했다. `1100741`/`1100742`는 새 action-return joint loss를 켰지만 라벨 진단 후 raw target 자체가 decision crop과 충돌함을 확인했으므로 first validation 전 중단했다.

코드 상태도 source-of-truth에 맞춰 고정했다. `model.py`에는 return logits 전용 BCE/listwise loss와 action decoder residual이 들어 있고, `train_mobilecropnet_v4.py`와 `run_mobilecropnet_v4_quality_relaxed_joint.sh`는 해당 CLI/env를 모두 받는다. 로컬 `py_compile`, runner `bash -n`, `tests/test_mobilecropnet_v4.py -q` 10개가 통과했고, 원격 shared project에도 재동기화한 뒤 원격 `py_compile`/`bash -n`을 통과했다. 다음 승격 조건은 train gate 통과가 아니라 같은 checkpoint의 internal no-prior head audit, Product-AR direct eval, qualitative review pack, equal-4/public, GAIC official, latency까지 모두 닫히는 것이다.

현재 후속 판단은 best selection score가 아니라 release gate 기준으로 한다. 즉 action이 크게 올라가도 top-return, route balanced accuracy, subject bbox IoU/valid calibration, generated proposal alignment, policy/action agreement, target-AR compatibility, checklist/why/risk agreement, qualitative catastrophic bucket 중 하나가 반복적으로 무너지면 final bundle로 보내지 않는다.

### 20.8 2026-04-29 09시대 return-pairwise loss와 strict curriculum 보강

현재 반복되는 병목은 crop action source와 exact returned-crop 선택이 분리된다는 점이다. action top1은 `0.96~1.00`까지 올라가는 row가 반복적으로 나오지만, top-return은 `0.55~0.57` plateau에 머물고, top-return이 `0.575852`까지 근접한 `1100702`는 action이 `0.937448`로 무너졌다. 따라서 후속 판단은 return head의 독립 BCE 증량이 아니라, teacher top candidate와 나머지 후보 사이의 명시적 pairwise margin을 final `return_logits`에 직접 거는 방향으로 재정렬했다.

이를 위해 `build_mobilecropnet_v4_return_pairwise_labels.py`를 추가해 gap005 top-agreement label에서 top-vs-rest pairwise JSONL을 만들었다. 산출물은 `artifacts/mobilecropnet_v4/quality_first_20260428/return_pairwise_gap005_topagree/` 아래에 두었고, train pair는 `43,314`, val pair는 `7,877`이다. `model.py`에는 `return_explicit_pairwise_weight`와 `return_explicit_pairwise_margin` 손실을 추가했고, `train_mobilecropnet_v4.py`와 `run_mobilecropnet_v4_quality_relaxed_joint.sh`도 동일 CLI/env를 받도록 맞췄다. 로컬 `py_compile`, runner `bash -n`, `tests/test_mobilecropnet_v4.py -q` 10개, 원격 `py_compile`/`bash -n`을 통과했다.

실행 축은 `1100729` return-pair gap005와 `1100730` gap006 single-positive로 나눴다. `1100729`는 pairwise acc가 실제 top-return/action gate로 전이되는지 확인하는 구조 검증이고, `1100730`은 positive 범위가 넓어서 action decoder가 안전한 crop class만 맞추고 exact candidate를 놓치는지 검증한다. `1100726` gap005 jointguard와 `1100724` setonly-actionguard는 모두 terminal 제외됐고, `1100725`는 `1100704` action-pass checkpoint에서 set-refiner와 action decoder를 공동 보정하는 남은 축이다.

2026-04-29 09시대 첫 확인 기준 `1100724`는 best epoch1 action `0.954697`, top-return `0.566085`이고 epoch2는 action만 `0.957190`으로 회복했지만 top-return `0.563799`라 제외했다. `1100725`는 action `0.956775`를 통과했으나 top-return `0.566708` 단독 실패다. `1100726`은 best epoch1 action `0.946800`, top-return `0.570241`로 둘 다 미달이라 제외했다. `1100729`는 top-return `0.572943`로 근접했지만 action `0.939111`이 낮고, return explicit pairwise acc도 `0.531270`으로 전이 신호가 약해 제외했다. `1100730` epoch1은 action `0.948254`, top-return `0.568786`라 아직 실패권이다. interactive setonly-gap006 epoch1은 action `0.961347`을 넘겼지만 top-return `0.562552`라 epoch2 확인 전까지 보류한다. interactive listwise와 policy-actionguard도 terminal 제외됐다. 아직 배포 후보는 없으며, 남은 run이 train gate를 통과하는 경우에만 같은 checkpoint/config로 no-prior head audit, Product-AR direct, qualitative pack, equal-4/public, GAIC official, latency를 순서대로 확장한다.

### 20.9 2026-04-29 09시대 action-return joint loss 추가

return-pairwise와 strict single-positive만으로는 action source와 exact returned candidate가 여전히 분리된다. 그래서 `model.py`에 `action_return_joint_loss`를 추가했다. 이 loss는 `decision_target`이 crop이면 best crop 후보 bag을, baseline/keep 계열이면 best base 후보 bag을 target으로 삼고, final `return_logits`의 log-sum-exp bag loss와 non-target margin을 동시에 건다. 기존 `action_consistency_loss`가 best crop과 best base의 source만 비교했다면, 새 loss는 source와 exact returned-crop candidate를 한 objective 안에서 묶는다.

`train_mobilecropnet_v4.py`에는 `--action_return_joint_weight`, `--action_return_joint_score_margin`, `--action_return_joint_logit_margin`, `--action_return_joint_temperature`를 추가했고, `run_mobilecropnet_v4_quality_relaxed_joint.sh`도 같은 env/CLI를 받도록 패치했다. 이어서 라벨 진단 결과를 반영해 `--return_target_mode`도 추가했다. `decision_source` 모드에서는 decision이 crop이면 baseline 후보를 return target에서 제외하고 crop 후보 안에서 top-return/return-exact를 계산한다. 반대로 baseline/keep 계열이면 base 후보 안에서 return target을 계산한다. raw score 기준 top-return은 `raw_top_return_*` metric으로 남긴다. 기본값은 `score`라 기존 checkpoint와 진행 중 run에는 영향을 주지 않는다. 로컬 `py_compile`, runner `bash -n`, `tests/test_mobilecropnet_v4.py -q` 10개와 원격 `py_compile`/`bash -n`을 통과했다.

라벨 진단 artifact는 `artifacts/mobilecropnet_v4/quality_first_20260428/public_uctr_label_diagnostics_20260429/summary.json`이다. 핵심 수치는 gap005/gap006 train에서 `decision_type=crop`이 `100%`인데 blend score top이 baseline 후보인 비율이 `0.697208`이고, top crop area가 public-heavy보다 넓다는 점이다. 즉 기존 train gate는 action target으로 crop을 요구하면서 top-return target으로 baseline을 요구하는 row가 다수였고, 이것이 action/top-return 동시 gate의 큰 원인이다. public60은 blend top baseline rate가 `0.419696`, top full-image rate가 `0.056305`로 낮아 student가 더 배우기 쉬운 분포였고, UCTR80 gap005는 baseline rate `0.697208`, full-image rate `0.131851`, top area mean `0.743879`로 더 느슨한 crop을 많이 만든다.

corrected 병렬 workload는 `1100746`/`1100747`/`1100748`/`1100749`다. `1100746`은 gap006 single-positive와 `1100702/best.pt` policy/actiondecoder init, `1100747`은 gap006 single-positive와 `1100704/best.pt` setonly/action-pass init, `1100748`은 gap005 curriculum과 `1100702/best.pt`, `1100749`는 original topagree와 `1100704/best.pt`를 사용한다. 네 run 모두 `RETURN_TARGET_MODE=decision_source`와 `action_return_joint_loss`를 사용하고, raw target 충돌을 숨기지 않기 위해 raw top-return metric도 같이 기록한다. 추가로 `1100752` public60/UCTR40 setonly control과 `1100753` vertical2 setonly control을 생성했다. 두 run은 route/subject를 동결한 채 label surface 차이만 분리해, public-heavy student 전이가 쉬웠던 이유와 UCTR-heavy label의 full-image/large-area bias를 실제 gate에서 검증한다.

### 20.10 2026-04-29 10시대 decision-source correction 및 raw/generated joint gate

09시대 corrected run은 decision-source top-return을 올리는 데는 일부 성공했지만, 제품 release gate에서는 통과로 볼 수 없다. `1100746`과 `1100748`은 action 또는 raw/generated metric이 미달했고, `1100747`, `1100749`, `1100752`, `1100753`은 기존 wrapper의 train gate가 raw top-return과 generated proposal alignment를 충분히 보지 못해 pass처럼 보였다. 따라서 각 산출물에 `product_gate_correction_decision.json`을 남기고 모두 제외했다. 특히 public60 control `1100752`는 action `0.994389`, decision-source top-return `0.639235`까지 올랐지만 raw top-return `0.567332`와 generated_align `0.0`이므로 `image + target_ar` no-prior 제품 계약을 만족하지 않는다.

이 correction을 코드로 고정하기 위해 `compute_mobilecropnet_v4_loss()`에 `raw_top_return_weight`를 추가하고, `train_mobilecropnet_v4.py`와 `run_mobilecropnet_v4_quality_relaxed_joint.sh`에 같은 CLI/env를 연결했다. wrapper train gate도 `RETURN_TARGET_MODE=decision_source` 계열에서 raw top-return gate와 generated alignment gate를 함께 보도록 고쳤다. 로컬 `py_compile`, wrapper `bash -n`, 원격 `py_compile`/`bash -n`을 통과했고, shared storage에 동기화했다.

새 source-of-truth 실험은 네 축이다. `1100785`는 score target을 유지하면서 raw-action/generated-align full-head를 본다. `1100787`은 decision-source target에 generated-align과 raw0.55 loss를 더한 vertical2 축이다. interactive GPU `10.2.4.134`의 `mcn-q-eva02l448-dsrc-genalign-raw035-setonly-ip134-r1`은 setonly/action-pass init에 raw0.35를 건 fast 축이다. `1100788`은 사용자 추가 지시에 맞춘 DINOv2-L/518 pretrained 품질 우선 smoke다. 10:20 KST 첫 validation 기준 interactive raw0.35는 action `0.973400`, decision-source top-return `0.609726`, generated_align `0.948254`를 확보했지만 raw_top_return `0.552369`가 아직 부족하다. `1100785`는 generated_align `0.948254`는 유지했지만 action `0.949501`, top-return `0.559019`가 낮다. 따라서 아직 배포 후보는 없고, raw/generated joint 축이 실패하면 다음 후속은 같은 loss 반복이 아니라 `winner_post_gate_candidate_id` 또는 product-consistent crop candidate 중심의 return target 재생성이다.

corrected gate를 통과하지 못한 `dsrc_public60_setonly`의 후속 GAIC/direct/latency 평가가 새로 생긴 것을 확인해 중복 GPU 사용을 중단했다. `1100810` direct rerun은 2026-04-29 10:20 KST에 stop했고, 이미 끝난 `1100808`/`1100809`는 excluded checkpoint의 부가 산출물로만 보관한다. 이후 public/equal-4/GAIC/latency 확장은 raw/generated joint train gate, no-prior head audit, Product-AR direct, qualitative pack을 먼저 통과한 survivor에게만 수행한다.

### 20.11 2026-04-29 11시대 product-consistent label repair 및 action-source 재정렬

10시대 raw/generated joint gate도 배포 후보를 만들지 못했다. `1100785`와 `1100787`, interactive raw0.35 setonly는 route/subject/proposal/generated align 일부를 유지했지만 action 또는 raw/top-return gate가 닫히지 않았고, DINOv2-L/518 smoke `1100788`은 validation 전 실행 예산으로 종료되어 품질 결과로 쓰지 않는다. excluded checkpoint 기반 public60 direct/public follow-up이 반복 생성된 것도 확인해 GPU 프로세스와 MLP workload를 종료했다.

원인은 단순 loss weight 부족이 아니라 label contract 충돌로 본다. 기존 gap005/gap006/vertical2 라벨은 `decision_target=crop`인 row가 많지만 `winner_post_gate_candidate_id`와 raw score top은 baseline/full 또는 target-AR minimal baseline을 가리키는 경우가 많았다. 따라서 `src/scripts/build_mobilecropnet_v4_product_consistent_labels.py`를 추가해 winner action source를 기준으로 `decision_target`을 재작성하고, baseline winner row에서는 baseline score fields도 함께 보정했다. 생성 산출물은 `product_consistent_gap006_singlepos_labels_20260429` train `9,670` / val `1,275`, 변경률 train `70.13%` / val `70.04%`; `product_consistent_vertical2_labels_20260429` train `16,017` / val `1,601`, 변경률 train `67.61%` / val `67.33%`다.

fast gate 두 축은 모두 제외됐다. interactive GPU `10.2.4.134`의 `mcn-q-eva02l448-pc-gap006-policyreturnfast-ip134-r2-20260429`는 route `0.620925`, subject IoU `0.670790`, subject-valid `0.619775`, proposal `0.939063`, generated align `0.945925`를 유지했지만 baseline-preserve `0.627743`, crop-action `0.419540`, top-return `0.533960`으로 실패했다. MLP `1100850` vertical2 fast도 route `0.625289`, subject IoU `0.664631`, valid `0.609726`, proposal `0.938929`, generated align `0.942020`은 유지했지만 baseline-preserve `0.603907`, crop-action `0.464879`, top-return `0.517041`로 실패했다.

이 결과 때문에 추가 병렬 축을 바로 만들었다. 그러나 `1100873`, `1100893`, actionstrong r4 로그에서 raw0.35 init checkpoint와 새 wrapper 기본값이 맞지 않아 `return_score_action_decoder` 일부 key가 재초기화되는 것을 확인했다. init checkpoint config는 `return_score_action_decoder_hidden_mult=1.1`, `return_score_action_decoder_layers=2`, `return_score_head_depth=2`, `return_score_head_hidden_mult=0.75`, action/source bias enabled, decision-source detach off, max scale `2.2`인데 mismatch run은 hidden/layer 기본값이 달랐다. 따라서 이 run들은 품질 결과가 아니라 config mismatch evidence로만 남긴다.

corrected source-of-truth는 세 축이다. interactive GPU `10.2.4.134`의 `mcn-q-eva02l448-pc-gap006-actionstrong-cfgmatch-r5-ip134-20260429`는 score target을 유지하되 action consistency/action-return loss를 강화한다. MLP `1100905` `mcn-q-eva02l448-pc-gap006-decisionsrc-cfgmatch-r2-20260429`는 gap006 product-consistent label에서 `RETURN_TARGET_MODE=decision_source`를 쓴다. MLP `1100906` `mcn-q-eva02l448-pc-vertical2-decisionsrc-cfgmatch-r3-20260429`는 vertical2 label surface에서 같은 config를 검증한다. `r5`와 `1100906`은 raw0.35 checkpoint key `684 / 684`를 shape/type skip 없이 로드했음을 확인했다. 이 중 하나가 train gate를 통과해야만 no-prior head audit, Product-AR direct, qualitative review pack, public/equal-4, GAIC official, latency로 확장한다. 통과하지 못하면 다음 단계는 품질 우선 대형 pretrained backbone 반복이 아니라 product-consistent label과 generated proposal/action coupling을 결합한 bounded smoke로 넘어간다.

config-match first validation은 baseline-preserve를 `0.69~0.71`까지 올렸지만 crop-action을 `0.34`대로 떨어뜨렸다. 이는 product-consistent label에서 baseline target rate가 `0.63~0.65`인 source imbalance가 loss를 baseline 쪽으로 밀었기 때문이다. 이를 직접 다루기 위해 `action_source_balanced` 옵션을 `model.py`, `train_mobilecropnet_v4.py`, `run_mobilecropnet_v4_quality_relaxed_joint.sh`에 추가했다. 이 옵션은 action consistency loss와 action-return joint loss의 row weight를 target source별 inverse-frequency로 보정하되, gate metric은 원래 source별 hit rate로 그대로 보고한다. 로컬/원격 `py_compile` 및 wrapper `bash -n`을 통과했다. 비균형 `1100905`/`1100906`/`r5`는 중단했고, 새 source-of-truth는 interactive `mcn-q-eva02l448-pc-gap006-actionstrong-srcbal-r6-ip134-20260429`, MLP `1100925` gap006 decision-source src-balanced, MLP `1100926` vertical2 decision-source src-balanced다.

`1100925`의 first validation은 source balance만으로는 부족하다는 결론을 줬다. baseline-preserve `0.684169`, crop-action `0.371473`, top-return `0.550679`로 crop-action은 config-match `0.345350`보다 조금 회복됐지만 release gate와는 여전히 멀고 top-return도 `0.558255 -> 0.550679`로 낮아졌다. 따라서 `1100925`는 중단하고, return score가 decision-source bias를 쓰는 구조를 고려해 decision head 자체의 crop/non-crop calibration을 보정하는 `DECISION_CROP_REBALANCE=1`, `DECISION_WEIGHT=2.0` replacement `1100945` `mcn-q-eva02l448-pc-gap006-srcbal-cropreb-r4-20260429`를 생성했다. `1100926` vertical2 source-balanced와 interactive r6 actionstrong은 계속 진행 중이다.

`1100926` vertical2 source-balanced도 first validation에서 baseline-preserve `0.601621`, crop-action `0.465919`, top-return `0.519950`에 그쳤다. vertical2 surface는 crop-action을 조금 살리지만 top-return/action joint가 더 낮아져 우선순위를 내린다. `1100926`은 중단했고, 같은 GPU에는 gap006 crop rebalance를 더 강하게 건 `1100951` `mcn-q-eva02l448-pc-gap006-srcbal-crop8-r5-20260429`를 생성했다. 이제 비교는 mild crop rebalance `1100945`, aggressive crop8 `1100951`, score-target actionstrong r6의 세 축으로 좁힌다.

interactive actionstrong r6도 first validation에서 baseline-preserve `0.685214`, crop-action `0.375131`, top-return `0.556688`에 머물러 score-target actionstrong 축을 제외했다. Pending 중이던 `1100951`은 중단하고 같은 crop8 설정을 interactive `mcn-q-eva02l448-pc-gap006-srcbal-crop8-r5-ip134-20260429`로 옮겼다. 남는 MLP에는 `RETURN_SCORE_DECISION_SOURCE_BIAS=0`인 `1100953` `mcn-q-eva02l448-pc-gap006-srcbal-nobias-r6-20260429`를 생성했다. 이 ablation은 decision head calibration이 아직 낮을 때 decision-source bias가 return score를 baseline 쪽으로 과도하게 끌고 가는지 확인하기 위한 것이다.

### 20.12 2026-04-29 13시대 action-source direct weighting 및 source-scale calibration

11시대 후속 crop8/nobias/validprop run은 모두 배포 후보가 아니었다. `1100945` mild crop rebalance는 baseline `0.685475`, crop-action `0.367294`, top-return `0.551724`; `1100950` validprop gap006은 baseline `0.567659`, crop-action `0.470219`, top-return `0.523511`; `1100953` nobias는 baseline `0.687827`, crop-action `0.385319`, top-return `0.558777`이다. vertical2 validprop `1100949`는 crop-action `0.577307`까지 올렸지만 baseline `0.513508`, top-return `0.487116`이라 제외했고, interactive crop8도 baseline `0.692006`, crop-action `0.361547`, top-return `0.551724`라 제외했다. route/subject/proposal은 대체로 유지되지만 source/action calibration과 final returned-crop selection이 동시에 닫히지 않는다.

이를 직접 보기 위해 `ACTION_SOURCE_CROP_WEIGHT_MULT`를 추가했다. 이 옵션은 `action_source_balanced` 이후 crop target row의 loss weight만 추가 배율로 키우며, gate metric은 원래 baseline/crop source별 hit rate로 그대로 기록한다. 패치 범위는 `src/mobilecropnet_v4/model.py`, `src/scripts/train_mobilecropnet_v4.py`, `src/scripts/run_mobilecropnet_v4_quality_relaxed_joint.sh`이고, 로컬 `py_compile`, wrapper `bash -n`, 원격 `py_compile`/`bash -n`을 통과했다. 그러나 단순 crop-row 증량은 기대한 방향이 아니었다. MLP `1100958` actcrop40은 baseline `0.589864`, crop-action `0.472571`, top-return `0.521421`; interactive actcrop25는 baseline `0.621996`, crop-action `0.430512`, top-return `0.532915`로 baseline-preserve를 크게 잃었다. 두 run은 즉시 중단했고 no-prior/direct/qualitative로 승격하지 않는다.

Pending이던 `1100961` actsrcscale은 user 지시에 맞춰 중단했다. 이후 같은 가설을 interactive GPU `10.2.4.134`에서 `mcn-q-eva02l448-pc-gap006-actsrcscale-r2-ip134-20260429`로 깨끗하게 재시작했다. r1은 launch collision으로 동일 output dir에 두 process가 붙어 artifact 오염 위험이 있으므로 source-of-truth에서 제외한다. r2는 `return_score_action_source_bias`와 `return_score_decision_source_scale`까지 trainable prefix에 넣어 decision-source bias를 학습 가능한 calibration으로 다룬다. 이 run의 first validation에서 baseline/crop action, top-return, generated align이 동시에 회복되지 않으면 같은 crop-weight 반복을 중단하고, 다음 단계는 margin-filtered product-consistent label, decision-conditioned return mask, best-pretrained 대형 backbone/대입력 품질 우선 smoke로 넘어간다.

사용자가 지적한 UCTR teacher 우세와 SSTK public 학생 우세의 불일치는 계속 유효한 blocker다. `UNIFIED_PUBLIC_BENCHMARK_EQUAL4_TEACHER_REPORT_KO_2026-04-23.md`의 teacher 표에서는 `universal_crop_teacher_h_stage3_gate128`가 public cropper ensemble보다 높지만, 학생 전이에서는 UCTR positive가 더 큰/느슨한 crop과 낮은 policy/utility surface를 만들고, product-consistent repair에서는 baseline/full 또는 minimal source로 target이 대량 이동한다. 즉 UCTR teacher 품질 자체가 약한 것이 아니라, 현재 학생의 final action/source/return head가 UCTR label surface를 제품 runtime 계약으로 안정적으로 증류하지 못하는 것이 핵심 원인이다.

### 20.13 2026-04-29 13시대 margin005 curriculum 및 decision-bias ablation

`actsrcscale` 계열도 gate를 닫지 못했다. interactive r2는 route `0.620925`, subject IoU `0.670790`, valid `0.619775`, proposal `0.939063`, generated align `0.944357`을 유지했지만 baseline `0.618339`, crop-action `0.452978`, top-return `0.535005`, decision acc `0.558516`라 제외했다. MLP `1100965` mild source-scale은 crop 가중을 낮췄지만 baseline `0.654389`, crop-action `0.396290`, top-return `0.538662`에 그쳐 중단했다. 이 결과는 source-scale/crop-row weighting만으로는 final action source와 returned-crop selection이 함께 회복되지 않음을 보여준다.

후속으로 source margin curriculum을 만들었다. 기존 product-consistent gap006 label에서 source margin `<0.05`인 row가 train `1,803 / 9,670`, val `259 / 1,275` 수준임을 확인했고, train에서만 이 row를 제외하고 val은 full로 유지하는 `product_consistent_gap006_margin005_trainonly_labels_20260429_v2`를 생성했다. train row는 `7,867`, val row는 `1,275`다. 최초 raw-input margin output은 baseline score copy 전 margin을 계산해 negative margin이 과도했으므로 source-of-truth에서 제외했고, builder는 baseline score copy 이후 margin을 계산하도록 수정했다.

당시 active source-of-truth는 두 축이었다. MLP `1100966` `mcn-q-eva02l448-pc-gap006-margin005-srcscale-r1-20260429`는 margin005 train label에서 source-scale/full return heads를 검증했다. interactive GPU `10.2.4.134`의 `mcn-q-eva02l448-pc-gap006-margin005-decisionbias-r1-ip134-20260429`는 set-refiner/action-decoder를 고정하고 decision/policy/source bias만 열어 return surface를 보존한 채 source calibration만 되는지 확인했다. 두 축 모두 아래 20.14에서 제외 판정으로 닫혔다.

### 20.14 2026-04-29 13시대 source-specific 실패와 best-pretrained warm-head 품질 우선 재개

margin005/source-scale 계열도 배포 후보가 아니었다. `1100966`은 route `0.620925`, subject IoU `0.670790`, subject-valid `0.619775`, proposal `0.939063`을 유지했지만 baseline `0.573929`, crop-action `0.429206`, top-return `0.497649`로 실패했다. interactive margin005 decision-bias는 crop-action `0.708725`까지 올렸으나 baseline `0.100836`, top-return `0.239289`로 source collapse가 발생했다. 따라서 decision bias만 강하게 여는 방식은 catastrophic qualitative risk가 크다.

source별 residual return head도 같은 결론이다. full gap006 source-specific run은 baseline `0.677638`, generated align `0.947492`는 좋았지만 crop-action `0.325496`, top-return `0.540230`, decision acc `0.585946`로 crop branch가 무너졌다. MLP `1100976` margin005 source-specific은 첫 validation 전에 platform `Session terminated, killing shell`로 종료되어 모델 품질 판정에는 쓰지 않는다. decision-conditioned mask smoke `1101001`은 Succeeded로 종료됐지만 기본 crop-action `0.325496`, decision-conditioned crop-action `0.250000`, decision-conditioned top-return `0.574451`로 train gate에서 제외했다.

다음 축은 사용자 지시의 품질 우선 제약 완화다. 단순히 큰 backbone을 scratch로 반복하지 않고, best-pretrained backbone은 반드시 보존하면서 기존 EVA02-L checkpoint에서 head만 warm-start하도록 학습 entrypoint를 수정했다. `src/scripts/train_mobilecropnet_v4.py`는 `--init_checkpoint_skip_prefixes`를 받아 warm-start state_dict에서 `backbone.*`을 skip할 수 있고, wrapper는 `INIT_CHECKPOINT_SKIP_PREFIXES`를 launch config와 CLI에 기록한다. 이 패치로 DINOv2-L/518과 ConvNeXtV2-Huge/512의 local pretrained weight를 덮어쓰지 않고 route/policy/return/checklist head 지식만 전이한다.

이후 두 warm-head smoke도 배포 후보가 되지 못했다. ConvNeXtV2-Huge r2 `1101018`은 route balanced `0.325000`, subject IoU `0.408930`, subject-valid `0.500000`, crop-action `0.122500`, top-return `0.560000`으로 train gate에서 제외했다. DINOv2-L/518 IP134 run은 epoch1에서 route `0.428750`, subject IoU `0.471552`, subject-valid `0.520000`, proposal `0.913155`, crop-action `0.150000`, top-return `0.600000`을 기록해 route/subject/proposal/action이 동시에 미달했고, `manual_stop_decision.json`을 남기고 epoch2 중 중단했다. 이 결과는 best-pretrained backbone과 입력 크기 확대만으로 Product-AR source/action/return alignment가 닫히지 않는다는 증거다.

### 20.15 2026-04-29 14시대 binary source-head gate 분리 패치

반복 실패 패턴을 재분석한 결과, `keep_full`/`minimal_crop`/`crop` 3-class decision head가 keep/minimal 구분과 crop-vs-baseline source 선택을 동시에 담당하면서 decision-conditioned return mask가 crop branch를 과하게 잃는 문제가 핵심 blocker로 남았다. 이를 분리하기 위해 `MobileCropNetV4`에 binary `decision_source_head`를 추가했다. 이 head는 외부 teacher subject prior 없이 policy token에서 crop source logit을 예측하며, `return_score_decision_source_from_source_head`와 `return_score_decision_conditioned_from_source_head`를 켜면 return source bias와 decision-conditioned mask가 3-class decision logit 대신 binary source logit을 사용한다.

학습 entrypoint에는 `decision_source_weight`, `decision_source_balanced_bce`, `decision_source_crop_rebalance_max_weight`를 추가했다. 이 loss는 crop/non-crop source target만 직접 최적화하고 `policy_source_acc`, `policy_source_base_recall`, `policy_source_crop_recall`을 기록한다. 로컬 `/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m py_compile src/mobilecropnet_v4/model.py src/scripts/train_mobilecropnet_v4.py`, wrapper `bash -n`, 원격 `/usr/local/bin/python3 -m py_compile`과 `bash -n`을 모두 통과했다.

현재 active source-of-truth는 두 개다. interactive GPU `10.2.4.134`의 `mcn-q-eva02l448-pc-margin005-sourcehead-r1-ip134-20260429`는 EVA02-L raw0.35 checkpoint를 warm-start하고 source-head balanced BCE `1.40`을 적용한 균형형이다. MLP `1101034` `mcn-q-eva02l448-pc-margin005-sourcehead-cropboost-r1-20260429`는 source BCE `2.00`, crop source multiplier `1.35`로 crop branch 회수력을 더 강하게 보는 변형이다. 두 run 모두 train gate에서 route/subject/proposal/action/top-return/decision-conditioned source 균형을 보이기 전에는 final bundle, GAIC official, public/equal-4, Product-AR direct, qualitative pack, latency로 승격하지 않는다.

### 20.16 2026-04-29 15시대 source-mixture와 sampler 재정렬

14시대 source-head 결과는 route/subject/proposal geometry가 문제가 아니라 crop-vs-baseline source 학습 표본과 selection surface가 엇갈리는 문제임을 보였다. IP134 hard source-head는 route `0.617473`, subject IoU `0.652459`, subject-valid `0.582500`, proposal `0.940154`, generated-align `0.948750`을 유지했지만 decision-conditioned crop action `0.100000`, baseline preserve `0.895417`로 실패했다. MLP `1101034`도 crop action `0.061250`, baseline preserve `0.902500`으로 같은 collapse를 보였다. decision-logit source-mixture `1101051`/`1101052`는 top-return `0.612500`/`0.620000`까지 올렸지만 crop action `0.172500`/`0.227500`에 머물렀다.

이에 따라 source-mixture만으로는 부족하고 train sampling 자체를 source-balanced로 바꿔야 한다고 판단했다. `src/scripts/train_mobilecropnet_v4.py`에 `--train_decision_source_balanced_sampler`와 power/min/max 옵션을 추가했고, route-balanced sampler와 decision-source sampler를 곱한 composite weight로 `WeightedRandomSampler`를 만들도록 했다. wrapper에는 `TRAIN_DECISION_SOURCE_BALANCED_SAMPLER` env와 launch config 기록을 추가했다. 로컬/원격 `py_compile`, `bash -n`을 통과했고, IP134 replacement의 `dataset_summary.json`에서 base `1043`, crop `557` source target count와 route+source composite sampler 활성화를 확인했다.

현재 active source-of-truth는 sampler-enabled 다섯 축이다. `1101088`은 gap006 single-positive label에서 source-mixture strength `0.50`을 검증하는 MLP workload이고, `1101090`은 margin005 train-only label에서 source-mixture strength `0.70`, crop multiplier `1.50`을 검증한다. interactive GPU `10.2.4.134`의 `mcn-eva02l-pc-m005-shsamp-s125c20-ip134-r1-20260429`는 source-head 기반 mixture strength `1.25`, crop multiplier `2.0`의 강한 recovery axis다. mild source-mixture bracket `1101096`/`1101097`은 start time 없이 10분 이상 Pending에 머물러 종료했다. 이후 r2 재생성 중 `s025c12-r2`가 `1101117`/`1101119` 같은 output path 중복 실행으로 오염되어 제외했고, clean mild bracket은 `1101120` strength `0.10 + crop 1.00`, `1101124` strength `0.25 + crop 1.20`이다. 다섯 축 모두 train gate에서 source-mixture crop action, baseline preserve, top-return, route/subject/proposal이 동시에 닫히기 전에는 Product-AR direct, GAIC official, public/equal-4, qualitative pack으로 승격하지 않는다.

### 20.17 2026-04-29 15시대 source-margin loss와 active gate 재정렬

sampler-enabled source-mixture bracket은 모두 같은 failure mode로 닫혔다. `1101088`은 route `0.619140`, subject IoU `0.665351`, subject-valid `0.585625`, proposal `0.941589`, generated-align `0.946250`을 유지했지만 baseline `0.632500`, crop-action `0.429583`, top-return `0.540000`, policy crop recall `0.044167`이라 제외했다. `1101120`은 baseline `0.406667`, crop-action `0.560833`, top-return `0.421250`, `1101124`는 baseline `0.405417`, crop-action `0.577083`, top-return `0.419583`으로 더 강한 crop-action 쪽으로 움직였지만 baseline preserve와 final return이 같이 무너졌다. IP134 sourcecal-croprec도 epoch2에서 crop-action `0.585833`까지 올렸으나 baseline `0.438333`, top-return `0.440417`이라 배포 후보가 아니다.

이 결과로 현재 원인은 “crop source recall이 낮다” 하나가 아니라, final return surface가 base 후보군과 crop 후보군 사이의 상대 logit gap을 제품 action target에 맞춰 분리하지 못하는 문제로 좁혀졌다. 따라서 단순 source/crop row weighting 반복을 중단하고 `return_source_margin_loss`를 추가했다. 이 loss는 active return surface에서 crop target row는 best crop logit이 best base logit보다 margin 이상 높아지도록, baseline/minimal target row는 반대로 best base logit이 best crop logit보다 margin 이상 높아지도록 직접 압박한다. source-balanced row weighting과 crop multiplier는 옵션으로 유지하되, gate metric은 원래 baseline/crop hit rate와 top-return을 그대로 본다.

구현 변경은 `src/mobilecropnet_v4/model.py`, `src/scripts/train_mobilecropnet_v4.py`, `src/scripts/run_mobilecropnet_v4_quality_relaxed_joint.sh`에 한정했다. 새 CLI/env는 `return_source_margin_weight`, `return_source_margin_logit_margin`, `return_source_margin_balanced`, `return_source_margin_balance_max_weight`, `return_source_margin_crop_weight_mult`이며, metric은 `return_source_margin_acc`, `return_source_margin_crop_hit`, `return_source_margin_base_hit`, `return_source_margin_target_gap`을 기록한다. 로컬 지정 Python의 `py_compile`, wrapper `bash -n`, `git diff --check`, 원격 `/usr/local/bin/python3 -m py_compile` 및 원격 `bash -n`을 모두 통과했다.

처음 시작한 source-margin r1 두 개는 active source-of-truth에서 제외했다. interactive GPU `10.2.4.134`의 `mcn-eva02l-pc-m005-srcmargin-honly-ip134-r1-20260429`와 MLP `1101145` `mcn-eva02l-pc-m005-srcmargin-honly-w2-r1-20260429`는 route expert distill checkpoint를 로드했지만, 실제 trainable scope는 policy/decision/source/return/checklist head-only였다. 따라서 route/backbone은 고정된 상태에서 expert forward 비용만 추가되는 구조라 first validation 전 manual stop으로 닫았다.

현재 active source-of-truth는 no-distill r2 두 개다. interactive GPU `10.2.4.134`의 `mcn-eva02l-pc-m005-srcmargin-honly-nodistill-ip134-r2-20260429`는 source-margin weight `3.0`, margin `0.30`, source-mixture strength `0.25`의 강한 gap 보정 축이다. MLP `1101159` `mcn-eva02l-pc-m005-srcmargin-nodistill-w2-r2-20260429`는 weight `2.0`, margin `0.25`, strength `0.15`의 보수 축이다. 둘 다 `ROUTE_EXPERT_DISTILL_WEIGHT=0.0`, `ROUTE_EXPERT_DISTILL_HARD_WEIGHT=0.0`이며, backbone/geometry를 보존하고 policy/decision/source/return/checklist 계열 head만 학습한다. train gate에서 return-source margin이 좋아져도 route, subject bbox IoU/valid calibration, proposal recall, generated proposal alignment, source-mixture baseline/crop action, top-return, checklist/why/risk agreement가 동시에 닫히지 않으면 Product-AR direct, qualitative pack, public/equal-4, GAIC official, latency로 승격하지 않는다.

현재 배포 후보는 여전히 없다. `best_selection_score`는 checkpoint 내부 선택 참고값일 뿐이며, 이번 source-margin 축도 catastrophic qualitative failure 가능성을 배제하기 전에는 배포 후보 checkpoint/config로 고정하지 않는다.

### 20.18 2026-04-29 16시대 source-gate hard mask와 decision-source margin

no-distill source-margin r2는 route/subject/proposal/generated-align geometry를 유지했지만 release gate의 핵심인 crop/base source lane을 닫지 못했다. IP134 `mcn-eva02l-pc-m005-srcmargin-honly-nodistill-ip134-r2-20260429`는 epoch1에서 route `0.627058`, subject IoU `0.667017`, subject-valid balanced `0.628542`, proposal recall `0.941440`, generated-align `0.951250`을 유지했지만 baseline preserve `0.528333`, crop-action `0.497917`, top-return `0.479167`, source-margin acc `0.597083`에 그쳤다. MLP `1101159`도 baseline `0.514583`, crop-action `0.510417`, top-return `0.475833`, source-margin acc `0.592917`로 같은 failure mode를 보였으므로 종료했다. 즉 final return surface에서 crop/base best logit 간격을 직접 주는 것만으로는 source head threshold와 runtime lane 선택이 안정화되지 않는다.

agreement-curriculum r3도 해법이 아니었다. `mcn-eva02l-pc-m005-agree-srcmargin-nodistill-ip134-r3-20260429`는 agreement row만 사용했지만 epoch1에서 baseline preserve `0.187917`, crop-action `0.685833`, top-return `0.288750`, decision-source acc `0.412500`, base recall `0.347917`, crop recall `0.494167`로 baseline lane이 붕괴했다. 이는 label filtering이 crop branch를 강하게 만들 수는 있어도, product runtime에서 필요한 baseline/crop 동시 calibration을 보장하지 못한다는 증거다.

따라서 구조적으로 source head logit 자체를 release gate에 넣기 위해 `decision_source_margin_loss`를 추가했다. 이 loss는 crop row에서는 source logit이 `+margin` 이상, baseline/minimal row에서는 `-margin` 이하가 되도록 강제하며, class-balanced row weight와 crop multiplier를 지원한다. `src/mobilecropnet_v4/model.py`, `src/scripts/train_mobilecropnet_v4.py`, `src/scripts/run_mobilecropnet_v4_quality_relaxed_joint.sh`에 CLI/env/config/metrics를 추가했고, 로컬/원격 문법 검증을 통과했다.

새 active source-of-truth는 두 축이다. IP134 `mcn-eva02l-pc-m005-agree-srcgate-hardmask-ip134-r4-20260429`는 agreement curriculum에 `decision_source_margin_loss`를 더하고, MLP `1101195` `mcn-eva02l-pc-m005-srcgate-hardmask-r2-20260429`는 full margin005 label에 같은 source-gate hard-mask를 적용한다. 둘 다 `RETURN_SCORE_DECISION_CONDITIONED=1`, `RETURN_SCORE_DECISION_CONDITIONED_FROM_SOURCE_HEAD=1`, `RETURN_SCORE_SOURCE_MIXTURE=0`으로 active return surface를 source head hard mask 기준으로 평가한다. `1101183`은 같은 MLP 가설의 r1이지만 wrapper env export 오류로 즉시 실패했으므로 모델 결과로 보지 않는다.

hard-mask와 threshold 보정도 release gate를 닫지 못했다. full-label hard-mask MLP `1101195`는 route `0.627058`, subject IoU `0.667017`, subject-valid `0.628542`, proposal `0.941440`, generated-align `0.955000`을 유지했지만 baseline `0.709167`, crop-action `0.262083`, top-return `0.547083`으로 제외했다. neutral IP134 run은 top-return `0.582083`까지 올랐지만 baseline `0.778750`, crop-action `0.207083`, decision-source acc `0.627500`이라 crop lane recall이 너무 낮았다. threshold `-0.20`/`-0.35` 보정도 top-return `0.593750~0.595000`과 baseline `0.785000~0.786667`을 보였지만 crop-action은 `0.222083`에 머물렀다.

단일 threshold 가능성도 별도 진단으로 닫았다. `decision_source_threshold_sweep`이 hard-mask 적용 후 logits를 다시 sweep하던 문제를 수정해 raw `action_logits` 기준으로 sweep하도록 했고, best-min-recall metric을 추가했다. `mcn-eva02l-pc-m005-dsrc-thrsweepfast-ip134-r2-20260429` 결과, active threshold surface는 baseline `0.877500`, crop-action `0.137083`, top-return `0.636667`이다. raw source threshold sweep의 best balanced point도 base recall `0.853333`, crop recall `0.543333`, top-return `0.708333`이고, min-recall best는 base `0.545417`, crop `0.569167`, top-return `0.473750`이다. 따라서 source logit threshold만 이동해서는 product release gate를 만족할 수 없다.

후속 구조 축은 `decision_source_pair_head`다. 기존 `decision_source_head`는 policy token만 보고 crop/base source를 결정해 best-base/best-crop 후보 표면을 직접 보지 못했다. 새 head는 기존 source logit에 zero-init residual로 붙고, best-base/best-crop candidate token, return/utility gap, route probability, decision context를 함께 입력으로 사용한다. warm-start checkpoint의 동작은 pair residual `0`으로 보존되며, 이후 pair residual만 학습하거나 source head와 함께 여는 두 변형을 비교한다. 로컬 smoke와 원격 문법 검증을 통과했고, 현재 IP134 `mcn-eva02l-pc-m005-pairsrc-resid-ip134-r1-20260429`는 pair residual만 학습 중이며, MLP `1101228` `mcn-eva02l-pc-m005-pairsrc-open-r1-20260429`는 pair residual + source head + source scale 변형이다. 두 run 모두 train gate에서 source/action/top-return과 route/subject/proposal/generated-align이 동시에 닫히기 전에는 Product-AR direct, qualitative pack, public/equal-4, GAIC official로 승격하지 않는다.

### 20.19 2026-04-29 18시대 late-pair action calibration, source-logit audit, competition/context 축

pair-source 이후 late-pair/action calibration bracket도 release gate를 닫지 못했다. MLP `1101253` `mcn-eva02l-pc-m005-latepair-actioncal-r1-20260429`는 epoch3에서 route balanced `0.625808`, subject IoU `0.667017`, subject-valid balanced `0.628542`, proposal recall `0.941440`, generated-align top1 `0.951250`을 유지했지만 baseline preserve `0.553333`, crop-action `0.545833`, top-return `0.512917`에 그쳤다. threshold `-0.20` MLP와 threshold `-0.13` IP134 bracket은 crop-action을 `0.574167`까지 올렸지만 baseline preserve가 `0.502500~0.503750`, top-return이 `0.481667`로 더 낮아져 제외했다.

no-sampler full-label bracket도 같은 결론이다. MLP `1101270` fullnosrcsam epoch1은 baseline preserve `0.681557`, crop-action `0.351881`, top-return `0.548589`였고, IP134 softmix no-sampler는 baseline `0.703239`, crop-action `0.324713`, top-return `0.565047`였다. source-balanced sampler를 끄면 baseline lane은 올라가지만 crop lane이 무너지고, soft mixture를 더해도 crop branch가 회복되지 않는다. 따라서 sampler on/off 자체가 원인이 아니라, source head와 final returned-crop surface가 같은 후보 pair를 일관되게 비교하지 못하는 구조적 문제로 본다.

`1101253` best checkpoint의 source-logit dump도 threshold-only 해법을 배제한다. `source_logit_route_ar_analysis.json` 기준 image count는 `1,275`, crop target rate는 `0.299608`이다. zero threshold는 acc `0.663529`, base recall `0.674132`, crop recall `0.638743`, balanced `0.656438`이다. best-balanced threshold `-0.807115`는 crop recall `0.814136`을 만들지만 base recall이 `0.545353`으로 떨어지고, best-min-recall threshold `-0.091291`도 base recall `0.659574`, crop recall `0.667539`, top-return `0.473750`에 그친다. 즉 global threshold, route/AR threshold table, hard mask는 배포 모델의 source/action agreement를 만들 수 없다.

이에 따라 다음 GPU 축은 단순 scalar 가중치가 아니라 candidate competition/context 구조를 켠 bounded smoke로 옮겼다. IP134 `mcn-eva02l-pc-m005-compctx-srcbal-softmix035-ip134-r1-20260429`는 context/policy-match/competition/set-refiner/action-decoder head를 열고 soft source-mixture strength `0.35`를 적용했지만 epoch1에서 source-mixture baseline `0.673333`, crop-action `0.408750`, top-return `0.552500`이라 manual stop으로 닫았다. geometry head는 route `0.625808`, subject IoU `0.667017`, proposal `0.941440`, generated-align `0.958750`으로 유지됐기 때문에 실패 원인은 geometry가 아니라 final action/source lane이다.

competition/context bracket도 first validation에서 제외했다. MLP `1101293` no-softmix는 route `0.627474`, subject IoU `0.667017`, proposal `0.941440`, generated-align `0.955000`을 유지했지만 baseline preserve `0.636250`, crop-action `0.450833`, top-return `0.538333`이라 manual stop으로 닫았다. IP134 crop-source weight `1.8` 보완축은 crop-action을 `0.484583`까지 올렸지만 baseline preserve `0.596667`, top-return `0.522083`으로 더 악화됐다. 따라서 crop-source loss 배율만 더 주는 방향은 train에서는 움직이지만 val product surface로 전이되지 않는다.

현재 남은 active source-of-truth는 두 개다. 첫째, high-margin train-only label `product_consistent_gap006_margin015_trainonly_labels_20260429`를 새로 만들었다. train은 source margin `<0.15` row `3,082`개를 제거해 `6,588` rows가 남고, baseline winner rate는 `0.674560`이다. val은 full `1,275` rows를 유지한다. IP134 `mcn-eva02l-pc-m015-compctx-srcbal-ip134-r1-20260429`은 이 label set에서 source ambiguity 제거가 product action surface로 전이되는지 확인한다. 둘째, MLP `1101320` `mcn-eva02l-pc-m005-compctx-srcbal-unfreeze-r1-20260429`은 head-only 한계를 확인하기 위해 backbone까지 low LR `1.0e-6`로 여는 bounded smoke다. 둘 다 first validation에서 route/subject/proposal/generated-align을 유지하면서 baseline preserve, crop-action, top-return을 동시에 개선하지 못하면 Product-AR direct, qualitative review pack, public/equal-4, GAIC official로 승격하지 않는다.

리더보드 공백 상태도 재확인했다. public60 보강 이후 주요 트랙/프로파일의 `FCDB IoU`, `CPC weighted`, `GNMC IoU`, `GAIC top1 MOS`, `GAIC SRCC`, `GAIC Accw4@10`, `GAIC primary`, Product-AR direct, latency 공백은 모두 채워졌다. 남은 3개 row는 checkpoint가 원격 shared storage에 없어 재평가할 수 없는 invalid checkpoint row다. 따라서 현재 남은 작업은 리더보드 표 채우기가 아니라, train gate를 실제로 통과하는 단일 checkpoint를 만든 뒤 같은 checkpoint/config/inference command로 no-prior direct, qualitative taxonomy, latency, public/equal-4, GAIC official을 순서대로 닫는 것이다.

### 20.20 2026-04-29 20시대 sourcegate 종료와 mixed-real preserve 전환

calibrated source-action gate는 mixed validation에서 배포 gate를 닫지 못했다. `mcn-srcgate-m005-mixval-20260429`는 final positive hit `0.952157`, subject IoU `0.703673`, proposal target-AR recall@5 `0.960000`을 유지했지만 crop-action hit `0.069804`, action-source acc `0.633725`, decision acc `0.435294`라 final action/source lane이 붕괴했다. `mcn-eva02l-rtcropv3-srcsel-pc-m015-calibbb-srcgate-mixedval-20260429`도 final positive hit `0.951373`, subject IoU `0.703673`은 비슷하지만 crop-action hit `0.031373`, decision acc `0.450980`라 더 약하다. 두 run 모두 qualitative bucket에서 subject-valid mismatch `24.55%`, subject IoU low `10.67%`, final_not_positive 약 `4.8%`가 반복되어 배포 후보에서 제외한다.

runtime crop-policy v3 set-only checkpoint는 내부 direct 계약의 upper-bound로만 보존한다. embedded Swin route checkpoint `mcn-eva02l-pc-m005-rtcropv3-setonly-r1-manual-swinembed-vthr045-r2.pt`는 Product-AR direct TEST에서 final hit `0.967615`, action/decision acc `0.998339`, subject IoU `0.708175`, target-AR compatibility `1.0`을 기록했다. 그러나 public/GAIC full eval은 FCDB `0.730538`, CPC weighted `0.769527`, GNMC `0.702191`, GAIC top1 MOS `3.697220`, SRCC `0.191504`, Accw4@10 `0.307569`, GAIC primary `0.482884`라 ranking 품질이 낮다. 따라서 이 checkpoint는 "image + target_ar no-prior direct는 가능하지만 public/GAIC crop quality가 부족하다"는 기준선이지 최종 배포 후보가 아니다.

all-crop v4 real-crop 라벨도 source-of-truth에서 제외했다. `1101354` ConvNeXtV2-Huge/512와 `1101355` DINOv2-L/518 epoch1은 route `0.430000 / 0.466250`, subject IoU `0.484996 / 0.482448`, top-return `0.238750 / 0.256250`, baseline target rate `0`으로 train gate와 멀었다. EVA02-L all-crop `1101353`, warm-head 재생성 `1101366`~`1101369`, IP134 `mcn-cnv2l512-rtcropv4real-dsrcraw-r1`도 같은 라벨 설계 결함을 공유하므로 manual stop으로 닫았다. 원인은 crop row를 real-crop으로 고치는 과정에서 `keep_full/minimal_crop` supervision까지 제거한 것이다.

새 source-of-truth는 `product_consistent_gap006_margin005_mixed_realcrop_labels_20260429`이다. `src/scripts/build_mobilecropnet_v4_runtime_crop_policy_labels.py`에 `--preserve_baseline_decisions`를 추가해 baseline/minimal decision은 그대로 보존하고, crop decision row만 baseline과 다른 real-crop winner를 갖도록 보정했다. manifest 기준 train `7,867` rows는 `keep_full 995 / minimal_crop 4,225 / crop 2,647`, val `1,275` rows는 `keep_full 144 / minimal_crop 749 / crop 382`이다. train crop row `2,646 / 2,647`, val crop row `382 / 382`는 baseline과 다른 winner를 갖고, baseline/minimal 보존 row는 train `5,220`, val `893`이다.

이 라벨로 병렬 GPU 축을 재구성했다. 현재 활성 run은 MLP `1101370` `mcn-eva02l-mixedreal-scoremix-ft-r1-20260429`, `1101371` `mcn-eva02l-mixedreal-utilityraw-ft-r1-20260429`, `1101373` `mcn-eva02l-mixedreal-dsrcraw-ft-r1-20260429`, `1101372` `mcn-cnv2h512-mixedreal-warmheads-r1-20260429`, `1101374` `mcn-dinov2l518-mixedreal-warmheads-r1-20260429`, interactive GPU `10.2.4.134`의 `mcn-cnv2l512-mixedreal-dsrcraw-r1-ip134-20260429`이다. ConvNeXtV2-Huge/512, ConvNeXtV2-L/512, DINOv2-L/518은 local timm pretrained weight를 보존하고 `INIT_CHECKPOINT_SKIP_PREFIXES=backbone`으로 EVA02-L set-only head만 warm-start한다. 이 축은 사용자 신규 지시대로 속도보다 품질을 우선하지만, release contract는 계속 `image + target_ar` 입력과 external teacher subject prior 없음이다.

다음 gate 순서는 고정한다. 먼저 train gate에서 route balanced, subject IoU/valid calibration, proposal recall, generated-align, baseline preserve, crop-action, top-return을 본다. 통과한 checkpoint만 embedded/no-prior Product-AR direct, qualitative review pack, latency로 승격한다. 여기서 subject-valid mismatch, low-subject-IoU, final_not_positive, risk_hit, checklist/why/risk agreement failure가 반복되면 public/equal-4가 좋아도 제외한다. direct/qualitative를 통과한 단일 checkpoint에 대해서만 FCDB/CPC/GNMC/equal-4와 GAIC official을 산출하고, checkpoint/config/inference command/evaluation artifacts/qualitative pack/blocker를 하나의 배포 후보 패키지로 고정한다.

### 20.21 2026-04-29 21시대 mixed-real gate 진행 상태

`mcn-srcgate-m005-mixval-20260429`와 `mcn-eva02l-rtcropv3-srcsel-pc-m015-calibbb-srcgate-mixedval-20260429`는 final positive hit와 subject IoU는 높았지만 crop-action/source-action lane이 붕괴해 배포 후보에서 제외했다. all-crop v4 real-crop label은 baseline/minimal supervision을 제거하는 구조적 결함 때문에 train gate source-of-truth에서 제외했고, 새 기준 라벨은 `artifacts/mobilecropnet_v4/quality_first_20260428/product_consistent_gap006_margin005_mixed_realcrop_labels_20260429/`로 고정한다. 이 라벨은 baseline/minimal row를 보존하고 crop row만 real-crop winner로 교체한다.

MLP `1101370`/`1101371`은 epoch1 validation 전 플랫폼 종료로 checkpoint/summary가 없어서 제외했다. `1101372` ConvNeXtV2-Huge/512와 `1101374` DINOv2-L/518 warm-head는 epoch1에서 route/subject/crop-action이 모두 낮아 중단했다. `1101373` EVA02-L decision-source r1은 epoch2에서 route balanced `0.613345`, subject IoU `0.678367`, subject-valid best balanced `0.619465`, proposal recall `0.945534`, generated-align `0.947761`, top-return `0.584950`까지 유지했지만 decision-conditioned baseline preserve `0.814925`와 crop-action `0.360075`가 release gate `0.955 / 0.96`에 못 미쳐 `excluded_train_gate`로 닫았다. IP134 ConvNeXtV2-L mixed-real dsrcraw output은 두 parent process가 같은 output dir에 쓴 collision 때문에 `invalid_output_collision_manual_stop`으로 보존한다.

현재 비중복 병렬 GPU 축은 `1101391` scoremix r2, `1101390` utility/raw r2, `1101397` decision-source source-balanced/margin, `1101398` mild source-mixture/action-calibration, `1101400` late-pair source-balanced, `1101401` dsrcraw best checkpoint 기반 head-only source-balanced, IP134 late-pair run이다. `1101401`은 `1101373`에서 이미 확보한 route/subject/proposal/top-return을 크게 흔들지 않고 decision/return head만 source-balanced로 다시 맞추는 보완축이다. 어떤 축도 train gate를 통과하기 전에는 public/GAIC/full leaderboard row로 승격하지 않는다.

### 20.22 2026-04-29 23시대 risk-conservative 전환과 GPU 정리

mixed-real preserve 라벨은 baseline/minimal supervision을 보존했지만, 검증 bracket에서 baseline-preserve, crop-action, top-return을 동시에 닫지 못했다. `1101391`, `1101390`, `1101397`, `1101398`, `1101394`, IP134 late-pair, `1101400`, `1101401`, `1101409`는 모두 첫 validation 또는 manual gate에서 제외했다. MLP `1101410` `mcn-eva02l-mixedreal-headpaironly-r1-20260429`도 epoch1에서 route `0.580159`, subject IoU `0.697853`, proposal `0.944057`, generated-align `0.944704`는 유지했지만 crop-action `0.675389`, decision-conditioned crop-action `0.413140`, top-return `0.534555`라 release gate와 거리가 커 `excluded_manual_gpu_reclaim`으로 종료했다.

새 source-of-truth는 mixed-real 라벨 위에 `--min_crop_source_margin`을 적용한 risk-conservative 라벨이다. m0.30 라벨은 train `7,867` rows 중 low-margin crop row `1,136`개를 baseline/minimal로 되돌려 `keep_full 1,141 / minimal_crop 5,215 / crop 1,511` 분포를 만들었고, val은 demote `180`개로 `keep_full 167 / minimal_crop 906 / crop 202`가 됐다. m0.50 라벨은 train demote `1,540`, val demote `232`로 더 보수적인 분포를 만든다. 이 라벨은 crop row를 무조건 늘리는 대신, crop이 source/action 관점에서 충분히 우세한 row만 crop supervision으로 남겨 baseline-preserve와 crop-action 간 충돌을 줄이는 목적이다.

현재 active gate는 MLP `1101413` mixed-real headtoprepair 비교축, MLP `1101414` riskcon m0.30 head-only, MLP `1101415` riskcon m0.30 full fine-tune, MLP `1101423` riskcon m0.30 headpair, MLP `1101425` riskcon m0.50 head-only smoke, IP134 `mcn-eva02l-riskcon-m030-headpair-smix015-r1-ip134-20260429`이다. 이 축들은 `return_score_set_refiner_head`, decision-source head, action/source loss를 포함하는 bounded gate이며, 입력 계약은 여전히 `image + target_ar`이고 external teacher subject prior를 쓰지 않는다. IP134에 이미 smix015 run이 활성화된 것을 확인해, 새로 시작한 ConvNeXtV2-Huge/512 quality-first smoke `mcn-cnv2h512-riskcon-m050-setrefiner-smoke-ip134-r1-20260429`는 training 전 `stopped_before_training_due_ip134_existing_active_run`으로 중단했다. 이는 품질 우선 대형 backbone 축을 폐기한다는 뜻이 아니라, 같은 GPU 동시 학습으로 artifact를 오염시키지 않기 위한 운영 정리다.

다음 판단은 best selection score가 아니라 release surface 전체로 한다. train gate에서 route balanced, subject box IoU/valid calibration, proposal recall, generated-align, baseline preserve, crop-action, top-return, decision-source balance가 동시에 닫히지 않으면 Product-AR direct, qualitative review pack, latency, public FCDB/CPC/GNMC/equal-4, GAIC official로 승격하지 않는다. risk-conservative도 실패하면 다음 bounded axis는 단순 threshold/source loss 반복이 아니라, public crop quality와 UCTR subject/geometry를 분리한 high-confidence policy target, return/action dual-surface supervision, 또는 대형 pretrained backbone의 clean single-GPU smoke로 전환한다.

### 20.23 2026-04-29 23시대 riskcon 결과와 품질 우선 backbone 확장

risk-conservative m0.20~m0.50 bracket의 공통 결론은 명확하다. top-return은 회복되지만 crop-action/source-action이 release gate까지 올라오지 않는다. `1101414` m0.30 head-only는 baseline `0.498995`, crop-action `0.557232`, top-return `0.512410`이고, `1101425` m0.50 head-only는 baseline `0.567729`, crop-action `0.540387`, top-return `0.584239`다. `1101415` m0.30 full fine-tune은 top-return `0.641549`까지 올라갔지만 crop-action이 `0.316901`로 더 낮아졌다. `1101430` m0.50 headpair는 top-return `0.741585`, decision-conditioned top-return `0.857897`까지 높지만 crop-action `0.241544`, decision-conditioned crop-action `0.054517`로 final action lane이 사실상 baseline 쪽으로 쏠렸다.

m0.20/m0.40 no-distill bracket도 같은 결론이다. `1101438` m0.20 no-distill은 baseline `0.698980`, crop-action `0.567668`, top-return `0.621901`이고, `1101435` m0.40 no-distill은 baseline `0.752141`, crop-action `0.377281`, top-return `0.696523`이다. IP134 m0.20 soft-mixture/crop-source multiplier 축도 crop-action을 `0.572741`까지밖에 올리지 못했고, decision-conditioned crop-action은 `0.260236`에 그쳤다. 따라서 단순 margin threshold, no-distill, source-mixture strength, crop-source multiplier는 crop branch under-recall을 해소하지 못한다.

사용자 지시에 맞춰 온디바이스 제약을 완화한 품질 우선 backbone 축을 병렬로 확장했다. MLP `1101433`은 ConvNeXtV2-Huge/512 pretrained backbone을 유지하고 EVA02-L head만 `backbone` 제외 warm-start하는 m0.50 set-refiner smoke다. MLP `1101440`은 DINOv2-L/518 pretrained warm-head smoke이며, IP134는 SwinV2-L/512 pretrained warm-head smoke `mcn-swinv2l512-riskcon-m020-crop2-warmheads-r1-ip134-20260429`를 실행한다. 세 축 모두 속도보다 정량/정성 품질을 우선하지만, 최종 입력 계약은 여전히 `image + target_ar`이고 external teacher subject prior를 사용하지 않는다. train gate를 통과하지 못하면 Product-AR direct, qualitative review pack, public/equal-4, GAIC official로 승격하지 않는다.

### 20.24 2026-04-30 02시대 product-topagree hybrid와 큰 입력 품질 우선 축

top-agree crop-only 라벨은 crop-action을 올렸지만 baseline/minimal policy coverage가 `0`이어서 배포 학습 source에서 제외했다. 이를 보완하기 위해 `product_topagree_hybrid_baseline_preserve_labels_20260429`를 만들었다. 이 라벨은 risk-conservative m0.20의 baseline/minimal decision row를 보존하고, crop row에만 UCTR80/public20 top-agree vertical2의 ranking/checklist/why/risk target을 병합한다. 따라서 사용자 지적대로 UCTR teacher 우위 신호를 반영하면서도, 최종 모델이 `image + target_ar`만으로 policy/action/checklist/risk를 일관되게 내는 release contract를 유지한다.

이 라벨의 초기 product-topagree hybrid gate에서는 `1101464`, `1101465`, `1101468`, `1101469`를 모두 제외했다. `1101464` DINOv2-L hardjoint는 route `0.437053`, baseline `0.475417`, crop-action `0.484583`, top-return `0.492917`로 낮다. `1101465` EVA02-L hardjoint는 route `0.613721`, subject IoU `0.641871`, proposal `0.949406`, top-return `0.632917`은 좋았지만 crop-action `0.259583`, baseline `0.749583`이라 action/policy gate가 닫히지 않았다. `1101468`/`1101469` gap-mask 계열은 baseline/top-return 쪽으로 수렴하면서 crop-action이 각각 `0.005000`/`0.000000`까지 붕괴했으므로 catastrophic source/action failure로 본다.

후속 축은 단순 threshold 반복이 아니라 온디바이스 제약을 완화한 큰 입력/강한 pretrained backbone 검증으로 전환했다. `run_mobilecropnet_v4_quality_relaxed_joint.sh`에 `INPUT_SIZE`, `CANDIDATE_K`, `PROPOSAL_Q`, `TOKEN_DIM`, `BACKBONE_NAME`, `IMAGE_MEAN`, `IMAGE_STD` override를 추가했고, 기본값은 기존 재현성을 유지한다. 1차 큰 입력 bracket에서는 `1101472` EVA02-L/560 hardjoint, `1101474` EVA02-L/560 gap-mask, `1101475` EVA02-L/560 baseguard, IP134 EVA02-L baseguard를 모두 제외했다. hardjoint는 baseline/crop-action/top-return이 동시에 낮았고, gap-mask/baseguard는 baseline/top-return을 올리는 대신 crop-action이 `0.000000~0.009167`까지 붕괴해 final action release gate를 닫지 못했다. `1101476` ConvNeXtV2-Huge/640 baseguard도 route `0.352500`, proposal `0.557804`, generated-align `0.561250`, crop-action `0.000000`이라 같은 계열의 반복 실패로 중단했다. `1101481` EVA02-L/560 action-source crop4는 route `0.582500`, subject IoU `0.511494`, generated-align `0.923750`은 일부 회복했지만 subject-valid `0.520625`, proposal `0.910393`, baseline `0.611250`, crop-action `0.175000`, raw top-return `0.570000`으로 종료했다. `1101478`/`1101482`/`1101477`은 selection score가 `0.76~0.77`로 높았지만 crop-action이 각각 `0.313333`/`0.329167`/`0.202083`, baseline이 `0.725833`/`0.630417`/`0.787083`이라 best-selection 단독 승격 금지 기준으로 제외했다. `1101484`는 crop-action `0.432500`까지 올라갔지만 baseline `0.431667`, top-return `0.458333`, raw top-return `0.469444`로 정책 보존과 최종 return이 무너졌고, `1101485` ConvNeXtV2-Huge/512 topagree warm-head도 route `0.483750`, subject IoU `0.483634`, baseline `0.398750`, crop-action `0.266250`으로 제외했다. IP134 `mcn-eva02l-prodhyb-focal2-dcond-thrm050-r1-ip134-20260429`도 selection `0.767942`였지만 baseline `0.643333`, crop-action `0.384583`, raw top-return `0.538750`이라 중단했다.

2026-04-30 03:53 KST 기준으로 이 product-topagree hybrid 축을 다시 정리했다. ConvNeXtV2-Huge/512 topagree source-mixture `1101486`은 route `0.477500`, subject IoU `0.490040`, baseline `0.377500`, crop-action `0.232500`, top-return `0.430000`으로 조기 제외했다. EVA02-L/560 topagree crop-expert init low source-mixture crop2.5 `1101487`은 route `0.632500`은 유지했지만 subject-valid `0.470625`, baseline `0.500000`, crop-action `0.236250`, top-return `0.543750`이라 제외했다. EVA02-L/448 topagree crop scorer 보존 source-only staged fine-tune `1101488`은 route `0.616222`, subject IoU `0.667017`, subject-valid `0.628542`, proposal `0.944757`, generated-align `0.951250`은 통과했지만 baseline `0.181667`, top-return `0.275833`, decision-source acc `0.376250`으로 source/base recall collapse가 명확해 epoch1에서 종료했다. reset decision-source smoke `1101490`/`1101491`은 실험 의도와 달리 `INIT_CHECKPOINT_SKIP_PREFIXES="decision_source_head decision_source_pair_head"` 공백 구분을 학습 스크립트가 파싱하지 못해 decision-source head reset이 실제 적용되지 않았다. 따라서 이 둘은 모델 품질 실패가 아닌 invalid launch로 분리하고 종료했으며, `train_mobilecropnet_v4.py`와 `run_mobilecropnet_v4_quality_relaxed_joint.sh`는 skip-prefix를 쉼표/공백 모두로 파싱하도록 패치했다. 원격 shared project에도 반영했고 로컬/원격 `py_compile` 및 wrapper `bash -n`을 통과했다. corrected reset-source `1101497` 로그에서 `skipped_prefixes=["decision_source_head","decision_source_pair_head"]`, skipped key `8`개를 확인해 실제 head reset이 적용됐음을 검증했다. 이후 중복으로 생성한 `1101500`/`1101501`은 `1101497`/`1101498`과 목적이 겹쳐 validation 전 회수했고 source-of-truth에서 제외했다. IP134 `mcn-eva02l-prodhyb-decisionlogit-mask-nopre-r1-ip134-20260429`는 selection `0.756244`, route `0.618727`, subject IoU `0.651750`, subject-valid balanced `0.580208`, proposal `0.934997`, generated-align `0.932500`, decision-source acc `0.705417`까지는 통과했지만, 실제 active source-mixture surface가 baseline `0.481667`, crop-action `0.383333`, top-return `0.474167`, raw top-return `0.478333`으로 release gate와 멀어 제외했다. decision-conditioned diagnostic top-return `0.712917`은 높지만 crop-action `0.171250`이 낮아, action/policy agreement가 닫힌 모델로 볼 수 없다. MLP `1101493` head-only balanced는 route `0.616222`, subject IoU `0.667017`, subject-valid `0.628542`, proposal `0.944757`, generated-align `0.951250`을 유지했지만 source-mixture baseline `0.084167`, crop-action `0.592500`, top-return `0.214167`, raw top-return `0.233333`으로 source/base lane이 붕괴했다. MLP `1101497` corrected reset-source crop4는 selection `0.760315`, route `0.585390`, subject IoU `0.642011`, subject-valid `0.573958`, proposal `0.936964`, generated-align `0.921250`, decision-conditioned top-return `0.665417`, raw top-return `0.603333`까지는 확보했지만, decision-conditioned baseline `0.791250`과 crop-action `0.238333`이 release gate를 크게 밑돌았다.

2026-04-30 04:28 KST 기준 추가 결과까지 반영하면 `1101498`, `1101503`, `1101504`, IP134 `dcondonly-postmask`도 모두 train gate에서 제외됐다. `1101498`은 route/subject/proposal은 유지했으나 source-mixture crop-action `0.330833`, top-return `0.576250`, baseline `0.644167`로 실패했고, `1101503`은 top-return `0.593750`은 넘겼지만 crop-action `0.299167`, baseline `0.681250`, raw top-return `0.549167`이 낮았다. `1101504`는 source-specific/policy-match를 켰음에도 crop-action `0.433750`, top-return `0.452083`, baseline `0.451250`으로 제외됐고, IP134 postmask는 decision-conditioned crop-action `0.447083`, top-return `0.363750`, baseline `0.301250`으로 active surface가 붕괴했다. 결론은 source threshold, source-mixture alpha, source-specific head, policy-match 단독 residual이 모두 baseline preserve와 crop-action recall을 동시에 닫지 못했다는 것이다.

이 실패 패턴을 반영해 `return_score_policy_source_head`를 새로 추가했다. 이 head는 policy token/candidate token/route/decision/utility/positive/risk/geometry match feature를 보되, source index별로 crop 후보 delta와 baseline 후보 delta를 별도로 출력한다. 마지막 linear는 zero-init이므로 warm-start 직후 기존 checkpoint 출력은 보존된다. 로컬 `py_compile`, wrapper `bash -n`, `tests/test_mobilecropnet_v4.py -q`, forward smoke, 원격 `py_compile`/`bash -n`을 통과했다. 새 active source-of-truth는 MLP `1101525` `polsrc-smix050`, `1101526` `polsrc-srcspec025`, `1101527` `polsrc-dcondhard`와 IP134 `mcn-eva02l-prodhyb-polsrc-policycoupled-smix035-r1-ip134-20260429`이다. 병렬 composite 진단은 full-val r3 `1101522` runtime-crop v3 expert, `1101523` top-agree expert, `1101524` public60 expert가 담당한다. 어느 축도 train gate 또는 composite gate를 통과하기 전에는 Product-AR direct, qualitative review pack, latency, GAIC official, public FCDB/CPC/GNMC로 승격하지 않는다.

2026-04-30 04:40 KST 기준 composite r3는 모두 terminal `Succeeded`지만 release 후보로 승격하지 않는다. `1101522` runtime-crop v3, `1101523` top-agree, `1101524` public60 모두 base route `0.655956`, subject IoU `0.616869`, proposal recall `0.944354`, generated-align `0.949843`은 유지했으나 subject-valid best balanced가 `0.497257`로 낮다. 실제 learned gate에서는 runtime-crop v3 `gap_m0500`이 baseline `0.677725`, crop-action `0.452830`, top-return `0.527502`, public60 `gap_m0500`이 baseline `0.777251`, crop-action `0.316981`, top-return `0.564472`, top-agree `gap_m0500`이 baseline `0.548578`, crop-action `0.637736`, top-return `0.495942`로 서로 다른 tradeoff를 보였다. 반면 oracle gate는 baseline `0.989336`, crop-action `1.000000`, top-return `0.776375~0.783589`라 crop/base expert coverage는 존재한다. 따라서 다음 구조 실험은 composite threshold 반복이 아니라 selector/gate head와 subject-valid calibration을 함께 학습해 oracle coverage를 단일 checkpoint 또는 명시적 내부 inference package로 전이하는 축이어야 한다. 같은 시각 MLP `1101528` `mcn-prodhyb-polsrc-mixed-smix020-r1-20260429`와 `1101529` `mcn-prodhyb-polsrc-risk50-smix035-r1-20260429`도 Running으로 확인되어 policy-source train gate 추적 대상에 추가했다.

이 판단을 바로 실행으로 연결해 `train_mobilecropnet_v4_policy_composite_selector.py`의 기존 selector feasibility path를 사용했다. MLP `1101537` `mcn-sel-risk50-rtcropv3-r1-20260429`, `1101538` `mcn-sel-risk50-topagree-r1-20260429`, `1101539` `mcn-sel-risk50-public60-r1-20260429`는 각각 r3와 같은 base/crop checkpoint를 쓰고, train `2,400` rows와 full val `1,275` rows에서 learned selector threshold sweep을 만든다. 이 축은 배포 후보 checkpoint가 아니라 oracle coverage가 학습 가능한 internal selector로 전이되는지 확인하는 bounded diagnostic이다. 통과하더라도 subject-valid balanced `0.497257` blocker가 남기 때문에, 최종 제품 후보가 되려면 selector packaging과 subject-valid calibration/head 보정, no-prior direct/qualitative/latency/public/GAIC gate를 모두 다시 닫아야 한다.

단, r1 selector 생성 직후 feature key mismatch를 발견했다. 스크립트가 현재 모델 output key인 `pred_subject_valid_logit`/`pred_subject_box` 대신 과거 key `subject_valid_logit`/`subject_box`를 읽고 있어 subject-valid/box feature가 selector 입력에서 사라졌다. `train_mobilecropnet_v4_policy_composite_selector.py`를 패치하고 로컬/원격 `py_compile`와 rsync를 통과시킨 뒤, r1 `1101537`/`1101538`/`1101539`는 종료했다. source-of-truth selector 전이 실험은 r2 `1101540` runtime-crop v3, `1101541` top-agree, `1101542` public60이다.

policy-source 첫 terminal 결과도 바로 반영한다. `1101526` `polsrc-srcspec025`는 MLP official `Succeeded`로 끝났지만 wrapper gate는 `excluded_train_gate`다. route `0.594562`, subject IoU `0.653056`, subject-valid `0.554375`, proposal `0.945841`, generated-align `0.931250`, source-mixture top-return `0.642083`, raw top-return `0.595417`은 유지했지만 crop-action `0.317083`과 baseline preserve `0.735417`이 각각 gate `0.96`/`0.955`에 크게 못 미친다. 즉 policy-source residual이 exact return score를 올리는 데는 도움이 됐지만, 최종 제품 action/source 선택을 닫지 못했다. 따라서 이 checkpoint는 no-prior direct/qualitative/public/GAIC로 승격하지 않는다.

IP134 `polsrc-policycoupled-smix035`도 같은 패턴으로 제외한다. route `0.597641`, subject IoU `0.672057`, proposal `0.934193`, generated-align `0.931000`, source-mixture top-return `0.647667`, raw top-return `0.590667`은 높지만 subject-valid `0.548833`, crop-action `0.265333`, baseline preserve `0.735667`로 train gate를 통과하지 못했다. decision-conditioned top-return `0.688667`도 crop-action이 `0.211333`에 그쳐, 높은 return score가 제품 final action 선택으로 전이되지 않는다.

같은 결론은 `1101525`/`1101527`에서도 반복된다. `1101525`는 route `0.623726`, subject IoU `0.654071`, subject-valid `0.563333`, proposal `0.935870`, generated-align `0.930000`을 유지했지만 source-mixture baseline `0.669583`, crop-action `0.340417`, raw top-return `0.561250`로 제외한다. `1101527`은 decision-conditioned top-return `0.695417`, raw top-return `0.631667`까지 올렸지만 crop-action `0.197083`, baseline `0.840833`으로 action/source gate가 더 멀어졌다. 즉 policy-source residual은 return selector 표면을 강화할 수 있지만, crop/base action policy를 제품 target과 동시에 맞추는 충분조건이 아니다.

남은 policy-source `1101528`/`1101529`도 같은 결론으로 닫았다. `1101528`은 route `0.600975`, subject IoU `0.672485`, subject-valid `0.552333`, proposal `0.947814`, generated-align `0.936000`, top-return `0.672333`, raw top-return `0.606667`까지는 유지했지만 crop-action `0.214333`, baseline preserve `0.792000`이라 release action/source gate와 거리가 컸다. `1101529`는 route `0.626976`, subject IoU `0.669019`, subject-valid `0.552500`, proposal `0.945589`, generated-align `0.938000`, top-return `0.679667`, raw top-return `0.613000`을 보였지만 crop-action `0.240667`, baseline preserve `0.796667`로 제외했다. 따라서 policy-source residual 축은 모두 no-prior direct/qualitative/public/GAIC 승격 없이 종료한다.

selector r2도 배포 후보로 전이되지 않았다. `1101540` runtime-crop v3 selector, `1101541` top-agree selector, `1101542` public60 selector는 모두 base route `0.655956`, subject IoU `0.616869`를 유지했지만 subject-valid best balanced가 `0.497257`로 낮다. threshold `0.40~0.50`의 source balanced accuracy는 약 `0.63`이고, baseline preserve를 `0.90` 안팎으로 올리는 threshold에서는 crop-action이 `0.25~0.30`대로 떨어진다. 반대로 crop-action을 `0.76~0.78`까지 올리는 낮은 threshold에서는 baseline preserve가 `0.43~0.45`로 무너진다. 결론은 two-expert oracle coverage가 존재해도 learned selector와 subject-valid calibration이 함께 닫히지 않으면 제품 release gate로 전이되지 않는다는 것이다.

후속 active diagnostic은 oracle-label selector `1101543`/`1101544`/`1101545`와 실패 checkpoint 기반 no-prior composite probe `1101546`/`1101547`이다. `1101543`~`1101545`는 selector label을 oracle score advantage로 바꾸어 separability 상한을 보는 진단이고, `1101546`/`1101547`은 실패한 policy-source checkpoint를 runtime-crop v3 crop expert와 결합했을 때 no-prior composite가 살아나는지 보는 원인 분리 평가다. 두 축 모두 최종 모델이 아니라 diagnostic이며, 결과가 좋아도 단일 checkpoint 또는 명확한 내부 package로 재구현하고 subject bbox/proposal/route/policy/final crop action/target-AR repair/checklist/why/risk를 같은 `image + target_ar` 계약에서 다시 검증해야 배포 후보가 된다.

사용자 지적의 UCTR teacher 우위와 student target 충돌을 반영해 consensus-preserve label도 원격 CPU에서 재생성했다. `product_topagree_consensus_preserve_labels_20260430`은 public/UCTR top-id agreement가 있는 gap005 top-agree row를 crop target으로 병합하고, policy row는 risk-conservative product-consistent base를 보존하는 목적이었다. 그러나 기존 `product_topagree_hybrid_baseline_preserve_labels_20260429`와 비교하면 train/val winner 차이 `0`, matching target 차이 `0`, candidate score target 차이 `0`이고, line diff도 train `98/7,867`, val `6/1,275`에 그친다. 즉 학습 target은 사실상 동일하므로 이 label로 새 GPU 학습을 만들면 중복 실험이다. artifact는 보존하지만 후속 GPU 축으로 승격하지 않는다.

반대로 `1101528`/`1101529`는 policy-source train gate에서 제외됐지만 top-return/raw-top-return은 가장 강한 편이었다. 그래서 no-prior composite 상한 확인을 `1101548` `polsrc-risk50 + runtime-crop v3`와 `1101549` `polsrc-mixed + runtime-crop v3`로 확장했다. 이 두 run은 새 학습이 아니라 bounded 평가이며, 결과가 좋아도 two-checkpoint diagnostic일 뿐이다. 최종 배포 후보가 되려면 같은 behavior를 단일 checkpoint 또는 명확한 internal package로 구현하고, direct/qualitative/public/GAIC/latency gate를 같은 config로 다시 통과해야 한다.

`1101543`~`1101547` terminal 결과는 two-expert/selector 축을 더 좁혔다. oracle-label selector `1101543` runtime-crop v3, `1101544` top-agree, `1101545` public60은 모두 source balanced accuracy가 최대 약 `0.6568` 수준이고 subject-valid best balanced가 `0.497257`로 고정되어 release gate와 멀다. selector threshold를 낮추면 crop-action은 `0.88~0.92`까지 올라가지만 baseline preserve가 `0.29~0.38`로 붕괴하고, threshold를 높이면 baseline preserve는 `0.91~0.94`까지 올라가지만 crop-action이 `0.25~0.33`대로 떨어진다. 따라서 label을 oracle score advantage로 바꾸어도 selector feature가 제품 action/source 결정을 충분히 분리하지 못한다.

`1101546`/`1101547` no-prior composite도 배포 후보가 아니다. `1101546` `smix050 + runtime-crop v3`는 route `0.640282`, subject IoU `0.609448`을 보였지만 subject-valid `0.481583`이고, best tradeoff에서도 baseline `0.659953`, crop-action `0.513208`, top-return `0.532913`에 그쳤다. `1101547` `dcondhard + runtime-crop v3`는 route `0.623041`, subject IoU `0.618381`, subject-valid `0.480799`, best tradeoff baseline `0.755924`, crop-action `0.426415`, top-return `0.575293`이다. oracle gate 자체는 높으므로 expert coverage는 남아 있지만, 실제 no-prior learned gate는 subject-valid/action-source/top-return을 동시에 만족시키지 못한다.

### 20.25 2026-04-30 05시대 composite 종료와 품질 우선 active 축

2026-04-30 05:42 KST에는 남은 no-prior composite `1101548`/`1101549`까지 terminal 판정했다. 두 run 모두 external teacher subject prior 없이 `image + target_ar` 런타임 계약으로 평가됐지만 release gate를 닫지 못했다. `1101548` risk50 base + rtcropv3 crop은 oracle top-return `0.789901`, oracle base/crop recall `1.0/1.0`로 candidate pool 상한은 있지만, 실제 gate 최선권 `gap_p0750`은 top-return `0.634806`, baseline preserve `0.899289`, crop-action `0.211321`, subject-valid best balanced `0.474922`에 그쳤다. `1101549` mixed base + rtcropv3 crop도 oracle top-return `0.789901`이지만 `gap_p0750`은 top-return `0.647430`, baseline `0.959716`, crop-action `0.101887`, subject-valid `0.474138`이다. 따라서 문제는 crop 후보 상한의 부재가 아니라, 제품 런타임에서 쓸 수 있는 internal source/action gate와 no-prior subject-valid calibration이 동시에 약하다는 점으로 확정한다.

이에 따라 two-expert/gap/selector 반복은 중단하고, 사용자가 새로 지시한 품질 우선 모델군을 실제 GPU 축으로 전환했다. 현재 active source-of-truth는 `1101553` EVA02-L/448 product-hybrid valid/source full run, `1101554` ConvNeXtV2-Huge/512 product-hybrid valid/source full run, IP134 `mcn-qfirst-dinov2l518-prodhyb-setdec-validsrc-smoke-r1-ip134-20260429` DINOv2-L/518 pretrained set-refiner/action-decoder smoke, `1101557` SwinV2-L/384 pretrained set-refiner/action-decoder smoke다. DINO/Swin smoke는 큰 backbone과 입력 크기 증가에 더해 final action selector 구조(`return_score_set_refiner_head`, `return_score_action_decoder_head`, binary/pair source head)를 함께 켠 비중복 축이다. 05:47 KST에는 subject-state action-decoder 변형 `1101561` ConvNeXtV2-Huge/512와 `1101562` EVA02-L/448도 Running으로 전환됐다. 이 둘은 output path가 기존 full run과 달라 중복 output collision이 아니며, final action selector가 subject-valid/subject box state를 더 직접적으로 보게 하는 구조 축이다. 여섯 축 중 train gate를 통과하는 checkpoint만 no-prior Product-AR direct, qualitative review pack, latency, GAIC official, public FCDB/CPC/GNMC/equal-4로 승격한다. 아직 배포 후보 checkpoint는 없다.

같은 시간대에 subject-valid calibration과 final action selector가 분리된 구조 문제를 직접 겨냥하는 패치도 추가했다. `MobileCropNetV4`의 `return_score_action_decoder`에 `return_score_action_decoder_subject_state` 옵션을 넣어, 내부 예측 subject bbox의 geometry, subject-valid logit/probability, 후보별 subject IoU/center/size 관계를 action decoder query, memory, head feature에 전달한다. 외부 teacher subject prior는 쓰지 않으며 최종 입력 계약은 계속 `image + target_ar`이다. 로컬 `py_compile`, wrapper `bash -n`, `tests/test_mobilecropnet_v4.py -q` 10개, interactive A100 CUDA forward smoke를 통과했고 원격 shared storage에 동기화했다. 이 패치 위에서 `1101562` EVA02-L/448 `mcn-qfirst-eva02l448-prodhyb-subjstate-actdec-r1-20260429`와 `1101561` ConvNeXtV2-Huge/512 `mcn-qfirst-cnv2h512-prodhyb-subjstate-actdec-r1-20260429`를 병렬 생성했으며 둘 다 Running으로 전환됐다. 05:55 KST에는 DINOv2-L/518에도 같은 subject-state action-decoder를 붙인 `1101567` `mcn-qfirst-dinov2l518-prodhyb-subjstate-actdec-r1-20260429`를 비중복 MLP smoke로 추가 생성했다. 이 run들은 기존 `1101553`/`1101554` valid/source full run 및 IP134 DINO valid/source smoke와 구분되는 구조 실험이고, train gate 통과 전에는 leaderboard row나 배포 후보로 승격하지 않는다.

후속 release gate 실행 준비도 보강했다. `run_mobilecropnet_v4_quality_first_candidate_eval.sh`에 `release_full` 모드를 추가해, survivor checkpoint가 나오면 같은 checkpoint/config/inference path로 final bundle(public/replay/head/direct), GAIC official, latency, qualitative pack을 순차 생성할 수 있게 했다. 이는 새 학습을 만들지 않고 release gate 평가만 묶는 runner 변경이며, `bash -n` 검증 후 원격 shared storage에 동기화했다.

2026-04-30 06:00 KST에는 `1101567`도 Running으로 전환되어 Pending 오류가 아님을 확인했다. 현재 active GPU 축은 MLP `1101553`, `1101554`, `1101557`, `1101561`, `1101562`, `1101567` 및 IP134 DINO smoke `launcher.pid=991102`이다. 아직 첫 validation/summary는 나오지 않았으므로 배포 후보는 없고, 승격 규칙은 그대로 train gate 통과 후 no-prior direct, qualitative pack, latency, GAIC official, public FCDB/CPC/GNMC/equal-4 순서다. 추가로 후속 재실행 감시 안정성을 위해 `train_mobilecropnet_v4.py`의 `train_status.json`을 step-level로 갱신하는 progress callback을 넣었고 로컬 `py_compile`, `tests/test_mobilecropnet_v4.py -q`, 원격 `py_compile`를 통과했다. 이미 시작된 run은 기존 프로세스라 `train.log`와 terminal `summary.json`을 source of truth로 유지한다.

2026-04-30 06:14 KST에는 과거 pending 오류로 지목됐던 `1098655`를 다시 확인했다. 해당 run은 현재 `Pending`이 아니라 `Terminated`이고, `release_gate_v2 rank_320` 과거 row의 `nvidia-smi` 선행 command 후 실제 학습 없이 종료된 상태다. 이 row는 최신 leaderboard에서 `SSTK UCTR release_gate_v2 / rank_320` checkpoint-source-missing invalid row로 이미 분리되어 있으므로, 현재 GPU를 써서 동일 profile을 중복 재생성하지 않는다. 또한 `run_mobilecropnet_v4_quality_first_candidate_eval.sh`의 `release_full` runner는 survivor 승격 시 `set -u`로 qualitative 단계가 깨지지 않도록 `DIRECT_SELECTION_POLICY`와 route expert 변수 기본값을 보강했고, 하위 `final_bundle`/`gaic_official`/`latency`/`qualitative` 재호출이 같은 `CHECKPOINT`, `METHOD_ID`, `OUTPUT_DIR`, dataset/eval 설정을 상속하도록 export를 추가했다. 로컬/원격 `bash -n`을 통과했다.

2026-04-30 06:20 KST에는 IP134 DINOv2-L/518 smoke를 terminal `excluded_train_gate`로 닫았다. `mcn-qfirst-dinov2l518-prodhyb-setdec-validsrc-smoke-r1-ip134-20260429`는 best selection score `0.645227`이지만 route `0.440000`, subject IoU `0.438240`, subject-valid `0.499608`, proposal `0.914118`, source-mixture crop-action `0.054902`, raw top-return `0.524706`, baseline preserve `0.563137`이라 route/subject/proposal/action/source lane이 동시에 gate 미달이다. 따라서 DINO backbone 반복은 중단하고, 회수된 IP134 A100에는 `mcn-qfirst-eva02l448-polsrc-subjgrad-actdec-smoke-r1-ip134-20260429`를 새로 시작했다. 이 run은 `mcn-prodhyb-polsrc-risk50-smix035`의 높은 top-return/raw-top-return checkpoint를 warm-start하되 pretrained EVA02-L backbone은 보존하고, `return_score_action_decoder_subject_state=1`, `return_score_action_decoder_subject_detach=0`, pairwise labels, stronger action/source margins로 subject-valid/subject-box state가 final action/source selector에 실제로 결합되는지 검증하는 bounded smoke다.

2026-04-30 06:32 KST에는 SwinV2-L detach=1 smoke도 `excluded_train_gate_manual_backfill`로 닫았다. `1101557` `mcn-qfirst-swinv2l384-prodhyb-validsrc-smoke-r1-20260429`는 `summary.json`은 생성했지만 `train_gate_decision.json`이 비어 있어 같은 threshold로 수동 decision을 남겼다. best selection score `0.657230`, route `0.445925`, subject IoU `0.488126`, subject-valid `0.527429`, proposal `0.911505`, generated-align `0.932602`, source-mixture baseline `0.254702`, crop-action `0.313480`, top-return `0.364420`, raw top-return `0.371473`이라 generated-align 외 주요 gate가 모두 미달이다. IP134 EVA02-L subject-gradient smoke는 정확한 EVA02-L large pretrained weight가 local/remote에 없어 Hugging Face timeout으로 invalid stop 처리했고, no-pretrained fallback을 허용하지 않았다. 이후 정확한 local pretrained weight가 staged된 SwinV2-L/384로 `mcn-qfirst-swinv2l384-subjgrad-actdec-smoke-r1-ip134-20260429`를 시작했다. 이 run은 `1101557`과 같은 backbone 계열이지만 `return_score_action_decoder_subject_detach=0`으로 subject state gradient coupling만 분리해 보는 비중복 bounded smoke다.

같은 시각 `1101587` `mcn-qfirst-eva02l448-setwarm-subjgrad-r1-20260429`도 Running으로 확인했다. 이 run은 `BACKBONE_PRETRAINED=0`으로 표시되지만 random backbone 실험이 아니라, 이미 route expert가 embedded된 set-only checkpoint를 init으로 로드하고 `TRAINABLE_PREFIXES`를 return/action/decision head 계열로 제한한 fast head-scope 실험이다. 따라서 사용자 지시의 pretrained backbone 조건을 새로 깨는 scratch run으로 보지 않고, set-only internal upper-bound에서 subject-gradient action decoder가 action/source lane을 회복하는지 보는 별도 비중복 축으로 추적한다.

2026-04-30 06:42 KST에는 `1101587`도 terminal `excluded_train_gate`로 닫았다. best selection score는 `0.723457`이고 route `0.625283`, subject IoU `0.531136`, proposal recall `0.938669`, generated-align `0.958431`, crop-action `1.000000`은 통과했지만 subject-valid best balanced `0.530196`과 source-mixture top-return `0.543922`가 각각 gate `0.55`/`0.58`에 미달했다. baseline target rate가 `0.0`인 set-only head-scope 축이라 baseline preserve는 제품 release evidence로 쓰기 어렵고, subject-valid calibration과 top-return agreement가 닫히지 않았다. 따라서 `1101587`은 Product-AR direct, qualitative review pack, latency, GAIC official, public FCDB/CPC/GNMC/equal-4로 승격하지 않는다. 현재 남은 active source-of-truth는 `1101553`, `1101554`, `1101561`, `1101562`, `1101567`, IP134 SwinV2-L subject-gradient smoke이며, 이들 중 train gate를 통과한 단일 checkpoint만 release_full runner로 승격한다.

2026-04-30 06:52 KST에는 `1101567` DINOv2-L/518 subject-state action-decoder도 `excluded_train_gate`로 닫았다. best selection score `0.641090`, route `0.450980`, subject IoU `0.437686`, subject-valid `0.500000`, proposal recall `0.903007`, generated-align `0.912941`, source-mixture baseline `0.405490`, crop-action `0.123922`, top-return `0.487059`, raw top-return `0.473725`이다. IP134 DINO smoke와 MLP subject-state DINO 모두 같은 route/subject/action/source collapse를 보였으므로 DINO backbone 반복은 현 단계에서 중단한다. 같은 시각 새 MLP `1101588` ConvNeXtV2-Huge/512 top-agree subject-gradient와 `1101589` EVA02-L/448 top-agree subject-gradient가 Running으로 확인됐다. 이 둘은 기존 full/subject-state와 달리 decision-conditioned/source-mixture/policy-source/action-decoder subject-gradient를 동시에 켠 1-epoch 구조축이며, train gate 통과 전에는 release 후보가 아니다.

2026-04-30 06:58 KST에는 `1101553` EVA02-L/448 product-hybrid valid/source full run도 `excluded_train_gate`로 닫았다. best selection score `0.742894`와 route `0.606849`, subject IoU `0.600702`, subject-valid `0.562353`, generated-align `0.938824`, source-mixture top-return `0.591373`만 보면 강해 보이지만, proposal recall `0.913995`, crop-action `0.245490`, raw top-return `0.556078`, baseline preserve `0.650588`이 release threshold를 닫지 못했다. 이 결과는 높은 selection score가 제품 final action/source alignment를 보장하지 않는다는 근거이며, full public/GAIC/direct/qualitative 승격 없이 제외한다. 남은 active source-of-truth는 `1101554`, `1101561`, `1101562`, IP134 SwinV2-L subject-gradient, `1101588`, `1101589`이다.

2026-04-30 07:01 KST에는 `1101554` ConvNeXtV2-Huge/512 valid/source full run을 `excluded_incomplete_terminated_epoch1_gate_fail`로 닫았다. MLP 세션은 epoch2 train `5,900 / 6,400` samples 지점에서 `Terminated`됐고 terminal summary는 없지만, epoch1 validation 기준 best selection score `0.591806`, route `0.432602`, subject IoU `0.493281`, subject-valid `0.540752`, proposal `0.921645`, source-mixture baseline `0.345611`, crop-action `0.274295`, top-return `0.415361`, raw top-return `0.420846`으로 이미 release train gate와 멀었다. 따라서 같은 workload를 재생성하지 않는다. 남은 active source-of-truth는 `1101561`, `1101562`, IP134 SwinV2-L subject-gradient, top-agree subject-gradient `1101588`/`1101589`이다.

2026-04-30 07:03 KST에는 IP134 `mcn-qfirst-swinv2l384-subjgrad-actdec-smoke-r1-ip134-20260429`도 `excluded_train_gate`로 닫았다. best selection score `0.657356`, route `0.445141`, subject IoU `0.486548`, subject-valid `0.518025`, proposal `0.898866`, generated-align `0.916144`, source-mixture baseline `0.333856`, crop-action `0.297022`, top-return `0.409875`, raw top-return `0.394984`이다. detach=0 subject-gradient로도 detach=1 Swin smoke의 route/subject/proposal/action/source collapse가 개선되지 않았으므로, SwinV2-L subject-gradient 반복은 중단한다. 남은 active source-of-truth는 `1101561`, `1101562`, `1101588`, `1101589`이다.

2026-04-30 07:08 KST에는 `1101562` EVA02-L/448 subject-state action-decoder도 terminal `excluded_train_gate`로 닫았다. best selection score `0.723571`, route `0.580376`, subject IoU `0.608393`, source-mixture top-return `0.618039`는 기존 DINO/Swin smoke보다 낫지만, release gate의 핵심인 subject-valid `0.540588`, proposal recall `0.910408`, crop-action `0.246275`, raw top-return `0.576471`, baseline preserve `0.701569`가 모두 부족하다. 즉 내부 subject bbox/valid state를 action decoder에 직접 넣는 구조만으로는 teacher-weak head인 subject-valid calibration과 final action-source alignment가 동시에 회복되지 않았다. 이 checkpoint는 no-prior direct, qualitative review pack, latency, GAIC official, public FCDB/CPC/GNMC/equal-4로 승격하지 않는다. 현재 남은 active source-of-truth는 `1101561` ConvNeXtV2-Huge subject-state full run, `1101588` ConvNeXtV2-Huge top-agree subject-gradient, `1101589` EVA02-L top-agree subject-gradient 세 개다.

2026-04-30 07:12 KST에는 IP134 `mcn-qfirst-eva02l448-setwarm-topagree-srcgrad-r1-ip134-20260429`도 제외했다. 이 run은 이전 active top-agree subject-gradient와 달리 `RETURN_SCORE_SOURCE_MIXTURE_DETACH=0`으로 source-mixture prior까지 gate loss가 흐르게 하고, `decision_source_pair_after_return_deltas=1`로 final return delta 이후의 base/crop gap을 source head가 보게 한 실험이다. route `0.626452`, subject IoU `0.651493`, subject-valid `0.591373`, proposal `0.938200`, generated-align `0.944314`는 좋았지만, source-mixture baseline preserve `0.217255`, crop-action `0.473333`, top-return `0.317647`, raw top-return `0.324706`으로 final product action surface가 붕괴했다. 결론은 smix prior gradient나 source pair late-head만으로는 제품 최종 action/source를 닫지 못한다는 것이다. 남은 source-of-truth는 `1101561`, `1101588`, `1101589`, `1101593`이다.

2026-04-30 07:22 KST에는 추가 durable artifact poll로 `1101561`, `1101589`, `1101593`을 모두 제외했다. `1101561` ConvNeXtV2-Huge subject-state full run은 best selection score `0.650547`, route `0.497649`, subject IoU `0.496102`, subject-valid `0.539577`, proposal `0.927758`, generated-align `0.941223`, baseline preserve `0.250000`, crop-action `0.329154`, top-return `0.350313`, raw top-return `0.373041`로 train gate와 멀다. `1101589` EVA02-L top-agree subject-gradient는 route `0.606583`을 유지했지만 subject-valid `0.539969`, proposal `0.898944`, baseline `0.566614`, crop-action `0.150470`, top-return `0.549373`, raw `0.511755`가 부족하다. `1101593` setwarm top-agree gap005는 route `0.626452`, subject IoU `0.651493`, subject-valid `0.591373`, proposal `0.938200`이 통과했지만 baseline `0.203922`, crop-action `0.478039`, top-return `0.310196`, raw `0.321961`로 source-mixture surface가 붕괴했다.

2026-04-30 07:25 KST에는 MLP 목록에서 `1101597` `mcn-qfirst-eva02l448-setwarm-topagree-pairsrc-dcond-r1-20260429`가 이미 Running임을 확인했다. 이 run은 방금 추가한 `decision_source_pair_subject_state` 구조를 실제로 사용해 내부 subject bbox/valid state를 decision-source pair head에 전달하고, embedded setwarm checkpoint에서 `return_score`, `decision_source_head`, `decision_source_pair_head`만 학습하는 비중복 축이다. 현재 active source-of-truth는 `1101588` ConvNeXtV2-Huge top-agree subject-gradient와 `1101597` EVA02-L setwarm pair-source/decision-conditioned 두 개다. 새 workload는 만들지 않고 두 run의 durable artifact를 폴링한다.

2026-04-30 07:30 KST에는 `1101597`도 `excluded_train_gate`로 닫았다. 이 run은 route `0.626452`, subject IoU `0.651493`, subject-valid `0.591373`, proposal `0.938200`, decision-conditioned top-return `0.734510`을 보였지만 generated-align `0.000000`, decision-conditioned crop-action `0.050196`, baseline preserve `0.907843`으로 gate를 통과하지 못했다. subject state를 decision-source pair head에 넣는 구조가 source ranking은 강하게 만들었지만 generated proposal/action consistency를 끊었으므로, 이 형태의 head-only pair-source 반복은 중단한다. 현재 active source-of-truth는 `1101588` ConvNeXtV2-Huge top-agree subject-gradient 하나다.

2026-04-30 07:32 KST에는 마지막 active였던 `1101588` ConvNeXtV2-Huge top-agree subject-gradient도 `excluded_train_gate`로 닫았다. best selection score `0.591774`, route `0.350588`, subject IoU `0.441861`, subject-valid `0.490980`, proposal `0.909477`, generated-align `0.928627`, source-mixture baseline `0.158431`, crop-action `0.183529`, top-return `0.281569`, raw top-return `0.301961`이다. 대형 pretrained backbone을 완전히 열어도 짧은 top-agree subject-gradient 학습은 route/subject/proposal/action/source를 동시에 망가뜨렸으므로, 이 checkpoint는 no-prior direct, qualitative review pack, latency, GAIC official, public FCDB/CPC/GNMC/equal-4로 승격하지 않는다. 현재 survivor checkpoint는 없으며, 다음 축은 generated proposal/action consistency를 보존한 상태에서 final source selector만 안정화하는 구조/라벨 수정을 우선한다.

2026-04-30 07:43 KST에는 후속 active source-of-truth를 6개로 제한해 다시 잠갔다. `1101599`는 ConvNeXtV2-B/448 top-agree route-polish로 route/image-residual head만 보정하고, `1101600`은 ConvNeXtV2-B/448 mixed-real source-polish로 policy/source/return/action decoder를 함께 연다. `1101601` top-hybrid decision-conditioned cropboost, `1101602` top-hybrid low source-mixture, `1101603` risk20 decision-conditioned cropboost, IP134 `risk40-lowmix012`는 set-only embedded EVA02-L checkpoint에서 return/source/action 계열만 빠르게 보정하되 generated proposal alignment loss를 유지한다. 설계 의도는 `1101597`처럼 source ranking top-return만 올리고 generated proposal/action consistency를 끊는 실패를 막는 것이다. 이 6개 중 train gate를 통과한 단일 checkpoint가 나오면 같은 checkpoint/config/inference command로 Product-AR direct, qualitative taxonomy, latency, GAIC official, public/equal-4를 순차 실행한다. 통과 전에는 배포 후보도 새 Unified Final Leaderboard row도 없다.

2026-04-30 07:48 KST에는 `1101603` risk20 decision-conditioned cropboost를 제외했다. best selection score는 `0.752478`이고 route `0.626452`, subject IoU `0.651493`, subject-valid `0.591373`, proposal `0.938507`, generated-align `0.949804`, decision-conditioned top-return `0.662745`, raw top-return `0.589020`으로 geometry/proposal/ranking 일부는 좋다. 그러나 제품 gate가 요구하는 decision-conditioned crop-action `0.170588`과 baseline preserve `0.770980`이 각각 `0.96`/`0.955` 기준에 크게 못 미친다. 이는 generated proposal consistency는 보존했지만 final source/action selector가 여전히 crop lane과 baseline lane을 동시에 맞추지 못한다는 evidence다. 따라서 direct/qualitative/GAIC/public 승격 없이 제외하고, 남은 active source-of-truth는 `1101599`, `1101600`, `1101601`, `1101602`, IP134 risk40 다섯 개다.

2026-04-30 07:54 KST에는 `1101601` top-hybrid decision-conditioned cropboost도 제외했다. best selection score `0.752619`, route `0.626452`, subject IoU `0.651493`, subject-valid `0.591373`, proposal `0.938200`, generated-align `0.944314`, decision-conditioned top-return `0.686667`, raw top-return `0.597255`로 return ranking 일부는 좋았지만, decision-conditioned crop-action `0.158824`와 baseline preserve `0.812157`이 release gate를 닫지 못했다. 즉 generated proposal consistency를 유지해도 decision-conditioned cropboost 단독으로는 crop lane recall과 baseline lane 보존을 동시에 맞추지 못한다. 이 checkpoint는 direct/qualitative/latency/GAIC/public 승격 없이 제외하고, 남은 active source-of-truth는 `1101599`, `1101600`, `1101602`, IP134 risk40 네 개다.

2026-04-30 07:55 KST에는 `1101602` top-hybrid low source-mixture도 제외했다. route `0.626452`, subject IoU `0.651493`, subject-valid `0.591373`, proposal `0.938200`, generated-align `0.944314`는 유지했지만, source-mixture baseline preserve `0.076078`, crop-action `0.498039`, top-return `0.214902`, raw top-return `0.229020`으로 제품 final source surface가 붕괴했다. decision-conditioned diagnostic은 top-return `0.727843`, raw `0.631373`을 보였으나 crop-action `0.067843`이라 배포 후보로 승격할 수 없다. 남은 active source-of-truth는 `1101599`, `1101600`, IP134 risk40 세 개다.

2026-04-30 08:08 KST에는 IP134 `mcn-qfirst-eva02l448-risk40-lowmix012-r1-ip134-20260430`도 `excluded_train_gate`로 닫았다. best selection score `0.730298`, route `0.626452`, subject IoU `0.651493`, subject-valid `0.591373`, proposal `0.938613`, generated-align `0.951373`, decision-conditioned top-return `0.826667`, raw top-return `0.638824`는 강하지만, 실제 active source-mixture surface는 baseline preserve `0.285098`, crop-action `0.296078`, top-return `0.365098`, raw top-return `0.343529`다. 이 결과는 ranking diagnostic이 좋아도 runtime final source/action surface가 닫히지 않으면 배포 후보가 될 수 없다는 점을 다시 확인한다.

2026-04-30 08:12 KST에는 source-gate 구조를 실제 remote shared tree에 다시 반영했다. `return_score_source_gate`는 base-source와 crop-source 안의 후보 순위는 유지하되, policy token, best-base/best-crop candidate token, route probability, decision logits, return/utility/positive/risk gap, 내부 subject bbox/valid state를 결합해 row-level base/crop source gate를 별도 학습한다. 최초 r1 MLP `1101613`/`1101614`/`1101615`는 stale remote code 때문에 source-gate 패치가 반영되지 않았고 산출물 없이 종료됐으므로 invalid launch로 분리했다. 이후 `model.py`, train/eval/infer/wrapper/backfill 파일을 remote shared tree에 재동기화하고 원격 `py_compile`/`bash -n`을 통과했다.

2026-04-30 08:20 KST에는 source-gate r2도 label filename이 stale path를 가리킨 invalid launch였음을 확인했다. `1101617`은 `product_consistent_gap006_margin005_riskconservative_m020_labels_20260429/train.jsonl`, `1101618`은 `product_topagree_hybrid_baseline_preserve_labels_20260429/train.jsonl`, `1101619`는 `product_consistent_gap006_margin005_riskconservative_m040_labels_20260429/train.jsonl`을 참조해 실패했다. 실제 파일명은 각각 `train_riskconservative_m020.jsonl`, `train_product_topagree_hybrid.jsonl`, `train_riskconservative_m040.jsonl`이므로, r2는 모델 결과가 아니라 invalid launch로 닫고 corrected r3 `1101621`/`1101622`/`1101623`을 재등록했다. 중복 등록된 후발 Pending `1101624`/`1101625`/`1101626`은 즉시 종료했다. 08:20 KST 기준 `1101621`/`1101622`는 Running, `1101623`은 Pending이며 10분 Pending rule로 관리한다.

같은 시각 headdirect single-checkpoint 후보도 별도 release gate로 확장했다. `mcn-q-cnv2b448-topagree-imgresroute-r1-routeembed-headdirect-r1-20260430`은 embedded route expert를 외부 prior 없이 checkpoint 내부에 포함한 후보이며, head audit은 route balanced `0.571399`, subject IoU `0.523292`, subject-valid balanced `0.706891`, proposal recall@5 `0.951563`으로 numeric gate를 넘었다. 그러나 top1_not_positive `69.03%`, route_mismatch `43.47%`, subject_valid_mismatch `28.44%`, low_subject_iou `27.03%`가 남아 있어 배포 후보로 확정하지 않는다. crop-only Product-AR direct는 final positive hit `0.961875`, target-AR compatible `1.0`, proposal recall@5 `0.971875`, route acc `0.565313`까지 확인했고, IP134에서 `mcn-q-cnv2b448-topagree-imgresroute-r1-routeembed-releasefull-rerun2-20260430` public/GAIC/latency/qualitative 평가를 실행 중이다.

2026-04-30 08:20 KST 기준 현재 active source-of-truth는 `1101599` ConvNeXtV2-B/448 top-agree route-polish, `1101600` ConvNeXtV2-B/448 mixed-real source-polish, `1101609` mixed-real decision-conditioned source-polish smoke, `1101610` risk40 decision-conditioned source-polish smoke, `1101612` risk40 lowmix006 source-polish r2, source-gate r3 `1101621`/`1101622`/`1101623`, IP134 headdirect release_full이다. train gate 또는 release_full gate 통과 checkpoint가 나오기 전에는 Product-AR direct, qualitative pack, latency, GAIC official, public/equal-4로 최종 승격하지 않는다. 후속 보고서와 release 판단 문구는 모두 국문으로 유지한다.

2026-04-30 08:30 KST에는 headdirect release_full의 실행 순서를 바꿨다. 기존 `release_full` wrapper는 public/equal-4 export를 가장 먼저 수행하는데, IP134 serial export가 `1,920 / 62,511` rows에 `467s`로 너무 느렸고 이 후보는 아직 head taxonomy blocker가 남아 있다. 그래서 public export를 중단하고 `release_gate_reorder_status.json`에 기록한 뒤, 같은 checkpoint로 no-prior Product-AR direct test, qualitative pack, GAIC official, latency를 먼저 실행하도록 재정렬했다. 08:32 KST 기준 direct test는 `1,200 / 4,817` images까지 정상 진행 중이다. catastrophic qualitative blocker가 없을 때만 public/equal-4를 sharded workload로 재개한다.

같은 시각 `1101599` route-polish r2도 정리했다. 이 run은 epoch1 train `5,400 / 12,366` samples에서 platform termination으로 끊겼고 `summary.json`, `best.pt`, `last.pt`가 없다. 따라서 모델 실패가 아니라 `incomplete_no_checkpoint`로 분리한다. 동일한 장시간 설정을 반복하면 같은 termination 위험이 크므로, `MAX_TRAIN_ROWS=3600`, `EPOCHS=1`의 bounded route-polish smoke r3 `1101630`을 생성했다. r3가 route balanced와 기존 action/source surface를 함께 통과할 때만 headdirect/direct/qualitative gate로 승격한다.

2026-04-30 08:24 KST에는 `1101599` route-polish를 `excluded_incomplete_terminated_no_summary`로 분리했다. 이 run은 epoch1 train step `450 / 500`에서 MLP 세션이 종료됐고 `summary.json`, `train_gate_decision.json`, checkpoint가 없다. 또한 route head hidden dimension mismatch로 route/head 계열 `42`개 key가 shape skip되어, route-polish 성능 자체를 검증한 결과로 쓰기 어렵다. 따라서 동일 workload를 즉시 반복하지 않고, 남은 active 학습 source-of-truth는 `1101600`, `1101609`, `1101610`, `1101612`, `1101621`, `1101622`, `1101623`으로 줄인다.

2026-04-30 08:36 KST에는 active source-of-truth를 다시 점검했다. `1101623`은 Pending 오류가 아니라 Running/IP 할당 상태로 전환됐고, `1101600`, `1101609`, `1101610`, `1101612`, `1101621`, `1101622`, `1101623`, `1101630` 모두 GPU workload가 살아 있다. durable status 기준 `1101600`은 val `900` samples, `1101609`/`1101610`은 val `600` samples, `1101612`는 val `300` samples까지 진행했고, source-gate r3는 각각 train `900/825/750` samples까지 증가했다. `1101630` route-polish bounded r3는 route-expert distill load 단계라 아직 checkpoint/gate 판단 전이다. 따라서 현재는 GPU 재생성 없이 summary/gate 파일을 폴링하고, `summary.json`은 생겼지만 `train_gate_decision.json`이 없을 때만 backfill을 수행한다.

같은 시각 IP134 headdirect release-full은 public/equal-4 serial export를 중단하고 direct/qualitative/GAIC/latency 우선순위로 재정렬한 상태다. direct test는 `1,200 / 4,817` images까지 정상 증가했으며, 이전 public export `2,304` rows는 stale partial로 분리한다. 이 후보는 numeric head gate만으로는 배포 후보가 아니고, qualitative taxonomy와 release-full gate가 catastrophic blocker 없이 닫힌 뒤에만 public/equal-4 sharded 재개 및 Unified Final Leaderboard row 승격을 허용한다.

리더보드 누락도 다시 확인했다. 38-row quality-first snapshot에서 주요 public/GAIC/equal-4 수치가 실제로 비어 있는 row는 `SSTK UCTR release_gate_v1 / plus_384`, `SSTK UCTR release_gate_v2 / rank_320`, `SSTK UCTR release_gate_v2_finetune_subjectpatch / rank_320` 세 개이며 모두 checkpoint/eval artifact가 없는 invalid row다. 완료 row는 `fcdb_primary`, `cpc_primary`, `gnmc_primary`, `official_top1_mos`, `official_srcc`, `official_accw4_top10`, `gaic_primary`, `equal4_raw_mean`, `equal4_zscore`를 가진다. 따라서 현재 남은 작업은 공백 채우기용 재평가가 아니라, active run에서 배포 가능한 단일 checkpoint/config/inference command/evaluation artifacts/qualitative pack/blocker를 닫는 것이다.

2026-04-30 08:42 KST에는 `1101609`와 `1101610`이 terminal `excluded_train_gate`로 닫혔다. `1101609` mixed-real decision-conditioned source-polish는 route balanced `0.491879`, decision-conditioned top-return/raw `0.243976/0.243976`, baseline preserve `0.058246`으로 route와 return/baseline이 모두 gate 미달이다. `1101610` risk40 decision-conditioned source-polish는 route balanced `0.491879`, decision-conditioned crop-action `0.804517`, top-return/raw `0.214303/0.245140`, baseline preserve `0.077757`이다. crop-action 회수력만 일부 보였고, final source/action/return 및 baseline preserve가 동시에 닫히지 않았으므로 release-full 승격 없이 제외한다. 남은 active 학습 source-of-truth는 `1101600`, `1101612`, `1101621`, `1101622`, `1101623`, `1101630`이다.

2026-04-30 08:45 KST에는 `1101600`과 `1101612`도 닫았다. `1101600` mixed-real source-polish는 platform termination 뒤 manual backfill된 gate 기준 route balanced `0.499349`, crop-action `0.867802`, top-return/raw `0.471740/0.471740`, baseline preserve `0.459587`로 제외한다. `1101612` risk40 lowmix006 source-polish r2는 MLP official `Succeeded`지만 route balanced `0.491879`, source-mixture crop-action `0.850467`, top-return/raw `0.174238/0.228569`, baseline preserve `0.025231`로 final source surface가 붕괴했다. 따라서 source-polish bracket은 모두 Product-AR direct/qualitative/GAIC/public/equal-4 승격 대상이 아니며, 남은 학습 source-of-truth는 source-gate r3 `1101621`, `1101622`, `1101623`과 route-polish bounded r3 `1101630`이다.

2026-04-30 08:55 KST에는 IP134 headdirect `release_full-rerun2`도 `excluded_release_gate`로 닫았다. 같은 checkpoint/config로 no-prior Product-AR direct, qualitative pack, GAIC official, A100 latency를 완료했고, 판정 JSON은 `artifacts/mobilecropnet_v4/quality_first_20260428/final_eval_bundle/mcn-q-cnv2b448-topagree-imgresroute-r1-routeembed-releasefull-rerun2-20260430/release_gate_decision.json`이다. Direct final hit `0.967822`, final IoU-to-best-positive `0.846002`, target-AR compatibility `1.0`, proposal recall@5 `0.974465`, A100 mean latency `41.736 ms`는 강하지만, qualitative failure summary가 subject_valid_mismatch `29.29%`, subject_iou_low `22.63%`, risk_hit `3.86%`이고 subject IoU `0.545387`, route acc `0.549305`, GAIC top1 MOS `3.741880`, SRCC `0.409028`, Accw4@10 `0.349432`, GAIC primary `0.569763`이라 release gate를 통과하지 못한다. 기존 public export partial `2,304 / 62,511` rows는 stale partial로 분리하며, catastrophic qualitative blocker가 이미 확인됐으므로 full public/equal-4 shard 재개와 Unified Final Leaderboard row 승격은 보류한다. 현재 남은 active source-of-truth는 source-gate r3 `1101621`/`1101622`/`1101623`과 route-polish bounded r3 `1101630` 네 개뿐이며, 모두 검증 단계에 들어갔고 `summary.json`/`train_gate_decision.json`은 아직 생성되지 않았다.

2026-04-30 08:50 KST에는 IP134 headdirect release-full도 terminal 완료로 닫았다. `mcn-q-cnv2b448-topagree-imgresroute-r1-routeembed-releasefull-rerun2-20260430`은 embedded route expert를 포함한 single-checkpoint/no-prior 후보이고, TEST direct에서 final positive hit `0.967822`, final IoU-to-best-positive `0.846002`, proposal recall@5 `0.974465`, target-AR compatible `1.0`, route acc `0.549305`, subject IoU `0.545387`, subject valid acc `0.707079`, risk hit `0.038613`을 기록했다. GAIC official은 top1 MOS `3.741880`, SRCC `0.409028`, Accw4@10 `0.349432`, GAIC primary `0.569762`이고, A100 latency는 candidate 64/86 모두 mean 약 `41.74ms`다. 하지만 qualitative review pack의 subject_valid_mismatch `29.29%`, subject_iou_low `22.63%`, risk_hit `3.86%`, final_not_positive `3.22%`, proposal_top5_miss `2.55%`가 release blocker다. 즉 crop geometry와 target-AR repair는 강하지만 subject bbox/valid calibration failure가 반복되므로, 이 checkpoint는 배포 후보가 아니며 public/equal-4 sharded 재개와 Unified Final Leaderboard row 승격을 하지 않는다.

같은 시점의 artifact 위치도 정리했다. workflow prompt 파일은 local tree에 없고, recovery source-of-truth는 `artifacts/mobilecropnet_v4/unified_recovery_20260424/subject_box_integration_report.json`, `runtime_no_prior_rerun_report.json`, `figure_prompt_set.json`, `final_deployment_leaderboard_refresh_20260428_quality_first_snapshot/final_deployment_leaderboard_quality_first_snapshot.json`이다. headdirect 최신 release-full bundle은 remote shared storage의 `artifacts/mobilecropnet_v4/quality_first_20260428/final_eval_bundle/mcn-q-cnv2b448-topagree-imgresroute-r1-routeembed-releasefull-rerun2-20260430/`를 source of truth로 유지한다.

2026-04-30 08:57 KST에는 headdirect failure 원인을 기준으로 후속 실험을 재정렬했다. direct TEST predictions에서 subject-valid threshold/policy sweep을 계산하면 `route_and_conf@0.30`은 balanced acc `0.706373`, mismatch `29.29%`이고, `route_or_conf@0.85`는 mismatch `24.87%`까지 낮추지만 balanced acc `0.694440`으로 gate 아래다. 후처리만으로는 subject valid calibration과 qualitative taxonomy를 동시에 닫지 못하므로, 같은 embedded checkpoint에서 subject calibration smoke 두 개를 시작했다. IP134 `mcn-cnv2b-headdirect-subjvalid-cal-r1-ip134-20260430`은 PID `1043234`로 valid-only heads를 학습하고, MLP `1101642` `mcn-cnv2b-headdirect-subjspatial-cal-r1-20260430`은 subject bbox/spatial/valid stack을 함께 연다. 두 run 모두 외부 teacher subject prior 없이 기존 checkpoint의 내부 subject head를 보정하는 bounded smoke이며, direct/qualitative 개선이 확인될 때만 release gate 재평가로 승격한다.

2026-04-30 09:00 KST에는 source-gate r3 중 `1101621`과 `1101622`도 terminal train gate에서 제외했다. `1101621` risk20 crop125 source-gate는 route `0.626452`, subject IoU `0.651493`, subject-valid `0.591373`, proposal `0.938507`, generated-align `0.952157`을 보존했지만 source-gate crop-action `0.321961`, top-return `0.496863`, raw top-return `0.455294`, baseline preserve `0.499608`이다. `1101622` top-hybrid source-gate도 route/subject/proposal/generated-align은 `0.626452`/`0.651493`/`0.938200`/`0.949020`으로 비슷하지만 crop-action `0.324314`, top-return `0.494510`, raw top-return `0.453725`, baseline preserve `0.496078`이라 동일하게 실패했다. 따라서 source-gate 구조만으로는 crop lane과 baseline lane을 동시에 맞추지 못했고, 남은 r3 source-of-truth는 `1101623`, route-polish `1101630`, subject calibration `1101642`와 IP134 valid-only smoke다.

2026-04-30 09:04 KST에는 subject calibration 실행 설정도 정정했다. IP134 valid-only smoke와 MLP `1101642` subject-spatial r1은 headdirect checkpoint의 route hidden mult `0.75`가 아니라 wrapper 기본 `0.70`으로 모델을 만들면서 route head key가 shape-skip됐다. 따라서 이 둘은 release evidence가 아니라 ablation으로만 보존하고, `1101642`는 중단했다. corrected subject-spatial r2 `1101645`는 `ROUTE_HEAD_HIDDEN_MULT=0.75`, `ROUTE_IMAGE_RESIDUAL=1`을 명시해 재등록했고, IP134 corrected valid-only r2 `mcn-cnv2b-headdirect-subjvalid-cal-r2-ip134-20260430`도 PID `1051156`으로 시작했다. 두 corrected run만 subject calibration 판단에 사용한다.

2026-04-30 09:06 KST에는 source-gate r3 마지막 축 `1101623`도 닫았다. risk40 source-gate는 route `0.626452`, subject IoU `0.651493`, subject-valid `0.591373`, proposal `0.938613`, generated-align `0.952941`, top-return `0.739216`까지 올렸지만 crop-action `0.083922`, raw top-return `0.574118`, baseline preserve `0.819608`로 release gate를 통과하지 못했다. 이로써 source-gate r3는 모두 실패이며, 남은 active source-of-truth는 route-polish bounded r3 `1101630`, corrected subject-spatial `1101645`, IP134 corrected valid-only r2다.

2026-04-30 09:18 KST에는 route-polish bounded r3 `1101630`도 `excluded_train_gate`로 닫았다. route balanced `0.473411`이 gate `0.52` 아래였고, subject IoU `0.527490`, subject-valid best balanced `0.678317`, proposal recall `0.938755`, generated-align `0.946517`, crop-action `0.972458`, top-return `0.639018`은 일부 통과했지만 route taxonomy failure 때문에 release 승격 대상이 아니다.

같은 시각 corrected subject-spatial `1101645`는 학습/route-expert embedding을 완료했고 head audit이 `1,596 / 1,600` image까지 진행 중이다. IP134 corrected valid-only r2는 head audit 완료 후 direct 단계에 들어갔다. 두 run 모두 headdirect release-full의 subject valid/bbox qualitative blocker를 겨냥한 bounded calibration smoke이며, direct/qualitative 결과가 확인되기 전에는 새 leaderboard row나 배포 후보가 아니다.

source-gate r3 실패 원인은 `source_gate_logit`이 return-logit 기반 간접 손실만 받고 직접 base/crop BCE/margin supervision을 받지 않았다는 점으로 재정의했다. 이에 따라 `src/mobilecropnet_v4/model.py`, `src/scripts/train_mobilecropnet_v4.py`, `src/scripts/run_mobilecropnet_v4_quality_relaxed_joint.sh`, `tests/test_mobilecropnet_v4.py`에 `SOURCE_GATE_SUPERVISION_WEIGHT`, balanced BCE, focal, margin 계열 옵션과 smoke test를 추가했고 로컬/원격 검증을 통과했다. 후속 비중복 r4 workload는 `1101652` risk20, `1101650` top-hybrid, `1101651` risk40이며, 세 run은 source-gate 직접 지도 구조 검증으로 병렬 Pending 상태에서 10분 규칙으로 관리한다.

2026-04-30 09:23 KST 정정: r4 `1101650`/`1101651`/`1101652`는 학습 전 CLI 인자 오류로 종료됐다. `SELECTION_METRIC=best_selection_score`는 trainer enum에 없으므로 모델 결과가 아니며, 같은 가설을 `SELECTION_METRIC=release_gate_joint`로 바꾼 r5 `1101659` risk20, `1101661` top-hybrid, `1101660` risk40으로 재등록했다. 이 세 run이 현재 source-gate 직접 지도 판단 대상이다.

2026-04-30 09:24 KST에는 corrected headdirect subject calibration 두 축도 배포 후보에서 제외했다. MLP `1101645` subject-spatial r2는 final hit `0.956250`, route acc `0.527500`, subject IoU `0.559585`, subject-valid acc/neg/pos `0.655000/0.756198/0.611111`이지만 qualitative subject_valid_mismatch `34.50%`, subject_iou_low `25.13%`가 남았다. IP134 valid-only r2는 final hit `0.959375`, subject IoU `0.522148`, subject-valid acc/neg/pos `0.693125/0.667355/0.704301`이고 subject_valid_mismatch `30.69%`, subject_iou_low `24.13%`다. valid-only는 mismatch를 일부 줄였지만 headdirect release-full의 catastrophic subject calibration failure를 닫지 못했으므로 GAIC/public/latency 확장 대상이 아니다.

2026-04-30 09:26 KST에는 source-gate 직접 지도 r5도 invalid launch로 정리했다. 같은 run name으로 생성된 후발 중복 `1101663`/`1101665`와 pending 중복 `1101660`은 output collision 방지를 위해 종료했고, 남은 r5 `1101659`/`1101661`/`1101667`은 init checkpoint route head `[538]` 대비 실행 모델 route head `[576]`로 `21`개 shape-skip이 발생했다. 이는 `ROUTE_HEAD_HIDDEN_MULT=0.75` 실행이 init checkpoint의 `0.70` 설정과 맞지 않은 문제라 모델 결과로 보지 않는다. 수정 r6는 `ROUTE_HEAD_HIDDEN_MULT=0.70`을 명시한 `1101669` risk40, `1101670` top-hybrid, `1101671` risk20이며, 등록 직후 Pending 상태라 10분 Pending rule로 관리한다.

2026-04-30 09:27 KST에는 public60 subject calibration을 유효 smoke로 유지한다. public60 base row는 direct final hit `0.967615`, subject IoU `0.708175`, GAIC primary `0.570683`, public FCDB/CPC/GNMC `0.746991/0.829701/0.733964`까지 완료됐지만 subject-valid negative `0.468610`과 qualitative subject_valid_mismatch `25.78%`가 release blocker였다. threshold sweep은 negative acc와 mismatch를 동시에 닫지 못했으므로, checkpoint 내부 subject head를 학습하는 `1101664` valid-only와 `1101666` subject-spatial을 실행 중이다. 두 run 모두 public60 checkpoint를 `loaded_keys=614`, `skipped_shape_key_count=0`으로 정상 로드했으므로 현재 release 판단에 유효하다.

2026-04-30 09:33 KST에는 r6 source-gate를 효율 문제로 성능 판단 전 분리하고 r7로 재실행했다. r6 `1101669`/`1101670`/`1101671`은 `ROUTE_HEAD_HIDDEN_MULT=0.70`으로 shape mismatch는 해결했지만, head-scope source-gate 학습에서 route/backbone이 frozen인데 route expert distill load가 켜져 있었다. 이후 실수로 `ROUTE_HEAD_HIDDEN_MULT=0.75`를 넣어 생성한 `1101674`/`1101675`와 IP134 r6는 shape-skip 위험 때문에 즉시 중단했다. 실제 유효 r7 source-of-truth는 IP134 `mcn-qfirst-eva02l448-risk20-srcgate-sup-r7-ip134-20260430`, MLP `1101676` top-hybrid, MLP `1101677` risk40, MLP `1101678` risk20이며, 세 run 모두 `ROUTE_HEAD_HIDDEN_MULT=0.70`, `ROUTE_EXPERT_DISTILL_WEIGHT=0`, `ROUTE_EXPERT_DISTILL_HARD_WEIGHT=0`으로 shape-skip 없이 학습됐다. 이후 같은 run name/output dir로 생긴 duplicate `1101680`/`1101681`은 output collision을 막기 위해 중단했고, r7 판단은 `summary.json`/`train_gate_decision.json`을 source of truth로 고정한다.

2026-04-30 09:43 KST r7 결과도 source-gate 직접 지도 가설을 배포 후보로 만들지 못했다. risk20 IP134는 route balanced `0.628999`, subject IoU `0.651493`, subject-valid best balanced `0.591373`, proposal `0.938507`, generated-align `0.945098`, top-return/raw `0.696078/0.604706`이지만 baseline preserve `0.850196`, crop-action `0.079216`이다. `1101676` top-hybrid는 crop-action `0.116471`, baseline `0.799216`, raw top-return `0.576471`로 실패했고, `1101677` risk40은 baseline `0.960784`, top-return/raw `0.828235/0.641176`을 통과했지만 crop-action `0.011765`로 crop lane이 사실상 닫혔다. `1101678` risk20도 crop-action `0.111765`, baseline `0.802745`, raw top-return `0.579216`으로 gate를 닫지 못했다. 따라서 `source_gate_logit` 직접 BCE/margin 손실만 추가하는 방식은 base/crop source를 동시에 보존하는 충분조건이 아니며, 다음 구조 개선은 action-source 후보 선택을 binary source gate가 아니라 crop 후보 상위 선택과 joint로 묶는 방향으로 넘어간다.

같은 시각 public60 subject calibration도 판정을 업데이트했다. `1101664` valid-only train summary는 route balanced `0.638232`, subject IoU `0.637812`, subject-valid acc/balanced/best-balanced `0.748333/0.498274/0.625119`이고, `1101666` subject-spatial은 route balanced `0.637399`, subject IoU `0.640340`, subject-valid acc/balanced/best-balanced `0.765417/0.506786/0.548750`이었다. 그러나 head audit `1,600` image 기준으로는 `1101664`가 subject IoU `0.478556`, valid acc/negative/positive `0.722500/0.657025/0.750896`, `1101666`이 subject IoU `0.481314`, valid acc/negative/positive `0.726250/0.743802/0.718638`에 그쳤다. 기존 public60 base의 subject IoU `0.708175`보다 크게 낮으므로 release gate를 실패한 것으로 확정하고, direct eval이 약 `12 / 1,600` image partial인 상태에서 두 workload를 `Terminated`로 회수했다. 부분 direct output은 release evidence가 아니라 head-gate 실패 후 중단된 artifact다.

2026-04-30 09:52 KST에는 raw035 near-miss checkpoint를 후속 source-of-truth로 올렸다. `mcn-q-eva02l448-dsrc-genalign-raw035-setonly-ip134-r1-20260429`는 route `0.625289`, subject IoU `0.664631`, subject-valid best-balanced `0.609726`, proposal `0.938929`, generated-align `0.948254`, crop-action `0.972776`, top-return `0.612219`를 통과했지만 raw top-return `0.557980 < 0.58`만 실패했다. direct/qualitative smoke `1101690`과 보정 run `1101691` top-hybrid source-gate, `1101692` risk40 source-gate, `1101693` risk20 source-gate, `1101694` top-agree retcal을 병렬 생성했다. 이 batch는 source-gate를 반복하는 것이 아니라, crop/action lane이 살아 있는 checkpoint에서 raw top-return과 source/action 균형이 닫히는지 검증하는 bounded recovery다.

2026-04-30 09:56 KST에는 raw035 보정 r1을 즉시 정정했다. `1101691`~`1101694`는 action decoder에 없던 subject-state feature를 켜면서 action decoder 핵심 3개 key가 shape-skip되어 warm-start 보정 evidence로 쓸 수 없다. 네 run은 학습 초기에 중단했고, action decoder subject-state를 끈 r2 `1101697` retcal, `1101698` risk40 source-gate, `1101699` risk20 source-gate, `1101700` top-hybrid source-gate를 재생성했다. direct/qual smoke `1101690`은 평가 전용이므로 그대로 유지한다.

### 20.26 2026-04-30 11시대 raw035 release gate와 subject-valid calibration 재정렬

`mcn-q-eva02l448-dsrc-genalign-raw035-setonly-ip134-r1-20260429`는 route, subject IoU, proposal, generated-align, crop-action, top-return을 동시에 가장 가깝게 보존한 near-miss checkpoint였지만, raw top-return과 subject-valid calibration이 release gate를 막았다. 이를 기준으로 실행한 retcal r2/r3, decision-conditioned cropboost, route ensemble threshold, UCTR retcal dcond는 모두 배포 후보에서 제외했다.

중요한 판정은 다음과 같다. `1101740` dcond-cropboost r3와 IP134 topagree r3는 train gate 일부를 통과했지만 no-prior head/direct/qualitative surface에서 subject-valid mismatch와 route acc blocker가 반복됐다. `1101751`/`1101752` routeens threshold `0.30`/`0.85`도 final hit는 유지했지만 route acc `0.506250`과 subject-valid mismatch `31.13~35.00%`로 실패했다. `1101760` UCTR retcal dcond는 train gate에서 top-return `0.570625`와 subject-valid best-balanced `0.549375`가 경계 미달이라 full bundle로 승격하지 않는다.

운영 측면에서는 IP134에 남아 있던 제외 checkpoint의 stale direct/qualitative rerun2를 종료했고, `mcn-raw035-validrefine-neg3-r2-ip134-20260430`은 `subject_refine_valid_head`에 실제 trainable parameter가 없어 invalid launch로 분리했다. 당시 유효한 source-of-truth는 subject-valid logit만 보정하는 세 개의 비중복 bracket이었지만, 이후 모두 train gate 또는 invalid 상태로 닫혔다.

| run | 위치 | trainable scope | valid negative scale | 상태 |
| --- | --- | --- | ---: | --- |
| `1101777` `mcn-raw035-validlogit-neg5-r1-20260430` | MLP A100 | `subject_valid_head,subject_spatial_valid_head` | 5.0 | `excluded_train_gate`; subject-valid/action/top-return 미달 |
| `1101778` `mcn-raw035-validlogit-neg8-r1-20260430` | MLP A100 | `subject_valid_head,subject_spatial_valid_head` | 8.0 | `excluded_train_gate`; subject-valid/action/top-return 미달 |
| `mcn-raw035-validlogit-neg3-r2-ip134-20260430` | IP134 A100 | `subject_valid_head,subject_spatial_valid_head` | 3.0 | `excluded_train_gate`; subject-valid/action/top-return 미달 |

이 bracket의 판단 기준은 단순 validation `best_selection_score`가 아니다. subject-valid best-balanced, negative/positive acc, Product-AR direct subject_valid_mismatch, subject IoU 보존, route acc 보존, proposal@5, action/top-return, target-AR compatibility, qualitative taxonomy가 함께 통과해야 한다. 세 run 모두 실패하면 다음 실험은 threshold/negative-scale 반복이 아니라 teacher-valid target 재정의, subject-valid head 구조 변경, crop/action head가 내부 subject-valid state를 조건으로 쓰는 joint 구조 변경으로 넘어간다.

2026-04-30 12:10 KST 1차 terminal 결과로 `1101797` `mcn-dcondcrop-validonly-neg8-r2-20260430`은 `excluded_train_gate`로 닫았다. route balanced `0.625719`, proposal recall `0.952377`, generated-align `0.956875`, decision-conditioned crop-action `1.0`은 유지됐지만 subject-valid best-balanced `0.530875 < 0.62`, decision-conditioned top-return `0.576946 < 0.58`이 미달했다. 이는 valid-only negative scale을 키워도 val split에서 specificity/recall 동시 calibration이 회복되지 않는다는 evidence다.

같은 시각 후속 구조 축을 실제 구현했다. `subject_valid_route_head`는 기존 subject-valid logit/prob, route softmax, image condition token을 입력으로 valid logit delta를 예측하며, zero-init으로 기존 checkpoint surface를 보존한다. 로컬 검증은 `py_compile`, runner `bash -n`, `tests/test_mobilecropnet_v4.py -q` 기준 `11 passed`이고, 원격 root package import smoke도 `remote_ok 689`로 확인했다. 13:29 KST 기준 `1101810` proposal-context refine, `1101817` route-conditioned valid calibrator, `1101829` clean route-calibrator, `1101834`/`1101836` primary-or-reliable target repair, `1101835`/IP134 box-exists target repair는 train gate 제외로 닫았다. `1101830` r1은 검증 도중 platform termination되어 summary가 없어 release evidence로 쓰지 않는다. 이후 생성된 `1101851` `box_reliability + route_scene_background_or_conf`, `1101852` `box_reliability + route_scene_or_conf`, `1101854`/`1101855` `route_reliable` target repair는 `Session terminated` 패턴으로 train gate 전 끊겼고, unique-tag 복구 `1101862`~`1101865`도 bootstrap 단계 종료라 성능 근거가 아니다.

2026-04-30 12:20 KST 추가 terminal 결과로 `1101778` `mcn-raw035-validlogit-neg8-r1-20260430`도 `excluded_train_gate`로 닫았다. best epoch 1 기준 route balanced `0.625302`, subject IoU `0.631945`, proposal recall `0.952672`, generated-align `0.953750`은 유지됐지만 subject-valid best-balanced `0.530875`, crop-action `0.945429`, top-return `0.555083`이 release train gate를 통과하지 못했다. epoch 2의 subject-valid best-balanced는 `0.529000`으로 더 낮아져 negative scale 증대만으로 calibration blocker를 해결하지 못한다는 결론을 보강한다.

2026-04-30 12:23 KST 추가 terminal 결과로 `1101796` `mcn-dcondcrop-routevalid-neg5-r2-20260430`도 `excluded_train_gate`로 닫았다. route balanced `0.622352`, subject IoU `0.631945`, proposal recall `0.952377`, generated-align `0.957500`, decision-conditioned crop-action `1.000000`은 살아 있었지만 subject-valid best-balanced `0.534938`과 decision-conditioned top-return `0.576946`이 미달했다. route head와 subject-valid head를 함께 열어도 release blocker가 닫히지 않았으므로, 다음 판단은 `subject_valid_route_head` residual 계열과 proposal-context refine 계열의 결과로 넘긴다.

2026-04-30 12:32 KST에는 `1101777` `mcn-raw035-validlogit-neg5-r1-20260430`도 `excluded_train_gate`로 닫았다. best epoch 2 기준 route balanced `0.625302`, subject IoU `0.631945`, proposal recall `0.952672`, generated-align `0.953125`은 유지됐지만 subject-valid best-balanced `0.526083`, crop-action `0.945429`, top-return `0.555083`으로 미달했다. raw035 valid-logit negative-scale bracket은 neg5/neg8 모두 실패했으므로 같은 축의 단순 반복은 중단한다.

사용자 지적의 UCTR teacher 우위와 student target 충돌을 반영해 target repair 코드도 추가했다. `MobileCropNetV4BatchDataset`은 이제 `subject_box_valid_target_mode`를 지원하며 기본값 `route_gated`는 기존 동작을 유지한다. 후속 실험에서만 `box_reliability`, `box_primary_or_reliable`, `box_exists`를 켜서 scene/background-like row의 valid subject supervision을 route hard-gate에서 분리할 수 있다. `box_exists`는 val/test target이 사실상 전부 positive라 specificity를 만들 수 없고, `box_primary_or_reliable`도 positive-heavy라 valid head가 collapse하는 것을 확인했으므로 신규 `route_reliable` target을 추가했다. 이 target은 foreground route에서도 reliability가 낮은 box를 invalid로 두어 train/test positive rate를 낮추고, UCTR의 subject support 신뢰도를 학생 valid head에 더 직접 전이한다. 이 변경은 로컬 `py_compile`, wrapper `bash -n`, `tests/test_mobilecropnet_v4.py -q` 11개, 원격 import smoke를 통과했다. 실제 bounded GPU 검증은 platform/bootstrap 종료 때문에 완료되지 않았으며, `1101851`, `1101852`, `1101854`, `1101855`, `1101862`~`1101865`는 release evidence가 아니다. 이 축의 후속 판단 기준은 subject-valid best-balanced뿐 아니라 route/valid agreement, decision-conditioned crop action, top-return, target-AR compatibility, qualitative taxonomy가 되어야 한다.

2026-04-30 12:38 KST 추가 terminal 결과로 IP134 `mcn-raw035-validlogit-neg3-r2-ip134-20260430`도 `excluded_train_gate`로 닫았다. best epoch 1 기준 subject-valid best-balanced `0.530875`, crop-action `0.945429`, top-return `0.555083`이 미달했으며 epoch 2는 subject-valid `0.524982`로 더 낮았다. 따라서 raw035 valid-logit neg3/5/8 bracket은 단순 negative-scale 보정 축으로 더 진행하지 않는다.

같은 시각 `1101819` `mcn-dcondcrop-routevalidcal-full1-r1-20260430`도 `excluded_train_gate`다. full train rows `39,002`로 route-conditioned residual을 학습하자 route `0.627977`, subject IoU `0.642984`, valid balanced `0.592366`까지는 개선됐지만, valid gate `0.62`를 넘지 못했고 decision-conditioned top-return도 `0.559735`로 낮았다. 결론적으로 full-data routecal은 valid calibration을 일부 개선하지만, target definition과 action/return alignment를 함께 바꾸지 않으면 배포 후보가 되지 않는다.

2026-04-30 13:45 KST 기준 운영 상태는 다음과 같다. `1101810`, `1101817`, `1101829`는 subject-valid/top-return 또는 subject IoU 미달로 제외됐다. `1101835`와 IP134 box-exists target repair는 subject IoU를 높였지만 subject-valid best-balanced `0.500000`, specificity `0.000000`으로 제외됐다. `1101834`/`1101836` primary-or-reliable target repair도 subject IoU `0.684656`은 강하지만 subject-valid best-balanced `0.522500`, top-return `0.571173`으로 제외됐다. `1101830` r1은 platform termination으로 summary가 없어 release evidence가 아니다. 같은 bracket을 복구한 `1101851`, `1101852`, `1101854`, `1101855`도 `Session terminated` 패턴으로 train gate 전 종료되어 partial artifact로 분리한다. unique tag와 `NUM_WORKERS=2`로 재생성된 `1101862`, `1101863`, `1101864`, `1101865`는 Start 후 7~37초 안에 bootstrap 단계에서 종료되어 성능 row가 아니다. 따라서 이 bracket은 아직 모델 가설 실패로 닫힌 것이 아니며, spot 재반복 대신 안정 실행 경로에서 `route_reliable`/`box_reliability` target repair를 재검증한다.

2026-04-30 13:32 KST에는 `route_reliable` target 분포도 별도 artifact로 고정했다. `artifacts/mobilecropnet_v4/quality_first_20260428/diagnostics/subject_valid_target_modes_20260430/target_mode_distribution_with_route_reliable.json` 기준 positive rate는 train `0.659992`, val `0.678795`, test `0.635873`이다. 이는 `box_reliability`의 train/val/test `0.795728/0.822721/0.780154`보다 보수적이고, `box_exists`의 전부 양성 붕괴를 피한다. val split route별로는 `portrait_single`/`portrait_group`이 `1.0`, `object_single` `0.874245`, `object_multi` `0.866071`, `scene_general`/`background_texture_copyspace` `0.0`이다. 따라서 다음 재개 시 핵심 질문은 "foreground 신뢰도 기반 valid target이 subject-valid specificity를 회복하면서 action/top-return을 보존하는가"다.

### 20.27 2026-04-30 프로파일군별 평가 정리 및 계속 실행 우선순위

`1101851`, `1101852`, `1101854`, `1101855`는 2026-04-30 13:27 KST에 모두 `Session terminated` 패턴으로 끊겨 train gate 전 partial artifact로만 분리한다. 같은 output path를 이어 쓰지 않고 `NUM_WORKERS=2`와 unique tag로 복구한 `1101862`, `1101863`, `1101864`, `1101865`도 Start 후 bootstrap 단계에서 `Terminated`되었고, `nvidia-smi`, train log, checkpoint, `summary.json`을 남기지 못했다. 따라서 이 섹션의 현재 상태에서 active source-of-truth는 없고, Product-AR direct/qualitative/latency/GAIC/public/equal-4로 승격할 새 checkpoint도 없다.

계속 실행 우선순위는 문서 복구가 아니라 배포 가능한 모델 확보 기준으로 재정렬한다. 첫째, 동일 spot workload 반복을 멈추고 standard P3 또는 사용 가능한 interactive GPU에서 `route_reliable + route_scene_or_conf` bounded smoke를 먼저 실행한다. 둘째, 이 smoke가 bootstrap을 통과하면 같은 안정 경로로 `box_reliability + route_scene_or_conf`, `box_reliability + route_scene_background_or_conf`, `route_reliable + route_scene_background_or_conf`를 병렬 확장한다. 셋째, train gate 통과 checkpoint만 Product-AR direct, qualitative review pack, latency, GAIC official, public/equal-4를 같은 checkpoint로 묶어 평가한다. 넷째, final hit가 높아도 subject-valid mismatch, subject IoU low, risk hit, route mismatch가 반복되면 Unified Final Leaderboard release row로 승격하지 않는다.

2026-04-30 13:42~13:53 KST에는 P3 standard quota `575`와 group pool `45` A100 availability를 확인하고 standard smoke `1101871` `mcn-dcondcrop-routerel-stdsmoke-r1-20260430`을 생성했다. 이 run은 `route_reliable + route_scene_or_conf` target/policy 조합의 첫 안정성 검증이었지만, 10분 이상 `Pending`에 머물러 Start Time/IP/run dir 없이 `Terminated`로 회수했다. 따라서 `1101871`은 모델 성능 근거가 아니라 MLP allocation blocker이며, 같은 MLP 방식의 즉시 재생성은 중단한다.

성능 평가는 앞으로도 온디바이스용과 비온디바이스/품질 우선 프로파일을 분리해 읽어야 한다.

| 프로파일군 | 포함 프로파일 | 2026-04-30 13:45 KST 기준 결론 |
| --- | --- | --- |
| 온디바이스/저지연 | `q24_288_w075`, `balanced_288`, `rank_320`, `hybrid_384`, `plus_384` | public/GAIC benchmark 수치는 상대적으로 안정적이지만, no-prior runtime에서 subject bbox/valid, route, action/top-return을 동시에 닫지 못했다. 대표 상위 row인 `SSTK public / balanced_288`도 FCDB `0.766093`, CPC `0.855655`, GNMC `0.768181`, GAIC primary `0.650690`임에도 release gate는 미통과다. |
| 비온디바이스/품질 우선 | `cnv2b_448*`, `eva02l_448*`, `cnv2l_512`, `cnv2h_512`, `dinov2l_518`, `swinv2*` | 속도 제약을 완화해 subject IoU 또는 final hit는 올라갔지만 qualitative blocker가 반복됐다. public60 `eva02l_448_public60_setonly`는 subject IoU `0.708175`, final hit `0.967615`였으나 subject-valid mismatch로 제외됐고, raw035 dcond 계열은 subject IoU `0.686101`, final hit `0.965625`였으나 route/valid mismatch로 제외됐다. |

따라서 현재 제품 판정은 `아직 배포 후보 없음`이며, GPU 재개는 MLP allocation blocker 해소 또는 접근 가능한 interactive GPU 확보가 필요하다. 가장 가까운 이전 축은 raw035 dcond/primary-reliable 계열처럼 subject IoU와 final crop hit를 올린 후보였지만, subject-valid specificity/negative calibration과 top-return/action agreement가 닫히지 않았다. 다음 판단은 단순 threshold, negative-scale, public/UCTR mix ratio 반복이 아니라 `route_reliable`류의 target definition을 유지하되 subject-valid state-conditioned action selector와 action/return coupling loss를 더 강하게 묶은 bounded smoke의 train gate 결과를 따른다.

## 부록 A. Figure Manifest

| Figure | 파일 | 용도 |
| --- | --- | --- |
| Fig. 4-1 | `assets_mobilecropnet_v4_master_20260427/fig01_mobilecropnet_v4_runtime_architecture.png` | runtime architecture |
| Fig. 5-1 | `assets_mobilecropnet_v4_master_20260427/fig02_training_contract_and_losses.png` | training row contract and losses |
| Fig. 7-1 | `assets_mobilecropnet_v4_master_20260427/fig11_candidate_bank_vs_generated_runtime.png` | candidate-bank replay vs generated-proposal runtime schematic |
| Fig. 8-1 | `assets_mobilecropnet_v4_master_20260427/fig07_subject_box_patch_status.png` | subject-box patch status |
| Fig. 9-1 | `assets_mobilecropnet_v4_master_20260427/fig05_route_subject_smoke_progress.png` | route/subject smoke progress |
| Fig. 10-1 | `assets_mobilecropnet_v4_master_20260427/fig06_runtime_no_prior_rerun_matrix.png` | runtime no-prior rerun |
| Fig. 14-1 | `assets_mobilecropnet_v4_master_20260427/fig10_teacher_public_cropper_ceiling.png` | teacher/public cropper ceiling |
| Fig. 15-1 | `assets_mobilecropnet_v4_master_20260427/fig03_strict_release_gate_matrix.png` | strict release gate matrix |
| Fig. 15-2 | `assets_mobilecropnet_v4_master_20260427/fig04_current_equal4_leaderboard.png` | current equal-4 leaderboard |
| Fig. 16-1~16-10 | `assets_mobilecropnet_v4_master_20260427/fig12_route_stratified_crop_sample_*.png` | teacher-route-stratified no-prior audit gallery with actual portrait_single/portrait_group coverage |
| Fig. 16-11~16-20 | `assets_mobilecropnet_v4_master_20260427/fig13_joint_head_inference_crop_sample_*.png` | joint head-quality no-prior inference gallery selected by crop, route, decision, target-AR, and subject-valid metrics |
| Fig. 16-21~16-24 | `assets_mobilecropnet_v4_master_20260427/fig16_mcn_public_cropper_qualitative_headgood_*.png` | MCN AR/head output vs public GAIC/CGS free-form crop comparison on teacher-matched release-gate head-good samples |
| Fig. 16-25~16-26 | `assets_mobilecropnet_v4_master_20260427/fig16_*_crop_head_success_gallery.png` | teacher and MobileCropNet crop/head success overview with subject mode, checklist, and why-tags |
| Fig. 17-1 | `assets_mobilecropnet_v4_master_20260427/fig08_inference_latency_profile.png` | inference latency profile |
| Fig. 17-2 | `assets_mobilecropnet_v4_master_20260427/fig17_latency_profile_public_cropper_comparison.png` | MobileCropNet profile and public cropper latency comparison |
| Fig. 17-3 | `assets_mobilecropnet_v4_master_20260427/fig17_latency_subset_mcn_public_cropper.png` | Representative MobileCropNet profile and public cropper latency subset |

전체 machine-readable manifest는 `Implement_Docs/assets_mobilecropnet_v4_master_20260427/figure_manifest.json`에 있다.

## 부록 B. Source 문서 Mapping

| 문서/산출물 | 반영 내용 |
| --- | --- |
| `MobileCropNet_v4_0_Implementation_Master_Report_Figure_Prompt_Plan_KO_2026-04-27.md` | figure guardrail, no-prior runtime 시각화 원칙, release gate status |
| `MobileCropNet_v4_0_Final_Benchmark_And_Deployment_Status_KO_2026-04-24.md` | 37-row leaderboard, deploy candidate 없음, route/head audit, qualitative blocker |
| `MobileCropNet_v4_0_Unified_Recovery_Report_KO_2026-04-24.md` | route smoke, runtime no-prior rerun, subject-box integration, recovery interpretation |
| `MobileCropNet_v4_0_Runtime_NoPrior_DeployAlign_Implementation_KO_2026-04-24.md` | no-prior runtime contract, executor, baseline lane, gate ablation |
| `MobileCropNet_v4_0_SubjectBox_IoU_Improvement_Plan_and_Patch_KO_2026-04-24.md` | subject-box v2/patch design, content-relative box, proposal-conditioned refinement |
| `MobileCropNet_v4_0_Product_AR_Head_Audit_and_Execution_Status_KO_2026-04-22.md` | route collapse, proposal philosophy mismatch, policy decision label collapse |
| `MobileCropNet_v4_0_UCTR_CPU_Recheck_and_Head_Improvement_Plan_KO_2026-04-22.md` | subjectprior route/proposal branch, policy score-first, route-balanced loss |
| `MobileCropNet_v4_0_Interim_Progress_Report_KO_2026-04-23.md` | teacher label completion, branch rule, profile matrix interpretation |
| `MobileCropNet_v4_0_PublicCropper_Comparison_KO_2026-04-16.md` | GAIC/CGS/CACNet public cropper의 구조, GAIC candidate-ranking 비교, Product-AR direct 비교 가능 범위 |
| `MobileCropNet_v4_0_bundle/MobileCropNet_v4_0.md` | 4.2절 대표 문헌 비교표 기반 GAIC/CGS 외 public cropper 후보 확장 검토 |
| `UNIFIED_PUBLIC_BENCHMARK_EQUAL4_TEACHER_REPORT_KO_2026-04-23.md` | teacher/public cropper ceiling and equal-4 interpretation |
| `analysis_summary.json` | consolidated release, leaderboard, route smoke, runtime, subject patch, qualitative values |
| `final_deployment_leaderboard.json` | top student rows and equal-4/direct/head metrics |
| `release_gate_manifest.json` | strict gate decision, blockers, thresholds |
| `runtime_no_prior_rerun_report.json` | no-prior runtime direct/risk/route/decision metrics |
| `subject_box_integration_report.json` | subject-box patch row metrics |
| `equal4_teacher_leaderboard.json` | teacher/public cropper ceiling |

## 부록 C. 코드 Index

| 경로 | 역할 |
| --- | --- |
| `src/mobilecropnet_v4/data.py` | vocab, letterbox transform, dataset adapter, candidate role merge, pairwise/listwise join |
| `src/mobilecropnet_v4/model.py` | MobileCropNetV4 architecture, proposal/ranker/heads/loss |
| `src/mobilecropnet_v4/eval_utils.py` | runtime baselines, exact AR repair, executor, prediction decode |
| `src/mobilecropnet_v4/gaic_benchmark.py` | GAIC official MOS benchmark evaluation |
| `src/mobilecropnet/` | legacy static micro-bank MobileCropNet baseline |

## 부록 D. Script Index

| 그룹 | 대표 script |
| --- | --- |
| training | `src/scripts/train_mobilecropnet_v4.py`, `src/scripts/run_mobilecropnet_v4_*experiment.sh` |
| inference/visualization | `src/scripts/infer_mobilecropnet_v4.py`, `src/scripts/visualize_mobilecropnet_v4_predictions.py` |
| replay/product eval | `src/scripts/evaluate_mobilecropnet_v4.py`, `src/scripts/evaluate_mobilecropnet_v4_product_ar_direct.py` |
| GAIC/public benchmark | `src/scripts/evaluate_mobilecropnet_v4_gaic_benchmark.py`, `src/scripts/export_mobilecropnet_v4_public_benchmark_predictions.py` |
| Product-AR label | `src/scripts/build_mobilecropnet_v4_product_ar_score_labels.py`, `src/scripts/score_mobilecropnet_v4_product_ar_candidates_with_uctr.py`, `src/scripts/score_mobilecropnet_v4_product_ar_candidates_with_public_cropper.py` |
| head/subject audit | `src/scripts/analyze_mobilecropnet_v4_head_predictions.py`, `src/scripts/audit_mobilecropnet_v4_release_heads.py`, `src/scripts/build_mobilecropnet_v4_subject_box_report.py` |
| leaderboard/release | `src/scripts/build_mobilecropnet_v4_final_deployment_leaderboard.py`, `src/scripts/build_mobilecropnet_v4_release_gate_manifest.py`, `src/scripts/build_mobilecropnet_v4_qualitative_review_pack.py` |

## 부록 E. Artifact Index

| 경로 | 내용 |
| --- | --- |
| `Implement_Docs/assets_mobilecropnet_v4_master_20260427/` | 이 문서의 Figure PNG, manifest, analysis summary |
| `artifacts/mobilecropnet_v4/unified_recovery_20260424/` | 최신 recovery, release gate, leaderboard 산출물 |
| `artifacts/mobilecropnet_v4/head_variant_async_20260423/` | SSTK/GAIC/public/T1 async profile runs |
| `artifacts/mobilecropnet_v4/shortlist_final_eval_20260423/` | original shortlist final eval bundle |
| `artifacts/mobilecropnet_v4/release_gate_v2_20260424/` | route recovery and release-gate smoke runs |
| `artifacts/mobilecropnet_v4/runtime_no_prior_rerun_20260424/` | no-prior runtime rerun |
| `artifacts/mobilecropnet_v4/subject_box_iou_patch_20260424_full/` | subject-box IoU patch runs |
| `artifacts/mobilecropnet_v4/inference_latency/` | latency benchmark 산출물 |
| `artifacts/unified_public_benchmark_20260423_equal4_teacher/` | teacher/public cropper equal-4 leaderboard |

## 부록 F. 재현 명령

모든 local Python 실행은 프로젝트 지정 interpreter를 사용한다.

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m pytest tests/test_mobilecropnet_v4.py -q
```

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/evaluate_mobilecropnet_v4.py --checkpoint <best.pt> --jsonl <test.jsonl> --project_root . --output_dir <eval_dir>
```

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/infer_mobilecropnet_v4.py --checkpoint <best.pt> --image <image.jpg> --target_ar 16:9 --output_json <out.json> --output_png <out.png> --selection_policy proposal_topk_rerank
```

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/build_mobilecropnet_v4_release_gate_manifest.py --leaderboard_json <final_deployment_leaderboard.json> --output_json <release_gate_manifest.json> --output_md <release_gate_manifest.md>
```

Remote CPU/GPU 작업은 project AGENTS 지침의 launch job, poll durable status files, collect summaries pattern을 따른다. GPU workload는 placeholder가 아니라 bounded command로 실제 smoke/full/eval 작업을 수행하고 종료해야 한다.

## 부록 G. Release Checklist

| gate | release condition |
| --- | --- |
| public equal-4 | top-set z-score와 raw mean이 높고 worst-dataset collapse가 없음 |
| GAIC official | top1 MOS, SRCC, Accw4@10 균형 |
| Product-AR direct | final hit@0.5, best IoU, low risk hit, target-AR compatibility |
| generated proposal | proposal recall@K와 generated alignment가 replay 대비 collapse하지 않음 |
| route | route collapse 없음, balanced accuracy threshold 이상 |
| subject box | IoU, confidence, valid acc, negative acc, positive acc 동시 충족 |
| policy/action | decision label collapse 없음, executor action과 utility gap 정합 |
| explanation | checklist/why/risk agreement, mode-inapplicable hallucination 없음 |
| qualitative | catastrophic bucket이 release threshold 아래 |
| latency | target profile/device latency budget 충족 |
| artifact | checkpoint, config, status, metrics, summary, release manifest가 durable path에 존재 |

## 부록 H. Figure Prompt Plan 요약

Figure prompt plan의 핵심 원칙은 다음과 같다. architecture, training contract, release funnel, candidate-bank vs runtime 같은 conceptual schematic은 이미지 생성 또는 programmatic schematic으로 만들 수 있다. leaderboard, gate matrix, route smoke, latency, qualitative crop panel처럼 실제 수치와 좌표가 의미를 결정하는 Figure는 JSON artifact와 실제 crop/overlay data를 source of truth로 사용해야 한다. 축소 contact sheet는 본문 qualitative evidence에서 제외하고, 샘플별 per-AR crop panel로 대체한다. AI-generated 이미지는 experimental evidence로 쓰지 않는다. 이미지 내부 label은 짧은 English를 유지하고, 한국어 해석은 Markdown caption과 본문에서 처리한다.
