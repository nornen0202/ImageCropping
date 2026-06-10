# MobileCropNet 마스터 리포트 AI Application Specialist 평가 대응 리뷰

작성일: `2026-05-07`

## 1. 리뷰 목적과 대상

본 문서는 `AI Application Specialist 양성과정 인증과제 평가표`의 6개 항목을 기준으로 아래 두 마스터 문서를 검토하고, 평가 관점에서 보완해야 할 내용을 정리한 별도 리뷰 문서다.

| 구분 | 대상 문서 |
| --- | --- |
| 학습 데이터 생성 | `Implement_Docs/MobileCropNet_Training_Data_Generation_Master_Report_KO_2026-04-24.md` |
| 모델 구현/평가 | `Implement_Docs/MobileCropNet_v4_0_Implementation_Master_Report_KO_2026-04-27.md` |
| 평가 기준 | `Implement_Docs/AI_Application_Specialist_Eval.md` |

핵심 결론은 다음과 같다. 두 마스터 문서는 문제 정의, AI 모델링, 데이터 팩토리, no-prior runtime, 평가/릴리스 게이트 측면에서는 매우 강하다. 반면 평가표에 직접 존재하는 `프로그래밍 & 툴활용` 항목은 부록의 코드/명령/산출물 색인으로 흩어져 있어, 평가자가 역량 증거로 바로 인식하기 어렵다. 또한 `AI Governance`와 `현업성과`도 현재 내용만으로는 간접 증거가 많으므로, 별도 요약 섹션을 추가하는 것이 좋다.

## 2. 평가 항목별 리뷰 요약

| 평가 항목 | 현재 문서의 강점 | 평가 리스크 | 보완 방향 |
| --- | --- | --- | --- |
| AI 기술/지식이해, AI 모델링 | crop 문제를 단일 bbox 회귀가 아닌 candidate, ranking, policy, route, subject, rationale 결합 문제로 재정의했다. GAIC/FCDB/CPC/GNMC/SSTK teacher 차이를 분리하고 baseline 및 teacher ceiling도 제시한다. | 초기 문서에는 구조 구성 요소와 benchmark ceiling은 있으나, GAIC/CGS/CACNet 및 heuristic/sequential/VLM cropper 대비 v4를 채택한 이유가 한 화면에 정리되어 있지 않았다. | v4 구현 문서 1.6에 public cropper 방법론 비교, 공개 benchmark 지표, 제품 요구사항 기반 설계 채택 근거를 추가한다. |
| 프로그래밍 & 툴활용 | `src/mobilecropnet_v4/`, `src/scripts/`, `tests/`, artifact, pytest 명령이 부록에 존재한다. 학습/평가/시각화/release gate가 코드화되어 있다. | 본문에서 도구 활용 역량이 명시적으로 드러나지 않는다. 사내 AI 개발 보조 도구 활용 서술도 없다. | 본 리뷰 문서의 5장 문안을 마스터 문서에 추가한다. 구현 보조 도구는 `Cline SR`로 통일한다. |
| AI App 개발, On-Device 활용 | no-prior runtime, `image + target_ar` 입력 계약, latency profile, exact target-AR postprocess, lightweight profile 비교가 명확하다. | 실제 모바일/NPU export, INT8/QAT, operator coverage는 아직 계획 수준이다. release gate 통과 checkpoint가 없다는 점도 제품 완성도 측면의 리스크다. | "배포 후보 없음"을 실패가 아니라 strict product gate가 차단한 상태로 설명하고, on-device 최적화 후속 계획을 명확히 둔다. |
| AI Governance | safety/reject, fatal demotion, safe conversion, split leakage 방지, provenance, 생성 이미지 비증거 원칙이 있다. | 데이터 권리, public benchmark 사용 범위, 인물/텍스트 포함 이미지의 윤리 검토, 생성형 산출물 검수 절차가 한 섹션에 정리되어 있지 않다. | `AI Governance 및 데이터 사용 적합성` 섹션을 추가해 source별 사용 목적, 라이선스/권한 확인, 개인정보/초상 리스크, 산출물 provenance를 표로 정리한다. |
| 과제 정의 | 제품형 이미지 크롭 문제의 배경, 기대효과, 접근방법, 검증방법이 매우 구체적이다. | 평가자가 전체 기술보고서를 읽기 전에 현업 문제와 AI 적용 필요성을 빠르게 파악할 요약이 부족하다. | 서론 뒤에 `평가 대응용 과제 정의 요약`을 추가한다. |
| 평가 및 성과 | equal-4, GAIC official, Product-AR direct, qualitative pack, latency, release gate 등 다면 평가가 강하다. | 현재 strict release gate 통과 모델이 없으므로 "최종 성과 부족"으로 오해될 수 있다. | 성과를 "배포 후보 확정"이 아니라 "제품 실패를 감추지 않는 검증 체계와 개선 방향 확보"로 정리하고, business impact를 업무 효율화/품질 게이트 관점에서 설명한다. |

## 3. 문서별 상세 리뷰

### 3.1 학습 데이터 생성 마스터 리포트

이 문서는 평가표의 `AI 기술/지식이해, AI 모델링`, `과제 정의`, `AI App 개발` 항목에 강하게 대응한다. 특히 SSTK, GAIC, public crop benchmark를 하나의 정답 crop 데이터셋으로 합치지 않고, 각 데이터가 가르치는 supervision 의미를 구분한 점이 좋다. C1-C7 perception precompute, candidate bank, teacher scoring, safe conversion, pairwise/listwise/decision/checklist/why-tag contract로 이어지는 구조는 AI 모델링 역량을 잘 보여준다.

정량 근거도 충분하다. `Full_10000` 기준 curated image `10,000`, conditional-DETR batch `48,766`, pairwise `151,049`, listwise `48,766`, candidate `7,533,038` 등 데이터 생성 규모가 구체적이다. GAIC corrected와 Product-AR label의 차이를 candidate richness와 task 정의 차이로 설명한 부분도 평가자에게 기술적 성숙도를 보여 줄 수 있다.

보완할 점은 세 가지다.

1. `프로그래밍 & 툴활용`이 부록의 코드 경로와 재현 명령에만 남아 있다. 본문 또는 결론 직전 별도 섹션에서 Python/PyTorch, JSONL/manifest 기반 데이터 계약, Cline SR 기반 구현 보조, 로컬/원격 검증 자동화, pytest/figure builder/release artifact 검증을 명확히 적어야 한다.
2. `AI Governance`는 safe conversion과 reject tag 관점으로는 강하지만, 평가표가 요구하는 법적/윤리적 체크 관점의 문구가 부족하다. SSTK와 public benchmark의 사용 범위, license/권한 확인, 인물/텍스트 이미지 안전 검토, 생성 산출물의 evidence 사용 금지 원칙을 별도 표로 정리하는 것이 좋다.
3. 본문 첫머리에 "구현 완료 보고서가 아니다"라고 되어 있는데, 인증과제 평가 관점에서는 구현 역량이 약해 보일 수 있다. "본 문서는 학습 데이터 생성 체계의 방법론 중심 보고서이며, 구현 증거는 부록 D-E 및 별도 구현 마스터 리포트와 연결된다" 정도로 조정하는 편이 안전하다.

### 3.2 MobileCropNet v4.0 구현 마스터 리포트

이 문서는 `AI App 개발/On-Device 활용`, `프로그래밍 & 툴활용`, `평가 및 성과` 항목에 대응할 수 있는 실제 구현 증거가 많다. `src/mobilecropnet_v4/data.py`는 vocab, dataset adapter, pairwise/listwise join, subject-box target mode, letterbox transform을 구현한다. `src/mobilecropnet_v4/model.py`는 AR-conditioned proposal, relation/set ranker, route/policy/subject/checklist/why-tag heads, runtime baseline과 decision source 계열을 포함한다. `src/scripts/train_mobilecropnet_v4.py`는 profile preset, route-balanced sampler, route expert distillation, long-running status/summary/checkpoint 저장, 다양한 loss 및 ablation option을 제공한다.

평가 코드도 잘 분리되어 있다. `evaluate_mobilecropnet_v4_product_ar_direct.py`는 no-prior direct evaluation과 exact target-AR postprocess를 확인하고, `build_mobilecropnet_v4_final_deployment_leaderboard.py`와 `build_mobilecropnet_v4_release_gate_manifest.py`는 public/GAIC/direct/head/latency/qualitative gate를 하나의 제품 판단으로 묶는다. 이 점은 단순 모델 학습이 아니라 AI application 개발 및 검증 체계를 구현했다는 강한 증거다.

보완할 점은 다섯 가지다.

1. v4 구조 자체는 설명되어 있지만, 왜 public cropper를 그대로 쓰지 않고 AR-conditioned proposal, relation ranker, route/policy/subject/explanation head를 결합한 구조로 갔는지에 대한 비교 분석이 부족하다. GAIC/CGS/CACNet, static/heuristic cropper, sequential/view-adjustment, VLM cropper와 비교해 "benchmark scorer"와 "제품형 no-prior cropper"의 차이를 명시해야 한다.
2. 20장 이후 실행 로그성 섹션이 길어 평가자가 핵심 성과를 놓칠 수 있다. 최신 상태는 "strict release gate 통과 checkpoint 없음, 그러나 full pipeline/release gate 구현 완료"로 요약하고, 장기 실행 기록은 부록 또는 실행 일지로 분리하는 것이 좋다.
3. `현재 강점` 섹션의 "다음 외부 개발 보조 도구/GPU workflow가 상태를 이어받을 수 있다"는 취지의 표현은 사내 도구명으로 정리해야 한다. 권장 문구는 "training/eval/release 산출물이 JSON/PNG/MD로 남아 있어 다음 Cline SR 기반 GPU workflow가 상태를 이어받을 수 있다"이다.
4. `프로그래밍 & 툴활용`은 현재 부록 C-F에 흩어져 있다. 평가표 항목과 같은 제목으로 본문에 1개 섹션을 추가하면 평가 안정성이 크게 올라간다.
5. on-device 항목은 latency profile이 있지만 모바일 배포 후속 단계가 아직 열려 있다. `operator coverage`, `INT8/QAT`, `NPU delegate`, `memory footprint`, `CPU fallback`, `pre/postprocess 비용`을 후속 검증 항목으로 별도 표에 넣어야 한다.

## 4. 사내 AI 도구 및 외부 도구 표기 전략

평가 문서에서는 구현 단계의 코딩 보조 및 코드베이스 탐색 도구를 `Cline SR`로 통일한다. 사전 리서치, 문제 정의, 비교 모델 조사, 보고서 구조화 단계에서 범용 LLM을 제한적으로 활용했다는 정도는 허용 가능하지만, 강조점은 사내 개발 환경과 재현 가능한 코드/산출물이어야 한다.

권장 표기 원칙은 다음과 같다.

| 구분 | 권장 표기 |
| --- | --- |
| 코딩/리팩터링/테스트 보조 | `Cline SR 기반 코드 탐색, 구현 보조, 실험 로그 분석, 재현 명령 정리` |
| 사전 리서치/아이디어 비교 | `범용 LLM을 활용한 관련 연구/설계 옵션 초안 검토` 정도로 최소 언급 |
| 최종 구현 근거 | `Python/PyTorch 코드, shell runner, pytest, MLP/remote artifact, JSON summary, release manifest` |
| 피해야 할 표현 | 외부 코딩 에이전트 명칭, 개인용 도구명, 재현 불가능한 대화 기반 판단 |
| 강조할 메시지 | AI 도구는 보조 수단이며, 최종 판단은 durable artifact와 정량/정성 gate에 의해 결정됨 |

## 5. 마스터 문서에 추가 권장: 프로그래밍 & 툴활용 섹션 초안

아래 문안은 두 마스터 문서 중 구현 마스터 리포트 본문 또는 부록 앞에 그대로 삽입할 수 있는 형태다.

### AI 프로그래밍 및 사내 Tool 활용

본 과제는 데이터 생성, 모델 설계, 학습, 평가, 릴리스 게이트를 모두 Python/PyTorch 기반 코드로 구현했다. 데이터 계층에서는 SSTK/GAIC/public benchmark를 JSONL, sidecar, manifest, summary artifact로 표준화하고, `MobileCropNetV4BatchDataset`에서 image-targetAR task, candidate role, pairwise/listwise label, route/decision/checklist/subject-box target을 tensor contract로 변환했다. 모델 계층에서는 `MobileCropNetV4`가 AR-conditioned proposal query, ROI-style candidate tokenization, RelationLite/set-transformer ranker, utility/positive/risk head, route head, policy/decision head, subject-box head, checklist/why-tag head를 통합한다.

구현 과정에서는 사내 AI 개발 보조 도구인 `Cline SR`을 활용해 대규모 코드베이스 탐색, 실험 스크립트 작성, 오류 로그 분석, 회귀 테스트 범위 확인, 보고서/산출물 연결 정리를 수행했다. 단, 최종 성능 판단은 도구 출력이 아니라 코드 실행 결과와 machine-readable artifact에 기반했다. 각 장기 학습/평가 run은 `train_status.json`, `metrics.json`, `summary.json`, `best.pt`, `latest.pt`, release manifest를 남기도록 설계했으며, downstream 단계는 free-form log가 아니라 JSON summary를 읽어 이어지도록 구성했다.

실험 운영은 로컬 개발 PC, 사내 원격 CPU 서버, MLP GPU workload를 목적에 맞게 분리했다. 단순 검증과 보고서 생성은 프로젝트 지정 Python interpreter로 실행하고, label conversion이나 report build처럼 CPU 부하가 큰 작업은 원격 CPU를 사용했으며, VLM teacher, full model training, A100 latency 같은 GPU 작업은 bounded MLP workload 또는 사용 가능한 interactive GPU run에서 실행했다. 모든 장기 작업은 launch/poll/collect 패턴을 따르며, checkpoint와 summary artifact가 존재하지 않는 partial run은 release evidence에서 제외했다.

검증 체계도 코드화했다. 단위 테스트는 `tests/test_mobilecropnet_v4.py` 및 label conversion, GAIC benchmark, Product-AR label, subject region 관련 테스트로 구성했고, 스크립트 검증은 `py_compile`, `bash -n`, pytest, schema/manifest validation, qualitative review pack, final deployment leaderboard, strict release gate manifest로 수행했다. 이 구조를 통해 단일 metric이 좋은 모델을 바로 배포 후보로 올리지 않고, public benchmark, GAIC official, Product-AR direct eval, head sanity, subject-box calibration, latency, qualitative catastrophic bucket을 같은 checkpoint 기준으로 통합 판정했다.

## 6. AI Governance 보강 문안

아래 문안은 두 마스터 문서의 `AI Governance` 또는 `데이터 사용 적합성` 섹션으로 추가하는 것을 권장한다.

### AI Governance 및 데이터 사용 적합성

본 과제의 데이터 사용은 source별 역할과 사용 범위를 분리해 관리한다. SSTK 계열 데이터는 제품형 pseudo-label factory와 policy/rationale supervision의 중심 source로 사용하고, GAIC/FCDB/CPC/GNMC 등 public benchmark 계열은 public crop quality 비교, teacher ceiling, ranking distillation 검증 목적으로 사용한다. public benchmark의 MOS나 crop preference는 제품 정책 truth로 직접 사용하지 않고, image-local rank, pairwise/listwise target, safe conversion을 거친 보조 supervision으로만 사용한다.

법적/윤리적 리스크 관점에서는 다음 항목을 확인 대상으로 둔다.

| 항목 | 관리 방식 |
| --- | --- |
| 데이터 출처와 사용 범위 | SSTK, GAIC, FCDB, CPC, GNMC 등 source별 목적과 산출물 경로를 manifest에 기록 |
| train/test leakage | image-level deterministic split, Product-AR task 확장 시 동일 image 분리 유지 |
| 인물/텍스트 안전 | face/joint/text cut, headroom/lookroom, copy-space/text cutoff를 reject/checklist/risk label로 관리 |
| public score 오용 방지 | raw MOS/public score를 global truth로 쓰지 않고 safe cap, fatal demotion, contradiction audit 적용 |
| 생성 산출물 사용 | 개념도나 보고서 보조 그림은 evidence로 사용하지 않으며, 성능/좌표/leaderboard는 JSON artifact와 실제 overlay를 source of truth로 사용 |
| 재현성과 감사 가능성 | label_generation_manifest, conversion_summary, train_status, metrics, release_gate_manifest를 durable path에 저장 |

이 원칙은 "좋아 보이는 crop"을 무조건 정답으로 삼는 위험을 막는다. 예를 들어 public score가 높더라도 인물 얼굴/관절/텍스트를 자르거나 SSTK product policy와 충돌하는 후보는 positive로 승격하지 않는다. 또한 모델 평가에서는 성공 예시만 보여 주지 않고 route mismatch, subject-valid mismatch, risk hit, target-AR incompatibility, qualitative catastrophic bucket을 함께 공개해 배포 전 실패를 은폐하지 않는다.

## 7. 현업성과 보강 방향

현재 문서는 기술적으로 깊지만, 평가표의 `현업성과` 항목을 위해서는 비즈니스 관점의 문장을 더 직접적으로 넣는 것이 좋다. 권장 메시지는 다음과 같다.

| 현업성과 관점 | 보강 문구 |
| --- | --- |
| 제품 품질 개선 | MobileCropNet은 다양한 target AR에서 주피사체, 인물 안전, 텍스트/여백, policy action을 함께 고려해 기존 static cropper보다 실패 원인을 구조적으로 줄이는 방향의 cropper다. |
| 차별화 | 단일 bbox 출력이 아니라 route, subject-box, risk, checklist, why-tag를 함께 출력해 제품 검수와 fallback 정책에 사용할 수 있는 설명 가능한 crop surface를 제공한다. |
| 업무 효율화 | candidate generation, teacher scoring, label conversion, benchmark, qualitative pack, release gate가 자동화되어 수작업 crop 검수와 반복 실험 추적 비용을 줄인다. |
| 품질 게이트 | strict release gate가 배포 후보를 보수적으로 차단하므로, public score만 높은 모델이 제품에 들어가는 위험을 줄인다. |
| 향후 제품화 | latency profile과 no-prior runtime path가 구현되어 있어, route/subject/action blocker가 해결되면 모바일/NPU 최적화 단계로 전환할 수 있다. |

주의할 점은 "현재 release gate 통과 모델 없음"을 숨기지 않는 것이다. 대신 이를 "제품 배포 기준을 낮추지 않고 failure mode를 식별한 상태"로 해석해야 한다. 인증과제 관점에서는 최종 배포 성공뿐 아니라 문제 정의, 구현 체계, 검증 체계, 실패 분석 및 다음 실험 설계까지 평가 대상이므로, strict gate의 존재는 오히려 실무형 AI 개발 역량의 근거가 된다.

## 8. 우선순위별 수정 권고

| 우선순위 | 수정/보완 항목 | 대상 문서 | 기대 효과 |
| --- | --- | --- | --- |
| 높음 | `프로그래밍 & 툴활용` 섹션 추가 | 두 문서 모두, 특히 v4 구현 문서 | 평가표의 빈 항목 리스크 제거 |
| 높음 | 사내 도구명 `Cline SR`로 통일 | v4 구현 문서 | 외부 도구 의존 인상 제거 |
| 높음 | `AI Governance 및 데이터 사용 적합성` 섹션 추가 | 학습 데이터 문서 우선, v4 문서에도 요약 | 법적/윤리적 체크 항목 대응 |
| 높음 | 초반에 `평가표 대응 요약` 1페이지 추가 | 두 문서 모두 | 평가자가 빠르게 등급 판단 가능 |
| 높음 | public cropper 비교와 v4 채택 근거 추가 | v4 구현 문서 | `AI 기술/지식이해, AI 모델링` 항목에서 모델 구조 선택의 타당성 강화 |
| 중간 | v4 문서 20.x 실행 일지 정리 또는 부록화 | v4 구현 문서 | 핵심 성과와 현재 상태 가독성 개선 |
| 중간 | on-device 후속 검증 계획 표 추가 | v4 구현 문서 | NPU/INT8/export 미완료 리스크 완화 |
| 중간 | business impact 문구 추가 | 두 문서 결론부 | 현업성과 30점 항목 보강 |
| 중간 | code/artifact index를 본문 평가 항목과 연결 | 두 문서 부록 | 구현 증거 탐색 비용 감소 |
| 낮음 | figure/evidence source 원칙을 governance와 연결 | 두 문서 | 생성 이미지와 실험 증거 혼동 방지 |

## 9. 평가자용 종합의견 초안

본 과제는 제품형 이미지 크롭핑을 단일 좌표 예측 문제가 아니라, target aspect ratio, 후보 생성, 후보 간 ranking, 주피사체 보존, 인물/텍스트 안전, route별 policy action, explanation을 함께 출력하는 온디바이스 AI application 문제로 재정의했다. 학습 데이터 생성 단계에서는 SSTK, GAIC, public crop benchmark의 supervision 의미를 분리하고, teacher score를 raw regression target으로 직접 쓰지 않고 safe conversion, pairwise/listwise preference, decision/checklist/why-tag/subject-box target으로 변환했다.

구현 측면에서는 Python/PyTorch 기반으로 dataset adapter, MobileCropNet v4 모델, 학습 entrypoint, no-prior inference, Product-AR direct evaluation, GAIC/public benchmark, qualitative review pack, final deployment leaderboard, strict release gate를 코드화했다. 사내 AI 개발 보조 도구 `Cline SR`은 코드 탐색, 구현 보조, 로그 분석, 실험 정리, 보고서 연결에 활용했고, 최종 판단은 `train_status.json`, `metrics.json`, `summary.json`, checkpoint, release manifest 같은 재현 가능한 artifact에 기반했다.

현재 strict release gate를 통과한 checkpoint는 없지만, 이는 성과 부재라기보다 제품 배포 기준을 보수적으로 적용해 route collapse, subject-box calibration, policy/action consistency, qualitative catastrophic failure를 식별한 결과로 해석하는 것이 타당하다. 본 과제는 AI 모델링, 사내 개발 도구 활용, 데이터/학습/평가 자동화, on-device 제약을 고려한 runtime 설계, governance-aware safe conversion 측면에서 실무형 AI application 개발 역량을 충분히 보여준다. 후속 보완은 release gate 통과 모델 확보, 모바일/NPU export, INT8/QAT 및 operator coverage 검증, 데이터 사용 권한/윤리 체크 문서화에 집중하면 된다.

## 10. 마스터 문서 반영 내역

본 섹션은 리뷰 권고 후 두 마스터 문서에 실제로 수정/보완 반영한 내용만 기록한다.

### 10.1 학습 데이터 생성 마스터 리포트 반영

대상 문서: `Implement_Docs/MobileCropNet_Training_Data_Generation_Master_Report_KO_2026-04-24.md`

| 반영 위치 | 수정/보완된 내용 |
| --- | --- |
| 1장 서론 | "구현 완료 보고서가 아니다"라는 문구를 "학습 데이터 생성 체계의 방법론 중심 기술보고서"로 완화하고, 구현 증거가 부록 C-E와 v4 구현 마스터 문서에 연결되어 있음을 명시했다. |
| 1.1 `AI Application Specialist 평가표 대응 요약` | 평가표 6개 항목별로 본 문서의 대응 내용과 근거 섹션/산출물을 표로 추가했다. |
| 1.2 `AI 프로그래밍 및 사내 Tool 활용` | 데이터 curation, candidate/subject-support, teacher scoring, label conversion, student contract 구현 경로를 명시했다. JSONL/sidecar/manifest 기반 데이터 계약, Cline SR 활용 범위, 로컬/원격/MLP 실행 운영, partial run 제외 원칙을 추가했다. |
| 1.3 `AI Governance 및 데이터 사용 적합성` | SSTK와 public benchmark source의 역할 분리, 법적/권한 확인 대상, image-level split, 인물/텍스트 안전, raw public score 오용 방지, 생성 산출물 비증거 원칙, durable artifact 감사를 표로 추가했다. |
| 1.4 `현업성과와 기대효과` | 제품 품질 개선, 제품 차별화, 업무 효율화, 품질 게이트, 온디바이스 모델 학습 연결성을 현업성과 관점으로 정리했다. strict release gate 미통과 상태를 제품 실패 요인 식별 결과로 해석하는 문구를 추가했다. |

### 10.2 MobileCropNet v4.0 구현 마스터 리포트 반영

대상 문서: `Implement_Docs/MobileCropNet_v4_0_Implementation_Master_Report_KO_2026-04-27.md`

| 반영 위치 | 수정/보완된 내용 |
| --- | --- |
| 1.1 `AI Application Specialist 평가표 대응 요약` | 평가표 6개 항목별로 v4 구현 문서의 대응 내용과 근거 섹션/산출물을 표로 추가했다. |
| 1.6 `AI 모델링 설계 채택 근거와 Public Cropper 비교` | 기존 문서에 부족했던 public cropper 대비 설계 채택 근거를 별도 섹션으로 추가했다. GAIC/CGS/CACNet, static/heuristic cropper, sequential/view-adjustment, VLM cropper, UCTR/public hybrid teacher의 구조, 학습 방식, 강점, 제품형 MobileCropNet 관점의 한계, v4 설계 반영점을 비교했다. 또한 GAIC v2 test 기준 public GAIC/CGS와 초기 MobileCropNet v4 q24 profile의 SRCC/top1 MOS 차이를 명시해, v4의 목적이 단순 GAIC scorer 대체가 아니라 no-prior proposal, target-AR compatibility, action policy, subject safety, route/rationale consistency, latency profile을 하나의 student contract로 묶는 것임을 설명했다. |
| 1.2 `AI 프로그래밍 및 사내 Tool 활용` | `data.py`, `model.py`, `eval_utils.py`, `train_mobilecropnet_v4.py`, direct/GAIC/public/head/subject/qualitative/leaderboard/release gate scripts의 구현 역할을 정리했다. Cline SR 활용 범위와 최종 판단 기준이 code/test/artifact임을 명시했다. |
| 1.3 `AI Governance 및 데이터/모델 사용 적합성` | teacher supervision과 deployment runtime input 분리, data source 분리, raw score 오용 방지, 개인정보/인물 안전 bucket, train/test leakage 방지, 생성 산출물 관리, strict release gate 책임성을 표로 추가했다. |
| 1.4 `On-Device 제품화 보강 계획` | no-prior runtime, latency, operator coverage, INT8/QAT, memory footprint, pre/postprocess를 현재 상태와 후속 검증 기준으로 정리했다. 품질 gate 통과 후 모바일/NPU 최적화로 넘어가는 순서를 명시했다. |
| 1.5 `현업성과와 비즈니스 기여` | 제품 품질 개선, 설명 가능한 crop 결과, 자동화된 반복 실험, 배포 리스크 감소, profile 확장성을 현업성과 표로 추가했다. "배포 후보 없음"을 end-to-end 체계 구현 후 strict gate가 남은 blocker를 식별한 상태로 재해석했다. |
| 18장 `현재 강점` | 외부 도구명으로 오해될 수 있는 GPU workflow 표현을 `Cline SR 기반 GPU workflow`로 교체했다. |
