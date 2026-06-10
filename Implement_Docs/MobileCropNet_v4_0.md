---
title: "SSTK-MobileCropNet v4.0"
subtitle: "Teacher-topology-agnostic learned proposal + relation-aware set ranking cropper\n최상의 결과/서비스와 온디바이스 배포를 동시에 겨냥한 재설계 완성본"
author: "OpenAI"
date: "2026-04-15"
lang: ko-KR
numbersections: true
toc: true
toc-depth: 3
fontsize: 10.5pt
geometry: margin=0.82in
mainfont: "Noto Sans CJK KR"
CJKmainfont: "Noto Sans CJK KR"
sansfont: "Noto Sans CJK KR"
monofont: "Noto Sans Mono CJK KR"
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
header-includes:
  - |
    ```{=latex}
    \usepackage[dvipsnames]{xcolor}
    \usepackage{amsmath,amssymb,mathtools}
    \usepackage{booktabs,longtable,array,tabularx,multirow}
    \usepackage{graphicx,float,caption}
    \usepackage{enumitem}
    \usepackage{fvextra}
    \fvset{breaklines=true,breakanywhere=true,fontsize=\small}
    \captionsetup{font=small,labelfont=bf}
    \setlength{\parskip}{0.35em}
    \setlength{\parindent}{0pt}
    \setlength{\emergencystretch}{3em}
    \renewcommand{\arraystretch}{1.12}
    \linespread{1.05}
    ```
---

# Executive Summary

본 문서는 `SSTK-MobileCropNet v3.3`의 핵심 중 **유지해야 할 것**과 **버려야 할 것**을 분리한 뒤, `SSTK teacher semantics`와의 런타임 정합성을 **의도적으로 배제**하고도 더 강한 품질과 더 나은 서비스를 줄 수 있는 `v4.0` 설계를 원점부터 다시 세운 완성본이다.

핵심 판단은 단순하다.

> **v4.0의 연구/품질 메인라인은 더 이상 `static micro-bank`도 아니고, teacher 6-block runtime 복제도 아니다.**  
> **기본형은 `learned global proposal generator + relation-aware set ranker + baseline-aware policy head + optional local refiner`다.**

이 판단의 이유는 세 가지다.

1. 현재 SSTK는 이미 `train_conditional_detr_batch.jsonl`, `train_pairwise.jsonl`, `train_listwise.jsonl`, `train_decision.jsonl`, `train_checklist.jsonl`, `train_regression.jsonl`로 이어지는 **풍부한 감독 신호**를 갖고 있다. 즉, teacher 파이프라인 모양을 student runtime에 복제하지 않아도 proposal, ranking, policy, explanation을 따로 학습할 수 있다. [I1][I2][I3]
2. 기존 이미지 크롭핑 문헌을 다시 정리하면, 좌표를 직접 회귀하는 방법은 **global diversity / no-crop 정책 / explainability**가 약하고, pure candidate ranking은 **candidate recall ceiling**이 있으며, pure set prediction은 **pairwise/listwise/policy supervision 활용**이 약하다. 따라서 **proposal과 ranking을 분리한 하이브리드**가 가장 타당하다. [E1]–[E24]
3. 온디바이스 배포를 생각하더라도, `split runtime`과 `host analytic bank`는 **알고리즘적 중심 원리**가 아니라 **export 안정화용 최적화 옵션**으로 위치를 낮추는 것이 맞다. v4.0의 알고리즘은 end-to-end learned proposal + learned ranking + learned policy이며, 배포 단계에서만 필요한 범위로 fixed-shape/quantization/split export를 적용한다. [E22]–[E28]

v4.0의 최종 권고는 아래 다섯 줄로 요약된다.

1. **candidate-first와 baseline-relative decision semantics는 유지한다.**
2. **`static micro-bank`는 연구 메인라인이 아니라 fallback/distillation/deployment baseline으로 내린다.**
3. **core model은 `learned proposal + relation-aware ranking + policy + refiner`로 재구성한다.**
4. **backbone은 `RepViT-M2`를 기본 shipping 후보, `MobileNetV4-Hybrid-M`를 고품질 대안으로 둔다.**
5. **export 성공, INT8 drift 관리, NPU coverage, CPU fallback 0은 품질 지표와 같은 급의 release gate로 둔다.**

![설계 공간과 v4.0 선택](assets/fig01_paradigm_map.png){ width=96% }

# 1. 문서 목적과 범위

이 문서는 다음 독자를 상정한다.

- SSTK를 처음 보는 모델 엔지니어
- `v3.3`의 split-runtime/static-bank가 왜 바뀌는지 이해해야 하는 연구 엔지니어
- 실제로 `PyTorch -> ONNX/LiteRT/QNN` 배포까지 책임지는 온디바이스 엔지니어
- 추후 논문화와 실험 설계를 함께 염두에 두는 연구 PM

문서가 해결하려는 질문은 네 개다.

1. teacher topology 정합성을 버리면 어떤 모델이 **가장 타당한 메인라인**인가?
2. 그 모델은 기존 이미지 크롭핑 논문들과 비교해 **무엇을 가져오고 무엇을 버리는가?**
3. 현재 SSTK 라벨 산출물은 그 모델의 **어떤 head / loss / 학습 단계**로 들어가는가?
4. 품질 우선 설계를 해치지 않으면서 **실제 모바일/NPU 배포**까지 어떻게 가져갈 것인가?

본 문서는 다음 내부 문서를 직접 이어받는다.

- [I1] `SSTK_Current_Implementation_Master_KO_2026-04-08-v2.md`
- [I2] `SSTK_Cropping_DataFactory_Master_KO_v3_1.md`
- [I3] `SSTK-MobileCropNet_v3_3.md`
- [I4] `Conditional DETR 모델 분석.txt`
- [I5] `문서 통합 및 업데이트.txt`
- [I6] `AI 기반 이미지 크로핑.txt`

단, `v4.0`의 출발점은 명확하다.

- **유지**: candidate-first, baseline-relative decision, 현재 라벨 계약, explainability/checklist 철학
- **배제**: `teacher semantics`와 동일한 런타임 topology, `split runtime + static bank`를 core algorithm으로 두는 전제

# 2. 문제 재정의: 무엇을 유지하고 무엇을 버릴 것인가

## 2.1 v3.3에서 유지해야 하는 것

`v3.3`이 옳게 잡아 둔 축은 여전히 유효하다.

- SSTK는 본질적으로 **candidate-aware selection + policy** 문제다.
- 최종 선택은 단순 top-1 bbox가 아니라 **baseline 대비 정말 자를 가치가 있는가**의 문제다.
- `keep_full / minimal_crop / crop`는 class prediction처럼 보이지만 실제로는 **policy calibration** 문제다.
- 실패는 보통 `candidate recall miss + ranking error + policy error`로 분해해야 원인이 보인다. [I1][I3][I5]

## 2.2 v3.3에서 버려야 하는 것

반대로 다음 두 전제는 더 이상 v4.0의 중심이 아니어야 한다.

### A. `static micro-bank`를 quality-optimal candidate source로 보는 전제

`static micro-bank`는 export 안정성, host analytic simplicity, fixed-shape runtime에는 유리하지만, 좋은 crop이 여러 개 존재하는 문제에서 **candidate recall ceiling**을 만든다. 특히 사람/오브젝트/scene/copyspace/text가 섞인 SSTK 상황에서는, 고정 슬롯이 실제 teacher positive manifold를 전부 덮는다는 보장이 없다. [I3][E5][E14]

### B. `split runtime`을 설계 원리로 보는 전제

`Gate + bank + ScoreA + ScoreB (+ Refine)`는 shipping-friendly일 수는 있지만, teacher topology mirror링이 강해질수록 라벨이 가진 감독 신호를 **task별로 최적으로 쓰기 어렵다.** v4.0에서 split runtime은 **deployment profile**이지 **알고리즘 정의**가 아니다. 알고리즘은 하나이고, export 단계에서만 필요한 범위로 분할한다. [I3][E27][E28]

## 2.3 v4.0에서의 핵심 재정의

v4.0은 teacher가 만든 산출물을 **모듈 복제용 spec**이 아니라 **감독 신호 묶음**으로 읽는다. 현재 라벨은 자연스럽게 네 과업으로 분해된다.

| 라벨/산출물 | 본질적 의미 | v4.0에서의 사용처 |
|---|---|---|
| `matching_targets` | positive crop set | proposal / positive assignment |
| `candidate_pool` + `pairwise` + `listwise` | 상대 선호와 shortlist 구조 | relation-aware ranker |
| `baseline` + `decision_target` | no-crop / minimal-crop 정책 | policy head |
| `checklist` + `macro` + `why_tags` | 설명 가능한 품질 축 | auxiliary head / audit |

즉 v4.0은 teacher semantics를 버리는 것이 아니라, **teacher가 이미 준 proposal/ranking/policy/explanation supervision을 더 정직한 구조로 재배열**하는 것이다.

![현재 라벨 산출물과 v4.0 헤드의 대응 관계](assets/fig03_label_mapping.png){ width=96% }

# 3. 현재 SSTK 데이터/라벨 계약에서 출발해야 하는 이유

## 3.1 현재 구현과 산출물은 이미 충분히 성숙하다

현재 SSTK 파이프라인은 단순 bbox 생성기가 아니라 **filter -> precompute(C1~C7) -> subject routing -> crop guidance -> candidate generation -> teacher scoring -> training labels -> benchmark/CI/report**로 닫힌 end-to-end crop data factory다. 현재 채택 lane은 `single_stage2` profile이며, synced full validation에서 `Gc/Ge Spearman = 0.510`, production winner IoU to GT MOS best `0.689`, GT percentile `0.747`, training-label validation `ok`, main pool monotonic violation `0`가 보고되어 있다. [I1]

이 말은 곧 다음을 뜻한다.

- 지금 필요한 것은 “새 데이터 포맷을 또 만든다”가 아니다.
- 지금 필요한 것은 **현재 라벨 산출물을 가장 잘 쓰는 모델 구조**를 고르는 일이다.

## 3.2 v4.0이 바로 읽어야 하는 현재 라벨 계약

`v3.3` 문서가 정리하듯, `train_conditional_detr_batch.jsonl`의 한 row는 이미 `(image_id, target_ar)` 그룹 하나에 대한 완전한 supervision 묶음이다. 현재 평균값은 대략 아래와 같다. [I3]

| 항목 | 의미 | 현재 평균 |
|---|---|---:|
| `matching_targets` | positive supervision set | 2.151 / group |
| `candidate_pool` | main negative / comparison pool | 20.584 / group |
| `ignored_candidates` | safe leftover / monotonic prune | 3.048 / group |
| `overflow_candidates` | unsafe / hard audit bucket | 1.868 / group |

현재 group 분포도 충분히 다양하다. [I3]

- target AR: `FREE, 1:1, 9:16, 16:9, 3:4, 4:3`
- subject mode: `object_single`, `background_texture_copyspace`, `portrait_single`, `scene_general`, `object_multi`, `portrait_group`, `other_ambiguous`
- decision type: `minimal_crop`, `crop`, `keep_full`

즉, v4.0은 “이 라벨로 proposal을 못 배운다”가 아니라, **proposal용 signal과 ranking/policy용 signal을 분리해서 읽지 않았기 때문에 아직 못 쓴 것**에 가깝다.

## 3.3 v4.0의 데이터 사용 원칙

- `batch/canonical/pairwise/listwise/decision/checklist/regression` artifact는 그대로 사용한다.
- 새 `v4` 전용 물리 JSON을 먼저 만들지 않는다.
- 학습 파이프라인은 **adapter view**만 추가한다.
- 현재 `validation_summary.status == ok`, main monotonic violation `0`, `safe_leftover_policy=ignore`를 전제로 한다. [I1][I3]

# 4. 기존 이미지 크롭핑 방법론 전수 재검토

## 4.1 큰 그림: 이미지 크롭핑은 다섯 계열로 나뉜다

기존 논문은 세부 구현은 달라도, 큰 틀에서는 아래 다섯 계열로 묶인다.

1. **pairwise/listwise ranking 계열**
2. **candidate scoring / relation modeling 계열**
3. **직접 좌표 회귀 / cascaded regression 계열**
4. **RL / action / meta-learning / set-prediction 계열**
5. **weak supervision / retrieval / VLM / composition-aware hybrid 계열**

v4.0을 정하기 위해서는 이들을 “어떤 논문이 유명한가”가 아니라, 아래 기준으로 다시 봐야 한다.

- 다중 정답(non-uniqueness)을 얼마나 잘 수용하는가?
- pairwise/listwise + decision supervision을 얼마나 직접 활용하는가?
- no-crop / minimal-crop을 자연스럽게 다루는가?
- relation modeling과 explainability에 얼마나 유리한가?
- mobile/NPU export를 얼마나 현실적으로 만들 수 있는가?

## 4.2 대표 문헌 비교표

아래 표는 SSTK와 직접적으로 연결되는 핵심 문헌만 추려 **무엇을 배워야 하는지** 중심으로 다시 정리한 것이다.

| 연도 | 방법 | 계열 | 핵심 아이디어 | 장점 | 한계 | v4.0에 남기는 교훈 |
|---:|---|---|---|---|---|---|
| 2017 | Learning to Compose with Professional Photographs on the Web [E1] | pairwise ranking | professional photo에서 ranking pair 학습 | comparative preference 학습이 강함 | explicit no-crop 정책이 없음 | pairwise는 여전히 핵심 supervision |
| 2017 | Deep Cropping [E2] | regression + candidate refinement | attention box regression + aesthetic classification | global한 초기 box를 빠르게 찾음 | 다중 정답/정책 분리가 약함 | refiner는 top-1/top-2 보정에만 쓰는 것이 안전 |
| 2018 | A2-RL [E3] | RL/action | crop window를 sequential action으로 이동 | 연속 탐색 가능 | 학습/배포 복잡도 높음 | RL은 주력보다 확장 옵션이 적절 |
| 2018 | Good View Hunting / CPC [E4] | pairwise + fast proposal | 100만+ comparative view pairs, fast VPN | dense comparative supervision | no-crop 정책 부재 | pairwise와 proposal 분리는 유효 |
| 2019 | GAIC [E5] | grid-anchor benchmark | exhaustive annotation, reliable benchmark | non-unique crop와 candidate evaluation 정당화 | static candidate space ceiling | candidate-first는 맞지만 proposal recall이 핵심 |
| 2019 | LVRN [E6] | listwise ranking | image cropping은 listwise task라고 명시 | shortlist 구조 학습 | refined sampling 의존 | listwise는 v4.0 ranker의 핵심 |
| 2020 | ASM-Net [E7] | score map | composition-aware + saliency-aware score map | 해석 가능성이 좋음 | score map만으로 policy를 설명하긴 약함 | composition prior를 중간표현으로 심는 건 유효 |
| 2020 | Composing Good Shots [E8] | relation graph | 후보들 사이 mutual relation 모델링 | comparative nature를 잘 반영 | 별도 candidate set 필요 | relation-aware ranker 필요 |
| 2020 | MARS [E9] | meta-learning / AR-conditioned | target AR별 adaptation | AR conditioning을 명시적으로 다룸 | policy/no-crop 설명은 약함 | AR token은 유지해야 함 |
| 2021 | CACNet [E10] | composition-aware hybrid | composition branch + cropping branch | explicit composition rule 학습 | crop ranking/정책 전체를 대신하진 못함 | composition auxiliary가 유용 |
| 2021 | TransView [E11] | transformer relation modeling | inside/outside/across boundary 관계 모델링 | 경계 바깥 정보의 중요성 반영 | operator cost 증가 | ROI-free inside/boundary/context pooling으로 흡수 |
| 2022 | Re-Compose [E12] | design constraint | 단순 점수 이상, 구성요소별 평가 | constrained crop에 적합 | 일반 crop 메인라인으로는 과함 | copyspace/text-safe constraint는 별도 head가 낫다 |
| 2022 | Human-centric Image Cropping [E13] | human-centric ranking | partition-aware + content-preserving feature | 인물 장면에서 매우 강함 | 범용 crop 전체를 대체하긴 어려움 | person slice에 partition-aware cue 필요 |
| 2022 | Rethinking Image Cropping [E14] | set prediction | image cropping을 set prediction으로 재정의 | global proposal + diversity | pair/list/policy supervision 활용이 약함 | proposal module에는 매우 적합 |
| 2023 | Spatial-aware Feature and Rank Consistency [E15] | spatial-aware ranking | crop mask + aggregated feature + rank consistency | spatial relationship를 명시화 | generator 전체를 대체하진 않음 | mask-aware box pooling이 중요 |
| 2023 | ClipCrop [E16] | VLM-conditioned crop | text/image query로 user intent 반영 | conditioned crop에 강함 | general mobile crop mainline으로는 무거움 | intent-conditioned 서비스의 서버형 옵션 |
| 2024 | GenCrop [E17] | weak supervision | stock photo + outpainting으로 pair 생성 | 수작업 라벨 없이도 강함 | runtime 모델 그 자체는 아님 | weakly-supervised pretraining source로 매우 유효 |
| 2025 | Cropper [E18] | VLM + ICL | free-form/subject-aware/AR-aware unified | 범용성과 설명력이 좋음 | 런타임 비용이 큼 | server/HQ teacher나 hard-case adjudicator 용도 |
| 2025 | ProCrop [E19] | retrieval composition | professional composition retrieval + fusion | weak-supervised 규모 확대 | retrieval 비용이 큼 | teacher/HQ re-ranker·pretraining source에 유용 |
| 2025 | AesCrop [E21] | composition-aware hybrid | composition bias + decoder hybrid | composition guidance의 최근형 | VMamba/decoder export 부담 | composition attention은 연구용 HQ에만 참고 |

## 4.3 문헌이 공통으로 말하는 것

문헌을 다시 모아 보면 결국 세 문장으로 요약된다.

### 첫째, 좋은 crop은 하나가 아니다

GAIC가 가장 강하게 보여 준 사실은 **single GT + IoU**로는 image cropping을 제대로 정의하기 어렵다는 점이다. `Rethinking Image Cropping`도 같은 문제를 set prediction으로 다시 푼다. 즉 v4.0은 단일 bbox 회귀로 가면 안 된다. [E5][E14]

### 둘째, crop quality는 절대 점수보다 상대 비교가 강하다

Learning to Compose, CPC, LVRN, Composing Good Shots, TransView는 모두 서로 다른 방식으로 **comparative judgment**가 핵심이라고 말한다. 이는 현재 SSTK 라벨의 `pairwise + listwise + decision`이 오히려 문헌 정류에 아주 잘 맞는다는 뜻이다. [E1][E4][E6][E8][E11]

### 셋째, global proposal과 relative ranking은 분리하는 편이 낫다

Deep Cropping, A2-RL, MARS, Rethinking, AesCrop은 “어떻게 전역적으로 좋은 box 후보를 찾을 것인가”를 다루고, TransView / Human-centric / Spatial-aware / Composing Good Shots는 “그 후보들을 어떻게 더 잘 비교할 것인가”를 다룬다. 둘을 한 블록에 무리하게 합치기보다 **proposal과 ranking을 분리**하는 것이 가장 안정적이다. [E2][E3][E9][E14][E21]

# 5. 설계 공간 분석: 왜 최종 선택은 하이브리드인가

## 5.1 비교 기준

v4.0은 아래 여섯 질문으로 설계 후보를 평가한다.

1. **다중 정답**을 얼마나 자연스럽게 다루는가?
2. **pairwise/listwise/decision** 라벨을 얼마나 직접 활용하는가?
3. **no-crop / minimal-crop** 정책을 얼마나 깔끔하게 분리하는가?
4. **global search와 local refinement**를 동시에 달성하는가?
5. **설명 가능성**과 체크리스트 복원이 쉬운가?
6. **모바일/NPU export**까지 현실적으로 연결되는가?

## 5.2 후보 비교표

| 옵션 | global search | diversity | pair/list 활용 | no-crop policy | explainability | mobile fit | v4.0 판단 |
|---|---|---|---|---|---|---|---|
| static micro-bank + ranker | 낮음~중간 | 중간 | 높음 | 높음 | 높음 | 매우 높음 | **fallback/deployment baseline** |
| pure candidate ranker on teacher replay | 없음 | 낮음 | 매우 높음 | 높음 | 높음 | 높음 | recall 병목 때문에 메인라인 불가 |
| pure coordinate regression | 높음 | 낮음 | 낮음 | 낮음 | 낮음 | 중간 | 메인라인 부적합 |
| RL / action cropper | 높음 | 중간 | 낮음 | 중간 | 낮음 | 낮음 | 확장 옵션 |
| pure set prediction (Crop-DETR) | 높음 | 높음 | 중간 | 낮음~중간 | 중간 | 중간 | proposal module로는 훌륭, final scorer로는 아쉬움 |
| VLM / retrieval cropper | 높음 | 높음 | 낮음 | 중간 | 높음 | 매우 낮음 | server/HQ only |
| **learned proposal + relation-aware ranker + policy** | **높음** | **높음** | **매우 높음** | **높음** | **높음** | **중간~높음** | **최종 선택** |

## 5.3 왜 pure Crop-DETR가 아니고, 왜 pure ranking도 아닌가

### pure Crop-DETR가 아닌 이유

set prediction은 proposal 다양성과 global search에는 강하다. 하지만 SSTK의 가장 강한 감독 신호인 `pairwise/listwise/decision/checklist`를 끝까지 살리기에는 부족하다. 또한 DETR의 one-to-one matching만 쓰면 cropping의 다중 정답 구조에 비해 supervision이 너무 sparse해진다. 따라서 proposal 모듈로는 좋지만 final scorer/policy 전체를 맡기기에는 아깝다. [E14][E22][E23][E24]

### pure ranking이 아닌 이유

반대로 teacher candidate replay만으로 ranker를 잘 학습할 수는 있어도, 좋은 crop이 replay 후보 풀 바깥에 있으면 그 순간 ceiling에 부딪힌다. 즉 pure ranking은 `candidate recall miss`를 절대 해결하지 못한다. [I5][E5]

### 그래서 하이브리드다

v4.0의 선택은 명확하다.

- **proposal**은 set-prediction / real-time DETR 계열에서 가져오고,
- **ranking**은 pairwise/listwise/relation modeling 계열에서 가져오며,
- **policy**는 baseline-aware decision head로 분리하고,
- **refiner**는 deep cropping류의 local delta만 제한적으로 사용한다.

![v4.0 전체 아키텍처](assets/fig02_v40_architecture.png){ width=96% }

# 6. MobileCropNet v4.0: 최종 제안 아키텍처

## 6.1 이름과 정의

- **이름**: `SSTK-MobileCropNet v4.0`
- **정식 해석**: *Teacher-topology-agnostic learned proposal + relation-aware set ranking cropper*
- **핵심 구조**:
  1. shared mobile encoder
  2. learned proposal generator
  3. operator-safe box pooling
  4. relation-aware set ranker
  5. baseline-aware policy head
  6. optional local refiner
  7. optional explanation / macro auxiliary heads

## 6.2 기본 표기

- 입력 이미지: $I \in \mathbb{R}^{H \times W \times 3}$
- target aspect ratio: $r$
- baseline 후보 집합: $\mathcal{B}_{base}(I,r)$
- learned proposal 집합: $\mathcal{B}_{prop}(I,r)$
- 최종 비교 후보 집합:

$$
\mathcal{C}(I,r)=\mathcal{B}_{base}(I,r) \cup \mathcal{B}_{prop}(I,r)
$$

- rank utility: $u_i$
- risk / safety logit: $\rho_i$
- policy score: $s_i^{policy}$
- baseline: $b_{base}$
- 최선 후보: $b^\star$

## 6.3 모듈 1 — shared mobile encoder

### 설계 목표

- one-pass 특징 추출
- multi-scale 정보 유지
- fixed-shape NPU export 친화성
- proposal과 ranking이 feature cache를 공유

### 입력 전처리

온디바이스에서는 variable-size 원본 이미지를 그대로 모델에 넣기보다,

1. 원본 비율을 유지한 채 long-side resize
2. square tensor로 zero/reflect padding
3. 원본 image AR과 padding meta를 별도 scalar/token으로 전달

하는 편이 훨씬 안전하다. v4.0은 기본적으로 `256 / 288 / 320` long-side profile을 권장한다.

### 권장 backbone

| 백본 | 구조적 성격 | 장점 | 약점 | 권장 역할 |
|---|---|---|---|---|
| **RepViT-M2** [E25] | pure lightweight CNN + re-parameterization | conv 중심이라 export/INT8/NPU에 유리, latency 안정적 | representation ceiling은 hybrid 모델보다 낮을 수 있음 | **기본 shipping backbone** |
| **MobileNetV4-Hybrid-M** [E26] | UIB + Mobile MQA hybrid | 정확도/지연 균형이 매우 좋고 다양한 accelerator에 강함 | attention op 지원이 약한 backend에서는 검증 필요 | **품질 우선 대안** |
| ViT-Small / DeiT-Tiny | transformer | 학습 편의성 | 모바일 지연/메모리 불리 | 비권장 |
| VMamba 계열 | state-space | 최근 composition hybrid 연구에 쓰임 | mobile export와 연산자 안전성 검증 부담 | HQ research only |

### 최종 권고

- **기본 제품형**: `RepViT-M2`
- **고품질 실험형**: `MobileNetV4-Hybrid-M`
- **미선정 이유가 명확한 백본**은 문서에 남기되, shipping shortlist에서는 제외한다.

## 6.4 모듈 2 — learned proposal generator

### 왜 필요한가

현재 SSTK 라벨은 ranking/policy에는 매우 강하지만, 좋은 crop이 후보 집합에 없으면 아무 소용이 없다. proposal generator는 바로 이 recall ceiling을 깨기 위한 모듈이다.

### 구조

proposal generator는 `Conditional DETR`와 `RT-DETR`에서 아이디어를 가져온 **AR-conditioned lightweight query decoder**로 둔다. [E22][E23]

- shared encoder feature $F$
- AR token $e_r$
- global image token $g$
- learnable query $q_i, i=1..Q$

각 query는 아래를 예측한다.

- proposal box $\hat b_i$
- objectness / proposal quality $\hat o_i$
- proposal token $t_i^0$

### AR-constrained proposal parameterization

2026-04-17 패치부터 fixed target AR(`1:1`, `9:16`, `16:9`, `3:4`, `4:3`)에서는 proposal box를 자유 `cx,cy,w,h` 회귀로 두지 않는다. 이전 구현은 target AR token을 조건으로 넣었지만 bbox head 자체는 4자유도 `cxcywh`를 직접 예측했기 때문에, `target_ar=3:4` row에서도 proposal top-1이 1:1에 가까운 박스로 나올 수 있었다. 이는 ranking crop(red)과 proposal head top-1(orange)이 항상 달라 보이는 주요 원인이었다.

새 구현은 bbox head의 raw sigmoid 출력 중 `cx`, `cy`, `scale`만 fixed AR proposal에 사용한다. 단순히 square letterbox 전체 `[0,1]`에 target AR을 맞추면 box가 padding 영역으로 넘어간 뒤 원본 좌표 복원에서 clipping되어 실제 crop AR이 깨질 수 있다. 따라서 data adapter가 실제 resized content 영역 `letterbox_content_box=[x_0,y_0,x_1,y_1]`를 모델에 전달하고, fixed AR proposal은 이 content rect 내부에서만 생성한다.

원본 target AR을 $r$, 원본 이미지 AR을 $a=W/H$, letterbox content 크기를 $C_w,C_h$라고 하면, letterbox 좌표계에서 써야 하는 보정 AR은 $r_{lb}=r(C_w/C_h)/a$다. 이 값을 기준으로 `scale`이 content rect 안의 최대 fixed-AR crop 크기를 조절한다.

$$ w=s\min(C_w,r_{lb}C_h),\quad h=w/r_{lb},\quad c_x=x_0+w/2+\sigma_x(C_w-w),\quad c_y=y_0+h/2+\sigma_y(C_h-h) $$

`FREE` target AR만 기존 자유 `cxcywh` 경로를 유지한다. 이렇게 하면 fixed AR proposal은 모델 구조상 원본 crop target AR을 만족하고, AR 불일치 correction을 ranking이나 visualizer에 떠넘기지 않는다. 평가 JSON에는 각 proposal별 `target_ar`, `target_ar_value`, `crop_ar`, `letterbox_ar`, `target_ar_log_error`, `target_ar_compatible`를 기록한다. 시각화의 orange box는 raw proposal top-1이 아니라 target-AR compatible proposal top-1을 우선 표시한다.

### query 수

- HQ: $Q=32$
- Balanced: $Q=24$
- Turbo: $Q=16$

### 왜 `Conditional DETR-lite` 사고방식인가

Conditional DETR은 decoder가 conditional spatial query를 학습해 localization difficulty를 줄인다. crop proposal도 object detection처럼 “어디를 볼지”가 중요하므로, vanilla DETR보다 conditional spatial query가 훨씬 적합하다. [E22]

### 왜 `RT-DETR` 사고방식도 섞는가

RT-DETR은 multi-scale feature 처리를 효율화하고, end-to-end real-time detection을 실제 지연 예산 안으로 밀어 넣는다. crop proposal도 detection과 비슷하게 **Q fixed, NMS 없는 learned proposal**이므로, real-time DETR의 효율화 트릭을 가져오는 것이 합리적이다. [E23]

### 왜 one-to-many dense positive가 필요한가

cropping은 detection보다 다중 정답성이 강하다. 따라서 Hungarian one-to-one만 쓰면 query supervision이 지나치게 sparse해진다. v4.0은 primary Hungarian match는 유지하되, RT-DETRv3의 문제의식처럼 **dense positive auxiliary**를 함께 둔다. [E24]

positive set을 다음처럼 정의한다.

$$
\mathcal{P}^+(I,r)=\mathcal{M}(I,r) \cup \left\{c \in \mathcal{C}_{teach} : p_{util}(c) \ge \tau_{soft+},\; c\;\text{admissible} \right\}
$$

여기서 $\mathcal{M}$은 `matching_targets`다.

query별 objectness target은

$$
y_i^{obj} = \mathbf{1}\left[\max_{b \in \mathcal{P}^+} \operatorname{IoU}(\hat b_i, b) \ge \eta_{pos}\right]
$$

으로 둔다.

### proposal loss

$$
L_{prop} = \lambda_b L_{box} + \lambda_o L_{obj} + \lambda_d L_{dense+} + \lambda_v L_{div}
$$

- $L_{box}$: matched positive에 대한 box regression / GIoU
- $L_{obj}$: objectness BCE 또는 focal BCE
- $L_{dense+}$: one-to-many dense positive assignment
- $L_{div}$: proposal collapse를 막는 diversity regularization

### proposal이 꼭 baseline을 포함해야 하는가?

proposal generator는 learned branch지만, **baseline 후보는 항상 별도로 합친다.** 즉 baseline은 proposal failure에 휘둘리지 않는 safety lane이다. 이것이 v4.0이 candidate-first semantics를 유지하는 방식이다.

## 6.5 모듈 3 — operator-safe box pooling

### 문제

relation-aware ranking을 하려면 box feature가 필요하다. 하지만 `ROIAlign / CropAndResize / grid_sample`는 backend별 export risk가 크고, NPU coverage가 불안정할 수 있다.

### 해결책

v4.0은 low-resolution feature map 위에서 **inside / boundary / context**를 추출하되, operator-friendly하게 한다.

기본 구현은 16×16 또는 18×18 feature grid 위에 box raster mask를 만든 뒤 masked average pooling을 하는 방식이다.

$$
f_i^{in} = \frac{\sum_{u,v} M_i^{in}(u,v) F(u,v)}{\sum_{u,v} M_i^{in}(u,v)+\epsilon}
$$

$$
f_i^{bd} = \frac{\sum_{u,v} M_i^{bd}(u,v) F(u,v)}{\sum_{u,v} M_i^{bd}(u,v)+\epsilon}
$$

$$
f_i^{ctx} = \frac{\sum_{u,v} M_i^{ctx}(u,v) F(u,v)}{\sum_{u,v} M_i^{ctx}(u,v)+\epsilon}
$$

이 방식은 TransView의 inside/outside/across 아이디어를 살리면서도, 배포 경로에서는 `Mul + Reduce` 위주로 구현할 수 있다. [E11][E15]

### 왜 이게 중요한가

- crop 내부만 보면 안 된다.
- 경계에서 무엇을 자르는지 봐야 한다.
- 경계 밖 context가 얼마나 남는지도 봐야 한다.
- 그러나 mobile export에서 ROI hot path는 피해야 한다.

따라서 v4.0은 **box-aware이지만 ROIAlign-free**인 특징 추출을 기본값으로 둔다.

## 6.6 모듈 4 — relation-aware set ranker

### 역할

proposal generator가 “좋은 후보가 후보 공간 안에 존재하게 하는 모듈”이라면, relation-aware set ranker는 “그 후보들 중 무엇을 실제로 채택할지 결정하는 모듈”이다.

### 왜 relation-aware인가

Composing Good Shots는 candidate 간 mutual relation이 중요하다고 말하고, TransView는 inside/outside/boundary across relation modeling이 중요하다고 말한다. Human-centric와 Spatial-aware 역시 crop 자체의 local feature만으로는 부족하고, crop와 주변/다른 후보의 관계가 중요함을 보였다. [E8][E11][E13][E15]

### HQ와 Mobile의 구현 차이

#### HQ ranker

- 2-layer lightweight self-attention / Set Transformer
- token dimension 160~192
- top-K = 16

#### Mobile ranker (`RelationLite`)

full self-attention 대신 다음 형태의 pairwise summary를 쓴다.

$$
r_i = \operatorname{Agg}_j \phi\left([t_i, t_j, \Delta g_{ij}, \operatorname{IoU}_{ij}]\right)
$$

$$
\tilde t_i = \psi([t_i, r_i, t_{base}, g, e_r])
$$

여기서

- $t_i$: candidate token
- $\Delta g_{ij}$: 두 후보의 geometry 차이
- $t_{base}$: baseline token
- $g$: global image token
- $e_r$: AR token

즉 full transformer보다 계산량이 적지만, “후보들 사이의 비교 구조”는 유지한다.

### 출력

ranker는 candidate별로 아래를 낸다.

- rank utility $\hat u_i$
- risk logit $\hat \rho_i$
- optional macro heads $\hat A_i, \hat S_i, \hat C_i$
- checklist heads
- uncertainty / confidence

### final policy score

최종 candidate score는 rank utility, risk, 그리고 area prior를 결합한다.

$$
\hat s_i^{policy} = \hat u_i - \lambda_r \hat \rho_i + \lambda_a(m,r)\log(a_i + \epsilon)
$$

여기서 $\lambda_a(m,r)$는 subject mode와 target AR에 따라 다르게 둘 수 있다.

## 6.7 모듈 5 — baseline-aware policy head

### 왜 ranking과 분리하는가

이 부분은 SSTK가 기존 crop 논문들과 가장 다른 지점이다. top-1 crop를 찾는 것과 “정말 자를지”는 같은 문제가 아니다.

### baseline 집합

v4.0은 baseline을 최소 두 개 둔다.

- $b_{full}$: 원본 전체
- $b_{min}$: target AR을 만족하는 max-area baseline

실제 서비스에서는 strict AR 여부에 따라 두 baseline 모두를 비교군으로 둘 수 있다.

### decision definition

$$
b^\star = \arg\max_i \hat s_i^{policy}
$$

$$
\hat \Delta = \hat s^{policy}(b^\star) - \hat s^{policy}(b_{base})
$$

정책은 ordinal하게 학습한다.

- `keep_full`
- `minimal_crop`
- `crop`

가능하면 threshold를 직접 박아 넣기보다, `delta` 자체와 class를 함께 예측하는 편이 낫다.

$$
L_{policy} = L_{ord}(\hat d, d^\star) + \lambda_\Delta \operatorname{Huber}(\hat \Delta, \Delta^\star)
$$

### early exit keep path

모바일 latency를 줄이기 위해, 아래 조건에서는 baseline을 조기 반환할 수 있다.

$$
\hat p_{keep} > \tau_{keep}^{hi} \;\land\; \hat \Delta < \tau_{gray}
$$

이 경우 refiner와 추가 ranking pass를 건너뛴다.

## 6.8 모듈 6 — optional local refiner

### 왜 optional인가

refiner는 geometry resolution을 보완하지만, 항상 켜 두면 latency variance가 커지고 proposal/ranker가 약한 문제를 가릴 수 있다. 따라서 v4.0에서 refiner는 **gray-zone only**가 원칙이다.

### parameterization

fixed-AR refiner의 기본 파라미터화는 다음 3자유도면 충분하다.

$$
\hat c_x = c_x + w\Delta x, \qquad
\hat c_y = c_y + h\Delta y, \qquad
\hat s = \exp(\Delta \log s)
$$

즉 AR은 유지하고 center/scale만 움직인다.

### trigger

- top-1과 top-2 margin이 작음
- policy delta가 threshold 근처
- face/text/head-top risk가 gray band에 있음
- uncertainty가 높음

### loss

$$
L_{ref} = \lambda_c L_{local-rank} + \lambda_b L_{box} + \lambda_g (1-\operatorname{GIoU})
$$

## 6.9 모듈 7 — optional `A_gen + A_crop + A_comp` teacher prior stack

이 모듈은 v4.0의 **모바일 runtime 필수 요소가 아니라** teacher/HQ training과 explanation distillation용 보조축이다.

- `A_gen`: general aesthetics / quality prior
- `A_crop`: crop-specific learned rank prior
- `A_comp`: composition classifier / embedding prior

### 권장 사용법

- mobile runtime에 그대로 넣지 않는다.
- teacher rescoring, HQ distillation, explanation/why-tag generation에 우선 쓴다.

### `A_gen` 추천

현재 기준으로는 `NIMA`보다 `TOPIQ-IAA`나 `Q-Align` 계열이 더 강한 선택지다. 다만 `Q-Align`은 비용이 크므로 mobile mainline이 아니라 teacher/HQ용이 적합하다. [I6][E29][E30][E31]

### `A_comp` 추천

- `KU-PCP` / `CACNet` 류의 9-class composition prior [E10]
- `PICD`의 24-class composition embedding [E20]

### 핵심 원칙

Aesthetic/Composition prior는 **final crop box를 직접 고르는 주 runtime**이 아니라, proposal/ranker 표현을 더 잘 정렬시키는 **training-time auxiliary**로 쓰는 것이 가장 안전하다.

# 7. 현재 라벨을 v4.0 학습에 어떻게 연결할 것인가

## 7.1 기본 철학

현재 라벨을 “teacher module output”으로 읽지 말고, “학습 과업별 supervision”으로 읽는다.

- proposal supervision
- ranking supervision
- policy supervision
- explanation supervision

## 7.2 positive set 정의

현재 `matching_targets`는 proposal 학습의 핵심 positive set이다. 여기에 admissible high-utility safe candidate를 soft positive로 추가해 dense positive를 만든다.

$$
\mathcal{P}^+ = \mathcal{M} \cup \left\{c \in \mathcal{C}_{pool} : \text{crop\_utility\_prob}(c) \ge \tau_{soft+},\; c\;\text{safe} \right\}
$$

여기서

- $\mathcal{M}$: `matching_targets`
- $\mathcal{C}_{pool}$: `candidate_pool`

## 7.3 negative set 정의

`candidate_pool` 전체를 그대로 negative로 쓰지 않고, 현재 builder semantics를 따라 나눈다.

- safe near-negative
- ordinary negative
- ignored safe leftover
- overflow unsafe/hard

v4.0은 `ignored_candidates`와 `overflow_candidates`를 main negative loss와 같은 weight로 쓰지 않는다. 이는 current monotonic split의 의도와 맞다. [I1][I3]

## 7.4 module별 매핑 표

| 현재 artifact / field | v4.0 모듈 | primary loss | 비고 |
|---|---|---|---|
| `matching_targets` | proposal | Hungarian + dense positive | positive recall 핵심 |
| `candidate_pool` | ranker | pairwise/listwise | shortlist 구조 학습 |
| `pairwise` | ranker | margin ranking / logistic ranking | hard boundary 강화 |
| `listwise` | ranker | softmax CE / ListNet | top region mass 유지 |
| `baseline` + `decision_target` | policy | ordinal CE + delta regression | no-crop/minimal-crop |
| `checklist` | auxiliary | BCE / ordinal / regression | explanation / debug |
| `macro` | auxiliary | Huber / distillation | A/S/C 구조 고정 |
| `ignored_candidates` | audit / hard-case mining | optional | main negative와 분리 |
| `overflow_candidates` | risk auxiliary | optional | unsafe/hard audit |

## 7.5 현재 row 수와 왜 충분한가

`v3.3` current alignment 기준으로, 현재 run은 대략 아래 규모를 이미 가진다. [I3]

| 산출물 | 규모 |
|---|---:|
| `train_conditional_detr_batch.jsonl` | 6,820 rows |
| `train_conditional_detr_canonical.jsonl` | 6,820 rows |
| `train_regression.jsonl` candidate rows | 155,051 |
| `train_checklist.jsonl` candidate rows | 155,051 |
| `train_pairwise.jsonl` pair rows | 44,282 |

즉 v4.0은 “라벨이 없어서 proposal/ranker/policy를 못 나눈다”가 아니라, **이미 있는 라벨을 그 목적에 맞게 쓸 설계가 없었던 것**에 더 가깝다.

# 8. 학습 방법론: 권장 커리큘럼과 loss 설계

## 8.1 학습은 staged가 정답이다

`proposal + ranking + policy + refiner`를 처음부터 한 번에 joint training하면, 문제 원인을 분리하기 어렵고 초기 불안정성이 커진다. 따라서 staged training이 안전하다.

![v4.0 권장 학습 커리큘럼](assets/fig04_training_curriculum.png){ width=96% }

## 8.2 Stage 0 — label contract audit

이 단계의 목적은 새 모델을 돌리기 전에 현재 라벨 계약이 깨지지 않았는지 확인하는 것이다.

필수 체크:

- `validation_summary.status == ok`
- `error_count == 0`
- `score_profile.profile_name == single_stage2`
- main monotonic violation `0`
- `crop_utility_prob` field 존재
- GAIC-like split image path 정합성

## 8.3 Stage 1 — ranker warm-up

proposal generator를 아직 켜지 않고, teacher candidate replay만으로 relation-aware ranker와 policy head를 먼저 안정화한다.

활성 loss:

- pairwise ranking
- listwise ranking
- decision loss
- optional macro/checklist auxiliary

### pairwise loss

$$
L_{pair} = \sum_{(i,j)\in\mathcal{P}} w_{ij}\log\left(1 + \exp\big(-y_{ij}(\hat u_i - \hat u_j - m_{ij})\big)\right)
$$

### listwise loss

$$
p_i^\star = \frac{\exp(u_i^\star/\tau_t)}{\sum_j \exp(u_j^\star/\tau_t)}, \qquad
\hat p_i = \frac{\exp(\hat u_i/\tau_p)}{\sum_j \exp(\hat u_j/\tau_p)}
$$

$$
L_{list} = -\sum_i p_i^\star \log \hat p_i
$$

### decision loss

$$
L_{dec} = L_{ord}(\hat d, d^\star) + \lambda_\Delta \operatorname{Huber}(\hat \Delta, \Delta^\star)
$$

### stage-1 총합

$$
L_1 = \lambda_p L_{pair} + \lambda_l L_{list} + \lambda_d L_{dec} + \lambda_a L_{aux}
$$

권장 시작값:

- $\lambda_p = 1.0$
- $\lambda_l = 1.0$
- $\lambda_d = 0.5$
- $\lambda_a = 0.25$

## 8.4 Stage 2 — proposal pretraining

이 단계부터 proposal generator를 켠다.

### 1:1 + dense positive hybrid

primary matching은 Hungarian이나 cost-based assignment를 유지하되, 추가 query에도 dense positive를 준다.

$$
L_2 = L_{Hungarian}^{1:1} + \lambda_{dense} L_{dense+} + \lambda_{obj} L_{obj} + \lambda_{div} L_{div}
$$

### diversity regularization

proposal collapse를 줄이기 위해 지나치게 비슷한 query 간 penalty를 준다.

$$
L_{div} = \sum_{i<j} \max\left(0, \operatorname{IoU}(\hat b_i, \hat b_j) - \eta_{dup}\right)
$$

## 8.5 Stage 3 — joint HQ training

proposal, box pooling, HQ ranker, policy head를 연결해 joint fine-tuning한다.

이때 batch를 두 종류로 섞는 것이 좋다.

- **teacher replay batch**: current candidate pool 그대로 사용
- **model proposal batch**: learned proposal top-K 사용

이렇게 해야 proposal이 아직 흔들릴 때 ranker까지 함께 무너지지 않는다.

## 8.6 Stage 4 — HQ to Mobile distillation

HQ model에서 Mobile model로 지식을 distill한다. 핵심 차이는 다음뿐이다.

- backbone 축소
- query 수 축소
- self-attention ranker를 `RelationLite`로 교체
- top-K 축소
- refiner 기본 OFF

### distillation loss

$$
L_{KD} = \tau^2 \operatorname{KL}\left(\sigma(z^{HQ}/\tau)\;\|\;\sigma(z^M/\tau)\right) + \lambda_f \sum_l \lVert \phi_l^M - \operatorname{sg}(\phi_l^{HQ}) \rVert_1
$$

## 8.7 Stage 5 — QAT + export hardening

INT8 양자화는 v4.0에서 optional이 아니라 사실상 필수다. 다만 QAT 없이 PTQ만으로 끝내면 policy threshold 근처 calibration drift가 커질 수 있다.

### 권장 QAT loss

$$
L_{QAT} = L_{task} + \lambda_q \sum_l \lVert z_l^{fp} - z_l^{int8} \rVert_2^2
$$

즉 태스크 loss 외에 fp/int8 activation drift를 작게 유지하는 보조항을 둔다.

# 9. v4.0에서의 구체적 tensor / module 계약

## 9.1 입력

| 텐서 | shape (B=1) | 설명 |
|---|---:|---|
| `image` | `[1, S, S, 3]` | padded RGB tensor, `S ∈ {256,288,320}` |
| `image_ar_log` | `[1,1]` | 원본 이미지 AR의 log 값 |
| `target_ar_id` | `[1]` | target AR bucket |
| `target_ar_log` | `[1,1]` | log AR |
| `optional_intent` | `[1,D_i]` | intent-conditioned service가 필요할 때만 |

## 9.2 proposal generator 출력

| 텐서 | shape | 설명 |
|---|---:|---|
| `prop.boxes` | `[1,Q,4]` | normalized xyxy |
| `prop.obj` | `[1,Q]` | proposal objectness |
| `prop.token0` | `[1,Q,D]` | initial proposal token |
| `prop.uncert` | `[1,Q]` | optional proposal uncertainty |

fixed AR inference/debug JSON에서는 proposal별로 다음 메타데이터를 추가로 기록한다.

| 필드 | 의미 |
|---|---|
| `target_ar_value` | `1:1`, `3:4` 등 target AR의 실수값. `FREE`는 `null` |
| `crop_ar` | 원본 이미지 좌표로 복원한 proposal의 실제 crop pixel AR |
| `letterbox_ar` | letterbox 좌표계 proposal AR |
| `target_ar_log_error` | `abs(log(crop_ar / target_ar_value))` |
| `target_ar_compatible` | fixed AR에서 log error가 허용 범위 안인지 여부 |
| `proposal_top1_raw` | objectness 기준 raw top-1 proposal |
| `proposal_top1_target_ar` | target AR compatibility를 우선한 표시/진단용 proposal |

평가 metric은 raw objectness top-1과 target-AR-selected top-1을 분리한다. `proposal_raw_top1_target_ar_log_error`와 `proposal_raw_top1_target_ar_compatible`는 raw proposal head가 곧바로 표시 가능한지를 보고, `proposal_top1_target_ar_log_error`와 `proposal_top1_target_ar_compatible`는 target-AR selection 후 사용자-facing orange box가 target AR을 만족하는지 본다.

## 9.3 box pooling 출력

| 텐서 | shape | 설명 |
|---|---:|---|
| `pool.token` | `[1,K,D_p]` | pooled candidate token |
| `pool.geom` | `[1,K,D_g]` | geometry / AR / baseline-relative feature |
| `pool.is_base` | `[1,K]` | baseline flag |

## 9.4 ranker 출력

| 텐서 | shape | 설명 |
|---|---:|---|
| `rank.utility` | `[1,K]` | rank utility |
| `rank.risk` | `[1,K]` | risk logit |
| `rank.ASC` | `[1,K,3]` | optional A/S/C auxiliary |
| `rank.check_bin` | `[1,K,B]` | binary checklist |
| `rank.check_ord` | `[1,K,O,C]` | ordinal checklist |
| `rank.conf` | `[1,K]` | confidence |

## 9.5 policy / refiner 출력

| 텐서 | shape | 설명 |
|---|---:|---|
| `policy.logits` | `[1,3]` | keep_full / minimal_crop / crop |
| `policy.delta_vs_base` | `[1,1]` | baseline 대비 개선량 |
| `refine.delta` | `[1,2,3]` | top-1/top-2 local delta |

## 9.6 explainable output 계약

최종 제품 출력은 crop box만 반환하면 안 된다. 정성 검증, 디버깅, 고객/PM 리뷰를 위해 모델은 crop 결과와 함께 사람이 읽을 수 있는 labelized explanation을 반환한다.

1차 구현에서 바로 반환 가능한 항목은 다음이다.

| 항목 | source tensor | 출력 예 |
|---|---|---|
| crop decision | `policy.logits` | `{"id": 2, "label": "crop", "score": 0.83}` |
| photo / subject mode | `route.logits` | `{"id": 6, "label": "scene_general", "score": 0.71}` |
| candidate utility | `rank.utility` | `{"score": 0.78, "label": "good"}` |
| candidate risk | `rank.risk` | `{"score": 0.12, "label": "safe"}` |
| positive likelihood | `rank.positive` | `{"score": 0.67, "label": "likely"}` |
| macro checklist | `rank.ASC/T` | `aesthetic`, `subject`, `composition`, `technical` 각각 `good/ok/bad` |

1차 label rule은 calibration 전까지 단순 threshold를 쓴다. score `>=0.72`는 `good`, `0.45~0.72`는 `ok`, `<0.45`는 `bad`로 표시한다. 이 label은 사람 검증용 display label이며, 최종 품질 판단은 official benchmark metric과 slice QA로 한다.

현재 구현에서는 `train_checklist.jsonl` 계열의 실제 candidate field를 이용해 세부 checklist head를 분리했다. 기존 `macro_head`만으로는 `composition: ok 0.63` 같은 거친 설명만 가능했다. SSTK data factory label에는 이미 `checklist_labels`, `checklist_scores`, `composition_focus`, `why_tags`, `reject_tags`, `safety_penalty`가 들어 있으므로, v4.0 student는 이들을 별도 target으로 받아 아래 세 종류의 detailed explanation head를 학습한다.

- `checklist_class_head`: 13개 checklist group의 multi-class label을 예측한다. 예: `headroom_loose`, `lookroom_excessive`, `rule_of_thirds_strong`, `phi_grid_weak`, `context_preserved`.
- `checklist_applicability_head`: subject mode와 candidate crop을 기준으로 각 checklist group이 적용 가능한지 예측한다. 예: `object_single`에서는 `headroom`, `lookroom`, `face_cut`, `joint_cut`을 known-not-applicable로 둔다.
- `detail_score_head`: 22개 normalized numeric score를 예측한다. 예: `third_strength`, `phi_strength`, `center_strength`, `headroom_ratio_norm`, `lookroom_ratio_norm`, `symmetry_score`, `safety_penalty_soft`.
- `why_tag_head`: 31개 why/reject tag를 multi-label로 예측한다. 예: `avoid_face_cut`, `avoid_person_cut`, `rule_of_thirds`, `phi_grid`, `headroom_violation`, `lookroom_violation`.

`na`는 설명 가능한 구도 상태가 아니라 teacher label 결측을 뜻한다. 따라서 class loss에서는 `checklist_labels.<key>`가 없거나 값이 `na`, `none`, `null`, 빈 문자열인 경우 해당 group의 valid mask를 0으로 둔다. vocab에는 checkpoint 호환성과 inference fallback을 위해 `na` class를 유지하지만, 새 학습에서는 `na`를 맞히는 방향으로 설명 head를 최적화하지 않는다. 추론/시각화에서는 `na` class 예측을 품질 라벨처럼 표시하지 않고 `unlabeled p=<probability>`로 표시하며, 실제 구도 판단은 `detail_scores`와 non-NA class label을 함께 본다.

또한 모든 checklist가 모든 subject mode에 적용되는 것은 아니다. `portrait_single/group`에서는 headroom/lookroom/face/person cut이 의미 있지만, `object_single` 또는 `object_multi`에서는 사람 전용 항목이 의미 없다. 새 데이터 adapter는 `EXPLANATION_APPLICABILITY_BY_MODE`를 기준으로 `checklist_applicability_target`, `checklist_applicability_valid`, `why_tag_applicable`를 생성한다. class/detail/why loss는 이 mask를 반영한다. 추론 decode는 mode-inapplicable 항목을 `available=false`, `display_label=not_applicable`로 표시하고 score를 숨긴다. mode상 가능하지만 모델이 applicability를 낮게 본 항목은 `display_label=unavailable`로 표시해 `not_applicable`과 구분한다. 따라서 사람이 없는 object crop에서 `headroom_loose 0.89` 같은 hallucination성 explanation이 사용자-facing 산출물에 나오지 않는다.

| 세부 head | label 예 | 주요 학습 신호 |
|---|---|---|
| rule-of-thirds / composition | `rule_of_thirds_strong`, `phi_grid_weak`, `center_comp_strong` | `checklist_labels.third_dist/phi_dist/center_dist`, `composition_focus` |
| headroom / lookroom | `headroom_loose`, `headroom_ok`, `lookroom_excessive` | `checklist_labels.headroom/lookroom`, normalized ratio score, reject tag |
| subject coverage / scale | `good`, `marginal`, `too_loose`, `ideal_scale` | subject prior, `checklist_labels.subject_coverage/subject_scale` |
| cut/context/copyspace safety | `no_face_cut`, `joint_cut_mild`, `context_preserved`, `copyspace_partial` | reject/safety tag, context/copyspace checklist |
| why / reject tags | `avoid_face_cut`, `rule_of_thirds`, `headroom_violation` | `why_tags`, `reject_tags`, destructive-negative bucket |

따라서 inference JSON의 권장 형태는 아래와 같다.

```json
{
  "decision": {"label": "crop", "score": 0.83},
  "subject_mode": {"label": "portrait_single", "score": 0.72},
  "predictions": [
    {
      "rank": 1,
      "bbox_norm_xyxy": [0.12, 0.08, 0.88, 0.92],
      "score": 0.78,
      "checklist": {
        "composition": {"label": "good", "score": 0.76},
        "subject": {"label": "good", "score": 0.84},
        "technical": {"label": "ok", "score": 0.62}
      },
      "detailed_checklist": {
        "third_dist": {"label": "rule_of_thirds_strong", "score": 0.82},
        "headroom": {"label": "headroom_loose", "display_label": "headroom_loose", "available": true, "score": 0.71},
        "lookroom": {"label": "lookroom_excessive", "display_label": "lookroom_excessive", "available": true, "score": 0.66}
      },
      "detail_scores": {
        "third_strength": 0.78,
        "phi_strength": 0.61,
        "symmetry_score": 0.58
      },
      "why_tags": [
        {"tag": "rule_of_thirds", "score": 0.74},
        {"tag": "avoid_face_cut", "score": 0.68}
      ],
      "explanation": {
        "risk": {"label": "safe", "score": 0.11},
        "positive": {"label": "likely", "score": 0.69}
      }
    }
  ]
}
```

이 구조를 쓰면 PNG contact sheet에는 crop box와 함께 `decision`, `subject_mode`, `composition/subject/technical` label뿐 아니라 세부 checklist, detail score, why tag, label-side teacher checklist를 함께 표시할 수 있다. JSONL 평가 산출물에서는 label별 failure slice, contradiction slice, teacher-label imitation 품질을 바로 집계할 수 있다.

## 9.7 Product AR/full label score replacement 계약

2026-04-20 이후 MobileCropNet의 제품 track은 `Product AR cropper + explanation`으로 고정한다. 따라서 학습 label도 `FREE`, `1:1`, `3:4`, `4:3`, `16:9`, `9:16` target AR row를 모두 포함하는 Product AR/full schema를 기본으로 사용한다. GAIC official MOS benchmark나 GAIC public cropper score track은 MobileCropNet student 자체를 GAIC scorer로 만드는 목표가 아니라, crop score teacher 또는 label curation을 진단하는 별도 track이다.

T6 deep ranker 또는 GAIC public cropper score를 Product AR/full label에 적용할 때는 crop score supervision만 교체한다. SSTK data factory의 subject mode, checklist label, applicability, fatal/safety/reject metadata는 계속 설명/안전 audit target으로 보존한다. 이때 external score가 높지만 SSTK가 hard reject 또는 fatal로 표시한 후보는 positive로 승격하지 않고 다음 정책을 적용한다.

| 상황 | 학습 label 처리 |
|---|---|
| external score 높음 + SSTK explanation 양호 | 같은 `(image_id, target_ar)` group의 positive 후보로 선택 가능 |
| external score 높음 + SSTK contradiction | raw external score는 보존하되 낮은 weight 또는 conflict bucket으로 audit |
| external score 높음 + SSTK hard reject/fatal | adjusted score를 `unsafe_score_cap`으로 cap하고 `external_high_sstk_hard_reject_demoted` bucket으로 demotion |
| external score 낮음 + SSTK negative | normal/hard negative로 유지 |

이 계약의 핵심은 external scorer의 ranking signal과 SSTK product safety signal을 혼동하지 않는 것이다. 모델의 utility/ranking head는 T6/public score의 순서를 학습할 수 있지만, 제품에서 명백히 위험한 crop을 top positive로 학습하지 않도록 converter가 `best_safe_external_with_sstk_fallback` 정책을 적용한다. 원본 external score와 demotion 사유는 `external_score_teacher` metadata에 남겨서 정성 분석, slice QA, label builder ablation에 사용한다.

구현은 `export_mobilecropnet_v4_product_ar_candidates.py`, `score_mobilecropnet_v4_product_ar_candidates_with_public_cropper.py`, `run_mobilecropnet_v4_product_ar_t6_label_generation.sh`, `build_mobilecropnet_v4_product_ar_score_labels.py`, `run_mobilecropnet_v4_product_ar_score_track_matrix.sh`로 분리했다. 생성 label은 `training_labels_public_score_product_ar_v1`, `training_labels_t6_score_product_ar_v1`이며, GAIC v2 official split 기준 Train 14,915 row / Val 1,136 row / Test 2,824 row의 Product AR/full schema를 유지한다. 2026-04-20 full-profile 실험에서는 `T6/turbo_256`이 top-return 계열, `public/rank_320`과 `public/balanced_288`이 ranking correlation 계열에서 가장 강했다. 세부 수치와 산출물 경로는 `MobileCropNet_v4_0_GAIC_Experiment_Report_KO_2026-04-16.md`의 7.12 절을 기준으로 관리한다.

2026-04-21부터는 이 Product AR/full label 위에 `deploy-align` 학습을 추가한다. 핵심은 label candidate bank에서만 utility를 맞추는 것이 아니라, **모델이 실제 배포 시 생성하는 proposal set에도 ranking/positive/risk 정렬 신호를 직접 주는 것**이다. 배포 selection policy는 `proposal_topk_rerank(top-8)`를 기본으로 두고, 관련 설계와 실험은 `MobileCropNet_v4_0_Product_AR_DeployAlign_Report_KO_2026-04-21.md`에서 별도로 관리한다. 최종 결과 기준으로는 `prod_rank_288`, `rank_320`보다 `turbo_256`이 배포 안정성이 훨씬 높았고, `deploy-align 0.30/0.15` ablation 모두 `arx2`는 넘었지만 기존 `T6/turbo_256` baseline 자체는 넘지 못했다. 따라서 제품 승격 모델은 여전히 `mcn-t6prodar-turbo-256-20260420-220603`이며, deploy-align lane은 배포 정합형 evaluator/visualization 계약과 loss ablation을 검증한 연구 결과로 유지한다. 또한 현재 deploy-align lane의 checkpoint 선택은 training loop 내부 exact direct evaluator가 아니라 `deploy_align_topreturn` proxy metric 기반이므로, direct Product AR primary metric과의 selection gap을 함께 해석해야 한다.

## 9.8 현재 구현을 이해하기 위한 핵심 요약

### 9.8.1 현재 모델은 무엇을 학습하는가

현재 MobileCropNet v4는 크게 두 모듈로 보면 이해가 쉽다.

| 모듈 | 역할 | 현재 구현 포인트 |
|---|---|---|
| Proposal generator | 이미지와 target AR을 보고 crop 후보 box 집합을 생성 | `proposal_queries`, `proposal_box_head`, `proposal_logit_head`로 구성된 learned proposal head. fixed AR row에서는 AR-constrained parameterization을 사용한다. |
| Ranker + policy/explanation | 생성된 후보 또는 label candidate bank를 비교해 최종 crop과 설명을 고름 | ROI pooled candidate token, relation refinement, optional set-transformer ranker, utility/positive/risk/policy/checklist/route head로 구성된다. |

즉 배포 시 모델은 `proposal 생성 -> 후보 scoring/ranking -> 최종 crop 선택 -> decision/subject mode/explanation 출력` 순서로 동작한다. 현재 제품 권장 selection policy는 `proposal_topk_rerank(top-8)`이다. 단순 `proposal objectness top-1`이나 `utility top-1`만으로 바로 내보내는 방식이 아니다.

### 9.8.2 학습 라벨은 어떻게 만들어지는가

학습의 출발점은 SSTK data factory가 만든 Product AR/full label이다. 이 label은 이미지마다 `FREE`, `1:1`, `3:4`, `4:3`, `16:9`, `9:16` row를 만들고, 각 row에 대해 `candidate_pool`, `matching_targets`, `pairwise/listwise`, `decision`, `route`, `checklist`, `why tag`를 함께 담는다.

라벨 생성 흐름은 아래처럼 이해하면 된다.

1. SSTK teacher pipeline이 이미지별 crop 후보 집합을 만든다.
2. 각 후보에 대해 subject mode, checklist, fatal/reject, safety, decision 같은 product semantics를 붙인다.
3. score supervision은 track에 따라 다음 셋 중 하나를 사용한다.
   - 기본 SSTK score: SSTK teacher의 `score_prob`, `crop_utility_prob`, `rank_pct`
   - T6 score track: 같은 후보 box 위에 T6 deep ranker score를 다시 계산해 score supervision만 교체
   - public score track: 같은 후보 box 위에 GAIC public cropper score를 다시 계산해 score supervision만 교체
4. external score가 높아도 SSTK가 `hard reject`, `fatal`, `unsafe`로 본 후보는 positive로 승격하지 않는다. converter가 `best_safe_external_with_sstk_fallback` 정책으로 safe 후보를 positive로 고른다.

중요한 점은 box 자체와 explanation ontology는 SSTK가 제공하고, score ordering만 T6/public으로 교체할 수 있다는 것이다. 즉 `T6/public score 기반 Product AR/full label`은 "다른 crop 후보를 새로 만든 것"이 아니라 "같은 후보 집합에 대해 utility supervision을 재정의한 것"이다.

### 9.8.3 Proposal generator는 무엇으로 학습되는가

proposal generator는 row 안의 positive crop 집합으로 학습된다. 여기서 positive box는 `matching_targets`와 safe high-score positive 후보를 합쳐 만든다.

핵심 손실은 다음과 같다.

| proposal 학습 신호 | 의미 |
|---|---|
| `proposal_obj_loss` | 어떤 generated proposal이 positive와 충분히 겹치는지 objectness로 학습 |
| `proposal_box_loss` | positive box에 더 가깝게 위치/크기를 맞추도록 학습 |
| `proposal_diversity_loss` | proposal collapse를 막고 후보 다양성을 유지 |
| `proposal_recall_0_5` | positive를 proposal set이 실제로 포함하는지 보는 핵심 진단 지표 |

`deploy-align` 실험에서는 여기에 generated proposal 자체를 다시 ranker target에 맞추는 `generated_proposal_alignment_loss`를 추가했다. 즉 배포 시 생성되는 proposal 집합에도 score/listwise/positive/risk 정렬 신호를 직접 주는 lane이다.

### 9.8.4 Ranker는 무엇으로 학습되는가

ranker는 candidate box 집합을 입력으로 받아 후보 간 상대 순서를 학습한다. 이때 입력 candidate는 학습 시에는 주로 Product AR/full label의 `candidate_pool`이고, 배포 시에는 모델이 스스로 생성한 proposal set이다.

핵심 손실은 다음과 같다.

| ranker 학습 신호 | 의미 |
|---|---|
| `score_loss` | 각 후보의 normalized score target을 직접 맞춤 |
| `listwise_loss` | 후보 전체 순서 분포를 맞춤 |
| `pairwise_loss` / `explicit_pairwise_loss` | 좋은 후보가 나쁜 후보보다 높게 오르도록 비교 학습 |
| `positive_loss` | top crop에 가까운 후보를 positive로 분리 |
| `risk_loss`, `top1_risk_loss` | hard negative / unsafe / overflow / destructive reject를 상위로 올리지 않도록 억제 |
| `top_return_loss`, `topk_coverage_loss` | top-1, top-k 실사용 성능을 직접 끌어올리는 보조 손실 |
| `teacher_distill_loss` | teacher soft distribution을 student ranking 분포에 주입하는 보조 경로 |

여기에 checklist/why/applicability/route/decision head가 붙어서 explanation과 policy를 같이 학습한다. 따라서 현재 ranker는 단순 score head가 아니라 `crop scorer + product decision + explanation decoder`에 가깝다.

### 9.8.5 모듈별 성능에 가장 크게 영향을 주는 요소

Proposal generator 쪽에서 가장 중요한 것은 아래 셋이다.

1. positive recall: 좋은 crop이 proposal set 안에 실제로 존재하는가
2. target AR compatibility: fixed AR row에서 생성 box가 정말 해당 종횡비를 만족하는가
3. positive bag 정의: 어떤 후보를 positive로 볼지, unsafe candidate를 어떻게 배제할지

Ranker 쪽에서 가장 중요한 것은 아래 넷이다.

1. score target 품질: SSTK/T6/public 중 어떤 ordering을 주는가
2. pairwise/listwise/top-return loss 구성: correlation을 밀지, top-return을 밀지
3. candidate coverage: 학습 시 candidate pool이 충분히 다양한가
4. selection surface 정합성: 학습 때 보던 candidate bank와 배포 때 보는 generated proposal set이 얼마나 비슷한가

시스템 전체로 보면 결국 두 가지가 가장 크다.

1. proposal set 안에 괜찮은 crop이 들어오는가
2. 들어온 후보들 중 ranker가 실제 배포 selection policy에서 올바른 것을 고르는가

즉 proposal recall이 낮으면 ranker가 아무리 좋아도 최종 crop이 나빠지고, 반대로 proposal이 좋아도 ranking/selection이 틀리면 사용자 결과는 망가진다.

### 9.8.6 모듈과 시스템은 어떻게 평가하는가

평가는 두 표면으로 나눠서 봐야 한다.

| 평가 표면 | 목적 | 주 평가 데이터 | 핵심 지표 |
|---|---|---|---|
| Direct Product AR evaluation | 실제 배포 crop 성능 평가 | GAIC v2 official split을 Product AR/full schema로 변환한 label | `final_positive_hit_iou_0_5`, `final_best_iou_to_positive`, `proposal_positive_recall_at_5_iou_0_5`, `route_acc`, `decision_acc` |
| Official GAIC MOS benchmark | scorer/ranker의 FREE-form ranking quality 평가 | GAIC official benchmark annotation / candidate set | `PCC`, `SRCC`, `Acc1/5`, `Acc1/10`, `Acc4/10`, `Accw4/10`, `top1 MOS` |

여기서 중요한 해석은 다음과 같다.

- Direct Product AR evaluation은 proposal generator와 ranker를 함께 본다.
- Official GAIC MOS benchmark는 주로 scorer/ranker 품질을 보는 표면이다. benchmark candidate set이 이미 정해져 있기 때문에 proposal generator 자체를 직접 재는 평가는 아니다.
- 따라서 GAIC MOS가 좋아도 Product AR 배포 crop이 항상 좋은 것은 아니고, 반대로 Product AR direct 성능이 좋아도 GAIC ranking correlation이 최고가 아닐 수 있다.

### 9.8.6.1 Public cropper와 Direct Product AR를 어떻게 비교할 수 있는가

이 질문은 평가 표면을 분리해서 봐야 한다.

1. `GAIC`, `CGS` 같은 public cropper는 **candidate scorer**이므로, Product AR/full label의 `candidate_pool` 위에 점수를 붙여 `(image_id, target_ar)` 그룹별 top-1 candidate를 고르게 만들 수 있다.
2. 따라서 `final_positive_hit_iou_0_5`, `final_best_iou_to_positive`, `final_risk_hit_iou_0_5`, `final_target_ar_compatible` 같은 **candidate-bank direct Product AR 지표**는 계산 가능하다.
3. 반면 public cropper는 MobileCropNet처럼 learned proposal generator, route head, decision head를 갖지 않으므로 `proposal_positive_recall_at_k`, `route_acc`, `decision_acc`까지 같은 의미로 직접 비교할 수는 없다.
4. `CACNet`은 단일 crop 회귀 모델이라 candidate별 native score가 없기 때문에, Product AR/full direct 비교에서도 projection 기반 근사 비교만 가능하다.

즉 public cropper와 MobileCropNet의 Direct Product AR 비교는 두 단계로 나뉜다.

| 비교 수준 | 가능 여부 | 설명 |
|---|---|---|
| candidate-bank direct 비교 | 가능 (`GAIC`, `CGS`) | 같은 Product AR 후보 집합 위에서 점수를 붙이고 top-1을 고르는 scorer baseline으로 비교 |
| deployed system full 비교 | 제한적 | MobileCropNet은 proposal generator + ranker + policy를 함께 평가하지만 public cropper는 proposal/policy 모듈이 없음 |
| projection-only 비교 | 제한적 (`CACNet`) | 단일 예측 crop을 가장 가까운 candidate에 투영하는 진단값만 가능 |

현재 코드베이스에는 두 가지가 이미 구현돼 있다. 첫째, `score_mobilecropnet_v4_product_ar_candidates_with_public_cropper.py`가 Product AR 후보 row를 `GAIC/CGS` public cropper로 scoring한다. 둘째, 이 점수를 이용한 `training_labels_public_score_product_ar_v1` label generation과 MobileCropNet 학습 실험이 이미 완료돼 있다. 다만 **public cropper 자체를 Product AR/full label 위에서 top-1 선택하게 한 뒤, MobileCropNet direct evaluator와 동일 형식의 전용 정량 report를 생성하는 실험은 아직 별도 구현/정리되지 않았다.**

따라서 현재 시점의 가장 타당한 해석은 다음과 같다. public cropper에 대해서는 이미 `GAIC official benchmark`, `unified public benchmark`, `Product AR score-label generation`은 완료돼 있지만, **Product AR direct evaluator 기준의 apples-to-apples candidate-bank baseline report는 추가 구현 항목**이다.

### 9.8.7 배포 시 사용자가 체감하는 성능을 가장 크게 좌우하는 것

사용자가 실제로 보는 것은 "최종 crop 1개"이기 때문에 아래 항목이 체감 성능을 가장 크게 좌우한다.

1. **selection policy**: 현재는 `proposal_topk_rerank(top-8)`가 가장 중요하다. 같은 checkpoint라도 `utility_top1`와 결과가 달라질 수 있다.
2. **proposal coverage**: 생성된 후보 안에 좋은 crop이 없으면 최종 crop은 반드시 나빠진다.
3. **route/decision coherence**: `keep_full`, `minimal_crop`, `crop` 판단과 subject mode가 어긋나면 crop과 explanation이 함께 이상해진다.
4. **unsafe/fatal filtering**: score가 높아도 product safety 관점에서 위험한 crop을 막아야 한다.
5. **training surface와 deployment surface의 정합성**: label candidate bank에서만 잘 맞고 generated proposal set에서는 무너지면, 오프라인 metric 대비 실제 사용자 결과가 나빠진다.

실무적으로는 `ranker score` 하나보다 `proposal recall + rerank policy + safety demotion`의 합이 사용자 경험을 결정한다. 그래서 현재 제품 track에서는 "GAIC scorer 최고점"보다 "Product AR direct 성능과 시각화 품질"을 우선해서 해석한다.

# 10. 온디바이스 배포 최적화 설계

## 10.1 원칙: 알고리즘은 하나, 프로파일은 셋

v4.0은 HQ, Balanced, Turbo 세 프로파일을 정의한다.

![배포 프로파일](assets/fig05_deployment_profiles.png){ width=96% }

### 권장 프로파일 표

| 프로파일 | 입력 | query 수 | ranker | refiner | 권장 백본 | 목표 |
|---|---:|---:|---|---|---|---|
| HQ-320 | 320 | 32 | 2-layer set transformer | ON | MobileNetV4-Hybrid-M | 최고 품질 / distillation teacher |
| Balanced-288 | 288 | 24 | RelationLite+ | gray-zone only | RepViT-M2 또는 MobileNetV4-Hybrid-M | 기본 제품형 |
| Turbo-256 | 256 | 16 | RelationLite | 기본 OFF | RepViT-M2 | 30ms급 latency 우선 |

### 구현 정렬 상태와 향후 기본 프로파일

2026-04-16 기준 현재까지 학습/평가된 GAIC MobileCropNet v4 checkpoint는 외부 pretrained backbone을 사용하지 않았다. 대표 Balanced run은 `input_size=288`, `proposal_q=24`, `candidate_k=24`, `token_dim=128`, `width_mult=0.75`, `backbone_name=custom_depthwise`, `backbone_pretrained=false`였고, 이후 HQ-320 검증 run도 `backbone_name=mobilenetv4_hybrid_medium`, `backbone_pretrained=false`로 수행됐다. 따라서 기존 checkpoint 성능은 pretrained backbone을 적용한 HQ-320/Balanced/Turbo 성능으로 해석하면 안 된다.

2026-04-17 업데이트: 설계상 권장 프로파일은 여전히 `HQ-320`이지만, GAIC v2 official MOS 기준 SSTK-only Product Track에서는 `HQ-320` 자체보다 `rank_320` corrected sweep과 `hybrid384` 확장이 더 좋은 결과를 냈다. 따라서 설계 검증/고품질 후보 실험은 `HQ-320`을 유지하되, 제품 후보 실험과 report의 primary candidate는 `rank_320 tr06-s19`와 `hybrid384`를 함께 본다. 모든 구현 preset은 `backbone_pretrained=true`를 기본값으로 하며, checkpoint 재로드 시에는 저장된 state dict를 사용하므로 외부 pretrained download를 반복하지 않는다. HQ-320 구현 preset은 다음과 같이 고정한다.

| 항목 | HQ-320 구현 preset |
|---|---|
| `model_profile` | `hq_320` |
| `input_size` | 320 |
| `proposal_q` | 32 |
| `candidate_k` | 32 |
| `token_dim` | 192 |
| `width_mult` | 1.0 |
| `backbone_name` | `mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k` |
| `backbone_pretrained` | `true` |
| `image_mean/std` | ImageNet mean/std, `0.485,0.456,0.406` / `0.229,0.224,0.225` |
| `ranker_type` | `set_transformer` |
| `ranker_depth` | 2 |
| `refiner` | strict HQ 요구사항이며, 현재 구현에서는 별도 local box refiner 학습 loss가 아직 미완료 |

Balanced/Turbo preset은 제품 latency와 품질 ablation을 위해 별도로 유지한다.

| 프로파일 | 구현 `backbone_name` | pretrained source | 정규화 | 선택 근거 |
|---|---|---|---|---|
| HQ-320 | `mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k` | ImageNet-12k pretrain + ImageNet-1k finetune | ImageNet mean/std | 설계상 HQ backbone family와 일치하고, timm 공개 MobileNetV4-Hybrid-M 중 320 test config에 가장 잘 맞는 고성능 weight다. |
| Balanced-288 | `mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k` | ImageNet-12k pretrain + ImageNet-1k finetune | ImageNet mean/std | RepViT-M2는 timm `repvit_m2.dist_in1k` weight가 있어 deployment ablation 후보로 유지하지만, 현재 품질 preset은 더 강한 in12k+in1k MobileNetV4-Conv-M weight를 사용한다. |
| Turbo-256 | `mobilenetv4_conv_small.e3600_r256_in1k` | ImageNet-1k pretrained | `0.5,0.5,0.5` / `0.5,0.5,0.5` | 256 입력 profile에 맞는 MobileNetV4 small 계열 중 가장 강한 공개 timm weight이며, profile의 latency 목적을 유지한다. |

Product Track에서 추가로 쓰는 실험 프로파일은 다음과 같이 해석한다.

| 프로파일 | 구현 요약 | 현재 판단 |
|---|---|---|
| `rank_320` | 320 입력, MobileNetV4 Conv-M pretrained, 32 candidate/proposal, set-transformer ranker depth 2 | SSTK-only 균형형 product candidate. 2026-04-17 corrected sweep 기준 `tr06-s19`가 Acc4/10, Accw4/10, top-return 균형이 가장 좋다. |
| `hybrid384` | 384 입력, MobileNetV4 Hybrid-M pretrained, 48 candidate/proposal, token 256, set-transformer ranker depth 3 | GAIC v2 official MOS top-return best. 단 비용과 Acc4/10/Accw4/10을 함께 봐야 한다. |

학습 CLI는 명시적 model-size override가 없으면 `hq_320` preset을 적용한다. 단, product sweep runner는 `rank_320`/`hybrid384` 같은 명시 프로파일을 넘겨 실행한다. regression smoke test나 latency ablation처럼 `--input_size`, `--proposal_q`, `--backbone_name` 등 모델 크기 인자를 직접 넘긴 경우에는 그 run을 custom profile로 기록한다. 논문/제품 성능 표에서는 `HQ-320 selected`, `rank_320 product`, `hybrid384 heavy`, `Balanced/Turbo ablation`을 분리한다. 엄밀한 HQ-320 완료 판정은 MobileNetV4-Hybrid-M, 320 input, 32 query, 2-layer set transformer, local refiner loss/출력까지 모두 켜진 checkpoint에만 부여한다.

## 10.2 latency model

지연은 대략 다음처럼 분해할 수 있다.

$$
T_{total} = T_{enc} + T_{prop} + T_{pool} + T_{rank} + T_{policy} + \mathbf{1}_{ref} T_{ref}
$$

v4.0이 줄여야 하는 것은 absolute latency뿐 아니라 **variance**다. 따라서 refiner는 항상 돌리지 않는다.

## 10.3 operator whitelist

배포 경로에서 가급적 유지해야 할 연산자는 아래다.

- `Conv2D`, `DepthwiseConv2D`, `PointwiseConv`
- `Linear/MatMul`
- `Add`, `Mul`, `Concat`
- `ReduceMean`, `ReduceSum`
- `Reshape`, `Transpose`
- `Softmax` (지원 시)

가능하면 피할 연산은 아래다.

- `ROIAlign`
- `grid_sample`
- dynamic `TopK`와 graph 내부 control flow
- `NonZero`, `Where` 기반 복잡한 분기
- dynamic shape dependent custom ops

## 10.4 split runtime을 어떻게 재정의할 것인가

v4.0은 split runtime을 **허용**하지만 **필수**로 두지 않는다.

### preferred path

- encoder + proposal + box pooling + ranker + policy를 one graph 또는 두 graph 안에 묶는다.

### conservative path

- graph 1: encoder + proposal
- host: fixed-grid mask rasterization / top-K gather
- graph 2: ranker + policy

즉 split은 export stability fallback일 뿐이다. 핵심 차이는 크다. `v3.3`에서는 split이 설계 중심이었지만, `v4.0`에서는 **same algorithm / different packaging**이다.

## 10.5 INT8, mixed precision, early exit

### INT8 우선 대상

- backbone stem / stages
- proposal MLP
- ranker MLP
- policy head

### FP16 유지 후보

- softmax / layernorm 민감 구간
- refiner 마지막 회귀 head
- threshold 근처 policy calibration 구간

### early exit 조건

- keep probability high
- delta to baseline gray zone 이하
- proposal confidence가 낮은데 baseline confidence는 충분히 높음

이때 baseline을 바로 반환하면 평균 latency를 크게 줄일 수 있다.

## 10.6 LiteRT / QNN export 경로

- LiteRT는 현재 NPU acceleration과 zero-copy hardware buffer를 지원하는 unified interface를 제공한다. [E27]
- ONNX Runtime QNN EP는 Snapdragon/Qualcomm 경로에서 실제 배포성이 좋다. [E28]
- 따라서 shipping stack은 `PyTorch -> ONNX/QDQ -> LiteRT or ORT+QNN` 두 갈래로 준비하는 것이 안전하다.

# 11. 평가 방법론: 품질, 정책, 설명, 시스템을 따로 본다

## 11.1 proposal 평가

proposal은 top-1 accuracy보다 **positive recall**이 중요하다.

$$
\operatorname{PosRecall@K}_\eta = \mathbf{1}\left[\exists \hat b \in \operatorname{TopK}(\mathcal{B}_{prop}), \exists b \in \mathcal{P}^+ : \operatorname{IoU}(\hat b, b) \ge \eta \right]
$$

함께 봐야 할 것:

- best-positive coverage@IoU
- baseline-preserve recall
- mode/AR별 proposal recall
- raw fixed AR proposal top-1 `proposal_raw_top1_target_ar_log_error`
- target-AR-selected proposal `proposal_top1_target_ar_compatible` rate
- raw proposal top-1과 target-AR filtered proposal top-1의 분리 지표
- unsafe proposal rate

## 11.2 ranking 평가

ranking은 다음 지표를 함께 본다.

- pairwise accuracy
- Top-1 hit
- NDCG@K
- shortlist overlap
- best-vs-baseline ordering accuracy

## 11.3 policy 평가

policy는 단순 3-way accuracy만 보면 안 된다.

- `keep_full / minimal_crop / crop` accuracy
- delta calibration error
- over-crop rate
- under-crop rate
- baseline regret

## 11.4 end-to-end crop 평가

### 내부 SSTK / GAIC

현재 파이프라인이 이미 쓰고 있는 `Gc/Ge` 프로토콜은 계속 유지한다. [I1][I2]

- `Gc`: GT crop 집합 안에서 scorer/policy가 MOS 순서를 얼마나 복원하는가
- `Ge`: GT crop을 production candidate pool에 주입했을 때 winner drift가 얼마나 생기는가

### 공개 benchmark CI

- FCDB
- CPC
- GNMC

이 셋은 최소 CI regression suite로 항상 돌리는 것이 맞다. [I6]

### slice 평가

SSTK는 generic average만 보면 안 된다. 최소 아래 slice를 따로 본다.

- portrait single / portrait group
- object single / object multi
- scene general
- background_texture_copyspace
- text/document heavy
- FREE vs AR-conditioned
- keep_full rare cases

## 11.5 explanation 평가

v4.0은 explanation head를 “있으면 좋은 부가 기능”이 아니라, quality와 함께 측정해야 한다.

권장 지표:

- checklist agreement
- why-tag consistency
- explanation fidelity (선택 이유가 실제 metric과 일치하는가)
- contradiction rate
- mode consistency violation rate
- not-applicable false positive rate
- applicability precision/recall
- label agreement when applicable
- person-tag-on-object rate

## 11.6 시스템 평가

- p50 / p90 latency
- peak memory
- NPU coverage
- CPU fallback 0 여부
- warm/cold start 차이
- energy per image
- export reproducibility

## 11.7 최종 evaluation matrix

| 축 | 핵심 지표 | release gate 예시 |
|---|---|---|
| proposal | PosRecall@K, unsafe proposal rate | proposal recall regression 없음 |
| ranking | pairwise acc, NDCG@K, top-1 | baseline lane 대비 개선 |
| policy | decision acc, over-crop rate | keep/minimal drift 허용범위 내 |
| end-to-end | Gc/Ge, IoU to GT MOS best, GT percentile | adopted lane 이상 |
| explanation | checklist agreement, contradiction rate | contradiction 낮음 |
| deployment | p50/p90 latency, NPU coverage, CPU fallback 0 | shipping pass |

# 12. v3.3 대비 무엇이 어떻게 달라지는가

## 12.1 요약 표

| 항목 | v3.3 | v4.0 |
|---|---|---|
| core candidate source | static micro-bank | learned proposal generator |
| runtime packaging | split runtime이 설계 중심 | split runtime은 deployment option |
| final scorer | ScoreA/ScoreB static bank ranker | relation-aware learned set ranker |
| policy | baseline-relative | baseline-relative 유지 |
| refiner | optional | optional 유지, 그러나 top-1/top-2 local-only |
| teacher alignment | 핵심 판단 기준 중 하나 | 배제 |
| pair/list/decision 활용 | ScoreB 중심 | ranker/policy 전체 중심 |
| backbones | RepViT/MobileNetV4 언급 | RepViT-M2 기본, MobileNetV4-Hybrid-M 대안 |
| explainability | auxiliary | auxiliary지만 release QA에 포함 |

## 12.2 유지되는 것

- candidate-first semantics
- baseline-relative decision
- no-crop / minimal-crop의 중요성
- checklist / why-tags / macro의 가치
- current label contract와 monotonic split 철학

## 12.3 달라지는 것

- teacher routing/bank topology를 runtime에서 복제하지 않는다.
- proposal을 learned module로 바꾼다.
- ranking을 relation-aware set reasoning으로 바꾼다.
- split runtime을 설계 원리에서 deployment packaging으로 낮춘다.

# 13. 리스크와 대응책

## 13.1 learned proposal이 불안정할 수 있다

**리스크**: proposal collapse, positive recall 부족

**대응**:

- Stage 1 ranker warm-up 후 proposal을 붙인다.
- Hungarian + dense positive hybrid를 사용한다.
- baseline 후보는 learned proposal과 무관하게 별도 합친다.
- PosRecall@K를 first-class KPI로 둔다.

## 13.2 relation-aware ranker가 모바일에서 비쌀 수 있다

**리스크**: self-attention cost와 export 불안정

**대응**:

- HQ와 Mobile ranker를 분리한다.
- Mobile은 `RelationLite`로 대체한다.
- token 수를 8~12 수준으로 제한한다.
- box pooling을 operator-safe하게 구현한다.

## 13.3 policy가 rare `keep_full`에 과적응/과소적응할 수 있다

**리스크**: keep_full rare class imbalance

**대응**:

- baseline set을 명시적으로 유지한다.
- class weight / focal CE를 적용한다.
- delta regression을 병행한다.
- `FREE`와 AR-compatible slice에서 keep_full을 따로 보고한다.

## 13.4 checklist가 noisy할 수 있다

**리스크**: aux head가 main objective를 오염

**대응**:

- aux weight를 낮게 둔다.
- applicability mask를 엄격히 쓴다.
- release 기준은 ranking/policy를 우선한다.

## 13.5 quantization drift가 policy threshold 근처에서 커질 수 있다

**리스크**: INT8에서 keep/minimal/crop 경계 흔들림

**대응**:

- QAT를 넣는다.
- threshold gray-zone과 early exit을 재보정한다.
- fp16 keep path를 남긴다.

# 14. 구현 A-to-Z 체크리스트

## 14.1 데이터/어댑터

1. current batch/canonical artifact schema 고정
2. adapter dataloader 작성
3. matching_targets / candidate_pool / decision_target / checklist decode 검증
4. image path, padding meta, AR bucket 변환 검증

## 14.2 모델 구현

1. shared encoder 구현
2. proposal generator 구현
3. box raster / integral pooling 구현
4. HQ ranker 구현
5. RelationLite mobile ranker 구현
6. policy head 구현
7. optional refiner 구현
8. optional A/S/C + checklist auxiliary 구현

## 14.3 학습

1. ranker warm-up
2. proposal pretrain
3. joint HQ training
4. distillation to mobile
5. QAT

## 14.4 배포

1. fixed input profiles 생성
2. ONNX export
3. QDQ / INT8 calibration
4. LiteRT path 검증
5. ORT + QNN path 검증
6. CPU fallback audit
7. latency / memory / energy 보고서 자동화

## 14.5 평가

1. internal SSTK slices
2. GAIC `Gc/Ge`
3. FCDB/CPC/GNMC CI
4. explanation fidelity report
5. failure gallery

# 15. 30 / 60 / 90일 권장 로드맵

## Day 0–30

- current label adapter 완성
- relation-aware ranker + policy warm-up
- baseline lane와의 first comparison
- proposal-free replay model로 빠른 smoke test

## Day 31–60

- learned proposal generator 연결
- joint HQ training
- PosRecall@K / Gc/Ge / decision calibration 첫 측정
- A/S/C + checklist auxiliary 켜기

## Day 61–90

- mobile distillation
- QAT + export hardening
- Balanced/Turbo profile 측정
- public benchmark CI + internal slice report 자동화

# 16. 최종 권고

본 문서의 최종 권고는 아래 한 문장으로 압축된다.

> **Teacher의 구조를 닮게 만들지 말고, 현재 라벨이 이미 정의한 네 과업 — proposal, ranking, policy, explanation — 을 각각 가장 잘 푸는 learned 하이브리드로 다시 짜라.**

이를 실무적 언어로 다시 쓰면 다음과 같다.

1. **연구 메인라인은 `learned proposal + relation-aware ranker + policy + refiner`다.**
2. **`static micro-bank`는 폐기 대상이 아니라 distillation/fallback/deployment baseline으로 재배치한다.**
3. **모바일 기본 백본은 `RepViT-M2`, 고품질 대안은 `MobileNetV4-Hybrid-M`으로 둔다.**
4. **배포 최적화는 알고리즘을 바꾸지 말고 packaging, quantization, early exit, operator-safe pooling으로 해결한다.**
5. **품질 평가는 proposal/ranking/policy/system을 분리해서 보고, `Gc/Ge + public CI + slice QA`를 함께 운영한다.**

이 구성을 따르면 v3.3이 가졌던 장점인 candidate-first와 baseline-relative semantics는 유지하면서도, teacher topology mirror링에 묶여 있던 후보 공간과 품질 상한을 함께 풀 수 있다.

# Appendix A. 추천 기본 하이퍼파라미터

| 항목 | HQ | Balanced | Turbo |
|---|---:|---:|---:|
| input long-side | 320 | 288 | 256 |
| query 수 | 32 | 24 | 16 |
| top-K to ranker | 16 | 12 | 8 |
| token dim | 192 | 160 | 128 |
| ranker depth | 2 | 1~2 | 1 |
| refiner | on | gray-zone only | off |
| batch size (train) | GPU 메모리 기준 | GPU 메모리 기준 | GPU 메모리 기준 |

# Appendix B. 간단한 inference pseudocode

```python
def infer_mobilecropnet_v40(image, target_ar, profile):
    x, meta = preprocess_keep_ar(image, profile.input_size)
    F = backbone(x)
    proposals = proposal_head(F, target_ar, meta)
    candidates = attach_baselines(proposals, image_ar=meta.image_ar, target_ar=target_ar)
    pooled = box_pool(F, candidates, meta, profile)
    ranked = ranker(pooled, candidates, target_ar)
    decision = policy_head(ranked, candidates, target_ar, meta)

    if should_early_exit_keep(decision, ranked, profile):
        return pick_baseline(candidates, decision)

    if should_refine(decision, ranked, profile):
        refined = local_refiner(F, top2_candidates(ranked))
        ranked = rerank_with_refined(ranked, refined)

    return assemble_output(ranked, decision)
```

# References

## Internal references

- [I1] `SSTK_Current_Implementation_Master_KO_2026-04-08-v2.md`
- [I2] `SSTK_Cropping_DataFactory_Master_KO_v3_1.md`
- [I3] `SSTK-MobileCropNet_v3_3.md`
- [I4] `Conditional DETR 모델 분석.txt`
- [I5] `문서 통합 및 업데이트.txt`
- [I6] `AI 기반 이미지 크로핑.txt`

## External references

- [E1] Yi-Ling Chen et al., *Learning to Compose with Professional Photographs on the Web*, 2017. <https://homepage.ntu.edu.tw/~lgchen/publication/paper/%5BC%5D%5B2017%5D%5BACM%5D%5BJan.Klopp%5D%5B1%5D.pdf>
- [E2] Wenguan Wang and Jianbing Shen, *Deep Cropping via Attention Box Prediction and Aesthetics Assessment*, ICCV 2017. <https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_Deep_Cropping_via_ICCV_2017_paper.pdf>
- [E3] Debang Li et al., *A2-RL: Aesthetics Aware Reinforcement Learning for Image Cropping*, CVPR 2018. <https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_A2-RL_Aesthetics_Aware_CVPR_2018_paper.pdf>
- [E4] Zijun Wei et al., *Good View Hunting: Learning Photo Composition From Dense View Pairs*, CVPR 2018. <https://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Good_View_Hunting_CVPR_2018_paper.pdf>
- [E5] Hui Zeng et al., *Reliable and Efficient Image Cropping: A Grid Anchor Based Approach*, CVPR 2019. <https://openaccess.thecvf.com/content_CVPR_2019/papers/Zeng_Reliable_and_Efficient_Image_Cropping_A_Grid_Anchor_Based_Approach_CVPR_2019_paper.pdf>
- [E6] Weirui Lu et al., *Listwise View Ranking for Image Cropping*, 2019. <https://arxiv.org/abs/1905.05352>
- [E7] Yi Tu et al., *Image Cropping with Composition and Saliency Aware Aesthetic Score Map*, AAAI 2020. <https://cdn.aaai.org/ojs/6889/6889-13-10118-1-10-20200525.pdf>
- [E8] Debang Li et al., *Composing Good Shots by Exploiting Mutual Relations*, CVPR 2020. <https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Composing_Good_Shots_by_Exploiting_Mutual_Relations_CVPR_2020_paper.pdf>
- [E9] Debang Li et al., *Learning to Learn Cropping Models for Different Aspect Ratio Requirements*, CVPR 2020. <https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Learning_to_Learn_Cropping_Models_for_Different_Aspect_Ratio_Requirements_CVPR_2020_paper.pdf>
- [E10] Chaoyi Hong et al., *Composing Photos Like a Photographer*, CVPR 2021. <https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_Composing_Photos_Like_a_Photographer_CVPR_2021_paper.pdf>
- [E11] Zhiyuan Pan et al., *TransView: Inside, Outside, and Across the Cropping View Boundaries*, ICCV 2021. <https://openaccess.thecvf.com/content/ICCV2021/papers/Pan_TransView_Inside_Outside_and_Across_the_Cropping_View_Boundaries_ICCV_2021_paper.pdf>
- [E12] Yang Cheng et al., *Re-Compose the Image by Evaluating the Crop on More Than Just a Score*, WACV 2022. <https://openaccess.thecvf.com/content/WACV2022/papers/Cheng_Re-Compose_the_Image_by_Evaluating_the_Crop_on_More_Than_WACV_2022_paper.pdf>
- [E13] Bo Zhang et al., *Human-centric Image Cropping with Partition-aware and Content-preserving Features*, ECCV 2022. <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670176.pdf>
- [E14] Gengyun Jia et al., *Rethinking Image Cropping: Exploring Diverse Compositions From Global Views*, CVPR 2022. <https://openaccess.thecvf.com/content/CVPR2022/papers/Jia_Rethinking_Image_Cropping_Exploring_Diverse_Compositions_From_Global_Views_CVPR_2022_paper.pdf>
- [E15] Chao Wang et al., *Image Cropping With Spatial-aware Feature and Rank Consistency*, CVPR 2023. <https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Image_Cropping_With_Spatial-Aware_Feature_and_Rank_Consistency_CVPR_2023_paper.pdf>
- [E16] Zhihang Zhong et al., *ClipCrop: Conditioned Cropping Driven by Vision-Language Model*, ICCVW 2023. <https://openaccess.thecvf.com/content/ICCV2023W/MMFM/papers/Zhong_ClipCrop_Conditioned_Cropping_Driven_by_Vision-Language_Model_ICCVW_2023_paper.pdf>
- [E17] James Hong et al., *Learning Subject-Aware Cropping by Outpainting Professional Photos*, AAAI 2024. <https://jhong93.github.io/pdf/crop-aaai24.pdf>
- [E18] Seung Hyun Lee et al., *Cropper: Vision-Language Model for Image Cropping through In-Context Learning*, CVPR 2025. <https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Cropper_Vision-Language_Model_for_Image_Cropping_through_In-Context_Learning_CVPR_2025_paper.pdf>
- [E19] Ke Zhang et al., *ProCrop: Learning Aesthetic Image Cropping from Professional Compositions*, 2025. <https://arxiv.org/abs/2505.22490>
- [E20] Zhaoran Zhao et al., *Can Machines Understand Composition? Dataset and Benchmark for Photographic Image Composition Embedding and Understanding*, CVPR 2025. <https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Can_Machines_Understand_Composition_Dataset_and_Benchmark_for_Photographic_Image_CVPR_2025_paper.pdf>
- [E21] Yen-Hong Wong and Lai-Kuan Wong, *AesCrop: Aesthetic-driven Cropping Guided by Composition*, ICCV 2025 Workshop. <https://openaccess.thecvf.com/content/ICCV2025W/MIPI/papers/Wong_AesCrop_Aesthetic-driven_Cropping_Guided_by_Composition_ICCVW_2025_paper.pdf>
- [E22] Depu Meng et al., *Conditional DETR for Fast Training Convergence*, ICCV 2021. <https://arxiv.org/abs/2108.06152>
- [E23] Yian Zhao et al., *RT-DETR: Real-Time DEtection TRansformer*, 2024 version of 2023 paper. <https://arxiv.org/abs/2304.08069>
- [E24] Shuo Wang et al., *RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision*, 2024. <https://arxiv.org/abs/2409.08475>
- [E25] Ao Wang et al., *RepViT: Revisiting Mobile CNN From ViT Perspective*, CVPR 2024. <https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.pdf>
- [E26] Danfeng Qin et al., *MobileNetV4: Universal Models for the Mobile Ecosystem*, ECCV 2024. <https://arxiv.org/abs/2404.10518>
- [E27] Google AI Edge, *LiteRT overview* / *NPU acceleration with LiteRT*. <https://ai.google.dev/edge/litert/overview>, <https://ai.google.dev/edge/litert/next/npu>
- [E28] ONNX Runtime, *QNN Execution Provider*. <https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html>
- [E29] IQA-PyTorch / pyiqa model cards and benchmark. <https://github.com/chaofengc/IQA-PyTorch>, <https://iqa-pytorch.readthedocs.io/en/latest/benchmark.html>
- [E30] Haoning Wu et al., *Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels*, 2023. <https://q-future.github.io/Q-Align/fig/Q_Align_v0_1_preview.pdf>
- [E31] Chaofeng Chen et al., *TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment*, 2023. <https://arxiv.org/pdf/2308.03060>
