# Shutterstock 400M 기반 Cropping 데이터 팩토리 & Qwen2.5-VL Teacher 라벨 생성기 — 재구성본 (KO v1.9)

본 문서는 원본 **통합 설계서 v1.9**의 내용을 논문 흐름에 가깝게 (파이프라인 개요 → Phase별 상세 → 데이터 스키마/패키징 → 평가/지표) 재구성한 버전입니다. 원본 내용은 누락 없이 포함하며, 원본 섹션 번호는 하위 헤딩에 유지했습니다.


## 0) Changelog / Scope / Principles

> **Changelog v1.9 (Teacher AB 하네스 구체화 + Proposal 주입 PoC + Free-form 후보 전략)**  
> - (v1.9) Teacher AB 하네스를 “문서 수준”이 아니라 **바로 구현 가능한 컴포넌트**로 구체화:  
>   - **입력/출력 JSON 스키마(고정)** + 실험 설정(YAML) + **Python/Ray 코드 스켈레톤**(ProposalTeacher / VLMTeacher / Aggregator / Evaluator 인터페이스)까지 내림  
> - (v1.9) “공개 Cropping teacher 레포” 중 **사전학습 가중치 + 데모 추론 코드가 명확한 3종**을 1차 PoC 대상으로 선정:  
>   - **GAIC‑Pytorch(Grid‑Anchor)** / **CACNet‑Pytorch(Photographer)** / **CGS‑Pytorch(Relation Mining)**  
> - (v1.9) Candidate Generator 단계에 **Teacher proposal 주입을 명시적 알고리즘으로 추가**:  
>   - (a) teacher free‑form/AR crop 제안 → (b) **AR projection** → (c) **local jitter neighborhood**(shift/scale) → (d) NMS/dedupe  
> - (v1.9) “proposal 주입만”으로 품질 개선폭을 측정하는 **Mini PoC 플랜**(변수 고정, 후보 set만 변화)과 **측정 지표(Recall@GT, Win‑rate, Violation Δ)**를 추가  
> - (v1.9) Free‑form 후보 생성 필요성 재고찰: 기본은 “grid+teacher seed+local search”로 충분하되, **어떤 조건에서 별도 free‑form generator가 필요해지는지** 의사결정 규칙을 명시  



> **Changelog v1.8 (Teacher Model Zoo + Cropping-model Teachers)**  
> - (v1.8) `Qwen2.5‑VL` 단일 의존을 탈피: Teacher 라벨 생성기를 **플러그인형 VLM/MLLM**으로 재정의하고, Qwen2.5‑VL(기본) + InternVL2.5/Molmo/MiniCPM‑V/Idefics3/LLaVA‑OneVision/DeepSeek‑VL2 등 **오픈 모델 후보군**을 “품질 상한/운영 비용/JSON 안정성” 관점에서 비교·선정할 수 있도록 **AB 하네스(평가 루프)**를 추가  
> - (v1.8) 공개 크롭 연구 레포의 **사전학습 Cropping 모델(예: GAIC Grid‑Anchor, CACNet, S2CNet, A2‑RL, CGS, UNIC 등)**을 Teacher로 편입:  
>   - (a) Candidate Generator에 **Teacher proposal 후보**를 주입(“Grid/룰”의 blind spot 보완)  
>   - (b) Cheap/Expensive Score에 `R_teach`(teacher consensus) **선택적 항**을 추가(가중치 소)  
>   - (c) Verify/QA에 `teacher_disagreement_rate`/`teacher_override_rate` 게이트를 추가(회귀 감시 + hard-case mining)  
> - (v1.8) 데이터 스키마에 `teacher_candidates[]`, `teacher_outputs[]`, `teacher_consensus_conf`, `teacher_disagreement_entropy` 등을 추가해 **디버깅·릴리즈 게이팅·Silver 재판정**을 강화  
> - (v1.8) VLM Teacher는 “좌표 생성”이 아니라 “후보 선택/체크리스트 검증/설명(why) 생성”에 집중한다는 원칙은 유지하되, Teacher ensemble(2~3개 모델)로 **라벨 신뢰도(confidence)**를 정량화(=자동 릴리즈 게이트의 핵심 시그널)  


> **Changelog v1.7 (PICD 통합)**
> - (v1.7) **PICD(Photographic Image Composition Dataset) 기반 “구도 임베딩/분류(24-class)” 모듈**을 데이터 팩토리에 정식 편입: Stage-2 Feature에 `composition_emb / comp24_logits`를 추가하고, Cheap/Expensive Score에 `R_picd` 항을 추가
> - (v1.7) PICD의 **24개 구도 taxonomy**(3분할/중앙/대각/수평/수직/삼각/곡선/원형/방사/원근/패턴/밀집/흩뿌림 등)을 `composition_tags`로 표준화하여 **설명(why) 라벨/QA/향후 Intent-crop UX**까지 확장
> - (v1.7) PICD가 제안한 **CDA(Composition Discrimination Accuracy)**를 “구도 임베딩 품질”의 **경량 프록시 지표**로 도입(=retrieval mAP과 상관) → 릴리즈 게이트에 `CDA@PICD` / `CDA@semantic_interference`를 추가
> - (v1.7) Candidate Generator에 **composition-guided anchors/jitter**를 추가(예: horizon-third, diagonal axis, triangle, curve, perspective/vanishing, pattern/dense/scatter)하여 “3분할/중앙만 살아남는” 단조로움을 방지
> - (v1.7) Qwen2.5-VL은 PICD 결과를 반영하여 “구도 분류기”가 아니라 **(a) 후보 선택 보조, (b) 체크리스트 기반 검증, (c) 설명 생성**에 집중하도록 프롬프트/스키마를 보강(구도 판단은 deterministic feature + comp-encoder가 1차)

> **Changelog v1.6**
> - (v1.6) **Portrait 전용 Headroom/Lookroom**: Shutterstock tag 기반 *샷타입(headshot/half/full)* 자동 추정 규칙 + 파라미터(범위/가중치) 산출 규칙을 추가하고, **Feature→Score→QA**까지 닫히도록 설계를 보강
> - (v1.6) **Horizon(Leveling) / Gaze(시선)** 모듈을 Stage-2 Feature Extraction에 정식 편입: SOTA 후보군 비교 + 10k 이미지 대량 처리 비용표를 업데이트
> - (v1.6) Cheap Scorer에 `R_headroom`, `R_lookroom`, `R_horizon`, `R_symmetry/center_bias`, `R_context`를 명시적으로 추가(카테고리/intent에 따른 on/off & 가중치 포함)
> - (v1.6) Candidate Generator에 **saliency-guided jittering(미세 이동 후보)** 및 (선택) **phi-grid(황금비율) 후보**를 추가하여 “Grid 간극”으로 좋은 구도 누락을 줄임
> - (v1.6) Qwen Stage-2 프롬프트/스키마에 **Composition Checklist(헤드룸/룩룸/수평/대칭/배경)** 필드와 *수치 근거*를 포함해 일관성과 설명 가능성을 강화(환각 완화)
> - (v1.6) Verify/QA에 **headroom/lookroom/horizon/symmetry** 위반율·분포·카테고리별 게이트를 추가하고, copy-space 태그에 대한 편향/예외 규칙을 명확화
>
> **Changelog v1.5**
> - Teacher Scoring의 `R_comp`에 **rule-of-thirds(삼분할) prior 수식/구현 규칙**을 명시적으로 추가 (v1.1 누락 보완)
> - Cheap 단계에서 `R_comp`가 **후보 N→M(20~40) 축소 전용**임을 명확화
> - (v1.3) **원본(프로 프레이밍)이 정답**인 경우를 위해 baseline 후보(b_full / b_maxarea) 강제 포함 + keep-vs-crop 게이팅(Δ개선 임계치) + over-crop QA 지표를 추가
> - Qwen rationale tag vocabulary에 `rule_of_thirds` 등 구도 태그를 추가하고, 자동/결정적 태깅 규칙(권장)을 명시
> - 이산 action head 표기를 `{ZOOM_IN, ZOOM_OUT}`로 정리(문서 일관성 보완)
> - (v1.5) EPIC C / Stage 2 **Feature Extraction**에 대해: (1) 다른 제안안 비교 분석(Feature별), (2) 3-스택(품질/효율/초고속) 추천, (3) **A100 80GB × 8노드 기준 10k 이미지 소요시간 추정표**를 추가
> - (v1.5) FastSAM vs EfficientViT-SAM/SAM2.1의 포지션을 정리하고, 대량 처리에서는 **bbox prompt 기반 EfficientViT-SAM 우선**을 권장하도록 보완


> **통합 대상 문서**:  
> - `Shutterstock Cropping Data Factory & Qwen2.5-VL Labeler Design Spec (v1)`  
> - `Shutterstock 기반 Image Cropping “데이터 팩토리” & Qwen2.5-VL 라벨 생성기 설계서 (v1.0)`  
> - `Shutterstock 400M 기반 Image Cropping 데이터셋 구축 실행 문서 (통합본)`  
>
> **목적**: Shutterstock 이미지(상용 라이선스 보유) + 메타데이터(alt-text/tags/caption)만을 입력으로,  
> 오픈 모델(Detection/Seg/OCR/CLIP/Aesthetic/MLLM·VLM)을 **Teacher**로 활용하여  
> **상용 학습 데이터셋(Training/Val)** + **비상용 평가셋(Eval/Benchmark)** + **Golden 품질 게이트**를  
> “품질 우선”으로 구축하는 **데이터 팩토리(제조 파이프라인) 설계**를 제공한다.
>
> **핵심 원칙**
> - 400M 전수 라벨링 금지: `Filter → Generate → Verify` 반복으로 “고품질 수백만”을 만든다.
> - 좌표는 VLM이 직접 생성하지 않게(불안정): **후보(Candidate) 생성 + 수치 스코어링 + Qwen은 선택/정당화/검증**에 집중.
> - “정답 crop”은 비유일: **AR-조건부 Top-K(3~5)** + 점수/플래그 + 설명(why)을 저장.
> - 데이터 릴리즈는 항상 **GoldenSet 게이팅 + QA 리포트 + Manifest(재현성)**로 통제한다.

---


## 1) 파이프라인 개요 및 핵심 원칙


### 3) 데이터 팩토리 전체 아키텍처

#### 3.1 파이프라인 개요: Filter → Generate → Verify → Release

```
Shutterstock 400M + metadata
  └─ Phase A: FILTER (400M → 1~5M Curated Pool)
       A1 Dedupe/near-dup clustering (pHash + CLIP embedding)
       A2 Technical quality filter (blur/lowres/watermark/template/text-heavy)
       A3 Category balancing (tags 기반 층화)
       A4 Aesthetic pre-score (카테고리별 percentile cut)
  └─ Phase B: GENERATE (Curated Pool → 라벨)
       B0 Perception precompute (det/seg/face/pose/ocr/saliency)
       B0.5 Composition precompute (CompEnc(PICD): comp24/emb + optional line/pattern/vanishing features)
       B1 Candidate generator (AR별 후보 150~300 → 다양성 샘플링)
       B2 Cheap Score (빠른 축소 N → M=20~40)
       B3 Expensive Score (정밀 스코어 + Top-K diversity)
- **composition embedding(PICD) QA (v1.7)**:
  - `comp24_preserve_rate` (keep_full|minimal_crop subset): \( \Pr[\arg\max p(I_b)=\arg\max p(I)] \) 또는 \( \cos(e(I),e(I_b))>\tau \) 비율
  - `comp24_entropy`: \(\mathcal{H}(\text{comp24})\) (분포 붕괴 감시; “3분할/중앙만” 과잉 방지)
  - `CDA@PICD` / `CDA@semantic_interference` (아래 정의): **릴리즈 간 회귀 금지** + 목표치(초기: 공개 베이스라인 상회)로 게이팅

  \[
  \text{CDA}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\big(\hat{neg}_i=neg_i\big)
  \]
  - (운영) PICD triplet task를 그대로 돌리거나, 내부에서도 `comp24` 고신뢰 샘플로 triplet을 구성해 **일일/주간 회귀 테스트**로 사용
       B4 Qwen2.5-VL (선택/설명/검증) on TopM candidates (10~20)
       B5 Silver/Golden 라우팅 (불확실 샘플은 재판정/인간 검수)
  └─ Phase C: VERIFY/QA (자동 + 샘플링 QC)
       C1 Structural checks (bbox/AR/면적/Top-K 형식)
       C2 Semantic checks (cut-off/text/coverage/caption drift)
       C3 Human spot-check (category×difficulty strata)
       C4 GoldenSet gate 통과 시 Release vX.Y
       C5 Composition regression (CDA@PICD + semantic_interference CDA)
```

#### 3.2 권장 컴퓨팅/저장 스택(품질 우선형)
- 대규모 ETL/샘플링/조인/클러스터링: **Spark**
- GPU 추론/후보 평가/라벨 생성: **Ray (A100 Actor 상주)**
- 저장:
  - 인덱스/피처/라벨: **Delta/Iceberg + Parquet**
  - 학습 패키징: **WebDataset(tar shards)** 또는 TFRecord
- 버전 관리: DVC/lakeFS/manifest(권장) - “재현성”이 품질의 일부

---

#### 3.3 (v1.7) PICD를 “어디에, 어떻게” 붙일까 - 가장 효과적인 결합 방식(권장)

PICD는 (1) **24-class 구도 taxonomy**, (2) **CDA**라는 경량 평가, (3) “semantic interference에 취약”이라는 실패 분석을 제공합니다.  
v1.7에서는 이를 아래처럼 **데이터 팩토리의 ‘규칙→피처→스코어→QA’ 루프**에 직접 연결합니다.

1) **Stage-2 Feature에 CompEnc(PICD) 추가 (B0.5)**
- 목표: “3분할/중앙/수평선” 같은 일부 룰만이 아니라, **대각/삼각/곡선/원근/패턴/밀집/흩뿌림**을 포함한 *구도 다양성*을 수치 피처로 갖는다.
- 출력(저장): `composition_emb(d=256)`, `comp24_logits`, `comp24_label/conf` (+선택 `arrangement12`, `element_type`)

2) **Candidate Generator를 composition-guided로 확장 (B1)**
- full-image `comp24_label`(고신뢰)에 따라 **특화 후보(anchor/jitter)를 추가**:
  - `LS-Hori3/LS-Hori2`: horizon-third / horizon-middle 후보 강화
  - `P-Dia/LS-Dia`: diagonal axis(주축) 정렬 후보
  - `P-Tri/LS-Tri`: triangle(3점/3변) 커버 후보
  - `LS-S-Cur/LS-C-Cur/LS-O-Cur`: curve 구조 보존 후보
  - `S-Per/LS-Dif`: vanishing/perspective / diffuse 후보
  - `PL-Pat/PL-Den/P-Scat`: pattern/dense/scatter 영역 유지 후보
- 구현은 “선 기반(semantic line/hough) + mask PCA + CompEnc re-rank” 조합으로 충분(모든 카테고리를 완벽 검출하려 하지 말고, **후보 다양성만 늘리면** 됨)

3) **Cheap/Expensive Score에 \(R_{\text{picd}}\) 추가 (B2/B3)**
- Cheap: `R_picd`를 **N→M pruning**에만 사용(과도 비용 방지 + 다양성 유지)
- Expensive: keep-vs-crop 게이팅과 함께 `R_picd-preserve`로 “원본 구도 파괴”를 억제

4) **Verify/QA에 CDA를 릴리즈 게이트로 추가 (C5)**
- `CDA@PICD`(Task I) + `CDA@semantic_interference`(Task II)로 CompEnc 품질을 추적
- 내부에서도 Shutterstock tags 기반으로 “semantic interference triplet”을 자동 구성해 주간 회귀 테스트로 운영

5) **Qwen은 ‘구도 판정자’가 아니라 ‘설명/검증자’**
- `comp24_full/crop`, `comp_preserve_sim`, `horizon/headroom/lookroom` 등 **수치 피처**를 제공하고,
- Qwen은 체크리스트(pass/fail) + why_tags/why_text를 작성(환각 최소화)


### 4) 거버넌스/스플릿/법무 체크

#### 4.1 Commercial vs Research 분리 규칙
- **상용 학습(Training/Val)**: Shutterstock(권리 확보)만 사용
- **공개 데이터셋**: 기본적으로 Eval/Benchmark 용도(비상용). 상용 학습에 섞지 않는다.

#### 4.2 Never-train 세트
- `SS-Golden-v1`: 학습 금지(절대 게이트)
- `Stress-v1`: 실패 유발 케이스 모음(회귀 테스트)

#### 4.3 Split 정책(누수 방지)
- near-duplicate cluster 단위로 split 할당
- 동일 cluster의 이미지가 train/val/test에 분산되지 않도록 강제
- 릴리즈마다 “누수 리포트 = 0”이 필수 AC

---




## 2) 사전 정의: 표기, 입력/출력, 태스크 분해


### 0) 용어/표기

- 원본 이미지: \(I\), 크기 \(W\times H\)
- 목표 종횡비(Aspect Ratio): \(r\in\mathcal{R}\) (예: 1:1, 9:16, 16:9, 3:4, 4:3, 4:5 …)
- 크롭 박스: \(b=(x_1,y_1,x_2,y_2)\) (정규화 좌표 [0,1] 권장)
- AR별 후보 집합: \(\mathcal{B}_r=\{b_{r,i}\}_{i=1..N}\)
- AR별 Top-K 크롭: \(\{b_{r,1..K}\}\)
- IoU:
  \[
  IoU(b_1,b_2)=\frac{|b_1\cap b_2|}{|b_1\cup b_2|}
  \]

---


### 1) 목표/제약 (Input → Output)

#### 1.1 입력(제약)
- 사용 가능한 원천 데이터: **Shutterstock 이미지 + 메타데이터(alt-text/tags/caption)**
- 내부 크롭핑 모델/GT는 없다고 가정(초기) → Teacher 기반으로 라벨을 “생성”해야 함
- 오픈 모델 사용 가능(상용 사용 가능 여부는 별도 법무 확인)
- 컴퓨팅: **A100-80GB 다수**, 시간보다 **품질 우선**

#### 1.2 출력(산출물)
**(A) Commercial Training/Val**
- `SS-CropTrain-v1`: AR-조건부 Top-K 크롭 + 점수/플래그 + 설명 + provenance
- `SS-CropVal-v1`: 동일 스키마의 검증용 split

**(B) Commercial Golden 품질 게이트 (학습 금지)**
- `SS-Golden-v1`: 전문가 검수 기반 고정밀 GT(2k~10k)

**(C) Non-commercial Eval/Benchmark**
- `CropBench-Mix-v1`: GAICD/FCDB/FLMS/CPC/SACD 등 공개 벤치 + 내부 Stress set

**(D) 운영 산출물**
- QA Report (분포/위반률/다양성/누수 검사)
- Release Manifest (scorer/prompt/candidate 파라미터/코드 커밋 해시 등 재현성)

---


### 2) 태스크 분해 (데이터 관점)

크롭/프레이밍은 실제로 여러 태스크가 섞인다. 데이터는 태스크별로 분리해 디버깅 가능하게 만든다.

1) **Aesthetic crop (free-form)**: AR 조건 없이 “보기 좋은” 크롭 1개(옵션)  
2) **AR-conditioned crop (필수)**: 목표 AR별로 최적 Top-K  
3) **Subject-preserving crop (필수)**: 주 피사체 절단 방지(인물/얼굴/제품/동물 등)  
4) **View recommendation / shift-zoom (선택)**: UNIC 스타일 “초기 뷰 → 최적 뷰” 액션 라벨

---




## 3) Phase A: Filter & Curate (400M → Curated Pool)


### 5) Phase A - Filter (400M → Curated Pool)

#### 5.1 Ingestion & Index 테이블
`images` 테이블(또는 parquet) 기본 필드:
- `image_id`, `uri`, `W`, `H`
- `tags[]`, `alt_text`, `caption`
- `license_ok_train` (상용 학습 가능 플래그)
- `phash`, `clip_emb_ref`, `cluster_id`, `split`
- 품질 메트릭: `blur_score`, `watermark_prob`, `frame_prob`, `text_density`

#### 5.2 Dedup / Near-dup clustering
- Stage 1: pHash로 bucketing
- Stage 2: OpenCLIP embedding으로 bucket 내 클러스터링
- Split은 cluster 단위로 할당

#### 5.3 Technical quality filter
- 해상도: min side ≥ 512(최소), ≥1024(권장)
- 블러/노이즈/워터마크/프레임/캘린더/템플릿 제거
- 텍스트 밀집(포스터/문서)은 별도 트랙(`doc_poster`)으로 분리

#### 5.4 Metadata consistency filter
- `clip_sim(image, caption)`, `clip_sim(image, main_tag_phrase)`를 계산해 극단 mismatch 제거/다운랭크

#### 5.5 Category balancing
- tags 기반 상위 카테고리 층화(인물/풍경/음식/제품/동물/건축/실내/스포츠/문서 등)
- long-tail 태그는 oversample

#### 5.6 Aesthetic pre-score percentile cut
- 카테고리별 aesthetic 분포가 다름 → **카테고리별 percentile**로 컷(상위 10~30% 등)

**Output:** `CuratedPool-v1` (1~5M 권장 시작)

---




## 4) 메타데이터 정규화 & Intent 추정


### 6) 메타데이터 정규화 룰셋 (Shutterstock tags/alt/caption → 주체/의도/카테고리)

> 목표: raw metadata를 모델/라벨링 파이프라인에서 안정적으로 쓰도록 **정규화된 구조**로 변환한다.

#### 6.1 Tag 정규화
- lower-case, 공백/하이픈 통일: `copy-space`, `copyspace` → `copy space`
- 중복 제거
- **4종 분류**:
  - `SUBJECT`: person, woman, dog, car, burger, building …
  - `CONTEXT`: beach, cafe, city, indoor, night …
  - `STYLE/TECH`: bokeh, studio, HDR …
  - `COMPOSITION_HINT`: copy space, portrait, isolated, panoramic, flat lay, close up, full length, group …

**Stop-tags(주체 선정에서 제외)** 예시:
- horizontal/vertical, outdoor/indoor, color image, background, template, banner, closeup, daylight…

#### 6.2 Caption noun-phrase 추출
- spaCy/Stanza로 noun-chunks 추출
- 역할 가중(초기):
  - 주어(nsubj): +1.0
  - 목적어(dobj/pobj): +0.7
  - 기타: +0.4
- 너무 일반적인 head(thing, photo, image 등)는 제거

#### 6.3 main_subject 후보 풀
```
C = topN_subject_tags (N=10) ∪ topM_caption_nps (M=5) ∪ (optional) detected_labels_topK
```

#### 6.4 main_subject 선정 점수(구현 룰)
각 후보 s에 대해:

- `tag_rank_score(s)`: tags에서 앞일수록 가중
- `caption_np_score(s)`: caption noun phrase 포함 여부
- `clip_confirm(s)`: CLIP(image, text=s) similarity
- `det_confirm(s)`: detector 존재 여부

추천 결합식:
```
S_subject(s) = 0.35*tag_rank + 0.15*caption_np + 0.35*clip + 0.15*det
main_subject = argmax_s S_subject(s)
confidence = softmax(max)
```
- `confidence < 0.45`이면 main_subject="unknown"으로 둔다.

#### 6.5 multi-subject 판정
`multi_subject=true` if any:
1) detector에서 인물/객체 인스턴스 2개 이상(각 면적 ≥ 0.02)
2) caption/tags 패턴: couple, group, crowd, family, “two/three …”
3) 후보 2위가 1위와 근접: `S2/S1 ≥ 0.75` & `S2 ≥ 0.55`

#### 6.6 특수 태그 처리(의도/제약 생성)
| hint tag | intent | 크롭 제약/선호 |
|---|---|---|
| copy space / negative space | preserve_copy_space | 빈 공간 비율 유지(예: >15%), subject 중앙 고정 불필요 |
| portrait / headshot / face | avoid_face_cut + keep_headroom + keep_lookroom | 얼굴 절단 금지, headroom + lookroom(시선 여백) 확보 |
| full length | avoid_person_cut | 머리/발/관절 절단 패널티 강화 |
| isolated / white background | product_packshot | 객체 주변 margin 5~15%로 타이트 |
| flat lay / top view | preserve_layout | 배치 유지, 가장자리 cut 강화 |
| panoramic / wide | include_context_wide | wide AR 선호, 너무 타이트한 크롭 회피 |
| landscape / seascape / horizon / skyline | keep_horizon_thirds + keep_leveling | 수평선 1/3·2/3 위치 + 수평(roll) 보정/QA |
| poster / template / sign | preserve_text | OCR text cut 금지(트랙 분리 권장) |
| group / family | keep_all_faces | 모든 얼굴 포함 최우선 |

intent는 priority로 정렬해 저장:
- 예: 인물+copy space → `avoid_face_cut(1.0) > subject_emphasis(0.9) > preserve_copy_space(0.7)`

---



#### 6.7 Portrait 전용: **ShotType(Headshot/Half/Full)** + **Headroom/Lookroom** 파라미터 자동 추정 (Shutterstock tags 기반)

> 목적: (1) **portrait에서는 ‘안 잘림’만으로는 부족**(머리 위가 너무 벙벙/답답, 시선 앞이 막힘 등)하므로,  
> (2) Shutterstock 메타(tag/caption)만으로도 **초기 파라미터(prior)**를 안정적으로 만들고,  
> (3) Stage-2 Feature(얼굴/포즈/시선)와 결합해 **Score→QA까지 닫히는 규칙 기반 시스템**을 만든다.

##### 6.7.1 입력 신호(메타) & 정규화
- `tags/keywords[]`: lowercase, 공백/하이픈 통일, 중복 제거
- `caption/alt_text`: noun-phrase(NP) 추출(간단 규칙으로도 충분)
  - 예: `"... portrait of a smiling woman with copy space"` → `{portrait, woman, copy space}`
- 결과: `meta_tokens = set(tags ∪ NP(caption) ∪ NP(alt_text))`

##### 6.7.2 ShotType Prior 추정 룰(태그 기반)
우선순위는 **Group → Full → Half → Headshot → Unknown** (충돌 시 상위가 승리).

| shot_type | 매칭(예: meta_tokens에 포함) | 비고 |
|---|---|---|
| `group` | `group`, `family`, `team`, `friends`, `couple`, `crowd`, `people` | group은 lookroom 대신 “group 균형/여백”을 중점 |
| `full` | `full body`, `full length`, `full-length`, `whole body`, `standing`, `head to toe`, `feet`, `legs` | “발/발목” cut-off가 중요 |
| `half` | `waist up`, `upper body`, `half body`, `half length`, `mid shot`, `medium shot`, `torso` | “팔꿈치/손목” cut-off 중요 |
| `headshot` | `headshot`, `close up`, `close-up`, `closeup`, `portrait`, `face`, `beauty`, `selfie`, `profile picture` | “헤드룸/눈 위치” 중요 |
| `unknown` | 위에 해당 없음 | 추후 Stage-2 feature(얼굴 크기/키포인트)로 보정 |

추가 플래그(동시에 가질 수 있음):
- `is_profile_view`: `profile`, `side view`, `looking left`, `looking right`
- `has_copy_space`: `copy space`, `copyspace`, `negative space`, `blank space`, `text space`, `banner`, `background`
- `is_isolated_packshot`: `isolated`, `cut out`, `on white`, `white background` (제품/인물 “팩샷” 가능)

##### 6.7.3 Portrait Category(옵션) 추정 룰 - 파라미터 미세 조정용
Shutterstock tag는 도메인이 풍부하므로, 아래 카테고리 분류로 **center vs thirds / headroom 범위 / lookroom 가중치**를 미세 조정할 수 있다.

| portrait_category | 매칭 힌트 | 의미/조정 |
|---|---|---|
| `formal_id` | `passport`, `id photo`, `mugshot`, `profile`, `linkedin`, `resume` | **중앙구도**↑, lookroom↓, headroom max↓ |
| `corporate` | `business`, `corporate`, `professional`, `office`, `executive` | 중앙구도↑, lookroom↓(과도한 lead space 억제) |
| `beauty_fashion` | `beauty`, `makeup`, `fashion`, `model`, `glamour`, `hairstyle` | hair 포함 위해 headroom max↑ |
| `lifestyle` | `lifestyle`, `outdoor`, `travel`, `candid` | thirds/phi 가중치↑ |
| `generic` | default | 기본값 |

##### 6.7.4 Headroom 파라미터 산출(메타 prior → feature로 보정)
**정의(후보 crop \(b\)에서 관측 headroom 비율)**  
- crop: \(b=(x_1,y_1,x_2,y_2)\) (pixel 또는 norm)  
- 머리 최상단 \(y_{\text{head}}\)는 Stage-2 feature로 추정:
  - (권장) pose/landmark로 **top-of-head**가 있으면 사용
  - (대체) face bbox 기반: \(y_{\text{head}}\approx y_{\text{face\_top}}-0.15\cdot h_{\text{face}}\)

\[
r_{\text{head}}(b)=\frac{y_{\text{head}}-y_1}{y_2-y_1}
\]

**샷타입별 기본 prior(권장 시작값, 이후 GoldenSet으로 캘리브레이션)**

| shot_type | \(r_{\text{head}}^{\min}\) | \(r_{\text{head}}^{*}\) (target) | \(r_{\text{head}}^{\max}\) | 비고 |
|---|---:|---:|---:|---|
| headshot | 0.03 | 0.07 | 0.12 | 눈(eye-line)이 상단 1/3 근처면 자연스럽게 만족 |
| half | 0.04 | 0.08 | 0.14 | 팔/손 cut-off와 trade-off |
| full | 0.02 | 0.06 | 0.10 | 발/발목 보존을 위해 headroom 과다 억제 |
| group | 0.03 | 0.08 | 0.14 | “최상단 머리” 기준 |

**카테고리/특수태그 보정(단순 가산/감산 룰)**
- `formal_id`/`corporate`: \(r^{\max}\leftarrow r^{\max}-0.03\), center_bias ↑
- `beauty_fashion`: \(r^{\max}\leftarrow r^{\max}+0.05\) (헤어/헤드기어)
- `has_copy_space`: \(r^{\max}\leftarrow \min(0.35, r^{\max}+0.15)\) (의도적 여백 허용)

**스코어 항(cheap scorer에 사용)**
\[
R_{\text{headroom}}(b)=
-\frac{|r_{\text{head}}(b)-r_{\text{head}}^{*}|}{\sigma_h}
-\gamma_h\cdot \max(0,\;r_{\text{head}}^{\min}-r_{\text{head}}(b),\;r_{\text{head}}(b)-r_{\text{head}}^{\max})
\]

권장: \(\sigma_h=0.05,\ \gamma_h=2.0\). (GoldenSet으로 튜닝)

##### 6.7.5 Lookroom/Nozeroom 파라미터 산출(메타 prior → gaze/headpose로 보정)
**핵심 아이디어:** “시선/얼굴 방향” 쪽 여백을 더 준다.

- Stage-2에서 `gaze_dir` 또는 `headpose_yaw`(대체)를 얻어 \(g\in\{-1,0,+1\}\)로 양자화
  - \(g=+1\): 오른쪽을 봄, \(g=-1\): 왼쪽을 봄, \(g=0\): 정면/불명확
- subject anchor \(x_s\): 얼굴 중심 또는 상체 중심

\[
m_L = x_s-x_1,\quad m_R=x_2-x_s
\]
\[
m_{\text{fwd}}=\begin{cases}m_R & g=+1\\ m_L & g=-1\end{cases},\quad
m_{\text{back}}=\begin{cases}m_L & g=+1\\ m_R & g=-1\end{cases}
\]
\[
r_{\text{look}}(b)=\frac{m_{\text{fwd}}+\epsilon}{m_{\text{back}}+\epsilon}
\]

**샷타입별 prior(권장 시작값)**

| shot_type | \(r_{\text{look}}^{\min}\) | \(r_{\text{look}}^{*}\) | \(r_{\text{look}}^{\max}\) | 비고 |
|---|---:|---:|---:|---|
| headshot | 1.20 | 1.50 | 2.50 | profile이면 상향(최대 3~4 허용) |
| half | 1.15 | 1.30 | 2.00 | |
| full | 1.05 | 1.15 | 1.60 | |
| group | - | - | - | group은 “양쪽 균형”으로 대체 |

**특수태그 보정**
- `is_profile_view`: \(r^{*}\leftarrow r^{*}+0.2\), \(r^{\max}\leftarrow r^{\max}+0.5\)
- `has_copy_space`: \(r^{\max}\leftarrow \min(4.0, r^{\max}+1.0)\) (의도적 lead space)

**스코어 항**
\[
R_{\text{lookroom}}(b)=\mathbb{1}[|g|=1]\cdot
\left(
-\frac{|r_{\text{look}}(b)-r_{\text{look}}^{*}|}{\sigma_\ell}
-\gamma_\ell\cdot \max(0,\;r_{\text{look}}^{\min}-r_{\text{look}}(b),\;r_{\text{look}}(b)-r_{\text{look}}^{\max})
\right)
\]

권장: \(\sigma_\ell=0.25,\ \gamma_\ell=1.0\).  
정면(\(g=0\))이면 \(R_{\text{lookroom}}=0\)으로 두고, 대신 **center/symmetry** 항을 사용.

##### 6.7.6 meta_norm 출력 필드(권장)
- `portrait.shot_type_prior ∈ {headshot, half, full, group, unknown}`
- `portrait.category ∈ {formal_id, corporate, beauty_fashion, lifestyle, generic}`
- `portrait.flags = {is_profile_view, has_copy_space, is_isolated_packshot}`
- `portrait.params = {headroom_min, headroom_target, headroom_max, lookroom_min, lookroom_target, lookroom_max, w_headroom, w_lookroom, w_center_bias}`

> **중요**: 위 값들은 “학습 정답”이 아니라 **cheap scorer/QA의 prior**입니다.  
> GoldenSet(전문가 GT)로 분포를 측정해 분기/파라미터를 재학습(보정)하는 것이 안정적입니다.




## 5) Phase B: 라벨 생성 (Curated Pool → Labels)


### 7) Perception Precompute (좌표 안정성의 핵심)

대규모 좌표 품질을 위해 “좌표를 생성”하기보단 “좌표를 검증/패널티”할 수 있는 시그널을 선계산한다.

#### 7.1 필수 모듈 & 출력 (v1.6)

| 모듈 | 출력(저장) | 활용(Score/QA) |
|---|---|---|
| object detection | bbox + class + score | 주체 coverage, multi-subject, context objects |
| segmentation / salient mask | mask(RLE) + bbox + centroid | 경계 절단/거리 패널티, subject coverage, context ratio |
| face detection (+landmarks) | face bbox + (5~106) landmarks | 얼굴 cut-off, **headroom(머리)**, (gaze 입력) |
| person pose (keypoints) | 17-kp(또는 wholebody) + person bbox | 관절 cut-off(목/무릎/발목/손목), full-body 보존 |
| **head pose / gaze** | headpose(yaw/pitch/roll) 또는 gaze target/dir + conf | **lookroom/noseroom**, (선택) movement lead-room |
| **horizon / roll** | horizon line(y, θ) + conf, roll_deg | **horizon_on_third**, leveling/tilt QA, landscape 구도 |
| OCR | text boxes + conf | 텍스트 보존/트랙 분리, text_cut 패널티 |
| symmetry(cheap, optional) | symmetry_score(0~1) | **center composition** 가중/게이트(대칭이면 중앙을 살림) |
| aesthetic pre-score(optional) | A(I), A(I_b) 일부 | Cheap 단계 prior, hard-case sampling |

> 원칙: Stage-2는 “크롭 좌표를 직접 생성”하지 않고, **크롭 후보 평가에 필요한 근거 피처를 선계산**하여  
> Cheap/QA에서 **결정적으로(Deterministic)** 사용한다.


#### 7.3 (추가) 모델 선택 가이드 - **품질 최우선 / 고효율(대량 처리) / 초고속** 3-스택

아래는 “라이선스/상용 제한은 고려하지 않는다”는 조건에서, **데이터 팩토리(오프라인 대량 추출)** 관점으로 *현실적으로* 쓸 수 있는 3가지 우선순위 스택입니다.

> **중요 해석(클러스터 스펙)**  
> 사용자가 적어준 “A100 80GB × 8 노드”는 환경마다 **(a) 8 GPU(total)** 또는 **(b) 64 GPU(8노드×8GPU)**로 해석될 수 있습니다.  
> 아래 시간 표는 **두 경우를 모두** 계산해 제공합니다(원하는 값으로 GPU 개수만 바꿔 선형 스케일링 가능).

##### 7.3.1 “다른 챗GPT 제안안” 검토/비교 분석(Feature별 핵심 결론)

- **C1 CLIP(이미지/텍스트 임베딩)**
  - (제안안의 장점) SigLIP2/MetaCLIP/DFN/EVA-CLIP 같은 대형 임베딩은 **구도·디테일·의미 정합**이 좋고, 특히 상위 hard-case에서 “caption drift(주제가 crop에서 사라짐)”을 잡는 데 유리합니다.
  - (현실적 보완) 하지만 전수(수백만~) 임베딩을 대형 모델로 돌리면 비용이 급증하므로,
    - **1-pass**: 중간급(OpenCLIP ViT-L/14 또는 SigLIP2 so400m)으로 전수 임베딩
    - **2-pass**: 상위 1~5%(또는 hard-case)만 대형(PE bigG / EVA-CLIP / DFN-5B)으로 재스코어  
    구조가 “품질 최우선” 조건에서도 가장 실용적입니다.
  - (추가 포인트) 텍스트 임베딩은 **중복 caption/tags**가 많으므로 text encoder는 캐시(Unique text hash)하면 비용이 크게 줄어듭니다.

- **C2 Seg/Saliency(주 피사체 마스크)**
  - (제안안의 장점) SAM2 계열은 마스크 품질 상한이 높고, 복잡한 경계/가림에서 강합니다.
  - (핵심 보완) 대량 처리에서는 “prompt 기반 1-mask”가 목표이므로, **EfficientViT-SAM이 처리량 관점에서 압도적**입니다(단 bbox prompt 필요).
  - (prompt 없는 트랙) 만약 bbox prompt가 없는 이미지가 많다면 DIS/IS-Net 같은 **전경 1-mask(SOD)** 트랙을 별도로 두는 것이 운영적으로 유리합니다.
  - (FastSAM 재평가) FastSAM은 “SAM 대비 50× 빠름”을 주장하지만, repo 기준 **~40ms/img(≈25fps, RTX3090)** 수준이라  
    “A100에서 bbox prompt 1-mask” 목적이라면 EfficientViT-SAM이 더 적합한 경우가 많습니다(아래 상세).

- **C3 Face/Pose(사람 cut-off)**
  - (제안안의 장점) ViTPose는 keypoint 품질 상한이 높고, RTMPose/YOLOv8-pose는 throughput이 좋습니다.
  - (데이터 팩토리 관점 결론)
    - **초고속/대량 1-pass**는 “탐지+키포인트”를 한 번에 내는 **YOLOv8-pose(또는 RTMDet+RTMPose)**가 운영이 쉽습니다.
    - **품질 최우선**은 ViTPose-H(또는 유사 SOTA) + (필요 시) SCRFD 같은 강한 face detector로 보완하는 2-stage가 좋습니다.
  - (실무 팁) 얼굴 bbox는 **전용 face detector**가 가장 안정적이지만, 비용이 부담되면 1-pass에서는 **키포인트로 얼굴 bbox 근사** 후  
    hard-case(얼굴이 작거나 가림)에서만 face detector를 선택적으로 실행하는 방식이 효율적입니다.

- **C4 OCR(텍스트 박스)**
  - (제안안의 장점) docTR/MMOCR는 PyTorch 실험/교체가 편하고, DBNet/CRAFT는 det-only 베이스라인으로 좋습니다.
  - (데이터 팩토리 결론) “박스만 필요(det-only)” + “대량 처리”에서는 **PaddleOCR가 문서화/최적화(TensorRT)·운영 성숙도** 측면에서 유리합니다.
  - (운영 핵심) 텍스트가 거의 없는 트랙(풍경/인물 등)은 OCR을 스킵하고, poster/document/banner 등 텍스트 트랙에만 적용하세요.


- **C5 Horizon/Leveling(수평선/롤 추정)**
  - (필요성) 풍경/해변/도시 스카이라인 등에서 **수평선 위치(horizon on 1/3·2/3) + 기울기(roll)**는 “좋은 구도”를 결정하는 핵심 신호입니다.
  - (품질 최우선) **ScaleLSD(CVPR 2025)** 같은 최신 line segment detector로 라인을 뽑고, RANSAC으로 **가장 강한 수평 후보**를 horizon으로 추정(실무적으로 가장 튼튼).
    - 장점: 다양한 장면에서 안정적, “수평선이 없는 장면”도 conf로 걸러낼 수 있음.
  - (대안/전용) **NG-DSAC Horizon**(horizon line 전용 추정, differentiable RANSAC 계열)로 y-position/angle을 직접 예측하는 방식도 있음(HLW 계열).
  - (초고속) OpenCV LSD/EDLines(+RANSAC) 같은 classical line detector는 **GPU 없이도** 충분히 빠르며, “QA용 roll 추정”에는 꽤 유용합니다.

- **C6 Gaze/HeadPose(시선/고개 방향)**
  - (필요성) 인물이 오른쪽을 보고 있는데 오른쪽 여백이 없으면 **답답함**이 생깁니다(lookroom/noseroom). 이는 “안 잘림”과 별개로 품질을 크게 좌우합니다.
  - (품질 최우선) **Gaze-LLE(CVPR 2025 highlight)** 같은 gaze target 추정 모델을 사용하면 “시선이 향하는 위치(heatmap/point)”를 얻을 수 있어 lookroom을 강하게 닫을 수 있습니다.
  - (고효율) **L2CS-Net**(coarse gaze classification)처럼 “좌/우/정면” 근사만 안정적으로 얻어도 lookroom 최적화에는 충분한 경우가 많습니다.
  - (초고속/백업) gaze가 어려운 경우 **head pose yaw**(예: 6DRepNet 계열)를 쓰면 “대략 왼쪽/오른쪽”은 상당히 잘 잡힙니다.

> 운영 팁: C5/C6는 **모든 이미지에 전수 적용하지 말고**,  
> - `portrait`/`people` 트랙에서만 C6(gaze) 실행,  
> - `landscape`/`seascape`/`city skyline` 트랙에서만 C5(horizon) 실행  
> 같은 **메타 기반 라우팅**으로 비용을 크게 줄일 수 있습니다.

##### 7.3.2 스택 개요(권장)
| Priority | C1 CLIP | C2 Seg/Saliency | C3 Face/Pose | C4 OCR(det-only) | C5 Horizon/Roll | C6 Gaze/HeadPose | 사용 맥락 |
|---|---|---|---|---|---|---|---|
| **품질 최우선 1순위** | SigLIP2/MetaCLIP 계열(고품질 임베딩) | SAM2(또는 HQ-SAM) | **RTMO-l 또는 ViTPose++** + SCRFD | PP-OCRv5 server det | **ScaleLSD(+RANSAC)** *(landscape만)* | **Gaze-LLE** *(portrait만)* | GoldenSet/Calibration, Hard-case 재검증, Teacher 품질 상한 측정 |
| **고효율(대량 처리) 1순위** | SigLIP2 so400m/ViT-L (INT8) | EfficientViT-SAM(L2) 또는 DIS | **RTMO-s/RTMPose-m** + 경량 face det | PP-OCRv5 mobile det 또는 DBNet++ | ScaleLSD-small/DeepLSD *(landscape만)* | L2CS-Net 또는 MobileGaze *(portrait만)* | **기본 대량 추출(수백만~)**: throughput/품질 균형 |
| **초고속 1순위** | OpenCLIP ViT-B (INT8) | DIS/IS-Net(전경1-mask) | YOLOv8n-pose 또는 RTMO-tiny | CRAFT/DBNet det | OpenCV LSD(+RANSAC) | headpose-yaw(6DRepNet) 또는 MobileGaze-S0 | 빠른 반복/사전 필터/프리뷰용, “먼저 돌려보고” 전략 |


> **FastSAM에 대한 재평가(다른 챗GPT 제안 대비)**  
> FastSAM은 “SAM 대비 50× 빠름”을 주장하지만, **repo에서 제시한 RTX3090 기준 inference ~40ms/img** 수준(≈25fps, 포인트 프롬프트 조건)이라  
> **A100에서 prompt 기반 1-mask**를 뽑는 용도라면 EfficientViT-SAM(L0/L2)의 *img/s 수백~수천* 레벨이 훨씬 유리합니다.  
> FastSAM은 오히려 “prompt 없이 대충 마스크 후보를 여러 개 뽑는” 간편한 베이스라인/프로토타이핑 용도로 남기는 편이 합리적입니다.

---

#### 7.4 (v1.6) **1만 장(10k) 처리 시간 추정** - A100 80GB × 8노드

아래 시간은 “**GPU 추론 순수 시간(대략)**” 기준입니다.  
실제 E2E는 **이미지 디코딩/리사이즈/후처리/저장(I/O)** + Ray 오버헤드로 인해 보통 `×(1.5~3.0)` 늘어납니다.

##### 7.4.1 계산식/가정
- 계산식(유틸리티 계수 포함):
  \[
  t_{\text{sec}} \approx \frac{N_{\text{img}}}{(\text{img/s/GPU})\cdot N_{\text{GPU}}\cdot u}
  \]
  - \(u\): 파이프라인 유틸리티(전처리/후처리 포함). **권장 \(u=0.7\)** 로 보수 추정.
- 트랙별 라우팅을 적용하면(권장) 전체 시간은 더 줄어듭니다.
  - 예: `C6 gaze`는 `portrait`에만, `C5 horizon`은 `landscape`에만, `C4 OCR`은 `text-heavy`에만.

> **주의**: 아래 throughput은 “계획 수립용 러프 숫자”입니다.  
> 최종 확정은 반드시 **mini-benchmark(1~5만장)**로 모델/배치/해상도별 실측을 권장합니다.

##### (A) 품질 최우선 스택 - 10k/feature (u=0.7 가정)
| Feature | Suggested model | img/s/GPU (rough) | 10k @8GPU | 10k @64GPU | 실행 트랙 |
|---|---|---:|---:|---:|---|
| C1 CLIP (img+text) | SigLIP2/ViT-H | 150 | 11.9s | 1.5s | all |
| C2 Seg/Saliency | SAM2(HQ 옵션) | 40 | 44.6s | 5.6s | all (또는 subject-heavy) |
| C3 Face/Pose | RTMO-l + face det | 160 | 11.2s | 1.4s | people/portrait |
| C4 OCR(det) | PP-OCRv5 server det | 60 | 29.8s | 3.7s | text-heavy |
| C5 Horizon/Roll | ScaleLSD(+RANSAC) | 150 | 11.9s | 1.5s | landscape |
| C6 Gaze/HeadPose | Gaze-LLE(또는 headpose+보정) | 30 | 59.5s | 7.4s | portrait |

- (참고) **모든 feature를 순차 실행**한다고 가정하면 총합:
  - 8 GPU(total) 기준: **2m49s**
  - 64 GPU(8노드×8GPU) 기준: **21.1s**

##### (B) 고효율(대량 처리) 스택 - 10k/feature (u=0.7)
| Feature | Suggested model | img/s/GPU (rough) | 10k @8GPU | 10k @64GPU | 실행 트랙 |
|---|---|---:|---:|---:|---|
| C1 CLIP | SigLIP2/ViT-L(INT8) | 300 | 6.0s | 0.7s | all |
| C2 Seg/Saliency | EfficientViT-SAM / DIS | 250 | 7.1s | 0.9s | all |
| C3 Face/Pose | RTMO-s / RTMPose | 400 | 4.5s | 0.6s | people/portrait |
| C4 OCR(det) | PP-OCR mobile det / DBNet++ | 250 | 7.1s | 0.9s | text-heavy |
| C5 Horizon/Roll | ScaleLSD-small / DeepLSD | 300 | 6.0s | 0.7s | landscape |
| C6 Gaze/HeadPose | L2CS-Net / MobileGaze | 250 | 7.1s | 0.9s | portrait |

- (순차 실행 가정 총합) 8GPU: **37.8s**, 64GPU: **4.7s**

##### (C) 초고속 스택 - 10k/feature (u=0.7)
| Feature | Suggested model | img/s/GPU (rough) | 10k @8GPU | 10k @64GPU | 실행 트랙 |
|---|---|---:|---:|---:|---|
| C1 CLIP | ViT-B(INT8) | 600 | 3.0s | 0.4s | all |
| C2 Seg/Saliency | DIS/IS-Net | 600 | 3.0s | 0.4s | all |
| C3 Face/Pose | YOLOv8n-pose / RTMO-tiny | 1000 | 1.8s | 0.2s | people/portrait |
| C4 OCR(det) | CRAFT/DBNet det | 400 | 4.5s | 0.6s | text-heavy |
| C5 Horizon/Roll | OpenCV LSD(+RANSAC) | 2000 | 0.9s | 0.1s | landscape |
| C6 Gaze/HeadPose | headpose-yaw / MobileGaze-S0 | 800 | 2.2s | 0.3s | portrait |

- (순차 실행 가정 총합) 8GPU: **15.3s**, 64GPU: **1.9s**


### 8) Candidate Generator (AR-조건부 후보 생성) - 구현 스펙

#### 8.1 기본 파라미터(권장 시작값)
| 파라미터 | 의미 | 기본값 |
|---|---|---:|
| `AR_LIST` | 목표 AR | ["1:1","9:16","16:9","3:4","4:3"] (+ "4:5" 옵션) |
| `GRID_M, GRID_N` | 그리드 | 12, 12 |
| `SCALE_SET` | 면적비 스케일 | [0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95] |
| `A_MIN, A_MAX` | 면적비 하한/상한 | 0.15, 0.95 |
| `AR_TOL` | AR 허용 오차 | 0.01~0.03 |
| `MAX_CANDIDATES_PER_AR` | 후보 cap | 150~300 (품질 우선이면 300) |
| `NMS_IOU` | 중복 제거 | 0.90 |
| `DIVERSITY_CLUSTER_K` | 다양성 샘플링 | 150~300 유지 시 optional |
| `BORDER_MARGIN_ALPHA` | 절단 margin 기준 | 0.02~0.04 * min(w,h) |

#### 8.2 후보 생성 방법(조합 권장)

##### 8.2.0 Baseline 후보(원본/최대면적) - “원본이 정답” 케이스를 **반드시 포함**

Shutterstock에는 이미 **전문 사진가가 의도한 프레이밍이 ‘완성품’**으로 존재하는 샘플이 상당히 많습니다(GenCrop의 핵심 가정).  
따라서 후보 풀에는 항상 **(i) 원본 유지(keep)** 또는 **(ii) 목표 AR에서의 최대면적(minimal crop)** 후보를 *강제로 포함*해야 합니다.  
그렇지 않으면 Teacher가 *불필요한 크롭을 항상 선택*하여 데이터가 과도하게 타이트해지고, Student가 **과도 줌/과도 크롭** 성향으로 학습됩니다.

- **Free-form(AR 비조건부)**  
  - `b_full = [0,0,1,1]` (원본 전체) 를 **항상 후보에 포함**  
  - 이후 스코어링에서 `b_full`이 **Top-1이 될 수 있어야 함**(= “no-crop” 허용)

- **AR-conditioned(AR 조건부)**  
  - 원본 이미지 AR: `ar_img = W/H`  
  - 목표 AR: `ar_t = w_r / h_r`  
  - (1) **원본 전체 후보(조건부 포함)**  
    - `|ar_img - ar_t| <= eps_ar` (예: `0.01`) 인 경우에만 `b_full=[0,0,1,1]`을 유효 후보로 포함  
  - (2) **최대면적 후보(항상 포함; minimal crop baseline)**  
    - 목표 AR을 만족하면서 이미지 밖으로 나가지 않는 “가장 큰 윈도우”를 baseline으로 정의  
    - (가장 단순) *center crop*:
      - if `ar_img >= ar_t` (이미지가 더 가로로 넓음) → **높이 100% 유지, 좌우만 자름**  
        - `bw = ar_t / ar_img`  
        - `b_maxarea_center = [ (1-bw)/2, 0, (1+bw)/2, 1 ]`
      - else (이미지가 더 세로로 김) → **너비 100% 유지, 상하만 자름**  
        - `bh = ar_img / ar_t`  
        - `b_maxarea_center = [ 0, (1-bh)/2, 1, (1+bh)/2 ]`
  - (3) **최대면적 + 주체/카피스페이스 정렬 후보(권장; 3~5개만 추가해도 큰 효과)**  
    - `b_maxarea_center`를 crop 축 방향으로 “슬라이드”하여 **주 피사체 중심** 또는 **copy-space side**를 보존  
    - 슬랙:
      - if `ar_img >= ar_t`: `slack_x = 1 - bw`, window width = `bw`
      - else: `slack_y = 1 - bh`, window height = `bh`
    - 원하는 중심 `c_des` (기본: subject centroid, 옵션: thirds 정렬점)로 정렬:
      - if width-crop: `x1 = clamp(c_des_x - bw/2, 0, slack_x)`, `x2=x1+bw`
      - if height-crop: `y1 = clamp(c_des_y - bh/2, 0, slack_y)`, `y2=y1+bh`
    - offset 템플릿(초기): `[-0.25, 0, +0.25] * slack` (3개)만 추가해도 “중심 고정 강제” 편향이 크게 줄어듭니다.

> 구현 팁: baseline 후보(`b_full`, `b_maxarea_*`)는 **Cheap prune 전에 미리 candidate list에 prepend**하고, Qwen에 전달하는 Top-M 후보에도 **항상 포함**시키세요(“원본이 정답” 판단이 가능해짐).


- **Grid anchor + multi-scale**(기본): center를 그리드에 두고 scale_set으로 박스 생성
- **AR sliding windows**(보강): 특정 AR에서 stride 기반 추가 후보
- **Object-centered**(주체 보존 강화): main subject bbox 중심 + margin 템플릿(1.1/1.25/1.45×) 생성
- **Copy-space candidates**: copy_space_side를 보존하는 구도(주체 한쪽 배치)
- **Text-preserve candidates(doc_poster)**: OCR box 중심/포함을 우선




##### 8.2.0a (v1.9) 공개 Cropping Teacher “proposal 주입” (free‑form seed) — Grid/룰의 blind spot 보완

v1.8에서 `R_teach`(teacher consensus)를 **약한 prior**로 추가했지만, 더 큰 품질 개선폭은 보통 “스코어 항 추가”보다 **후보 공간(candidate space) 자체의 개선**에서 먼저 나옵니다.  
즉, *좋은 크롭이 후보에 없으면* 스코어러/Teacher(Qwen)가 아무리 좋아도 뽑을 수 없습니다.

따라서 v1.9에서는 공개 레포의 사전학습 크롭 모델을 **“Score teacher”가 아니라 “Proposal teacher(후보 생성 보강)”**로 먼저 사용합니다.

---

###### (1) 1차 도입 대상(권장 3종) — “바로 PoC 가능한 레포” 기준
아래 3종은 **(a) pretrained weight 제공, (b) demo inference 코드 존재, (c) bbox 출력이 명확**해서 “proposal 주입만으로 개선폭”을 측정하기에 적합합니다.

- **GAIC‑Pytorch (Grid Anchor 기반, AR 지정 가능)**  
  - 장점: AR‑conditioned crop을 직접 뽑을 수 있어 projection 필요성이 낮음. 임의 AR도 지원.
- **CACNet‑Pytorch (Photographer 스타일, 앵커포인트 회귀 기반)**  
  - 장점: “정석 구도”에 강한 경향. 인물/제품에서 안정적 proposal을 주는 경우가 많음.
- **CGS‑Pytorch (후보 간 관계를 이용한 scoring / crop 선택)**  
  - 장점: 단일 객체 타이트닝 대신 “관계/균형”을 반영한 crop이 나오는 편(데이터/장면에 따라).

> 운영 원칙: **초기에는 “proposal 주입만”**(= scoring/프롬프트/게이팅 고정)으로 개선폭을 분리 측정하고,  
> 유효성이 확인되면 그때 `R_teach`(consensus prior)까지 단계적으로 추가합니다.

---

###### (2) Proposal 주입 파이프라인(권장; AR‑조건부 Top‑K 라벨 생성용)

**입력:** 이미지 \(I\), target AR 집합 \(\mathcal{R}\), baseline 후보 \(\mathcal{B}^{grid}_r\)  
**추가 입력:** teacher set \(\mathcal{T}=\{t_1,t_2,t_3\}\)

**출력:** 최종 후보 \(\mathcal{B}_r=\mathcal{B}^{grid}_r \cup \mathcal{B}^{base}_r \cup \mathcal{B}^{teach\_proj+jitter}_r\)

구성 단계:

1) **Teacher proposal 생성 (free‑form 또는 AR‑specific)**
- 각 teacher \(t\)에 대해:
  - (가능하면) \(r\in\mathcal{R}\)별로 crop을 직접 예측: \(\hat b^{t}_r\)
  - (그렇지 않으면) free‑form crop \(\hat b^{t}_{ff}\) 1개(또는 Top‑K)만 생성

2) **AR Projection (free‑form → target AR)**
- teacher가 free‑form만 주는 경우, 이를 각 target AR로 사상(projection)하여 후보로 만든다.
- 기본 원칙: **가능하면 “확장(expand)”을 우선**, 불가하면 “축소(shrink)”  
  (확장은 컨텍스트 보존에 유리, 축소는 타이트닝 편향을 강화할 수 있음)

**AR projection 의사코드(중심 보존 + 경계 clamp):**
```python
def project_box_to_ar(b, ar_t, prefer_expand=True):
    # b: [x1,y1,x2,y2] normalized in [0,1]
    x1,y1,x2,y2 = b
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = (x2-x1), (y2-y1)
    ar = w / max(h, 1e-6)

    if abs(ar - ar_t) < 1e-3:
        return b

    if ar < ar_t:
        # too tall → need wider OR shorter
        if prefer_expand:
            w2 = min(h * ar_t, 1.0)
            h2 = w2 / ar_t
        else:
            h2 = min(w / ar_t, 1.0)
            w2 = h2 * ar_t
    else:
        # too wide → need taller OR narrower
        if prefer_expand:
            h2 = min(w / ar_t, 1.0)
            w2 = h2 * ar_t
        else:
            w2 = min(h * ar_t, 1.0)
            h2 = w2 / ar_t

    x1n, x2n = cx - w2/2, cx + w2/2
    y1n, y2n = cy - h2/2, cy + h2/2

    # clamp by shifting (preserve size as much as possible)
    dx = 0.0
    if x1n < 0: dx = -x1n
    if x2n > 1: dx = 1 - x2n
    x1n, x2n = x1n + dx, x2n + dx

    dy = 0.0
    if y1n < 0: dy = -y1n
    if y2n > 1: dy = 1 - y2n
    y1n, y2n = y1n + dy, y2n + dy

    # final clamp (rare)
    x1n, y1n = max(0.0, x1n), max(0.0, y1n)
    x2n, y2n = min(1.0, x2n), min(1.0, y2n)
    return [x1n,y1n,x2n,y2n]
```

3) **Local Jitter Neighborhood (seed 주변에서만 미세 탐색)**
- projection된 seed \(\tilde b_r^t\) 주변에 shift/scale을 주어 \(K_j\)개의 근방 후보를 추가:
  - shift: \(\Delta x,\Delta y \in \{-\alpha,0,+\alpha\}\) (권장 \(\alpha=0.03\sim0.06\))
  - scale: \(s \in \{0.92, 1.00, 1.08\}\)
- AR은 target AR을 유지하도록 재‑projection(또는 width/height 동시 스케일)한다.
- 보통 teacher seed 1개당 **9~27개** 정도면 충분(quality-first는 27까지 허용).

4) **Dedupe / NMS**
- teacher 후보는 grid 후보와 중복될 수 있으므로 IoU 기준으로 제거:
  - `NMS_IOU_TEACH = 0.95` (거의 동일 박스만 제거)
- 후보의 `source`를 반드시 기록:
  - `source = "grid" | "baseline" | "teacher:gaic" | "teacher:cacnet" | "teacher:cgs" | "teacher:jitter"`

5) **저장 필드(운영/디버깅 필수)**
- `teacher_candidates[]`: teacher별 seed/proj/jitter 후보 및 meta
  - `teacher_id`, `stage = seed|proj|jitter`, `bbox_norm`, `teacher_score(optional)`
- `iou_to_teacher_top1`: 최종 선택이 teacher seed와 얼마나 가까운지(후보 생성 성능의 직접 지표)
- `proposal_injected = True/False`

---

###### (3) “proposal 주입만” PoC에서의 핵심 지표(후보 공간 개선 확인)
- `candidate_recall@GT(τ)` (GoldenSet 기준):  
  \[
  \text{Recall@GT}=\Pr\left[\max_{b\in\mathcal{B}_r}\mathrm{IoU}(b,b_{GT}) \ge \tau\right],\ \tau\in\{0.7,0.8\}
  \]
  - 이 값이 오르면 “스코어러가 아니라 후보 공간이 좋아졌다”는 강한 증거
- `oracle_top1_iou`: 후보 중 GT에 가장 가까운 IoU 평균(upper bound)
- `proposal_use_rate`: 최종 Top‑1이 teacher 계열 후보(source=teacher*)에서 나온 비율  
  (너무 높으면 teacher 스타일 과의존 가능 → 게이팅/다양성/스코어 조정)

> 결론: v1.9에서 teacher proposal은 “최종 답을 강제”하지 않습니다.  
> **후보 다양성을 넓히고, 선택은 기존 Scorer+Qwen+QA가 계속 담당**합니다.


##### 8.2.1 Grid Anchor(기본) - AR-조건부 “전역 커버리지” 확보
- v1.6 기본 후보는 **Grid Anchor + multi-scale** 조합:
  - 중심점 grid: \(M\times N\) (권장 `M=N=12`)
  - scale(area) set: `scale_set` (권장 0.25~0.95, 8~10개)
  - 각 후보는 목표 AR을 만족하도록 \((w,h)\)를 결정하고, 이미지 밖으로 나가는 후보는 제외
- 후보 수 폭증 방지:
  - `nms_iou=0.90`로 near-duplicate 제거
  - `max_k=240` 같은 상한으로 제한 + diversity sampling(k-means)

##### 8.2.2 **Saliency-Guided Jittering(권장)** - “Grid 사이 간극”에서 좋은 구도 누락 방지
Gemini 리뷰에서 지적한 것처럼, 순수 grid 후보는 피사체가 grid 사이에 걸리면 좋은 구도를 놓칠 수 있습니다.  
따라서 **saliency/segmentation centroid \(c_s\)** 주변으로 **미세 이동(jitter)** 후보를 추가합니다.

- 입력: subject centroid \(c_s=(c_x,c_y)\), subject bbox size \(w_s,h_s\)
- offset set(권장):  
  - \(\Delta x \in \{0,\pm0.03,\pm0.06\}\cdot w_s\)  
  - \(\Delta y \in \{0,\pm0.03,\pm0.06\}\cdot h_s\)
- 적용 방법:
  - (A) grid 후보의 중심 \((c_x,c_y)\)를 위 오프셋으로 이동한 후보를 추가
  - (B) 또는 `b_maxarea_subject`(최대면적+주체정렬 baseline)만 jitter하여 **후보 수를 억제**

##### 8.2.3 (선택) Phi-Grid/Thirds-Target 후보 - “황금비율/3분할”을 후보 단계에서 보장
- portrait/lifestyle 트랙에서 구도 prior를 더 강하게 넣고 싶다면,
  - 중심점 후보를 **3분할 교차점(4개)** 또는 **phi-grid 포인트** 근처로 강제하는 후보를 소수(예: 8~16개/AR) 추가
- horizon이 있는 landscape 트랙이라면,
  - horizon y를 1/3 또는 2/3에 맞추도록 “상하 슬라이드” 후보를 추가(cheap score에서 강하게 선택됨)

##### 8.2.4 (v1.8) 공개 Cropping 모델 Teacher Proposal 주입 (강력 추천)

Grid/규칙 후보만으로는 “연구 SOTA가 잡는 미묘한 구도”를 놓치는 경우가 많습니다.  
특히 Shutterstock에는 **프로가 이미 잘 프레이밍한 원본**도 많아, 데이터가 “과도 크롭”으로 치우치지 않게 하려면 **학습된 크롭 모델의 제안**을 후보 풀에 넣어두는 것이 효과적입니다.

**아이디어:** 오픈 레포에서 제공하는 사전학습 크롭 모델을 여러 개 실행하여, 각 모델이 제안한 crop box를 `candidate`에 추가합니다.  
이 후보들은 이후 Cheap/Expensive Score와 VLM Candidate-Pick에서 동일하게 경쟁합니다(=Teacher가 “좌표를 생성”하지 않고도 SOTA 구도를 고려 가능).

- 추천 Teacher 후보군(예시)
  - **GAIC Grid‑Anchor** (anchor 기반, 빠름, pretrained 제공)
  - **CACNet** (“Composing Photos Like a Photographer”; composition map 기반)
  - **S2CNet** (Spatial‑Semantic Collaborative Cropping; complex scene에 강함)
  - **A2‑RL** (Aspect‑aware RL cropping; AR‑conditioned에 특히 유용)
  - **CGS** (Composing Good Shots; ranking 기반)
  - **UNIC** (unbounded composition; view adjustment 라벨/검증용)

**운영 규칙**
1) Teacher crop은 보통 free-form이므로, target AR이 필요한 경우 `project_to_ar()`로 보정합니다.  
   - `mode="expand_then_clip"`: teacher crop을 최대한 보존하며 AR 맞추기(권장)  
   - `mode="center_inside"`: teacher crop 내부에서 AR center-crop(보수적)  
2) Teacher 후보는 AR당 1~3개만 추가(후보 폭증 방지).  
3) Teacher 후보에는 `source`와 `teacher_conf`를 메타로 저장하여 QA에서 추적합니다.

**의사코드**
```python
def inject_teacher_candidates(img, ar, cand_list):
    teacher_boxes = []
    for teacher in TEACHER_CROP_MODELS:   # ["gaic", "cacnet", "s2cnet", "a2rl", "cgs"]
        b0, conf = teacher.predict(img, ar=ar)  # teacher에 따라 ar 미지원일 수 있음
        b1 = project_to_ar(b0, ar, mode="expand_then_clip")
        teacher_boxes.append({"bbox": b1, "conf": conf, "source": f"teacher:{teacher.name}"})

    # dedupe vs existing candidates
    cand_list = add_and_nms(cand_list, teacher_boxes, iou=0.90)
    return cand_list
```

> **팁:** Teacher 후보 주입은 “전수”로 해도 되지만, 계산량이 크면  
> (a) `hard-case`(multi-subject, heavy text, low comp confidence)나  
> (b) `Golden/Calibration`에만 먼저 적용해도 효과를 빠르게 확인할 수 있습니다.

#### 8.3 후보 생성 의사코드(AR-조건부, 구현 가능)

```python
def generate_candidates_ar(
    target_ar: float,
    M: int = 12, N: int = 12,
    scale_set=(0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95),
    a_min: float = 0.15, a_max: float = 0.95,
    ar_tol: float = 0.02,
    max_k: int = 240,
    nms_iou: float = 0.90,
    # --- v1.6 additions ---
    saliency_centroid=None,   # (cx, cy) in [0,1] from seg/saliency
    subj_size=None,           # (ws, hs) in [0,1] from subj bbox
    jitter_fracs=(0.00, 0.03, 0.06),
    jitter_scales=(0.45, 0.55, 0.65),
    use_phi_thirds: bool = True,
):
    # Return: list of candidate boxes [x1,y1,x2,y2] in normalized coords.
    # Note: baseline candidates (b_full / b_maxarea_center / b_maxarea_subject) should be
    #       added outside this function as "must-include" candidates.

    xs = [i / M for i in range(M + 1)]
    ys = [j / N for j in range(N + 1)]
    cand = []

    def add_box(cx, cy, area_t):
        bw = (area_t * target_ar) ** 0.5
        bh = (area_t / target_ar) ** 0.5
        x1 = cx - bw / 2; y1 = cy - bh / 2
        x2 = cx + bw / 2; y2 = cy + bh / 2
        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
            return
        ar = (x2 - x1) / (y2 - y1)
        if abs(ar - target_ar) > ar_tol:
            return
        cand.append([x1, y1, x2, y2])

    # (1) global grid anchors
    for area_t in scale_set:
        if not (a_min <= area_t <= a_max):
            continue
        for cx in xs:
            for cy in ys:
                add_box(cx, cy, area_t)

    # (2) saliency-guided jittering around main subject
    if saliency_centroid is not None and subj_size is not None:
        cx0, cy0 = saliency_centroid
        ws, hs = subj_size
        dxs = [s * f * ws for f in jitter_fracs for s in (-1, 1)] + [0.0]
        dys = [s * f * hs for f in jitter_fracs for s in (-1, 1)] + [0.0]
        for area_t in jitter_scales:
            if not (a_min <= area_t <= a_max):
                continue
            for dx in dxs:
                for dy in dys:
                    add_box(cx0 + dx, cy0 + dy, area_t)

    # (3) (optional) phi-grid / thirds targeted centers
    if use_phi_thirds:
        third = [1/3, 2/3]
        phi = [0.382, 0.618]
        centers = [(x, y) for x in third + phi for y in third + phi]
        for area_t in (0.35, 0.55, 0.75):
            for cx, cy in centers:
                add_box(cx, cy, area_t)

    cand = nms_boxes(cand, iou_thr=nms_iou)

    if len(cand) > max_k:
        cand = diversity_sample(cand, k=max_k)  # e.g., k-means on (cx,cy,logw,logh)

    return cand
```

#### 8.4 Candidate ID 규칙
- `"{ar}_g{M}x{N}_s{scale_idx}_i{seq:04d}"`
- 예: `4x3_g12x12_s3_i0087`
- 파라미터 해시(`candidate_gen_hash`)를 레코드에 저장

#### 8.5 (v1.9+) Candidate Generator 멀티프로세싱 가속(구현)

`generate_candidates.py`의 병목은 per-image `build_output_record(...)` 반복 구간입니다.  
v1.9+ 구현에서는 해당 구간을 **멀티프로세싱(ProcessPoolExecutor)** 으로 병렬 처리하도록 확장했습니다.

- 결과 결정성:
  - 입력 순서를 유지하는 `executor.map(...)` 사용으로 출력 JSONL 순서를 고정
  - `num_workers=1`이면 기존 단일 프로세스 경로와 동일 결과
- Worker 공유 상태:
  - `c2/c3/teacher/actual_size_map`은 worker initializer에서 1회 주입
  - Linux 권장 start method는 `fork` (copy-on-write 메모리 효율)

신규 CLI 옵션 (`src/generate_candidates.py`):

- `--num_workers INT`
  - `0`: auto (권장)
  - `1`: single-process
  - `N>=2`: 멀티프로세스 worker 수 고정
- `--mp_chunksize INT` (default: 64)
  - worker map 청크 크기
- `--mp_start_method {auto|fork|forkserver|spawn}` (default: `auto`)
  - 대용량 맵 공유 효율을 위해 Linux에서는 `fork` 권장

E2E 파이프라인 전달 옵션 (`run_phaseA_to_teacher_e2e.sh`):

- `--cand_num_workers`
- `--cand_mp_chunksize`
- `--cand_mp_start_method`

권장 시작값(서버, Linux):

```bash
--cand_num_workers 0 \
--cand_mp_chunksize 64 \
--cand_mp_start_method fork
```

---


### 9) Teacher Scorer 설계 (Cheap → Expensive → Top-K Diversity)

> 목표: “미학>피사체>텍스트” 우선순위를 반영하되, **치명적 절단**은 Hard reject로 차단.

#### 9.0 (v1.6) **Good Composition Closed-Loop** - “구도 규칙 → 피처 → 스코어 → QA”가 완전히 닫히는 설계

아래 표는 **사진학적 구도 규칙**이 Stage-2 Feature, Cheap/Expensive Score, Verify/QA로 **일관되게 연결**되도록 고정한 “운영 매트릭스”입니다.

| Composition Rule | meta routing(추천) | Feature(Stage-2) | Score term(cheap/expensive) | QA metric / gate |
|---|---|---|---|---|
| Rule of Thirds(3분할) | lifestyle/landscape/일반 | subject centroid \(c_s\) | \(R_{\text{third}}\) (또는 \(R_{\text{comp}}\)) | `third_dist_p50/p90` |
| Golden Ratio(Phi grid) | lifestyle/portrait(선택) | \(c_s\) | \(R_{\phi}\) | `phi_dist_p50` |
| Center composition(중앙) | formal_id/corporate/대칭 | symmetry_score, \(c_s\) | \(R_{\text{center}}\) (symmetry 조건부 가중) | `center_selected_when_symmetry_high` |
| Headroom(헤드룸) | portrait | face/pose → \(y_{\text{head}}\) | \(R_{\text{headroom}}\) | `headroom_violation_rate` (shot_type별) |
| Lookroom/Noseroom(시선 여백) | portrait(profile 우선) | gaze/headpose → \(g\), face center | \(R_{\text{lookroom}}\) | `lookroom_violation_rate` (g≠0 subset) |
| Joint cut-off(관절 절단 회피) | people | pose keypoints | \(P_{\text{joint}}\), \(P_{\text{cutoff}}\) (hard/soft) | `joint_cut_rate` (목/무릎/발목/손목) |
| Horizon on Thirds(수평선 위치) | landscape/seascape | horizon(y,θ,conf) | \(R_{\text{horizon\_y}}\) | `horizon_third_dist`, `horizon_detect_rate` |
| Leveling/Roll(수평 맞춤) | landscape/architecture | roll_deg | \(P_{\text{roll}}\) (QA 중심) | `roll_violation_rate` (|roll|>θ) |
| Visual Balance(배경/맥락) | non-packshot | subject area ratio, caption tokens | \(R_{\text{context}}\) | `context_loss_rate` (too tight/too loose) |
| Copy-space 의도 보존 | has_copy_space | saliency density, blank ratio | \(R_{\text{copyspace}}\) | `copyspace_preserve_rate` |
| Text safety(텍스트 보존) | text-heavy | OCR boxes | \(P_{\text{text}}\) | `text_keep_ratio` |
| PICD 24-class(구도 유형: 대각/삼각/곡선/방사/원근/패턴/밀집/흩뿌림 등) | intent-aware(선택) | comp-encoder(`comp24_logits/emb`), line/edge(선택), vanishing/pattern(선택) | \(R_{\text{picd-preserve}}/R_{\text{picd-target}}\) | `CDA@PICD`, `comp24_preserve_rate`, `comp24_entropy`, `CDA@semantic_interference` |

> **운영 원칙**  
> - “룰”은 **피처로 측정 가능해야** 하고,  
> - “피처”는 **스코어/QA에서 동일한 정의**로 쓰여야 하며,  
> - “QA 게이트”는 데이터 릴리즈(버전업)의 **stop/go 기준**으로 강제합니다.


#### 9.1 Hard Constraints (정책 매트릭스)
위반 시 후보 \(b\)는 즉시 reject (score = -∞):

- **Face hard rule**: 얼굴 bbox \(\{F_k\}\) 존재 시, 모든 \(F_k \subset b\)
- **Severe keypoint cut rule(사람)**: keypoint가 경계에 너무 근접하면 reject 또는 강한 패널티
  \[
  m = \alpha \cdot \min(w_b,h_b), \quad \alpha=0.02\sim0.04
  \]
- **면적 비율**: \(a(b)\in[a_{min},a_{max}]\) (기본 0.15~0.95)
- **AR 오차**: \(|\frac{w_b}{h_b}-r| \le \epsilon_r\) (권장 0.01)

#### 9.2 Cheap Score (빠른 후보 축소: N→M) - v1.6 확장(Headroom/Lookroom/Horizon/Symmetry/Context)

\[
\begin{aligned}
S_{\text{cheap}}(I,b,r)=
&\;\lambda_{\text{cov}}C_{\text{subj}}(I,b)
-\lambda_{\text{cut}}P_{\text{cut}}(I,b)
-\lambda_{\text{text}}P_{\text{text}}(I,b)\\
&+\lambda_{\text{comp}}R_{\text{comp}}(I,b,r)
+\lambda_{\text{hr}}R_{\text{headroom}}(I,b)
+\lambda_{\text{lr}}R_{\text{lookroom}}(I,b)
+\lambda_{\text{sym}}R_{\text{sym}}(I,b)
+\lambda_{\text{ctx}}R_{\text{context}}(I,b)
+\lambda_{\text{cs}}R_{\text{copyspace}}(I,b)
+\lambda_{\text{picd}}R_{\text{picd}}(I,b)
+\lambda_{\text{teach}}R_{\text{teach}}(I,b,r)
\end{aligned}
\]

> 모든 항은 **항상 켜두는 것이 아니라**, meta_norm(§6.6~6.7)에서 추정한 `intent/shot_type/category`에 따라  
> \(\lambda\)를 0으로 두거나(비활성) 작은 값으로 둡니다(soft).

##### (A) Subject coverage / Context ratio
- Subject coverage(주체 포함률):
  \[
  C_{\text{subj}}=\frac{|M\cap b|}{|M|}\quad(\text{mask}) \;\;\text{or}\;\; \frac{|B_s\cap b|}{|B_s|}\quad(\text{bbox})
  \]
- Subject area ratio(주체가 crop 안에서 차지하는 비율):
  \[
  a_{\text{subj}}(b)=\frac{|M\cap b|}{|b|}
  \]
  - `portrait/headshot`에서는 \(a_{\text{subj}}\)가 상대적으로 커도 자연스럽지만,
  - `lifestyle/landscape`에서는 너무 커지면 “배경 스토리”가 사라집니다.

##### (B) Cut-off penalty(신체/얼굴/관절 절단)
\[
P_{\text{cut}}=
\alpha_f \cdot \mathbb{1}[\text{face\_cut}]
+\alpha_j \cdot \sum_{k\in\mathcal{J}}\mathbb{1}[\text{joint}_k\ \text{near edge}]
+\alpha_b \cdot \mathbb{1}[\text{subj\_touch\_border}]
\]
- \(\mathcal{J}=\{\text{neck,knee,ankle,wrist}\}\) (필요 시 elbow/hip 추가)
- `near edge`는 keypoint가 crop 경계로부터 margin(예: 2~4% of crop) 안에 들어오면 true

##### (C) Text safety penalty(OCR 기반)
\[
P_{\text{text}} = 1 - \text{text\_keep\_ratio}(b)
\]
- `copy space`/광고 템플릿 트랙에서는 \(P_{\text{text}}\) 가중치를 키움.

##### (D) Composition prior \(R_{\text{comp}}\) (3분할/황금비율/중앙/수평선)
\[
R_{\text{comp}} = 
w_{\text{third}}R_{\text{third}}+
w_{\phi}R_{\phi}+
w_{\text{center}}(\text{sym})R_{\text{center}}+
w_{\text{hor}}R_{\text{horizon\_y}}
\]

- Rule-of-thirds:
  \[
  R_{\text{third}}=-\min_{t\in T}\|c_s-t\|
  \]
  - \(T\): 3분할 교차점 4개(정규좌표)
- Phi grid(선택):
  \[
  R_{\phi}=-\min_{p\in \Phi}\|c_s-p\|,\quad \Phi=\{0.382,0.618\}^2
  \]
- Center composition:
  \[
  R_{\text{center}}=-\|c_s-(0.5,0.5)\|
  \]
- Horizon on thirds(landscape에서만 활성):
  \[
  R_{\text{horizon\_y}}=-\min\left(|y_h-\tfrac13|,\;|y_h-\tfrac23|\right)
  \]
  - horizon detector가 conf 낮으면 \(w_{\text{hor}}=0\).

> **중요(대칭/중앙구도 살리기)**  
> \(w_{\text{center}}(\text{sym})\)는 symmetry_score가 높을수록 증가시켜 “3분할 점수는 낮지만 중앙이 정답”인 케이스(증명사진/건축 대칭)를 살립니다.

##### (E) Headroom / Lookroom (portrait에서만 강하게)
- \(R_{\text{headroom}}\), \(R_{\text{lookroom}}\) 정의는 §6.7.4~6.7.5 참조  
- meta_norm이 `portrait.shot_type_prior`를 주면 그 파라미터를 사용, 아니면 feature 기반(얼굴비율)으로 shot_type을 후보별 보정

##### (F) Symmetry / Visual balance
- Symmetry reward(간단 버전):
  \[
  R_{\text{sym}}=\text{symmetry\_score}(I_b)\in[0,1]
  \]
  - cheap 단계에서는 full-image symmetry_score를 사용해도 충분(정교한 것은 expensive에서)
- Context preservation(배경/맥락 보존):
  \[
  R_{\text{context}}=-\frac{|a_{\text{subj}}(b)-a_{\text{subj}}^{*}|}{\sigma_a}
  \]
  - \(a_{\text{subj}}^{*}\)는 intent에 따라 다르게:
    - headshot: 0.55~0.75
    - half: 0.40~0.65
    - lifestyle: 0.20~0.45
    - landscape: 0.05~0.25

##### (G) Copy-space 의도 보존(특수태그)
\[
R_{\text{copyspace}}=
\mathbb{1}[\text{has\_copy\_space}]\cdot \text{blank\_ratio}(b)
\]
- blank_ratio는 “낮은 saliency 영역 비율”로 근사(세그/살리언시 기반)

##### (H) PICD 기반 Composition Embedding 항 \(R_{\text{picd}}\) - v1.7 추가

PICD는 **24개 구도 카테고리**로 사진 구도를 정의하고(3분할/중앙/대각/수평/수직/삼각/곡선/방사/원근/패턴/밀집/흩뿌림 등), “구도 임베딩이 실제로 구도를 구분하는지”를 **CDA**로 평가합니다.  
v1.7에서는 Qwen이 구도를 “판정”하기보다, **전용 composition encoder(CompEnc)**가 산출한 임베딩/로짓을 스코어러와 QA에 직접 연결합니다.

- CompEnc 출력:
  \[
  e(I)\in\mathbb{R}^d,\quad p(I)=\text{softmax}(g(I))\in\mathbb{R}^{24}
  \]
  \[
  e(I_b),\quad p(I_b)
  \]

- **보존(prior) 목적**: 원본의 구도 타입을 과도하게 파괴하지 않게(특히 landscape/copy-space/architecture)
  \[
  R_{\text{picd-preserve}}(b)=\cos(e(I),e(I_b))-\eta\cdot \mathrm{KL}\big(p(I)\,\|\,p(I_b)\big)
  \]
  - 권장 \(\eta=0.2\sim0.5\)

- **타깃(intent) 목적**: 특정 구도(예: diagonal, symmetry, pattern)를 “의도적으로” 강화하고 싶을 때
  \[
  R_{\text{picd-target}}(b)=p(I_b)[c^*]
  \]
  - \(c^*\)는 (a) `intent.composition_target`가 있으면 그 값, (b) 없으면 `argmax p(I)`(=원본 구도)로 둔다.

- 최종:
  \[
  R_{\text{picd}}(b)=\gamma_{\text{keep}}R_{\text{picd-preserve}}(b)+\gamma_{\text{tgt}}R_{\text{picd-target}}(b)
  \]
  - 기본 권장: \(\gamma_{\text{keep}}=1,\ \gamma_{\text{tgt}}=0\) (대부분 “보존”이 우선)
  - intent가 명확할 때만 \(\gamma_{\text{tgt}}>0\)

**Cheap 단계에서의 사용 원칙**
- `R_picd`는 **N→M 후보 축소에서만** 사용(=과도한 계산을 피하면서도 “구도 다양성”을 살림).
- CompEnc가 없는 초기(bootstrap)에는 `R_comp(3분할/중앙/수평선)`만으로 시작하고, CompEnc가 준비되면 \(\lambda_{\text{picd}}\)를 점진적으로 올립니다.

**(중요) Qwen에 의존하지 않는 이유**
- PICD 벤치마크는 MLLM들이 구도 구분(Triplet)에서 **랜덤에 가까운 정확도**를 보이고, **semantic interference**에 취약함을 보고합니다.  
  따라서 v1.7에서는 Qwen을 “설명 생성/체크리스트 검증”으로 제한하고, 구도 타입은 CompEnc + deterministic features를 1차로 사용합니다.

##### 출력(cheap 단계)
- AR별 후보 \(N\rightarrow M\) 축소: **\(M=20\sim40\)** 권장
- **Top-K 다양성**을 위해 동일 후보 반복 시 패널티 또는 거리 기반 re-rank 적용
- (학습 강화) 최종 Top-K 외에도 **점수 40~60대의 ‘애매한 후보’**를 hard negative로 저장(§13 QA에서 추적)

##### (I) Cropping Teacher consensus 보너스 \(R_{\text{teach}}\) - v1.8

오픈 크롭 모델 Teacher들이 제안한 crop들과의 “합의(consensus)”를 **약한 prior**로 사용합니다.  
목표는 **스코어러의 blind spot을 줄이되**, 특정 teacher 스타일에 과적합하지 않도록 **가중치를 작게** 두는 것입니다.

- 입력: target AR \(r\)에 대해 teacher proposal 집합 \(\mathcal{B}^{teach}_r=\{b^{(m)}\}_{m=1}^M\)
- 1차: teacher 근접도
  \[
  \rho(b)=\max_{b^{(m)}\in \mathcal{B}^{teach}_r}\mathrm{IoU}(b, b^{(m)})
  \]
- 2차(선택): teacher 간 pairwise IoU가 \(\ge \tau_c\) 인 쌍이 1개 이상이면 consensus로 간주(권장 \(\tau_c=0.85\)).
- 최종:
  \[
  R_{\text{teach}}(b)=\sigma\left(\frac{\rho(b)-\tau}{\beta}\right)\cdot \mathbb{1}[\text{consensus}]
  \]
  - 권장 \(\tau=0.75,\ \beta=0.05\)  
  - consensus가 없으면 \(R_{\text{teach}}=0\) (teacher disagreement은 오히려 hard-case)

**권장 가중치(초기)**  
- Cheap: \(\lambda_{\text{teach}}=0.05\sim0.15\)  
- Expensive: \(w_{\text{teach}}=0.05\sim0.20\)

**QA 연결:** §13에 `teacher_disagreement_rate`, `teacher_consensus_rate`, `teacher_override_rate`를 추가해 회귀를 감시합니다.


#### 9.3 Expensive Score (정밀 스코어)

##### 9.3.1 Keep-vs-Crop 게이팅(원본/최대면적 우선) - **과도 크롭 방지 핵심**

**문제:** Shutterstock 프로 사진은 이미 “완성 구도”인 경우가 많아, *조금이라도 점수가 높게 나오는 타이트 crop*이 항상 정답이 되면 제품 품질이 망가질 수 있습니다(과도 줌/negative space 파괴/컨텍스트 손실).

**해결:** 각 AR에 대해 “baseline 후보”(`b_base`)를 정의하고, 최상 후보(`b_best`)가 baseline 대비 **충분히 개선될 때만** 크롭을 허용합니다.

- baseline 선택:
  - AR가 맞으면 `b_base = b_full`
  - AR가 다르면 `b_base = b_maxarea_center` 또는 `b_maxarea_subject`

- 최종 스코어(예시):
  \[
  S_{final}(b)=S_{exp}(b) + w_{area}\cdot \log(\text{area}(b)+\epsilon)
  \]
  - `w_area`는 “불필요한 타이트닝”을 억제(특히 landscape/copy-space에서 효과 큼)

- 게이팅 규칙:
  \[
  \Delta = S_{final}(b_{best}) - S_{final}(b_{base})
  \]
  \[
  b^* =
  \begin{cases}
  b_{base} & \text{if } \Delta < \tau_{improve}(cat,intent,r) \\
  b_{best} & \text{otherwise}
  \end{cases}
  \]

- 권장 시작값(예시; GoldenSet으로 반드시 재튜닝)
  - **copy_space / banner / background / panoramic / landscape**: `τ_improve = 0.04 ~ 0.07`
  - **portrait / 사람 단일**: `τ_improve = 0.02 ~ 0.05`
  - **group(다중 얼굴)**: `τ_improve = 0.03 ~ 0.06` + face-coverage hard 강화
  - **product/packshot/isolated**: `τ_improve = 0.00 ~ 0.03` (타이트 crop이 유리한 경우가 많음)

- (v1.8, 선택) **Teacher-consensus 보정**: teacher proposal의 다수결이 `b_base`와 IoU>0.90이면, `τ_improve`를 +0.02 상향(=더 보수적으로 keep)하여 **불필요한 과도 크롭**을 억제합니다.

- 데이터에 반드시 저장할 필드(디버깅/운영에 결정적)
  - `baseline_bbox`, `baseline_score`
  - `delta_improve = Δ`
  - `decision_type = keep_full | minimal_crop | crop`
  - rationale.tags에 `no_crop_needed` / `minimal_crop_preferred` 등을 명시

> 운영 팁: Top-K를 생성할 때도 `b_base`를 **항상 포함**시키면(Top-K 밖으로 밀려나지 않게) Student가 “필요할 때는 덜 자르는 선택지”를 학습할 수 있습니다.


\[
S_{\text{exp}}(I,b,r)=
w_a A(I_b)
+ w_{ca}\cos(E_I(I_b),E_T(T))
+ w_{cov}C_{\text{subj}}
- w_{cut}P_{\text{cut}}
- w_{text}P_{\text{text}}
+ w_{picd}R_{\text{picd}}(b)
+ w_{teach}R_{\text{teach}}(b)
+ w_{edge}R_{\text{edge}}
\]

- \(A(I_b)\): crop aesthetic score
- \(\cos(\cdot)\): crop과 텍스트(meta) 의미 정합성(주제 유실 방지)
- 경계 패널티(SBL 계열):
  \[
  R_{\text{edge}}=-\sum_{side}\max(0, m-d(B_s,\partial b))
  \]

**가중치 초기값(권장)**
- \(w_a=1.0\), \(w_{cut}=2.0\), \(w_{cov}=0.5\), \(w_{edge}=0.5\), \(w_{ca}=0.3\), \(w_{text}=0.2\), \(w_{picd}=0.2\) (보존 우선, Golden로 튜닝; v1.8에서 `w_teach`는 0.05~0.20 범위에서 시작 권장)

#### 9.4 Top-K + Diversity 선택
- 점수 상위부터 greedy 선택
- 다양성 제약:
  \[
  \max_{k\in selected} IoU(b,b_k) < \tau_{div}
  \]
- \(\tau_{div}=0.75\) 권장
- K=5(권장), K=3(초기)

#### 9.5 GoldenSet 기반 스코어러 캘리브레이션
**(A) Pairwise Logistic(Bradley-Terry)**
- 특징 \(\phi(I,b,r)\) → 점수 \(s=f_\theta(\phi)\)
- 선호쌍 손실:
  \[
  \mathcal{L}(\theta)=\sum -\log \sigma(s^+-s^-)
  \]
- \(f_\theta\): 선형(가중치) → GBDT/XGBoost로 고도화(권장)
- 일부 피처(monotonic constraint: cut, coverage)는 단조 제약 가능

**(B) 임계치 최적화**
- 위반률 vs 선호도 Pareto로 coverage 하한/마진 등을 자동 결정

---


### 10) VLM/MLLM Teacher 라벨 생성기 (기본: Qwen2.5‑VL, 플러그인 가능)

#### 10.1 권장 모드: Candidate-Pick(좌표 안정)
- 시스템이 후보 박스와 수치 features를 만들고,
- Qwen은 **Top-K 선택 + 설명 생성 + 검증 플래그**를 출력한다.
- 좌표는 후보에서 **그대로 사용**(Qwen이 수정 금지).

#### 10.2 Stage-1: 메타 정규화(`meta_norm_v1`) - 선택(룰 기반이 기본)
룰 기반이 기본이지만, multi-lingual/잡다한 태그에서 보조로 사용 가능.

**출력 스키마(요약)**:
```json
{
  "schema_version":"meta_norm_v1",
  "image_id":"...",
  "language":"en",
  "category":"person",
  "subcategory":"single_person_portrait",
  "main_subject":{"label":"woman","type":"person","confidence":0.92,
    "evidence":{"tags_top":["woman","portrait"],"noun_phrases_top":["woman"],"notes":"..."}},
  "secondary_subjects":[{"label":"coffee cup","type":"object","confidence":0.63}],
  "multi_subject":false,
  "special_flags":{"copy_space":true,"copy_space_side":"right","portrait":true,"isolated":false,"panoramic":false,"close_up":false,"text_overlay_likely":false},
  "intent":[{"code":"avoid_face_cut","priority":5,"detail":"..."},{"code":"preserve_copy_space","priority":3,"detail":"..."}]
}
```

#### 10.3 Stage-2: 크롭 라벨(`crop_label_v1`) - 핵심

##### 입력 컨텍스트(JSON)

> **중요(원본이 정답 케이스 대응)**: Stage-2 입력 JSON에 아래를 추가하세요.
> - `baseline_candidates`: `b_full`(가능 시), `b_maxarea_center`, `b_maxarea_subject` (각각 candidate_id 포함)
> - `keep_policy`: `tau_improve`, `w_area`, `force_include_baseline_in_topm=true`
> - `baseline_scores`: baseline 후보들의 `score_numeric`/feature를 별도 요약(토큰 절약)

예시(입력 JSON 확장 조각):
```json
{
  "keep_policy": {
    "tau_improve": 0.05,
    "w_area": 0.10,
    "prefer_baseline_when_uncertain": true
  },
  "baseline_candidates": [
    {"candidate_id":"baseline_full","bbox_norm_xyxy":[0,0,1,1]},
    {"candidate_id":"baseline_maxarea_center","bbox_norm_xyxy":[0.125,0,0.875,1]},
    {"candidate_id":"baseline_maxarea_subject","bbox_norm_xyxy":[0.05,0,0.80,1]}
  ]
}
```

Stage-2 **출력 JSON**에는 아래 필드를 추가하는 것을 권장합니다.
- `decision_type`: `"keep_full" | "minimal_crop" | "crop"`
- `delta_improve_vs_baseline`: float
- `baseline_used_candidate_id`: string


- `norm`(정규화), `objects`, `text_boxes`, `candidates`(AR별 TopM 후보 + features)
- hard constraints와 rationale_vocab을 함께 제공



##### rationale.tags 어휘(vocab) & 부여 규칙(권장)
> 목적: “왜 이 크롭인가”를 **일관된 태그/근거**로 기록해, (1) 사용자 설명, (2) 디버깅/회귀테스트, (3) 선호학습에 재사용.

- `rationale.tags`는 아래 vocab 중에서만 선택(LLM 자유서술 금지 권장):
  - `subject_preserved`, `background_context_ok`, `clutter_reduced`
  - `avoid_face_cut`, `avoid_person_cut`, `avoid_object_cut`
  - `copy_space_kept`, `text_kept`
  - `rule_of_thirds`, `centered_subject`, `symmetry`, `leading_lines`, `balanced_negative_space`, `horizon_on_third`
  - `ar_fits_well`, `tight_crop`, `wide_crop`
- **결정적 태깅(권장)**: Qwen이 태그를 “추론”하게 두기보다, 시스템이 numeric feature로 태그를 산출하고 Qwen은 **설명 문장만 생성**(가장 안정적).
  - `rule_of_thirds`: `thirds_distance <= τ_third` (권장 \(τ_{third}=0.18\)) 또는 `R_third >= -0.18`
  - `centered_subject`: `center_distance <= τ_center` (권장 \(τ_{center}=0.15\))
  - `copy_space_kept`: `copy_space=true` AND `empty_ratio(side)=≥ τ_empty` (권장 0.25)
  - `text_kept`: `min(text_keep_ratio)>=0.9`
  - `avoid_face_cut`: `face_cut=false`
- `rationale.evidence`에 최소 포함(권장):
  - `subject_center_in_crop`=[cx,cy], `thirds_distance`, `center_distance`
  - `subject_coverage`, `face_cut/person_cut/text_cut`
  - `kept_object_ids` / `dropped_context_keywords`(가능하면)

##### 출력(JSON) - **좌표/설명/검증 항목 포함 (v1.6)**

Stage-2(Teacher: Qwen2.5-VL)은 “이미지+메타+후보 박스+후보별 요약 피처”를 입력받아,
- **Top-K crop 선택**
- **‘왜 이 크롭인가’ 설명**
- **Composition Checklist(헤드룸/룩룸/수평/대칭/맥락/텍스트/잘림) 검증 결과**
를 **엄격 JSON**으로 출력합니다.

권장 스키마(요약):

```jsonc
{
  "record_id": "uuid",
  "image_id": "sstk_...",
  "target_ar": "1:1",
  "decision_type": "topk_and_explain",
  "selected_topk": [
    {
      "rank": 1,
      "candidate_id": "ar1x1#017",
      "bbox_norm_xyxy": [0.12, 0.05, 0.88, 0.95],
      "why_tags": ["rule_of_thirds", "headroom_ok", "lookroom_ok", "context_preserved"],
      "why_text": "..."
    }
  ],
  "also_considered": [
    {
      "candidate_id": "ar1x1#003",
      "reject_tags": ["joint_cutoff", "lookroom_violation"],
      "reject_text": "..."
    }
  ],
  "composition_checks": {
    "headroom": {"value": 0.07, "target_range": [0.03, 0.12], "pass": true},
    "lookroom":  {"value": 1.52, "target_range": [1.20, 2.50], "pass": true, "gaze_dir": "right"},
    "horizon":   {"y": 0.33, "roll_deg": 1.4, "pass": true, "conf": 0.82},
    "symmetry":  {"value": 0.81, "pass": true},
    "picd_comp": {"full": "LS-Hori3", "crop": "LS-Hori3", "preserve_sim": 0.84, "pass": true},
    "context":   {"subject_area": 0.38, "target_range": [0.20, 0.45], "pass": true},
    "cutoff":    {"joint_cutoff_score": 0.00, "face_cut": false, "pass": true},
    "text":      {"text_keep_ratio": 0.98, "pass": true}
  },
  "original_policy": {
    "original_is_candidate": true,
    "kept_original": false,
    "why_not_original": "..."
  },
  "explanations": {
    "short": "1~2문장",
    "long": "3~6문장(근거 포함, 과장 금지)"
  },
  "validator": {
    "schema_ok": true,
    "numeric_consistency_ok": true,
    "notes": ""
  }
}
```

- **필수 원칙**
  - `bbox_norm_xyxy`는 반드시 **입력 후보 중 하나**를 그대로 사용(새 좌표 생성 금지)
  - `composition_checks.*.value`는 입력으로 제공된 피처를 “복사/요약”하는 형태로만 채움(환각 최소화)
  - `why_tags/reject_tags`는 “룰→피처→스코어→QA” 매트릭스(§9.0)의 용어 집합에서만 선택


#### 10.4 프롬프트 템플릿(JSON only) - v1.6 (Composition Checklist 강제)

> 목적: Qwen2.5-VL이 “감(주관)”으로 점수만 찍지 않도록,  
> **(1) 적용 규칙을 먼저 선언**하고 → **(2) 체크리스트를 통과/실패로 채운 뒤** → **(3) 최종 선택/설명**을 출력하게 만든다.

```text
SYSTEM:
You are a strict JSON generator. Output MUST be a single valid JSON object. No markdown. No extra text.

USER:
[Image]
<image>

[Task]
Select Top-K crops for target aspect ratio and explain "why this crop" with a composition checklist.
You MUST choose from the given candidate boxes. Do NOT invent new coordinates.

[Target AR]
{{target_ar}}

[Normalized metadata (meta_norm)]
- main_subject: {{main_subject}}
- intent: {{intent}}                    # e.g., portrait / landscape / product / copy_space / text_heavy ...
- category: {{category}}
- portrait:
  - shot_type_prior: {{shot_type_prior}} # headshot / half / full / group / unknown
  - portrait_category: {{portrait_category}}
  - flags: {{portrait_flags}}
  - params:
      headroom_range: {{headroom_min}}, {{headroom_target}}, {{headroom_max}}
      lookroom_range: {{lookroom_min}}, {{lookroom_target}}, {{lookroom_max}}
      weights: {{w_headroom}}, {{w_lookroom}}, {{w_center_bias}}
- copy_space: {{has_copy_space}}
- text_heavy: {{text_heavy}}
- multi_subject_hint: {{multi_subject_hint}}

[Candidates]  # each candidate already has numeric features computed from Stage-2
For each candidate, you will receive:
- candidate_id
- bbox_norm_xyxy
- features:
  - subj_coverage, subj_area
  - face_cut, joint_cutoff_score
  - headroom_ratio (if portrait), eye_y (if available)
  - gaze_dir (left/right/center/unknown), lookroom_ratio (if available)
  - horizon_y, roll_deg, horizon_conf (if available)
  - symmetry_score (if available)
  - comp24_full (label/conf), comp24_crop (label/conf), comp_preserve_sim, comp_target_prob (if available)
  - text_keep_ratio (if text detected)
  - clip_img_txt (optional), aesthetic_pre (optional)

{{candidates_json}}

[Original policy]
The FULL original image (no crop) is also a candidate: candidate_id="full_image".
It is allowed to select original if it has the best composition and preserves context.

[Output rules]
1) First, decide which composition rules apply for this sample:
   - portrait -> headroom, lookroom, joint-cutoff are important
   - landscape -> horizon_on_thirds + leveling are important
   - formal_id/corporate or high symmetry -> center composition can override thirds
   - copy_space -> preserve blank area; do not over-tight crop
   - text_heavy -> preserve text boxes

2) Fill composition_checks using ONLY the provided numeric features.
3) Select Top-K (K={{top_k}}). Rank them.
4) Provide why_tags/reject_tags using allowed tags only:
   ["rule_of_thirds","phi_grid","center_comp","symmetry","headroom_ok","headroom_violation",
    "lookroom_ok","lookroom_violation","joint_cutoff","face_cut","horizon_on_third","roll_tilt",
    "context_preserved","context_lost","copy_space_preserved","text_preserved","text_cut",
    "no_crop_needed","crop_improves_comp",
    "diagonal","triangle","s_curve","c_curve","o_curve","radial","perspective","pattern","dense","scatter"]

5) Output JSON strictly following the schema in the spec:
- selected_topk[0] must be the best candidate.
- also_considered should include 1~3 near-miss hard negatives.

Return JSON only.
```

**운영 팁(환각 방지)**
- 후보별 `features`를 가능한 한 **수치로** 제공하고, Qwen에는 “그 수치를 기반으로 yes/no”만 판단하게 하면 일관성이 크게 올라갑니다.
- `also_considered`에 hard negative를 남기면, 후속 학습에서 contrastive/ordering supervision이 쉬워집니다.


#### 10.5 Self-Critique 2nd pass(하드케이스만)
- `overall_confidence < 0.55` 또는 `failure_modes != []`인 샘플만 2nd pass 실행
- 목적: 얼굴/텍스트 절단 같은 치명 위반 감소

#### 10.6 Post-validation & retry
- JSON parse 실패 / schema 불일치 / candidate_id 미존재 / AR 위반 → 최대 2회 재시도
- 실패 시 fallback: numeric scorer top1 + low_confidence 플래그

#### 10.7 대규모 비용/품질 최적화
- AR별 후보를 Qwen에 **TopM=10~20**만 전달(그 이상은 토큰 폭발)
- **한 호출에 여러 AR** 동시 처리(이미지 인코딩 1회)
- 캐시 키: `(image_id, candidate_gen_hash, scorer_hash, prompt_version)`

---


### 11) UNIC 스타일 View Adjustment 라벨 생성(Action Head)

#### 11.0 “STOP(변경 없음)” 라벨을 충분히 넣어야 하는 이유(원본이 정답 ↔ 뷰 조정 없음)

카메라 프리뷰/오토프레이밍의 실제 UX에서는 **‘움직이지 않는 것’**이 매우 중요한 정답입니다.  
따라서 view adjustment 데이터셋에서는 반드시 다음을 보장해야 합니다.

- `decision_type in {keep_full, minimal_crop}` 인 샘플에 대해,
  - `v_init`을 `v_target`과 거의 동일하게 샘플링하는 케이스를 일정 비율 포함(예: 20~40%)
  - 이 경우 `action_discrete.primary="STOP"` / 회귀도 `dx,dy,dlog_scale≈0` 라벨을 생성
- 이렇게 해야 Student가 불필요한 좌/우 이동이나 줌 드리프트를 학습하지 않습니다.



#### 11.1 v_init 샘플링(재현 규칙)
- 카메라 AR 고정(예: 4:3, 3:4)
- 원본에서 v_init 랜덤 샘플하되
  - 크기 제한: `size >= alpha` (alpha=0.70 권장)
  - 타깃과 IoU 제한: `IoU(v_init, v_target) >= beta` (beta=0.70 권장)
- v_target은 “동일 camera AR에서의 최적 crop” 라벨

#### 11.2 회귀 액션 라벨(Δx, Δy, dlog_scale)
```python
def action_regression(v_init, v_target):
    cxi,cyi,wi,hi = box_to_center_scale(v_init)
    cxt,cyt,wt,ht = box_to_center_scale(v_target)
    dx = (cxt - cxi) / wi
    dy = (cyt - cyi) / hi
    dlog_scale = log(wt/wi)
    return dx, dy, dlog_scale
```

#### 11.3 이산 액션 시퀀스(LEFT/RIGHT/UP/DOWN/ZOOM_IN/ZOOM_OUT/STOP)
- greedy로 오차 축을 줄이는 1-step 라벨 시퀀스 생성
- `primary`(첫 액션) + `steps`(teacher forcing용) 저장

---






#### 10.8 (v1.8) Qwen2.5‑VL보다 더 나은 선택지가 있을까? — Open VLM/MLLM Teacher 후보군

v1.7까지는 Qwen2.5‑VL을 “기본 Teacher”로 가정했습니다.  
v1.8에서는 Teacher를 **교체 가능한 플러그인(VLM/MLLM)** 으로 정의하고, 아래 후보군을 **GoldenSet + PICD 기반 AB 하네스**로 검증하여 최적 조합을 선택합니다.

| 후보 | 장점(라벨 생성 관점) | 약점/주의 | 추천 역할 |
|---|---|---|---|
| **Qwen2.5‑VL (3B/7B/72B)** | JSON/좌표/근거 출력이 안정적, instruction-following 강함 | 72B는 추론비용 큼, 일부 구도 용어 환각 가능 → checklist/verify 중심으로 사용 | **Primary Teacher**(candidate pick + checklist + rationale) |
| **InternVL2.5 (예: 78B)** | 멀티모달 벤치 강력, 대형 비전 인코더로 grounding 강함 | JSON 안정성은 모델/서빙옵션에 따라 편차 → strict validator+retry 필요 | **Adjudicator Teacher**(hard-case 2nd opinion) |
| **Molmo (예: 72B)** | open VLM 중 강한 grounding/문서/도구사용 지향(AllenAI) | 출력 포맷 편차 가능 → 스키마 검증 필수 | 2nd opinion, 설명(why) 다양화 |
| **MiniCPM‑V 2.6** | 상대적으로 가벼워 대량 운영 용이, OCR/문서/대화 성능 강점 | 72B급 대비 미학 판단 상한은 낮을 수 있음 | **High-throughput verifier**(checklist, sanity) |
| **Idefics3 (8B)** | 경량, 오픈, 구조화 출력에 비교적 안정적 | 구도/미학 상한은 제한적일 수 있음 | triage(사전검증), 저비용 fallback |
| LLaVA‑OneVision / DeepSeek‑VL2 등 | 최신 오픈 SOTA 후보 | 운영 안정성/JSON 포맷은 사전 검증 필요 | 후보군(AB 하네스에 포함) |

**핵심 결론(실무):** “어떤 모델이 절대적으로 최고”라기보다,  
- **Primary(Qwen‑72B) + Adjudicator(InternVL‑78B 또는 Molmo‑72B)** 2‑teacher 체계가 라벨 품질을 가장 크게 끌어올립니다.  
- Teacher disagreement가 높은 샘플은 GoldenSet/QC로 보내는 것이 장기 품질을 만듭니다.

#### 10.9 (v1.8) Teacher AB 하네스 & Ensemble 운영 규칙

- 입력: 동일한 `(image, target_ar, candidate_set, meta_norm, deterministic_features)`에 대해 여러 Teacher 실행
- 출력: `rank_topk`, `checklist`, `rationale`, `composition_tags`, `confidence`
- 지표(자동):
  - `json_valid_rate` / `bbox_valid_rate`
  - `checklist_consistency` (예: headroom_violation인데 `headroom_ok=True`이면 불일치)
  - `pairwise_stability` (temperature/seed 변화에 따른 rank 변동)
  - `human_pref_corr` (GoldenSet)
  - `comp24_tag_consistency` (PICD subset)
- Ensemble(권장):
  - **rank**: Borda/Condorcet 또는 평균 점수로 Top-K 합의
  - **checklist**: 다수결 + deterministic feature로 override
  - **confidence**: agreement 기반(예: vote entropy) + deterministic violation penalty

> 추천 운영  
> - 전수 라벨링은 Primary 1개로 돌리고,  
> - `confidence < τ` 또는 `teacher_disagreement` high 인 경우에만 Adjudicator를 추가 실행합니다.  
> - “품질 우선”이라면 Adjudicator 비율을 높여도 OK이며, 대신 **GoldenSet 게이트**를 더 빡세게 둡니다.




#### 10.10 (v1.9) Teacher AB 하네스 — “구체 입력/출력 스키마” + 코드 스켈레톤

v1.8의 AB 하네스는 개념 수준이었고, v1.9에서는 **에이전트가 바로 코드로 내릴 수 있게**  
(1) 고정 스키마, (2) 인터페이스, (3) 실행/리포트 산출물까지 명확히 정의합니다.

---

##### 10.10.1 AB 하네스의 목표(정의)
AB 하네스는 “Teacher를 바꾸면 좋아질까?”를 넘어, 아래 3가지를 분리해서 측정합니다.

1) **VLM Teacher 성능 차이**: 동일 후보 set에서 *선택/설명/체크리스트* 품질 비교  
2) **Cropping‑model Teacher proposal 효과**: scoring은 고정하고 *후보 set만* 바꿔서 개선폭 비교  
3) **Ensemble 정책 효과**: disagreement 기반 라우팅/합의 규칙이 label 안정성을 얼마나 올리는지 비교

---

##### 10.10.2 AB Harness Input Schema (teacher_ab_input_v1)

> 단위: 1 레코드 = (image_id, target_ar)  
> 목적: Teacher가 “좌표를 직접 생성”하지 않고 **candidate pick + grounded checklist**를 하도록 강제.

```json
{
  "schema_version": "teacher_ab_input_v1",
  "sample_id": "bigstock_image_123|ar=9:16",
  "image": {
    "image_id": "bigstock_image_123",
    "uri": "s3://bucket/path.jpg",
    "width": 4000,
    "height": 3000
  },
  "target_ar": "9:16",
  "meta": {
    "tags": ["portrait", "woman", "copy space", "..."],
    "caption": "A woman reading a book ...",
    "alt_text": "..."
  },
  "meta_norm": {
    "category": "people",
    "shot_type": "half",
    "main_subject": "person",
    "multi_subject": false,
    "has_copy_space": true,
    "copy_space_side": "right",
    "intent": {
      "primary": "subject_preserve",
      "composition_target": null,
      "avoid_text_cut": false
    }
  },
  "features": {
    "subject": {
      "mask_rle": "...",
      "bbox_norm": [0.12, 0.18, 0.76, 0.95],
      "centroid_norm": [0.42, 0.61],
      "area_ratio": 0.31
    },
    "faces": [
      {"bbox_norm": [0.18,0.22,0.30,0.34], "score": 0.99, "landmarks": {"eye_l":[...],"eye_r":[...],"nose":[...]}}
    ],
    "pose": {
      "person_bbox_norm": [0.12,0.18,0.76,0.95],
      "keypoints17_norm": [[...],[...]],
      "kp_conf": [...]
    },
    "gaze": {"yaw_deg": 18.0, "pitch_deg": -2.0, "conf": 0.72},
    "horizon": {"roll_deg": 1.2, "y_norm": 0.63, "conf": 0.81},
    "ocr": {
      "boxes_norm": [[0.02,0.05,0.31,0.12]],
      "conf": [0.91]
    },
    "composition": {
      "comp24_label_full": "P-RoT",
      "comp24_conf_full": 0.61,
      "emb_full_256": "base64(fp16)",
      "symmetry_score": 0.12
    },
    "clip": {
      "image_emb": "base64(fp16)",
      "text_emb": "base64(fp16)",
      "cos_full": 0.32
    }
  },
  "candidates": [
    {
      "cand_id": "base_full",
      "bbox_norm": [0,0,1,1],
      "source": "baseline",
      "pre": {"area": 1.0}
    },
    {
      "cand_id": "grid_000123",
      "bbox_norm": [0.05,0.12,0.92,0.97],
      "source": "grid",
      "pre": {"area": 0.79, "ar": 0.56}
    },
    {
      "cand_id": "teach_gaic_seed",
      "bbox_norm": [0.10,0.15,0.88,0.95],
      "source": "teacher:gaic",
      "pre": {"teacher_score": 0.71}
    }
  ],
  "policy": {
    "topk": 5,
    "diversity_iou_thr": 0.92,
    "hard_constraints": {
      "min_face_coverage": 0.98,
      "max_roll_deg": 3.0,
      "text_keep_ratio_min": 0.95
    }
  }
}
```

---

##### 10.10.3 AB Harness Output Schema (teacher_ab_output_v1)

```json
{
  "schema_version": "teacher_ab_output_v1",
  "sample_id": "bigstock_image_123|ar=9:16",
  "teacher_stack_id": "baseline+qwen72b",
  "teacher_id": "Qwen2.5-VL-72B",
  "selected_topk": [
    {
      "rank": 1,
      "cand_id": "teach_gaic_seed",
      "bbox_norm": [0.10,0.15,0.88,0.95],
      "final_score": 0.812,
      "rationale_tags": ["subject_preserved","good_headroom","lookroom_ok","rule_of_thirds"],
      "rationale_text": "인물이 잘리지 않고 ...",
      "checklist": {
        "headroom_ok": true,
        "lookroom_ok": true,
        "horizon_level_ok": true,
        "joint_cutoff_ok": true,
        "text_cut_ok": true,
        "context_ok": true
      }
    }
  ],
  "teacher_confidence": 0.74,
  "teacher_disagreement": {
    "n_teachers": 2,
    "vote_entropy": 0.31,
    "top1_iou_min": 0.66
  },
  "validation": {
    "json_valid": true,
    "bbox_valid": true,
    "hard_violation": false,
    "violation_tags": []
  },
  "timing": {
    "latency_ms": 1820,
    "n_candidates_in": 28
  },
  "debug": {
    "raw_response_path": "s3://.../raw.json",
    "seed": 1234
  }
}
```

---

##### 10.10.4 코드 스켈레톤 (Python/Ray)

> 목표: “Teacher를 바꿔도 I/O 포맷이 절대 흔들리지 않게”  
> (스키마 검증 → 실패 시 retry/쿼런틴 → 리포트 자동 생성)

```python
# teacher_ab/schemas.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

BBox = list[float]  # [x1,y1,x2,y2] normalized

@dataclass
class Candidate:
    cand_id: str
    bbox_norm: BBox
    source: str  # "grid"|"baseline"|"teacher:*"
    pre: dict[str, Any]

@dataclass
class TeacherABInput:
    schema_version: Literal["teacher_ab_input_v1"]
    sample_id: str
    image: dict[str, Any]
    target_ar: str
    meta: dict[str, Any]
    meta_norm: dict[str, Any]
    features: dict[str, Any]
    candidates: list[Candidate]
    policy: dict[str, Any]
```

```python
# teacher_ab/teachers/base.py
from typing import Protocol

class ProposalTeacher(Protocol):
    teacher_id: str
    def propose(self, image_bgr, target_ars: list[str]) -> dict[str, list[BBox]]:
        # return proposals per target_ar (sorted best->worst)
        ...

class VLMTeacher(Protocol):
    teacher_id: str
    def pick_and_explain(self, sample: TeacherABInput):
        # select Top-K from candidates + checklist + rationale (JSON-only)
        ...
```

```python
# teacher_ab/candidate_injector.py
def inject_teacher_proposals(
    base_candidates: list[Candidate],
    proposals_by_teacher: dict[str, dict[str, list[BBox]]],
    target_ar: str,
    jitter_cfg: dict,
) -> list[Candidate]:
    # 1) add teacher seeds
    # 2) AR-project if needed
    # 3) jitter neighborhood around seeds
    # 4) IoU dedupe / NMS
    return merged
```

```python
# teacher_ab/run_ab.py (Ray)
import ray

@ray.remote(num_gpus=1)
class ProposalActor:
    def __init__(self, teacher_name: str, ckpt_path: str):
        self.teacher = load_teacher(teacher_name, ckpt_path)

    def run(self, batch_images, ars: list[str]):
        return [self.teacher.propose(img, ars) for img in batch_images]

@ray.remote(num_gpus=1)
class VLMActor:
    def __init__(self, model_name: str):
        self.vlm = load_vlm(model_name)

    def run(self, samples: list[TeacherABInput]):
        return [self.vlm.pick_and_explain(s) for s in samples]

def run_experiment(config):
    # 0) load dataset slice
    # 1) build base candidates (grid + baseline)
    # 2) optional: proposal injection (GAIC/CACNet/CGS)
    # 3) compute cheap/expensive score + pick TopM
    # 4) VLM teacher pick TopK + explain
    # 5) validate + write jsonl + metrics report
    pass
```

---

##### 10.10.5 AB Harness Report (필수 산출물)
- `outputs/<exp_id>/predictions.<teacher_stack_id>.jsonl`
- `outputs/<exp_id>/metrics.<teacher_stack_id>.json`
- `outputs/<exp_id>/diff.<A>_vs_<B>.md` (핵심 KPI 비교)
- `outputs/<exp_id>/pairs_for_human_review.jsonl` (GoldenSet pairwise UI 입력)

AB는 “데이터팩토리의 자동 릴리즈 게이트”와 직결되므로, **리포트 포맷을 고정**시키는 것이 중요합니다.

---

#### 10.11 (v1.9) Mini PoC: “proposal 주입만”으로 개선폭 측정 (2~3개 teacher 선택)

**목표:** scoring/게이팅/프롬프트를 고정한 상태에서,  
**Candidate set에 teacher proposal을 주입하는 것만으로** GoldenSet/QA 지표가 개선되는지 측정.

##### 10.11.1 실험 변형(Variants)
- **V0 Baseline**: grid+baseline 후보만
- **V1 +GAIC**: V0 + GAIC proposal(seed+proj+jitter)
- **V2 +CACNet**: V0 + CACNet proposal(seed+proj+jitter)
- **V3 +CGS**: V0 + CGS proposal(seed+proj+jitter)
- **V4 Ensemble**: V0 + (GAIC∪CACNet∪CGS)

> 원칙: v1차 PoC에서는 `R_teach` 가중치는 **0으로 두고**, 후보 공간 효과만 본다.

##### 10.11.2 데이터 슬라이스(권장)
- Curated Pool에서 **카테고리×샷타입×난이도 층화**로 20k~100k
- GoldenSet(학습 금지) 1k~5k (가능하면 AR 5종 모두 라벨)
- Stress set 1k (edge subject / multi-person / text heavy / diagonal / perspective 등)

##### 10.11.3 평가 지표(“후보 공간이 좋아졌는지”를 먼저 본다)
- **candidate_recall@GT(0.8)**: GT에 가까운 후보가 후보군에 존재하는지(핵심)
- **oracle_top1_iou**: 후보군 상한(upper bound)
- **human win-rate(A vs B)**: GoldenSet pairwise(최소 300쌍/AR)
- violation Δ: face_cut/headroom/lookroom/joint_cut/text_cut/roll 위반률 변화
- `proposal_rescue_rate`: Baseline에서 GT 근접 후보가 없었는데, proposal 주입 후 생긴 비율

##### 10.11.4 합격 기준(예시)
- candidate_recall@GT(0.8) **+2~5%p 이상 상승**(카테고리별 최소 +1%p)
- violation은 악화되지 않거나, 악화 시 원인(특정 teacher/특정 트랙) 명확히 분리 가능
- human win-rate가 통계적으로 유의하게 상승(>55% 수준부터 의미 있음)

---

#### 10.12 (v1.9) “Free‑form 후보 생성”이 별도로 필요한가?

결론부터 말하면, **AR‑조건부 라벨 생성(제품용)**만 본다면  
v1.9의 “grid + (teacher seed + local jitter) + keep/minimal baseline” 조합으로 *대부분 충분*합니다.

다만 아래 조건이면 별도의 free‑form generator(또는 refiner)가 필요합니다.

##### (A) 필요 조건(진단 규칙)
- GoldenSet에서 **candidate_recall@GT가 낮은 카테고리**가 존재  
  - 예: lifestyle(사람+컨텍스트) / framing(문/창틀) / perspective(원근) / pattern(반복)
- “좋은 크롭”이 grid spacing 사이에 자주 숨어서, jitter로도 커버가 안 됨
- teacher proposal이 특정 스타일로 쏠려 diversity가 급격히 떨어짐

##### (B) 추천 해법(별도 모델 훈련 없이도 가능)
1) **Seed‑based Local Search Refiner (score만으로 탐색)**
- Top‑M 후보(cheap 통과)에서 시작해, \((cx,cy,\log s)\) 공간을 작은 step으로 탐색하며 score를 올리는 방향으로 업데이트  
- 장점: 새 모델 없이도 “grid 누락”을 메움, compute는 후보 수×step으로 통제 가능

2) **Free‑form label을 별도 트랙으로 생성(선택)**
- AR을 고정하지 않고 \(r\sim \text{Uniform}(\log r)\)로 샘플링하여 “best aesthetic crop”을 하나 더 저장  
- 이는 향후 “사용자 임의 AR/스토리텔링 크롭” UX 확장에 유리

##### (C) 공개 Cropping teacher만으로 free‑form이 충분한가?
- **대부분의 경우 “seed로는 충분”**합니다(teacher가 좋은 초기점 제공).  
- 하지만 “Top‑K 다양성/의도 컨트롤”까지 고려하면, teacher 출력 1개만으로는 부족할 수 있으므로  
  **(teacher seed + jitter + (선택) local search)**가 가장 안정적인 운영 조합입니다.


## 6) 데이터 스키마, 저장, 패키징


### 12) 데이터 스키마(테이블) & 패키징

#### 12.1 Lake Tables (Delta/Iceberg + Parquet)
- `images`: 메타/클러스터/split/품질
- `features`: CLIP emb, seg/saliency, face/landmarks, pose keypoints, **headpose/gaze**, **horizon/roll**, OCR boxes, symmetry_score, aesthetic_full, **composition_emb/comp24_logits(PICD)**
- `candidates_stage1`: AR별 후보 + cheap score 상위 M (+ `source`/`teacher_conf`/`iou_to_teacher` 저장)
- `crops_pseudo`: AR×Top-K 최종 라벨(+score components + flags + rationale + **composition_checks(headroom/lookroom/horizon/...)** + `teacher_outputs[]` + `teacher_consensus_conf`)
- `golden_labels`(학습 금지): GT bbox + reason tags + pairwise(가능하면)
- `qa_report`: 릴리즈별 지표/분포/위반률
- `manifest`: 빌드 파라미터/코드 해시/버전

#### 12.2 학습 패키징(WebDataset)
- shard 단위 tar: `img.jpg` + `meta.json` + `labels.json`
- cluster 단위 split을 유지하는 방식으로 패키징

---


### 16) 부록: JSON 스키마/프롬프트 파일 구조(권장)

```
/data_factory/
  /schemas/
    meta_norm_v1.json
    crop_label_v1.json
  /prompts/
    meta_norm_v1.txt
    crop_label_candidate_pick_v1.txt
    crop_label_self_critique_v1.txt
  /manifests/
    SS-CropTrain-v1.0.json
```

---




## 7) Verify/QA, 평가 지표 및 Release Gate


### 13) Verify/QA & Release Gate

#### 13.1 필수 QA 지표(릴리즈 조건)

##### (추가) 과도 크롭/과도 줌 감시 지표(원본이 정답 케이스 대응)

- **keep_full_rate**: `target_ar`가 원본 AR과 일치(허용오차 내)일 때 `b_full`이 선택된 비율
- **minimal_crop_rate**: AR 불일치 상황에서 `b_maxarea_*`가 선택된 비율
- **overcrop_rate**: 선택된 `area_ratio`가 `area(b_maxarea_center)` 대비 과도하게 작은 비율  
  - 예: `area(selected) < 0.75 * area(maxarea_center)` 를 overcrop으로 정의(카테고리별 threshold)
- **delta_improve_distribution**: `Δ = S(best)-S(baseline)` 분포(카테고리×AR별)  
  - `Δ`가 0 근처에 몰리는데도 과도 크롭이 많으면 스코어러/게이팅이 잘못된 것
- **STOP_action_rate**(view adjustment): `action=STOP` 비율과 카테고리별 편향(너무 낮으면 드리프트 위험)

이 지표들을 Release Manifest에 자동 기록해, 버전 간 회귀를 즉시 탐지합니다.

- (v1.8) **teacher-consensus QA 지표**
  - `teacher_consensus_rate`: AR별로 teacher 간 합의(consensus)가 성립한 비율(너무 낮으면 teacher pool/서빙 오류 의심)
  - `teacher_disagreement_rate`: teacher 간 top1 IoU < 0.6 등 “강한 불일치” 비율(상위면 hard-case로 채굴)
  - `teacher_disagreement_entropy`: teacher vote entropy 평균(의사결정 불확실성)
  - `teacher_override_rate`: 최종 선택이 teacher consensus / baseline을 얼마나 자주 뒤집는지(과도하면 스코어러/프롬프트 문제)


- (v1.9) **proposal‑injection QA 지표(후보 공간 개선 모니터링)**  
  - `proposal_use_rate`: 최종 Top‑1이 `source=teacher*`에서 나온 비율(과의존 감시)  
  - `proposal_rescue_rate`: Baseline에서 `Recall@GT(0.8)=0` 이었는데 injection 후 `1`로 바뀐 비율(핵심 개선 지표)  
  - `candidate_recall@GT(0.7/0.8)`: GoldenSet 기준 후보군 리콜(= 후보 생성 성능)  
  - `oracle_top1_iou@GT`: 후보군 상한(upper bound) — 스코어러 문제인지 후보 문제인지 분리  
  - `candidate_count_delta`: AR별 후보 수 증가량(비용/품질 trade-off 추적)


- leakage: cluster 기반 split 누수 0
- violation rates (v1.6 확장):
  - face_cut_rate (portrait) ≤ 0.2% 권장
  - **headroom_violation_rate** (portrait, shot_type별) ≤ 1.0% 권장
  - **lookroom_violation_rate** (portrait, g≠0 subset) ≤ 2.0% 권장
  - **joint_cut_rate** (neck/knee/ankle/wrist) ≤ 0.5% 권장
  - **roll_violation_rate** (landscape/architecture, |roll|>3°) ≤ 1.0% 권장
  - **horizon_third_dist_p90** (landscape, horizon_conf>τ subset) ≤ 0.08 권장
  - text_cut_rate (doc_poster) ≤ 0.5% 권장
  - subject_coverage_fail (<0.9) ≤ 3%
  - **copyspace_preserve_rate** (has_copy_space subset) ≥ 0.85 권장
- Top-K diversity: mean IoU(top1,top2) < 0.92 권장
- 카테고리/AR 분포 커버리지

#### 13.2 GoldenSet Gate(제품 승인)
- human preference win-rate(베이스라인 대비) 목표 달성
- hard violation 거의 0
- Top-K utility: Top-K 안에 “전문가 선호급” 포함 비율 ≥ 0.85 권장

---




## 8) 실행 계획 / 로드맵


### 14) 실행 백로그/로드맵(예시)

#### 14.1 Epics(P0)
- A. Data lake & versioning (manifest, split, 누수 리포트)
- B. Curate pipeline (quality filter, balancing, aesthetic cut)
- C. Feature extraction (CLIP/seg/face/kp/OCR)
- D. Candidate generator (AR별 후보 + diversity)
- E. Scorer v1 + calibration (cheap→expensive + Golden 캘리브레이션)
- F. GoldenSet 구축/운영 (샘플링/라벨링 UI/게이트 대시보드)
- G. Dataset build & QA release (v1.0 릴리즈/롤백)

#### 14.2 12주 로드맵(참고; 품질 우선이므로 조정 가능)
- W1-2: 인덱스/중복클러스터/split/manifest
- W3-4: 후보 생성 + cheap score + 50k E2E
- W5-6: GoldenSet v1(2k) + 캘리브레이션(선형 pairwise)
- W7-8: scale-up(500k~2M) + hard-case mining + 게이팅
- W9-10: (옵션) 사람/제품 트랙 강화 + view adjustment 라벨
- W11-12: v1.0 릴리즈 + 운영 자동화

---




## 9) 리스크 & 대응


### 15) 리스크 & 대응

1) **스타일 편향(스톡 룩)** → 카테고리/스타일 균형 + Golden에 제품 UX 케이스 추가  
2) **스코어러 편향** → Golden 기반 캘리브레이션 + monotonic 제약  
3) **누수(유사 이미지)** → cluster split 강제 + 자동 누수 리포트  
4) **텍스트/로고 이슈** → OCR 트랙 분리 + 패널티/감시 유지  
5) **라벨 품질 붕괴** → QA 지표/분포 감시 + hard-case mining + Silver 재판정  

---




## 10) 부록


### (부록) PICD 24-class Composition Taxonomy 치트시트(v1.7)

PICD의 24개 구도 카테고리(약어)는 아래와 같습니다(논문 Figure 기준).  
운영에서는 `comp24_label`(원본/크롭)과 함께, 설명/UX에 쓰기 쉬운 `composition_tags`로 매핑해 저장하는 것을 권장합니다.

| # | PICD label | Abbrev | 추천 `composition_tags`(예시) |
|---:|---|---|---|
| 1 | Single Point RoT | P-RoT | `rule_of_thirds` |
| 2 | Single Shape RoT | S-RoT | `rule_of_thirds` |
| 3 | Centered SinglePoint | P-Cent | `center_comp` |
| 4 | Centered SingleShape | S-Cent | `center_comp` |
| 5 | Diagonal Multi-points | P-Dia | `diagonal` |
| 6 | Diagonal Lines/Shapes | LS-Dia | `diagonal` |
| 7 | Horizontal Arranged Points | P-Hori | `horizontal` |
| 8 | Horizontal Arranged Shapes | S-Hori | `horizontal` |
| 9 | Horizontal ThirdPart | LS-Hori3 | `horizon_on_third` |
| 10 | Horizontal MiddlePart | LS-Hori2 | `horizon_middle` |
| 11 | Vertical Multi-points | P-Ver | `vertical` |
| 12 | Vertical Middle Line | L-Ver2 | `vertical_middle` |
| 13 | Vertical Third Line | L-Ver3 | `vertical_third` |
| 14 | Vertical Multi-lines | L-Ver-mul | `vertical_multiline` |
| 15 | Three Points Triangle | P-Tri | `triangle` |
| 16 | Line/Shape Triangle | LS-Tri | `triangle` |
| 17 | C-Curve | LS-C-Cur | `c_curve` |
| 18 | O-Curve | LS-O-Cur | `o_curve` |
| 19 | S-Curve | LS-S-Cur | `s_curve` |
| 20 | Diffuse | LS-Dif | `diffuse` |
| 21 | Perspective | S-Per | `perspective` |
| 22 | Random-Dense | PL-Den | `dense` |
| 23 | Pattern | PL-Pat | `pattern` |
| 24 | Scatter | P-Scat | `scatter` |

> 주의: 위 매핑은 “설명/UX”용입니다.  
> 학습/QA/분석에는 반드시 원문 `comp24_label`(24-class)도 함께 보관하세요.





### 17) 참고 문헌(키워드)

- GAIC(Grid Anchor, 후보 축소/지표), Rethinking(Set prediction), GenCrop(outpainting + SBL),
  UNIC(view adjustment), Cropper(VLM ICL), InstructCrop(explanation/intent)

- RTMO(CVPR 2024): https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo
- ViTPose / MMPose: https://github.com/open-mmlab/mmpose
- ScaleLSD(CVPR 2025): https://github.com/ant-research/scalelsd
- NG-DSAC Horizon: https://github.com/vislearn/ngdsac_horizon
- Gaze-LLE(official): https://github.com/azerroug/gaze-lle
- Gazelle(gaze target estimation): https://github.com/idiap/gazelle
- L2CS-Net(gaze): https://github.com/Ahmednull/L2CS-Net
- MobileGaze(gaze, 경량): https://github.com/uosie-cvlab/MobileGaze
- 6DRepNet 기반 head pose(estimation): https://github.com/yakhyo/head-pose-estimation


#### (v1.8 추가) Teacher 모델/오픈 레포 참고

- Open VLM/MLLM Teacher 모델카드: Qwen2.5‑VL / InternVL2.5 / Molmo / MiniCPM‑V / Idefics3 / LLaVA‑OneVision / DeepSeek‑VL2
- Open Cropping Teacher 모델(사전학습 가중치 공개): GAIC(Grid‑Anchor) / CACNet / S2CNet / A2‑RL / CGS / UNIC
- 크롭/미학 종합 리스트(코드 링크 포함): Awesome Aesthetic Evaluation & Cropping
