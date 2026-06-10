# SSTK Multi-Mode Training Labels Figure Generation Prompts

작성일: 2026-06-02

대상 문서:

```text
Implement_Docs/SSTK_MultiMode_TrainingLabels_Enhanced_Onboarding_KO_v1_0.md
```

이 문서는 위 온보딩 문서의 이해를 돕기 위해 ChatGPT 이미지 생성으로 추가하거나 교체할 Figure 후보를 정리하고, 각 Figure를 생성하기 위한 영문 프롬프트를 제공한다. 특히 현재 문서에서 텍스트/수식만으로는 한눈에 들어오기 어려운 `Score Components`와 `Mode별 Utility`는 필수 생성 대상으로 둔다.

## 1. 전체 Figure 설계 원칙

### 1.1 공통 시각 스타일

모든 Figure는 같은 보고서에 들어가므로 시각 언어를 통일한다.

- 형태: clean technical infographic, vector-style diagram.
- 배경: white or very light gray background.
- 비율: 16:9 landscape, high resolution, 3840x2160 preferred.
- 색상: restrained enterprise palette.
  - subject/safety: blue
  - composition/placement: green
  - teacher/GAIC: purple
  - hard gate/reject: red or amber
  - output/artifact: dark gray
- 텍스트: 영어 label 중심, 짧고 크게. 작은 문장과 긴 수식은 피한다.
- 톤: engineering onboarding document, not marketing, not cartoon.
- 금지: photorealistic stock photo, decorative gradient orb, busy background, tiny unreadable text, 3D render, comic style.

### 1.2 이미지 생성 후 후처리 권장

ChatGPT image generation은 긴 텍스트와 수식을 틀리게 그릴 수 있다. 따라서 생성 프롬프트는 가능한 label을 짧게 유지하고, 최종 문서 삽입 전 다음을 확인한다.

- key label spelling: `q_subj`, `q_scale`, `q_place`, `q_ar`, `q_ctx`, `q_scene`, `q_head`, `q_look`, `q_group`, `q_teach`.
- `target_ar_only`가 category 변경이 아니라 attributes 축약이라는 점.
- `COCO-format`이 COCO dataset이 아니라 schema라는 점.
- `Checklist`와 `Why-tags`가 같은 것이 아니라 서로 다른 용도라는 점.
- 빨간색은 hard reject나 risk에만 사용했는지.
- 화살표 방향이 실제 pipeline과 맞는지.

### 1.3 공통 프롬프트 Prefix

각 Figure 프롬프트 앞에 아래 prefix를 붙여서 스타일을 고정한다.

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs. Make the layout suitable for a Markdown technical report. Use consistent colors: blue for subject preservation, green for composition and placement, purple for teacher/GAIC signals, amber/red for gates and rejects, dark gray for artifacts and outputs.
```

## 2. 생성 대상 Figure 목록

| ID | 우선순위 | 문서 삽입 위치 | Figure 목적 |
|---|---|---|---|
| F01 | P0 | 7.5 Score Components | score component가 어떤 입력에서 나오고 어떤 score family로 묶이는지 설명 |
| F02 | P0 | 7.6 Mode별 Utility | mode family별 utility 가중치 차이를 heatmap 형태로 설명 |
| F03 | P0 | 9.2 Checklist와 Why-Tag의 차이 | checklist와 why-tag의 역할 차이와 연결 관계 설명 |
| F04 | P1 | 2.2 v14 산출물 구조 | run directory와 주요 artifact의 역할 설명 |
| F05 | P1 | 6.3 Query Open Rule | entity atom에서 mode query가 열리는 정책 설명 |
| F06 | P1 | 6.5 Landscape Candidate Policy | landscape GAIC-Qteach / subject-safe 후보 선택 흐름 설명 |
| F07 | P1 | 7.2-7.3 Score/Gate/Positive Selection | hard gate, utility, threshold, positive/negative selection funnel 설명 |
| F08 | P1 | 8 Label JSON Schema | main JSON, target_ar_only JSON, mode_query_status sidecar 관계 설명 |
| F09 | P2 | 10 QA와 검증 | validation/audit/review pack 전체 QA dashboard 설명 |
| F10 | P2 | 12 v14까지의 보완 히스토리 | v3부터 v14까지 주요 개선 흐름 설명 |

P0 세 개는 반드시 생성하는 것을 권장한다. 현재 문서의 기존 Figure들은 pipeline과 mental model 중심이라, score와 explainability 학습 계약을 설명하는 그림이 부족하다.

## 3. Figure Prompts

## F01. Score Components Map

삽입 위치:

```text
7.5 Score Components 직후
```

생성 목적:

`score_components`가 단순히 많은 숫자의 나열이 아니라, subject preservation, composition, safety, group completeness, teacher score, penalty로 묶이고 최종 `score_mode` / `checklist_scores`에 연결된다는 점을 한눈에 보여준다.

포함해야 할 개념:

- Input: crop candidate + mode query.
- Subject preservation: `core_recall`, `env_recall`, `secondary_recall` -> `q_subj`.
- Scale/axis: `subject_ratio`, `occ_x`, `occ_y` -> `q_scale`, `q_axis`.
- Placement: `anchor_rx`, `anchor_ry` -> `q_place`.
- AR/context/scene: `q_ar`, `q_ctx`, `q_scene`.
- Portrait safety: `q_head`, `q_look`, `face_recall`, `joint_cut_score`, `intrusion_ratio`.
- Group: `group_recall`, `group_balance` -> `q_group`.
- Teacher: GAIC teacher -> `q_teach`.
- Penalties/gates: hard rejects, route penalty, safety penalty.
- Output: `score_mode`, `checklist_scores`, `checklist_labels`, `why_tags`.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs. Make the layout suitable for a Markdown technical report. Use consistent colors: blue for subject preservation, green for composition and placement, purple for teacher/GAIC signals, amber/red for gates and rejects, dark gray for artifacts and outputs.

Title: "Multi-Mode Score Components"

Design a left-to-right architecture diagram. On the far left, draw a simple image crop rectangle labeled "Crop candidate + Mode query". From it, split into six grouped lanes:

1. Blue lane: "Subject preservation" with small boxes "core_recall", "env_recall", "secondary_recall", merging into "q_subj".
2. Blue-green lane: "Scale and axis" with "subject_ratio", "occ_x / occ_y", merging into "q_scale" and "q_axis".
3. Green lane: "Placement" with "anchor_rx / anchor_ry", merging into "q_place".
4. Green-gray lane: "AR, context, scene" with "q_ar", "q_ctx", "q_scene".
5. Amber lane: "Portrait safety" with "q_head", "q_look", "face_recall", "joint_cut_score", "intrusion_ratio".
6. Purple lane: "Teacher signal" with "GAIC teacher" merging into "q_teach".

Add a separate blue group lane: "Group completeness" with "group_recall" and "group_balance" merging into "q_group".

All lanes should flow into a central dark node labeled "Mode utility score U_q(c)". Add a red side gate labeled "Hard rejects: face cut, joint cut, subject cut, AR mismatch" pointing into the utility node as a blocking gate.

On the far right, show three output boxes:
- "score_mode"
- "checklist_scores"
- "threshold labels + why_tags"

Use simple mathematical hints only as small callouts, not full equations: "q_subj = recall mix", "q_group = recall + balance", "q_teach = normalized GAIC". Keep text large and legible. Avoid long sentences.
```

후처리 체크포인트:

- `q_subj`, `q_group`, `q_teach` spelling이 정확해야 한다.
- hard reject가 score component가 아니라 blocking gate로 그려졌는지 확인한다.
- `checklist_labels`와 `why_tags`가 `checklist_scores` 이후에 파생되는 구조인지 확인한다.

## F02. Mode Utility Matrix / Heatmap

삽입 위치:

```text
7.6 Mode별 Utility 직후
```

생성 목적:

mode별 utility 식을 수식으로만 읽으면 어떤 mode가 어떤 component를 중요하게 보는지 파악하기 어렵다. 이 Figure는 mode family별 component weight 차이를 heatmap 또는 matrix로 보여준다.

포함해야 할 개념:

- Row groups:
  - Single person / face
  - Group
  - Landscape
  - Object single
  - Object multi
- Columns:
  - `q_subj`
  - `q_group`
  - `q_scale`
  - `q_place`
  - `q_axis`
  - `q_head`
  - `q_look`
  - `q_ctx`
  - `q_scene`
  - `q_ar`
  - `q_teach`
  - penalties
- Highlight:
  - person/face: head/look/safety important.
  - group: `q_group` important.
  - landscape: `q_teach` dominant, subject-safe only if active.
  - object: subject, placement, context, scene/object recall important.
  - rot modes: `q_place` for thirds/ROT.
  - center modes: center placement.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs. Make the layout suitable for a Markdown technical report. Use consistent colors: blue for subject preservation, green for composition and placement, purple for teacher/GAIC signals, amber/red for gates and rejects, dark gray for artifacts and outputs.

Title: "Mode Utility: What Each Mode Cares About"

Create a matrix-style heatmap with mode families as rows and score components as columns.

Rows:
1. "Single person / Face"
2. "Group"
3. "Landscape"
4. "Object single"
5. "Object multi"

Columns:
"q_subj", "q_group", "q_scale", "q_place", "q_axis", "q_head", "q_look", "q_ctx", "q_scene", "q_ar", "q_teach", "penalties"

Use circle size or heatmap intensity to show importance. Do not write exact numeric weights in every cell. Use a legend: "larger / darker = higher influence".

Highlight key patterns with short callouts:
- "Person: subject + headroom + lookroom + safety"
- "Group: member recall + balance"
- "Landscape: GAIC q_teach first, subject-safe only when reliable"
- "Object: foreground preservation + placement + context"
- "Rot modes boost thirds placement"

Add a bottom caption strip: "Final score = mode-specific weighted utility minus penalties, after hard gates".

Make the heatmap clean and readable, with no crowded text. Use colored column families: blue for subject, green for composition, purple for teacher, red for penalties.
```

후처리 체크포인트:

- landscape row에서 `q_teach`가 가장 강하게 표시되어야 한다.
- group row에서 `q_group`이 강하게 표시되어야 한다.
- person/face row에서 `q_head`, `q_look`, safety/penalties가 눈에 보여야 한다.
- 수식의 정확한 계수까지 이미지에 넣으려 하지 않는다. 계수는 문서 본문이 담당한다.

## F03. Checklist vs Why-Tag

삽입 위치:

```text
9.2 Checklist와 Why-Tag의 차이 직후
```

생성 목적:

처음 보는 엔지니어가 “checklist와 why-tag가 둘 다 설명 label 아닌가?”라고 물을 수 있다. 이 Figure는 checklist가 fixed schema / loss target이고, why-tag가 sparse human-readable summary라는 차이를 보여준다.

포함해야 할 개념:

- `checklist_scores`: continuous regression targets.
- thresholding -> `checklist_labels`.
- `checklist_applicable`: loss mask.
- deterministic summary -> `why_tags`.
- checklist는 dense fixed table, why-tag는 sparse badges.
- 둘은 통합하지 않고 유지한다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs. Make the layout suitable for a Markdown technical report. Use consistent colors: blue for subject preservation, green for composition and placement, purple for teacher/GAIC signals, amber/red for gates and rejects, dark gray for artifacts and outputs.

Title: "Checklist vs Why-Tags"

Create a two-column comparison diagram.

Left column title: "Checklist = structured training targets"
Show a fixed table with rows:
"subject_coverage", "subject_scale", "headroom", "lookroom", "face_cut", "joint_cut", "context", "center_dist", "ar"
Show three small fields beside the table:
"checklist_scores: continuous regression values"
"checklist_labels: threshold-derived labels"
"checklist_applicable: loss mask"

Right column title: "Why-tags = sparse explanation summary"
Show badge-like tags:
"subject_preserved", "headroom_ok", "centered_subject", "ar_fits_well", "avoid_face_cut"
Show a short note: "Only key reasons appear"

Between the columns, draw arrows:
"scores -> thresholds -> labels"
"labels + rejects -> why_tags"

At the bottom, add a clear statement:
"Keep both: checklist trains and audits every axis; why-tags explain the main reasons."

Use large, readable text. Make the checklist look dense and structured, and the why-tags look sparse and human-readable.
```

후처리 체크포인트:

- why-tag가 독립 수동 truth처럼 보이면 안 된다.
- `checklist_scores -> threshold -> checklist_labels -> why_tags` 흐름이 보여야 한다.
- “Keep both” 결론이 명확해야 한다.

## F04. v14 Artifact Map

삽입 위치:

```text
2.2 v14 산출물 구조 직후
```

생성 목적:

run directory 안의 `summary.json`, `mode_query_status.jsonl`, `label_json`, split, explainability summary의 관계를 시각화한다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs. Make the layout suitable for a Markdown technical report.

Title: "v14 Multi-Mode Label Run Directory"

Draw a file-tree style diagram of one run directory. Use grouped boxes:

Top: "<run_dir>"

Group 1: "Run status and summaries"
- status.json
- summary.json
- dataset_stats.json
- validation_summary.json

Group 2: "Label JSON"
- label_json/multimode_labels_full.json
- label_json/multimode_labels_train.json
- label_json/multimode_labels_val.json
- label_json/multimode_labels_target_ar_only_*.json

Group 3: "Query and statistics"
- mode_query_status.jsonl
- mode_stats.csv
- target_ar_stats.csv
- mode_target_ar_stats.csv

Group 4: "Explainability"
- explainability_enrichment_summary.json
- explainability_label_policy.json
- explainability_validation_summary.json

Group 5: "Splits"
- train_image_ids.txt
- val_image_ids.txt
- split_manifest.json

Add a note near target_ar_only: "same annotations, attributes only target_ar".
Add a note near label_json: "COCO-format schema, not COCO images".
```

후처리 체크포인트:

- `target_ar_only` 설명이 명확해야 한다.
- `COCO-format schema, not COCO images`가 보이면 좋다.

## F05. Query Open Rule Diagram

삽입 위치:

```text
6.3 Query Open Rule 직후
```

생성 목적:

entity atom에서 어떤 mode query가 열리고, secondary person / group fallback / scene pseudo-subject gate가 어떻게 작동하는지 보여준다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "From Entity Atoms to Mode Queries"

Create a flow diagram with four input atom types on the left:
1. "Person atom"
2. "Group atom"
3. "Object atom"
4. "Scene atom"

For each atom, draw arrows to opened mode queries:
- Person atom -> single_person_center, single_person_rot, optional face
- Group atom -> group_center, group_rot
- Object atom -> object_single_center, object_single_rot, object_multi_center, object_multi_rot
- Scene atom -> landscape

Add small gate icons:
- Person: "trusted person required"
- Secondary person: "primary-only default; rare exception"
- Group: "trusted multi-person or group_all fallback"
- Object: "foreground evidence required"
- Scene: "low-reliability pseudo-subject neutralized"

On the right, show two possible outcomes for every query:
"Positive annotation" or "No annotation, status saved"

Use blue for person/group, green for object, purple for scene, amber for gates.
```

후처리 체크포인트:

- query open과 positive 생성이 분리되어야 한다.
- secondary person policy가 기본 primary-only임을 나타낸다.

## F06. Landscape GAIC-Qteach Subject-Safe Policy

삽입 위치:

```text
6.5 Landscape Candidate Policy 직후 또는 7.6 landscape utility 설명 직후
```

생성 목적:

landscape mode가 “대부분 minimal crop”으로 보였던 혼동, GAIC cropper를 최대한 따르라는 요구, subject-safe 보정 적용 조건을 하나의 flow로 설명한다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Landscape Policy: GAIC-Qteach First, Subject-Safe When Reliable"

Design a left-to-right decision flow.

Left: "Landscape query candidates"

Step 1: Candidate preference stack:
1. "Direct GAIC teacher for FREE"
2. "GAIC-lineage teacher candidates"
3. "Other teacher fallback"
4. "All candidates fallback"

Step 2: Score:
Large purple node: "q_teach from GAIC"
Small green nodes: "q_scene" and "q_place"

Step 3: Subject-safe branch:
If "reliable subject exists" -> add blue branch "core_recall + q_subj gate, subject cut penalty"
If "low-reliability scene pseudo-subject" -> gray branch "neutralized, no SUBJ overlay, no subject-safe bonus/penalty"

Right output:
"landscape positive" or "no-label if low q_teach / hard reject"

Add a short bottom note:
"Teacher score is primary; subject-safe only prevents reliable subjects from being cut."
```

후처리 체크포인트:

- `q_teach`가 primary로 보여야 한다.
- low-reliability scene pseudo-subject가 neutralized 되는 branch가 있어야 한다.

## F07. Hard Gate, Utility, Positive Selection Funnel

삽입 위치:

```text
7.2 공통 원칙 또는 7.3 Positive와 Negative 직후
```

생성 목적:

utility score와 hard reject가 다르다는 점, positive가 없는 query는 annotation을 만들지 않고 sidecar에 남는다는 점을 설명한다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Score Funnel: Gate First, Then Select Positive"

Draw a funnel from left to right:

1. "Candidate bank"
2. "Mode-specific hard gates"
   - red examples: face cut, joint cut, subject cut, AR mismatch
3. "Utility score U_q(c)"
   - components: subject, composition, teacher, penalties
4. "Threshold tau_pos"
5. "Best positive max 1"
6. "Optional negatives max 2"

Add a side path after hard gates and threshold:
"No positive -> mode_query_status.jsonl only"

Add a final output box:
"label_json annotations: gt_flag=1 positive, gt_flag=0 optional negative"

Make hard gates visibly separate from utility scoring.
```

후처리 체크포인트:

- hard gate가 score 이전 또는 별도 block으로 보여야 한다.
- positive 없음이 label JSON이 아니라 sidecar로 가는 구조가 있어야 한다.

## F08. Label JSON Schema and target_ar_only

삽입 위치:

```text
8. Label JSON Schema 직후 또는 기존 schema map 교체
```

생성 목적:

main label JSON, target_ar_only label JSON, mode_query_status sidecar의 역할 차이를 명확히 한다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "COCO-Format Label JSON: Three Views"

Create three side-by-side panels:

Panel 1: "Main multimode_labels_*.json"
Show top-level boxes:
"images[]"
"annotations[]"
"categories[]"
Inside annotations, show fields:
"bbox xywh", "category_id = mode", "gt_flag", "query_id", "score_mode", "attributes: score + debug + explainability"

Panel 2: "target_ar_only JSON"
Show same images/annotations/categories, but attributes box contains only:
"attributes = { target_ar }"
Add note: "same annotation rows, smaller attributes"

Panel 3: "mode_query_status.jsonl"
Show row list with:
"query_id", "decision", "no_positive_reason", "best_score", "negative_count"
Add note: "records queries with no annotation"

Add a top note:
"COCO-format schema does not mean COCO dataset images."
```

후처리 체크포인트:

- `category_id = mode`가 보여야 한다.
- target_ar_only가 category 변경이 아니라 attributes 축약임이 보여야 한다.

## F09. QA Dashboard

삽입 위치:

```text
10. QA와 검증 직후
```

생성 목적:

QA가 단일 metric이 아니라 query open, positive yield, mode/AR imbalance, hard reject, explainability validation, visual review를 함께 보는 구조임을 설명한다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Multi-Mode Label QA Dashboard"

Draw a dashboard-style diagram with six cards:

1. "Coverage"
   - source images
   - train / val split
   - missing files = 0

2. "Query lifecycle"
   - positive
   - below_tau
   - no_positive
   - no_candidates

3. "Mode and AR balance"
   - mode_stats.csv
   - target_ar_stats.csv
   - mode_target_ar_stats.csv

4. "Safety and intent"
   - face cut
   - joint cut
   - center vs rot
   - object foreground

5. "Explainability validation"
   - schema coverage 1.0
   - missing score components 0
   - final_score mismatch 0

6. "Visual review"
   - crop review packs
   - v3 vs v14 comparisons
   - subject support overlays

Use green check marks for passed validation, amber warning icons for audit areas, and no decorative elements.
```

후처리 체크포인트:

- `final_score mismatch 0`가 포함되면 좋다.
- QA file names가 너무 작거나 길면 후처리에서 overlay한다.

## F10. v3 to v14 Improvement Timeline

삽입 위치:

```text
12. v14까지의 보완 히스토리 직후
```

생성 목적:

v3 baseline에서 v14까지 어떤 문제를 어떤 버전에서 해결했는지 빠르게 이해시키는 timeline이다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "From v3 Baseline to v14 Quality-Gated Labels"

Create a horizontal timeline with six milestones:

1. "v3 Full 10K baseline"
   - first full 9:1 split
   - many noisy positives

2. "v4-v5 Quality gates"
   - trusted face/person
   - object saliency foreground

3. "v6-v8 Secondary policy"
   - primary-only default
   - rare dominant secondary exception

4. "v9 GAIC-Qteach landscape"
   - GAIC teacher priority
   - landscape score uses q_teach

5. "v10-v11 Strict intent"
   - group_all fallback
   - center vs rot hard gates
   - subject-safe landscape

6. "v13-v14 Scene subject neutralized"
   - no false SUBJ overlay
   - low-reliability pseudo-subject disabled
   - explainability attributes added

Use colored icons: red warning for old issues, blue/green fixes, purple teacher signal, gray final artifacts.
```

후처리 체크포인트:

- v14에는 explainability attributes added가 포함되어야 한다.
- v9 landscape가 GAIC teacher priority였다는 점이 보여야 한다.

## 4. 7.5 Score Components 심화 Figure Pack

이 절은 `SSTK_MultiMode_TrainingLabels_Enhanced_Onboarding_KO_v1_0.md`의 `7.5 Score Components`를 직관적으로 설명하기 위한 전용 Figure pack이다. 기존 F01은 전체 map 역할을 하고, 아래 SC-series Figure는 각 component family를 예시 이미지, crop overlay, 수식, score, checklist label, why-tag로 연결한다.

Figure 수에는 제한을 두지 않는다. 오히려 component를 한 장에 모두 넣으면 텍스트가 작아지고 이해가 어려워지므로, 아래처럼 component family별로 쪼개는 편이 낫다.

### 4.1 공통 레이아웃 규칙

각 SC-series Figure는 가능한 한 같은 시각 문법을 쓴다.

- 왼쪽: synthetic example image panel. 실제 SSTK 이미지를 쓰지 않고, 이해를 돕는 pseudo-photo 또는 clean illustrative image로 생성한다.
- 중앙: crop box와 overlay.
  - selected crop: thick white or black rectangle
  - core subject box: solid blue
  - envelope/support box: dashed blue
  - secondary subject/member box: light blue
  - thirds/center grid: green
  - hard reject or cut region: red
  - GAIC/teacher source: purple
- 오른쪽: score card.
  - formula
  - example numeric value
  - derived checklist label
  - emitted why-tags
- 하단: one-line takeaway.

이미지 생성 모델은 수식과 작은 텍스트를 틀릴 수 있다. 따라서 프롬프트에는 수식을 넣되, 최종 보고서 삽입 전 수식/숫자/label은 필요하면 수동 overlay로 보정한다.

### 4.2 Score Component 연결 표

| Figure | Component | 핵심 수식/값 | 연결 checklist | 연결 why-tags |
|---|---|---|---|---|
| SC01 | `q_subj` | `0.55 core + 0.25 env + 0.20 secondary` | `subject_coverage` | `subject_preserved`, `subject_marginal`, `subject_poor` |
| SC02 | `subject_ratio`, `q_scale` | subject area / crop area, Gaussian around mode target | `subject_scale` | `subject_scale_ideal`, `subject_scale_loose`, `subject_scale_tight`, `wide_crop`, `tight_crop` |
| SC03 | `occ_x`, `occ_y`, `q_axis` | subject occupancy along crop axes | indirect via scale/intent QA | `balanced_crop`, `wide_crop`, `tight_crop` |
| SC04 | `anchor_rx`, `anchor_ry`, `q_place` | center or thirds distance | `center_dist`, `third_dist` | `centered_subject`, `rule_of_thirds` |
| SC05 | `q_ar`, `ar_rel_error` | target AR fit | `ar` | `ar_fits_well`, `ar_choice_freeform`, `ar_extreme_penalty` |
| SC06 | `q_ctx`, `context_value`, copyspace | context amount around subject | `context`, `copyspace` | `context_preserved`, `context_loss`, `copy_space_kept`, `copy_space_lost` |
| SC07 | `q_scene` | horizon + symmetry + wide crop | `horizon_state` | `horizon_on_target`, `horizon_off_target`, `needs_leveling` |
| SC08 | `q_head`, `q_look` | headroom and gaze-side lookroom | `headroom`, `lookroom` | `headroom_ok`, `headroom_violation`, `lookroom_ok`, `lookroom_violation` |
| SC09 | `face_recall`, `joint_cut_score`, `intrusion_ratio` | safety and cut checks | `face_cut`, `joint_cut` | `avoid_face_cut`, `avoid_person_cut` |
| SC10 | `group_recall`, `group_balance`, `q_group` | member recall + balance + envelope | group completeness QA | `subject_preserved`, `balanced_crop` |
| SC11 | `object_recall`, `saliency_crop_overlap` | foreground object preservation | `subject_coverage`, object QA | `subject_preserved`, `subject_poor` |
| SC12 | `q_teach` | normalized GAIC teacher score | not direct checklist, used in final score | teacher/GAIC explanation context |
| SC13 | hard rejects / penalties | safety gate and soft penalties | cut/AR/context labels | `ar_extreme_penalty`, `context_loss`, violation tags |
| SC14 | full worked example | score components -> final score -> labels -> tags | multiple checklist fields | compact why-tag summary |

## SC00. Score Component Visual Grammar Legend

삽입 위치:

```text
7.5 Score Components 시작 직후
```

생성 목적:

이어지는 SC-series Figure가 모두 같은 overlay 문법을 쓰도록 legend를 만든다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Score Component Visual Legend"

Create a legend-only figure that explains the visual language used in score component examples.

Show a simple synthetic image thumbnail with a crop rectangle. Add labeled overlays:
- "Selected crop" as a thick black or white rectangle
- "Core subject" as a solid blue rectangle
- "Envelope / support" as a dashed blue rectangle
- "Secondary subject" as a light blue rectangle
- "Center line" as a green crosshair
- "Rule-of-thirds grid" as green thirds lines
- "Hard reject region" as a red warning outline
- "Teacher / GAIC candidate" as a purple outline
- "Checklist label" as a small dark badge
- "Why-tag" as a rounded pill badge

At the bottom, add a simple flow:
"overlay measurements -> score components -> checklist labels -> why-tags"

Keep all labels large and readable. This is a legend, not a dense pipeline diagram.
```

## SC01. Subject Preservation: `q_subj`

삽입 위치:

```text
7.5 Score Components의 q_subj 설명 직후
```

생성 목적:

`core_recall`, `env_recall`, `secondary_recall`이 어떻게 `q_subj`로 합쳐지고, 이것이 `subject_coverage` checklist와 `subject_preserved`/`subject_poor` why-tag로 이어지는지 보여준다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Subject Preservation Score: q_subj"

Create a three-panel figure using synthetic example image thumbnails of the same person subject:

Panel A: "Good coverage"
Show a crop box that fully includes the blue core subject box and most of the dashed envelope box. Add score card:
"core_recall = 1.00"
"env_recall = 0.94"
"secondary_recall = 0.90"
"q_subj = 0.55 core + 0.25 env + 0.20 secondary = 0.96"
"checklist: subject_coverage = good"
"why-tags: subject_preserved"

Panel B: "Marginal coverage"
Show a crop that cuts part of the envelope and slightly clips the subject support. Score card:
"q_subj = 0.78"
"checklist: subject_coverage = marginal"
"why-tags: subject_marginal"

Panel C: "Poor coverage"
Show a crop that cuts much of the core subject. Use red clipped area. Score card:
"q_subj = 0.52"
"checklist: subject_coverage = poor"
"why-tags: subject_poor"

Use blue overlays for core/envelope, red for clipped subject, and large readable score cards. Add bottom takeaway:
"q_subj measures how much of the intended subject support remains inside the crop."
```

## SC02. Subject Scale: `subject_ratio` and `q_scale`

삽입 위치:

```text
7.5 Score Components의 q_scale 설명 직후
```

생성 목적:

subject가 crop 안에서 너무 작거나, 적절하거나, 너무 큰 경우가 `subject_scale` checklist와 why-tag로 어떻게 바뀌는지 보여준다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Subject Scale: subject_ratio and q_scale"

Create a three-panel comparison with synthetic portrait crop thumbnails:

Panel A: "Too loose"
The person is very small inside a wide crop with lots of empty context. Add score card:
"subject_ratio below target"
"q_scale low"
"checklist: subject_scale = too_loose"
"why-tags: subject_scale_loose, wide_crop"

Panel B: "Ideal scale"
The person fills a natural amount of the crop, with comfortable context. Add score card:
"subject_ratio near mode target"
"q_scale high"
"checklist: subject_scale = ideal_scale"
"why-tags: subject_scale_ideal"

Panel C: "Too tight"
The person is overly large and near the crop edges. Add score card:
"subject_ratio above target"
"q_scale low"
"checklist: subject_scale = too_tight"
"why-tags: subject_scale_tight, tight_crop"

Add a small curve diagram below the panels: x-axis "subject_ratio", peak labeled "mode target", y-axis "q_scale". Keep the curve simple and large.
```

## SC03. Axis Occupancy: `occ_x`, `occ_y`, `q_axis`

삽입 위치:

```text
7.5 Score Components의 occ_x/occ_y/q_axis 설명 직후
```

생성 목적:

subject의 가로/세로 점유율이 mode intent에 맞는지 설명한다. `q_axis`는 직접 checklist key는 아니지만 `scale`, `tight/wide` 해석과 mode intent QA에 중요하다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Axis Occupancy: occ_x, occ_y, q_axis"

Create a four-panel instructional diagram with simplified crop thumbnails:

Panel 1: "Tall portrait fit"
Show a standing person with high vertical occupancy and moderate horizontal occupancy. Label:
"occ_y high, occ_x moderate"
"q_axis good for single_person"

Panel 2: "Over-wide subject"
Show a subject too spread horizontally for a tight portrait intent. Label:
"occ_x too high"
"possible tight_crop"

Panel 3: "Tiny subject"
Show a small subject in a large crop. Label:
"occ_x low, occ_y low"
"wide_crop"

Panel 4: "Object fit"
Show a product object centered with balanced x/y occupancy. Label:
"axis occupancy matches object mode"
"balanced_crop"

Add a side mini-formula:
"occ_x = subject width / crop width"
"occ_y = subject height / crop height"
"q_axis = mode-specific fit"

Use simple measurement arrows inside each crop.
```

## SC04. Placement: `anchor_rx`, `anchor_ry`, `q_place`

삽입 위치:

```text
7.5 Score Components의 q_place 설명 직후
```

생성 목적:

center mode와 rot mode에서 같은 subject라도 좋은 위치가 다르다는 점을 보여준다. `center_dist`, `third_dist`, `centered_subject`, `rule_of_thirds`로 연결한다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Placement Score: Center vs Rule of Thirds"

Create a split figure with two large example crop thumbnails.

Left panel: "Center mode"
Show a subject centered in the crop. Overlay a green center crosshair. Mark the anchor point with a green dot labeled "anchor_rx, anchor_ry". Score card:
"center distance small"
"q_place high"
"checklist: center_dist = center_comp_strong"
"why-tags: centered_subject"

Right panel: "ROT mode"
Show the same subject placed near the left rule-of-thirds vertical line. Overlay a green rule-of-thirds grid. Score card:
"thirds distance small"
"q_place high"
"checklist: third_dist = rule_of_thirds_strong"
"why-tags: rule_of_thirds"

Add a red mini-example between them: "Wrong intent" showing a centered subject under ROT mode or thirds subject under center mode, labeled "strict_v11 may reject".

Use clear grid lines and large labels.
```

## SC05. Aspect Ratio Fit: `q_ar`

삽입 위치:

```text
7.5 Score Components의 q_ar 설명 직후
```

생성 목적:

FREE AR와 fixed AR의 차이, `q_ar`와 `ar_rel_error`가 `ar_fits_well` / `ar_choice_freeform` / `ar_extreme_penalty`로 이어지는 구조를 보여준다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Target Aspect Ratio Fit: q_ar"

Create a three-panel figure.

Panel A: "FREE"
Show a flexible crop rectangle around a scene. Score card:
"target_ar = FREE"
"checklist: ar = ar_choice_freeform"
"why-tags: ar_choice_freeform"

Panel B: "Fixed AR fits"
Show a 16:9 crop that matches the target frame. Add small ruler labels "target 16:9" and "crop 16:9". Score card:
"q_ar high"
"ar_rel_error low"
"checklist: ar = ar_fits_well"
"why-tags: ar_fits_well"

Panel C: "AR mismatch"
Show a crop that visibly does not match a 9:16 target frame, with red warning outline. Score card:
"q_ar low"
"hard reject risk"
"why-tags: ar_extreme_penalty"

Add bottom takeaway: "q_ar checks whether the crop geometry satisfies the requested target AR."
```

## SC06. Context and Copyspace: `q_ctx`, `context_value`, `copyspace`

삽입 위치:

```text
7.5 Score Components의 q_ctx/context 설명 직후
```

생성 목적:

subject 주변 맥락이 부족하거나 과도한 경우, copyspace intent에서 빈 여백을 보존해야 하는 경우를 보여준다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Context and Copyspace"

Create a four-panel example figure:

Panel A: "Context preserved"
Show a person or object with useful surrounding environment inside the crop. Score card:
"context_value in target range"
"checklist: context = context_preserved"
"why-tags: context_preserved"

Panel B: "Context lost"
Show a crop too tight around the subject with surrounding scene cut away. Red context-loss outline. Score card:
"context_value too low"
"checklist: context = context_poor or context_partial"
"why-tags: context_loss"

Panel C: "Context excessive"
Show subject too small with too much background. Score card:
"context_value too high"
"checklist: context = context_excessive"

Panel D: "Copyspace kept"
Show a product or person on one side and blank space on the other side for text. Add a translucent label "copyspace". Score card:
"copyspace_intent = true"
"checklist: copyspace = copyspace_preserved"
"why-tags: copy_space_kept"

Use subtle gray for context area, blue for subject, and green for preserved copyspace.
```

## SC07. Scene Quality: `q_scene`

삽입 위치:

```text
7.5 Score Components의 q_scene 설명 직후
```

생성 목적:

landscape/scene에서 horizon, symmetry, wide crop이 `q_scene`과 `horizon_state`, `horizon_on_target` / `horizon_off_target`으로 이어지는 구조를 보여준다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Scene Score: q_scene"

Create a three-panel landscape scene figure:

Panel A: "Stable horizon"
Show a clean wide scene crop with a level horizon line. Overlay a green horizon guide. Score card:
"q_horizon high"
"q_scene high"
"checklist: horizon_state = strong"
"why-tags: horizon_on_target"

Panel B: "Tilted or cut horizon"
Show a scene crop with horizon tilted or badly cut. Red warning line. Score card:
"q_horizon low"
"checklist: horizon_state = weak"
"why-tags: horizon_off_target, needs_leveling"

Panel C: "Wide scene balance"
Show a balanced scene with symmetry and open context. Score card:
"q_symmetry + q_wide"
"q_scene = horizon + symmetry + wide"

Add small formula strip:
"q_scene = 0.45 horizon + 0.30 symmetry + 0.25 wide"
```

## SC08. Portrait Framing: `q_head`, `q_look`

삽입 위치:

```text
7.5 Score Components의 q_head/q_look 설명 직후
```

생성 목적:

headroom과 lookroom을 예시 이미지로 설명하고, label과 why-tag로 연결한다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Portrait Framing: Headroom and Lookroom"

Create a two-row, three-column figure using synthetic portrait thumbnails.

Top row: Headroom
1. "Too tight" - head near or clipped by top crop edge. Red top margin. Label: "headroom_tight", why-tag "headroom_violation".
2. "OK" - comfortable top margin. Green top margin. Label: "headroom_ok", why-tags "headroom_ok, head_top_safe".
3. "Too loose" - excessive empty space above head. Amber top margin. Label: "headroom_loose".

Bottom row: Lookroom
1. "Insufficient" - face looking right but little space on right side. Red side margin. Label: "lookroom_insufficient", why-tag "lookroom_violation".
2. "Adequate" - gaze direction has natural forward space. Green side margin. Label: "lookroom_adequate", why-tag "lookroom_ok".
3. "Excessive" - too much empty forward space. Amber side margin. Label: "lookroom_excessive".

Add measurement arrows for headroom and lookroom. Add score cards: "q_head" and "q_look".
```

## SC09. Face, Joint, and Intrusion Safety

삽입 위치:

```text
7.5 Score Components의 face_recall/joint_cut_score/intrusion_ratio 설명 직후
```

생성 목적:

안전 component가 score 향상용이 아니라 hard gate와 checklist/why-tag에 직접 연결되는 것을 보여준다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Portrait Safety Components"

Create four example panels:

Panel A: "No face cut"
Show a portrait crop fully containing the face box. Score card:
"face_recall >= 0.98"
"checklist: face_cut = no_face_cut"
"why-tags: avoid_face_cut"

Panel B: "Face cut"
Show crop clipping the forehead or side of face. Use red clipped face area. Score card:
"face_recall low"
"checklist: face_cut = face_cut"
"hard reject risk"

Panel C: "No joint cut"
Show visible keypoints inside crop. Score card:
"joint_cut_score <= 0.05"
"checklist: joint_cut = no_joint_cut"
"why-tags: avoid_person_cut"

Panel D: "Intrusion"
Show another person entering the crop and occupying area. Use amber overlay. Score card:
"intrusion_ratio high"
"penalty or reject"

Add bottom note: "Safety components can block positives even when utility is high."
```

## SC10. Group Completeness: `group_recall`, `group_balance`, `q_group`

삽입 위치:

```text
7.5 Score Components의 group_recall/group_balance/q_group 설명 직후
```

생성 목적:

group crop은 전체 union만 보면 충분하지 않고, member별 recall과 balance가 중요하다는 점을 보여준다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Group Completeness: q_group"

Create a three-panel group portrait figure:

Panel A: "Balanced group crop"
Show three people fully inside the crop. Each member has a blue box, all similar recall. Score card:
"group_recall high"
"group_balance high"
"q_group high"
"why-tags: subject_preserved, balanced_crop"

Panel B: "One member cut"
Show group crop clipping one person at the edge. Use red clipped member. Score card:
"member recall low"
"group_balance low"
"hard reject risk"

Panel C: "Union included but unbalanced"
Show group union mostly inside crop but one person much less visible. Score card:
"group_recall moderate"
"group_balance low"
"q_group lower"

Add formula strip:
"q_group = 0.55 group_recall + 0.25 group_balance + 0.20 env_recall"
```

## SC11. Object Foreground Preservation

삽입 위치:

```text
7.5 Score Components의 object_recall/saliency_crop_overlap 설명 직후
```

생성 목적:

object mode에서 saliency foreground와 object recall이 왜 필요한지 보여준다. 사람/얼굴 오검출이 아닌 실제 foreground object만 positive로 남기는 정책을 설명한다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Object Foreground Preservation"

Create four panels with synthetic product/object scenes:

Panel A: "Foreground object preserved"
Show a product object inside crop with green saliency mask overlapping the object. Score card:
"object_recall high"
"saliency_crop_overlap high"
"checklist: subject_coverage = good"
"why-tags: subject_preserved"

Panel B: "Wrong foreground"
Show crop around background or irrelevant item while true object saliency is outside. Red warning. Score card:
"saliency overlap low"
"hard reject"

Panel C: "Object truncated"
Show crop cutting a major part of the object. Score card:
"object_recall low"
"checklist: subject_coverage = poor"
"why-tags: subject_poor"

Panel D: "Multi-object"
Show two related objects with union envelope. Score card:
"multi-object support"
"q_group for object_multi"

Use green translucent mask for saliency foreground and blue boxes for object boxes.
```

## SC12. Teacher Signal: `q_teach`

삽입 위치:

```text
7.5 Score Components의 q_teach 설명 직후 또는 6.5 Landscape Candidate Policy와 함께
```

생성 목적:

`q_teach`가 checklist 항목 자체는 아니지만 landscape final score에 강하게 들어가며, GAIC teacher lineage가 어떤 역할을 하는지 보여준다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Teacher Signal: q_teach"

Create a diagram with three candidate crops for the same landscape scene:

Candidate 1: "Direct GAIC teacher"
Purple crop outline, highest teacher score.

Candidate 2: "GAIC-lineage candidate"
Purple dashed outline, medium teacher score.

Candidate 3: "Fallback candidate"
Gray outline, lower teacher score.

Show a simple conversion box:
"teacher_score_raw -> normalized q_teach"

On the right, show how q_teach connects:
"Landscape utility: q_teach is primary"
"Other modes: teacher can be tie-breaker or component"

Add note:
"q_teach informs final score, while checklist labels describe interpretable crop properties."
```

## SC13. Penalties and Hard Rejects

삽입 위치:

```text
7.2 공통 원칙 또는 7.7 Strict Intent Gate 직후
```

생성 목적:

soft penalty와 hard reject의 차이를 예시 이미지와 checklist/why-tag로 연결한다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Penalties vs Hard Rejects"

Create a split figure:

Left side: "Soft penalties"
Show candidate still eligible but with lower score. Examples:
- route_dom_penalty
- safety_penalty_soft
- context_loss
Use amber arrows reducing a score bar.

Right side: "Hard rejects"
Show candidate blocked before positive selection. Examples:
- face cut
- severe joint cut
- subject cut
- target AR mismatch
Use red stop gates and blocked arrows.

At the bottom, show:
"Soft penalty: lowers U_q(c)"
"Hard reject: H_q(c) = 0, no positive"

Add derived outputs:
"checklist labels record the issue"
"why-tags summarize key reasons"
```

## SC14. Full Worked Example: From Crop to Attributes

삽입 위치:

```text
7.5 Score Components 마지막 또는 9.3 Added Attributes 직후
```

생성 목적:

하나의 예시 이미지에서 overlay 측정값이 `score_components`, `checklist_scores`, `checklist_labels`, `why_tags`로 어떻게 변환되는지 end-to-end로 보여준다.

영문 프롬프트:

```text
Create a clean vector-style technical infographic for an engineering onboarding document. Use a white background, sharp thin lines, restrained enterprise colors, and large readable English labels. Use a 16:9 landscape canvas, high resolution, minimal decoration, no photorealistic imagery, no cartoon style, no tiny text, no dense paragraphs.

Title: "Worked Example: One Crop, Many Supervision Signals"

Create an end-to-end diagram using one synthetic portrait image.

Left: image thumbnail with selected crop, core subject box, envelope box, center crosshair, headroom arrow, lookroom arrow.

Middle top: "Measured score components"
Show a compact table:
"q_subj = 0.96"
"q_scale = 0.88"
"q_place = 0.91"
"q_head = 0.82"
"q_look = 0.76"
"q_ar = 1.00"

Middle bottom: "Final score"
Show:
"score_mode = weighted utility - penalties"
"final_score = 0.82"

Right top: "Checklist labels"
Show badges:
"subject_coverage = good"
"subject_scale = ideal_scale"
"headroom = headroom_ok"
"lookroom = lookroom_adequate"
"center_dist = center_comp_strong"
"ar = ar_fits_well"

Right bottom: "Why-tags"
Show sparse tags:
"subject_preserved"
"subject_scale_ideal"
"headroom_ok"
"lookroom_ok"
"centered_subject"
"ar_fits_well"

Add bottom takeaway:
"Scores train regression heads; labels and why-tags explain the crop."
```

### 4.3 SC-series 생성 후 문서 삽입 권장

7.5에는 모든 SC Figure를 한꺼번에 넣기보다, 아래 순서로 나누어 삽입하는 것을 권장한다.

1. `SC00` legend
2. `SC01` q_subj
3. `SC02` q_scale
4. `SC04` q_place
5. `SC08` headroom/lookroom
6. `SC10` group
7. `SC11` object foreground
8. `SC12` q_teach
9. `SC13` penalties/hard rejects
10. `SC14` full worked example

`SC03`, `SC05`, `SC06`, `SC07`, `SC09`는 문서 길이에 따라 추가한다. score component를 완전히 그림으로 설명하려면 전체 SC00-SC14를 생성해도 된다.

## 5. 문서 삽입 권장 순서

권장 생성/삽입 순서:

1. F01 `Score Components Map`
2. SC00-SC14 `Score Components 심화 Figure Pack`
3. F02 `Mode Utility Matrix / Heatmap`
4. F03 `Checklist vs Why-Tag`
5. F06 `Landscape GAIC-Qteach Subject-Safe Policy`
6. F07 `Hard Gate, Utility, Positive Selection Funnel`
7. F04/F08 artifact/schema 계열
8. F09/F10 QA/history 계열

P0 Figure를 먼저 생성하면 문서의 가장 어려운 부분인 score와 explainability가 크게 읽기 쉬워진다. 기존 mental model, ontology, pipeline 그림은 이미 있으므로, 새 Figure는 score/gate/schema/explainability 중심으로 보강하는 것이 효율적이다.

## 6. 파일명 제안

생성 이미지 파일은 기존 bundle과 분리해 아래 경로를 권장한다.

```text
Implement_Docs/SSTK_MultiMode_TrainingLabels_Markdown_Bundle/assets_v14_explainability/
```

권장 파일명:

```text
fig_v14_score_components_map.png
fig_v14_mode_utility_heatmap.png
fig_v14_checklist_vs_whytags.png
fig_v14_artifact_map.png
fig_v14_query_open_rules.png
fig_v14_landscape_gaic_subjectsafe.png
fig_v14_score_gate_positive_funnel.png
fig_v14_label_json_schema_views.png
fig_v14_qa_dashboard.png
fig_v14_improvement_timeline.png
```

SC-series 권장 파일명:

```text
fig_v14_sc00_visual_legend.png
fig_v14_sc01_subject_preservation_qsubj.png
fig_v14_sc02_subject_scale_qscale.png
fig_v14_sc03_axis_occupancy_qaxis.png
fig_v14_sc04_placement_qplace.png
fig_v14_sc05_aspect_ratio_qar.png
fig_v14_sc06_context_copyspace_qctx.png
fig_v14_sc07_scene_qscene.png
fig_v14_sc08_headroom_lookroom.png
fig_v14_sc09_portrait_safety.png
fig_v14_sc10_group_qgroup.png
fig_v14_sc11_object_foreground.png
fig_v14_sc12_teacher_qteach.png
fig_v14_sc13_penalties_hard_rejects.png
fig_v14_sc14_worked_example_attributes.png
```

문서 삽입 예시:

```markdown
![Score components map](SSTK_MultiMode_TrainingLabels_Markdown_Bundle/assets_v14_explainability/fig_v14_score_components_map.png){ width=100% }
```
