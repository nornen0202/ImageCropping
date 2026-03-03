# SSTK(Shutterstock) Cropping 데이터셋: 공개 데이터셋 대비 차별점/한계, 모델 R&D 로드맵, 그리고 **Subject‑Mode Routing + C2 Top‑N Seg 확장** “패치 수준 설계” (KO)

본 문서는 아래 2가지를 **하나로 통합**한 실행 지향 문서입니다.

1) **기존 공개 크롭 데이터셋들과 비교 분석** → *우리가 만드는 SSTK 데이터셋의 특징/차별점/한계*  
2) v1.9 설계 + 현재 구현(README/REPORT 기반)에서 드러난 문제(특히 비인물/자연/배경에서 주피사체 오류) 해결을 위해,  
   **“subject_mode routing을 어느 JSONL에 어떤 필드로 추가할지”**,  
   **“C2(c2_seg)를 topN + importance_score로 확장하는 스키마/QA 항목”**까지 **패치 수준 설계**로 내림

---

## 0. 현재 구현 상태 요약(문서/리포트 기준)

현재 end‑to‑end 파이프라인은 다음 범위까지 실제 실행이 가능합니다.

- Phase A(Filter) → Phase B(Perception C1/C2/C3/C5) → Candidate(AR별 후보) → Teacher Scorer(Cheap→Expensive→Top‑K) → QA/Viz

`10K_local` 런(균형 샘플 500장)에서 관측된 대표 지표:

- Filtered 이미지 수: **500장**
- Candidate: 평균 **531.7개/이미지**, public teacher proposal 주입률 **1.0**
- Public teacher raw: **1500 records**(GAIC/CACNet/CGS 각 500)
- Teacher decision: `crop` 0.388 / `minimal_crop` 0.612  
- Teacher final score: mean −3.6476 / p90 1.0597 / neg_rate 0.7144

> 해석: 파이프라인 골격은 완성됐지만, **(a) 후보 수 폭증**, **(b) decision이 minimal 쪽으로 기울어짐**,  
> **(c) score 분포가 카테고리별/모드별로 정규화되지 않음**,  
> 그리고 무엇보다 **(d) 비인물/자연/배경에서 ‘대표 피사체 1개’ 선택이 깨질 때 전부 무너지는 구조**가 핵심 리스크로 보입니다.

---

## 1. 기존 공개 Cropping 데이터셋들과 비교(특징/차별점/한계)

### 1.1 공개 데이터셋들의 전형적 구성과 공통 한계

공개 크롭 데이터셋은 보통 아래 2계열입니다.

- **Sparse GT(정답 crop 1개 or 소수)**: 이미지당 1~10개의 GT crop  
- **Dense candidate + ranking/pairwise**: 이미지당 일정 후보(예: 24개 뷰)를 두고 쌍대 비교/랭킹

공통 한계(제품/상용 관점):

- **AR‑조건부 라벨이 약함**: “1:1/9:16/4:5 등 여러 AR에서 각각 최적”이 필요하지만 대부분 단일 GT 또는 단일 목적.
- **“원본이 정답(keep_full)” 케이스를 레이블로 다루지 않음**: 스톡/프로 사진 분포에선 매우 중요.
- **메타 기반 의도(카피스페이스/isolated/portrait 등) 결합이 약함**
- **설명 가능한 레이블(왜 이 크롭인가)**이 거의 없음(있어도 소규모 ICIT류).
- **상용 학습/배포 관점의 권리/운영/버전/QA 게이트**가 데이터 자체에 내장되어 있지 않음.

### 1.2 대표 공개 셋들의 포지션(요약)

| 데이터셋 | 라벨 형태 | 강점 | 한계(우리 관점) |
|---|---|---|---|
| **GAICD/GAICv2** | Grid‑anchor 후보 + 점수/랭킹 | grid 후보 기반 학습과 궁합 좋음 | 멀티 AR/설명/상용 운영(메타+QA)까지는 직접 제공 X |
| **CPC** | 24 view + pairwise ranking(매우 많음) | ranking 학습에 최적 | AR‑조건부/설명/상용 운영 X |
| **FCDB** | crop/ranking(규모 중간) | 실사진 기반 | 규모/도메인 다양성 한계, AR‑조건부/설명 약함 |
| **FLMS/HCDB** | GT crop 소수 | 빠른 벤치 | 규모 작음, AR‑조건부/설명 약함 |
| **SACD** | dense 후보 + optimal boxes(+pair) | semantic-aware/랭킹 풍부 | 상용 운영/메타 기반 정책/설명은 직접 제공 X |
| **ICIT(InstructCrop류)** | crop + instruction + 설명(소규모) | 설명 가능한 크롭 연구에 적합 | 규모 작음, 대량 AR 라벨/상용 운영에 그대로 쓰기 어려움 |
| **Unsplash** | 이미지 소스(크롭 GT 아님) | GenCrop/outpainting 약지도 생성에 유용 | 라벨 없음, 상용학습은 별도 권리/정책 필요 |

> XPView는 공개적으로 “정식 1차 출처/정의”가 검색/문헌에서 혼재되는 경우가 있어, **정확한 출처 확인 후** 비교표에 넣는 것이 안전합니다.

### 1.3 SSTK(우리) 데이터셋의 차별점(공개 셋 대비 “제품 지향” 포인트)

SSTK 데이터셋/팩토리(v1.9~)가 공개 셋과 다른 지점은 “라벨이 많다”가 아니라, **제품 운영에 필요한 상태(state)와 provenance를 데이터로 남기는 것**입니다.

1) **AR‑조건부 Top‑K crop**  
- AR마다 Top‑K(3~5) + score + flags를 저장(학습/평가/UX에 직접 대응)

2) **`keep_full / minimal_crop / crop` 정책 레이블**  
- 스톡 사진에선 “원본이 이미 정답”이 자주 발생  
- `minimal_crop`이 높은 비율로 나온다는 사실 자체가 “도메인 특성”을 반영하는 중요한 레이블

3) **메타데이터 기반 의도/정책 학습 가능**  
- tags/alt/caption으로 `portrait/headshot/selfie/group/copy-space/isolated/background/texture…` 같은 정책 조건을 학습에 넣을 수 있음

4) **설명 가능한 학습 데이터로 확장 가능(체크리스트 기반 rationale)**  
- headroom/lookroom/horizon/symmetry/copyspace/text_keep 같은 항목을 pass/fail + why_tags로 저장하면  
  모델이 “왜 이 크롭인가”를 **다중 라벨**로 학습 가능

5) **공개 Cropping teacher 모델을 ‘proposal generator’로 편입(이미 구현)**  
- GAIC/CACNet/CGS를 “최종 judge”가 아니라 후보 공간 확장 도구로 사용  
- v1.9의 “proposal 주입만” PoC로 **후보 공간 개선폭**을 분리 측정 가능

### 1.4 SSTK(우리) 데이터셋의 한계/리스크(공개 셋과 대비하여 솔직히)

| 축 | SSTK의 리스크/한계 | 공개 셋은 어떤가 |
|---|---|---|
| **Teacher bias** | pseudo‑label이므로 스코어러/teacher의 취향/편향이 데이터에 그대로 남음 | 공개 셋도 편향은 있으나 GT/human label 비중이 상대적으로 높은 경우가 있음 |
| **도메인 과적합** | 스톡 사진 스타일로 과적합 가능(“스톡 느낌”의 구도 편향) | FCDB/Unsplash/Flickr 기반은 상대적으로 다양할 수 있음 |
| **주피사체 정의** | “대표 피사체 1개” 가정이 자연/배경/추상에서 깨짐 → 후보/스코어/QA 전부 붕괴 | 공개 셋도 완벽하진 않지만, 데이터 정의가 단순한 대신 문제 범위가 좁음 |
| **대량 운영 비용** | 후보 수가 크면 teacher/QA 비용 폭증(특히 VLM) | 공개 셋은 규모가 작아 비용 문제가 덜하지만, 제품 규모를 커버 못함 |
| **메타 노이즈** | tags/alt/caption은 노이즈/비일관성 존재 | 공개 셋은 메타가 없는 대신 “정책 학습”도 못함 |
| **법무/권리** | Shutterstock 라이선스 범위에 따라 파생물/모델학습 허용이 달라질 수 있음(내부 정책 필요) | 공개 셋은 연구용 라이선스가 많고 상용 전환이 어려움 |

---

## 2. 우리 데이터셋으로 가능한 모델 R&D 계획(연구 방향)

SSTK 데이터는 아래 연구/제품 트랙을 동시에 열어줍니다.

### 2.1 AR‑조건부 Cropping 모델(주력)
- 입력: (Image, target_AR)
- 출력: AR별 Top‑K crop(or Top‑1)
- 학습:
  - box regression + listwise ranking(Top‑K) + diversity loss(Top‑K 중복 방지)
- 포인트: 공개 데이터만으로는 “멀티 AR 제품 수준” 학습이 어려움 → SSTK의 핵심 가치

### 2.2 keep/minimal/crop 정책 모델(실서비스 핵심)
- 출력: `decision_type ∈ {keep_full, minimal_crop, crop}`
- `delta_improve` 기반 gate를 학습(또는 rule+학습 혼합)하면  
  “괜히 크롭해서 망치는” 사례를 구조적으로 줄일 수 있음

### 2.3 타입별 특화(MoE/Conditioned) 모델
- `subject_mode`(아래 4~5장에서 추가)로 라우팅:
  - portrait_single/group: headroom/lookroom/cutoff
  - object/product: centered + context + copyspace
  - scene/landscape: horizon + composition embedding(PICD) + negative space
  - text/doc: OCR keep ratio
- 구조:
  - (A) 단일 모델 + mode token
  - (B) Router + 전문가(MoE)

### 2.4 설명 가능한 Cropping(=rationale head)
- 출력에 `rationale_tags`(headroom_ok, lookroom_violation, horizon_tilted, symmetry_high, copyspace_preserved…)를 포함시키면  
  사용자가 납득 가능한 설명 UX가 가능해짐.

### 2.5 Multi‑subject / Group‑aware cropping
- 다중 인물/다중 객체를 “하나”로 강제하지 않고, **피사체 집합(set)** 을 보존하는 모델  
- 가족/단체/스포츠/거리 사진에서 품질이 급상승

---

## 3. 현재 구현 결과 기반 보완 제안(우선순위)

### 3.1 후보 수 폭증(평균 531.7개/이미지)
- 장점: recall↑
- 단점: teacher/QA 비용 폭증, Top‑K 다양성 붕괴 위험

**권장**
- cheap score로 AR별 **N→M(예: 500→40)** 강제(설계대로)
- QA에 Top‑K 다양성 지표 의무화(평균 IoU, center‑dist, unique_source_count 등)
- proposal 주입 효과 지표 추가:
  - `proposal_rescue_rate` (oracle recall 개선폭)
  - `teacher_seed_winrate` (주입 후보가 최종 top에 들어간 비율)

### 3.2 decision이 minimal 쪽으로 기울어짐(crop 0.388 / minimal 0.612)
- “원본이 정답” 분포가 실제로 많을 수 있음(스톡 사진 특성)  
- 하지만 과도하면 스코어러가 **risk‑averse**로 설계됐을 가능성도 있음

**권장**
- `subject_mode`별로 decision 분포를 나누어 봐야 함(특히 scene/landscape에서 minimal이 많아야 자연스러움)
- `delta_improve`의 정규화(AR별/카테고리별 percentile) 도입

### 3.3 비인물/자연/배경에서 주피사체 오류 → 후보/스코어 붕괴
이 문서의 핵심 해결 과제이며, 아래 **4)~5)** 의 패치 설계로 해결합니다.

---

## 4. 문제 정의 재정의: “대표 피사체 1개”가 항상 정답이 아니다

비인물/풍경/배경은 다음이 더 중요합니다.

- 객체 1개보다 **장면 구성(scene composition)** 이 목적
- SAM류는 큰 영역(하늘/바다/벽)을 “아주 잘” 분리 → 오히려 대표 피사체로 오인하기 쉬움
- 그래서 “주피사체 1개” 기반 후보 생성은 구조적으로 취약

### 핵심 해법: **Subject‑Mode Routing + Multi‑Subject Set + Scene Anchors**

- `subject_mode`를 먼저 분류하고,
- 모드별로 **(a) 피사체 정의**, **(b) 후보 생성 템플릿**, **(c) 스코어 가중치**, **(d) QA 게이트**를 바꿉니다.

---

## 5. (패치 수준 설계) Subject‑Mode Routing: 어디에, 어떤 필드를 추가할까?

### 5.1 subject_mode enum(초기 v1)

아래 enum으로 시작하고, 운영 중 필요에 따라 세분화합니다.

- `portrait_single`
- `portrait_group`
- `object_single`
- `object_multi`
- `scene_landscape`
- `background_texture_copyspace`
- `text_document`
- `other_ambiguous`

> `super_cat`(12종)과는 별개입니다. `super_cat=landscape_nature`라도 copy‑space가 강하면 `background_texture_copyspace`로 라우팅할 수 있습니다.

---

### 5.2 subject_mode를 “어느 JSONL에” 추가할까? (권장: 3곳)

현재 파이프라인의 데이터 흐름상, 최소한 아래 3곳에 넣으면 **Generate→Score→QA가 닫힙니다.**

#### (A) **Precompute merged JSONL**  
파일:  
`data/SSTK/<DATANAME>/artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl`

- 이유: candidate 생성과 teacher scoring 모두 precompute를 읽습니다. 여기서 mode가 결정되면 downstream이 일관됩니다.
- 추가 필드(Top‑level):

```jsonc
{
  "image_id": "...",
  // (기존) c2_seg, c3_pose, c5_geom ...
  "routing": {
    "subject_mode": "scene_landscape",
    "subject_mode_conf": 0.86,
    "subject_mode_reasons": ["tag:landscape", "person_count=0", "text_ratio<0.02"],
    "subject_mode_flags": {
      "has_person": false,
      "has_text_heavy": false,
      "has_copyspace_tag": false,
      "is_background_like": true
    },
    "shot_type": null,                   // portrait일 때 headshot/half/full
    "primary_subject_type": "scene",     // person/object/scene/text
    "primary_subject_source": "c2",      // c2|c3|union|none
    "subject_set": {
      "num_person": 0,
      "num_subject_inst": 3,             // c2 topN 개수
      "union_box_xyxy": [..],            // multi-subject union
      "primary_idx": -1                  // scene 모드면 -1 허용
    },
    "policy_id": "scene_v1"              // scorer/candidate weights preset
  }
}
```

#### (B) **Candidate JSONL(AR 후보 결과)**  
파일:  
`data/SSTK/<DATANAME>/artifacts/candidates/candidates_ar_<run_tag>.jsonl`

- 이유: 후보가 만들어질 때 **어떤 모드/정책으로 만들어졌는지**가 provenance로 남아야 디버깅/회귀 테스트가 가능합니다.
- 추가 필드(이미지 레벨 헤더 또는 메타):

```jsonc
{
  "image_id": "...",
  "target_ar": "1:1",
  "routing": { "subject_mode": "...", "policy_id": "..." },
  "subject": { "primary_box": [..], "union_box": [..] },
  "candidates": [ ... ],
  "candidate_meta": {
    "source_mix": {"grid": 320, "teacher_seed": 15, "phi": 40, "jitter": 120},
    "mode_overrides": ["scene_mode_disable_subject_centering"]
  }
}
```

#### (C) **Teacher score JSONL(최종 라벨/Top‑K)**  
파일:  
`data/SSTK/<DATANAME>/artifacts/teacher/scores/teacher_scores_ar_<run_tag>.jsonl`

- 이유: `decision_type`/위반율/개선량이 **subject_mode별로 해석이 달라**집니다.
- 추가 필드:

```jsonc
{
  "image_id": "...",
  "target_ar": "1:1",
  "routing": { "subject_mode": "...", "policy_id": "..." },
  "decision_type": "minimal_crop",
  "delta_improve": 0.012,
  "tau_improve": 0.035,
  "qa_flags": {
    "headroom_pass": true,
    "lookroom_pass": true,
    "horizon_pass": true,
    "context_pass": false,
    "subject_mode_consistent": true
  },
  "topk": [ ... ] // 기존 구조 유지
}
```

---

### 5.3 subject_mode는 “어디서 계산”할까? (권장 구현 패치)

현재 코드 흐름을 크게 깨지 않으면서 넣는 **가장 안전한 패치**:

- **옵션 1(권장)**: `merge` 이후 enrich 스크립트 추가  
  - 이미 `enrich_c3_pose_jsonl.py`가 존재하듯이,
  - `enrich_subject_mode_jsonl.py`를 추가해 merged precompute jsonl에 routing을 주입

**새 스크립트**: `src/scripts/enrich_subject_mode_jsonl.py` (추가)

입력:
- `--input_feats_jsonl artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl`
- `--input_filtered_parquet filtered_<bucket>.parquet` (tags/super_cat)
- (선택) `--input_ocr_jsonl` (OCR 도입 후)
출력:
- `.../feats_c2c3c5_v2_strict_enriched_routed.jsonl` (또는 overwrite)

---

### 5.4 subject_mode 라우팅 규칙(초기 v1 룰셋)

아래는 **구현 가능한 규칙**(tag + precompute feature)입니다.

#### 입력 신호
- `tags_norm`: 정규화된 tags(문자열 리스트)
- `super_cat`: (예: people_single, landscape_nature, documents_text …)
- `c3_person_count`: C3에서 검출된 사람 수
- `c2_top_masks_stats`: C2에서 topN 마스크의 면적 분포/경계 접촉 등
- (선택) `ocr_text_area_ratio`: OCR 이후

#### 룰(우선순위 기반)

```python
def infer_subject_mode(tags_norm, super_cat, c3_person_count, ocr_text_area_ratio=None):
    # 1) 텍스트/문서
    if super_cat == "documents_text" or has_any(tags_norm, ["poster","banner","typography","document","text"]):
        return "text_document", 0.95, ["super_cat/doc_tag"]
    # 2) 인물
    if c3_person_count >= 2 or has_any(tags_norm, ["group","family","team","crowd","couple"]):
        return "portrait_group", 0.9, ["person_count>=2 or group_tag"]
    if c3_person_count == 1 or has_any(tags_norm, ["portrait","headshot","selfie","person"]):
        return "portrait_single", 0.85, ["person_count==1 or portrait_tag"]
    # 3) 카피스페이스/배경
    if has_any(tags_norm, ["copy space","background","texture","pattern","isolated","minimal","negative space"]):
        return "background_texture_copyspace", 0.8, ["copyspace/background_tag"]
    # 4) 풍경/장면
    if super_cat in ["landscape_nature","architecture_exterior","indoor_interior"] or has_any(tags_norm, ["landscape","cityscape","interior","architecture"]):
        return "scene_landscape", 0.75, ["scene_tag/super_cat"]
    # 5) 객체(제품/동물/음식/차량)
    if super_cat in ["animals","food","product_object","transportation","sports"] or has_any(tags_norm, ["animal","pet","food","vehicle","product"]):
        # single vs multi는 C2 topN 분포로 후판정
        return "object_single", 0.6, ["object_super_cat"]
    return "other_ambiguous", 0.3, ["fallback"]
```

#### object_single vs object_multi 후판정(=C2 topN 필요)
- `importance_score` 상위 1개가 압도적이면 single  
- 상위 2~3개가 비슷하면 multi로 보고 union 기반 후보 생성

---

### 5.5 policy_id(모드별 후보/스코어 가중치 preset)

`routing.policy_id`는 downstream에서 단순 switch로 쓰기 위함입니다.

예시:

- `portrait_single_v1`  
  - 후보: person box 중심 + headroom/lookroom jitter 강화  
  - 스코어: cutoff/headroom/lookroom 가중치↑, horizon 중간, context 중간
- `portrait_group_v1`  
  - 후보: group union 유지 + 좌우 여백 + headroom 완화  
  - 스코어: 사람 cut‑off 최우선, symmetry/center 보너스(단체사진)
- `scene_v1`  
  - 후보: horizon‑third / comp24 / negative space 템플릿 강화  
  - 스코어: horizon/roll/comp24/context↑, subject coverage↓, keep_full gate 완화
- `copyspace_v1`  
  - 후보: copy‑space preserve 템플릿(좌/우/상단 여백) 강화  
  - 스코어: copyspace 보존↑, center bias↓
- `text_v1`  
  - 스코어: OCR text_keep_ratio 최우선(문서 트랙)

---

## 6. (패치 수준 설계) C2(c2_seg)를 **Top‑N + importance_score**로 확장

### 6.1 왜 top‑1이 아니라 top‑N인가?
- 비인물/풍경/복합 장면은 “대표 피사체 1개”로 요약하기 어렵습니다.
- top‑N을 저장하면:
  - multi‑subject union/coverage 후보 생성 가능
  - scene 모드에서 “피사체 없음”을 합리적으로 처리 가능(=primary_idx=-1)
  - QA에서 “배경 선택률/오류율”을 측정 가능

### 6.2 C2 출력 스키마 확장(호환 유지)

현재 `c2_seg`는 “list of instances” 형태이므로, **list 구조는 유지**하면서 instance dict를 확장합니다.

#### (A) instance dict 확장(Top‑N)

```jsonc
{
  "box": [x1,y1,x2,y2],         // 기존 유지 (px)
  "class_id": -1,
  "score": 1.23,                // 기존 score(내부 heuristic) or det_conf
  "mask_rle": "...",            // (선택) top1만 저장해도 됨
  "area": 12345,                // 기존 유지
  "area_ratio": 0.182,          // 추가
  "center_xy": [cx, cy],        // 추가(px)
  "center_dist_norm": 0.34,     // 추가
  "border_touch": 0.25,         // 추가(0~1, bbox가 경계에 닿는 정도)
  "bg_like": false,             // 추가(배경성 마스크 판정)
  "importance_score": 2.71,     // 추가(핵심)
  "importance_comp": {          // 추가(디버깅용)
    "det": 0.8,
    "area_pref": 0.3,
    "center": 0.4,
    "border_pen": -0.1,
    "person_overlap": 0.0,
    "clip_align": 0.0
  },
  "source": "sam2_amg"          // 추가(sam2_amg | yolo_prompt | efficient_sam_prompt)
}
```

#### (B) image‑level summary 추가(Top‑level)

```jsonc
{
  "image_id": "...",
  "c2_seg": [ ... topN instances ... ],
  "c2_primary_idx": 0,               // -1 가능(scene/copyspace)
  "c2_union_box_xyxy": [..],         // topM union (M<=N)
  "c2_topn": 5,                      // config
  "c2_stats": {
    "num_raw_masks": 213,
    "num_kept": 5,
    "importance_gap_1_2": 0.92,
    "max_area_ratio": 0.81
  }
}
```

> 저장 비용이 걱정이면: `mask_rle`은 **top1 + union_mask**만 저장하고, 나머지는 bbox 중심으로 유지해도 candidate/teacher 대부분은 동작합니다.

---

### 6.3 importance_score 설계(실행 가능한 수식/규칙)

핵심: “중앙+면적”만으로는 모자/책 같은 주변 물체가 주피사체로 뽑히기 쉽습니다.  
importance는 최소한 **(a) objectness**, **(b) meta alignment**, **(c) 배경 억제**를 포함해야 합니다.

#### (A) 공통 features
- `area_ratio = area / (H*W)`
- `dist = center_dist_norm`
- `border_touch ∈ [0,1]`
- `det_conf`: YOLO box와 매칭되면 그 confidence(없으면 0)
- `person_overlap`: C3 person union box와 IoU(없으면 0)
- (선택) `clip_align`: crop embedding vs (tags/caption) embedding cosine

#### (B) 배경 억제(background‑like) 판정(간단 휴리스틱)

```python
bg_like = (
    area_ratio > 0.75 and border_touch >= 0.5   # 프레임 대부분 + 가장자리 접촉
) or (
    area_ratio > 0.60 and aspect_is_extreme(box) and border_touch >= 0.5
)
```

#### (C) importance_score(모드별 가중치)

```python
# area 선호: 너무 작거나 너무 큰 것을 억제하는 형태(예: 삼각형/가우시안)
area_pref = exp(-((area_ratio - mu)**2) / (2*sigma**2))   # mu=0.18, sigma=0.12 (초기)
center_pref = 1.0 - dist
border_pref = 1.0 - border_touch

S = (
  w_det   * det_conf +
  w_area  * area_pref +
  w_center* center_pref +
  w_border* border_pref +
  w_person* person_overlap +
  w_clip  * clip_align -
  w_bg    * int(bg_like)
)
```

초기 권장(quality_first):
- portrait: `w_person=1.0, w_det=0.6, w_bg=1.0, w_center=0.3, w_area=0.3`
- object:   `w_det=0.8, w_bg=1.0, w_center=0.2, w_area=0.4`
- scene:    `w_bg=0.2, w_center=0.1, w_area=0.1`(사실상 primary를 강제하지 않음)  
  → 대신 `c2_primary_idx=-1` 허용, union_box만 사용하거나 scene anchors로 후보 생성

---

### 6.4 Top‑N 선택/정렬 규칙

1) raw masks 생성(SAM2 AMG 등)  
2) 면적 필터(너무 작은 노이즈 제외)  
3) importance_score 계산  
4) **Top‑N 선택**(N=5~10)  
5) NMS/dedupe(유사 마스크 제거)  
6) `primary_idx` 결정:
- portrait/object: top1
- object_multi: top2~3 union 사용
- scene/copyspace: **primary_idx=-1** 허용(“주피사체 없음”)

---

## 7. Candidate/Teacher/QA까지 “구도 규칙 → 피처 → 스코어 → QA”가 닫히는 운영 설계

### 7.1 Candidate Generator 패치(모드 기반 분기)

- **portrait_single/group**
  - subject prior: C3 person union 중심, C2는 보조
  - 후보: headroom/lookroom jitter, group union 유지 템플릿
- **object_single/multi**
  - subject prior: C2 top1 또는 union_box
  - 후보: subject‑center templates + context padding
- **scene_landscape**
  - subject prior: “없음” 가능(=primary_idx=-1)
  - 후보: horizon‑third anchors, comp24 anchors, negative space templates, symmetry‑center 후보
- **background_texture_copyspace**
  - 후보: copy‑space preserve(좌/우/상단 여백), minimal crop 우선
- **text_document**
  - OCR 도입 시 text‑keep 우선 후보(텍스트 박스 보존을 기준으로 후보 prune)

> 이 라우팅이 없으면 “주피사체 잘못 선택”이 후보 생성 전체를 망칩니다.

### 7.2 Teacher Scorer 패치(모드별 스코어 가중치 스케줄)

- `routing.policy_id`에 따라 cheap score 항들의 가중치를 바꿉니다.
- scene 모드에서는 `S_subject`/`coverage` 비중을 낮추고 `R_horizon/R_picd/R_context` 비중을 올립니다.
- copyspace 모드에서는 `R_copyspace_preserve`를 올리고 center bias를 내립니다.

---

## 8. (패치 수준 설계) QA 리포트 항목 추가(필수)

현재 QA는 decision/score/일부 pass‑fail을 제공하지만, **subject_mode 도입 후에는 “모드별” 리포트가 필수**입니다.

### 8.1 QA CSV/JSON에 추가할 “공통 컬럼”(image‑level)

- `subject_mode`
- `policy_id`
- `subject_mode_conf`
- `subject_mode_conflict` (tags 기반 scene인데 portrait로 나온다 등)
- `c2_num_inst` (topN)
- `c2_primary_idx`
- `c2_primary_area_ratio`
- `c2_primary_bg_like`
- `multi_subject` (top2 중요도 근접 or 사람≥2)
- `union_used` (candidate가 union_box 기반인지)
- `candidate_oracle_iou@0.7` (GoldenSet/소량 GT 있을 때)
- `proposal_rescue` (teacher seed 주입으로 oracle 향상 여부)

### 8.2 집계 지표(모드×AR×super_cat)

- `decision_keep/minimal/crop_rate` by subject_mode
- `headroom/lookroom/horizon/context_fail_rate` by subject_mode
- **NEW: `bg_selected_rate`**  
  - 정의: `c2_primary_bg_like==true` 인 비율 (scene/landscape에서 높으면 정상일 수 있으나, object 모드에서 높으면 치명적)
- **NEW: `multi_subject_detect_rate`, `union_used_rate`**
- `mean/p95 candidate_count` by subject_mode
- `teacher_seed_winrate` / `proposal_rescue_rate` by subject_mode
- `score_norm_stats` (z-score/percentile)

### 8.3 Stress Set(회귀 테스트) 생성 규칙(추천)
- landscape(하늘/바다 큰 비중), 야경, 인테리어 대칭, texture/background, multi-object(비인물) 등을 자동 샘플링하여  
  릴리즈마다 고정 평가로 돌립니다.

---

## 9. Future Works(데이터셋/모델 모두)

### 9.1 데이터셋 Future Works
- **GoldenSet 구축(학습 금지)**: 모드×난이도 균형(2k~10k), AR 5~8개, why_tags 포함
- **OCR 트랙(text_document) 정식 편입**: 텍스트 유지/잘림 패널티를 GT로 고정
- **PICD/Composition Embedding 강화(v1.7 연장)**: comp24 분포 붕괴 감시 + “구도 다양성”을 데이터 KPI로 관리
- **Video/preview 확장**: keyframe 라벨 + 트래킹 안정성(temporal consistency) 데이터 생성
- **Outpainting anchor(GenCrop류) 제한적 도입**: 상위 aesthetic 1~5%에만 적용하여 구도 prior 확보

### 9.2 모델 Future Works
- AR‑조건부 cropping + policy head + rationale head **멀티태스크 통합**
- subject_mode router + MoE(전문가 모델)  
- multi-subject set transformer(집합 기반)  
- 온디바이스 NPU를 위한 distillation/quantization(추후)

---

## 10. 구현 체크리스트(패치 적용 순서)

1) `c2_seg.py`를 Top‑N + importance_score 출력하도록 확장  
2) `enrich_subject_mode_jsonl.py` 추가(merged precompute에 routing 주입)  
3) `generate_candidates.py`에 subject_mode 분기 + union/scene anchors 적용  
4) `score_teacher.py`에서 policy_id 기반 가중치 스케줄 적용  
5) `qa_teacher_report.py`에 모드별 KPI/게이트 추가  
6) Stress Set 자동 생성 및 회귀 테스트 파이프라인 추가

---

## 부록 A. subject_mode router 스켈레톤(패치 수준)

```python
# src/routing/subject_mode_router.py (신규)
from dataclasses import dataclass

@dataclass
class RoutingOut:
    subject_mode: str
    conf: float
    reasons: list
    flags: dict
    shot_type: str | None
    policy_id: str
    primary_source: str
    primary_idx: int
    union_box: list | None

def route_subject_mode(tags_norm, super_cat, feats):
    n_person = feats.get("c3_person_count", 0)
    text_ratio = feats.get("ocr_text_area_ratio", None)

    mode, conf, reasons = infer_subject_mode(tags_norm, super_cat, n_person, text_ratio)
    flags = {
        "has_person": n_person > 0,
        "has_copyspace_tag": has_any(tags_norm, ["copy space","negative space","background"]),
        "has_text_heavy": (text_ratio is not None and text_ratio > 0.08),
    }

    # C2 topN 기반 single/multi 후판정
    c2 = feats.get("c2_seg", [])
    c2_sorted = sorted(c2, key=lambda x: x.get("importance_score", -1), reverse=True)
    primary_idx = -1
    union_box = None
    primary_source = "none"

    if mode.startswith("portrait"):
        primary_source = "c3"
        primary_idx = 0
        # union_box는 c3 union에서 생성(별도)
    elif mode.startswith("object"):
        primary_source = "c2"
        primary_idx = 0 if len(c2_sorted) else -1
        # importance 상위가 근접하면 object_multi로 override + union 사용
        if len(c2_sorted) >= 2 and (c2_sorted[0]["importance_score"] - c2_sorted[1]["importance_score"] < 0.15):
            mode = "object_multi"
            conf = max(conf, 0.7)
            reasons.append("top2_close=>multi")
            union_box = union_xyxy([c2_sorted[0]["box"], c2_sorted[1]["box"]])
    elif mode in ["scene_landscape","background_texture_copyspace","text_document"]:
        primary_idx = -1
        primary_source = "none"

    policy_id = {
        "portrait_single": "portrait_single_v1",
        "portrait_group": "portrait_group_v1",
        "object_single": "object_v1",
        "object_multi": "object_multi_v1",
        "scene_landscape": "scene_v1",
        "background_texture_copyspace": "copyspace_v1",
        "text_document": "text_v1",
    }.get(mode, "generic_v1")

    return RoutingOut(mode, conf, reasons, flags, shot_type=None,
                     policy_id=policy_id, primary_source=primary_source,
                     primary_idx=primary_idx, union_box=union_box)
```

---

## 부록 B. C2 importance scoring 스켈레톤(패치 수준)

```python
def compute_importance(mask_inst, img_w, img_h, det_matches, person_union_box=None, mode="generic"):
    area_ratio = mask_inst["area"] / float(img_w*img_h)
    cx, cy = mask_inst["center_xy"]
    dist = mask_inst["center_dist_norm"]
    border = mask_inst["border_touch"]
    det_conf = mask_inst.get("det_conf", 0.0)
    person_overlap = iou(mask_inst["box"], person_union_box) if person_union_box else 0.0
    bg_like = mask_inst.get("bg_like", False)

    mu, sigma = 0.18, 0.12
    area_pref = math.exp(-((area_ratio - mu)**2) / (2*sigma**2))
    center_pref = 1.0 - dist
    border_pref = 1.0 - border

    if mode.startswith("portrait"):
        w_det,w_area,w_center,w_border,w_person,w_bg = 0.6,0.3,0.3,0.2,1.0,1.0
    elif mode.startswith("object"):
        w_det,w_area,w_center,w_border,w_person,w_bg = 0.8,0.4,0.2,0.2,0.2,1.0
    elif mode.startswith("scene"):
        w_det,w_area,w_center,w_border,w_person,w_bg = 0.2,0.1,0.1,0.1,0.0,0.2
    else:
        w_det,w_area,w_center,w_border,w_person,w_bg = 0.5,0.3,0.2,0.2,0.2,0.8

    score = (
        w_det*det_conf + w_area*area_pref + w_center*center_pref + w_border*border_pref +
        w_person*person_overlap - w_bg*(1.0 if bg_like else 0.0)
    )
    return score
```

---

## 부록 C. QA report 집계(모드별) 스켈레톤

```python
# qa_teacher_report.py 내부에 추가할 집계 예시
group_keys = ["subject_mode","target_ar","super_cat"]
agg = df.groupby(group_keys).agg(
    n=("image_id","count"),
    keep_rate=("decision_type", lambda s: (s=="keep_full").mean()),
    minimal_rate=("decision_type", lambda s: (s=="minimal_crop").mean()),
    crop_rate=("decision_type", lambda s: (s=="crop").mean()),
    bg_selected_rate=("c2_primary_bg_like","mean"),
    union_used_rate=("union_used","mean"),
    headroom_fail=("headroom_pass", lambda s: 1.0 - s.mean()),
    lookroom_fail=("lookroom_pass", lambda s: 1.0 - s.mean()),
    horizon_fail=("horizon_pass", lambda s: 1.0 - s.mean()),
).reset_index()
```

---

### 끝.

