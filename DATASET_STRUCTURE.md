# Dataset Structure Reference

이 문서는 `data/` 디렉터리에 저장된 각 이미지 크로핑 연구용 데이터셋의 **파일 구조**, **어노테이션 포맷**, 그리고 **통합 변환 방법**을 설명합니다.

---

## 1. CPC (Comparative Photo Composition)

**출처**: Good View Hunting (Stony Brook)

### 파일 구조

```
data/cpc/extracted/CPCDataset/
├── images/                          # 12,569장 JPG
│   ├── ava_obj_0_10711.jpg
│   └── ...
├── CollectedAnnotationsRaw/         # 10,797개 어노테이션 (.txt)
│   ├── ava_obj_0_10711.jpg.txt
│   └── ...
├── raw_annotations/                 # 5개 MTurk 원본 배치 파일
│   ├── assignments-p2_10010_Approved.txt
│   └── ...
├── pdefined_anchor.pkl              # VFN 학습용 predefined anchor boxes
└── ReadMe.txt
```

### 어노테이션 포맷 (`CollectedAnnotationsRaw/*.txt`)

각 파일은 **double-JSON encoded** 문자열입니다:

```json
{
  "bboxes": [[x1, y1, x2, y2], ...],   // 24개 candidate crop boxes (pixel coords)
  "scores": [[s1, s2, ...], ...]        // 6명 annotator × 24 crops, 점수 0~5
}
```

- **Bbox 좌표계**: `[x1, y1, x2, y2]` = `[left, top, right, bottom]` (pixel)
- **Score**: 각 annotator가 각 crop에 매긴 점수 (0=최악, 5=최고)
- `pdefined_anchor.pkl`: VFN 모델 학습에 사용된 사전 정의 anchor boxes (pickle)

### 통합 변환 시 매핑

| 원본 필드 | 통합 포맷 |
|-----------|-----------|
| `bboxes[i]` → `[x1,y1,x2,y2]` | `crop_x1, crop_y1, crop_x2, crop_y2` (pixel) |
| `mean(scores[:, i])` | `score` (0~5 → 0.0~1.0 정규화) |

---

## 2. XPView (Expert-level Photo Viewing)

**출처**: Good View Hunting (Stony Brook)

### 파일 구조

```
data/xpview/extracted/XPDataset/
├── images/                          # 1,045장 JPG
│   ├── UMD_PQ_27773.jpg
│   ├── init1K_COCO_train2014_000000436979.jpg
│   ├── FCDB_3294446343_070405d6ff_b.jpg
│   └── ...
├── annotations_strict/              # 992개 어노테이션 (.txt)
│   └── <image_name>.txt
├── candidate_crops/                 # 1,045개 candidate crop 목록 (.txt)
│   └── <image_name>.txt
├── collected_annotations/           # 3개 sub-directory (3명의 전문가 원본)
├── effective_image_list.txt         # 유효 이미지 목록 (992줄)
└── ReadMe.txt
```

### 어노테이션 포맷

**`annotations_strict/*.txt`** (CSV, 헤더 없음):
```
190,190,650,768,1.0000
156,117,1024,696,1.0000
285,285,1000,768,1.0000
0,0,511,767,0.0000
```

| 컬럼 | 의미 |
|------|------|
| col 0 | `h_min` (y1, top) |
| col 1 | `w_min` (x1, left) |
| col 2 | `h_max` (y2, bottom) |
| col 3 | `w_max` (x2, right) |
| col 4 | `score` (expert vote count, 0~3: 0 = no expert approved, 3 = all 3 approved) |

**`candidate_crops/*.txt`** (CSV, 헤더 없음):
```
304,380,917,768
```
→ `h_min,w_min,h_max,w_max` (score 없음, candidate 목록)

### 통합 변환 시 매핑

| 원본 필드 | 통합 포맷 |
|-----------|-----------|
| `h_min, w_min, h_max, w_max` | `crop_x1=w_min, crop_y1=h_min, crop_x2=w_max, crop_y2=h_max` |
| `score` (0~3 votes) | `score = votes / 3.0` (0.0~1.0 정규화) |

---

## 3. FLMS (Fang et al., MM 2014)

**출처**: http://fangchen.org/proj_page/FLMS_mm14/

### 파일 구조

```
data/flms/extracted/
├── image/                           # 500장 JPG
│   ├── 1010203631_Large.jpg
│   └── ...
├── 500_image_dataset.mat            # MATLAB annotation
└── readme                           # 포맷 설명
```

### 어노테이션 포맷 (`500_image_dataset.mat`)

MATLAB `.mat` 파일로, `img_gt` 구조체 배열 포함:

- **`img_gt[i].img`**: 파일명 (문자열)
- **`img_gt[i].bbox`**: N×4 행렬, 각 행 = `[h_min, w_min, h_max, w_max]` (MATLAB 1-indexed)

> **주의**: MATLAB 좌표는 1-indexed. Python으로 변환 시 그대로 사용하거나 명시적으로 -1 보정 필요.
> Turker 실패 시 음수 좌표가 포함될 수 있음 → 필터링 필요.

### 통합 변환 시 매핑

| 원본 필드 | 통합 포맷 |
|-----------|-----------|
| `h_min, w_min, h_max, w_max` | `crop_x1=w_min, crop_y1=h_min, crop_x2=w_max, crop_y2=h_max` |
| (점수 없음) | `score = 1.0` (all ground truth) |

---

## 4. GAIC / GAIC_v2 (Grid Anchor Image Cropping)

**출처**: https://github.com/HuiZeng/Image-Cropping

### 파일 구조

```
data/GAIC/                              data/GAIC_v2/
├── images/                             ├── images/
│   ├── train/ (1,036 jpg)              │   ├── train/ (2,636 jpg)
│   └── test/  (200 jpg)                │   ├── val/   (200 jpg)
├── annotations/ (1,236 txt)            │   └── test/  (500 jpg)
                                        ├── annotations/
                                        │   ├── train/ (2,636 txt)
                                        │   ├── val/   (200 txt)
                                        │   └── test/  (500 txt)
                                        └── untitled.m
```

### 어노테이션 포맷 (`annotations/*.txt`)

Space-delimited, 헤더 없음. 이미지당 ~90줄 (grid anchor crops):

```
  28   43  480  811 2.14
  28   43  480  896 3.43
  28   43  537  725 1.00
```

| 컬럼 | 의미 |
|------|------|
| col 0 | `y1` (top) |
| col 1 | `x1` (left) |
| col 2 | `y2` (bottom) |
| col 3 | `x2` (right) |
| col 4 | `MOS score` (Mean Opinion Score, ~1.0~5.0) |

### 통합 변환 시 매핑

| 원본 필드 | 통합 포맷 |
|-----------|-----------|
| `y1, x1, y2, x2` | `crop_x1=x1, crop_y1=y1, crop_x2=x2, crop_y2=y2` |
| `MOS` (1.0~5.0) | `score` = `(MOS - 1) / 4` (0.0~1.0 정규화) |

---

## 5. Unsplash Lite

**출처**: https://unsplash.com/data

### 파일 구조

```
data/unsplash-research-dataset-lite-latest/   # 원본 (TSV)
├── photos.csv000          # 25,000 photos
├── keywords.csv000        # photo-keyword pairs
├── collections.csv000     # photo-collection pairs
├── colors.csv000          # dominant colors per photo
├── conversions.csv000     # search conversion logs
├── DOCS.md / README.md / TERMS.md

data/unsplash/                                 # 정제본 (unsplash_prepare.py 출력)
├── images/
│   └── <photo_id>.jpg     # CDN에서 다운로드된 리사이즈 이미지
├── annotations/
│   └── <photo_id>.json    # 통합 어노테이션 sidecar
├── index.jsonl            # 전체 JSONL 인덱스
├── photos_refined.tsv     # 정제된 photos 테이블
├── keywords_pivot.tsv     # photo_id별 keywords
└── colors_pivot.tsv       # photo_id별 dominant color
```

### 어노테이션 포맷 (`annotations/<photo_id>.json`)

```json
{
  "photo_id": "oSf8ePoG9NU",
  "photo_width": 4000,
  "photo_height": 2667,
  "aspect_ratio": 1.5,
  "orientation": "landscape",
  "photo_featured": true,
  "ai_description": "black road ...",
  "keywords": "fog,forest,frost,...",
  "keyword_count": 42,
  "dominant_hex": "121312",
  "dominant_color_name": "black",
  "dominant_coverage": 0.384
}
```

> **참고**: Unsplash Lite에는 crop bounding box가 없습니다. 이미지 자체가 "좋은 구도"의 예시로 사용됩니다. 통합 포맷에서는 **전체 이미지 = best crop** (`crop = [0,0,W,H], score=1.0`)으로 처리합니다.

---

## 통합 어노테이션 포맷 (Unified Format)

모든 데이터셋을 하나의 JSON Lines (`.jsonl`) 파일로 변환합니다.

### 스키마

```json
{
  "image_id": "cpc_ava_obj_0_10711",
  "dataset": "cpc",
  "image_path": "data/cpc/extracted/CPCDataset/images/ava_obj_0_10711.jpg",
  "image_width": 600,
  "image_height": 428,
  "crops": [
    {
      "crop_x1": 42,
      "crop_y1": 0,
      "crop_x2": 469,
      "crop_y2": 427,
      "score": 0.4333,
      "source": "annotator_mean"
    }
  ]
}
```

### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `image_id` | string | `{dataset}_{filename_stem}` 형식의 고유 ID |
| `dataset` | string | 데이터셋 이름 (cpc, xpview, flms, gaic, gaic_v2, unsplash) |
| `image_path` | string | 이미지 파일 상대 경로 |
| `image_width` | int | 이미지 너비 (px) |
| `image_height` | int | 이미지 높이 (px) |
| `crops[].crop_x1` | int | crop 좌상단 x (left) |
| `crops[].crop_y1` | int | crop 좌상단 y (top) |
| `crops[].crop_x2` | int | crop 우하단 x (right) |
| `crops[].crop_y2` | int | crop 우하단 y (bottom) |
| `crops[].score` | float | 정규화된 평점 (0.0 ~ 1.0) |
| `crops[].source` | string | 점수 출처 (annotator_mean, expert, mos, gt, full_image) |

### Score 정규화

| 데이터셋 | 원본 범위 | 변환 공식 |
|----------|-----------|-----------|
| CPC | 0~5 (int) | `mean(scores) / 5.0` |
| XPView | 0~3 (expert votes) | `votes / 3.0` |
| FLMS | 없음 (GT crop) | `1.0` |
| GAIC/GAIC_v2 | 1.0~5.0 (MOS) | `(MOS - 1) / 4` |
| Unsplash | 없음 (full image) | `1.0` |

---

## 통합 변환 명령

```bash
python -m crop_datasets.unify \
    --data_root ./data \
    --output    ./data/unified_crops.jsonl

# 빠른 테스트 (데이터셋별 10장만)
python -m crop_datasets.unify \
    --data_root ./data \
    --output    ./data/unified_crops.jsonl \
    --limit 10

# 특정 데이터셋만
python -m crop_datasets.unify \
    --data_root ./data \
    --output    ./data/unified_crops.jsonl \
    --datasets cpc gaic_v2
```

**출력 통계 예시**:
```
Dataset     | Images | Crops   | Avg crops/img
------------|--------|---------|-------------
cpc         | 10,797 | 259,128 | 24.0
xpview      |    992 |   6,456 | ~6.5
flms        |    500 |   5,000 | ~10
gaic        |  1,236 | 111,240 | ~90
gaic_v2     |  3,336 | 300,240 | ~90
unsplash    | 25,000 |  25,000 | 1.0
```
