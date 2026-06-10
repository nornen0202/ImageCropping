# crop_datasets (ImageCropping)

Image cropping / composition 연구용 공개 데이터셋을 **동일한 UX**로 관리하는 Python 패키지 + CLI입니다.

| 다운로드 방식 | 데이터셋 |
|---|---|
| 자동 (GDrive) | **CPC, XPView** |
| 자동 (HTTP) | **FLMS** |
| 수동 등록 | **FCDB, SACD, Unsplash, GAICD, ICIT** |

> 본 도구는 데이터셋 라이선스/ToS 준수를 우선합니다. 접근 권한이 필요한 데이터는 우회 다운로드를 제공하지 않습니다.

---

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Google Drive 자동 다운로드(CPC, XPView)까지 사용하려면:

```bash
pip install -e '.[gdrive]'
```

> **가상환경 이름이 `ImageCropping_Py310`인 경우**: 위 대신 해당 환경에서 `pip install -e '.[gdrive]'`를 실행하세요.

---

## CLI 빠른 시작

```bash
# 지원 데이터셋 목록 + 상태 확인
crop-datasets list

# 자동 다운로드 (--root 기본값: ./data)
crop-datasets download cpc
crop-datasets download xpview
crop-datasets download flms

# 수동 등록 (다운로드 후 로컬 경로 등록)
crop-datasets register sacd --path /data/SACD
crop-datasets register gaicd --path /data/GAICD
crop-datasets register icit --path /data/ICIT

# 등록 확인
crop-datasets verify flms
crop-datasets verify sacd

# 저장 위치 변경 (--root)
crop-datasets download flms --root /mnt/storage/datasets
```

`python -m crop_datasets.cli ...` 형태도 동일하게 사용할 수 있습니다.

---

## 주요 명령 상세

### `list` — 전체 데이터셋 현황 출력

```bash
crop-datasets list
crop-datasets list --root /mnt/storage/datasets
```

지원 데이터셋 목록, 다운로드 유형, 설치 상태, 라이선스, 용량 추정치를 출력합니다.

---

### `download <name>` — 자동 다운로드

자동 다운로드 가능한 데이터셋(`http`, `gdrive`)만 수행합니다.

- **스트리밍 + resume**: Range 헤더 기반 이어받기
- **retry/backoff**: 네트워크 오류 시 자동 재시도 (최대 3회)
- **자동 압축 해제**: `.tar`, `.tar.gz`, `.zip` 지원 (idempotent)

```bash
# CPC (Google Drive, ~2-5 GB) — gdown 필요
crop-datasets download cpc

# XPView (Google Drive, ~2-5 GB) — gdown 필요
crop-datasets download xpview

# FLMS (HTTP, ~1 GB) — 추가 의존성 없음
crop-datasets download flms
```

다운로드 실패 시 에러 메시지에 **수동 다운로드 URL + register 안내**를 함께 출력합니다.

---

### `register <name> --path <local_path>` — 수동 등록

수동으로 받은 데이터셋의 로컬 경로를 등록하고 구조를 검증합니다.

```bash
crop-datasets register fcdb  --path /data/FCDB
crop-datasets register sacd  --path /data/SACD
crop-datasets register gaicd --path /data/GAICD
crop-datasets register icit  --path /data/ICIT
```

---

### `verify <name>` — 등록 경로 구조 검증

```bash
crop-datasets verify flms
crop-datasets verify cpc
```

---

### `prepare unsplash` — Unsplash 로컬 인덱싱

Unsplash Dataset은 자동 다운로드를 지원하지 않습니다. 승인된 export를 받은 후 인덱스를 생성하세요.

```bash
crop-datasets prepare unsplash --unsplash_root /data/unsplash_export
# --max-rows 옵션으로 인덱싱 행 수 제한 (기본 50000)
crop-datasets prepare unsplash --unsplash_root /data/unsplash_export --max-rows 100000
```

생성 결과: `<root>/unsplash/index.jsonl`

---

## 데이터셋별 준비 안내

### CPC / XPView
- **다운로드**: 자동 (`gdrive`) — `pip install 'crop_datasets[gdrive]'` 필요
- **명령**: `crop-datasets download cpc` / `crop-datasets download xpview`

### FLMS
- **다운로드**: 자동 (`http`)
- **명령**: `crop-datasets download flms`

### FCDB (Flickr Cropping Dataset)
- **방식**: 공식 Flickr 스크립트로 이미지 다운로드 + annotation 파일 별도 제공
- **준비**:
  1. [FCDB 공식 repo](https://github.com/gy20073/FCDB) 클론
  2. Flickr API key 발급 후 환경변수 설정:
     ```bash
     export FLICKR_KEY=your_key
     export FLICKR_SECRET=your_secret
     ```
  3. 공식 다운로드 스크립트 실행
  4. 등록:
     ```bash
     crop-datasets register fcdb --path /data/FCDB
     ```

### SACD (Subject-Aware Composition Dataset)
- **방식**: 수동 (링크 만료 가능성 있음)
- **준비**: [프로젝트 페이지](https://www.yingcong.me/projects/subject-aware-composition/)에서 수동 다운로드 후:
  ```bash
  crop-datasets register sacd --path /data/SACD
  ```

### Unsplash Lite Dataset

Lite Dataset은 상업/비상업 모두 사용 가능하며, 직접 다운로드할 수 있습니다 (~700 MB).

**Step 1 — TSV 아카이브 다운로드**

```bash
mkdir -p ./data/unsplash-research-dataset-lite-latest
cd ./data/unsplash-research-dataset-lite-latest
curl -L "https://unsplash.com/data/lite/latest" -o unsplash-lite.zip
unzip unsplash-lite.zip
cd -
```

**Step 2 — 이미지 다운로드 + 어노테이션 정제**

```bash
# pandas 필요
pip install pandas tqdm

python -m crop_datasets.datasets.unsplash_prepare \
    --dataset_root ./data/unsplash-research-dataset-lite-latest \
    --output_root  ./data/unsplash \
    --img_size 640 \
    --workers 8
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--img_size` | 640 | Unsplash CDN에서 요청할 이미지 너비(픽셀). 0이면 원본 해상도 |
| `--workers` | 8 | 병렬 다운로드 스레드 수 |
| `--limit` | 0 (전체) | N장으로 제한 (테스트용) |
| `--skip-images` | — | 이미지 다운로드 없이 어노테이션/인덱스만 생성 |

**출력 구조**

```
./data/unsplash/
├── images/
│   └── <photo_id>.jpg          # 다운로드된 이미지 (~25,000장)
├── annotations/
│   └── <photo_id>.json         # 사진별 통합 어노테이션 sidecar
├── index.jsonl                 # 전체 JSONL 인덱스 (스트리밍용)
├── photos_refined.tsv          # 정제된 photos 테이블 (aspect_ratio, orientation 포함)
├── keywords_pivot.tsv          # photo_id별 keywords 집계
└── colors_pivot.tsv            # photo_id별 주요 색상
```

**사이드카 JSON 예시** (`annotations/oSf8ePoG9NU.json`):

```json
{
  "photo_id": "oSf8ePoG9NU",
  "photo_width": 4000,
  "photo_height": 2667,
  "aspect_ratio": 1.5,
  "orientation": "landscape",
  "photo_featured": true,
  "ai_description": "black road in between white and brown grass ...",
  "keywords": "fog,forest,frost,landscape,nature,sunrise,sunset,...",
  "keyword_count": 42,
  "dominant_hex": "121312",
  "dominant_color_name": "black",
  "download_ok": true,
  "image_path": "images/oSf8ePoG9NU.jpg"
}
```

> **라이선스**: Unsplash Dataset Terms 준수 필수. 이미지 재배포 불가. 자세한 내용은 [`LICENSES.md`](./LICENSES.md) 참고.

### GAICD / GAIC
- **방식**: 수동 — [HuiZeng/Image-Cropping](https://github.com/HuiZeng/Image-Cropping)
- **명령**: `crop-datasets register gaicd --path /data/GAICD`

### ICIT (InstructCrop)
- **방식**: 수동 (공식 배포 채널 확인 필요)
- **명령**: `crop-datasets register icit --path /data/ICIT`

---

## 상태 관리

| 항목 | 기본값 |
|---|---|
| 데이터 저장 루트 | `./data` (`--root`로 변경) |
| Registry 파일 | `<root>/registry.json` |
| Manifest 파일 | `datasets.yaml` (`--manifest`로 변경) |
| 기록 내용 | 설치 경로, 다운로드 유형, 소스 URL, 검증 결과, 업데이트 시각 |

---

## 테스트

```bash
pip install -e '.[test]'
python -m pytest -v
```

네트워크를 실제로 사용하지 않고 mocking/샘플 파일만 사용합니다.  
13개 테스트 포함: CLI argparse, manifest 파싱, 구조 검증, HTTP resume, tar/zip 추출, GDrive URL 생성.

---

## 주의사항

- 비공개/권한 필요한 데이터를 우회 다운로드하지 않습니다.
- 링크 만료/권한 오류 시, 에러 메시지에 수동 다운로드 + `register` 경로를 안내합니다.
- 각 데이터셋별 라이선스/약관 링크는 [`LICENSES.md`](./LICENSES.md) 참고.
