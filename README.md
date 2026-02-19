# crop_datasets (ImageCropping)

Image cropping / composition 연구용 공개 데이터셋을 **동일한 UX**로 관리하는 Python 패키지 + CLI입니다.

- 자동 다운로드 가능: **CPC, XPView, FLMS**
- 권한/정책/링크 안정성 이슈로 수동 등록 중심: **FCDB, SACD, Unsplash, GAICD, ICIT**

> 본 도구는 데이터셋 라이선스/ToS 준수를 우선합니다. 접근 권한이 필요한 데이터는 우회 다운로드를 제공하지 않습니다.

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Google Drive 자동 다운로드까지 쓰려면
pip install -e '.[gdrive]'
```

## CLI 빠른 시작

```bash
crop-datasets list --root ./data
crop-datasets download flms --root ./data
crop-datasets register sacd --path /data/SACD --root ./data
crop-datasets verify sacd --root ./data
crop-datasets prepare unsplash --unsplash_root /data/unsplash_export --root ./data
```

`python -m crop_datasets.cli ...` 형태도 동일하게 사용할 수 있습니다.

## 주요 명령

### 1) list
지원 데이터셋, 자동/수동 유형, 라이선스 링크, 용량 추정, 설치 상태 출력.

```bash
crop-datasets list --root ./data
```

### 2) download `<name>`
자동 다운로드 가능한 데이터셋만 수행합니다.

- `http`: 스트리밍, retry/backoff, resume(Range), 진행률 표시
- `gdrive`: `gdown` 기반 resume 다운로드
- archive(tar/zip) 자동 해제 (idempotent)

예시:

```bash
crop-datasets download cpc --root ./data
crop-datasets download xpview --root ./data
crop-datasets download flms --root ./data
```

다운로드 실패 시 에러 메시지에 다음 액션(수동 다운로드 URL + register 안내)을 제공합니다.

### 3) register `<name> --path <local_path>`
수동 다운로드/권한 필요 데이터셋의 로컬 경로 등록 + 구조 검증.

```bash
crop-datasets register fcdb --path /mnt/datasets/FCDB --root ./data
crop-datasets register sacd --path /mnt/datasets/SACD --root ./data
crop-datasets register gaicd --path /mnt/datasets/GAICD --root ./data
crop-datasets register icit --path /mnt/datasets/ICIT --root ./data
```

### 4) verify `<name>`
registry에 등록된 경로를 기준으로 필수 파일/디렉터리 구조 확인.

```bash
crop-datasets verify flms --root ./data
crop-datasets verify unsplash --root ./data
```

### 5) prepare unsplash
Unsplash는 자동 다운로드를 지원하지 않습니다.
로컬 export를 받아 구조 검증 후 인덱스 캐시(`index.jsonl`)를 생성합니다.

```bash
crop-datasets prepare unsplash --unsplash_root /data/unsplash_export --root ./data
```

## 지원 데이터셋 정책

1. **CPC** (Good View Hunting, GDrive file id: `1TMvuCSONEN1_9y7KnzKgy_7_fSFTHzyO`)
   - 모드: auto (`gdrive`)
2. **XPViewDataset** (GDrive file id: `1DpNY_Fb9eCabwROYF02eMzy4gK3I4Pzj`)
   - 모드: auto (`gdrive`)
3. **FLMS**
   - 모드: auto (`http`)
4. **FCDB**
   - 모드: script/manual
   - 공식 배포 방식(Flickr API/ToS 준수)으로 사용 권장
5. **SACD**
   - 모드: manual 우선
   - 링크 만료 시 수동 다운로드 후 register
6. **Unsplash (GenCrop/stock 기반)**
   - 모드: manual only (접근 권한 필요)
7. **GAICD/GAIC**
   - 모드: manual only
8. **ICIT**
   - 모드: manual only

## 상태/재현성

- 기본 저장 루트: `./data` (옵션 `--root`로 변경)
- registry: `<root>/registry.json`
- manifest: `datasets.yaml`
- 기록 항목: 설치 경로, 다운로드 유형, 소스 URL, 검증 결과, 업데이트 시각

## 테스트

```bash
pytest
```

테스트는 네트워크를 실제로 사용하지 않고 mocking/샘플 파일만 사용합니다.

## 주의사항

- 비공개/권한 필요한 데이터를 우회 다운로드하지 않습니다.
- 링크 만료/권한 오류 시, 에러 메시지에 수동 다운로드 + `register` 경로를 안내합니다.
- 각 데이터셋별 라이선스/약관 링크는 `LICENSES.md` 참고.
