# Artifact Archive Guide

`data/<DatasetGroup>/<DatasetName>` 아래에 쌓이는 산출물 중, 최신 run만 남기고 이전 run은 같은 그룹의 `*_olds` 디렉터리로 옮겨 보관하는 절차를 정리한 문서입니다.

이번에 실제 적용한 예시는 아래입니다.

- source: `data/GAIC/All`
- cutoff: `260330_r0`
- archive root: `data/GAIC/All_olds`

이 작업 후 `data/GAIC/All/artifacts/training_labels` 에는 `gaic_260330_r0_leftover_ignore_monotonic` 만 남고, 이전 run 및 기존 `training_labels/olds/*` 내용은 `data/GAIC/All_olds` 로 정리되었습니다.

## Added Scripts

- Python core: [src/scripts/archive_old_artifacts.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/archive_old_artifacts.py)
- Shell wrapper: [src/scripts/run_archive_old_artifacts.sh](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/run_archive_old_artifacts.sh)

## What Gets Moved

- 이름 안에 `260330_r0`, `260316_r2`, `260319_cond_detr_v1` 같은 run token 이 들어 있는 산출물
- `--cutoff` 보다 이전 run으로 판단되는 파일/디렉터리
- 기존에 중간 정리용으로 넣어둔 `artifacts/.../olds/*`

다음은 유지됩니다.

- cutoff 이상인 최신 run
- run token 이 없는 공용 디렉터리/파일
- 기본적으로 `artifacts`, `logs` 외 경로
- downstream 입력으로 재사용되는 보호 산출물

현재 보호되는 재사용 산출물은 아래입니다.

- `artifacts/metadata/*`: GAIC/TestImages synthetic caption jsonl 및 summary
- `artifacts/public_teachers/*`: raw/proposal 재주입용 공개 teacher 산출물
- `data/GAIC/All` 한정 `artifacts/candidates/candidates_ar_gaic_260320_r0*`
- `data/GAIC/All` 한정 `artifacts/reports/gaic_benchmark_eval_gaic_260320_r0*`

이동 대상은 원래 dataset root 기준 상대 경로를 유지한 채 sibling archive root 로 갑니다.

- `data/GAIC/All/artifacts/reports/gaic_260324_r2_detailed`
- `data/GAIC/All_olds/artifacts/reports/gaic_260324_r2_detailed`

`training_labels/olds/gaic_260324_r2` 처럼 `olds` 내부에 있던 항목은 `All_olds/artifacts/training_labels/gaic_260324_r2` 로 정규화됩니다.

## Basic Usage

쉘 스크립트 사용:

```bash
src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 data/GAIC/All
```

파이썬 스크립트 직접 사용:

```bash
python3 src/scripts/archive_old_artifacts.py data/GAIC/All --cutoff 260330_r0
```

archive destination 이 이미 있을 때는 아래 옵션을 선택할 수 있습니다.

```bash
src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 --skip-existing data/GAIC/All
src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 --overwrite data/GAIC/All
```

## Dry Run

실제 이동 전에 반드시 dry-run 으로 확인할 수 있습니다.

```bash
src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 --dry-run data/GAIC/All
python3 src/scripts/archive_old_artifacts.py data/SSTK/Test_100 --cutoff 260319_cond_detr_v1 --dry-run
```

## Target Forms

개별 dataset root 를 직접 줄 수 있습니다.

```bash
src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 data/GAIC/All
src/scripts/run_archive_old_artifacts.sh --cutoff 260331_r0 data/TestImages/All
src/scripts/run_archive_old_artifacts.sh --cutoff 260319_cond_detr_v1 data/SSTK/Test_100
```

상위 그룹 디렉터리를 주면 immediate child dataset root 들을 자동으로 찾아 처리합니다.

```bash
src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 --dry-run data/GAIC
src/scripts/run_archive_old_artifacts.sh --cutoff 260331_r0 --dry-run data/TestImages
src/scripts/run_archive_old_artifacts.sh --cutoff 260319_cond_detr_v1 --dry-run data/SSTK
```

## Scope Control

기본 스캔 범위는 `artifacts`, `logs` 입니다. 특정 범위만 제한하고 싶으면 `--scope` 를 반복 지정합니다.

```bash
src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 --scope artifacts data/GAIC/All
src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 --scope logs data/GAIC/All
src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 --scope artifacts --scope logs data/GAIC/All
```

## Recommended Workflow

1. 최신으로 유지할 run token 을 정합니다.
2. dry-run 으로 이동 목록을 확인합니다.
3. 실제 실행합니다.
4. `*_olds` 와 원본 dataset root 를 한 번씩 확인합니다.

예시:

```bash
src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 --dry-run data/GAIC/All
src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 data/GAIC/All
find data/GAIC/All/artifacts/training_labels -maxdepth 2 -type d | sort
find data/GAIC/All_olds -maxdepth 3 -type d | sort | sed -n '1,80p'
```

## Notes

- 기본값은 destination 이 이미 존재하면 스크립트가 중단됩니다.
- `--skip-existing` 는 기존 archive destination 을 유지하고 해당 move 만 건너뜁니다.
- `--overwrite` 는 기존 archive destination 을 삭제한 뒤 source 를 이동합니다.
- 보호된 재사용 산출물이 `*_olds`에 이미 있으면, 실행 시 원래 dataset root 로 없는 파일만 자동 복구합니다.
- 빈 디렉터리는 이동 후 자동 정리됩니다.
- `python3` 외 인터프리터를 쓰려면 `PYTHON_BIN` 환경변수를 지정하면 됩니다.

```bash
PYTHON_BIN=/path/to/python src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 --dry-run data/GAIC/All
```
