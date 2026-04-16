# Public Cropper GAIC v2 Benchmark Evaluation

- annotations_json: `/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/data/Publics/GAIC_v2/annotations_json/instances_test.json`
- image_count: 500
- candidate_count: 43123
- device: `cuda`

## Availability

| method | status | reason / note |
| --- | --- | --- |
| `cacnet` | evaluated | CACNet emits one crop, so candidate scores are IoU to the emitted crop for GAIC annotation-space projection. |
| `cgs` | evaluated |  |
| `gaic` | evaluated | GAIC's bundled Python2/3.5 roi_align/rod_align artifacts are incompatible with Py3.10; the evaluation uses the Py3.10-built CGS extensions with the same API. |
| `gaic_mos_oracle` | reference |  |

## Metric Definitions

- `PCC`, `SRCC`: official GAIC MOS and predicted candidate scores의 이미지별 상관계수를 평균한 값이며 높을수록 좋다.
- `AccK/N`: 모델이 반환한 상위 K개 후보 중 official MOS 상위 N개에 포함되는 비율이며 높을수록 좋다.
- `AccwK/N`: `AccK/N`에 official rank와 반환 순서의 차이를 반영한 rank-weighted 지표이며 높을수록 좋다.
- `top1 MOS`: 모델 top-1 crop의 official MOS이며 높을수록 좋다.
- `MOS regret`: 이미지 내 최고 official MOS와 top-1 MOS의 차이며 낮을수록 좋다.
- `top1 rank pct`: top-1 crop의 official rank percentile이며 1.0에 가까울수록 좋다.
- `-`: 모델 출력 형식상 해당 GAIC candidate-ranking 지표를 원 방식으로 계산할 수 없음을 뜻한다.

## Method Comparison

| method | images | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/5 | Accw4/5 | top1 MOS | top1 rank pct | MOS regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cacnet` | 500 | - | - | - | - | - | - | - | - | - |
| `cgs` | 500 | 0.838836 | 0.799759 | 0.550000 | 0.728000 | 0.462000 | 0.401560 | 3.921020 | 0.898081 | 0.310880 |
| `gaic` | 500 | 0.872629 | 0.845874 | 0.604000 | 0.804000 | 0.494000 | 0.432030 | 3.985300 | 0.930537 | 0.246600 |
| `gaic_mos_oracle` | 500 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 4.231900 | 1.000000 | 0.000000 |

## CACNet Projection Diagnostics

CACNet은 단일 crop만 출력하므로 위 GAIC candidate-ranking 표에서는 해당 지표를 `-`로 표시했다. 아래 값은 예측 crop과 official candidate 간 IoU 투영으로 계산한 참고 진단값이며, 원 GAIC 평가 지표로 해석하면 안 된다.

| images | projected top1 MOS | projected MOS regret | projected top1 rank pct | candidate IoU max | candidate IoU mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 3.734740 | 0.497160 | 0.833031 | 0.896787 | 0.645612 |

## Notes

- `gaic_mos_oracle` is an upper bound that sorts candidates by the official MOS itself.
- CACNet is a single-crop regressor, so native GAIC candidate-ranking metrics are marked as not applicable in the primary comparison table.
- CGS and GAIC are candidate rankers and are scored directly on the official GAIC annotation candidates.
