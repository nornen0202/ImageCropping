# Multi-Mode Crop Review Guide

이 시각화는 비교 기준 없이 multi-mode 라벨 자체를 정성 검수하기 위한 산출물이다.

## 읽는 법

- 왼쪽 패널: 원본 이미지 위에 선택된 positive crop box를 표시한다.
- 오른쪽 패널: `(target AR, crop/query mode)` 조합별 score가 가장 높은 best crop 1개만 카드 형태로 보여준다.
- 패널 상단의 컬러 `IMAGE ROUTE` 바는 image-level classifier target이며 placement를 포함하지 않는다.
- 패널 상단의 `positive crop/query modes`와 crop card의 `crop_mode=`는 실제 crop annotation/query mode이다.
- crop 카드 하단의 `crop_route=` 값은 annotation-level `route_family_v2/person_shot_type/placement_intent/context_intent`이며, 긴 텍스트는 카드 안에서 줄바꿈된다.
- `v16 summary aux`는 image row의 보조 요약값이며 image-level primary target이 아니다.
- crop 카드의 query/AR/score 텍스트는 crop 이미지 위가 아니라 별도 header/footer 영역에 배치했다.
- cyan 영역은 subject core, orange box는 support/envelope, magenta box는 member subject를 의미한다.
- crop box 번호와 오른쪽 crop 카드 번호가 대응한다.
- mode별 crop/카드 경계선은 서로 구분되는 고정 팔레트를 사용한다.

## 산출물

- selected panels: `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v17_routing_v2_precompute_native_nofood_petstrict_cropstrict_nms_v5/crop_review_routing_v2_v17_precompute_native/panels_unified_220/panels`
- contact sheets: `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v17_routing_v2_precompute_native_nofood_petstrict_cropstrict_nms_v5/crop_review_routing_v2_v17_precompute_native/panels_unified_220/v17_precompute_native_crop_review_contact_sheet_part001.jpg`
- manifest JSON: `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v17_routing_v2_precompute_native_nofood_petstrict_cropstrict_nms_v5/crop_review_routing_v2_v17_precompute_native/panels_unified_220/v17_precompute_native_crop_review_manifest.json`
- manifest CSV: `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v17_routing_v2_precompute_native_nofood_petstrict_cropstrict_nms_v5/crop_review_routing_v2_v17_precompute_native/panels_unified_220/v17_precompute_native_crop_review_manifest.csv`

## 요약

- selected images: 220
- written panels: 220
- max crops per image: 12
- crop columns: 3
