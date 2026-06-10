# SSTK Curated Media-Type Audit 및 Lookroom/Gaze QA 분리 안내

작성일: 2026-05-29

이 문서는 초기에 SSTK Curated media-type 감사와 Lookroom/Gaze QA를 함께 기록했지만, 두 주제의 의사결정 축이 달라서 아래 두 문서로 분리했다.

| 주제 | 문서 |
| --- | --- |
| SSTK Curated pool media-type 감사, v4 camera-primary 정책, SigLIP2/CLIP prompt classifier, Phase A pre-sampling gate | `Implement_Docs/SSTK_Curated_MediaType_Audit_KO_2026-05-29.md` |
| Gaze-LLE 출력 해석, 2D gaze 기준, lookroom 라벨/score 비활성화, runtime display gate | `Implement_Docs/Lookroom_Gaze_QA_KO_2026-05-29.md` |

운영 기준은 다음과 같다.

- Curated pool의 main 학습 입력은 deterministic v4 audit의 `filtered_sstk_photo_primary.parquet` 또는 Phase A pre-sampling media-type gate를 통과한 photo-primary 후보를 사용한다.
- Lookroom은 side gaze인 `left/right`에서만 활성화한다. 정면, 위, 아래, unknown gaze는 `lookroom_na`이며 `r_lookroom`/`C_lookroom` score target도 학습에서 제외한다.
