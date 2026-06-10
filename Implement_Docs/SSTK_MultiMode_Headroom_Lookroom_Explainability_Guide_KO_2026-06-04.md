# SSTK Multi-Mode Headroom / Lookroom Explainability Guide

이 문서는 `SSTK_MultiMode_TrainingLabels_Enhanced_Onboarding_KO_v1_0.md`의 SC08 내용을 실제 PhaseA 산출물 기준으로 풀어 쓴 실무자용 가이드다. ChatGPT 생성 이미지가 아니라 `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529`의 원본 이미지, v14 multimode label JSON, 실제 crop box를 사용했다.

## 1. 이 문서를 보는 목적

처음 보는 엔지니어는 `q_head`, `q_look`, `headroom_value`, `lookroom_value`, checklist label, why-tag가 서로 다른 항목처럼 보일 수 있다. 실제로는 하나의 crop을 아래 순서로 해석한다.

1. crop box가 정해진다.
2. person/group/face 계열이면 머리 또는 얼굴 anchor를 찾는다.
3. crop top과 head top 사이의 비율로 headroom을 계산한다.
4. side gaze가 있으면 gaze 앞쪽/뒤쪽 여백 비율로 lookroom을 계산한다.
5. 연속 score인 `q_head`, `q_look`을 만들고, threshold로 사람이 읽는 checklist label을 파생한다.
6. label 상태에 따라 `headroom_ok`, `headroom_violation`, `lookroom_ok`, `lookroom_violation` 같은 why-tag를 붙인다.

## 2. 사용한 산출물

| 항목 | 경로 |
|---|---|
| PhaseA 원본 이미지 | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/images/` |
| v14 label JSON | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010/label_json/multimode_labels_full.json` |
| explainability policy | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010/explainability_label_policy.json` |
| 이 문서용 panel 생성 스크립트 | `src/scripts/build_phasea_headroom_lookroom_guide_assets.py` |
| 생성 asset 디렉터리 | `Implement_Docs/assets_phasea_v14_headroom_lookroom_guide/` |

재생성 명령:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/build_phasea_headroom_lookroom_guide_assets.py
```

## 3. 전체 contact sheet

아래 이미지는 실제 PhaseA v14 positive annotation에서 뽑은 대표 예시다. 각 패널은 왼쪽에 원본 이미지와 geometry overlay, 가운데에 실제 crop 결과, 오른쪽에 해당 annotation의 score/label/tag를 표시한다.

![PhaseA v14 headroom lookroom contact sheet](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headroom_lookroom_contact_sheet.jpg){ width=100% }

Legend:

- 빨간 box: 실제 학습 label crop box.
- 파란 box/line/band: head 또는 face 기준 headroom 영역.
- 노란 점/화살표: gaze anchor와 gaze 방향.
- 초록 선: gaze 방향 앞쪽 lookroom margin.
- 회색 선: gaze 뒤쪽 margin.
- 오른쪽 텍스트: `score_mode`, `q_head`, `q_look`, checklist label, why-tag.

## 4. Headroom 계산과 라벨

Headroom은 crop 상단에서 머리 top까지의 상대 여백이다.

$$ headroom\_value = \frac{y_{head\_top} - y_{crop\_top}}{h_{crop}} $$

`q_head`는 이 값이 mode별 목표값에 가까울수록 높아지는 연속 score다. 구현 기준 목표값은 face `0.16`, single person `0.12`, group `0.10`이다. label은 score 자체가 아니라 `headroom_value`의 threshold로 파생된다.

| mode | tight | ok | loose |
|---|---:|---:|---:|
| `face` | `< 0.03` | `0.03 ~ 0.26` | `> 0.26` |
| `single_person_*` | `< 0.04` | `0.04 ~ 0.20` | `> 0.20` |
| `group_*` | `< 0.035` | `0.035 ~ 0.18` | `> 0.18` |

해석:

- `headroom_ok`: 머리 위 여백이 mode 기준으로 자연스럽다. why-tag에 `headroom_ok`, `head_top_safe`가 붙는다.
- `headroom_tight`: crop top이 머리/얼굴 top에 너무 가깝거나 일부가 crop 밖으로 나갈 수 있다. why-tag에 `headroom_violation`이 붙는다.
- `headroom_loose`: 머리 위 빈 공간이 mode 기준보다 크다. 인물 scale이 작아지거나 crop 의도가 느슨해질 수 있다.
- person/group/face가 아닌 mode에서는 headroom label이 `na`다.

### 4.1 Headroom OK

![face balanced headroom lookroom](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_01_face_balanced_headroom_lookroom.jpg){ width=100% }

이 예시는 `face` mode, `16:9` crop이다. `headroom_value=0.159565`로 face target `0.16`에 거의 맞고, `q_head=0.999985`, label은 `headroom_ok`다. 오른쪽 label에 `head_top_safe`, `headroom_ok` why-tag가 같이 붙어 있으므로 score, checklist, why-tag가 일관된다.

### 4.2 Headroom Tight

![headroom tight](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_04_headroom_tight.jpg){ width=100% }

주의: 이 panel은 legacy guide sample이며 group aggregate gaze marker가 함께 보인다. 이 섹션에서 확인할 대상은 gaze arrow가 아니라 파란 head line과 crop top 사이의 headroom이다. group mode의 단일 gaze marker 해석 문제는 `review_jy_260604_followup_report_KO.md` 6.1의 gaze anchor audit을 따른다.

이 예시는 `group_center` mode의 tight 사례다. `headroom_value=0.019989`이고 group tight threshold `<0.035`에 걸려 `headroom_tight`가 된다. `q_head=0.411009`로 낮고, why-tag에는 `headroom_violation`이 붙는다. 실무 검수에서는 파란 head line이 crop top에 붙어 있는지, crop 결과에서 머리 위 공간이 답답한지 확인하면 된다.

### 4.3 Headroom Loose

![headroom loose](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_05_headroom_loose.jpg){ width=100% }

이 예시는 `single_person_center`, `1:1` crop이다. `headroom_value=0.239945`로 single-person loose threshold `>0.20`을 넘기 때문에 `headroom_loose`다. `q_head=0.692802`는 완전 실패는 아니지만, `headroom_ok`보다는 낮다. crop 결과에서 머리 위 배경이 넓고 subject scale이 느슨해지는지 확인한다.

## 5. Lookroom 계산과 라벨

Lookroom은 gaze 방향이 `left` 또는 `right`로 신뢰 가능할 때만 적용된다. gaze가 `center`, `unknown`, 또는 side direction이 없으면 `lookroom`은 `na`이고 학습/표시 mask도 `0`이다.

Side gaze가 있는 경우:

$$ lookroom\_value = \frac{margin_{gaze\_forward}}{margin_{gaze\_backward}} $$

`q_look`은 이 비율이 target `1.45` 근처일수록 높다. label threshold는 아래와 같다.

| label | 기준 |
|---|---|
| `lookroom_insufficient` | `< 1.05` |
| `lookroom_adequate` | `1.05 ~ 2.50` |
| `lookroom_excessive` | `> 2.50` |
| `na` | person/group/face가 아니거나 side gaze가 없음 |

해석:

- `lookroom_adequate`: 인물이 바라보는 쪽 여백이 뒤쪽보다 적절히 크다. why-tag에 `lookroom_ok`가 붙는다.
- `lookroom_insufficient`: gaze 앞쪽이 짧아 시선이 crop edge에 막혀 보인다. why-tag에 `lookroom_violation`이 붙고, 너무 심하면 `lookroom_cut` hard reject가 될 수 있다.
- `lookroom_excessive`: gaze 앞쪽 여백이 지나치게 커서 subject가 한쪽에 밀리거나 crop이 비어 보일 수 있다.

### 5.1 Lookroom Adequate

![single person balanced lookroom](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_02_single_person_balanced_headroom_lookroom.jpg){ width=100% }

이 예시는 `single_person_center`, `FREE` crop이다. gaze가 `right`이고 `lookroom_value=1.45383`으로 target `1.45`에 거의 맞다. `q_look=0.999964`, label은 `lookroom_adequate`, why-tag는 `lookroom_ok`다. 초록 선이 회색 선보다 약간 긴지 확인하면 된다.

같은 이미지 `sstk_image_789389968`를 기준으로 headroom/lookroom threshold를 모두 비교한 부록 예시는 [10. 부록 A](#10-부록-a-같은-이미지로-보는-headroomlookroom-threshold)를 참조한다.

### 5.2 Lookroom Insufficient

![lookroom insufficient](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_06_lookroom_insufficient.jpg){ width=100% }

주의: 이 panel은 legacy guide sample이며 group aggregate gaze가 섞여 있어 gaze circle/arrow가 단일 인물의 정확한 lookroom anchor처럼 보이지 않을 수 있다. lookroom threshold 자체는 아래 설명처럼 해석하되, 실무 교육용 기준 예시는 [10. 부록 A](#10-부록-a-같은-이미지로-보는-headroomlookroom-threshold)의 same-image single-person panel을 우선 사용한다.

이 예시는 gaze가 `right`인데 `lookroom_value=0.187505`로 앞쪽 여백이 뒤쪽보다 훨씬 짧다. `q_look=0.019535`, label은 `lookroom_insufficient`, why-tag는 `lookroom_violation`이다. 실제 crop 결과에서도 인물이 바라보는 오른쪽 공간이 부족하게 보이는지 확인한다.

### 5.3 Lookroom Excessive

![lookroom excessive](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_07_lookroom_excessive.jpg){ width=100% }

주의: 이 panel은 `group_rot` legacy sample이며 member face가 여러 개인 group aggregate gaze가 단일 yellow arrow로 표시된다. 이 때문에 화살표 위치/방향이 실제 주 피사체 gaze처럼 보이지 않을 수 있다. group lookroom은 후속 정책에서 별도 적용 조건을 두는 것이 맞고, 기본 lookroom 학습/해석 예시는 single-person/face panel을 우선한다.

이 예시는 `group_rot`, `FREE` crop이다. gaze가 `left`이고 `lookroom_value=63.820856`으로 앞쪽 여백이 지나치게 크다. `q_look=0.0`, label은 `lookroom_excessive`, why-tag는 `lookroom_violation`이다. 이런 경우 crop이 subject를 한쪽에 몰아두고 빈 공간을 과도하게 남길 수 있다.

### 5.4 Lookroom Not Applicable

![lookroom not applicable](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_08_lookroom_not_applicable.jpg){ width=100% }

이 예시는 `face`, `1:1` crop이지만 gaze가 `center`다. 따라서 `lookroom_label=na`, `lookroom applicable=0`, `q_look=0.5` 중립값이다. 이 값은 “lookroom이 나쁘다”가 아니라 “side gaze 기준으로 판단하지 않는다”는 뜻이다. 학습 시에도 `lookroom` label head는 이 항목을 valid target으로 보지 않아야 한다.

## 6. Checklist와 Why-Tag의 관계

`checklist_labels`는 사람이 읽는 상태값이고, `why_tags`는 label과 score를 요약해 crop 선택 또는 실패 이유를 설명하는 sparse tag다.

| Checklist 상태 | 대표 why-tag | 의미 |
|---|---|---|
| `headroom_ok` | `headroom_ok`, `head_top_safe` | 머리 위 여백이 mode 기준 안에 있음 |
| `headroom_tight`, `headroom_loose` | `headroom_violation` | 머리 위 여백이 너무 작거나 큼 |
| `lookroom_adequate` | `lookroom_ok` | side gaze 앞쪽 여백이 적절함 |
| `lookroom_insufficient`, `lookroom_excessive` | `lookroom_violation` | side gaze 앞쪽 여백이 부족하거나 과도함 |
| `lookroom=na` | 없음 | side gaze가 없어 lookroom 판단을 하지 않음 |

중요한 점은 why-tag가 별도 truth가 아니라 checklist/score를 사람이 빠르게 읽게 하는 요약이라는 것이다. 모델 학습에서는 `C_headroom`, `C_lookroom`, `headroom_ratio`, `lookroom_ratio` 같은 연속 target을 우선 회귀하고, label/tag는 보조 학습 또는 평가/debug 용도로 쓰는 편이 안전하다.

## 7. PhaseA v14 분포

`multimode_labels_full.json` 기준 전체 annotation은 `167,387`개, image는 `10,000`장이다. positive annotation 기준 headroom/lookroom 분포는 아래와 같다.

| 항목 | positive count |
|---|---:|
| `headroom=na` | 70,427 |
| `headroom_ok` | 13,169 |
| `headroom_loose` | 2,626 |
| `headroom_tight` | 332 |
| `lookroom=na` | 75,206 |
| `lookroom_adequate` | 5,407 |
| `lookroom_insufficient` | 5,289 |
| `lookroom_excessive` | 652 |

`na`가 많은 이유는 landscape/object mode에서는 headroom/lookroom을 적용하지 않고, lookroom은 person/group/face mode라도 side gaze가 없으면 적용하지 않기 때문이다.

## 8. 실무 검수 체크리스트

1. `mode_name`이 person/group/face 계열인지 확인한다. 아니면 headroom/lookroom은 대체로 `na`가 맞다.
2. `lookroom applicable=1`인데 gaze 화살표가 실제 시선 방향과 반대로 보이면 C6 gaze/precompute 품질을 의심한다.
3. `headroom_tight`인데 실제 crop에서 머리 위가 충분해 보이면 `head_bbox_norm_xyxy` 또는 `head_y_norm` 추정 오류를 확인한다.
4. `headroom_loose`인데 crop이 사진학적으로 자연스러운 environmental portrait라면 mode/shot-type threshold가 너무 엄격한지 검토한다.
5. `lookroom_insufficient`는 단순히 사람이 중앙이 아니라는 뜻이 아니다. 시선 앞쪽 여백이 뒤쪽보다 부족한지를 봐야 한다.
6. `lookroom_excessive`는 빈 공간이 항상 나쁘다는 뜻이 아니다. copy-space 의도가 있는 이미지는 별도 mode/copyspace 신호와 함께 판단해야 한다.
7. 최종 학습에서는 `checklist_applicable` mask를 반드시 적용한다. `lookroom=na`를 부정 label처럼 학습하면 center gaze/unknown gaze 샘플에서 잘못된 penalty가 생긴다.

## 9. 생성된 개별 asset 목록

| case | image | panel |
|---|---|---|
| face balanced | `sstk_image_1019674000` | `assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_01_face_balanced_headroom_lookroom.jpg` |
| single-person balanced | `sstk_image_789389968` | `assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_02_single_person_balanced_headroom_lookroom.jpg` |
| group balanced | `pond5_image_134178914` | `assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_03_group_balanced_headroom_lookroom.jpg` |
| headroom tight | `sstk_image_1972270181` | `assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_04_headroom_tight.jpg` |
| headroom loose | `pond5_image_58696654` | `assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_05_headroom_loose.jpg` |
| lookroom insufficient | `sstk_image_1633699804` | `assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_06_lookroom_insufficient.jpg` |
| lookroom excessive | `bigstock_image_428715728` | `assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_07_lookroom_excessive.jpg` |
| lookroom not applicable | `bigstock_image_308173657` | `assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_08_lookroom_not_applicable.jpg` |

원본 manifest와 machine-readable summary는 각각 아래에 저장했다.

- `Implement_Docs/assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headroom_lookroom_guide_manifest.csv`
- `Implement_Docs/assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headroom_lookroom_guide_assets_summary.json`

## 10. 부록 A. 같은 이미지로 보는 Headroom/Lookroom Threshold

이 부록은 5.1의 `sstk_image_789389968` 하나만 사용해 headroom과 lookroom 상태를 모두 비교한다. 첫 번째 panel은 실제 v14 positive annotation이다. `lookroom_insufficient` panel은 같은 이미지에 존재하는 실제 v14 negative candidate를 사용했다. `headroom_tight`, `headroom_loose`, `lookroom_excessive` panel은 같은 이미지와 같은 subject/gaze anchor를 유지하고 crop box만 인위적으로 바꾼 설명용 synthetic crop이다. 이 synthetic crop들은 학습 라벨이 아니라 threshold 동작을 설명하기 위한 시각화 산출물이다.

![same image headroom lookroom variants](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_02_same_image_variant_contact_sheet.jpg){ width=100% }

### 10.1 기준 Annotation

기준 annotation은 section 5.1의 실제 positive crop이다.

| 항목 | 값 |
|---|---|
| source image | `sstk_image_789389968` |
| base annotation | `id=165313` |
| mode / AR | `single_person_center` / `FREE` |
| 실제 crop | `[0.0, 0.0, 512.0, 605.471]` |
| headroom | `headroom_value=0.121665`, `q_head=0.999929`, `label=headroom_ok` |
| lookroom | `lookroom_value=1.45383`, `q_look=0.999964`, `label=lookroom_adequate` |
| why-tags | `headroom_ok`, `head_top_safe`, `lookroom_ok` |

이 crop은 single-person headroom target `0.12`와 lookroom target `1.45`에 거의 맞는다. 따라서 headroom과 lookroom이 모두 high score이며, 사람이 읽는 checklist도 `headroom_ok`, `lookroom_adequate`로 파생된다.

### 10.2 같은 이미지 Variant 요약

| variant | crop source | crop xyxy | headroom value / label | lookroom value / label | 해석 |
|---|---|---|---|---|---|
| actual ok/adequate | 실제 positive | `[0.0, 0.0, 512.0, 605.471]` | `0.121665 / headroom_ok` | `1.45383 / lookroom_adequate` | 머리 위 여백과 gaze 앞쪽 여백이 모두 target 근처다. |
| headroom tight | synthetic | `[0.0, 60.029, 512.0, 605.471]` | `0.025 / headroom_tight` | `1.45383 / lookroom_adequate` | crop top을 아래로 내려 머리 line과 crop top 사이가 너무 좁아진다. |
| headroom loose | synthetic | `[0.0, 0.0, 512.0, 306.936]` | `0.24 / headroom_loose` | `1.45383 / lookroom_adequate` | crop bottom을 올려 crop height 대비 머리 위 여백 비율이 커진다. |
| lookroom insufficient | 실제 negative candidate | `[4.875, 0.0, 414.99, 614.371]` | `0.119903 / headroom_ok` | `1.012554 / lookroom_insufficient` | gaze가 오른쪽인데 오른쪽 forward margin이 threshold `1.05`보다 짧다. |
| lookroom excessive | synthetic | `[107.538, 0.0, 512.0, 605.471]` | `0.121665 / headroom_ok` | `3.0 / lookroom_excessive` | 왼쪽 edge를 subject 쪽으로 당겨 오른쪽 forward margin이 과도하게 커진다. |

### 10.3 Headroom Tight를 보는 법

![same image headroom tight](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_02_same_image_02_synthetic_headroom_tight.jpg){ width=100% }

이 synthetic crop은 원래 crop의 top을 아래로 내려 `headroom_value=0.025`가 되도록 만들었다. single-person mode에서는 `<0.04`가 `headroom_tight`이므로 label이 tight로 바뀐다. lookroom 방향의 x 좌표는 건드리지 않았기 때문에 `lookroom_value=1.45383`과 `lookroom_adequate`는 유지된다.

실무자가 확인할 부분은 파란 head line이 crop top에 거의 붙는지다. 이 경우 why-tag는 `headroom_violation`으로 바뀌고, head top safety가 약해진다.

### 10.4 Headroom Loose를 보는 법

![same image headroom loose](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_02_same_image_03_synthetic_headroom_loose.jpg){ width=100% }

이 synthetic crop은 top은 그대로 두고 bottom을 올려 crop height를 줄였다. 같은 head line이라도 crop height가 짧아지면 `headroom_value`가 커진다. 여기서는 `0.24`가 되어 single-person loose threshold `>0.20`을 넘는다.

주의할 점은 headroom loose가 “절대 픽셀 기준으로 위쪽 공간이 많다”는 뜻만은 아니라는 것이다. 정의가 crop height 대비 비율이므로, bottom을 올려 tighter crop을 만들 때도 상대 headroom은 loose로 갈 수 있다. 따라서 시각 검수에서는 crop 결과에서 머리 위 공간이 얼굴/상반신 scale 대비 과하게 보이는지 같이 봐야 한다.

### 10.5 Lookroom Insufficient를 보는 법

![same image lookroom insufficient](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_02_same_image_04_actual_negative_lookroom_insufficient.jpg){ width=100% }

이 panel은 synthetic이 아니라 같은 이미지의 실제 v14 negative candidate다. gaze는 `right`이고, 오른쪽 forward margin이 왼쪽 backward margin과 거의 같거나 더 짧아져 `lookroom_value=1.012554`가 된다. threshold가 `<1.05`이므로 label은 `lookroom_insufficient`다.

`q_look=0.623447`로 완전히 0에 가깝지는 않다. 이유는 이 crop이 threshold 바로 아래의 경계 사례이기 때문이다. checklist label은 threshold로 끊기지만, regression score는 target `1.45`에서 얼마나 멀어졌는지를 연속값으로 표현한다. 이 차이 때문에 학습에서는 score regression과 label/tag를 분리해서 해석해야 한다.

### 10.6 Lookroom Excessive를 보는 법

![same image lookroom excessive](assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_02_same_image_05_synthetic_lookroom_excessive.jpg){ width=100% }

이 synthetic crop은 오른쪽 gaze-forward 공간이 지나치게 커지도록 왼쪽 edge를 subject 쪽으로 당겼다. `lookroom_value=3.0`으로 excessive threshold `>2.50`을 넘고, `q_look=0.002653`까지 떨어진다. headroom y 좌표는 그대로 유지했기 때문에 `headroom_ok`는 유지된다.

이 예시는 “시선 앞 공간이 많으면 항상 좋은가?”에 대한 반례다. lookroom은 target보다 부족해도 문제지만, 과도하면 subject가 한쪽으로 밀리고 crop의 균형이 깨질 수 있다.

### 10.7 부록 Asset

| 산출물 | 경로 |
|---|---|
| same-image contact sheet | `Implement_Docs/assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_02_same_image_variant_contact_sheet.jpg` |
| same-image manifest | `Implement_Docs/assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_02_same_image_variant_manifest.csv` |
| same-image summary | `Implement_Docs/assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_02_same_image_variant_summary.json` |
