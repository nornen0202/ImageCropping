# Lookroom/Gaze QA 정리 보고서

작성일: 2026-05-29

## 1. 결론

현재 pipeline에서 Gaze-LLE/Gazelle backend는 3D yaw/pitch degree를 직접 예측하는 모델로 쓰이지 않는다. gaze target heatmap의 peak를 2D image coordinate로 읽고, head center와 gaze target의 x/y 차이를 proxy로 변환한다. 따라서 `left/right/center` 기준도 degree가 아니라 normalized 2D displacement threshold다.

Lookroom은 side gaze인 `left/right`에서만 의미가 있다. 정면, 위, 아래, unknown gaze는 `lookroom_na`로 두고, `r_lookroom`과 `C_lookroom` score target도 `None`으로 제외하는 것이 맞다. 이번 구현에서 이 score target 누수도 차단했다.

## 2. Gaze-LLE 결과는 2D인가 3D인가

현재 C6 quality-first 경로의 Gazelle/Gaze-LLE backend는 gaze target estimation 모델로 사용된다. 즉 "사람이 scene 안의 어느 2D 위치를 보고 있는가"를 heatmap으로 예측하고, 그 peak를 `gaze_target_norm_xy`로 저장한다.

구현 흐름:

1. Gaze-LLE/Gazelle이 scene heatmap과 optional in/out score를 예측한다.
2. heatmap peak를 normalized image coordinate `(gx, gy)`로 읽는다.
3. head box center `(cx, cy)`와 비교해 `dx = gx - cx`, `dy = gy - cy`를 만든다.
4. `yaw_proxy = clamp(dx * 2.0, -1.0, 1.0)`, `pitch_proxy = clamp(dy * 2.0, -1.0, 1.0)`로 저장한다.

이 값은 degree 단위 3D angle이 아니다. "2D 화면에서 head 중심 대비 gaze target이 어느 방향에 있는가"를 나타내는 proxy다.

## 3. left/right/center 기준

현재 기준은 다음이다.

| Path | 기준 |
| --- | --- |
| Gazelle/Gaze-LLE path | `dx > 0.05`면 `right`, `dx < -0.05`면 `left`, 그 사이면 `center` |
| C3 pose proxy fallback | `yaw_proxy > 0.12`면 `right`, `< -0.12`면 `left`, 그 사이면 `center` |

따라서 "2D gaze 각도 기준 몇 도까지 left/right/center인가"라는 질문에 대한 답은, 현재 구현상 degree 기준은 없다는 것이다. normalized 2D displacement threshold로 분류한다. 실제 3D gaze vector 또는 yaw/pitch degree 모델을 도입한다면 별도의 degree threshold policy를 새로 정의해야 한다.

## 4. 정면, 위, 아래 gaze에서 lookroom 계산

`compute_lookroom_term()`은 `gaze_dir`가 `left/right`가 아니면 lookroom을 활성화하지 않는다.

- `gaze_dir == center`: `value=None`, `pass=True`, label은 `lookroom_na`
- `gaze_dir == unknown`: `value=None`, `pass=True`, label은 `lookroom_na`
- 위/아래 gaze: `pitch_proxy`에는 반영될 수 있지만 lookroom scoring은 x축 여백만 보므로 `left/right`가 아니면 비활성

위/아래 gaze는 headroom이나 subject placement와 연결될 수는 있지만, lookroom penalty로 처리하면 안 된다. Lookroom은 "시선이 향하는 좌/우 방향에 crop 내부 여백이 충분한가"를 보는 항목이다.

## 5. 오른쪽 gaze 각도가 다양할 때 계산

현재 lookroom ratio는 gaze angle magnitude가 아니라 direction과 crop 내부 anchor margin으로 계산한다.

$$ r_{\text{lookroom}}=\frac{m_{\text{forward}}+\epsilon}{m_{\text{back}}+\epsilon} $$

오른쪽을 보는 경우:

- `forward = crop_x2 - anchor_x`
- `back = anchor_x - crop_x1`

왼쪽을 보는 경우:

- `forward = anchor_x - crop_x1`
- `back = crop_x2 - anchor_x`

따라서 오른쪽 10도와 40도를 구분해 lookroom target을 다르게 주지는 않는다. `right`로 분류된 뒤에는 "얼마나 오른쪽을 보고 있느냐"보다 "crop 안에서 오른쪽 여백이 뒤쪽 여백보다 충분한가"가 중요하다.

후속 개선으로 강한 side gaze일수록 더 많은 lookroom을 요구하려면 `abs(dx)`, heatmap confidence, in/out score를 이용해 target range를 동적으로 키우는 정책이 필요하다. 현재 구현에는 넣지 않았다.

## 6. 정면 인물 사진의 학습 라벨 영향

정면 인물 사진은 lookroom label이 `lookroom_na`로 처리된다. class loss 쪽에서는 `na`, `n/a`, `none`, `null`, `nan`이 meaningful label이 아니므로 training mask에서 제외된다.

기존 score path에서는 side gaze가 아닌데도 `C_lookroom=0.5` 같은 중립 score target이 남을 수 있었다. 이 값은 명시적 negative는 아니지만, lookroom이 적용되지 않는 정면 사진까지 score head가 학습하게 만드는 불필요한 신호다.

이번 패치에서 다음 조건을 만족할 때만 lookroom score component를 저장하도록 바꿨다.

```text
lookroom_active = gaze_dir in {left, right} AND value is not None
```

그 외에는 다음처럼 빠진다.

- `scores.components.r_lookroom = None`
- `macro_components.C_lookroom = None`
- detail score target의 `C_lookroom` valid mask도 0

따라서 정면 인물 사진은 lookroom 부족 negative도, lookroom 적절 positive도 학습하지 않는다. 이는 lookroom score 모델 학습에 부정적인 영향을 주는 `na` 샘플 누수를 줄이는 방향이다.

## 7. Runtime 표시 Gate와 Applicability Head

runtime에서 특정 checklist 또는 score를 보여줄지는 다음 세 값을 분리해서 봐야 한다.

| Gate | 의미 |
| --- | --- |
| `route_applicable` | subject mode상 해당 항목이 의미 있는가 |
| `observed_active` | teacher/feature artifact가 실제 관측값을 제공했는가 |
| `predicted_active` | runtime model이 해당 항목을 활성으로 예측했는가 |

질문했던 "이 부분도 인식하는 head를 추가해서 학습해야 하는가?"에 대한 답은 다음이다.

- offline teacher artifact나 debug viz처럼 관측값이 있는 환경에서는 `route_applicable AND observed_active`로 충분하다.
- 실제 runtime에서 teacher 관측값 없이 모델 출력만으로 표시 여부를 정해야 한다면 `predicted_active`가 필요하다.
- 이 repo에는 이미 `checklist_applicability_head`가 있다. 새 head를 추가하는 것이 아니라 기존 applicability head를 학습/사용하면 된다.

권장 display rule:

- 제품 runtime: `display = route_applicable AND predicted_active`
- teacher/debug artifact 화면: `display = route_applicable AND observed_active`
- review/debug UI: `display = route_applicable AND (observed_active OR predicted_active)`

## 8. 구현 및 검증

구현 파일:

- `src/score_teacher.py`: side gaze가 아닐 때 `r_lookroom`/`C_lookroom`을 `None`으로 비활성화
- `src/mobilecropnet_v4/data.py`: `C_lookroom=None`이면 detail score target valid mask에서 제외
- `src/mobilecropnet_v4/model.py`: 기존 `checklist_applicability_head`가 runtime active/display 예측에 사용 가능
- `tests/test_teacher_scorer.py`: center gaze에서 `lookroom_na`, `r_lookroom is None`, `C_lookroom is None` 회귀 테스트

검증:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m pytest tests/test_teacher_scorer.py -q
```

통합 검증에서는 `tests/test_teacher_scorer.py`, `tests/test_sstk_media_type_audit.py`, `tests/test_filter_sstk_dataset_export.py`를 함께 실행한다.

## 9. 참고 문헌과 출처

- Ryan et al., "Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders", CVPR 2025. <https://openaccess.thecvf.com/content/CVPR2025/papers/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.pdf>
- Gaze-LLE arXiv. <https://arxiv.org/abs/2412.09586>
- Gazelle/Gaze-LLE GitHub. <https://github.com/fkryan/gazelle>
