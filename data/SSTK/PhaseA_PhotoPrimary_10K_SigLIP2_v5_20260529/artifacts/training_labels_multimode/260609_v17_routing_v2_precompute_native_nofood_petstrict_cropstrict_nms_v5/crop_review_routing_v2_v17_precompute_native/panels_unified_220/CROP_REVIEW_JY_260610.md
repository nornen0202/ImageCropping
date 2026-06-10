# Multi-Mode Crop Review 

## 잘못 라벨링 된 것처럼 보이는 것들 -> 원인 분석 후 구체적인 수정/보완 방안을 강구하나거나 의도한 결과(고칠필요가 없는)이면 이유를 설명하기 바람. 

- 원본 이미지에는 사람 전신이 전부 보이는데 image_route가 face_headshot 혹은 scene으로 분류됨: review_pond5_image_72309116, review_pond5_image_86336860, review_pond5_image_114358384, review_pond5_image_120403442, review_pond5_image_143683670, review_pond5_image_144387333
- 사람 2명이 보이는데 image_route: scene 혹은 person_single_* 으로 분류됨: review_pond5_image_92200704, review_pond5_image_114639246
- 사람 1명만 보이는데 image_route: person_group 으로 분류됨:  review_pond5_image_101326243, review_pond5_image_143273930
- review_pond5_image_136488302 의 crop #3, #5 를 보면 피사체(개)가 너무 tight 하게 (상단 혹은 좌측 여백이 너무 없어보임) 보임.

## 수정할 사항들
- 시각화 결과에 대해서 crop bbox 라인 색상을 crop_mode 별로 다르게 눈에 띄는 색상으로 변경하도록 하고, 우측 패널의 크롭 결과 테두리 부분과 매칭시키기 바람.