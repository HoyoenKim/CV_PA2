1. 모든 기능 구현은 main.py에 되어 있습니다.
2. save_images() 을 통해서 이미지들의 특징점을 추출합니다.
3. growing_step() 을 통해서 그로잉 스텝을 진행합니다.
이때 생긴 3d point를 points_3d.obj에 저장합니다.
4. 이 결과를 이용해서 번들 어드저스트먼트를 수행합니다.
이때 생긴 3d point를 ba_points_3d.obj에 저장합니다.