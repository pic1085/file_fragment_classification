# scripts

실행 가능한 파이프라인/시각화 스크립트 모음입니다.

**파일**
- `main.py`: 베이스라인 파이프라인(로드→학습→평가→혼동쌍)
- `train_hierarchical.py`: 계층 분류기 학습 및 베이스라인 비교
- `extract_features.py`: 특징 추출 후 `.npz` 저장
- `visualize_confusion.py`: 혼동쌍/분포/tsne 시각화 생성
