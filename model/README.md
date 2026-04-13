# model

딥러닝 모델과 클러스터링 실험 코드입니다.

**주요 파일**
- `2_stage.py`: 2-stage 계층 분류 파이프라인
- `Clustering.py`: 군집 기반 분류 실험
- `Dual_SE.py`: Depthwise+SE CNN 모델
- `Multi_scale_SE.py`: 멀티스케일 SE 변형 모델
- `lightweight_fft11_model.py`: 경량 FFT-11 모델
- `train_fft11.py`: FFT-11 학습 스크립트
- `eval_fft11_model.py`: FFT-11 평가 스크립트
- `eval_class_model.py`: 클래스 수준 평가
- `cluster_hdbscan.py`: HDBSCAN 분석
- `ffc_hdbscan.py`: HDBSCAN 시각화
- `*.pth`: 학습된 가중치
