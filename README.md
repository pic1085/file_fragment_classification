# FFT-75 DEV Workspace

![Python](docs/badges/python.svg)

이 레포지토리는 FFT-75 파일 타입 분류를 위한 실험/파이프라인을 모은 작업 공간입니다.
바이트 단위 특징 추출, 베이스라인 분류기, 계층적 분류기, 혼동쌍 분석, 시각화, 딥러닝 실험까지 한 곳에서 정리했습니다.

**무엇을 진행했나요?**
- `.npz` 데이터셋 로드 및 클래스 매핑 정리
- 바이트 히스토그램/통계/바이그램/매직바이트 기반 특징 추출
- RandomForest 베이스라인 학습 및 평가
- 카테고리→서브클래스 계층적 분류기 학습 및 비교
- 혼동쌍 추출 및 혼동행렬/분포 시각화
- 2-stage/클러스터링/경량 CNN 등 확장 실험

**출력된 결과물(예시)**
- `outputs/confusion_matrix.png`: 베이스라인 혼동행렬
- `outputs/fig1_confusion_rate_bar.png`: 혼동률 막대 그래프
- `outputs/fig2_feature_distributions.png`: 혼동쌍 feature 분포 비교
- `outputs/fig3_histogram_overlay.png`: 바이트 히스토그램 오버레이
- `outputs/fig4_tsne_humanreadable.png`: Human-readable 클래스 t-SNE
- `outputs/` 내 CSV/NPY 등 비이미지 결과물은 로컬 생성물이며 `.gitignore`로 제외됨

**실행 방법**
- `python scripts/main.py --data_dir /path/to/Data_set`
- `python scripts/train_hierarchical.py --data_dir /path/to/Data_set`
- `python scripts/extract_features.py --data_dir /path/to/Data_set`
- `python scripts/visualize_confusion.py --data_dir /path/to/Data_set`

**데이터셋**
- FFT-75 (File Fragment Type, FFT-75 Dataset)
- 링크: https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset
- 라이선스/이용조건: 데이터포트 페이지 기준

루트에 호환용 엔트리도 제공됩니다.
- `python main.py`
- `python train_hierarchical.py`
- `python extract_features.py`
- `python visualize_confusion.py`

**폴더 구조**
- `core/`: 공통 유틸(경로, 데이터 로더, 클래스명, 특징 추출)
- `analysis/`: 혼동쌍/혼동행렬 분석 유틸
- `models/`: 베이스라인/계층 분류기 등 클래식 ML 모델
- `scripts/`: 파이프라인 실행 스크립트
- `Preprocessing/`: 데이터 점검/서브셋 생성 유틸
- `model/`: 딥러닝 모델 및 클러스터링 실험
- `N/`: n-gram 및 2-stage 실험
- `unknown/`: 불확실 샘플/보정 관련 CNN 실험
- `outputs/`: 실행 결과물(이미지 위주, 비이미지 산출물은 gitignore)
- `result/`: 학습된 모델 가중치
- `local/`: 로컬 경로 오버라이드

**경로 설정**
- `core/paths.py`가 기본 경로를 관리합니다.
- 개인 환경에 맞춘 경로는 `local/paths.py`에서 오버라이드하세요.
