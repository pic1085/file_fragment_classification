"""
main.py
베이스라인 파이프라인 실행 진입점.

실행 예시:
  python main.py --data_dir /path/to/Data_set

npz 키를 모를 때 먼저 실행:
  python main.py --inspect_only --data_dir /path/to/Data_set
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import os
import argparse
from tqdm import tqdm

from core.dataset import load_fft75_npz, inspect_npz
from core.paths import DATA_DIR, OUTPUT_DIR
from models.baseline_classifier import BaselineClassifier
from analysis.confusion_analysis import plot_confusion_matrix, extract_confusing_pairs


# 전체 파이프라인 단계 정의
_STAGES = [
    "1단계: 데이터 로드",
    "2단계: 모델 학습",
    "3단계: Confusion Matrix",
    "4단계: 혼동 쌍 추출",
]


def run_baseline(data_dir: str, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    stage_bar = tqdm(_STAGES, desc="파이프라인",
                     bar_format="{l_bar}{bar:25}| {n_fmt}/{total_fmt} 단계",
                     ncols=80, position=0, leave=True)

    for stage in stage_bar:
        stage_bar.set_description(f"[{stage}]")

        # ── 1. 데이터 로드 ──────────────────────────────────────────
        if "로드" in stage:
            print(f"\n{'='*60}\n{stage}\n{'='*60}")
            X_train, X_val, X_test, y_train, y_val, y_test, classes = \
                load_fft75_npz(data_dir=data_dir)

        # ── 2. 모델 학습 + 평가 ────────────────────────────────────
        elif "학습" in stage:
            print(f"\n{'='*60}\n{stage}\n{'='*60}")
            clf = BaselineClassifier(n_estimators=100, max_depth=30, scale_features=True,
                                     train_batch=10)
            clf.fit(X_train, y_train, classes)

            print("\n[검증셋]")
            clf.evaluate(X_val, y_val, desc="검증셋 평가", verbose=True)

            print("\n[테스트셋]")
            test_result = clf.evaluate(X_test, y_test,
                                       desc="테스트셋 평가", verbose=True)
            clf.save(os.path.join(output_dir, "rf_baseline"))

        # ── 3. Confusion Matrix ────────────────────────────────────
        elif "Confusion" in stage:
            print(f"\n{'='*60}\n{stage}\n{'='*60}")
            y_pred = test_result['y_pred']
            save_path = os.path.join(output_dir, "confusion_matrix.png")
            print("confusion matrix 생성 중...")
            plot_confusion_matrix(y_test, y_pred, classes, save_path=save_path)

        # ── 4. 혼동 쌍 추출 ───────────────────────────────────────
        elif "혼동" in stage:
            print(f"\n{'='*60}\n{stage}\n{'='*60}")
            confusing_pairs = extract_confusing_pairs(
                y_test, y_pred, classes, top_k=20, min_confusion=3
            )
            csv_path = os.path.join(output_dir, "confusing_pairs.csv")
            with open(csv_path, "w") as f:
                f.write("class_a,class_b,a_to_b,b_to_a,total,confusion_rate\n")
                for p in tqdm(confusing_pairs, desc="CSV 저장",
                              bar_format="{l_bar}{bar:20}{r_bar}", ncols=65):
                    f.write(f"{p.class_a},{p.class_b},"
                            f"{p.a_misclassified_as_b},{p.b_misclassified_as_a},"
                            f"{p.total_confusion},{p.confusion_rate:.4f}\n")
            print(f"혼동 쌍 CSV 저장: {csv_path}")

    print(f"\n{'='*60}")
    print("모든 단계 완료.")
    print(f"  결과 디렉토리: {os.path.abspath(output_dir)}")
    print(f"  다음 단계: analysis.confusion_analysis.analyze_pair_features() 로")
    print(f"  각 혼동 쌍 엔트로피/카이제곱 분포 비교 → 논문 Figure 2, 3")
    print(f"{'='*60}\n")

    return clf, confusing_pairs, classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFT-75 베이스라인 파이프라인")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--inspect_only", action="store_true",
                        help="npz 키 구조만 확인하고 종료")
    args = parser.parse_args()

    if args.inspect_only:
        for fname in ["train.npz", "val.npz", "test.npz"]:
            path = os.path.join(args.data_dir, fname)
            if os.path.exists(path):
                inspect_npz(path)
            else:
                print(f"[없음] {path}")
    else:
        run_baseline(data_dir=args.data_dir, output_dir=args.output_dir)
