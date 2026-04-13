"""
baseline_classifier.py
Random Forest 베이스라인 분류기.
warm_start 방식으로 트리를 배치 단위로 추가하여 tqdm 진행바 표시.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Tuple, Optional
from tqdm import tqdm


class BaselineClassifier:

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        n_jobs: int = -1,
        random_state: int = 42,
        scale_features: bool = True,
        train_batch: int = 10,      # tqdm 업데이트 단위 (트리 수)
    ):
        self.n_estimators  = n_estimators
        self.train_batch   = train_batch
        self.scale_features = scale_features
        self.classes_: Optional[List[str]] = None

        # warm_start=True: fit() 호출마다 트리를 누적 추가
        self.rf = RandomForestClassifier(
            n_estimators=train_batch,   # 초기값; fit마다 증가
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight='balanced',
            warm_start=True,
            verbose=0,
        )
        self.scaler = StandardScaler() if scale_features else None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            classes: List[str]) -> None:
        self.classes_ = classes

        if self.scale_features:
            print("특징 스케일링 중...")
            X_train = self.scaler.fit_transform(X_train)

        # warm_start 배치 학습 + tqdm
        total     = self.n_estimators
        batch     = self.train_batch
        n_batches = (total + batch - 1) // batch   # 올림 나눗셈

        current = 0
        with tqdm(total=total, desc="RF 트리 학습",
                  unit="trees", bar_format="{l_bar}{bar:35}{r_bar}",
                  ncols=85) as pbar:
            for _ in range(n_batches):
                current = min(current + batch, total)
                self.rf.n_estimators = current
                self.rf.fit(X_train, y_train)
                pbar.update(min(batch, total - pbar.n))  # 마지막 배치 보정

        print(f"학습 완료: {X_train.shape[0]:,}개 샘플  |  "
              f"{len(classes)}개 클래스  |  {total}그루 트리")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scale_features:
            X = self.scaler.transform(X)
        return self.rf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.scale_features:
            X = self.scaler.transform(X)
        return self.rf.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 desc: str = "평가", verbose: bool = True) -> dict:
        """
        배치 단위 예측으로 tqdm 진행바 표시.
        """
        if self.scale_features:
            X = self.scaler.transform(X)

        batch = 1000
        preds = []
        n = len(X)

        with tqdm(total=n, desc=desc, unit="samples",
                  bar_format="{l_bar}{bar:35}{r_bar}", ncols=85) as pbar:
            for start in range(0, n, batch):
                end = min(start + batch, n)
                preds.append(self.rf.predict(X[start:end]))
                pbar.update(end - start)

        y_pred = np.concatenate(preds)
        acc    = accuracy_score(y, y_pred)
        report = classification_report(
            y, y_pred, target_names=self.classes_, digits=4
        )
        if verbose:
            print(f"\n전체 정확도: {acc:.4f}  ({acc*100:.2f}%)")
            print(report)
        return {'accuracy': acc, 'report': report, 'y_pred': y_pred}

    def feature_importance(self, feature_names: Optional[List[str]] = None,
                           top_k: int = 20) -> List[Tuple[str, float]]:
        importances = self.rf.feature_importances_
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(len(importances))]
        pairs = sorted(zip(feature_names, importances),
                       key=lambda x: x[1], reverse=True)
        top = pairs[:top_k]
        print(f"\n상위 {top_k} 중요 특징:")
        for name, imp in top:
            print(f"  {name}: {imp:.4f}")
        return top

    def save(self, save_dir: str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with tqdm(total=2, desc="모델 저장",
                  bar_format="{l_bar}{bar:20}{r_bar}", ncols=65) as pbar:
            joblib.dump(self.rf, save_dir / "rf_model.pkl"); pbar.update(1)
            if self.scaler:
                joblib.dump(self.scaler, save_dir / "scaler.pkl")
            pbar.update(1)
        print(f"저장 완료: {save_dir}")

    @classmethod
    def load(cls, save_dir: str, classes: List[str]) -> "BaselineClassifier":
        save_dir = Path(save_dir)
        obj = cls.__new__(cls)
        obj.rf      = joblib.load(save_dir / "rf_model.pkl")
        scaler_path = save_dir / "scaler.pkl"
        obj.scaler  = joblib.load(scaler_path) if scaler_path.exists() else None
        obj.scale_features = obj.scaler is not None
        obj.classes_ = classes
        print(f"모델 로드 완료: {save_dir}")
        return obj