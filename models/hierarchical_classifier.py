"""
hierarchical_classifier.py
계층적 분류기 구현.

구조:
  1단계: 카테고리 분류기 (75클래스 -> 11개 그룹)
  2단계: 그룹별 서브 분류기
         - Human-readable 전용: 구조적 feature 추가
         - 나머지 그룹: 전체 feature 사용
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from typing import List, Dict
from sklearn.utils.class_weight import compute_class_weight
from core.class_names import (
    FFT75_CATEGORIES, IDX_TO_CATEGORY
)

CATEGORIES    = sorted(FFT75_CATEGORIES.keys())
CAT_TO_LABEL  = {cat: i for i, cat in enumerate(CATEGORIES)}
HIST_SLICE    = slice(0, 256)


def make_cat_labels(y: np.ndarray) -> np.ndarray:
    y_cat = np.zeros(len(y), dtype=np.int32)
    for i, label in enumerate(y):
        cat_name = IDX_TO_CATEGORY.get(int(label), 'Other')
        y_cat[i] = CAT_TO_LABEL[cat_name]
    return y_cat


def extract_hr_extra_features(X: np.ndarray) -> np.ndarray:
    """Human-readable 전용 추가 feature (12차원)."""
    hist = X[:, HIST_SLICE]
    printable  = hist[:, 0x20:0x7F].sum(axis=1, keepdims=True)
    alpha      = (hist[:, 0x41:0x5B].sum(axis=1, keepdims=True) +
                  hist[:, 0x61:0x7B].sum(axis=1, keepdims=True))
    digit      = hist[:, 0x30:0x3A].sum(axis=1, keepdims=True)
    whitespace = hist[:, [0x20, 0x09, 0x0A, 0x0D]].sum(axis=1, keepdims=True)
    special    = hist[:, [0x3C, 0x3E, 0x7B, 0x7D, 0x5B, 0x5D,
                           0x3A, 0x2C, 0x5C, 0x2F]].sum(axis=1, keepdims=True)
    null_byte  = hist[:, [0x00]].sum(axis=1, keepdims=True)
    brace_open = hist[:, [0x7B]].sum(axis=1, keepdims=True)
    lt_sign    = hist[:, [0x3C]].sum(axis=1, keepdims=True)
    hash_sign  = hist[:, [0x23]].sum(axis=1, keepdims=True)
    percent    = hist[:, [0x25]].sum(axis=1, keepdims=True)
    comma      = hist[:, [0x2C]].sum(axis=1, keepdims=True)
    high_byte  = hist[:, 0x80:].sum(axis=1, keepdims=True)
    return np.hstack([printable, alpha, digit, whitespace, special,
                      null_byte, brace_open, lt_sign, hash_sign,
                      percent, comma, high_byte]).astype(np.float32)


def build_sub_features(X: np.ndarray, category: str) -> np.ndarray:
    if category == 'Human-readable':
        return np.hstack([X, extract_hr_extra_features(X)])
    return X


def _train_rf(rf, X, y, n_estimators, batch, desc):
    current = 0
    with tqdm(total=n_estimators, desc=desc, unit='trees',
              bar_format='{l_bar}{bar:30}{r_bar}', ncols=80) as pbar:
        for _ in range((n_estimators + batch - 1) // batch):
            current = min(current + batch, n_estimators)
            rf.n_estimators = current
            rf.fit(X, y)
            pbar.update(min(batch, n_estimators - pbar.n))


class CategoryClassifier:
    def __init__(self, n_estimators=500, max_depth=15, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.n_jobs       = n_jobs
        self.clf          = None
        self.scaler       = StandardScaler()

    def fit(self, X, y):
        y_cat = make_cat_labels(y)
        X_s   = self.scaler.fit_transform(X)
        print('  LightGBM 카테고리 분류기 학습 중...')
        self.clf = LGBMClassifier(
            n_estimators=1000,       # 500 → 1000
            max_depth=20,            # 15 → 20
            learning_rate=0.03,      # 0.05 → 0.03 (더 세밀하게)
            num_leaves=255,          # 127 → 255
            min_child_samples=10,    # 추가
            subsample=0.8,           # 추가 (과적합 방지)
            colsample_bytree=0.8,    # 추가
            class_weight='balanced',
            n_jobs=self.n_jobs,
            random_state=42,
            verbose=-1
        )
        self.clf.fit(X_s, y_cat)
        acc = accuracy_score(y_cat, self.clf.predict(X_s))
        print(f'카테고리 분류기 완료  train_acc={acc:.4f}')

    def predict(self, X):
        return self.clf.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))

    def save(self, path):
        joblib.dump({'clf': self.clf, 'scaler': self.scaler}, path, compress=3)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        d = joblib.load(path)
        obj.clf = d['clf']; obj.scaler = d['scaler']
        return obj



class SubClassifier:
    def __init__(self, category, class_indices,
                 n_estimators=300, max_depth=15, n_jobs=-1):
        self.category      = category
        self.class_indices = class_indices
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.n_jobs        = n_jobs
        self.clf           = None
        self.scaler        = StandardScaler()

    def fit(self, X, y):
        mask = np.isin(y, self.class_indices)
        X_f, y_f = X[mask], y[mask]
        if len(X_f) == 0:
            return
        X_feat = build_sub_features(X_f, self.category)
        X_s    = self.scaler.fit_transform(X_feat)
        self.clf = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            num_leaves=127,
            class_weight='balanced',
            n_jobs=self.n_jobs,
            random_state=42,
            verbose=-1
        )
        self.clf.fit(X_s, y_f)
        acc = accuracy_score(y_f, self.clf.predict(X_s))
        print(f'  [{self.category}] 완료  n={len(X_f)}  train_acc={acc:.4f}')

    def predict(self, X):
        X_feat = build_sub_features(X, self.category)
        return self.clf.predict(self.scaler.transform(X_feat))

    def save(self, path):
        joblib.dump({'clf': self.clf, 'scaler': self.scaler,
                     'category': self.category,
                     'class_indices': self.class_indices}, path, compress=3)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        d = joblib.load(path)
        obj.clf = d['clf']; obj.scaler = d['scaler']
        obj.category = d['category']
        obj.class_indices = d['class_indices']
        return obj

class HierarchicalClassifier:
    def __init__(self, cat_estimators=200, sub_estimators=150, max_depth=30):
        self.cat_clf = CategoryClassifier(cat_estimators, max_depth)  # ← 이것으로 교체
        self.sub_clfs: Dict[str, SubClassifier] = {
            cat: SubClassifier(cat, indices, sub_estimators, max_depth)
            for cat, indices in FFT75_CATEGORIES.items()
        }

    def fit(self, X, y):
        print('\n[1단계] 카테고리 분류기 학습')
        self.cat_clf.fit(X, y)  # ← y 그대로 전달 (내부에서 make_cat_labels 처리)

        print('\n[2단계] 그룹별 서브 분류기 학습')
        for cat in CATEGORIES:
            self.sub_clfs[cat].fit(X, y)  # ← y 그대로 전달 (내부에서 필요한 클래스만 선택)

    def predict(self, X, batch_size=1000):
        n = len(X)
        y_pred   = np.full(n, -1, dtype=np.int32)
        cat_pred = np.empty(n, dtype=np.int32)

        with tqdm(total=n, desc='1단계 카테고리 예측', unit='samples',
                bar_format='{l_bar}{bar:30}{r_bar}', ncols=80) as pbar:
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                cat_pred[s:e] = self.cat_clf.predict(X[s:e])
                pbar.update(e - s)

        # ↓ 핵심 수정: cat_idx 대신 cat_name으로 직접 비교
        for cat_name in tqdm(CATEGORIES, desc='2단계 서브 분류기',
                            bar_format='{l_bar}{bar:30}{r_bar}', ncols=80):
            cat_idx = CAT_TO_LABEL[cat_name]        # ← 정수 인덱스 명시적으로 가져옴
            mask = (cat_pred == cat_idx)
            if mask.sum() == 0:
                continue
            y_pred[mask] = self.sub_clfs[cat_name].predict(X[mask])

        missed = (y_pred == -1).sum()
        if missed > 0:
            print(f'[경고] 예측 실패 {missed}개 -> 0으로 대체')
            y_pred[y_pred == -1] = 0
        return y_pred
    
    def evaluate(self, X, y, classes, verbose=True):
        y_pred = self.predict(X)
        acc    = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred,
                                       target_names=classes, digits=4)
        if verbose:
            print(f'\n전체 정확도: {acc:.4f}  ({acc*100:.2f}%)')
            print(report)
        return {'accuracy': acc, 'report': report, 'y_pred': y_pred}

    def save(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.cat_clf.save(save_dir / 'cat_clf.pkl')
        for cat, sub in self.sub_clfs.items():
            fname = cat.replace('-', '_').replace(' ', '_') + '_sub.pkl'
            sub.save(save_dir / fname)
        print(f'저장 완료: {save_dir}')

    @classmethod
    def load(cls, save_dir):
        save_dir = Path(save_dir)
        obj = cls.__new__(cls)
        obj.cat_clf  = CategoryClassifier.load(save_dir / 'cat_clf.pkl')
        obj.sub_clfs = {}
        for cat in CATEGORIES:
            fname = cat.replace('-', '_').replace(' ', '_') + '_sub.pkl'
            obj.sub_clfs[cat] = SubClassifier.load(save_dir / fname)
        print(f'로드 완료: {save_dir}')
        return obj
