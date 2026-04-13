"""
confusion_analysis.py
혼동 쌍 분석 모듈.
class_names.py와 연동하여 카테고리 정보까지 활용.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from core.class_names import idx_to_category, is_same_category


@dataclass
class ConfusingPair:
    class_a: str
    class_b: str
    a_misclassified_as_b: int
    b_misclassified_as_a: int
    total_confusion: int
    confusion_rate: float
    same_category: bool   # 같은 카테고리 내 혼동인지 여부

    def __repr__(self):
        tag = "[동일 카테고리]" if self.same_category else "[cross-category]"
        return (f"{tag} {self.class_a} ↔ {self.class_b}  "
                f"총 {self.total_confusion}회  rate={self.confusion_rate:.3f}")


def build_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    return cm_norm


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (24, 20)
) -> None:
    cm_norm = build_confusion_matrix(y_true, y_pred, classes)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        xticklabels=classes,
        yticklabels=classes,
        cmap='Blues',
        ax=ax,
        vmin=0, vmax=1,
        linewidths=0.3,
        linecolor='#e0e0e0'
    )
    ax.set_xlabel('예측 클래스', fontsize=13)
    ax.set_ylabel('실제 클래스', fontsize=13)
    ax.set_title('FFT-75 Confusion Matrix (베이스라인 RF)', fontsize=15)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"confusion matrix 저장: {save_path}")
    plt.show()


def extract_confusing_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    top_k: int = 20,
    min_confusion: int = 5
) -> List[ConfusingPair]:
    cm = confusion_matrix(y_true, y_pred)
    n = len(classes)
    pairs = []
    class_counts = np.bincount(y_true, minlength=n)

    for i in range(n):
        for j in range(i + 1, n):
            a_to_b = cm[i, j]
            b_to_a = cm[j, i]
            total = int(a_to_b + b_to_a)
            if total < min_confusion:
                continue
            denom = class_counts[i] + class_counts[j]
            rate = total / max(denom, 1)
            same_cat = is_same_category(i, j)
            pairs.append(ConfusingPair(
                class_a=classes[i],
                class_b=classes[j],
                a_misclassified_as_b=int(a_to_b),
                b_misclassified_as_a=int(b_to_a),
                total_confusion=total,
                confusion_rate=rate,
                same_category=same_cat,
            ))

    pairs.sort(key=lambda p: p.total_confusion, reverse=True)
    top_pairs = pairs[:top_k]

    # 카테고리별 집계 출력
    same_cat  = sum(1 for p in top_pairs if p.same_category)
    cross_cat = len(top_pairs) - same_cat
    print(f"\n상위 {len(top_pairs)}개 혼동 쌍  "
          f"(동일 카테고리: {same_cat}개 / cross-category: {cross_cat}개)")
    print("-" * 70)
    for p in top_pairs:
        print(p)
    return top_pairs


def plot_category_confusion_summary(
    confusing_pairs: List[ConfusingPair],
    save_path: Optional[str] = None
) -> None:
    """
    혼동 쌍을 카테고리별로 집계하여 막대 차트로 표시.
    논문 Figure: "어느 카테고리에서 혼동이 집중되는가"
    """
    from collections import defaultdict
    from core.class_names import idx_to_category, FFT75_CLASSES

    cat_total: dict = defaultdict(int)
    for p in confusing_pairs:
        # class 이름으로 인덱스 찾기
        try:
            idx_a = FFT75_CLASSES.index(p.class_a)
            cat = idx_to_category(idx_a)
        except ValueError:
            cat = "Unknown"
        cat_total[cat] += p.total_confusion

    cats   = sorted(cat_total, key=cat_total.get, reverse=True)
    totals = [cat_total[c] for c in cats]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(cats, totals, color='#4C9BE8', alpha=0.8, edgecolor='white')
    ax.bar_label(bars, fmt='%d', fontsize=10)
    ax.set_ylabel('총 혼동 횟수')
    ax.set_title('카테고리별 혼동 집중도')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"카테고리 혼동 차트 저장: {save_path}")
    plt.show()


def analyze_pair_features(
    fragments_a: List[bytes],
    fragments_b: List[bytes],
    class_a: str,
    class_b: str,
    save_path: Optional[str] = None
) -> Dict:
    """혼동 쌍 두 클래스의 feature 분포 비교."""
    from core.feature_extractor import shannon_entropy, chi_square_stat

    def get_stats(fragments):
        entropies = [shannon_entropy(f) for f in fragments]
        chi2s     = [chi_square_stat(f) for f in fragments]
        means     = [np.mean(np.frombuffer(f, dtype=np.uint8)) for f in fragments]
        return np.array(entropies), np.array(chi2s), np.array(means)

    ent_a, chi_a, mean_a = get_stats(fragments_a)
    ent_b, chi_b, mean_b = get_stats(fragments_b)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"혼동 쌍 분석: {class_a} vs {class_b}", fontsize=14)
    color_a, color_b = '#4C9BE8', '#E86B4C'

    for ax, da, db, xlabel, title in [
        (axes[0], ent_a,  ent_b,  'Shannon Entropy',      '엔트로피 분포'),
        (axes[1], chi_a,  chi_b,  'Chi-square statistic', '카이제곱 분포'),
        (axes[2], mean_a, mean_b, '바이트 평균값',          '바이트 평균 분포'),
    ]:
        ax.hist(da, bins=30, alpha=0.6, color=color_a, label=class_a, density=True)
        ax.hist(db, bins=30, alpha=0.6, color=color_b, label=class_b, density=True)
        ax.set_xlabel(xlabel); ax.set_title(title); ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"혼동 쌍 분석 저장: {save_path}")
    plt.show()

    return {
        class_a: {'entropy_mean': float(np.mean(ent_a)), 'entropy_std': float(np.std(ent_a)), 'chi2_mean': float(np.mean(chi_a))},
        class_b: {'entropy_mean': float(np.mean(ent_b)), 'entropy_std': float(np.std(ent_b)), 'chi2_mean': float(np.mean(chi_b))},
    }
