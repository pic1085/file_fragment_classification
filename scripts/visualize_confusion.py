"""
visualize_confusion.py
혼동 쌍 feature-level 시각화 스크립트.
논문 Figure 2, 3을 생성합니다.

실행:
    python visualize_confusion.py --data_dir /path/to/Data_set
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from collections import defaultdict

from core.dataset import load_fft75_npz
from core.paths import DATA_DIR, OUTPUT_DIR
from core.class_names import FFT75_CLASSES


# ── 한글 폰트 설정 ────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic',
                  'AppleGothic', 'UnDotum']
    for font in candidates:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            return font
        except Exception:
            continue
    # 폰트 없으면 영문 레이블로 대체 (경고 억제)
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning,
                            message='Glyph.*missing from font')
    return None


# ── Feature 추출 유틸 ─────────────────────────────────────────────────────────
# npz에서 이미 추출된 feature 벡터를 사용합니다.
# feature 구조: [hist_0..255 (256) | entropy,chi2,mean,std,mode,unique,zero (7) | bigram_0..99 (100)]
ENTROPY_IDX = 256
CHI2_IDX    = 257
MEAN_IDX    = 258
STD_IDX     = 259


def get_class_samples(X, y, class_idx, max_n=2000):
    """특정 클래스의 feature 벡터 샘플 반환."""
    mask = y == class_idx
    X_cls = X[mask]
    if len(X_cls) > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_cls), max_n, replace=False)
        X_cls = X_cls[idx]
    return X_cls


# ── Figure 1: 혼동 쌍 엔트로피/카이제곱/평균 분포 비교 ───────────────────────
def plot_feature_distributions(X, y, pairs, save_dir, top_n=6):
    """
    상위 N개 혼동 쌍의 feature 분포를 비교.
    논문 Figure 2: "왜 혼동되는가"의 feature-level 근거.
    """
    set_korean_font()
    features = [
        (ENTROPY_IDX, 'Shannon Entropy',    'entropy'),
        (CHI2_IDX,    'Chi-square',         'chi2'),
        (MEAN_IDX,    'Byte mean',          'mean'),
        (STD_IDX,     'Byte std',           'std'),
    ]

    pairs_to_plot = pairs[:top_n]
    fig, axes = plt.subplots(len(pairs_to_plot), len(features),
                             figsize=(16, 3.2 * len(pairs_to_plot)))
    fig.suptitle('Confusing pair feature distributions (baseline RF)',
                 fontsize=14, y=1.01)

    color_a, color_b = '#3A7FC1', '#E05C3A'

    for row, (name_a, name_b, total, rate) in enumerate(
            tqdm(pairs_to_plot, desc='분포 시각화')):
        idx_a = FFT75_CLASSES.index(name_a)
        idx_b = FFT75_CLASSES.index(name_b)
        Xa = get_class_samples(X, y, idx_a)
        Xb = get_class_samples(X, y, idx_b)

        for col, (feat_idx, xlabel, _) in enumerate(features):
            ax = axes[row, col]
            va = Xa[:, feat_idx]
            vb = Xb[:, feat_idx]
            ax.hist(va, bins=40, alpha=0.55, color=color_a,
                    label=name_a, density=True)
            ax.hist(vb, bins=40, alpha=0.55, color=color_b,
                    label=name_b, density=True)
            if col == 0:
                ax.set_ylabel(f'{name_a} vs {name_b}\n(rate={rate:.3f})',
                              fontsize=9)
            if row == 0:
                ax.set_title(xlabel, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig2_feature_distributions.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    print(f'\n[저장] {path}')
    plt.close()


# ── Figure 2: 바이트 히스토그램 오버레이 (상위 5쌍) ──────────────────────────
def plot_histogram_overlay(X, y, pairs, save_dir, top_n=5):
    """
    두 클래스의 평균 바이트 히스토그램을 겹쳐 보여줌.
    분포가 겹치는 구간이 시각적으로 명확하게 드러남.
    논문 Figure 3.
    """
    set_korean_font()
    pairs_to_plot = pairs[:top_n]
    fig, axes = plt.subplots(top_n, 1, figsize=(14, 3.5 * top_n))
    if top_n == 1:
        axes = [axes]
    fig.suptitle('Byte histogram overlay — top confusing pairs', fontsize=13)

    color_a, color_b = '#3A7FC1', '#E05C3A'
    x = np.arange(256)

    for ax, (name_a, name_b, total, rate) in zip(
            axes, tqdm(pairs_to_plot, desc='히스토그램 오버레이')):
        idx_a = FFT75_CLASSES.index(name_a)
        idx_b = FFT75_CLASSES.index(name_b)
        Xa = get_class_samples(X, y, idx_a)
        Xb = get_class_samples(X, y, idx_b)

        # 평균 히스토그램 (처음 256 feature = 바이트 히스토그램)
        hist_a = Xa[:, :256].mean(axis=0)
        hist_b = Xb[:, :256].mean(axis=0)

        ax.bar(x, hist_a, alpha=0.5, color=color_a, width=1.0, label=name_a)
        ax.bar(x, hist_b, alpha=0.5, color=color_b, width=1.0, label=name_b)

        # 겹치는 영역 강조
        overlap = np.minimum(hist_a, hist_b)
        ax.bar(x, overlap, alpha=0.4, color='#888888', width=1.0, label='overlap')

        ax.set_title(f'{name_a} vs {name_b}  (confusion rate={rate:.3f})',
                     fontsize=10)
        ax.set_xlabel('Byte value (0–255)', fontsize=8)
        ax.set_ylabel('Mean frequency', fontsize=8)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig3_histogram_overlay.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    print(f'[저장] {path}')
    plt.close()


# ── Figure 3: Human-readable 그룹 t-SNE 임베딩 ───────────────────────────────
def plot_tsne_humanreadable(X, y, save_dir, max_per_class=800):
    """
    Human-readable 그룹(MD,RTF,TXT,TEX,JSON,HTML,XML,LOG,CSV) t-SNE 시각화.
    feature space에서 클래스들이 얼마나 겹치는지 직관적으로 확인.
    논문 Figure 4.
    """
    set_korean_font()
    hr_classes = ['MD','RTF','TXT','TEX','JSON','HTML','XML','LOG','CSV']
    hr_indices = [FFT75_CLASSES.index(c) for c in hr_classes]
    colors = ['#E05C3A','#3A7FC1','#2CA05A','#F5A623','#9B59B6',
              '#1ABC9C','#E74C3C','#34495E','#F39C12']

    X_sub, y_sub, labels = [], [], []
    for cls_idx, name in zip(hr_indices,
                             tqdm(hr_classes, desc='t-SNE 샘플 수집')):
        Xc = get_class_samples(X, y, cls_idx, max_n=max_per_class)
        X_sub.append(Xc)
        y_sub.extend([name] * len(Xc))

    X_sub = np.vstack(X_sub)

    # PCA로 50차원 축소 후 t-SNE (속도 향상)
    print('PCA 전처리 중...')
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_sub)

    print('t-SNE 실행 중 (수 분 소요)...')
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
                random_state=42, n_jobs=-1)
    X_2d = tsne.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(11, 9))
    for name, color in zip(hr_classes, colors):
        mask = np.array(y_sub) == name
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=name, alpha=0.4, s=8, linewidths=0)

    ax.set_title('t-SNE: Human-readable group (FFT-75)', fontsize=13)
    ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
    ax.legend(markerscale=3, fontsize=9, loc='best')
    plt.tight_layout()
    path = os.path.join(save_dir, 'fig4_tsne_humanreadable.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    print(f'[저장] {path}')
    plt.close()


# ── Figure 4: 혼동 쌍 confusion_rate 막대 차트 ───────────────────────────────
def plot_confusion_rate_bar(pairs, save_dir):
    """상위 20개 혼동 쌍의 confusion_rate 막대 차트. 논문 Figure 1."""
    set_korean_font()
    labels = [f"{a}↔{b}" for a, b, *_ in pairs]
    rates  = [r for *_, r in pairs]
    colors_bar = ['#3A7FC1' if a in ('RTF','ELF','MACH-O','HTML','MD','TEX','JSON')
                  else '#AAAAAA' for a, *_ in pairs]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.barh(labels[::-1], rates[::-1], color=colors_bar[::-1], height=0.65)
    ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
    ax.set_xlabel('Confusion rate')
    ax.set_title('Top-20 confusing pairs — baseline RF (FFT-75)', fontsize=12)
    ax.set_xlim(0, max(rates) * 1.15)
    ax.tick_params(axis='y', labelsize=9)

    patch_hl = mpatches.Patch(color='#3A7FC1', label='Key confusion source')
    patch_ot = mpatches.Patch(color='#AAAAAA', label='Other')
    ax.legend(handles=[patch_hl, patch_ot], fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig1_confusion_rate_bar.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    print(f'[저장] {path}')
    plt.close()


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main(data_dir, output_dir, skip_tsne):
    os.makedirs(output_dir, exist_ok=True)

    # 데이터 로드 (test셋 기준으로 시각화)
    print('데이터 로드 중...')
    _, _, X_test, _, _, y_test, classes = load_fft75_npz(data_dir)

    # CSV에서 혼동 쌍 로드
    csv_path = os.path.join(output_dir, 'confusing_pairs.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f'{csv_path} 없음. main.py를 먼저 실행해 confusing_pairs.csv를 생성하세요.')

    pairs = []
    with open(csv_path) as f:
        next(f)  # 헤더 skip
        for line in f:
            a, b, atob, btoa, total, rate = line.strip().split(',')
            pairs.append((a, b, int(total), float(rate)))

    print(f'\n총 {len(pairs)}개 혼동 쌍 로드 완료\n')

    # Figure 생성
    print('=' * 50)
    print('Figure 1: confusion rate 막대 차트')
    print('=' * 50)
    plot_confusion_rate_bar(pairs, output_dir)

    print('=' * 50)
    print('Figure 2: feature 분포 비교 (상위 6쌍)')
    print('=' * 50)
    plot_feature_distributions(X_test, y_test, pairs, output_dir, top_n=6)

    print('=' * 50)
    print('Figure 3: 바이트 히스토그램 오버레이 (상위 5쌍)')
    print('=' * 50)
    plot_histogram_overlay(X_test, y_test, pairs, output_dir, top_n=5)

    if not skip_tsne:
        print('=' * 50)
        print('Figure 4: Human-readable 그룹 t-SNE')
        print('=' * 50)
        plot_tsne_humanreadable(X_test, y_test, output_dir)
    else:
        print('[skip] --skip_tsne 옵션으로 t-SNE 건너뜀')

    print(f'\n모든 Figure 저장 완료: {os.path.abspath(output_dir)}')
    print('  fig1_confusion_rate_bar.png   → 논문 Figure 1')
    print('  fig2_feature_distributions.png → 논문 Figure 2')
    print('  fig3_histogram_overlay.png    → 논문 Figure 3')
    if not skip_tsne:
        print('  fig4_tsne_humanreadable.png   → 논문 Figure 4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str, default=str(DATA_DIR))
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--skip_tsne',  action='store_true',
                        help='t-SNE는 시간이 걸려요. 빠른 확인 시 사용')
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.skip_tsne)
