"""
train_hierarchical.py
계층적 분류기 학습 + 베이스라인 비교.

실행:
    python train_hierarchical.py --data_dir /path/to/Data_set
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import os
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from core.dataset import load_fft75_npz
from core.paths import DATA_DIR, OUTPUT_DIR
from models.hierarchical_classifier import HierarchicalClassifier
from core.class_names import FFT75_CATEGORIES, IDX_TO_CATEGORY


def compare_and_save(y_test, y_pred_base, y_pred_hier,
                     classes, save_dir):
    base_acc = accuracy_score(y_test, y_pred_base)
    hier_acc = accuracy_score(y_test, y_pred_hier)

    p_b, r_b, f_b, _ = precision_recall_fscore_support(
        y_test, y_pred_base, average=None, zero_division=0)
    p_h, r_h, f_h, _ = precision_recall_fscore_support(
        y_test, y_pred_hier, average=None, zero_division=0)

    csv_path = os.path.join(save_dir, 'comparison_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'category',
                         'base_f1', 'hier_f1', 'f1_delta'])
        for i, cls in enumerate(classes):
            delta = f_h[i] - f_b[i]
            writer.writerow([cls, IDX_TO_CATEGORY.get(i, '?'),
                              f'{f_b[i]:.4f}', f'{f_h[i]:.4f}',
                              f'{delta:+.4f}'])

    print(f'\n{"="*55}')
    print(f'  베이스라인 정확도   : {base_acc:.4f} ({base_acc*100:.2f}%)')
    print(f'  계층적 분류기 정확도: {hier_acc:.4f} ({hier_acc*100:.2f}%)')
    delta = hier_acc - base_acc
    sign  = '+' if delta >= 0 else ''
    print(f'  향상폭             : {sign}{delta:.4f} ({sign}{delta*100:.2f}%p)')
    print(f'{"="*55}')

    hr_indices = FFT75_CATEGORIES['Human-readable']
    mask = np.isin(y_test, hr_indices)
    if mask.sum() > 0:
        acc_b = accuracy_score(y_test[mask], y_pred_base[mask])
        acc_h = accuracy_score(y_test[mask], y_pred_hier[mask])
        d     = acc_h - acc_b
        sign  = '+' if d >= 0 else ''
        print(f'\n  [Human-readable 그룹]')
        print(f'  베이스라인  : {acc_b:.4f}')
        print(f'  계층적 분류 : {acc_h:.4f}')
        print(f'  향상폭      : {sign}{d:.4f} ({sign}{d*100:.2f}%p)')

    print(f'\n  비교표 저장: {csv_path}')

def save_charts(df, output_dir):
    df = df.dropna(subset=['class'])
    df = df[df['class'].str.strip() != '']

    # ── 1. 카테고리별 F1 비교 막대 그래프 ──────────────────────
    fig, axes = plt.subplots(4, 3, figsize=(22, 28))
    axes = axes.flatten()
    categories = df['category'].unique()

    for i, cat in enumerate(categories):
        ax = axes[i]
        sub = df[df['category'] == cat].sort_values('hier_f1', ascending=True)
        x = range(len(sub))
        ax.barh([v - 0.2 for v in x], sub['base_f1'], height=0.4,
                color='steelblue', alpha=0.7, label='Baseline')
        ax.barh([v + 0.2 for v in x], sub['hier_f1'], height=0.4,
                color='tomato', alpha=0.7, label='Hierarchical')
        ax.set_yticks(list(x))
        ax.set_yticklabels(sub['class'], fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_title(f'[{cat}]', fontsize=12, fontweight='bold')
        ax.axvline(0.7, color='gray', linestyle='--', linewidth=0.8)
        ax.legend(fontsize=8)

    # 남는 subplot 숨기기
    for j in range(len(categories), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Baseline vs Hierarchical Classifier (F1 by Category)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/f1_by_category.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  차트 저장: {output_dir}/f1_by_category.png')

    # ── 2. 전체 F1 delta 히트맵 ────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 20))
    pivot = df.set_index('class')[['base_f1', 'hier_f1', 'f1_delta']]
    pivot = pivot.sort_values('f1_delta', ascending=True)

    colors = ['tomato' if v < 0 else 'mediumseagreen'
              for v in pivot['f1_delta']]
    bars = ax.barh(pivot.index, pivot['f1_delta'], color=colors, alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('F1 Delta (Hierarchical - Baseline)', fontsize=12)
    ax.set_title('F1 Score Change: Hierarchical vs Baseline', fontsize=14,
                 fontweight='bold')

    pos_patch = mpatches.Patch(color='mediumseagreen', label='개선')
    neg_patch = mpatches.Patch(color='tomato', label='악화')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/f1_delta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  차트 저장: {output_dir}/f1_delta.png')

    # ── 3. 카테고리별 평균 F1 요약 바 차트 ────────────────────
    cat_summary = df.groupby('category')[['base_f1','hier_f1']].mean().sort_values('hier_f1')
    fig, ax = plt.subplots(figsize=(12, 7))
    x = range(len(cat_summary))
    ax.bar([v - 0.2 for v in x], cat_summary['base_f1'], width=0.4,
           color='steelblue', alpha=0.8, label='Baseline')
    ax.bar([v + 0.2 for v in x], cat_summary['hier_f1'], width=0.4,
           color='tomato', alpha=0.8, label='Hierarchical')
    ax.set_xticks(list(x))
    ax.set_xticklabels(cat_summary.index, rotation=30, ha='right', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Average F1 Score', fontsize=12)
    ax.set_title('Average F1 by Category', fontsize=14, fontweight='bold')
    ax.axhline(0.7, color='gray', linestyle='--', linewidth=0.8, label='0.7 기준선')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/f1_category_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  차트 저장: {output_dir}/f1_category_summary.png')
    

def main(data_dir, output_dir, baseline_pred_path):
    os.makedirs(output_dir, exist_ok=True)

    print('='*55)
    print('1. 데이터 로드')
    print('='*55)
    X_train, X_val, X_test, y_train, y_val, y_test, classes = \
        load_fft75_npz(data_dir)

    print('\n' + '='*55)
    print('2. 계층적 분류기 학습')
    print('='*55)
    clf = HierarchicalClassifier(
        cat_estimators=200,
        sub_estimators=150,
        max_depth=30,
    )
    clf.fit(X_train, y_train)

    print('\n' + '='*55)
    print('3. 검증셋 평가')
    print('='*55)
    clf.evaluate(X_val, y_val, classes, verbose=True)

    print('\n' + '='*55)
    print('4. 테스트셋 최종 평가')
    print('='*55)
    test_result = clf.evaluate(X_test, y_test, classes, verbose=True)

    hier_dir = os.path.join(output_dir, 'hierarchical')
    clf.save(hier_dir)

    print('\n' + '='*55)
    print('5. 베이스라인 vs 계층적 분류기 비교')
    print('='*55)

    if os.path.exists(baseline_pred_path):
        print(f'베이스라인 예측 로드: {baseline_pred_path}')
        y_pred_base = np.load(baseline_pred_path)
    else:
        print('베이스라인 예측 파일 없음 -> rf_baseline 재실행')
        from models.baseline_classifier import BaselineClassifier
        base_clf = BaselineClassifier.load(
            os.path.join(output_dir, 'rf_baseline'), classes)
        res = base_clf.evaluate(X_test, y_test,
                                desc='베이스라인 재예측', verbose=False)
        y_pred_base = res['y_pred']
        np.save(baseline_pred_path, y_pred_base)
        print(f'베이스라인 예측 저장: {baseline_pred_path}')

    compare_and_save(y_test, y_pred_base,
                     test_result['y_pred'], classes, output_dir)
    df_comp = pd.read_csv(f'{args.output_dir}/comparison_table.csv')
    save_charts(df_comp, args.output_dir)

    print('\n완료.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',      type=str, default=str(DATA_DIR))
    parser.add_argument('--output_dir',    type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--baseline_pred', type=str,
                        default=str(OUTPUT_DIR / 'y_pred_baseline.npy'))
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.baseline_pred)
