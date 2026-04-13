import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.paths import DATA_DIR, MODEL_DIR
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import hdbscan


# ==========================
# Config
# ==========================
NPZ_PATH = str(DATA_DIR / "train.npz")   # 여기 바꾸면 됨 (val/test도 가능)
OUT_DIR = MODEL_DIR / "runs_hdbscan_raw"
SEED = 42

# 너무 크면 일단 샘플링 권장 (예: 20k~100k)
MAX_N = 30000   # None이면 전체 사용

# HDBSCAN 파라미터 (처음엔 이렇게 시작 추천)
MIN_CLUSTER_SIZE = 80
MIN_SAMPLES = 20   # None이면 기본값(min_cluster_size 근처로 내부 설정됨)


# ==========================
# Utils
# ==========================
def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    x = data["x"]  # (N,4096) uint8
    y = data["y"]  # (N,) int (0~74)
    return x, y

def subsample(x, y, max_n, seed=42):
    if max_n is None or len(x) <= max_n:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_n, replace=False)
    return x[idx], y[idx]

def cluster_summary(labels, y_true, out_csv):
    """
    labels: HDBSCAN cluster labels (-1 = noise)
    y_true: ground-truth extension label (0~74)
    """
    clusters = sorted(set(labels))
    rows = []
    for c in clusters:
        mask = (labels == c)
        n = int(mask.sum())
        if n == 0:
            continue

        # noise cluster
        if c == -1:
            rows.append([c, n, "NOISE", 0.0, ""])
            continue

        ys = y_true[mask].tolist()
        cnt = Counter(ys)
        top_label, top_count = cnt.most_common(1)[0]
        purity = top_count / n
        top5 = ",".join([f"{lab}:{k}" for lab, k in cnt.most_common(5)])
        rows.append([c, n, int(top_label), float(purity), top5])

    # save csv
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "size", "top_y_label", "purity", "top5_label_counts"])
        w.writerows(rows)

def plot_cluster_sizes(labels, save_path):
    uniq, counts = np.unique(labels, return_counts=True)
    # 보기 좋게 noise(-1)는 맨 앞으로
    order = np.argsort(uniq)
    uniq, counts = uniq[order], counts[order]

    plt.figure(figsize=(10,4))
    plt.bar(np.arange(len(uniq)), counts)
    plt.xticks(np.arange(len(uniq)), uniq, rotation=45)
    plt.ylabel("count")
    plt.title("HDBSCAN cluster sizes (including -1 noise)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_pca_2d(X, color, save_path, title):
    """
    X: (N,4096) float32
    color: (N,) labels (cluster id or y)
    """
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(7,6))
    plt.scatter(Z[:,0], Z[:,1], s=3, c=color, alpha=0.7)
    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


# ==========================
# Main
# ==========================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Load data:", NPZ_PATH)
    x_u8, y = load_npz(NPZ_PATH)
    print("  x:", x_u8.shape, x_u8.dtype, " y:", y.shape, y.dtype)

    x_u8, y = subsample(x_u8, y, MAX_N, SEED)
    print(f"[2] After sampling: N={len(x_u8)}")

    # HDBSCAN은 float 입력이 편함
    # (0~255를 0~1로 스케일) -> 거리 계산 안정성에 도움
    X = x_u8.astype(np.float32) / 255.0

    print("[3] Run HDBSCAN (raw 4096D)...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        core_dist_n_jobs=os.cpu_count()
    )
    labels = clusterer.fit_predict(X)   # (N,)

    n_noise = int((labels == -1).sum())
    n_total = len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print("\n===== HDBSCAN Result =====")
    print("clusters:", n_clusters)
    print("noise:", n_noise, f"({n_noise/n_total*100:.2f}%)")

    # 군집 품질 참고용(정답 y를 활용): ARI/NMI
    # (군집은 비지도라 '정답'이랑 1:1은 아니지만 참고는 됨)
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    print(f"ARI (y vs cluster): {ari:.4f}")
    print(f"NMI (y vs cluster): {nmi:.4f}")

    # 저장물
    cluster_summary(labels, y, out_dir / "cluster_summary.csv")
    plot_cluster_sizes(labels, out_dir / "cluster_sizes.png")

    # 시각화(2D) - PCA로만(빠르고 설치 기본)
    plot_pca_2d(X, labels, out_dir / "pca_clusters.png", "PCA 2D (colored by cluster id)")
    plot_pca_2d(X, y,      out_dir / "pca_true_y.png",  "PCA 2D (colored by true y label)")

    print("\nSaved to:", out_dir.resolve())
    print(" - cluster_summary.csv")
    print(" - cluster_sizes.png")
    print(" - pca_clusters.png")
    print(" - pca_true_y.png")


if __name__ == "__main__":
    main()
