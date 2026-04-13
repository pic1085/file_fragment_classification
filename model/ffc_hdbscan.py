import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.paths import DATA_DIR, MODEL_DIR
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import hdbscan

# (선택) UMAP 있으면 2D 시각화가 더 예쁨
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# ==========================
# 0) Config
# ==========================
NPZ_PATH = str(DATA_DIR / "train.npz")
OUT_DIR  = MODEL_DIR / "runs_hdbscan_pca-umap3"

# === UMAP for clustering ===
USE_UMAP_FOR_CLUSTER = True
UMAP_DIM = 20
UMAP_N_NEIGHBORS = 50
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

# HDBSCAN 핵심 파라미터
MIN_CLUSTER_SIZE = 200
# min_samples=None이면 기본적으로 min_cluster_size로 동작 (공식 문서)   [oai_citation:3‡HDBSCAN](https://hdbscan.readthedocs.io/en/0.8.2/api.html)
MIN_SAMPLES = None

# 특징 추출: byte histogram(256) -> (optional) PCA
USE_PCA = True
PCA_DIM = 30

# 시각화(2D)
USE_UMAP_FOR_PLOT = True
RANDOM_SEED = 42

# HDBSCAN 옵션(트리/플롯 관련)
HDBSCAN_METRIC = "euclidean"
CLUSTER_SELECTION_METHOD = "eom"   # "eom" or "leaf"
APPROX_MIN_SPAN_TREE = False       # MST가 필요하면 False 권장   [oai_citation:4‡HDBSCAN](https://hdbscan.readthedocs.io/en/0.8.2/api.html)
GEN_MIN_SPAN_TREE = True           # minimum_spanning_tree_ 생성 시도   [oai_citation:5‡HDBSCAN](https://hdbscan.readthedocs.io/en/0.8.2/api.html)
PREDICTION_DATA = False            # approximate_predict 쓸거면 True


# ==========================
# 1) Helpers
# ==========================
def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    x = data["x"].astype(np.uint8, copy=False)   # (N,4096)
    y = data["y"].astype(np.int64, copy=False)   # (N,)
    assert x.ndim == 2 and x.shape[1] == 4096
    assert len(x) == len(y)
    return x, y

def bytes_to_hist_features(x_uint8: np.ndarray) -> np.ndarray:
    """
    x_uint8: (N,4096) uint8
    return:  (N,256) float32  (정규화된 빈도)
    """
    N = x_uint8.shape[0]
    feats = np.empty((N, 256), dtype=np.float32)

    for i in tqdm(range(N), desc="Extract byte-hist features"):
        h = np.bincount(x_uint8[i], minlength=256).astype(np.float32)
        h /= 4096.0
        feats[i] = h

    return feats

def make_2d_embedding(feats: np.ndarray) -> np.ndarray:
    """
    시각화용 2D 임베딩
    """
    if USE_UMAP_FOR_PLOT and HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=RANDOM_SEED,
        )
        return reducer.fit_transform(feats)
    else:
        pca2 = PCA(n_components=2, random_state=RANDOM_SEED)
        return pca2.fit_transform(feats)

def summarize_clusters(cluster_labels: np.ndarray, true_labels: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = cluster_labels
    N = len(labels)
    unique = np.unique(labels)

    n_noise = int((labels == -1).sum())
    noise_ratio = n_noise / max(N, 1)

    clusters = [c for c in unique if c != -1]
    n_clusters = len(clusters)

    rows = []
    topk_rows = []

    for c in clusters:
        idx = np.where(labels == c)[0]
        size = len(idx)
        ys = true_labels[idx]

        vals, cnts = np.unique(ys, return_counts=True)
        order = np.argsort(-cnts)
        vals = vals[order]
        cnts = cnts[order]

        top_label = int(vals[0])
        top_count = int(cnts[0])
        purity = top_count / max(size, 1)

        rows.append({
            "cluster_id": int(c),
            "size": int(size),
            "purity": float(purity),
            "top_label": int(top_label),
            "top_label_count": int(top_count),
        })

        k = min(5, len(vals))
        for j in range(k):
            topk_rows.append({
                "cluster_id": int(c),
                "rank": j + 1,
                "label": int(vals[j]),
                "count": int(cnts[j]),
                "ratio_in_cluster": float(cnts[j] / max(size, 1)),
            })

    df = pd.DataFrame(rows).sort_values(["size"], ascending=False)
    df_topk = pd.DataFrame(topk_rows).sort_values(["cluster_id", "rank"])

    df.to_csv(out_dir / "cluster_summary.csv", index=False)
    df_topk.to_csv(out_dir / "cluster_top5_labels.csv", index=False)

    print("\n=== HDBSCAN Summary ===")
    print(f"Total samples : {N}")
    print(f"#clusters     : {n_clusters}")
    print(f"noise (-1)    : {n_noise} ({noise_ratio*100:.2f}%)")
    print("\nTop clusters (by size):")
    print(df.head(10).to_string(index=False))

    return df, noise_ratio

def plot_cluster_scatter(emb2d, cluster_labels, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(emb2d[:, 0], emb2d[:, 1], c=cluster_labels, s=3, alpha=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)

def plot_noise_only(emb2d, labels, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    is_noise = (labels == -1)
    ax.scatter(emb2d[~is_noise, 0], emb2d[~is_noise, 1], s=2, alpha=0.2)
    ax.scatter(emb2d[is_noise, 0], emb2d[is_noise, 1], s=3, alpha=0.9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)

def plot_cluster_sizes(df_summary: pd.DataFrame, out_path: Path, title: str, top_n: int = 30):
    d = df_summary.sort_values("size", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(d)), d["size"].values)
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(d["cluster_id"].values, rotation=45)
    ax.set_title(title)
    ax.set_xlabel("cluster_id (top by size)")
    ax.set_ylabel("size")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)

def plot_persistence(clusterer: hdbscan.HDBSCAN, out_path: Path, title: str):
    pers = getattr(clusterer, "cluster_persistence_", None)
    if pers is None:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(pers)), pers)
    ax.set_title(title)
    ax.set_xlabel("cluster index (internal)")
    ax.set_ylabel("persistence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)

def save_tree_plots(clusterer: hdbscan.HDBSCAN, out_dir: Path):
    # condensed tree
    try:
        ax = clusterer.condensed_tree_.plot(select_clusters=True)
        ax.figure.savefig(out_dir / "condensed_tree.png", dpi=250, bbox_inches="tight")
        plt.close(ax.figure)
    except Exception as e:
        print("condensed_tree plot skipped:", e)

    # single linkage tree
    try:
        ax = clusterer.single_linkage_tree_.plot(cmap="viridis", colorbar=True)
        ax.figure.savefig(out_dir / "single_linkage_tree.png", dpi=250, bbox_inches="tight")
        plt.close(ax.figure)
    except Exception as e:
        print("single_linkage_tree plot skipped:", e)

    # minimum spanning tree (NOTE: gen_min_span_tree=True여도 최적화 케이스에서 없을 수 있음)   [oai_citation:6‡HDBSCAN](https://hdbscan.readthedocs.io/en/0.8.2/api.html)
    try:
        mst = getattr(clusterer, "minimum_spanning_tree_", None)
        if mst is None:
            print("mst not available: minimum_spanning_tree_ is None")
            return
        ax = mst.plot(node_size=5, edge_alpha=0.6)
        ax.figure.savefig(out_dir / "min_spanning_tree.png", dpi=250, bbox_inches="tight")
        plt.close(ax.figure)
    except Exception as e:
        print("mst plot skipped:", e)


# ==========================
# 2) Main
# ==========================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Load: {NPZ_PATH}")
    x, y = load_npz(NPZ_PATH)
    print(f"x: {x.shape} | y: {y.shape} | unique y: {len(np.unique(y))}")

    feats = bytes_to_hist_features(x)
    feats_norm = normalize(feats, norm="l2").astype(np.float32, copy=False)

    # (Optional) PCA
    feats_mid = feats_norm
    if USE_PCA:
        pca = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
        feats_mid = pca.fit_transform(feats_norm).astype(np.float32, copy=False)
        print(f"PCA: {feats_norm.shape[1]} -> {feats_mid.shape[1]}")

    # (Optional) UMAP for clustering
    feats_for_cluster = feats_mid
    if USE_UMAP_FOR_CLUSTER:
        if not HAS_UMAP:
            raise RuntimeError("UMAP is not installed but USE_UMAP_FOR_CLUSTER=True")
        reducer = umap.UMAP(
            n_components=UMAP_DIM,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=RANDOM_SEED,
            low_memory=True,
        )
        feats_for_cluster = reducer.fit_transform(feats_mid).astype(np.float32, copy=False)
        print(f"UMAP: {feats_mid.shape[1]} -> {feats_for_cluster.shape[1]}")

    # HDBSCAN (공식 API 옵션들 사용)   [oai_citation:7‡HDBSCAN](https://hdbscan.readthedocs.io/en/0.8.2/api.html)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        approx_min_span_tree=APPROX_MIN_SPAN_TREE,
        gen_min_span_tree=GEN_MIN_SPAN_TREE,
        prediction_data=PREDICTION_DATA,
    )

    print("\nRun HDBSCAN...")
    labels = clusterer.fit_predict(feats_for_cluster)  # -1 = noise

    # Trees / plots
    save_tree_plots(clusterer, out_dir)

    # Save scores
    np.save(out_dir / "outlier_scores.npy", clusterer.outlier_scores_)
    np.save(out_dir / "probabilities.npy", clusterer.probabilities_)

    # Summary
    df_summary, noise_ratio = summarize_clusters(labels, y, out_dir)

    # 2D plot embedding (시각화는 원래 feature space 기준)
    print("\nMake 2D embedding for plot...")
    emb2d = make_2d_embedding(feats_norm)

    plot_cluster_scatter(emb2d, labels, out_dir/"clusters_by_id.png", "Clusters by ID")
    plot_noise_only(emb2d, labels, out_dir/"noise_only.png", "Noise only")

    plot_cluster_sizes(df_summary, out_dir / "cluster_sizes_top.png", "Cluster sizes (top clusters)")
    plot_persistence(clusterer, out_dir / "cluster_persistence.png", "HDBSCAN cluster persistence (stability)")

    np.save(out_dir / "cluster_labels.npy", labels)

    print("\nSaved outputs to:", out_dir.resolve())
    print("- condensed_tree.png / single_linkage_tree.png / min_spanning_tree.png")
    print("- clusters_by_id.png / noise_only.png")
    print("- cluster_sizes_top.png / cluster_persistence.png")
    print("- cluster_summary.csv / cluster_top5_labels.csv")
    print("- cluster_labels.npy / outlier_scores.npy / probabilities.npy")


if __name__ == "__main__":
    main()
