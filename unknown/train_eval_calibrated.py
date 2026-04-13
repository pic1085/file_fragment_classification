# train_eval_75class_calibrated.py
import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.paths import DATA_DIR, UNKNOWN_RUNS_DIR
from tqdm import tqdm
from collections import defaultdict

from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# ==========================
# Config
# ==========================
TRAIN_NPZ = str(DATA_DIR / "train.npz")
VAL_NPZ   = str(DATA_DIR / "val.npz")
OUT_DIR   = UNKNOWN_RUNS_DIR / "runs_75class_calibrated"

SEED = 42
N_JOBS = -1  # <= 너가 질문한 부분(병렬 CPU 코어 수, -1이면 all)

# Feature switches
USE_BYTE_HIST = True
USE_ENTROPY   = True
USE_STATS     = True
USE_SIGNATURE = True
USE_HASHED_NGRAM = True

NGRAM_N = 2
HASH_DIM = 65536
NGRAM_REGION = "headtail"  # all/head/tail/headtail
HEAD_BYTES = 512
TAIL_BYTES = 512

# Calibration
CALIB_METHOD = "sigmoid"   # "sigmoid" or "isotonic"(느림)
CALIB_CV = 3               # 3~5 추천

# Low-confidence export
LOWCONF_TAU = 0.6          # 이 값 아래는 "재검증 큐"로 뽑기
LOWCONF_MAX_ROWS = 300000  # 너무 커지는 것 방지 (원하면 늘려)

# ==========================
# Label mapping (네가 준 그대로)
# ==========================
EXT_BY_ID = [
  "ARW","CR2","DNG","GPR","NEF","NRW","ORF","PEF","RAF","RW2",
  "3FR","JPG","TIFF","HEIC","BMP","GIF","PNG",
  "AI","EPS","PSD",
  "MOV","MP4","3GP","AVI","MKV","OGV","WEBM",
  "APK","JAR","MSI","DMG",
  "7Z","BZ2","DEB","GZ","PKG","RAR","RPM","XZ","ZIP",
  "EXE","MACH-O","ELF","DLL",
  "DOC","DOCX","KEY","PPT","PPTX","XLS","XLSX",
  "DJVU","EPUB","MOBI","PDF","MD","RTF","TXT","TEX","JSON","HTML","XML","LOG","CSV",
  "AIFF","FLAC","M4A","MP3","OGG","WAV","WMA",
  "PCAP","TTF","DWG","SQLITE"
]
assert len(EXT_BY_ID) == 75

# ==========================
# Load
# ==========================
def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    x = data["x"].astype(np.uint8, copy=False)   # (N,4096)
    y = data["y"].astype(np.int64, copy=False)   # (N,)
    assert x.ndim == 2 and x.shape[1] == 4096
    return x, y

# ==========================
# Feature blocks
# ==========================
def byte_hist(x_u8: np.ndarray) -> np.ndarray:
    N = x_u8.shape[0]
    out = np.empty((N, 256), dtype=np.float32)
    for i in tqdm(range(N), desc="byte_hist"):
        h = np.bincount(x_u8[i], minlength=256).astype(np.float32)
        h /= 4096.0
        out[i] = h
    return out

def shannon_entropy_from_hist(hist: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(hist, eps, 1.0)
    ent = -(p * np.log2(p)).sum(axis=1, keepdims=True).astype(np.float32)
    return ent

def simple_stats(x_u8: np.ndarray) -> np.ndarray:
    x = x_u8.astype(np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std  = x.std(axis=1, keepdims=True)
    mn   = x.min(axis=1, keepdims=True)
    mx   = x.max(axis=1, keepdims=True)
    zero = (x_u8 == 0).mean(axis=1, keepdims=True).astype(np.float32)
    return np.concatenate([mean, std, mn, mx, zero], axis=1).astype(np.float32)

SIGS = [
    ("ZIP_PK",   b"PK\x03\x04"),
    ("RAR",      b"Rar!\x1A\x07"),
    ("7Z",       b"7z\xBC\xAF\x27\x1C"),
    ("GZ",       b"\x1F\x8B"),
    ("PDF",      b"%PDF"),
    ("PNG",      b"\x89PNG\r\n\x1A\n"),
    ("JPG",      b"\xFF\xD8\xFF"),
    ("GIF87",    b"GIF87a"),
    ("GIF89",    b"GIF89a"),
    ("MP3_ID3",  b"ID3"),
    ("MP4_FTYP", b"ftyp"),
]

def signature_features(x_u8: np.ndarray) -> np.ndarray:
    N = x_u8.shape[0]
    out = np.zeros((N, len(SIGS)), dtype=np.float32)
    for i in tqdm(range(N), desc="signature"):
        head = bytes(x_u8[i, :64].tolist())
        for j, (_, sig) in enumerate(SIGS):
            out[i, j] = 1.0 if sig in head else 0.0
    return out

def _select_region(row: np.ndarray) -> np.ndarray:
    if NGRAM_REGION == "all":
        return row
    if NGRAM_REGION == "head":
        return row[:HEAD_BYTES]
    if NGRAM_REGION == "tail":
        return row[-TAIL_BYTES:]
    if NGRAM_REGION == "headtail":
        return np.concatenate([row[:HEAD_BYTES], row[-TAIL_BYTES:]])
    raise ValueError("bad NGRAM_REGION")

def hashed_ngram_sparse(x_u8: np.ndarray, n: int, dim: int, batch_size: int = 2000):
    N = x_u8.shape[0]
    blocks = []
    for s in tqdm(range(0, N, batch_size), desc="hashed_ngram_sparse"):
        e = min(N, s + batch_size)
        rows, cols, vals = [], [], []
        for i in range(s, e):
            seq = _select_region(x_u8[i])
            if len(seq) < n:
                continue
            d = {}
            if n == 2:
                for k in range(len(seq) - 1):
                    key = (int(seq[k]) << 8) | int(seq[k + 1])
                    h = (key * 2654435761) & 0xFFFFFFFF
                    c = h % dim
                    d[c] = d.get(c, 0) + 1
            elif n == 3:
                for k in range(len(seq) - 2):
                    key = (int(seq[k]) << 16) | (int(seq[k + 1]) << 8) | int(seq[k + 2])
                    h = (key * 2654435761) & 0xFFFFFFFF
                    c = h % dim
                    d[c] = d.get(c, 0) + 1
            else:
                raise ValueError("n must be 2 or 3")

            r = i - s
            for c, v in d.items():
                rows.append(r)
                cols.append(c)
                vals.append(float(v))

        m = sparse.csr_matrix((vals, (rows, cols)), shape=(e - s, dim), dtype=np.float32)
        blocks.append(m)

    X = sparse.vstack(blocks, format="csr")
    return X

def build_features(x_u8: np.ndarray):
    dense_parts = []
    sparse_parts = []

    H = None
    if USE_BYTE_HIST:
        H = byte_hist(x_u8)
        dense_parts.append(H)
    if USE_ENTROPY:
        if H is None:
            H = byte_hist(x_u8)
        dense_parts.append(shannon_entropy_from_hist(H))
    if USE_STATS:
        dense_parts.append(simple_stats(x_u8))
    if USE_SIGNATURE:
        dense_parts.append(signature_features(x_u8))

    if len(dense_parts) > 0:
        X_dense = np.concatenate(dense_parts, axis=1).astype(np.float32, copy=False)
        X_dense = normalize(X_dense, norm="l2").astype(np.float32, copy=False)
        sparse_parts.append(sparse.csr_matrix(X_dense))

    if USE_HASHED_NGRAM:
        X_ng = hashed_ngram_sparse(x_u8, n=NGRAM_N, dim=HASH_DIM, batch_size=2000)
        X_ng = normalize(X_ng, norm="l2")
        sparse_parts.append(X_ng)

    X = sparse.hstack(sparse_parts, format="csr").astype(np.float32)
    return X

# ==========================
# Plot helpers
# ==========================
def save_heatmap(cm, labels, out_path, title):
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=90, fontsize=6)
    plt.yticks(tick, labels, fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

def save_bar(values, names, out_path, title, ylabel):
    plt.figure(figsize=(22, 6))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), names, rotation=90, fontsize=8)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

def save_hist(x, out_path, title, xlabel):
    plt.figure(figsize=(10, 5))
    plt.hist(x, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

# ==========================
# Top confusions
# ==========================
def compute_top_confusions(y_true, y_pred, topn=40):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(EXT_BY_ID)))
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            c = cm[i, j]
            if c > 0:
                pairs.append((int(c), i, j))
    pairs.sort(reverse=True)
    rows = []
    for c, i, j in pairs[:topn]:
        rows.append({"count": c, "true": EXT_BY_ID[i], "pred": EXT_BY_ID[j], "true_id": i, "pred_id": j})
    return cm, pd.DataFrame(rows)

def plot_top_confusion_subset(cm, df_pairs, out_path):
    # subset classes appearing in top confusions
    cls_ids = sorted(set(df_pairs["true_id"].tolist() + df_pairs["pred_id"].tolist()))
    sub = cm[np.ix_(cls_ids, cls_ids)]
    labels = [EXT_BY_ID[i] for i in cls_ids]
    save_heatmap(sub, labels, out_path, "Top confusions (subset, top40)")

# ==========================
# Confidence bins
# ==========================
def pred1_prob_bins(pred1_prob, correct):
    bins = np.linspace(0.0, 1.0, 11)
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (pred1_prob >= a) & (pred1_prob < b)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"[{a:.1f},{b:.1f})",
            "count": int(m.sum()),
            "acc": float(correct[m].mean())
        })
    return pd.DataFrame(rows)

# ==========================
# Main
# ==========================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Load]")
    x_tr, y_tr = load_npz(TRAIN_NPZ)
    x_va, y_va = load_npz(VAL_NPZ)
    print(" train:", x_tr.shape, y_tr.shape)
    print(" val  :", x_va.shape, y_va.shape)

    print("\n[Features]")
    X_tr = build_features(x_tr)
    X_va = build_features(x_va)
    print("X_tr:", X_tr.shape, X_tr.dtype)
    print("X_va:", X_va.shape, X_va.dtype)

    # -------- Model: LinearSVC + Calibration --------
    print("\n[Train: LinearSVC + CalibratedClassifierCV]")
    base = LinearSVC(class_weight="balanced", random_state=SEED)
    clf = CalibratedClassifierCV(base, method=CALIB_METHOD, cv=CALIB_CV)
    clf.fit(X_tr, y_tr)

    # Predict
    proba = clf.predict_proba(X_va)               # (N,75)
    pred  = proba.argmax(axis=1)
    pred1_prob = proba.max(axis=1)

    acc = accuracy_score(y_va, pred)
    print(f"\n[Val] accuracy: {acc:.6f}")

    # Report CSV
    rep = classification_report(
        y_va, pred,
        labels=np.arange(len(EXT_BY_ID)),
        target_names=EXT_BY_ID,
        output_dict=True,
        zero_division=0
    )
    df_rep = pd.DataFrame(rep).T
    df_rep.to_csv(out_dir / "val_class_report.csv", index=True)

    # Confusion matrix
    cm, df_top = compute_top_confusions(y_va, pred, topn=40)
    save_heatmap(cm, EXT_BY_ID, out_dir / "confusion_matrix_75.png", "Confusion Matrix (75 classes)")
    df_top.to_csv(out_dir / "val_top_confusions.csv", index=False)
    plot_top_confusion_subset(cm, df_top, out_dir / "top_confusions_subset_top40.png")

    # F1 + support plot
    df_cls = df_rep.loc[EXT_BY_ID].copy()
    df_cls["class"] = df_cls.index
    df_cls_sorted = df_cls.sort_values("f1-score", ascending=False)
    save_bar(
        df_cls_sorted["f1-score"].values,
        df_cls_sorted["class"].values,
        out_dir / "f1_by_class_sorted.png",
        "F1-score by class (sorted)",
        "f1-score"
    )

    df_sup_sorted = df_cls.sort_values("support", ascending=False)
    save_bar(
        df_sup_sorted["support"].values,
        df_sup_sorted["class"].values,
        out_dir / "support_by_class_sorted.png",
        "Support by class (sorted)",
        "support"
    )

    # Pred prob histogram + bins
    save_hist(pred1_prob, out_dir / "pred1_prob_hist.png", "Top-1 predicted probability histogram", "pred1_prob")
    correct = (pred == y_va)
    df_bins = pred1_prob_bins(pred1_prob, correct)
    df_bins.to_csv(out_dir / "val_pred1_prob_bins.csv", index=False)

    # Save per-sample predictions (compressed)
    # (CSV가 너무 크면 gzip 추천)
    df_pred = pd.DataFrame({
        "i": np.arange(len(y_va)),
        "true_id": y_va,
        "true": [EXT_BY_ID[i] for i in y_va],
        "pred_id": pred,
        "pred": [EXT_BY_ID[i] for i in pred],
        "pred1_prob": pred1_prob,
        "correct": correct.astype(np.int8),
    })
    df_pred.to_csv(out_dir / "val_predictions.csv.gz", index=False, compression="gzip")

    # Low-confidence samples export (for CNN stage)
    low = df_pred[df_pred["pred1_prob"] < LOWCONF_TAU].copy()
    if len(low) > LOWCONF_MAX_ROWS:
        low = low.sample(LOWCONF_MAX_ROWS, random_state=SEED)
    low.to_csv(out_dir / "val_lowconf_samples.csv.gz", index=False, compression="gzip")

    print("\nSaved to:", out_dir.resolve())
    print("- val_class_report.csv")
    print("- val_pred1_prob_bins.csv")
    print("- val_top_confusions.csv")
    print("- confusion_matrix_75.png / f1_by_class_sorted.png / pred1_prob_hist.png / support_by_class_sorted.png / top_confusions_subset_top40.png")
    print("- val_predictions.csv.gz / val_lowconf_samples.csv.gz")

if __name__ == "__main__":
    main()
