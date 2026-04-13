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

from core.paths import DATA_DIR, N_RUNS_DIR
from tqdm import tqdm
from collections import defaultdict

from scipy import sparse

from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier

# =========================================================
# Config
# =========================================================
TRAIN_NPZ = str(DATA_DIR / "train.npz")
VAL_NPZ   = str(DATA_DIR / "val.npz")
OUT_DIR   = N_RUNS_DIR / "runs_2stage_svc_calibrated_3"

SEED = 42
rng = np.random.default_rng(SEED)

# ---- feature switches (dense block)
USE_BYTE_HIST = True
USE_ENTROPY   = True
USE_STATS     = True
USE_SIGNATURE = True

# ---- stage-wise feature design
USE_HASHED_NGRAM_STAGE1 = True   # stage1은 빠르게(273차원)
USE_HASHED_NGRAM_STAGE2 = True    # stage2는 강하게(65k차원)

# ---- hashed ngram
NGRAM_N = 2
HASH_DIM = 65536
NGRAM_REGION = "headtail"  # "all" | "head" | "tail" | "headtail"
HEAD_BYTES = 256
TAIL_BYTES = 256
NGRAM_BATCH = 2000

# ---- model
# LinearSVC는 확률이 없어서 calibration을 붙임
SVC_C_STAGE1 = 1.0
SVC_C_STAGE2 = 1.0
CALIB_CV = 2          # 2~3 권장(데이터 크면 2가 현실적)
CALIB_METHOD = "sigmoid"  # "sigmoid" or "isotonic"(isotonic은 더 무겁다)

# ---- Top-k
TOPK_CAT = 3
TOPK_EXT = 3

# ---- threshold (Stage-1 max prob 기준으로 confident면 top1-cat만 쓰고,
# 아니면 topK-cat 혼합)
TAU = 0.25


# =========================================================
# Label mapping (너가 준 그대로)
# =========================================================
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

EXT2CAT = {
  "ARW":"RAW","CR2":"RAW","DNG":"RAW","GPR":"RAW","NEF":"RAW","NRW":"RAW","ORF":"RAW","PEF":"RAW","RAF":"RAW","RW2":"RAW","3FR":"RAW",
  "JPG":"IMG","JPEG":"IMG","TIFF":"IMG","HEIC":"IMG","BMP":"IMG","GIF":"IMG","PNG":"IMG","AI":"IMG","EPS":"IMG","PSD":"IMG",
  "MOV":"VID","MP4":"VID","3GP":"VID","AVI":"VID","MKV":"VID","OGV":"VID","WEBM":"VID",
  "APK":"APP","JAR":"APP","MSI":"APP","DMG":"APP",
  "EXE":"BIN","DLL":"BIN","ELF":"BIN","MACH-O":"BIN",
  "7Z":"ARCH","BZ2":"ARCH","DEB":"ARCH","GZ":"ARCH","PKG":"ARCH","RAR":"ARCH","RPM":"ARCH","XZ":"ARCH","ZIP":"ARCH",
  "DOC":"DOC","DOCX":"DOC","KEY":"DOC","PPT":"DOC","PPTX":"DOC","XLS":"DOC","XLSX":"DOC",
  "DJVU":"DOC","EPUB":"DOC","MOBI":"DOC","PDF":"DOC","MD":"DOC","RTF":"DOC","TXT":"DOC","TEX":"DOC",
  "JSON":"DOC","HTML":"DOC","XML":"DOC","LOG":"DOC","CSV":"DOC",
  "AIFF":"AUD","FLAC":"AUD","M4A":"AUD","MP3":"AUD","OGG":"AUD","WAV":"AUD","WMA":"AUD",
  "PCAP":"ETC","TTF":"ETC","DWG":"ETC","SQLITE":"ETC",
}

CAT_NAMES = ["APP","ARCH","AUD","BIN","DOC","ETC","IMG","RAW","VID"]
CAT2ID = {c:i for i,c in enumerate(CAT_NAMES)}
EXTID2CATID = np.array([CAT2ID[EXT2CAT[e]] for e in EXT_BY_ID], dtype=np.int64)

def ext_to_cat_id(ext_id: int) -> int:
    return int(EXTID2CATID[ext_id])


# =========================================================
# Load
# =========================================================
def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    x = data["x"].astype(np.uint8, copy=False)   # (N,4096)
    y = data["y"].astype(np.int64, copy=False)   # (N,)
    assert x.ndim == 2 and x.shape[1] == 4096
    return x, y


# =========================================================
# Feature blocks (dense)
# =========================================================
def byte_hist(x_u8: np.ndarray) -> np.ndarray:
    N = x_u8.shape[0]
    out = np.empty((N,256), dtype=np.float32)
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
    return np.concatenate([mean,std,mn,mx,zero], axis=1).astype(np.float32)

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


# =========================================================
# Hashed n-gram sparse
# =========================================================
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
                for k in range(len(seq)-1):
                    key = (int(seq[k]) << 8) | int(seq[k+1])
                    h = (key * 2654435761) & 0xFFFFFFFF
                    c = h % dim
                    d[c] = d.get(c, 0) + 1
            elif n == 3:
                for k in range(len(seq)-2):
                    key = (int(seq[k]) << 16) | (int(seq[k+1]) << 8) | int(seq[k+2])
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

    return sparse.vstack(blocks, format="csr")


# =========================================================
# Build features (stage-wise)
# =========================================================
def build_features(x_u8: np.ndarray, use_hashed_ngram: bool):
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

    if use_hashed_ngram:
        X_ng = hashed_ngram_sparse(x_u8, n=NGRAM_N, dim=HASH_DIM, batch_size=NGRAM_BATCH)
        X_ng = normalize(X_ng, norm="l2")
        sparse_parts.append(X_ng)

    return sparse.hstack(sparse_parts, format="csr").astype(np.float32)


# =========================================================
# Plot helpers
# =========================================================
def save_confusion(cm, labels, out_path, title):
    plt.figure(figsize=(10,9))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=45, ha="right")
    plt.yticks(tick, labels)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


# =========================================================
# Top-k utilities
# =========================================================
def topk_from_proba(proba: np.ndarray, k: int):
    idx = np.argpartition(-proba, kth=min(k-1, proba.shape[1]-1), axis=1)[:, :k]
    row = np.arange(proba.shape[0])[:, None]
    idx_sorted = idx[np.arange(idx.shape[0])[:, None], np.argsort(-proba[row, idx], axis=1)]
    prob_sorted = proba[row, idx_sorted]
    return idx_sorted, prob_sorted

def topk_hit_rate(y_true: np.ndarray, topk_idx: np.ndarray):
    return np.mean([y_true[i] in topk_idx[i] for i in range(len(y_true))])

# =========================================================
# sweep tau for pipeline eval
# =========================================================

def sweep_tau_pipeline(stage1_clf, stage2_models, X1_va, X2_va, y_va, taus, topk_cat, topk_ext):
    rows = []
    for tau in taus:
        top1, topk = pipeline_eval_only(
            stage1_clf, stage2_models, X1_va, X2_va, y_va,
            tau=tau, topk_cat=topk_cat, topk_ext=topk_ext
        )
        rows.append({"tau": float(tau), "top1": float(top1), f"top{topk_ext}": float(topk)})
        print(f"tau={tau:.2f} | top1={top1:.4f} | top{topk_ext}={topk:.4f}")
    return pd.DataFrame(rows)

def pipeline_eval_only(stage1_clf, stage2_models, X1_va, X2_va, y_ext_va, tau, topk_cat, topk_ext):
    proba_cat = stage1_clf.predict_proba(X1_va)
    top_cat_idx, top_cat_prob = topk_from_proba(proba_cat, k=topk_cat)
    maxp = proba_cat.max(axis=1)

    N = X1_va.shape[0]
    hit1 = 0
    hitk = 0

    for i in range(N):
        true_ext = int(y_ext_va[i])

        if maxp[i] >= tau:
            cats = [int(top_cat_idx[i, 0])]
            cats_w = [float(top_cat_prob[i, 0])]
        else:
            cats = [int(c) for c in top_cat_idx[i]]
            cats_w = [float(p) for p in top_cat_prob[i]]

        merged = defaultdict(float)
        for c, w in zip(cats, cats_w):
            if c not in stage2_models:
                continue
            m = stage2_models[c]
            p = m.predict_proba(X2_va[i])
            for ext_id, prob in zip(m.classes_, p[0]):
                merged[int(ext_id)] += w * float(prob)

        if len(merged) == 0:
            continue

        items = sorted(merged.items(), key=lambda x: -x[1])[:topk_ext]
        pred_ext_ids = [e for e, _ in items]

        if pred_ext_ids[0] == true_ext:
            hit1 += 1
        if true_ext in pred_ext_ids:
            hitk += 1

    return hit1 / N, hitk / N

# =========================================================
# Stage-1: category (SVC + Calibration)
# =========================================================
def train_stage1(X1_tr, y_ext_tr, X1_va, y_ext_va, out_dir: Path):
    ycat_tr = np.array([ext_to_cat_id(int(e)) for e in y_ext_tr], dtype=np.int64)
    ycat_va = np.array([ext_to_cat_id(int(e)) for e in y_ext_va], dtype=np.int64)

    print("\n[Stage-1: category]")
    print("Unique cat:", len(np.unique(ycat_tr)))
    print(pd.Series([CAT_NAMES[i] for i in ycat_tr]).value_counts())

    base = SGDClassifier(
        loss="log_loss",
        alpha=3e-6,
        penalty="l2",
        max_iter=30,
        tol=1e-3,
        random_state=SEED,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.05,
        n_iter_no_change=3,
    )
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X1_tr, ycat_tr)

    proba = clf.predict_proba(X1_va)
    pred1 = proba.argmax(axis=1)
    acc1 = accuracy_score(ycat_va, pred1)

    topk_idx, topk_prob = topk_from_proba(proba, k=TOPK_CAT)
    hitk = topk_hit_rate(ycat_va, topk_idx)

    print(f"Stage1 acc (Top-1): {acc1:.6f}")
    print(f"Stage1 Top-{TOPK_CAT} hit-rate: {hitk:.6f}")

    cm = confusion_matrix(ycat_va, pred1, labels=np.arange(len(CAT_NAMES)))
    save_confusion(cm, CAT_NAMES, out_dir / "stage1_confusion_cat.png", "Stage-1 Confusion (Category)")

    # 샘플 로그 저장
    rows = []
    for i in range(min(200, len(ycat_va))):
        r = {"i": i, "true_cat": CAT_NAMES[ycat_va[i]]}
        for j in range(TOPK_CAT):
            r[f"pred{j+1}_cat"] = CAT_NAMES[int(topk_idx[i, j])]
            r[f"pred{j+1}_prob"] = float(topk_prob[i, j])
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "stage1_topk_samples.csv", index=False)

    return clf, ycat_va


# =========================================================
# Stage-2: per-category extension (SVC + Calibration)
# =========================================================
def train_stage2_models(X2_tr, y_ext_tr, out_dir: Path):
    ycat_tr = np.array([ext_to_cat_id(int(e)) for e in y_ext_tr], dtype=np.int64)

    models = {}
    for cat_id, cat_name in enumerate(CAT_NAMES):
        idx = np.where(ycat_tr == cat_id)[0]
        if len(idx) < 50:
            print(f"[Stage2] skip {cat_name} (too few samples)")
            continue

        y_sub = y_ext_tr[idx]
        if len(np.unique(y_sub)) < 2:
            print(f"[Stage2] skip {cat_name} (only one ext)")
            continue

        base = LinearSVC(C=SVC_C_STAGE2, class_weight="balanced", max_iter=8000)
        clf = CalibratedClassifierCV(base, method=CALIB_METHOD, cv=CALIB_CV)
        clf.fit(X2_tr[idx], y_sub)

        models[cat_id] = clf
        print(f"[Stage2] trained cat={cat_name} | n={len(idx)} | #ext={len(np.unique(y_sub))}")

    # 저장(메타)
    pd.Series({CAT_NAMES[k]: len(models[k].classes_) for k in models}).to_csv(out_dir / "stage2_ext_count_by_cat.csv")
    return models


# =========================================================
# Pipeline: stage1 topk-cat -> stage2 mixture -> topk-ext
# =========================================================
def pipeline_predict_topk_ext(stage1_clf, stage2_models, X1_va, X2_va, y_ext_va,
                             out_dir: Path, tau: float, topk_cat: int, topk_ext: int):

    proba_cat = stage1_clf.predict_proba(X1_va)
    top_cat_idx, top_cat_prob = topk_from_proba(proba_cat, k=topk_cat)
    maxp = proba_cat.max(axis=1)

    N = X1_va.shape[0]
    hit1 = 0
    hitk = 0
    out_rows = []

    for i in tqdm(range(N), desc="pipeline_topk_ext"):
        true_ext = int(y_ext_va[i])
        true_ext_name = EXT_BY_ID[true_ext]
        true_cat = ext_to_cat_id(true_ext)

        # confident면 top1-cat만, 아니면 topk-cat 혼합
        if maxp[i] >= tau:
            cats = [int(top_cat_idx[i, 0])]
            cats_w = [float(top_cat_prob[i, 0])]
        else:
            cats = [int(c) for c in top_cat_idx[i]]
            cats_w = [float(p) for p in top_cat_prob[i]]

        merged = defaultdict(float)  # ext_id -> score
        for c, w in zip(cats, cats_w):
            if c not in stage2_models:
                continue
            m = stage2_models[c]
            p = m.predict_proba(X2_va[i])  # (1, #ext_in_cat)
            cls = m.classes_
            for ext_id, prob in zip(cls, p[0]):
                merged[int(ext_id)] += w * float(prob)

        if len(merged) == 0:
            pred_exts, pred_scores = [], []
        else:
            items = sorted(merged.items(), key=lambda x: -x[1])[:topk_ext]
            pred_exts = [EXT_BY_ID[e] for e, _ in items]
            pred_scores = [s for _, s in items]

        if len(pred_exts) > 0:
            if pred_exts[0] == true_ext_name:
                hit1 += 1
            if true_ext_name in pred_exts:
                hitk += 1

        row = {
            "i": i,
            "true_ext": true_ext_name,
            "true_cat": CAT_NAMES[true_cat],
            "stage1_maxp": float(maxp[i]),
        }
        for j in range(topk_cat):
            row[f"cat{j+1}"] = CAT_NAMES[int(top_cat_idx[i, j])]
            row[f"cat{j+1}_p"] = float(top_cat_prob[i, j])

        for j in range(topk_ext):
            if j < len(pred_exts):
                row[f"pred_ext{j+1}"] = pred_exts[j]
                row[f"pred_ext{j+1}_score"] = float(pred_scores[j])
            else:
                row[f"pred_ext{j+1}"] = ""
                row[f"pred_ext{j+1}_score"] = 0.0

        out_rows.append(row)

    top1_acc = hit1 / N
    topk_hit = hitk / N
    print(f"\n[Pipeline ext]")
    print(f"tau={tau:.2f} | Top-1 ext acc: {top1_acc:.6f} | Top-{topk_ext} ext hit-rate: {topk_hit:.6f}")

    pd.DataFrame(out_rows).to_csv(out_dir / "pipeline_topk_ext.csv", index=False)


# =========================================================
# Main
# =========================================================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Load]")
    x_tr, y_tr = load_npz(TRAIN_NPZ)
    x_va, y_va = load_npz(VAL_NPZ)
    print(" train:", x_tr.shape, y_tr.shape)
    print(" val  :", x_va.shape, y_va.shape)

    # -------- Stage-1 features --------
    print("\n[Features: train | Stage-1]")
    X1_tr = build_features(x_tr, use_hashed_ngram=USE_HASHED_NGRAM_STAGE1)
    print("X1_tr:", X1_tr.shape, X1_tr.dtype)

    print("\n[Features: val | Stage-1]")
    X1_va = build_features(x_va, use_hashed_ngram=USE_HASHED_NGRAM_STAGE1)
    print("X1_va:", X1_va.shape, X1_va.dtype)

    # -------- Stage-2 features --------
    print("\n[Features: train | Stage-2]")
    X2_tr = build_features(x_tr, use_hashed_ngram=USE_HASHED_NGRAM_STAGE2)
    print("X2_tr:", X2_tr.shape, X2_tr.dtype)

    print("\n[Features: val | Stage-2]")
    X2_va = build_features(x_va, use_hashed_ngram=USE_HASHED_NGRAM_STAGE2)
    print("X2_va:", X2_va.shape, X2_va.dtype)

    # Stage-1 train
    stage1_clf, _ = train_stage1(X1_tr, y_tr, X1_va, y_va, out_dir)

    # Stage-2 train
    print("\n[Train Stage-2 models]")
    stage2_models = train_stage2_models(X2_tr, y_tr, out_dir)
    
    taus = np.linspace(0.05, 0.8, 16)
    df_tau = sweep_tau_pipeline(stage1_clf, stage2_models, X1_va, X2_va, y_va, taus, TOPK_CAT, TOPK_EXT)
    df_tau.to_csv(out_dir / "pipeline_tau_sweep.csv", index=False)
    print(df_tau.sort_values("top1", ascending=False).head(10))

    # -------- Choose best tau and run final pipeline save --------
    best_tau = float(df_tau.sort_values("top1", ascending=False).iloc[0]["tau"])
    print(f"\n[Final pipeline] best_tau={best_tau:.3f}")
    
    # Pipeline
    pipeline_predict_topk_ext(
        stage1_clf, stage2_models,
        X1_va, X2_va, y_va,
        out_dir,
        tau=best_tau,
        topk_cat=TOPK_CAT,
        topk_ext=TOPK_EXT
    )

    print("\nSaved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
