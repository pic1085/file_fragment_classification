# ngram_entropy_2stage.py
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

from scipy import sparse

from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


# ==========================
# Config
# ==========================
TRAIN_NPZ = str(DATA_DIR / "train.npz")
VAL_NPZ   = str(DATA_DIR / "val.npz")
OUT_DIR   = N_RUNS_DIR / "runs_2stage_ngram_entropy_sig"

SEED = 42
rng = np.random.default_rng(SEED)

# --- feature switches
USE_BYTE_HIST = True
USE_ENTROPY   = True
USE_STATS     = True

USE_HASHED_NGRAM = True
NGRAM_N = 2                 # 2-gram 추천 (3-gram 비용↑)
HASH_DIM = 65536            # 2^16
NGRAM_REGION = "headtail"   # "all" or "head" or "tail" or "headtail"
HEAD_BYTES = 512
TAIL_BYTES = 512

USE_SIGNATURE = True        # magic-number / signature feature

# --- training
STAGE1_CLASS_WEIGHT = "balanced"   # None or "balanced"
STAGE2_CLASS_WEIGHT = None         # stage2는 None부터
SOLVER = "saga"
MAX_ITER = 5000
C_STAGE1 = 1.0
C_STAGE2 = 1.0
N_JOBS = -1

DENSE_DTYPE = np.float32


# ==========================
# Label mapping (YOUR PROJECT)
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

CAT_NAMES = ["APP","ARCH","AUD","BIN","DOC","ETC","IMG","RAW","VID"]
CAT2ID = {name:i for i,name in enumerate(CAT_NAMES)}

EXT2CAT = {
  # RAW
  "ARW":"RAW","CR2":"RAW","DNG":"RAW","GPR":"RAW","NEF":"RAW","NRW":"RAW","ORF":"RAW","PEF":"RAW","RAF":"RAW","RW2":"RAW","3FR":"RAW",

  # IMG
  "JPG":"IMG","JPEG":"IMG","TIFF":"IMG","HEIC":"IMG","BMP":"IMG","GIF":"IMG","PNG":"IMG","AI":"IMG","EPS":"IMG","PSD":"IMG",

  # VID
  "MOV":"VID","MP4":"VID","3GP":"VID","AVI":"VID","MKV":"VID","OGV":"VID","WEBM":"VID",

  # APP/BIN
  "APK":"APP","JAR":"APP","MSI":"APP","DMG":"APP",
  "EXE":"BIN","DLL":"BIN","ELF":"BIN","MACH-O":"BIN",

  # ARCH
  "7Z":"ARCH","BZ2":"ARCH","DEB":"ARCH","GZ":"ARCH","PKG":"ARCH","RAR":"ARCH","RPM":"ARCH","XZ":"ARCH","ZIP":"ARCH",

  # OFFICE/DOC
  "DOC":"DOC","DOCX":"DOC","KEY":"DOC","PPT":"DOC","PPTX":"DOC","XLS":"DOC","XLSX":"DOC",
  "DJVU":"DOC","EPUB":"DOC","MOBI":"DOC","PDF":"DOC","MD":"DOC","RTF":"DOC","TXT":"DOC","TEX":"DOC",
  "JSON":"DOC","HTML":"DOC","XML":"DOC","LOG":"DOC","CSV":"DOC",

  # AUDIO
  "AIFF":"AUD","FLAC":"AUD","M4A":"AUD","MP3":"AUD","OGG":"AUD","WAV":"AUD","WMA":"AUD",

  # ETC
  "PCAP":"ETC","TTF":"ETC","DWG":"ETC","SQLITE":"ETC",
}

# ext_id(0~74) -> cat_id(0~8)
EXTID2CATID = np.array([CAT2ID[EXT2CAT[e]] for e in EXT_BY_ID], dtype=np.int64)

def extid_to_catid(y_ext: np.ndarray) -> np.ndarray:
    """Vectorized ext_id -> cat_id"""
    y_ext = y_ext.astype(np.int64, copy=False)
    return EXTID2CATID[y_ext]


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
# Feature blocks (dense)
# ==========================
def byte_hist(x_u8: np.ndarray) -> np.ndarray:
    # (N,4096) -> (N,256) freq
    N = x_u8.shape[0]
    out = np.empty((N,256), dtype=DENSE_DTYPE)
    for i in tqdm(range(N), desc="byte_hist"):
        h = np.bincount(x_u8[i], minlength=256).astype(np.float32)
        h /= 4096.0
        out[i] = h
    return out

def shannon_entropy_from_hist(hist: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(hist, eps, 1.0)
    ent = -(p * np.log2(p)).sum(axis=1, keepdims=True).astype(DENSE_DTYPE)
    return ent

def simple_stats(x_u8: np.ndarray) -> np.ndarray:
    # mean/std/min/max + zero_ratio
    x = x_u8.astype(np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std  = x.std(axis=1, keepdims=True)
    mn   = x.min(axis=1, keepdims=True)
    mx   = x.max(axis=1, keepdims=True)
    zero = (x_u8 == 0).mean(axis=1, keepdims=True).astype(np.float32)
    return np.concatenate([mean,std,mn,mx,zero], axis=1).astype(DENSE_DTYPE)


# ==========================
# Signature features (dense)
# ==========================
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
    # check in head only
    N = x_u8.shape[0]
    out = np.zeros((N, len(SIGS)), dtype=DENSE_DTYPE)
    for i in tqdm(range(N), desc="signature"):
        head = bytes(x_u8[i, :64].tolist())
        for j, (_, sig) in enumerate(SIGS):
            out[i, j] = 1.0 if sig in head else 0.0
    return out


# ==========================
# Hashed n-gram sparse
# ==========================
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
    """
    build CSR matrix (N, dim) using hashed n-grams on selected region
    memory-safe: batch -> vstack
    """
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
                rows.append(r); cols.append(c); vals.append(float(v))

        m = sparse.csr_matrix((vals, (rows, cols)), shape=(e - s, dim), dtype=np.float32)
        blocks.append(m)

        # batch마다 메모리 정리
        del rows, cols, vals, m
        gc.collect()

    X = sparse.vstack(blocks, format="csr")
    return X


# ==========================
# Build final feature matrix
# ==========================
def build_features(x_u8: np.ndarray):
    dense_parts = []
    sparse_parts = []

    # byte hist
    if USE_BYTE_HIST:
        H = byte_hist(x_u8)
        dense_parts.append(H)
    else:
        H = None

    # entropy (from hist is fastest)
    if USE_ENTROPY:
        if H is None:
            H = byte_hist(x_u8)
        E = shannon_entropy_from_hist(H)  # (N,1)
        dense_parts.append(E)

    # stats
    if USE_STATS:
        S = simple_stats(x_u8)           # (N,5)
        dense_parts.append(S)

    # signature
    if USE_SIGNATURE:
        G = signature_features(x_u8)     # (N,|SIGS|)
        dense_parts.append(G)

    # dense concat -> sparse
    if len(dense_parts) > 0:
        X_dense = np.concatenate(dense_parts, axis=1).astype(np.float32, copy=False)
        X_dense = normalize(X_dense, norm="l2").astype(np.float32, copy=False)
        sparse_parts.append(sparse.csr_matrix(X_dense))
        del X_dense
        gc.collect()

    # sparse hashed n-gram
    if USE_HASHED_NGRAM:
        X_ng = hashed_ngram_sparse(x_u8, n=NGRAM_N, dim=HASH_DIM, batch_size=2000)
        X_ng = normalize(X_ng, norm="l2")
        sparse_parts.append(X_ng)
        del X_ng
        gc.collect()

    X = sparse.hstack(sparse_parts, format="csr").astype(np.float32)
    return X


# ==========================
# Plot helpers
# ==========================
def save_confusion(cm, labels, out_path, title):
    plt.figure(figsize=(10, 9))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=45, ha="right")
    plt.yticks(tick, labels)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


# ==========================
# 2-stage training/eval
# ==========================
def train_stage1(X_tr, y_ext_tr, X_va, y_ext_va, out_dir: Path):
    cat_tr = extid_to_catid(y_ext_tr)
    cat_va = extid_to_catid(y_ext_va)

    print("\n[Stage-1: category]")
    print("Unique ext:", len(np.unique(y_ext_tr)))
    print("Unique cat:", len(np.unique(cat_tr)))
    print("Top cats:\n", pd.Series(cat_tr).map(lambda i: CAT_NAMES[i]).value_counts().head(10))

    clf = LogisticRegression(
        solver=SOLVER,
        max_iter=MAX_ITER,
        n_jobs=N_JOBS,
        C=C_STAGE1,
        class_weight=STAGE1_CLASS_WEIGHT,
        verbose=0,
    )
    clf.fit(X_tr, cat_tr)

    pred = clf.predict(X_va)
    acc = accuracy_score(cat_va, pred)
    print(classification_report(cat_va, pred, target_names=CAT_NAMES, digits=4))
    print("Stage1 acc:", acc)

    cm = confusion_matrix(cat_va, pred, labels=np.arange(len(CAT_NAMES)))
    save_confusion(cm, CAT_NAMES, out_dir / "stage1_confusion_cat.png", "Stage-1 Confusion (Category)")

    return clf


def train_stage2_per_cat(X_tr, y_ext_tr, X_va, y_ext_va, stage1_clf, out_dir: Path):
    cat_tr = extid_to_catid(y_ext_tr)
    cat_va = extid_to_catid(y_ext_va)

    print("\n[Stage-2: per-category extension]")
    models = {}
    val_acc_by_cat = {}

    # per-category extension classifier
    for cat_id, cat_name in enumerate(CAT_NAMES):
        idx_tr = np.where(cat_tr == cat_id)[0]
        idx_va = np.where(cat_va == cat_id)[0]

        if len(idx_tr) < 2 or len(np.unique(y_ext_tr[idx_tr])) < 2:
            print(f"  - {cat_name}: skip (too few classes/samples)")
            continue

        clf = LogisticRegression(
            solver=SOLVER,
            max_iter=MAX_ITER,
            n_jobs=N_JOBS,
            C=C_STAGE2,
            class_weight=STAGE2_CLASS_WEIGHT,
            verbose=0,
        )
        clf.fit(X_tr[idx_tr], y_ext_tr[idx_tr])

        pred_va = clf.predict(X_va[idx_va])
        acc = accuracy_score(y_ext_va[idx_va], pred_va)
        val_acc_by_cat[cat_name] = acc
        models[cat_id] = clf
        print(f"  - {cat_name}: val_acc={acc:.4f} (val={len(idx_va)})")

    # pipeline evaluation (stage1 predicted cat -> stage2)
    print("\n[Pipeline]")
    pred_cat = stage1_clf.predict(X_va)

    final_pred_ext = np.full_like(y_ext_va, fill_value=-1)
    for cat_id in range(len(CAT_NAMES)):
        idx = np.where(pred_cat == cat_id)[0]
        if len(idx) == 0:
            continue
        if cat_id not in models:
            continue
        final_pred_ext[idx] = models[cat_id].predict(X_va[idx])

    covered = (final_pred_ext != -1)
    cov = covered.mean()
    acc = accuracy_score(y_ext_va[covered], final_pred_ext[covered]) if cov > 0 else 0.0
    print(f"Pipeline ext acc (covered): {acc:.4f} | coverage={cov*100:.2f}%")

    # save
    pd.Series(val_acc_by_cat).sort_values(ascending=False).to_csv(out_dir / "stage2_val_acc_by_cat.csv")
    np.save(out_dir / "pipeline_pred_ext.npy", final_pred_ext)

    return models


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # quick mapping sanity print
    df_map = pd.DataFrame({
        "ext_id": np.arange(75),
        "ext": EXT_BY_ID,
        "cat": [CAT_NAMES[c] for c in EXTID2CATID],
    })
    df_map.to_csv(out_dir / "extid_to_cat.csv", index=False)

    print("[Load]")
    x_tr, y_tr = load_npz(TRAIN_NPZ)
    x_va, y_va = load_npz(VAL_NPZ)
    print(" train:", x_tr.shape, y_tr.shape)
    print(" val  :", x_va.shape, y_va.shape)

    # features
    print("\n[Features: train]")
    X_tr = build_features(x_tr)
    print("X_tr:", X_tr.shape, X_tr.dtype)

    print("\n[Features: val]")
    X_va = build_features(x_va)
    print("X_va:", X_va.shape, X_va.dtype)

    # stage1
    stage1_clf = train_stage1(X_tr, y_tr, X_va, y_va, out_dir)

    # stage2
    _ = train_stage2_per_cat(X_tr, y_tr, X_va, y_va, stage1_clf, out_dir)

    print("\nSaved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
