#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.paths import DATA_DIR, N_RUNS_DIR
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from scipy.sparse import csr_matrix, vstack, hstack


# =========================
# 0) Config
# =========================
NPZ_TRAIN = str(DATA_DIR / "train.npz")
NPZ_VAL   = None   # val.npz 있으면 경로로 지정
OUT_DIR   = N_RUNS_DIR / "runs_2stage_ngram_entropy_safe"

SEED = 42

# --- Feature switches ---
USE_HIST = True
USE_ENTROPY = True
USE_STATS = True
USE_HASHED_2GRAM = True

WINDOW = 256          # entropy window
NGRAM_N = 2
HASH_DIM = 2**16      # 65536 (처음엔 이거 추천). 더 높이면 성능↑, 메모리/시간↑
BATCH_SIZE = 2000     # 1000~5000 권장 (RAM에 따라)
POWER_OF_TWO_HASHDIM = True  # HASH_DIM이 2^k면 True (빠른 마스크)

# --- Training ---
STAGE1_C = 2.0
STAGE2_C = 2.0
MAX_ITER = 300
N_JOBS = os.cpu_count() or 8

MAX_N_TRAIN = None   # 너무 크면 200000 등으로 줄여 테스트
MAX_N_VAL   = None


# =========================
# 1) Label mapping (YOU MUST EDIT)
# =========================
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
DEFAULT_CAT="ETC"
DEFAULT_CAT = "OTHER"


# =========================
# 2) Data loading
# =========================
def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    x = d["x"].astype(np.uint8, copy=False)  # (N,4096)
    y = d["y"].astype(np.int64, copy=False)  # (N,)
    assert x.ndim == 2 and x.shape[1] == 4096
    return x, y

def subsample(x, y, max_n, seed):
    if max_n is None or len(x) <= max_n:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_n, replace=False)
    return x[idx], y[idx]


# =========================
# 3) Feature blocks (dense)
# =========================
def byte_hist(x_uint8: np.ndarray) -> np.ndarray:
    N = x_uint8.shape[0]
    out = np.empty((N, 256), dtype=np.float32)
    for i in tqdm(range(N), desc="byte_hist"):
        h = np.bincount(x_uint8[i], minlength=256).astype(np.float32)
        out[i] = h / 4096.0
    return out

def shannon_entropy(counts: np.ndarray) -> float:
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def entropy_features(x_uint8: np.ndarray, window: int = 256) -> np.ndarray:
    N = x_uint8.shape[0]
    out = np.empty((N, 5), dtype=np.float32)
    nwin = 4096 // window
    for i in tqdm(range(N), desc="entropy"):
        b = x_uint8[i]
        cg = np.bincount(b, minlength=256).astype(np.float32)
        eg = shannon_entropy(cg)
        ws = []
        for w in range(nwin):
            seg = b[w*window:(w+1)*window]
            c = np.bincount(seg, minlength=256).astype(np.float32)
            ws.append(shannon_entropy(c))
        ws = np.asarray(ws, dtype=np.float32)
        out[i] = np.array([eg, ws.mean(), ws.std(), ws.min(), ws.max()], dtype=np.float32)
    return out

def simple_stats(x_uint8: np.ndarray) -> np.ndarray:
    N = x_uint8.shape[0]
    out = np.empty((N, 6), dtype=np.float32)
    printable = np.zeros(256, dtype=bool)
    printable[32:127] = True
    printable[9] = printable[10] = printable[13] = True
    for i in tqdm(range(N), desc="stats"):
        b = x_uint8[i]
        mean = float(b.mean()) / 255.0
        std = float(b.std()) / 255.0
        zero_ratio = float((b == 0).mean())
        printable_ratio = float(printable[b].mean())
        high_ratio = float((b >= 128).mean())
        uniq_ratio = float(len(np.unique(b)) / 256.0)
        out[i] = np.array([mean, std, zero_ratio, printable_ratio, high_ratio, uniq_ratio], dtype=np.float32)
    return out


# =========================
# 4) Hashed 2-gram sparse (OOM-safe)
# =========================
def _hash_u16_to_bucket(u16: np.ndarray, hash_dim: int) -> np.ndarray:
    """
    u16: uint16 array
    return: int32 bucket indices in [0, hash_dim)
    """
    # 가벼운 multiplicative hash
    # (uint32로 확장 후 곱)
    x = u16.astype(np.uint32)
    x = (x * 2654435761) & 0xFFFFFFFF  # Knuth multiplicative hash
    if POWER_OF_TWO_HASHDIM:
        return (x & (hash_dim - 1)).astype(np.int32)
    else:
        return (x % hash_dim).astype(np.int32)

def hashed_2gram_sparse_batched(x_uint8: np.ndarray, hash_dim: int, batch_size: int) -> csr_matrix:
    """
    4096 bytes -> 4095 2-gram
    각 샘플에서 2-gram을 uint16로 만들고, hash bucket에 count를 누적하여 sparse row 생성
    배치로 CSR block을 만든 후 vstack
    """
    N = x_uint8.shape[0]
    blocks = []
    for start in tqdm(range(0, N, batch_size), desc="hashed_2gram_sparse"):
        end = min(N, start + batch_size)
        xb = x_uint8[start:end]  # (B,4096)

        rows = []
        cols = []
        data = []

        for i in range(end - start):
            b = xb[i]
            # uint16 2-gram code: b[t]*256 + b[t+1]
            u16 = (b[:-1].astype(np.uint16) << 8) | b[1:].astype(np.uint16)
            buckets = _hash_u16_to_bucket(u16, hash_dim)

            # row compression: unique bucket + counts
            uniq, cnt = np.unique(buckets, return_counts=True)

            r = start + i
            rows.append(np.full_like(uniq, r, dtype=np.int32))
            cols.append(uniq.astype(np.int32))
            data.append(cnt.astype(np.float32))

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)

        block = csr_matrix((data, (rows - start, cols)), shape=(end - start, hash_dim))
        blocks.append(block)

    return vstack(blocks).tocsr()


def build_features(x_uint8: np.ndarray) -> csr_matrix:
    dense_parts = []
    if USE_HIST:
        dense_parts.append(byte_hist(x_uint8))
    if USE_ENTROPY:
        dense_parts.append(entropy_features(x_uint8, window=WINDOW))
    if USE_STATS:
        dense_parts.append(simple_stats(x_uint8))

    dense = np.concatenate(dense_parts, axis=1).astype(np.float32, copy=False)
    X_dense = csr_matrix(dense)  # sparse로 바꿔서 결합

    if USE_HASHED_2GRAM:
        X_ng = hashed_2gram_sparse_batched(x_uint8, hash_dim=HASH_DIM, batch_size=BATCH_SIZE)
        X = hstack([X_dense, X_ng]).tocsr()
        return X

    return X_dense


# =========================
# 5) Labels -> category
# =========================
def y_to_ext(y: np.ndarray) -> np.ndarray:
    return np.array([EXT_BY_ID[int(i)] for i in y], dtype=object)

def ext_to_cat(ext: np.ndarray) -> np.ndarray:
    return np.array([EXT2CAT.get(str(e).upper(), DEFAULT_CAT) for e in ext], dtype=object)

def make_cat_ids(cats: np.ndarray):
    uniq = sorted(set(cats.tolist()))
    c2i = {c: i for i, c in enumerate(uniq)}
    i2c = {i: c for c, i in c2i.items()}
    ycat = np.array([c2i[c] for c in cats], dtype=np.int64)
    return ycat, c2i, i2c


# =========================
# 6) Models
# =========================
def make_lr(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            C=C,
            max_iter=MAX_ITER,
            n_jobs=N_JOBS,
            solver="saga",
            multi_class="auto",
        ))
    ])

def plot_confusion(cm: np.ndarray, labels: list, title: str, save_path: Path, max_labels: int = 30):
    plt.figure(figsize=(10, 8))
    if len(labels) > max_labels:
        labels = labels[:max_labels]
        cm = cm[:max_labels, :max_labels]
        title += f" (top {max_labels})"
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(labels)), labels, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


# =========================
# 7) Main
# =========================
def main():
    out_dir = Path(OUT_DIR)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    print("[Load]")
    x_tr, y_tr = load_npz(NPZ_TRAIN)
    x_tr, y_tr = subsample(x_tr, y_tr, MAX_N_TRAIN, SEED)

    if NPZ_VAL is not None:
        x_va, y_va = load_npz(NPZ_VAL)
        x_va, y_va = subsample(x_va, y_va, MAX_N_VAL, SEED + 1)
    else:
        x_tr, x_va, y_tr, y_va = train_test_split(
            x_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr
        )

    print(" train:", x_tr.shape, y_tr.shape)
    print(" val  :", x_va.shape, y_va.shape)

    # label -> ext -> cat
    ext_tr = y_to_ext(y_tr)
    ext_va = y_to_ext(y_va)
    cat_tr = ext_to_cat(ext_tr)
    cat_va = ext_to_cat(ext_va)

    ycat_tr, c2i, i2c = make_cat_ids(cat_tr)
    ycat_va = np.array([c2i.get(c, -1) for c in cat_va], dtype=np.int64)
    assert (ycat_va >= 0).all(), "Validation has unseen category label. EXT2CAT 확인 필요"

    # Features
    print("\n[Features: train]")
    X_tr = build_features(x_tr)
    print("X_tr:", X_tr.shape, X_tr.dtype)

    print("\n[Features: val]")
    X_va = build_features(x_va)
    print("X_va:", X_va.shape, X_va.dtype)

    # Stage-1
    print("\n[Stage-1: category]")
    print("Unique ext:", len(np.unique(ext_tr)))
    print("Unique cat:", len(np.unique(cat_tr)))
    print("Top cats:", pd.Series(cat_tr).value_counts().head(10))
    stage1 = make_lr(STAGE1_C)
    stage1.fit(X_tr, ycat_tr)
    joblib.dump(stage1, out_dir / "models" / "stage1_cat.joblib")

    pred_cat = stage1.predict(X_va)
    acc_cat = accuracy_score(ycat_va, pred_cat)
    cat_names = [i2c[i] for i in range(len(i2c))]

    rep1 = classification_report(ycat_va, pred_cat, target_names=cat_names, digits=4)
    (out_dir / "stage1_report.txt").write_text(rep1, encoding="utf-8")
    print(rep1)
    print("Stage1 acc:", acc_cat)

    cm1 = confusion_matrix(ycat_va, pred_cat, labels=list(range(len(cat_names))))
    plot_confusion(cm1, cat_names, "Stage-1 Confusion (Category)", out_dir / "confusion_stage1.png")

    # Stage-2
    print("\n[Stage-2: per-category extension]")
    pipeline_pred_ext = np.full_like(y_va, fill_value=-1)

    for cat, cat_id in c2i.items():
        idx_tr = np.where(ycat_tr == cat_id)[0]
        idx_va = np.where(ycat_va == cat_id)[0]

        if len(idx_tr) < 100 or len(idx_va) < 20:
            print(f"  - skip {cat}: train={len(idx_tr)} val={len(idx_va)}")
            continue

        model = make_lr(STAGE2_C)
        model.fit(X_tr[idx_tr], y_tr[idx_tr])
        joblib.dump(model, out_dir / "models" / f"stage2_ext_{cat}.joblib")

        pred_ext = model.predict(X_va[idx_va])
        pipeline_pred_ext[idx_va] = pred_ext

        acc = accuracy_score(y_va[idx_va], pred_ext)
        print(f"  - {cat}: val_acc={acc:.4f} (val={len(idx_va)})")

    covered = (pipeline_pred_ext >= 0)
    pipeline_acc = accuracy_score(y_va[covered], pipeline_pred_ext[covered]) if covered.any() else 0.0
    print("\n[Pipeline]")
    print(f"Pipeline ext acc (covered): {pipeline_acc:.4f} | coverage={covered.mean()*100:.2f}%")

    metrics = {
        "stage1_cat_acc": float(acc_cat),
        "pipeline_ext_acc_covered": float(pipeline_acc),
        "coverage_ratio": float(covered.mean()),
        "hash_dim": int(HASH_DIM),
        "batch_size": int(BATCH_SIZE),
        "use_2gram": bool(USE_HASHED_2GRAM),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
