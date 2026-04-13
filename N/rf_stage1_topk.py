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
from sklearn.ensemble import RandomForestClassifier


# ==========================
# Config
# ==========================
TRAIN_NPZ = str(DATA_DIR / "train.npz")
VAL_NPZ   = str(DATA_DIR / "val.npz")
OUT_DIR   = N_RUNS_DIR / "runs_stage1_rf_topk"

SEED = 42
rng = np.random.default_rng(SEED)

# ---- Stage-1 feature switches (RF용: dense만 권장)
USE_BYTE_HIST = True
USE_ENTROPY   = True
USE_STATS     = True
USE_SIGNATURE = True

# ---- Top-k
TOP_K = 3     # 2 또는 3 추천

# ---- RandomForest hyperparams (처음엔 보수적으로)
RF_N_ESTIMATORS = 400
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 2
RF_N_JOBS = -1

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

  # DOC
  "DOC":"DOC","DOCX":"DOC","KEY":"DOC","PPT":"DOC","PPTX":"DOC","XLS":"DOC","XLSX":"DOC",
  "DJVU":"DOC","EPUB":"DOC","MOBI":"DOC","PDF":"DOC","MD":"DOC","RTF":"DOC","TXT":"DOC","TEX":"DOC",
  "JSON":"DOC","HTML":"DOC","XML":"DOC","LOG":"DOC","CSV":"DOC",

  # AUD
  "AIFF":"AUD","FLAC":"AUD","M4A":"AUD","MP3":"AUD","OGG":"AUD","WAV":"AUD","WMA":"AUD",

  # ETC
  "PCAP":"ETC","TTF":"ETC","DWG":"ETC","SQLITE":"ETC",
}

EXTID2CATID = np.array([CAT2ID[EXT2CAT[e]] for e in EXT_BY_ID], dtype=np.int64)

def extid_to_catid(y_ext: np.ndarray) -> np.ndarray:
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
# Dense feature blocks
# ==========================
def byte_hist(x_u8: np.ndarray) -> np.ndarray:
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
    x = x_u8.astype(np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std  = x.std(axis=1, keepdims=True)
    mn   = x.min(axis=1, keepdims=True)
    mx   = x.max(axis=1, keepdims=True)
    zero = (x_u8 == 0).mean(axis=1, keepdims=True).astype(np.float32)
    return np.concatenate([mean,std,mn,mx,zero], axis=1).astype(DENSE_DTYPE)

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
    out = np.zeros((N, len(SIGS)), dtype=DENSE_DTYPE)
    for i in tqdm(range(N), desc="signature"):
        head = bytes(x_u8[i, :64].tolist())
        for j, (_, sig) in enumerate(SIGS):
            out[i, j] = 1.0 if sig in head else 0.0
    return out


def build_dense_features(x_u8: np.ndarray) -> np.ndarray:
    dense_parts = []
    H = None

    if USE_BYTE_HIST:
        H = byte_hist(x_u8)
        dense_parts.append(H)

    if USE_ENTROPY:
        if H is None:
            H = byte_hist(x_u8)
        E = shannon_entropy_from_hist(H)
        dense_parts.append(E)

    if USE_STATS:
        S = simple_stats(x_u8)
        dense_parts.append(S)

    if USE_SIGNATURE:
        G = signature_features(x_u8)
        dense_parts.append(G)

    X = np.concatenate(dense_parts, axis=1).astype(np.float32, copy=False)
    # RF는 스케일링 필수는 아니지만, 값 범위 정리(히스토그램+통계 섞였으니) 도움됨
    X = normalize(X, norm="l2").astype(np.float32, copy=False)
    return X


# ==========================
# Top-k helper
# ==========================
def topk_from_proba(proba: np.ndarray, k: int):
    """
    proba: (N, C)
    return:
      topk_idx: (N, k)
      topk_prob: (N, k)
    """
    k = min(k, proba.shape[1])
    idx = np.argpartition(-proba, kth=k-1, axis=1)[:, :k]  # unsorted top-k
    # sort within top-k
    row = np.arange(proba.shape[0])[:, None]
    p = proba[row, idx]
    order = np.argsort(-p, axis=1)
    idx_sorted = idx[row, order]
    p_sorted = p[row, order]
    return idx_sorted, p_sorted


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
# Main
# ==========================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # mapping sanity
    df_map = pd.DataFrame({
        "ext_id": np.arange(75),
        "ext": EXT_BY_ID,
        "cat": [CAT_NAMES[c] for c in EXTID2CATID],
    })
    df_map.to_csv(out_dir / "extid_to_cat.csv", index=False)

    print("[Load]")
    x_tr, y_tr_ext = load_npz(TRAIN_NPZ)
    x_va, y_va_ext = load_npz(VAL_NPZ)
    y_tr_cat = extid_to_catid(y_tr_ext)
    y_va_cat = extid_to_catid(y_va_ext)

    print(" train:", x_tr.shape, y_tr_ext.shape, "cats:", len(np.unique(y_tr_cat)))
    print(" val  :", x_va.shape, y_va_ext.shape, "cats:", len(np.unique(y_va_cat)))
    print("Top cats:\n", pd.Series(y_tr_cat).map(lambda i: CAT_NAMES[i]).value_counts().head(20))

    # features
    print("\n[Features: train (dense for RF)]")
    X_tr = build_dense_features(x_tr)
    print("X_tr:", X_tr.shape, X_tr.dtype)

    print("\n[Features: val (dense for RF)]")
    X_va = build_dense_features(x_va)
    print("X_va:", X_va.shape, X_va.dtype)

    # Stage-1 RandomForest
    print("\n[Stage-1: RandomForest category]")
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS,
        random_state=SEED,
        class_weight="balanced_subsample",
        bootstrap=True,
    )
    clf.fit(X_tr, y_tr_cat)

    pred1 = clf.predict(X_va)
    acc1 = accuracy_score(y_va_cat, pred1)
    print(classification_report(y_va_cat, pred1, target_names=CAT_NAMES, digits=4))
    print("Stage1 acc:", acc1)

    cm = confusion_matrix(y_va_cat, pred1, labels=np.arange(len(CAT_NAMES)))
    save_confusion(cm, CAT_NAMES, out_dir / "stage1_confusion_cat.png", "Stage-1 Confusion (RF Category)")

    # ---- Top-k outputs
    # RF는 predict_proba 지원
    proba = clf.predict_proba(X_va)  # (N, 9)
    topk_idx, topk_prob = topk_from_proba(proba, TOP_K)

    # top-1 / top-k accuracy
    top1 = topk_idx[:, 0]
    top1_acc = (top1 == y_va_cat).mean()

    # top-k hit rate: 정답이 상위 k안에 들어가면 성공
    hit = np.any(topk_idx == y_va_cat[:, None], axis=1)
    topk_hit = hit.mean()

    print(f"\nTop-1 acc: {top1_acc:.4f}")
    print(f"Top-{TOP_K} hit-rate: {topk_hit:.4f}")

    # 저장: 각 샘플의 top-k 예측 + 확률
    out_rows = []
    for i in range(len(y_va_cat)):
        row = {
            "i": i,
            "true_cat": CAT_NAMES[int(y_va_cat[i])],
        }
        for k in range(TOP_K):
            row[f"pred{k+1}_cat"] = CAT_NAMES[int(topk_idx[i, k])]
            row[f"pred{k+1}_prob"] = float(topk_prob[i, k])
        out_rows.append(row)

    df_topk = pd.DataFrame(out_rows)
    df_topk.to_csv(out_dir / "val_topk_predictions.csv", index=False)

    # summary 저장
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Stage1 acc: {acc1:.6f}\n")
        f.write(f"Top-1 acc: {top1_acc:.6f}\n")
        f.write(f"Top-{TOP_K} hit-rate: {topk_hit:.6f}\n")

    print("\nSaved to:", out_dir.resolve())
    print("- extid_to_cat.csv")
    print("- stage1_confusion_cat.png")
    print("- val_topk_predictions.csv (top-k 후보/확률)")
    print("- summary.txt")

if __name__ == "__main__":
    main()
