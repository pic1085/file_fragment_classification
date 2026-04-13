import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.paths import DATA_DIR, UNKNOWN_RUNS_DIR
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================
# Config
# ==========================
TRAIN_NPZ = str(DATA_DIR / "train.npz")
VAL_NPZ   = str(DATA_DIR / "val.npz")
OUT_DIR   = UNKNOWN_RUNS_DIR / "runs_lite_cnn"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 75
SEQ_LEN = 4096

# 학습 속도/메모리 맞춰 조절
BATCH_SIZE = 512
EPOCHS = 6
LR = 2e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# (선택) train 서브샘플로 빠르게 시작
TRAIN_SUBSAMPLE = 300_000   # None이면 전체 사용

# ==========================
# Your label list (75)
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
    d = np.load(npz_path, allow_pickle=True)
    x = d["x"].astype(np.uint8, copy=False)   # (N,4096)
    y = d["y"].astype(np.int64, copy=False)   # (N,)
    return x, y

# ==========================
# Dataset
# ==========================
class BytesDataset(Dataset):
    def __init__(self, x_u8: np.ndarray, y: np.ndarray, indices=None):
        self.x = x_u8
        self.y = y
        if indices is None:
            self.idx = np.arange(len(x_u8), dtype=np.int64)
        else:
            self.idx = indices.astype(np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        # uint8 -> int64 (embedding index)
        x = torch.from_numpy(self.x[j].astype(np.int64, copy=False))
        y = int(self.y[j])
        return x, y, j

# ==========================
# Model (Lightweight 1D-CNN)
# - Embedding(256->16)
# - Depthwise Separable Conv blocks
# ==========================
class DSConv1d(nn.Module):
    """Depthwise separable 1D conv: depthwise + pointwise"""
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return F.silu(x)

class LiteCNNBytes(nn.Module):
    def __init__(self, num_classes=75, emb_dim=16, base_ch=48, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(256, emb_dim)

        # input: (B, emb_dim, 4096)
        self.block1 = DSConv1d(emb_dim, base_ch,   k=9, s=2, p=4)   # 4096 -> 2048
        self.block2 = DSConv1d(base_ch, base_ch,   k=7, s=2, p=3)   # 2048 -> 1024
        self.block3 = DSConv1d(base_ch, base_ch*2, k=5, s=2, p=2)   # 1024 -> 512
        self.block4 = DSConv1d(base_ch*2, base_ch*2, k=5, s=2, p=2) # 512 -> 256

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(base_ch*2, num_classes)

    def forward(self, x):  # x: (B,4096) int64
        x = self.emb(x)          # (B,4096,emb)
        x = x.transpose(1, 2)    # (B,emb,4096)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B,C)
        x = self.drop(x)
        return self.fc(x)

# ==========================
# Metrics
# ==========================
@torch.no_grad()
def eval_topk(model, loader, k=3):
    model.eval()
    total = 0
    hit1 = 0
    hitk = 0
    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)

        topk = torch.topk(prob, k=k, dim=1).indices
        pred1 = topk[:, 0]
        hit1 += (pred1 == y).sum().item()
        hitk += (topk == y.view(-1,1)).any(dim=1).sum().item()
        total += y.size(0)
    return hit1/total, hitk/total

@torch.no_grad()
def predict_to_csv(model, loader, out_csv: Path, topk=3):
    model.eval()
    rows = []
    for x, y, idx in tqdm(loader, desc="predict"):
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        vals, inds = torch.topk(prob, k=topk, dim=1)

        inds = inds.cpu().numpy()
        vals = vals.cpu().numpy()
        y = y.numpy()
        idx = idx.numpy()

        for i in range(len(idx)):
            r = {
                "i": int(idx[i]),
                "true_ext_id": int(y[i]),
                "true_ext": EXT_BY_ID[int(y[i])]
            }
            for t in range(topk):
                r[f"pred{t+1}_id"] = int(inds[i, t])
                r[f"pred{t+1}"] = EXT_BY_ID[int(inds[i, t])]
                r[f"pred{t+1}_prob"] = float(vals[i, t])
            rows.append(r)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
def save_confusion_matrix_png(cm, labels, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=90, fontsize=6)
    plt.yticks(tick, labels, fontsize=6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

def save_bar_sorted(values, labels, out_png: Path, title: str, xlabel: str, ylabel: str, top_n: int = None):
    import matplotlib.pyplot as plt
    order = np.argsort(-values)
    if top_n is not None:
        order = order[:top_n]
    v = values[order]
    l = [labels[i] for i in order]

    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(v)), v)
    plt.xticks(np.arange(len(v)), l, rotation=90, fontsize=7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

def save_top_confusions(cm, labels, out_csv: Path, out_png: Path, top_n: int = 30):
    """
    off-diagonal 중 많이 헷갈린 pair (true->pred) 상위 top_n 저장 + 간단 heatmap
    """
    import matplotlib.pyplot as plt

    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            if v > 0:
                pairs.append((v, i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    pairs = pairs[:top_n]

    rows = []
    for v, i, j in pairs:
        rows.append({
            "count": v,
            "true": labels[i],
            "pred": labels[j]
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # heatmap (top_n 항목만)
    if len(pairs) > 0:
        # top_n에 등장한 label만 뽑아 작은 CM 만들기
        used = sorted(set([i for _, i, _ in pairs] + [j for _, _, j in pairs]))
        sub = cm[np.ix_(used, used)]

        plt.figure(figsize=(10, 9))
        plt.imshow(sub, interpolation="nearest")
        plt.title(f"Top confusions (subset, top{top_n})")
        plt.colorbar()
        tick = np.arange(len(used))
        plt.xticks(tick, [labels[i] for i in used], rotation=90, fontsize=7)
        plt.yticks(tick, [labels[i] for i in used], fontsize=7)
        plt.tight_layout()
        plt.savefig(out_png, dpi=250)
        plt.close()

def save_confidence_analysis(y_true, prob, labels, out_dir: Path):
    """
    top-1 confidence 분포 + bin별 accuracy 저장
    """
    import matplotlib.pyplot as plt

    pred = prob.argmax(axis=1)
    maxp = prob.max(axis=1)
    correct = (pred == y_true)

    # histogram
    plt.figure(figsize=(10, 4))
    plt.hist(maxp, bins=30)
    plt.title("Top-1 confidence histogram")
    plt.xlabel("max probability")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "val_confidence_hist.png", dpi=250)
    plt.close()

    # bins accuracy
    bins = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (maxp >= a) & (maxp < b)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"[{a:.1f},{b:.1f})",
            "count": int(m.sum()),
            "acc": float(correct[m].mean()),
        })
    pd.DataFrame(rows).to_csv(out_dir / "val_confidence_bins.csv", index=False)

def evaluate_and_visualize_from_predictions(pred_csv: Path, out_dir: Path, labels: list[str]):
    """
    predict_to_csv로 만든 val_predictions_top3.csv를 읽어서
    - report(csv)
    - f1/support bar
    - confusion matrix
    - top confusions
    - confidence 분석
    """
    df = pd.read_csv(pred_csv)

    y_true = df["true_ext_id"].to_numpy(dtype=np.int64)
    y_pred = df["pred1_id"].to_numpy(dtype=np.int64)
    top1_prob = df["pred1_prob"].to_numpy(dtype=np.float32)

    # (선택) prob 전체가 아니라 top1_prob만 있으니,
    # confidence 분석은 top1_prob 히스토그램/빈 정확도 중심으로 간단히 처리
    # (full softmax prob 저장까지 하고 싶으면 predict 단계에서 prob 전체 저장 가능)
    # 여기서는 top1_prob 기준 confidence 분석:
    import matplotlib.pyplot as plt
    correct = (y_pred == y_true)

    plt.figure(figsize=(10, 4))
    plt.hist(top1_prob, bins=30)
    plt.title("Top-1 predicted probability histogram")
    plt.xlabel("pred1_prob")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "val_pred1_prob_hist.png", dpi=250)
    plt.close()

    bins = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (top1_prob >= a) & (top1_prob < b)
        if m.sum() == 0:
            continue
        rows.append({"bin": f"[{a:.1f},{b:.1f})", "count": int(m.sum()), "acc": float(correct[m].mean())})
    pd.DataFrame(rows).to_csv(out_dir / "val_pred1_prob_bins.csv", index=False)

    # accuracy
    acc = accuracy_score(y_true, y_pred)
    (out_dir / "val_summary.txt").write_text(f"top1_acc={acc}\n", encoding="utf-8")

    # classification report
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T
    rep_df.to_csv(out_dir / "val_class_report.csv")

    # per-class f1/support
    # (classification_report의 각 클래스 key는 "0","1",... string)
    f1 = np.zeros(len(labels), dtype=np.float32)
    sup = np.zeros(len(labels), dtype=np.float32)
    for i in range(len(labels)):
        key = str(i)
        if key in rep:
            f1[i] = float(rep[key].get("f1-score", 0.0))
            sup[i] = float(rep[key].get("support", 0.0))

    save_bar_sorted(f1, labels, out_dir / "val_f1_by_class.png",
                    "F1-score by class (sorted)", "class", "f1-score", top_n=75)
    save_bar_sorted(sup, labels, out_dir / "val_support_by_class.png",
                    "Support by class (sorted)", "class", "support", top_n=75)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    np.save(out_dir / "val_confusion_matrix.npy", cm)
    save_confusion_matrix_png(cm, labels, out_dir / "val_confusion_matrix.png", "Confusion Matrix (75 classes)")

    # top confusions
    save_top_confusions(cm, labels, out_dir / "val_top_confusions.csv", out_dir / "val_top_confusions.png", top_n=40)

# ==========================
# Train
# ==========================
def main():
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    print("DEVICE:", DEVICE)

    print("[Load]")
    x_tr, y_tr = load_npz(TRAIN_NPZ)
    x_va, y_va = load_npz(VAL_NPZ)
    print(" train:", x_tr.shape, y_tr.shape)
    print(" val  :", x_va.shape, y_va.shape)

    # train subsample
    if TRAIN_SUBSAMPLE is not None and TRAIN_SUBSAMPLE < len(x_tr):
        tr_idx = np.random.choice(len(x_tr), size=TRAIN_SUBSAMPLE, replace=False)
    else:
        tr_idx = None

    ds_tr = BytesDataset(x_tr, y_tr, tr_idx)
    ds_va = BytesDataset(x_va, y_va, None)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)

    model = LiteCNNBytes(num_classes=NUM_CLASSES, emb_dim=16, base_ch=48, dropout=0.1).to(DEVICE)

    # class weight (imbalanced 대응)
    counts = np.bincount(y_tr[tr_idx] if tr_idx is not None else y_tr, minlength=NUM_CLASSES).astype(np.float32)
    w = (counts.sum() / (counts + 1.0))
    w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE, dtype=torch.float32))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best = 0.0
    ckpt = out / "best.pt"

    for ep in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"train ep{ep}")
        for x, y, _ in pbar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                logits = model(x)
                loss = crit(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=float(loss.item()))

        va1, va3 = eval_topk(model, dl_va, k=3)
        print(f"[val] ep{ep} top1={va1:.4f} top3={va3:.4f}")

        if va1 > best:
            best = va1
            torch.save({"model": model.state_dict()}, ckpt)
            print("  saved ->", ckpt)

    # load best + export predictions
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state["model"])

    va1, va3 = eval_topk(model, dl_va, k=3)
    print(f"\n[Best] top1={va1:.4f} top3={va3:.4f}")

    pred_csv = out / "val_predictions_top3.csv"
    predict_to_csv(model, dl_va, pred_csv, topk=3)
    print("saved ->", pred_csv)
        # --- extra reports & plots ---
    evaluate_and_visualize_from_predictions(pred_csv, out, EXT_BY_ID)
    print("saved -> val_class_report.csv / val_f1_by_class.png / val_support_by_class.png / val_confusion_matrix.png / val_top_confusions.*")
if __name__ == "__main__":
    main()
