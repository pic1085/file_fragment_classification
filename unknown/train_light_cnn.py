import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.paths import DATA_DIR, UNKNOWN_RUNS_DIR
from dataclasses import dataclass
from typing import Tuple, List

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix


# ==========================
# Config
# ==========================
@dataclass
class CFG:
    train_npz: str = str(DATA_DIR / "train.npz")
    val_npz: str   = str(DATA_DIR / "val.npz")
    out_dir: str   = str(UNKNOWN_RUNS_DIR / "runs_light_cnn_75")

    seed: int = 42
    num_classes: int = 75

    # training
    epochs: int = 15
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 1e-4
    num_workers: int = 8
    amp: bool = True

    # model
    base_channels: int = 64
    dropout: float = 0.1

    # evaluation / reporting
    prob_bins: Tuple[float, ...] = (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
    top_confusions_k: int = 40
    save_lowconf_samples: bool = True
    lowconf_tau: float = 0.20   # top1 prob < tau 인 샘플을 low-confidence로 저장


# ==========================
# Label names (네 매핑 그대로)
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
# Reproducibility
# ==========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================
# Dataset
# ==========================
class NPZFragmentDataset(Dataset):
    """
    x: (N,4096) uint8
    y: (N,) int64
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.x = data["x"].astype(np.uint8, copy=False)
        self.y = data["y"].astype(np.int64, copy=False)
        assert self.x.ndim == 2 and self.x.shape[1] == 4096
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        # normalize to [0,1]
        x = self.x[idx].astype(np.float32) / 255.0  # (4096,)
        y = int(self.y[idx])
        # return (C,L) = (1,4096)
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(y, dtype=torch.long)


# ==========================
# Lightweight 1D CNN
# - Depthwise separable conv blocks + global average pooling
# ==========================
class DWConv1d(nn.Module):
    def __init__(self, ch: int, k: int = 7, s: int = 1):
        super().__init__()
        pad = (k // 2)
        self.dw = nn.Conv1d(ch, ch, kernel_size=k, stride=s, padding=pad, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return F.silu(x)

class LightCNN1D(nn.Module):
    def __init__(self, num_classes: int = 75, base: int = 64, dropout: float = 0.1):
        super().__init__()
        # stem: (1,4096) -> (base,1024) 정도로 줄이기
        self.stem = nn.Sequential(
            nn.Conv1d(1, base, kernel_size=9, stride=4, padding=4, bias=False),
            nn.BatchNorm1d(base),
            nn.SiLU(),
        )
        # stages: downsample + DW blocks
        self.stage1 = nn.Sequential(
            nn.Conv1d(base, base, kernel_size=5, stride=2, padding=2, bias=False),  # /2
            nn.BatchNorm1d(base),
            nn.SiLU(),
            DWConv1d(base, k=7),
            DWConv1d(base, k=7),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(base, base*2, kernel_size=5, stride=2, padding=2, bias=False),  # /2
            nn.BatchNorm1d(base*2),
            nn.SiLU(),
            DWConv1d(base*2, k=7),
            DWConv1d(base*2, k=7),
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(base*2, base*4, kernel_size=5, stride=2, padding=2, bias=False),  # /2
            nn.BatchNorm1d(base*4),
            nn.SiLU(),
            DWConv1d(base*4, k=7),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base*4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        # global average pooling
        x = x.mean(dim=-1)
        x = self.dropout(x)
        return self.fc(x)


# ==========================
# Train / Eval
# ==========================
def train_one_epoch(model, loader, optim, scaler, device):
    model.train()
    total_loss = 0.0
    total = 0

    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        bs = yb.size(0)
        total_loss += loss.item() * bs
        total += bs

    return total_loss / max(total, 1)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_y = []
    all_pred = []
    all_prob1 = []
    all_proba = []  # optionally store for analysis (can be big; but 115k x 75 is okay)

    for xb, yb in tqdm(loader, desc="val", leave=False):
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        proba = torch.softmax(logits, dim=1)

        pred = torch.argmax(proba, dim=1)
        prob1 = torch.max(proba, dim=1).values

        all_y.append(yb.numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob1.append(prob1.cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    p1 = np.concatenate(all_prob1)
    proba_full = np.concatenate(all_proba)
    acc = (y_true == y_pred).mean()
    return acc, y_true, y_pred, p1, proba_full


# ==========================
# Reporting / Plots
# ==========================
def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path, title: str):
    plt.figure(figsize=(18, 18))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=90, fontsize=7)
    plt.yticks(tick, labels, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

def save_bar(values: np.ndarray, names: List[str], out_path: Path, title: str, ylabel: str, sort_desc: bool = True):
    order = np.argsort(-values) if sort_desc else np.argsort(values)
    v = values[order]
    n = [names[i] for i in order]

    plt.figure(figsize=(22, 6))
    plt.bar(np.arange(len(v)), v)
    plt.xticks(np.arange(len(v)), n, rotation=90, fontsize=8)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

def save_prob_hist(p1: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(10, 5))
    plt.hist(p1, bins=30)
    plt.title(title)
    plt.xlabel("pred1_prob")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

def prob_bins_table(p1: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, bins: Tuple[float, ...]):
    correct = (y_true == y_pred)
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p1 >= a) & (p1 < b)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"[{a:.1f},{b:.1f})",
            "count": int(m.sum()),
            "acc": float(correct[m].mean()),
        })
    return pd.DataFrame(rows)

def top_confusions(cm: np.ndarray, labels: List[str], k: int = 40):
    # off-diagonal largest counts
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                pairs.append((c, labels[i], labels[j]))
    pairs.sort(reverse=True, key=lambda x: x[0])
    top = pairs[:k]
    return pd.DataFrame(top, columns=["count","true","pred"])

def plot_top_confusions_subset(cm: np.ndarray, labels: List[str], conf_df: pd.DataFrame, out_path: Path, title: str):
    # build subset labels from top confusions true/pred
    used = []
    for _, r in conf_df.iterrows():
        used.append(r["true"])
        used.append(r["pred"])
    # unique preserve order
    seen = set()
    subset = []
    for x in used:
        if x not in seen:
            seen.add(x)
            subset.append(x)
        if len(subset) >= 40:
            break

    idx = [labels.index(x) for x in subset]
    sub_cm = cm[np.ix_(idx, idx)]

    plt.figure(figsize=(14, 14))
    plt.imshow(sub_cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(subset))
    plt.xticks(tick, subset, rotation=90, fontsize=8)
    plt.yticks(tick, subset, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


# ==========================
# Main
# ==========================
def main():
    cfg = CFG()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # data
    tr_ds = NPZFragmentDataset(cfg.train_npz)
    va_ds = NPZFragmentDataset(cfg.val_npz)

    tr_loader = DataLoader(
        tr_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )
    va_loader = DataLoader(
        va_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )

    # model
    model = LightCNN1D(cfg.num_classes, base=cfg.base_channels, dropout=cfg.dropout).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_acc = -1.0
    best_path = out_dir / "best.pt"

    for ep in range(1, cfg.epochs + 1):
        print(f"\nEpoch {ep}/{cfg.epochs}")
        tr_loss = train_one_epoch(model, tr_loader, optim, scaler if scaler.is_enabled() else None, device)
        val_acc, _, _, _, _ = eval_model(model, va_loader, device)
        print(f"train_loss={tr_loss:.6f} | val_acc={val_acc:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)
            print("saved best:", best_path)

    # load best and final eval
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    val_acc, y_true, y_pred, p1, proba_full = eval_model(model, va_loader, device)
    print("\nFinal val_acc:", val_acc)

    # 1) classification report -> csv
    report = classification_report(
        y_true, y_pred,
        labels=np.arange(cfg.num_classes),
        target_names=EXT_BY_ID,
        output_dict=True,
        digits=6,
        zero_division=0
    )
    # per-class rows only
    rows = []
    for i, name in enumerate(EXT_BY_ID):
        r = report.get(name, None)
        if r is None:
            continue
        rows.append({
            "class_id": i,
            "class": name,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1-score": r["f1-score"],
            "support": r["support"],
        })
    df_report = pd.DataFrame(rows)
    df_report.to_csv(out_dir / "val_class_report.csv", index=False)

    # 2) confusion matrix (75)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(cfg.num_classes))
    save_confusion_matrix(cm, EXT_BY_ID, out_dir / "confusion_matrix_75.png", "Confusion Matrix (75 classes)")

    # 3) f1 bar (sorted)
    f1 = df_report["f1-score"].values
    support = df_report["support"].values
    save_bar(f1, EXT_BY_ID, out_dir / "f1_by_class_sorted.png", "F1-score by class (sorted)", "f1-score", sort_desc=True)
    save_bar(support, EXT_BY_ID, out_dir / "support_by_class_sorted.png", "Support by class (sorted)", "support", sort_desc=True)

    # 4) top1 prob histogram + bins table
    save_prob_hist(p1, out_dir / "pred1_prob_hist.png", "Top-1 predicted probability histogram")
    df_bins = prob_bins_table(p1, y_true, y_pred, cfg.prob_bins)
    df_bins.to_csv(out_dir / "val_pred1_prob_bins.csv", index=False)

    # 5) top confusions
    df_conf = top_confusions(cm, EXT_BY_ID, k=cfg.top_confusions_k)
    df_conf.to_csv(out_dir / "val_top_confusions.csv", index=False)
    if len(df_conf) > 0:
        plot_top_confusions_subset(
            cm, EXT_BY_ID, df_conf,
            out_dir / "top_confusions_subset_top40.png",
            "Top confusions (subset, top40)"
        )

    # 6) (optional) low-confidence samples list
    if cfg.save_lowconf_samples:
        low_idx = np.where(p1 < cfg.lowconf_tau)[0]
        df_low = pd.DataFrame({
            "i": low_idx,
            "true": [EXT_BY_ID[int(y_true[i])] for i in low_idx],
            "pred": [EXT_BY_ID[int(y_pred[i])] for i in low_idx],
            "pred1_prob": p1[low_idx],
        })
        df_low.to_csv(out_dir / "val_lowconf_samples.csv", index=False)

    # summary text
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"val_acc={val_acc}\n")
        f.write(f"saved: val_class_report.csv, val_pred1_prob_bins.csv, val_top_confusions.csv\n")
        f.write(f"plots: confusion_matrix_75.png, top_confusions_subset_top40.png, f1_by_class_sorted.png, support_by_class_sorted.png, pred1_prob_hist.png\n")

    print("\nSaved to:", out_dir.resolve())
    print("- val_class_report.csv")
    print("- val_pred1_prob_bins.csv")
    print("- val_top_confusions.csv")
    print("- (optional) val_lowconf_samples.csv")
    print("- plots: confusion_matrix_75.png / top_confusions_subset_top40.png / f1_by_class_sorted.png / support_by_class_sorted.png / pred1_prob_hist.png")


if __name__ == "__main__":
    main()
