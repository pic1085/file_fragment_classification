"""
2-Stage Hierarchical Classifier for FFT-75
Stage 1: Category (11-way)
Stage 2: Extension within predicted category (category-conditional heads)

- Shared ultra-light backbone
- During training: Stage2 uses TRUE category for stable learning
- During eval/test: Pipeline inference uses PREDICTED category (realistic)
- Logs: metrics.csv (small columns)
- Saves: per_extension_acc_test.csv, per_category_acc_test.csv, confusion_test.npy (optional)
"""

import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.paths import DATA_DIR, MODEL_RUNS_DIR
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==========================
# 0) Config
# ==========================

TRAIN_PATH = str(DATA_DIR / "train.npz")
VAL_PATH   = str(DATA_DIR / "val.npz")
TEST_PATH  = str(DATA_DIR / "test.npz")

BATCH_SIZE = 256
NUM_EPOCHS = 50
LR = 2e-3
WEIGHT_DECAY = 1e-4

# Loss weights
LAMBDA_CAT = 0.5
LAMBDA_EXT = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAGMENT_SIZE = 4096

INDEX2NAME = [
    "ARW:Raw","CR2:Raw","DNG:Raw","GPR:Raw","NEF:Raw","NRW:Raw","ORF:Raw","PEF:Raw","RAF:Raw","RW2:Raw",
    "3FR:Raw","JPG:Bitmap","TIFF:Bitmap","HEIC:Bitmap","BMP:Bitmap","GIF:Bitmap","PNG:Bitmap",
    "AI:Vector","EPS:Vector","PSD:Vector",
    "MOV:Video","MP4:Video","3GP:Video","AVI:Video","MKV:Video","OGV:Video","WEBM:Video",
    "APK:Archive","JAR:Archive","MSI:Archive","DMG:Archive","7Z:Archive","BZ2:Archive","DEB:Archive","GZ:Archive",
    "PKG:Archive","RAR:Archive","RPM:Archive","XZ:Archive","ZIP:Archive",
    "EXE:Executables","MACH-O:Executables","ELF:Executables","DLL:Executables",
    "DOC:Office","DOCX:Office","KEY:Office","PPT:Office","PPTX:Office","XLS:Office","XLSX:Office",
    "DJVU:Published","EPUB:Published","MOBI:Published","PDF:Published",
    "MD:Human-readable","RTF:Human-readable","TXT:Human-readable","TEX:Human-readable","JSON:Human-readable",
    "HTML:Human-readable","XML:Human-readable","LOG:Human-readable","CSV:Human-readable",
    "AIFF:Audio","FLAC:Audio","M4A:Audio","MP3:Audio","OGG:Audio","WAV:Audio","WMA:Audio",
    "PCAP:Other","TTF:Other","DWG:Other","SQLITE:Other"
]
assert len(INDEX2NAME) == 75
NUM_EXT_CLASSES = 75

CATEGORIES = [
    "Raw", "Bitmap", "Vector", "Video", "Archive",
    "Executables", "Office", "Published", "Human-readable", "Other", "Audio"
]
CAT2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2CAT = {i: c for c, i in CAT2ID.items()}
NUM_CAT = len(CATEGORIES)

LABEL2CAT = np.array([CAT2ID[s.split(":")[1]] for s in INDEX2NAME], dtype=np.int64)

# category -> list of global extension labels
CAT2LABELS: Dict[int, List[int]] = {i: [] for i in range(NUM_CAT)}
for ext_id in range(NUM_EXT_CLASSES):
    CAT2LABELS[int(LABEL2CAT[ext_id])].append(ext_id)

# global extension label -> within-category index (0..len(cat)-1)
EXT2WITHIN = np.full((NUM_EXT_CLASSES,), -1, dtype=np.int64)
for cat_id, labels in CAT2LABELS.items():
    for j, ext_id in enumerate(labels):
        EXT2WITHIN[ext_id] = j
assert (EXT2WITHIN >= 0).all()

# within-category -> global extension label (for each category)
WITHIN2EXT: Dict[int, np.ndarray] = {}
for cat_id, labels in CAT2LABELS.items():
    WITHIN2EXT[cat_id] = np.array(labels, dtype=np.int64)

# ==========================
# 1) Dataset
# ==========================

class FFTDataset2Stage(Dataset):
    """
    x: uint8 (4096,)
    y_ext: 0..74 (global extension label)
    y_cat: 0..10
    y_within: within-category label
    """
    def __init__(self, npz_path: str, augment=False):
        data = np.load(npz_path, allow_pickle=True)
        self.x = data["x"].astype(np.uint8, copy=False)
        self.y_ext = data["y"].astype(np.int64, copy=False)
        self.augment = augment

        assert self.x.ndim == 2 and self.x.shape[1] == FRAGMENT_SIZE
        assert self.x.shape[0] == self.y_ext.shape[0]
        uniq = np.unique(self.y_ext)
        print(f"[{npz_path}] x={self.x.shape} | uniq labels={len(uniq)}")

    def __len__(self):
        return len(self.x)

    def _augment_bytes(self, arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        # Byte dropout
        if np.random.rand() < 0.5:
            k = np.random.randint(1, 1 + FRAGMENT_SIZE // 200)  # ~0.5% 이하
            idx = np.random.randint(0, FRAGMENT_SIZE, size=k)
            out[idx] = 0
        # Bit flip
        if np.random.rand() < 0.5:
            k = np.random.randint(1, 1 + FRAGMENT_SIZE // 300)
            idx = np.random.randint(0, FRAGMENT_SIZE, size=k)
            bit = (1 << np.random.randint(0, 8))
            out[idx] = np.bitwise_xor(out[idx], bit)
        # 아주 작은 shift(선택): header 의존 줄이려면 켜도 됨 (너무 크게는 비추)
        if np.random.rand() < 0.2:
            shift = np.random.randint(-4, 5)
            out = np.roll(out, shift)
        return out

    def __getitem__(self, idx: int):
        frag = self.x[idx]
        if self.augment:
            frag = self._augment_bytes(frag)

        y_ext = int(self.y_ext[idx])
        y_cat = int(LABEL2CAT[y_ext])
        y_within = int(EXT2WITHIN[y_ext])

        x = torch.from_numpy(frag).long()  # for embedding
        return x, torch.tensor(y_ext), torch.tensor(y_cat), torch.tensor(y_within)

# ==========================
# 2) Blocks / Backbone
# ==========================

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, expand_ratio=2, use_residual=True):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = use_residual and (in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv1d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size//2, groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU6(inplace=True),
        ]
        layers += [
            nn.Conv1d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class LightweightSEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class UltraLightBackbone(nn.Module):
    def __init__(self, width_mult=0.5):
        super().__init__()
        emb_dim = int(16 * width_mult)
        self.emb = nn.Embedding(256, emb_dim)

        c1 = int(32 * width_mult)
        c2 = int(48 * width_mult)
        c3 = int(64 * width_mult)

        self.stem = nn.Sequential(
            nn.Conv1d(emb_dim, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU6(inplace=True),
        )
        self.stage1 = nn.Sequential(
            InvertedResidual(c1, c1, kernel_size=3, expand_ratio=2),
            InvertedResidual(c1, c2, kernel_size=5, expand_ratio=2, use_residual=False),
            nn.MaxPool1d(2),
        )
        self.stage2 = nn.Sequential(
            InvertedResidual(c2, c2, kernel_size=3, expand_ratio=2),
            LightweightSEBlock(c2, reduction=4),
            InvertedResidual(c2, c3, kernel_size=5, expand_ratio=2, use_residual=False),
            nn.MaxPool1d(2),
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(c3, c3, kernel_size=3, expand_ratio=2),
            LightweightSEBlock(c3, reduction=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feat_dim = c3

    def forward(self, x):
        x = self.emb(x)          # (B,4096,emb)
        x = x.permute(0, 2, 1)   # (B,emb,4096)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).squeeze(-1)  # (B, D)
        return x

# ==========================
# 3) 2-Stage Model
# ==========================

class TwoStageHierModel(nn.Module):
    """
    Stage1: category head (11-way)
    Stage2: per-category extension heads
    """
    def __init__(self, width_mult=0.5, dropout=0.2):
        super().__init__()
        self.backbone = UltraLightBackbone(width_mult=width_mult)
        D = self.backbone.feat_dim
        self.dropout = nn.Dropout(dropout)

        # Stage 1: category head
        self.cat_head = nn.Linear(D, NUM_CAT)

        # Stage 2: category-conditional heads
        self.ext_heads = nn.ModuleList()
        for cat_id in range(NUM_CAT):
            n = len(CAT2LABELS[cat_id])
            self.ext_heads.append(nn.Linear(D, n))

    def forward_features(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return feat

    def forward_stage1(self, feat):
        return self.cat_head(feat)

    def forward_stage2_truecat(self, feat, y_cat):
        """
        Training-time: use TRUE category to pick the right head.
        Returns:
          ext_logits: (B, variable by sample) -> we pack as list segments then concat
          also returns list of indices to reconstruct if needed
        """
        B = feat.size(0)
        device = feat.device
        ext_logits_out = torch.empty((B, 1), device=device)  # placeholder to be overwritten per sample segment

        # We'll build (B, max_n) and use per-sample CE with different head sizes is messy.
        # Better: compute loss per-cat groups.
        # So here just return dict cat_id -> logits for samples in that cat.
        per_cat = {}
        for cat_id in range(NUM_CAT):
            mask = (y_cat == cat_id)
            if mask.any():
                logits = self.ext_heads[cat_id](feat[mask])  # (n_samples, n_ext_in_cat)
                per_cat[cat_id] = (mask, logits)
        return per_cat

    @torch.no_grad()
    def predict_pipeline(self, x):
        """
        Inference-time pipeline:
          1) predict category
          2) choose that category head
          3) predict extension within that category
          4) map back to global extension id (0..74)
        """
        feat = self.forward_features(x)
        cat_logits = self.forward_stage1(feat)
        cat_pred = cat_logits.argmax(dim=1)  # (B,)

        ext_pred_global = torch.empty_like(cat_pred)
        for cat_id in range(NUM_CAT):
            mask = (cat_pred == cat_id)
            if mask.any():
                logits = self.ext_heads[cat_id](feat[mask])
                within = logits.argmax(dim=1)  # (n,)
                global_ids = torch.from_numpy(WITHIN2EXT[cat_id]).to(x.device)  # (n_ext_in_cat,)
                ext_pred_global[mask] = global_ids[within]
        return cat_pred, ext_pred_global

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================
# 4) Train / Eval
# ==========================

def train_one_epoch(model: TwoStageHierModel, loader, opt, epoch,
                    ce_cat, ce_ext, scheduler=None):
    model.train()
    total_loss = 0.0
    total_cat_correct = 0
    total_ext_correct_truecat = 0
    total_n = 0

    pbar = tqdm(loader, desc=f"Train {epoch:02d}")
    for x, y_ext, y_cat, y_within in pbar:
        x = x.to(DEVICE, non_blocking=True)
        y_ext = y_ext.to(DEVICE, non_blocking=True)
        y_cat = y_cat.to(DEVICE, non_blocking=True)
        y_within = y_within.to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        feat = model.forward_features(x)
        cat_logits = model.forward_stage1(feat)
        cat_loss = ce_cat(cat_logits, y_cat)

        # Stage2 loss: group-by-true-cat
        per_cat = model.forward_stage2_truecat(feat, y_cat)
        ext_loss = torch.tensor(0.0, device=DEVICE)
        ext_correct = 0
        for cat_id, (mask, logits) in per_cat.items():
            yy = y_within[mask]  # within-cat labels
            ext_loss = ext_loss + ce_ext(logits, yy)
            ext_correct += (logits.argmax(dim=1) == yy).sum().item()

        loss = LAMBDA_CAT * cat_loss + LAMBDA_EXT * ext_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if scheduler is not None:
            scheduler.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_cat_correct += (cat_logits.argmax(dim=1) == y_cat).sum().item()
        total_ext_correct_truecat += ext_correct
        total_n += bs

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            cat_acc=f"{(total_cat_correct/total_n)*100:.1f}%",
            ext_acc=f"{(total_ext_correct_truecat/total_n)*100:.1f}%"
        )

    return total_loss / total_n, total_cat_correct / total_n, total_ext_correct_truecat / total_n


@torch.no_grad()
def eval_epoch(model: TwoStageHierModel, loader, epoch, phase,
               ce_cat, ce_ext, save_dir=None):
    model.eval()
    total_loss = 0.0
    total_n = 0

    cat_correct = 0
    ext_correct_truecat = 0
    ext_correct_pipeline = 0

    # per-extension accuracy (pipeline)
    per_total_ext = np.zeros(NUM_EXT_CLASSES, dtype=np.int64)
    per_correct_ext = np.zeros(NUM_EXT_CLASSES, dtype=np.int64)

    # per-category accuracy (pipeline ext)
    per_total_cat = np.zeros(NUM_CAT, dtype=np.int64)
    per_correct_cat = np.zeros(NUM_CAT, dtype=np.int64)

    for x, y_ext, y_cat, y_within in tqdm(loader, desc=f"{phase} {epoch:02d}", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y_ext = y_ext.to(DEVICE, non_blocking=True)
        y_cat = y_cat.to(DEVICE, non_blocking=True)
        y_within = y_within.to(DEVICE, non_blocking=True)

        feat = model.forward_features(x)
        cat_logits = model.forward_stage1(feat)
        cat_loss = ce_cat(cat_logits, y_cat)

        per_cat = model.forward_stage2_truecat(feat, y_cat)
        ext_loss = torch.tensor(0.0, device=DEVICE)
        ext_correct = 0
        for cat_id, (mask, logits) in per_cat.items():
            yy = y_within[mask]
            ext_loss = ext_loss + ce_ext(logits, yy)
            ext_correct += (logits.argmax(dim=1) == yy).sum().item()

        loss = LAMBDA_CAT * cat_loss + LAMBDA_EXT * ext_loss

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        cat_pred = cat_logits.argmax(dim=1)
        cat_correct += (cat_pred == y_cat).sum().item()
        ext_correct_truecat += ext_correct

        # pipeline prediction
        _, ext_pred_global = model.predict_pipeline(x)
        ext_correct_pipeline += (ext_pred_global == y_ext).sum().item()

        # per-extension stats (pipeline)
        y_ext_np = y_ext.cpu().numpy()
        ext_pred_np = ext_pred_global.cpu().numpy()
        np.add.at(per_total_ext, y_ext_np, 1)
        np.add.at(per_correct_ext, y_ext_np, (ext_pred_np == y_ext_np).astype(np.int64))

        # per-category stats (pipeline ext)
        y_cat_np = y_cat.cpu().numpy()
        ok = (ext_pred_np == y_ext_np).astype(np.int64)
        np.add.at(per_total_cat, y_cat_np, 1)
        np.add.at(per_correct_cat, y_cat_np, ok)

    avg_loss = total_loss / total_n
    cat_acc = cat_correct / total_n
    ext_acc_truecat = ext_correct_truecat / total_n
    ext_acc_pipeline = ext_correct_pipeline / total_n

    # macro per-extension acc (pipeline)
    per_ext_acc = per_correct_ext / np.maximum(per_total_ext, 1)
    ext_macc = float(per_ext_acc.mean())

    print(
        f"[{phase} {epoch:02d}] "
        f"Loss={avg_loss:.4f} | "
        f"CatAcc={cat_acc*100:.2f}% | "
        f"ExtAcc(trueCat)={ext_acc_truecat*100:.2f}% | "
        f"ExtAcc(pipeline)={ext_acc_pipeline*100:.2f}% | "
        f"ExtmAcc={ext_macc*100:.2f}%"
    )

    # optional plots
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        # category bar (pipeline ext correctness by true category)
        cat_bar = per_correct_cat / np.maximum(per_total_cat, 1) * 100
        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(NUM_CAT), cat_bar)
        plt.xticks(np.arange(NUM_CAT), [ID2CAT[i] for i in range(NUM_CAT)], rotation=45, ha="right")
        plt.ylabel("Extension accuracy (%)")
        plt.title(f"Per-category extension accuracy ({phase})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"per_category_extacc_{phase.lower()}.png"), dpi=200)
        plt.close()

    return {
        "loss": avg_loss,
        "cat_acc": cat_acc,
        "ext_acc_truecat": ext_acc_truecat,
        "ext_acc_pipeline": ext_acc_pipeline,
        "ext_macc": ext_macc,
        "per_ext_acc": per_ext_acc,
        "per_cat_extacc": (per_correct_cat / np.maximum(per_total_cat, 1)),
    }

def save_per_extension_csv(per_ext_acc: np.ndarray, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ext_id", "ext_name", "category", "acc"])
        for i in range(NUM_EXT_CLASSES):
            ext = INDEX2NAME[i].split(":")[0]
            cat = INDEX2NAME[i].split(":")[1]
            w.writerow([i, ext, cat, f"{per_ext_acc[i]:.6f}"])

def save_per_category_csv(per_cat_acc: np.ndarray, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cat_id", "category", "ext_acc"])
        for i in range(NUM_CAT):
            w.writerow([i, ID2CAT[i], f"{per_cat_acc[i]:.6f}"])

# ==========================
# 5) Main
# ==========================

def main():
    print("Device:", DEVICE)

    train_set = FFTDataset2Stage(TRAIN_PATH, augment=True)
    val_set   = FFTDataset2Stage(VAL_PATH, augment=False)
    test_set  = FFTDataset2Stage(TEST_PATH, augment=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE*2, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE*2, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = TwoStageHierModel(width_mult=0.5, dropout=0.2).to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")

    ce_cat = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_ext = nn.CrossEntropyLoss(label_smoothing=0.1)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine scheduler per-iteration이 아니라 per-epoch 쓰고 싶으면 여기서 변경
    # 여기선 train_one_epoch에서 step()을 batch마다 호출하니까, T_max는 total_steps로 잡는 게 맞음
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-5)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = MODEL_RUNS_DIR / f"2stage_{run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "train_loss", "train_cat_acc", "train_ext_acc_truecat",
            "val_loss", "val_cat_acc", "val_ext_acc_truecat", "val_ext_acc_pipeline", "val_ext_macc"
        ])

    best_val = 0.0
    best_state = None
    patience = 7
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_cat_acc, tr_ext_acc = train_one_epoch(
            model, train_loader, opt, epoch, ce_cat, ce_ext, scheduler
        )

        val = eval_epoch(
            model, val_loader, epoch, "Val", ce_cat, ce_ext, save_dir=None
        )

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch,
                f"{tr_loss:.4f}", f"{tr_cat_acc:.6f}", f"{tr_ext_acc:.6f}",
                f"{val['loss']:.4f}", f"{val['cat_acc']:.6f}",
                f"{val['ext_acc_truecat']:.6f}", f"{val['ext_acc_pipeline']:.6f}",
                f"{val['ext_macc']:.6f}",
            ])

        print(
            f"[Summary] epoch={epoch} | "
            f"Train(cat={tr_cat_acc*100:.2f}%, ext_trueCat={tr_ext_acc*100:.2f}%) | "
            f"Val(cat={val['cat_acc']*100:.2f}%, ext_pipeline={val['ext_acc_pipeline']*100:.2f}%, mAcc={val['ext_macc']*100:.2f}%)"
        )
        print("-" * 70)

        # best model 기준: "pipeline extension acc"가 현실 성능
        if val["ext_acc_pipeline"] > best_val:
            best_val = val["ext_acc_pipeline"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    print("\n" + "=" * 70)
    print("FINAL TEST")
    test = eval_epoch(
        model, test_loader, epoch+1, "Test", ce_cat, ce_ext, save_dir=str(log_dir)
    )

    print("\nFinal Results (Pipeline 기준):")
    print(f"  Best Val ExtAcc(pipeline): {best_val*100:.2f}%")
    print(f"  Test CatAcc             : {test['cat_acc']*100:.2f}%")
    print(f"  Test ExtAcc(trueCat)    : {test['ext_acc_truecat']*100:.2f}%")
    print(f"  Test ExtAcc(pipeline)   : {test['ext_acc_pipeline']*100:.2f}%")
    print(f"  Test Ext mAcc           : {test['ext_macc']*100:.2f}%")

    # Save per-extension and per-category acc (compact separate files)
    save_per_extension_csv(test["per_ext_acc"], str(log_dir / "per_extension_acc_test.csv"))
    save_per_category_csv(test["per_cat_extacc"], str(log_dir / "per_category_acc_test.csv"))

    # Save checkpoint
    ckpt_path = log_dir / "two_stage_best.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_val_extacc_pipeline": float(best_val),
        "test_cat_acc": float(test["cat_acc"]),
        "test_ext_acc_pipeline": float(test["ext_acc_pipeline"]),
        "test_ext_macc": float(test["ext_macc"]),
        "param_count": int(count_parameters(model)),
    }, ckpt_path)

    print(f"\nSaved logs to: {log_dir}")
    print(f"Checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
