import os
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.paths import DATA_DIR, N_RUNS_DIR, UNKNOWN_RUNS_DIR
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================
# Config
# ==========================
TRAIN_NPZ = str(DATA_DIR / "train.npz")
VAL_NPZ   = str(DATA_DIR / "val.npz")

# 네 파이프라인이 이미 만들어둔 결과 파일(예: pipeline_topk_ext.csv)
PIPELINE_CSV = N_RUNS_DIR / "runs_2stage_reco_pipeline" / "pipeline_topk_ext.csv"
OUT_DIR = UNKNOWN_RUNS_DIR / "runs_cnn_refine"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# “불확실 샘플” 정의
# - p1이 작거나
# - (p1-p2) margin이 작으면
THR_P1 = 0.55
THR_MARGIN = 0.10

# CNN이 분류할 목표
# 1) ext(75-way)로 바로 재분류  ✅ 추천
NUM_CLASSES = 75

# 학습은 불확실 샘플만
BATCH_SIZE = 256
EPOCHS = 8
LR = 2e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

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
EXT2ID = {e:i for i,e in enumerate(EXT_BY_ID)}

# ==========================
# Utils
# ==========================
def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    x = d["x"].astype(np.uint8, copy=False)   # (N,4096)
    y = d["y"].astype(np.int64, copy=False)   # (N,)
    return x, y

def make_uncertain_indices(pipeline_csv: str, thr_p1: float, thr_margin: float, split: str):
    """
    pipeline_topk_ext.csv 형태를 가정:
      - pred_ext1, pred_ext1_score, pred_ext2, pred_ext2_score ...
      - 혹은 pred1_prob, pred2_prob...
    네 파일 컬럼이 조금 다를 수 있어서 아래는 유연하게 처리.
    """
    df = pd.read_csv(pipeline_csv)
    # true가 들어있다고 가정: true_ext
    # split 구분이 없다면 그냥 전체를 대상으로 뽑고, 나중에 val에서만 refine해도 됨.
    # 여기서는 val refine가 목적이므로 df 전체가 val이라고 가정.

    # score 컬럼 찾기
    # (A) pred_ext1_score / pred_ext2_score
    if "pred_ext1_score" in df.columns and "pred_ext2_score" in df.columns:
        p1 = df["pred_ext1_score"].to_numpy(np.float32)
        p2 = df["pred_ext2_score"].to_numpy(np.float32)
    # (B) pred1_prob / pred2_prob
    elif "pred1_prob" in df.columns and "pred2_prob" in df.columns:
        p1 = df["pred1_prob"].to_numpy(np.float32)
        p2 = df["pred2_prob"].to_numpy(np.float32)
    else:
        raise ValueError("pipeline csv에서 확률/스코어 컬럼을 못 찾았어. 컬럼명을 확인해줘.")

    margin = p1 - p2
    uncertain = (p1 < thr_p1) | (margin < thr_margin)
    idx = df.index[uncertain].to_numpy(np.int64)

    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    df_unc = df.loc[idx].copy()
    df_unc["p1"] = p1[uncertain]
    df_unc["p2"] = p2[uncertain]
    df_unc["margin"] = margin[uncertain]
    df_unc.to_csv(out / "uncertain_samples.csv", index=False)
    print(f"[uncertain] count={len(idx)} / {len(df)} ({len(idx)/len(df)*100:.2f}%) saved -> {out/'uncertain_samples.csv'}")
    return idx

# ==========================
# Dataset (uncertain only)
# ==========================
class ByteFragmentDataset(Dataset):
    def __init__(self, x_u8: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.x = x_u8
        self.y = y
        self.idx = indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        # uint8 -> long (embedding index)
        x = torch.from_numpy(self.x[j].astype(np.int64, copy=False))  # (4096,)
        y = int(self.y[j])
        return x, y, j

# ==========================
# Model: Embedding + 1D CNN (fast & strong)
# ==========================
class CNN1DBytes(nn.Module):
    """
    bytes(0~255) -> embedding -> conv blocks -> global pooling -> classifier
    """
    def __init__(self, num_classes=75, emb_dim=32, channels=128, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(256, emb_dim)

        self.conv1 = nn.Conv1d(emb_dim, channels, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.bn2   = nn.BatchNorm1d(channels)

        self.conv3 = nn.Conv1d(channels, channels*2, kernel_size=5, stride=2, padding=2)
        self.bn3   = nn.BatchNorm1d(channels*2)

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels*2, num_classes)

    def forward(self, x):  # x: (B,4096) int64
        x = self.emb(x)          # (B,4096,emb)
        x = x.transpose(1, 2)    # (B,emb,4096)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # global max pool
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B,C)
        x = self.drop(x)
        logits = self.fc(x)
        return logits

# ==========================
# Train / Eval
# ==========================
@torch.no_grad()
def eval_topk(model, loader, k=3):
    model.eval()
    total = 0
    hit1 = 0
    hitk = 0
    for x, y, _ in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)

        topk = torch.topk(prob, k=k, dim=1).indices  # (B,k)
        pred1 = topk[:, 0]
        hit1 += (pred1 == y).sum().item()

        # top-k hit
        y2 = y.view(-1, 1)
        hitk += (topk == y2).any(dim=1).sum().item()

        total += y.size(0)
    return hit1/total, hitk/total

def train(model, tr_loader, va_loader):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # class imbalance 대응(불확실 샘플은 분포가 더 찌그러질 수 있음)
    # -> 간단히 “불확실 train subset”에서 class weight 계산
    y_all = []
    for _, y, _ in tr_loader:
        y_all.append(y.numpy())
    y_all = np.concatenate(y_all)
    counts = np.bincount(y_all, minlength=NUM_CLASSES).astype(np.float32)
    w = (counts.sum() / (counts + 1.0))  # inverse-ish
    w = w / w.mean()
    w_t = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w_t)

    best_va = 0.0
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "cnn_refiner.pt"

    for ep in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(tr_loader, desc=f"train ep{ep}")
        for x, y, _ in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        va1, va3 = eval_topk(model, va_loader, k=3)
        print(f"[val] ep{ep} top1={va1:.4f} top3={va3:.4f}")

        if va1 > best_va:
            best_va = va1
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"  saved best -> {ckpt_path}")

    # load best
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    return model

@torch.no_grad()
def predict_on_indices(model, x_u8: np.ndarray, indices: np.ndarray, topk=3):
    model.eval()
    preds = {}
    # batch 추론
    bs = 512
    for s in tqdm(range(0, len(indices), bs), desc="cnn_predict"):
        idx = indices[s:s+bs]
        x = torch.from_numpy(x_u8[idx].astype(np.int64, copy=False)).to(DEVICE)
        prob = torch.softmax(model(x), dim=1)
        vals, inds = torch.topk(prob, k=topk, dim=1)
        preds_batch = []
        for i in range(len(idx)):
            preds_batch.append((inds[i].cpu().numpy(), vals[i].cpu().numpy()))
        for j, pack in zip(idx, preds_batch):
            preds[int(j)] = pack
    return preds

# ==========================
# Main
# ==========================
def main():
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    print("DEVICE:", DEVICE)

    print("[Load]")
    x_tr, y_tr = load_npz(TRAIN_NPZ)
    x_va, y_va = load_npz(VAL_NPZ)
    print(" train:", x_tr.shape, y_tr.shape)
    print(" val  :", x_va.shape, y_va.shape)

    # 1) 파이프라인 결과에서 “val 불확실 인덱스” 뽑기
    uncertain_va_idx = make_uncertain_indices(PIPELINE_CSV, THR_P1, THR_MARGIN, split="val")

    # 2) (선택) train에서도 불확실을 뽑을 수 있지만,
    #    일단은 val 불확실을 “학습/검증”에 쓰면 누수라서 안 됨.
    #    대신 train은 전체를 쓰되, 학습량 줄이고 싶으면 “train의 일정 비율 random + 불확실 규칙”을 적용해야 함.
    # 여기서는 간단하게: train은 랜덤 서브샘플로 줄이자 (CNN은 불확실에 집중하니까)
    train_sub = 200000  # GPU 상황에 맞춰 조절 (예: 200k)
    tr_idx = np.random.choice(len(x_tr), size=min(train_sub, len(x_tr)), replace=False)

    # val은 “불확실만” 평가/추론할 거라서 그 subset만
    va_idx = uncertain_va_idx

    ds_tr = ByteFragmentDataset(x_tr, y_tr, tr_idx)
    ds_va = ByteFragmentDataset(x_va, y_va, va_idx)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)

    # 3) CNN 학습
    model = CNN1DBytes(num_classes=NUM_CLASSES, emb_dim=32, channels=128, dropout=0.1)
    model = train(model, dl_tr, dl_va)

    # 4) 불확실 샘플에 대해 top-k 예측 저장
    pred_map = predict_on_indices(model, x_va, va_idx, topk=3)

    rows = []
    for j in va_idx:
        j = int(j)
        true = int(y_va[j])
        topk_ids, topk_ps = pred_map[j]
        r = {
            "i": j,
            "true_ext_id": true,
            "true_ext": EXT_BY_ID[true],
        }
        for t in range(3):
            r[f"cnn_pred{t+1}_id"] = int(topk_ids[t])
            r[f"cnn_pred{t+1}"] = EXT_BY_ID[int(topk_ids[t])]
            r[f"cnn_pred{t+1}_prob"] = float(topk_ps[t])
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out / "cnn_uncertain_predictions.csv", index=False)
    print("saved ->", out / "cnn_uncertain_predictions.csv")

    # 5) (선택) 기존 pipeline_topk_ext.csv에 CNN 결과로 “덮어쓰기”한 최종 파일 생성
    pipe = pd.read_csv(PIPELINE_CSV)
    pipe["final_pred1_ext"] = pipe["pred_ext1"] if "pred_ext1" in pipe.columns else ""
    pipe["final_pred1_source"] = "pipeline"

    # val 인덱스는 pipe의 0..len(val)-1 순서라고 가정했을 때만 바로 매칭됨.
    # (너 지금 출력도 그 형태였음: i가 0.. 로 시작)
    for _, r in df.iterrows():
        i = int(r["i"])
        pipe.loc[i, "final_pred1_ext"] = r["cnn_pred1"]
        pipe.loc[i, "final_pred1_source"] = "cnn_refine"

    pipe.to_csv(out / "pipeline_with_cnn_refine.csv", index=False)
    print("saved ->", out / "pipeline_with_cnn_refine.csv")


if __name__ == "__main__":
    main()
