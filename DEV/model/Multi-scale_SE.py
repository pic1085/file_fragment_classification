"""
FFT-75 Subset (11 classes)용 모델
- Inception + Depthwise Separable Conv 구조 유지
- Dual-SE (Channel + Position) + Multi-Scale Position SE 적용 버전
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==========================
# 0. 설정값
# ==========================

TRAIN_PATH = "path"
VAL_PATH   = "path"
TEST_PATH  = "path"

BATCH_SIZE = 256
NUM_EPOCHS = 50
LR = 1e-3

LABEL_IDS = [
    45,  # DOCX
    61,  # XML
    47,  # PPT
    21,  # MP4
    59,  # JSON
    63,  # CSV
    57,  # TXT
    39,  # ZIP
    23,  # AVI
    16,  # PNG
    54,  # PDF
]
LABEL2INDEX = {orig: idx for idx, orig in enumerate(LABEL_IDS)}

NUM_CLASSES = len(LABEL_IDS)   # 11
FRAGMENT_SIZE = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================
# 1. Dataset 정의
# ==========================

class FFTSubsetDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        x = data["x"]      # (N, 4096), uint8
        y = data["y"]      # (N,)

        assert x.ndim == 2 and x.shape[1] == FRAGMENT_SIZE, f"x shape 이상함: {x.shape}"
        assert x.shape[0] == y.shape[0], "x, y 길이가 다름"

        self.x = torch.from_numpy(x.astype(np.uint8))
        self.y_orig = y.astype(np.int64)

        uniq = np.unique(self.y_orig)
        print(f"[{npz_path}] unique labels:", uniq)

        for v in uniq:
            if v not in LABEL2INDEX:
                raise ValueError(
                    f"{npz_path}: 라벨 {v} 는 LABEL2INDEX에 없음. "
                    f"서브셋 만들 때 다른 클래스가 섞인 것 같음."
                )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fragment = self.x[idx].long()   # (4096,)
        orig_label = int(self.y_orig[idx])
        new_label = LABEL2INDEX[orig_label]      # 0~10
        label_tensor = torch.tensor(new_label, dtype=torch.long)
        return fragment, label_tensor


# ==========================
# 2. Multi-Scale Dual-SE Block
# ==========================

class MultiScaleDualSEBlock(nn.Module):
    """
    Dual-SE + Multi-Scale Position SE
    - Channel SE: 채널 방향 압축/확장
    - Position SE: 여러 스케일(예: 1, 4, 16)에서 위치 정보를 pooling 후 합성
    """
    def __init__(self, channels, reduction=16, seq_len=FRAGMENT_SIZE, pos_scales=(1, 4, 16)):
        super().__init__()

        # ---- Channel SE ----
        hidden_c = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden_c)
        self.fc2 = nn.Linear(hidden_c, channels)

        # ---- Multi-Scale Position SE ----
        # pos_scales: 예) 1, 4, 16 → 원본, 4배 다운, 16배 다운
        self.pos_scales = pos_scales

        # 채널 평균 후 (B, 1, L)에서 작업할 것이므로 conv 채널은 1로 고정
        self.pos_convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
            for _ in pos_scales
        ])

    def forward(self, x):
        B, C, L = x.size()

        # ===== Channel SE =====
        y = self.avg_pool(x).view(B, C)      # (B, C)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(B, C, 1)                  # (B, C, 1)

        # ===== Multi-Scale Position SE =====
        # 채널 평균 → (B, 1, L)
        p = x.mean(dim=1, keepdim=True)      # (B, 1, L)

        pos_out = 0.0
        for s, conv in zip(self.pos_scales, self.pos_convs):
            if s == 1:
                p_s = p                      # 원본 길이
            else:
                # 길이를 L/s 로 다운샘플링
                p_s = F.avg_pool1d(p, kernel_size=s, stride=s)

            # local mixing
            p_s = conv(p_s)

            # 다시 원래 길이 L로 업샘플링
            p_s_up = F.interpolate(p_s, size=L, mode="linear", align_corners=False)
            pos_out = pos_out + p_s_up

        pos_out = pos_out / len(self.pos_scales)
        pos_gate = torch.sigmoid(pos_out)    # (B, 1, L)

        # 최종 scale: channel gate * position gate
        return x * y * pos_gate


# ==========================
# 3. Inception + DSC Block 
# ==========================

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.depthwise = nn.Conv1d(
            in_ch, in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=False
        )
        self.pointwise = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InceptionDSCBlock(nn.Module):
    """
    Inception + Depthwise Separable block
    - Branch1 : DSConv k=11
    - Branch2 : DSConv k=19
    - Branch3 : DSConv k=27
    - Shortcut: 1x1 conv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch // 4

        self.branch1 = DepthwiseSeparableConv1d(in_ch, mid_ch, kernel_size=11)
        self.branch2 = DepthwiseSeparableConv1d(in_ch, mid_ch, kernel_size=19)
        self.branch3 = DepthwiseSeparableConv1d(in_ch, mid_ch, kernel_size=27)

        self.shortcut = nn.Conv1d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.shortcut_bn = nn.BatchNorm1d(mid_ch)

        self.fuse = nn.Conv1d(mid_ch * 4, out_ch, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm1d(out_ch)
        self.act = nn.Hardswish()

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        sc = self.shortcut_bn(self.shortcut(x))

        out = torch.cat([b1, b2, b3, sc], dim=1)  # (B, mid_ch*4, L)
        out = self.fuse(out)
        out = self.fuse_bn(out)
        out = self.act(out)
        return out


# ==========================
# 4. 최종 분류기 (Dual-SE + Multi-Scale Position SE)
# ==========================

class DSCSEClassifier(nn.Module):
    """
    - Embedding (0~255 -> 32차원)
    - Conv1d
    - InceptionDSCBlock x3 + MultiScaleDualSEBlock x3
    - Global AvgPool
    - 1x1 Conv -> num_classes
    """
    def __init__(self, num_classes=NUM_CLASSES, emb_dim=32):
        super().__init__()
        self.emb = nn.Embedding(256, emb_dim)

        self.conv_in = nn.Conv1d(emb_dim, 32, kernel_size=19, padding=9, bias=False)
        self.bn_in = nn.BatchNorm1d(32)
        self.act = nn.Hardswish()

        self.block1 = InceptionDSCBlock(32, 64)
        self.se1    = MultiScaleDualSEBlock(64, reduction=16, seq_len=FRAGMENT_SIZE)

        self.block2 = InceptionDSCBlock(64, 64)
        self.se2    = MultiScaleDualSEBlock(64, reduction=16, seq_len=FRAGMENT_SIZE)

        self.block3 = InceptionDSCBlock(64, 128)
        self.se3    = MultiScaleDualSEBlock(128, reduction=16, seq_len=FRAGMENT_SIZE)

        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Conv1d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 4096)
        x = self.emb(x)            # (B, 4096, 32)
        x = x.permute(0, 2, 1)     # (B, 32, 4096)

        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.act(x)

        x = self.block1(x)
        x = self.se1(x)

        x = self.block2(x)
        x = self.se2(x)

        x = self.block3(x)
        x = self.se3(x)

        x = self.global_avg(x)     # (B, C, 1)
        x = self.classifier(x)     # (B, num_classes, 1)
        x = x.squeeze(-1)          # (B, num_classes)
        return x


# ==========================
# 5. 학습 / 평가 루프
# ==========================

def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def train_one_epoch(model, loader, optimizer, criterion, epoch: int):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for x, y in tqdm(loader, desc=f"Train {epoch:02d}", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        total_samples += bs

    return total_loss / total_samples, total_acc / total_samples


@torch.no_grad()
def eval_model(model, loader, criterion, epoch: int, phase: str = "Val"):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []

    for x, y in tqdm(loader, desc=f"{phase} {epoch:02d}", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        total_samples += bs

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    num_classes = NUM_CLASSES
    class_accs = []
    for cls in range(num_classes):
        cls_mask = (all_labels == cls)
        if cls_mask.sum() == 0:
            continue
        cls_correct = (all_preds[cls_mask] == cls).sum()
        cls_total = cls_mask.sum()
        class_accs.append(cls_correct / cls_total)

    macro_acc = sum(class_accs) / len(class_accs)
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples

    print(f"[{phase} {epoch:02d}] Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.2f}% | mAcc: {macro_acc*100:.2f}%")

    return avg_loss, avg_acc, macro_acc


def save_model(model, path="dscse_fft11_best.pth"):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    abs_path = os.path.abspath(path)
    torch.save(state_dict, abs_path)
    print(f"\n[✔] Model saved to: {abs_path}")
    return abs_path


# ==========================
# 6. main
# ==========================

def main():
    print("Device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU count:", torch.cuda.device_count())

    print("Loading datasets...")
    train_set = FFTSubsetDataset(TRAIN_PATH)
    val_set   = FFTSubsetDataset(VAL_PATH)
    test_set  = FFTSubsetDataset(TEST_PATH)

    print(f"Train: {len(train_set)} samples")
    print(f"Val  : {len(val_set)} samples")
    print(f"Test : {len(test_set)} samples")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = DSCSEClassifier(num_classes=NUM_CLASSES)

    if DEVICE == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc, val_macc = eval_model(model, val_loader, criterion, epoch, phase="Val")

        scheduler.step()

        print(f"[Epoch {epoch:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%  ||  "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val mAcc: {val_macc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_macc = eval_model(model, test_loader, criterion, NUM_EPOCHS + 1, phase="Test")
    print("\n=== TEST RESULT ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc*100:.2f}%")
    print(f"Test mAcc: {test_macc*100:.2f}%")

    save_model(model, "dscse_fft11_best.pth")
    print("Saved model to dscse_fft11_best.pth")


if __name__ == "__main__":
    main()