"""
경량화된 FFT-75 Subset (11 classes) 모델
- 파라미터 수 대폭 감소
- MobileNet 스타일 아키텍처
- Grouped Convolution + Channel Shuffle
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.paths import DATA_DIR, MODEL_RUNS_DIR
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==========================
# 0. 설정값
# ==========================

TRAIN_PATH = str(DATA_DIR / "train.npz")
VAL_PATH   = str(DATA_DIR / "val.npz")
TEST_PATH  = str(DATA_DIR / "test.npz")

BATCH_SIZE = 256  # 경량 모델이므로 배치 크기 증가 가능
NUM_EPOCHS = 50   # 빠른 수렴
LR = 2e-3

LABEL_IDS = list(range(75))
LABEL2INDEX = {orig: idx for idx, orig in enumerate(LABEL_IDS)}

NUM_CLASSES = len(LABEL_IDS)
FRAGMENT_SIZE = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================
# 1. Dataset (메모리 효율적)
# ==========================

class LightweightFFTDataset(Dataset):
    def __init__(self, npz_path: str, augment=False):
        data = np.load(npz_path, allow_pickle=True)
        # uint8로 유지하여 메모리 절약
        self.x = data["x"]  # numpy array로 유지
        self.y_orig = data["y"].astype(np.int64)
        self.augment = augment
        
        print(f"[{npz_path}] shape: {self.x.shape}, unique labels: {np.unique(self.y_orig)}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 필요할 때만 tensor로 변환
        fragment = self.x[idx].astype(np.int64)
        
        if self.augment and np.random.random() > 0.5:
            # 간단한 byte-level augmentation
            shift = np.random.randint(-10, 11)
            fragment = np.roll(fragment, shift)
        
        fragment = torch.from_numpy(fragment)
        orig_label = int(self.y_orig[idx])
        new_label = LABEL2INDEX[orig_label]
        return fragment, torch.tensor(new_label, dtype=torch.long)


# ==========================
# 2. 경량 빌딩 블록
# ==========================

class ChannelShuffle(nn.Module):
    """Channel Shuffle for grouped convolution"""
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
    
    def forward(self, x):
        B, C, L = x.shape
        g = self.groups
        # (B, g, C//g, L) -> (B, C//g, g, L) -> (B, C, L)
        return x.view(B, g, C//g, L).transpose(1, 2).contiguous().view(B, C, L)


class InvertedResidual(nn.Module):
    """MobileNetV2 스타일 Inverted Residual Block"""
    def __init__(self, in_ch, out_ch, kernel_size=3, expand_ratio=2, use_residual=True):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = use_residual and (in_ch == out_ch)
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU6(inplace=True),
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size//2, 
                     groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU6(inplace=True),
        ])
        
        # Project
        layers.extend([
            nn.Conv1d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class LightweightSEBlock(nn.Module):
    """경량 Squeeze-and-Excitation"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 1x1 conv 대신 linear 사용 (더 효율적)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Hardsigmoid(inplace=True)
        )
    
    def forward(self, x):
        B, C, L = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1)
        return x * y


# ==========================
# 3. 경량 분류 모델
# ==========================

class UltraLightClassifier(nn.Module):
    """초경량 모델 - 파라미터 < 100K 목표"""
    def __init__(self, num_classes=NUM_CLASSES, width_mult=0.5):
        super().__init__()
        
        # 경량 embedding (8차원)
        emb_dim = int(16 * width_mult)
        self.emb = nn.Embedding(256, emb_dim)
        
        # Stem (간단한 시작)
        c1 = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv1d(emb_dim, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU6(inplace=True)
        )
        
        # 경량 블록들
        c2 = int(48 * width_mult)
        c3 = int(64 * width_mult)
        
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
        
        # Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(c3, num_classes)
        
    def forward(self, x):
        # x: (B, 4096)
        x = self.emb(x)           # (B, 4096, emb_dim)
        x = x.permute(0, 2, 1)    # (B, emb_dim, 4096)
        
        x = self.stem(x)          # (B, c1, 2048)
        x = self.stage1(x)        # (B, c2, 1024)
        x = self.stage2(x)        # (B, c3, 512)
        x = self.stage3(x)        # (B, c3, 512)
        
        x = self.global_pool(x)   # (B, c3, 1)
        x = x.squeeze(-1)         # (B, c3)
        x = self.dropout(x)
        x = self.classifier(x)    # (B, num_classes)
        return x


class MediumLightClassifier(nn.Module):
    """중간 크기 경량 모델 - 더 나은 성능"""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Byte embedding
        self.emb = nn.Embedding(256, 24)
        
        # Multi-scale feature extraction (경량화)
        self.conv1 = nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU6(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU6(inplace=True),
        )
        
        # Grouped convolution for efficiency
        self.grouped_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, groups=4, bias=False),
            nn.BatchNorm1d(64),
            ChannelShuffle(groups=4),
            nn.ReLU6(inplace=True),
        )
        
        # Depthwise separable blocks
        self.blocks = nn.Sequential(
            InvertedResidual(64, 64, kernel_size=5, expand_ratio=3),
            nn.MaxPool1d(2),
            InvertedResidual(64, 96, kernel_size=5, expand_ratio=3, use_residual=False),
            LightweightSEBlock(96, reduction=8),
            nn.MaxPool1d(2),
            InvertedResidual(96, 96, kernel_size=3, expand_ratio=2),
        )
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU6(inplace=True),
            nn.Linear(48, num_classes)
        )
    
    def forward(self, x):
        # x: (B, 4096)
        x = self.emb(x)           # (B, 4096, 24)
        x = x.permute(0, 2, 1)    # (B, 24, 4096)
        
        # Multi-scale features
        x1 = self.conv1(x)        # (B, 32, 4096)
        x2 = self.conv2(x)        # (B, 32, 4096)
        x = torch.cat([x1, x2], dim=1)  # (B, 64, 4096)
        
        x = self.grouped_conv(x)  # (B, 64, 2048)
        x = self.blocks(x)        # (B, 96, 512)
        
        x = self.global_pool(x).squeeze(-1)  # (B, 96)
        x = self.dropout(x)
        x = self.fc(x)            # (B, num_classes)
        return x


# ==========================
# 4. 학습 유틸리티 (Knowledge Distillation 지원)
# ==========================

class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss"""
    def __init__(self, alpha=0.7, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, labels, teacher_logits=None):
        # Hard target loss
        loss = self.ce_loss(student_logits, labels)
        
        # Soft target loss (if teacher provided)
        if teacher_logits is not None:
            student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
            distill_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
            loss = self.alpha * loss + (1 - self.alpha) * distill_loss
        
        return loss


def count_parameters(model):
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, criterion, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    
    pbar = tqdm(loader, desc=f"Train {epoch:02d}")
    for x, y in pbar:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # 메모리 효율
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        bs = x.size(0)
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total_samples += bs
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc*100:.1f}%"})
    
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / total_samples, total_acc / total_samples


@torch.no_grad()
def eval_model(model, loader, criterion, epoch, phase="Val"):
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
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
        
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total_samples += bs
        
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
    
    # Per-class accuracy
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    class_accs = []
    for cls in range(NUM_CLASSES):
        cls_mask = (all_labels == cls)
        if cls_mask.sum() > 0:
            cls_acc = (all_preds[cls_mask] == cls).mean()
            class_accs.append(cls_acc)
    
    macro_acc = np.mean(class_accs) if class_accs else 0
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    
    print(f"[{phase} {epoch:02d}] Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.2f}% | mAcc: {macro_acc*100:.2f}%")
    
    return avg_loss, avg_acc, macro_acc


# ==========================
# 5. Pruning 유틸리티 (선택적)
# ==========================

def apply_pruning(model, amount=0.2):
    """구조적 프루닝 적용"""
    import torch.nn.utils.prune as prune
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            prune.l1_unstructured(module, name='weight', amount=amount)
        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    return model


# ==========================
# 6. Main
# ==========================

def main():
    print(f"Device: {DEVICE}")
    print("=" * 50)
    
    # 데이터 로드
    print("Loading datasets...")
    train_set = LightweightFFTDataset(TRAIN_PATH, augment=True)
    val_set = LightweightFFTDataset(VAL_PATH, augment=False)
    test_set = LightweightFFTDataset(TEST_PATH, augment=False)
    
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    
    # 데이터 로더
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE*2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE*2, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    model_choice = 1
    
    if model_choice == 1:
        model = UltraLightClassifier(num_classes=NUM_CLASSES, width_mult=0.5)
        model_name = "ultra_light"
    else:
         model = MediumLightClassifier(num_classes=NUM_CLASSES)
         model_name = "medium_light"

    model = model.to(DEVICE)
    
    # 파라미터 수 출력
    param_count = count_parameters(model)
    print(f"\nModel: {model_name}")
    print(f"Parameters: {param_count:,}")
    print(f"Model size: ~{param_count * 4 / 1024 / 1024:.2f} MB (FP32)")
    print("=" * 50)
    
    # 학습 설정
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = MODEL_RUNS_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macc"])
        
    # 학습
    best_val_acc = 0.0
    best_state = None
    patience = 7
    no_improve = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, scheduler
        )
        val_loss, val_acc, val_macc = eval_model(
            model, val_loader, criterion, epoch, "Val"
        )
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", 
                             f"{val_loss:.4f}", f"{val_acc:.4f}", f"{val_macc:.4f}"])
        
        print(f"[Summary] Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        print("-" * 50)
        
        # Best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # 최고 모델로 복원
    if best_state is not None:
        model.load_state_dict(best_state)
    # CSV 읽어서 그래프 저장
    rows = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    epochs = rows[:, 0]
    train_loss_arr = rows[:, 1]
    train_acc_arr  = rows[:, 2]
    val_loss_arr   = rows[:, 3]
    val_acc_arr    = rows[:, 4]

    plt.figure()
    plt.plot(epochs, train_loss_arr, label="train_loss")
    plt.plot(epochs, val_loss_arr, label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.savefig(log_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc_arr, label="train_acc")
    plt.plot(epochs, val_acc_arr, label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend()
    plt.savefig(log_dir / "acc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved logs to: {log_dir}")
    
    # 테스트
    print("\n" + "=" * 50)
    print("FINAL TEST")
    test_loss, test_acc, test_macc = eval_model(
        model, test_loader, criterion, NUM_EPOCHS + 1, "Test"
    )
    
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test Macro Acc: {test_macc*100:.2f}%")
    print(f"  Best Val Acc: {best_val_acc*100:.2f}%")
    
    # 모델 저장
    save_path = f"{model_name}_fft11_best.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'test_macc': test_macc,
        'param_count': param_count,
    }, save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Quantization 예시 (추가 경량화)
    print("\n" + "=" * 50)
    print("Quantization Preview:")
    print(f"  FP32 size: ~{param_count * 4 / 1024 / 1024:.2f} MB")
    print(f"  INT8 size: ~{param_count * 1 / 1024 / 1024:.2f} MB")
    print(f"  Compression ratio: 4x")


if __name__ == "__main__":
    main()
