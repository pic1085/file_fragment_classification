import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm.auto import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.paths import DATA_DIR, MODEL_DIR, RESULT_DIR

# ==============================
# 0. 공통 설정
# ==============================

TEST_PATH = str(DATA_DIR / "test.npz")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
FRAGMENT_SIZE = 4096

# FFT-75 11클래스 서브셋 라벨
# LABEL_IDS = [
#     45,  # DOCX
#     61,  # XML
#     47,  # PPT
#     21,  # MP4
#     59,  # JSON
#     63,  # CSV
#     57,  # TXT
#     39,  # ZIP
#     23,  # AVI
#     16,  # PNG
#     54,  # PDF
# ]
LABEL_IDS = list(range(75))
LABEL2INDEX = {orig: idx for idx, orig in enumerate(LABEL_IDS)}
NUM_CLASSES = len(LABEL_IDS)
# ==============================
# INDEX2NAME (클래스 인덱스 -> "확장자:카테고리")
# ==============================
INDEX2NAME = [
    "ARW:Raw",            # 0
    "CR2:Raw",            # 1
    "DNG:Raw",            # 2
    "GPR:Raw",            # 3
    "NEF:Raw",            # 4
    "NRW:Raw",            # 5
    "ORF:Raw",            # 6
    "PEF:Raw",            # 7
    "RAF:Raw",            # 8
    "RW2:Raw",            # 9
    "3FR:Raw",            # 10
    "JPG:Bitmap",         # 11
    "TIFF:Bitmap",        # 12
    "HEIC:Bitmap",        # 13
    "BMP:Bitmap",         # 14
    "GIF:Bitmap",         # 15
    "PNG:Bitmap",         # 16
    "AI:Vector",          # 17
    "EPS:Vector",         # 18
    "PSD:Vector",         # 19
    "MOV:Video",          # 20
    "MP4:Video",          # 21
    "3GP:Video",          # 22
    "AVI:Video",          # 23
    "MKV:Video",          # 24
    "OGV:Video",          # 25
    "WEBM:Video",         # 26
    "APK:Archive",        # 27
    "JAR:Archive",        # 28
    "MSI:Archive",        # 29
    "DMG:Archive",        # 30
    "7Z:Archive",         # 31
    "BZ2:Archive",        # 32
    "DEB:Archive",        # 33
    "GZ:Archive",         # 34
    "PKG:Archive",        # 35
    "RAR:Archive",        # 36
    "RPM:Archive",        # 37
    "XZ:Archive",         # 38
    "ZIP:Archive",        # 39
    "EXE:Executables",    # 40
    "MACH-O:Executables", # 41
    "ELF:Executables",    # 42
    "DLL:Executables",    # 43
    "DOC:Office",         # 44
    "DOCX:Office",        # 45
    "KEY:Office",         # 46
    "PPT:Office",         # 47
    "PPTX:Office",        # 48
    "XLS:Office",         # 49
    "XLSX:Office",        # 50
    "DJVU:Published",     # 51
    "EPUB:Published",     # 52
    "MOBI:Published",     # 53
    "PDF:Published",      # 54
    "MD:Human-readable",  # 55
    "RTF:Human-readable", # 56
    "TXT:Human-readable", # 57
    "TEX:Human-readable", # 58
    "JSON:Human-readable",# 59
    "HTML:Human-readable",# 60
    "XML:Human-readable", # 61
    "LOG:Human-readable", # 62
    "CSV:Human-readable", # 63
    "AIFF:Audio",         # 64
    "FLAC:Audio",         # 65
    "M4A:Audio",          # 66
    "MP3:Audio",          # 67
    "OGG:Audio",          # 68
    "WAV:Audio",          # 69
    "WMA:Audio",          # 70
    "PCAP:Other",         # 71
    "TTF:Other",          # 72
    "DWG:Other",          # 73
    "SQLITE:Other",       # 74
]

# ==============================
# 1. Dataset 정의 (기존과 동일)
# ==============================

class FFTSubsetDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        x = data["x"]      # (N, 4096)
        y = data["y"]      # (N,)

        assert x.ndim == 2 and x.shape[1] == FRAGMENT_SIZE
        assert x.shape[0] == y.shape[0]

        self.x = torch.from_numpy(x.astype(np.uint8))
        self.y_orig = y.astype(np.int64)

        uniq = np.unique(self.y_orig)
        print(f"[{npz_path}] unique labels:", uniq)

        for v in uniq:
            if v not in LABEL2INDEX:
                raise ValueError(f"{npz_path}: label {v} not in LABEL2INDEX")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fragment = self.x[idx].long()
        orig_label = int(self.y_orig[idx])
        new_label = LABEL2INDEX[orig_label]
        return fragment, torch.tensor(new_label, dtype=torch.long)


# ==============================
# 2. 각 모델 아키텍처 불러오기
#    - 실제 파일 이름에 맞게 import 부분만 수정해서 사용
# ==============================

# (1) Dual-SE / Multi-Scale 모델: 둘 다 DSCSEClassifier 구조 사용
#   예: dual_se_fft11.py, multi_scale_fft11.py 이런 식으로 분리했다고 가정
from Dual_SE import DSCSEClassifier as DualSEClassifier
from Multi_scale_SE import DSCSEClassifier as MultiScaleClassifier
from lightweight_fft11_model import UltraLightClassifier, MediumLightClassifier

# ==============================
# 3. 체크포인트 로더 (state_dict vs dict 보호)
# ==============================

def load_state_dict_flexible(path: str):
    ckpt = torch.load(path, map_location="cpu")
    # 1) 그냥 state_dict 저장된 경우
    if isinstance(ckpt, dict) and all(
        not isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        # medium_light 쪽은 {'model_state_dict': ..., 'test_acc': ...} 형식
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        # 그 외 특이 형식이면 직접 key 확인 필요
    return ckpt  # 순수 state_dict라고 가정


# ==============================
# 4. 평가 함수 (per-class accuracy 계산)
# ==============================

@torch.no_grad()
def eval_per_class(model: nn.Module, loader: DataLoader, model_name: str):
    model.eval()
    total_correct = 0
    total_samples = 0

    # 클래스별 정답/전체 카운트
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_total   = np.zeros(NUM_CLASSES, dtype=np.int64)

    for x, y in tqdm(loader, desc=f"[{model_name}] Test", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x)
        preds = logits.argmax(dim=1)

        correct = (preds == y)
        total_correct += correct.sum().item()
        total_samples += y.size(0)

        # per-class 카운트
        for cls in range(NUM_CLASSES):
            mask = (y == cls)
            class_total[cls] += mask.sum().item()
            if mask.any():
                class_correct[cls] += (preds[mask] == cls).sum().item()

    overall_acc = total_correct / total_samples
    per_class_acc = class_correct / np.maximum(class_total, 1)

    print(f"\n[{model_name}] Overall Test Acc: {overall_acc*100:.2f}%")
    for i, name in enumerate(INDEX2NAME):
        print(f" - {name:4s}: {per_class_acc[i]*100:6.2f}% (n={class_total[i]})")

    return overall_acc, per_class_acc


# ==============================
# 5. 메인: 3개 모델 테스트 + 그림 저장
# ==============================

def main():
    print("Device:", DEVICE)

    # Test 데이터셋 & 로더
    test_set = FFTSubsetDataset(TEST_PATH)
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # # -------- (1) Dual-SE 모델 --------
    # dual_model = DualSEClassifier(num_classes=NUM_CLASSES)
    # dual_state = load_state_dict_flexible(str(RESULT_DIR / "Dual-SE_best.pth"))
    # dual_model.load_state_dict(dual_state)
    # dual_model = dual_model.to(DEVICE)

    # dual_overall, dual_per_class = eval_per_class(
    #     dual_model, test_loader, "Dual-SE"
    # )

    # # -------- (2) Multi-Scale SE 모델 --------
    # ms_model = MultiScaleClassifier(num_classes=NUM_CLASSES)
    # ms_state = load_state_dict_flexible(str(RESULT_DIR / "Multi-scale_SE_best.pth"))
    # ms_model.load_state_dict(ms_state)
    # ms_model = ms_model.to(DEVICE)

    # ms_overall, ms_per_class = eval_per_class(
    #     ms_model, test_loader, "Multi-Scale SE"
    # )

    # -------- (3) 경량 모델 (medium_light_fft11_best.pth) --------
    # 🔴 여기서 실제 학습한 구조에 맞게 선택!
    #   - UltraLight로 학습했다면 UltraLightClassifier 사용
    #   - MediumLight로 학습했다면 MediumLightClassifier 사용
    #
    # 예시: MediumLight로 학습했다고 가정
    # light_model = MediumLightClassifier(num_classes=NUM_CLASSES)
    # UltraLight였다면 ↓ 이렇게 바꾸면 됨
    light_model = UltraLightClassifier(num_classes=NUM_CLASSES, width_mult=0.5)

    light_state = load_state_dict_flexible(str(MODEL_DIR / "ultra_light_fft11_best.pth"))
    light_model.load_state_dict(light_state)
    light_model = light_model.to(DEVICE)

    light_overall, light_per_class = eval_per_class(
        light_model, test_loader, "Ultra-Light"
        # Medium-Light였다면 "Medium-Light"
    )
    
    # medium_light_model = MediumLightClassifier(num_classes=NUM_CLASSES)
    # medium_light_state = load_state_dict_flexible(str(RESULT_DIR / "medium_light_fft11_best.pth"))
    # medium_light_model.load_state_dict(medium_light_state)
    # medium_light_model = medium_light_model.to(DEVICE)

    # medium_light_overall, medium_light_per_class = eval_per_class(
    #     medium_light_model, test_loader, "Medium-Light"
    # )
    # ==========================
    # 6. per-class accuracy 시각화
    # ==========================

    x = np.arange(NUM_CLASSES)
    width = 0.25

    plt.figure(figsize=(12, 6))
    # plt.bar(x - width, dual_per_class * 100, width=width, label="Dual-SE")
    # plt.bar(x,         ms_per_class * 100,   width=width, label="Multi-Scale SE")
    plt.bar(x + width, light_per_class * 100, width=width, label="Ultra-Light")
    # plt.bar(x + 2*width, medium_light_per_class * 100, width=width, label="Medium-Light")

    plt.xticks(x, INDEX2NAME, rotation=45)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Per-Class Accuracy Comparison (Test set)")
    plt.legend()
    plt.tight_layout()

    os.makedirs("result/plots", exist_ok=True)
    save_path = "result/plots/per_class_accuracy_comparison.png"
    plt.savefig(save_path, dpi=200)
    print(f"\n[✔] per-class accuracy plot saved to: {save_path}")


if __name__ == "__main__":
    main()
