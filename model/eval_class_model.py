import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.paths import DATA_DIR, MODEL_DIR

# ==========================
# 0. 경로/설정
# ==========================
TEST_PATH = str(DATA_DIR / "test.npz")
CKPT_PATH = str(MODEL_DIR / "ultra_light_cluster_best.pth")  # ✅ 너 저장된 경로로 수정

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
FRAGMENT_SIZE = 4096

# 확장자(0~74) -> "EXT:CATEGORY"
EXT_INDEX2NAME = [
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
assert len(EXT_INDEX2NAME) == 75

CATEGORIES = [
    "Raw", "Bitmap", "Vector", "Video", "Archive",
    "Executables", "Office", "Published", "Human-readable", "Other", "Audio"
]
CAT2ID = {c:i for i,c in enumerate(CATEGORIES)}
NUM_CLASSES = len(CATEGORIES)

# 확장자 라벨(0~74) -> 카테고리 라벨(0~10)
LABEL2CATID = np.array([CAT2ID[s.split(":")[1]] for s in EXT_INDEX2NAME], dtype=np.int64)

# ==========================
# 1. Dataset (학습 코드와 동일 변환!)
# ==========================
class ClusterDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.x = data["x"].astype(np.uint8, copy=False)
        self.y_ext = data["y"].astype(np.int64, copy=False)  # 0~74

        assert self.x.ndim == 2 and self.x.shape[1] == FRAGMENT_SIZE
        assert self.x.shape[0] == self.y_ext.shape[0]

        # ✅ 확장자 -> 카테고리로 변환
        self.y_cat = LABEL2CATID[self.y_ext]

        print(f"[{npz_path}] y_ext unique: {np.unique(self.y_ext)[:10]} ...")
        print(f"[{npz_path}] y_cat unique: {np.unique(self.y_cat)} (should be 0~10)")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        frag = torch.from_numpy(self.x[idx]).long()
        y = int(self.y_cat[idx])
        return frag, torch.tensor(y, dtype=torch.long)

# ==========================
# 2. 모델 (학습 코드와 동일 클래스 import)
# ==========================
from lightweight_fft11_model import UltraLightClassifier  # 너가 쓰던 파일 그대로

def load_state_dict_flexible(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt

@torch.no_grad()
def evaluate(model, loader, save_dir="result/plots"):
    model.eval()
    total_correct = 0
    total = 0

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)

    for x, y in tqdm(loader, desc="Test", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        pred = model(x).argmax(dim=1)

        total_correct += (pred == y).sum().item()
        total += y.size(0)

        y_np = y.cpu().numpy()
        p_np = pred.cpu().numpy()

        np.add.at(conf, (y_np, p_np), 1)
        np.add.at(class_total, y_np, 1)
        np.add.at(class_correct, y_np, (p_np == y_np).astype(np.int64))

    overall = total_correct / total
    per_cls = class_correct / np.maximum(class_total, 1)

    print(f"\nOverall Acc: {overall*100:.2f}%")
    print("Per-category accuracy:")
    for i, a in enumerate(per_cls):
        print(f" - {CATEGORIES[i]:15s}: {a*100:6.2f}% (n={class_total[i]})")

    os.makedirs(save_dir, exist_ok=True)

    # ==========================
    # ✅ 1) Bar plot 저장
    # ==========================
    plt.figure(figsize=(10, 5))
    x = np.arange(NUM_CLASSES)

    plt.bar(x, per_cls * 100)
    plt.xticks(x, CATEGORIES, rotation=45, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title(f"Per-Category Accuracy (11 classes) | Overall: {overall*100:.2f}%")
    plt.tight_layout()

    bar_path = os.path.join(save_dir, "per_category_accuracy_bar.png")
    plt.savefig(bar_path, dpi=250)
    plt.close()
    print(f"[✔] Saved bar plot: {bar_path}")

    # ==========================
    # ✅ 2) Confusion matrix 저장 (기존)
    # ==========================
    conf_norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(10, 9))
    im = plt.imshow(conf_norm, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(NUM_CLASSES), CATEGORIES, rotation=45, ha="right")
    plt.yticks(np.arange(NUM_CLASSES), CATEGORIES)
    plt.title("Confusion Matrix (row-normalized) - Category 11")
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout()

    cm_path = os.path.join(save_dir, "confusion_category11.png")
    plt.savefig(cm_path, dpi=250)
    plt.close()
    print(f"[✔] Saved confusion matrix: {cm_path}")

def main():
    ds = ClusterDataset(TEST_PATH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = UltraLightClassifier(num_classes=NUM_CLASSES, width_mult=0.5).to(DEVICE)
    sd = load_state_dict_flexible(CKPT_PATH)
    model.load_state_dict(sd)

    evaluate(model, dl, save_dir="result/plots")

if __name__ == "__main__":
    main()
