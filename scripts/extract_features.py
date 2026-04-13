import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
from tqdm import tqdm
from core.feature_extractor import extract_all_features
from core.paths import DATA_DIR, DATA_FEAT_DIR

OUT_DIR = DATA_FEAT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

for split in ['train', 'val', 'test']:
    data = np.load(DATA_DIR / f'{split}.npz', allow_pickle=True)
    X_raw = data['x']  # (N, 4096)
    y     = data['y']

    X_feat = np.zeros((len(X_raw), 397), dtype=np.float32)  # entropy_profile 추가시 371

    for i in tqdm(range(len(X_raw)), desc=f'{split} 특징 추출'):
        fragment = X_raw[i].astype(np.uint8).tobytes()
        X_feat[i] = extract_all_features(fragment)

    np.savez_compressed(OUT_DIR / f'{split}.npz', x=X_feat, y=y)
    print(f'{split} 저장 완료: {X_feat.shape}')
    
