"""
dataset.py
FFT-75 .npz 파일 로더.
npz에 classes 키가 없으면 class_names.py의 FFT75_CLASSES를 자동 사용.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from .class_names import get_class_names


_X_CANDIDATES     = ['X', 'data', 'features', 'x', 'arr_0']
_Y_CANDIDATES     = ['y', 'labels', 'label', 'targets', 'arr_1']
_CLASS_CANDIDATES = ['classes', 'class_names', 'extensions', 'class_labels']


def _find_key(npz, candidates, role):
    for k in candidates:
        if k in npz:
            return k
    raise KeyError(
        f"[dataset] '{role}' 키를 찾을 수 없습니다.\n"
        f"  시도한 후보: {candidates}\n"
        f"  npz에 존재하는 키: {list(npz.keys())}\n"
        f"  → dataset.py 상단의 후보 리스트에 실제 키를 추가하세요."
    )


def inspect_npz(path: str) -> None:
    npz = np.load(path, allow_pickle=True)
    print(f"\n[inspect] {path}")
    print(f"  키 목록: {list(npz.keys())}")
    for k in npz.keys():
        arr = npz[k]
        if hasattr(arr, 'shape'):
            print(f"  '{k}': shape={arr.shape}, dtype={arr.dtype}")
            if arr.dtype.kind in ('U', 'S', 'O'):
                print(f"        샘플값: {arr[:5]}")
        else:
            print(f"  '{k}': {arr}")
    print()


def load_npz_split(path: str, desc: str = ""):
    npz = np.load(path, allow_pickle=True)
    x_key = _find_key(npz, _X_CANDIDATES, 'X (특징 행렬)')
    y_key = _find_key(npz, _Y_CANDIDATES, 'y (레이블)')

    steps = ["npz 읽기", "X 변환", "y 변환"]
    with tqdm(total=3, desc=desc or Path(path).name,
              bar_format="{l_bar}{bar:30}{r_bar}", ncols=80) as pbar:
        pbar.set_postfix_str(steps[0]); raw_x = npz[x_key]; pbar.update(1)
        pbar.set_postfix_str(steps[1]); X = raw_x.astype(np.float32); pbar.update(1)
        pbar.set_postfix_str(steps[2]); y = npz[y_key].astype(np.int32).ravel(); pbar.update(1)

    return X, y


def load_class_names_from_npz(npz_path: str, num_classes: Optional[int] = None) -> List[str]:
    """npz에서 클래스 이름을 읽고, 없으면 class_names.py 사용."""
    npz = np.load(npz_path, allow_pickle=True)
    for k in _CLASS_CANDIDATES:
        if k in npz:
            print(f"  클래스 이름: npz의 '{k}' 키에서 로드")
            return [str(n) for n in npz[k]]

    # npz에 없으면 class_names.py 사용
    builtin = get_class_names()
    if num_classes is not None and num_classes == len(builtin):
        print(f"  클래스 이름: class_names.py (FFT-75 기본값, {len(builtin)}개)")
        return builtin
    if num_classes is not None:
        print(f"  [경고] 클래스 수 불일치 (npz={num_classes}, 내장={len(builtin)}) → 숫자 대체")
        return [f"class_{i}" for i in range(num_classes)]
    return builtin


def load_fft75_npz(
    data_dir: str,
    train_file: str = "train.npz",
    val_file:   str = "val.npz",
    test_file:  str = "test.npz",
):
    data_dir = Path(data_dir)
    print(f"데이터 디렉토리: {data_dir}\n")

    splits = [
        (train_file, "train.npz 로드"),
        (val_file,   "val.npz   로드"),
        (test_file,  "test.npz  로드"),
    ]

    results = []
    for fname, desc in tqdm(splits, desc="전체 데이터 로드",
                            bar_format="{l_bar}{bar:20}{r_bar}",
                            ncols=80, position=0):
        p = data_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")
        X, y = load_npz_split(str(p), desc=desc)
        results.append((X, y))

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = results

    num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
    classes = load_class_names_from_npz(str(data_dir / train_file), num_classes)

    print(f"\n클래스 수    : {num_classes}")
    print(f"클래스 샘플  : {classes[:10]}{'...' if len(classes) > 10 else ''}")
    print(f"Train        : {X_train.shape[0]:,}개  |  Val: {X_val.shape[0]:,}개  |  Test: {X_test.shape[0]:,}개")
    print(f"Feature 차원 : {X_train.shape[1]}")

    return X_train, X_val, X_test, y_train, y_val, y_test, classes
