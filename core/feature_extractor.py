"""
feature_extractor.py
FFT-75 파일 조각(4096 byte)에서 세 가지 특징을 추출합니다.
  - 바이트 히스토그램 (256-bin)
  - 통계적 특징 (엔트로피, 카이제곱, 평균, 표준편차 등)
  - N-gram 빈도 (1-gram = 히스토그램, 2-gram 추가)
"""

import numpy as np
from scipy.stats import chisquare
from collections import Counter
from typing import Union
import math


FRAGMENT_SIZE = 4096

MAGIC_SIGNATURES = [
    (b'PK\x03\x04',      'ZIP/JAR/APK/DOCX'),   # ZIP 계열
    (b'Rar!\x1a\x07',    'RAR'),
    (b'\x1f\x8b',        'GZ'),
    (b'7z\xbc\xaf',      '7Z'),
    (b'BZh',             'BZ2'),
    (b'\xfd7zXZ',        'XZ'),
    (b'\xed\xab\xee\xdb','RPM'),
    (b'!<arch>',         'DEB'),
    (b'\xca\xfe\xba\xbe','MACH-O/DMG'),
    (b'MZ',              'EXE/DLL'),
    (b'\x7fELF',         'ELF'),
    (b'\xff\xd8\xff',    'JPG'),
    (b'\x89PNG',         'PNG'),
    (b'GIF8',            'GIF'),
    (b'BM',              'BMP'),
    (b'II\x2a\x00',      'TIFF-LE'),
    (b'MM\x00\x2a',      'TIFF-BE'),
    (b'%PDF',            'PDF'),
    (b'\xd0\xcf\x11\xe0','DOC/XLS/PPT'),   # OLE2 계열
    (b'fLaC',            'FLAC'),
    (b'ID3',             'MP3'),
    (b'OggS',            'OGG'),
    (b'RIFF',            'WAV/AVI'),
    (b'ftyp',            'MP4/MOV/M4A'),   # 오프셋 4에 위치
    (b'\x1aSQLite',      'SQLITE'),
    (b'\x00\x01\x00\x00','TTF'),
]

def magic_features(fragment: bytes) -> np.ndarray:
    """파일 시그니처 기반 매직 바이트 특징 (26차원)"""
    feat = np.zeros(len(MAGIC_SIGNATURES), dtype=np.float32)
    for i, (magic, _) in enumerate(MAGIC_SIGNATURES):
        # 앞부분 확인
        if fragment[:len(magic)] == magic:
            feat[i] = 1.0
        # ftyp는 오프셋 4에 위치 (MP4/MOV)
        elif magic == b'ftyp' and fragment[4:8] == magic:
            feat[i] = 1.0
    return feat

def byte_histogram(fragment: bytes) -> np.ndarray:
    """
    256-bin 바이트 빈도 히스토그램 (정규화 포함).
    파일 타입별로 특정 바이트 값의 분포가 다른 점을 활용.
    예: 텍스트 파일은 0x20-0x7E 범위에 집중, 압축 파일은 균일 분포.
    """
    hist = np.zeros(256, dtype=np.float32)
    for byte in fragment:
        hist[byte] += 1
    total = len(fragment)
    if total > 0:
        hist /= total  # 정규화: 비율로 변환
    return hist


def shannon_entropy(fragment: bytes) -> float:
    """
    Shannon 엔트로피 계산 (0 ~ 8 bits).
    - 압축/암호화 파일: ~8.0 (최대 무작위)
    - 텍스트 파일: ~4.0~5.0
    - 실행 파일: ~6.0~7.0
    혼동 쌍 분석에서 핵심 feature: zip vs gz가 모두 ~8.0으로 수렴하는 문제 확인 가능.
    """
    if not fragment:
        return 0.0
    freq = Counter(fragment)
    total = len(fragment)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def chi_square_stat(fragment: bytes) -> float:
    hist = np.zeros(256, dtype=np.float32)
    for byte in fragment:
        hist[byte] += 1
    expected = len(fragment) / 256.0
    # observed와 expected 합을 맞추기 위해 0 제거하지 않고 그대로 사용
    exp_vals = np.full(256, expected)
    stat, _ = chisquare(hist, f_exp=exp_vals)
    return float(stat)


def statistical_features(fragment: bytes) -> np.ndarray:
    """
    통계적 특징 벡터 (7차원).
    [엔트로피, 카이제곱, 평균, 표준편차, 최빈값, 고유 바이트 수, 0바이트 비율]
    """
    arr = np.frombuffer(fragment, dtype=np.uint8).astype(np.float32)
    entropy   = shannon_entropy(fragment)
    chi2      = chi_square_stat(fragment)
    mean      = float(np.mean(arr))
    std       = float(np.std(arr))
    mode_val  = float(Counter(fragment).most_common(1)[0][0])
    unique    = float(len(set(fragment)))
    zero_ratio = float(fragment.count(0)) / max(len(fragment), 1)

    return np.array([entropy, chi2, mean, std, mode_val, unique, zero_ratio],
                    dtype=np.float32)


def bigram_features(fragment: bytes, top_k: int = 100) -> np.ndarray:
    """
    2-gram(연속 두 바이트 쌍) 빈도 특징.
    전체 65536개 중 top_k개만 사용하여 차원 제어.
    파일 포맷별 고유한 바이트 시퀀스 패턴을 포착.
    예: JPEG의 0xFF 0xD8, ZIP의 0x50 0x4B 등.
    """
    bigram_counts = Counter()
    for i in range(len(fragment) - 1):
        bigram = (fragment[i], fragment[i + 1])
        bigram_counts[bigram] += 1

    total = max(sum(bigram_counts.values()), 1)
    # 가장 빈번한 top_k 개 bigram의 비율
    top_bigrams = bigram_counts.most_common(top_k)
    features = np.zeros(top_k, dtype=np.float32)
    for idx, (_, count) in enumerate(top_bigrams):
        features[idx] = count / total
    return features
def entropy_profile(fragment: bytes, n_blocks: int = 8) -> np.ndarray:
    block_size = len(fragment) // n_blocks
    return np.array([
        shannon_entropy(fragment[i*block_size:(i+1) * block_size])
        for i in range(n_blocks)
    ], dtype=np.float32)

def extract_all_features(fragment, use_bigram=True, bigram_topk=100):
    hist   = byte_histogram(fragment)       # 256
    stats  = statistical_features(fragment) # 7
    eprof  = entropy_profile(fragment)      # 8
    magic  = magic_features(fragment)       # 26  ← 추가
    parts  = [hist, stats, eprof, magic]
    if use_bigram:
        parts.append(bigram_features(fragment, bigram_topk))   # 100
    return np.concatenate(parts)  # 371 → 397차원


def load_fragment(path: str, offset: int = 0) -> bytes:
    """파일에서 4096 바이트 조각 하나를 읽어옵니다."""
    with open(path, 'rb') as f:
        f.seek(offset)
        return f.read(FRAGMENT_SIZE)
