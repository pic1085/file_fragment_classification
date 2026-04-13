"""
class_names.py
FFT-75 클래스 인덱스 → 확장자 매핑.
npz에 classes 키가 없을 때 이 파일에서 직접 로드합니다.

사용법:
    from core.class_names import FFT75_CLASSES, get_class_names
    classes = get_class_names()   # List[str], 길이 75
"""

FFT75_CLASSES = [
    "ARW",    # 0  Raw
    "CR2",    # 1  Raw
    "DNG",    # 2  Raw
    "GPR",    # 3  Raw
    "NEF",    # 4  Raw
    "NRW",    # 5  Raw
    "ORF",    # 6  Raw
    "PEF",    # 7  Raw
    "RAF",    # 8  Raw
    "RW2",    # 9  Raw
    "3FR",    # 10 Raw
    "JPG",    # 11 Bitmap
    "TIFF",   # 12 Bitmap
    "HEIC",   # 13 Bitmap
    "BMP",    # 14 Bitmap
    "GIF",    # 15 Bitmap
    "PNG",    # 16 Bitmap
    "AI",     # 17 Vector
    "EPS",    # 18 Vector
    "PSD",    # 19 Vector
    "MOV",    # 20 Video
    "MP4",    # 21 Video
    "3GP",    # 22 Video
    "AVI",    # 23 Video
    "MKV",    # 24 Video
    "OGV",    # 25 Video
    "WEBM",   # 26 Video
    "APK",    # 27 Archive
    "JAR",    # 28 Archive
    "MSI",    # 29 Archive
    "DMG",    # 30 Archive
    "7Z",     # 31 Archive
    "BZ2",    # 32 Archive
    "DEB",    # 33 Archive
    "GZ",     # 34 Archive
    "PKG",    # 35 Archive
    "RAR",    # 36 Archive
    "RPM",    # 37 Archive
    "XZ",     # 38 Archive
    "ZIP",    # 39 Archive
    "EXE",    # 40 Executables
    "MACH-O", # 41 Executables
    "ELF",    # 42 Executables
    "DLL",    # 43 Executables
    "DOC",    # 44 Office
    "DOCX",   # 45 Office
    "KEY",    # 46 Office
    "PPT",    # 47 Office
    "PPTX",   # 48 Office
    "XLS",    # 49 Office
    "XLSX",   # 50 Office
    "DJVU",   # 51 Published
    "EPUB",   # 52 Published
    "MOBI",   # 53 Published
    "PDF",    # 54 Published
    "MD",     # 55 Human-readable
    "RTF",    # 56 Human-readable
    "TXT",    # 57 Human-readable
    "TEX",    # 58 Human-readable
    "JSON",   # 59 Human-readable
    "HTML",   # 60 Human-readable
    "XML",    # 61 Human-readable
    "LOG",    # 62 Human-readable
    "CSV",    # 63 Human-readable
    "AIFF",   # 64 Audio
    "FLAC",   # 65 Audio
    "M4A",    # 66 Audio
    "MP3",    # 67 Audio
    "OGG",    # 68 Audio
    "WAV",    # 69 Audio
    "WMA",    # 70 Audio
    "PCAP",   # 71 Other
    "TTF",    # 72 Other
    "DWG",    # 73 Other
    "SQLITE", # 74 Other
]

# 카테고리 매핑 (혼동 쌍 분석 시 같은 카테고리 여부 판단에 활용)
FFT75_CATEGORIES = {
    "Raw":           list(range(0, 11)),
    "Bitmap":        list(range(11, 17)),
    "Vector":        list(range(17, 20)),
    "Video":         list(range(20, 27)),
    "Archive":       list(range(27, 40)),
    "Executables":   list(range(40, 44)),
    "Office":        list(range(44, 51)),
    "Published":     list(range(51, 55)),
    "Human-readable":list(range(55, 64)),
    "Audio":         list(range(64, 71)),
    "Other":         list(range(71, 75)),
}

# 인덱스 → 카테고리 역방향 매핑
IDX_TO_CATEGORY = {
    idx: cat
    for cat, indices in FFT75_CATEGORIES.items()
    for idx in indices
}


def get_class_names() -> list:
    """75개 확장자 이름 리스트 반환."""
    return FFT75_CLASSES.copy()


def idx_to_name(idx: int) -> str:
    """클래스 인덱스 → 확장자 이름."""
    return FFT75_CLASSES[idx]


def idx_to_category(idx: int) -> str:
    """클래스 인덱스 → 카테고리 이름."""
    return IDX_TO_CATEGORY.get(idx, "Unknown")


def is_same_category(idx_a: int, idx_b: int) -> bool:
    """두 클래스가 같은 카테고리인지 확인."""
    return IDX_TO_CATEGORY.get(idx_a) == IDX_TO_CATEGORY.get(idx_b)
