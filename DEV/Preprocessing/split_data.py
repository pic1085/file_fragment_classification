import numpy as np

# 원본 npz 파일 경로
in_path = "경로"
out_path = "경로"

# 데이터 로드
data = np.load(in_path, allow_pickle=True)
x = data["x"]
y = data["y"]

print("원본 x shape:", x.shape)
print("원본 y shape:", y.shape)

# 선택 클래스: JPG 제거 + PDF 추가
selected_ids = np.array([
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
    54,  # PDF  (추가)
], dtype=y.dtype)

# 해당 레이블만 남기기
mask = np.isin(y, selected_ids)

print("선택된 샘플 수:", mask.sum())

# 서브셋 추출
x_sub = x[mask]
y_sub = y[mask]

print("서브셋 x shape:", x_sub.shape)
print("서브셋 y shape:", y_sub.shape)

# 새 npz 저장
np.savez_compressed(out_path, x=x_sub, y=y_sub)
print("저장 완료:", out_path)

# 원본 x shape: (6144000, 4096)
# 원본 y shape: (6144000,)
# 선택된 샘플 수: 901466
# 서브셋 x shape: (901466, 4096)
# 서브셋 y shape: (901466,)
# 저장 완료: 경로