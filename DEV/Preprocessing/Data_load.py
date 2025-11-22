import numpy as np

data = np.load("경로")

y = data["y"]

unique, counts = np.unique(y, return_counts=True)

print("class distribution: ", len(unique))
print("\n======= class counts =========")

for cls, cnt in zip(unique, counts):
    print(f"class {cls}: {cnt} samples")
    

# train.npz 갯수
# class 16: 81784 samples - png
# class 21: 82077 samples - mp4
# class 23: 82155 samples - avi
# class 39: 81751 samples - zip
# class 45: 81979 samples - docx
# class 47: 81855 samples - ppt
# class 54: 81908 samples - pdf
# class 57: 81929 samples - txt
# class 59: 82015 samples - json
# class 61: 82233 samples - xml
# class 63: 81780 samples - csv

# test.npz 갯수 
# class 16: 10376 samples
# class 21: 10150 samples
# class 23: 10180 samples
# class 39: 10399 samples
# class 45: 10285 samples
# class 47: 10349 samples
# class 54: 10203 samples
# class 57: 10270 samples
# class 59: 10258 samples
# class 61: 9995 samples
# class 63: 10286 samples

# val.npz 갯수
# class 16: 10240 samples
# class 21: 10173 samples
# class 23: 10065 samples
# class 39: 10250 samples
# class 45: 10136 samples
# class 47: 10196 samples
# class 54: 10289 samples
# class 57: 10201 samples
# class 59: 10127 samples
# class 61: 10172 samples
# class 63: 10334 samples