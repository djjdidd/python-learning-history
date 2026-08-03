import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage

x = np.linspace(-1.0, 1.0, 201)
y = np.linspace(-1.0, 1.0, 201)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

left_rect = ((X >= -0.35) & (X <= -0.12) & (np.abs(Y) <= 0.25))
right_rect = ((X >= 0.12) & (X <= 0.35) & (np.abs(Y) <= 0.25))
mask_2d = np.where(left_rect | right_rect, 1.0, 0.0)
sigma = 0.08
psf_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf_2d = psf_2d / np.sum(psf_2d)
aerial_2d = convolve2d(mask_2d, psf_2d, mode='same')
cd_target=1.0
dose_a=1.00
threshold_a=0.20
dose_b=0.95
threshold_b=0.22
def extract_cd_from_middle_row(binary_2d, x_coords):
    row_mid = binary_2d.shape[0] // 2
    profile = binary_2d[row_mid, :]
    printed_indices = np.where(profile == 1)[0]

    if len(printed_indices) == 0:
        return 0.0

    left_edge = x_coords[printed_indices[0]]
    right_edge = x_coords[printed_indices[-1]]
    cd_value = right_edge - left_edge
    return cd_value
def noise(noise_level,dose,threshold):
    noise=noise_level*np.random.randn(*aerial_2d.shape)
    pri=noise+aerial_2d
    pri=np.clip(pri,0.0,1.0)
    printed=(dose*pri>=threshold).astype(float)
    cd_mid=extract_cd_from_middle_row(printed,x)
    labeled_array, num_features = ndimage.label(printed)
    return printed,pri,cd_mid,num_features
noise_level = 0.02
num_trials = 50
cd_result_a=[]
fea_a=[]
cd_result_b=[]
fea_b=[]
for i in range(num_trials):
    printed,pri,cd_mid,num_features=noise(noise_level,dose_a,threshold_a)
    cd_result_a.append(cd_mid)
    fea_a.append(num_features)
for i in range(num_trials):
    printed,pri,cd_mid,num_features=noise(noise_level,dose_b,threshold_b)
    cd_result_b.append(cd_mid)
    fea_b.append(num_features)
cd_result_a = np.array(cd_result_a)
fea_a = np.array(fea_a)
cd_result_b = np.array(cd_result_b)
fea_b = np.array(fea_b)
cd_meana = np.mean(cd_result_a)
cd_stda = np.std(cd_result_a)
cd_meanb = np.mean(cd_result_b)
cd_stdb = np.std(cd_result_b)
unique_featuresa = np.unique(fea_a)#把数组里出现过的不同值找出来，并且去重
unique_featuresb = np.unique(fea_b)
for f in unique_featuresa:
    count = np.sum(fea_a == f)
    print(f"num_featuresa = {f}, 次数 = {count}")
unique_featuresa = np.unique(fea_a)#把数组里出现过的不同值找出来，并且去重
for f in unique_featuresb:
    count = np.sum(fea_b == f)
    print(f"num_featuresb = {f}, 次数 = {count}")
plt.figure(figsize=(15,9))
plt.subplot(2, 3, 1)
plt.plot(cd_result_a, marker='o')#折线图，表示每个点都画一个圆点
plt.axhline(cd_meana, color='red', linestyle='--', label='mean')
plt.xlabel("Trial a index")
plt.ylabel("CD")
plt.grid(True)
plt.legend()
plt.subplot(2, 3, 2)
plt.plot(cd_result_b, marker='o')#折线图，表示每个点都画一个圆点
plt.axhline(cd_meanb, color='red', linestyle='--', label='mean')
plt.xlabel("Trial b index")
plt.ylabel("CD")
plt.grid(True)
plt.legend()
plt.subplot(2, 3, 3)
plt.hist(cd_result_a, bins=15, edgecolor='black')#直方图，表示把 CD 的取值范围分成 15 个小区间，然后统计每个区间里有多少次试验结果落进去
plt.axvline(cd_meana, color='red', linestyle='--', label='mean')

plt.title("Histogram of CD under Noise a")
plt.xlabel("CD")
plt.ylabel("Count")
plt.legend()
plt.subplot(2, 3, 4)
plt.hist(cd_result_b, bins=15, edgecolor='black')#直方图，表示把 CD 的取值范围分成 15 个小区间，然后统计每个区间里有多少次试验结果落进去
plt.axvline(cd_meanb, color='red', linestyle='--', label='mean')

plt.title("Histogram of CD under Noise b")
plt.xlabel("CD")
plt.ylabel("Count")
plt.legend()
plt.subplot(2, 3, 5)
plt.plot(fea_a, marker='o',color='green',label='a')
plt.plot(fea_b, marker='o',color='pink',label='b')
plt.title("Connected Components over Trials")
plt.xlabel("Trial index")
plt.ylabel("num_features")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()