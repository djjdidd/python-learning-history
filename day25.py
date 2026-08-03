#局部失效图 —— 把每一行分成“安全 / 偏弱 / 失效”
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# ============================================================
# 1. 建立二维网格
# ============================================================
nx, ny = 401, 241
x = np.linspace(-200, 200, nx)
y = np.linspace(-120, 120, ny)
X, Y = np.meshgrid(x, y)

# ============================================================
# 2. 构造一个 gap 随 y 变化的 line-space-line mask
#    下半部分 gap 大，上半部分 gap 小
# ============================================================
mask = np.zeros((ny, nx), dtype=float)

left_width = 36
right_width = 36

y_min_pattern = -80
y_max_pattern = 80

for i in range(ny):
    yy = y[i]

    if y_min_pattern <= yy <= 0:
        # 左线中心位置随 y 连续变化：上面更靠近中间，gap 更小
        left_center = np.interp(yy, [y_min_pattern, y_max_pattern], [-70, -30])
        right_center = np.interp(yy, [y_min_pattern, y_max_pattern], [70, 30])

        left_x1 = left_center - left_width / 2
        left_x2 = left_center + left_width / 2

        right_x1 = right_center - right_width / 2
        right_x2 = right_center + right_width / 2

        mask[i, (x >= left_x1) & (x <= left_x2)] = 1.0
        mask[i, (x >= right_x1) & (x <= right_x2)] = 1.0
    if 0 <= yy <= 80:
        # 左线中心位置随 y 连续变化：上面更靠近中间，gap 更小
        left_center = np.interp(yy, [y_min_pattern, y_max_pattern], [-30, -70])
        right_center = np.interp(yy, [y_min_pattern, y_max_pattern], [30, 70])

        left_x1 = left_center - left_width / 2
        left_x2 = left_center + left_width / 2

        right_x1 = right_center - right_width / 2
        right_x2 = right_center + right_width / 2

        mask[i, (x >= left_x1) & (x <= left_x2)] = 1.0
        mask[i, (x >= right_x1) & (x <= right_x2)] = 1.0
# ============================================================
# 3. 定义二维高斯 PSF
# ============================================================
sigma = 12.0
psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf = psf / psf.sum()

# ============================================================
# 4. 卷积成像 + 阈值打印
# ============================================================
dose = 1.0
threshold = 0.18

aerial = fftconvolve(mask, psf, mode='same')
printed = (dose * aerial >= threshold).astype(int)

# ============================================================
# 5. 测量函数：一行里的 left / gap / right / num_features
# ============================================================
def measure_row_lsl(row, x):
    idx = np.where(row > 0)[0]

    if len(idx) == 0:
        return None, None, None, 0

    diff_idx = np.diff(idx)
    split_points = np.where(diff_idx > 1)[0]

    num_features = len(split_points) + 1

    if num_features != 2:
        return None, None, None, num_features

    split = split_points[0]

    left_part = idx[:split + 1]
    right_part = idx[split + 1:]

    left_cd = x[left_part[-1]] - x[left_part[0]]
    right_cd = x[right_part[-1]] - x[right_part[0]]
    gap_cd = x[right_part[0]] - x[left_part[-1]]

    return left_cd, gap_cd, right_cd, num_features

# ============================================================
# 6. 对每一行做分类
# ============================================================
fail_left_cd = 24
fail_gap_cd = 40
fail_right_cd = 24

weak_left_cd = 30
weak_gap_cd = 50
weak_right_cd = 30

status_list = []
valid_y_list = []

left_cd_list = []
gap_cd_list = []
right_cd_list = []
num_features_list = []

for i in range(ny):
    row = printed[i, :]
    left_cd, gap_cd, right_cd, num_features = measure_row_lsl(row, x)

    # 记录所有行的状态
    yy = y[i]
    valid_y_list.append(yy)
    num_features_list.append(num_features)

    if num_features != 2:
        status_list.append('fail')
        left_cd_list.append(np.nan)#这里没有有效数值
        gap_cd_list.append(np.nan)
        right_cd_list.append(np.nan)
    else:
        left_cd_list.append(left_cd)
        gap_cd_list.append(gap_cd)
        right_cd_list.append(right_cd)

        if (left_cd < fail_left_cd) or (gap_cd < fail_gap_cd) or (right_cd < fail_right_cd):
            status_list.append('fail')
        elif (left_cd < weak_left_cd) or (gap_cd < weak_gap_cd) or (right_cd < weak_right_cd):
            status_list.append('weak')
        else:
            status_list.append('safe')

valid_y_arr = np.array(valid_y_list)
left_cd_arr = np.array(left_cd_list)
gap_cd_arr = np.array(gap_cd_list)
right_cd_arr = np.array(right_cd_list)
num_features_arr = np.array(num_features_list)
status_arr = np.array(status_list)

# ============================================================
# 7. 把状态转成数字，方便画局部失效图
#    safe=2, weak=1, fail=0
# ============================================================
status_num = np.zeros_like(valid_y_arr, dtype=float)

status_num[status_arr == 'fail'] = 0
status_num[status_arr == 'weak'] = 1
status_num[status_arr == 'safe'] = 2

# ============================================================
# 8. 统计可用区域比例
# ============================================================
safe_ratio = np.mean(status_arr == 'safe')#这样就能算比例
weak_ratio = np.mean(status_arr == 'weak')
fail_ratio = np.mean(status_arr == 'fail')

# ============================================================
# 9. 可视化
# ============================================================
plt.figure(figsize=(15, 11))

# (1) printed pattern
plt.subplot(2, 2, 1)
plt.imshow(printed, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='gray')
plt.title('Printed Pattern')
plt.xlabel('x')
plt.ylabel('y')

# (2) 各行指标
plt.subplot(2, 2, 2)
plt.plot(valid_y_arr, left_cd_arr, label='left_cd')
plt.plot(valid_y_arr, gap_cd_arr, label='gap_cd')
plt.plot(valid_y_arr, right_cd_arr, label='right_cd')
plt.axhline(fail_left_cd, color='red', linestyle='--', alpha=0.6, label='fail threshold')
plt.axhline(weak_left_cd, color='orange', linestyle='--', alpha=0.6, label='weak threshold')
plt.xlabel('y')
plt.ylabel('CD / gap')
plt.title('Row-wise Metrics')
plt.legend()

# (3) 局部失效图
plt.subplot(2, 2, 3)
plt.plot(valid_y_arr, status_num, 'ko')#黑色圆点
plt.yticks([0, 1, 2], ['fail', 'weak', 'safe'])#把 y 轴上的数字刻度 0、1、2，改成文字标签。
plt.xlabel('y')
plt.ylabel('status')
plt.title('Local Failure Map')

# (4) 在 printed pattern 上标出不同状态区域
plt.subplot(2, 2, 4)
plt.imshow(printed, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='gray')

for yy, st in zip(valid_y_arr, status_arr):
    if st == 'fail':
        plt.axhline(yy, color='red', alpha=0.18)
    elif st == 'weak':
        plt.axhline(yy, color='orange', alpha=0.15)
    elif st == 'safe':
        plt.axhline(yy, color='green', alpha=0.12)

plt.title('Status Overlay on Printed Pattern')
plt.xlabel('x')
plt.ylabel('y')

summary_text = (
    f"safe ratio = {safe_ratio:.2%}\n"
    f"weak ratio = {weak_ratio:.2%}\n"
    f"fail ratio = {fail_ratio:.2%}"
)

plt.gcf().text(0.64, 0.03, summary_text, fontsize=11,
               bbox=dict(facecolor='white', alpha=0.85))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()