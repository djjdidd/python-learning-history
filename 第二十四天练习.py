#构造gap随y变化的mask
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

dx = x[1] - x[0]
dy = y[1] - y[0]

# ============================================================
# 2. 定义一个 line-space-line mask
#    这里故意做成 y 方向不完全均匀，制造局部薄弱区
# ============================================================
mask = np.zeros((ny, nx), dtype=float)

# 左线：在上半部分稍微变窄
left_center1 = -60
left_width=40
left_center2=-90
# 右线：基本保持不变
right_center = 60
right_width = 40

# 图形总高度
y_min_pattern = -80
y_max_pattern = 80

for i in range(ny):
    yy = y[i]

    if y_min_pattern <= yy <= y_max_pattern:
        # 左线宽度随 y 变化：上面更窄，下面更宽
        if yy < 0:
            current_left_centre = left_center1
        else:
            current_left_centre = left_center2

        left_x1 = current_left_centre - left_width / 2
        left_x2 = current_left_centre + left_width / 2

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
# 4. 卷积得到 aerial image
# ============================================================
aerial = fftconvolve(mask, psf, mode='same')

# ============================================================
# 5. 阈值打印
# ============================================================
dose = 1.0
threshold = 0.18

printed = (dose * aerial >= threshold).astype(int)

# ============================================================
# 6. 定义：测量一行中的两个 line 和中间 gap
# ============================================================
def measure_line_space_line_row(row, x):
    """
    输入:
        row : 一维 0/1 数组
        x   : x 坐标

    输出:
        left_cd, gap_cd, right_cd
    如果这一行结构不完整，则返回 None, None, None
    """
    idx = np.where(row > 0)[0]

    if len(idx) == 0:
        return None, None, None

    # 找连续区域的断点
    diff_idx = np.diff(idx)

    split_points = np.where(diff_idx > 1)[0]

    # 如果不是两个独立 feature，则认为结构异常
    if len(split_points) != 1:
        return None, None, None

    split = split_points[0]

    left_part = idx[:split + 1]
    right_part = idx[split + 1:]

    left_cd = x[left_part[-1]] - x[left_part[0]]
    right_cd = x[right_part[-1]] - x[right_part[0]]
    gap_cd = x[right_part[0]] - x[left_part[-1]]

    return left_cd, gap_cd, right_cd

# ============================================================
# 7. 对每一行做扫描测量
# ============================================================
left_cd_list = []
gap_cd_list = []
right_cd_list = []
valid_y_list = []

for i in range(ny):
    row = printed[i, :]
    left_cd, gap_cd, right_cd = measure_line_space_line_row(row, x)

    if left_cd is not None:
        left_cd_list.append(left_cd)
        gap_cd_list.append(gap_cd)
        right_cd_list.append(right_cd)
        valid_y_list.append(y[i])

left_cd_arr = np.array(left_cd_list)
gap_cd_arr = np.array(gap_cd_list)
right_cd_arr = np.array(right_cd_list)
valid_y_arr = np.array(valid_y_list)

# ============================================================
# 8. 找最薄弱位置
# ============================================================
min_left_idx = np.argmin(left_cd_arr)
min_gap_idx = np.argmin(gap_cd_arr)
min_right_idx = np.argmin(right_cd_arr)

min_left_cd = left_cd_arr[min_left_idx]
min_gap_cd = gap_cd_arr[min_gap_idx]
min_right_cd = right_cd_arr[min_right_idx]

min_left_y = valid_y_arr[min_left_idx]
min_gap_y = valid_y_arr[min_gap_idx]
min_right_y = valid_y_arr[min_right_idx]

# ============================================================
# 9. 定义“薄弱区”
#    这里先用简单规则：
#    小于各自最大值的 90% 就算偏薄弱
# ============================================================
left_weak_mask = left_cd_arr < 0.9 * np.max(left_cd_arr)
gap_weak_mask = gap_cd_arr < 0.9 * np.max(gap_cd_arr)
right_weak_mask = right_cd_arr < 0.9 * np.max(right_cd_arr)

# ============================================================
# 10. 可视化
# ============================================================
plt.figure(figsize=(14, 10))

# (1) mask
plt.subplot(2, 2, 1)
plt.imshow(mask, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='gray')
plt.title('Mask')
plt.xlabel('x')
plt.ylabel('y')

# (2) printed pattern
plt.subplot(2, 2, 2)
plt.imshow(printed, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='gray')
plt.axhline(min_left_y, color='red', linestyle='--', label='min left_cd row')
plt.axhline(min_gap_y, color='blue', linestyle='--', label='min gap_cd row')
plt.axhline(min_right_y, color='green', linestyle='--', label='min right_cd row')
plt.title('Printed Pattern')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# (3) 各行测量结果
plt.subplot(2, 2, 3)
plt.plot(valid_y_arr, left_cd_arr, label='left_cd')
plt.plot(valid_y_arr, gap_cd_arr, label='gap_cd')
plt.plot(valid_y_arr, right_cd_arr, label='right_cd')
plt.axvline(min_left_y, color='red', linestyle='--')
plt.axvline(min_gap_y, color='blue', linestyle='--')
plt.axvline(min_right_y, color='green', linestyle='--')
plt.xlabel('y')
plt.ylabel('CD / gap')
plt.title('Row-wise Metrics')
plt.legend()

# (4) 薄弱区标记
plt.subplot(2, 2, 4)
plt.plot(valid_y_arr, left_cd_arr, label='left_cd')#'ro'为简写red o
plt.plot(valid_y_arr[left_weak_mask], left_cd_arr[left_weak_mask], 'ro', label='left weak')
plt.plot(valid_y_arr, gap_cd_arr, label='gap_cd')
plt.plot(valid_y_arr[gap_weak_mask], gap_cd_arr[gap_weak_mask], 'bo', label='gap weak')
plt.plot(valid_y_arr, right_cd_arr, label='right_cd')
plt.plot(valid_y_arr[right_weak_mask], right_cd_arr[right_weak_mask], 'go', label='right weak')
plt.xlabel('y')
plt.ylabel('CD / gap')
plt.title('Weak Regions')
plt.legend()

summary_text = (
    f"min left_cd = {min_left_cd:.2f} at y = {min_left_y:.1f}\n"
    f"min gap_cd = {min_gap_cd:.2f} at y = {min_gap_y:.1f}\n"
    f"min right_cd = {min_right_cd:.2f} at y = {min_right_y:.1f}"
)

plt.gcf().text(0.52, 0.02, summary_text, fontsize=11,
               bbox=dict(facecolor='white', alpha=0.8))
#gcf获取当前整张图 0.52,0.02这里不是数据坐标，而是整张图的相对坐标。
#fontsize字体大小是 11 bbox这个是在给文字加一个背景框 dict给这段文字配一个白色半透明背景框
plt.tight_layout(rect=[0, 0.05, 1, 1])#给整张图留出一个矩形范围，让子图只排布在这个范围里
plt.show()