#二维插值与 focus-dose 工艺数据可视化

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# ==============================
# 1. 定义离散的工艺参数采样点
# ==============================
# focus_points: 焦距偏差的采样点
# dose_points: 曝光剂量的采样点
focus_points = np.array([-0.20, -0.10, 0.00, 0.10, 0.20])
dose_points = np.array([0.80, 0.90, 1.00, 1.10, 1.20])

# ==============================
# 2. 定义每组 (focus, dose) 下对应的 CD 数据
# ==============================
# 这是一个 5x5 矩阵：
# 行对应 focus_points
# 列对应 dose_points
# 例如 cd_matrix[2, 2] 表示 focus=0.00, dose=1.00 时的 CD
cd_matrix = np.array([
    [45.0, 42.0, 39.0, 37.0, 36.0],
    [43.0, 40.0, 37.0, 35.0, 34.0],
    [42.0, 39.0, 36.0, 34.0, 33.0],
    [43.0, 40.0, 37.0, 35.0, 34.0],
    [45.0, 42.0, 39.0, 37.0, 36.0]
])

# ==============================
# 3. 建立二维插值函数
# ==============================
# method='linear' 表示线性插值
# 也就是在相邻已知点之间做线性估计
interp_func = RegularGridInterpolator(
    (focus_points, dose_points),
    cd_matrix,
    method='linear'
)

# ==============================
# 4. 查询某一个中间参数点的 CD
# ==============================
# 这个点不在原始采样点上，所以需要插值估计
query_point = np.array([0.05, 1.05])
cd_query = interp_func(query_point)

print("在 focus=0.05, dose=1.05 时，插值得到的 CD =", cd_query)

# ==============================
# 5. 为了画更平滑的二维图，建立更密的参数网格
# ==============================
focus_fine = np.linspace(-0.20, 0.20, 100)
dose_fine = np.linspace(0.80, 1.20, 100)

# meshgrid 把一维坐标变成二维坐标网格
# indexing='ij' 表示第一个维度对应 focus，第二个维度对应 dose
FINE_FOCUS, FINE_DOSE = np.meshgrid(focus_fine, dose_fine, indexing='ij')

# ==============================
# 6. 对整个细网格上的所有点进行插值
# ==============================
# 先把二维网格展平成很多个 (focus, dose) 点
query_points = np.stack([FINE_FOCUS.ravel(), FINE_DOSE.ravel()], axis=-1)

# 对这些点逐一插值，然后再恢复成二维数组形状
cd_fine = interp_func(query_points).reshape(FINE_FOCUS.shape)

# ==============================
# 7. 画图：左边画原始矩阵，右边画插值后的平滑结果
# ==============================
plt.figure(figsize=(12, 5))

# ---- 左图：原始 5x5 数据矩阵 ----
plt.subplot(1, 2, 1)
plt.imshow(#把二维矩阵画成颜色图
    cd_matrix,
    extent=[dose_points.min(), dose_points.max(), focus_points.min(), focus_points.max()],
    origin='lower',
    aspect='auto'
)
plt.colorbar(label='CD')

# 在图上把原始采样点位置标出来
plt.scatter(
    np.repeat(dose_points, len(focus_points)),
    np.tile(focus_points, len(dose_points)),
    color='white',
    s=10
)

plt.xlabel("Dose")
plt.ylabel("Focus")
plt.title("Original Focus-Dose CD Matrix")

# ---- 右图：插值后的平滑二维图 ----
plt.subplot(1, 2, 2)
plt.imshow(
    cd_fine,
    extent=[dose_fine.min(), dose_fine.max(), focus_fine.min(), focus_fine.max()],
    origin='lower',
    aspect='auto'
)
plt.contour(FINE_DOSE, FINE_FOCUS, cd_fine, levels=10, colors='black')#把相同数值的位置画成线
plt.colorbar(label='CD')
plt.xlabel("Dose")
plt.ylabel("Focus")
plt.title("Interpolated Focus-Dose CD Matrix")

plt.tight_layout()
plt.show()