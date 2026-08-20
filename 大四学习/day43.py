#从逐行 EPE 走向 contour——target contour 和 printed contour 到底是什么
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


# =========================================================
# 1. 建立二维网格
# =========================================================
nx, ny = 401, 241
x = np.linspace(-2.0, 2.0, nx)
y = np.linspace(-1.2, 1.2, ny)
xx, yy = np.meshgrid(x, y)


# =========================================================
# 2. 构造二维 line-space-line mask
# =========================================================
mask = np.zeros((ny, nx), dtype=float)

y_min_pattern = -0.75
y_max_pattern = 0.75

left_x1, left_x2 = -1.10, -0.35
right_x1, right_x2 = 0.35, 1.10

pattern_region = (yy >= y_min_pattern) & (yy <= y_max_pattern)

left_line = (xx >= left_x1) & (xx <= left_x2) & pattern_region
right_line = (xx >= right_x1) & (xx <= right_x2) & pattern_region

mask[left_line] = 1.0
mask[right_line] = 1.0

# 制造右边线局部薄弱区
middle_weak_region = (yy >= -0.20) & (yy <= 0.20)
right_line_trim = (xx >= 0.35) & (xx <= 0.48) & middle_weak_region
mask[right_line_trim] = 0.0


# =========================================================
# 3. PSF 和 aerial image
# =========================================================
sigma = 0.12
psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
psf /= psf.sum()

aerial = fftconvolve(mask, psf, mode='same')


# =========================================================
# 4. 阈值打印
# =========================================================
dose = 0.98
threshold = 0.28
printed = (dose * aerial >= threshold).astype(float)


# =========================================================
# 5. 简单二维边界提取函数
# =========================================================
def extract_boundary(binary_image):
    """
    输入：
        binary_image: 二值图,1 表示图形区域,0 表示背景

    输出：
        boundary: 二值边界图,1 表示边界点,0 表示非边界点

    简化规则：
        如果某个像素是 1,
        但它的上下左右至少有一个邻居是 0,
        那它就是边界。
    """

    ny, nx = binary_image.shape
    boundary = np.zeros_like(binary_image)

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            if binary_image[i, j] == 0:
                continue

            up = binary_image[i - 1, j]
            down = binary_image[i + 1, j]
            left = binary_image[i, j - 1]
            right = binary_image[i, j + 1]

            if (up == 0) or (down == 0) or (left == 0) or (right == 0):
                boundary[i, j] = 1

    return boundary


# =========================================================
# 6. 提取 target contour 和 printed contour
# =========================================================
target_contour = extract_boundary(mask)
printed_contour = extract_boundary(printed)


# =========================================================
# 7. 构造 contour 差异图
# =========================================================
# 0: 背景
# 1: target contour
# 2: printed contour
# 3: target 和 printed 重合
contour_compare = np.zeros_like(mask)

contour_compare[target_contour == 1] = 1
contour_compare[printed_contour == 1] = 2
contour_compare[(target_contour == 1) & (printed_contour == 1)] = 3


# =========================================================
# 8. 计算简单 contour overlap 指标
# =========================================================
target_count = np.sum(target_contour == 1)
printed_count = np.sum(printed_contour == 1)
overlap_count = np.sum((target_contour == 1) & (printed_contour == 1))

target_overlap_ratio = overlap_count / target_count if target_count > 0 else 0
printed_overlap_ratio = overlap_count / printed_count if printed_count > 0 else 0


# =========================================================
# 9. 作图
# =========================================================
plt.figure(figsize=(16, 10))

# (1) target mask
plt.subplot(2, 3, 1)
plt.imshow(
    mask,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Target Mask")
plt.xlabel("x")
plt.ylabel("y")

# (2) printed pattern
plt.subplot(2, 3, 2)
plt.imshow(
    printed,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title(f"Printed Pattern\n(dose={dose:.2f}, threshold={threshold:.2f})")
plt.xlabel("x")
plt.ylabel("y")

# (3) target contour
plt.subplot(2, 3, 3)
plt.imshow(
    target_contour,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Target Contour")
plt.xlabel("x")
plt.ylabel("y")

# (4) printed contour
plt.subplot(2, 3, 4)
plt.imshow(
    printed_contour,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Printed Contour")
plt.xlabel("x")
plt.ylabel("y")

# (5) contour compare
plt.subplot(2, 3, 5)
plt.imshow(
    contour_compare,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='viridis'
)
plt.colorbar()
plt.title("Contour Compare\n1 target, 2 printed, 3 overlap")
plt.xlabel("x")
plt.ylabel("y")

# (6) summary
plt.subplot(2, 3, 6)
plt.axis('off')

summary_text = (
    "[Contour statistics]\n"
    f"target contour points  = {target_count}\n"
    f"printed contour points = {printed_count}\n"
    f"overlap contour points = {overlap_count}\n\n"
    f"target overlap ratio   = {target_overlap_ratio:.3f}\n"
    f"printed overlap ratio  = {printed_overlap_ratio:.3f}\n\n"
    "[Interpretation]\n"
    "Target contour: desired boundary.\n"
    "Printed contour: actual printed boundary.\n"
    "Overlap means the two contours agree at the pixel level.\n"
    "Low overlap indicates contour mismatch."
)

plt.text(0.02, 0.98, summary_text, va='top', fontsize=10)
plt.title("Summary")

plt.tight_layout()
plt.show()