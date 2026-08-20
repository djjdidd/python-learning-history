# 第54天：从 OPC 过渡到 ILT
# Toy ILT: 把 mask 当作二维变量，用误差反馈做简化反演优化
#ILT：Inverse Lithography Technology，反演光刻技术
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
# 2. 固定参数
# =========================================================
y_min_pattern = -0.75
y_max_pattern = 0.75

left_x1, left_x2 = -1.10, -0.35

# 右边三段式 target
right_x1_target = 0.35
right_x3_target = 0.55
right_x2 = 1.10

sigma = 0.12
dose = 0.98
threshold = 0.28


# =========================================================
# 3. 构造三段式 target mask
# =========================================================
def build_target_mask():
    """
    构造目标图形 target。

    这里 target 仍然是：
        左边一条矩形线
        右边三段式矩形线
    """

    target = np.zeros((ny, nx), dtype=float)

    pattern_region = (yy >= y_min_pattern) & (yy <= y_max_pattern)

    # 左线
    left_line = (
        (xx >= left_x1)
        & (xx <= left_x2)
        & pattern_region
    )

    # 右线下段
    lower_region = (
        (yy >= -0.75)
        & (yy <= -0.25)
    )

    right_lower = (
        (xx >= right_x1_target)
        & (xx <= right_x2)
        & lower_region
    )

    # 右线中段
    middle_region = (
        (yy >= -0.25)
        & (yy <= 0.25)
    )

    right_middle = (
        (xx >= right_x3_target)
        & (xx <= right_x2)
        & middle_region
    )

    # 右线上段
    upper_region = (
        (yy >= 0.25)
        & (yy <= 0.75)
    )

    right_upper = (
        (xx >= right_x1_target)
        & (xx <= right_x2)
        & upper_region
    )

    target[left_line] = 1.0
    target[right_lower] = 1.0
    target[right_middle] = 1.0
    target[right_upper] = 1.0

    return target


# =========================================================
# 4. 构造一个初始 mask
# =========================================================
def build_initial_mask():
    """
    初始 mask 先直接等于 target。

    真实 ILT 里，初始 mask 可以是 target layout、
    OPC 后 mask，或者某种初始猜测。
    """

    return build_target_mask().copy()


# =========================================================
# 5. PSF
# =========================================================
def build_psf():
    """
    构造高斯 PSF。

    这里仍然用简化的 Gaussian PSF 模拟光学模糊。
    """

    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()

    return psf


# =========================================================
# 6. sigmoid 函数
# =========================================================
def sigmoid(z, sharpness=40):
    """
    平滑阈值函数。

    之前我们用的是：
        printed = (dose * aerial >= threshold).astype(float)

    这是硬阈值，不方便做连续优化。

    今天为了讲 ILT 直觉，用 sigmoid 做一个软阈值：
        value 接近 1：表示更像 printed 区域
        value 接近 0：表示更像非 printed 区域
    """

    return 1.0 / (1.0 + np.exp(-sharpness * z))


# =========================================================
# 7. 成像：得到 aerial 和 soft printed
# =========================================================
def simulate_soft_print(mask, psf):
    """
    mask -> aerial image -> soft printed pattern

    注意：
        soft_printed 不是 0/1 二值图，
        而是 0 到 1 之间的连续值。
    """

    aerial = fftconvolve(mask, psf, mode='same')

    soft_printed = sigmoid(dose * aerial - threshold)

    return aerial, soft_printed


# =========================================================
# 8. 硬阈值 printed，用于最后显示
# =========================================================
def hard_print(aerial):
    """
    把 aerial image 用硬阈值得到 printed pattern。
    """

    printed = (dose * aerial >= threshold).astype(float)

    return printed


# =========================================================
# 9. 计算 loss
# =========================================================
def compute_loss(soft_printed, target):
    """
    简化 loss:
        loss = mean((soft_printed - target)^2)

    也就是 soft printed 和 target 的均方误差。
    """

    error = soft_printed - target
    loss = np.mean(error**2)

    return loss


# =========================================================
# 10. Toy ILT 迭代
# =========================================================
def run_toy_ilt(initial_mask, target, psf, num_iter=30, learning_rate=0.8):
    """
    玩具版 ILT 迭代。

    重要说明：
        这不是严格真实 ILT 的完整梯度推导。
        它是一个用于教学的 error back-projection 版本。

    核心思想：
        1. 当前 mask 经过成像，得到 soft_printed
        2. 比较 soft_printed 和 target，得到 error
        3. 把 error 反向卷积回 mask 空间
        4. 根据反向误差更新 mask
        5. 用 clip 限制 mask 在 [0, 1]
    """

    mask = initial_mask.copy()

    loss_history = []

    for it in range(num_iter):

        # 正向成像
        aerial, soft_printed = simulate_soft_print(mask, psf)

        # printed 和 target 的误差
        error = soft_printed - target

        # 当前 loss
        loss = compute_loss(soft_printed, target)
        loss_history.append(loss)

        # 把误差反投影回 mask 空间
        # 因为 PSF 是对称高斯，这里仍然用同一个 psf 做卷积
        back_projected_error = fftconvolve(error, psf, mode='same')

        # 更新 mask
        # 如果 printed 太亮，error > 0，mask 应该减小
        # 如果 printed 太暗，error < 0，mask 应该增大
        mask = mask - learning_rate * back_projected_error

        # 限制 mask 值在 0 到 1 之间
        # 这相当于最简单的 mask range constraint
        mask = np.clip(mask, 0.0, 1.0)

    return mask, loss_history


# =========================================================
# 11. 主流程
# =========================================================
target = build_target_mask()
initial_mask = build_initial_mask()
psf = build_psf()

# 初始结果
aerial_initial, soft_initial = simulate_soft_print(initial_mask, psf)
printed_initial = hard_print(aerial_initial)
initial_loss = compute_loss(soft_initial, target)

# Toy ILT 优化
ilt_mask, loss_history = run_toy_ilt(
    initial_mask=initial_mask,
    target=target,
    psf=psf,
    num_iter=40,
    learning_rate=0.8
)

# ILT 后结果
aerial_ilt, soft_ilt = simulate_soft_print(ilt_mask, psf)
printed_ilt = hard_print(aerial_ilt)
ilt_loss = compute_loss(soft_ilt, target)

# 为了观察 mask 是否变成复杂灰度图，做一个二值化 mask
ilt_mask_binary = (ilt_mask >= 0.5).astype(float)


# =========================================================
# 12. 作图
# =========================================================
plt.figure(figsize=(16, 10))


# (1) target
plt.subplot(2, 3, 1)
plt.imshow(
    target,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Target Pattern")
plt.xlabel("x")
plt.ylabel("y")


# (2) initial mask
plt.subplot(2, 3, 2)
plt.imshow(
    initial_mask,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Initial Mask")
plt.xlabel("x")
plt.ylabel("y")


# (3) initial printed
plt.subplot(2, 3, 3)
plt.imshow(
    printed_initial,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Printed from Initial Mask")
plt.xlabel("x")
plt.ylabel("y")


# (4) ILT soft mask
plt.subplot(2, 3, 4)
plt.imshow(
    ilt_mask,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Toy ILT Soft Mask")
plt.xlabel("x")
plt.ylabel("y")


# (5) ILT printed
plt.subplot(2, 3, 5)
plt.imshow(
    printed_ilt,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Printed from Toy ILT Mask")
plt.xlabel("x")
plt.ylabel("y")


# (6) loss history
plt.subplot(2, 3, 6)
plt.plot(loss_history)
plt.title("ILT Loss History")
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()


# =========================================================
# 13. 第二张图：对比 soft mask 和 binary mask
# =========================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(
    ilt_mask,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("ILT Soft Mask")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 3, 2)
plt.imshow(
    ilt_mask_binary,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("ILT Binary Mask")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 3, 3)
plt.imshow(
    target,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Target")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()


# =========================================================
# 14. 输出总结
# =========================================================
print("Initial loss =", initial_loss)
print("Toy ILT loss =", ilt_loss)
print("Loss reduction =", initial_loss - ilt_loss)
print("Final mask min =", np.min(ilt_mask))
print("Final mask max =", np.max(ilt_mask))