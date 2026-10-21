import numpy as np
import torch

# =========================
# 1. 定义 Mask
# =========================

mask_np = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]
], dtype=np.float32)

# Mask 是需要求梯度的变量
mask = torch.tensor(
    mask_np,
    requires_grad=True
)

# 目标图形（固定，不需要求梯度）
target = torch.tensor(mask_np.copy())


# =========================
# 2. 构造 Pupil
# =========================

N = 4
cutoff = 0.26

freq = np.fft.fftfreq(N)

FX, FY = np.meshgrid(
    freq, freq, indexing="ij"
)

pupil_np = (
    FX**2 + FY**2 <= cutoff**2
).astype(np.float32)

pupil = torch.tensor(pupil_np)


# =========================
# 3. Forward：光学成像
# =========================

F = torch.fft.fft2(mask)

E = torch.fft.ifft2(F * pupil)

I = torch.abs(E) ** 2


# =========================
# 4. 计算 Loss
# =========================

loss = torch.mean((I - target) ** 2)

print("Aerial Image:")
print(I)

print("\nLoss =", loss.item())


# =========================
# 5. Backward：自动计算梯度
# =========================

loss.backward()

print("\nMask Gradient:")
print(mask.grad)

# =========================
# 6. Finite Difference
# =========================

eps = 1e-4

# 复制两份 Mask
mask_plus = mask_np.astype(np.float64).copy()
mask_minus = mask_np.astype(np.float64).copy()

# 只修改第二行、第二列这个像素
mask_plus[1, 1] += eps
mask_minus[1, 1] -= eps

# 使用 NumPy 独立计算 Loss
def compute_loss(M):

    F = np.fft.fft2(M)

    E = np.fft.ifft2(F * pupil_np)

    I = np.abs(E) ** 2

    return np.mean((I - mask_np) ** 2)


# 分别计算两次 Loss
loss_plus = compute_loss(mask_plus)
loss_minus = compute_loss(mask_minus)

# 利用有限差分估算梯度
grad_fd = (loss_plus - loss_minus) / (2 * eps)

# 读取 PyTorch 自动计算的梯度
grad_auto = mask.grad[1, 1].item()

print("\nFinite Difference =", grad_fd)
print("Autograd =", grad_auto)

print("Difference =", abs(grad_fd - grad_auto))