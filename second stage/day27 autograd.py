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