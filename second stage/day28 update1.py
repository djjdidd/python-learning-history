import numpy as np
import torch

# =========================
# 1. 初始化 Mask 和 Target
# =========================

mask_np = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]
], dtype=np.float32)

mask = torch.tensor(
    mask_np,
    requires_grad=True
)

target = torch.tensor(mask_np.copy())


# =========================
# 2. 构造 Pupil
# =========================

N = 4
cutoff = 0.26

freq = np.fft.fftfreq(N)

FX, FY = np.meshgrid(freq, freq, indexing="ij")

pupil_np = (
    FX**2 + FY**2 <= cutoff**2
).astype(np.float32)

pupil = torch.tensor(pupil_np)


# =========================
# 3. Forward
# =========================

F = torch.fft.fft2(mask)

E = torch.fft.ifft2(F * pupil)

I = torch.abs(E) ** 2

loss = torch.mean((I - target) ** 2)

print("Initial Loss =", loss.item())


# =========================
# 4. Backward
# =========================

loss.backward()

print("\nMask Gradient:")
print(mask.grad)


# =========================
# 5. 更新一次 Mask
# =========================

learning_rate = 0.5

with torch.no_grad():

    # 梯度下降
    mask -= learning_rate * mask.grad

    # 限制 Mask 在 [0, 1] 范围内
    mask.clamp_(0.0, 1.0)

print("\nUpdated Mask:")
print(mask)


# =========================
# 6. 重新计算 Loss
# =========================

with torch.no_grad():

    F_new = torch.fft.fft2(mask)

    E_new = torch.fft.ifft2(F_new * pupil)

    I_new = torch.abs(E_new) ** 2

    new_loss = torch.mean((I_new - target) ** 2)

print("\nNew Loss =", new_loss.item())