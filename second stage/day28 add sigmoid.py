import numpy as np
import torch

# =========================
# 1. 定义 Target
# =========================

target = torch.tensor([
    [0., 0., 0., 0.],
    [0., 1., 1., 0.],
    [0., 1., 1., 0.],
    [0., 0., 0., 0.]
])


# =========================
# 2. 初始化优化变量 Z
# =========================

Z = torch.tensor([
    [-2., -2., -2., -2.],
    [-2.,  2.,  2., -2.],
    [-2.,  2.,  2., -2.],
    [-2., -2., -2., -2.]
], requires_grad=True)


# =========================
# 3. 构造 Pupil
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
# 4. 定义 Forward Model
# =========================

def forward(Z):

    # Z 先转换为 Mask
    M = torch.sigmoid(Z)

    # 光学成像
    F = torch.fft.fft2(M)

    E = torch.fft.ifft2(F * pupil)

    I = torch.abs(E) ** 2

    return I


# =========================
# 5. 创建优化器
# =========================

optimizer = torch.optim.SGD([Z], lr=1.0)


# =========================
# 6. 优化前
# =========================

M_initial = torch.sigmoid(Z)

I = forward(Z)

loss = torch.mean((I - target) ** 2)

print("Initial Mask:")
print(M_initial.detach())

print("\nInitial Loss =", loss.item())


# =========================
# 7. 自动求导并更新 Z
# =========================

optimizer.zero_grad()

loss.backward()

print("\nGradient of Z:")
print(Z.grad)

optimizer.step()


# =========================
# 8. 更新后
# =========================

with torch.no_grad():

    M_new = torch.sigmoid(Z)

    I_new = forward(Z)

    new_loss = torch.mean((I_new - target) ** 2)

print("\nUpdated Mask:")
print(M_new)

print("\nNew Loss =", new_loss.item())