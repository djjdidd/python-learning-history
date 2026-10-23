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
# 2. 初始化 Z
# =========================

Z = (4 * target - 2).clone().detach()

Z.requires_grad_(True)

# =========================
# 3. 构造 Pupil
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
# 4. 定义光学成像模型
# =========================

def forward_mask(M):

    F = torch.fft.fft2(M)

    E = torch.fft.ifft2(F * pupil)

    I = torch.abs(E) ** 2

    return I


# =========================
# 5. 路径 A：直接计算 Z 的梯度
# =========================

M = torch.sigmoid(Z)

I = forward_mask(M)

loss = torch.mean((I - target) ** 2)

loss.backward()

# 保存 Autograd 得到的梯度
grad_z_auto = Z.grad.detach().clone()


# =========================
# 6. 路径 B：独立计算 M 的梯度
# =========================

M_check = M.detach().clone().requires_grad_(True)

I_check = forward_mask(M_check)

loss_check = torch.mean(
    (I_check - target) ** 2
)

loss_check.backward()

grad_m = M_check.grad


# =========================
# 7. 手动应用 Sigmoid 链式法则
# =========================

sigmoid_derivative = M_check * (1 - M_check)

grad_z_chain = grad_m * sigmoid_derivative


# =========================
# 8. 比较结果
# =========================

difference = torch.max(
    torch.abs(grad_z_auto - grad_z_chain)
)

print("Initial Loss =", loss.item())

print("\nCenter Pixel [1,1]:")

print("Mask =", M_check[1, 1].item())

print(
    "Gradient of M =",
    grad_m[1, 1].item()
)

print(
    "Sigmoid Derivative =",
    sigmoid_derivative[1, 1].item()
)

print(
    "Gradient of Z (Chain Rule) =",
    grad_z_chain[1, 1].item()
)

print(
    "Gradient of Z (Autograd) =",
    grad_z_auto[1, 1].item()
)

print(
    "\nMaximum Gradient Difference =",
    difference.item()
)