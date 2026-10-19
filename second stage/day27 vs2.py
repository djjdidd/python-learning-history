import numpy as np
import torch

# 1. 定义二维 Mask
mask_np = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]
], dtype=np.float32)

mask_torch = torch.tensor(mask_np)

# 2. 构造 Pupil
N = 4
cutoff = 0.26

freq = np.fft.fftfreq(N)

FX, FY = np.meshgrid(freq, freq, indexing="ij")

pupil_np = (
    FX**2 + FY**2 <= cutoff**2
).astype(np.float32)

# 转换为 PyTorch Tensor
pupil_torch = torch.tensor(pupil_np)

# 3. NumPy 光学成像
F_np = np.fft.fft2(mask_np)

E_np = np.fft.ifft2(F_np * pupil_np)

I_np = np.abs(E_np) ** 2

# 4. PyTorch 光学成像
F_torch = torch.fft.fft2(mask_torch)

E_torch = torch.fft.ifft2(F_torch * pupil_torch)

I_torch = torch.abs(E_torch) ** 2

# 5. 比较结果
print("Pupil:")
print(pupil_np)

print("\nNumPy Aerial Image:")
print(I_np)

print("\nPyTorch Aerial Image:")
print(I_torch)

difference = np.max(
    np.abs(I_np - I_torch.numpy())
)

print("\nMax Difference =", difference)