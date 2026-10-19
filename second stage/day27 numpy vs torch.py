import numpy as np
import torch

# 1. 定义二维 Mask
mask_np = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]
], dtype=np.float32)

# 2. 转换为 PyTorch Tensor
mask_torch = torch.tensor(mask_np)

# 3. NumPy 二维 FFT
F_numpy = np.fft.fft2(mask_np)

# 4. PyTorch 二维 FFT
F_torch = torch.fft.fft2(mask_torch)

# 5. 比较结果
print("NumPy FFT amplitude:")
print(np.abs(F_numpy))

print("\nPyTorch FFT amplitude:")
print(torch.abs(F_torch))

# 6. 计算两者的最大误差
difference = np.max(
    np.abs(F_numpy - F_torch.numpy())
)

print("\nMax Difference =", difference)