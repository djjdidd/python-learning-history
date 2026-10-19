import sys
import torch
import numpy as np

# 查看当前使用的 Python
print("Python path:", sys.executable)

# 查看安装的库
print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)

# 第一个 Tensor 实验
x = torch.tensor([1.0, 2.0, 3.0])

print("x =", x)
print("x * 2 =", x * 2)