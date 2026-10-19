import torch

# 1. 创建需要优化的二维变量
Z = torch.tensor([
    [0.0, 1.0],
    [2.0, 3.0]
], requires_grad=True)

# 2. 创建目标矩阵
target = torch.tensor([
    [1.0, 1.0],
    [1.0, 1.0]
])

# 3. 计算 Loss
loss = torch.sum((Z - target) ** 2)

# 4. 自动计算梯度
loss.backward()

# 5. 查看结果
print("Z =")
print(Z)

print("Loss =", loss.item())

print("Gradient =")
print(Z.grad)