import torch

# 1. 初始化二维变量
Z = torch.tensor([
    [0.0, 1.0],
    [2.0, 3.0]
], requires_grad=True)

target = torch.ones((2, 2))

# 2. 创建优化器
optimizer = torch.optim.SGD([Z], lr=0.1)

# 3. 初始 Loss
loss = torch.sum((Z - target) ** 2)

print("Initial Loss =", loss.item())

# 4. 连续进行三次优化
for step in range(3):

    # 清除旧梯度
    optimizer.zero_grad()

    # Forward
    loss = torch.sum((Z - target) ** 2)

    # Backward
    loss.backward()

    # 更新参数
    optimizer.step()

    # 查看更新后的 Loss
    with torch.no_grad():
        new_loss = torch.sum((Z - target) ** 2)

    print(f"Step {step+1}: Loss = {new_loss.item():.6f}")

# 5. 查看最终矩阵
print("Final Z =")
print(Z)