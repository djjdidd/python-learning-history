import torch

# 1. 初始化二维变量
Z = torch.tensor([
    [0.0, 1.0],
    [2.0, 3.0]
], requires_grad=True)

target = torch.ones((2, 2))

learning_rate = 0.1

# 2. 查看初始 Loss
loss = torch.sum((Z - target) ** 2)

print("Initial Loss =", loss.item())

# 3. 连续进行三次梯度下降
for step in range(3):

    # 清除上一次的梯度
    if Z.grad is not None:
       # Z.grad.zero_()
        pass
    # Forward
    loss = torch.sum((Z - target) ** 2)

    # Backward
    loss.backward()

    # 更新参数
    with torch.no_grad():
        Z -= learning_rate * Z.grad

        # 计算更新后的 Loss
        new_loss = torch.sum((Z - target) ** 2)

    print(f"Step {step+1}: Loss = {new_loss.item():.6f}")

# 4. 输出最终优化结果
print("Final Z =")
print(Z)