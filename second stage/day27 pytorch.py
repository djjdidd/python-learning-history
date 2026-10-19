import torch

# 1. 定义变量
x = torch.tensor(0.0, requires_grad=True)

# 2. 定义 Loss
loss = (x - 3) ** 2

# 3. 自动计算梯度
loss.backward()

# 4. 输出结果
print("x =", x.item())
print("Loss =", loss.item())
print("Gradient =", x.grad.item())

learning_rate = 0.1

new_x = x.item() - learning_rate * x.grad.item()

print("Updated x =", new_x)
print("New Loss =", (new_x - 3) ** 2)
