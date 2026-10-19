import torch

# 1. 初始化二维变量
Z = torch.tensor([
    [0.0, 1.0],
    [2.0, 3.0]
], requires_grad=True)

target = torch.ones((2, 2))

# 2. Forward
loss = torch.sum((Z - target) ** 2)

print("Initial Loss =", loss.item())

# 3. Backward
loss.backward()

print("Gradient =")
print(Z.grad)

# 4. Gradient Descent
learning_rate = 0.1

with torch.no_grad():
    Z -= learning_rate * Z.grad

# 5. 查看更新结果
new_loss = torch.sum((Z - target) ** 2)

print("Updated Z =")
print(Z)

print("New Loss =", new_loss.item())