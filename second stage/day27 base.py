import torch

# 1. 创建需要求导的变量
x = torch.tensor(0.0, requires_grad=True)

# 2. 将函数拆成两步
a = x - 3

loss = a ** 2

# 3. 观察计算图
print("x.grad_fn =", x.grad_fn)
print("a.grad_fn =", a.grad_fn)
print("loss.grad_fn =", loss.grad_fn)

# 4. 反向传播
loss.backward()

print("Gradient =", x.grad.item())