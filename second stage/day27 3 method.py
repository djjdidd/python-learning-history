import torch

# 目标矩阵
target = torch.ones((2, 2))

# 三种优化器
methods = ["GD", "Momentum", "Adam"]

for method in methods:

    # 每组实验重新初始化
    Z = torch.tensor([
        [0.0, 1.0],
        [2.0, 3.0]
    ], requires_grad=True)

    # 根据方法创建优化器
    if method == "GD":
        optimizer = torch.optim.SGD([Z], lr=0.1)

    elif method == "Momentum":
        optimizer = torch.optim.SGD(
            [Z], lr=0.1, momentum=0.9
        )

    else:
        optimizer = torch.optim.Adam([Z], lr=0.1)

    print(f"\nMethod = {method}")

    # 连续优化五次
    for step in range(5):

        optimizer.zero_grad()

        # Forward
        loss = torch.sum((Z - target) ** 2)

        # Backward
        loss.backward()

        # 更新参数
        optimizer.step()

        # 计算更新后的 Loss
        with torch.no_grad():
            new_loss = torch.sum((Z - target) ** 2)

        print(
            f"Step {step+1}: "
            f"Z[0,0] = {Z[0,0].item():.4f}, "
            f"Loss = {new_loss.item():.6f}"
        )