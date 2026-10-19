import torch

target = torch.ones((2, 2))

# 比较普通 GD 和 Momentum
momentum_list = [0.0, 0.9]

for m in momentum_list:

    # 每组实验重新初始化
    Z = torch.tensor([
        [0.0, 1.0],
        [2.0, 3.0]
    ], requires_grad=True)

    optimizer = torch.optim.SGD(
        [Z],
        lr=0.1,
        momentum=m
    )

    print(f"\nMomentum = {m}")

    for step in range(5):

        # 1. 清除梯度
        optimizer.zero_grad()

        # 2. Forward
        loss = torch.sum((Z - target) ** 2)

        # 3. Backward
        loss.backward()

        # 4. 更新参数
        optimizer.step()

        # 5. 观察结果
        with torch.no_grad():
            new_loss = torch.sum((Z - target) ** 2)

            print(
                f"Step {step+1}: "
                f"Z[0,0] = {Z[0,0].item():.4f}, "
                f"Loss = {new_loss.item():.6f}"
            )