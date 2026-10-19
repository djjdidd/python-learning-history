
import matplotlib.pyplot as plt


# ==========================================
# 1. Define the loss function
# ==========================================

def loss_function(x):
    return (x - 3.0) ** 2


# ==========================================
# 2. Define the analytical gradient
# ==========================================

def gradient(x):
    return 2.0 * (x - 3.0)


# ==========================================
# 3. Gradient Descent
# ==========================================

def gradient_descent(x0, learning_rate, steps):

    x = x0

    loss_history = []

    for step in range(steps):

        # Record the current loss
        loss = loss_function(x)

        loss_history.append(loss)

        # Calculate the gradient
        grad = gradient(x)

        # Update x
        x = x - learning_rate * grad

    return x, loss_history


# ==========================================
# 4. Compare different learning rates
# ==========================================

learning_rates = [0.1, 0.5, 1.1]

steps = 12
x0 = 0.0

plt.figure(figsize=(8, 5))

for lr in learning_rates:

    final_x, history = gradient_descent(
        x0=x0,
        learning_rate=lr,
        steps=steps
    )

    plt.plot(
        range(steps),
        history,
        marker="o",
        label=f"Learning Rate = {lr}"
    )

    print(
        f"LR = {lr}, "
        f"Final x = {final_x:.6f}, "
        f"Final Loss = {loss_function(final_x):.6f}"
    )


# ==========================================
# 5. Plot the optimization curves
# ==========================================

plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.title("Gradient Descent: Learning Rate Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
