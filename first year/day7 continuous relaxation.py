from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

# --------------------------------------------------
# 1. Build 2D coordinate grid
# --------------------------------------------------
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

# --------------------------------------------------
# 2. Define target rectangle
# --------------------------------------------------
width_x = 0.6
width_y = 0.4

mask_2d = ((np.abs(X) <= width_x / 2) &
           (np.abs(Y) <= width_y / 2)).astype(float)

target = mask_2d.copy()

# --------------------------------------------------
# 3. Define PSF
# --------------------------------------------------
sigma = 0.05
psf_2d = np.exp(-((X ** 2 + Y ** 2) / (2 * sigma ** 2)))
psf_2d /= np.sum(psf_2d)

threshold = 0.5

# --------------------------------------------------
# 4. Soft resist model
# --------------------------------------------------
def soft_resist(aerial, threshold, beta=30):
    return 1 / (1 + np.exp(-beta * (aerial - threshold)))

# --------------------------------------------------
# 5. Serif template
# --------------------------------------------------
serif_size = 0.10

serif_template = (
    ((np.abs(X - 0.3) <= serif_size / 2) &
     (np.abs(Y - 0.2) <= serif_size / 2))
    |
    ((np.abs(X + 0.3) <= serif_size / 2) &
     (np.abs(Y - 0.2) <= serif_size / 2))
    |
    ((np.abs(X - 0.3) <= serif_size / 2) &
     (np.abs(Y + 0.2) <= serif_size / 2))
    |
    ((np.abs(X + 0.3) <= serif_size / 2) &
     (np.abs(Y + 0.2) <= serif_size / 2))
).astype(float)

# --------------------------------------------------
# 6. Continuous parameterization
# --------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def make_continuous_mask(z):
    alpha = sigmoid(z)
    mask = mask_2d + alpha * serif_template
    mask = np.clip(mask, 0.0, 1.0)
    return mask

# --------------------------------------------------
# 7. Objective function
# --------------------------------------------------
def continuous_objective(z_array):
    z = z_array[0]

    mask = make_continuous_mask(z)

    aerial = signal.fftconvolve(mask, psf_2d, mode="same")
    resist = soft_resist(aerial, threshold, beta=30)

    loss = np.mean((resist - target) ** 2)
    return loss

# --------------------------------------------------
# 8. Numerical gradient
# --------------------------------------------------
def gradient_z(z, h=0.1):
    loss_plus = continuous_objective(np.array([z + h]))
    loss_minus = continuous_objective(np.array([z - h]))
    return (loss_plus - loss_minus) / (2 * h)

# --------------------------------------------------
# 9. Gradient descent
# --------------------------------------------------
z = 0.0
learning_rate = 1.0

z_history = []
alpha_history = []
loss_history = []

for i in range(20):
    grad = gradient_z(z)
    z = z - learning_rate * grad
    z = np.clip(z, -10, 10)

    loss = continuous_objective(np.array([z]))
    alpha = sigmoid(z)

    z_history.append(z)
    alpha_history.append(alpha)
    loss_history.append(loss)

    print(i, "z =", z, "alpha =", alpha, "loss =", loss, "gradient =", grad)

# --------------------------------------------------
# 10. Plot results
# --------------------------------------------------
optimized_mask = make_continuous_mask(z)
optimized_aerial = signal.fftconvolve(optimized_mask, psf_2d, mode="same")
optimized_resist = soft_resist(optimized_aerial, threshold, beta=30)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(mask_2d, cmap="gray", origin="lower")
plt.title("Original Mask")

plt.subplot(1, 3, 2)
plt.imshow(optimized_mask, cmap="gray", origin="lower")
plt.title("Optimized Continuous Mask")

plt.subplot(1, 3, 3)
plt.imshow(optimized_resist, cmap="gray", origin="lower")
plt.title("Resulting Resist")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 11. Plot optimization history
# --------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(loss_history, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss during Gradient Descent")
plt.grid(True)
plt.show()