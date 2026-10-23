import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic grid
# =========================================================

N = 128
L = 2.0
dx = L / N

x = np.arange(N) * dx - L / 2
y = np.arange(N) * dx - L / 2

X, Y = np.meshgrid(x, y)


# =========================================================
# 2. Target pattern
# =========================================================

target_width_x = 0.60
target_width_y = 0.40

target = (
    (np.abs(X) <= target_width_x / 2)
    &
    (np.abs(Y) <= target_width_y / 2)
).astype(float)


# =========================================================
# 3. Initial mask -> Z
# =========================================================

initial_mask = target.copy()

eps = 0.1

mask_clipped = np.clip(
    initial_mask,
    eps,
    1 - eps
)

Z = np.log(
    mask_clipped
    /
    (1 - mask_clipped)
)


# =========================================================
# 4. Sigmoid
# =========================================================

def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))


# =========================================================
# 5. Optical parameters
# =========================================================

wavelength = 0.193
NA = 0.85
cutoff = NA / wavelength

freq = np.fft.fftfreq(N, d=dx)
FX, FY = np.meshgrid(freq, freq)

pupil = (
    np.sqrt(FX**2 + FY**2) <= cutoff
).astype(float)


# =========================================================
# 6. Optical forward model
# =========================================================

def optical_forward(mask_input):

    spectrum = np.fft.fft2(mask_input)

    filtered_spectrum = spectrum * pupil

    field = np.fft.ifft2(filtered_spectrum)

    intensity = np.abs(field) ** 2

    return intensity, field


# =========================================================
# 7. Reference normalization
# =========================================================

reference_intensity, _ = optical_forward(target)
reference_peak = np.max(reference_intensity)


# =========================================================
# 8. Resist model
# =========================================================

threshold = 0.5
beta = 20.0
dose = 1.0

def soft_resist(intensity):

    return 1.0 / (
        1.0 + np.exp(-beta * (intensity - threshold))
    )


# =========================================================
# 9. Loss
# =========================================================

def printing_loss(resist, target_pattern):

    return np.mean((resist - target_pattern) ** 2)


# =========================================================
# 10. Forward pass
# =========================================================

def forward_pass(Z_input):

    mask = sigmoid(Z_input)

    intensity, field = optical_forward(mask)

    intensity_norm = intensity / reference_peak

    effective_intensity = dose * intensity_norm

    resist = soft_resist(effective_intensity)

    loss = printing_loss(resist, target)

    return loss, mask, intensity, field, resist


# =========================================================
# 11. Backward pass
# =========================================================

def backward_pass(Z_input):

    loss, mask, intensity, field, resist = forward_pass(Z_input)

    # dL/dR
    dL_dR = 2.0 * (resist - target) / resist.size

    # dR/dIeff
    dR_dIeff = beta * resist * (1.0 - resist)

    # dL/dIeff
    dL_dIeff = dL_dR * dR_dIeff

    # dL/dI
    dL_dI = dL_dIeff * dose / reference_peak

    # dL/dE
    dL_dE = 2.0 * dL_dI * field

    # dL/dM
    dL_dM_complex = np.fft.ifft2(
        np.fft.fft2(dL_dE) * np.conj(pupil)
    )

    dL_dM = np.real(dL_dM_complex)

    # dM/dZ
    dM_dZ = mask * (1.0 - mask)

    # dL/dZ
    dL_dZ = dL_dM * dM_dZ

    return loss, dL_dZ, mask, resist


# =========================================================
# 12. Save initial result
# =========================================================

loss_initial, mask_initial, intensity_initial, field_initial, resist_initial = forward_pass(Z)


# =========================================================
# 13. Optimization loop
# =========================================================

learning_rate = 10.0
num_iters = 100

loss_history = []

for it in range(num_iters):

    loss, dL_dZ, mask_now, resist_now = backward_pass(Z)

    loss_history.append(loss)

    Z = Z - learning_rate * dL_dZ

    if it % 10 == 0 or it == num_iters - 1:
        print(
            f"Iteration {it:3d}: loss = {loss:.10f}"
        )


# =========================================================
# 14. Final result
# =========================================================

loss_final, mask_final, intensity_final, field_final, resist_final = forward_pass(Z)

print()
print("=" * 60)
print("Final Result")
print("=" * 60)
print("Initial loss =", loss_initial)
print("Final loss   =", loss_final)


# =========================================================
# 15. Plot loss history
# =========================================================

plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Printing Loss")
plt.title("Loss History")
plt.grid(True)
plt.show()


# =========================================================
# 16. Plot target
# =========================================================

plt.figure(figsize=(5, 4))
plt.imshow(
    target,
    extent=[x[0], x[-1], y[0], y[-1]],
    origin="lower"
)
plt.title("Target")
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.colorbar()
plt.show()


# =========================================================
# 17. Plot initial mask
# =========================================================

plt.figure(figsize=(5, 4))
plt.imshow(
    mask_initial,
    extent=[x[0], x[-1], y[0], y[-1]],
    origin="lower"
)
plt.title("Initial Mask")
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.colorbar()
plt.show()


# =========================================================
# 18. Plot final mask
# =========================================================

plt.figure(figsize=(5, 4))
plt.imshow(
    mask_final,
    extent=[x[0], x[-1], y[0], y[-1]],
    origin="lower"
)
plt.title("Final Mask")
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.colorbar()
plt.show()


# =========================================================
# 19. Plot initial resist
# =========================================================

plt.figure(figsize=(5, 4))
plt.imshow(
    resist_initial,
    extent=[x[0], x[-1], y[0], y[-1]],
    origin="lower"
)
plt.title("Initial Resist")
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.colorbar()
plt.show()


# =========================================================
# 20. Plot final resist
# =========================================================

plt.figure(figsize=(5, 4))
plt.imshow(
    resist_final,
    extent=[x[0], x[-1], y[0], y[-1]],
    origin="lower"
)
plt.title("Final Resist")
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.colorbar()
plt.show()