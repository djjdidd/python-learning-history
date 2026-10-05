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
# 3. Initial mask
# =========================================================

initial_mask = target.copy()


# =========================================================
# 4. Convert mask to optimization variable Z
# =========================================================

eps = 0.01

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
# 5. Sigmoid
# =========================================================

def sigmoid(z):

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(-z)
        )
    )


# =========================================================
# 6. Optical parameters
# =========================================================

wavelength = 0.193
NA = 0.85

cutoff = NA / wavelength


# =========================================================
# 7. Frequency grid
# =========================================================

freq = np.fft.fftfreq(
    N,
    d=dx
)

FX, FY = np.meshgrid(
    freq,
    freq
)


# =========================================================
# 8. Circular pupil
# =========================================================

pupil = (
    np.sqrt(
        FX ** 2
        +
        FY ** 2
    )
    <= cutoff
).astype(float)


# =========================================================
# 9. Optical forward model
# =========================================================

def optical_forward(
    mask_input
):

    spectrum = np.fft.fft2(
        mask_input
    )

    filtered_spectrum = (
        spectrum
        *
        pupil
    )

    field = np.fft.ifft2(
        filtered_spectrum
    )

    intensity = (
        np.abs(field) ** 2
    )

    return (
        intensity,
        field
    )


# =========================================================
# 10. Fixed reference intensity
# =========================================================

reference_intensity, _ = optical_forward(
    target
)

reference_peak = np.max(
    reference_intensity
)


# =========================================================
# 11. Soft resist
# =========================================================

threshold = 0.5
beta = 20.0
dose = 1.0


def soft_resist(
    intensity
):

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -beta
                *
                (
                    intensity
                    -
                    threshold
                )
            )
        )
    )


# =========================================================
# 12. Printing loss
# =========================================================

def printing_loss(
    resist,
    target_pattern
):

    return np.mean(
        (
            resist
            -
            target_pattern
        ) ** 2
    )


# =========================================================
# 13. Complete forward pass
# =========================================================

def forward_pass(
    Z_input
):

    # Z -> mask
    mask = sigmoid(
        Z_input
    )

    # mask -> aerial image
    intensity, field = optical_forward(
        mask
    )

    # fixed normalization
    intensity_norm = (
        intensity
        /
        reference_peak
    )

    # dose
    effective_intensity = (
        dose
        *
        intensity_norm
    )

    # resist
    resist = soft_resist(
        effective_intensity
    )

    # loss
    loss = printing_loss(
        resist,
        target
    )

    return (
        loss,
        mask,
        intensity,
        field,
        resist
    )


# =========================================================
# 14. Initial forward pass
# =========================================================

loss_before, mask, intensity, field, resist = forward_pass(
    Z
)


print()
print("=" * 60)
print("Before Gradient Update")
print("=" * 60)

print(
    "Printing loss =",
    loss_before
)


# =========================================================
# 15. dL / dR
# =========================================================

dL_dR = (
    2.0
    *
    (
        resist
        -
        target
    )
    /
    resist.size
)


# =========================================================
# 16. dR / dI_effective
# =========================================================

dR_dIeff = (
    beta
    *
    resist
    *
    (
        1.0
        -
        resist
    )
)


# =========================================================
# 17. dL / dI_effective
# =========================================================

dL_dIeff = (
    dL_dR
    *
    dR_dIeff
)


# =========================================================
# 18. dL / dI
# =========================================================
# I_effective =
# dose * intensity / reference_peak

dL_dI = (
    dL_dIeff
    *
    dose
    /
    reference_peak
)


# =========================================================
# 19. dL / dE
# =========================================================
# intensity = |E|^2

dL_dE = (
    2.0
    *
    dL_dI
    *
    field
)


# =========================================================
# 20. Optical adjoint
# dL/dE -> dL/dM
# =========================================================

dL_dM_complex = np.fft.ifft2(
    np.fft.fft2(
        dL_dE
    )
    *
    np.conj(
        pupil
    )
)


dL_dM = np.real(
    dL_dM_complex
)


# =========================================================
# 21. Sigmoid gradient
# dL/dM -> dL/dZ
# =========================================================

dM_dZ = (
    mask
    *
    (
        1.0
        -
        mask
    )
)


dL_dZ = (
    dL_dM
    *
    dM_dZ
)


print()
print("=" * 60)
print("Gradient Information")
print("=" * 60)

print(
    "dL/dZ shape =",
    dL_dZ.shape
)

print(
    "dL/dZ max =",
    np.max(dL_dZ)
)

print(
    "dL/dZ min =",
    np.min(dL_dZ)
)


# =========================================================
# 22. One gradient descent step
# =========================================================

learning_rate = 10.0


Z_new = (
    Z
    -
    learning_rate
    *
    dL_dZ
)


# =========================================================
# 23. Forward pass after update
# =========================================================

loss_after, mask_after, intensity_after, field_after, resist_after = forward_pass(
    Z_new
)


print()
print("=" * 60)
print("After One Gradient Update")
print("=" * 60)

print(
    "Loss before =",
    loss_before
)

print(
    "Loss after =",
    loss_after
)

print(
    "Loss change =",
    loss_after - loss_before
)


# =========================================================
# 24. Target
# =========================================================

plt.figure(
    figsize=(5, 4)
)

plt.imshow(
    target,
    extent=[
        x[0],
        x[-1],
        y[0],
        y[-1]
    ],
    origin="lower"
)

plt.title(
    "Target"
)

plt.xlabel("x (um)")
plt.ylabel("y (um)")

plt.colorbar()

plt.show()


# =========================================================
# 25. Initial mask
# =========================================================

plt.figure(
    figsize=(5, 4)
)

plt.imshow(
    mask,
    extent=[
        x[0],
        x[-1],
        y[0],
        y[-1]
    ],
    origin="lower"
)

plt.title(
    "Mask Before Update"
)

plt.xlabel("x (um)")
plt.ylabel("y (um)")

plt.colorbar()

plt.show()


# =========================================================
# 26. Mask after one update
# =========================================================

plt.figure(
    figsize=(5, 4)
)

plt.imshow(
    mask_after,
    extent=[
        x[0],
        x[-1],
        y[0],
        y[-1]
    ],
    origin="lower"
)

plt.title(
    "Mask After One Update"
)

plt.xlabel("x (um)")
plt.ylabel("y (um)")

plt.colorbar()

plt.show()


# =========================================================
# 27. Resist before update
# =========================================================

plt.figure(
    figsize=(5, 4)
)

plt.imshow(
    resist,
    extent=[
        x[0],
        x[-1],
        y[0],
        y[-1]
    ],
    origin="lower"
)

plt.title(
    "Printed Resist Before Update"
)

plt.xlabel("x (um)")
plt.ylabel("y (um)")

plt.colorbar()

plt.show()


# =========================================================
# 28. Resist after update
# =========================================================

plt.figure(
    figsize=(5, 4)
)

plt.imshow(
    resist_after,
    extent=[
        x[0],
        x[-1],
        y[0],
        y[-1]
    ],
    origin="lower"
)

plt.title(
    "Printed Resist After One Update"
)

plt.xlabel("x (um)")
plt.ylabel("y (um)")

plt.colorbar()

plt.show()


# =========================================================
# 29. Gradient map
# =========================================================

plt.figure(
    figsize=(5, 4)
)

plt.imshow(
    dL_dZ,
    extent=[
        x[0],
        x[-1],
        y[0],
        y[-1]
    ],
    origin="lower"
)

plt.title(
    "dL/dZ"
)

plt.xlabel("x (um)")
plt.ylabel("y (um)")

plt.colorbar()

plt.show()