import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic functions
# =========================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def soft_resist(aerial, threshold=0.5, beta=20):
    return 1.0 / (
        1.0 + np.exp(
            -beta * (aerial - threshold)
        )
    )


# =========================================================
# 2. Spatial grid
# =========================================================

N = 21

x = np.linspace(-1.0, 1.0, N)   # unit: μm
y = np.linspace(-1.0, 1.0, N)

X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]


# =========================================================
# 3. Target pattern
# =========================================================

target = (
    (np.abs(X) <= 0.30) &
    (np.abs(Y) <= 0.20)
).astype(float)


# =========================================================
# 4. Initial latent variable Z
# =========================================================

Z_initial = np.where(
    target == 1,
    2.0,
    -2.0
).astype(float)


# Initial continuous mask
mask_initial = sigmoid(
    Z_initial
)


# =========================================================
# 5. Frequency grid
# =========================================================

fx = np.fft.fftfreq(
    N,
    d=dx
)

fy = np.fft.fftfreq(
    N,
    d=dy
)

FX, FY = np.meshgrid(
    fx,
    fy
)


# =========================================================
# 6. Optical parameters
# =========================================================

wavelength = 0.193   # μm = 193 nm


# =========================================================
# 7. Build pupil
# =========================================================

def make_pupil(
    NA,
    wavelength=0.193
):

    cutoff = (
        NA / wavelength
    )

    frequency_radius = np.sqrt(
        FX**2 + FY**2
    )

    pupil = (
        frequency_radius <= cutoff
    ).astype(float)

    return pupil


# =========================================================
# 8. Fourier forward model
# =========================================================

def forward_fourier(
    Z,
    pupil,
    threshold=0.5,
    beta=20
):

    # Z -> continuous mask
    mask = sigmoid(Z)


    # Mask -> frequency domain
    spectrum = np.fft.fft2(
        mask
    )


    # Pupil filtering
    filtered_spectrum = (
        spectrum * pupil
    )


    # Back to spatial domain
    field = np.fft.ifft2(
        filtered_spectrum
    )


    # Complex field -> intensity
    aerial = (
        np.abs(field) ** 2
    )


    # Soft resist
    resist = soft_resist(
        aerial,
        threshold=threshold,
        beta=beta
    )


    return (
        mask,
        spectrum,
        filtered_spectrum,
        field,
        aerial,
        resist
    )


# =========================================================
# 9. Loss function
# =========================================================

def loss_fourier(
    Z,
    target,
    pupil,
    threshold=0.5,
    beta=20
):

    (
        mask,
        spectrum,
        filtered_spectrum,
        field,
        aerial,
        resist
    ) = forward_fourier(
        Z,
        pupil,
        threshold,
        beta
    )

    loss = np.mean(
        (resist - target) ** 2
    )

    return loss


# =========================================================
# 10. Backward propagation
# =========================================================

def backward_fourier(
    Z,
    target,
    pupil,
    threshold=0.5,
    beta=20
):

    # -----------------------------
    # Forward
    # -----------------------------

    (
        mask,
        spectrum,
        filtered_spectrum,
        field,
        aerial,
        resist
    ) = forward_fourier(
        Z,
        pupil,
        threshold,
        beta
    )


    # -----------------------------
    # Loss -> Resist
    # -----------------------------

    dL_dR = (
        2.0
        * (resist - target)
        / target.size
    )


    # -----------------------------
    # Resist -> Intensity
    # -----------------------------

    dR_dI = (
        beta
        * resist
        * (1.0 - resist)
    )

    dL_dI = (
        dL_dR
        * dR_dI
    )


    # -----------------------------
    # Intensity -> Complex field
    # -----------------------------

    dL_dE = (
        2.0
        * dL_dI
        * field
    )


    # -----------------------------
    # Optical adjoint
    # Field -> Mask
    # -----------------------------

    dL_dM_complex = np.fft.ifft2(
        np.conj(pupil)
        *
        np.fft.fft2(
            dL_dE
        )
    )

    dL_dM = np.real(
        dL_dM_complex
    )


    # -----------------------------
    # Mask -> Z
    # -----------------------------

    dM_dZ = (
        mask
        * (1.0 - mask)
    )

    dL_dZ = (
        dL_dM
        * dM_dZ
    )


    return dL_dZ


# =========================================================
# 11. ILT optimization function
# =========================================================

def run_ilt(
    Z_start,
    target,
    pupil,
    num_iterations=100,
    learning_rate=5.0
):

    Z_opt = Z_start.copy()

    loss_history = []


    for iteration in range(
        num_iterations
    ):

        # Current loss
        current_loss = loss_fourier(
            Z_opt,
            target,
            pupil
        )

        loss_history.append(
            current_loss
        )


        # Gradient
        grad = backward_fourier(
            Z_opt,
            target,
            pupil
        )


        # Gradient descent update
        Z_opt = (
            Z_opt
            -
            learning_rate
            * grad
        )


    # Final forward result
    (
        mask_final,
        spectrum_final,
        filtered_final,
        field_final,
        aerial_final,
        resist_final
    ) = forward_fourier(
        Z_opt,
        pupil
    )


    final_loss = loss_fourier(
        Z_opt,
        target,
        pupil
    )


    return (
        Z_opt,
        mask_final,
        aerial_final,
        resist_final,
        loss_history,
        final_loss
    )


# =========================================================
# 12. Build two pupils
# =========================================================

pupil_050 = make_pupil(
    NA=0.50
)

pupil_085 = make_pupil(
    NA=0.85
)


# =========================================================
# 13. Run ILT for NA = 0.50
# =========================================================

(
    Z_050,
    mask_050,
    aerial_050,
    resist_050,
    loss_050,
    final_loss_050
) = run_ilt(
    Z_initial,
    target,
    pupil_050,
    num_iterations=100,
    learning_rate=5.0
)


# =========================================================
# 14. Run ILT for NA = 0.85
# =========================================================

(
    Z_085,
    mask_085,
    aerial_085,
    resist_085,
    loss_085,
    final_loss_085
) = run_ilt(
    Z_initial,
    target,
    pupil_085,
    num_iterations=100,
    learning_rate=5.0
)


# =========================================================
# 15. Mask correction
# =========================================================

change_050 = (
    mask_050
    -
    mask_initial
)

change_085 = (
    mask_085
    -
    mask_initial
)


# =========================================================
# 16. Quantify correction amount
# =========================================================

mean_change_050 = np.mean(
    np.abs(change_050)
)

mean_change_085 = np.mean(
    np.abs(change_085)
)


max_change_050 = np.max(
    np.abs(change_050)
)

max_change_085 = np.max(
    np.abs(change_085)
)


# =========================================================
# 17. Print results
# =========================================================

print()
print("=" * 60)

print("NA = 0.50")
print(
    "Initial loss =",
    loss_050[0]
)

print(
    "Final loss   =",
    final_loss_050
)

print(
    "Mean absolute mask change =",
    mean_change_050
)

print(
    "Max absolute mask change  =",
    max_change_050
)


print()

print("NA = 0.85")
print(
    "Initial loss =",
    loss_085[0]
)

print(
    "Final loss   =",
    final_loss_085
)

print(
    "Mean absolute mask change =",
    mean_change_085
)

print(
    "Max absolute mask change  =",
    max_change_085
)

print("=" * 60)


# =========================================================
# 18. Plot comparison
# =========================================================

plt.figure(
    figsize=(14, 10)
)


# -----------------------------------------
# Target
# -----------------------------------------

plt.subplot(
    3,
    4,
    1
)

plt.imshow(
    target,
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Target"
)

plt.colorbar()


# -----------------------------------------
# Initial mask
# -----------------------------------------

plt.subplot(
    3,
    4,
    2
)

plt.imshow(
    mask_initial,
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Initial Mask"
)

plt.colorbar()


# -----------------------------------------
# Pupil NA = 0.50
# -----------------------------------------

plt.subplot(
    3,
    4,
    3
)

plt.imshow(
    np.fft.fftshift(
        pupil_050
    ),
    origin="lower",
    cmap="gray"
)

plt.title(
    "Pupil, NA=0.50"
)

plt.colorbar()


# -----------------------------------------
# Pupil NA = 0.85
# -----------------------------------------

plt.subplot(
    3,
    4,
    4
)

plt.imshow(
    np.fft.fftshift(
        pupil_085
    ),
    origin="lower",
    cmap="gray"
)

plt.title(
    "Pupil, NA=0.85"
)

plt.colorbar()


# -----------------------------------------
# Optimized mask NA = 0.50
# -----------------------------------------

plt.subplot(
    3,
    4,
    5
)

plt.imshow(
    mask_050,
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Optimized Mask, NA=0.50"
)

plt.colorbar()


# -----------------------------------------
# Optimized mask NA = 0.85
# -----------------------------------------

plt.subplot(
    3,
    4,
    6
)

plt.imshow(
    mask_085,
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Optimized Mask, NA=0.85"
)

plt.colorbar()


# -----------------------------------------
# Mask correction NA = 0.50
# -----------------------------------------

plt.subplot(
    3,
    4,
    7
)

plt.imshow(
    change_050,
    origin="lower"
)

plt.title(
    "Mask Change, NA=0.50"
)

plt.colorbar()


# -----------------------------------------
# Mask correction NA = 0.85
# -----------------------------------------

plt.subplot(
    3,
    4,
    8
)

plt.imshow(
    change_085,
    origin="lower"
)

plt.title(
    "Mask Change, NA=0.85"
)

plt.colorbar()


# -----------------------------------------
# Resist NA = 0.50
# -----------------------------------------

plt.subplot(
    3,
    4,
    9
)

plt.imshow(
    resist_050,
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Resist, NA=0.50"
)

plt.colorbar()


# -----------------------------------------
# Resist NA = 0.85
# -----------------------------------------

plt.subplot(
    3,
    4,
    10
)

plt.imshow(
    resist_085,
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Resist, NA=0.85"
)

plt.colorbar()


# -----------------------------------------
# Loss curves
# -----------------------------------------

plt.subplot(
    3,
    4,
    11
)

plt.plot(
    loss_050,
    label="NA=0.50"
)

plt.plot(
    loss_085,
    label="NA=0.85"
)

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "Print Loss"
)

plt.title(
    "Loss Comparison"
)

plt.legend()

plt.grid()


# -----------------------------------------
# Correction magnitude comparison
# -----------------------------------------

plt.subplot(
    3,
    4,
    12
)

labels = [
    "NA=0.50",
    "NA=0.85"
]

values = [
    mean_change_050,
    mean_change_085
]

plt.bar(
    labels,
    values
)

plt.ylabel(
    "Mean |Mask Change|"
)

plt.title(
    "Correction Magnitude"
)

plt.grid(
    axis="y"
)


plt.tight_layout()

plt.show()