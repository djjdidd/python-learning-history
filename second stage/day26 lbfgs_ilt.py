import numpy as np
import matplotlib.pyplot as plt

from scipy.special import expit
from scipy.optimize import minimize


# ============================================================
# 1. Basic settings
# ============================================================

N = 128
L = 2.0
dx = L / N

wavelength = 0.193
NA = 0.85
cutoff = NA / wavelength

dose = 1.0
beta_resist = 20.0
threshold = 0.5

reference_peak = 1.0


# ============================================================
# 2. Spatial grid and target
# ============================================================

x = (
    np.arange(N)
    -
    N // 2
) * dx

X, Y = np.meshgrid(
    x,
    x
)

target = (
    (np.abs(X) <= 0.60 / 2)
    &
    (np.abs(Y) <= 0.40 / 2)
).astype(float)


# ============================================================
# 3. Frequency grid and pupil
# ============================================================

freq = np.fft.fftfreq(
    N,
    d=dx
)

FX, FY = np.meshgrid(
    freq,
    freq
)

pupil = (
    FX**2 + FY**2
    <= cutoff**2
).astype(float)


# ============================================================
# 4. Initial mask
# ============================================================

mask_initial = (
    0.1
    +
    0.8 * target
)

Z_initial = np.log(
    mask_initial
    /
    (1.0 - mask_initial)
)


# ============================================================
# 5. Forward + backward
# ============================================================

def compute_loss_and_gradient(Z):

    # --------------------------------------------------------
    # Z -> Mask
    # --------------------------------------------------------

    mask = expit(Z)

    # --------------------------------------------------------
    # Optical imaging
    # --------------------------------------------------------

    spectrum = np.fft.fft2(
        mask
    )

    field = np.fft.ifft2(
        spectrum
        *
        pupil
    )

    intensity = (
        np.abs(field)**2
    )

    # --------------------------------------------------------
    # Resist
    # --------------------------------------------------------

    effective_intensity = (
        dose
        *
        intensity
        /
        reference_peak
    )

    resist = expit(
        beta_resist
        *
        (
            effective_intensity
            -
            threshold
        )
    )

    # --------------------------------------------------------
    # Printing loss
    # --------------------------------------------------------

    loss = np.mean(
        (resist - target)**2
    )

    # ========================================================
    # Backward
    # ========================================================

    # Loss -> Resist
    dL_dR = (
        2.0
        *
        (resist - target)
        /
        resist.size
    )

    # Resist -> Intensity
    dL_dI = (
        dL_dR
        *
        beta_resist
        *
        resist
        *
        (1.0 - resist)
        *
        dose
        /
        reference_peak
    )

    # Intensity -> Field
    dL_dfield = (
        2.0
        *
        dL_dI
        *
        field
    )

    # Field -> Mask
    dL_dmask = np.real(
        np.fft.ifft2(
            np.fft.fft2(
                dL_dfield
            )
            *
            np.conj(pupil)
        )
    )

    # Mask -> Z
    gradient = (
        dL_dmask
        *
        mask
        *
        (1.0 - mask)
    )

    return loss, gradient


# ============================================================
# 6. History for L-BFGS
# ============================================================

lbfgs_history = {
    "loss": [],
    "gradient_norm": []
}


# ============================================================
# 7. Wrapper function for SciPy
# ============================================================

def objective_lbfgs(Z_flat):

    # SciPy gives us a 1D vector.
    # Convert it back to a 128 x 128 array.
    Z = Z_flat.reshape(
        N,
        N
    )

    # Our original ILT model
    loss, gradient = (
        compute_loss_and_gradient(Z)
    )

    # Record each function/gradient evaluation
    lbfgs_history[
        "loss"
    ].append(
        loss
    )

    lbfgs_history[
        "gradient_norm"
    ].append(
        np.linalg.norm(
            gradient
        )
    )

    # SciPy wants a 1D gradient
    gradient_flat = (
        gradient.ravel()
    )

    return (
        loss,
        gradient_flat
    )


# ============================================================
# 8. Run L-BFGS
# ============================================================

result_lbfgs = minimize(

    fun=objective_lbfgs,

    x0=Z_initial.ravel(),

    method="L-BFGS-B",

    jac=True,

    options={

        "maxiter": 150,

        "maxcor": 10,

        "ftol": 1e-12,

        "gtol": 1e-8
    }
)


# ============================================================
# 9. Read optimization result
# ============================================================

Z_lbfgs = (
    result_lbfgs.x.reshape(
        N,
        N
    )
)

mask_lbfgs = expit(
    Z_lbfgs
)

final_loss, final_gradient = (
    compute_loss_and_gradient(
        Z_lbfgs
    )
)

final_gradient_norm = (
    np.linalg.norm(
        final_gradient
    )
)


# ============================================================
# 10. Print result
# ============================================================

print(
    "\n"
    "============================================"
)

print(
    "L-BFGS Result"
)

print(
    "============================================"
)

print(
    "Success =",
    result_lbfgs.success
)

print(
    "Message =",
    result_lbfgs.message
)

print(
    "Initial Loss =",
    lbfgs_history["loss"][0]
)

print(
    "Final Loss =",
    final_loss
)

print(
    "Final Gradient Norm =",
    final_gradient_norm
)

print(
    "Iterations =",
    result_lbfgs.nit
)

print(
    "Function Evaluations =",
    result_lbfgs.nfev
)

print(
    "Gradient Evaluations =",
    result_lbfgs.njev
)


# ============================================================
# 11. Plot loss history
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    lbfgs_history["loss"]
)

plt.xlabel(
    "Function / Gradient Evaluation"
)

plt.ylabel(
    "Printing Loss"
)

plt.title(
    "L-BFGS Optimization"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 12. Plot gradient norm
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.semilogy(
    lbfgs_history[
        "gradient_norm"
    ]
)

plt.xlabel(
    "Function / Gradient Evaluation"
)

plt.ylabel(
    "Gradient Norm"
)

plt.title(
    "L-BFGS Gradient Norm"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 13. Show initial and optimized mask
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    mask_initial,
    origin="lower",
    extent=[
        x[0],
        x[-1],
        x[0],
        x[-1]
    ]
)

plt.xlabel("x (um)")
plt.ylabel("y (um)")

plt.title(
    "Initial Mask"
)

plt.colorbar()

plt.tight_layout()

plt.show()


plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    mask_lbfgs,
    origin="lower",
    extent=[
        x[0],
        x[-1],
        x[0],
        x[-1]
    ]
)

plt.xlabel("x (um)")
plt.ylabel("y (um)")

plt.title(
    "L-BFGS Optimized Mask"
)

plt.colorbar()

plt.tight_layout()

plt.show()