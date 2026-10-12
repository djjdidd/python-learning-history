import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Spatial grid
# =========================================================

N = 128
L = 2.0                     # um
dx = L / N

x = np.arange(N) * dx - L / 2
y = np.arange(N) * dx - L / 2

X, Y = np.meshgrid(x, y)


# =========================================================
# 2. Target
# =========================================================

target = (
    (np.abs(X) <= 0.60 / 2)
    &
    (np.abs(Y) <= 0.40 / 2)
).astype(float)


# =========================================================
# 3. Fixed mask
# =========================================================
# For this stage we ONLY optimize the source.
#
# The mask is fixed.
# Use a continuous target-like mask first.
# =========================================================

eps = 0.10

mask_fixed = np.clip(
    target,
    eps,
    1.0 - eps
)


# =========================================================
# 4. Optical parameters
# =========================================================

wavelength = 0.193            # um
NA = 0.85

cutoff = NA / wavelength


freq = np.fft.fftfreq(
    N,
    d=dx
)

FX, FY = np.meshgrid(
    freq,
    freq
)


# =========================================================
# 5. Process conditions
# =========================================================

focus_list = np.array([
    -0.05,
     0.00,
    +0.05
])

dose_list = np.array([
    0.95,
    1.00,
    1.05
])


# =========================================================
# 6. Soft pupil parameters
# =========================================================
# alpha controls how sharp the pupil edge is.
#
# Larger alpha -> more like hard pupil
# Smaller alpha -> smoother
# =========================================================

alpha = 20.0
small_number = 1e-12


# =========================================================
# 7. Soft complex pupil + gradient
# =========================================================

def soft_pupil_and_gradient(
    source_x,
    source_y,
    focus
):

    # -----------------------------------------------------
    # Shifted frequency coordinates
    # -----------------------------------------------------

    fx = FX - source_x
    fy = FY - source_y


    # -----------------------------------------------------
    # Radius
    # -----------------------------------------------------

    r = np.sqrt(
        fx**2
        +
        fy**2
        +
        small_number
    )


    # =====================================================
    # Soft aperture
    #
    # A =
    # 1 / [1 + exp(alpha * (r - cutoff))]
    # =====================================================

    argument = (
        alpha
        *
        (
            r - cutoff
        )
    )


    # Numerical protection
    argument = np.clip(
        argument,
        -60.0,
        60.0
    )


    aperture = (
        1.0
        /
        (
            1.0
            +
            np.exp(argument)
        )
    )


    # =====================================================
    # dA / d(source_x)
    # =====================================================

    dA_dsx = (
        alpha
        *
        aperture
        *
        (
            1.0 - aperture
        )
        *
        fx
        /
        r
    )


    # =====================================================
    # Defocus phase
    # =====================================================

    q = (
        fx**2
        +
        fy**2
    )


    phase = np.exp(
        -1j
        *
        np.pi
        *
        wavelength
        *
        focus
        *
        q
    )


    # =====================================================
    # dPhase / d(source_x)
    # =====================================================

    dphase_dsx = (
        1j
        *
        2.0
        *
        np.pi
        *
        wavelength
        *
        focus
        *
        fx
        *
        phase
    )


    # =====================================================
    # Complex pupil
    # =====================================================

    pupil = (
        aperture
        *
        phase
    )


    # =====================================================
    # dP / d(source_x)
    # =====================================================

    dpupil_dsx = (
        dA_dsx
        *
        phase
        +
        aperture
        *
        dphase_dsx
    )


    return (
        pupil,
        dpupil_dsx
    )


# =========================================================
# 8. Partial coherent imaging
#    + source-shift derivative
# =========================================================
#
# Symmetric dipole:
#
# source 1 = (+s, 0)
# source 2 = (-s, 0)
#
# Both weights = 0.5
# =========================================================

def optical_forward_with_source_gradient(
    mask,
    focus,
    source_shift
):

    spectrum = np.fft.fft2(
        mask
    )


    total_intensity = np.zeros_like(
        mask,
        dtype=float
    )


    dI_ds = np.zeros_like(
        mask,
        dtype=float
    )


    source_info = [

        (
            +source_shift,
            0.0,
            0.5,
            +1.0
        ),

        (
            -source_shift,
            0.0,
            0.5,
            -1.0
        )
    ]


    for (
        source_x,
        source_y,
        source_weight,
        source_sign
    ) in source_info:


        # -------------------------------------------------
        # P and dP/d(source_x)
        # -------------------------------------------------

        (
            pupil,
            dpupil_dsx
        ) = soft_pupil_and_gradient(
            source_x,
            source_y,
            focus
        )


        # -------------------------------------------------
        # Because source_x = +/- source_shift
        #
        # dP/ds =
        #
        # dP/d(source_x)
        # *
        # d(source_x)/ds
        # -------------------------------------------------

        dpupil_ds = (
            dpupil_dsx
            *
            source_sign
        )


        # -------------------------------------------------
        # Field
        # -------------------------------------------------

        field = np.fft.ifft2(
            spectrum
            *
            pupil
        )


        # -------------------------------------------------
        # dE / ds
        # -------------------------------------------------

        dfield_ds = np.fft.ifft2(
            spectrum
            *
            dpupil_ds
        )


        # -------------------------------------------------
        # Intensity
        # -------------------------------------------------

        intensity_source = (
            np.abs(field)**2
        )


        total_intensity += (
            source_weight
            *
            intensity_source
        )


        # -------------------------------------------------
        # dI / ds
        #
        # = 2 Re[
        #     E* dE/ds
        #   ]
        # -------------------------------------------------

        dI_source_ds = (
            2.0
            *
            np.real(
                np.conj(field)
                *
                dfield_ds
            )
        )


        dI_ds += (
            source_weight
            *
            dI_source_ds
        )


    return (
        total_intensity,
        dI_ds
    )


# =========================================================
# 9. Fixed normalization reference
# =========================================================
#
# IMPORTANT:
#
# reference_peak does NOT change while source is optimized.
# =========================================================

reference_source_shift = 0.60


reference_intensity, _ = (
    optical_forward_with_source_gradient(
        target,
        focus=0.0,
        source_shift=reference_source_shift
    )
)


reference_peak = np.max(
    reference_intensity
)


# =========================================================
# 10. Soft resist
# =========================================================

threshold = 0.5
beta = 20.0


def soft_resist(
    intensity
):

    argument = (
        -beta
        *
        (
            intensity
            -
            threshold
        )
    )


    argument = np.clip(
        argument,
        -60.0,
        60.0
    )


    return (
        1.0
        /
        (
            1.0
            +
            np.exp(argument)
        )
    )


# =========================================================
# 11. One process condition:
#     loss + dL/ds
# =========================================================

def condition_source_loss_gradient(
    mask,
    focus,
    dose,
    source_shift
):

    # =====================================================
    # Optical imaging
    # =====================================================

    (
        intensity,
        dI_ds
    ) = optical_forward_with_source_gradient(
        mask,
        focus,
        source_shift
    )


    # =====================================================
    # Fixed normalization + dose
    # =====================================================

    effective_intensity = (
        dose
        *
        intensity
        /
        reference_peak
    )


    # =====================================================
    # Resist
    # =====================================================

    resist = soft_resist(
        effective_intensity
    )


    # =====================================================
    # Printing loss
    # =====================================================

    loss = np.mean(
        (
            resist
            -
            target
        )**2
    )


    # =====================================================
    # dL / dR
    # =====================================================

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


    # =====================================================
    # dR / dIeff
    # =====================================================

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


    # =====================================================
    # dL / dI
    # =====================================================

    dL_dI = (
        dL_dR
        *
        dR_dIeff
        *
        dose
        /
        reference_peak
    )


    # =====================================================
    # dL / ds
    #
    # source_shift is ONE scalar,
    # so sum contributions from all pixels.
    # =====================================================

    dL_ds = np.sum(
        dL_dI
        *
        dI_ds
    )


    return (
        loss,
        dL_ds,
        resist
    )


# =========================================================
# 12. Soft worst-case objective
# =========================================================

tau = 0.002


def robust_source_loss_gradient(
    mask,
    source_shift
):

    condition_losses = []
    condition_gradients = []


    # =====================================================
    # Evaluate 3 x 3 process conditions
    # =====================================================

    for focus in focus_list:

        for dose in dose_list:

            (
                loss,
                dL_ds,
                _
            ) = condition_source_loss_gradient(
                mask,
                focus,
                dose,
                source_shift
            )


            condition_losses.append(
                loss
            )


            condition_gradients.append(
                dL_ds
            )


    condition_losses = np.array(
        condition_losses
    )

    condition_gradients = np.array(
        condition_gradients
    )


    # =====================================================
    # Soft worst-case weights
    # =====================================================

    max_loss = np.max(
        condition_losses
    )


    exp_values = np.exp(
        (
            condition_losses
            -
            max_loss
        )
        /
        tau
    )


    weights = (
        exp_values
        /
        np.sum(
            exp_values
        )
    )


    # =====================================================
    # Soft worst-case loss
    # =====================================================

    robust_loss = (
        max_loss
        +
        tau
        *
        np.log(
            np.mean(
                exp_values
            )
        )
    )


    # =====================================================
    # Robust source gradient
    #
    # dLrobust/ds =
    #
    # sum_k w_k dL_k/ds
    # =====================================================

    robust_gradient = np.sum(
        weights
        *
        condition_gradients
    )


    return (
        robust_loss,
        robust_gradient,
        condition_losses,
        condition_gradients,
        weights
    )


# =========================================================
# 13. Continuous source optimization
# =========================================================

source_shift = 0.60

source_learning_rate = 5.0

num_iters = 200


source_history = []
robust_loss_history = []
average_loss_history = []
worst_loss_history = []
gradient_history = []


print()
print("=" * 70)
print("CONTINUOUS SOURCE OPTIMIZATION")
print("=" * 70)


for it in range(
    num_iters
):

    (
        robust_loss,
        dL_ds,
        condition_losses,
        condition_gradients,
        weights
    ) = robust_source_loss_gradient(
        mask_fixed,
        source_shift
    )


    # -----------------------------------------------------
    # Save current values BEFORE update
    # -----------------------------------------------------

    source_history.append(
        source_shift
    )


    robust_loss_history.append(
        robust_loss
    )


    average_loss_history.append(
        np.mean(
            condition_losses
        )
    )


    worst_loss_history.append(
        np.max(
            condition_losses
        )
    )


    gradient_history.append(
        dL_ds
    )


    # =====================================================
    # Gradient descent update
    # =====================================================

    source_shift = (
        source_shift
        -
        source_learning_rate
        *
        dL_ds
    )


    # =====================================================
    # Keep source shift in a sensible range
    # =====================================================

    source_shift = np.clip(
        source_shift,
        0.0,
        1.2
    )


    if (
        it % 10 == 0
        or
        it == num_iters - 1
    ):

        print(
            f"Iteration {it:3d} | "
            f"shift = {source_shift:.8f} | "
            f"soft loss = {robust_loss:.10f} | "
            f"average = {np.mean(condition_losses):.10f} | "
            f"worst = {np.max(condition_losses):.10f} | "
            f"dL/ds = {dL_ds:.6e}"
        )


# =========================================================
# 14. Final evaluation
# =========================================================

(
    final_robust_loss,
    final_gradient,
    final_condition_losses,
    final_condition_gradients,
    final_weights
) = robust_source_loss_gradient(
    mask_fixed,
    source_shift
)


loss_matrix = final_condition_losses.reshape(
    len(focus_list),
    len(dose_list)
)


weight_matrix = final_weights.reshape(
    len(focus_list),
    len(dose_list)
)


# =========================================================
# 15. Final nominal resist
# =========================================================

(
    nominal_loss,
    _,
    nominal_resist
) = condition_source_loss_gradient(
    mask_fixed,
    focus=0.0,
    dose=1.0,
    source_shift=source_shift
)


# =========================================================
# 16. Print final result
# =========================================================

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)


print(
    "Initial source shift =",
    source_history[0]
)

print(
    "Final source shift   =",
    source_shift
)


print()

print(
    "Nominal loss =",
    nominal_loss
)

print(
    "Average process loss =",
    np.mean(
        final_condition_losses
    )
)

print(
    "Worst process loss =",
    np.max(
        final_condition_losses
    )
)

print(
    "Soft worst-case loss =",
    final_robust_loss
)

print(
    "Final dL/ds =",
    final_gradient
)


print()
print(
    "Process loss matrix:"
)

print(
    loss_matrix
)


print()
print(
    "Process weight matrix:"
)

print(
    weight_matrix
)


# =========================================================
# 17. Plot everything in one large figure
# =========================================================

extent = [
    x[0],
    x[-1],
    y[0],
    y[-1]
]


fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)


# =========================================================
# Target
# =========================================================

im = axes[0, 0].imshow(
    target,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[0, 0].set_title(
    "Target"
)

axes[0, 0].set_xlabel(
    "x (um)"
)

axes[0, 0].set_ylabel(
    "y (um)"
)

fig.colorbar(
    im,
    ax=axes[0, 0]
)


# =========================================================
# Fixed mask
# =========================================================

im = axes[0, 1].imshow(
    mask_fixed,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[0, 1].set_title(
    "Fixed Mask"
)

axes[0, 1].set_xlabel(
    "x (um)"
)

axes[0, 1].set_ylabel(
    "y (um)"
)

fig.colorbar(
    im,
    ax=axes[0, 1]
)


# =========================================================
# Final nominal resist
# =========================================================

im = axes[0, 2].imshow(
    nominal_resist,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[0, 2].set_title(
    f"Final Resist\nSource shift = {source_shift:.4f}"
)

axes[0, 2].set_xlabel(
    "x (um)"
)

axes[0, 2].set_ylabel(
    "y (um)"
)

fig.colorbar(
    im,
    ax=axes[0, 2]
)


# =========================================================
# Source-shift history
# =========================================================

axes[1, 0].plot(
    source_history
)

axes[1, 0].set_xlabel(
    "Iteration"
)

axes[1, 0].set_ylabel(
    "Source Shift (1/um)"
)

axes[1, 0].set_title(
    "Source Shift History"
)

axes[1, 0].grid(
    True
)


# =========================================================
# Loss history
# =========================================================

axes[1, 1].plot(
    robust_loss_history,
    label="Soft Worst"
)

axes[1, 1].plot(
    average_loss_history,
    label="Average"
)

axes[1, 1].plot(
    worst_loss_history,
    label="Worst"
)

axes[1, 1].set_xlabel(
    "Iteration"
)

axes[1, 1].set_ylabel(
    "Loss"
)

axes[1, 1].set_title(
    "Process Loss History"
)

axes[1, 1].legend()

axes[1, 1].grid(
    True
)


# =========================================================
# Final process weight map
# =========================================================

im = axes[1, 2].imshow(
    weight_matrix,
    origin="lower",
    aspect="auto",
    vmin=0
)

axes[1, 2].set_title(
    "Final Soft-Worst Weight Map"
)

axes[1, 2].set_xticks(
    range(
        len(dose_list)
    )
)

axes[1, 2].set_xticklabels(
    dose_list
)

axes[1, 2].set_yticks(
    range(
        len(focus_list)
    )
)

axes[1, 2].set_yticklabels(
    focus_list
)

axes[1, 2].set_xlabel(
    "Dose"
)

axes[1, 2].set_ylabel(
    "Focus (um)"
)

fig.colorbar(
    im,
    ax=axes[1, 2]
)


plt.tight_layout()

plt.show()

# =========================================================
# 18. Scan robust loss vs source shift
# =========================================================
#
# Purpose:
# Fix the mask and directly observe
#
#        L_robust(s)
#
# This lets us compare:
#
# 1. actual loss curve
# 2. analytical gradient dL/ds
# 3. gradient-descent result
# =========================================================


source_scan = np.arange(
    0.0,
    0.3001,
    0.005
)


scan_loss = []
scan_gradient = []
scan_average = []
scan_worst = []


for s in source_scan:

    (
        robust_loss,
        dL_ds,
        condition_losses,
        condition_gradients,
        weights
    ) = robust_source_loss_gradient(
        mask_fixed,
        s
    )


    scan_loss.append(
        robust_loss
    )


    scan_gradient.append(
        dL_ds
    )


    scan_average.append(
        np.mean(
            condition_losses
        )
    )


    scan_worst.append(
        np.max(
            condition_losses
        )
    )


scan_loss = np.array(
    scan_loss
)

scan_gradient = np.array(
    scan_gradient
)

scan_average = np.array(
    scan_average
)

scan_worst = np.array(
    scan_worst
)


# =========================================================
# Find minimum
# =========================================================

best_index = np.argmin(
    scan_loss
)


best_source_scan = (
    source_scan[
        best_index
    ]
)


best_loss_scan = (
    scan_loss[
        best_index
    ]
)


print()
print("=" * 70)
print("SOURCE-SHIFT LOSS SCAN")
print("=" * 70)


print(
    "Best source shift from scan =",
    best_source_scan
)


print(
    "Minimum robust loss =",
    best_loss_scan
)


print(
    "Gradient at scan minimum =",
    scan_gradient[
        best_index
    ]
)


# =========================================================
# Plot robust loss vs source shift
# =========================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    source_scan,
    scan_loss,
    label="Soft Worst"
)


plt.plot(
    source_scan,
    scan_average,
    label="Average"
)


plt.plot(
    source_scan,
    scan_worst,
    label="Worst"
)


plt.axvline(
    best_source_scan,
    linestyle="--",
    label=f"Best shift = {best_source_scan:.3f}"
)


plt.xlabel(
    "Source Shift (1/um)"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Process Loss vs Source Shift"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# =========================================================
# Plot gradient vs source shift
# =========================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    source_scan,
    scan_gradient
)


plt.axhline(
    0.0,
    linestyle="--"
)


plt.xlabel(
    "Source Shift (1/um)"
)

plt.ylabel(
    "dL / ds"
)

plt.title(
    "Source Gradient vs Source Shift"
)

plt.grid(
    True
)

plt.tight_layout()

plt.show()