import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Spatial grid
# =========================================================

N = 128
L = 2.0                  # um
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
# We still use the sigmoid representation:
#
# M = sigmoid(Z)
#
# For this experiment the mask remains fixed.
# =========================================================

eps = 0.10

mask_initial = np.clip(
    target,
    eps,
    1.0 - eps
)

Z_initial = np.log(
    mask_initial
    /
    (1.0 - mask_initial)
)


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


mask = sigmoid(
    Z_initial
)


# =========================================================
# 4. Optical parameters
# =========================================================

wavelength = 0.193       # um
NA = 0.85

cutoff = NA / wavelength


# =========================================================
# 5. Frequency grid
# =========================================================

freq = np.fft.fftfreq(
    N,
    d=dx
)

FX, FY = np.meshgrid(
    freq,
    freq
)


print(
    "Optical cutoff =",
    cutoff,
    "1/um"
)


# =========================================================
# 6. Soft pupil parameters
# =========================================================
#
# Larger alpha:
#     sharper pupil edge
#
# Smaller alpha:
#     smoother pupil edge
#
# We use a moderate value first so that source-shift
# gradients are numerically visible and stable.
# =========================================================

alpha = 20.0

small_number = 1e-12


# =========================================================
# 7. Soft complex pupil + dP/d(source_x)
# =========================================================

def soft_pupil_and_gradient(
    source_x,
    source_y,
    focus
):

    # -----------------------------------------------------
    # Shifted frequency coordinates
    # -----------------------------------------------------

    fx = (
        FX - source_x
    )

    fy = (
        FY - source_y
    )


    # -----------------------------------------------------
    # Radius in shifted frequency coordinates
    # -----------------------------------------------------

    r = np.sqrt(
        fx**2
        +
        fy**2
        +
        small_number
    )


    # =====================================================
    # A. Soft aperture
    #
    # A = 1 / [1 + exp(alpha * (r - cutoff))]
    # =====================================================

    sigmoid_argument = (
        alpha
        *
        (
            r - cutoff
        )
    )


    # Prevent numerical overflow in exp()
    sigmoid_argument = np.clip(
        sigmoid_argument,
        -60.0,
        60.0
    )


    aperture = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                sigmoid_argument
            )
        )
    )


    # =====================================================
    # B. dA / d(source_x)
    #
    # dA/dsx =
    #
    # alpha * A * (1-A) * fx / r
    # =====================================================

    dA_dsx = (
        alpha
        *
        aperture
        *
        (
            1.0
            -
            aperture
        )
        *
        fx
        /
        r
    )


    # =====================================================
    # C. Defocus phase
    #
    # phase =
    #
    # exp[
    #   -i*pi*lambda*z*(fx^2 + fy^2)
    # ]
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
    # D. dPhase / d(source_x)
    #
    # = i * 2*pi*lambda*z*fx*phase
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
    # E. Complete complex pupil
    #
    # P = A * phase
    # =====================================================

    pupil = (
        aperture
        *
        phase
    )


    # =====================================================
    # F. Product rule
    #
    # dP/dsx
    #
    # = dA/dsx * phase
    # + A * dphase/dsx
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
# 8. Partial coherent forward +
#    source-shift derivative
# =========================================================
#
# Symmetric horizontal dipole:
#
# source 1 = (+s, 0)
# source 2 = (-s, 0)
#
# The optimization variable is the common distance s.
#
# Therefore:
#
# d(+s)/ds = +1
# d(-s)/ds = -1
# =========================================================

def optical_forward_with_source_gradient(
    mask_input,
    focus,
    source_shift
):

    spectrum = np.fft.fft2(
        mask_input
    )


    total_intensity = np.zeros_like(
        mask_input,
        dtype=float
    )


    dI_ds = np.zeros_like(
        mask_input,
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


        # =================================================
        # P and dP/d(source_x)
        # =================================================

        (
            pupil,
            dpupil_dsx
        ) = soft_pupil_and_gradient(
            source_x,
            source_y,
            focus
        )


        # =================================================
        # Chain rule:
        #
        # dP/ds =
        # dP/d(source_x)
        # *
        # d(source_x)/ds
        # =================================================

        dpupil_ds = (
            dpupil_dsx
            *
            source_sign
        )


        # =================================================
        # Field
        #
        # E = IFFT[
        #       FFT(M) * P
        #     ]
        # =================================================

        field = np.fft.ifft2(
            spectrum
            *
            pupil
        )


        # =================================================
        # Field derivative
        #
        # dE/ds =
        #
        # IFFT[
        #   FFT(M) * dP/ds
        # ]
        # =================================================

        dfield_ds = np.fft.ifft2(
            spectrum
            *
            dpupil_ds
        )


        # =================================================
        # Intensity
        #
        # I_s = |E_s|^2
        # =================================================

        intensity_source = (
            np.abs(field)**2
        )


        total_intensity += (
            source_weight
            *
            intensity_source
        )


        # =================================================
        # Intensity derivative
        #
        # dI_s/ds =
        #
        # 2 Re[
        #     E_s^* dE_s/ds
        # ]
        # =================================================

        dI_source_ds = (
            2.0
            *
            np.real(
                np.conj(
                    field
                )
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
# reference_peak must remain fixed while source_shift
# changes.
#
# Otherwise source optimization would partly optimize
# the normalization itself.
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


print(
    "Reference peak =",
    reference_peak
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
            np.exp(
                argument
            )
        )
    )


# =========================================================
# 11. One process condition:
#     loss + dL/d(source_shift)
# =========================================================

def source_loss_and_gradient(
    mask_input,
    focus,
    dose,
    source_shift
):

    # =====================================================
    # Optical forward
    # =====================================================

    (
        intensity,
        dI_ds
    ) = optical_forward_with_source_gradient(
        mask_input,
        focus,
        source_shift
    )


    # =====================================================
    # Dose + fixed normalization
    #
    # I_eff =
    #
    # dose * I / reference_peak
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
    #
    # L =
    #
    # mean[
    #   (R - T)^2
    # ]
    # =====================================================

    loss = np.mean(
        (
            resist
            -
            target
        )**2
    )


    # =====================================================
    # dL/dR
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
    # dR/dI_eff
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
    # dL/dI
    #
    # I_eff =
    # dose * I / reference_peak
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
    # dL/ds
    #
    # L is one scalar
    #
    # so contributions from every pixel are summed.
    # =====================================================

    dL_ds = np.sum(
        dL_dI
        *
        dI_ds
    )


    return (
        loss,
        dL_ds,
        resist,
        intensity,
        dI_ds
    )


# =========================================================
# 12. Source-shift gradient check
# =========================================================
#
# Use non-zero defocus deliberately.
#
# This tests BOTH:
#
# aperture derivative
# +
# defocus-phase derivative
# =========================================================

source_shift_check = 0.10

focus_check = 0.05
dose_check = 0.95

epsilon_fd = 1e-5


# =========================================================
# Analytical gradient
# =========================================================

(
    loss_now,
    analytical_gradient,
    resist_check,
    intensity_check,
    dI_ds_check
) = source_loss_and_gradient(
    mask,
    focus_check,
    dose_check,
    source_shift_check
)


# =========================================================
# s + epsilon
# =========================================================

(
    loss_plus,
    _,
    _,
    _,
    _
) = source_loss_and_gradient(
    mask,
    focus_check,
    dose_check,
    source_shift_check
    +
    epsilon_fd
)


# =========================================================
# s - epsilon
# =========================================================

(
    loss_minus,
    _,
    _,
    _,
    _
) = source_loss_and_gradient(
    mask,
    focus_check,
    dose_check,
    source_shift_check
    -
    epsilon_fd
)


# =========================================================
# Central finite difference
# =========================================================

finite_difference = (
    loss_plus
    -
    loss_minus
) / (
    2.0
    *
    epsilon_fd
)


# =========================================================
# Relative error
# =========================================================

relative_error = (
    abs(
        analytical_gradient
        -
        finite_difference
    )
    /
    max(
        abs(
            analytical_gradient
        ),
        abs(
            finite_difference
        ),
        1e-12
    )
)


# =========================================================
# 13. Print gradient-check result
# =========================================================

print()
print("=" * 70)
print("SOURCE SHIFT GRADIENT CHECK")
print("=" * 70)


print(
    "Source shift =",
    source_shift_check
)


print(
    "Focus =",
    focus_check
)


print(
    "Dose =",
    dose_check
)


print()

print(
    "Loss =",
    loss_now
)


print(
    "Analytical gradient =",
    analytical_gradient
)


print(
    "Finite difference   =",
    finite_difference
)


print(
    "Relative error      =",
    relative_error
)


# =========================================================
# 14. Visualize current result
# =========================================================

extent = [
    x[0],
    x[-1],
    y[0],
    y[-1]
]


fig, axes = plt.subplots(
    2,
    2,
    figsize=(11, 9)
)


# ---------------------------------------------------------
# Target
# ---------------------------------------------------------

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

fig.colorbar(
    im,
    ax=axes[0, 0]
)


# ---------------------------------------------------------
# Aerial image
# ---------------------------------------------------------

im = axes[0, 1].imshow(
    intensity_check
    /
    reference_peak,
    extent=extent,
    origin="lower"
)

axes[0, 1].set_title(
    "Normalized Aerial Image"
)

fig.colorbar(
    im,
    ax=axes[0, 1]
)


# ---------------------------------------------------------
# Resist
# ---------------------------------------------------------

im = axes[1, 0].imshow(
    resist_check,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[1, 0].set_title(
    "Printed Resist"
)

fig.colorbar(
    im,
    ax=axes[1, 0]
)


# ---------------------------------------------------------
# dI / ds
# ---------------------------------------------------------

im = axes[1, 1].imshow(
    dI_ds_check,
    extent=extent,
    origin="lower"
)

axes[1, 1].set_title(
    "dI / ds"
)

fig.colorbar(
    im,
    ax=axes[1, 1]
)


for ax in axes.flat:

    ax.set_xlabel(
        "x (um)"
    )

    ax.set_ylabel(
        "y (um)"
    )


plt.tight_layout()

plt.show()