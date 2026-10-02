import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic parameters
# =========================================================

N = 128
L = 6.4
dx = L / N

x = np.arange(N) * dx - L / 2
x_fft = np.arange(N) * dx

period = 0.20
line_width = 0.10

wavelength = 0.193
NA = 0.85

cutoff = NA / wavelength


# =========================================================
# 2. Target / initial mask
# =========================================================

position_in_period = (
    np.mod(
        x + period / 2,
        period
    )
    -
    period / 2
)

mask = (
    np.abs(position_in_period)
    <= line_width / 2
).astype(float)

target = mask.copy()


# =========================================================
# 3. Fourier frequency grid
# =========================================================

freq = np.fft.fftfreq(
    N,
    d=dx
)


# =========================================================
# 4. Multi-point illumination source
# =========================================================

source_positions = np.array([
    -2.0,
    -1.5,
    -1.0,
    -0.5,
     0.5,
     1.0,
     1.5,
     2.0
])

source_weights = np.ones(
    len(source_positions)
)

source_weights = (
    source_weights
    /
    np.sum(source_weights)
)


# =========================================================
# 5. Shifted pupil
# =========================================================

def make_shifted_pupil(
    freq,
    cutoff,
    source_position
):

    pupil = (
        np.abs(
            freq - source_position
        )
        <= cutoff
    ).astype(float)

    return pupil


# =========================================================
# 6. Build TCC
# =========================================================

TCC = np.zeros(
    (N, N),
    dtype=complex
)

for source_position, weight in zip(
    source_positions,
    source_weights
):

    pupil = make_shifted_pupil(
        freq,
        cutoff,
        source_position
    )

    TCC += (
        weight
        *
        np.outer(
            pupil,
            np.conj(pupil)
        )
    )


# =========================================================
# 7. TCC eigendecomposition
# =========================================================

eigenvalues, eigenvectors = np.linalg.eigh(
    TCC
)

sort_index = np.argsort(
    eigenvalues
)[::-1]

eigenvalues = eigenvalues[
    sort_index
]

eigenvectors = eigenvectors[
    :,
    sort_index
]


# Small negative values may appear because of numerical error
eigenvalues = np.clip(
    eigenvalues,
    0,
    None
)


# =========================================================
# 8. Full TCC image
#    Used as reference normalization
# =========================================================

def full_tcc_image(
    mask_input
):

    spectrum_input = np.fft.fft(
        mask_input
    )

    intensity = np.zeros(N)

    for ix, x_value in enumerate(
        x_fft
    ):

        q = (
            spectrum_input
            *
            np.exp(
                1j
                *
                2
                *
                np.pi
                *
                freq
                *
                x_value
            )
        )

        intensity[ix] = (
            np.real(
                q
                @
                TCC
                @
                np.conj(q)
            )
            /
            (N ** 2)
        )

    return intensity


# =========================================================
# 9. Truncated coherent-mode imaging
# =========================================================

def partial_coherent_image(
    mask_input,
    num_modes
):

    spectrum_input = np.fft.fft(
        mask_input
    )

    intensity = np.zeros(N)

    for ix, x_value in enumerate(
        x_fft
    ):

        q = (
            spectrum_input
            *
            np.exp(
                1j
                *
                2
                *
                np.pi
                *
                freq
                *
                x_value
            )
        )

        intensity_value = 0.0

        for k in range(
            num_modes
        ):

            mode_k = eigenvectors[
                :,
                k
            ]

            lambda_k = eigenvalues[
                k
            ]

            E_k = (
                q
                @
                mode_k
            )

            intensity_value += (
                lambda_k
                *
                np.abs(E_k) ** 2
            )

        intensity[ix] = (
            intensity_value
            /
            (N ** 2)
        )

    return intensity


# =========================================================
# 10. Soft resist model
# =========================================================

def soft_resist(
    intensity,
    threshold=0.5,
    beta=20.0
):

    resist = (
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

    return resist


# =========================================================
# 11. ILT loss
# =========================================================

def compute_loss(
    resist,
    target
):

    loss = np.mean(
        (
            resist
            -
            target
        ) ** 2
    )

    return loss


# =========================================================
# 12. Full TCC reference
# =========================================================

intensity_full = full_tcc_image(
    mask
)

reference_peak = np.max(
    intensity_full
)

full_image_norm = (
    intensity_full
    /
    reference_peak
)


# =========================================================
# 13. Partial-coherent forward model
# =========================================================

num_modes = 8

intensity_modes = partial_coherent_image(
    mask,
    num_modes
)

intensity_modes_norm = (
    intensity_modes
    /
    reference_peak
)


# =========================================================
# 14. Resist response
# =========================================================

threshold = 0.5
beta = 20.0

resist = soft_resist(
    intensity_modes_norm,
    threshold,
    beta
)

def forward_modes(
    mask_input,
    num_modes
):

    spectrum = np.fft.fft(
        mask_input
    )

    intensity = np.zeros(N)

    fields = []

    for k in range(num_modes):

        mode_k = eigenvectors[:, k]
        lambda_k = eigenvalues[k]

        field_k = np.fft.ifft(
            spectrum * mode_k
        )

        intensity += (
            lambda_k
            *
            np.abs(field_k) ** 2
        )

        fields.append(field_k)

    return intensity, fields
def backward_modes(
    fields,
    resist,
    target,
    num_modes,
    reference_peak,
    beta
):

    # -----------------------------------------
    # Loss -> Resist
    # -----------------------------------------

    dL_dR = (
        2
        *
        (resist - target)
        /
        N
    )


    # -----------------------------------------
    # Resist -> normalized intensity
    # -----------------------------------------

    dL_dInorm = (
        dL_dR
        *
        beta
        *
        resist
        *
        (1 - resist)
    )


    # -----------------------------------------
    # normalized intensity -> raw intensity
    # -----------------------------------------

    dL_dI = (
        dL_dInorm
        /
        reference_peak
    )


    # -----------------------------------------
    # Intensity -> Mask
    # -----------------------------------------

    dL_dM = np.zeros(
        N
    )


    for k in range(num_modes):

        mode_k = eigenvectors[:, k]
        lambda_k = eigenvalues[k]

        field_k = fields[k]


        # dL / dE_k
        dL_dE = (
            2
            *
            lambda_k
            *
            dL_dI
            *
            field_k
        )


        # Optical adjoint
        gradient_k = np.fft.ifft(
            np.fft.fft(dL_dE)
            *
            np.conj(mode_k)
        )


        dL_dM += np.real(
            gradient_k
        )


    return dL_dM
num_modes = 20

mask_current = mask.copy()


# Forward
intensity, fields = forward_modes(
    mask_current,
    num_modes
)

intensity_norm = (
    intensity
    /
    reference_peak
)

resist = soft_resist(
    intensity_norm,
    threshold,
    beta
)

loss_before = compute_loss(
    resist,
    target
)


# Backward
gradient = backward_modes(
    fields,
    resist,
    target,
    num_modes,
    reference_peak,
    beta
)


# One gradient-descent step
learning_rate = 0.1

mask_new = (
    mask_current
    -
    learning_rate
    *
    gradient
)

mask_new = np.clip(
    mask_new,
    0.0,
    1.0
)


# Forward again
intensity_new, _ = forward_modes(
    mask_new,
    num_modes
)

resist_new = soft_resist(
    intensity_new / reference_peak,
    threshold,
    beta
)

loss_after = compute_loss(
    resist_new,
    target
)


print()
print("=" * 50)
print("One ILT Gradient Step")
print("=" * 50)

print(
    "Loss before =",
    loss_before
)

print(
    "Loss after  =",
    loss_after
)

print(
    "Gradient max =",
    np.max(np.abs(gradient))
)
# =========================================================
# Partial-Coherent ILT + Binarization Penalty
# =========================================================
lambda_bin = 0.02
num_modes = 20

num_iterations = 100
learning_rate = 0.1

lambda_bin = 0.02

mask_current = mask.copy()

total_loss_history = []
print_loss_history = []
bin_loss_history = []


for iteration in range(num_iterations):

    # -----------------------------------------
    # Forward
    # -----------------------------------------

    intensity, fields = forward_modes(
        mask_current,
        num_modes
    )

    intensity_norm = (
        intensity
        /
        reference_peak
    )

    resist = soft_resist(
        intensity_norm,
        threshold,
        beta
    )


    # -----------------------------------------
    # Printing loss
    # -----------------------------------------

    print_loss = compute_loss(
        resist,
        target
    )


    # -----------------------------------------
    # Binarization loss
    # -----------------------------------------

    bin_loss = np.mean(
        mask_current
        *
        (1 - mask_current)
    )


    # -----------------------------------------
    # Total loss
    # -----------------------------------------

    total_loss = (
        print_loss
        +
        lambda_bin
        *
        bin_loss
    )


    print_loss_history.append(
        print_loss
    )

    bin_loss_history.append(
        bin_loss
    )

    total_loss_history.append(
        total_loss
    )


    # -----------------------------------------
    # Optical gradient
    # -----------------------------------------

    gradient_print = backward_modes(
        fields,
        resist,
        target,
        num_modes,
        reference_peak,
        beta
    )


    # -----------------------------------------
    # Binarization gradient
    # -----------------------------------------

    gradient_bin = (
        1
        -
        2 * mask_current
    ) / N


    # -----------------------------------------
    # Total gradient
    # -----------------------------------------

    gradient_total = (
        gradient_print
        +
        lambda_bin
        *
        gradient_bin
    )


    # -----------------------------------------
    # Gradient descent
    # -----------------------------------------

    mask_current = (
        mask_current
        -
        learning_rate
        *
        gradient_total
    )

    mask_current = np.clip(
        mask_current,
        0.0,
        1.0
    )


    if (
        iteration % 10 == 0
        or
        iteration == num_iterations - 1
    ):

        print(
            f"Iteration {iteration:3d}: "
            f"Print = {print_loss:.6f}, "
            f"Bin = {bin_loss:.6f}, "
            f"Total = {total_loss:.6f}"
        )
intensity_final, _ = forward_modes(
    mask_current,
    num_modes
)

resist_final = soft_resist(
    intensity_final / reference_peak,
    threshold,
    beta
)

final_print_loss = compute_loss(
    resist_final,
    target
)

final_bin_loss = np.mean(
    mask_current
    *
    (1 - mask_current)
)


print()
print("=" * 50)
print("ILT with Binarization")
print("=" * 50)

print(
    "Final printing loss =",
    final_print_loss
)

print(
    "Final binarization loss =",
    final_bin_loss
)

print(
    "Mask minimum =",
    np.min(mask_current)
)

print(
    "Mask maximum =",
    np.max(mask_current)
)
plt.figure(figsize=(9, 5))

plt.plot(
    x,
    target,
    label="Target"
)

plt.plot(
    x,
    mask_current,
    label="Optimized Mask"
)

plt.plot(
    x,
    resist_final,
    label="Printed Resist"
)

plt.xlim(-0.5, 0.5)

plt.xlabel("x (um)")
plt.ylabel("Response")

plt.title(
    "Partial-Coherent ILT with Binarization"
)

plt.grid()
plt.legend()

plt.show()