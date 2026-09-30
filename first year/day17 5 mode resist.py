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
# 2. Periodic mask
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


# =========================================================
# 3. Fourier grid and mask spectrum
# =========================================================

freq = np.fft.fftfreq(
    N,
    d=dx
)

spectrum = np.fft.fft(
    mask
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

    return (
        np.abs(
            freq - source_position
        )
        <= cutoff
    ).astype(float)


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
# 7. Full Hopkins / TCC image
# =========================================================

intensity_tcc = np.zeros(
    N
)

for ix, x_value in enumerate(
    x_fft
):

    q = (
        spectrum
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

    intensity_tcc[ix] = np.real(
        q
        @
        TCC
        @
        np.conj(q)
    ) / (N ** 2)


# =========================================================
# 8. TCC eigendecomposition
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


# =========================================================
# 9. Reconstruct with truncated coherent modes
# =========================================================

num_modes = 5

intensity_modes = np.zeros(
    N
)

for ix, x_value in enumerate(
    x_fft
):

    q = (
        spectrum
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

    intensity_modes[ix] = (
        intensity_value
        /
        (N ** 2)
    )


# =========================================================
# 10. Use the SAME normalization reference
# =========================================================

reference_peak = np.max(
    intensity_tcc
)

full_image = (
    intensity_tcc
    /
    reference_peak
)

mode_image = (
    intensity_modes
    /
    reference_peak
)


# =========================================================
# 11. Soft resist model
# =========================================================

def soft_resist(
    intensity,
    threshold=0.5,
    beta=20.0
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


threshold = 0.5
beta = 20.0

resist_full = soft_resist(
    full_image,
    threshold,
    beta
)

resist_modes = soft_resist(
    mode_image,
    threshold,
    beta
)


# =========================================================
# 12. Compare aerial image and resist error
# =========================================================

image_difference = np.abs(
    full_image
    -
    mode_image
)

resist_difference = np.abs(
    resist_full
    -
    resist_modes
)


print()
print("=" * 60)
print("5-Mode Approximation")
print("=" * 60)

print(
    "Maximum aerial-image difference =",
    np.max(image_difference)
)

print(
    "Mean aerial-image difference =",
    np.mean(image_difference)
)

print()

print(
    "Maximum resist difference =",
    np.max(resist_difference)
)

print(
    "Mean resist difference =",
    np.mean(resist_difference)
)


# =========================================================
# 13. Final plot only
# =========================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    x,
    resist_full,
    label="Full TCC"
)

plt.plot(
    x,
    resist_modes,
    linestyle="--",
    label="5 Coherent Modes"
)

plt.xlim(
    -0.5,
    0.5
)

plt.xlabel(
    "x (um)"
)

plt.ylabel(
    "Resist Response"
)

plt.title(
    "Full TCC vs 5-Mode Resist"
)

plt.grid()
plt.legend()

plt.show()
# =========================================================
# 14. Edge interpolation
# =========================================================

def interpolate_edge(
    x1,
    x2,
    y1,
    y2,
    threshold
):

    if y2 == y1:
        return 0.5 * (x1 + x2)

    return (
        x1
        +
        (
            threshold - y1
        )
        /
        (
            y2 - y1
        )
        *
        (
            x2 - x1
        )
    )


# =========================================================
# 15. Measure CD from resist response
# =========================================================

def measure_cd_from_resist(
    x,
    resist,
    threshold=0.5
):

    binary = (
        resist
        >= threshold
    ).astype(float)

    transition = np.diff(
        binary
    )

    left_edges = np.where(
        transition == 1
    )[0]

    right_edges = np.where(
        transition == -1
    )[0]

    center_index = np.argmin(
        np.abs(x)
    )

    left_candidates = left_edges[
        left_edges < center_index
    ]

    right_candidates = right_edges[
        right_edges > center_index
    ]

    if (
        len(left_candidates) == 0
        or
        len(right_candidates) == 0
    ):
        return np.nan, np.nan, np.nan

    left_index = left_candidates[-1]
    right_index = right_candidates[0]

    x_left = interpolate_edge(
        x[left_index],
        x[left_index + 1],
        resist[left_index],
        resist[left_index + 1],
        threshold
    )

    x_right = interpolate_edge(
        x[right_index],
        x[right_index + 1],
        resist[right_index],
        resist[right_index + 1],
        threshold
    )

    cd = (
        x_right
        -
        x_left
    )

    return cd, x_left, x_right


# =========================================================
# 16. Measure Full-TCC CD
# =========================================================

cd_full, x_left_full, x_right_full = (
    measure_cd_from_resist(
        x,
        resist_full,
        threshold=0.5
    )
)


# =========================================================
# 17. Measure 5-mode CD
# =========================================================

cd_modes, x_left_modes, x_right_modes = (
    measure_cd_from_resist(
        x,
        resist_modes,
        threshold=0.5
    )
)


# =========================================================
# 18. CD error
# =========================================================

cd_error = (
    cd_modes
    -
    cd_full
)


print()
print("=" * 60)
print("CD Comparison")
print("=" * 60)

print(
    "Full TCC CD =",
    cd_full * 1000,
    "nm"
)

print(
    "5-mode CD =",
    cd_modes * 1000,
    "nm"
)

print(
    "CD error =",
    cd_error * 1000,
    "nm"
)

print()
print(
    "Full TCC left edge =",
    x_left_full * 1000,
    "nm"
)

print(
    "5-mode left edge =",
    x_left_modes * 1000,
    "nm"
)

print()

print(
    "Full TCC right edge =",
    x_right_full * 1000,
    "nm"
)

print(
    "5-mode right edge =",
    x_right_modes * 1000,
    "nm"
)
# =========================================================
# Measure CD from resist
# =========================================================

def interpolate_edge(x1, x2, y1, y2, threshold=0.5):

    return (
        x1
        +
        (threshold - y1)
        /
        (y2 - y1)
        *
        (x2 - x1)
    )


def measure_cd(x, resist, threshold=0.5):

    binary = (resist >= threshold).astype(float)

    transition = np.diff(binary)

    left_edges = np.where(
        transition == 1
    )[0]

    right_edges = np.where(
        transition == -1
    )[0]

    center_index = np.argmin(
        np.abs(x)
    )

    left_candidates = left_edges[
        left_edges < center_index
    ]

    right_candidates = right_edges[
        right_edges > center_index
    ]

    left_index = left_candidates[-1]
    right_index = right_candidates[0]

    x_left = interpolate_edge(
        x[left_index],
        x[left_index + 1],
        resist[left_index],
        resist[left_index + 1],
        threshold
    )

    x_right = interpolate_edge(
        x[right_index],
        x[right_index + 1],
        resist[right_index],
        resist[right_index + 1],
        threshold
    )

    cd = x_right - x_left

    return cd, x_left, x_right


# =========================================================
# Full TCC CD
# =========================================================

cd_full, left_full, right_full = measure_cd(
    x,
    resist_full
)


# =========================================================
# 5-mode CD
# =========================================================

cd_modes, left_modes, right_modes = measure_cd(
    x,
    resist_modes
)


# =========================================================
# CD error
# =========================================================

cd_error = cd_modes - cd_full


print()
print("=" * 50)
print("CD Comparison")
print("=" * 50)

print(
    "Full TCC CD =",
    cd_full * 1000,
    "nm"
)

print(
    "5-mode CD =",
    cd_modes * 1000,
    "nm"
)

print(
    "CD error =",
    cd_error * 1000,
    "nm"
)
# =========================================================
# Scan CD error vs number of coherent modes
# =========================================================

mode_list = []
cd_error_list = []

for num_modes in range(1, 9):

    intensity_modes = np.zeros(N)

    for ix, x_value in enumerate(x_fft):

        q = (
            spectrum
            *
            np.exp(
                1j * 2 * np.pi * freq * x_value
            )
        )

        intensity_value = 0.0

        for k in range(num_modes):

            mode_k = eigenvectors[:, k]
            lambda_k = eigenvalues[k]

            E_k = q @ mode_k

            intensity_value += (
                lambda_k
                *
                np.abs(E_k) ** 2
            )

        intensity_modes[ix] = (
            intensity_value / (N ** 2)
        )


    # Use same normalization reference
    mode_image = (
        intensity_modes
        /
        reference_peak
    )


    # Soft resist
    resist_modes = soft_resist(
        mode_image,
        threshold,
        beta
    )


    # Measure CD
    cd_modes, _, _ = measure_cd(
        x,
        resist_modes
    )


    # CD error
    cd_error_nm = (
        cd_modes - cd_full
    ) * 1000


    mode_list.append(
        num_modes
    )

    cd_error_list.append(
        cd_error_nm
    )


    print(
        f"Modes = {num_modes}: "
        f"CD = {cd_modes * 1000:.4f} nm, "
        f"CD error = {cd_error_nm:.4f} nm"
    )
plt.figure(figsize=(8, 5))

plt.plot(
    mode_list,
    cd_error_list,
    marker="o"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Number of Coherent Modes"
)

plt.ylabel(
    "CD Error (nm)"
)

plt.title(
    "CD Error vs Number of Coherent Modes"
)

plt.grid()

plt.show()
# =========================================================
# Automatically choose modes using CD tolerance
# =========================================================

cd_tolerance_nm = 1.0

selected_modes = None

for num_modes, cd_error_nm in zip(
    mode_list,
    cd_error_list
):

    if abs(cd_error_nm) < cd_tolerance_nm:

        selected_modes = num_modes
        break


print()
print("=" * 50)
print("CD-Based Mode Selection")
print("=" * 50)

print(
    "CD tolerance =",
    cd_tolerance_nm,
    "nm"
)

if selected_modes is not None:

    print(
        "Minimum number of modes =",
        selected_modes
    )

else:

    print(
        "No mode number satisfies the CD tolerance."
    )