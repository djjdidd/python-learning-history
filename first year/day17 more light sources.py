import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic parameters
# =========================================================

N = 128

L = 6.4                 # total window, um
dx = L / N

# Centered coordinate for plotting / mask construction
x = (
    np.arange(N) * dx
    -
    L / 2
)

period = 0.20           # um
line_width = 0.10       # um

wavelength = 0.193      # um
NA = 0.85

cutoff = (
    NA / wavelength
)


print("dx =", dx, "um")
print("Frequency resolution =", 1 / L, "1/um")
print("Cutoff =", cutoff, "1/um")
print("First diffraction order =", 1 / period, "1/um")


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
# 4. Define dipole illumination
# =========================================================

# Frequency resolution:
# df = 1/L = 0.15625 1/um
#
# 4 frequency bins:
# source_shift = 4 * 0.15625 = 0.625 1/um

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
source_weights / np.sum(source_weights)

# =========================================================
# 5. Shifted pupil function
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
# 6. Abbe formulation
#
# Each source point:
#
# mask spectrum
# -> shifted pupil
# -> field
# -> intensity
#
# Then add INTENSITIES
# =========================================================

intensity_abbe = np.zeros(
    N
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

    filtered_spectrum = (
        spectrum
        *
        pupil
    )

    field = np.fft.ifft(
        filtered_spectrum
    )

    intensity_source = (
        np.abs(field) ** 2
    )

    intensity_abbe += (
        weight
        *
        intensity_source
    )


# =========================================================
# 7. Build TCC matrix
#
# TCC[m,n]
# =
# sum_s w_s P_s[m] P_s*[n]
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

    # Outer product:
    #
    # P_s(f_m) * P_s*(f_n)

    source_tcc = np.outer(
        pupil,
        np.conj(pupil)
    )

    TCC += (
        weight
        *
        source_tcc
    )


# =========================================================
# 8. Hopkins / TCC image reconstruction
# =========================================================

# IMPORTANT:
# FFT reconstruction uses FFT sample coordinates
#
# 0, dx, 2dx, ...
#
# rather than the centered plotting coordinate.

x_fft = (
    np.arange(N)
    *
    dx
)


intensity_tcc = np.zeros(
    N
)


for ix, x_value in enumerate(
    x_fft
):

    # q_m =
    #
    # M(f_m) * exp(i 2 pi f_m x)

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

    # Hopkins intensity:
    #
    # I(x) = q^T TCC q*
    #
    # 1/N^2 comes from NumPy FFT normalization

    intensity_tcc[ix] = np.real(
        q
        @
        TCC
        @
        np.conj(q)
    ) / (N ** 2)


# =========================================================
# 9. Normalize for comparison
# =========================================================

intensity_abbe_norm = (
    intensity_abbe
    /
    np.max(intensity_abbe)
)

intensity_tcc_norm = (
    intensity_tcc
    /
    np.max(intensity_tcc)
)


# =========================================================
# 10. Numerical difference
# =========================================================

difference = np.abs(
    intensity_abbe_norm
    -
    intensity_tcc_norm
)

max_difference = np.max(
    difference
)

mean_difference = np.mean(
    difference
)


print()
print("=" * 60)
print("Abbe vs Hopkins / TCC")
print("=" * 60)

print(
    "Maximum difference =",
    max_difference
)

print(
    "Mean difference =",
    mean_difference
)


# =========================================================
# 11. Plot Abbe vs Hopkins
# =========================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    x,
    intensity_abbe_norm,
    label="Abbe"
)

plt.plot(
    x,
    intensity_tcc_norm,
    linestyle="--",
    label="Hopkins / TCC"
)

plt.xlim(
    -0.5,
    0.5
)

plt.xlabel(
    "x (um)"
)

plt.ylabel(
    "Normalized Intensity"
)

plt.title(
    "Abbe vs Hopkins / TCC"
)

plt.grid()
plt.legend()

plt.show()


# =========================================================
# 12. Plot numerical difference
# =========================================================

plt.figure(
    figsize=(9, 4)
)

plt.plot(
    x,
    difference
)

plt.xlim(
    -0.5,
    0.5
)

plt.xlabel(
    "x (um)"
)

plt.ylabel(
    "|Abbe - TCC|"
)

plt.title(
    "Numerical Difference"
)

plt.grid()

plt.show()


# =========================================================
# 13. Visualize TCC
# =========================================================

TCC_display = np.abs(
    np.fft.fftshift(
        np.fft.fftshift(
            TCC,
            axes=0
        ),
        axes=1
    )
)

freq_display = np.fft.fftshift(
    freq
)


plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    TCC_display,
    extent=[
        freq_display.min(),
        freq_display.max(),
        freq_display.min(),
        freq_display.max()
    ],
    origin="lower",
    aspect="auto"
)

plt.colorbar(
    label="|TCC|"
)

plt.xlabel(
    "f2 (1/um)"
)

plt.ylabel(
    "f1 (1/um)"
)

plt.title(
    "Transmission Cross Coefficient"
)

plt.xlim(
    -8,
    8
)

plt.ylim(
    -8,
    8
)

plt.show()


# =========================================================
# 14. TCC eigendecomposition
#
# TCC v_k = lambda_k v_k
#
# eigenvector  -> coherent mode
# eigenvalue   -> mode importance
# =========================================================

eigenvalues, eigenvectors = np.linalg.eigh(
    TCC
)


# =========================================================
# 15. Sort eigenvalues from large to small
# =========================================================

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


# Very small negative values may appear because of
# floating-point numerical error.

positive_eigenvalues = np.clip(
    eigenvalues,
    0,
    None
)


# =========================================================
# 16. Print largest eigenvalues
# =========================================================

print()
print("=" * 60)
print("TCC Eigenvalues")
print("=" * 60)


number_to_print = min(
    10,
    len(eigenvalues)
)


for k in range(
    number_to_print
):

    print(
        f"Mode {k + 1}:",
        eigenvalues[k]
    )


# =========================================================
# 17. Cumulative contribution
# =========================================================

total_eigenvalue = np.sum(
    positive_eigenvalues
)


if total_eigenvalue > 0:

    cumulative_ratio = (
        np.cumsum(
            positive_eigenvalues
        )
        /
        total_eigenvalue
    )

else:

    cumulative_ratio = np.zeros_like(
        positive_eigenvalues
    )


print()
print("=" * 60)
print("Cumulative Mode Contribution")
print("=" * 60)


for k in range(
    number_to_print
):

    print(
        f"First {k + 1} modes:",
        cumulative_ratio[k] * 100,
        "%"
    )


# =========================================================
# 18. Eigenvalue spectrum
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    np.arange(
        1,
        len(eigenvalues) + 1
    ),
    positive_eigenvalues,
    marker="o"
)

plt.xlabel(
    "Mode Index"
)

plt.ylabel(
    "Eigenvalue"
)

plt.title(
    "TCC Eigenvalue Spectrum"
)

plt.grid()

plt.show()


# =========================================================
# 19. Cumulative contribution plot
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    np.arange(
        1,
        len(cumulative_ratio) + 1
    ),
    cumulative_ratio * 100,
    marker="o"
)

plt.axhline(
    90,
    linestyle="--",
    label="90%"
)

plt.axhline(
    99,
    linestyle="--",
    label="99%"
)

plt.xlabel(
    "Number of Modes"
)

plt.ylabel(
    "Cumulative Contribution (%)"
)

plt.title(
    "Cumulative TCC Mode Contribution"
)

plt.grid()
plt.legend()

plt.show()


# =========================================================
# 20. Visualize first two coherent modes
# =========================================================

mode_1 = eigenvectors[
    :,
    0
]

mode_2 = eigenvectors[
    :,
    1
]


mode_1_display = np.fft.fftshift(
    np.abs(mode_1)
)

mode_2_display = np.fft.fftshift(
    np.abs(mode_2)
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    freq_display,
    mode_1_display,
    label="Mode 1"
)

plt.plot(
    freq_display,
    mode_2_display,
    label="Mode 2"
)

plt.xlim(
    -8,
    8
)

plt.xlabel(
    "Spatial Frequency (1/um)"
)

plt.ylabel(
    "Mode Magnitude"
)

plt.title(
    "First Two Coherent Modes"
)

plt.grid()
plt.legend()

plt.show()

# =========================================================
# 21. Coherent mode reconstruction
#
# Full TCC:
#
# I(x) = q^T TCC q*
#
# After eigendecomposition:
#
# TCC = sum_k lambda_k v_k v_k^H
#
# Therefore:
#
# I(x) = sum_k lambda_k |q^T v_k|^2
# =========================================================


# Number of coherent modes we want to keep
num_modes = 2


# Prepare an empty array
# to store reconstructed intensity
intensity_modes = np.zeros(
    N
)


# =========================================================
# 22. Reconstruct intensity at every x position
# =========================================================

for ix, x_value in enumerate(
    x_fft
):

    # -----------------------------------------
    # Build q at this spatial position
    #
    # q_m =
    # M(f_m) * exp(i 2 pi f_m x)
    # -----------------------------------------

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


    # -----------------------------------------
    # Add contributions from coherent modes
    # -----------------------------------------

    intensity_value = 0.0


    for k in range(
        num_modes
    ):

        # kth eigenvector
        mode_k = (
            eigenvectors[
                :,
                k
            ]
        )


        # kth eigenvalue
        lambda_k = (
            eigenvalues[
                k
            ]
        )


        # -------------------------------------
        # Coherent field of kth mode
        #
        # E_k(x) = q^T v_k
        # -------------------------------------

        E_k = (
            q
            @
            mode_k
        )


        # -------------------------------------
        # Intensity contribution
        #
        # lambda_k * |E_k|^2
        # -------------------------------------

        intensity_value += (
            lambda_k
            *
            np.abs(
                E_k
            ) ** 2
        )


    # -----------------------------------------
    # Match NumPy FFT normalization
    # -----------------------------------------

    intensity_modes[ix] = (
        intensity_value
        /
        (N ** 2)
    )


# =========================================================
# 23. Normalize reconstructed image
# =========================================================

intensity_modes_norm = (
    intensity_modes
    /
    np.max(
        intensity_modes
    )
)


# =========================================================
# 24. Compare with full TCC
# =========================================================

difference_modes = np.abs(
    intensity_tcc_norm
    -
    intensity_modes_norm
)


max_difference_modes = np.max(
    difference_modes
)


mean_difference_modes = np.mean(
    difference_modes
)


print()
print("=" * 60)

print(
    "Full TCC vs Coherent Mode Reconstruction"
)

print("=" * 60)

print(
    "Number of modes =",
    num_modes
)

print(
    "Maximum difference =",
    max_difference_modes
)

print(
    "Mean difference =",
    mean_difference_modes
)


# =========================================================
# 25. Plot:
# Full TCC vs coherent-mode reconstruction
# =========================================================

plt.figure(
    figsize=(9, 5)
)


plt.plot(
    x,
    intensity_tcc_norm,
    label="Full TCC"
)


plt.plot(
    x,
    intensity_modes_norm,
    linestyle="--",
    label=f"First {num_modes} Coherent Modes"
)


plt.xlim(
    -0.5,
    0.5
)


plt.xlabel(
    "x (um)"
)


plt.ylabel(
    "Normalized Intensity"
)


plt.title(
    "Full TCC vs Coherent Mode Reconstruction"
)


plt.grid()

plt.legend()

plt.show()


# =========================================================
# 26. Plot reconstruction error
# =========================================================

plt.figure(
    figsize=(9, 4)
)


plt.plot(
    x,
    difference_modes
)


plt.xlim(
    -0.5,
    0.5
)


plt.xlabel(
    "x (um)"
)


plt.ylabel(
    "|Full TCC - Mode Reconstruction|"
)


plt.title(
    "Coherent Mode Reconstruction Error"
)


plt.grid()

plt.show()
# =========================================================
# 27. Scan number of coherent modes
# =========================================================

max_modes = 8

mode_number_list = []

max_error_list = []

mean_error_list = []


for num_modes in range(
    1,
    max_modes + 1
):

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

            mode_k = (
                eigenvectors[
                    :,
                    k
                ]
            )

            lambda_k = (
                eigenvalues[
                    k
                ]
            )

            E_k = (
                q
                @
                mode_k
            )

            intensity_value += (
                lambda_k
                *
                np.abs(
                    E_k
                ) ** 2
            )


        intensity_modes[ix] = (
            intensity_value
            /
            (N ** 2)
        )


    # -----------------------------------------
    # Normalize reconstructed image
    # -----------------------------------------

    intensity_modes_norm = (
        intensity_modes
        /
        np.max(
            intensity_modes
        )
    )


    # -----------------------------------------
    # Calculate error
    # -----------------------------------------

    difference_modes = np.abs(
        intensity_tcc_norm
        -
        intensity_modes_norm
    )


    max_error = np.max(
        difference_modes
    )


    mean_error = np.mean(
        difference_modes
    )


    mode_number_list.append(
        num_modes
    )

    max_error_list.append(
        max_error
    )

    mean_error_list.append(
        mean_error
    )


    print(
        f"Modes = {num_modes}:",
        "Max error =",
        max_error,
        "Mean error =",
        mean_error
    )
# =========================================================
# 28. Plot reconstruction error vs number of modes
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    mode_number_list,
    max_error_list,
    marker="o",
    label="Maximum Error"
)

plt.plot(
    mode_number_list,
    mean_error_list,
    marker="o",
    label="Mean Error"
)

plt.xlabel(
    "Number of Coherent Modes"
)

plt.ylabel(
    "Reconstruction Error"
)

plt.title(
    "Reconstruction Error vs Number of Modes"
)

plt.grid()

plt.legend()

plt.show()
# =========================================================
# 29. Automatically choose minimum number of modes
# =========================================================

error_threshold = 0.01   # 1%

selected_modes = None

for num_modes, max_error in zip(
    mode_number_list,
    max_error_list
):

    if max_error < error_threshold:

        selected_modes = num_modes

        break


print()
print("=" * 60)
print("Automatic Mode Selection")
print("=" * 60)

print(
    "Error threshold =",
    error_threshold
)

if selected_modes is not None:

    print(
        "Minimum number of modes =",
        selected_modes
    )

else:

    print(
        "No mode number satisfies the error threshold."
    )