import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic parameters
# =========================================================

N = 128

L = 6.4                 # total window, um
dx = L / N

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

print(
    "dx =",
    dx,
    "um"
)

print(
    "Frequency resolution =",
    1 / L,
    "1/um"
)

print(
    "Cutoff =",
    cutoff,
    "1/um"
)

print(
    "First diffraction order =",
    1 / period,
    "1/um"
)


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
# 3. Fourier grid
# =========================================================

freq = np.fft.fftfreq(
    N,
    d=dx
)

spectrum = np.fft.fft(
    mask
)


# =========================================================
# 4. Define a two-point dipole source
# =========================================================

# Frequency resolution:
# df = 1/L = 0.15625 1/um
#
# Use 4 frequency bins:
# source_shift = 4 * 0.15625 = 0.625 1/um

source_shift = 0.625

source_positions = [
    -source_shift,
    +source_shift
]

source_weights = [
    0.5,
    0.5
]


# =========================================================
# 5. Function: shifted pupil
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
# 6. ABbe formulation
#
# Each source point:
#
# spectrum -> shifted pupil -> field -> intensity
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

    field = np.fft.ifft(
        spectrum
        *
        pupil
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
    # pupil[m] * pupil*[n]

    TCC += (
        weight
        *
        np.outer(
            pupil,
            np.conj(pupil)
        )
    )


# =========================================================
# 8. Hopkins / TCC image reconstruction
# =========================================================

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
# 9. Normalize for visual comparison
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
# 10. Calculate numerical difference
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

print(
    "Abbe vs Hopkins/TCC"
)

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
# 11. Plot aerial images
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
    "Abbe vs Hopkins/TCC"
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