import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Spatial grid
# =========================================================

N = 4096

x = np.linspace(
    -8.0,
    8.0,
    N
)

dx = x[1] - x[0]


# =========================================================
# 2. Periodic line-space mask
# =========================================================

period = 0.4       # μm
line_width = 0.2   # μm


# Make the center line symmetric around x = 0
position_in_period = (
    np.mod(
        x + period / 2,
        period
    )
    - period / 2
)

mask = (
    np.abs(position_in_period)
    <=
    line_width / 2
).astype(float)


# =========================================================
# 3. Frequency grid
# =========================================================

freq = np.fft.fftfreq(
    N,
    d=dx
)


# =========================================================
# 4. Optical parameters
# =========================================================

wavelength = 0.193   # μm
NA = 0.85

cutoff = (
    NA / wavelength
)

print(
    "Cutoff frequency =",
    cutoff,
    "1/μm"
)


# =========================================================
# 5. Mask spectrum
# =========================================================

spectrum = np.fft.fft(
    mask
)


# =========================================================
# 6. Defocus pupil function
# =========================================================

def make_defocus_pupil(
    freq,
    cutoff,
    defocus_strength
):

    # -------------------------------------
    # Aperture
    # -------------------------------------

    aperture = (
        np.abs(freq)
        <=
        cutoff
    ).astype(float)


    # -------------------------------------
    # Defocus phase
    #
    # This is a simplified teaching model.
    # defocus_strength is dimensionless here.
    # -------------------------------------

    normalized_frequency = (
        freq / cutoff
    )

    phase = (
        defocus_strength
        *
        normalized_frequency**2
    )


    # -------------------------------------
    # Complex pupil
    # -------------------------------------

    pupil = (
        aperture
        *
        np.exp(
            1j * phase
        )
    )

    return pupil


# =========================================================
# 7. Fourier imaging function
# =========================================================

def imaging_with_defocus(
    spectrum,
    freq,
    cutoff,
    defocus_strength
):

    # Build defocused pupil
    pupil = make_defocus_pupil(
        freq,
        cutoff,
        defocus_strength
    )


    # Optical filtering
    filtered_spectrum = (
        spectrum
        *
        pupil
    )


    # Back to spatial domain
    field = np.fft.ifft(
        filtered_spectrum
    )


    # Complex field -> intensity
    intensity = (
        np.abs(field) ** 2
    )


    # Normalize only for shape comparison
    intensity = (
        intensity
        /
        np.max(intensity)
    )


    return intensity


# =========================================================
# 8. Defocus values
# =========================================================

defocus_list = [
    0.0,
    1.0,
    2.0,
    3.0
]


# =========================================================
# 9. Calculate aerial images
# =========================================================

intensity_list = []


for defocus_strength in defocus_list:

    intensity = imaging_with_defocus(
        spectrum,
        freq,
        cutoff,
        defocus_strength
    )

    intensity_list.append(
        intensity
    )


# =========================================================
# 10. Plot mask
# =========================================================

plt.figure(
    figsize=(10, 4)
)

plt.plot(
    x,
    mask
)

plt.xlim(
    -0.5,
    0.5
)

plt.xlabel(
    "x (μm)"
)

plt.ylabel(
    "Mask Transmission"
)

plt.title(
    "Periodic Line-Space Mask"
)

plt.grid()

plt.show()


# =========================================================
# 11. Plot aerial images
# =========================================================

plt.figure(
    figsize=(10, 5)
)


for defocus_strength, intensity in zip(
    defocus_list,
    intensity_list
):

    plt.plot(
        x,
        intensity,
        label=(
            f"Defocus = "
            f"{defocus_strength}"
        )
    )


# Threshold reference line
threshold = 0.5

plt.axhline(
    threshold,
    linestyle="--",
    label="Threshold = 0.5"
)


# Zoom into the central feature
plt.xlim(
    -0.4,
    0.4
)

plt.ylim(
    0,
    1.05
)

plt.xlabel(
    "x (μm)"
)

plt.ylabel(
    "Normalized Intensity"
)

plt.title(
    "Effect of Defocus on Aerial Image"
)

plt.legend()

plt.grid()

plt.show()


# =========================================================
# 12. Plot pupil phase
# =========================================================

plt.figure(
    figsize=(10, 5)
)


for defocus_strength in defocus_list:

    pupil = make_defocus_pupil(
        freq,
        cutoff,
        defocus_strength
    )

    pupil_shifted = np.fft.fftshift(
        pupil
    )

    freq_shifted = np.fft.fftshift(
        freq
    )

    phase_shifted = np.angle(
        pupil_shifted
    )

    plt.plot(
        freq_shifted,
        phase_shifted,
        label=(
            f"Defocus = "
            f"{defocus_strength}"
        )
    )


plt.xlim(
    -cutoff * 1.2,
    cutoff * 1.2
)

plt.xlabel(
    "Spatial Frequency (1/μm)"
)

plt.ylabel(
    "Pupil Phase (rad)"
)

plt.title(
    "Defocus Phase Across the Pupil"
)

plt.legend()

plt.grid()

plt.show()