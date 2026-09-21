import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Spatial grid
# =========================================================

N = 1024

x = np.linspace(
    -2.0,
    2.0,
    N
)

dx = x[1] - x[0]


# =========================================================
# 2. 1D periodic line-space mask
# =========================================================

period = 0.4      # μm
line_width = 0.2  # μm

position_in_period = np.mod(
    x,
    period
)

mask = (
    position_in_period
    <
    line_width
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
NA = 0.5

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
# 6. Normal illumination pupil
# =========================================================

pupil_normal = (
    np.abs(freq)
    <=
    cutoff
).astype(float)


# =========================================================
# 7. Normal coherent imaging
# =========================================================

filtered_normal = (
    spectrum
    *
    pupil_normal
)

field_normal = np.fft.ifft(
    filtered_normal
)

intensity_normal = (
    np.abs(field_normal) ** 2
)


# =========================================================
# 8. Dipole illumination
# =========================================================

source_shift = 1.5   # 1/μm

source_points = [
    source_shift,
    -source_shift
]

weight = (
    1.0
    /
    len(source_points)
)


intensity_dipole = np.zeros_like(
    x,
    dtype=float
)


# =========================================================
# 9. Loop over two dipole source points
# =========================================================

for source_f in source_points:

    # Shifted pupil
    pupil_shifted = (
        np.abs(
            freq - source_f
        )
        <=
        cutoff
    ).astype(float)


    # Filter spectrum
    filtered = (
        spectrum
        *
        pupil_shifted
    )


    # Back to spatial domain
    field = np.fft.ifft(
        filtered
    )


    # Intensity from this source point
    intensity = (
        np.abs(field) ** 2
    )


    # Weighted incoherent sum
    intensity_dipole += (
        weight
        *
        intensity
    )


# =========================================================
# 10. Normalize only for visualization
# =========================================================

intensity_normal_plot = (
    intensity_normal
    /
    np.max(intensity_normal)
)

intensity_dipole_plot = (
    intensity_dipole
    /
    np.max(intensity_dipole)
)


# =========================================================
# 11. Plot mask
# =========================================================

plt.figure(
    figsize=(10, 4)
)

plt.plot(
    x,
    mask
)

plt.xlabel(
    "x (μm)"
)

plt.ylabel(
    "Mask transmission"
)

plt.title(
    "1D Line-Space Mask"
)

plt.grid()

plt.show()


# =========================================================
# 12. Plot spectrum and pupil coverage
# =========================================================

plt.figure(
    figsize=(10, 5)
)

spectrum_display = np.abs(
    np.fft.fftshift(
        spectrum
    )
)

freq_display = np.fft.fftshift(
    freq
)


plt.plot(
    freq_display,
    spectrum_display
    /
    np.max(spectrum_display),
    label="Mask Spectrum"
)


# Normal pupil range
plt.axvspan(
    -cutoff,
    cutoff,
    alpha=0.2,
    label="Normal Pupil"
)


# Dipole + source
plt.axvspan(
    source_shift - cutoff,
    source_shift + cutoff,
    alpha=0.2,
    label="+ Dipole Pupil"
)


# Dipole - source
plt.axvspan(
    -source_shift - cutoff,
    -source_shift + cutoff,
    alpha=0.2,
    label="- Dipole Pupil"
)


plt.xlim(
    -8,
    8
)

plt.xlabel(
    "Spatial Frequency (1/μm)"
)

plt.ylabel(
    "Normalized Spectrum"
)

plt.title(
    "Spectrum and Pupil Coverage"
)

plt.legend()

plt.grid()

plt.show()


# =========================================================
# 13. Compare aerial images
# =========================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    x,
    mask,
    label="Mask",
    linestyle="--"
)

plt.plot(
    x,
    intensity_normal_plot,
    label="Normal Illumination"
)

plt.plot(
    x,
    intensity_dipole_plot,
    label="Dipole Illumination"
)

plt.xlabel(
    "x (μm)"
)

plt.ylabel(
    "Normalized Intensity"
)

plt.title(
    "Normal vs Dipole Illumination"
)

plt.legend()

plt.grid()

plt.show()

# =========================================================
# 14. Image contrast
# =========================================================

Imax_normal = np.max(intensity_normal_plot)
Imin_normal = np.min(intensity_normal_plot)

contrast_normal = (
    (Imax_normal - Imin_normal)
    /
    (Imax_normal + Imin_normal)
)


Imax_dipole = np.max(intensity_dipole_plot)
Imin_dipole = np.min(intensity_dipole_plot)

contrast_dipole = (
    (Imax_dipole - Imin_dipole)
    /
    (Imax_dipole + Imin_dipole)
)


print(
    "Normal contrast =",
    contrast_normal
)

print(
    "Dipole contrast =",
    contrast_dipole
)