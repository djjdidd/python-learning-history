import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic parameters
# =========================================================

N = 4096

x = np.linspace(
    -8.0,
    8.0,
    N
)

dx = x[1] - x[0]

period = 0.20          # um
line_width = 0.10      # um

wavelength = 0.193     # um
NA = 0.85

cutoff = (
    NA
    /
    wavelength
)


print(
    "Cutoff frequency =",
    cutoff,
    "1/um"
)

print(
    "First diffraction order =",
    1 / period,
    "1/um"
)


# =========================================================
# 2. Build centered periodic mask
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
    np.abs(
        position_in_period
    )
    <= line_width / 2
).astype(float)


# =========================================================
# 3. Frequency grid
# =========================================================

freq = np.fft.fftfreq(
    N,
    d=dx
)


# =========================================================
# 4. FFT of mask
# =========================================================

spectrum = np.fft.fft(
    mask
)


# =========================================================
# 5. Normal illumination
# =========================================================

pupil_normal = (
    np.abs(freq)
    <= cutoff
).astype(float)


filtered_spectrum_normal = (
    spectrum
    *
    pupil_normal
)


field_normal = np.fft.ifft(
    filtered_spectrum_normal
)


intensity_normal_raw = (
    np.abs(field_normal) ** 2
)


intensity_normal = (
    intensity_normal_raw
    /
    np.max(intensity_normal_raw)
)


# =========================================================
# 6. Dipole illumination
# =========================================================

source_shift = 1.5     # 1/um


# Right source point
pupil_right = (
    np.abs(
        freq - source_shift
    )
    <= cutoff
).astype(float)


filtered_spectrum_right = (
    spectrum
    *
    pupil_right
)


field_right = np.fft.ifft(
    filtered_spectrum_right
)


intensity_right = (
    np.abs(field_right) ** 2
)


# Left source point
pupil_left = (
    np.abs(
        freq + source_shift
    )
    <= cutoff
).astype(float)


filtered_spectrum_left = (
    spectrum
    *
    pupil_left
)


field_left = np.fft.ifft(
    filtered_spectrum_left
)


intensity_left = (
    np.abs(field_left) ** 2
)


# =========================================================
# 7. Incoherent intensity sum
# =========================================================

intensity_dipole_raw = (
    0.5 * intensity_right
    +
    0.5 * intensity_left
)


intensity_dipole = (
    intensity_dipole_raw
    /
    np.max(intensity_dipole_raw)
)


# =========================================================
# 8. Contrast function
# =========================================================

def image_contrast(
    intensity
):

    Imax = np.max(
        intensity
    )

    Imin = np.min(
        intensity
    )

    contrast = (
        (Imax - Imin)
        /
        (Imax + Imin)
    )

    return contrast


# =========================================================
# 9. Calculate contrast
# =========================================================

contrast_normal = (
    image_contrast(
        intensity_normal
    )
)


contrast_dipole = (
    image_contrast(
        intensity_dipole
    )
)


print()
print(
    "=" * 60
)

print(
    "Image Contrast"
)

print(
    "=" * 60
)

print(
    "Normal contrast =",
    contrast_normal
)

print(
    "Dipole contrast =",
    contrast_dipole
)


# =========================================================
# 10. Plot mask
# =========================================================

plt.figure(
    figsize=(9, 4)
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
    "x (um)"
)

plt.ylabel(
    "Mask transmission"
)

plt.title(
    "Periodic Mask"
)

plt.grid()

plt.show()


# =========================================================
# 11. Plot mask spectrum
# =========================================================

freq_shifted = np.fft.fftshift(
    freq
)

spectrum_shifted = np.fft.fftshift(
    np.abs(spectrum)
)


spectrum_shifted = (
    spectrum_shifted
    /
    np.max(spectrum_shifted)
)


plt.figure(
    figsize=(9, 4)
)

plt.plot(
    freq_shifted,
    spectrum_shifted
)

plt.axvline(
    cutoff,
    linestyle="--",
    label="+cutoff"
)

plt.axvline(
    -cutoff,
    linestyle="--",
    label="-cutoff"
)

plt.xlim(
    -15,
    15
)

plt.xlabel(
    "Spatial Frequency (1/um)"
)

plt.ylabel(
    "Normalized Spectrum"
)

plt.title(
    "Mask Spectrum and Optical Cutoff"
)

plt.grid()

plt.legend()

plt.show()


# =========================================================
# 12. Plot pupils
# =========================================================

plt.figure(
    figsize=(9, 4)
)

plt.plot(
    freq_shifted,
    np.fft.fftshift(
        pupil_normal
    ),
    label="Normal pupil"
)

plt.plot(
    freq_shifted,
    np.fft.fftshift(
        pupil_right
    ),
    label="Right source pupil"
)

plt.plot(
    freq_shifted,
    np.fft.fftshift(
        pupil_left
    ),
    label="Left source pupil"
)

plt.xlim(
    -10,
    10
)

plt.xlabel(
    "Spatial Frequency (1/um)"
)

plt.ylabel(
    "Pupil transmission"
)

plt.title(
    "Normal and Shifted Pupils"
)

plt.grid()

plt.legend()

plt.show()


# =========================================================
# 13. Compare aerial images
# =========================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    x,
    intensity_normal,
    label="Normal illumination"
)

plt.plot(
    x,
    intensity_dipole,
    label="Dipole illumination"
)

plt.xlim(
    -0.5,
    0.5
)

plt.xlabel(
    "x (um)"
)

plt.ylabel(
    "Normalized intensity"
)

plt.title(
    "Normal vs Dipole Illumination"
)

plt.grid()

plt.legend()

plt.show()
# =========================================================
# 14. Scan source shift
# =========================================================

source_shift_scan = np.linspace(
    0.0,
    3.0,
    121
)

contrast_scan = []


for source_shift in source_shift_scan:

    # Right source
    pupil_right = (
        np.abs(
            freq - source_shift
        )
        <= cutoff
    ).astype(float)

    field_right = np.fft.ifft(
        spectrum
        *
        pupil_right
    )

    intensity_right = (
        np.abs(field_right) ** 2
    )


    # Left source
    pupil_left = (
        np.abs(
            freq + source_shift
        )
        <= cutoff
    ).astype(float)

    field_left = np.fft.ifft(
        spectrum
        *
        pupil_left
    )

    intensity_left = (
        np.abs(field_left) ** 2
    )


    # Incoherent sum
    intensity_dipole = (
        0.5 * intensity_right
        +
        0.5 * intensity_left
    )

    # Normalize
    intensity_dipole = (
        intensity_dipole
        /
        np.max(intensity_dipole)
    )


    # Contrast
    contrast = image_contrast(
        intensity_dipole
    )

    contrast_scan.append(
        contrast
    )


contrast_scan = np.array(
    contrast_scan
)
# =========================================================
# 15. Find best source shift
# =========================================================

best_index = np.argmax(
    contrast_scan
)

best_source_shift = (
    source_shift_scan[
        best_index
    ]
)

best_contrast = (
    contrast_scan[
        best_index
    ]
)


print()
print("=" * 60)

print(
    "Source Optimization"
)

print("=" * 60)

print(
    "Best source shift =",
    best_source_shift,
    "1/um"
)

print(
    "Best dipole contrast =",
    best_contrast
)
# =========================================================
# 16. Plot contrast vs source shift
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    source_shift_scan,
    contrast_scan
)

plt.axvline(
    best_source_shift,
    linestyle="--",
    label=f"Best shift = {best_source_shift:.3f}"
)

plt.xlabel(
    "Source Shift (1/um)"
)

plt.ylabel(
    "Image Contrast"
)

plt.title(
    "Contrast vs Source Shift"
)

plt.grid()

plt.legend()

plt.show()