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

threshold = 0.5

cutoff = NA / wavelength


print("Cutoff frequency =", cutoff, "1/um")
print("First diffraction order =", 1 / period, "1/um")


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
    np.abs(position_in_period)
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
# 4. Fourier spectrum of mask
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

field_normal = np.fft.ifft(
    spectrum
    *
    pupil_normal
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
# 6. Contrast function
# =========================================================

def image_contrast(intensity):

    Imax = np.max(intensity)
    Imin = np.min(intensity)

    return (
        (Imax - Imin)
        /
        (Imax + Imin)
    )


contrast_normal = image_contrast(
    intensity_normal
)


# =========================================================
# 7. Manual dipole example
# =========================================================

source_shift_manual = 1.5


pupil_right_manual = (
    np.abs(
        freq - source_shift_manual
    )
    <= cutoff
).astype(float)

pupil_left_manual = (
    np.abs(
        freq + source_shift_manual
    )
    <= cutoff
).astype(float)


field_right_manual = np.fft.ifft(
    spectrum
    *
    pupil_right_manual
)

field_left_manual = np.fft.ifft(
    spectrum
    *
    pupil_left_manual
)


intensity_right_manual = (
    np.abs(field_right_manual) ** 2
)

intensity_left_manual = (
    np.abs(field_left_manual) ** 2
)


# Different source points:
# add INTENSITY, not field
intensity_dipole_manual_raw = (
    0.5 * intensity_right_manual
    +
    0.5 * intensity_left_manual
)

intensity_dipole_manual = (
    intensity_dipole_manual_raw
    /
    np.max(intensity_dipole_manual_raw)
)


contrast_dipole_manual = image_contrast(
    intensity_dipole_manual
)


print()
print("=" * 60)
print("Normal vs Manual Dipole")
print("=" * 60)

print(
    "Normal contrast =",
    contrast_normal
)

print(
    "Manual dipole contrast =",
    contrast_dipole_manual
)


# =========================================================
# 8. Plot normal vs manual dipole
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
    intensity_dipole_manual,
    label="Dipole illumination"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Threshold"
)

plt.xlim(
    -0.5,
    0.5
)

plt.xlabel("x (um)")
plt.ylabel("Normalized intensity")

plt.title(
    "Normal vs Dipole Illumination"
)

plt.grid()
plt.legend()

plt.show()


# =========================================================
# 9. Source shift scan for contrast
# =========================================================

source_shift_scan = np.linspace(
    0.0,
    3.0,
    121
)

contrast_scan = []


for source_shift in source_shift_scan:

    pupil_right = (
        np.abs(
            freq - source_shift
        )
        <= cutoff
    ).astype(float)

    pupil_left = (
        np.abs(
            freq + source_shift
        )
        <= cutoff
    ).astype(float)


    field_right = np.fft.ifft(
        spectrum
        *
        pupil_right
    )

    field_left = np.fft.ifft(
        spectrum
        *
        pupil_left
    )


    intensity_right = (
        np.abs(field_right) ** 2
    )

    intensity_left = (
        np.abs(field_left) ** 2
    )


    intensity_dipole = (
        0.5 * intensity_right
        +
        0.5 * intensity_left
    )


    intensity_dipole = (
        intensity_dipole
        /
        np.max(intensity_dipole)
    )


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
# 10. Best source shift for contrast
# =========================================================

best_contrast_index = np.argmax(
    contrast_scan
)

best_source_shift = (
    source_shift_scan[
        best_contrast_index
    ]
)

best_contrast = (
    contrast_scan[
        best_contrast_index
    ]
)


print()
print("=" * 60)
print("Contrast Optimization")
print("=" * 60)

print(
    "Best source shift for contrast =",
    best_source_shift,
    "1/um"
)

print(
    "Best contrast =",
    best_contrast
)


# =========================================================
# 11. Edge interpolation
# =========================================================

def interpolate_edge(
    x1,
    x2,
    I1,
    I2,
    threshold
):

    if I2 == I1:
        return (
            x1 + x2
        ) / 2

    return (
        x1
        +
        (
            threshold - I1
        )
        /
        (
            I2 - I1
        )
        *
        (
            x2 - x1
        )
    )


# =========================================================
# 12. General value interpolation
# =========================================================

def interpolate_value(
    x1,
    x2,
    y1,
    y2,
    x_target
):

    if x2 == x1:
        return (
            y1 + y2
        ) / 2

    ratio = (
        (x_target - x1)
        /
        (x2 - x1)
    )

    return (
        y1
        +
        ratio
        *
        (y2 - y1)
    )


# =========================================================
# 13. Pixel-level NILS
# =========================================================

def calculate_nils_pixel(
    x,
    intensity,
    threshold,
    target_cd
):

    binary = (
        intensity
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


    left_candidates = (
        left_edges[
            left_edges < center_index
        ]
    )

    right_candidates = (
        right_edges[
            right_edges > center_index
        ]
    )


    if (
        len(left_candidates) == 0
        or
        len(right_candidates) == 0
    ):

        return np.nan


    left_index = (
        left_candidates[-1]
    )

    right_index = (
        right_candidates[0]
    )


    dI_dx = np.gradient(
        intensity,
        x
    )


    slope_left = np.abs(
        dI_dx[
            left_index
        ]
    )

    slope_right = np.abs(
        dI_dx[
            right_index
        ]
    )


    slope_edge = (
        slope_left
        +
        slope_right
    ) / 2


    I_edge = threshold


    nils = (
        target_cd
        /
        I_edge
        *
        slope_edge
    )


    return nils


# =========================================================
# 14. Sub-pixel NILS
# =========================================================

def calculate_nils_subpixel(
    x,
    intensity,
    threshold,
    target_cd
):

    # -----------------------------------------
    # Find binary threshold crossings
    # -----------------------------------------

    binary = (
        intensity
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


    # -----------------------------------------
    # Select center feature
    # -----------------------------------------

    center_index = np.argmin(
        np.abs(x)
    )


    left_candidates = (
        left_edges[
            left_edges < center_index
        ]
    )

    right_candidates = (
        right_edges[
            right_edges > center_index
        ]
    )


    if (
        len(left_candidates) == 0
        or
        len(right_candidates) == 0
    ):

        return np.nan


    left_index = (
        left_candidates[-1]
    )

    right_index = (
        right_candidates[0]
    )


    # -----------------------------------------
    # Sub-pixel edge positions
    # -----------------------------------------

    x_left = interpolate_edge(
        x[left_index],
        x[left_index + 1],
        intensity[left_index],
        intensity[left_index + 1],
        threshold
    )


    x_right = interpolate_edge(
        x[right_index],
        x[right_index + 1],
        intensity[right_index],
        intensity[right_index + 1],
        threshold
    )


    # -----------------------------------------
    # Calculate image gradient
    # -----------------------------------------

    dI_dx = np.gradient(
        intensity,
        x
    )


    # -----------------------------------------
    # Interpolate gradient at true edge position
    # -----------------------------------------

    slope_left = interpolate_value(
        x[left_index],
        x[left_index + 1],
        dI_dx[left_index],
        dI_dx[left_index + 1],
        x_left
    )


    slope_right = interpolate_value(
        x[right_index],
        x[right_index + 1],
        dI_dx[right_index],
        dI_dx[right_index + 1],
        x_right
    )


    slope_left = np.abs(
        slope_left
    )

    slope_right = np.abs(
        slope_right
    )


    slope_edge = (
        slope_left
        +
        slope_right
    ) / 2


    I_edge = threshold


    nils = (
        target_cd
        /
        I_edge
        *
        slope_edge
    )


    return nils


# =========================================================
# 15. Normal NILS
# =========================================================

nils_normal_pixel = calculate_nils_pixel(
    x,
    intensity_normal,
    threshold,
    line_width
)


nils_normal_subpixel = calculate_nils_subpixel(
    x,
    intensity_normal,
    threshold,
    line_width
)


print()
print("=" * 60)
print("Normal Illumination NILS")
print("=" * 60)

print(
    "Pixel-level NILS =",
    nils_normal_pixel
)

print(
    "Sub-pixel NILS =",
    nils_normal_subpixel
)


# =========================================================
# 16. Scan source shift for both NILS methods
# =========================================================

nils_pixel_scan = []

nils_subpixel_scan = []


for source_shift in source_shift_scan:

    # Right source
    pupil_right = (
        np.abs(
            freq - source_shift
        )
        <= cutoff
    ).astype(float)


    # Left source
    pupil_left = (
        np.abs(
            freq + source_shift
        )
        <= cutoff
    ).astype(float)


    field_right = np.fft.ifft(
        spectrum
        *
        pupil_right
    )


    field_left = np.fft.ifft(
        spectrum
        *
        pupil_left
    )


    intensity_right = (
        np.abs(field_right) ** 2
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


    intensity_dipole = (
        intensity_dipole
        /
        np.max(intensity_dipole)
    )


    # Pixel-level NILS
    nils_pixel = calculate_nils_pixel(
        x,
        intensity_dipole,
        threshold,
        line_width
    )


    # Sub-pixel NILS
    nils_subpixel = calculate_nils_subpixel(
        x,
        intensity_dipole,
        threshold,
        line_width
    )


    nils_pixel_scan.append(
        nils_pixel
    )

    nils_subpixel_scan.append(
        nils_subpixel
    )


nils_pixel_scan = np.array(
    nils_pixel_scan
)

nils_subpixel_scan = np.array(
    nils_subpixel_scan
)


# =========================================================
# 17. Best pixel-level NILS
# =========================================================

best_nils_pixel_index = np.nanargmax(
    nils_pixel_scan
)

best_nils_pixel_shift = (
    source_shift_scan[
        best_nils_pixel_index
    ]
)

best_nils_pixel = (
    nils_pixel_scan[
        best_nils_pixel_index
    ]
)


# =========================================================
# 18. Best sub-pixel NILS
# =========================================================

best_nils_subpixel_index = np.nanargmax(
    nils_subpixel_scan
)

best_nils_subpixel_shift = (
    source_shift_scan[
        best_nils_subpixel_index
    ]
)

best_nils_subpixel = (
    nils_subpixel_scan[
        best_nils_subpixel_index
    ]
)


print()
print("=" * 60)
print("NILS Optimization")
print("=" * 60)

print(
    "Best source shift for contrast =",
    best_source_shift,
    "1/um"
)

print(
    "Best contrast =",
    best_contrast
)

print()

print(
    "Best source shift for pixel-level NILS =",
    best_nils_pixel_shift,
    "1/um"
)

print(
    "Best pixel-level NILS =",
    best_nils_pixel
)

print()

print(
    "Best source shift for sub-pixel NILS =",
    best_nils_subpixel_shift,
    "1/um"
)

print(
    "Best sub-pixel NILS =",
    best_nils_subpixel
)


# =========================================================
# 19. Contrast vs source shift
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
    label=f"Best contrast = {best_source_shift:.3f}"
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


# =========================================================
# 20. Pixel vs sub-pixel NILS
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    source_shift_scan,
    nils_pixel_scan,
    label="Pixel-level NILS"
)

plt.plot(
    source_shift_scan,
    nils_subpixel_scan,
    label="Sub-pixel NILS"
)

plt.axvline(
    best_nils_subpixel_shift,
    linestyle="--",
    label=f"Best sub-pixel shift = {best_nils_subpixel_shift:.3f}"
)

plt.xlabel(
    "Source Shift (1/um)"
)

plt.ylabel(
    "NILS"
)

plt.title(
    "Pixel vs Sub-pixel NILS"
)

plt.grid()
plt.legend()

plt.show()


# =========================================================
# 21. Recalculate aerial image at best contrast source
# =========================================================

pupil_right_contrast = (
    np.abs(
        freq - best_source_shift
    )
    <= cutoff
).astype(float)


pupil_left_contrast = (
    np.abs(
        freq + best_source_shift
    )
    <= cutoff
).astype(float)


field_right_contrast = np.fft.ifft(
    spectrum
    *
    pupil_right_contrast
)


field_left_contrast = np.fft.ifft(
    spectrum
    *
    pupil_left_contrast
)


intensity_best_contrast = (
    0.5 * np.abs(field_right_contrast) ** 2
    +
    0.5 * np.abs(field_left_contrast) ** 2
)


intensity_best_contrast = (
    intensity_best_contrast
    /
    np.max(intensity_best_contrast)
)


# =========================================================
# 22. Recalculate aerial image at best sub-pixel NILS source
# =========================================================

pupil_right_nils = (
    np.abs(
        freq - best_nils_subpixel_shift
    )
    <= cutoff
).astype(float)


pupil_left_nils = (
    np.abs(
        freq + best_nils_subpixel_shift
    )
    <= cutoff
).astype(float)


field_right_nils = np.fft.ifft(
    spectrum
    *
    pupil_right_nils
)


field_left_nils = np.fft.ifft(
    spectrum
    *
    pupil_left_nils
)


intensity_best_nils = (
    0.5 * np.abs(field_right_nils) ** 2
    +
    0.5 * np.abs(field_left_nils) ** 2
)


intensity_best_nils = (
    intensity_best_nils
    /
    np.max(intensity_best_nils)
)


# =========================================================
# 23. Compare optimized aerial images
# =========================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    x,
    intensity_best_contrast,
    label="Best Contrast Source"
)

plt.plot(
    x,
    intensity_best_nils,
    label="Best Sub-pixel NILS Source"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Threshold"
)

plt.xlim(
    -0.3,
    0.3
)

plt.xlabel(
    "x (um)"
)

plt.ylabel(
    "Normalized Intensity"
)

plt.title(
    "Best Contrast vs Best NILS Source"
)

plt.grid()
plt.legend()

plt.show()