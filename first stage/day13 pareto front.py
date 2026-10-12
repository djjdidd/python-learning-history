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

period = 0.30       # μm
line_width = 0.15   # μm

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

print(
    "First diffraction order =",
    1 / period,
    "1/μm"
)


# =========================================================
# 5. Mask spectrum
# =========================================================

spectrum = np.fft.fft(
    mask
)


# =========================================================
# 6. Function:
# calculate dipole aerial image
# =========================================================

def dipole_image(
    source_shift
):

    source_points = [
        source_shift,
        -source_shift
    ]

    intensity_total = np.zeros_like(
        x,
        dtype=float
    )

    for source_f in source_points:

        pupil_shifted = (
            np.abs(
                freq - source_f
            )
            <=
            cutoff
        ).astype(float)

        filtered = (
            spectrum
            *
            pupil_shifted
        )

        field = np.fft.ifft(
            filtered
        )

        intensity = (
            np.abs(field) ** 2
        )

        intensity_total += (
            0.5
            *
            intensity
        )

    # Normalize only for metric comparison
    intensity_total = (
        intensity_total
        /
        np.max(intensity_total)
    )

    return intensity_total


# =========================================================
# 7. Function:
# calculate contrast
# =========================================================

def calculate_contrast(
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
# 8. Function:
# calculate NILS
# =========================================================

def calculate_nils(
    intensity
):

    gradient = np.gradient(
        intensity,
        x
    )

    edge_position = (
        line_width
    )

    edge_index = np.argmin(
        np.abs(
            x - edge_position
        )
    )

    I_edge = intensity[
        edge_index
    ]

    slope = np.abs(
        gradient[
            edge_index
        ]
    )

    NILS = (
        line_width
        /
        I_edge
        *
        slope
    )

    return NILS


# =========================================================
# 9. Scan source shift
# =========================================================

source_shift_scan = np.linspace(
    0.0,
    2.0,
    81
)

contrast_scan = []
nils_scan = []


for source_shift in source_shift_scan:

    intensity = dipole_image(
        source_shift
    )

    contrast = calculate_contrast(
        intensity
    )

    nils = calculate_nils(
        intensity
    )

    contrast_scan.append(
        contrast
    )

    nils_scan.append(
        nils
    )


contrast_scan = np.array(
    contrast_scan
)

nils_scan = np.array(
    nils_scan
)


# =========================================================
# 10. Find best contrast source
# =========================================================

best_contrast_index = np.argmax(
    contrast_scan
)

best_contrast_shift = (
    source_shift_scan[
        best_contrast_index
    ]
)

best_contrast_value = (
    contrast_scan[
        best_contrast_index
    ]
)


# =========================================================
# 11. Find best NILS source
# =========================================================

best_nils_index = np.argmax(
    nils_scan
)

best_nils_shift = (
    source_shift_scan[
        best_nils_index
    ]
)

best_nils_value = (
    nils_scan[
        best_nils_index
    ]
)


# =========================================================
# 12. Find Pareto front
# =========================================================

num_points = len(
    source_shift_scan
)

is_pareto = np.ones(
    num_points,
    dtype=bool
)


for i in range(
    num_points
):

    for j in range(
        num_points
    ):

        if i == j:
            continue

        better_or_equal_contrast = (
            contrast_scan[j]
            >=
            contrast_scan[i]
        )

        better_or_equal_nils = (
            nils_scan[j]
            >=
            nils_scan[i]
        )

        strictly_better = (
            (contrast_scan[j] > contrast_scan[i])
            or
            (nils_scan[j] > nils_scan[i])
        )

        if (
            better_or_equal_contrast
            and
            better_or_equal_nils
            and
            strictly_better
        ):

            is_pareto[i] = False

            break


# =========================================================
# 13. Extract Pareto points
# =========================================================

pareto_shift = (
    source_shift_scan[
        is_pareto
    ]
)

pareto_contrast = (
    contrast_scan[
        is_pareto
    ]
)

pareto_nils = (
    nils_scan[
        is_pareto
    ]
)


# =========================================================
# 14. Print results
# =========================================================

print()
print("=" * 70)

print(
    "Best source for contrast =",
    best_contrast_shift
)

print(
    "Maximum contrast =",
    best_contrast_value
)

print()

print(
    "Best source for NILS =",
    best_nils_shift
)

print(
    "Maximum NILS =",
    best_nils_value
)

print("=" * 70)


print()
print("Pareto-optimal source shifts:")

for shift, contrast, nils in zip(
    pareto_shift,
    pareto_contrast,
    pareto_nils
):

    print(
        f"shift = {shift:.3f}, "
        f"contrast = {contrast:.6f}, "
        f"NILS = {nils:.6f}"
    )


# =========================================================
# 15. Plot:
# source shift -> metrics
# =========================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    source_shift_scan,
    contrast_scan,
    label="Contrast"
)

plt.plot(
    source_shift_scan,
    nils_scan,
    label="NILS"
)

plt.xlabel(
    "Source Shift (1/μm)"
)

plt.ylabel(
    "Metric Value"
)

plt.title(
    "Contrast and NILS vs Source Shift"
)

plt.legend()

plt.grid()

plt.show()


# =========================================================
# 16. Plot Pareto front
# =========================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    contrast_scan,
    nils_scan,
    label="All source shifts"
)

plt.scatter(
    pareto_contrast,
    pareto_nils,
    marker="x",
    s=100,
    label="Pareto-optimal"
)

for shift, contrast, nils in zip(
    pareto_shift,
    pareto_contrast,
    pareto_nils
):

    plt.text(
        contrast,
        nils,
        f"{shift:.2f}"
    )

plt.xlabel(
    "Image Contrast"
)

plt.ylabel(
    "NILS"
)

plt.title(
    "Pareto Front"
)

plt.legend()

plt.grid()

plt.show()