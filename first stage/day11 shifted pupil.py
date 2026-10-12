import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic function
# =========================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# =========================================================
# 2. Spatial grid
# =========================================================

N = 101

x = np.linspace(-1.0, 1.0, N)   # unit: μm
y = np.linspace(-1.0, 1.0, N)

X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]


# =========================================================
# 3. Target pattern
# =========================================================

target = (
    (np.abs(X) <= 0.30) &
    (np.abs(Y) <= 0.20)
).astype(float)


# =========================================================
# 4. Initial latent variable Z
# =========================================================

Z_initial = np.where(
    target == 1,
    2.0,
    -2.0
).astype(float)


# Continuous mask
mask = sigmoid(
    Z_initial
)


# =========================================================
# 5. Frequency grid
# =========================================================

fx = np.fft.fftfreq(
    N,
    d=dx
)

fy = np.fft.fftfreq(
    N,
    d=dy
)

FX, FY = np.meshgrid(
    fx,
    fy
)


# =========================================================
# 6. Optical parameters
# =========================================================

wavelength = 0.193   # μm = 193 nm

NA = 0.85

cutoff = (
    NA / wavelength
)

print(
    "Cutoff spatial frequency =",
    cutoff,
    "1/μm"
)


# =========================================================
# 7. Define 4-point illumination source
# =========================================================

source_shift = 1.5   # unit: 1/μm

source_points = [

    # Right
    (
        source_shift,
        0.0
    ),

    # Left
    (
        -source_shift,
        0.0
    ),

    # Up
    (
        0.0,
        source_shift
    ),

    # Down
    (
        0.0,
        -source_shift
    )
]


# =========================================================
# 8. Shifted pupil function
# =========================================================

def make_shifted_pupil(
    source_fx,
    source_fy,
    NA,
    wavelength
):

    # Pupil cutoff
    cutoff = (
        NA / wavelength
    )


    # Distance from shifted pupil center
    shifted_radius = np.sqrt(

        (FX - source_fx) ** 2

        +

        (FY - source_fy) ** 2
    )


    # Circular shifted pupil
    pupil_shifted = (
        shifted_radius <= cutoff
    ).astype(float)


    return pupil_shifted


# =========================================================
# 9. Mask spectrum
# =========================================================

spectrum = np.fft.fft2(
    mask
)


# =========================================================
# 10. Prepare storage
# =========================================================

pupil_list = []

intensity_list = []

field_list = []


# Final partially coherent intensity
intensity_total = np.zeros_like(
    mask,
    dtype=float
)


# Equal weight for all source points
weight = (
    1.0
    /
    len(source_points)
)


print(
    "Number of source points =",
    len(source_points)
)

print(
    "Weight of each source point =",
    weight
)


# =========================================================
# 11. Loop over source points
# =========================================================

for source_fx, source_fy in source_points:


    # -----------------------------------------------------
    # Step 1:
    # Build shifted pupil
    # -----------------------------------------------------

    pupil_s = make_shifted_pupil(
        source_fx,
        source_fy,
        NA,
        wavelength
    )


    pupil_list.append(
        pupil_s
    )


    # -----------------------------------------------------
    # Step 2:
    # Filter mask spectrum
    # -----------------------------------------------------

    filtered_spectrum = (
        spectrum
        *
        pupil_s
    )


    # -----------------------------------------------------
    # Step 3:
    # Back to spatial domain
    # -----------------------------------------------------

    field_s = np.fft.ifft2(
        filtered_spectrum
    )


    field_list.append(
        field_s
    )


    # -----------------------------------------------------
    # Step 4:
    # Complex field -> intensity
    # -----------------------------------------------------

    intensity_s = (
        np.abs(field_s) ** 2
    )


    intensity_list.append(
        intensity_s
    )


    # -----------------------------------------------------
    # Step 5:
    # Add weighted intensity
    # -----------------------------------------------------

    intensity_total += (
        weight
        *
        intensity_s
    )


# =========================================================
# 12. Print source information
# =========================================================

print()

print("=" * 60)

for i, (
    source_fx,
    source_fy
) in enumerate(
    source_points
):

    print(
        f"Source {i + 1}: "
        f"({source_fx:.2f}, "
        f"{source_fy:.2f}) 1/μm"
    )

print("=" * 60)


# =========================================================
# 13. Plot target and mask
# =========================================================

plt.figure(
    figsize=(10, 4)
)


plt.subplot(
    1,
    2,
    1
)

plt.imshow(
    target,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Target"
)

plt.xlabel(
    "x (μm)"
)

plt.ylabel(
    "y (μm)"
)

plt.colorbar()


plt.subplot(
    1,
    2,
    2
)

plt.imshow(
    mask,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Continuous Mask"
)

plt.xlabel(
    "x (μm)"
)

plt.ylabel(
    "y (μm)"
)

plt.colorbar()


plt.tight_layout()

plt.show()


# =========================================================
# 14. Plot shifted pupils
# =========================================================

plt.figure(
    figsize=(10, 8)
)


for i in range(
    len(source_points)
):

    plt.subplot(
        2,
        2,
        i + 1
    )


    # fftshift only for visualization
    pupil_display = np.fft.fftshift(
        pupil_list[i]
    )


    plt.imshow(
        pupil_display,
        origin="lower",
        cmap="gray"
    )


    source_fx, source_fy = (
        source_points[i]
    )


    plt.title(
        f"Shifted Pupil\n"
        f"Source = "
        f"({source_fx:.1f}, "
        f"{source_fy:.1f})"
    )


    plt.colorbar()


plt.tight_layout()

plt.show()


# =========================================================
# 15. Plot individual coherent intensities
# =========================================================

plt.figure(
    figsize=(10, 8)
)


for i in range(
    len(source_points)
):

    plt.subplot(
        2,
        2,
        i + 1
    )


    plt.imshow(
        intensity_list[i],
        extent=[
            x.min(),
            x.max(),
            y.min(),
            y.max()
        ],
        origin="lower"
    )


    source_fx, source_fy = (
        source_points[i]
    )


    plt.title(
        f"Coherent Intensity {i + 1}\n"
        f"Source = "
        f"({source_fx:.1f}, "
        f"{source_fy:.1f})"
    )


    plt.xlabel(
        "x (μm)"
    )

    plt.ylabel(
        "y (μm)"
    )

    plt.colorbar()


plt.tight_layout()

plt.show()


# =========================================================
# 16. Plot final partially coherent intensity
# =========================================================

plt.figure(
    figsize=(6, 5)
)


plt.imshow(
    intensity_total,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower"
)


plt.title(
    "Partially Coherent Intensity"
)


plt.xlabel(
    "x (μm)"
)

plt.ylabel(
    "y (μm)"
)


plt.colorbar()


plt.tight_layout()

plt.show()


# =========================================================
# 17. Verify intensity summation manually
# =========================================================

intensity_check = (

    weight
    * intensity_list[0]

    +

    weight
    * intensity_list[1]

    +

    weight
    * intensity_list[2]

    +

    weight
    * intensity_list[3]
)


difference = np.max(
    np.abs(
        intensity_total
        -
        intensity_check
    )
)


print()

print(
    "Maximum difference between"
)

print(
    "loop result and manual sum =",
    difference
)