import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Spatial grid
# =========================================================

N = 101

x = np.linspace(-1.0, 1.0, N)   # unit: μm
y = np.linspace(-1.0, 1.0, N)

X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]


# =========================================================
# 2. Target / mask
# =========================================================

mask = (
    (np.abs(X) <= 0.30) &
    (np.abs(Y) <= 0.20)
).astype(float)


# =========================================================
# 3. Frequency grid
# =========================================================

fx = np.fft.fftfreq(N, d=dx)
fy = np.fft.fftfreq(N, d=dy)

FX, FY = np.meshgrid(fx, fy)


# =========================================================
# 4. Physical parameters
# =========================================================

wavelength = 0.193   # μm

NA_1 = 0.50
NA_2 = 0.85


# =========================================================
# 5. Build pupil from NA
# =========================================================

def make_pupil(NA):

    cutoff = NA / wavelength

    pupil = (
        np.sqrt(FX**2 + FY**2) <= cutoff
    ).astype(float)

    return pupil


# =========================================================
# 6. Fourier imaging
# =========================================================

def fourier_imaging(mask, pupil):

    # Spatial domain -> frequency domain
    spectrum = np.fft.fft2(mask)

    # Optical filtering
    filtered_spectrum = spectrum * pupil

    # Frequency domain -> spatial domain
    field = np.fft.ifft2(filtered_spectrum)

    # Complex field -> intensity
    intensity = np.abs(field) ** 2

    # Normalize
    intensity = intensity / np.max(intensity)

    return intensity


# =========================================================
# 7. Two pupils
# =========================================================

pupil_1 = make_pupil(NA_1)
pupil_2 = make_pupil(NA_2)


# =========================================================
# 8. Two aerial images
# =========================================================

image_1 = fourier_imaging(
    mask,
    pupil_1
)

image_2 = fourier_imaging(
    mask,
    pupil_2
)


# =========================================================
# 9. Plot
# =========================================================

plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)
plt.imshow(
    np.fft.fftshift(pupil_1),
    origin="lower",
    cmap="gray"
)
plt.title("Pupil, NA = 0.50")
plt.colorbar()


plt.subplot(2, 2, 2)
plt.imshow(
    np.fft.fftshift(pupil_2),
    origin="lower",
    cmap="gray"
)
plt.title("Pupil, NA = 0.85")
plt.colorbar()


plt.subplot(2, 2, 3)
plt.imshow(
    image_1,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)
plt.title("Aerial Image, NA = 0.50")
plt.colorbar()


plt.subplot(2, 2, 4)
plt.imshow(
    image_2,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)
plt.title("Aerial Image, NA = 0.85")
plt.colorbar()


plt.tight_layout()
plt.show()