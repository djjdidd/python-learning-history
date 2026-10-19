import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


# ============================================================
# 1. Basic settings
# ============================================================

N = 128
L = 2.0
dx = L / N

wavelength = 0.193
NA = 0.85
cutoff = NA / wavelength

dose = 1.0
beta_resist = 20.0
threshold = 0.5

# Fixed normalization reference
reference_peak = 1.0

# Compare these learning rates
learning_rates = [1.0, 10.0, 100.0, 1000.0]

steps = 150


# ============================================================
# 2. Spatial grid and target
# ============================================================

x = (np.arange(N) - N // 2) * dx

X, Y = np.meshgrid(x, x)

target = (
    (np.abs(X) <= 0.60 / 2)
    &
    (np.abs(Y) <= 0.40 / 2)
).astype(float)


# ============================================================
# 3. Frequency grid and pupil
# ============================================================

freq = np.fft.fftfreq(
    N,
    d=dx
)

FX, FY = np.meshgrid(
    freq,
    freq
)

pupil = (
    FX**2 + FY**2
    <= cutoff**2
).astype(float)


# ============================================================
# 4. Initial mask
# ============================================================

# 0.1 outside target
# 0.9 inside target
mask_initial = (
    0.1 + 0.8 * target
)

# Convert mask to unconstrained variable Z
Z_initial = np.log(
    mask_initial
    /
    (1.0 - mask_initial)
)


# ============================================================
# 5. ILT forward + backward
# ============================================================

def compute_loss_and_gradient(Z):

    # --------------------------------------------------------
    # Mask parameterization
    # --------------------------------------------------------

    mask = expit(Z)

    # --------------------------------------------------------
    # Optical forward
    # --------------------------------------------------------

    spectrum = np.fft.fft2(mask)

    field = np.fft.ifft2(
        spectrum * pupil
    )

    intensity = np.abs(field)**2

    # --------------------------------------------------------
    # Resist
    # --------------------------------------------------------

    effective_intensity = (
        dose
        *
        intensity
        /
        reference_peak
    )

    resist = expit(
        beta_resist
        *
        (
            effective_intensity
            -
            threshold
        )
    )

    # --------------------------------------------------------
    # Printing loss
    # --------------------------------------------------------

    loss = np.mean(
        (resist - target)**2
    )

    # ========================================================
    # Backward
    # ========================================================

    # Loss -> Resist
    dL_dR = (
        2.0
        *
        (resist - target)
        /
        resist.size
    )

    # Resist -> Intensity
    dL_dI = (
        dL_dR
        *
        beta_resist
        *
        resist
        *
        (1.0 - resist)
        *
        dose
        /
        reference_peak
    )

    # Intensity -> Field
    dL_dfield = (
        2.0
        *
        dL_dI
        *
        field
    )

    # Field -> Mask
    dL_dmask = np.real(
        np.fft.ifft2(
            np.fft.fft2(
                dL_dfield
            )
            *
            np.conj(pupil)
        )
    )

    # Mask -> Z
    gradient = (
        dL_dmask
        *
        mask
        *
        (1.0 - mask)
    )

    return loss, gradient


# ============================================================
# 6. Gradient Descent
# ============================================================

def run_gradient_descent(
    Z0,
    learning_rate,
    steps
):

    # Very important:
    # each experiment starts from the SAME initial Z
    Z = Z0.copy()

    history = {
        "loss": [],
        "gradient_norm": [],
        "parameter_change": []
    }

    for step in range(steps):

        loss, gradient = (
            compute_loss_and_gradient(Z)
        )

        # ------------------------------------------
        # Record current state
        # ------------------------------------------

        history["loss"].append(
            loss
        )

        grad_norm = np.linalg.norm(
            gradient
        )

        history[
            "gradient_norm"
        ].append(
            grad_norm
        )

        # ------------------------------------------
        # Update
        # ------------------------------------------

        Z_old = Z.copy()

        Z = (
            Z
            -
            learning_rate
            *
            gradient
        )

        parameter_change = np.linalg.norm(
            Z - Z_old
        )

        history[
            "parameter_change"
        ].append(
            parameter_change
        )

    # Final loss after all updates
    final_loss, final_gradient = (
        compute_loss_and_gradient(Z)
    )

    return (
        Z,
        history,
        final_loss,
        np.linalg.norm(
            final_gradient
        )
    )

def run_momentum(
    Z0,
    learning_rate,
    beta_momentum,
    steps
):

    Z = Z0.copy()

    # Momentum state
    velocity = np.zeros_like(Z)

    history = {
        "loss": [],
        "gradient_norm": [],
        "parameter_change": []
    }

    for step in range(steps):

        # Forward + backward
        loss, gradient = compute_loss_and_gradient(Z)

        # Record loss
        history["loss"].append(loss)

        grad_norm = np.linalg.norm(gradient)

        history["gradient_norm"].append(
            grad_norm
        )

        # Save old parameters
        Z_old = Z.copy()

        # ==========================================
        # Momentum
        # ==========================================

        velocity = (
            beta_momentum * velocity
            +
            gradient
        )

        Z = (
            Z
            -
            learning_rate * velocity
        )

        # Record parameter change
        parameter_change = np.linalg.norm(
            Z - Z_old
        )

        history[
            "parameter_change"
        ].append(
            parameter_change
        )

    final_loss, final_gradient = (
        compute_loss_and_gradient(Z)
    )

    return (
        Z,
        history,
        final_loss,
        np.linalg.norm(final_gradient)
    )


def run_adam(
    Z0,
    learning_rate,
    steps,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
):

    Z = Z0.copy()

    # First moment
    m = np.zeros_like(Z)

    # Second moment
    v = np.zeros_like(Z)

    history = {
        "loss": [],
        "gradient_norm": [],
        "parameter_change": []
    }

    for step in range(1, steps + 1):

        # ==========================================
        # Forward + backward
        # ==========================================

        loss, gradient = (
            compute_loss_and_gradient(Z)
        )

        history["loss"].append(loss)

        grad_norm = np.linalg.norm(
            gradient
        )

        history[
            "gradient_norm"
        ].append(
            grad_norm
        )

        Z_old = Z.copy()

        # ==========================================
        # Adam
        # ==========================================

        # First moment
        m = (
            beta1 * m
            +
            (1.0 - beta1) * gradient
        )

        # Second moment
        v = (
            beta2 * v
            +
            (1.0 - beta2) * gradient**2
        )

        # Bias correction
        m_hat = (
            m
            /
            (1.0 - beta1**step)
        )

        v_hat = (
            v
            /
            (1.0 - beta2**step)
        )

        # Parameter update
        Z = (
            Z
            -
            learning_rate
            *
            m_hat
            /
            (
                np.sqrt(v_hat)
                +
                epsilon
            )
        )

        parameter_change = np.linalg.norm(
            Z - Z_old
        )

        history[
            "parameter_change"
        ].append(
            parameter_change
        )

    final_loss, final_gradient = (
        compute_loss_and_gradient(Z)
    )

    return (
        Z,
        history,
        final_loss,
        np.linalg.norm(
            final_gradient
        )
    )



# ============================================================
# 7. Run all learning-rate experiments
# ============================================================

all_results = {}

print(
    "\n"
    "============================================"
)
print(
    "ILT Learning Rate Comparison"
)
print(
    "============================================\n"
)

for lr in learning_rates:

    (
        Z_final,
        history,
        final_loss,
        final_grad_norm
    ) = run_gradient_descent(
        Z0=Z_initial,
        learning_rate=lr,
        steps=steps
    )

    all_results[lr] = {
        "Z": Z_final,
        "history": history,
        "final_loss": final_loss,
        "final_gradient_norm": final_grad_norm
    }

    print(
        f"Learning Rate = {lr:7.1f}"
    )

    print(
        f"Initial Loss  = "
        f"{history['loss'][0]:.8f}"
    )

    print(
        f"Final Loss    = "
        f"{final_loss:.8f}"
    )

    improvement = (
        (
            history["loss"][0]
            -
            final_loss
        )
        /
        history["loss"][0]
        *
        100.0
    )

    print(
        f"Improvement   = "
        f"{improvement:.4f}%"
    )

    print(
        f"Final GradNorm= "
        f"{final_grad_norm:.8e}"
    )

    print()


# ============================================================
# 8. Plot Loss Curves
# ============================================================

plt.figure(
    figsize=(9, 5)
)

for lr in learning_rates:

    history = (
        all_results[lr]["history"]
    )

    plt.plot(
        history["loss"],
        label=f"LR = {lr:g}"
    )

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "Printing Loss"
)

plt.title(
    "ILT Learning Rate Comparison"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 9. Plot Gradient Norm
# ============================================================

plt.figure(
    figsize=(9, 5)
)

for lr in learning_rates:

    history = (
        all_results[lr]["history"]
    )

    plt.plot(
        history["gradient_norm"],
        label=f"LR = {lr:g}"
    )

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "Gradient Norm"
)

plt.title(
    "Gradient Norm During Optimization"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 10. Plot Parameter Change
# ============================================================

plt.figure(
    figsize=(9, 5)
)

for lr in learning_rates:

    history = (
        all_results[lr]["history"]
    )

    plt.plot(
        history["parameter_change"],
        label=f"LR = {lr:g}"
    )

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "||Z(t+1) - Z(t)||"
)

plt.title(
    "Parameter Change During Optimization"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================
# GD baseline
# ==========================================

(
    Z_gd,
    history_gd,
    loss_gd,
    grad_gd
) = run_gradient_descent(
    Z0=Z_initial,
    learning_rate=1000.0,
    steps=150
)


# ==========================================
# Momentum
# ==========================================

(
    Z_momentum,
    history_momentum,
    loss_momentum,
    grad_momentum
) = run_momentum(
    Z0=Z_initial,
    learning_rate=100.0,
    beta_momentum=0.9,
    steps=150
)


print("\nGD")
print("Final Loss =", loss_gd)
print("Final Gradient Norm =", grad_gd)

print("\nMomentum")
print("Final Loss =", loss_momentum)
print("Final Gradient Norm =", grad_momentum)

plt.figure(figsize=(9, 5))

plt.plot(
    history_gd["loss"],
    label="GD, LR=1000"
)

plt.plot(
    history_momentum["loss"],
    label="Momentum, LR=100, beta=0.9"
)

plt.xlabel("Iteration")
plt.ylabel("Printing Loss")

plt.title("GD vs Momentum")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

momentum_learning_rates = [
    30.0,
    100.0,
    300.0,
    1000.0
]

momentum_results = {}

for lr in momentum_learning_rates:

    (
        Z_final,
        history,
        final_loss,
        final_grad_norm
    ) = run_momentum(
        Z0=Z_initial,
        learning_rate=lr,
        beta_momentum=0.9,
        steps=150
    )

    momentum_results[lr] = {
        "Z": Z_final,
        "history": history,
        "final_loss": final_loss,
        "final_gradient_norm": final_grad_norm
    }

    print(
        f"Momentum LR = {lr:7.1f} | "
        f"Final Loss = {final_loss:.8f} | "
        f"Grad Norm = {final_grad_norm:.8e}"
    )

plt.figure(figsize=(9, 5))

# GD baseline
plt.plot(
    history_gd["loss"],
    label="GD, LR=1000",
    linewidth=2
)

# Momentum
for lr in momentum_learning_rates:

    plt.plot(
        momentum_results[lr]["history"]["loss"],
        label=f"Momentum, LR={lr:g}"
    )

plt.xlabel("Iteration")
plt.ylabel("Printing Loss")

plt.title(
    "Momentum Learning Rate Comparison"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


adam_learning_rates = [
    0.001,
    0.01,
    0.05,
    0.1
]

adam_results = {}

print("\nAdam Learning Rate Comparison\n")

for lr in adam_learning_rates:

    (
        Z_final,
        history,
        final_loss,
        final_grad_norm
    ) = run_adam(
        Z0=Z_initial,
        learning_rate=lr,
        steps=150
    )

    adam_results[lr] = {
        "Z": Z_final,
        "history": history,
        "final_loss": final_loss,
        "final_gradient_norm": final_grad_norm
    }

    print(
        f"Adam LR = {lr:7.3f} | "
        f"Final Loss = {final_loss:.8f} | "
        f"Grad Norm = {final_grad_norm:.8e}"
    )

plt.figure(figsize=(9, 5))

for lr in adam_learning_rates:

    plt.plot(
        adam_results[lr]["history"]["loss"],
        label=f"Adam, LR={lr:g}"
    )

plt.xlabel("Iteration")
plt.ylabel("Printing Loss")

plt.title(
    "Adam Learning Rate Comparison"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()