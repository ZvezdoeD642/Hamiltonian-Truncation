import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.special import hermite


def harmonic_oscillator_wavefunction(n, x):
    normalization = 1.0 / np.sqrt((2.0**n) * math.factorial(n) * np.sqrt(np.pi))
    # Hermite polynomial H_n(x)
    H_n = hermite(n)
    return normalization * H_n(x) * np.exp(-(x**2) / 2.0)


def compute_ground_state_profile(N_cutoff, g_coupling, x_range):
    """Constructs the truncated Hamiltonian, finds eigenvectors, and superposes

    basis functions to create the ground-state field profile.
    """
    # 1. Construct H0 and V (Anharmonic Oscillator: H = H0 + g*x^4)
    H0 = np.diag(np.arange(N_cutoff) + 0.5)

    a_dag = np.diag(np.sqrt(np.arange(1, N_cutoff)), k=-1)
    a = np.diag(np.sqrt(np.arange(1, N_cutoff)), k=1)
    x_op = (a + a_dag) / np.sqrt(2.0)

    x2 = np.matmul(x_op, x_op)
    x4 = np.matmul(x2, x2)
    V = g_coupling * x4

    H = H0 + V

    # 2. Diagonalize to get eigenvalues and eigenvectors
    energies, eigenvectors = np.linalg.eigh(H)

    # Extract Ground State eigenvector c_n (first column)
    ground_state_coeffs = eigenvectors[:, 0]

    # 3. Superpose basis wavefunctions across spatial grid x
    field_profile = np.zeros_like(x_range)
    for n in range(N_cutoff):
        # Contribution of n-th basis state scaled by expansion coefficient c_n
        c_n = ground_state_coeffs[n]
        field_profile += c_n * harmonic_oscillator_wavefunction(n, x_range)

    return field_profile, energies[0]


# --- Main Execution & Plotting ---
N_cutoff = 30  # Basis truncation size (E_max cutoff)
g = 0.5  # Non-perturbative interaction strength
x_grid = np.linspace(-4, 4, 400)  # Spatial domain

# Compute spatial field profile
profile, E0 = compute_ground_state_profile(N_cutoff, g, x_grid)

# Plotting with Matplotlib
plt.figure(figsize=(8, 5))
plt.plot(
    x_grid,
    profile,
    label=f"Ground State Profile $\\Psi_0(x)$ (E0 = {E0:.4f})",
    color="navy",
    lw=2,
)
plt.plot(
    x_grid,
    profile**2,
    label="Probability Density $|\\Psi_0(x)|^2$",
    color="crimson",
    linestyle="--",
    lw=1.5,
)

plt.title(
    f"Non-Perturbative Ground State Profile via HT ($N={N_cutoff}, g={g}$)",
    fontsize=12,
)
plt.xlabel("Position ($x$)", fontsize=11)
plt.ylabel("Field Amplitude", fontsize=11)
plt.axhline(0, color="black", linewidth=0.8, linestyle=":")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

plt.show()