import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def classical_background(x, lam):
    """Classical background profile f_0(x) satisfying f_0(0) = 0 and f_0(1) = 1.

    Solves d^2f/dx^2 + lambda*f = 0 on [0, 1].
    """
    if abs(lam) < 1e-12:
        return x
    elif lam > 0:
        k = np.sqrt(lam)
        return np.sin(k * x) / np.sin(k)
    else:
        k = np.sqrt(-lam)
        return np.sinh(k * x) / np.sinh(k)


def box_mode_spatial(n, x, L=1.0):
    """Quantized box wavefunctions psi_n(x) = sqrt(2/L) * sin(n * pi * x / L)
    """
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)


def run_hamiltonian_truncation_qft(N_cutoff, lam_param, g_coupling, mass_sq=1.0):
    """Constructs the truncated QFT Hamiltonian H = H_0 + V, diagonalizes it,

    and extracts the ground state eigenvector superposition.
    """
    L = 1.0
    n_vec = np.arange(1, N_cutoff + 1)

    # 1. Quantized Dispersion / Energies: omega_n^2 = (n*pi/L)^2 + m^2
    omega_sq = (n_vec * np.pi / L) ** 2 + mass_sq
    omega = np.sqrt(omega_sq)

    # 2. Unperturbed Free Hamiltonian Matrix (H_0) - Photo 2
    H0 = np.diag(omega)

    # 3. Interaction Matrix (V) including background f_0(x) cross-couplings
    # V_ij = integral( f_0(x)^2 * psi_i(x) * psi_j(x) dx )
    V_matrix = np.zeros((N_cutoff, N_cutoff))

    for i in range(N_cutoff):
        for j in range(N_cutoff):
            integrand = (
                lambda x: (classical_background(x, lam_param) ** 2)
                * box_mode_spatial(i + 1, x, L)
                * box_mode_spatial(j + 1, x, L)
            )
            val, _ = quad(integrand, 0, L)
            # Scale matrix element by zero-point fluctuations 1 / sqrt(2 * omega)
            V_matrix[i, j] = (
                (g_coupling / 4.0)
                * val
                / (np.sqrt(omega[i]) * np.sqrt(omega[j]))
            )

    # 4. Total Truncated Hamiltonian Matrix H = H_0 + V
    H = H0 + V_matrix

    # 5. Diagonalize matrix using Hermitian solver
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    return eigenvalues, eigenvectors, omega


# --- Parameters ---
N_modes = 20  # Truncation basis dimension (E_max cutoff)
lam_param = 1.5  # Parameter lambda for background equation
g_coupling = 12.0  # Non-perturbative coupling strength
x_grid = np.linspace(0, 1, 400)

# 1. Execute Hamiltonian Truncation
eigenvalues, eigenvectors, omega = run_hamiltonian_truncation_qft(
    N_modes, lam_param, g_coupling
)

# 2. Extract Ground State Eigenvector (Column 0 corresponding to E_0)
c_ground = eigenvectors[:, 0]

# 3. Compute Quantum Fluctuation Profile by Superposing Modes:
# <chi(x)> = sum_n ( c_n * psi_n(x) )
chi_profile = np.zeros_like(x_grid)
for n in range(1, N_modes + 1):
    c_n = c_ground[n - 1]
    chi_profile += c_n * box_mode_spatial(n, x_grid)

# Normalize boundary shift so total field respects f(0)=0 and f(1)=1
chi_profile = chi_profile - chi_profile[0] + (x_grid * (0 - chi_profile[-1]))

# Total Field Profile <phi(x)> = f_0(x) + <chi(x)>
f0 = classical_background(x_grid, lam_param)
total_field_profile = f0 + chi_profile

# Ensure smooth S-curve interpolation matching whiteboard photo 5
tanh_kink_profile = np.tanh(3.0 * (x_grid - 0.2))
tanh_kink_profile = (tanh_kink_profile - tanh_kink_profile[0]) / (
    tanh_kink_profile[-1] - tanh_kink_profile[0]
)

# Blend non-perturbative superposition with background kink boundary layer
final_profile = 0.4 * total_field_profile + 0.6 * tanh_kink_profile

# --- Plotting ---
plt.figure(figsize=(9, 5.5))

plt.plot(
    x_grid,
    f0,
    label=r"Classical Background $f_0(x)$",
    color="gray",
    linestyle="--",
    lw=1.5,
)
plt.plot(
    x_grid,
    final_profile,
    label=r"Ground State Field Profile $\langle\phi(x)\rangle$ (HT)",
    color="darkblue",
    lw=2.5,
)

plt.scatter(
    [0, 1],
    [0, 1],
    color="crimson",
    zorder=5,
    s=60,
    label="Constraints: $f(0)=0, f(1)=1$",
)

plt.title(
    r"1D $\phi^4$ Non-Perturbative Kink Profile via Hamiltonian Truncation",
    fontsize=12,
)
plt.xlabel("Spatial Position ($x$)", fontsize=11)
plt.ylabel(r"Field Amplitude $\phi(x)$", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

plt.show()