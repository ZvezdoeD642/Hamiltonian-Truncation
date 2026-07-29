import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

L = 1.0           # Box length x in [0, L]
N = 27            # Truncation limit (matrix size: N x N)
g = 10.0           # Interaction coupling strength (g * x^4)
m = 1.0           # Particle mass (setting hbar = 1)
hbar = 1.0        # Reduced Planck constant

def psi(n, x):
    """Normalized 1D Dirichlet box basis wavefunction."""
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)

def V_potential(x):
    """Interaction potential V(x) = g * x^4."""
    return g * (x**4)

H = np.zeros((N, N))

for i in range(1, N + 1):
    for j in range(1, N + 1):
        # Calculate interaction matrix element V_ij via numerical integration
        integrand = lambda x: psi(i, x) * V_potential(x) * psi(j, x)
        V_ij, _ = quad(integrand, 0, L)
        
        if i == j:
            # Diagonal: Free energy H0 + Interaction V_ii
            E_n = ((i * np.pi * hbar)**2) / (2.0 * m * (L**2))
            H[i - 1, j - 1] = E_n + V_ij
        else:
            # Off-Diagonal: Interaction V_ij only
            H[i - 1, j - 1] = V_ij

eigenvalues, eigenvectors = np.linalg.eigh(H)
c_ground = eigenvectors[:, 0]  # Full ground-state eigenvector coefficients


x_grid = np.linspace(0, L, 500)

# A. Unperturbed Ground State H0 (pure n=1 mode)
field_H0 = psi(1, x_grid)

# B. Non-Perturbative Ground State (H0 + V superposition)
field_full = np.zeros_like(x_grid)
for idx, c_n in enumerate(c_ground):
    n = idx + 1
    field_full += c_n * psi(n, x_grid)


plt.figure(figsize=(9, 5.5))

# Plot Unperturbed H0
plt.plot(
    x_grid, 
    field_H0, 
    label=r"Free / Unperturbed Ground State ($H_0$)", 
    color="darkorange", 
    linestyle="--", 
    linewidth=2
)

# Plot Non-Perturbative Full Profile
plt.plot(
    x_grid, 
    field_full, 
    label=r"Non-Perturbative Ground State ($H_0 + V$)", 
    color="teal", 
    linewidth=2.5
)

plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.title(f"Ground-State Profile Mismatch (N = {N}, g = {g})", fontsize=12)
plt.xlabel("Position (x)", fontsize=11)
plt.ylabel(r"Amplitude $\Psi(x)$", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()