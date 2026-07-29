import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

L = 1.0           
N = 15            
g = 5.0           
m = 1.0           
hbar = 1.0        

def psi(n, x):
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)

def V_potential(x):
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

# Ground-state energy and eigenvector coefficients (c_1, c_2, ..., c_N)
E_0 = eigenvalues[0]
c_ground = eigenvectors[:, 0]

print(f"Ground State Energy E_0: {E_0:.5f}")

x_grid = np.linspace(0, L, 500)
field_profile = np.zeros_like(x_grid)

# Superpose: Psi_0(x) = sum_n ( c_n * psi_n(x) )
for idx, c_n in enumerate(c_ground):
    n = idx + 1
    field_profile += c_n * psi(n, x_grid)


plt.figure(figsize=(8, 5))
plt.plot(x_grid, field_profile, label=r"Ground State $\Psi_0(x)$", color="teal", linewidth=2)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.title(f"Non-Perturbative Ground-State Field Profile (N = {N}, g = {g})", fontsize=12)
plt.xlabel("Position (x)", fontsize=11)
plt.ylabel(r"Amplitude $\Psi(x)$", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()