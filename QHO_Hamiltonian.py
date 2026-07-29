import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import eval_hermite
import math

def main():
    N = 20           
    g = 5.0
    hbar = 1.0
    m = 1.0

    # Define the spatial grid for plotting
    x_grid = np.linspace(-5.0, 5.0, 500)

    # Pre-calculate normalization constants for speed
    norms = [1.0 / np.sqrt((2.0**n) * math.factorial(n) * np.sqrt(np.pi)) for n in range(N)]

    # Optimized QHO wavefunction
    def psi_harmonic(n, x):
        return norms[n] * np.exp(-0.5 * x**2) * eval_hermite(n, x)

    H = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if (i + j) % 2 == 0:
                integrand = lambda x: psi_harmonic(i, x) * (g * (x**4)) * psi_harmonic(j, x)
                # Integrate over bounds [-10, 10]
                V_ij, _ = quad(integrand, -10.0, 10.0)
            else:
                V_ij = 0.0  # Odd parity terms are exactly 0

            if i == j:
                # Diagonal: Unperturbed QHO energy E_n = (n + 1/2) + Interaction V_ii
                E_n = i + 0.5
                H[i, j] = E_n + V_ij
            else:
                # Off-diagonal: Interaction V_ij only
                H[i, j] = V_ij

    # Diagonalize truncated matrix
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    E_0 = eigenvalues[0]

    field_H0 = psi_harmonic(0, x_grid)


    plt.figure(figsize=(10, 8))

    field_full = np.zeros_like(x_grid)
    for k in range(5):
        c_k = eigenvectors[:, k]  # k-th eigenstate coefficients
        psi_k = np.zeros_like(x_grid)
        for n in range(N):
            psi_k += c_k[n] * psi_harmonic(n, x_grid)
        E_k = eigenvalues[k]
        plt.plot(x_grid, psi_k + E_k, label=f"Eigenstate {k} (E={eigenvalues[k]:.2f})")

    E_min = np.min(eigenvalues)
    E_max = eigenvalues[4]

    # Unperturbed QHO
    plt.plot(
        x_grid, 
        field_H0, 
        label=r"Unperturbed QHO ($g = 0$)", 
        color="darkorange", 
        linestyle="--", 
        linewidth=2
    )

    # Full Anharmonic Ground State

    # Scaled Potential V(x) for reference
    V_visual = 0.5 * x_grid**2 + g * x_grid**4
    plt.plot(
        x_grid, 
        V_visual, 
        label=r"Potential $V(x) = \frac{1}{2}x^2 + gx^4$ (scaled)", 
        color="gray", 
        linestyle=":", 
        alpha=0.6
    )

    plt.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    plt.ylim(E_min - 1, E_max + 2)
    plt.xlim(-10.0, 10.0)

    plt.title(f"QHO Anharmonic Ground State via Numerical Integration ($N = {N}$, $E_0 = {E_0:.4f}$)", fontsize=11)
    plt.xlabel("Position (x)", fontsize=11)
    plt.ylabel(r"Amplitude $\Psi(x)$", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()