import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from SimpleHamiltonian import psi

def main():
    # System Parameters
    N = 10
    g = 100.0
    L = 1.0
    hbar = 1.0
    m_mass = 1.0

    ground_state_energies = []
    # Pre-factor for H0 ground state energy
    epsilon_0 = (hbar ** 2 * np.pi ** 2) / (2 * m_mass * L ** 2)

    # Initialize empty N x N matrix
    H = np.zeros((N, N))

    # Loop over matrix rows (m) and columns (n)
    for m in range(1, N + 1):
        for n in range(1, N + 1):


            if m == n:
                H0_elem = (n ** 2) * epsilon_0
                x4_elem = L ** 4 * (0.2 - 1.5 / (n ** 2 * np.pi ** 2) + 1.5 / (n ** 4 * np.pi ** 4))
                H[m - 1, n - 1] = H0_elem + g * x4_elem
                ground_state_energies.append(H[m - 1, n - 1])


            elif (m + n) % 2 == 0:
                x4_elem = L ** 4 * (24 * m * n * (m ** 2 + n ** 2)) / (np.pi ** 4 * (m ** 2 - n ** 2) ** 4)
                H[m - 1, n - 1] = g * x4_elem


            else:
                x4_elem = -L ** 4 * (8 * m * n * (m ** 2 + n ** 2)) / (np.pi ** 2 * (m ** 2 - n ** 2) ** 3)
                H[m - 1, n - 1] = g * x4_elem


    eigenvalues, eigenvectors = np.linalg.eigh(H)
    c_ground = eigenvectors[:, 0]  # Full ground-state eigenvector coefficients

    x_grid = np.linspace(0, L, 500)

    field_H0 = psi(1, x_grid)

    print(ground_state_energies)

    field_full = np.zeros_like(x_grid)
    for idx, c_n in enumerate(c_ground):
        n = idx + 1
        field_full += c_n * np.sqrt(2.0 / L) * np.sin(n * np.pi * x_grid / L)


    plt.figure(figsize=(9, 5.5))

    # Display the resulting Hamiltonian matrix
    plt.plot(
        x_grid, 
        field_H0, 
        label=r"Free / Unperturbed Ground State ($H_0$)", 
        color="darkorange", 
        linestyle="--", 
        linewidth=2
    )

    plt.plot(
        x_grid, 
        field_full, 
        label=r"Non-Perturbative Ground State ($H_0 + V$)", 
        color="teal", 
        linewidth=2.5
    )

    plt.xlabel("Position (x)", fontsize=11)
    plt.ylabel(r"Amplitude $\Psi(x)$", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    plt.show()

if __name__ == "__main__":
    main()


