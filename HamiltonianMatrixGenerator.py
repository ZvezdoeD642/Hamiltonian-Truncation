import numpy as np
# System Parameters
N = 5
g = 0.5
L = 1.0
hbar = 1.0
m_mass = 1.0
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


        elif (m + n) % 2 == 0:
            x4_elem = L ** 4 * (24 * m * n * (m ** 2 + n ** 2)) / (np.pi ** 4 * (m ** 2 - n ** 2) ** 4)
            H[m - 1, n - 1] = g * x4_elem


        else:
            x4_elem = -L ** 4 * (8 * m * n * (m ** 2 + n ** 2)) / (np.pi ** 2 * (m ** 2 - n ** 2) ** 3)
            H[m - 1, n - 1] = g * x4_elem

# Display the resulting Hamiltonian matrix
print(f"{N}x{N} Truncated Hamiltonian Matrix:")
print(np.round(H, 4))


