import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp


L = 1.0           # Box length x in [0, L]
N = 25            # Truncation limit (matrix size: N x N)
g = 20.0           # Coupling constant for V(x) = g * x^4

def V_potential(x):
    return g * (x**4)

def psi_basis(n, x):
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)


H = np.zeros((N, N))
for i in range(1, N + 1):
    for j in range(1, N + 1):
        integrand = lambda x: psi_basis(i, x) * V_potential(x) * psi_basis(j, x)
        V_ij, _ = quad(integrand, 0, L)
        
        if i == j:
            E_n = ((i * np.pi)**2) / 2.0  # Kinetic energy -1/2 d^2/dx^2
            H[i - 1, j - 1] = E_n + V_ij
        else:
            H[i - 1, j - 1] = V_ij

# Extract Matrix Ground State Energy E0
eigenvalues, eigenvectors = np.linalg.eigh(H)
E0_HT = eigenvalues[0]
c_ground = eigenvectors[:, 0]

x_grid = np.linspace(0, L, 500)
y_HT = np.zeros_like(x_grid)
for idx, c_n in enumerate(c_ground):
    y_HT += c_n * psi_basis(idx + 1, x_grid)


# Differential Equation: -1/2 y'' + V(x)y = E0 * y  ==>  y'' = 2*(V(x) - E0)*y
def ode_system(x, Y):
    y, dydx = Y
    d2ydx2 = 2.0 * (V_potential(x) - E0_HT) * y
    return [dydx, d2ydx2]


initial_slope = c_ground[0] * np.sqrt(2.0 / L) * (np.pi / L) # Match slope at x=0
sol = solve_ivp(ode_system, [0, L], [0.0, initial_slope], t_eval=x_grid, rtol=1e-8, atol=1e-10)
y_ODE = sol.y[0]

norm_factor = np.sqrt(np.trapezoid(y_ODE**2, x_grid))
y_ODE = y_ODE / norm_factor

# Ensure both wavefunctions have positive phase orientation
if y_HT[10] < 0: y_HT = -y_HT
if y_ODE[10] < 0: y_ODE = -y_ODE

diff = np.abs(y_HT - y_ODE)

mae = np.mean(diff)
rmse = np.sqrt(np.mean((y_HT - y_ODE)**2))
max_err = np.max(diff)

print("\n=== QUANTITATIVE MATCH METRICS ===")
print(f"Mean Absolute Error (MAE) : {mae:.6e}")
print(f"Root Mean Square Error   : {rmse:.6e}")
print(f"Maximum Difference       : {max_err:.6e}")


print(f"Hamiltonian Truncation Ground State Energy E_0 = {E0_HT:.6f}")
print(f"Endpoint Value of ODE Wave at x=L: y(L) = {y_ODE[-1]:.6e}")

plt.figure(figsize=(9, 5.5))
plt.plot(x_grid, y_HT, label=f"Hamiltonian Truncation ($N={N}$)", color="teal", linewidth=2.5)
plt.plot(x_grid, y_ODE, label=f"ODE Integration with $E_0 = {E0_HT:.4f}$", color="crimson", linestyle="--", linewidth=2)
plt.axhline(0, color="gray", linestyle=":", alpha=0.7)
plt.title(f"Ground State Match (Energy $E_0 = {E0_HT:.4f}$)", fontsize=12)
plt.xlabel("Position (x)", fontsize=11)
plt.ylabel(r"Amplitude $\Psi(x)$", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()