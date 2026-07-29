import numpy as np
import matplotlib.pyplot as plt

def solve_power_series(x_grid, a0=0.0, a1=1.0, num_terms=40):
    
    a = np.zeros(num_terms)
    
    # Base cases from initial conditions y(0) = a0, y'(0) = a1
    a[0] = a0
    a[1] = a1
    a[2] = a0 / 2.0
    a[3] = a1 / 6.0
    a[4] = a[2]/12
    a[5] = a[3]/20
    
    # Calculate terms using the recurrence relation
    for n in range(4, num_terms - 2):
        a[n + 2] = (a[n] - a[n - 4]) / ((n + 2) * (n + 1))
        
    # Evaluate y(x) = sum_n (a_n * x^n) across the spatial grid
    y = np.zeros_like(x_grid)
    for n in range(num_terms):
        y += a[n] * (x_grid ** n)
        
    return y, a


# Domain chosen where the Taylor series converges smoothly
x_grid = np.linspace(-1.5, 1.5, 400)

# Initial conditions at x = 0
# Set a0 = 0, a1 = 1 for a symmetric/even ground-state-like profile
a0_val = 0.0  # y(0)
a1_val = 1.0  # y'(0)

# 2. Compute Series Solution
y_series, coefficients = solve_power_series(x_grid, a0=a0_val, a1=a1_val, num_terms=50)

# 3. Compute Effective Potential V(x) = x^4 + 1
V_potential = x_grid**4 + 1.0

plt.figure(figsize=(9, 5))

# Plot power series solution y(x)
plt.plot(x_grid, y_series, label=r'Power Series Solution $y(x)$', color='teal', linewidth=2.5)

# Plot potential V(x)
plt.plot(x_grid, V_potential, '--', label=r'Effective Potential $V(x) = x^4 + 1$', color='orange', alpha=0.8)

plt.axhline(0, color='black', linewidth=0.8, linestyle=':')
plt.axvline(0, color='black', linewidth=0.8, linestyle=':')

plt.title(r'Power Series Solution', fontsize=13)
plt.xlabel('x', fontsize=12)
plt.ylabel('y(x)', fontsize=12)
plt.ylim(0, 3.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()