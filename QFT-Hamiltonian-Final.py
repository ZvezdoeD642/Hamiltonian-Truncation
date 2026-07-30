import numpy as np
import matplotlib.pyplot as plt
import itertools

# =====================================================================
# 1. GLOBAL SETUPS & PHYSICS PARAMETERS
# =====================================================================
E_max = 6.0
mass = 1.0
g = 0.5       # Interaction coupling strength
L = 10.0      # Box size

momenta = [-1, 0, 1]
mode_indices = [0, 1, 2] 

# Pre-calculate 4-operator momentum conservation (For the Hamiltonian)
valid_mode_combos = []
for combo in itertools.product(mode_indices, repeat=4):
    if sum(momenta[idx] for idx in combo) == 0:
        valid_mode_combos.append(combo)

# Operator combinations
op_4_combos = list(itertools.product([-1, 1], repeat=4)) # For phi^4
op_2_combos = list(itertools.product([-1, 1], repeat=2)) # For phi^2

# =====================================================================
# 2. CORE ENGINE FUNCTIONS
# =====================================================================
def generate_fock_basis(E_max, mass, modes):
    omegas = [np.sqrt(k**2 + mass**2) for k in modes]
    basis = []
    max_particles = int(E_max / min(omegas)) + 1
    
    # Generate all tuples and apply the TRUNCATION & SYMMETRY filters
    for state_tuple in itertools.product(range(max_particles), repeat=len(modes)):
        total_energy = sum(state_tuple[i] * omegas[i] for i in range(len(modes)))
        total_particles = sum(state_tuple)
        
        # Keep only states under budget AND with Even Parity (Z2 Symmetry)
        if total_energy <= E_max and total_particles % 2 == 0:
            basis.append(state_tuple)
            
    return basis, omegas

def creation(coeff, state_tuple, mode_index):
    n = state_tuple[mode_index]
    new_coeff = coeff * np.sqrt(n + 1)
    temp_list = list(state_tuple)
    temp_list[mode_index] += 1
    return new_coeff, tuple(temp_list)

def annihilation(coeff, state_tuple, mode_index):
    n = state_tuple[mode_index]
    if n == 0: return 0.0, state_tuple # Protect the vacuum!
    new_coeff = coeff * np.sqrt(n)
    temp_list = list(state_tuple)
    temp_list[mode_index] -= 1
    return new_coeff, tuple(temp_list)

# =====================================================================
# 3. MAIN SCRIPT: DIAGONALIZATION & OBSERVABLES
# =====================================================================
def main():
    print(f"--- QFT Hamiltonian Truncation (E_max = {E_max}) ---")
    
    # 1. Generate Basis
    my_basis, omegas = generate_fock_basis(E_max, mass, momenta)
    N = len(my_basis) 
    print(f"Basis generated. Matrix Dimension: {N} x {N}")

    # 2. Build Hamiltonian Matrix
    H = np.zeros((N, N))
    print("Building Hamiltonian Matrix...")
    for i in range(N):
        for j in range(N):
            start_state = my_basis[j]
            end_state = my_basis[i]
            cell_total = 0.0

            # Diagonal (Free Energy)
            if i == j:
                cell_total += sum(start_state[k] * omegas[k] for k in range(3))

            # Off-Diagonal (phi^4 Interaction)
            for mode_combo in valid_mode_combos:
                for op_combo in op_4_combos:
                    current_state = start_state
                    current_coeff = 1.0
                    
                    for op, idx in zip(reversed(op_combo), reversed(mode_combo)):
                        if op == -1: current_coeff, current_state = annihilation(current_coeff, current_state, idx)
                        elif op == 1: current_coeff, current_state = creation(current_coeff, current_state, idx)
                        if current_coeff == 0.0: break
                        
                    if current_state == end_state:
                        cell_total += g * current_coeff

            H[i, j] = cell_total

    # 3. Diagonalize
    print("Diagonalizing...")
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    E_0 = eigenvalues[0]
    ground_state = eigenvectors[:, 0]
    print(f"Ground State Energy (Vacuum): {E_0:.4f}")

    # 4. Simple Observable: Total Virtual Particles
    N_matrix = np.zeros((N, N))
    for i in range(N):
        N_matrix[i, i] = sum(my_basis[i])
    expected_particles = ground_state.T @ N_matrix @ ground_state
    print(f"Expected Virtual Particles in Vacuum: {expected_particles:.4f}")

    # =====================================================================
    # 4. THE 2-POINT SPATIAL CORRELATION FUNCTION <phi(x)phi(0)>
    # =====================================================================
    print("Calculating Exact 2-Point Correlation Function...")
    
    # Step A: Precompute the operator matrix for every pair of modes (m, n)
    E_mn = np.zeros((3, 3)) 
    for m in range(3):
        for n in range(3):
            Op_matrix = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    start_state = my_basis[j]
                    end_state = my_basis[i]
                    cell_total = 0.0
                    
                    for op_combo in op_2_combos:
                        curr_state, curr_coeff = start_state, 1.0
                        
                        # Apply operator to mode n (phi(0)), then mode m (phi(x))
                        if op_combo[1] == -1: curr_coeff, curr_state = annihilation(curr_coeff, curr_state, n)
                        else: curr_coeff, curr_state = creation(curr_coeff, curr_state, n)
                        if curr_coeff == 0.0: continue
                        
                        if op_combo[0] == -1: curr_coeff, curr_state = annihilation(curr_coeff, curr_state, m)
                        else: curr_coeff, curr_state = creation(curr_coeff, curr_state, m)
                        if curr_coeff == 0.0: continue
                        
                        if curr_state == end_state:
                            cell_total += curr_coeff
                            
                    Op_matrix[i, j] = cell_total
                    
            # Expectation value <Omega | (a_m + a_m^dag)(a_n + a_n^dag) | Omega>
            E_mn[m, n] = ground_state.T @ Op_matrix @ ground_state

    # Step B: Scan across physical space and apply the spatial waves
    x_grid = np.linspace(0, L, 100)
    correlation_values = []
    
    for x in x_grid:
        c_val = 0.0
        for m in range(3):
            for n in range(3):
                # Calculate the standing waves at x and 0
                k_m, k_n = momenta[m], momenta[n]
                
                wave_m_x = 1.0 if k_m == 0 else (np.cos(2*np.pi*k_m*x/L) if k_m > 0 else np.sin(2*np.pi*abs(k_m)*x/L))
                wave_n_0 = 1.0 if k_n == 0 else (np.cos(0) if k_n > 0 else np.sin(0))
                
                # Combine with QFT Normalization (1 / 2*sqrt(w_m * w_n))
                weight = (wave_m_x * wave_n_0) / (2.0 * np.sqrt(omegas[m] * omegas[n]))
                
                c_val += weight * E_mn[m, n]
                
        correlation_values.append(c_val)

    # =====================================================================
    # 5. PLOTTING THE PUBLISHABLE DATA
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Vacuum Composition (Notice how clean it is now!)
    ax1.plot(range(N), np.abs(ground_state)**2, color='teal', marker='o', linestyle='-')
    ax1.set_title("Exact Vacuum Composition (Even Parity Basis)", fontsize=12)
    ax1.set_xlabel("State Index (Truncated Tuple ID)")
    ax1.set_ylabel("Probability $|c_i|^2$")
    ax1.grid(alpha=0.3)
    
    # Plot 2: The Correlation Function (The Mass Gap Decay)
    ax2.plot(x_grid, correlation_values, color='crimson', linewidth=2.5)
    ax2.set_title(r"2-Point Spatial Correlation $\langle \Omega | \phi(x)\phi(0) | \Omega \rangle$", fontsize=12)
    ax2.set_xlabel("Distance from origin (x)")
    ax2.set_ylabel("Correlation Strength")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()