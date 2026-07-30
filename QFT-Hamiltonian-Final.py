import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

E_max = 15.0
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


def generate_fock_basis_recursive(E_max, mass, modes):
    """
    Generates a truncated Fock space basis using Depth-First Search.
    Strictly enforces the energy cutoff (E_max) and Z2 symmetry (Even Parity).
    """
    omegas = [np.sqrt(k**2 + mass**2) for k in modes]
    num_modes = len(modes)
    basis = []
    
    # We maintain a single list that represents our current path down the tree
    current_state = [0] * num_modes
    
    def dfs(mode_idx, current_energy, current_particles):
        # BASE CASE: We have assigned a particle count to every mode
        if mode_idx == num_modes:
            # Apply the Z2 Symmetry filter (Even Parity)
            if current_particles % 2 == 0:
                basis.append(tuple(current_state))
            return
        
        # RECURSIVE STEP: How many particles can we afford in this specific mode?
        omega = omegas[mode_idx]
        budget_remaining = E_max - current_energy
        max_particles_for_this_mode = int(budget_remaining / omega)
        
        # Loop through all affordable particle counts for this mode
        for n in range(max_particles_for_this_mode + 1):
            current_state[mode_idx] = n
            new_energy = current_energy + n * omega
            new_particles = current_particles + n
            
            # Recurse deeper into the next mode
            dfs(mode_idx + 1, new_energy, new_particles)

    # Kick off the recursion starting at mode 0, with 0 energy and 0 particles
    dfs(mode_idx=0, current_energy=0.0, current_particles=0)
    
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


def main():
    print(f"--- QFT Hamiltonian Truncation (E_max = {E_max}) ---")
    
    # 1. Generate Basis
    my_basis, omegas = generate_fock_basis_recursive(E_max, mass, momenta)
    N = len(my_basis) 
    print(f"Basis generated. Matrix Dimension: {N} x {N}")

    # 1. The Reverse Lookup Dictionary (O(1) lookups instead of O(N) searches)
    print("Building reverse lookup dictionary...")
    state_to_idx = {state: idx for idx, state in enumerate(my_basis)}

    # 2. Initialize a Sparse LIL Matrix (List of Lists, optimized for adding elements)
    print("Building Sparse Hamiltonian Matrix...")
    H = lil_matrix((N, N))

    # We only loop through the starting states ONCE (O(N) instead of O(N^2))
    for j, start_state in enumerate(my_basis):
        
        # --- Diagonal Elements (Free Energy) ---
        free_energy = sum(start_state[k] * omegas[k] for k in range(len(momenta)))
        H[j, j] = free_energy

        # --- Off-Diagonal Elements (Interactions) ---
        for mode_combo in valid_mode_combos:
            for op_combo in op_4_combos:
                
                current_state = start_state
                current_coeff = 1.0
                
                # Apply the 4 operators sequentially
                for op, idx in zip(reversed(op_combo), reversed(mode_combo)):
                    if op == -1: 
                        current_coeff, current_state = annihilation(current_coeff, current_state, idx)
                    elif op == 1: 
                        current_coeff, current_state = creation(current_coeff, current_state, idx)
                    
                    # If an annihilation hit 0, this interaction chain is dead
                    if current_coeff == 0.0: 
                        break
                
                # If the chain survived AND the resulting state is within our E_max budget...
                if current_coeff != 0.0 and current_state in state_to_idx:
                    # Look up the row index of the resulting state instantly
                    i = state_to_idx[current_state]
                    
                    # Add to the sparse matrix
                    H[i, j] += g * current_coeff

    # 3. Convert to CSR (Compressed Sparse Row) format for ultra-fast math
    H = H.tocsr()
    print(f"Matrix built! Non-zero elements: {H.nnz} (out of {N*N} total cells)")

    # 3. Diagonalize
    print("Diagonalizing Matrix using Lanczos algorithm...")
    
    eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
    
    E_0 = eigenvalues[0]
    ground_state = eigenvectors[:, 0]

    # 4. Simple Observable: Total Virtual Particles
    N_matrix = np.zeros((N, N))
    for i in range(N):
        N_matrix[i, i] = sum(my_basis[i])
    expected_particles = ground_state.T @ N_matrix @ ground_state
    print(f"Expected Virtual Particles in Vacuum: {expected_particles:.4f}")


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