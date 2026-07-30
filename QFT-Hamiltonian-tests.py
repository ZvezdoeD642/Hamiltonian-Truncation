import numpy as np
import matplotlib.pyplot as plt
import itertools

# --- 1. GLOBAL SETUPS ---
possible_combinations = list(itertools.product([-1, 1], repeat=4))
momenta = [-1, 0, 1]
mode_indices = [0, 1, 2] 
valid_mode_combos = []

for combo in itertools.product(mode_indices, repeat=4):
    total_momentum = sum(momenta[idx] for idx in combo)
    if total_momentum == 0:
        valid_mode_combos.append(combo)

# --- 2. CORE FUNCTIONS ---
def generate_fock_basis(E_max, mass, modes):
    omegas = [np.sqrt(k**2 + mass**2) for k in modes]
    basis = []
    max_particles = int(E_max / min(omegas)) + 1
    possible_counts = range(max_particles)
    
    for state_tuple in itertools.product(possible_counts, repeat=len(modes)):
        total_energy = sum(state_tuple[i] * omegas[i] for i in range(len(modes)))
        if total_energy <= E_max:
            basis.append(state_tuple)
            
    return basis, omegas

def creation(coeff, state_tuple, mode_index):
    current_particles = state_tuple[mode_index]
    new_coeff = coeff * np.sqrt(current_particles + 1)
    temp_list = list(state_tuple)
    temp_list[mode_index] += 1
    return new_coeff, tuple(temp_list)

def annihilation(coeff, state_tuple, mode_index):
    current_particles = state_tuple[mode_index]
    if current_particles == 0:
        return 0.0, state_tuple
    
    new_coeff = coeff * np.sqrt(current_particles)
    temp_list = list(state_tuple)
    temp_list[mode_index] -= 1
    return new_coeff, tuple(temp_list)

# --- 3. MAIN SCRIPT ---
def main():
    # Physics parameters
    E_max = 6.0
    mass = 1.0
    g = 0.5  # The phi^4 coupling strength!
    L = 10.0 # Length of the spatial box

    print(f"Generating Basis for E_max = {E_max}...")
    my_basis, omegas = generate_fock_basis(E_max=E_max, mass=mass, modes=momenta)
    
    N = len(my_basis) 
    print(f"Matrix Dimension: {N} x {N}\n")

    H = np.zeros((N, N))

    # Fill in the matrix
    print("Building the Hamiltonian Matrix...")
    for i in range(N):
        for j in range(N):
            
            # FIX: j is the starting column, i is the ending row!
            start_state = my_basis[j]
            end_state = my_basis[i]

            # FIX: Initialize the cell total to 0 at the start of every cell
            cell_total = 0.0

            # Diagonal elements (Free Energy H_0)
            if i == j:
                cell_total += sum(start_state[k] * omegas[k] for k in range(3))

            # Off-diagonal elements (Interaction V)
            for mode_combo in valid_mode_combos:
                for op_combo in possible_combinations:
        
                    current_state = start_state
                    current_coeff = 1.0
        
                    for op, aisle_index in zip(reversed(op_combo), reversed(mode_combo)):
                        if op == -1:
                            current_coeff, current_state = annihilation(current_coeff, current_state, aisle_index)
                        elif op == 1:
                            current_coeff, current_state = creation(current_coeff, current_state, aisle_index)
                            
                        if current_coeff == 0.0:
                            break
                        
                    # FIX: Orthogonality check inside op_combo loop
                    if current_state == end_state:
                        # Multiply by coupling constant g!
                        cell_total += g * current_coeff

            # FIX: Assign to matrix outside of the nested combos loop
            H[i, j] = cell_total

    print("Diagonalizing Matrix...")
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    E_0 = eigenvalues[0]
    ground_state_vector = eigenvectors[:, 0]
    
    print(f"\nSUCCESS! Non-Perturbative Ground State Energy: {E_0:.4f}")
    
    # --- 4. PLOTTING THE RESULTS ---
    # Plot 1: Show the dominant states in the vacuum
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Matrix Sparsity (The Physics Rules)")
    plt.spy(H, markersize=2, color='teal')
    plt.xlabel("Starting State (j)")
    plt.ylabel("Ending State (i)")

    plt.subplot(1, 2, 2)
    plt.title("Vacuum Composition (Ground State Eigenvector)")
    plt.plot(range(N), np.abs(ground_state_vector)**2, color='crimson', marker='o', linestyle='-')
    plt.xlabel("State Index (Tuple ID)")
    plt.ylabel("Probability inside Vacuum")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()