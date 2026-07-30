import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.animation as animation

# --- 1. GLOBAL SETUPS ---
possible_combinations = list(itertools.product([-1, 1], repeat=4))
momenta = [-1, 0, 1]  # Mode index 0: k = -1 | Mode index 1: k = 0 | Mode index 2: k = +1
mode_indices = [0, 1, 2] 

valid_mode_combos = []
for combo in itertools.product(mode_indices, repeat=4):
    if sum(momenta[idx] for idx in combo) == 0:
        valid_mode_combos.append(combo)

# --- 2. CORE FUNCTIONS ---

def spatial_integral(mode_combo, op_combo, momenta, L):
    net_k = sum(-op * momenta[idx] for op, idx in zip(op_combo, mode_combo))
    if abs(net_k) < 1e-12:
        return L + 0.0j
    else:
        return (np.exp(1j * net_k * L) - 1.0) / (1j * net_k)


def generate_fock_basis(E_max, mass, modes):
    omegas = [np.sqrt(k**2 + mass**2) for k in modes]
    basis = []
    max_particles = int(E_max / min(omegas)) + 1
    
    for state_tuple in itertools.product(range(max_particles), repeat=len(modes)):
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
    E_max = 7.0
    mass = 1.0
    lambda_coupling = 10
    L = 100
    
    coupling_factor = lambda_coupling / 24.0

    print(f"Generating Basis for E_max = {E_max}...")
    my_basis, omegas = generate_fock_basis(E_max=E_max, mass=mass, modes=momenta)
    
    norm_factors = [1.0 / np.sqrt(2.0 * w * L) for w in omegas]

    
    N = len(my_basis) 
    print(f"Matrix Dimension: {N} x {N}\n")

    H = np.zeros((N, N))

    print("Building Hamiltonian Matrix...")
    for i in range(N):
        for j in range(N):
            start_state = my_basis[j]
            end_state = my_basis[i]
            cell_total = 0.0

            if i == j:
                cell_total += sum(start_state[k] * omegas[k] for k in range(len(momenta)))

            for mode_combo in valid_mode_combos:
                field_norm = np.prod([norm_factors[idx] for idx in mode_combo])
                
                for op_combo in possible_combinations:
                    I_spatial = spatial_integral(mode_combo, op_combo, momenta, L)
                    if np.abs(I_spatial) < 1e-8:
                        continue

                    current_state = start_state
                    current_coeff = 1.0
        
                    for op, aisle_index in zip(reversed(op_combo), reversed(mode_combo)):
                        if op == -1:
                            current_coeff, current_state = annihilation(current_coeff, current_state, aisle_index)
                        elif op == 1:
                            current_coeff, current_state = creation(current_coeff, current_state, aisle_index)
                            
                        if current_coeff == 0.0:
                            break
                        
                    if current_state == end_state:
                        matrix_element = coupling_factor * field_norm * np.real(I_spatial) * current_coeff
                        cell_total += matrix_element

            H[i, j] = cell_total

    print("Diagonalizing Hamiltonian...")
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # --- 4. SUPERPOSITION OF NON-ZERO MOMENTUM MODES (k = -1, k = +1) ---
    idx_vac = my_basis.index((0, 0, 0))
    idx_km1 = my_basis.index((1, 0, 0))  # k = -1 mode excitation
    idx_kp1 = my_basis.index((0, 0, 1))  # k = +1 mode excitation

    psi_0 = np.zeros(N, dtype=complex)
    psi_0[idx_vac] = 1.0 / np.sqrt(3)
    psi_0[idx_km1] = 1.0 / np.sqrt(3)
    psi_0[idx_kp1] = 1.0 / np.sqrt(3)

    # Project state onto the energy eigenbasis
    c_n = eigenvectors.T.conj() @ psi_0

    # Build field operator phi(x) matrices
    x_grid = np.linspace(0, L, 400)
    phi_x_matrices = []
    
    for x in x_grid:
        phi_matrix = np.zeros((N, N), dtype=complex)
        
        for mode_idx, k in enumerate(momenta):
            norm = norm_factors[mode_idx]
            
            for j, start_state in enumerate(my_basis):
                # Annihilation operator term
                coeff_a, next_state_a = annihilation(1.0, start_state, mode_idx)
                if coeff_a > 0.0 and next_state_a in my_basis:
                    i = my_basis.index(next_state_a)
                    phi_matrix[i, j] += norm * coeff_a * np.exp(1j * (2 * np.pi * k / L) * x)
                    
                # Creation operator term
                coeff_c, next_state_c = creation(1.0, start_state, mode_idx)
                if coeff_c > 0.0 and next_state_c in my_basis:
                    i = my_basis.index(next_state_c)
                    phi_matrix[i, j] += norm * coeff_c * np.exp(-1j * (2 * np.pi * k / L) * x)
                    
        phi_x_matrices.append(phi_matrix)

    # --- 5. ANIMATION ---
    fig, ax = plt.subplots(figsize=(9, 5))
    line, = ax.plot(x_grid, np.zeros_like(x_grid), color='crimson', linewidth=2.0, label="$\\langle \\psi(t) | \\phi(x) | \\psi(t) \\rangle$")
    
    ax.set_ylim(-0.5, 0.5) 
    ax.set_title("Exact Quantum Field Wave Dynamics $\\langle \\phi(x, t) \\rangle$", fontsize=13)
    ax.set_xlabel("Spatial Coordinate (x)", fontsize=11)
    ax.set_ylabel("Field Expectation Value", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    def animate(frame):
        t = frame * 0.05
        phases = np.exp(-1j * eigenvalues * t)
        psi_t = eigenvectors @ (c_n * phases)
        
        field_profile = [np.real(np.vdot(psi_t, phi_mat @ psi_t)) for phi_mat in phi_x_matrices]
        line.set_ydata(field_profile)
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=400, interval=30, blit=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()