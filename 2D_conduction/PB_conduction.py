import sys
sys.path.append('..')
from func_plot import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time
import json

rc('font',**{'family':'serif','serif':['Computer Modern'], 'size' : 16})

def solve_implicit_heat_transfer():

    #Physical parameters
    qw = 7500 #W/m2
    T0 = 312.55 #K
    tw = 5 #s

    # Function for non-uniform heat flux
    def coeff(x):
        c=0
        if abs(x)>9e-3:
            c=0
        elif 7.63e-3<=abs(x)<=9e-3:
            c=0.78
        elif 2.75e-3<=abs(x)<=7.63e-3:
            c=6.256e3*(x**2)-1.511e2*x*np.sign(x)+1.575
        else:
            c=1.21
        return c


    # ==========================================
    # 1. Geometry & Grid Generation
    # ==========================================
    Lx = 0.04        # Width [m]
    Ly_solid = 0.005  # Solid thickness [m]
    Ly_liquid = 0.005 # Liquid thickness [m]
    Ly = Ly_solid + Ly_liquid

    Nx = 800          # Grid points in X
    Ny = 200        # Grid points in Y

    dx = Lx / Nx
    dy = Ly / Ny

    # 1D coordinates for plotting
    x = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, Nx)
    y = np.linspace(-Ly_solid+dy/2, Ly_liquid - dy/2, Ny)
    X, Y = np.meshgrid(x, y)

    R_roi = 5e-3
    n0_roi = int((Lx/2 - R_roi)/dx)
    n1_roi = int((Lx/2 + R_roi)/dx)

    # ==========================================
    # 2. Material Properties (Field Generation)
    # ==========================================
    # Example: Copper (Solid) & Water (Liquid)
    # Solid Properties
    k_s, rho_s, cp_s = 10.9, 4890.0, 410.0
    # Liquid Properties
    k_l, rho_l, cp_l = 6.387e-2, 1624, 1074.4

    # Initialize property maps
    K_map = np.zeros((Ny, Nx))
    RhoCp_map = np.zeros((Ny, Nx))

    # Identify the interface row index
    # We use cell centers. If y < 0, it's solid.
    interface_row_idx = int(Ly_solid / dy)

    # Fill Solid
    K_map[:interface_row_idx, :] = k_s
    RhoCp_map[:interface_row_idx, :] = rho_s * cp_s

    # Fill Liquid
    K_map[interface_row_idx:, :] = k_l
    RhoCp_map[interface_row_idx:, :] = rho_l * cp_l

    # ==========================================
    # 3. Source Term Definition
    # ==========================================
    # Source only in the LAST row of the solid (interface_row_idx - 1)
    # Non-uniform profile (e.g., Gaussian)
    Q_vol = np.zeros((Ny, Nx))

    source_row = interface_row_idx - 1
    center_x_idx = Nx // 2
    sigma_idx = Nx // 10
    x_coords = np.linspace(-Lx/2+dx/2, Lx/2-dx/2, Nx)
    q_vol = np.array([coeff(x)*qw/dy for x in x_coords])


    # Normalize to ensure total integrated power is correct
    # Power = Sum(Q_vol * Volume_cell)
    # Volume_cell = dx * dy * 1 (depth)

    Q_vol[source_row, :] = q_vol



    # ==========================================
    # 4. Implicit Matrix Assembly (Ax = b)
    # ==========================================
    print("Assembling System Matrix...")

    N = Nx * Ny  # Total number of unknowns
    A = lil_matrix((N, N)) # Use LIL format for efficient setup
    b = np.zeros(N)

    dt = 0.02    # Time step [s] (Much larger than explicit limit)
    t_final = tw
    steps = int(t_final / dt)

    # Helper to get 1D index from 2D (i, j)
    def get_idx(r, c):
        return r * Nx + c

    # Pre-calculate Harmonic Means for coefficients
    # This assumes properties don't change with time (Temperature independent)
    # If properties define T, this must move inside the time loop (Non-linear)

    for r in range(Ny):
        for c in range(Nx):
            idx = get_idx(r, c)

            # Current Cell Properties
            rho_cp = RhoCp_map[r, c]
            k_p = K_map[r, c]

            # Unsteady Term coeff: (rho*cp*dx*dy)/dt
            ap_0 = (rho_cp * dx * dy) / dt

            # Initialize neighbors coefficients
            ae = aw = an = as_ = 0.0

            # --- WEST NEIGHBOR (c-1) ---
            if c > 0:
                k_w = K_map[r, c-1]
                k_face = (2 * k_p * k_w) / (k_p + k_w) # Harmonic Mean
                aw = k_face * dy / dx

            # --- EAST NEIGHBOR (c+1) ---
            if c < Nx - 1:
                k_e = K_map[r, c+1]
                k_face = (2 * k_p * k_e) / (k_p + k_e)
                ae = k_face * dy / dx

            # --- SOUTH NEIGHBOR (r-1) ---
            if r > 0:
                k_s_neigh = K_map[r-1, c]
                k_face = (2 * k_p * k_s_neigh) / (k_p + k_s_neigh)
                as_ = k_face * dx / dy

            # --- NORTH NEIGHBOR (r+1) ---
            if r < Ny - 1:
                k_n = K_map[r+1, c]
                k_face = (2 * k_p * k_n) / (k_p + k_n)
                an = k_face * dx / dy

            # Boundary Conditions (Implicitly handled by coeff=0 for Adiabatic)
            # Top Boundary (Liquid Surface) -> Fixed Temperature (Dirichlet)
            # We modify the equation for the top row cells later or handle here.
            # Here, we treat r=Ny-1 as having a ghost node or fixed value.
            # Let's use Source Term Method for Dirichlet at Top: T = 300
            if r == Ny - 1:
                # To fix T, we set A[idx, idx] = very_large and b[idx] = very_large * T_fixed
                # This overrides the heat balance equation.
                large_num = 1e12
                A[idx, idx] = large_num
                # b[idx] will be set in loop
                continue

            # Central Coefficient (Implicit)
            ap = ap_0 + ae + aw + an + as_

            # Fill Matrix
            A[idx, idx] = ap
            if c > 0:      A[idx, idx - 1]  = -aw
            if c < Nx - 1: A[idx, idx + 1]  = -ae
            if r > 0:      A[idx, idx - Nx] = -as_
            if r < Ny - 1: A[idx, idx + Nx] = -an
    # Convert to CSR for fast solving
    A_csr = A.tocsr()
    print("Matrix Assembled. Starting Time Loop.")

    if N <= 100:
        # Assuming 'A_csr' is your assembled matrix from the previous code
        # 1. Convert sparse to dense
        A_dense = A_csr.toarray()
        # 2. Print with specific formatting for alignment
        print("Matrix A structure (Coefficients):")
        for row in A_dense:
            # Formats each number to 1 decimal place with a width of 6
            print(" ".join(f"{val:6.1f}" for val in row))


    # ==========================================
    # 5. Time Stepping Loop
    # ==========================================
    T = np.ones(N) * T0 # Initial Temp 300K

    # Store history for plotting center profile
    history_center = []
    history_Tis = []
    history_Til = []
    t_list = []

    start_time = time.time()

    print(f"Grid: {Nx}x{Ny}, Time Steps: {steps}, dt: {dt:.2e} s, dx: {dx:.2e}m, dy:{dy:.2e}m")
    for n in range(steps):
        t_list.append(n*dt)
        t_cpu_it = time.time()

        # Construct RHS (b vector)
        b[:] = 0.0

        # Iterate to fill b (Vectorization possible, but explicit loop clearer for logic)
        # b = ap_0 * T_old + Source

        # We need to rebuild b specifically because of the ap_0 * T_old term
        # Efficient way:
        # b = (RhoCp / dt * V) * T_old + Source * V

        # 1. Unsteady part
        # Flatten maps for vectorized operations
        rho_cp_flat = RhoCp_map.flatten()
        vol = dx * dy
        ap_0_vec = (rho_cp_flat * vol) / dt

        b = ap_0_vec * T

        # 2. Add Source Term
        b += Q_vol.flatten() * vol

        # 3. Apply Boundary Conditions to RHS
        # Top Wall Fixed T = 300
        # Indices of top row
        top_indices = np.arange(N - Nx, N)
        large_num = 1e12
        b[top_indices] = large_num * T0

        # Solve System
        T_new = spsolve(A_csr, b)
        T = T_new

        if n % 10 == 0:
            history_center.append(T[get_idx(source_row, center_x_idx)])

        T_field = T.reshape((Ny, Nx))
        history_Tis.append(np.mean(T_field[interface_row_idx-1,n0_roi:n1_roi]))
        history_Til.append(np.mean(T_field[interface_row_idx,n0_roi:n1_roi]))

        t_cpu_it_end = time.time()
        print(f"Iteration {n}: t={n*dt:.2e}s, CPU time={t_cpu_it_end - t_cpu_it:.2e}s")


    print(f"Simulation finished in {time.time() - start_time:.2f}s")

    # Reshape for plotting
    T_field = T.reshape((Ny, Nx))

    # ==========================================
    # 6. Visualization
    # ==========================================
    fig, ax = plt.subplots(1, 3, figsize=(21, 6))

    # Plot 1: 2D Temperature Field
    im = ax[0].imshow(T_field, origin='lower', extent=[-Lx/2, Lx/2, -Ly_solid, Ly_liquid], cmap='coolwarm', aspect='auto')
    ax[0].axhline(y=0, color='black', linestyle='--', linewidth=1, label='L/S Interface')
    ax[0].set_title(f"Temperature Map (t={t_final}s)")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[0].legend()
    plt.colorbar(im, ax=ax[0], label='T [K]')

    # Plot 2: Vertical Profile through the heat source center
    # Plot JK result
    with open('T2_JK.json', 'r') as json_file:
        Tc_json = json.load(json_file)

    center_idx = Nx // 2
    T_JK = [float(e['x']) for e in Tc_json]
    y_JK = [float(e['y'])*1e-3 for e in Tc_json]

    setup_ax(ax[1], xlabel = r'$T$ [K]', ylabel=r'$y$ [m]')
    ax[1].plot(T_JK, y_JK, '--og', label='J. Kind')
    ax[1].plot(T_field[:, center_idx], y, 'r-', label='Centerline Profile')
    ax[1].axhline(y=0, color='k', linestyle='--', label='L/S Interface')
    ax[1].set_title("Vertical Temperature Profile")
    ax[1].grid(True)
    ax[1].legend()

    # Plot 3: Avg interface temp. over time
    # Plot JK result
    with open('T1_JK.json', 'r') as json_file:
        Ti_json = json.load(json_file)

    center_idx = Nx // 2
    t_JK = [float(e['x']) for e in Ti_json]
    T_JK = [float(e['y']) for e in Ti_json]

    T_avg = (np.array(history_Tis)+np.array(history_Til))/2

    setup_ax(ax[2], ylabel = r'$T$ [K]', xlabel=r'$t$ [s]')
    ax[2].plot(t_JK, T_JK, '--og', label='J. Kind')
    ax[2].plot(t_list, history_Tis, 'r-', label=r'$T_w$')
    ax[2].plot(t_list, history_Til, 'b-', label=r'$T_l$')
    ax[2].plot(t_list, T_avg, ls='--', color='orange', label='Average')
    ax[2].set_title("Average $T$ at L/S interface, for -5mm$<x<$5mm")
    ax[2].grid(True)
    ax[2].legend()

    fig.suptitle(rf'Simulation results, $q_w=${qw/1e4:.2f}~W/cm$^2$, $t_w$={t_final:.2f}~s, $T_0=${T0:.2f}~K, $\mathrm{{d}}x=${dx:.2e}~m, $\mathrm{{d}}y=${dy:.2e}~m, $\mathrm{{d}}t$={dt:.2e}~s')

    plt.tight_layout()
    plt.savefig('Result.pdf')

if __name__ == "__main__":
    solve_implicit_heat_transfer()
