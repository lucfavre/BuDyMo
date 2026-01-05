import numpy as np
import os

# ==========================================
# 1. PARAMETERS & CONSTANTS
# ==========================================
class Params:
    def __init__(self):
        # Physical Constants (FC-72 properties at 0.6bar)
        self.A = 0.0              # Hamaker constant (0 if non-wetting)
        self.rhol = 1650.55         # Liquid density [kg/m3]
        self.rhov = 7.49        # Vapor density [kg/m3]
        self.mul = 5.382e-4        # Liquid viscosity [Pa.s]
        self.h_lat = 89282      # Latent heat [J/kg] (renamed from h to avoid confusion with heat transfer coeff)
        self.Tsat = 39.51+273.15        # Saturation Temperature [K]
        self.lambdal = 5.382e-2      # Liquid thermal conductivity [W/m.K]
        self.Rg = 8.3145/0.338           # Gas constant for water [J/kg.K]
        self.f = 1.0              # Accommodation factor
        self.sigma = 9.75e-3        # Surface tension [N/m]

        self.micro = 8e0         # Micro region angle [degrees]
        self.vs = 0.0             # Velocity
        self.vcl = 0.0        # Contact line velocity

        # Interfacial thermal resistance factor
        # Ri = Tsat * sqrt(2*Pi*Rg*Tsat) / (h^2 * rhov) * (2-f)/(2*f)
        self.Ri = (self.Tsat * np.sqrt(2.0 * np.pi * self.Rg * self.Tsat) /
                   (self.h_lat**2 * self.rhov) * (2.0 - self.f) / (2.0 * self.f))

params = Params()

# ==========================================
# 2. THE ODE FUNCTION (Derivative)
# ==========================================
def derivatives(x, y, loop_mode, Tw):
    """
    Calculates dy/dx.
    y[0] -> x (horizontal position)
    y[1] -> delta (film thickness)
    y[2] -> theta (contact angle)
    y[3] -> Pressure variable
    y[4] -> Q (Integrated Heat Flux)
    """
    dy = np.zeros(5)

    # Extract state variables for readability
    # Note: Fortran y(1) is Python y[0], etc.
    curr_x = y[0]
    delta  = y[1]
    theta  = y[2]
    press  = y[3]
    flux   = y[4]

    # Auxiliary variables
    ls = 1.0e-10
    # Avoid division by zero if theta is 0
    sin_theta = np.sin(theta) if abs(theta) > 1e-12 else 1e-12

    r = (delta + ls) / sin_theta
    m = (Tw - params.Tsat) / (params.h_lat * params.Ri)

    # --- Loop 1: Micro-layer Physics ---
    if loop_mode == 1:
        dy[0] = np.cos(theta)
        dy[1] = np.sin(theta)

        # Curvature/Pressure equation
        # Original: y(3) = yn(3) + dx*(yn(4)-m**2*(1.q0/rhov-1.q0/rhol))/sigma
        term_pressure = (press - m**2 * (1.0/params.rhov - 1.0/params.rhol)) / params.sigma
        # Disjoining pressure term A/delta^3 is commented out in some parts of original,
        # but active in others. Keeping strictly to the ACTIVE lines in 'source 19'.
        #dy[2] = term_pressure
        dy[2] = (1/params.sigma) * (press) # - A/delta**3)

        # Hydrodynamics / Pressure gradient
        # term1 represents evaporation recoil or flow resistance
        denom_hydro = 2.0*theta*np.cos(2.0*theta) - np.sin(2.0*theta)
        if abs(denom_hydro) < 1e-15: denom_hydro = 1e-15 # Safety
        fV1 = 8/denom_hydro

        term_hydro_1 =  fV1 * flux / (params.rhol * params.h_lat * (r**3))

        # vcl term (contact line speed)
        denom_vcl = 2.0*theta - np.sin(2.0*theta)
        if abs(denom_vcl) < 1e-15: denom_vcl = 1e-15
        fV2 = 4 * np.sin(theta)/denom_vcl

        term_hydro_2 = (params.vcl / (r**2)) * fV2

        dy[3] = params.mul * (term_hydro_1 - term_hydro_2)

        # Heat Flux
        # Tw - Tsat * (1 + (P + kinetic_terms)/(rho*h))
        #T_interface = params.Tsat * (1.0 + (press + m**2 * (1.0/params.rhov - 1.0/params.rhol)) / (params.rhol * params.h_lat))
        T_interface = params.Tsat * (1.0 + press / (params.rhol * params.h_lat))

        dy[4] = params.lambdal*(Tw - T_interface) / (r * theta + params.lambdal * params.Ri)

    # --- Loop 2: Macro/Adiabatic or Simplified Physics ---
    else:
        dy[0] = np.cos(theta)
        dy[1] = np.sin(theta)
        dy[2] = 0.0 # Angle constant?
        dy[3] = 0.0 # Pressure constant?

        # Simple conduction
        dy[4] = params.lambdal * (Tw - params.Tsat) / (r * theta + params.lambdal * params.Ri)

    return dy

# ==========================================
# 3. MAIN SIMULATION (Shooting Method)
# ==========================================
def solve_mathieu():
    # Setup Output
    output_file = 'python_output.dat'
    f_out = open(output_file, 'w')

    # Simulation Parameters
    Nb_Unknowns = 16 # Not all used in logic
    N_DTw = 5

    # Generate Superheat Array (DTw_list)
    # 0.05 to 30.0 linearly spaced
    DTw_list = np.linspace(0.05, 30.0, N_DTw)

    # Fixed step size
    dx = 1.0e-10

    integ_length = 0.5e-6
    # Main Loop over Superheats
    for i in range(N_DTw):
        DT = DTw_list[i]
        Tw = params.Tsat + DT

        print(f"Calculation {i+1}/{N_DTw}: Tw-Tsat = {DT:.4f}")

        for loop_idx in [1,2]:
            tir_active = True
            send = integ_length if loop_idx == 1 else 1.0e-6 # Integration length

            # Secant Method History
            history = [] # Stores (guess_val, error)

            # Shooting Parameters
            i_tir = 3 # Index of variable to shoot (Python index 3 -> y(4) pressure)
            # Note: Fortran i_tir=4 (y(4)). Python index is 3.

            # Target variable index (Fortran i_target=13, dyend index -> 3)
            # The logic checks dyend(i_target-10). 13-10 = 3. Python index 2 (dy[2] -> angle derivative)

            # Shooting Loop variables
            eps_shoot = 1e-9

            current_xend = 0e-12
            y0_shoot_val = 1e-6 # Initial guess for y[i_tir] (pressure if i_tir=3)

        # Initial Guesses
        angle_deg = params.micro # 28.0
        # Result placeholders
        final_results = None

        y = np.zeros(5)
        y[0] = 0e0

        while y[0] <= send:

            if tir_active:
                # Update the target length incrementally
                if current_xend < 1e-8:
                    current_xend += 1e-10
                elif current_xend < 6e-8:
                    current_xend += 1e-9
                elif current_xend < 2e-7:
                    current_xend += 1e-8
                elif current_xend <= 2e-6:
                    current_xend += 1e-7
                elif current_xend <= 1e-4:
                    current_xend += 1e-6
                else:
                    current_xend += 1e-5

                if current_xend >= 1e-8: eps_shoot = 1e-15
                if current_xend >= 2e-8: eps_shoot = 1e-27
                if current_xend >= 3e-8: eps_shoot = 1e-28
                if current_xend >= 4e-8: eps_shoot = 1e-29

            else:
                current_xend = send

            # --- Set Initial Conditions ---
            y[0] = 0.0 # x
            y[1] = 0.0 # thickness (approx 0)
            y[2] = angle_deg * np.pi / 180 # theta

            # Apply shooting guess
            if tir_active:
                y[3] = y0_shoot_val # Pressure guess
            else:
                # Fixed BC if not shooting
                y[3] = 0.0

            y[4] = 0.0 # Flux start

            iteration = 0
            max_iter = 10

            while iteration < max_iter:
                iteration += 1
                # Hack from source 8:
                # y0(2) = y0(2)*(1.q0+0.q-3) -> Modifies thickness slightly?
                # Keeping simple for now as y[1] starts at 0.

                # --- Integration (RK4 Manual Implementation) ---
                current_x_integ = 0.0

                # Integration Loop
                # Stop when x coordinate (y[0]) > send
                while y[0] <= current_xend:
                    yn = y.copy()

                    # RK4 Step
                    k1 = derivatives(current_x_integ, yn, loop_idx, Tw)

                    k2_y = yn + 0.5 * dx * k1
                    k2 = derivatives(current_x_integ + 0.5*dx, k2_y, loop_idx, Tw)

                    k3_y = yn + 0.5 * dx * k2
                    k3 = derivatives(current_x_integ + 0.5*dx, k3_y, loop_idx, Tw)

                    k4_y = yn + dx * k3
                    k4 = derivatives(current_x_integ + dx, k4_y, loop_idx, Tw)

                    # Update
                    dy_step = (1.0/6.0) * (k1 + 2*k2 + 2*k3 + k4)
                    y = yn + dx * dy_step

                    # Step x (global integration var, distinct from y[0])
                    current_x_integ += dx
#                    print(f'{y[2]*180/np.pi}, {send}')

                # --- Shooting Convergence Check ---
                # Calculate final derivative at end
                dy_end = derivatives(current_x_integ, y, loop_idx, Tw)

                # Target: Derivative of angle (dy[2]) should be 0?
                # Fortran: dyendn(i_target-10). i_target=13 => index 3 in Fortran => index 2 in Python (theta derivative)
                error = dy_end[2]

                if not tir_active:
                    break

                # Store history for Secant Method
                history.append((y0_shoot_val, error))

                if abs(error) < 1e-6:
                    print(f"  Converged after {iteration} iterations.")
                    break

                # --- Secant Method Update ---
                if iteration == 1:
                    # Perturb guess slightly
                    y0_shoot_val = y0_shoot_val * (1.0 + 1e-3)
                else:
                    # New guess = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
                    val_curr, err_curr = history[-1]
                    val_prev, err_prev = history[-2]

                    if abs(err_curr - err_prev) < 1e-16:
                        y0_shoot_val *= 1.01 # Avoid div by zero
                    else:
                        y0_shoot_val = val_curr - err_curr * (val_curr - val_prev) / (err_curr - err_prev)

            # Store results from Loop 1
            if loop_idx == 1:
                final_results = y

                # Write to file: DTw_list, x, delta, angle(deg), pressure, flux, initial_pressure_guess
                # Fortran: write(44,...) DTw_list(i),y(1), y(2), 180.q0/Pi*y(3), y(4), y(5), y0(i_tir)
                data_str = (f"{DT:.6e} {y[0]:.6e} {y[1]:.6e} {np.rad2deg(y[2]):.6e} "
                            f"{y[3]:.6e} {y[4]:.6e} {history[-1][0]:.6e}\n")
                f_out.write(data_str)
                f_out.flush() # Ensure write

    f_out.close()
    print(f"Simulation complete. Results saved to {output_file}")
