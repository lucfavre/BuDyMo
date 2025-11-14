import numpy as np
import CoolProp.CoolProp as CP

# Function returning a dictionnary containing the FC-72 properties at given Psat
# Based on the bibliography work of LOTHAR, University of Pisa, Italy
def fc72_ppt(Psat):
    # Input value of saturation pressure is in bar

    Tsat = -86.699 + 143.278 * (Psat**0.204)                                                # Saturation temperature [degree C]
    t = Tsat                                                                                # Variable name used in the following formulae
    beta =  0.00261 / (1.74 - 2.62e-3 * t)                                                  # Expansion coefficient beta [1/K]
    cp_l = 589.18 + 1.5443 * (t+273.15)                                                     # Liquid specific heat [J/kg/K]
    hlv = -0.4984 * (t**2) - 230.89 * t + 99179.99                                          # Vaporization enthalpy [J/kg]
    lam_l = 9.0672e-2 - (1.168e-4 * (t + 273.15))                                           # Liquid thermal conductivity [W/m/K]
    nu_l = 1e-7 * (-4.102e-6 * (t**3) + 8.759e-4 * (t**2) - 8.259e-2 * t + 5.409)           # Liquid kinematic viscosity [m2/s]
    rho_l = 1755.59 - 3.101e-4 * (t**3) + 4.503e-2 * (t**2) - 3.954 * t                     # Liquid density [kg/m3]
    rho_v = 2.063e-7* (t**4) + 2.893e-7 * (t**3) + 1.856e-3 * (t**2) + 6.842e-2 * t +1.3724 # Saturated vapor density [kg/m3]
    sigma = 4.2705e-2 * ((1 - (t + 273.5) / 451.65)**1.2532)                                # Surface tension [N/m]
    epsilon_r = 1.7834 - 0.00157 * t                                                        # Relative dielectric constant [-]
    ir = 1.268679 - 0.000644 * t                                                            # Refractive index [-]
    mu_l = rho_l * nu_l                                                                     # Liquid dynamic viscosity [Pa.s]
    alpha_l = lam_l / (rho_l * cp_l)                                                        # Liquid thermal diffusivity [m2/s]
    g = 9.8066                                                                              # Earth gravity constant [m/s2]

    vals = [Tsat, beta, cp_l, hlv, lam_l, nu_l, rho_l, rho_v, sigma, epsilon_r, ir, mu_l, alpha_l]
    headers = ['Tsat', 'beta', 'cp_l', 'h_lv', 'lam_l', 'nu_l', 'rho_l', 'rho_v', 'sigma', 'eps_r', 'ir', 'mu_l', 'alpha_l']
    res = {}
    for i in range(len(headers)):
        res[headers[i]] = vals[i]

    return res

### PHYSICAL PROPERTIES FUNCTIONS

## Functions returning the physcial properties of a given fluid at pressure P_Pa, at saturation temperature is no subcooling is specified
def fluid_ppt_sat(fluid, P_Pa):
    if fluid == 'FC72':
        res = fc72_ppt(P_Pa/1.01e5)
        cp_v = 0
        lam_v = 0
        mu_v = 0
        nu_v = 0
        alpha_v = 0
        res['cp_v'] = cp_v
        res['lam_v'] = lam_v
        res['mu_v'] = mu_v
        res['nu_v'] = nu_v
        res['alpha_v'] = alpha_v

        return res

    elif fluid == 'FC87': #Properties given in Thorncroft et al., IJHMT, 1998 at 1 atm
        Tsat = 29.3 # Celsius
        rho_l = 1750 # kg/m3
        rho_v = 12.5 # kg/m3
        mu_l = 458e-6 # kg/m/s = Pa.s
        nu_l = mu_l / rho_l # Pa.s
        mu_v = 0 # Not available
        nu_v = 0
        cp_l = 1.09e3 # J/kg/K
        cp_v = 0 # Not available
        Pr_l = 9.03 #Liquid Prandtl number
        lam_l = mu_l * cp_l / Pr_l # W/m/K
        alpha_l = lam_l / (rho_l * cp_l)
        lam_v = 0 # Not available
        alpha_v = 0

        h_lv = 31.3e3 # J/kg
        sigma = 8.97e-3 # N/m

        beta = 0
        epsilon_r = 0
        ir = 0

    else:
        Tsat = CP.PropsSI('T', 'P', P_Pa, 'Q', 0, fluid) - 273.15 #degC
        rho_l = CP.PropsSI('D', 'P', P_Pa, 'Q', 0, fluid) #kg/m3
        rho_v = CP.PropsSI('D', 'P', P_Pa, 'Q', 1, fluid) #kg/m3
        mu_l = CP.PropsSI('VISCOSITY', 'P', P_Pa, 'Q', 0, fluid) #Pa.s
        nu_l = mu_l / rho_l
        mu_v = CP.PropsSI('VISCOSITY', 'P', P_Pa, 'Q', 1, fluid) #Pa.s
        nu_v = mu_v / rho_v
        cp_l = CP.PropsSI('CPMASS', 'P', P_Pa, 'Q', 0, fluid) #J/kg/K
        cp_v = CP.PropsSI('CPMASS', 'P', P_Pa, 'Q', 1, fluid) #J/kg/K
        lam_l = CP.PropsSI('L', 'P', P_Pa, 'Q', 0, fluid) #W/m/K
        alpha_l = lam_l / (rho_l * cp_l)
        lam_v = CP.PropsSI('L', 'P', P_Pa, 'Q', 1, fluid) #W/m/K
        alpha_v = lam_v / (rho_v * cp_v)

        h_lv = CP.PropsSI('HMASS', 'P', P_Pa, 'Q', 1, fluid) - CP.PropsSI('HMASS', 'P', P_Pa, 'Q', 0, fluid) #J/kg/K
        sigma = CP.PropsSI('SURFACE_TENSION', 'P', P_Pa, 'Q', 0, fluid) #N/m = J/m2

        beta = 0
        epsilon_r = 0
        ir = 0

    vals = [Tsat, rho_l, rho_v, mu_l, nu_l, mu_v, nu_v,
            cp_l, cp_v, lam_l, alpha_l, lam_v, alpha_v,
            h_lv, sigma, beta, epsilon_r, ir]
    headers = ['Tsat', 'rho_l', 'rho_v', 'mu_l', 'nu_l', 'mu_v', 'nu_v',
               'cp_l', 'cp_v', 'lam_l', 'alpha_l', 'lam_v', 'alpha_v',
               'h_lv', 'sigma', 'beta', 'eps_r', 'ir']
    res = {}
    for i in range(len(headers)):
        res[headers[i]] = vals[i]

    return res
