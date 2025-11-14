import sys
sys.path.append('..')

import numpy as np

from fluid_ppt import *
from func_plot import *
from drag_coeffs import *

# Compute force balance over a bubble in given conditions
fluid = 'water'
P_Pa = 1.01e5 # Pressure [Pa]
qw = 1.0e4 # Heat flux [W/m2]
DTw = 1 # Wall superheat [K]
DTl = 5 # Liquid subcooling [K]
Gl = 100 # Liquid mass flow [kg/m2/s]
Dh = 10e-3 # Hydraulic diameter [m]
theta_deg = 40 # Contact angle [degrees]
theta = np.pi * theta_deg / 180 # Contact angle [rad]
dtheta_deg = 5 # Contact angle half hystersis [degrees]
dtheta = np.pi * dtheta_deg / 180 # Contact angle half hysteresis [rad]

# Computing fluid properties at given pressure
ppt = fluid_ppt_sat(fluid, P_Pa) # Properties dictionnary
sigma = ppt['sigma'] # Surface tension [N/m2]
rho_l = ppt['rho_l'] # Liquid density [kg/m3]
nu_l = ppt['nu_l'] # Liquid kinematic viscosity [m2/s]
mu_l = ppt['mu_l'] # Liquid dynamic viscosity [Pa.s]
cp_l = ppt['cp_l'] # Liquid heat capacity [J/kg/K]
rho_v = ppt['rho_v'] # Vapor density [kg/m3]
h_lv = ppt['h_lv'] # Enthalpy of vaporisation [J/kg]
alpha_l = ppt['alpha_l'] # Liquid heat diffusivity [m2/s]


# Function for equivalent radius calculation over time and its two first derivatives

def fReq(t):
    Jaw = rho_l * cp_l * DTw / (rho_v * h_lv)
    K = 2
    res = K * Jaw * np.sqrt(alpha_l * t)
    return res

def fRdot(t):
    Jaw = rho_l * cp_l * DTw / (rho_v * h_lv)
    K = 2
    res = K * Jaw * np.sqrt(alpha_l) / (2 * np.sqrt(t))
    return res

def fRddot(t):
    Jaw = rho_l * cp_l * DTw / (rho_v * h_lv)
    K = 2
    res = K * Jaw * np.sqrt(alpha_l) / (-4 * t**(3/2))
    return res

# Function for bubble foot radius evolution over time

def fRf(t):
    res = fReq(t) * np.sin(theta)
    return res


# Function for turbulent liquid velocity profile and its derivative

# Reichardt dimensionless liquid velocity profile
def uplus_reich(yplus):
    kappa = 0.41
    chi = 11
    c = 7.4
    uplus = (1 / kappa) * np.log(1 + kappa * yplus) + c * (1 - np.exp(-yplus / chi) + (yplus / chi) * np.exp(- yplus / 3)  ) #Reichardt law

    return uplus

# Reichardt dimensionless wall shear rate profile
def duplus_reich(yplus):
    kappa = 0.41
    chi = 11
    c = 7.4
    duplus=(1 / (1 + kappa * yplus)) + (c / chi) * (np.exp(-yplus / chi) + (1 - yplus / 3) * np.exp(-yplus / 3) )

    return duplus

# Calculation of the wall friction velocity using Mac Adams correlation

Re_Dh = Gl * Dh / mu_l # Bulk flow Reynolds number
Cf = 0.036 * (Re_Dh**(-0.1818)) #Mac Adams friction factor
tau_w = (Cf/2) * rho_l * (Gl / rho_l)**2 # Mac Adams correlation
u_tau = np.sqrt(tau_w / rho_l)


# Simulation setup and important properties
t0 = 1e-5 # Starting time in s
tmax = 0.5 # End time in s
time = np.linspace(t0, tmax, 1000) # Time instants list
indexes = [i for i in range(len(time))] # Indexes
n_t = len(indexes) # Number of instants

Req = [fReq(t) for t in time]
Rdot = [fRdot(t) for t in time]
Rf = [fRf(t) for t in time]
Vb = [(4/3) * np.pi * (r**3) for r in Req]

# List to save force results
Fsx_list = []
Fsy_list = []
Fcp_list = []
Fd_list = []
Fl_list = []
Fam_x_list = []
Fam_y_list = []

sum_Fx = []
sum_Fy = []

for i in range(len(time)):
    t = time[i]

    # Compute local liquid velocity and shear
    yplus = Req[i] * u_tau / nu_l
    uplus = uplus_reich(yplus)
    duplus = duplus_reich(yplus)

    Ul = uplus * u_tau
    gamma = (u_tau**2 / nu_l) * duplus

    Ub = 0 # Force balance for static bubble, no sliding velocity yet
    Ub_dot = 0 # No sliding: zero bubble velocity and acceleration
    Urel = Ul - Ub

    # Dimensionless numbers for drag and lift coefficients
    Re_b = rho_l * abs(Urel) * 2 * Req[i] / mu_l
    Sr = gamma * 2 * Req[i] / Urel
    Lr = 1 # Bubble height = Bubble radius
    Lu = Req[i] * abs(Urel) / nu_l
    Lom = Req[i] * np.sqrt(gamma / nu_l)

    # Capillary force (Klausner et al. 1993)
    Fsx = -np.pi * Req[i] * sigma * 2.5 * (Rf[i]/Req[i]) * dtheta / ((np.pi/2)**2 - dtheta**2) * np.sin(theta) * np.cos(dtheta)
    Fsy = -np.pi * Req[i] * sigma * 2 * (Rf[i]/Req[i]) * np.sin(theta) * np.sin(dtheta)/dtheta

    # Contact pressure force
    Fcp = 2 * np.pi * Req[i] * sigma * np.sin(theta)**2

    # Drag and lift forces
    Cd = CD_shi_adim(Re_b, Sr, Lr, Lu)
    Fd = 0.5 * Cd * rho_l * np.pi * Req[i]**2 * abs(Urel) * Urel

    Cl = CL_shi_adim(Re_b, Sr, Lr, Lu, Lom)
    Fl = 0.5 * Cl * rho_l * np.pi * Req[i]**2 * abs(Urel)**2

    # Added mass force
    Cam_x = 0.636
    Fam_x = rho_l * Vb[i] * Cam_x * (3 * Rdot[i]/Req[i] * Urel - Ub_dot)

    Cam_y1 = 0.27
    Cam_y2 = 0.326
    Fam_y = rho_l * Vb[i] * ((-3*Cam_y1 + Cam_y2) * Rdot[i]**2 / Req[i] - Cam_y1 * fRddot(t))

    # Saving values
    Fsx_list.append(Fsx)
    Fsy_list.append(Fsy)
    Fcp_list.append(Fcp)
    Fd_list.append(Fd)
    Fl_list.append(Fl)
    Fam_x_list.append(Fam_x)
    Fam_y_list.append(Fam_y)
    sum_Fx.append(Fsx + Fd + Fam_x)
    sum_Fy.append(Fsy + Fcp + Fl + Fam_y)


# Prepare the plots
fig, axs = plt.subplots(1,3, figsize=(18,6), constrained_layout=True)
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

plot_title = fr' $P$={round(P_Pa/1e5,2)}~bar, $\Delta T_w$={DTw}\textdegree C, $\Delta T_l$={DTl}\textdegree C, $G_l$={Gl}~kg/m$^2$/s'
setup_ax(ax1, r'$t$ [s]', r'$y$ [m]')
ax1b = ax1.twiny()
ax1b.set_xlabel(r'$U_l$ [m/s]')
setup_ax(ax2, r'$t$ [s]', r'$F_x$ [N]')
setup_ax(ax3, r'$t$ [s]', r'$F_y$ [N]')


# Plotting radius over time and local liquid velocity as function of height
yList = np.linspace(0, 1.15*max(Req), 300)
Ul_list = [u_tau * uplus_reich(y*u_tau/nu_l) for y in yList]
ax1b.plot(Ul_list, yList, '--k', label = r'$U_l$')
ax1.plot(time, Req, '-g', label = r'$R_{eq}$')
ax1.legend(loc='upper left')
ax1b.legend(loc='lower right')


# Plotting x force balance
ax = ax2
ax.plot(time, Fsx_list, c='purple', label=r'$F_{C,x}$')
ax.plot(time, Fd_list, c='green', label = r'$F_{D}$')
ax.plot(time, Fam_x_list, c='red', label = r'$F_{AM,x}$')
ax.plot(time, sum_Fx, c='black', linestyle='--', label=r'$\sum F_x$')

ax.legend(loc='best')

# Plotting y force balance
ax = ax3
ax.plot(time, Fsy_list, c='purple', label=r'$F_{C,y}$')
ax.plot(time, Fcp_list, c='blue', label=r'$F_{CP}$')
ax.plot(time, Fl_list, c='green', label=r'$F_{L}$')
ax.plot(time, Fam_y_list, c='red', label=r'$F_{AM,y}$')
ax.plot(time, sum_Fy, c='black', linestyle='--', label=r'$\sum F_y$')

ax.legend(loc='best')

plt.savefig(f'force_balance.pdf', format='pdf', bbox_inches='tight')
plt.clf()
plt.close()
print('Done')
