from fluid_ppt import *
# Compute force balance over a bubble in given conditions

fluid = 'water'
P_Pa = 1.01e5 # Pressure in Pa
qw = 1.0e4 # Heat flux in W/m2
DTw = 10 # Wall superheat in K
DTl = 5 # Liquid subcooling in K
Gl = 500 # Liquid mass flow in kg/m2/s
### FORCE BALANCE CALCULATION

# Computing FC-72 properties at the case pressure
ppt = fluid_ppt_sat(fluid, P_Pa)
sigma = ppt['sigma']
rho_l = ppt['rho_l']
nu_l = ppt['nu_l']
rho_v = ppt['rho_v']

# Simulation setup and important properties
t0 = 0 # Starting time in s
tmax = 0.1 # End time in s
time = np.linspace(0, 0.1, 1000) # Time instants list
indexes = [i for i in range(len(time))] # Indexes
n_t = len(indexes) # Number of instants

Req = [

# Useful quantities and dimensionless numbers
Rdot = [(R_m[i+1] - R_m[i-1])/(time[i+1] - time[i-1]) for i in img_l]
Ub_dot = [(vx_filtered[i+1] - vx_filtered[i-1])/(time[i+1] - time[i-1]) for i in img_l]
VbUb_dot = [(V_m3[i-1]*vx_filtered[i+1] - V_m3[i-1]*vx_filtered[i-1])/(time[i+1] - time[i-1])
           for i in img_l]


# Dimensionless numbers for Shi et al. lift and drag coeffs.
Re_b = [2*R_m[i]*abs(urel_GC[i])/nu_l for i in img_l]
dy = 1e-7 # y step for shear rate calculation
gamma = [(uliq_rubi(Ql_real, xGC_m[i], yGC_m[i]+dy) - uliq_rubi(Ql_real, xGC_m[i], yGC_m[i]-dy))/(2*dy)
         for i in img_l]
Sr = [gamma[i] * 2 * R_m[i] / urel_GC[i-1] for i in img_l]
Lr = [yGC_m[i] / R_m[i] for i in range(n_img)]
Lu = [yGC_m[i] * abs(urel_GC[i-1]) / nu_l for i in img_l]
Lom = [yGC_m[i] * np.sqrt(gamma[i] / nu_l) for i in img_l]

# Capillary force (Klausner et al. 1993)
Fsx = [-np.pi * R_m[i] * sigma * 2.5 * (Rf_m[i]/R_m[i]) * dtheta_j[i] / ((np.pi/2)**2 - dtheta_j[i]**2) * np.sin(theta_j[i]) * np.cos(dtheta_j[i])
       for i in img_l]
Fsy = [-np.pi * R_m[i] * sigma * 2 * (Rf_m[i]/R_m[i]) * np.sin(theta_j[i]) * np.sin(dtheta_j[i])/dtheta_j[i]
       for i in img_l]

# Contact pressure force
Fcp = [2 * np.pi * R_m[i] * sigma * np.sin(theta_j[i])**2
      for i in img_l]

# Drag and lift force
CD = [CD_shi_adim(Re_b[i], Sr[i], Lr[i], Lu[i]) for i in img_l]
Fd = [0.5 * CD[i] * rho_l * np.pi * R_m[i]**2 * abs(urel_GC[i]) * urel_GC[i]
      for i in img_l]

CL = [CL_shi_adim(Re_b[i], Sr[i], Lr[i], Lu[i], Lom[i]) for i in img_l]
Fl = [0.5 * CL[i] * rho_l * np.pi * R_m[i]**2 * abs(urel_GC[i])**2
      for i in img_l]

# Added mass force
Cam_x = 0.636
Fam_x = [rho_l * V_m3[i] * (3*Cam_x*Rdot[i]/R_m[i] * urel_GC[i] - Cam_x * Ub_dot[i])
        for i in img_l]

# Bubble inertia
bub_in = [rho_v * VbUb_dot[i] for i in img_l]

# Sum of forces
sum_Fx = [Fsx[i] + Fd[i] - bub_in[i] for i in range(n_img)]

sum_Fy = [Fsy[i] + Fcp[i] + Fl[i] for i in range(n_img)]


# Prepare the plots
fig, axs = plt.subplots(1,2, figsize=(10,4), constrained_layout=True)
ax1 = axs[0]
ax2 = axs[1]
plot_title = fr' $P$={Pmbar}~mbar, $q_w$={qw}~W/cm$^2$, $\Delta T_l$={DTl}\textdegree C, $Q_l$={Ql}~mL/min, $t_w$={tw}~s'
fig.suptitle(plot_title)
setup_ax(ax1, r'$t$ [s]', r'$F_x$ [N]')
setup_ax(ax2, r'$t$ [s]', r'$F_y$ [N]')

# Plotting x force balance
ax = ax1
ax.plot(time_l, Fsx, c='purple', label=r'$F_{C,x}$')
ax.plot(time_l, Fd, c='green', label = r'$F_{D}$')
ax.plot(time_l, bub_in, c='red', label = r'Inertia')
ax.plot(time_l, sum_Fx, c='black', linestyle='--', label=r'$\sum F_x$')

ax.legend(loc='best')

# Plotting y force balance
ax = ax2
ax.plot(time_l, Fsy, c='purple', label=r'$F_{C,y}$')
ax.plot(time_l, Fcp, c='blue', label=r'$F_{CP}$')
ax.plot(time_l, Fl, c='green', label=r'$F_{L}$')
ax.plot(time_l, sum_Fy, c='black', linestyle='--', label=r'$\sum F_y$')

ax.legend(loc='best')

plt.savefig(f'force_balance.pdf', format='pdf', bbox_inches='tight')
plt.clf()
plt.close()
print('Done')
