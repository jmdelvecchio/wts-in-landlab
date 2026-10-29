""" Script for comparison of 1D performance of the depth-integrated model with analytical solutions"""


#%%
import numpy as np
import matplotlib.pyplot as plt

from landlab import RasterModelGrid, imshow_grid
from model.water_track_model import WaterTrackModelThermal
from tqdm import tqdm

#%%

def stefan_mod(lam, k_u, T_s, t, phi, rho_w, L):
    """Modified Stefan Equation or Modified Berggren Equation (Kurylyk and Hayashi, 2016)"""
    return lam * np.sqrt((2*k_u*T_s*t)/(phi*rho_w*L))

def calc_st1(c_u, rho_u, rho_w, T_s, L, phi):
    """Calculate the Stefan number for the thaw case"""
    return (c_u * rho_u * T_s) / (rho_w * L * phi)

def calc_st2(c_f, rho_f, rho_w, T_s, L, phi):
    """Calculate the Stefan number for the freeze case"""
    return (c_f * rho_f * (-T_s)) / (rho_w * L * phi)

def calc_beta(c_f, rho_f, k_f, c_u, rho_u, k_u):
    return np.sqrt((k_f * rho_f * c_f)/(k_u * rho_u * c_u))

def calc_delta(c_f, rho_f, k_f, c_u, rho_u, k_u):
    return (k_u * c_f * rho_f)/(k_f * c_u * rho_u)

def lamda_thaw_0(S_T1):
    """Thaw correction factor when T_i=0 (Kurylyk and Hayashi, 2016, Equation 23)"""
    return 1 - 0.16*S_T1 + 0.038*S_T1**2

def lambda_thaw(S_T1, T_i, T_s, beta):
    """Thaw correction factor T_i < 0 (Kurylyk and Hayashi, 2016, Equation 24)"""
    return (1 + 0.147*S_T1*(beta * T_i/T_s)**2 + 0.535 * np.sqrt(S_T1) * beta * T_i/T_s) * (1 - 0.16*S_T1 + 0.038*S_T1**2)

def lambda_freeze_0(S_T2):
    """Freeze correction factor when T_i=0 (Kurylyk and Hayashi, 2016, Equation 25)"""
    return 1 - 0.16*S_T2 + 0.038*S_T2**2

def lambda_freeze(S_T2, T_i, T_s, beta):
    """Freeze correction factor T_i > 0 (Kurylyk and Hayashi, 2016, Equation 26)"""
    return (1 + 0.061 * (S_T2**0.88 * (- T_i/(beta * T_s))**1.65) - 0.43 * (S_T2**0.44 * (- T_i/(beta * T_s)) **0.825)) * (1 - 0.16*S_T2 + 0.038*S_T2**2)

class _GDP:
    pass

def set_null_gdp(wtm):

    nn = wtm._grid.number_of_nodes
    nl = wtm._grid.number_of_links

    wtm.gdp     = _GDP()
    wtm.gdp._q  = np.zeros(nl)
    wtm.gdp._vel = np.zeros(nl)
    wtm.gdp.recharge = np.zeros(nn)
    wtm.gdp._qsavg   = np.zeros(nn)
    wtm.gdp._hydr_grad = np.zeros(nl)
    wtm.gdp.run_with_adaptive_time_step_solver = lambda x: None


def plot_E_T(t, E, T_mean, save_name=None):

    fig, axs = plt.subplots(ncols=2, figsize=(8,4))
    axs[0].plot(t, E, label='Modeled')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Sensible heat content (J/m2)')

    axs[1].plot(t, T_mean, label='Modeled')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Mean Temperature (C)')
    fig.tight_layout()

    if save_name:
        plt.savefig(save_name, dpi=300)



#%% Set up simple grid for 1D

# grid and initial conditions
boundaries = {"top": "closed", "left": "closed", "bottom": "closed", "right": "closed"}
Nx = 3; Ny = 3; dx = 1
mg = RasterModelGrid((Ny,Nx), xy_spacing=dx, bc=boundaries)
z = mg.add_zeros('topographic__elevation', at='node')
zb = mg.add_zeros('aquifer_base__elevation', at='node')
zwt = mg.add_zeros("water_table__elevation", at="node")
tmean = mg.add_zeros("mean_unfrozen__temperature", at="node") # if unchanged, start temp is zero

zb[:] = -1e-2

params = {}
params['frozen_gradient'] = 0 # K/m, temperature gradient in the frozen soil (constant for now)
params['S0'] = 0 # W/m^2, peak solar irradiance
params['T_air'] = 10.0
params['dt'] = 10 # seconds
params['T'] = 90 * 24 * 3600
# params['beta'] = 1.0
params['surface_boundary_type'] = 'Dirichlet'

params['recharge_rate'] = 0.0 # recharge rate (constant, uniform here) m/s
params['hydraulic_conductivity'] = 1e-3 # hydraulic conductivity (constant, uniform here) m/s

wtm = WaterTrackModelThermal(mg, params)
set_null_gdp(wtm) # replace the actual gdp with a dummy version so that all lateral fluxes are zero


#%%

T = params['T']
Nt = int(T / params['dt'])

b = np.zeros(Nt)
E = np.zeros(Nt)
T_mean = np.zeros(Nt)

for i in range(Nt):
    wtm.run_step()
    b[i] = z[4] - mg.at_node['aquifer_base__elevation'][4]
    T_mean[i] = mg.at_node['mean_unfrozen__temperature'][4]
    E[i] = wtm._E[4]


# %%

t = np.arange(Nt) * params['dt']
St1 = calc_st1(wtm.rhoC_u, 1, wtm.rho_w, wtm.T_air, wtm.L, wtm.phi)
lam = lamda_thaw_0(St1)
X_pred = stefan_mod(lam, wtm.k_u, wtm.T_air, t, wtm.phi, wtm.rho_w, wtm.L)


plt.figure(figsize=(5,4))
plt.plot(t, b, label='b')
plt.plot(t, X_pred, label='Thaw 0 (K&H 2016 #23)')
# plt.plot(t, 8e-5 * t**(1/2), label='8e-5 * t**(1/2)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Active Layer Thickness (m)')


plot_E_T(t, E, T_mean)

# %%

alpha_u = wtm.k_u / (wtm.rhoC_u) # unfrozen thermal diffusivity
b0 = 1e-2 # m initial thickness 
# t0 = b0**2 / (4 * lam**2 * alpha_u)
L_v = wtm.phi * wtm.rho_w * wtm.L
t0 = b0**2 * L_v / (2.0 * wtm.k_u * wtm.T_air * lam**2)
t_shifted = t + t0
X_pred1 = stefan_mod(lam, wtm.k_u, wtm.T_air, t_shifted, wtm.phi, wtm.rho_w, wtm.L)



plt.figure(figsize=(5,4))
plt.plot(t+t0, b, label='b')
plt.plot(t+t0, X_pred1, label='Thaw 0 (K&H 2016 #23)')
# plt.plot(t, 8e-5 * t**(1/2), label='8e-5 * t**(1/2)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Active Layer Thickness (m)')


plot_E_T(t+t0, E, T_mean)


# %%
