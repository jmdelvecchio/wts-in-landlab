"""
Script to run the water track model in steady configuration
"""
#%%
import re
import glob
import shutil

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from landlab import RasterModelGrid, imshow_grid
from water_track_funcs import calc_growth_rate_1, calc_wavelenth_1
from model.water_track_model import WaterTrackModel

#%%
# grid and initial conditions
boundaries = {"top": "open", "left": "closed", "bottom": "closed", "right": "closed"}
Nx = 101; Ny = 200; dx = 5
mg = RasterModelGrid((Ny,Nx), xy_spacing=dx, bc=boundaries)
z = mg.add_zeros('topographic__elevation', at='node')
zb = mg.add_zeros('aquifer_base__elevation', at='node')
zwt = mg.add_zeros("water_table__elevation", at="node")
tmean = mg.add_zeros("mean_unfrozen__temperature", at="node")

# parabolic hillslope, uniform permeable thickness
x = mg.x_of_node
y = mg.y_of_node
a = 0.00005
b = 0.5 # permeable thickness m
z[:] = -a * y**2 + a * 2000**2
zb[:] = z - b
zwt[:] = zb + 0.1*b

params = {}
params['recharge_rate'] = 1.0e-6 # recharge rate (constant, uniform here) m/s
params['hydraulic_conductivity'] = 1e-1 # hydraulic conductivity (constant, uniform here) m/s
params['porosity'] = 0.9 # porosity (constant, uniform here) -- does not matter for steady state solution
params['S0'] = 10 # W/m^2, peak solar irradiance

params['frozen_gradient'] = 10 # K/m, Temperature gradient in the frozen soil. In most recent model, positive is increasing temp vertically. 
params['T_air'] = 1
params['dt'] = 6*3600 # seconds
params['T'] = 18 * 24 * 3600
params['courant_coefficient'] = 0.5
# params['gwdt'] = 1e3 # seconds, groundwater model timestep 
# params['tol'] = 1e-10 # tolerance for numerical solvers
# params['max_iter'] = 20 # maximum iterations for numerical solvers

params['use_melt_diffusion'] = False
params['use_steady_hydrology'] = False

## parameters generally kept constant:
params['kf'] = 2.728 # W/m/K, frozen soil
params['ku'] = 1.2682 # W/m/K, unfrozen soil
params['Tm'] = 0 # C, melting temperature
params['L'] = 334E3 # J/kg, latent heat of fusion
params['beta'] = 0.04 # insulation parameter (W/m^2 K)
params['rho_w'] = 1000 # kg/m^3
params['rho_s'] = 2600 # kg/m^3
params['C_s'] = 700 # J/kg/K, specific heat capacity of soil
params['C_w'] = 4.2e3 # J/kg/K, specific heat capacity of water at ~5C


output = {}
output["output_interval"] = 20
output["output_fields"] = [
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        ]
output["base_output_path"] = '/Users/tuv05476/Documents/Research Data/Local/water-tracks/wtm_steady_'

# get latest run ID so not to overwrite existing
matching_files = glob.glob(f"{output['base_output_path']}*.nc")
numbers = []
for file_path in matching_files:
    # Extract digits that appear right before the .nc extension
    match = re.search(r'(\d+)\.nc$', file_path)
    if match:
        numbers.append(int(match.group(1)))
output["run_id"] = max(numbers) + 1 if numbers else 0
print(f'Current run ID: {output["run_id"]}')


#%%
## Introduce random fluctuations to base elevation to seed water track formation
lam = 5 # correlation length for the random field
alpha = 0.01 # scaling factor for the random field
# fluct =  alpha * generate_correlated_random_field(Ny, Nx, lam/dx * 2, 2142025).flatten()
fluct = alpha * np.random.randn(Ny, Nx).flatten()

# calc average slope of hillslope
slope = np.arctan(np.mean(np.abs(np.gradient(z.reshape(mg.shape), dx, axis=0))))
slope_deg = np.rad2deg(slope)
print(f'Average slope of hillslope is {round(slope_deg, 2)} degrees')

rho_w = 1000
rho_s = 2600
rho_u = rho_s*(1-params['porosity']) + (rho_w * params['porosity'])
wavelength = calc_wavelenth_1(max(mg.y_of_node), params['kf'], params['frozen_gradient'], rho_w, slope_deg, params['hydraulic_conductivity'])
growth_rate = calc_growth_rate_1(max(mg.y_of_node), params['kf'], params['L'], params['frozen_gradient'], rho_w, rho_u, params['porosity'], slope_deg, params['hydraulic_conductivity'] )
print(f'Wavelength: {round(wavelength, 2)} meters')
print(f'Growth rate: {3600*24*365*growth_rate:.2e} meters/year')


# wavelength_target = round(wavelength, 2)*2.0  # meters
# kappa = 2 * np.pi / wavelength_target
amplitude = 0.01  # small perturbation, meters
# fluct = amplitude * np.sin(kappa * mg.x_of_node)

zb0 = zb.copy()

zb[:] = zb + fluct
zwt[:] = zb + 0.1 # near equilibrium thickness


# spectral analysis across the hillslope
plt.figure()
zb = zb.reshape(mg.shape)
N = zb.shape[0]
colors = plt.cm.viridis(np.linspace(0, 1, N))
for i in range(N):
    frequencies, psd = signal.welch(zb[i, 5:-5], 1/dx, nperseg=128)
    plt.loglog(1/frequencies[1:], psd[1:], color=colors[i], alpha=0.2)
plt.xlabel('Length (m)')
plt.ylabel('Power/Frequency (m^2 / 1/m)')
plt.title('Initial Power Spectral Density')
plt.show()

#%%

# copy scripts to output location
shutil.copy('../model/water_track_model.py', output['base_output_path'] + f"water_track_model_{output['run_id']}.py")
shutil.copy('./run_steady_model.py', output['base_output_path'] + f"run_steady_model_{output['run_id']}.py")

# mdl = WaterTrackModel(mg, params, output_dict=output)
mdl = WaterTrackModel(mg, params)
if params['use_steady_hydrology']:
    mdl.run_hydrology_steady()
else:
    mdl.run_hydrology_dynamic()

#%%
mdl.run_model()

# %%

mdl.make_plots()
# %%

plt.figure(figsize=(15,5))
plt.subplot(1, 3, 2)
imshow_grid(mg, mdl.melt_diffusion, cmap='plasma', colorbar_label='Melt diffusion rate (m/s)')

#%%

plt.figure(figsize=(15,5))
plt.subplot(1, 3, 2)
imshow_grid(mg, mdl._T_mean, cmap='plasma', colorbar_label='Temperature (C)')

# %%
plt.figure(figsize=(15,5))
plt.subplot(1, 3, 2)
imshow_grid(mg, mdl._dzb_dt, cmap='plasma', colorbar_label='Base melt rate (m/s)')

# %%

# final timestep cross sections of aquifer base 
plt.figure()
zb = mdl._zb.reshape(mg.shape)
N = zb.shape[0]
colors = plt.cm.viridis(np.linspace(0, 1, 10))
for i in range(8):
    row = N//10 * (i + 1)
    plt.plot(zb[row, :] - np.mean(zb[row, :]) + i * 0.1, color=colors[i])
plt.xlabel('Distance (m)')
plt.ylabel('Elevation (m)')
plt.show()
# %%


# spectral analysis across the hillslope
plt.figure()
zb = mdl._zb.reshape(mg.shape)
colors = plt.cm.viridis(np.linspace(0, 1, N))
for i in range(N):
    frequencies, psd = signal.welch(zb[i, 5:-5], 1/dx, nperseg=128)
    plt.loglog(1/frequencies[1:], psd[1:], color=colors[i], alpha=0.2)
plt.xlabel('Length (m)')
plt.ylabel('Power/Frequency (m^2 / 1/m)')
plt.show()

#%%

# final timestep map view
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
imshow_grid(mg, 'mean_unfrozen__temperature', cmap='Reds', colorbar_label='Mean Unfrozen Temperature (°C)')

plt.subplot(1, 3, 2)
imshow_grid(mg, mdl._dzb_dt, cmap='inferno', colorbar_label='Interface Velocity (m/s)')

plt.subplot(1, 3, 3)
imshow_grid(mg, mdl._z - mdl._zb, cmap='plasma', colorbar_label='Active Layer Thickness (m)')
plt.tight_layout()
plt.show()
# %%
