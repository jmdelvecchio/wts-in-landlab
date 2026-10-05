"""
Script to run the water track model in steady configuration
"""
#%%
import re
import glob
import shutil
import warnings

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from landlab import RasterModelGrid, imshow_grid
from landlab.grid.raster_mappers import map_link_vector_components_to_node_raster
from water_track_funcs import calc_growth_rate_1, calc_wavelenth_1, notional_equilibrium_temperature
from model.water_track_model import WaterTrackModelThermal

#%%
# grid and initial conditions
boundaries = {"top": "open", "left": "closed", "bottom": "closed", "right": "closed"}
Nx = 100; Ny = 200; dx = 10
mg = RasterModelGrid((Ny,Nx), xy_spacing=dx, bc=boundaries)
z = mg.add_zeros('topographic__elevation', at='node')
zb = mg.add_zeros('aquifer_base__elevation', at='node')
zwt = mg.add_zeros("water_table__elevation", at="node")
tmean = mg.add_zeros("mean_unfrozen__temperature", at="node")

# parabolic hillslope, uniform permeable thickness
x = mg.x_of_node
y = mg.y_of_node
a = 0.0001
b = 0.5 # permeable thickness m
z[:] = -a * y**2 + a * 2000**2
zb[:] = z - b
zwt[:] = zb + 0.1*b

# hillslope cross section
plt.figure()
plt.plot(y, z)
plt.xlabel('Y coordinate (m)')
plt.ylabel('Elevation (m)')
plt.show()

params = {}
params['recharge_rate'] = 1.0e-6 # recharge rate (constant, uniform here) m/s
params['hydraulic_conductivity'] = 1e-1 # hydraulic conductivity (constant, uniform here) m/s
params['porosity'] = 0.9 # porosity (constant, uniform here) -- does not matter for steady state solution
params['S0'] = 30.0 # 10 # W/m^2, peak solar irradiance

params['frozen_gradient'] = 10.0 #10 # K/m, Temperature gradient in the frozen soil. In most recent model, positive is increasing temp vertically. 
params['T_air'] = 1
params['dt'] = 6*3600 # seconds
params['T'] = 180 * 24 * 3600
params['courant_coefficient'] = 0.5
# params['gwdt'] = 1e3 # seconds, groundwater model timestep 
# params['tol'] = 1e-10 # tolerance for numerical solvers
# params['max_iter'] = 20 # maximum iterations for numerical solvers

params['use_melt_diffusion'] = False
params['use_steady_hydrology'] = False
params['use_fourier_frozen_gradient'] = True

## parameters generally kept constant:
params['kf'] = 2.728 # W/m/K, frozen soil
params['ku'] = 1.2682 # W/m/K, unfrozen soil
params['Tm'] = 0 # C, melting temperature
params['L'] = 334E3 # J/kg, latent heat of fusion
params['beta'] = 0.4 # 0.04 # insulation parameter (W/m^2 K)
params['rho_w'] = 1000 # kg/m^3
params['rho_s'] = 2600 # kg/m^3
params['C_s'] = 700 # J/kg/K, specific heat capacity of soil
params['C_w'] = 4.2e3 # J/kg/K, specific heat capacity of water at ~5C


T_eq = notional_equilibrium_temperature(
    S0=params['S0'], beta=params['beta'], T_air=params['T_air'], kf=params['kf'], frozen_gradient=params['frozen_gradient']
)
print(f"Notional equilibrium T_mean: {T_eq:.1f} C")


output = {}
output["output_interval"] = 20
output["output_fields"] = [
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        ]
output["base_output_path"] = '/Users/tuv05476/Documents/Research Data/Local/water-tracks/wtm_thermal_'

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
alpha = 0.01 # scaling factor for the random field #0.002
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


# wavelength_target = round(wavelength, 2) # meters
# kappa = 2 * np.pi / wavelength_target
# amplitude = 0.01  # small perturbation, meters
# fluct = amplitude * np.sin(kappa * mg.x_of_node)

zb0 = zb.copy()

zb[:] = zb + fluct
zwt[:] = zb + 0.5 # near equilibrium thickness


# spectral analysis across the hillslope
plt.figure()
zb1 = zb.reshape(mg.shape)
N = zb1.shape[0]
colors = plt.cm.viridis(np.linspace(0, 1, N))
for i in range(N):
    frequencies, psd = signal.welch(zb1[i, 5:-5], 1/dx, nperseg=128)
    plt.loglog(1/frequencies[1:], psd[1:], color=colors[i], alpha=0.2)
plt.xlabel('Length (m)')
plt.ylabel('Power/Frequency (m^2 / 1/m)')
plt.title('Initial Power Spectral Density')
plt.show()


#%%

plt.figure()
imshow_grid(mg, z-zb, colorbar_label="Topographic Elevation (m)")
plt.show()

#%%

# copy scripts to output location
shutil.copy('../model/water_track_model.py', output['base_output_path'] + f"water_track_model_{output['run_id']}.py")
shutil.copy('./run_wtthermal_model.py', output['base_output_path'] + f"run_wtthermal_model_{output['run_id']}.py")

mdl = WaterTrackModelThermal(mg, params, output_dict=output)
# mdl = WaterTrackModelThermal(mg, params)
if params['use_steady_hydrology']:
    mdl.run_hydrology_steady()
else:
    mdl.run_hydrology_dynamic()

#%%

## Check the domain-mean energy balance implied by the hydrology solution before
## running the full model: is this parameter combination even in a growth (thawing)
## regime, or will the active layer refreeze?
flux_frozen = mdl.k_f * mdl.frozen_gradient
Qdiss_mean = mdl._Qdiss[mg.core_nodes].mean()
BC_top_mean = (mdl.S0 + mdl.beta * (mdl.T_air - mdl._T_mean))[mg.core_nodes].mean()
net_flux = Qdiss_mean + BC_top_mean - flux_frozen

print(f'Flux frozen (conductive loss to permafrost): {flux_frozen:.4f} W/m^2')
print(f'Mean dissipative heating (Qdiss): {Qdiss_mean:.4f} W/m^2')
print(f'Mean top boundary flux (BC_top): {BC_top_mean:.4f} W/m^2')
print(f'Net flux (Qdiss + BC_top - flux_frozen): {net_flux:.4f} W/m^2')

if net_flux < 0:
    warnings.warn(
        f'Net flux is negative ({net_flux:.4f} W/m^2): mean dissipative + top-boundary heating '
        'cannot offset conductive loss to the frozen layer. This parameter combination is in a '
        'freezing regime.'
    )

#%%
mdl.run_model()

# %%

mdl.make_plots()
# mdl.make_plots(save_path=output['base_output_path'] + f"wtbasic_model_{output['run_id']}")

# %%

plt.figure(figsize=(15,5))
plt.subplot(1, 3, 2)
imshow_grid(mg, mdl._z - mdl._zb, cmap='plasma', colorbar_label='Active layer thickness (m)', vmin=0.5, vmax=0.7)

#%%

plt.figure(figsize=(15,5))
plt.subplot(1, 3, 2)
imshow_grid(mg, mdl._T_mean, cmap='plasma', colorbar_label='Temperature (C)') #, vmin=5.5, vmax=6.5)

# %%
plt.figure(figsize=(15,5))
plt.subplot(1, 3, 2)
imshow_grid(mg, mdl._dzb_dt, cmap='plasma', colorbar_label='Base melt rate (m/s)') #, vmin=0, vmax=5e-9)

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
    frequencies, psd = signal.welch(zb[i, 5:-5], 1/dx, nperseg=70)
    plt.loglog(1/frequencies[1:], psd[1:], color=colors[i], alpha=0.2)
plt.xlabel('Length (m)')
plt.ylabel('Power/Frequency (m^2 / 1/m)')
plt.ylim(1e-10, 1e-2)
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

plt.subplot(1, 3, 2)
imshow_grid(mg, np.log10(abs(mdl._dzb_dt - np.mean(mdl._dzb_dt))), cmap='inferno', colorbar_label='Interface Velocity (m/s)')

# %%

vels = abs(mdl.gdp._vel)
plt.figure(figsize=(5, 3))
plt.hist(np.log10(vels[vels > 0]), bins=50, density=True)
plt.xlabel('log10(Darcy velocity) (m/s)')
plt.ylabel('Probability density')


v_x, v_y = map_link_vector_components_to_node_raster(mdl._grid, mdl.gdp._vel) # get mean value in x and y directions at node



plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
imshow_grid(mg, np.log10(abs(v_x[mg.core_nodes])), cmap='Blues', colorbar_label='Darcy velocity x-component (m/s)', vmin=-8, vmax=-1.5)

plt.subplot(1, 2, 2)
imshow_grid(mg, np.log10(abs(v_y[mg.core_nodes])), cmap='Blues', colorbar_label='Darcy velocity y-component (m/s)', vmin=-8, vmax=-1.5)


# %%
