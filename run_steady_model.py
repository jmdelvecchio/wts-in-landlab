"""
Script to run the water track model in steady configuration
"""
#%%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from landlab import RasterModelGrid, imshow_grid
from water_track_funcs import calc_growth_rate_1, calc_wavelenth_1
from water_track_model import WaterTrackModel

#%%
# grid and initial conditions
boundaries = {"top": "open", "left": "closed", "bottom": "closed", "right": "closed"}
Nx = 201; Ny = 400; dx = 5
mg = RasterModelGrid((Ny,Nx), xy_spacing=dx, bc=boundaries)
z = mg.add_zeros('topographic__elevation', at='node')
zb = mg.add_zeros('aquifer_base__elevation', at='node')
zwt = mg.add_zeros("water_table__elevation", at="node")

# parabolic hillslope, uniform permeable thickness
x = mg.x_of_node
y = mg.y_of_node
a = 0.00005
b = 5 # permeable thickness m
z[:] = -a * y**2 + a * 2000**2
zb[:] = z - b

params = {}
params['recharge_rate'] = 1.0e-6 # recharge rate (constant, uniform here) m/s
params['hydraulic_conductivity'] = 1e-1 # hydraulic conductivity (constant, uniform here) m/s
params['porosity'] = 0.9 # porosity (constant, uniform here) -- does not matter for steady state solution
params['S0'] = 100 # W/m^2, peak solar irradiance
params['kf'] = 2.728 # W/m/K, frozen soil
params['ku'] = 1.2682 # W/m/K, unfrozen soil
params['Tm'] = 0 # C, melting temperature
params['L'] = 334E3 # J/kg, latent heat of fusion
params['frozen_gradient'] = -20 # -20 # K/m, temperature gradient in the frozen soil (constant for now)
params['T_surface'] = 1 #0 #-5 # C, surface temperature (constant for now)
params['T_air'] = 5
params['dt'] = 6*3600 # seconds
params['T'] = 365 * 24 * 3600
params['steady'] = False # whether to run the groundwater model to steady state or not

## Introduce random fluctuations to base elevation to seed water track formation
lam = 5 # correlation length for the random field
alpha = 0.01 # scaling factor for the random field
# fluct =  alpha * generate_correlated_random_field(Ny, Nx, lam/dx * 2, 2142025).flatten()
fluct = alpha * np.random.randn(Ny, Nx).flatten()

# calc average slope of hillslope
slope = np.arctan(np.mean(np.abs(np.gradient(z.reshape(mg.shape), dx, axis=0))))
slope_deg = np.rad2deg(slope)
print(f'Average slope of hillslope is {round(slope_deg, 2)} degrees')

# calc theoretical wavelength and growth rate for these parameters
# wavelength, growth_rate = calc_one_wavelength(
#     slope_deg,
#     params['frozen_gradient'],
#     params['ku'] * (params['T_surface'] - params['Tm']) / b, # use the conductive flux to
#     x_t = max(mg.y_of_node), # use the length of the hillslope as the characteristic length scale
#     porosity=params['porosity'],
#     beta=params['S0'], # not sure about this
#     flow_speed = params['hydraulic_conductivity']*slope # m/s, just a guess for now
#     )
# print(f'Wavelength: {round(wavelength, 2)} meters')
# print(f'Growth rate: {3600*24*365*growth_rate:.2e} meters/year')

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

mdl = WaterTrackModel(mg, params)
if params['steady']:
    mdl.run_hydrology_steady()
else:
    mdl.run_hydrology_dynamic()
mdl.run_model()

# %%

mdl.make_plots()
# %%

plt.figure(figsize=(15,5))
plt.subplot(1, 3, 2)
imshow_grid(mg, mdl.melt_diffusion, cmap='plasma', colorbar_label='Melt diffusion rate (m/s)')

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





# %%

# 1. Generate a dummy signal (Sampling rate: 1000 Hz, Duration: 2 seconds)
fs = 1000.0
time = np.arange(0, 2, 1/fs)
# Signal contains 50 Hz and 120 Hz sinusoids, plus random noise
signal_data = np.sin(2 * np.pi * 50 * time) + np.sin(2 * np.pi * 120 * time) + np.random.normal(scale=2, size=len(time))

# 2. Calculate Power Spectral Density using Welch's method
frequencies, psd = signal.welch(signal_data, fs, nperseg=1024)

# 3. Plot the results
plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, psd) # Logarithmic scale for better dynamic range
plt.title('Power Spectral Density (PSD)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V^2 / Hz)')
plt.grid(True)

# %%
