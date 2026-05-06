"""
Script to run the water track model in steady configuration
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from landlab import RasterModelGrid, imshow_grid
from water_track_funcs import generate_correlated_random_field, calc_one_wavelength
from water_track_model import WaterTrackModel

#%%
# grid and initial conditions
boundaries = {"top": "open", "left": "closed", "bottom": "closed", "right": "closed"}
Nx = 101; Ny = 200; dx = 10
mg = RasterModelGrid((Ny,Nx), xy_spacing=dx, bc=boundaries)
z = mg.add_zeros('topographic__elevation', at='node')
zb = mg.add_zeros('aquifer_base__elevation', at='node')
zwt = mg.add_zeros("water_table__elevation", at="node")

# parabolic hillslope, uniform permeable thickness
x = mg.x_of_node
y = mg.y_of_node
a = 0.00005
b = 10 # permeable thickness m
z[:] = -a * y**2 + a * max(y)**2
zb[:] = z - b

params = {}
params['recharge_rate'] = 1.0e-6 # recharge rate (constant, uniform here) m/s
params['hydraulic_conductivity'] = 1e-1 # hydraulic conductivity (constant, uniform here) m/s
params['porosity'] = 0.9 # porosity (constant, uniform here) -- does not matter for steady state solution
params['S0'] = 0.04 # W/m^2, peak solar irradiance
params['kf'] = 2.0 # W/m/K, frozen soil
params['ku'] = 0.5 # W/m/K, unfrozen soil
params['Tm'] = 0 # C, melting temperature
params['L'] = 334E3 # J/kg, latent heat of fusion
params['frozen_gradient'] = -20 # -20 # K/m, temperature gradient in the frozen soil (constant for now)
params['T_surface'] = 0 #-5 # C, surface temperature (constant for now)


## Introduce random fluctuations to base elevation to seed water track formation
lam = 5 # correlation length for the random field
alpha = 0.05 # scaling factor for the random field
# fluct =  alpha * generate_correlated_random_field(Ny, Nx, lam/dx * 2, 2142025).flatten()
fluct = alpha * np.random.randn(Ny, Nx).flatten()

# calc average slope of hillslope
slope = np.arctan(np.mean(np.abs(np.gradient(z.reshape(mg.shape), dx, axis=0))))
slope_deg = np.rad2deg(slope)
print(f'Average slope of hillslope is {round(slope_deg, 2)} degrees')

# calc theoretical wavelength and growth rate for these parameters
wavelength, growth_rate = calc_one_wavelength(
    slope_deg,
    params['frozen_gradient'],
    params['ku'] * (params['T_surface'] - params['Tm']) / b, # use the conductive flux to
    x_t = max(mg.y_of_node), # use the length of the hillslope as the characteristic length scale
    porosity=params['porosity'],
    beta=params['S0'], # not sure about this
    flow_speed = params['hydraulic_conductivity']*slope # m/s, just a guess for now
    )
print(f'Wavelength: {round(wavelength, 2)} meters')
print(f'Growth rate: {3600*24*365*growth_rate:.2e} meters/year')

wavelength_target = round(wavelength, 2)*2.0  # meters
kappa = 2 * np.pi / wavelength_target
amplitude = 0.01  # small perturbation, meters
# fluct = amplitude * np.sin(kappa * mg.x_of_node)

zb0 = zb.copy()

zb[:] = zb + fluct
zwt[:] = zb + 0.1 # near equilibrium thickness

#%%

mdl = WaterTrackModel(mg, params)
mdl.run_hydrology()
mdl.run_model()

# %%

mdl.make_plots()
# %%

plt.figure()
plt.subplot(1, 3, 2)
imshow_grid(mg, '', cmap='plasma', colorbar_label='Dissipation (W/m^2)')

# %%
