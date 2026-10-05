"""Demonstration of heat transport in thermal model"""


#%%
import numpy as np
import matplotlib.pyplot as plt

from landlab import RasterModelGrid, imshow_grid
from model.water_track_model import WaterTrackModelThermal

#%%

# grid and initial conditions
boundaries = {"top": "closed", "left": "closed", "bottom": "open", "right": "closed"}
Nx = 101; Ny = 200; dx = 5
mg = RasterModelGrid((Ny,Nx), xy_spacing=dx, bc=boundaries)
z = mg.add_zeros('topographic__elevation', at='node')
zb = mg.add_zeros('aquifer_base__elevation', at='node')
zwt = mg.add_zeros("water_table__elevation", at="node")
tmean = mg.add_zeros("mean_unfrozen__temperature", at="node")
tmean[mg.y_of_node > 900] = 2
tmean[np.logical_and(mg.x_of_node == 250, mg.y_of_node > 400)] = 2

a = 0.05
b = 10 # permeable thickness m

z[:] = a * mg.y_of_node #+ 0.2 * np.random.randn(len(z))
zb[:] = z - b
zwt[:] = z - 0.5 * b
plt.figure()
imshow_grid(mg, "topographic__elevation", colorbar_label="Elevation (m)")
plt.figure()
imshow_grid(mg, zwt-zb, colorbar_label="Aquifer thickness (m)")

params = {}
params['frozen_gradient'] = 0 # -20 # K/m, temperature gradient in the frozen soil (constant for now)
params['S0'] = 0 # W/m^2, peak solar irradiance
params['T_air'] = 1.0
params['dt'] = 1*3600 # seconds
params['dtgw'] = 100*6*3600 # seconds
params['T'] = 180 * 24 * 3600
params['courant_coefficient'] = 0.1
params['regularization_f'] = 0.001
# params['ku'] = 0.00001

params['recharge_rate'] = 5.0e-7 # recharge rate (constant, uniform here) m/s
params['hydraulic_conductivity'] = 1e-3 # hydraulic conductivity (constant, uniform here) m/s
params['verbose'] = True
params['max_iter'] = 5000

wtm = WaterTrackModelThermal(mg, params)



# %%

wtm.run_hydrology_steady()

#%%

plt.figure()
imshow_grid(mg, zwt-zb, colorbar_label="Aquifer thickness (m)")
plt.show()

plt.figure()
imshow_grid(mg, z-zwt, colorbar_label="Depth to wt (m)")
plt.show()

plt.figure()
imshow_grid(mg, "surface_water__specific_discharge", colorbar_label="Surface water specific discharge (m/s)")
plt.show()


# %%
plt.figure()
imshow_grid(mg, "mean_unfrozen__temperature", colorbar_label="Mean Unfrozen Temperature (K)", cmap='coolwarm')

plt.figure()
imshow_grid(mg, wtm._Qdiss, colorbar_label="Thermal Dissipation (W/m2)", cmap='plasma')
# %%

for i in range(500):
    wtm.run_heat_transport()


imshow_grid(mg, "mean_unfrozen__temperature", colorbar_label="Mean Unfrozen Temperature (K)", cmap='coolwarm')
plt.figure()
# %%


# final timestep map view
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
imshow_grid(mg, wtm._T_mean, cmap='Reds', colorbar_label='Mean Unfrozen Temperature (°C)')

plt.subplot(1, 3, 2)
imshow_grid(mg, wtm._dzb_dt, cmap='inferno', colorbar_label='Interface Velocity (m/s)')

plt.subplot(1, 3, 3)
imshow_grid(mg, wtm._Qdiss, cmap='inferno', colorbar_label='Dissipation (W/m^2)', vmax=0.5)

plt.tight_layout()
plt.show()
# %%
