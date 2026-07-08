"""Demonstration of heat transport in thermal model"""


#%%
import numpy as np
import matplotlib.pyplot as plt

from landlab import RasterModelGrid, imshow_grid
from model.water_track_model import WaterTrackModel

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

a = 0.05
b = 5 # permeable thickness m

z[:] = a * mg.y_of_node
zb[:] = z - b
zwt[:] = z - 0.5 * b
plt.figure()
imshow_grid(mg, "topographic__elevation", colorbar_label="Elevation (m)")
plt.figure()
imshow_grid(mg, zwt-zb, colorbar_label="Aquifer thickness (m)")

params = {}
params['frozen_gradient'] = -20 # -20 # K/m, temperature gradient in the frozen soil (constant for now)
params['T_air'] = 0.0
params['dt'] = 10*6*3600 # seconds
params['dtgw'] = 100*6*3600 # seconds
params['T'] = 180 * 24 * 3600
params['courant_coefficient'] = 0.1

params['recharge_rate'] = 5.0e-8 # recharge rate (constant, uniform here) m/s
params['hydraulic_conductivity'] = 1e-4 # hydraulic conductivity (constant, uniform here) m/s
params['verbose'] = True
params['max_iter'] = 5000

wtm = WaterTrackModel(mg, params)

# %%

wtm.run_hydrology_steady()

#%%

plt.figure()
imshow_grid(mg, zwt-zb, colorbar_label="Aquifer thickness (m)")

plt.figure()
imshow_grid(mg, z-zwt, colorbar_label="Depth to wt (m)")

plt.figure()
imshow_grid(mg, "surface_water__specific_discharge", colorbar_label="Surface water specific discharge (m/s)")
# %%

imshow_grid(mg, "mean_unfrozen__temperature", colorbar_label="Mean Unfrozen Temperature (K)", cmap='coolwarm')
plt.figure()
# %%

for i in range(100):
    wtm.run_heat_transport()


imshow_grid(mg, "mean_unfrozen__temperature", colorbar_label="Mean Unfrozen Temperature (K)", cmap='coolwarm')
plt.figure()
# %%
