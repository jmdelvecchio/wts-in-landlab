""" Scripts to analyze the results of a wavelength sweep. """

#%%

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from DupuitLEM.io import load_grid_from_dataset, load_fields_from_dataset

#%%

directory = '/Users/tuv05476/Documents/Research Data/Local/water-tracks/wavelength_sweep/'
# base_output_path = 'basic_nomeltdiff_wl1.00'
base_output_path = 'thermal_meltdiff_wl1.00'
i = 0

ds = xr.open_dataset('%s/%s/output_%d.nc'%(directory, base_output_path, i))
grid = load_grid_from_dataset(ds)
load_fields_from_dataset(ds, grid)

#%%

# final timestep cross sections of aquifer base 
plt.figure()
zb = grid.at_node['aquifer_base__elevation']
zb = zb.reshape(grid.shape)
N = zb.shape[0]
colors = plt.cm.viridis(np.linspace(0, 1, 10))
for i in range(8):
    row = N//10 * (i + 1)
    plt.plot(zb[row, :] - np.mean(zb[row, :]) + i * 0.1, color=colors[i])
plt.xlabel('Distance (m)')
plt.ylabel('Elevation (m)')
plt.show()
# %%
