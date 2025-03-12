"""
Set up and run the GroundwaterDupuitPercolator on a test hillslope
Calculate the factor Q, and evolve the subsurface assuming a linear relationship
between Q and rate of active layer deepening.

"""

#%%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.ndimage

from landlab import RasterModelGrid, imshow_grid
from landlab.components import GroundwaterDupuitPercolator, FlowAccumulator
from landlab.grid.raster_mappers import map_link_vector_components_to_node_raster

def generate_correlated_random_field(Nx, Ny, l):
    """
    Generate a 2D array of random values with spatial correlation.

    Parameters:
    Nx (int): Number of grid points in the x dimension.
    Ny (int): Number of grid points in the y dimension.
    l (float): Correlation length scale.

    Returns:
    np.ndarray: 2D array of spatially correlated random values.
    """
    # Generate a 2D array of random values
    random_field = np.random.randn(Nx, Ny)
    
    # Apply a Gaussian filter to introduce spatial correlation
    correlated_field = scipy.ndimage.gaussian_filter(random_field, sigma=l)
    
    return correlated_field


#%% Create the grid, add basic fields

boundaries = {"top": "open", "left": "closed", "bottom": "closed", "right": "closed"}

Nx = 101; Ny = 200; dx = 5
mg = RasterModelGrid((Ny,Nx), xy_spacing=dx, bc=boundaries)
z = mg.add_zeros('topographic__elevation', at='node')
zb = mg.add_zeros('aquifer_base__elevation', at='node')
zwt = mg.add_zeros("water_table__elevation", at="node")

# some parameters
b = 0.5 # permeable thickness m
r = 1.0e-7 # recharge rate (constant, uniform here) m/s
ksat = 1e-4 # hydraulic conductivity (constant, uniform here) m/s
n = 0.1 # porosity (constant, uniform here) -- does not matter for steady state solution
routing_method = 'MFD' # could also be 'D8' or 'Steepest'

# some example parabolic hillslopes, just made up
x = mg.x_of_node
y = mg.y_of_node
a = 0.0002
z[:] = -a * y**2 + a * max(y)**2

zinthat = 0.2
lam = 5
kappa = 2*np.pi/lam
# zb[:] = z - b + zinthat * np.cos(kappa * x)
# zb[:] = z - b # set constant permeable thickness b
zb[:] = z - b + 0.01 * generate_correlated_random_field(Ny, Nx, lam/dx).flatten()
zb0 = zb.copy()

zwt[:] = z # start water table at the surface


fig, axes = plt.subplots(1, 3, figsize=(12, 5))
plt.sca(axes[0])
imshow_grid(mg, z, colorbar_label='z')
axes[0].set_title('Topographic Elevation')

plt.sca(axes[1])
imshow_grid(mg, zb, colorbar_label='zb')
axes[1].set_title('Active Zone Base Elevation')

plt.sca(axes[2])
imshow_grid(mg, z-zb, colorbar_label='zc', cmap='viridis')
axes[2].set_title('Active Zone Thickness')

plt.tight_layout()
plt.show()


# 3D plot of the hillslope
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# X = x.reshape(mg.shape)
# Y = y.reshape(mg.shape)
# Z = z.reshape(mg.shape)
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap='pink',
#                        linewidth=0, antialiased=False)

#%% Initialize components

# initialize the groundwater model
gdp = GroundwaterDupuitPercolator(
    mg,
    recharge_rate=r,
    hydraulic_conductivity=ksat,
    porosity=n,
)

# surface water routing (not actually used for subsurface evolution at the moment)
fa = FlowAccumulator(
    mg,
    surface="topographic__elevation",
    flow_director=routing_method,
    runoff_rate="surface_water__specific_discharge",
)

# %% run groundwater model to steady state

diff = 1
tol = 1e-10
while diff > tol:
    zwt0 = zwt.copy()
    gdp.run_with_adaptive_time_step_solver(1e5)
    diff = np.max(zwt0-zwt)
    print(diff)

gwf = gdp.calc_gw_flux_at_node() # map groundwater flux to node (easier to plot)
fa.run_one_step() # calculate flow directions and route surface water (if any is generated)

# %% Some figures

# groundwater flux, mapped from links to nodes
plt.figure()
imshow_grid(mg, gwf, cmap='plasma')
plt.title('Groundwater Flux at Node')

# local produced runoff
plt.figure()
imshow_grid(mg, 'surface_water__specific_discharge', cmap='viridis')
plt.title('Local Runoff')

# routed runoff
plt.figure()
imshow_grid(mg, 'surface_water__discharge', cmap='viridis')
plt.title('Surface Water Discharge')

# saturated thickness
plt.figure()
imshow_grid(mg, (zwt-zb)/(z-zb), cmap='Blues')
plt.title('Relative Saturated Thickness')

# %% Test Lazy Evolution

T = 365*24*3600
dt = 3600
N = T//dt

thaw_rate_background = 1e-7
Q_coeff = 1e-3

slp = np.max(mg.at_node['topographic__steepest_slope'], axis=1) # we are not evolving topography, so the topographic slope stays constant
for i in tqdm(range(N)):

    # run groundwater model to get steady state solution
    diff = 1
    tol = 1e-10
    while diff > tol:
        zwt0 = zwt.copy()
        gdp.run_with_adaptive_time_step_solver(1e5)
        diff = np.max(zwt0-zwt)
        # print(diff)
    
    # calculate Q with the actual darcy velocity *  hydraulic gradient
    vel_x, vel_y = map_link_vector_components_to_node_raster(mg, gdp._vel) # get mean value in x and y directions at node
    hydgr_x, hydgr_y = map_link_vector_components_to_node_raster(mg, gdp._hydr_grad)
    Q_node = Q_coeff * np.abs(vel_x * hydgr_x + vel_y * hydgr_y) # absolute value of the dot product, just like in the model description

    ##  lazy way using a gdp function and topographic slope
    # gwf = gdp.calc_gw_flux_at_node() # the total groundwater flux out of a node
    # Q_node = Q_coeff * (gwf / (z-zb)) * slp  # a simple but not quite accurate way to calculate Q_node

    # evolve based on some simple criteria for z
    zb -= (Q_node + thaw_rate_background) * dt


# %%

f = zb - zb0
plt.figure()
imshow_grid(mg, f, colorbar_label='zb-zb0', cmap='viridis')
axes[2].set_title('Active Zone Change')

fg = f.reshape(mg.shape)

plt.figure()
for i in range(0,200,20):
    plt.plot(fg[i,1:-1])


# %%


from landlab.plot.graph import plot_graph

grid = RasterModelGrid((4, 5), xy_spacing=(3, 4))
plot_graph(grid, at="node")
#%%
plot_graph(grid, at="link")# %%

# %%
