"""
Set up and run the GroundwaterDupuitPercolator on a test hillslope
Calculate the internal heating factor Q, and evolve the subsurface assuming a linear relationship
between Q and rate of active layer deepening.

This is a steady forcing model, melt rate only varies with water flow velocity.

"""

#%%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.ndimage
from scipy.signal import welch

from landlab import RasterModelGrid, imshow_grid
from landlab.components import GroundwaterDupuitPercolator
from landlab.grid.raster_mappers import map_link_vector_components_to_node_raster

def generate_correlated_random_field(Nx, Ny, l, seed):
    """
    Generate a 2D array of random values with spatial correlation.

    Parameters:
    Nx (int): Number of grid points in the x dimension.
    Ny (int): Number of grid points in the y dimension.
    l (float): Correlation length scale.

    Returns:
    np.ndarray: 2D array of spatially correlated random values.
    """

    np.random.seed(seed)
    # Generate a 2D array of random values
    random_field = np.random.randn(Nx, Ny)
    
    # Apply a Gaussian filter to introduce spatial correlation
    correlated_field = scipy.ndimage.gaussian_filter(random_field, sigma=l)
    
    return correlated_field

def plot_average_psd(arr, fs=1.0):
    """
    Plot the average power spectral density (PSD) of the rows of the array.

    Parameters:
    arr (numpy.ndarray): Input array with dimensions (ny, nx).
    fs (float): Sampling frequency. Default is 1.0.
    """

    psd_list = []

    # Compute PSD for each row
    for row in arr:
        freqs, psd = welch(row, fs=fs)
        psd_list.append(psd)

    # Average the PSDs
    avg_psd = np.mean(psd_list, axis=0)

    # Plot the averaged PSD
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, avg_psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (PSD)')
    plt.title('Average Power Spectral Density of Rows')
    plt.grid(True)
    plt.show()

def const_melt_rate(
                S0=350,
                L=334E3,
                n=0.9,
                Ks=2.73, 
                dTdz=-20.0, 
                Q=0,
                rho_ice=917):
    """
    Computes the melt rate at the permafrost table.

    Parameters:
    S0 : peak solar irradiance (W/m^2) -- but may want to set this lower to account for insulation
    L : latent heat of fusion (J/kg)
    n : porosity of the soil (dimensionless)
    Ks : thermal conductivity of frozen soil (W/m*K)
    dTdz : temperature gradient in the frozen soil (K/m)
    Q : internal heat dissipation (W/m^2)
    rho_ice : density of ice (kg/m³)
    """

    # Compute total energy flux
    energy_flux = S0 + Ks * dTdz + Q

    # Convert energy flux to melt rate in meters per second
    # PLus or minus if you care about this "melt factor" fudge
    # melt_rate_per_second = (energy_flux / (L) * melt_factor
    melt_rate_per_second = (energy_flux / (L*n*rho_ice))#* melt_factor

    return melt_rate_per_second

#%% Visualize melt rate

S0 = 75 # W/m^2, peak solar irradiance
rho_w = 1000 # kg/m^3
g = 9.81 # m/s^2
u = np.geomspace(1e-5, 1e-1, 10) # m/s, velocity of water flow
hydr = np.sin(4.8 * np.pi/180) # use Warburton and others' value for slope gradient, 4.8 degrees

Q = rho_w * g * u * hydr # W/m^2 internal heat dissipation

plt.figure()
plt.plot(u, const_melt_rate(S0=S0, Q=Q)*3600*24*90, 'o-')
plt.xlabel('Water Flow Velocity (m/s)')
plt.ylabel('90 Day Melt Rate (m)')
plt.title('Melt Rate vs Water Flow Velocity')
# plt.xscale('log')
# plt.yscale('log')
plt.grid(True)

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
ksat = 1e-2 # hydraulic conductivity (constant, uniform here) m/s
n = 0.9 # porosity (constant, uniform here) -- does not matter for steady state solution
S0 = 75 # W/m^2, peak solar irradiance
rho_w = 1000 # kg/m^3
g = 9.81 # m/s^2

# some example parabolic hillslope, just made up
x = mg.x_of_node
y = mg.y_of_node
a = 0.0002
z[:] = -a * y**2 + a * max(y)**2

# generate a random field to perturb the base elevation
lam = 5 # correlation length for the random field
alpha = 0.1 # scaling factor for the random field
zb[:] = z - b + alpha * generate_correlated_random_field(Ny, Nx, lam/dx * 2, 2142025).flatten()
zb0 = zb.copy()
zwt[:] = zb + b # start water table at the surface

# overview plots
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
    regularization_f=0.1,
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

# calculate internal heating factor Q
Q_coeff = rho_w * g # convert from head gradient to pressure gradient 
vel_x, vel_y = map_link_vector_components_to_node_raster(mg, gdp._vel) # get mean value in x and y directions at node
hydgr_x, hydgr_y = map_link_vector_components_to_node_raster(mg, gdp._hydr_grad)
Q_node = Q_coeff * np.abs(vel_x * hydgr_x + vel_y * hydgr_y) # absolute value of the dot product, as in the model description


# %% Some figures

# groundwater flux, mapped from links to nodes
plt.figure()
imshow_grid(mg, gwf, cmap='plasma')
plt.title('Groundwater Flux at Node')

# local produced runoff
plt.figure()
imshow_grid(mg, 'surface_water__specific_discharge', cmap='viridis')
plt.title('Local Runoff')

# saturated thickness
plt.figure()
imshow_grid(mg, (zwt-zb)/(z-zb), cmap='Blues')
plt.title('Relative Saturated Thickness')

# These plots make it clear why Q_node remains so uniform: the flux is dominated by
# the downslope direction, which is not really affected by perturbations. Cross slope
# gradients are much smaller, don't really appear when added together.

# x velocity
plt.figure()
imshow_grid(mg, vel_x, cmap='RdBu')
plt.title('Vel_x')

# y velocity
plt.figure()
imshow_grid(mg, vel_y, cmap='Blues')
plt.title('Vel_y')

# Q (heat source) rate
plt.figure()
imshow_grid(mg, Q_node, cmap='plasma')
plt.title('Q_node')

# plot_average_psd(Q_node.reshape(mg.shape), fs=1.0)
# plot_average_psd(zb0.reshape(mg.shape), fs=1.0)
# q = mg.at_node['surface_water__specific_discharge']
# plot_average_psd(q.reshape(mg.shape), fs=1.0)


# %% Test Simple Hillslope Evolution Model

T = 90*24*3600
dt = 3600*6
N = T//dt

for i in tqdm(range(N)):

    # run groundwater model to get steady state solution
    diff = 1
    tol = 1e-10
    iter = 0
    while diff > tol and iter < 20:
        zwt0 = zwt.copy()
        gdp.run_with_adaptive_time_step_solver(0.1*dt)
        diff = np.max(zwt0-zwt)
        iter += 1
        # print(diff)

    # calculate Q with the actual darcy velocity *  hydraulic gradient
    vel_x, vel_y = map_link_vector_components_to_node_raster(mg, gdp._vel) # get mean value in x and y directions at node
    hydgr_x, hydgr_y = map_link_vector_components_to_node_raster(mg, gdp._hydr_grad)
    Q_node = Q_coeff * np.abs(vel_x * hydgr_x + vel_y * hydgr_y)

    melt_depth = const_melt_rate(S0=75, Q=Q_node) * dt # use the function initialized at the beginning
    zb[mg.core_nodes] = zb[mg.core_nodes] - melt_depth[mg.core_nodes] # subtract melt depth from zb (melt depth is positive when the active layer is deepening)

    #if zb > z (profile is entirely frozen) then make it equal to z
    if (zb > z).any():
        zb[zb > z] = z[zb > z] - 1e-3
        print('oh no zb > z setting it to z - delta <3')

    zwt[mg.core_nodes] = (zb + melt_depth + gdp._thickness)[mg.core_nodes] # update water table elevation (assume melted zone is fully saturated)

    if i % 100 == 0:
        f = zb - zb0
        # f = zwt0-zwt
        # plt.figure()
        # imshow_grid(mg, f, colorbar_label='zb-zb0', cmap='viridis')
        # axes[2].set_title(f'zb-zb0 at timestep {i}')
        print(f'Max melt at timestep {i} is {np.max(melt_depth)}')
        print(f'Max zb-zb0 at timestep {i} is {np.max(f)}')
        print(f'Max Q_node at timestep {i} is {np.max(Q_node)}')

# %%

f = zb - zb0
# f = zwt0-zwt
fg = f.reshape(mg.shape)

plt.figure()
for i in range(0,200,20):
    plt.plot(fg[i,1:-1])

plt.figure()
imshow_grid(mg, z-zb, colorbar_label='z-zb', cmap='viridis')
plt.title('Active Zone Thickness')

plt.figure()
imshow_grid(mg, f, colorbar_label='zb-zb0', cmap='viridis')
plt.title('Active Zone Change')
