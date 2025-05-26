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
from scipy.signal import welch

from landlab import RasterModelGrid, imshow_grid
from landlab.components import GroundwaterDupuitPercolator, FlowAccumulator
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

def melt_rate(t_seconds,
                S0=700, # 700 This is twice as high as it should be but I couldn't get the thaw to be deep enough
                L = 334E3 * 0.5, # ice content modulated by porosity?  # 334000,
                Ks=2.73, 
                dTdz=-20.0, 
                Q=0, 
                Tm=0, 
                k=1, 
                T_mean=-2, 
                T_range=15, 
                rho_ice=917):
    """
    Computes the melt rate at the permafrost table using time in seconds.
    t_seconds : time in seconds
    S0 : peak solar irradiance (W/m^2)
    L : latent heat of fusion (J/kg)
    Ks : thermal conductivity of frozen soil (W/m*K)
    dTdz : temperature gradient in the frozen soil (K/m)
    Q : internal heat dissipation (W/m^2)
    Tm : melting temperature threshold (°C)
    k : steepness of the melting transition
    T_mean : mean annual temperature (°C)
    T_range : annual temperature range (°C)
    rho_ice : density of ice (kg/m³)
    """
    seconds_per_year = 365 * 86400  # Total seconds in a year
    doy_shift_seconds = 180 * 86400  # Shift in seconds

    # Seasonal solar irradiance variation (in seconds)
    irradiance = S0 * np.cos(2 * np.pi * (t_seconds - doy_shift_seconds) / seconds_per_year)

    # Seasonal temperature variation (in seconds)
    T_surface = T_mean + (T_range / 2) * np.cos(2 * np.pi * (t_seconds - doy_shift_seconds) / seconds_per_year)

    # Compute total energy flux
    energy_flux = irradiance + Ks * dTdz + Q

    # Smooth melt function using logistic function
    melt_factor = 1 / (1 + np.exp(-k * (T_surface - Tm)))

    # Convert energy flux to melt rate in meters per second
    # PLus or minus if you care about this "melt factor" fudge
    # melt_rate_per_second = (energy_flux / (L) * melt_factor
    melt_rate_per_second = (energy_flux / (L*rho_ice))#* melt_factor

    return melt_rate_per_second

#%% Visualize melt rate

dt = 100
Q1 = 10
t = np.arange(0,60*60*24*365,dt)
plt.figure()


fig, ax = plt.subplots()
ax.plot(t, melt_rate(t), label='Q=0')
ax.plot(t, melt_rate(t,Q=Q1), label='Q>0')

ax1 = ax.twinx()
ax1.plot(t, np.cumsum(melt_rate(t))*dt, label='Q=0 cumulative', linestyle='--')
ax1.plot(t, np.cumsum(melt_rate(t, Q=Q1))*dt, label='Q>0 cumulative', linestyle='--')
plt.legend()


#%% Create the grid, add basic fields

boundaries = {"top": "open", "left": "closed", "bottom": "closed", "right": "closed"}

Nx = 101; Ny = 200; dx = 5
mg = RasterModelGrid((Ny,Nx), xy_spacing=dx, bc=boundaries)
z = mg.add_zeros('topographic__elevation', at='node')
zb = mg.add_zeros('aquifer_base__elevation', at='node')
zwt = mg.add_zeros("water_table__elevation", at="node")

# some parameters
b = 50 # permeable thickness m
r = 2.0e-8 # recharge rate (constant, uniform here) m/s
ksat = 1e-4 # hydraulic conductivity (constant, uniform here) m/s
n = 0.1 # porosity (constant, uniform here) -- does not matter for steady state solution
routing_method = 'MFD' # could also be 'D8' or 'Steepest'

# some example parabolic hillslopes, just made up
x = mg.x_of_node
y = mg.y_of_node
a = 0.0002
z[:] = -a * y**2 + a * max(y)**2

# zinthat = 0.2
# kappa = 2*np.pi/lam
# zb[:] = z - b + zinthat * np.cos(kappa * x)
# zb[:] = z - b # set constant permeable thickness b
lam = 5
zb[:] = z - b + 0.05 * generate_correlated_random_field(Ny, Nx, lam/dx * 2, 2142025).flatten()
zb0 = zb.copy()

zwt[:] = zb + 0.5 # start water table at the surface


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


Q_coeff = 1
# calculate Q with the actual darcy velocity *  hydraulic gradient
vel_x, vel_y = map_link_vector_components_to_node_raster(mg, gdp._vel) # get mean value in x and y directions at node
hydgr_x, hydgr_y = map_link_vector_components_to_node_raster(mg, gdp._hydr_grad)
Q_node = Q_coeff * np.abs(vel_x * hydgr_x + vel_y * hydgr_y) # absolute value of the dot product, just like in the model description


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

# x velocity
plt.figure()
imshow_grid(mg, vel_x, cmap='RdBu')
plt.title('Vel_x')

# y velocity
plt.figure()
imshow_grid(mg, vel_y, cmap='Blues')
plt.title('Vel_y')

#%%
# Q rate
plt.figure()
imshow_grid(mg, Q_node, cmap='Blues')
plt.title('Q_node')

# irradiance = 600
# melt_rate = ((irradiance + 2.73 * -20.0 + Q_node)/(3343e3 * 0.5)) 
# plt.figure()
# imshow_grid(mg, melt_rate, cmap='Blues')
# plt.title('Melt for 1 second\n ')
# print(f'Max melt rate: {np.max(melt_rate)}; min melt rate: {np.min(melt_rate)}')

# These plots make it clear why Q_node remains so uniform: the flux is dominated by
# the downslope direction, which is not really affected by perturbations. Cross slope
# gradients are much smaller, don't really appear when added together.
# plt.figure()
# imshow_grid(mg, vel_y, cmap='Blues')

# plt.figure()
# imshow_grid(mg, vel_x, cmap='Blues')

#%%

# plot_average_psd(Q_node.reshape(mg.shape), fs=1.0)
# plot_average_psd(zb0.reshape(mg.shape), fs=1.0)
# q = mg.at_node['surface_water__specific_discharge']
# plot_average_psd(q.reshape(mg.shape), fs=1.0)

# %% Test Lazy Evolution

T = 365*24*3600
dt = 3600*6
N = T//dt

Q_coeff = 1 #100

slp = np.max(mg.at_node['topographic__steepest_slope'], axis=1) # we are not evolving topography, so the topographic slope stays constant
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

    ## use for debuggin: it is usually zwt that gives nans.
    # a1 = np.isnan(gdp._vel).any()
    # a2 = np.isnan(gdp._hydr_grad).any()
    # a3 = np.isnan(zwt).any()
    # if a1 or a2 or a3:
    #     break
    
    # calculate Q with the actual darcy velocity *  hydraulic gradient
    vel_x, vel_y = map_link_vector_components_to_node_raster(mg, gdp._vel) # get mean value in x and y directions at node
    hydgr_x, hydgr_y = map_link_vector_components_to_node_raster(mg, gdp._hydr_grad)
    Q_node = Q_coeff * np.abs(vel_x * hydgr_x + vel_y * hydgr_y) # absolute value of the dot product, just like in the model description

    melt_depth = melt_rate(i*dt,Q=Q_node) * dt # use the function initialized at the beginning
    zb[mg.core_nodes] = zb[mg.core_nodes] - melt_depth[mg.core_nodes]

    #if zb > z then make it equal to z
    if (zb > z).any():
        zb[zb > z] = z[zb > z] - 1e-3
        print('oh no zb > z setting it to z - delta <3')

    zwt[mg.core_nodes] = (zb + melt_depth + gdp._thickness)[mg.core_nodes]

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
plt.figure()
imshow_grid(mg, f, colorbar_label='zb-zb0', cmap='viridis')
axes[2].set_title('Active Zone Change')

fg = f.reshape(mg.shape)

plt.figure()
for i in range(0,200,20):
    plt.plot(fg[i,1:-1])


# %%
print(f'Max change in zb= {np.max(f)}')
# # %%
# from landlab.plot.graph import plot_graph

# grid = RasterModelGrid((4, 5), xy_spacing=(3, 4))
# plot_graph(grid, at="node")
# #%%
# plot_graph(grid, at="link")# %%

# %%
