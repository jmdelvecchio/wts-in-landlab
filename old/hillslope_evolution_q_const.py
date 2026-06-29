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

import numpy as np
from scipy import optimize

def calc_one_wavelength(
    slope_deg,
    frozen_grad,
    unfrozen_grad,
    x_t = 1600,
    porosity=0.9,
    beta = 0.04,
    flow_speed = 0.1
    ):
    
    def get_wavelength(RHS, LHS_den_term):
        """Numerically solves for kappa and then wavelength"""

        def f(kappa):

            LHS_num = kappa**(8/3)
            # LHS_den = (kappa**2 + ((C_f*rho_f*(Q_bar_calc-(beta*unfrozen_grad)))/(K_f*rho_w*L_curly)))**(0.5)
            LHS_den = (kappa**2 + ((LHS_den_term)))**(0.5)
            LHS = LHS_num / LHS_den

            return LHS - RHS

        try:
            sol = optimize.root_scalar(f, bracket=[
                RHS**(3/5), # Keeps it the "right" solution
                3e2], method='brentq')
            wavelength = 2*3.1415/sol.root
        except ValueError:
            # print(f'No solution > RHS**(3/5) when RHS ={RHS}')
            wavelength = np.nan

        return wavelength
    
    # Match sign convention of original wavelength formulation
    frozen_grad = abs(frozen_grad)
    unfrozen_grad = abs(unfrozen_grad)
    
    # Constants
    rho_w = 1000 # kg m^3 
    rho_s = 2600
    rho_i = 900

    C_w = 4184 # J K-1 kg-1 
    C_s = 700 
    C_i = 2050

    K_w = 0.598 # W/m·K 
    K_s = 1.460 #* 5 # not sure what this 5 is doing here. 
    K_i = 2.220

    g = 9.8
    rho_f = rho_s*(1-porosity) + (rho_i * porosity)
    C_f = C_s*(1-porosity) + (C_i * porosity)
    K_f = K_s*(1-porosity) + (K_i * porosity)
    slope = np.tan(np.deg2rad(slope_deg))

    L_curly = 334E3 * porosity # Latent head of fusion modulated by porosity

    # Calculate Q_bar, which is based on both
    Q_bar_calc = (flow_speed * rho_w * g * slope)
    growth_rate = (1/(rho_w*porosity*L_curly)) * (Q_bar_calc - (beta * unfrozen_grad))

    RHS = (2.0374 * (Q_bar_calc)) / (x_t**(2/3) * (K_f * frozen_grad))
    # print(f'Using RHS {RHS}')
    LHS_den_term = ((C_f*rho_f*(Q_bar_calc-(beta*unfrozen_grad)))/(K_f*rho_w*L_curly))
    wavelength = get_wavelength(RHS, LHS_den_term)

    return [wavelength, growth_rate]

    
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
                phi=0.9,
                Ks=2.73, 
                dTdz=-20.0, 
                Q=0,
                rho_ice=917):
    """
    Computes the melt rate at the permafrost table.

    Parameters:
    S0 : peak solar irradiance (W/m^2) -- but may want to set this lower to account for insulation
    L : latent heat of fusion (J/kg)
    phi : porosity of the soil (dimensionless)
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
    melt_rate_per_second = (energy_flux / (L*phi*rho_ice))#* melt_factor

    return melt_rate_per_second

#%% Visualize melt rate

slope_deg = 4.8 # use Warburton and others' value for slope gradient, 4.8 degrees
dTdz = -20.0

S0 = 75 # W/m^2, peak solar irradiance
rho_w = 1000 # kg/m^3
g = 9.81 # m/s^2
u = np.geomspace(1e-5, 1e-1, 10) # m/s, velocity of water flow
hydr = np.sin(slope_deg * np.pi/180) 

Q = rho_w * g * u * hydr # W/m^2 internal heat dissipation

plt.figure()
plt.plot(u, const_melt_rate(S0=S0, Q=Q, dTdz=dTdz)*3600*24*90, 'o-')
plt.xlabel('Water Flow Velocity (m/s)')
plt.ylabel('90 Day Melt Rate (m)')
plt.title('Melt Rate vs Water Flow Velocity')
# plt.xscale('log')
# plt.yscale('log')
plt.grid(True)

#%% Find wavelength and growth rate of these parameters

wavelength, growth_rate = calc_one_wavelength(
    np.rad2deg(np.atan(0.2)), #slope_deg,
    dTdz,
    dTdz,
    beta=0.04,
    flow_speed=0.025,
    )

print(f'Wavelength: {round(wavelength, 2)} meters')
print(f'You can form it over: {round(1/(growth_rate * 3.15e7),3)} years')

#%%
xslope_var_all = []
#%% Create the grid, add basic fields

boundaries = {"top": "open", "left": "closed", "bottom": "closed", "right": "closed"}

Nx = 101; Ny = 200; dx = 10
mg = RasterModelGrid((Ny,Nx), xy_spacing=dx, bc=boundaries)
z = mg.add_zeros('topographic__elevation', at='node')
zb = mg.add_zeros('aquifer_base__elevation', at='node')
zwt = mg.add_zeros("water_table__elevation", at="node")

# some parameters
b = 10 # permeable thickness m
r = 1.0e-6 # recharge rate (constant, uniform here) m/s
ksat = 1e-1 # hydraulic conductivity (constant, uniform here) m/s
phi = 0.9 # porosity (constant, uniform here) -- does not matter for steady state solution
S0 = 0.04*20 # W/m^2, peak solar irradiance
rho_w = 1000 # kg/m^3
g = 9.81 # m/s^2

# thermal conductivities:
kf = 2.0 # W/m/K, frozen soil
ku = 0.5 # W/m/K, unfrozen soil

# other parameters
Tm = 0 # C, melting temperature
L = 334E3 # J/kg, latent heat of fusion

# constants for now
frozen_gradient = -20 # -20 # K/m, temperature gradient in the frozen soil (constant for now)
T_surface = 0 #-5 # C, surface temperature (constant for now)

# some example parabolic hillslope, just made up
x = mg.x_of_node
y = mg.y_of_node
a = 0.00005
z[:] = -a * y**2 + a * max(y)**2

# calc average slope of hillslope
slope = np.arctan(np.mean(np.abs(np.gradient(z.reshape(mg.shape), dx, axis=0))))
slope_deg = np.rad2deg(slope)
print(f'Average slope of hillslope is {round(slope_deg, 2)} degrees')

# calc theoretical wavelength and growth rate for these parameters
wavelength, growth_rate = calc_one_wavelength(
    slope_deg,
    frozen_gradient,
    ku * (T_surface - Tm) / b, # use the conductive flux to
    x_t = max(mg.y_of_node), # use the length of the hillslope as the characteristic length scale
    porosity=phi,
    beta = 0.04,
    flow_speed = ksat*slope # m/s, just a guess for now
    )
print(f'Wavelength: {round(wavelength, 2)} meters')
print(f'Growth rate: {3600*24*365*growth_rate:.2e} meters/year')


#%%

# generate a random field to perturb the base elevation
lam = 5 # correlation length for the random field
alpha = 0.05 # scaling factor for the random field
# zb[:] = z - b + alpha * generate_correlated_random_field(Ny, Nx, lam/dx * 2, 2142025).flatten()
zb[:] = z - b + alpha * np.random.randn(Ny, Nx).flatten()

# wavelength_target = round(wavelength, 2)*2.0  # meters
# kappa = 2 * np.pi / wavelength_target
# amplitude = 0.01  # small perturbation, meters
# zb[:] = z - b + amplitude * np.sin(kappa * mg.x_of_node)

zb0 = zb.copy()
zwt[:] = zb + 0.1 # near equilibrium thickness

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


#%% Initialize components

# initialize the groundwater model
gdp = GroundwaterDupuitPercolator(
    mg,
    recharge_rate=r,
    hydraulic_conductivity=ksat,
    porosity=phi,
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
gwf0 = gdp.calc_gw_flux_at_node() # map groundwater flux to node (easier to plot)


#%% Calc Q and plot

# calculate internal heating factor Q
Q_coeff = rho_w * g # convert from head gradient to pressure gradient 
vel_x, vel_y = map_link_vector_components_to_node_raster(mg, gdp._vel) # get mean value in x and y directions at node
hydgr_x, hydgr_y = map_link_vector_components_to_node_raster(mg, gdp._hydr_grad)
Q_node = Q_coeff * np.abs(vel_x * hydgr_x + vel_y * hydgr_y) # absolute value of the dot product, as in the model description

q_x, q_y = map_link_vector_components_to_node_raster(mg, gdp._q) # get mean value in x and y directions at node
Q_node_alt = Q_coeff * np.abs(q_x * hydgr_x + q_y * hydgr_y) # should be the same as above, just using q instead of vel*hydr
# cross_slope_mean_thickness = np.broadcast_to(np.mean((zwt-zb).reshape(mg.shape), axis=1)[:,np.newaxis], mg.shape) # mean thickness in cross slope direction
# Q_node_alt = Q_node_alt / cross_slope_mean_thickness.flatten() # convert from W/m^2 to W/m^3 by dividing by thickness of active layer
# Q_node_alt = Q_node_alt / np.mean(zwt-zb) # convert from W/m^2 to W/m^3 by dividing by thickness of active layer


# Q (heat source) rate
plt.figure()
imshow_grid(mg, Q_node, cmap='plasma')
plt.title('Q_node')

plt.figure()
imshow_grid(mg, Q_node_alt, cmap='plasma')
plt.title('Q_node_alt')

# %% Some figures

# groundwater flux, mapped from links to nodes
plt.figure()
imshow_grid(mg, gwf0, cmap='plasma')
plt.title('Groundwater Flux at Node')

# x velocity
plt.figure()
imshow_grid(mg, vel_x, cmap='RdBu')
plt.title('Vel_x')

# y velocity
plt.figure()
imshow_grid(mg, vel_y, cmap='Blues')
plt.title('Vel_y')

# local produced runoff
plt.figure()
imshow_grid(mg, 'surface_water__specific_discharge', cmap='viridis')
plt.title('Local Runoff')

# saturated thickness
plt.figure()
imshow_grid(mg, (zwt-zb)/(z-zb), cmap='Blues')
plt.title('Relative Saturated Thickness')

print(f'Median aquifer thickness is {np.median(zwt-zb)} meters')
print(f'Mean aquifer thickness is {np.mean(zwt-zb)} meters')

# These plots make it clear why Q_node remains so uniform: the flux is dominated by
# the downslope direction, which is not really affected by perturbations. Cross slope
# gradients are much smaller, don't really appear when added together.

# %% Test Simple Hillslope Evolution Model

T = 180*24*3600
dt = 3600*6
N = T//dt
Q_method = 'q'

h = mg.at_node['aquifer__thickness'] 

xslope_var = np.zeros(N)
for i in tqdm(range(N)):

    # run groundwater model to get steady state solution
    diff = 1
    tol = 1e-8
    iter = 0
    while diff > tol and iter < 20:
        zwt0 = zwt.copy()
        gdp.run_with_adaptive_time_step_solver(0.1*dt)
        diff = np.max(zwt0-zwt)
        iter += 1
        # print(diff)

    if Q_method == 'q':
        # alternative way to calculate Q with averaged q 
        q_x, q_y = map_link_vector_components_to_node_raster(mg, gdp._q) # get mean value in x and y directions at node
        Q_node_alt = Q_coeff * np.abs(q_x * hydgr_x + q_y * hydgr_y) # should be the same as above, just using q instead of vel*hydr
        # Q_node = Q_node_alt / np.mean(zwt-zb, axis=0) # convert from W/m^2 to W/m^3 by dividing by thickness of active layer
        Q_node = Q_node_alt
    else:
        # calculate Q with the actual darcy velocity *  hydraulic gradient
        vel_x, vel_y = map_link_vector_components_to_node_raster(mg, gdp._vel) # get mean value in x and y directions at node
        hydgr_x, hydgr_y = map_link_vector_components_to_node_raster(mg, gdp._hydr_grad)
        Q_node = Q_coeff * np.abs(vel_x * hydgr_x + vel_y * hydgr_y)

    # Background flux terms (uniform, just set the mean thaw rate)
    flux_solar = ku * (T_surface - Tm) / b  # uniform with b as thickness of the active layer
    flux_frozen = kf * frozen_gradient  # uniform background

    # Spatially variable term - this is where the instability lives
    flux_dissipation = Q_node  # varies with local flow conditions

    # Interface velocity
    dzb_dt = (flux_solar + flux_frozen + flux_dissipation) / (rho_w * phi * L) # flux frozen added because value is negative, so it reduces the melt rate
    zb[:] = zb - dzb_dt * dt # note this also updates the boundary condition for the groundwater model, which is important for the feedback to work
    # zwt keeps same position, aquifer adds water from deepening of permafrost table, so thickness increases by the melt depth
    h[:] = (zwt - zb) # update aquifer thickness

    # cross slope variability metric
    signal = np.std(zb.reshape(mg.shape), axis=1).mean()
    xslope_var[i] = signal

    # #if zb > z (profile is entirely frozen) then make it equal to z
    # if (zb > z).any():
    #     zb[zb > z] = z[zb > z] - 1e-3
    #     print('oh no zb > z setting it to z - delta <3')

    if i % 100 == 0:
        f = zb - zb0

        tqdm.write(f'Max melt at timestep {i} is {np.max(dzb_dt * dt)} meters')
        tqdm.write(f'Max zb-zb0 at timestep {i} is {np.max(f)}')
        tqdm.write(f'Max Q_node at timestep {i} is {np.max(Q_node)}')

# %%

# along-slope only (base state)
u_along = np.mean(np.abs(vel_y))

# cross-slope perturbation amplitude
u_cross = np.std(vel_x.reshape(mg.shape), axis=0).mean()

print(f'Along-slope mean velocity: {u_along:.2e} m/s')
print(f'Cross-slope perturbation velocity: {u_cross:.2e} m/s')

#%% Plots

f = zb - zb0
fg = f.reshape(mg.shape)
gwf = gdp.calc_gw_flux_at_node() # map groundwater flux to node (easier to plot)

plt.figure()
for i in range(0,200,20):
    plt.plot(fg[i,1:-1])

plt.figure()
imshow_grid(mg, zb, colorbar_label='z-zb', cmap='viridis')
plt.title('Base Elevation of Active Zone')

plt.figure()
imshow_grid(mg, z-zb, colorbar_label='z-zb', cmap='viridis')
plt.title('Active Zone Thickness')

plt.figure()
imshow_grid(mg, f, colorbar_label='zb-zb0', cmap='viridis')
plt.title('Active Zone Change')

plt.figure()
imshow_grid(mg, gwf0 - gwf, cmap='plasma')
plt.title('Change in Groundwater Flux at Node')

#%%
# hydraulic gradient at node
hydgr_x, hydgr_y = map_link_vector_components_to_node_raster(mg, gdp._hydr_grad)
hydgr_mag = np.sqrt(hydgr_x**2 + hydgr_y**2)
plt.figure()
imshow_grid(mg, hydgr_mag, cmap='plasma')
plt.title('Hydraulic Gradient Magnitude at Node')

vel_x, vel_y = map_link_vector_components_to_node_raster(mg, gdp._vel) # get mean value in x and y directions at node
vel_mag = np.sqrt(vel_x**2 + vel_y**2)
plt.figure()
imshow_grid(mg, vel_mag, cmap='plasma')
plt.title('Velocity Magnitude at Node')

# groundwater flux, mapped from links to nodes
plt.figure()
imshow_grid(mg, gwf, cmap='plasma')
plt.title('Groundwater Flux at Node')


#%%
t = np.arange(N) * dt
plt.figure()
plt.plot(t, xslope_var)
plt.xlabel('Time (s)')
plt.ylabel('Mean Std Dev of zb in Cross Slope Direction')

# %%

# Q (heat source) rate
plt.figure()
imshow_grid(mg, Q_node, cmap='plasma')
plt.title('Q_node')


plt.figure()
imshow_grid(mg, gwf/h, cmap='plasma')
plt.title('Groundwater Flux at Node')

# plot_average_psd(zb0.reshape(mg.shape), fs=1.0)
# plot_average_psd(zb.reshape(mg.shape), fs=1.0)
# plot_average_psd((gwf-gwf0).reshape(mg.shape), fs=1.0)
# q = mg.at_node['surface_water__specific_discharge']
# plot_average_psd(q.reshape(mg.shape), fs=1.0)

# %%


xslope_var_all.append(xslope_var)
# %%


run_names = ['0.5x wavelength', '1x wavelength', '2x wavelength']
plt.figure()
for i, xslope_var in enumerate(xslope_var_all):
    plt.plot(t, xslope_var, label=run_names[i])
plt.xlabel('Time (s)')
plt.ylabel('Mean Std Dev of zb in Cross Slope Direction')
plt.legend()
# %%

plt.figure()
plt.subplot(1, 3, 2)
imshow_grid(mg, Q_node, cmap='plasma', colorbar_label='Dissipation (W/m^2)')
plt.colorbar(ax=plt.gca(),label='Dissipation (W/m^2)')
# %%
