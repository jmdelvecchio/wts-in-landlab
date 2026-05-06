"""
Models for water track formation and evolution, including the main model class and supporting functions.

Model class: WaterTrackModel
Supporting functions: water_track_funcs.py
"""
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from landlab import imshow_grid
from landlab.components import GroundwaterDupuitPercolator
from landlab.grid.raster_mappers import map_link_vector_components_to_node_raster
from water_track_funcs import calc_one_wavelength


class WaterTrackModel:
    """A class to model water track formation and evolution on hillslopes.
    
    
    """

    def __init__(self, grid, params):
        """Initialize the model with a landlab grid and parameters."""
        self.grid = grid
        self.params = params
        # Initialize other model components here (e.g., groundwater flow, erosion)
        
        self.S0 = params.get('S0', 0.04) # 0.04 W/m^2, peak solar irradiance
        # thermal conductivities:
        self.kf = params.get('kf', 2.0) # 2.0 W/m/K, frozen soil
        self.ku = params.get('ku', 0.5) # 0.5 W/m/K, unfrozen soil
        self.frozen_gradient = params.get('frozen_gradient', -20) # -20 # K/m, temperature gradient in the frozen soil (constant for now)
        self.T_surface = params.get('T_surface', 0) # # C, surface temperature (constant for now)
        self.rho_w = params.get('rho_w', 1000) # kg/m^3
        self.phi = params.get('porosity', 0.9) # porosity
        
        self.g = params.get('g', 9.81) # m/s^2
        self.Tm = params.get('Tm', 0) # C, melting temperature
        self.L = params.get('L', 334E3) # J/kg, latent heat of fusion

        self.tol = params.get('tol', 1e-10) # tolerance for numerical solvers
        self.max_iter = params.get('max_iter', 20) # maximum iterations for numerical solvers

        # time stepping parameters
        self.dt = params.get('dt', 6*3600) # seconds
        self.gwdt = params.get('gwdt', 1e3) # seconds, groundwater model timestep #TODO: make this adaptive based on convergence of groundwater model
        self.T = params.get('T', 180*24*3600) # seconds, total simulation time


        self.gdp = GroundwaterDupuitPercolator(
                    self.grid,
                    recharge_rate=self.params['recharge_rate'],
                    hydraulic_conductivity=self.params['hydraulic_conductivity'],
                    porosity=self.phi,
                    regularization_f=0.1,
                    )
        self._z = self.grid.at_node['topographic__elevation']
        self._zb = self.grid.at_node['aquifer_base__elevation']
        self._zwt = self.grid.at_node['water_table__elevation']
        self._Qdiss = self.grid.add_zeros('node', 'thermal_dissipation')
        self._zb0 = self._zb.copy()
        self._h = self.grid.add_zeros('node', 'aquifer_thickness')
        
        def verbose_print(*args, **kwargs):
            if self.params.get('verbose', False):
                print(*args, **kwargs)
        self.verbose_print = verbose_print

    def estimate_wavelength(self):
        """Estimate the most unstable wavelength based on the current model parameters.
        Assumes uniform slope, with long edge in the y direction.

        This function might be wrong. Need to check the terms. 
        """

        slope = np.arctan(np.mean(np.abs(np.gradient(self._z.reshape(self.grid.shape), self.grid.dx, axis=0))))
        slope_deg = np.rad2deg(slope)
        print(f'Average slope of hillslope is {round(slope_deg, 2)} degrees')

        wavelength, growth_rate = calc_one_wavelength(
                                    slope_deg,
                                    self.frozen_gradient,
                                    self.S0 + self.ku * (self.T_surface - self.Tm) / np.mean(self._z-self._zb), # use the conductive flux to
                                    x_t = max(self.grid.y_of_node), # use the length of the hillslope as the characteristic length scale
                                    porosity=self.phi,
                                    beta = 0.04,
                                    flow_speed = self.gdp.ksat*slope # m/s, just a guess for now
                                    )
        print(f'Wavelength: {round(wavelength, 2)} meters')
        print(f'Growth rate: {3600*24*365*growth_rate:.2e} meters/year')



    def run_hydrology(self):
        """Run the groundwater flow model to get water table and fluxes."""


        # run groundwater model to get steady state solution
        diff = 1
        iter = 0
        while diff > self.tol and iter < self.max_iter:
            zwt0 = self._zwt.copy()
            self.gdp.run_with_adaptive_time_step_solver(self.gwdt)
            diff = np.max(zwt0 - self._zwt)
            iter += 1
        self.verbose_print(f'Groundwater model converged in {iter} iterations with max change {diff:.2e} m')

        # calculate internal heating factor Q
        Q_coeff = self.rho_w * self.g # convert from head gradient to pressure gradient 
    
        hydgr_x, hydgr_y = map_link_vector_components_to_node_raster(self.grid, self.gdp._hydr_grad)
        q_x, q_y = map_link_vector_components_to_node_raster(self.grid, self.gdp._q) # get mean value in x and y directions at node
        # self._Qdiss = Q_coeff * np.abs(q_x * hydgr_x + q_y * hydgr_y) # should be the same as above, just using q instead of vel*hydr
        self._Qdiss[:] = Q_coeff * np.abs(q_x * hydgr_x + q_y * hydgr_y) / np.mean(self._zwt - self._zb) # possibly wrong units?

    def run_step(self):
        """Run a single time step of the model."""
        self.run_hydrology()

        # Background flux terms (uniform, just set the mean thaw rate)
        flux_solar = self.S0 +  self.ku * (self.T_surface - self.Tm) / (self._z - self._zb)  # make this spatially variable by using local thickness instead of mean thickness
        flux_frozen = self.kf * self.frozen_gradient  # uniform background

        # Spatially variable term - this is where the instability lives
        flux_dissipation = self._Qdiss  # varies with local flow conditions

        # Interface velocity
        dzb_dt = (flux_solar + flux_frozen + flux_dissipation) / (self.rho_w * self.phi * self.L) # flux frozen added because value is negative, so it reduces the melt rate
        self._zb[:] = self._zb - dzb_dt * self.dt # note this also updates the boundary condition for the groundwater model, which is important for the feedback to work
        # zwt keeps same position, aquifer adds water from deepening of permafrost table, so thickness increases by the melt depth
        self._h[:] = (self._zwt - self._zb) # update aquifer thickness
    
    def run_model(self):
        """Run the model for the specified total time."""
        n_steps = int(self.T / self.dt)
        self.xslope_var = np.zeros(n_steps) # metric for cross slope variability of the water table, which should increase as water tracks form and evolve
        self.t = np.arange(n_steps) * self.dt
        for step in tqdm(range(n_steps)):
            self.run_step()

            # cross slope variability metric
            signal = np.std(self._zb.reshape(self.grid.shape), axis=1).mean()
            self.xslope_var[step] = signal
            
            if step % 10 == 0:
                self.verbose_print(f'Completed step {step}/{n_steps}')

    def make_plots(self):
        """Generate plots of the model results."""
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        imshow_grid(self.grid, 'aquifer_thickness', cmap='viridis', colorbar_label='Aquifer Thickness (m)')

        plt.subplot(1, 3, 2)
        imshow_grid(self.grid, self._Qdiss, cmap='inferno', colorbar_label='Dissipation (W/m^2)')

        plt.subplot(1, 3, 3)
        imshow_grid(self.grid, self._z - self._zb, cmap='plasma', colorbar_label='Active Layer Thickness (m)')
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(self.t, self.xslope_var)
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Std Dev of zb in Cross Slope Direction')

