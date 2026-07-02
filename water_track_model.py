"""
Models for water track formation and evolution, including the main model class and supporting functions.

Model class: WaterTrackModel
"""

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from landlab import imshow_grid
from landlab.components import GroundwaterDupuitPercolator
from landlab.grid.raster_mappers import map_link_vector_components_to_node_raster
from landlab.grid.mappers import map_mean_of_link_nodes_to_link

from DupuitLEM.io import (
    initialize_output_dataset,
    write_output_step,
)

class WaterTrackModel:
    """A class to model water track formation and evolution on hillslopes.
    """

    def __init__(self, grid, params, output_dict=None):
        """Initialize the model with a landlab grid and parameters."""
        self._grid = grid
        self.params = params
        self.output_dict = output_dict
        # Initialize other model components here (e.g., groundwater flow, erosion)
        
        self.S0 = params.get('S0', 0.04) # 0.04 W/m^2, peak solar irradiance
        # thermal conductivities:
        self.kf = params.get('kf', 2.0) # 2.0 W/m/K, frozen soil
        self.ku = params.get('ku', 0.5) # 0.5 W/m/K, unfrozen soil
        self.beta = params.get('beta', 0.04) # insulation parameter (W/m^2 K)
        self.frozen_gradient = params.get('frozen_gradient', -20) # -20 # K/m, temperature gradient in the frozen soil (constant for now)
        self.T_surface = params.get('T_surface', 0) # C, surface temperature (constant for now)
        self.T_air = params.get('T_air', 0) # C, air temperature (constant for now)
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
        self.n_steps = int(self.T / self.dt) 

        self.gdp = GroundwaterDupuitPercolator(
                    self._grid,
                    recharge_rate=self.params['recharge_rate'],
                    hydraulic_conductivity=self.params['hydraulic_conductivity'],
                    porosity=self.phi,
                    regularization_f=0.1,
                    # vn_coefficient=0.2
                    # courant_coefficient=0.1
                    )
        self._z = self._grid.at_node['topographic__elevation']
        self._zb = self._grid.at_node['aquifer_base__elevation']
        self._zwt = self._grid.at_node['water_table__elevation']
        self._Qdiss = self._grid.add_zeros('node', 'thermal_dissipation')
        self._zb0 = self._zb.copy()
        self._h = self._grid.add_zeros('node', 'aquifer_thickness')
        self._dzb_dt = np.zeros_like(self._zb) # initialize melt rate for use in correction term
        self._b = self._z - self._zb # initialize active layer thickness for use in correction term
        
        # configure outputs
        if output_dict:
            
            # set flag to save output, and store output dictionary
            self.save_output = True
            self.output = output_dict

            # store for easier access
            self.output_interval = output_dict["output_interval"]
            self.output_fields = output_dict["output_fields"]
            self.base_path = output_dict["base_output_path"]
            self.id = output_dict["run_id"]

            # initialize output dataset
            self._initialize_output()

        else:
            self.save_output = False


        def verbose_print(*args, **kwargs):
            if self.params.get('verbose', False):
                print(*args, **kwargs)
        self.verbose_print = verbose_print

    def _initialize_output(self):
        n_output = self.n_steps // self.output_interval

        self.output_times = (
            np.arange(n_output) * self.output_interval * self.dt
        )

        self._output_ds = initialize_output_dataset(
            self._grid,
            self.output,
            self.output_times,
        )

        self._output_path = self.base_path + f"{self.id}.nc"
        self._output_ds.to_netcdf(self._output_path, mode="w")

        self._output_index = 0


    def run_hydrology_steady(self):
        """Run the groundwater flow model to steady state to get water table, fluxes, and dissipative heating."""

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
    
        hydgr_x, hydgr_y = map_link_vector_components_to_node_raster(self._grid, self.gdp._hydr_grad)
        q_x, q_y = map_link_vector_components_to_node_raster(self._grid, self.gdp._q) # get mean value in x and y directions at node
        self._Qdiss = Q_coeff * np.abs(q_x * hydgr_x + q_y * hydgr_y) # should be the same as above, just using q instead of vel*hydr
        # self._Qdiss[:] = Q_coeff * np.abs(q_x * hydgr_x + q_y * hydgr_y) / np.mean(self._zwt - self._zb) # possibly wrong units?

    def run_hydrology_dynamic(self):
        """Run the groundwater flow model for a single timestep dt to get water table, fluxes, and dissipative heating."""

        self.gdp.run_with_adaptive_time_step_solver(self.dt)

        # calculate internal heating factor Q
        Q_coeff = self.rho_w * self.g # convert from head gradient to pressure gradient 
    
        hydgr_x, hydgr_y = map_link_vector_components_to_node_raster(self._grid, self.gdp._hydr_grad)
        q_x, q_y = map_link_vector_components_to_node_raster(self._grid, self.gdp._q) # get mean value in x and y directions at node
        self._Qdiss = Q_coeff * np.abs(q_x * hydgr_x + q_y * hydgr_y) # should be the same as above, just using q instead of vel*hydr
        # self._Qdiss[:] = Q_coeff * np.abs(q_x * hydgr_x + q_y * hydgr_y) / np.mean(self._zwt - self._zb) # possibly wrong units?
      

    def run_step(self):
        """Run a single time step of the model."""

        eps = 1e-4 # small value to prevent thickness from going to zero

        # Two options: steady state hydrology or dynamic
        if self.params.get('steady', False):
            self.run_hydrology_steady()
        else:
            self.run_hydrology_dynamic()

        # Flux terms: solar, frozen, and dissipative
        # flux_solar = self.S0 +  self.ku * (self.T_surface - self.Tm) / (self._z - self._zb)  # this version with thickness dependence
        flux_solar = self.S0 + self.beta * (self.T_air - self.T_surface) # this version assumes steady state in vertical profile: all energy from surface reaches the interface
        flux_frozen = self.kf * self.frozen_gradient  # uniform background (frozen gradient is a negative value)
        flux_dissipation = self._Qdiss  # varies with local flow conditions

        # correction term for flux spreading - melt diffusion 
        gradb = self._grid.calc_grad_at_link(self._b)
        bprod = map_mean_of_link_nodes_to_link(self._grid, self._dzb_dt * self._b) # map to links for later divergence calculation
        gradb_x, gradb_y = map_link_vector_components_to_node_raster(self._grid, gradb) # vector components
        gradb_sq_node = gradb_x**2 + gradb_y**2  # vector magnitude at nodes
        gradb_sq_link = map_mean_of_link_nodes_to_link(self._grid, gradb_sq_node)  # map to links for later divergence calculation
        self.melt_diffusion = self._grid.calc_flux_div_at_node((gradb * bprod) / (1 + gradb_sq_link)) # term all together (Warburton et al. 2024)

        # Interface velocity
        self._dzb_dt = (flux_solar + flux_frozen + flux_dissipation) / (self.rho_w * self.phi * self.L) + self.melt_diffusion  # flux frozen added because value is negative, so it reduces the melt rate
        self._zb[:] = self._zb - self._dzb_dt * self.dt # note this also updates the boundary condition for the groundwater model, which is important for the feedback to work
        self._zb[self._zb >= self._z] = self._z[self._zb >= self._z] - eps # make sure refreezing doesn't cause total freezing above the land surface
        
        # zwt keeps same position, aquifer adds water from deepening of permafrost table, so thickness increases by the melt depth
        self._h[:] = (self._zwt - self._zb) # update aquifer thickness
        self._b[:] = self._z - self._zb # update active layer thickness
    
    def run_model(self):
        """Run the model for the specified total time."""

        self.xslope_var = np.zeros(self.n_steps) # metric for cross slope variability of the water table, which should increase as water tracks form and evolve
        self.t = np.arange(self.n_steps) * self.dt
        for step in tqdm(range(self.n_steps)):
            self.run_step()

            # cross slope variability metric
            signal = np.std(self._zb.reshape(self._grid.shape), axis=1).mean()
            self.xslope_var[step] = signal
            
            if self.save_output and step % self.output_interval == 0:
                write_output_step(
                    self._output_ds,
                    self._grid,
                    self.output,
                    self._output_index,
                )

                self._output_index += 1

                self._output_ds.to_netcdf(self._output_path, mode="a")

    def make_plots(self):
        """Generate plots of the model results."""

        # final timestep map view
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        imshow_grid(self._grid, 'aquifer_thickness', cmap='viridis', colorbar_label='Aquifer Thickness (m)')

        plt.subplot(1, 3, 2)
        imshow_grid(self._grid, self._Qdiss, cmap='inferno', colorbar_label='Dissipation (W/m^2)')

        plt.subplot(1, 3, 3)
        imshow_grid(self._grid, self._z - self._zb, cmap='plasma', colorbar_label='Active Layer Thickness (m)')
        plt.tight_layout()
        plt.show()

        # time evolution of cross slope variability
        plt.figure()
        plt.plot(self.t, self.xslope_var)
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Std Dev of zb in Cross Slope Direction')

