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
from landlab.grid.mappers import map_mean_of_link_nodes_to_link, map_value_at_max_node_to_link

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
        
        self.S0 = params.get('S0', 10) # W/m^2, peak solar irradiance
        self.k_f = params.get('kf', 2.728) # 2.0 W/m/K, frozen soil
        self.k_u = params.get('ku', 1.2682) # 0.5 W/m/K, unfrozen soil
        self.beta = params.get('beta', 0.04) # insulation parameter (W/m^2 K)
        self.frozen_gradient = params.get('frozen_gradient', 10) # 10 # K/m, temperature gradient in the frozen soil (constant for now)
        self.T_air = params.get('T_air', 0) # C, air temperature (constant for now)
        self.rho_w = params.get('rho_w', 1000) # kg/m^3
        self.rho_s = params.get('rho_s', 2600) # kg/m^3
        self.C_s = params.get('C_s', 700) # J/kg/K, specific heat capacity of soil
        self.C_w = params.get('C_w', 4.2e3) # J/kg/K, specific heat capacity of water at ~5C
        self.phi = params.get('porosity', 0.9) # porosity

        self.C_u = self.phi * self.C_w + (1 - self.phi) * self.C_s # J/kg/K, unfrozen specific heat capacity 
        self.rho_u = self.phi * self.rho_w + (1 - self.phi) * self.rho_s # kg/m^3, unfrozen density
        
        self.g = params.get('g', 9.81) # m/s^2
        self.Tm = params.get('Tm', 0) # C, melting temperature
        self.L = params.get('L', 334e3) # J/kg, latent heat of fusion

        self.tol = params.get('tol', 1e-10) # tolerance for numerical solvers
        self.max_iter = params.get('max_iter', 20) # maximum iterations for numerical solvers

        # time stepping parameters
        self.dt = params.get('dt', 6*3600) # seconds
        self.gwdt = params.get('gwdt', 1e3) # seconds, groundwater model timestep #TODO: make this adaptive based on convergence of groundwater model
        self.T = params.get('T', 180*24*3600) # seconds, total simulation time
        self.n_steps = int(self.T / self.dt) 
        self._courant_coefficient = params.get('courant_coefficient', 0.5) # coefficient for advection in both gw model and thermal model

        self.use_melt_diffusion = params.get('use_melt_diffusion', False)
        self.use_steady_hydrology = params.get('use_steady_hydrology', False)

        self.gdp = GroundwaterDupuitPercolator(
                    self._grid,
                    recharge_rate=self.params['recharge_rate'],
                    hydraulic_conductivity=self.params['hydraulic_conductivity'],
                    porosity=self.phi,
                    regularization_f=0.1,
                    # vn_coefficient=0.2
                    courant_coefficient=self._courant_coefficient
                    )
        self._z = self._grid.at_node['topographic__elevation']
        self._zb = self._grid.at_node['aquifer_base__elevation']
        self._zwt = self._grid.at_node['water_table__elevation']
        self._T_mean = self._grid.at_node['mean_unfrozen__temperature']
        self._h = self._grid.at_node['aquifer__thickness']

        self._Qdiss = self._grid.add_zeros('node', 'thermal_dissipation')
        self._dzb_dt = np.zeros_like(self._zb) # initialize melt rate for use in correction term
        self._b = self._z - self._zb # initialize active layer thickness for use in correction term
        self._zb0 = self._zb.copy()

        # Boundary node bookkeeping. State variables on closed nodes stay at their initial values.
        # Open (fixed-value) nodes get a dynamic Dirichlet T_mean, set each substep
        # from the mean of their core neighbors, so heat/geometry can evolve there
        # too. 
        self._closed_nodes = self._grid.status_at_node == self._grid.BC_NODE_IS_CLOSED
        self._open_nodes = self._grid.status_at_node == self._grid.BC_NODE_IS_FIXED_VALUE

        open_node_ids = np.where(self._open_nodes)[0]
        nbrs = self._grid.active_adjacent_nodes_at_node[open_node_ids]  # (n_open, 4), -1 padded
        valid = nbrs >= 0
        nbrs_safe = np.where(valid, nbrs, 0)
        is_core_nbr = valid & (self._grid.status_at_node[nbrs_safe] == self._grid.BC_NODE_IS_CORE)

        self._open_node_ids = open_node_ids
        self._open_node_neighbors = nbrs_safe            # (n_open, 4)
        self._open_node_neighbor_mask = is_core_nbr       # (n_open, 4) bool
        self._open_node_has_core_neighbor = is_core_nbr.any(axis=1)

        # depth-integrated sensible heat, J/m2
        self._E = self._grid.add_zeros('node', 'sensible_heat_content')
        self._E = self.C_u * self.rho_u * self._b * (self._T_mean - self.Tm)

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
        self._Qdiss = Q_coeff * np.abs(q_x * hydgr_x + q_y * hydgr_y) # dissipated heat is rho_w g (q dot gradh)


    def run_hydrology_dynamic(self):
        """Run the groundwater flow model for a single timestep dt to get water table, fluxes, and dissipative heating."""

        self.gdp.run_with_adaptive_time_step_solver(self.dt)

        # calculate internal heating factor Q
        Q_coeff = self.rho_w * self.g # convert from head gradient to pressure gradient 
    
        hydgr_x, hydgr_y = map_link_vector_components_to_node_raster(self._grid, self.gdp._hydr_grad)
        q_x, q_y = map_link_vector_components_to_node_raster(self._grid, self.gdp._q) # get mean value in x and y directions at node
        self._Qdiss = Q_coeff * np.abs(q_x * hydgr_x + q_y * hydgr_y) # dissipated heat is rho_w g (q dot gradh)
    

    def _update_open_boundary_temperature(self):
        """Set T_mean at open (fixed-value) boundary nodes to the mean of their
        core neighbors. Acts as a dynamic Dirichlet condition: it lets heat
        cross the open boundary and tracks the domain's evolving state instead
        of pinning the boundary to a fixed value, while carrying no cross-
        boundary structure of its own. Nodes with no core neighbor (e.g.
        corners) are left unchanged.
        """
        has_nbr = self._open_node_has_core_neighbor
        if not np.any(has_nbr):
            return

        ids = self._open_node_ids[has_nbr]
        nbr_ids = self._open_node_neighbors[has_nbr]
        mask = self._open_node_neighbor_mask[has_nbr]

        nbr_vals = np.where(mask, self._T_mean[nbr_ids], 0.0)
        counts = mask.sum(axis=1)
        self._T_mean[ids] = nbr_vals.sum(axis=1) / counts

    def run_heat_transport(self):
        """
        Update depth-averaged active layer temperature T_mean for one timestep dt, using 
        a depth-integrated sensible heat formulation: E = Cu rho_u b (T_mean - Tm)
        Computes and stores dzb_dt via the Stefan condition for use in run_step().
        
        The depth-integrated heat equation for E is:
        
        dE/dt = div (ku b gradT) + Qdiss + BC_top - flux_unfrozen
                - Cu rho_u * [div (T q) - T div q]
        
        where the Stefan condition at the base gives dzb_dt. The term [div (T q) - T div q]
        comes from the product rule: q grad T = div (T q) - T div q
        """

        flux_frozen = self.k_f * self.frozen_gradient  # W/m2

        remaining_time = self.dt
        self._num_substeps = 0
        dz = np.zeros_like(self._zb)  # sum up the melt at each subtimestep

        # local copy of active layer thickness, evolved internally within this
        # call to keep flux_unfrozen and the diffusion coefficient consistent with the
        # current thaw depth as it changes over substeps. self._b is only updated in run_step.
        b_local = self._b.copy()

        grad_T = np.zeros_like(self._grid.length_of_link)
        while remaining_time > 0:

            # Lateral diffusion: div (ku b gradT) [W/m2]
            grad_T[self._grid.active_links] = self._grid.calc_grad_at_link(self._T_mean)[self._grid.active_links]
            b_at_link = map_mean_of_link_nodes_to_link(self._grid, b_local)
            diff_flux = self.k_u * b_at_link * grad_T
            lateral_diffusion = self._grid.calc_flux_div_at_node(diff_flux)
            # lateral_diffusion = np.zeros_like(self._T_mean)

            # Lateral advection: q gradT = div (Tq) - T(div q) [K m/s at nodes]
            T_at_link = map_value_at_max_node_to_link(self._grid, self._zwt, self._T_mean)
            adv_flux = T_at_link * self.gdp._q
            flux_div_Tq = self._grid.calc_flux_div_at_node(adv_flux)
            flux_div_q = self._grid.calc_flux_div_at_node(self.gdp._q)
            lateral_advection = flux_div_Tq - self._T_mean * flux_div_q  # K m/s at nodes

            # Top boundary condition [W/m2]
            BC_top = self.S0 + self.beta * (self.T_air - self._T_mean)

            # Bottom boundary condition: heat flux from active layer to interface [W/m2]
            # Linear approximation to temperature profile, T drops from T_mean to Tm over b/2
            flux_unfrozen = self.k_u * (self._T_mean - self.Tm) / (b_local / 2)

            # calculate courant minimum timestep
            dt_courant = self._courant_coefficient * np.min(
                np.divide(
                    self._grid.length_of_link,
                    abs(self.gdp._vel),
                    where=abs(self.gdp._vel) > 0,
                    out=np.ones_like(self.gdp._vel) * 1e15,
                )
            )
            substep_dt = min([dt_courant, remaining_time])

            # Stefan condition: local interface velocity for this substep [m/s]
            if self.use_melt_diffusion:
                gradb = self._grid.calc_grad_at_link(b_local)
                bprod = map_mean_of_link_nodes_to_link(self._grid, self._dzb_dt * b_local)
                gradb_x, gradb_y = map_link_vector_components_to_node_raster(self._grid, gradb)
                gradb_sq_node = gradb_x**2 + gradb_y**2
                gradb_sq_link = map_mean_of_link_nodes_to_link(self._grid, gradb_sq_node)
                self.melt_diffusion = self._grid.calc_flux_div_at_node((gradb * bprod) / (1 + gradb_sq_link))
            else:
                self.melt_diffusion = 0.0

            db_dt_local = (flux_unfrozen - flux_frozen) / (self.rho_w * self.phi * self.L) + self.melt_diffusion
            # Closed nodes carry no flux so do not evolve
            db_dt_local[self._closed_nodes] = 0.0

            # Sensible heat content update [W/m2]
            # Newly thawed material enters at Tm, contributing zero to E.
            dE_dt = (
                lateral_diffusion
                + self._Qdiss
                + BC_top
                - flux_unfrozen
                - self.C_u * self.rho_u * lateral_advection
            )
            self._E[self._grid.core_nodes] += dE_dt[self._grid.core_nodes] * substep_dt

            # advance the local geometry
            b_local = b_local + db_dt_local * substep_dt

            # recover T_mean diagnostically from updated E and b_local
            self._T_mean[self._grid.core_nodes] = (
                self.Tm + self._E[self._grid.core_nodes] / (self.C_u * self.rho_u * b_local[self._grid.core_nodes])
            )

            # Dynamic Dirichlet BC at open boundaries, refreshed each substep so
            # it stays current for the next substep fluxes
            self._update_open_boundary_temperature()

            dz += db_dt_local * substep_dt

            remaining_time -= substep_dt
            self._num_substeps += 1

        # Net interface velocity from flux balance plus geometric melt diffusion, averaged over subtimesteps
        self._dzb_dt = dz / self.dt

    def run_step(self):
        """Run a single time step of the model."""
        
        eps = 1e-4 # small value to prevent thickness from going to zero

        # Two options: steady state hydrology or dynamic. 
        # Updates water table, water fluxes, and calculates Qdiss
        if self.use_steady_hydrology:
            self.run_hydrology_steady()
        else:
            self.run_hydrology_dynamic()

        self.run_heat_transport()  # updates T_mean, calculates dzb_dt

        # Apply Stefan condition to update permafrost table. Closed nodes are excluded
        # explicitly here as well.
        active = ~self._closed_nodes
        self._zb[active] = self._zb[active] - self._dzb_dt[active] * self.dt # note this also updates the boundary condition for the groundwater model, which is important for the feedback to work
        self._zb[self._zb >= self._z] = self._z[self._zb >= self._z] - eps # make sure refreezing doesn't cause total freezing above the land surface
        

        # Update derived geometric quantities
        self._h[:] = self._zwt - self._zb  # aquifer thickness
        self._b[:] = self._z - self._zb    # active layer thickness

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
        minh = np.min(self._h[self._grid.core_nodes])
        maxh = np.max(self._h[self._grid.core_nodes])
        imshow_grid(self._grid, 'aquifer__thickness', cmap='viridis', colorbar_label='Aquifer Thickness (m)', vmin=minh, vmax=maxh)

        plt.subplot(1, 3, 2)
        minQ = np.min(self._Qdiss[self._grid.core_nodes])
        maxQ = np.max(self._Qdiss[self._grid.core_nodes])
        imshow_grid(self._grid, self._Qdiss, cmap='inferno', colorbar_label='Dissipation (W/m^2)', vmin=minQ, vmax=maxQ)

        plt.subplot(1, 3, 3)
        minb = np.min(self._b[self._grid.core_nodes])
        maxb = np.max(self._b[self._grid.core_nodes])
        imshow_grid(self._grid, self._b, cmap='plasma', colorbar_label='Active Layer Thickness (m)', vmin=minb, vmax=maxb)
        plt.tight_layout()
        plt.show()


        # final timestep map view
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        minT = np.min(self._T_mean[self._grid.core_nodes])
        maxT = np.max(self._T_mean[self._grid.core_nodes])
        imshow_grid(self._grid, 'mean_unfrozen__temperature', cmap='Reds', colorbar_label='Mean Unfrozen Temperature (°C)', vmin=minT, vmax=maxT)

        plt.subplot(1, 3, 2)
        mindzb = np.min(self._dzb_dt[self._grid.core_nodes])
        maxdzb = np.max(self._dzb_dt[self._grid.core_nodes])
        imshow_grid(self._grid, self._dzb_dt, cmap='inferno', colorbar_label='Interface Velocity (m/s)', vmin=mindzb, vmax=maxdzb)

        plt.subplot(1, 3, 3)
        minb = np.min(self._b[self._grid.core_nodes])
        maxb = np.max(self._b[self._grid.core_nodes])
        imshow_grid(self._grid, self._b, cmap='plasma', colorbar_label='Active Layer Thickness (m)', vmin=minb, vmax=maxb)
        plt.tight_layout()
        plt.show()


        # time evolution of cross slope variability
        plt.figure()
        plt.plot(self.t, self.xslope_var)
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Std Dev of zb in Cross Slope Direction')

