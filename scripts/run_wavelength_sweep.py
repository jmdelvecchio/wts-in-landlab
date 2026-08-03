"""
Parallel model-comparison sweep.

Runs WaterTrackModelBasic and WaterTrackModelThermal, each with melt
diffusion on and off (4 structural configurations), across a set of initial
sinusoidal cross-slope perturbation wavelengths expressed as multiples of the
linearly-predicted wavelength (calc_wavelenth_1). All runs share the same
physical parameters and grid geometry -- the only things varied are model
structure and initial wavelength -- so the resulting growth-rate-vs-wavelength
comparison isolates the effect of model structure on water track growth.

Each run gets its own output folder (netCDF output + make_plots figures,
named after its configuration and wavelength multiple). At the end, a single
summary figure plots the empirical growth rate of the cross-slope-variability
metric (xslope_var) against wavelength multiple, one curve per configuration.

Unlike the other run_*.py scripts in this folder, this one is meant to be run
end-to-end (not stepped through interactively cell by cell), since figures
are saved to disk rather than shown, and the heavy lifting happens in
parallel worker processes.
"""

#%%

import matplotlib
matplotlib.use('Agg')  # must happen before pyplot is imported anywhere (including via model.water_track_model),
                        # in both this process and any worker process spawned from it

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from landlab import RasterModelGrid
from water_track_funcs import calc_wavelenth_1, notional_equilibrium_temperature
from model.water_track_model import WaterTrackModelBasic, WaterTrackModelThermal

#%%
# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

OUTPUT_ROOT = '/Users/tuv05476/Documents/Research Data/Local/water-tracks/wavelength_sweep/'
OUTPUT_INTERVAL = 20
OUTPUT_FIELDS_BASIC = [
    "at_node:aquifer_base__elevation",
    "at_node:water_table__elevation",
]
OUTPUT_FIELDS_THERMAL = OUTPUT_FIELDS_BASIC + ["at_node:mean_unfrozen__temperature"]

# grid geometry -- parabolic hillslope, uniform initial permeable thickness,
# matching the pattern used in the other run_*.py scripts
GEOMETRY = {
    'boundaries': {"top": "open", "left": "closed", "bottom": "closed", "right": "closed"},
    'Nx': 100,
    'Ny': 150,
    'dx': 10,
    'a': 0.0001,      # parabola curvature
    'b': 0.5,         # initial permeable thickness, m
    'y_offset': 2000,
}
AMPLITUDE = 0.01  # initial sinusoidal perturbation amplitude, m

# multiples of the linearly-predicted wavelength to sweep over
WAVELENGTH_MULTIPLES = [0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# the 4 structural configurations to compare
MODEL_CONFIGS = [
    (WaterTrackModelBasic, False, "basic_nomeltdiff"),
    (WaterTrackModelBasic, True, "basic_meltdiff"),
    (WaterTrackModelThermal, False, "thermal_nomeltdiff"),
    (WaterTrackModelThermal, True, "thermal_meltdiff"),
]

# single shared physical parameter set -- identical across every run and
# every model configuration. WaterTrackModelBasic ignores the thermal-only
# keys (rho_s, C_s, C_w, ku) but they're harmless to include, and keeping
# them here means both classes see exactly the same params dict.
BASE_PARAMS = {
    'recharge_rate': 1.0e-6,          # m/s
    'hydraulic_conductivity': 1e-1,   # m/s
    'porosity': 0.9,
    'S0': 30.0,                       # W/m^2, peak solar irradiance
    'frozen_gradient': 10.0,          # K/m
    'T_air': 1,
    'dt': 6 * 3600,                   # seconds
    'T': 180 * 24 * 3600,             # seconds, total simulation time
    'courant_coefficient': 0.5,
    'kf': 2.728,                      # W/m/K, frozen soil
    'ku': 1.2682,                     # W/m/K, unfrozen soil (thermal only)
    'Tm': 0,
    'L': 334e3,
    'beta': 0.4,
    'rho_w': 1000,
    'rho_s': 2600,                    # thermal only
    'C_s': 700,                       # thermal only
    'C_w': 4.2e3,                     # thermal only
    'use_steady_hydrology': False,
}


#%%
# ------------------------------------------------------------------
# Grid / initial condition construction
# ------------------------------------------------------------------

def build_base_fields(mg, geometry):
    """Add the (unperturbed) parabolic-hillslope elevation fields to mg."""
    a, b, y_offset = geometry['a'], geometry['b'], geometry['y_offset']

    z = mg.add_zeros('topographic__elevation', at='node')
    zb = mg.add_zeros('aquifer_base__elevation', at='node')
    zwt = mg.add_zeros('water_table__elevation', at='node')

    y = mg.y_of_node
    z[:] = -a * y**2 + a * y_offset**2
    zb[:] = z - b
    zwt[:] = zb + 0.1 * b

    return z, zb, zwt


def build_grid_and_ic(wavelength_target, needs_tmean, geometry, amplitude, t_eq):
    """Build a fresh grid with a sinusoidal cross-slope perturbation to zb
    at the given wavelength. IC type is hardcoded to sinusoidal for now;
    a 'random field' option can be added here later without touching the
    sweep/runner logic.

    T_mean starts at t_eq (the notional 0-D equilibrium temperature) rather
    than 0, so the thermal model doesn't spend the run's early portion on an
    unrelated spin-up transient (T_mean racing from 0 toward equilibrium)
    that would otherwise dominate the growth-rate fit and mask the water-
    track signal.

    geometry, amplitude, and t_eq are passed in explicitly (rather than
    defaulted to module-level globals) so callers always use current values
    instead of whatever was bound when this function was defined.
    """
    mg = RasterModelGrid((geometry['Ny'], geometry['Nx']), xy_spacing=geometry['dx'], bc=geometry['boundaries'])
    z, zb, zwt = build_base_fields(mg, geometry)
    if needs_tmean:
        tmean = mg.add_zeros('mean_unfrozen__temperature', at='node')
        tmean[:] = t_eq

    kappa = 2 * np.pi / wavelength_target
    fluct = amplitude * np.sin(kappa * mg.x_of_node)
    zb[:] = zb + fluct
    zwt[:] = zb + 0.1 * geometry['b']

    return mg


#%%
# ------------------------------------------------------------------
# Predicted wavelength (computed once, from the fixed geometry/params)
# ------------------------------------------------------------------

_ref_grid = RasterModelGrid((GEOMETRY['Ny'], GEOMETRY['Nx']), xy_spacing=GEOMETRY['dx'], bc=GEOMETRY['boundaries'])
_z_ref, _, _ = build_base_fields(_ref_grid, GEOMETRY)

_slope = np.arctan(np.mean(np.abs(np.gradient(_z_ref.reshape(_ref_grid.shape), GEOMETRY['dx'], axis=0))))
_slope_deg = np.rad2deg(_slope)

BASE_WAVELENGTH = calc_wavelenth_1(
    max(_ref_grid.y_of_node),
    BASE_PARAMS['kf'],
    BASE_PARAMS['frozen_gradient'],
    BASE_PARAMS['rho_w'],
    _slope_deg,
    BASE_PARAMS['hydraulic_conductivity'],
)
print(f'Predicted wavelength: {BASE_WAVELENGTH:.1f} m (slope {_slope_deg:.2f} deg)')

BASE_T_EQ = notional_equilibrium_temperature(
    BASE_PARAMS['S0'], BASE_PARAMS['beta'], BASE_PARAMS['T_air'],
    BASE_PARAMS['kf'], BASE_PARAMS['frozen_gradient'],
)
print(f'Notional equilibrium T_mean: {BASE_T_EQ:.1f} C')


#%%
# ------------------------------------------------------------------
# Run specs
# ------------------------------------------------------------------

def build_run_specs(model_configs, wavelength_multiples, base_wavelength):
    specs = []
    for model_class, use_melt_diffusion, label in model_configs:
        for multiple in wavelength_multiples:
            specs.append({
                'model_class': model_class,
                'use_melt_diffusion': use_melt_diffusion,
                'label': label,
                'wavelength_multiple': multiple,
                'wavelength_target': multiple * base_wavelength,
            })
    return specs


#%%
# ------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------

def _init_worker(tqdm_lock):
    """ProcessPoolExecutor initializer: share one lock across all worker
    processes so their tqdm bars don't garble each other's terminal writes
    when two bars happen to redraw at the same instant.
    """
    tqdm.set_lock(tqdm_lock)


def _tqdm_position():
    """A small, stable integer identifying which pool worker this process
    is, so its tqdm bar always redraws to the same terminal line across the
    many specs that worker processes over its lifetime. multiprocessing
    assigns each worker process a unique, unchanging identity tuple at
    creation time; this is the standard trick for giving pool workers their
    own tqdm row (see tqdm's multiprocessing examples).
    """
    identity = multiprocessing.current_process()._identity
    return identity[0] - 1 if identity else 0


def _run_one(spec):
    """Build, run, and save a single model instance. Module-level so it can
    be pickled and sent to a worker process.
    """
    needs_tmean = spec['model_class'] is WaterTrackModelThermal
    mg = build_grid_and_ic(spec['wavelength_target'], needs_tmean, GEOMETRY, AMPLITUDE, BASE_T_EQ)

    params = dict(BASE_PARAMS)
    params['use_melt_diffusion'] = spec['use_melt_diffusion']

    run_name = f"{spec['label']}_wl{spec['wavelength_multiple']:.2f}"
    run_dir = os.path.join(OUTPUT_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)

    output_dict = {
        'output_interval': OUTPUT_INTERVAL,
        'output_fields': OUTPUT_FIELDS_THERMAL if needs_tmean else OUTPUT_FIELDS_BASIC,
        'base_output_path': os.path.join(run_dir, 'output_'),
        'run_id': 0,
    }

    mdl = spec['model_class'](mg, params, output_dict=output_dict)
    mdl.run_model(tqdm_position=_tqdm_position(), tqdm_desc=run_name)
    mdl.make_plots(save_path=run_dir)

    # save t/xslope_var for later fitting (e.g. linear-regime growth rate)
    # without needing to re-run the model
    np.savetxt(
        os.path.join(run_dir, 'xslope_var.csv'),
        np.column_stack([mdl.t, mdl.xslope_var]),
        delimiter=',',
        header='t,xslope_var',
        comments='',
    )

    return {
        'label': spec['label'],
        'wavelength_multiple': spec['wavelength_multiple'],
        't': mdl.t,
        'xslope_var': mdl.xslope_var,
    }


def fit_growth_rate(t, xslope_var):
    """Empirical exponential growth rate [1/s] of xslope_var: the slope of
    a linear fit to ln(xslope_var) vs. t.
    """
    slope, _ = np.polyfit(t, np.log(xslope_var), 1)
    return slope


def fit_linear_growth_rate(t, xslope_var):
    """Empirical linear growth rate [m/s] of xslope_var: the slope of a
    linear fit to xslope_var vs. t directly (as opposed to fit_growth_rate's
    exponential fit). Use this when xslope_var(t) looks linear rather than
    exponential past the early transient.
    """
    slope, _ = np.polyfit(t, xslope_var, 1)
    return slope


#%%
# ------------------------------------------------------------------
# Run the sweep and build the summary plot
# ------------------------------------------------------------------

if __name__ == '__main__':

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    specs = build_run_specs(MODEL_CONFIGS, WAVELENGTH_MULTIPLES, BASE_WAVELENGTH)
    print(f'Running {len(specs)} model instances...')

    results = []
    n_workers = min(len(specs), os.cpu_count() or 1)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(multiprocessing.RLock(),),
    ) as executor:
        futures = {executor.submit(_run_one, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            result = future.result()
            results.append(result)
            print(f"Finished {result['label']} (wavelength x{result['wavelength_multiple']:.2f})")

    def _plot_growth_rate_summary(fit_fn, seconds_per_year_scale, ylabel, filename):
        """Fit fit_fn to each run's (t, xslope_var), plot the resulting rate
        vs. wavelength multiple (one curve per configuration), and save.
        """
        growth_by_label = {label: ([], []) for _, _, label in MODEL_CONFIGS}
        for r in results:
            rate_per_year = fit_fn(r['t'], r['xslope_var']) * seconds_per_year_scale
            growth_by_label[r['label']][0].append(r['wavelength_multiple'])
            growth_by_label[r['label']][1].append(rate_per_year)

        plt.figure(figsize=(8, 6))
        for label, (multiples, rates) in growth_by_label.items():
            order = np.argsort(multiples)
            multiples_sorted = np.array(multiples)[order]
            rates_sorted = np.array(rates)[order]
            plt.plot(multiples_sorted, rates_sorted, marker='o', label=label)
        plt.axvline(1.0, color='gray', linestyle='--', linewidth=1, label='predicted wavelength')
        plt.xlabel(f'Wavelength / predicted ({BASE_WAVELENGTH:.1f} m)')
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        summary_path = os.path.join(OUTPUT_ROOT, filename)
        plt.savefig(summary_path)
        plt.close()
        print(f'Saved summary plot to {summary_path}')

    # ---- exponential growth rate vs. wavelength, one curve per configuration ----
    _plot_growth_rate_summary(
        fit_growth_rate,
        3600 * 24 * 365,
        'Growth rate of cross-slope std dev (1/year)',
        'growth_rate_vs_wavelength.png',
    )

    # ---- linear growth rate vs. wavelength, one curve per configuration ----
    _plot_growth_rate_summary(
        fit_linear_growth_rate,
        3600 * 24 * 365,
        'Growth rate of cross-slope std dev (m/year)',
        'growth_rate_vs_wavelength_linear.png',
    )
