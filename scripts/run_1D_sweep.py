"""
Parallel 1D sweep: depth-integrated thermal model vs. the modified Stefan
solution (Kurylyk and Hayashi, 2016), across air temperatures and frozen
gradients.

Every run is a single-column (3x3 closed grid, centre node) thaw problem with
zero initial unfrozen temperature, a Dirichlet surface temperature T_air, and
lateral fluxes switched off (see 1D_testing.py, which this generalises). The
analytical reference is the T_i = 0 solution, which strictly applies only for
frozen_gradient = 0. For non-zero frozen gradients it is kept as the reference
anyway, so the figures show how far the model departs from it.

Produces, for each frozen gradient:
  - active layer thickness vs. time (shifted by t0 to correct for the
    initial thickness), model and analytical, one colour per T_air
  - sensible heat content and mean temperature vs. time, one colour per T_air
and, for each non-zero gradient, a divergence row/figure:
  - ALT: (model - analytical) / analytical
  - E and T_mean: (model - model at zero gradient) / (model at zero gradient),
    since no analytical E / T_mean is defined here

Underlying data is saved as one .npz per run plus a combined sweep_results.npz.
The time series are saved at log-spaced time steps (N_SAVE points) to keep the
files small; the figures are all log-time.

Usage:  python scripts/run_1D_sweep.py [--days 180] [--output DIR]
"""

#%%

import matplotlib
matplotlib.use('Agg')  # before pyplot is imported anywhere, in this and worker processes

import os
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from landlab import RasterModelGrid
from model.water_track_model import WaterTrackModelThermal

#%%
# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

OUTPUT_ROOT = '/Users/tuv05476/Documents/Research Data/Local/water-tracks/1D_sweep/'

T_AIR_VALUES = [1, 2, 5, 10, 15]          # C
FROZEN_GRADIENTS = [0, 10, 20]            # K/m; 0 is the analytical-validity case
N_SAVE = 2000                             # log-spaced samples saved per run
B0 = 1e-2                                 # m, initial permeable thickness
CENTER_NODE = 4                           # centre of the 3x3 grid

BASE_PARAMS = {
    'S0': 0,
    'dt': 10,                             # seconds
    'T': 180 * 24 * 3600,                 # seconds
    'surface_boundary_type': 'Dirichlet',
    'recharge_rate': 0.0,
    'hydraulic_conductivity': 1e-3,
}

#%%
# ------------------------------------------------------------------
# Analytical solution (Kurylyk and Hayashi, 2016)
# ------------------------------------------------------------------

def stefan_mod(lam, k_u, T_s, t, phi, rho_w, L):
    """Modified Stefan equation (Kurylyk and Hayashi, 2016)"""
    return lam * np.sqrt((2 * k_u * T_s * t) / (phi * rho_w * L))


def calc_st1(c_u, rho_u, rho_w, T_s, L, phi):
    """Stefan number for the thaw case"""
    return (c_u * rho_u * T_s) / (rho_w * L * phi)


def lamda_thaw_0(S_T1):
    """Thaw correction factor when T_i = 0 (Kurylyk and Hayashi, 2016, Eq. 23)"""
    return 1 - 0.16 * S_T1 + 0.038 * S_T1**2


class _GDP:
    pass


def set_null_gdp(wtm):
    """Replace the groundwater component with a dummy so all lateral fluxes are zero."""
    nn = wtm._grid.number_of_nodes
    nl = wtm._grid.number_of_links

    wtm.gdp = _GDP()
    wtm.gdp._q = np.zeros(nl)
    wtm.gdp._vel = np.zeros(nl)
    wtm.gdp.recharge = np.zeros(nn)
    wtm.gdp._qsavg = np.zeros(nn)
    wtm.gdp._hydr_grad = np.zeros(nl)
    wtm.gdp.run_with_adaptive_time_step_solver = lambda x: None


#%%
# ------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------

def _init_worker(tqdm_lock):
    tqdm.set_lock(tqdm_lock)


def _tqdm_position():
    identity = multiprocessing.current_process()._identity
    return identity[0] - 1 if identity else 0


def run_name(spec):
    return f"Tair{spec['T_air']:g}_G{spec['frozen_gradient']:g}"


def _run_one(spec):
    """Run one 1D model and compare to the analytical solution. Module-level
    so it can be pickled for the worker pool."""
    boundaries = {"top": "closed", "left": "closed", "bottom": "closed", "right": "closed"}
    mg = RasterModelGrid((3, 3), xy_spacing=1, bc=boundaries)
    z = mg.add_zeros('topographic__elevation', at='node')
    zb = mg.add_zeros('aquifer_base__elevation', at='node')
    mg.add_zeros('water_table__elevation', at='node')
    mg.add_zeros('mean_unfrozen__temperature', at='node')  # starts at zero
    zb[:] = -B0

    params = dict(BASE_PARAMS)
    params['T'] = spec['T']
    params['T_air'] = float(spec['T_air'])
    params['frozen_gradient'] = float(spec['frozen_gradient'])

    wtm = WaterTrackModelThermal(mg, params)
    set_null_gdp(wtm)

    Nt = int(params['T'] / params['dt'])
    b = np.zeros(Nt)
    E = np.zeros(Nt)
    T_mean = np.zeros(Nt)
    T_mean_field = mg.at_node['mean_unfrozen__temperature']
    for i in tqdm(range(Nt), position=_tqdm_position(), desc=run_name(spec), leave=False):
        wtm.run_step()
        b[i] = z[CENTER_NODE] - zb[CENTER_NODE]
        T_mean[i] = T_mean_field[CENTER_NODE]
        E[i] = wtm._E[CENTER_NODE]

    t = np.arange(Nt) * params['dt']

    # analytical solution, shifted by t0 so it starts from the initial thickness B0
    St1 = calc_st1(wtm.rhoC_u, 1, wtm.rho_w, wtm.T_air, wtm.L, wtm.phi)
    lam = lamda_thaw_0(St1)
    L_v = wtm.phi * wtm.rho_w * wtm.L
    t0 = B0**2 * L_v / (2.0 * wtm.k_u * wtm.T_air * lam**2)
    t_shifted = t + t0
    b_pred = stefan_mod(lam, wtm.k_u, wtm.T_air, t_shifted, wtm.phi, wtm.rho_w, wtm.L)

    # log-spaced subsample (skip t=0 sample's role: index 0 is included via unique)
    idx = np.unique(np.geomspace(1, Nt, N_SAVE).astype(int) - 1)

    result = {
        'T_air': spec['T_air'],
        'frozen_gradient': spec['frozen_gradient'],
        't': t[idx],
        't_shifted': t_shifted[idx],
        'b': b[idx],
        'b_pred': b_pred[idx],
        'E': E[idx],
        'T_mean': T_mean[idx],
        'St1': St1,
        'lam': lam,
        't0': t0,
    }

    os.makedirs(spec['output_root'], exist_ok=True)
    np.savez(os.path.join(spec['output_root'], f"{run_name(spec)}.npz"), **result)
    return result


#%%
# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def _colors(n):
    return plt.cm.viridis(np.linspace(0.05, 0.9, n))


def _rel_diff(a, ref):
    return (a - ref) / ref


def plot_alt(results_G, results_G0, gradient, save_path):
    """Active layer thickness vs. shifted time, plus divergence from the
    analytical solution for non-zero gradients."""
    ncols = 2 if gradient != 0 else 1
    fig, axs = plt.subplots(ncols=ncols, figsize=(5 * ncols + 1, 4), squeeze=False)
    axs = axs[0]
    cols = _colors(len(results_G))
    for c, r in zip(cols, results_G):
        axs[0].plot(r['t_shifted'], r['b'], color=c, label=f"{r['T_air']:g} C")
        axs[0].plot(r['t_shifted'], r['b_pred'], color=c, linestyle='--')
    axs[0].set_xscale('log'); axs[0].set_yscale('log')
    axs[0].set_xlabel('Time + t0 (s)')
    axs[0].set_ylabel('Active Layer Thickness (m)')
    axs[0].set_title(f'Frozen gradient {gradient:g} K/m\n(solid: model, dashed: Thaw 0, K&H 2016 #23)', fontsize=9)
    axs[0].legend(title='T_air')
    if gradient != 0:
        for c, r in zip(cols, results_G):
            axs[1].plot(r['t_shifted'], _rel_diff(r['b'], r['b_pred']), color=c, label=f"{r['T_air']:g} C")
        axs[1].axhline(0, color='gray', linewidth=0.8)
        axs[1].set_xscale('log')
        axs[1].set_xlabel('Time + t0 (s)')
        axs[1].set_ylabel('(model - analytical) / analytical')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_E_T(results_G, results_G0, gradient, save_path):
    """Sensible heat content and mean temperature vs. shifted time, plus
    divergence from the zero-gradient model run for non-zero gradients."""
    nrows = 2 if gradient != 0 else 1
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(9, 4 * nrows), squeeze=False)
    cols = _colors(len(results_G))
    for c, r in zip(cols, results_G):
        axs[0, 0].plot(r['t_shifted'], r['E'], color=c, label=f"{r['T_air']:g} C")
        axs[0, 1].plot(r['t_shifted'], r['T_mean'], color=c, label=f"{r['T_air']:g} C")
    for ax, ylabel in zip(axs[0], ['Sensible heat content (J/m2)', 'Mean Temperature (C)']):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Time + t0 (s)')
        ax.set_ylabel(ylabel)
    axs[0, 0].set_title(f'Frozen gradient {gradient:g} K/m', fontsize=9)
    axs[0, 0].legend(title='T_air')

    if gradient != 0:
        ref_by_T = {r['T_air']: r for r in results_G0}
        for c, r in zip(cols, results_G):
            ref = ref_by_T[r['T_air']]
            axs[1, 0].plot(r['t_shifted'], _rel_diff(r['E'], ref['E']), color=c, label=f"{r['T_air']:g} C")
            axs[1, 1].plot(r['t_shifted'], _rel_diff(r['T_mean'], ref['T_mean']), color=c, label=f"{r['T_air']:g} C")
        for ax, ylabel in zip(axs[1], ['(E - E_G0) / E_G0', '(T_mean - T_mean_G0) / T_mean_G0']):
            ax.axhline(0, color='gray', linewidth=0.8)
            ax.set_xscale('log')
            ax.set_xlabel('Time + t0 (s)')
            ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


#%%
# ------------------------------------------------------------------
# Run the sweep
# ------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--days', type=float, default=BASE_PARAMS['T'] / 86400, help='simulated duration in days')
    parser.add_argument('--output', default=OUTPUT_ROOT, help='output directory')
    args = parser.parse_args()

    out = args.output
    os.makedirs(out, exist_ok=True)

    specs = [
        {'T_air': Ta, 'frozen_gradient': G, 'T': args.days * 86400, 'output_root': os.path.join(out, 'runs')}
        for G in FROZEN_GRADIENTS for Ta in T_AIR_VALUES
    ]
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
            r = future.result()
            results.append(r)
            print(f"Finished T_air={r['T_air']:g}, G={r['frozen_gradient']:g}")

    # stable ordering for plots and the combined file
    results.sort(key=lambda r: (r['frozen_gradient'], r['T_air']))
    by_G = {G: [r for r in results if r['frozen_gradient'] == G] for G in FROZEN_GRADIENTS}
    G0 = by_G.get(0, [])

    # combined data file: arrays stacked as [gradient, T_air, time]
    np.savez(
        os.path.join(out, 'sweep_results.npz'),
        T_air=np.array(T_AIR_VALUES),
        frozen_gradient=np.array(FROZEN_GRADIENTS),
        **{key: np.array([[r[key] for r in by_G[G]] for G in FROZEN_GRADIENTS])
           for key in ['t', 't_shifted', 'b', 'b_pred', 'E', 'T_mean', 'St1', 'lam', 't0']},
    )

    for G in FROZEN_GRADIENTS:
        plot_alt(by_G[G], G0, G, os.path.join(out, f'alt_G{G:g}.png'))
        plot_E_T(by_G[G], G0, G, os.path.join(out, f'E_T_G{G:g}.png'))
    print(f'Saved figures and data to {out}')
