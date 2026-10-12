"""
Diffusive behaviour of the two "smoothing" mechanisms in WaterTrackModelThermal,
isolated from everything else (flux_unfrozen, Q_diss, and the constant part of
the frozen flux are all zero).

1. use_melt_diffusion  -- directly alters b:
       db/dt = div[ grad(b) * (dzb_dt * b) / (1 + |grad b|^2) ]
   For small slopes and a uniform melt rate m = dzb_dt this is the linear
   diffusion equation with D = m * b0.
   NOTE: in the model dzb_dt is the *previous* step's db_dt. With every other
   term zero it would be identically zero, so here a background melt rate m is
   prescribed (uniform, constant) -- the diffusion coefficient scales with it.

2. use_fourier_frozen_gradient -- acts on b only through the frozen flux:
       flux_frozen = kf * G * (1 + F^-1[ |k| F[b] ])
       db/dt       = -flux_frozen / (rho_w phi L)
   Dropping the constant kf*G term (the "flux frozen" term set to zero), what
   is left is db/dt = -c F^-1[ |k| F[b] ],  c = kf G / (rho_w phi L)  [m/s].
   In Fourier space this decays each mode at rate c|k| (|k| in cycles/m, as in
   the model's _make_fft_matrix), i.e. a *fractional* (half-Laplacian)
   diffusion, not a Fickian one. Point-source solution is the 2D Poisson
   kernel, with s = c t / (2 pi)  (|k| is in cycles/m):
       b(r,t) = M s / (2 pi (s^2 + r^2)^(3/2))
   which has heavy tails (infinite variance) and a spreading width ~ t, not
   t^(1/2). So there is no single diffusivity; D_eff depends on scale/time.

Tests (grid spacing: A uses dx = 0.25 m, B uses 1 m)
   A. Point source: radial profile vs. Gaussian (linear diffusion) / Poisson
      kernel, half-mass radius r50 vs. time (Fickian: r50^2 ~ t; Cauchy: r50 ~ t),
      and effective diffusivity D_eff = r50^2 / (4 ln2 t).
   B. Single sinusoidal mode: decay rate vs. wavelength. Fickian: rate ~ k^2
      (D_eff constant). Fourier-frozen: rate ~ k, so D_eff(lambda) = c lambda /(4 pi^2).

Run end-to-end; saves figure(s) to scripts/output_diffusion_test/.
"""

#%%
import os
import numpy as np
import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab.grid.raster_mappers import map_link_vector_components_to_node_raster
from landlab.grid.mappers import map_mean_of_link_nodes_to_link

#%%
# ---------------------------------------------------------------- parameters
# (physical values as in WaterTrackModelBase defaults)
kf = 2.728          # W/m/K
G = 10.0            # K/m, frozen gradient
rho_w = 1000.0
phi = 0.9
L = 334e3
c_fourier = kf * G / (rho_w * phi * L)   # m/s, rate coefficient of |k| operator

b0 = 0.5            # m, background thickness
m_melt = 3e-8       # m/s, prescribed uniform melt rate for the melt-diffusion model
amp = 1e-3 * b0     # m, point-source height (small => |grad b| << 1, linear regime)

dx = 0.25
N = 513             # odd, so the source sits on a node at the centre
dt = 6 * 3600.0     # s
T = 5 * 365 * 24 * 3600.0
n_steps = int(T / dt)

outdir = os.path.join('/Users/tuv05476/Documents/Research/water_tracks_evolution', "output_diffusion_test")
os.makedirs(outdir, exist_ok=True)


#%%
# ------------------------------------------------------- the two update rules
def make_fft_matrix(shape, dx):
    """Same as WaterTrackModelThermal._make_fft_matrix: |k| in cycles/m."""
    Ny, Nx = shape
    kx, ky = np.meshgrid(np.fft.fftfreq(Nx, d=dx), np.fft.fftfreq(Ny, d=dx))
    return np.sqrt(kx**2 + ky**2)


def make_grid(N, dx):
    mg = RasterModelGrid((N, N), xy_spacing=dx)
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    return mg


class FourierFrozen:
    """db/dt from the frozen-gradient Fourier term only (constant kf*G part removed)."""
    name = "fourier frozen gradient"

    def __init__(self, shape, dx):
        self.shape = shape
        self.A = make_fft_matrix(shape, dx)

    def db_dt(self, b):
        # copied from run_heat_transport: coef = ifft2(A * fft2(b))
        coef = np.real(np.fft.ifft2(self.A * np.fft.fft2(b.reshape(self.shape)))).flatten()
        flux_frozen = kf * G * coef               # (kf*G*1 term dropped)
        return -flux_frozen / (rho_w * phi * L)


class MeltDiffusion:
    """db/dt from the melt-diffusion term only, with prescribed uniform dzb_dt = m."""
    name = "melt diffusion"

    def __init__(self, grid, m):
        self.grid = grid
        self.m = m
        self.closed = grid.status_at_node == grid.BC_NODE_IS_CLOSED

    def db_dt(self, b):
        g = self.grid
        dzb_dt = np.full_like(b, self.m)
        # copied from run_heat_transport
        gradb = g.calc_grad_at_link(b)
        bprod = map_mean_of_link_nodes_to_link(g, dzb_dt * b)
        gx, gy = map_link_vector_components_to_node_raster(g, gradb)
        gsq_link = map_mean_of_link_nodes_to_link(g, gx**2 + gy**2)
        out = g.calc_flux_div_at_node((gradb * bprod) / (1 + gsq_link))
        out[self.closed] = 0.0
        return out


def run(model, b_init, dt, n_steps, save_every=1):
    """Forward Euler on b (as in the model, substeps with db_dt_local)."""
    b = b_init.copy().ravel()
    snaps = {0: b.copy()}
    for i in range(1, n_steps + 1):
        b += model.db_dt(b) * dt
        if i % save_every == 0:
            snaps[i] = b.copy()
    return snaps


#%%
# ------------------------------------------------------ A. point source test
def radial_stats(pert, X, Y, xc, yc):
    """Return half-mass radius, second-moment-based radius, and radial profile."""
    r = np.hypot(X - xc, Y - yc).ravel()
    w = pert.ravel()
    M = w.sum()
    order = np.argsort(r)
    cum = np.cumsum(w[order]) / M
    r50 = r[order][np.searchsorted(cum, 0.5)]
    msd = np.sum(w * r**2) / M
    return M, r50, msd


mg = make_grid(N, dx)
X = mg.x_of_node.reshape(N, N)
Y = mg.y_of_node.reshape(N, N)
xc = yc = (N - 1) * dx / 2
ic = (N - 1) // 2

b_init = np.full((N, N), b0)
b_init[ic, ic] += amp                   # single-cell bump of height amp
M0 = amp * dx**2                         # volume of the perturbation

models = [FourierFrozen((N, N), dx), MeltDiffusion(mg, m_melt)]
save_every = 50
results = {}
for mod in models:
    snaps = run(mod, b_init, dt, n_steps, save_every)
    results[mod.name] = snaps
    print(f"{mod.name}: done, volume drift = "
          f"{(snaps[n_steps] - b0).sum() * dx**2 / M0 - 1:.2e}")

D_melt_pred = m_melt * b0
print(f"\nmelt diffusion: predicted D = m*b0 = {D_melt_pred:.3e} m2/s")
print(f"fourier frozen: c = kf G/(rho phi L) = {c_fourier:.3e} m/s")

#%%
# time series of r50 and D_eff
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
steps = sorted(results["melt diffusion"].keys())[1:]
tt = np.array(steps) * dt
colors = {"fourier frozen gradient": "C3", "melt diffusion": "C0"}
r50s = {}
for name, snaps in results.items():
    r50 = np.array([radial_stats(snaps[s] - b0, X, Y, xc, yc)[1] for s in steps])
    r50s[name] = r50
    Deff = r50**2 / (4 * np.log(2) * tt)
    axs[0].loglog(tt / 86400, r50, color=colors[name], label=name)
    axs[1].semilogx(tt / 86400, Deff, color=colors[name], label=name)

# theory
axs[0].loglog(tt / 86400, np.sqrt(4 * np.log(2) * D_melt_pred * tt), "C0--", label=r"Fickian $\sqrt{4\ln2\,D t}$, $D=m b_0$")
axs[0].loglog(tt / 86400, np.sqrt(3) * c_fourier * tt / (2 * np.pi), "C3--", label=r"Poisson kernel $\sqrt{3}\,c t/2\pi$")
axs[1].axhline(D_melt_pred, color="C0", ls="--", label="$m b_0$")
axs[0].set(xlabel="time (days)", ylabel="half-mass radius r50 (m)")
axs[1].set(xlabel="time (days)", ylabel=r"$D_{eff}=r_{50}^2/(4\ln2\,t)$ (m$^2$/s)")
axs[0].legend(fontsize=7); axs[1].legend(fontsize=7)
axs[0].set_title("A. point source: spread"); axs[1].set_title("A. effective diffusivity")

# final radial profiles vs theory
tfin = n_steps * dt
r_line = np.linspace(0, 12, 300)
rr = np.hypot(X - xc, Y - yc).ravel()
for name, snaps in results.items():
    p = (snaps[n_steps] - b0).ravel() / M0
    axs[2].semilogy(rr, p, ".", ms=2, color=colors[name], alpha=0.4, label=name)
axs[2].semilogy(r_line, np.exp(-r_line**2 / (4 * D_melt_pred * tfin)) / (4 * np.pi * D_melt_pred * tfin),
                "C0-", lw=1, label="Gaussian, D=m b0")
s = c_fourier * tfin / (2 * np.pi)
axs[2].semilogy(r_line, s / (2 * np.pi * (s**2 + r_line**2) ** 1.5), "C3-", lw=1, label="Poisson kernel")
axs[2].set(xlabel="r (m)", ylabel="perturbation / M0 (1/m$^2$)", xlim=(0, 12), ylim=(1e-5, None),
           title=f"A. profile at t = {tfin/86400:.0f} d")
axs[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "point_source.png"), dpi=150)

# scaling exponents of r50 vs t (late times)
for name, r50 in r50s.items():
    sel = tt > 0.3 * tt[-1]
    slope = np.polyfit(np.log(tt[sel]), np.log(r50[sel]), 1)[0]
    print(f"{name}: r50 ~ t^{slope:.2f}  (Fickian = 0.50, Poisson/fractional = 1.00); "
          f"D_eff(t_end) = {r50[-1]**2/(4*np.log(2)*tt[-1]):.3e} m2/s")

#%%
# --------------------------------------------- B. single mode decay vs wavelength
lams = np.array([16, 24, 32, 48, 64, 96, 128, 256]) * 1.0
dxm = 1.0
Nm = 257  # grid for mode test (periodic in x for FFT; landlab edges closed so
          # we use cos modes in x with matching wavelength, keep a few away from edges)
rates = {"fourier frozen gradient": [], "melt diffusion": []}
mgm = make_grid(Nm, dxm)
Xm = mgm.x_of_node.reshape(Nm, Nm)
dtm = 3600.0 * 3
for lam in lams:
    b_sin = b0 + amp * np.cos(2 * np.pi * (Xm - Xm.min()) / lam)
    for mod in (FourierFrozen((Nm, Nm), dxm), MeltDiffusion(mgm, m_melt)):
        # estimate rate from the instantaneous db_dt on the central row, projected on the mode
        # (initial shape is an exact eigenmode up to numerical dispersion / boundary rows)
        d = mod.db_dt(b_sin.ravel()).reshape(Nm, Nm)
        row = Nm // 2
        pert = (b_sin - b0)[row, 8:-8]
        rate = -np.sum(d[row, 8:-8] * pert) / np.sum(pert**2)  # 1/s
        rates[mod.name].append(rate)

fig2, axs2 = plt.subplots(1, 2, figsize=(11, 4.5))
k_ang = 2 * np.pi / lams
for name, col in colors.items():
    r = np.array(rates[name])
    axs2[0].loglog(lams, r, "o-", color=col, label=name)
    axs2[1].semilogx(lams, r / k_ang**2, "o-", color=col, label=name)
axs2[0].loglog(lams, 2 * np.pi * c_fourier / lams, "C3--", label=r"$c\,|k|$ (cycles) = $c/\lambda$")
axs2[0].loglog(lams, D_melt_pred * k_ang**2, "C0--", label=r"$D k^2$, $D=m b_0$")
axs2[1].plot(lams, c_fourier * lams / (4 * np.pi**2), "C3--", label=r"$c\lambda/4\pi^2$")
axs2[1].axhline(D_melt_pred, color="C0", ls="--", label="$m b_0$")
axs2[0].set(xlabel="wavelength (m)", ylabel="decay rate (1/s)", title="B. mode decay rate")
axs2[1].set(xlabel="wavelength (m)", ylabel=r"rate/$k^2$ = $D_{eff}$ (m$^2$/s)", title="B. effective diffusivity vs scale")
axs2[0].legend(fontsize=7); axs2[1].legend(fontsize=7)
fig2.tight_layout()
fig2.savefig(os.path.join(outdir, "mode_decay.png"), dpi=150)

print("\nlambda (m) | D_eff fourier (m2/s) | D_eff melt (m2/s)")
for lam, rf, rm, k in zip(lams, rates["fourier frozen gradient"], rates["melt diffusion"], k_ang):
    print(f"{lam:8.0f}   | {rf/k**2:.3e}           | {rm/k**2:.3e}")
for name in rates:
    sl = np.polyfit(np.log(lams), np.log(rates[name]), 1)[0]
    print(f"{name}: rate ~ lambda^{sl:.2f}  (Fickian = -2, fractional |k| = -1)")

# wavelength at which the two mechanisms have equal effective diffusivity
lam_eq = 4 * np.pi**2 * D_melt_pred / c_fourier
print(f"\nD_eff equal at lambda = 4 pi^2 m b0 / c = {lam_eq:.1f} m "
      f"(fourier term dominates smoothing at longer wavelengths)")
plt.show()

# %%
