"""
Unit tests for run_heat_transport()

Run with:  pytest test_thermal_model.py -v
"""

#%%
import numpy as np
import pytest
from landlab import RasterModelGrid
from model.water_track_model import WaterTrackModelThermal

#%%

class _Stub:
    """Empty object; attributes are set explicitly by make_model."""
    pass


def make_model(grid, dt=3600.0, **overrides):
    """
    Build a minimal stub with every attribute that run_heat_transport reads
    from self.  The method is borrowed directly from WaterTrackModelThermal so
    no implementation is duplicated.

    Defaults are Toolik-like values from Warburton et al. (2026).
    Pass keyword arguments to override individual attributes in a test.
    """
    m = _Stub()
    m._grid = grid

    nn = grid.number_of_nodes
    nl = grid.number_of_links

    # time
    m.dt = dt

    # thermal parameters
    m.k_u   = 1.2682   # W m-1 K-1  unfrozen bulk
    m.k_f   = 2.728    # W m-1 K-1  frozen bulk
    m.C_u   = 3835.6   # J kg-1 K-1 unfrozen bulk
    m.rho_u = 1136.0   # kg m-3     unfrozen bulk (phi*rho_w + (1-phi)*rho_s)
    m.rho_w = 1000.0   # kg m-3
    m.phi   = 0.9      # –
    m.L     = 334e3    # J kg-1
    m.Tm    = 0.0      # °C  melting point

    # surface forcing  (all off by default)
    m.S0              = 0.0   # W m-2
    m.beta            = 0.0   # W m-2 K-1
    m.T_air           = 0.0   # °C
    m.frozen_gradient = 0.0   # K m-1

    # numerics
    m._courant_coefficient = 0.5
    m.use_melt_diffusion   = False

    # node arrays  (neutral initial state)
    m._T_mean        = np.zeros(nn)
    m._E             = np.zeros(nn)
    m._b             = np.ones(nn)    # 1 m active layer
    m._zb            = np.zeros(nn)
    m._z             = np.ones(nn)
    m._zwt           = np.ones(nn)
    m._Qdiss         = np.zeros(nn)
    m._dzb_dt        = np.zeros(nn)
    m.melt_diffusion = np.zeros(nn)

    # minimal GDP mock: _q is used for heat flux in run_heat_transport,
    # _vel is used in courant condition.
    class _GDP:
        pass
    m.gdp     = _GDP()
    m.gdp._q  = np.zeros(nl)   # no lateral flow
    m.gdp._vel = np.zeros(nl)

    # boundary node bookkeeping, mirroring WaterTrackModelThermal.__init__ exactly so
    # closed/open node handling in run_heat_transport works on the stub too.
    m._closed_nodes = grid.status_at_node == grid.BC_NODE_IS_CLOSED
    m._open_nodes = grid.status_at_node == grid.BC_NODE_IS_FIXED_VALUE

    open_node_ids = np.where(m._open_nodes)[0]
    nbrs = grid.active_adjacent_nodes_at_node[open_node_ids]
    valid = nbrs >= 0
    nbrs_safe = np.where(valid, nbrs, 0)
    is_core_nbr = valid & (grid.status_at_node[nbrs_safe] == grid.BC_NODE_IS_CORE)

    m._open_node_ids = open_node_ids
    m._open_node_neighbors = nbrs_safe
    m._open_node_neighbor_mask = is_core_nbr
    m._open_node_has_core_neighbor = is_core_nbr.any(axis=1)
    m._update_open_boundary_temperature = (
        lambda: WaterTrackModelThermal._update_open_boundary_temperature(m)
    )

    # apply per-test overrides
    for key, val in overrides.items():
        setattr(m, key, val)

    return m


def _run(m):
    """Invoke the real run_heat_transport on the stub."""
    WaterTrackModelThermal.run_heat_transport(m)


# ============================================================
# Shared fixture
# ============================================================

@pytest.fixture
def flat_grid():
    """
    Small flat raster (7 x 7, 10 m spacing) with all boundaries closed.
    Closed boundaries mean no flux escapes the domain, so boundary nodes
    do not contaminate interior values.
    """
    mg = RasterModelGrid((7, 7), xy_spacing=10.0)
    mg.set_closed_boundaries_at_grid_edges(
        bottom_is_closed=True,
        left_is_closed=True,
        right_is_closed=True,
        top_is_closed=True,
    )
    return mg


# ============================================================
# Test 1 – source term heating
# ============================================================

def test_source_term_heating(flat_grid):
    """
    With only Qdiss active, T_mean should rise by

        ΔT = Q0 * dt / (C_u * rho_u * b)

    Terms zeroed and why:
    ┌──────────────────────────────────────────────────────────┐
    │  T_mean = Tm          →  flux_unfrozen = 0               │
    │  S0 = 0, beta = 0     →  BC_top = 0                      │
    │  frozen_gradient = 0  →  flux_frozen = 0 (Stefan = 0)    │
    │  q = 0, vel = 0       →  no advection; single substep    │
    │  uniform T_mean       →  lateral diffusion = 0           │
    └──────────────────────────────────────────────────────────┘
    """
    Q0 = 10.0     # W m-2
    b0 = 1.0      # m
    T0 = 0.0      # °C  == Tm  →  flux_unfrozen = 0
    dt = 3600.0   # s

    m = make_model(flat_grid, dt=dt, Tm=T0)
    m._T_mean[:] = T0
    m._E[:]      = m.C_u * m.rho_u * m._b * (m._T_mean - m.Tm)
    m._b[:]      = b0
    m._Qdiss[:]  = Q0

    _run(m)

    dT_expected = Q0 * dt / (m.C_u * m.rho_u * b0)

    np.testing.assert_allclose(
        m._T_mean[flat_grid.core_nodes],
        T0 + dT_expected,
        rtol=1e-6,
        err_msg="Source term heating: T_mean did not match Q0*dt/(C_u*rho_u*b)",
    )


# ============================================================
# Test 2 – Stefan condition (decoupled from temperature evolution)
# ============================================================

def test_stefan_condition_decoupled(flat_grid):
    """
    With prescribed uniform T_mean and frozen_gradient, dzb_dt should equal

        dzb_dt = (ku*(T_mean - Tm)/(b/2) - kf*frozen_gradient) / (rho_w*phi*L)

    dt = 1 s is chosen so T_mean changes by < 0.001 % during the step,
    keeping the initial and end-of-step flux values indistinguishable.

    Analytical check:
        flux_unfrozen = 1.2682 * 2.0 / 0.25  =  10.15 W m-2
        flux_frozen   = 2.728  * 20.0         =  54.56 W m-2
        dzb_dt        = (10.15 - 54.56) / (1000 * 0.9 * 334e3)
                      ≈ -1.48e-7 m s-1   (net refreezing)
    """
    T_val = 2.0    # °C  above Tm → flux_unfrozen > 0
    b0    = 0.5    # m
    fg    = 20.0   # K m-1
    dt    = 1.0    # s  ← tiny so T_mean barely moves

    m = make_model(flat_grid, dt=dt, frozen_gradient=fg)
    m._T_mean[:] = T_val
    m._E[:]      = m.C_u * m.rho_u * m._b * (m._T_mean - m.Tm)
    m._b[:]      = b0

    _run(m)

    flux_unfrozen   = m.k_u * (T_val - m.Tm) / (b0 / 2)
    flux_frozen     = m.k_f * fg
    dzb_dt_expected = (flux_unfrozen - flux_frozen) / (m.rho_w * m.phi * m.L)

    np.testing.assert_allclose(
        m._dzb_dt[flat_grid.core_nodes],
        dzb_dt_expected,
        rtol=1e-4,   # slightly relaxed: T_mean shifts by ~5e-6 K during dt
        err_msg="Stefan condition: dzb_dt did not match flux balance",
    )


# ============================================================
# Test 3 – top boundary condition (single forward-Euler step)
# ============================================================

def test_top_bc_single_step(flat_grid):
    """
    With only BC_top active, one forward-Euler step should give

        T_mean += (S0 + beta*(T_air - T_mean_0)) * dt / (C_u * rho_u * b)

    Terms zeroed and why:
    ┌──────────────────────────────────────────────────────────┐
    │  T_mean_0 = Tm        →  flux_unfrozen = 0               │
    │  Qdiss = 0                                               │
    │  frozen_gradient = 0  →  flux_frozen = 0 (Stefan = 0)    │
    │  q = 0, vel = 0       →  no advection; single substep    │
    │  uniform T_mean       →  lateral diffusion = 0           │
    └──────────────────────────────────────────────────────────┘

    Analytical check (with defaults):
        BC_top_0    = 50 + 0.04*(5 - 0) = 50.2 W m-2
        dT_expected = 50.2 * 3600 / (3835.6 * 1136 * 0.5)  ≈  0.083 K
    """
    S0    = 50.0    # W m-2
    beta  = 0.04    # W m-2 K-1
    T_air = 5.0     # °C
    b0    = 0.5     # m
    T0    = 0.0     # °C  == Tm  →  flux_unfrozen = 0
    dt    = 3600.0  # s

    m = make_model(flat_grid, dt=dt, S0=S0, beta=beta, T_air=T_air, Tm=T0)
    m._T_mean[:] = T0
    m._E[:]      = m.C_u * m.rho_u * m._b * (m._T_mean - m.Tm)
    m._b[:]      = b0

    _run(m)

    BC_top_0    = S0 + beta * (T_air - T0)
    dT_expected = BC_top_0 * dt / (m.C_u * m.rho_u * b0)

    np.testing.assert_allclose(
        m._T_mean[flat_grid.core_nodes],
        T0 + dT_expected,
        rtol=1e-5,
        err_msg="Top BC: T_mean after one step did not match forward-Euler prediction",
    )


# ============================================================
# Test 4 – Pure diffusion: Gaussian variance growth
# ============================================================

def test_gaussian_spreading():
    """
    Without advection, source terms, or vertical boundary fluxes, the second
    moment of a Gaussian temperature profile evolves as:

        <r²>(t) = 2σ₀² + 4Dt,   D = k_u / (C_u * rho_u)

    Key insight: flux_unfrozen acts as a uniform multiplicative decay
    (dT/dt ∝ -λT), so T(r,t) = exp(-λt) * G(r; σ²(t)).  The decay factor
    cancels in the normalised second moment:

        <r²> = ∫r²T dA / ∫T dA = 2σ²(t)  regardless of λ

    Parameter choices:
    ┌──────────────────────────────────────────────────────────────────┐
    │  k_u = C_u = rho_u = 1  →  D = 1 m²/s  (clean arithmetic)     │
    │  b = 1000 m              →  flux_unfrozen decay ≈ 2e-6 /s       │
    │                             τ_decay = 5e5 s ≫ t_test = 50 s    │
    │  σ₀ = 20 m, domain 200 m →  3σ = 60 m ≪ 100 m to wall         │
    │  50 steps × 1 s          →  <r²>: 800 → 1000 m²  (25 % rise)  │
    └──────────────────────────────────────────────────────────────────┘
    Closed boundaries → zero flux (Neumann); wall reflections negligible
    since T(wall) ≈ exp(−100²/800) ≈ 4e-6 throughout the run.
    """
    mg = RasterModelGrid((41, 41), xy_spacing=5.0)   # 200 m × 200 m
    mg.set_closed_boundaries_at_grid_edges(
        bottom_is_closed=True, top_is_closed=True,
        left_is_closed=True,   right_is_closed=True,
    )

    dt      = 1.0    # s
    n_steps = 50
    sigma0  = 20.0   # m

    m = make_model(mg, dt=dt, k_u=1.0, C_u=1.0, rho_u=1.0, Tm=0.0)
    m._b[:] = 1000.0   # suppresses flux_unfrozen term to negligible level

    D = m.k_u / (m.C_u * m.rho_u)   # 1.0 m²/s

    xc = np.mean(mg.x_of_node[mg.core_nodes])
    yc = np.mean(mg.y_of_node[mg.core_nodes])
    r2 = (mg.x_of_node - xc)**2 + (mg.y_of_node - yc)**2
    m._T_mean[:] = np.exp(-r2 / (2.0 * sigma0**2))
    m._E[:] = m.C_u * m.rho_u * m._b * (m._T_mean - m.Tm)
    # boundary nodes: never updated by run_heat_transport; closed BCs mean
    # their values do not enter the flux divergence at core nodes

    for _ in range(n_steps):
        _run(m)

    # Weighted second moment over core nodes only
    T_c   = m._T_mean[mg.core_nodes]
    r2_c  = r2[mg.core_nodes]
    r2_mean = np.sum(T_c * r2_c) / np.sum(T_c)

    r2_expected = 2.0 * sigma0**2 + 4.0 * D * (n_steps * dt)   # 1000 m²

    np.testing.assert_allclose(
        r2_mean, r2_expected,
        rtol=0.05,
        err_msg="Gaussian spreading: <r²> did not grow as 2σ₀² + 4Dt; "
                "check D = k_u/(C_u*rho_u) in lateral diffusion term",
    )


# ============================================================
# Test 5 – Advection-diffusion: Peclet profile at steady state
# ============================================================

def test_peclet_profile():
    """
    With uniform upward flow u and fixed Dirichlet temperatures at bottom
    (T_hot) and top (T_cold), the 1-D steady state satisfies:

        u dT/dy = D d²T/dy²,   Pe = uL/D

    with analytical solution:
        T(y) = T_hot + (T_cold - T_hot) * (exp(Pe y/L) - 1) / (exp(Pe) - 1)

    Boundary nodes (rows 0 and Ny-1) are held at fixed reservoir values via an
    explicit no-op override of _update_open_boundary_temperature: this test
    exercises bulk advection-diffusion physics, not the model's own open-
    boundary handling (which instead sets T dynamically from neighbor means;
    see test_open_boundary_tracks_neighbor_mean for that).

    Parameter choices:
    ┌───────────────────────────────────────────────────────────────────────┐
    │  k_u = C_u = rho_u = 1  →  D = 1.0 m²/s                                 │
    │  u = 0.2 m/s, L = 10 m  →  Pe = 2  (well-resolved: δ = D/u = 5 cells) │
    │  b = 1000 m              →  vertical fluxes suppressed by 1/b²        │
    │  dt = 1 s < dt_Courant = 2.5 s  →  single substep per call            │
    │  200 steps = 200 s ≫ τ_diff = L²/D = 100 s  →  converged              │
    └───────────────────────────────────────────────────────────────────────┘
    The test starts from a uniform temperature (not the analytical solution)
    to verify convergence TO the Peclet profile, not just its preservation.
    """
    Ny = 11; Nx = 3; dx = 1.0
    mg = RasterModelGrid((Ny, Nx), xy_spacing=dx)
    # left/right closed (1-D problem); top/bottom open (active boundary links
    # carry T_hot and T_cold into the flux divergence at adjacent core nodes)
    mg.set_closed_boundaries_at_grid_edges(
        left_is_closed=True,    right_is_closed=True,
        bottom_is_closed=False, top_is_closed=False,
    )

    D     = 1.0;    u  = 0.2          # m²/s, m/s
    L     = (Ny - 1) * dx             # 10 m
    Pe    = u * L / D                  # 2.0
    T_hot = 1.0;    T_cold = 0.0
    b_val = 1000.0                     # m

    y = mg.y_of_node
    T_analytical = (T_hot
                    + (T_cold - T_hot)
                    * (np.exp(Pe * y / L) - 1.0)
                    / (np.exp(Pe) - 1.0))

    dt = 0.2   # s

    m = make_model(mg, dt=dt, k_u=1.0, C_u=1.0, rho_u=1.0, Tm=0.0)
    m._b[:] = b_val
    # This test wants fixed reservoir BCs, not the model's dynamic
    # neighbor-mean open-boundary condition -- disable it here.
    m._update_open_boundary_temperature = lambda: None

    # Start from uniform T (not the steady state) to test convergence
    m._T_mean[:] = 0.5 * (T_hot + T_cold)
    # Fix Dirichlet values at all boundary nodes (they will not be updated)
    m._T_mean[mg.boundary_nodes] = T_analytical[mg.boundary_nodes]
    m._E[:] = m.C_u * m.rho_u * m._b * (m._T_mean - m.Tm)

    # Uniform upward flow: depth-integrated flux q = u * b on vertical links
    m.gdp._q[:] = 0.0
    m.gdp._q[mg.vertical_links] = u * b_val
    m.gdp._vel[:] = 0.0
    m.gdp._vel[mg.vertical_links] = u    # Courant condition: dt_c = 0.5*dx/u = 2.5 s

    for _ in range(1000):
        _run(m)

    # Check central column core nodes (node_id % Nx == 1)
    core_col1 = mg.core_nodes[mg.core_nodes % Nx == 1]

    np.testing.assert_allclose(
        m._T_mean[core_col1],
        T_analytical[core_col1],
        atol=0.05 * (T_hot - T_cold),
        err_msg="Peclet profile: converged T did not match analytical solution; "
                "check signs of advection and diffusion terms",
    )


# ============================================================
# Test 6 – Global energy conservation with spatially variable b
# ============================================================

def test_energy_conservation_variable_b(flat_grid):
    """
    With only Qdiss active, total thermal energy must increase at exactly:

        ΔE = Q0 x dt x Σ A_cell   (sum over core nodes)

    regardless of the spatial distribution of active-layer thickness b.

    All terms that would change or redistribute energy are eliminated:
    ┌──────────────────────────────────────────────────────────────┐
    │  T_mean = Tm = 0    →  flux_unfrozen = 0; lateral diff = 0  │
    │  S0 = 0, β = 0      →  BC_top = 0                           │
    │  q = 0, vel = 0     →  no advection; single substep         │
    │  frozen_gradient = 0 →  Stefan term = 0                     │
    └──────────────────────────────────────────────────────────────┘

    The algebraic cancellation is exact:
        ΔT_i  = Q0 x dt / (C_u ρ_u b_i)
        ΔE_i  = C_u ρ_u b_i x ΔT_i x A_cell = Q0 x dt x A_cell

    so ΔE_total = Q0 x dt x Σ A_cell,  independent of the b distribution.

    This specifically tests that the 1/b factor in dT_dt is implemented
    correctly for spatially variable b.
    """
    Q0  = 10.0     # W m-2  uniform dissipative heating
    dt  = 3600.0   # s

    m = make_model(flat_grid, dt=dt, Tm=0.0)
    m._T_mean[:] = 0.0   # T = Tm everywhere → flux_unfrozen = 0
    m._E[:] = m.C_u * m.rho_u * m._b * (m._T_mean - m.Tm)

    # Non-uniform b: the 1/b in ΔT must cancel with b in ΔE
    np.random.seed(1234)
    m._b[:] = 0.5 + np.random.rand(flat_grid.number_of_nodes)   # 0.5–1.5 m
    m._Qdiss[:] = Q0

    A_cell = flat_grid.dx**2   # m²  (uniform for raster grid)
    n_core = len(flat_grid.core_nodes)

    def total_energy(model):
        return (
            model.C_u * model.rho_u
            * np.sum(model._b[flat_grid.core_nodes]
                     * model._T_mean[flat_grid.core_nodes])
            * A_cell
        )

    E_before = total_energy(m)
    _run(m)
    E_after  = total_energy(m)

    delta_E_expected = Q0 * dt * n_core * A_cell

    np.testing.assert_allclose(
        E_after - E_before,
        delta_E_expected,
        rtol=1e-10,
        err_msg="Energy conservation: ΔE ≠ Q0·dt·ΣA; "
                "check 1/b weighting in dT_dt for variable b",
    )


# ============================================================
# Test 7 – Boundary handling: closed nodes, open nodes
# ============================================================

@pytest.fixture
def mixed_boundary_grid():
    """
    5x5 grid with one open edge (top) and three closed edges (left, right,
    bottom) -- the same boundary configuration used in the production
    scripts. Gives a mix of closed, open, and core nodes in one grid.
    """
    return RasterModelGrid(
        (5, 5), xy_spacing=10.0,
        bc={"top": "open", "left": "closed", "bottom": "closed", "right": "closed"},
    )


def test_closed_nodes_never_update(mixed_boundary_grid):
    """
    Closed nodes must not be touched by run_heat_transport at all: T_mean,
    E, and the resulting interface velocity (dzb_dt) must stay exactly at
    their initial values, even under forcing that clearly changes core
    nodes. The masking happens upstream (on db_dt_local, before b_local/dz
    accumulate it), so this also checks that no inconsistent value can
    build up over repeated steps.
    """
    mg = mixed_boundary_grid
    dt = 3600.0

    m = make_model(mg, dt=dt, S0=20.0, beta=0.04, T_air=5.0, frozen_gradient=1.0)
    m._T_mean[:] = 0.5
    m._b[:] = 1.0
    m._E[:] = m.C_u * m.rho_u * m._b * (m._T_mean - m.Tm)
    m._Qdiss[:] = 5.0

    closed = m._closed_nodes
    T_before = m._T_mean[closed].copy()
    E_before = m._E[closed].copy()

    for _ in range(5):
        _run(m)

    np.testing.assert_array_equal(
        m._T_mean[closed], T_before,
        err_msg="Closed node T_mean was modified by run_heat_transport",
    )
    np.testing.assert_array_equal(
        m._E[closed], E_before,
        err_msg="Closed node E was modified by run_heat_transport",
    )
    np.testing.assert_array_equal(
        m._dzb_dt[closed], np.zeros(np.sum(closed)),
        err_msg="Closed node dzb_dt should be exactly zero (no evolution)",
    )
    # sanity check: the forcing actually did change core nodes, otherwise
    # this test would pass vacuously
    assert not np.allclose(m._T_mean[mg.core_nodes], 0.5), (
        "Test forcing did not change core T_mean; test would be vacuous"
    )


def test_open_boundary_tracks_neighbor_mean(mixed_boundary_grid):
    """
    _update_open_boundary_temperature sets each open node's T_mean to the
    mean of its core neighbors' current T_mean -- a dynamic Dirichlet BC,
    generic to whichever edge(s) are marked open. Nodes with no core
    neighbor (e.g. corners) are left untouched.
    """
    mg = mixed_boundary_grid
    m = make_model(mg)

    # distinct value per node so picking the wrong neighbor set is detectable
    m._T_mean[:] = np.arange(mg.number_of_nodes, dtype=float)
    T_ic = m._T_mean.copy()

    m._update_open_boundary_temperature()

    has_nbr = m._open_node_has_core_neighbor
    for node in m._open_node_ids[has_nbr]:
        nbrs = mg.active_adjacent_nodes_at_node[node]
        valid_nbrs = nbrs[nbrs >= 0]
        core_nbrs = valid_nbrs[mg.status_at_node[valid_nbrs] == mg.BC_NODE_IS_CORE]
        expected = T_ic[core_nbrs].mean()
        assert m._T_mean[node] == pytest.approx(expected), (
            f"Open node {node} T_mean did not match the mean of its core neighbors"
        )

    unchanged = m._open_node_ids[~has_nbr]
    np.testing.assert_array_equal(
        m._T_mean[unchanged], T_ic[unchanged],
        err_msg="Open node with no core neighbor should be left unchanged",
    )


def test_open_boundary_active_layer_can_deepen(mixed_boundary_grid):
    """
    Because open-boundary T_mean now tracks its core neighbors instead of
    staying frozen at Tm, the open boundary should show net melting
    (dzb_dt > 0) under forcing that warms the domain, just like a core
    node -- unlike a closed node, which stays inert (dzb_dt == 0) under the
    same forcing.
    """
    mg = mixed_boundary_grid
    dt = 3600.0

    # Small frozen_gradient / large Qdiss so flux_unfrozen clearly overtakes
    # flux_frozen within a few steps.
    m = make_model(mg, dt=dt, S0=50.0, beta=0.04, T_air=5.0, frozen_gradient=0.05)
    m._T_mean[:] = 0.0
    m._b[:] = 1.0
    m._E[:] = m.C_u * m.rho_u * m._b * (m._T_mean - m.Tm)
    m._Qdiss[:] = 20.0

    for _ in range(5):
        _run(m)

    open_with_nbr = m._open_node_ids[m._open_node_has_core_neighbor]
    assert np.all(m._dzb_dt[open_with_nbr] > 0), (
        "Open boundary nodes with a core neighbor should show net melting "
        "under this warming forcing, same as core nodes"
    )
    assert np.all(m._dzb_dt[mg.core_nodes] > 0), (
        "Sanity check failed: core nodes should also show net melting"
    )
    np.testing.assert_array_equal(
        m._dzb_dt[m._closed_nodes], np.zeros(np.sum(m._closed_nodes)),
        err_msg="Closed nodes should remain inert even as the open boundary deepens",
    )