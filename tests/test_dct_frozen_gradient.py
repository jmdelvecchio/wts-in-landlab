"""
Tests for the DCT (mirror-extended) frozen-gradient correction in
WaterTrackModelThermal, compared against the periodic-FFT version.

Background
----------
The frozen-gradient correction multiplies the constant frozen flux by
(1 + coef), where coef = F^-1[ |k| F[b] ] is a half-derivative of the active
layer thickness b. np.fft.fft2 treats the domain as periodic, but ours has
closed walls whose b never changes, so any difference between the interior b
and the wall b looks like a *step* to the FFT, and |k| turns a step into a
spike (~ step / (pi dx)) at every wall. The DCT version transforms only the
evolving nodes and implicitly mirrors the field at the walls, so no step
exists.

What each test shows (in plain terms)
-------------------------------------
1. test_dct_block_excludes_closed_ring    -- which nodes are transformed
2. test_cosine_modes_scale_by_k           -- the operator really is |k|
3. test_coef_has_zero_mean                -- correction adds no net heat
4. test_uniform_interior_gives_no_edge_spike
                                          -- THE edge-artifact test: FFT
                                             spikes at the wall, DCT does not
5. test_edge_spike_scales_inversely_with_dx
                                          -- why it got worse at fine dx
6. test_fourier_and_dct_flags_are_exclusive

Run with:  pytest tests/test_dct_frozen_gradient.py -v
"""

#%%
import numpy as np
import pytest
from landlab import RasterModelGrid
from model.water_track_model import WaterTrackModelThermal

#%%

NY, NX = 40, 30      # small grid so the tests are fast
DX = 0.1             # m
KF, G = 2.728, 10.0  # W/m/K, K/m  -> constant frozen flux kf*G = 27.28 W/m2


class _Stub:
    """Empty object; attributes set by make_stub. We borrow the real methods
    from WaterTrackModelThermal so no implementation is duplicated."""
    pass


def make_stub(ny=NY, nx=NX, dx=DX):
    """Same boundary conditions as the real runs: top open, other sides closed."""
    grid = RasterModelGrid(
        (ny, nx), xy_spacing=dx,
        bc={"top": "open", "left": "closed", "bottom": "closed", "right": "closed"},
    )
    m = _Stub()
    m._grid = grid
    m._closed_nodes = grid.status_at_node == grid.BC_NODE_IS_CLOSED
    m.k_f = KF
    m.frozen_gradient = G

    m._fft_A = WaterTrackModelThermal._make_fft_matrix(m)
    m._dct_slice, m._dct_A = WaterTrackModelThermal._make_dct_matrix(m)
    return m


def fft_coef(m, b):
    """frozen_grad_coef from the existing periodic-FFT method (2D array)."""
    WaterTrackModelThermal.calc_fourier_frozen_gradient(m, b.ravel())
    return m.frozen_grad_coef.reshape(m._grid.shape)


def dct_coef(m, b):
    """frozen_grad_coef from the new DCT method (2D array)."""
    WaterTrackModelThermal.calc_dct_frozen_gradient(m, b.ravel())
    return m.frozen_grad_coef.reshape(m._grid.shape)


def wall_stepped_thickness(m, b_wall=0.5, b_interior=0.57):
    """Closed ring stays at its initial b_wall; every evolving node has
    b_interior. This is what the model looks like after thawing for a while:
    the interior thickens but the closed walls cannot."""
    b = np.full(m._grid.shape, b_interior)
    b[m._closed_nodes.reshape(m._grid.shape)] = b_wall
    return b


# ============================================================
# 1. Which nodes are transformed
# ============================================================

def test_dct_block_excludes_closed_ring():
    """The transformed block should be the core nodes plus the fully open top
    row. For a (NY, NX) grid that is rows 1..NY-1 and columns 1..NX-2 (the
    closed left/right/bottom lines are left out)."""
    m = make_stub()
    rows, cols = m._dct_slice
    assert (rows.start, rows.stop) == (1, NY), f"rows {rows} (expected 1..{NY - 1})"
    assert (cols.start, cols.stop) == (1, NX - 1), f"cols {cols} (expected 1..{NX - 2})"
    assert m._dct_A.shape == (NY - 1, NX - 2)


# ============================================================
# 2. The operator is |k|
# ============================================================

@pytest.mark.parametrize("n", [1, 3, 7])
def test_cosine_modes_scale_by_k(n):
    """A cosine with n half-wavelengths across the block is an exact eigenfunction
    of the DCT-II operator, so coef must equal |k| * perturbation, with
    |k| = n / (2 L) in cycles/m (the same convention as the FFT version).
    Test with a wave in x only, so k = n/(2 L_x)."""
    m = make_stub()
    rows, cols = m._dct_slice
    nx = cols.stop - cols.start
    Lx = nx * DX

    x = (np.arange(nx) + 0.5) * DX           # node-centred positions in the block
    wave = np.cos(n * np.pi * x / Lx)        # shape (nx,)
    amp = 0.01
    b = np.full(m._grid.shape, 0.5)
    b[rows, cols] += amp * wave[None, :]

    coef = dct_coef(m, b)[rows, cols]
    expected = np.broadcast_to(amp * (n / (2 * Lx)) * wave[None, :], coef.shape)
    np.testing.assert_allclose(coef, expected, atol=1e-12,
                               err_msg=f"mode n={n}: coef != |k| * perturbation")


# ============================================================
# 3. No net source
# ============================================================

def test_coef_has_zero_mean():
    """The k = 0 term is zero, so over the transformed block the correction
    averages to exactly zero: it moves heat around but adds none. The closed
    nodes (outside the block) get exactly zero."""
    m = make_stub()
    rng = np.random.default_rng(1)
    b = 0.5 + 0.01 * rng.random(m._grid.shape)

    coef = dct_coef(m, b)
    assert abs(coef[m._dct_slice].sum()) < 1e-12, "coef should sum to zero over the block"
    assert np.all(coef[m._closed_nodes.reshape(m._grid.shape)] == 0.0)


# ============================================================
# 4. The edge artifact: FFT spikes, DCT does not
# ============================================================

def test_uniform_interior_gives_no_edge_spike():
    """Interior thickness is uniform (0.57 m) but the closed walls are still
    0.5 m. Physically there is nothing to smooth inside the domain, so the
    correction should be zero.

      * FFT: sees the walls as a step -> large coef at the first core column
        next to the left wall.
      * DCT: only transforms the interior, which is uniform -> coef == 0.

    The test prints the flux numbers so you can see the size of the artifact.
    """
    m = make_stub()
    b = wall_stepped_thickness(m)
    mid = NY // 2      # a row well away from the top/bottom
    first_core_col = 1

    c_fft = fft_coef(m, b)[mid, first_core_col]
    c_dct = dct_coef(m, b)[mid, first_core_col]
    print(f"\ncoef at first core node next to left wall:  FFT = {c_fft:+.4f}, DCT = {c_dct:+.2e}")
    print(f"  as flux perturbation (kf*G*coef):         FFT = {KF * G * c_fft:+.3f} W/m2, "
          f"DCT = {KF * G * c_dct:+.2e} W/m2")

    assert abs(c_fft) > 0.01, "expected the FFT version to show an edge spike"
    assert abs(c_dct) < 1e-12, "DCT correction should vanish for a uniform interior"
    assert np.max(np.abs(dct_coef(m, b))) < 1e-12


# ============================================================
# 5. Why it got worse at fine resolution
# ============================================================

def test_edge_spike_scales_inversely_with_dx():
    """For a wall step the FFT edge coefficient goes like step / (pi * dx), so
    halving dx doubles it. This is why the artifact is a fine-resolution
    problem. (Same step size 0.07 m at each dx.)"""
    spikes = []
    for dx in (0.2, 0.1, 0.05):
        m = make_stub(dx=dx)
        c = fft_coef(m, wall_stepped_thickness(m))
        spikes.append(abs(c[NY // 2, 1]))
    print("\nFFT edge coef at dx = 0.2, 0.1, 0.05 m:", np.round(spikes, 4))

    ratio_1 = spikes[1] / spikes[0]
    ratio_2 = spikes[2] / spikes[1]
    assert 1.7 < ratio_1 < 2.3, f"expected ~2x per halving of dx, got {ratio_1:.2f}"
    assert 1.7 < ratio_2 < 2.3, f"expected ~2x per halving of dx, got {ratio_2:.2f}"


# ============================================================
# 6. Option handling
# ============================================================

def test_fourier_and_dct_flags_are_exclusive():
    """Turning on both frozen-gradient options is ambiguous, so the model
    refuses at construction."""
    grid = RasterModelGrid(
        (10, 10), xy_spacing=1.0,
        bc={"top": "open", "left": "closed", "bottom": "closed", "right": "closed"},
    )
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_zeros("aquifer_base__elevation", at="node")
    grid.add_zeros("water_table__elevation", at="node")
    grid.add_zeros("mean_unfrozen__temperature", at="node")
    params = dict(recharge_rate=1e-6, hydraulic_conductivity=1e-2,
                  use_fourier_frozen_gradient=True, use_dct_frozen_gradient=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        WaterTrackModelThermal(grid, params)
