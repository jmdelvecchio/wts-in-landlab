
#%%
from landlab import RasterModelGrid, grid, imshow_grid
import numpy as np
import matplotlib.pyplot as plt


def make_fft_matrix(grid):
    """Make a matrix for computing the 2D FFT of grid state variables.
    Used for computing the power spectrum of b (or zb) for update to frozen 
    gradient in the thermal model.
    """
    Ny, Nx = grid.shape
    kx = np.fft.fftfreq(Nx, d=grid.dx)
    ky = np.fft.fftfreq(Ny, d=grid.dy)
    kx, ky = np.meshgrid(kx, ky)
    A = np.sqrt(kx**2 + ky**2)
    return A


#%%
# grid and initial conditions
boundaries = {"top": "open", "left": "closed", "bottom": "closed", "right": "closed"}
Nx = 100; Ny = 200; dx = 10
mg = RasterModelGrid((Ny,Nx), xy_spacing=dx, bc=boundaries)
z = mg.add_zeros('topographic__elevation', at='node')
zb = mg.add_zeros('aquifer_base__elevation', at='node')
zwt = mg.add_zeros("water_table__elevation", at="node")
tmean = mg.add_zeros("mean_unfrozen__temperature", at="node")

# parabolic hillslope, uniform permeable thickness
x = mg.x_of_node
y = mg.y_of_node
a = 0.0001
b = 0.5 # permeable thickness m
z[:] = -a * y**2 + a * 2000**2
zb[:] = z - b
zwt[:] = zb + 0.1*b


wavelength_target = 100
kappa = 2 * np.pi / wavelength_target
amplitude = 0.01  # small perturbation, meters
fluct = amplitude * np.sin(kappa * mg.x_of_node)

zb0 = zb.copy()

zb[:] = zb + fluct
zwt[:] = zb + 0.5 # near equilibrium thickness


# hillslope cross section
plt.figure()
plt.plot(y, z)
plt.xlabel('Y coordinate (m)')
plt.ylabel('Elevation (m)')
plt.show()


plt.figure()
imshow_grid(mg, z-zb, colorbar_label="Topographic Elevation (m)")
plt.show()


# %%

b = z - zb
frozen_gradient = -10
k_f = 2.728
fft_A = make_fft_matrix(mg)



#%%

b_local = b.reshape(mg.shape)

b_fft = np.fft.fft2(b_local)

# product with precomputed matrix for frozen gradient coefficient
Ab_fft = np.multiply(fft_A, b_fft)

# convert back to spatial domain
frozen_grad_coef = np.real(np.fft.ifft2(Ab_fft)).flatten()

# scale the frozen gradient by the coefficient
frozen_grad = frozen_gradient * (1 + frozen_grad_coef) # check sign (+/-)?
flux_frozen = k_f * frozen_grad  # W/m2

# %%

plt.figure()
imshow_grid(mg, flux_frozen, colorbar_label="Flux Frozen")
plt.show()
# %%
