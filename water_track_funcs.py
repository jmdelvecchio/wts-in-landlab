""" 
Functions to support water track modelling
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import welch

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
