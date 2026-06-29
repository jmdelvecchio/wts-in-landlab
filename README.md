# Shallow Groundwater-Driven Formation of Water tracks in Landlab
<i>aka the greatest collaboration ever!</i>

<b>Hypothesis</b>: Water track-like features will develop over the course of a single thaw season if we seed a surface with random perturbations, given the flow-driven heating feedback proposed in the linear stability analysis of [Warburton et al](https://onlinelibrary.wiley.com/doi/abs/10.1029/2025WR040569). 

## Contents:
- `water_track_model.py` Core class WaterTrackModel that is used to conduct simulations in landlab. The model takes a landlab grid and dictionary of parameters, runs GroundwaterDupuitPercolator to calculate groundwater flux and dissipative heating, and then calculates a thermal flux balance to determine the rate of melt, which is used to update the active layer thickness.
- `run_steady_model.py` Script to set up a steady forcing simulation with the WaterTrackModel.
- `water_track_funcs.py` Some miscellaneous functions including those for calculating the wavelength and growth rate based on [Warburton et al.](https://onlinelibrary.wiley.com/doi/abs/10.1029/2025WR040569)


## Things to remember when we do this:
- The stability analysis predicts that perturbation size will control both the resulting most unstable wavelength as well as the growth rate, so you have to be mindful of (1) your model run time and (2) your grid resolution because potentially you will miss wavelength selection if you don't run it for long enough and/or the resulting most unstable wavelength won't be resolved by your grid.
- We wouldn't expect growth over a single season unless flow rates are > 10^-2 m/s
- Expected wavelengths for single season growth would be < 5 x 10^1 m, and the more closely spaced wavelengths grow faster (so you need a fine-ish grid to see the fastest-developing wavelengths)

<i>Thanks so much for David and GFZ for shipping me off to Potsdam for a week where we could hack away at this! - JD </i>