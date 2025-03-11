# Water tracks in Landlab, aka the greatest collaboration ever!

<b>Hypothesis</b>: Water track-like features will develop over the course of a single thaw season if we seed a surface with just the right size perturbations and the flow-driven heating feedback proposed in the linear stability analysis of [Warburton et al](https://eartharxiv.org/repository/view/8587/). 

## Contents:
- `hillslope.ipynb` David Litwin made this for me
- `heating_numba.ipynb` Solves a 1D Stefan problem with an internal heat dissipation term Q over the course of a year, numerically simulating the thermal profile of water track vs not water track soils might look like. Later I model Q as a function of slope and put it on a hillslope. 
- `analytic.ipynb` Analytical solution to the melt rate of a permafrost soil given an annually varying solar flux (and air temperature), plus the Q heating term inspired by the treatment of melt at a glacier bed implemented in [SHAKTI](https://gmd.copernicus.org/articles/11/2955/2018/), with a little help from our pal ChatGPT when I couldn't quite implement it the way I wanted. This is great for Landlab because then we don't have to solve a bajillion 1D Stefan problems. 

<i>Thanks so much for David and GFZ for shipping me off to Potsdam for a week where we could hack away at this! - JD</i>