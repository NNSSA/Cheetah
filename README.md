<p align="center">
  <img src="Cheetah_logo.png" alt="Cheetah Logo" width="50%" />
</p>

# Cheetah: An auto-differentiable gravity simulation in JAX

Authors: Nashwan Sabti, Keduse Worku

Cheetah is a code that allows to quickly compute the clustering of particles onto dark-matter halos.

The code calculates neutrino trajectories through backtracking from today (when z_0 is set to 0) to z_i (corresponding to ~4.85). All of the action happens in main.py. 

For each radial position at z_0, Cheetah initializes runs for a range of velocities and angles to accurately sample the neutrino phase space distribution at z_i. The neutrino distribution at z_0 is then calculated through Liouville’s theorem through my_function and saved in f_array. 

The calculated phase space distribution today is then integrated across all velocities and angles for each radial position to estimate the mass density. Through Cheetah’s highly parallized architecture, this process is repeated for many radial positions (as well as neutrino and DM mass values) simultaneously to produce the output results in final_density_ratios.npy. 

Most parameters are can be edited through the subscript input.py. To run with default parameters, type “python main.py” while in the Cheetah_JAX directory. 

