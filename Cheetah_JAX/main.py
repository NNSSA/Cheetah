import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from src.constants import n_FD_SM, Tnu_0
import matplotlib.pyplot as plt
from src.ode import *
from src.input import *
from src.distributions import f_today, f_FD, Gauss_bump, f_dropoff, f_declining_power
import src.utils as utils
from src.units import UNITS
import time
# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

@jit
def my_function(u_ini, theta, r_ini, Mh):
    uy_ini = -u_ini * jnp.cos(theta)
    uz_ini = -u_ini * jnp.sin(theta)

    ini_conds = jnp.array([r_ini, 0.0, uy_ini, uz_ini])

    args = (Mh,)

    result = solve_equations(time_span, ini_conds, args, time_eval)
    #returns velocities at z_i
    u_sol = result.ys[-1, 2:] 
    
    #Using Liouville's theorem to equate phase space distributions
    return vmap(f_today, in_axes=(None, 0, None))(
        f_FD, #alter initial neutrino distribution here
        mass_array,
        jnp.sqrt(jnp.sum(jnp.power(u_sol, 2))),
    )


if __name__ == "__main__":
    print("Solving equations...")
    start = time.time()
    result = vmap(
        vmap(
            vmap(
                vmap(my_function, in_axes=(0, None, None, None)),
                in_axes=(None, 0, None, None),
            ),
            in_axes=(None, None, 0, None),
        ),
        in_axes=(None, None, None, 0),
    )(u_ini_array, theta_array, r_array, Mh_array)
    f_array = result.T
    
    print("Solved equations, proceeding to computing overdensities...")
    density_contrast = jnp.zeros((Mh_samples, r_samples, mass_samples))
    n_FD_SM = 3.0 * 1.202 * Tnu_0**3 / 4.0 / jnp.pi**2 

    for num_Mh in range(Mh_samples):
        for num_r in range(r_samples):
            density_contrast = density_contrast.at[num_Mh, num_r, :].set(
                vmap(utils.density_trapz, in_axes=(0, 0, None))(
                    f_array[:, :, :, num_r, num_Mh],
                    p_array * mass_array[:, None] / mass_fid, #rescaling the momenta for each neutrino mass
                    theta_array,
                )
                / n_FD_SM
            )

    print(f'R_i: {Ri / UNITS.Mpc} Mpc')
    end = time.time()
    print("Saving overdensities...")
    #jnp.save('final_density_ratios.npy', density_contrast)
    #Plotting the central density values
    plt.loglog(mass_array / UNITS.eV, density_contrast[0, 0, :] - 1)
    plt.xlabel("mass in eV")
    plt.ylabel("n/n_average - 1")
    plt.savefig("density_contrast_plot.png")
    print("Time elapsed:", end - start)


    ##########################################################

    ### Integrands Check ###

    #density_integrands = jnp.zeros((Mh_samples, r_samples, mass_samples, p_samples))
    #for num_Mh in range(Mh_samples):
    #    for num_r in range(r_samples):
    #        for num_p in range(p_samples):
    #            density_integrands = density_integrands.at[num_Mh, num_r,:].set(
    #                vmap(utils.density_trapz_integrands, in_axes=(0, 0, None))(
    #                    f_array[:, :, :, num_r, num_Mh],
    #                    p_array * mass_array[:, None] / mass_fid,
    #                    theta_array,
    #                )
    #                * (p_array / Tnu_0)**2
    #            )
    #jnp.save('integrands.npy', density_integrands)


    #########################################################

    ### Mass Conservation Check ###

    #enclosed_mass_profile_volume = utils.compute_enclosed_mass(density_contrast, r_array)
    #enclosed_mass_background_volume = utils.compute_enclosed_mass_background(r_array)

    #at large radii
    #ratio_volume = enclosed_mass_profile_volume[:, -1, :] / enclosed_mass_background_volume[:, -1, :]
    #if jnp.all(ratio_volume < 1.05):
    #    print("Ratios are below 1.05: mass is effectively conserved.")
    #else:
    #    print("Some ratios are above 1.05: mass conservation may not be fully satisfied.")

    #print(f"Ratio (volume method): {ratio_volume}")


