import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from src.constants import n_FD_SM
import matplotlib.pyplot as plt
from src.ode import *
from src.loop_input import *
from src.distributions import f_today, f_FD
import src.utils as utils
from src.units import UNITS
import time
import subprocess
import glob
# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

@jit
def my_function(u_ini, theta, r_ini, Mh, time_span, time_eval):
    ux_ini = -u_ini * jnp.cos(theta)
    uy_ini = -u_ini * jnp.sin(theta)

    ini_conds = jnp.array([r_ini, 0.0, ux_ini, uy_ini])

    args = (Mh,)

    result = solve_equations(time_span, ini_conds, args, time_eval)
    u_sol = result.ys[-1, 2:] 

    return vmap(f_today, in_axes=(None, 0, None))(
        f_FD,
        mass_array,
        jnp.sqrt(jnp.sum(jnp.power(u_sol, 2))),
    )

if __name__ == "__main__":
    z_start_arr = jnp.geomspace(0.01, 3.11, 100)
    start = time.time()
    
    density_contrast_list = []

    for i, z_start in enumerate(z_start_arr):
        ## As most neutrino halo growth occurs in late times, z=3 is sufficient ##
        z_end = 3.00
        z_samples = 100
        z_array = jnp.linspace(z_start, z_end, z_samples)

        time_span = (jnp.min(z_array), jnp.max(z_array))
        time_eval = z_array

        print(f"Solving equations for z_start = {z_start}...")
        result = vmap(
            vmap(
                vmap(
                    vmap(my_function, in_axes=(0, None, None, None, None, None)),
                    in_axes=(None, 0, None, None, None, None),
                ),
                in_axes=(None, None, 0, None, None, None),
            ),
            in_axes=(None, None, None, 0, None, None),
        )(u_ini_array, theta_array, r_array, Mh_array, time_span, time_eval)
        f_array = result.T

        print("Solved equations, proceeding to computing overdensities...")

        density_contrast = jnp.zeros((Mh_samples, r_samples, mass_samples))
        n_FD_SM = 3.0 * 1.202 * Tnu_0**3 / 4.0 / jnp.pi**2

        for num_Mh in range(Mh_samples):
            for num_r in range(r_samples):
                density_contrast = density_contrast.at[num_Mh, num_r, :].set(
                    vmap(utils.density_trapz, in_axes=(0, 0, None))(
                        f_array[:, :, :, num_r, num_Mh],
                        p_array * mass_array[:, None] / mass_fid,
                        theta_array,
                    )
                    / n_FD_SM
                )
        print(f'central density contrast:{density_contrast[0,0,0]}')
        density_contrast_list.append(density_contrast)
        print(f"Progress: {i + 1} / {len(z_start_arr)}")

    print("All done with individual runs")

    # Combine all output files after the loop
    print("Combining all output files...")
    combined_array = jnp.stack(density_contrast_list, axis=0)
    jnp.save('combined_1e15_dens_pmax40.npy', combined_array)

    end = time.time()
    print("Time elapsed: ", end - start)
