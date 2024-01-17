import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from src.constants import n_FD_SM
import matplotlib.pyplot as plt
from src.ode import *
from src.input import *
from src.distributions import f_today, f_FD
import src.utils as utils
from src.units import UNITS
# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

@jit
def my_function(u_ini, theta, phi, r_ini, Mh):
    ux_ini = -u_ini * jnp.sin(theta) * jnp.cos(phi)
    uy_ini = -u_ini * jnp.sin(theta) * jnp.sin(phi)
    uz_ini = -u_ini * jnp.cos(theta)

    ini_conds = jnp.array([r_ini, 0.0, 0.0, ux_ini, uy_ini, uz_ini])

    args = (Mh,)

    result = solve_equations(time_span, ini_conds, args, time_eval)

    u_sol = result.ys[-1, 3:]

    return vmap(f_today, in_axes=(None, 0, None))(
        f_FD,
        mass_array,
        jnp.sqrt(jnp.sum(jnp.power(u_sol, 2))),
    )


if __name__ == "__main__":
    result = vmap(
        vmap(
            vmap(
                vmap(
                    vmap(my_function, in_axes=(0, None, None, None, None)),
                    in_axes=(None, 0, None, None, None),
                ),
                in_axes=(None, None, 0, None, None),
            ),
            in_axes=(None, None, None, 0, None),
        ),
        in_axes=(None, None, None, None, 0),
    )(u_ini_array, theta_array, phi_array, r_array, Mh_array)
    f_array = result.T

    print("Solved equations, proceeding to computing overdensities...")

    density_contrast = jnp.zeros((Mh_samples, r_samples, mass_samples))
    n_FD_SM = 3.0 * 1.202 * Tnu_0**3 / 4.0 / jnp.pi**2
    for num_Mh in range(Mh_samples):
        for num_r in range(r_samples):
            density_contrast = density_contrast.at[num_Mh, num_r, :].set(
                vmap(utils.density, in_axes=(0, 0, None, None))(
                    f_array[:, :, :, :, num_r, num_Mh],
                    p_array * mass_array[:, None] / mass_fid,
                    theta_array,
                    phi_array,
                )
                / n_FD_SM
            )

    print("Saving overdensities...")

    jnp.save("final_density_ratios_JAX.npy", density_contrast)
    plt.loglog(mass_array / UNITS.eV, density_contrast[0, 0, :] - 1)
    plt.xlabel("mass in eV")
    plt.ylabel("n/n_average - 1")
    plt.show()
