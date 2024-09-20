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
import time
# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

@jit
def my_function(u_ini, psi, r_ini, Mh):
    ur_ini = -u_ini * jnp.cos(psi)
    upsi_ini = -u_ini * jnp.sin(psi)

    ini_conds = jnp.array([r_ini, psi, ur_ini, upsi_ini])

    args = (Mh,)

    result = solve_equations(time_span, ini_conds, args, time_eval)

    #u_sol = result.ys[-1, 2:]

    #f_now = vmap(f_today, in_axes=(None, 0, None))(
    #    f_FD,
    #    mass_array,
    #    jnp.sqrt(jnp.sum(jnp.power(u_sol, 2))),
    #)

    full_u_sol = result.ys[:,:]

    return full_u_sol


if __name__ == "__main__":
    print("Solving equations...")
    start = time.time()
    result, orbits = vmap(
        vmap(
            vmap(
                vmap(my_function, in_axes=(0, None, None, None)),
                in_axes=(None, 0, None, None),
            ),
            in_axes=(None, None, 0, None),
        ),
        in_axes=(None, None, None, 0),
    )(u_ini_array, psi_array, r_array, Mh_array)
    #f_array = result.T

    #jnp.save("ftoday_simple_trial_1.npy", results)
    jnp.save("orbits_simple_trial_1.npy", orbits)

    end = time.time()
    print("Time elapsed: ", end - start)

