import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from src.constants import n_FD_SM
import matplotlib.pyplot as plt
from src.ode import *
from src.input_single import *
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

    ini_conds = jnp.array([r_ini, ur_ini])

    args = (Mh,)

    result = solve_equations(time_span, ini_conds, args, time_eval)


    full_u_sol = result.ys[:,:]

    return full_u_sol


if __name__ == "__main__":
    start = time.time()
    result = my_function(u_ini_array[0], psi_array[0], r_array[0], Mh_array[0])
    jnp.save("orbits_singlep_trial_1.npy", result)
    end = time.time()
    print("Time elapsed: ", end - start)

