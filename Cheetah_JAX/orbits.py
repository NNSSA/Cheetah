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
def my_function(u_ini, theta, r_ini, Mh):
    uy_ini = -u_ini * jnp.sin(theta)
    uz_ini = -u_ini * jnp.cos(theta)

    ini_conds = jnp.array([r_ini, 0.0, uy_ini, uz_ini])

    args = (Mh,)

    result = solve_equations(time_span, ini_conds, args, time_eval)
    sol = result.ys
    #usol = result.ys[:, :2] #for velocity components
    return sol

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
    orbits = result.T
    
    end = time.time()
    print("Saving orbits...")
    jnp.save('orbits.npy', orbits)
    print("Time elapsed: ", end - start)
