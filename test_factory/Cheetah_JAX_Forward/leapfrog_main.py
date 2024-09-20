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
def leapfrog_step(state, z, args, dt):
    # First half step for momentum
    state += 0.5 * dt * equations_of_motion(z, state, args)
    # Full step for position
    state += dt * equations_of_motion(z, state, args)
    # Second half step for momentum
    state += 0.5 * dt * equations_of_motion(z, state, args)
    return state

@jit
def leapfrog_integration(time_span, ini_conds, args, time_eval):
    t0, t1 = time_span
    dt = (t1 - t0) / max_steps  # Calculate the time step
    t_eval = jnp.linspace(t0, t1, max_steps + 1)  # Time points to evaluate
    states = jnp.zeros((len(t_eval), len(ini_conds)))  # Array to store states

    # Initialize the state
    states = jax.ops.index_update(states, jax.ops.index[0], ini_conds)

    for i in range(1, len(t_eval)):
        z = t_eval[i]  # Redshift value at current step
        states = jax.ops.index_update(states, jax.ops.index[i], leapfrog_step(states[i - 1], z, args, dt))

    return t_eval, states




@jit
def my_function(u_ini, psi, r_ini, Mh):
    ur_ini = -u_ini * jnp.cos(psi)
    upsi_ini = -u_ini * jnp.sin(psi)

    ini_conds = jnp.array([r_ini, ur_ini])

    args = (Mh,)

    t_eval, states = leapfrog_integration(time_span, ini_conds, args, time_eval)
    #result = solve_equations(time_span, ini_conds, args, time_eval)
 #   full_u_sol = result.ys[:,:]

    return t_eval, states


if __name__ == "__main__":
    start = time.time()
    t_eval, states  = my_function(u_ini_array[0], psi_array[0], r_array[0], Mh_array[0])
   # jnp.save("orbits_singlep_trial_1.npy", result)
    end = time.time()
    print("Time elapsed: ", end - start)



t_eval, states = leapfrog_integration(time_span, ini_conds, args, time_eval)
jnp.save("leapfrog_teval_1.npy", t_eval)
jnp.save("leapfrog_states_1.npy", states)
