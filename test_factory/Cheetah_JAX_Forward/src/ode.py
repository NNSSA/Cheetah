import jax.numpy as jnp
from jax import jit
from .profiles import dPhidxi_tot
from .utils import Hubble
from .input import rtol, atol, max_steps
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController


@jit
def equations_of_motion(z, state, args):
    dr = -state[1] * (1 + z)**2 / Hubble(z) * jnp.array([1]) #dr = -state[1] / Hubble(z) * jnp.array([1])
    dur = dPhidxi_tot(state[0], z, args[0]) / Hubble(z) #dur = dPhidxi_tot(state[0], z, args[0]) * (1 + z)**-2 / Hubble(z)
    return jnp.concatenate((dr, dur))


term = ODETerm(equations_of_motion)
solver = Dopri5()
stepsize_controller = PIDController(rtol=rtol, atol=atol)


@jit
def solve_equations(time_span, ini_conds, args, time_eval):
    return diffeqsolve(
        term,
        solver,
        t0=time_span[0],
        t1=time_span[1],
        dt0=None,
        y0=ini_conds,
        args=args,
        saveat=SaveAt(ts=time_eval),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
