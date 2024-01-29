import jax.numpy as jnp
from jax import jit
from .profiles import dPhidxi_tot
from .utils import Hubble
from .input import rtol, atol, max_steps
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController


@jit
def equations_of_motion(z, state, args):
    dx = -state[3:] * (1 + z) / Hubble(z)
    du = dPhidxi_tot(state[:3], z, args[0]) / (1 + z) / Hubble(z)
    return jnp.concatenate((dx, du))


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
