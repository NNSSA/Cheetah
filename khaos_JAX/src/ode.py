import jax.numpy as jnp
from jax import jit
from .profiles import dPhidxi_NFW
from .input import s_array, z_array, rtol, atol, max_steps
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController


@jit
def equations_of_motion(s_time, state, args):
    redshift = jnp.interp(s_time, s_array, z_array)
    dx = state[3:]
    du = -dPhidxi_NFW(state[:3], redshift, args[0]) / (1 + redshift) ** 2
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
