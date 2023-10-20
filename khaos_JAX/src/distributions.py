import jax.numpy as jnp
from jax import jit
from functools import partial
from .units import UNITS


@partial(jit, static_argnums=0)
def f_today(f, mass, u):
    return f(mass * u)


# Fermi-Dirac distribution
@jit
def f_FD(p):
    Tnu = 1.95 * UNITS.K
    return jnp.power(1 + jnp.exp(p / Tnu), -1.0)
