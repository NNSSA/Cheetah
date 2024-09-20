import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from .constants import c_light
from .units import UNITS
from .input import Omega_m, h
from functools import partial


points, weights = np.polynomial.legendre.leggauss(25)


@partial(jit, static_argnums=0)
def integrator(f, a, b):
    sub = (b - a) / 2.0
    add = (b + a) / 2.0

    return jax.lax.select(sub == 0, 0.0, sub * jnp.dot(f(sub * points + add), weights))


@jit
def Hubble(z):
    return (100.0 * h * UNITS.km / UNITS.s / UNITS.Mpc) * jnp.sqrt(Omega_m * jnp.power(1 + z, 3) + 1.0 - Omega_m)


@jit
def density(f_array, p_array, theta_array, phi_array):
    phi_integral = jax.scipy.integrate.trapezoid(f_array, phi_array, axis=-1)
    theta_integral = jax.scipy.integrate.trapezoid(
        phi_integral * jnp.sin(theta_array), theta_array, axis=-1
    )
    p_integral = jax.scipy.integrate.trapezoid(
        theta_integral * p_array**3, jnp.log(p_array)
    )
    return p_integral / (2.0 * np.pi) ** 3
