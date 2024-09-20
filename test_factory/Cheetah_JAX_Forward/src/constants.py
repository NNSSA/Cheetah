import jax.numpy as jnp
from .units import UNITS


# Neutrinos
Tnu_0 = 1.95 * UNITS.K
n_FD_SM = 3.0 * 1.202 * Tnu_0**3 / 4.0 / jnp.pi**2

# Other constants
Grav_const = 4.786486e-20 * UNITS.Mpc / UNITS.MSun
c_light = 299792.458 * UNITS.km / UNITS.s
rho_crit_0_over_hsq = 2.77463e11 * UNITS.MSun / UNITS.Mpc**3
