import jax
import jax.numpy as jnp
from jax import jit
from .constants import Grav_const
from .input import rho_crit_0, Omega_m, Omega_b
from .units import UNITS


@jit
def dPhidxi_NFW(pos, redshift, Mh):
    r = pos
    Delta_vir = 200.0
    conc = jax.lax.select(
        redshift >= 4,
        10
        ** (
            1.3081
            - 0.1078 * (1 + redshift)
            + 0.00398 * (1 + redshift) ** 2
            + (0.0223 - 0.0944 * (1 + redshift) ** (-0.3907))
            * jnp.log10(Mh / UNITS.MSun)
        ),
        10
        ** (
            1.7543
            - 0.2766 * (1 + redshift)
            + 0.02039 * (1 + redshift) ** 2
            + (0.2753 + 0.00351 * (1 + redshift) - 0.3038 * (1 + redshift) ** 0.0269)
            * jnp.log10(Mh / UNITS.MSun)
            * (
                1.0
                + (-0.01537 + 0.02102 * (1 + redshift) ** (-0.1475))
                * (jnp.log10(Mh / UNITS.MSun)) ** 2
            )
        ),
    )
    Rvir = (3.0 * Mh / (4 * jnp.pi * Delta_vir * rho_crit_0)) ** (1.0 / 3)
    Rs = Rvir / conc
    rhoS = Mh / (4.0 * jnp.pi * Rs**3 * (jnp.log(1 + conc) - conc / (1.0 + conc)))
    xs = conc * r / Rvir
    return (
        4.0
        * jnp.pi
        * Grav_const
        * rhoS
        * Rs
        * (jnp.log(1.0 + xs) / (xs**2) - 1.0 / (xs * (1.0 + xs)))
    ) *  jnp.array([1]) 


@jit
def dPhidxi_hernquist(pos, Mh):
    r, psi = pos
    Mb = (Omega_b / Omega_m) * Mh
    Delta_vir = 200.0
    Rvir = (3.0 * Mh / (4 * jnp.pi * Delta_vir * rho_crit_0)) ** (1.0 / 3)
    a = Rvir * 2e-1
    return (Grav_const * Mb / jnp.power(r + a, 2.0)) * jnp.array([x / r, y / r, z / r])


@jit
def dPhidxi_tot(pos, redshift, Mh):
    return dPhidxi_NFW(pos, redshift, Mh)# + dPhidxi_hernquist(pos, Mh)
