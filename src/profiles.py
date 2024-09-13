import jax
import jax.numpy as jnp
from jax import jit
from .constants import Grav_const
from .input import rho_crit_0, Omega_m, z_end, r200
from .units import UNITS
from functools import partial
from .input import r_array

@jit
def dPhidxi_tot(pos, redshift, Mh):
    return dPhidxi_NFW(pos, redshift, Mh) # + any additional potential term

@jit
def I_func(c):
    return jnp.log(1 + c) - (c / (1 + c))


@partial(jit, static_argnums=(1,))
def mod_func(redshift, power=(1)):
    ## modulation function for the DM halo growth rate. Power is optional arg, 1 for linear growth by default ##
    z_i = z_end
    #200 times critical density at previous redshift
    z_0 = ((1 + z_i) / 200**(1/3)) - 1
    mod = jnp.select([redshift <= z_0, redshift < z_i], [1, ((z_i - redshift) / (z_i - z_0))**power], default=0)
    return mod


@jit
def mu_func(z, alpha, beta):
    ## for small gravity modifications ## 
    return alpha * (z / (1 + z)) ** beta


@jit
def dPhidxi_NFW(pos, redshift, Mh):
    ## Time-evolving NFW profile with modulation function ##
    y, z = pos
    r = jnp.sqrt(jnp.power(y, 2.0) + jnp.power(z, 2.0))
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
    Ri = 1.77 * (Mh / 1e12 / UNITS.MSun)**(1/3) * UNITS.Mpc
    R200 = Ri / (1 + z_end)
    Rs = R200 / conc
    result = jnp.select([r < (R200 * (1 + redshift)), r < (R200 * (1 + z_end))], [(I_func((r / Rs) / (1 + redshift)) / I_func(conc)) - (r / ((1 + z_end) * R200)) ** 3, 1 - (r / ((1 + z_end) * R200)) ** 3], default=0)

    return (
        Grav_const
        * (1 + redshift)
        * (r ** -2)
        * mod_func(redshift)
        * Mh
        #* (1 + mu_func(redshift, alpha, beta))
        * result) * jnp.array([y / r, z / r], dtype=jnp.float64)


@jit
def dPhidxi_NFW_const(pos, redshift, Mh):
    ## constant concentration parameter ##
    y, z = pos
    r = jnp.sqrt(jnp.power(y, 2.0) + jnp.power(z, 2.0))
    conc = 4.433
    Ri = 1.77 * (Mh / 1e12 / UNITS.MSun)**(1/3) * UNITS.Mpc
    R200 = Ri / (1 + z_end)
    Rs = R200 / conc

    result = jnp.select([r < (R200 * (1 + redshift)), r < (R200 * (1 + z_end))], [(I_func((r / Rs) / (1 + redshift)) / I_func(conc)) - (r / ((1 + z_end) * R200)) ** 3, 1 - (r / ((1 + z_end) * R200)) ** 3], default=0)


    return (
        Grav_const
        * (1 + redshift)
        * (r ** -2)
        * mod_func(redshift)
        * Mh
        * result) * jnp.array([y / r, z / r], dtype=jnp.float64)



