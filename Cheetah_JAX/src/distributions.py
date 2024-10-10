import jax.numpy as jnp
from jax import jit, scipy
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



## Below are various alternate initial distributions for neutrinos ## 
## Note that the background density change must also be accounted for ##

@jit
def f_dropoff(p):
    # Values chosen to preserve Neff
    p_0 = 1.1
    plateau_value = 0.5
    alpha = 3.5
    Tnu = 1.3625 * UNITS.K
    p = p / Tnu

    return jnp.where(p <= p_0, plateau_value, plateau_value * jnp.exp(-alpha * (p - p_0)))


@jit
def f_declining_power(p):
    Tnu = 1.95 * UNITS.K
    p_arr = p / Tnu
    # Values chosen to preserve Neff
    power = 0.7898
    some_p_limit = 0.05
 
    return jnp.where(p_arr < some_p_limit,
                     0.5,
                     0.5 * (some_p_limit / p_arr)**power)



@jit
def Gaussian_amplitude(Neff, y, sigma, p):
    ## returns the Amplitude for the Gaussian distribution that corresponds with a given Neff ##
    factor = (360 / (7*jnp.pi**4)) * ((11/4)**(4/3)) * (1.39578**-4)
    
    Tnu = 1.95 * UNITS.K
    p_arr = p / Tnu
    Gaussian = jnp.exp(((p_arr - y)**2) / (sigma**2) / -2)

    integ = scipy.integrate.trapezoid(Gaussian * (p / Tnu)**3, p / Tnu)
    Amp = Neff / factor / integ
    return Amp


@jit
def Gaussian_distribution_simple(p):
    ## testing alternative initial neutrino distributions ##
    Tnu = 1.95 * UNITS.K
    p_arr = p / Tnu

    #Neff = 3.046
    y = 1.5
    sigma = 0.294

    Gaussian = jnp.exp(((p_arr - y)**2) / (sigma**2) / -2)
    Amp = 0.336177
    #Amp = Gaussian_amplitude(Neff, y, sigma, p)
    return (Amp * Gaussian)



def Gauss_bump(p):
    Tnu = 1.3625 * UNITS.K
    f_FD = jnp.power(1 + jnp.exp(p / Tnu), -1.0)
    return (Gaussian_distribution_simple(p) + f_FD)


