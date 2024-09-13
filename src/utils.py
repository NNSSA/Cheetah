import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from .constants import c_light, Tnu_0
from .units import UNITS
from .input import Omega_m, h, z_end, Mh_samples, r_samples, mass_samples
from functools import partial
#from .distributions import f_FD


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
def density_trapz(f_array, p_array, phi_array):
    mu_integral = jax.scipy.integrate.trapezoid(f_array, jnp.cos(phi_array), axis=-1)
    
    #mu_integral is negative because our theta range goes is [1,-1] while mu is [-1,1]
    p_integral = jax.scipy.integrate.trapezoid(-mu_integral * p_array**2, p_array)
    return p_integral / (2.0 * np.pi)**2



@jit
def density_trapz_integrands(f_array, p_array, phi_array):
    mu_integral = jax.scipy.integrate.trapezoid(f_array, jnp.cos(phi_array), axis=-1)

    #mu_integral is negative because our theta range goes is [1,-1] while mu is [-1,1]
    p_integrand = -mu_integral
    return p_integrand



@jit
def compute_enclosed_mass(density_profile, r_array):
    ## computes the enclosed mass through summing over spherical shells ##
    num_dm_halos, num_radial_positions, num_neutrino_masses = density_profile.shape

    # Volumes per radial shell
    volumes = (4 / 3) * jnp.pi * (r_array[1:]**3 - r_array[:-1]**3)
    volumes = jnp.pad(volumes, (1, 0), mode='constant', constant_values=0)  # Add zero for the central shell
    volumes = jnp.expand_dims(volumes, axis=(0, 2))  # Shape (1, num_radial_positions, 1)
    volumes = jnp.broadcast_to(volumes, (num_dm_halos, num_radial_positions, num_neutrino_masses))

    # Compute enclosed mass as function of position 
    enclosed_mass_profile = jnp.cumsum(density_profile * volumes, axis=1)

    return enclosed_mass_profile



def compute_enclosed_mass_background(r_array):
    ## computes the enclosed mass for the normalized background neutrino density through summing over spherical shells ##
    r_array = jnp.sort(r_array)
    volumes = (4 / 3) * jnp.pi * (r_array[1:]**3 - r_array[:-1]**3)
    enclosed_mass_background = jnp.pad(jnp.cumsum(volumes), (1, 0), mode='constant', constant_values=0)

    num_dm_halos = Mh_samples
    num_neutrino_masses = mass_samples
    num_radial_positions = len(r_array)

    return jnp.broadcast_to(jnp.expand_dims(enclosed_mass_background, axis=(0, 2)), (num_dm_halos, num_radial_positions, num_neutrino_masses))
