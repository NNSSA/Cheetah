import jax.numpy as jnp
from jax import vmap
from .constants import Tnu_0, rho_crit_0_over_hsq
from .units import UNITS
from .utils import s_time


# Cosmological parameters
Omega_m = 0.3
h = 0.67
rho_crit_0 = rho_crit_0_over_hsq * h**2

# Redshift range
z_start = 0.0
z_end = 4.0
z_samples = 100
z_array = jnp.linspace(z_start, z_end, z_samples)

s_array = vmap(s_time, in_axes=(0, None, None))(z_array, Omega_m, h)

time_span = (jnp.min(s_array), jnp.max(s_array))
time_eval = jnp.linspace(jnp.min(s_array), jnp.max(s_array), 10000)

# Halo mass range
Mh_min = 1e12 * UNITS.MSun
Mh_max = 1e15 * UNITS.MSun
Mh_samples = 10
Mh_array = jnp.geomspace(Mh_min, Mh_max, Mh_samples)

# Distance range
r_min = 1e-3 * UNITS.Mpc
r_max = 1e-1 * UNITS.Mpc
r_samples = 10
r_array = jnp.geomspace(r_min, r_max, r_samples)

# Particle mass range
mass_fid = 0.3 * UNITS.eV
mass_min = 0.01 * UNITS.eV
mass_max = 0.3 * UNITS.eV
mass_samples = 15
mass_array = jnp.geomspace(mass_min, mass_max, mass_samples)

# Momentum range
p_min = 0.01 * Tnu_0
p_max = 400 * Tnu_0
p_samples = 200
p_array = jnp.geomspace(p_min, p_max, p_samples)
u_ini_array = p_array / mass_fid

# Angular range
theta_samples = 20
theta_array = jnp.linspace(0.0, jnp.pi, theta_samples)

phi_samples = 20
phi_array = jnp.linspace(0.0, 2.0 * jnp.pi, phi_samples)

# Differential equation solver parameters
rtol = 1e-8
atol = 1e-8
max_steps = 2**16
