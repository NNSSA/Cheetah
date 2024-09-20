import jax.numpy as jnp
from .constants import Tnu_0, rho_crit_0_over_hsq
from .units import UNITS


# Cosmological parameters
Omega_m = 0.3
Omega_b = 0.05
h = 0.67
rho_crit_0 = rho_crit_0_over_hsq * h**2

# Redshift range
z_start = 0.0
z_end = 4.0
z_samples = 100
z_array = jnp.linspace(z_end, z_start, z_samples)

time_span = (jnp.max(z_array), jnp.min(z_array))
time_eval = z_array

# Halo mass range
Mh_min = 1e12 * UNITS.MSun
Mh_max = 1e15 * UNITS.MSun
Mh_samples = 1
Mh_array = jnp.array([1e12 * UNITS.MSun]) #jnp.geomspace(Mh_min, Mh_max, Mh_samples)

# Distance range
r_min = 1e-3 * UNITS.Mpc
r_max = 1e4 * UNITS.Mpc
r_samples = 3
r_array = jnp.geomspace(r_min, r_max, r_samples)

# Particle mass range
mass_fid = 0.3 * UNITS.eV
mass_min = 0.01 * UNITS.eV
mass_max = 0.3 * UNITS.eV
mass_samples = 1 #15
mass_array = jnp.array([0.3 * UNITS.eV])  #jnp.geomspace(mass_min, mass_max, mass_samples)

# Momentum range
p_min = 0.01 * Tnu_0
p_max = 13 * Tnu_0
p_samples = 2
p_array = jnp.geomspace(p_min, p_max, p_samples)
u_ini_array = p_array / mass_fid

# Angular range
psi_samples = 2
psi_array = jnp.linspace(0.0, jnp.pi, psi_samples)


# Differential equation solver parameters
rtol = 1e-6
atol = 1e-6
max_steps = 2**16
