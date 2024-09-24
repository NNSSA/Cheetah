import jax.numpy as jnp
from .constants import Tnu_0, rho_crit_0_over_hsq
from .units import UNITS


# Cosmological parameters
Omega_m = 0.315
h = 0.68
rho_crit_0 = rho_crit_0_over_hsq * h**2

# Redshift range
z_start = 0.0
z_end = 4.85 
z_samples = 100
z_array = jnp.linspace(z_start, z_end, z_samples)

time_span = (jnp.min(z_array), jnp.max(z_array))
time_eval = z_array

# Halo mass range
Mh_min = 1e12 * UNITS.MSun
Mh_max = 1e15 * UNITS.MSun
Mh_samples = 1
Mh_array = jnp.array([1e15 * UNITS.MSun]) #jnp.geomspace(Mh_min, Mh_max, Mh_samples)

# Distance range
Ri = 1.77 * (Mh_array[0] / 1e12 / UNITS.MSun)**(1/3) * UNITS.Mpc
r200 = Ri / (1 + z_end)

r_min = 1e-2 * UNITS.Mpc 
r_max = 5e1 * UNITS.Mpc
r_samples = 20 
r_array = jnp.geomspace(r_min, r_max, r_samples) #jnp.geomspace(r200/10, Ri*2, r_samples)

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
theta_samples = 10
theta_array = jnp.linspace(0.0, jnp.pi, theta_samples)

# Differential equation solver parameters
rtol = 1e-6    #1e-4
atol = 1e-6    #1e-4
max_steps = 2**16  
