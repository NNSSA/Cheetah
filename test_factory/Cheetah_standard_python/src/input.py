import numpy as np
from .constants import Tnu_0, rho_crit_0_over_hsq
from .units import UNITS
from .utils import s_time
from scipy.interpolate import PchipInterpolator


# Cosmological parameters
Omega_m = 0.3
h = 0.67
rho_crit_0 = rho_crit_0_over_hsq * h**2

# Redshift range
z_start = 0.0
z_end = 4.0
z_samples = 100
z_array = np.linspace(z_start, z_end, z_samples)

s_array = np.array([s_time(z, Omega_m, h) for z in z_array])
z_s_interpolator = PchipInterpolator(s_array, z_array)

time_span = (min(s_array), max(s_array))
time_eval = np.linspace(min(s_array), max(s_array), 10000)
z_eval = z_s_interpolator(time_eval)

# Halo mass range
Mh_min = 1e12 * UNITS.MSun
Mh_max = 1e15 * UNITS.MSun
Mh_samples = 10
Mh_array = [1e12 * UNITS.MSun]  # np.geomspace(Mh_min, Mh_max, Mh_samples)

# Distance range
r_min = 1e-3 * UNITS.Mpc
r_max = 1e-1 * UNITS.Mpc
r_samples = 10
r_array = [8e-3 * UNITS.Mpc]  # np.geomspace(r_min, r_max, r_samples)

# Particle mass range
mass_fid = 0.3 * UNITS.eV
mass_min = 0.01 * UNITS.eV
mass_max = 0.3 * UNITS.eV
mass_samples = 15
mass_array = np.geomspace(mass_min, mass_max, mass_samples)

# Momentum range
p_min = 0.01 * Tnu_0
p_max = 400 * Tnu_0
p_samples = 20
p_array = np.geomspace(p_min, p_max, p_samples)
u_ini_array = p_array / mass_fid

# Angular range
theta_samples = 10
theta_array = np.linspace(0.0, np.pi, theta_samples)

phi_samples = 10
phi_array = np.linspace(0.0, 2.0 * np.pi, phi_samples)

# Define distribution
f_array = np.zeros(
    (Mh_samples, r_samples, mass_samples, p_samples, theta_samples, phi_samples)
)
