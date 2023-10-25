import numpy as np
from .constants import c_light
from .units import UNITS
from scipy.integrate import simps


points, weights = np.polynomial.legendre.leggauss(25)


def integrator(f, a, b):
    sub = (b - a) / 2.0
    add = (b + a) / 2.0

    if sub == 0:
        return 0.0

    return sub * np.dot(f(sub * points + add), weights)


def s_time(z, Omega_m, h):
    return (
        c_light
        * integrator(
            lambda x: (1 + x) / np.sqrt(Omega_m * np.power(1 + x, 3) + 1.0 - Omega_m),
            0.0,
            z,
        )
        / (100.0 * h * UNITS.km / UNITS.s / UNITS.Mpc)
    )


def density(f_array, p_array, theta_array, phi_array):
    phi_integral = simps(f_array, phi_array, axis=-1)
    theta_integral = simps(phi_integral * np.sin(theta_array), theta_array, axis=-1)
    p_integral = simps(theta_integral * p_array**3, np.log(p_array))
    return p_integral / (2.0 * np.pi) ** 3
