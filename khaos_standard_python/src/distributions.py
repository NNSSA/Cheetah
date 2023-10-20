import numpy as np
from .units import UNITS


def f_today(f, mass, u):
    return f(mass * u)


# Fermi-Dirac distribution
def f_FD(p):
    Tnu = 1.95 * UNITS.K
    return np.power(1 + np.exp(p / Tnu), -1.0)
