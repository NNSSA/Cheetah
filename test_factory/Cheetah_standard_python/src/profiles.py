import numpy as np
from .constants import Grav_const
from .input import rho_crit_0
from .units import UNITS


def dPhidxi_NFW(pos, redshift, Mh):
    x, y, z = pos
    r = np.sqrt(x**2 + y**2 + z**2)
    Delta_vir = 200.0
    if redshift >= 4:
        conc = 10 ** (
            1.3081
            - 0.1078 * (1 + redshift)
            + 0.00398 * (1 + redshift) ** 2
            + (0.0223 - 0.0944 * (1 + redshift) ** (-0.3907))
            * np.log10(Mh / UNITS.MSun)
        )
    else:
        conc = 10 ** (
            1.7543
            - 0.2766 * (1 + redshift)
            + 0.02039 * (1 + redshift) ** 2
            + (0.2753 + 0.00351 * (1 + redshift) - 0.3038 * (1 + redshift) ** 0.0269)
            * np.log10(Mh / UNITS.MSun)
            * (
                1.0
                + (-0.01537 + 0.02102 * (1 + redshift) ** (-0.1475))
                * (np.log10(Mh / UNITS.MSun)) ** 2
            )
        )
    Rvir = (3.0 * Mh / (4 * np.pi * Delta_vir * rho_crit_0)) ** (1.0 / 3)
    Rs = Rvir / conc
    rhoS = Mh / (4.0 * np.pi * Rs**3 * (np.log(1 + conc) - conc / (1.0 + conc)))
    xs = conc * r / Rvir
    return (
        4.0
        * np.pi
        * Grav_const
        * rhoS
        * Rs
        * (np.log(1.0 + xs) / (xs**2) - 1.0 / (xs * (1.0 + xs)))
    ) * np.array([x / r, y / r, z / r])
