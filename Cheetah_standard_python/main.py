import numpy as np
from src.constants import n_FD_SM
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from src.ode import solve_equations
from src.input import *
from src.distributions import f_today, f_FD
import src.utils as utils
from src.units import UNITS


def my_function(Mh, r_ini, u_ini, theta, phi):
    ux_ini = -u_ini * np.sin(theta) * np.cos(phi)
    uy_ini = -u_ini * np.sin(theta) * np.sin(phi)
    uz_ini = -u_ini * np.cos(theta)

    ini_conds = [r_ini, 0.0, 0.0, ux_ini, uy_ini, uz_ini]

    args = (Mh,)
    result_solve_ivp = solve_equations(time_span, ini_conds, args, time_eval)
    u_sol = result_solve_ivp.y[3:, -1]

    fff_array = []
    for mass in mass_array:
        fff_array.append(f_today(f_FD, mass, np.sqrt(np.sum(u_sol**2))))

    return fff_array


if __name__ == "__main__":
    for num_Mh, Mh in enumerate(Mh_array):
        for num_r, r_ini in enumerate(r_array):
            for num_phi, phi in enumerate(phi_array):
                for num_theta, theta in enumerate(theta_array):
                    print(
                        "Halo mass: {}/{}, Position: {}/{}, phi: {}/{}, theta: {}/{}".format(
                            num_Mh + 1,
                            Mh_samples,
                            num_r + 1,
                            r_samples,
                            num_phi + 1,
                            phi_samples,
                            num_theta + 1,
                            theta_samples,
                        )
                    )

                    result = Parallel(n_jobs=-1)(
                        delayed(my_function)(Mh, r_ini, u_ini, theta, phi)
                        for u_ini in u_ini_array
                    )

                    f_array[num_Mh, num_r, :, :, num_theta, num_phi] = np.array(
                        result
                    ).T

    density_contrast = np.zeros((Mh_samples, r_samples, mass_samples))
    for num_Mh in range(Mh_samples):
        for num_r in range(r_samples):
            for num_mass, mass in enumerate(mass_array):
                density_contrast[num_Mh, num_r, num_mass] = (
                    utils.density(
                        f_array[num_Mh, num_r, num_mass, :, :, :],
                        p_array * mass / mass_fid,
                        theta_array,
                        phi_array,
                    )
                    / n_FD_SM
                )

    np.savetxt(
        "final_density_ratios_standard_python.txt",
        np.vstack((mass_array / UNITS.eV, density_contrast[0, 0, :])),
    )
    plt.loglog(mass_array / UNITS.eV, density_contrast[0, 0, :] - 1)
    plt.show()
