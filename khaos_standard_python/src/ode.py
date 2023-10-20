import numpy as np
from .profiles import dPhidxi_NFW
from .input import z_s_interpolator
from scipy.integrate import solve_ivp


def equations_of_motion(s_time, state, Mh):
    dx = state[3:]
    du = (
        -dPhidxi_NFW(state[:3], z_s_interpolator(s_time), Mh)
        / (1 + z_s_interpolator(s_time)) ** 2
    )
    return np.concatenate((dx, du))


def solve_equations(time_span, ini_conds, args, time_eval):
    return solve_ivp(
        equations_of_motion,
        t_span=time_span,
        y0=ini_conds,
        args=args,
        method="RK45",
        t_eval=time_eval,
        rtol=1e-8,
        atol=1e-8,
    )
