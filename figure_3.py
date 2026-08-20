import numpy as np
import matplotlib.pyplot as plt

def velocity_profile(y_over_h, Gamma, ns, np_, delta_over_h):
    """
    Nondimnl velocity u/<\bar{u_p}> for the two-layer NLIS flow
    Equations 3.3, 3.4, 3.5 in the paper
    y_over_h: array of y/h values (can be negative for the secondary layer)
    """
    exp_ns = (ns + 1.0)/ns
    exp_np = (np_ + 1.0)/np_

    result = np.zeros_like(y_over_h, dtype=float)

    # page 12 last para: no slip layer -> classical no-slip Poiseuille flow
    # if delta_over_h == 0:
    #     mask_p = (y_over_h >= 0) & (y_over_h <= 1)
    #     yp = y_over_h[mask_p]
    #     result[mask_p] = (2.0*np_ + 1.0)/(np_ + 1.0)*(1.0 - (1.0 - yp)**exp_np)
    #     return result

    # common term (delta/h+1)^((n_s+1)/n_s) - 1
    delta_term = (delta_over_h + 1.0)**exp_ns - 1.0

    # denom (n_p/(2n_p + 1)) + Gamma*(n_s/(n_s+1))*delta_term   [from <u_p>/A]
    denom = (np_/(2.0*np_ + 1.0)) + Gamma*(ns/(ns + 1.0))*delta_term

    # primary fluid region 0 <= y/h <= 1
    mask_p = y_over_h >= 0
    yp = y_over_h[mask_p]
    numer_p = (np_/(np_ + 1.0))*(1.0 - (1.0 - yp)**exp_np) + Gamma*(ns/(ns + 1.0))*delta_term
    result[mask_p] = numer_p/denom

    # secondary fluid region -delta_over_h <= y/h < 0
    mask_s = y_over_h < 0
    ys = y_over_h[mask_s]
    numer_s = Gamma*(ns/(ns + 1.0))*((delta_over_h + 1.0)**exp_ns - (1.0 - ys)**exp_ns)
    result[mask_s] = numer_s/denom

    return result