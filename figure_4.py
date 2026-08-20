import numpy as np
import matplotlib.pyplot as plt

def slip_length_beta_prime(Gamma, ns, delta_over_h):    # eqn 3.9
    exp_ns = (ns + 1.0)/ns
    return Gamma*(1/exp_ns)*((delta_over_h + 1.0)**exp_ns - 1.0)

def delta_over_h_newt(beta_prime, Gamma):               # eqn 3.11
    # newtonian: Gamma = kp/ks, thus ks/kp = 1/Gamma.
    return np.sqrt(2.0*(1.0/Gamma)*beta_prime + 1.0) - 1.0

def schon_beta(Fs, Lp_over_h, D, N):                    # from Table 1
    t1 = np.log(1.0/np.cos(Fs*np.pi/2.0))
    arg = (1.0 + np.sin(Fs*np.pi/2.0))/(1.0 - np.sin(Fs*np.pi/2.0))
    t2 = 2*np.pi + (1.0/(2.0*Fs*D*N))*np.log(arg)
    return (t1 / t2)*Lp_over_h

def schon_points(N_val, Fs_list, Lp_h_list, D=1.0):
    dh_pts, b_pts = [], []
    for Fs in Fs_list:
        for Lp_h in Lp_h_list:
            b = schon_beta(Fs, Lp_h, D, N_val)
            dh = delta_over_h_newt(b, N_val)
            dh_pts.append(dh)
            b_pts.append(b)
    return np.array(dh_pts), np.array(b_pts)