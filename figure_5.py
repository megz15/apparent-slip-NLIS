import numpy as np
import matplotlib.pyplot as plt

def slip_length_beta_prime(Gamma, ns, delta_over_h):    # eqn 3.9
    exp_ns = (ns + 1.0)/ns
    return Gamma*(1/exp_ns)*((delta_over_h + 1.0)**exp_ns - 1.0)

def schon_beta(Fs, Lp_over_h, D, N):                    # from Table 1
    t1 = np.log(1.0/np.cos(Fs*np.pi/2.0))
    arg = (1.0 + np.sin(Fs*np.pi/2.0))/(1.0 - np.sin(Fs*np.pi/2.0))
    t2 = 2*np.pi + (1.0/(2.0*Fs*D*N))*np.log(arg)
    return (t1 / t2)*Lp_over_h