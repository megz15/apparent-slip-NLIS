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

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
delta_h_arr = np.logspace(-2, 2, 500)

# Panel (a): Gamma = 0.5, varying ns
ax = axes[0]
for ns, c in zip([0.5, 0.7, 1.0, 1.3, 1.5], ['blue', 'green', 'black', 'red', 'magenta']):
    ax.plot(delta_h_arr, slip_length_beta_prime(0.5, ns, delta_h_arr), color=c, label=f'$n_s$ = {ns}')

# dh_s, b_s = schon_points(0.5, Fs_list, Lp_h_list, D=1.0)
# ax.plot(dh_s, b_s, 'o', color='red', ms=8, mfc='none', mew=1.5,
#         label='Schönecker et al. (2014)')

ax.set_xscale('log')
ax.set_xlabel('$\\delta/h$')
ax.set_ylabel("$\\beta'$")
ax.set_title('(a) $\\Gamma = 0.5$, varying $n_s$')
ax.legend(loc='lower right')
ax.grid(True, which='both', alpha=0.4)
ax.set_ylim(0, 50)

# Panel (b): ns = 1, varying Gamma
ax = axes[1]
for Gamma, c in zip([0.1, 0.2, 0.5, 0.7, 1.0], ['blue', 'green', 'black', 'red', 'magenta']):
    ax.plot(delta_h_arr, slip_length_beta_prime(Gamma, 1.0, delta_h_arr), color=c, label=f'$\\Gamma$ = {Gamma}')

# for N_val in [0.1, 0.2, 0.5, 0.7, 1.0]:
#     dh_s, b_s = schon_points(N_val, Fs_list, Lp_h_list, D=1.0)
#     ax.plot(dh_s, b_s, 'o', color='red', ms=8, mfc='none', mew=1.5)
# ax.plot([], [], 'o', color='red', ms=8, mfc='none', mew=1.5,
#         label='Schönecker et al. (2014)')

ax.set_xscale('log')
ax.set_xlabel('$\\delta/h$')
ax.set_ylabel("$\\beta'$")
ax.set_title('(b) $n_s = 1$, varying $\\Gamma$')
ax.legend(loc='lower right')
ax.grid(True, which='both', alpha=0.4)
ax.set_ylim(0, 50)

plt.tight_layout()
plt.savefig('figure4_slip_length.png', dpi=150, bbox_inches='tight')
plt.show()