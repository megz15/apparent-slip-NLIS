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

fig, ax = plt.subplots(figsize=(8, 6))

delta_over_h = 0.2
Gamma_arr = np.logspace(-2, 2.4, 500)
for ns, c in zip([0.2, 0.5, 1.0, 1.3, 1.5], ['blue', 'green', 'black', 'red', 'magenta']):
    ax.plot(Gamma_arr, slip_length_beta_prime(Gamma_arr, ns, delta_over_h), color=c, label=f'$n_s$ = {ns}')

# Schonecker
# sch_G, sch_B = [], []
# for N_val in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
#     target = 0.22 * N_val  # required β' for δ/h = 0.2
#     found = False
#     for Fs in np.linspace(0.1, 0.9, 20):
#         if found:
#             break
#         for D in [0.5, 1.0, 2.0, 5.0]:
#             t1 = np.log(1.0 / np.cos(Fs * np.pi / 2.0)) / (2.0 * np.pi)
#             arg = (1 + np.sin(Fs * np.pi / 2)) / (1 - np.sin(Fs * np.pi / 2))
#             t2 = (1.0 / (2.0 * Fs * D * N_val)) * np.log(arg)
#             denom = t1 + t2
#             if denom > 0:
#                 Lp_h = target / denom
#                 if 0.005 < Lp_h < 20:
#                     b = schon_beta(Fs, Lp_h, D, N_val)
#                     sch_G.append(N_val)
#                     sch_B.append(b)
#                     found = True
#                     break
# if sch_G:
#     ax.plot(sch_G, sch_B, 'o', color='green', ms=10, mfc='none', mew=2,
#             label='Schonecker et al. (2014)')

ax.set_xscale('log')
ax.set_xlabel('$\\Gamma$')
ax.set_ylabel("$\\beta'$")
ax.set_title(f"$\\beta'$ vs $\\Gamma$ at $\\delta/h = {delta_over_h}$")
ax.legend(loc='upper left')
ax.grid(True, which='both', alpha=0.4)
ax.set_ylim(0, 50)

plt.tight_layout()
plt.savefig('figure5_beta_vs_Gamma.png', dpi=150, bbox_inches='tight')
plt.show()