import numpy as np
import matplotlib.pyplot as plt

def slip_length_beta_prime(Gamma, ns, delta_over_h):    # eqn 3.9
    exp_ns = (ns + 1.0)/ns
    return Gamma*(1/exp_ns)*((delta_over_h + 1.0)**exp_ns - 1.0)

def Q_ratio_const_P(np_, beta_prime):                   # eqn 3.13
    # newtonian (np=1) then 1 + 3Beta' (choi)
    return 1.0 + ((2.0*np_ + 1.0)/np_)*beta_prime

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
beta_arr = np.linspace(0, 0.5, 300)
for np_, c in zip([0.5, 0.8, 1.0, 1.2, 1.5], ['blue', 'green', 'black', 'red', 'magenta']):
    ax.plot(beta_arr, Q_ratio_const_P(np_, beta_arr), color=c, label=f'$n_p = {np_}$')

ax.set_xlabel("$\\beta'$")
ax.set_ylabel('$Q_{slip}/Q_{no-slip}$')
ax.set_title("$Q_{slip}/Q_{no-slip}$ vs $\\beta'$")
ax.legend(loc='upper left')
ax.grid(True, alpha=0.4)
ax.set_xlim(0, 0.5)
ax.set_ylim(1.0, 3.5)

ax = axes[1]
Gamma_b, delta_b = 0.5, 0.5
ns_arr = np.linspace(0.5, 1.5, 200)
np_arr = np.linspace(0.5, 1.5, 200)
NS, NP = np.meshgrid(ns_arr, np_arr)

# computing BETAprime & Q_ratio on the grid
beta_grid = slip_length_beta_prime(Gamma_b, NS, delta_b)
Q_grid = Q_ratio_const_P(NP, beta_grid)

# contour plot with labeled levels
levels = np.linspace(1.8, 2.5, 30)
cs = ax.contourf(NS, NP, Q_grid, levels=levels, cmap='viridis')
fig.colorbar(cs, ax=ax, label='$Q_{slip}/Q_{no-slip}$')
cs2 = ax.contour(NS, NP, Q_grid, levels=levels, colors='k', linewidths=0.5)
ax.clabel(cs2, fmt='%.4f', inline=True)

# mark Newtonian point (np=ns=1, Q=1.9375)
ax.plot(1.0, 1.0, 'r*', markersize=15, zorder=5, label='$n_p=n_s=1$: $Q_{ratio}=1.9375$')
ax.set_xlabel('$n_s$')
ax.set_ylabel('$n_p$')
ax.set_title(f'(b) $Q_{{slip}}/Q_{{no-slip}}$ at $\\Gamma={Gamma_b}$, ' f'$\\delta/h={delta_b}$')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figure6_flow_rate_ratio.png', dpi=150, bbox_inches='tight')
plt.show()