import numpy as np
import matplotlib.pyplot as plt

def slip_length_beta_prime(Gamma, ns, delta_over_h):    # eqn 3.9
    exp_ns = (ns + 1)/ns
    return Gamma*(1/exp_ns)*((delta_over_h + 1)**exp_ns - 1)

def G_ratio_const_Q(np_, beta_prime):                   # eqn 3.15
    return (1 + ((2 * np_ + 1) / np_) * beta_prime)**np_

def PDR_percent(np_, beta_prime):        # pressure drag reduction
    return (1 - 1 / G_ratio_const_Q(np_, beta_prime)) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): G_no-slip/G_slip vs BETAprime for varying np
ax = axes[0]
beta_arr = np.linspace(0, 0.5, 300)
for np_, c in zip([0.5, 0.8, 1.0, 1.2, 1.5], ['blue', 'green', 'black', 'red', 'magenta']):
    ax.plot(beta_arr, G_ratio_const_Q(np_, beta_arr), color=c, label=f'$n_p = {np_}$')

ax.set_xlabel("$\\beta'$")
ax.set_ylabel('$G_{no-slip}/G_{slip}$')
ax.set_title("(a) $G_{no-slip}/G_{slip}$ vs $\\beta'$")
ax.legend(loc='upper left')
ax.grid(True, alpha=0.4)
ax.set_xlim(0, 0.5)
ax.set_ylim(1.0, 4)

# Panel (b): PDR(%) vs Gamma (log scale) at np=0.7, del/h=0.5
ax = axes[1]
np_b, delta_b = 0.7, 0.5
Gamma_arr = np.logspace(-2, 2, 500)
for ns, c in zip([0.2, 0.5, 1.0], ['red', 'blue', 'black']):
    beta_arr_b = slip_length_beta_prime(Gamma_arr, ns, delta_b)
    pdr_arr = PDR_percent(np_b, beta_arr_b)
    ax.plot(Gamma_arr, pdr_arr, color=c, label=f'$n_s = {ns}$')

ax.set_xscale('log')
ax.set_xlabel('$\\Gamma$')
ax.set_ylabel('PDR (%)')
ax.set_title(f'(b) PDR vs $\\Gamma$ ($n_p={np_b}$, $\\delta/h={delta_b}$)')
ax.legend()
ax.grid(True, which='both', alpha=0.4)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('figure7_pressure_ratio_PDR.png', dpi=150, bbox_inches='tight')
plt.show()