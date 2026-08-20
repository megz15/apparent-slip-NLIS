import numpy as np
import matplotlib.pyplot as plt

def wall_shear_ratio(np_, beta_prime):              # eqn B4 (appendix), simplified
    return (1.0/(1.0 + ((2.0*np_ + 1.0)/np_)*beta_prime))**np_

def poiseuille_number(np_, beta_prime):             # eqn B6 (appendix)
    return 2.0**(2.0*np_ + 3.0)*(1.0/((np_/(2.0*np_ + 1.0)) + beta_prime))**np_

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): tau_slip/tau_no-slip vs BETAprime for varying np
ax = axes[0]
beta_arr = np.linspace(0, 0.5, 300)
for np_, c in zip([0.5, 0.75, 1.0, 1.25, 1.5], ['blue', 'green', 'black', 'red', 'magenta']):
    ax.plot(beta_arr, wall_shear_ratio(np_, beta_arr), color=c, label=f'$n_p = {np_}$')

ax.set_xlabel("$\\beta'$")
ax.set_ylabel('$\\tau_w^{slip}/\\tau_w^{no-slip}$')
ax.set_title('Wall shear stress ratio vs $\\beta\'$')
ax.legend()
ax.grid(True, alpha=0.4)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.0)

# Panel (b): Po vs BETAprime for varying np
ax = axes[1]
for np_, c in zip([0.8, 0.9, 1.0, 1.1, 1.2], ['blue', 'green', 'black', 'red', 'magenta']):
    ax.plot(beta_arr, poiseuille_number(np_, beta_arr), color=c, label=f'$n_p = {np_}$')

# Davies et al: Po = 96/(1+3BETAprime) for np=1
ax.plot(beta_arr[::20], 96.0 / (1.0 + 3.0 * beta_arr[::20]), '+', color='magenta', ms=10, mew=2, label="Davies: $96/(1+3\\beta')$")

ax.set_xlabel("$\\beta'$")
ax.set_ylabel('Po')
ax.set_title('Poiseuille number vs $\\beta\'$')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.4)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 200)

plt.tight_layout()
plt.savefig('figure8_wall_shear_stress_Po.png', dpi=150, bbox_inches='tight')
plt.show()