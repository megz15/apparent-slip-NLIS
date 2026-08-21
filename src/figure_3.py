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

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel (a): vary Gamma (ns = 0.7, np = 1.3, delta/h = 0.5)
ax = axes[0, 0]
dh_a = 0.5
y_arr_a = np.linspace(-dh_a, 1.0, 500)
for Gamma, c in zip([0.1, 1.0, 10.0], ['blue', 'black', 'red']):
    u_prof = velocity_profile(y_arr_a, Gamma, 0.7, 1.3, dh_a)
    ax.plot(u_prof, y_arr_a, color=c, label=f'$\\Gamma$ = {Gamma}')

ax.set_xlabel('$u/\\langle\\bar{u_p}\\rangle$')
ax.set_ylabel('$y/h$')
ax.set_title('$n_s = 0.7, n_p = 1.3, \\delta/h = 0.5$')
ax.legend(loc='lower right')
ax.grid(True, which='both', alpha=0.4)
ax.axhline(0, color='gray', linestyle=':')

# Panel (b): vary ns (Gamma = 1, np = 1, delta/h = 0.5)
ax = axes[0, 1]
dh_b = 0.5
y_arr_b = np.linspace(-dh_b, 1.0, 500)
for ns, c in zip([0.25, 0.5, 1.0, 1.5, 1.75], ['blue', 'green', 'black', 'red', 'magenta']):
    u_prof = velocity_profile(y_arr_b, 1.0, ns, 1.0, dh_b)
    ax.plot(u_prof, y_arr_b, color=c, label=f'$n_s$ = {ns}')

ax.set_xlabel('$u/\\langle\\bar{u_p}\\rangle$')
ax.set_ylabel('$y/h$')
ax.set_title('$\\Gamma = 1, n_p = 1, \\delta/h = 0.5$')
ax.legend(loc='lower right')
ax.grid(True, which='both', alpha=0.4)
ax.axhline(0, color='gray', linestyle=':')

# Panel (c): vary np (Gamma = 1, ns = 1, delta/h = 0.5)
ax = axes[1, 0]
dh_c = 0.5
y_arr_c = np.linspace(-dh_c, 1.0, 500)
for np_, c in zip([0.25, 0.5, 1.0, 1.5, 1.75], ['blue', 'green', 'black', 'red', 'magenta']):
    u_prof = velocity_profile(y_arr_c, 1.0, 1.0, np_, dh_c)
    ax.plot(u_prof, y_arr_c, color=c, label=f'$n_p$ = {np_}')

ax.set_xlabel('$u/\\langle\\bar{u_p}\\rangle$')
ax.set_ylabel('$y/h$')
ax.set_title('$\\Gamma = 1, n_s = 1, \\delta/h = 0.5$')
ax.legend(loc='lower right')
ax.grid(True, which='both', alpha=0.4)
ax.axhline(0, color='gray', linestyle=':')

# Panel (d): vary delta/h (Gamma = 1, ns = 1, np = 1)
ax = axes[1, 1]
for dh, c in zip([0, 0.1, 0.25, 0.5, 1.0], ['blue', 'green', 'black', 'red', 'magenta']):
    y_arr_d = np.linspace(-dh, 1.0, 500)
    u_prof = velocity_profile(y_arr_d, 1.0, 1.0, 1.0, dh)
    ax.plot(u_prof, y_arr_d, color=c, label=f'$\\delta/h$ = {dh}')

ax.set_xlabel('$u/\\langle\\bar{u_p}\\rangle$')
ax.set_ylabel('$y/h$')
ax.set_title('$\\Gamma = 1, n_s = 1, n_p = 1$')
ax.legend(loc='lower right')
ax.grid(True, which='both', alpha=0.4)
ax.axhline(0, color='gray', linestyle=':')

plt.tight_layout()
plt.savefig('figure3_velocity_profiles.png', dpi=150, bbox_inches='tight')
plt.show()