import numpy as np
from scipy.optimize import root

def carreau_viscosity(du_dy, fluid):
    neu_0, neu_inf, lam, n = fluid['neu_0'], fluid['neu_inf'], fluid['lambda'], fluid['n']
    return neu_inf + (neu_0 - neu_inf) * (1.0 + (lam * np.abs(du_dy))**2)**((n - 1.0) / 2.0)
    # scalar strain rate becomes |du/dy| under assumption of fully developed steady 1-dim flow

def calc_fdm_residuals(u, G, dy_s, dy_p, N_s, pf, sf):
    R = np.zeros_like(u)
    
    # no-slip BC at bottom (y=-delta)
    R[0] = u[0]
    
    # secondary fluid
    du_dy_s = np.diff(u[:N_s+1]) / dy_s       # fwd difference approxn; np.diff(u_[:N_s+1]) = u[1:N_s+1] - u[:N_s]
    tau_s = carreau_viscosity(du_dy_s, sf) * du_dy_s
    R[1:N_s] = np.diff(tau_s) / dy_s + G      # dtau/dy = dP/dx = -G from the JFM paper; np.diff(tau_s) = (tau_s[1:] - tau_s[:-1])
    
    # interface stress continuity at y = 0
    du_dy_p = np.diff(u[N_s:]) / dy_p
    tau_p = carreau_viscosity(du_dy_p, pf) * du_dy_p
    R[N_s] = tau_p[0] - tau_s[-1]
    
    # primary fluid
    R[N_s+1:-1] = np.diff(tau_p) / dy_p + G
    
    # symm BC at y=h -> du/dy=0 (top of channel / centerline)
    R[-1] = u[-1] - u[-2]
    
    return R

def solve_carreau_profile(G, h, delta, pf, sf, N_s=100, N_p=200):
    dy_s = delta / N_s
    dy_p = h / N_p
    N_total = N_s + N_p
    
    u_guess = np.linspace(0, 0.05, N_total)     # init guess, random simple linear velocity profile
    
    sol = root(calc_fdm_residuals, u_guess, args=(G, dy_s, dy_p, N_s, pf, sf), method='krylov', options={'fatol': 1e-6})
    # try powell's hybr, levenberg-marquardt, quasi-newton instead of krylov
    if not sol.success: raise RuntimeError(f"FDM solver diverged: {sol.message}")
    u_full = sol.x
    
    # spatial grid from y = -delta to y = h
    y_s = np.linspace(-delta, 0, N_s + 1)
    y_p = np.linspace(0, h, N_p + 1)[1:] 
    y_full = np.concatenate((y_s, y_p))
    
    # get primary fluid vel to compute avg vel <u_p>
    u_p = u_full[N_s:]
    u_p_avg = np.trapezoid(u_p, dx=dy_p) / h    # numinteg
    
    # nondimensionalize
    y_over_h = y_full / h
    u_nondim = u_full / u_p_avg
    
    return y_over_h, u_nondim