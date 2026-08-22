import numpy as np

def carreau_viscosity(du_dy, fluid):
    neu_0, neu_inf, lam, n = fluid['neu_0'], fluid['neu_inf'], fluid['lambda'], fluid['n']
    return neu_inf + (neu_0 - neu_inf) * (1.0 + (lam * np.abs(du_dy))**2)**((n - 1.0) / 2.0)
    # scalar strain rate becomes |du/dy| under assumption of fully developed steady 1-dim flow

def calc_fdm_residuals(u, G, dy_s, dy_p, N_s, pf, sf):
    R = np.zeros_like(u)
    
    # no-slip BC at bottom (y=-delta)
    R[0] = u[0]
    
    # secondary fluid
    du_dy_s = np.diff(u[:N_s+1]) / dy_s                 # fwd difference approxn
    tau_s = carreau_viscosity(du_dy_s, sf) * du_dy_s
    R[1:N_s] = (tau_s[1:] - tau_s[:-1]) / dy_s + G      # dtau/dy = dP/dx = -G from the JFM paper
    
    # interface stress continuity at y = 0
    du_dy_p = np.diff(u[N_s:]) / dy_p
    tau_p = carreau_viscosity(du_dy_p, pf) * du_dy_p
    R[N_s] = tau_p[0] - tau_s[-1]
    
    # primary fluid
    R[N_s+1:-1] = (tau_p[1:] - tau_p[:-1]) / dy_p + G
    
    # symm BC at y=h -> du/dy=0 (top of channel / centerline)
    R[-1] = u[-1] - u[-2]
    
    return R