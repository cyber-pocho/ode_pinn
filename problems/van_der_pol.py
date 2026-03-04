"""
Problem: Van der Pol Oscillator (stiff ODE)
  x'' - mu*(1 - x²)*x' + x = 0
  x(0) = 2,  x'(0) = 0

Rewritten as system:
  dx/dt  = v
  dv/dt  = mu*(1 - x²)*v - x

Stiffness increases with mu. Classic test for numerical solvers.
"""
import torch
import numpy as np

def ode_fn(
        t:torch.Tensor, 
        xv: torch.Tensor, 
        dxv_dt: torch.Tensor, 
        ) -> torch.Tensor:
    """
    Residual for the VdP system
    """
    x,v=xv[:,0:1], xv[:,1:2]
    dx_dt, dv_dt = dxv_dt[:, 0:1], dxv_dt[:, 1:2]

    res_x = dx_dt - v
    res_v = dv_dt - (MU * (1 - x**2) * v - x)
    return torch.cat([res_x, res_v], dim=1)
def boundary_conditions()->list: 
    t0=torch.tensor([[0,0]], dtype=torch.float32)
    xv0=torch.tensor([[X0,V0]], dtype=torch.float32)
    return [(t0, xv0)]
def reference_solution(t:np.ndarray)->np.ndarray: 
    """Scipy reference"""
    def system(t, z):
        x,v=z
        return [v, MU*(1-x**2)*v-x]
    sol=solve_ivp(system,[t[0], t[-1]], [X0,V0], t_eval=t, method="Radau")
    return sol.y.T
