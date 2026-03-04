"""
Lokta-Volterra System: No closed form solution, we will compare to a numerically tested solution
"""
import torch
import numpy as np
from scipy.integrate import odeint

#constant parameters
ALPHA=1.0
BETA=0.1 
GAMMA=1.5
DELTA=0.075

def ode_fn(
        t: torch.Tensor, 
        xy: torch.Tensor, 
        dxy_dt: torch.Tensor, 
        )->torch.Tensor: 
    """
    Residual for the Lotka-Volterra system. 
    """
    x,y=xy[:,0:1], xy[:,1:2]
    dx_dt, dy_dt = dxy_dt[:,0:1], dxy_dt[:,1:2]

    res_x=dx_dt-(ALPHA*x-BETA*x*y)
    res_y=dy_dt-(-GAMMA*y+DELTA*x*y)
    return torch.cat([res_x, res_y], dim=1)
def boundary_conditions()-> list: 
    """Arbitrary initial conditions: x(0)=10, y(0)=5"""
    t0=torch.tensor([[0.0]],dtype=torch.float32)
    xy0=torch.tensor([[X0, Y0]],dtype=torch.float32)
    return [(t0, xy0)]
def reference_solution(t:np.ndarray)->np.ndarray:
    """Numerical reference from scipy"""
    def system(z, t): 
        x,y=z
        return [ALPHA*x-BETA*x*y, -GAMMA*y+DELTA*x*y]
    return odeint(system, [X0,Y0],t)
