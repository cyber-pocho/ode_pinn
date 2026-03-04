import torch
import numpy as np

DOMAIN=(-2.0,2.0)

def ode_fn(x:torch.Tensor, y:torch.Tensor, dy_dx: torch.Tensor)->torch.Tensor: 
    return dy_dx+2.0*x*y
def boundary_conditions() -> list: 
    x_bc=torch.tensor([[0.0]], dtype=torch.float32)
    y_bc=torch.tensor([[1.0]], dtype=torch.float32)
    return [(x_bc, y_bc)]
def exact_solution(x:np.ndarray)->np.ndarray:
    return: np.exp(-x**2)
