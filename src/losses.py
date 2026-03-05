import torch 
from typing import Callable

def compute_gradient(y: torch.Tensor, x:torch.Tensor) -> torch.Tensor: 
    """
    Compute y'(x) automatically

    Args: 
        y: Network output, shape (N,1)
        x: Input tensor with requires_grad(True), shape (N,1)
    Returns: 
        y'(x), shape(N,1)
    """
    grad = torch.autograd.grad(
            outputs=y, 
            inputs=x, 
            grad_outputs=torch.ones_like(y), 
            create_graph=True, 
            #retain_graph=True,
            )[0]
    return grad
def ode_residual_loss(
        model: torch.nn.Module, 
        x: torch.Tensor, 
        ode_fn: Callable, 
        ) -> torch.Tensor: 
    """
    Compute ODE residual at collocation points
    The ODE is written as y'(x) = f(x, y)
    Residual: r(x) y'(x)_nn-f(x, y'__n)
    Loss: mean(r^2)
    Args: 
        model: Physics Informed
        x: collocation points
        ode_fn: Function, returns 0 when ODE is satisfied
    Returns: 
        Scalar loss tensor
    """
    y = model(x)
    dy_dx=compute_gradient(y, x)
    residual = ode_fn(x, y, dy_dx)
    return torch.mean(residual**2)

def boundary_condition_loss(
        model: torch.nn.Module,
        bc_points: list,
) -> torch.Tensor:
    """
    Compute boundary condition loss.
    Args:
        model:     PINN model
        bc_points: List of (x_bc, y_target) tuples
    Returns:
        Scalar loss tensor
    """
    assert len(bc_points) > 0, "bc_points is empty — did you forget boundary conditions?"

    device = next(model.parameters()).device
    loss = torch.zeros(1, device=device).squeeze()

    for x_bc, y_target in bc_points:
        y_pred = model(x_bc)
        loss = loss + torch.mean((y_pred - y_target) ** 2)

    return loss

def total_loss(
        model: torch.nn.Module, 
        x_colloc: torch.Tensor, 
        ode_fn: Callable, 
        bc_points: list, 
        lambda_bc: float=10.0,
        )->tuple: 
    """
    Compute PINN total loss
    Args: 
        model: The PINN model
        x_colloc: Collocation points
        ode_fn: ODE residual function
        bc_points: Boundary conditions pairss
        lambda_bc: Weight for BC losss
    Returns: 
        (total_loss, ode_loss, bc_loss) - all scalar tensor
    """
    l_ode = ode_residual_loss(model, x_colloc, ode_fn)
    l_bc=boundary_condition_loss(model, bc_points)
    l_total=l_ode+lambda_bc*l_bc
    return l_total, l_ode, l_bc


