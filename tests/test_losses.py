"""
Unit tests for loss functions
"""
import torch
import pytest
from src.model import PINN
from src.losses import compute_gradient, ode_residual_loss, boundary_condition_loss

def test_compute_gradient_linear():
    x=torch.linspace(-1,1,50).unsqueeze(1).requires_grad_(True)
    y=2.0*x
    dy_dx=compute_gradient(y,x)
    assert torch.allclose(dy_dx, torch.full_like(dy_dx,2.0), atol=1e-5)
def test_compute_gradient_quadratic():
    x = torch.linspace(-2, 2, 100).unsqueeze(1).requires_grad_(True)
    y = x ** 2
    dy_dx = compute_gradient(y, x)
    expected = 2.0 * x.detach()
    assert torch.allclose(dy_dx.detach(), expected, atol=1e-4)
def test_ode_resexact():
    x=torch.linspace(-2,2,200).unsqueeze(1).requires_grad_(True)
    class ExactModel(torch.nn.Module):
        def forward(self, x):
            return torch.exp(-x**2)
    model=ExactModel()
    def ode_fn(x, y, dy_dx):
        return dy_dx+2.0*x*y
    loss=ode_residual_loss(model,x,ode_fn)
    assert loss.item() < 1e-6, f"Expected near-zero residual"
def test_boundary_conditions_loss_zero():
    class ConstantModel(torch.nn.Module):
        def forward(self, x): 
            return torch.ones_like(x)
    model=ConstantModel()
    x_bc=torch.tensor([[0.0]])
    y_bc=torch.tensor([[1.0]])
    loss=boundary_condition_loss(model, [(x_bc, y_bc)])
    assert loss.item() < 1e-8

def test_pinn_forward_shape():
    model=PINN(input_dim=1, output_dim=1, hidden_dim=32, num_layers=3)
    x=torch.randn(100,1)
    y=model(x)
    assert y.shape == (100, 1)


