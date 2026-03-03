"""
Training loops for PINNs. 

Two phases:
    1. Adam: Stochastic
    2. second-order and deterministic
Standard for PINN literature
"""
import torch
from tqdm import tqdm 
from typing import Callable
from losses import total_loss
from sampler import random_sampler

class PINNTrainer: 
    """
    Trainer for Physics-Informed Neural Networks.

    Args:
        model:       The PINN model
        ode_fn:      ODE residual function (x, y, dy_dx) -> residual
        bc_points:   List of (x_bc, y_target) boundary conditions
        domain:      (x_min, x_max) training domain
        n_colloc:    Number of collocation points per step
        lambda_bc:   Weight for BC loss
        device:      'cpu' or 'cuda'

    """
    def __init__(self, model: torch.nn.Module, ode_fn: Callable, bc_points: tuple(-2.0, 2.0), n_colloc: int=100, lambda_bc: float=10.0, device: str="cpu"): 
        self.model=model.to(device)
        self.ode_fn=ode_fn
        self.bc_points=[(x.to(device), y.to(device)) for x, y in bc_points]
        self.domain=domain
        self.n_colloc=n_colloc
        self.lambda_bc=lambda_bc
        self.device=device
        self.history={"total":[], "ode":[], "bc":[]}
    def _step(self) -> tuple:
        """Single training step: sample new collocation points and compute loss"""
        x=random_sampler(self.n_colloc, self.domain, self.device)
        return total_loss(self.model, x, self.ode_fn, self.bc_points, self.lambda_bc)
    def train_adam(self, epochs: int=500, lr: float=1e-3, log_every: int=500): 
        """
        Adam optimizer: 
        Args: 
            epochsL Number of Adam steps
            lr: Learning rate
            log_every: Prints loss every N steps
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        print(f"{'='*55}")
        print(f" Adam | epochs={epochs} | lr{lr}")
        print(f"{'='*55}")

        pbar=tqdm(range(epochs), desc="Adam", ncols=80)
        for epoch in pbar: 
            optimizer.zero_grad()
            l_total, l_ode, l_bc, self._step()
            l_total.backward()
            optimizer.step()
            scheduler.step()

            if epoch % log_every==0: 
                pbar.set_postfix({
                    "total":f"{l_total.item():.2e}", 
                    "ode":f"{l_ode.item():.2e}",
                    "bc": f"{l_bc.item():.2e}",
                    })
        print(f"Final Adam loss: {self.history['total'][-1]:.4e}\n")
    def train_lbfgs(self, max_iter: int=500, lr:float=1.00, log_every:int=100): 
        """
        L-BFGS optimizer
        Args: 
            max_iter: Maximum iterations
            lr: step size
            log_every: print losss every N steps
        """
        optimizer=torch.optim.LBFGS(
                self.model.parameters(), 
                lr=lr, 
                max_iter=max_iter, 
                tolerance_grad=1e-9, 
                tolerance_change=1e-11, 
                history_size=100, 
                line_search_fn="strong_wolfe",
                )
        print(f"{'='*55}")
        print(f" Optimizer | epochs={epochs} | lr{lr}")
        print(f"{'='*55}")
        step=[0]

        def closure(): 
            optimizer.zero_grad()
            l_total, l_ode, l_bc=self._step()
            l_total.backward()
            self.history["total"].append(l_total.item())
            self.history["ode"].append(l_ode.item())
            self.history["bc"].append(l_bc.item())
            if step[0]%log_every==0: 
                print(f" iter{step[0]:.4d} | total={l_total.item():4e} | ode={l_ode.item():.4e} | bc={l_bc.item():.4e}")
            step[0]+=1
            return l_total
        optimizer.step(closure)
        print(f"Final loss: {self.history['total'][-1]:.4e}\n")


