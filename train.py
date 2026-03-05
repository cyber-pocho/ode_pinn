"""
Main training point

Usage: 
    python train.py
    python train.py --problem exponential --epochs 5000 --lbfgs_iter 500
    python train.py --problem lotka_volterra --hidden 128 --layers 5
"""

import argparse
import torch
import numpy as np

from src.model import PINN
from src.trainer import PINNTrainer
from src.utils import plot_solution, plot_loss_history, compute_metrics

def parse_args(): 
    parser=argparse.ArgumentParser(description="PINN ODE solver")
    parser.add_argument("--problem", type=str, default="exponential", 
                        choices=["exponential", "lotka_volterra", "van_der_pol"]
                        )
    parser.add_argument("--hidden",      type=int,   default=64,   help="Hidden layer width")
    parser.add_argument("--layers",      type=int,   default=4,    help="Number of hidden layers")
    parser.add_argument("--epochs",      type=int,   default=5000, help="Adam epochs")
    parser.add_argument("--lr",          type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--lbfgs_iter",  type=int,   default=500,  help="L-BFGS iterations")
    parser.add_argument("--n_colloc",    type=int,   default=1000, help="Collocation points")
    parser.add_argument("--lambda_bc",   type=float, default=10.0, help="BC loss weight")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      type=str,   default="cpu")
    return parser.parse_args()
def load_problem(name: str): 
    """Load the ODE problem definition"""
    if name=="exponential": 
        from problems.exponential import ode_fn, boundary_conditions, exact_solution, DOMAIN
        return ode_fn, boundary_conditions(), exact_solution, DOMAIN, 1, 1
    elif name=="lotka_volterra": 
        from problems.lotka_volterra import ode_fn, boundary_conditions, reference_solution, DOMAIN
        return ode_fn, boundary_conditions(), reference_solution, DOMAIN, 1, 2
    elif name == "van_der_pol":
        from problems.van_der_pol import ode_fn, boundary_conditions, reference_solution, DOMAIN
        return ode_fn, boundary_conditions(), reference_solution, DOMAIN, 1, 2
    else:
        raise ValueError(f"Unknown problem: {name}")

def main(): 
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f" Problem: {args.problem}")
    print(f" Device: {args.device}")
    print(f" Network layers: {args.layers} layers x {args.hidden} neurons")

    ode_fn, bc_points, exact_fn, domain, input_dim, output_dim = load_problem(args.problem)
    model=PINN(
            input_dim=input_dim, 
            output_dim=output_dim, 
            hidden_dim=args.hidden, 
            num_layers=args.layers,
            )
    n_params=sum(p.numel() for p in model.parameters())
    print(f" Model parameters: {n_params:,}\\n")
    trainer = PINNTrainer(
        model=model,
        ode_fn=ode_fn,
        bc_points=bc_points,
        domain=domain,
        n_colloc=args.n_colloc,
        lambda_bc=args.lambda_bc,
        device=args.device,
    )
    trainer.train_adam(epochs=args.epochs, lr=args.lr)
    trainer.train_lbfgs(max_iter=args.lbfgs_iter)

    if args.problem=="exponential": 
        metrics=compute_metrics(model, exact_fn, domain)
        for k,v in metrics.items(): 
            print(f" {k:<22}{v:.4e}")
    if args.problem=="exponential":
        plot_solution(
                model, exact_fn, domain,
                title=f"PINN for the exponential problem"
                ) 
    plot_loss_history(trainer.history)

if __name__=="__main__": 
    main()
