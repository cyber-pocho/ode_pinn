import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_solution(
        model: torch.nn.Module, 
        exact_fn, 
        domain: tuple=(-2.0,2.0), 
        n_plot: int=500, 
        save_dir: str="results",
        title: str="PINN solution",
        ): 
    """
    Plot the PINN solution vs the exact analytical solution
    Args: 
        model: Trained PINN
        exact_fn: Function x - Exact
        domain: (x_min, x_max)
        n_plot: Number of plot points
        save_dir: Directory to save the figure
        title: Title plot
    """
    os.makedirs(save_dir, exist_ok=True)
    x_np=np.linspace(domain[0], domain[1], n_plot)
    x_t=torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(x_t).squeeze().numpy()
    model.train()

    y_exact = exact_fn(x_np)
    error = np.abs(y_pred - y_exact)
    fig, axes = plt.subplots(1, 2, figsize(14, 4))

    #solution
    ax=axes[0]
    ax.plot(x_np, y_exact, "k--", linewidth=1.5, label="Exact solution")
    ax.plot(x_np, y_pred, "r--", linewidth=1.0, label="Predicted solution")
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel("$y$", fontsize=11)
    ax.set_title("Solution", fontsize-14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    #error plot
    ax=axes[1]
    ax.semilogy(x_np, error, "b-", linewidth=2)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("|$y_{pred}-y_{exact}$|", fontsize=11)
    ax.set_title("Pointwise absolute error", fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path=ps.path.join(save_dir, "solution.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_loss_history(history:dict, save_dir:str="results"): 
    """
    Plot training loss curves on a log scale
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax=plt.subplots(figsize=(14,4))
    ax.semilogy(history["total"], label="Total loss", linewidth=1)
    ax.semilogy(history["ode"], label="ODE residual", linewidth=1, linestyle='--')
    ax.semilogy(history["bc"], label="Boundary condition", linewidth=1, linestyle=':')
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Loss (log scale)", fontsize=11)
    ax.set_title("Training History", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    path=os.path.join(save_dir, "loss_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()

def compute_metrics(
        model: torch.nn.Module, 
        exact_fn, 
        domain: tuple=(-2.0, 2.0), 
        n_points: int=1000, 
        )-> dict:
    """
    Compute accuracy metrics against the exact solution. 
    Returns: 
        Dict with max_abs_error, mean_abs_error, l2_relative_error
    """
    x_np=np.linspace(domain[0], domain[1], n_points)
    x_t=torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)

    model.eval()
    with torch.no_grad(): 
        y_pred=model(x_t).squeeze().numpy()
    model.train()

    y_exact=exact_fn(x_np)
    error=np.abs(y_pred-y_exact)
    metrics={
            "max_abs_error": float(error.max()), 
            "mean_abs_error": float(error.mean()), 
            "l2_relative_error": float(
                np.linalg.norm(y_pred-y_exact)/np.linalg.norm(y_exact)
                ),
            }
    return metrics

