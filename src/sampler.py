import torch
import numpy as np

def uniform_sampler(n_points: int, domain:tuple, device:str = "cpu") -> torch.Tensor: 
    """
    Collection of points uniformly distributed
    Args: 
        n_points: Number of points
        domain: (x_min, x_max)
        device: torch device
    Returns: 
        Tensor of shape(n_points, 1) with requires_grad=True
    """
    x=torch.linspace(domain[0], domain[1], n_points, device=device).unsqueeze(1)
    x.requires_grad_(True)
    return x
def random_sampler(n_points:int, domain:tuple, device:str="cpu") -> torch.Tensor: 
    """
    Random collection of points
        Same arguments and returns as uniform_sampler
    """
    x_min, x_max = domain
    x = torch.FloatTensor(n_points, 1).uniform_(x_min, x_max).to(device)
    x.requires_grad(True)
    return x
def latin_hypercube_sampler(n_points: int, domain:tuple, device: str="cpu")->torch.Tensor: 
    """
        Latin Hypercube Sampling
        Args:
            n_points: Number of points
            domain:   (x_min, x_max)
            device:   torch device

        Returns:
            Tensor of shape (n_points, 1) with requires_grad=True
    """
    x_min, x_max = domain
    intervals=np.linspace(0, 1, n_points+1)
    points=np.random.uniform(intervals[:1], intervals[1:])
    np.random.shuffle(points)
    x=torch.tensor(points*(x_max-x_min)+x_min, dtype=torch.float32, device=device).unsqueeze(1)
    x.requires_grad_(True)
    return x

