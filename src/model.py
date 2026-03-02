import torch 
import torch.nn as nn

class PINN(nn.Module):
    """
    Physics Informed Neural Network
    Args: 
        input_dim: Dimensions of the inputs
        output_dim: Dimensions of the output
        hidden_dim: Width of the hidden layer
        num_layers: Number of hidden layers
    """
    def __init__(self, input_dim: int = 1, output_dim int=1, hidden_dim: int=64, num_layers: int=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers-1):
            layers += [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()
    def _init_weights(self):
        """
        Initialization of weights
        """
        for layer in self.net: 
            if isinstance(layer, nn.Linear): 
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.net(x)

