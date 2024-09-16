import torch
from torch import nn


class Approximator(nn.Module):
    def __init__(self, *,
                 num_inputs = 2,     # Number of input features, default is 2
                 num_neurons = 5,    # Number of neurons in each hidden layer, default is 5
                 num_layers = 3,     # Number of hidden layers, default is 3
                 activation  =nn.Tanh(),  # Activation function, default is Tanh
                 num_outputs = 1):   # Number of output features, default is 1
        """
        Initializes the Approximator neural network.

        Args:
            num_inputs (int): Number of input features.
            num_neurons (int): Number of neurons in each hidden layer.
            num_layers (int): Number of hidden layers.
            activation (nn.Module): Activation function to use between layers.
            num_outputs (int): Number of output features.
        """

        super().__init__()  # Call the parent class (nn.Module) initializer

        start_layer = nn.Linear(num_inputs, num_neurons)
        layers = [start_layer, activation]

        # Add hidden layers and activation functions to the list
        for _ in range(num_layers):
            layers.extend([nn.Linear(num_neurons, num_neurons), activation])

        output_layer = nn.Linear(num_neurons, num_outputs)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, t, x):
        """
        Forward pass of the network.

        Args:
            t (torch.Tensor): Tensor representing the time component of the input.
            x (torch.Tensor): Tensor representing the spatial component of the input.

        Returns:
            torch.Tensor: The network output after processing the input tensors.
        """
        x = self.network(torch.vstack((t, x)).T)
        return torch.squeeze(x)