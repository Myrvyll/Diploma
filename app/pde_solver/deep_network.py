import torch
from torch import nn


class Approximator(nn.Module):
    def __init__(self, *,
                 num_inputs = 2,
                 num_neurons = 5,
                 num_layers = 3,
                 activation = nn.Tanh(),
                 num_outputs = 1):

        super().__init__()

        start_layer = nn.Linear(num_inputs, num_neurons)
        layers = [start_layer, activation]
        for _ in range(num_layers):
            layers.extend([nn.Linear(num_neurons, num_neurons), activation])

        # layers.extend([nn.Linear(num_neurons, num_neurons), nn.ReLU()])

        output_layer = nn.Linear(num_neurons, num_outputs)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, t, x):
        x = self.network(torch.vstack((t, x)).T)
        return torch.squeeze(x)