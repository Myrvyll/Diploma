import torch
from torch import nn

from pde_solver import util_functions
from pde_solver import problem

class PhysicsLoss(nn.Module):
    """
    A custom loss function for physics-informed neural networks (PINNs) that calculates
    the residuals of a PDE solution.
    """

    def __init__(self):
        """
        Initializes the PhysicsLoss class.
        """
        super().__init__()

    def forward(self, ts, xs, y_pred, problem:problem.Heat, avg = nn.MSELoss()):
        """
        Forward pass to compute the physics-informed loss.

        Args:
            ts (torch.Tensor): Tensor representing the time components of the input.
            xs (torch.Tensor): Tensor representing the spatial components of the input.

            y_pred (torch.Tensor): Predicted output tensor from the neural network.

            problem (problem.Heat): An instance of the problem class that defines the PDE.
            
            avg (nn.Module): A loss function to compute the average residual, default is Mean Squared Error (MSE).

        Returns:
            torch.Tensor: The computed physics-informed loss.
        """

        dt = util_functions.derivative(ts, y_pred, order=1)
        dxx = util_functions.derivative(xs, y_pred, order=2)

        resudials = problem.right_side(dt) - problem.left_side(ts, xs, dxx)
        total_resudial = avg(resudials, torch.zeros_like(resudials))
        return total_resudial




