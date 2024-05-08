import torch
from torch import nn

import util_functions
import problem

class PhysicsLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, ts, xs, y_pred, problem:problem.Heat, avg = nn.MSELoss()):

        dt = util_functions.derivative(ts, y_pred, order=1)
        dxx = util_functions.derivative(xs, y_pred, order=2)

        resudials = problem.right_side(dt) - problem.left_side(ts, xs, dxx)
        total_resudial = avg(resudials, torch.zeros_like(resudials))
        return total_resudial




