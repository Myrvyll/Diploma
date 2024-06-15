import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import torch.nn as nn


from pde_solver import problem

import logging
formatter = logging.Formatter('%(asctime)s:%(filename)s:%(levelname)s:\n%(message)s')
# formatter = logging.Formatter('%(filename)s:\n%(message)s')

logger = logging.getLogger("solver_fdm")

class FDMHeatSolver:
    """
    Class to solve the heat equation using the Finite Difference Method (FDM).
    """
    def __init__(self, task:problem.Heat):
        """
        Initializes the FDMHeatSolver class.

        Args:
            task (Heat): An instance of the Heat class representing the heat equation problem.
        """
        self.task = task
        self.ts = None
        self.xs = None

    def solve(self, t_n, x_n):
        """
        Solves the heat equation using the Crank-Nicolson method.

        Args:
            t_n (int): Number of time steps.
            x_n (int): Number of spatial steps.
        
        Returns:
            None
        """

        self.ts = torch.linspace(self.task.start_t, self.task.end_t, t_n)
        self.xs = torch.linspace(self.task.left_x, self.task.right_x, x_n)
        # logger.debug('Krank-Nicolson: ts, xs')
        # logger.debug(self.ts)
        # logger.debug(self.xs)
    
        dt = self.ts[1] - self.ts[0]
        dx = self.xs[1] - self.xs[0]
    
        F = self.task.alpha * dt / pow(dx, 2)
        logger.info(str(F) + ' is F value')
    
        if F > 0.5:
            logger.warning("It is not recommended to use dt, dx such that f > 1/2")

        init_t = torch.full_like(self.xs, self.task.start_t)
        u0 = self.task.initial_condition(init_t, self.xs)
        # logger.debug(u0)
    
        u_p = u0.clone().detach()
    
        u_approximated = [u_p]
        # logger.debug('INIT U approximation')
        # logger.debug(u_approximated)
    
        a_sides = -0.5*F
        a_center = 1+F 
        
        upper_diagonal = np.full(x_n-1, a_sides)
        upper_diagonal[0] = 0
    
        diagonal = np.full(x_n, a_center)
        diagonal[0] = 1
        diagonal[-1] = 1
    
        lower_diagonal = np.full(x_n-1, a_sides)
        lower_diagonal[-1] = 0
    
        A_sparse = sp.sparse.diags(diagonals=[upper_diagonal, diagonal, lower_diagonal], 
                                   offsets = [1, 0, -1], format='csr')
        left_x = torch.tensor(self.task.left_x).unsqueeze(0)
        right_x = torch.tensor(self.task.right_x).unsqueeze(0)
    
        for t in self.ts[1:]:
    
            right_side = torch.zeros_like(u0)
            u_c = torch.zeros_like(u0)
            t = t.unsqueeze(0)
    
            right_side[1:-1] = u_p[1:-1] + 0.5*F*(u_p[:-2] - 2*u_p[1:-1] + u_p[2:])
            right_side[0] = self.task.left_boundary(t, left_x)
            right_side[-1] = self.task.right_boundary(t, right_x)
    
            u_c = torch.tensor(sp.sparse.linalg.spsolve(A_sparse, right_side))
    
            u_approximated.append(u_c)
            u_p, u_c = u_c, u_p
            
    
        u_approximated = np.array(u_approximated)

        self.solution = u_approximated

    def plot(self, path):
        """
        Plots the computed solution and saves the plots to the specified path.

        Args:
            path (str): The path where the plot images will be saved.
        
        Returns:
            None
        """

        plt.tight_layout()
    
        x = self.xs
        t = self.ts
        T, X = torch.meshgrid(t, x, indexing="ij")
        # data = torch.cartesian_prod(t, x)
        # cmap = plt.colormaps['Reds'](28)
        fig, axes = plt.subplots(1, subplot_kw={"projection": "3d"})
    
        u = self.solution
        for angle in range(0, 180, 60):
        # for angle in range(40, 80, 20):
            axes.view_init(azim=(angle-90), elev=30)
            axes.plot_surface(X, T, u, cmap=plt.cm.coolwarm)
            axes.set_title("Скінченні різниці")
            axes.set_xlabel("X")
            axes.set_ylabel("T")
            fig.savefig(path + f"{angle}")
            plt.close()


    def count_metrics(self, average_over_n):
        """
        Calculates error metrics (MSE, MAE) for the computed solution against the exact solution.

        Args:
            average_over_n (int): Number of points over which to average the error metrics.
        
        Returns:
            dict: A dictionary containing the calculated MSE, MAE.
        
        Raises:
            ValueError: If the exact solution is not provided in the Heat problem instance.
        """
        
        if self.task.exact_solution is None:
            raise ValueError("There is no exact solution in this problem.")

        t_check, x_check = torch.meshgrid(self.ts, self.xs, indexing="ij")
        t_check = t_check.flatten()
        x_check = x_check.flatten()
    

        y_check = torch.tensor(self.solution)
        y_GT = self.task.exact_solution(t_check, x_check)
        y_GT = y_GT.reshape(len(self.ts), len(self.xs))

        
        lossL2 = nn.MSELoss()
        lossL1 = nn.L1Loss()

        mse = lossL2(y_check, y_GT)
        mae = lossL1(y_check, y_GT)

        mrae = ((y_check - y_GT).abs())/(y_GT.abs())
        mask = torch.isnan(mrae)
        mrae[mask] = y_check[mask]
        mrae = (mrae.sum())/average_over_n

        return {'MSE': mse.item(),
                'MAE': mae.item()}
