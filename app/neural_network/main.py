import torch
from torch import nn
# import json
import math
import matplotlib.pyplot as plt

import deep_network
# import physics_loss
import solver
import problem
import solver
# import util_functions


if  __name__ == "__main__":
    network = deep_network.Approximator(num_neurons=32)
    optimizer = torch.optim.Adam(network.parameters(), lr =0.003)
    loss = nn.MSELoss()
    
    answer = solver.HeatSolver(network=network, 
                               optimizer=optimizer, 
                               loss=loss)
    
    # data = []
    # with open("app_inputs\input_problem.json") as f:
    #     data = json.load(f)
    
    task = problem.Heat(alpha = 1, 
                        initial_condition = lambda t, x: 3*torch.pow(torch.sin(x*2*math.pi), 3), 
                        left_boundary_condition = torch.vmap(lambda t, x: torch.tensor(0, dtype=torch.float32)), 
                        right_boundary_condition = torch.vmap(lambda t, x: torch.tensor(0, dtype=torch.float32)), 
                        left_x = 0, 
                        right_x = 1, 
                        start_t = 0, 
                        end_t = 1,
                        fn = torch.vmap(lambda t, x: torch.tensor(0, dtype=torch.float32)),  
                        solution = None)
    
    answer.fit(task=task, epochs = 3000)
    mesh = 1000

    x = torch.linspace(0, 1, mesh)
    t = torch.linspace(0, 0.06, mesh)
    T, X = torch.meshgrid(t, x, indexing="ij")
    # data = torch.cartesian_prod(t, x)
    # cmap = plt.colormaps['Reds'](28)
    _, axes = plt.subplots(1, subplot_kw={"projection": "3d"})
    
    with torch.inference_mode():
        u = answer.predict(T.flatten(), X.flatten())
    
    u = u.reshape((mesh, mesh))
    
    axes.plot_surface(X, T, u, cmap=plt.cm.coolwarm)
    # for xs in ys_to_plot:
    #     line = axes.plot(x_train_1d, xs)
    #     # line[0].set_color(color)
    # axes.set_zlim(-1, 1)
    # axes.set_ylim(-1, 1)
    plt.show()
    