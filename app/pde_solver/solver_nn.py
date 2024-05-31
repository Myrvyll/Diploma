import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


from pde_solver import problem
from pde_solver import physics_loss


import torch
import torch.nn as nn
import torch.optim

import logging
formatter = logging.Formatter('%(asctime)s:%(filename)s:%(levelname)s:\n%(message)s')
# formatter = logging.Formatter('%(filename)s:\n%(message)s')

logger = logging.getLogger("solver_nn")

# file_handler = logging.FileHandler('logs/logs_fd_pde.log')
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setFormatter(formatter)

logger.setLevel("DEBUG")
logger.addHandler(ch)
# logger.addHandler(file_handler)

class NNHeatSolver:
    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer, loss: torch.nn.Module):
        self.network = network
        self.optimizer = optimizer
        self.loss_for_sp = loss

        self.task = None
        
        self.t_train_inner = None
        self.x_train_inner = None

        # data in init
        self.x_train_init = None
        self.t_train_init = None
        self.y_train_init = None

        # data in left boundary
        self.t_train_left = None
        self.x_train_left = None
        self.y_train_left = None

        # data in right boundary
        self.t_train_right = None
        self.x_train_right = None
        self.y_train_right = None

        self.loss_values = None
        self.learning_rates = []
        self.y_min = None
        self.y_max = None
        self.test_data = {}

    def generate_test_data(self, task: problem.Heat):
        
        # m = torch.distributions.Exponential(torch.tensor([7.0]))
        # t_train_inner_exp = m.rsample(torch.Size([60])).squeeze()
        t_train_inner_un = torch.rand(80) * (task.end_t - task.start_t) + task.start_t

        # self.t_train_inner = torch.cat([t_train_inner_exp, t_train_inner_un])
        # self.t_train_inner = t_train_inner_exp
        self.test_data['t_inner'] = t_train_inner_un
        self.test_data['x_inner'] = torch.rand(80) * (task.right_x - task.left_x) + task.left_x

        # data in init
        self.test_data['x_init'] = torch.linspace(task.left_x, task.right_x, 50)
        self.test_data['t_init'] = torch.zeros_like(self.x_train_init)
        self.test_data['y_init'] = task.initial_condition(self.t_train_init, self.x_train_init)

        # data in left boundary
        self.test_data['t_left'] = torch.linspace(task.start_t, task.end_t, 20)
        self.test_data['x_left'] = torch.full_like(self.t_train_left, task.left_x)
        self.test_data['y_left'] = task.left_boundary(self.t_train_left, self.x_train_left)

        # data in right boundary
        self.test_data['t_right'] = torch.linspace(task.start_t, task.end_t, 20)
        self.test_data['x_right'] = torch.full_like(self.t_train_right, task.right_x)
        self.test_data['y_right'] = task.right_boundary(self.t_train_right, self.x_train_right)


    def generate_data(self, task: problem.Heat):
        
        m = torch.distributions.Exponential(torch.tensor([7.0]))
        # t_train_inner_exp = m.rsample(torch.Size([60])).squeeze()
        t_train_inner_un = torch.rand(80) * (task.end_t - task.start_t) + task.start_t

        # self.t_train_inner = torch.cat([t_train_inner_exp, t_train_inner_un])
        # self.t_train_inner = t_train_inner_exp
        self.t_train_inner = t_train_inner_un
        self.x_train_inner = torch.rand(80) * (task.right_x - task.left_x) + task.left_x

        # data in init
        self.x_train_init = torch.linspace(task.left_x, task.right_x, 50)
        self.t_train_init = torch.zeros_like(self.x_train_init)
        self.y_train_init = task.initial_condition(self.t_train_init, self.x_train_init)

        # data in left boundary
        self.t_train_left = torch.linspace(task.start_t, task.end_t, 20)
        self.x_train_left = torch.full_like(self.t_train_left, task.left_x)
        self.y_train_left = task.left_boundary(self.t_train_left, self.x_train_left)

        # data in right boundary
        self.t_train_right = torch.linspace(task.start_t, task.end_t, 20)
        self.x_train_right = torch.full_like(self.t_train_right, task.right_x)
        self.y_train_right = task.right_boundary(self.t_train_right, self.x_train_right)

        # Collect all the y-values to find the min and max
        all_y_values = torch.cat([self.y_train_init, self.y_train_left, self.y_train_right])

        self.y_min = torch.min(all_y_values)
        self.y_max = torch.max(all_y_values)

    def rescale_output(self, y_normalized):
        # Rescale the output back to the original scale
        return y_normalized * (self.y_max - self.y_min) + self.y_min

    def fit(self, task: problem.Heat, epochs=10000):

        self.task = task
        phys_loss_fn = physics_loss.PhysicsLoss()


        # lr_adjuster = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[epochs*0.5], gamma = 0.1)
        self.generate_data(task)
        self.generate_test_data(task)
        self.t_train_inner.requires_grad = True
        self.x_train_inner.requires_grad = True
        self.test_data['t_inner'].requires_grad = True
        self.test_data['x_inner'].requires_grad = True


        loss_values = {'physics_loss':[], 
                       'init_loss':[], 
                       'left_loss':[],
                       'right_loss':[], 
                       'total_loss': []}
        
        loss_values_test = {'physics_loss':[], 
                       'init_loss':[], 
                       'left_loss':[],
                       'right_loss':[], 
                       'total_loss': []}

        for epoch in range(epochs):

            self.network.train()

            # t_id = torch.randint(0, len(self.t_train_inner), (300,))
            # x_id = torch.randint(0, len(self.x_train_inner), (300,))
            self.optimizer.zero_grad()
            
            # training
            y_pred_inner = self.predict_in_training(self.t_train_inner, self.x_train_inner)
            y_pred_init = self.predict_in_training(self.t_train_init, self.x_train_init)
            y_pred_left = self.predict_in_training(self.t_train_left, self.x_train_left)
            y_pred_right = self.predict_in_training(self.t_train_right, self.x_train_right)

            phys_loss = phys_loss_fn(self.t_train_inner, self.x_train_inner, y_pred_inner, task)
            init_loss = self.loss_for_sp(y_pred_init, self.y_train_init)
            left_loss = self.loss_for_sp(y_pred_left, self.y_train_left)
            right_loss = self.loss_for_sp(y_pred_right, self.y_train_right)

            loss_values['physics_loss'].append(phys_loss.item())
            loss_values['init_loss'].append(init_loss.item())
            loss_values['left_loss'].append(left_loss.item())
            loss_values['right_loss'].append(right_loss.item())

            total_loss = phys_loss + 75*init_loss + left_loss + right_loss
            loss_values['total_loss'].append(total_loss.item())


            total_loss.backward()

            self.optimizer.step()
            # self.learning_rates.append(lr_adjuster.get_last_lr())
            if epoch % (epochs / 10) == 0:
                logger.info(f'{epoch}/{epochs}|Loss phys: {phys_loss:.8f} | loss left: {left_loss:.8f} | loss right: {right_loss:.8f} | loss init: {init_loss:.8f} | Total: {total_loss:.8f}')
            # lr_adjuster.step()
            self.network.eval()

            y_pred_inner = self.predict_in_training(self.test_data['t_inner'], self.test_data['x_inner'])
            y_pred_init = self.predict_in_training(self.test_data['t_init'], self.test_data['x_init'])
            y_pred_left = self.predict_in_training(self.test_data['t_left'], self.test_data['x_left'])
            y_pred_right = self.predict_in_training(self.test_data['t_right'], self.test_data['x_right'])

            phys_loss = phys_loss_fn(self.test_data['t_inner'], self.test_data['x_inner'], y_pred_inner, task)
            init_loss = self.loss_for_sp(y_pred_init, self.test_data['y_init'])
            left_loss = self.loss_for_sp(y_pred_left, self.test_data['y_left'])
            right_loss = self.loss_for_sp(y_pred_right, self.test_data['y_right'])

            total_loss = phys_loss + 75*init_loss + left_loss + right_loss
            total_loss.backward()

            loss_values_test['physics_loss'].append(phys_loss.item())
            loss_values_test['init_loss'].append(init_loss.item())
            loss_values_test['left_loss'].append(left_loss.item())
            loss_values_test['right_loss'].append(right_loss.item())
            loss_values_test['total_loss'].append(total_loss.item())

            # self.test_data['t_inner'].grad.zeros_()
            # self.test_data['x_inner'].grad.zeros_()
            

            if epoch % (epochs / 10) == 0:
                self.generate_data(task)
                self.t_train_inner.requires_grad = True
                self.x_train_inner.requires_grad = True
                logger.info(f'{epoch}/{epochs} Test |Loss phys: {phys_loss:.8f} | loss left: {left_loss:.8f} | loss right: {right_loss:.8f} | loss init: {init_loss:.8f} | Total: {total_loss:.8f}')

            
            self.loss_values = loss_values
            self.loss_values_test = loss_values_test

    def predict_in_training(self, t, x):

        x = (x - self.task.left_x) / (self.task.right_x - self.task.left_x)
        t = (t - self.task.start_t) / (self.task.end_t - self.task.start_t)
        ys_normalized = self.network(t, x)
        ys = ys_normalized
        ys = self.rescale_output(ys_normalized)
        return ys


    def predict(self, t, x):

        self.network.eval()
        with torch.inference_mode():

            x = (x - self.task.left_x) / (self.task.right_x - self.task.left_x)
            t = (t - self.task.start_t) / (self.task.end_t - self.task.start_t)

            ys_normalized = self.network(t, x)
            ys = ys_normalized
            ys = self.rescale_output(ys_normalized)
            return ys
    
    def plot(self, path):

        plt.tight_layout()
        mesh = 1000
    
        x = torch.linspace(self.task.left_x, self.task.right_x, mesh)
        t = torch.linspace(0, self.task.end_t, mesh)
        T, X = torch.meshgrid(t, x, indexing="ij")
        # data = torch.cartesian_prod(t, x)
        # cmap = plt.colormaps['Reds'](28)
        fig, axes = plt.subplots(1, subplot_kw={"projection": "3d"})
        
        with torch.inference_mode():
            u = self.predict(T.flatten(), X.flatten())
        
        u = u.reshape((mesh, mesh))
        for angle in range(0, 180, 60):
        # for angle in range(40, 80, 20):
            axes.view_init(azim=angle-90, elev=30)
            axes.plot_surface(X, T, u, cmap=plt.cm.coolwarm)
            axes.set_title("Нейронна мережа")
            axes.set_xlabel("X")
            axes.set_ylabel("T")
            fig.savefig(path + f"{angle}")
            plt.close()
        # for xs in ys_to_plot:
        #     line = axes.plot(x_train_1d, xs)
        #     # line[0].set_color(color)
        # axes.set_zlim(-1, 1)
        # axes.set_ylim(-1, 1)

    def plot_loss(self, path):

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        indexes = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2]]

        for (key, value), id in zip(self.loss_values.items(), indexes):
            ax = axes[id[0], id[1]]
            ax.set_title(key)
            ax.plot(value)
            ax.plot(self.loss_values_test[key])
        
        axes[1][2].plot(self.learning_rates)
        axes[1][2].set_title("Learning rate")

        fig.savefig(path)
        # plt.close()

    def count_metrics(self, n_points):
        
        if self.task.exact_solution is None:
            raise ValueError("There is no exact solution in this problem.")

        t_check = torch.rand(n_points) * (self.task.end_t - self.task.start_t) + self.task.start_t
        x_check = torch.rand(n_points) * (self.task.right_x - self.task.left_x) + self.task.left_x

        y_check = self.predict(t_check, x_check)

        y_GT = self.task.exact_solution(t_check, x_check)

        lossL2 = nn.MSELoss()
        lossL1 = nn.L1Loss()

        mse = lossL2(y_check, y_GT)
        mae = lossL1(y_check, y_GT)

        mrae = ((y_check - y_GT).abs())/(y_GT.abs())
        mask = torch.isnan(mrae)
        mrae[mask] = y_check[mask]
        mrae = (mrae.sum())/n_points

        return {'MSE': mse.item(),
                'MAE': mae.item(),
                'MRAE': mrae.item()}














    


    






        

        

