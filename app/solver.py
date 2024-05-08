import torch
from torch import nn
import problem
import physics_loss


class HeatSolver:
    def __init__ (self, network: nn.Module, optimizer:torch.optim.Optimizer, loss:torch.nn.Module):

        self.network = network
        self.optimizer = optimizer
        self.loss_for_sp = loss

        self.task = None


    def generate_data(self, task:problem.Heat):

        self.t_train_inner = torch.rand(300, requires_grad=True)
        self.x_train_inner = torch.rand(300, requires_grad=True)

        # data in init
        self.x_train_init = torch.linspace(task.left_x, task.right_x, 20)
        self.t_train_init = torch.zeros_like(self.x_train_init)
        self.y_train_init = task.initial_condition(self.t_train_init, self.x_train_init)

        # data in left boundary
        self.t_train_left = torch.linspace(0, task.end_t, 20)
        self.x_train_left = torch.full_like(self.t_train_left, task.left_x)
        self.y_train_left = task.left_boundary(self.t_train_left, self.t_train_left)

        
        # data in right boundary
        self.t_train_right = torch.linspace(0, task.end_t, 20)
        self.x_train_right = torch.full_like(self.t_train_right, task.right_x)
        self.y_train_right = task.right_boundary(self.t_train_right, self.x_train_right)


    def fit(self, task: problem.Heat, epochs = 1000):

        self.generate_data(task)
        phys_loss_fn = physics_loss.PhysicsLoss()

        for epoch in range(epochs):

            self.network.train()

            y_pred_inner = self.network(self.t_train_inner, self.x_train_inner)

            y_pred_init = self.network(self.t_train_init, self.x_train_init)
            y_pred_left = self.network(self.t_train_left, self.x_train_left)
            y_pred_right = self.network(self.t_train_right, self.x_train_right)

            phys_loss = phys_loss_fn(self.t_train_inner, self.x_train_inner, y_pred_inner, task)

            init_loss = self.loss_for_sp(y_pred_init, self.y_train_init)
            left_loss = self.loss_for_sp(y_pred_left, self.y_train_left)
            right_loss = self.loss_for_sp(y_pred_right, self.y_train_right)

            total_loss = phys_loss + init_loss + left_loss + right_loss
            self.optimizer.zero_grad()

            total_loss.backward()

            self.optimizer.step()

            if epoch % (epochs/10) == 0:
               print(f'Loss phys: {phys_loss:.8f}| loss left: {left_loss:.8f} | loss right: {right_loss:.8f} | loss init: {init_loss:.8f} | Total: {total_loss}')
            
    def predict(self, *args):
        
        self.network.eval()
        with torch.inference_mode():
            ys = self.network(*args)

        return ys

        

        

