import torch
from torch import autograd
import sympy as smp



def is_not_empty(string):
    if (string is not None) and \
       (string != 'None') and \
       (string != ' ') and \
       (string != ''):
        return True
    else:
        return False


def get_attr_dictionary(form):
    
    attr_dict = {}
    for key, value in form.items():
        value = value.strip()
        if is_not_empty(value):
            attr_dict[key] = value
        else:
            attr_dict[key] = None
    return attr_dict


def derivative(xs, ys, order = 1):
    
    derivatives = [ys]
    for i in range(order):
        der = autograd.grad(derivatives[i], xs, 
                            grad_outputs=torch.ones_like(derivatives[i]), 
                            retain_graph=True, 
                            create_graph=True)[0]
        derivatives.append(der)
        
    return derivatives[order]



    

