import torch
from torch import autograd


def insert_newline(text):
    """
    Inserts a newline character (\n) after every 30 characters in the given text.
    
    Args:
        text (str): The input text to be modified.
        
    Returns:
        str: The modified text with newline characters inserted.
    """
    result = ""
    for i in range(len(text)):
        if i > 0 and i % 30 == 0:
            result += "\n"
        result += text[i]
    return result


def derivative(xs, ys, order = 1):
    
    derivatives = [ys]
    for i in range(order):
        der = autograd.grad(derivatives[i], xs, 
                            grad_outputs=torch.ones_like(derivatives[i]), 
                            retain_graph=True, 
                            create_graph=True)[0]
        derivatives.append(der)
        
    return derivatives[order]