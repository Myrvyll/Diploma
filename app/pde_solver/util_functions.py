import torch
from torch import autograd
import math



def is_not_empty(string):
    """
    Checks if a given string is not empty or null.

    This function evaluates if a string is not None, not equal to 'None', 
    not just a space, and not an empty string.

    Args:
        string (str): The string to check.

    Returns:
        bool: True if the string is not empty or null, False otherwise.
    """
    if (string is not None) and \
       (string != 'None') and \
       (string != ' ') and \
       (string != ''):
        return True
    else:
        return False


def get_attr_dictionary(form):
    """
    Creates a dictionary from form data, setting empty or invalid values to None.

    This function iterates over the key-value pairs in the form, trims whitespace
    from the values, and uses `is_not_empty` to determine if the value should be 
    included as-is or set to None.

    Args:
        form (dict): A dictionary representing form data.

    Returns:
        dict: A dictionary with cleaned values from the form.
    """
    attr_dict = {}
    for key, value in form.items():
        value = value.strip()
        if is_not_empty(value):
            attr_dict[key] = value
        else:
            attr_dict[key] = None
    return attr_dict


def derivative(xs, ys, order = 1):
    """
    Computes the derivative of a tensor with respect to another tensor.

    This function uses PyTorch's autograd to compute the derivative of `ys` with
    respect to `xs`. It allows for computing higher-order derivatives by specifying
    the order.

    Args:
        xs (torch.Tensor): The input tensor with respect to which the derivative is computed.
        ys (torch.Tensor): The output tensor for which the derivative is computed.
        order (int, optional): The order of the derivative to compute. Default is 1.

    Returns:
        torch.Tensor: The computed derivative of the specified order.
    """
    derivatives = [ys]
    for i in range(order):
        der = autograd.grad(derivatives[i], xs, 
                            grad_outputs=torch.ones_like(derivatives[i]), 
                            retain_graph=True, 
                            create_graph=True)[0]
        derivatives.append(der)
        
    return derivatives[order]

def get_batch_idx(epoch, batch_size, dataset_size):
    """
    Calculates the start and end indices for a batch of data.

    This function computes the indices for the current batch based on the epoch number,
    batch size, and total dataset size. It ensures that the indices are within the valid range.

    Args:
        epoch (int): The current epoch number.
        batch_size (int): The size of the batch.
        dataset_size (int): The total size of the dataset.

    Returns:
        tuple: A tuple containing the start and end indices for the batch.
    """

    batches_quantity = math.ceil(dataset_size/batch_size)
    batch_num = epoch % batches_quantity

    start = batch_num*batch_size
    end = ((batch_num+1)*batch_size) - 1

    if end >= dataset_size:
        end = dataset_size - 1

    return (start, end)



    

