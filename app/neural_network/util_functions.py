import torch
from torch import autograd
import sympy as smp


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


def validate_input_dict(input_dict: dict[str]) -> None:
    """
    Validate the input dictionary against the specified requirements.

    Args:
        input_dict (Dict[str, str]): A dictionary with the following keys:
            'coefficient', 'left-x', 'right-x', 'end-t', 'init-boundary', 'left-boundary', 'right-boundary'
            All values in the dictionary are strings.

    Raises:
        ValueError: If any of the validation checks fail.

    Returns:
        None
    """
    t, x = smp.symbols('t x')

    # Check 'coefficient', 'left-x', and 'right-x' for float values
    for key in ['coefficient', 'left-x', 'right-x']:
        value = input_dict.get(key)
        if value is not None:
            try:
                float_value = float(value)
            except ValueError:
                raise ValueError(f"The value for '{key}' is not a valid float.")
        else:
            raise ValueError(f"The value for '{key}' can not be None.")
    
    # check left and right sides to be left-x < right-x
    if input_dict['left-x'] > input_dict['right-x']:
        raise ValueError(f'Left x {input_dict["left-x"]} should be less than right x {input_dict["right-x"]}')

    # Check 'end-t' for positive float value
    end_t_value = input_dict.get('end-t')
    if end_t_value is not None:
        try:
            end_t = float(end_t_value)
            if end_t <= 0:
                raise ValueError("The value for 'end-t' must be a positive float or None.")
        except ValueError:
            raise ValueError("The value for 'end-t' is not a valid float.")
    else:
        raise ValueError(f"The value for 'end-t' can not be None.")

    if input_dict.get('init-boundary') is None:
        raise ValueError(f"The value for initial condition can not be None.")
    
    # Check other keys for valid SymPy expressions
    for key in ['init-boundary', 'left-boundary', 'right-boundary']:
        value = input_dict.get(key)
        if value is not None:
            try:
                expr = smp.parsing.sympy_parser.parse_expr(value)
                lambdified_expr = smp.utilities.lambdify([t, x], expr)
                _ = lambdified_expr(3, 4)
            except (SyntaxError, TypeError, ValueError):
                raise ValueError(f"The value for '{key}' is not a valid SymPy expression.")
    

