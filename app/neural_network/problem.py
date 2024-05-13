import sympy as smp
import torch

import util_functions

"""
    One-line summary of the class.

    Extended description of the class, providing more details
    about its purpose and functionality. This can span multiple
    lines.

    Attributes:
        attribute1 (type): Description of attribute1.
        attribute2 (type): Description of attribute2.
        ...

    Parameters:
        param1 (type): Description of parameter 1 for the constructor.
        param2 (type): Description of parameter 2 for the constructor.
        ...

    Examples:
        >>> # Example usage of the class
        >>> instance = ClassName(param1, param2)
        >>> result = instance.method(args)
        # Expected output or behavior

    Notes:
        Any additional notes or references related to the class.
    """



class Heat():
    ''' Клас для опису теплового одновимірного диференціального рівняння.
    u_t = a*u_xx + f(x, t)
    u(0, x) = g(x) - початкова умова
    u(t, a) = l(t)  - ліва крайова умова
    u(t, b) = r(t)  - права крайова умова
    '''       
    def __init__(self, 
                 alpha, 
                 initial_condition, 
                 left_boundary_condition, 
                 right_boundary_condition, 
                 left_x = 0, 
                 right_x = 1, 
                 start_t = 0, 
                 end_t = 1,
                 fn = None,  
                 solution = None) -> None:
        
        self.alpha = alpha            # coefficient in the equation
        self.fn = fn                  # component of heterogeneousity

        self.left_x = left_x          # ліва границя області по х
        self.right_x = right_x        # права границя області по х
        self.start_t = start_t        # start of time
        self.end_t = end_t            # end of time count

        self.initial_condition = initial_condition       # initial condition
        self.left_boundary = left_boundary_condition       # left boundary condition
        self.right_boundary = right_boundary_condition      # right boundary condition

        self.exact_solution = solution    # solution for checking an algorithm

        self._text = None                 # things for formatted printing
    
    @classmethod
    def from_string_dict(cls, string_dict):

        try:
            util_functions.validate_input_dict(string_dict)
        except e:
            
      
        
    def right_side(self, u_t):
        return u_t
        
    def left_side(self, ts, xs, u_xx):
        return self.alpha*u_xx #+ self.fn(ts, xs)