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

    # def __init__(self, json_data):
    #     ''' Reads from json with format:
    # {
    #     "number of example to use": 1,
    #     "examples": [

    #        {"left x": 0,       #number
    #         "right x": 2,      #number
    #         "start t": 0,      #number
    #         "end t": 1,        #number
        
    #         "left boundary condition": "0",       # sympy functions
    #         "right boundary condition": "0",
    #         "initial condition":"2*sin(pi*x/2) - sin(pi*x) + 4*sin(2*pi*x)",
    #         "solution":"2*sin(pi*x/2)*exp(-(pi**2)*t/16) - sin(pi*x)*exp(-(pi**2)*t/4) + 4*sin(2*pi*x)*exp(-(pi**2)*t)",
        
    #         "alpha": 0.25},        #also number
    #     ]
    # } 
    # '''
    #     try: 
    #         _ = json_data["number of example to use"] + 1
    #     except TypeError:
    #         raise TypeError("Number of example to use should be number")

    #     if json_data["number of example to use"] > len(json_data['examples']) or (json_data["number of example to use"] < 0):
    #         raise ValueError("Numbers of example to use is nonexistent.")
        

    #     json_data = json_data['examples'][json_data["number of example to use"]]
    #     # logger.debug(json_data)
        
    #     self.left_x = json_data["left x"]
    #     self.right_x = json_data["right x"]
        
    #     self.start_t = json_data["start t"]
    #     self.end_t = json_data["end t"]
    #     self.alpha = json_data["alpha"]
        
    #     # logger.debug(f"x0: {self.left_x}")
    #     # logger.debug(f"x1: {self.right_x}")
    #     # logger.debug(f"t0: {self.start_t}")
    #     # logger.debug(f"t1: {self.end_t}")
        
    #     x = smp.Symbol("x")
    #     t = smp.Symbol("t")
        
    #     try:
    #         self.left_boundary = smp.lambdify((x, t), json_data["left boundary condition"])
    #         self.right_boundary = smp.lambdify((x, t), json_data["right boundary condition"])
    #         self.initial_condition = smp.lambdify((x, t), json_data["initial condition"])
    #         if json_data['solution'] is not None:
    #             self.exact_solution = smp.lambdify((x, t), json_data["solution"])
    #             # logger.debug(f"solution from (3, 4): {self.solution(3, 4)}")
    #         else:
    #             self.exact_solution = None

    #     except NameError as e:
    #         # logger.error("Problems were encountered while reading given problem. All conditions should be functions from (x, t), no other letter variables allowed.")
    #         # logger.error(e)
    #         print(e)
    #     except SyntaxError as e:
    #         # logger.error("Syntax error. Probably, there is at least 1 unclosed parenthesis.")
    #         # logger.error(e)
    #         print(e)

        # self._text = [util_functions.insert_newline(json_data["left boundary condition"]), 
        #              util_functions.insert_newline(json_data["right boundary condition"]), 
    #                  util_functions.insert_newline(json_data["initial condition"])]
        
    def right_side(self, u_t):
        return u_t
        
    def left_side(self, ts, xs, u_xx):
        return self.alpha*u_xx #+ self.fn(ts, xs)