import sympy as smp
import torch
import matplotlib.pyplot as plt

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
                 solution = None,
                 expressions = None) -> None:
        
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

        self._expressions = expressions           # things for formatted printing
    
    @classmethod
    def from_string_dict(cls, string_dict):
        '''
        Converts from dict with structure:

        {'coefficient': '1', 
        'left-x': '0', 
        'right-x': '1', 
        'end-t': '1', 
        'init-boundary': 'sin(x)', 
        'left-boundary': '0', 
        'right-boundary': '0'}


        '''
        cls.validate_input_dict(string_dict)

        t, x = smp.symbols('t x')

        temp_dict = {}
        for key in ['init-boundary', 'left-boundary', 'right-boundary', 'solution']:
            if string_dict[key] is not None:
                try:
                    numba = float(string_dict[key])
                    expr = torch.vmap(lambda t, x: torch.tensor(numba, dtype=torch.float32))
                except Exception:
                    expr = smp.parsing.sympy_parser.parse_expr(string_dict[key])
                    expr = smp.utilities.lambdify([t, x], expr)
            temp_dict[key] = expr


        return cls(alpha = float(string_dict['coefficient']), 
                   initial_condition = temp_dict['init-boundary'], 
                   left_boundary_condition = temp_dict['left-boundary'], 
                   right_boundary_condition = temp_dict['right-boundary'], 
                   left_x = float(string_dict['left-x']), 
                   right_x = float(string_dict['right-x']), 
                   start_t = 0, 
                   end_t = float(string_dict['end-t']),
                   solution = temp_dict['solution'],
                   expressions = string_dict)  
    
    def to_latex(self):

        temp_dict = self._expressions.copy()
        for key in ['init-boundary', 'left-boundary', 'right-boundary']:
            try:
                numba = float(temp_dict[key])
            except ValueError:
                expr = smp.parsing.sympy_parser.parse_expr(temp_dict[key])
                temp_dict[key] = smp.printing.latex(expr)
        return temp_dict

    def __str__(self):
        return str(self._expressions)
        
    def right_side(self, u_t):
        return u_t
        
    def left_side(self, ts, xs, u_xx):
        return self.alpha*u_xx #+ self.fn(ts, xs)
    
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
        for key in ['init-boundary', 'left-boundary', 'right-boundary', 'solution']:
            value = input_dict.get(key)
            if value is not None:
                try:
                    expr = smp.parsing.sympy_parser.parse_expr(value)
                    lambdified_expr = smp.utilities.lambdify([t, x], expr)
                    _ = lambdified_expr(3, 4)
                except (SyntaxError, TypeError, ValueError):
                    raise ValueError(f"The value for '{key}' is not a valid SymPy expression.")
                

    def plot_solution(self, path):

        if self.exact_solution is None:
            return None

        plt.tight_layout()
        mesh = 1000
    
        x = torch.linspace(self.left_x, self.right_x, mesh)
        t = torch.linspace(0, self.end_t, mesh)
        T, X = torch.meshgrid(t, x, indexing="ij")
        # data = torch.cartesian_prod(t, x)
        # cmap = plt.colormaps['Reds'](28)
        _, axes = plt.subplots(1, subplot_kw={"projection": "3d"})
        
        u = self.exact_solution(T.flatten(), X.flatten())
        u = u.reshape((mesh, mesh))

        for angle in range(0, 180, 60):
        # for angle in range(40, 80, 20):
            axes.view_init(azim=angle-90, elev=30)
            axes.plot_surface(X, T, u, cmap=plt.cm.coolwarm)
            axes.set_title("Точний розв'язок")
            plt.savefig(path + f"{angle}")