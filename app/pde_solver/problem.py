import sympy as smp
import torch
import matplotlib.pyplot as plt


class Heat():
    ''' 
    Class to describe a one-dimensional heat differential equation.
    
    u_t = a*u_xx + f(x, t)
    u(0, x) = g(x)  - initial condition
    u(t, a) = l(t)  - left boundary condition
    u(t, b) = r(t)  - right boundary condition
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
        """
        Initializes the Heat class.

        Args:
            alpha (float): Coefficient in the heat equation.

            initial_condition (callable): Function representing the initial condition g(x).
            left_boundary_condition (callable): Function representing the left boundary condition l(t).
            right_boundary_condition (callable): Function representing the right boundary condition r(t).
            
            left_x (float): Left boundary of the spatial domain, default is 0.
            right_x (float): Right boundary of the spatial domain, default is 1.
            start_t (float): Start time, default is 0.
            end_t (float): End time, default is 1.
            fn (callable, optional): Function representing the heterogeneity f(x, t), default is None.
            solution (callable, optional): Exact solution for verification, default is None.
            expressions (dict, optional): Dictionary for formatted printing, default is None.
        """
        
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
        Creates class instance from a dictionary with structure:
        
        {
            'coefficient': '1', 
            'left-x': '0', 
            'right-x': '1', 
            'end-t': '1', 
            'init-boundary': 'sin(x)', 
            'left-boundary': '0', 
            'right-boundary': '0'
        }
        
        Args:
            string_dict (dict): Dictionary containing the parameters as strings.
        
        Returns:
            Heat: An instance of the Heat class initialized with the provided parameters.
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
        """
        Converts the conditions to LaTeX format for formatted printing.
        
        Returns:
            dict: Dictionary containing the LaTeX strings for all class parameters that
            describe a parabolic problem.
        """

        temp_dict = self._expressions.copy()
        for key in ['init-boundary', 'left-boundary', 'right-boundary']:
            try:
                numba = float(temp_dict[key])
            except ValueError:
                expr = smp.parsing.sympy_parser.parse_expr(temp_dict[key])
                temp_dict[key] = smp.printing.latex(expr)
        return temp_dict

    def __str__(self):
        """
        String representation of the Heat object.
        
        Returns:
            str: String representation of the expressions dictionary.
        """
        return str(self._expressions)
        
    def right_side(self, u_t):
        """
        Returns the right-hand side of the PDE (u_t).
        
        Args:
            u_t (torch.Tensor): Time derivative of the predicted output.
        
        Returns:
            torch.Tensor: The right-hand side of the PDE.
        """
        return u_t
        
    def left_side(self, ts, xs, u_xx):
        """
        Returns the left-hand side of the PDE (a * u_xx + f(x, t)).
        
        Args:
            ts (torch.Tensor): Time components.
            xs (torch.Tensor): Spatial components.
            u_xx (torch.Tensor): Second-order spatial derivative of the predicted output.
        
        Returns:
            torch.Tensor: The left-hand side of the PDE.
        """
        return self.alpha*u_xx #+ self.fn(ts, xs)
    
    @staticmethod
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
        """
        Plots the exact solution if available and saves the plot to the specified path.
        
        Args:
            path (str): The path where the plot images will be saved.
        
        Returns:
            None
        """

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
            axes.set_xlabel("X")
            axes.set_ylabel("T")
            plt.savefig(path + f"{angle}")