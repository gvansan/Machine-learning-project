import numpy as np
import scipy as sp
import math
import torch

from typing import Callable

class diffeq():
    """A class for the differential equations."""
    
    def __init__(self, odes: Callable, n_var:int, n_args: int):
        """Initializes the differential equation.

        Parameters
        ----------
        odes
            A function f(t, X, args) that returns [D(X), D^2(X), ..., D^N(X)], 
            with variables X = (x_1, x_2, ..., x_N).
        n_var
            The number of variables = N = dim(X).
        n_args
            The number of arguments = dim(args).
        """
        self.odes = odes
        self.n_var = n_var
        self.n_args = n_args

    def solve(self, x0: list, t_span: tuple, n_steps: int, args: tuple, method: str):
        """Numearically solves the differential equation.

        Parameters
        ----------
        x0
            Initial variables X_0 = (x1_0, x2_0, ..., xN_0).
        t_span
            The time interval [t_i, t_e] in which the differential equation will be calculated.
        n_steps
            The number of time steps.
        args
            Arguments for the set of differential equations f(t, X, args).
        method
            The method for the numerical calculation (RK45, RK23, DOP853, Radau, BDF, LSODA).

        Returns
        -------
        sol
            scipy.integrate.solve_ivp solution
        """
        t_eval = np.linspace(*t_span, n_steps)

        if args == ():
            sol = sp.integrate.solve_ivp(self.odes, t_span, x0, t_eval=t_eval, method=method)
        else:
            sol = sp.integrate.solve_ivp(self.odes, t_span, x0, args=args, t_eval=t_eval, method=method)
            
        return sol

def create_trainig_test_set(eq: diffeq, t_span: tuple, n_steps: int, n_data: int, coeff_test: float, method:str):
    """Creates training and test data sets.

    Parameters
    ----------
    eq
        The differential equation of class diffeq.
    t_span
        The time interval [t_i, t_e] in which the differential equation will be calculated.
    n_steps
        The number of time steps.
    n_data
        The total number (training and test) of required data samples.
    coeff_test
        The fraction of data samples that belong to the test set.
    method
        The method for the numerical calculation (RK45, RK23, DOP853, Radau, BDF, LSODA).

    Returns
    -------
    training_set
        A dictionary with:
            - "args_tensor" = arguments belonging to each of the data samples.
            - "x0_tensor" = initial variables of each data sample.
            - "t_tensor" = time timestaps belonging to each data sample.
            - "y_tensor" = values of variables at each timestamp belonging to data samples.
            - "X" = a torch.tensor of tensors t = (args_tensor[i], x0_tensor[i], t_tensor[i]),
                    it is used as input in a neural network.
    test_set
        A dictionary with:
            - "args_tensor" = arguments belonging to each of the data samples.
            - "x0_tensor" = initial variables of each data sample.
            - "t_tensor" = time timestaps belonging to each data sample.
            - "y_tensor" = values of variables at each timestamp belonging to data samples.
            - "X" = a torch.tensor of tensors t = (args_tensor[i], x0_tensor[i], t_tensor[i]),
                    it is used as input in a neural network.
    """
    if coeff_test < 1 and coeff_test > 0:
        n_train = math.floor((1 - coeff_test) * n_data)
    else:
        raise ValueError("coeff_test has to be in ]0,1[")
    
    args_tensor = torch.rand(n_data, eq.n_args)
    x0_tensor = torch.rand(n_data, eq.n_var)
    t_tensor = torch.zeros(n_data, n_steps)
    y_tensor = torch.zeros(n_data, eq.n_var, n_steps)
    
    # There's maybe a more efficient way...
    for i, (args, x0) in enumerate(zip(args_tensor, x0_tensor)):
        sol = eq.solve(x0, t_span, n_steps, args, method)
        t_tensor[i] = torch.from_numpy(sol.t)
        y_tensor[i] = torch.from_numpy(sol.y)

    training_tensors = [args_tensor[:n_train], x0_tensor[:n_train], t_tensor[:n_train], y_tensor[:n_train], y_tensor[:n_train].reshape(n_train, len(y_tensor[0]) * len(y_tensor[0][0]))]
    test_tensors = [args_tensor[n_train:], x0_tensor[n_train:], t_tensor[n_train:], y_tensor[n_train:], y_tensor[n_train:].reshape(n_data - n_train, len(y_tensor[0]) * len(y_tensor[0][0]))]
    training_set = {"args_tensor": training_tensors[0],
                    "x0_tensor": training_tensors[1],
                    "t_tensor": training_tensors[2],
                    "y_tensor": training_tensors[3],
                    "X": torch.cat([training_tensors[i] for i in [0, 1, 2]], dim=1)}
    test_set = {"args_tensor": test_tensors[0],
                    "x0_tensor": test_tensors[1],
                    "t_tensor": test_tensors[2],
                    "y_tensor": test_tensors[3],
                    "X": torch.cat([test_tensors[i] for i in [0, 1, 2]], dim=1)}

    return training_set, test_set