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

def create_trainig_test_set(eq: diffeq, t_span: tuple, n_steps: int, n_data: int, coeff_test: float, method:str, device="cpu", seed=0):
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
    seed
        Random seed used for generation of data.

    Returns
    -------
    training_set
        A dictionary with:
            - "args_tensor" = arguments belonging to each of the data samples.
            - "x0_tensor" = initial variables of each data sample.
            - "t_tensor" = time timestaps belonging to each data sample.
            - "y_tensor" = values of variables at each timestamp belonging to data samples.
            - "X" = a torch.tensor of tensors t = (args_tensor[i], x0_tensor[i], t_tensor[i, j]),
                    it is used as input in a neural network.
    test_set
        A dictionary with:
            - "args_tensor" = arguments belonging to each of the data samples.
            - "x0_tensor" = initial variables of each data sample.
            - "t_tensor" = time timestaps belonging to each data sample.
            - "y_tensor" = values of variables at each timestamp belonging to data samples.
            - "X" = a torch.tensor of tensors t = (args_tensor[i], x0_tensor[i], t_tensor[i, j]),
                    it is used as input in a neural network.
    """
    if coeff_test < 1 and coeff_test > 0:
        n_train = math.floor((1 - coeff_test) * n_data)
    else:
        raise ValueError("coeff_test has to be in ]0,1[")

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    
    args_tensor = torch.rand(n_data, eq.n_args)
    x0_tensor = torch.rand(n_data, eq.n_var)
    t_tensor = torch.zeros(n_data, n_steps)
    y_tensor = torch.zeros(n_data, n_steps, eq.n_var)
    
    for i, (args, x0) in enumerate(zip(args_tensor, x0_tensor)):
        sol = eq.solve(x0, t_span, n_steps, args, method)
        t_tensor[i] = torch.from_numpy(sol.t).to(device)
        y_tensor[i] = torch.from_numpy(sol.y).to(device).T   # transpose to (n_steps, n_var)

    args_tensor.to(device)
    x0_tensor.to(device)
    t_tensor.to(device)
    y_tensor.to(device)

    X = torch.zeros(n_data, n_steps, eq.n_args + eq.n_var + 1).to(device)
    
    for i in range(n_data):
        args_rep = args_tensor[i].repeat(n_steps, 1)   # (n_steps, n_args)
        x0_rep = x0_tensor[i].repeat(n_steps, 1)       # (n_steps, n_var)
        t_col = t_tensor[i].unsqueeze(1)               # (n_steps, 1)
        
        # Concatenate into [args..., x0..., t]
        X[i] = torch.cat([args_rep, x0_rep, t_col], dim=1).to(device)

    training_set = {
        "args_tensor": args_tensor[:n_train],
        "x0_tensor": x0_tensor[:n_train],
        "t_tensor": t_tensor[:n_train],
        "y_tensor": y_tensor[:n_train],
        "X": X[:n_train],
    }

    test_set = {
        "args_tensor": args_tensor[n_train:],
        "x0_tensor": x0_tensor[n_train:],
        "t_tensor": t_tensor[n_train:],
        "y_tensor": y_tensor[n_train:],
        "X": X[n_train:],
    }

    return training_set, test_set

def create_trainig_validation_test_set(eq: diffeq, t_span: tuple, n_steps: int, n_data: int, coeff_valtest: list, method:str, device="cpu", seed=0):
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
    seed
        Random seed used for generation of data.

    Returns
    -------
    training_set
        A dictionary with:
            - "args_tensor" = arguments belonging to each of the data samples.
            - "x0_tensor" = initial variables of each data sample.
            - "t_tensor" = time timestaps belonging to each data sample.
            - "y_tensor" = values of variables at each timestamp belonging to data samples.
            - "X" = a torch.tensor of tensors t = (args_tensor[i], x0_tensor[i], t_tensor[i, j]),
                    it is used as input in a neural network.
    validation_set
        A dictionary with:
            - "args_tensor" = arguments belonging to each of the data samples.
            - "x0_tensor" = initial variables of each data sample.
            - "t_tensor" = time timestaps belonging to each data sample.
            - "y_tensor" = values of variables at each timestamp belonging to data samples.
            - "X" = a torch.tensor of tensors t = (args_tensor[i], x0_tensor[i], t_tensor[i, j]),
                    it is used as input in a neural network.
    test_set
        A dictionary with:
            - "args_tensor" = arguments belonging to each of the data samples.
            - "x0_tensor" = initial variables of each data sample.
            - "t_tensor" = time timestaps belonging to each data sample.
            - "y_tensor" = values of variables at each timestamp belonging to data samples.
            - "X" = a torch.tensor of tensors t = (args_tensor[i], x0_tensor[i], t_tensor[i, j]),
                    it is used as input in a neural network.
    """
    if sum(coeff_valtest) > 1:
        raise ValueError("the sum of all coefficients should less than 1.")

    n_set = [0, 0, 0]
    
    for i, coeff in enumerate(coeff_valtest):
        if coeff < 1 and coeff > 0:
            n_set[i] = math.floor(coeff * n_data)
        else:
            raise ValueError("coefficient has to be in ]0,1[")

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    
    args_tensor = torch.rand(n_data, eq.n_args)
    x0_tensor = torch.rand(n_data, eq.n_var)
    t_tensor = torch.zeros(n_data, n_steps)
    y_tensor = torch.zeros(n_data, n_steps, eq.n_var)
    
    for i, (args, x0) in enumerate(zip(args_tensor, x0_tensor)):
        sol = eq.solve(x0, t_span, n_steps, args, method)
        t_tensor[i] = torch.from_numpy(sol.t).to(device)
        y_tensor[i] = torch.from_numpy(sol.y).to(device).T   # transpose to (n_steps, n_var)

    args_tensor.to(device)
    x0_tensor.to(device)
    t_tensor.to(device)
    y_tensor.to(device)

    X = torch.zeros(n_data, n_steps, eq.n_args + eq.n_var + 1).to(device)
    
    for i in range(n_data):
        args_rep = args_tensor[i].repeat(n_steps, 1)   # (n_steps, n_args)
        x0_rep = x0_tensor[i].repeat(n_steps, 1)       # (n_steps, n_var)
        t_col = t_tensor[i].unsqueeze(1)               # (n_steps, 1)
        
        # Concatenate into [args..., x0..., t]
        X[i] = torch.cat([args_rep, x0_rep, t_col], dim=1).to(device)

    training_set = {
        "args_tensor": args_tensor[:n_set[0]],
        "x0_tensor": x0_tensor[:n_set[0]],
        "t_tensor": t_tensor[:n_set[0]],
        "y_tensor": y_tensor[:n_set[0]],
        "X": X[:n_set[0]],
    }

    i_val = n_set[0] + n_set[1]
    validation_set = {
        "args_tensor": args_tensor[n_set[0]:i_val],
        "x0_tensor": x0_tensor[n_set[0]:i_val],
        "t_tensor": t_tensor[n_set[0]:i_val],
        "y_tensor": y_tensor[n_set[0]:i_val],
        "X": X[n_set[0]:i_val],
    }

    test_set = {
        "args_tensor": args_tensor[i_val:],
        "x0_tensor": x0_tensor[i_val:],
        "t_tensor": t_tensor[i_val:],
        "y_tensor": y_tensor[i_val:],
        "X": X[i_val:],
    }

    return training_set, validation_set, test_set