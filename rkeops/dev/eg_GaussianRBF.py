import numpy as np
from pykeops.numpy import Genred


def pykeops_GaussianRBF(*args, dtype="float64"):
    """Redo eg_GaussianRBF.R example with same data and PyKeOps only."""

    args = [np.array(arg, dtype=dtype) for arg in args]

    # Define a custom formula
    formula = "Exp(-G * SqDist(X,Y)) * B"
    variables = [
        "G = Pm(1)",
        "X = Vi(3)",
        "Y = Vj(3)",
        "B = Vj(2)",
    ]

    # Compile corresponding operator
    gaussian_conv = Genred(formula, variables, reduction_op="Sum", axis=1)
    # Perform reduction
    res_pknp = gaussian_conv(*args, backend="CPU")

    return res_pknp


def pykeops_GaussianRBF_grad(*args, dtype="float64"):
    """Redo eg_GaussianRBF.R example with same data and PyKeOps only."""

    args = [np.array(arg, dtype=dtype) for arg in args]

    # Define a custom formula
    formula = "Grad(Exp(-G * SqDist(X,Y)) * B, Y, e)"
    variables = [
        "G = Pm(1)",
        "X = Vi(3)",
        "Y = Vj(3)",
        "B = Vj(2)",
        "e = Vi(2)",
    ]

    # Compile corresponding operator
    gaussian_conv_grad = Genred(formula, variables, reduction_op="Sum", axis=1)
    # Perform reduction
    res_pknp_grad = gaussian_conv_grad(*args, backend="CPU")

    return res_pknp_grad
