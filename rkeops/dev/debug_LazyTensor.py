import numpy as np
from pykeops.numpy import Genred

formula = "V0"
variables = ["V0=Vi(4)"]

## Sum
# Compile corresponding operator
op = Genred(formula, variables, reduction_op="Sum", axis=0)

# data
arg = np.random.randn(10, 4)

# Perform reduction
res = op(arg, backend="CPU")
res.shape

## KMin
# Compile corresponding operator
op = Genred(formula, variables, reduction_op="KMin", axis=0, opt_arg=3)

# data
arg = np.random.randn(10, 4)

# Perform reduction
res = op(arg, backend="CPU")

## Min_Argmin
# Compile corresponding operator
op = Genred(formula, variables, reduction_op="Min_ArgMin", axis=0)

# data
arg = np.random.randn(10, 4)

# Perform reduction
res = op(arg, backend="CPU")
res.shape

