import numpy as np
from pykeops.numpy import Genred

# single operand
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

# single operand
formula = "Sum(V0-V1)"
variables = ["V0=Vi(3)", "V1=Vj(3)"]

## Sum
# Compile corresponding operator
op = Genred(formula, variables, reduction_op="Sum", axis=0)

# data
x = np.random.randn(4, 3)
y = np.random.randn(5, 3)

# Perform reduction
res = op(x, y, backend="CPU")

expected_res = np.sum(np.sum(x.T[:,:,np.newaxis] - y.T[:,np.newaxis,:], axis = 0), axis = 0)
