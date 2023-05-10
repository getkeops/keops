import numpy as np
from pykeops.numpy import Genred
# Simple formula
formula = "V0"
variables = ["V0=Vi(4)"]
# Sum on Vi
op = Genred(formula, variables, reduction_op="Sum", axis=0)
# data
arg = np.random.randn(10, 4)
# Perform reduction (add a dimension)
res = op(arg, backend="CPU")
print(res.shape)
#> (1, 4)
# Sum on Vj
op = Genred(formula, variables, reduction_op="Sum", axis=1)
# data
arg = np.random.randn(10, 4)
# Perform reduction (should do nothing?)
res = op(arg, backend="CPU")
print(res.shape)
#> (10, 4)
print(sum(res - arg))
#> array([0., 0., 0., 0.])