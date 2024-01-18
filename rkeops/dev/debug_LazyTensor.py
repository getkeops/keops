import numpy as np
from pykeops.numpy import Genred

### single operand
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

### more complex formula
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

expected_res = np.sum(
    np.sum(x.T[:, :, np.newaxis] - y.T[:, np.newaxis, :], axis=0), axis=0
)

## LogSumExp
# Compile corresponding operator
op = Genred(formula, variables, reduction_op="LogSumExp", axis=0)

# data
x = np.random.randn(4, 3)
y = np.random.randn(5, 3)

# Perform reduction
res = op(x, y, backend="CPU")
res.shape

expected_res = np.log(
    np.sum(
        np.exp(np.sum(x.T[:, :, np.newaxis] - y.T[:, np.newaxis, :], axis=0)), axis=0
    )
)

np.sum(np.square(res.flatten() - expected_res))

## LogSumExpWeight
formula = "Sum(V0-V1)"
variables = ["V0=Vi(3)", "V1=Vj(3)", "V2=Vj(3)"]

# Compile corresponding operator
op = Genred(formula, variables, reduction_op="LogSumExpWeight", axis=0, formula2="V2")

# data
x = np.random.randn(4, 3)
y = np.random.randn(5, 3)
w = np.ones([5, 3])

# Perform reduction
res = op(x, y, w, backend="CPU")
res.shape

expected_res = np.log(
    np.sum(
        np.exp(np.sum(x.T[:, :, np.newaxis] - y.T[:, np.newaxis, :], axis=0)), axis=0
    )
)

np.sum(np.square(res.flatten() - expected_res))

## SumSoftMaxWeight
formula = "Sum(V0-V1)"
variables = ["V0=Vi(3)", "V1=Vj(3)", "V2=Vj(3)"]

# Compile corresponding operator
op = Genred(formula, variables, reduction_op="SumSoftMaxWeight", axis=1, formula2="V2")

# data
x = np.random.randn(4, 3)
y = np.random.randn(5, 3)
w = np.ones([5, 3])

# Perform reduction
res = op(x, y, w, backend="CPU")
res.shape

tmp = np.sum((x[:, None, :] - y[None, :, :]), axis=2)
tmp -= np.max(tmp, axis=1)[:, None]  # Subtract the max to prevent numeric overflows
tmp = np.exp(tmp) @ w / np.sum(np.exp(tmp), axis=1)[:, None]

expected_res = tmp

np.sum(np.square(res - expected_res))

# data
x = np.random.randn(4, 3)
y = np.random.randn(5, 3)
w = np.random.randn(5, 3)

# Perform reduction
res = op(x, y, w, backend="CPU")
res.shape

tmp = np.sum((x[:, None, :] - y[None, :, :]), axis=2)
tmp -= np.max(tmp, axis=1)[:, None]  # Subtract the max to prevent numeric overflows
tmp = np.exp(tmp) @ w / np.sum(np.exp(tmp), axis=1)[:, None]

expected_res = tmp

np.sum(np.square(res - expected_res))
