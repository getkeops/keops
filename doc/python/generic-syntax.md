
# Using the syntax in the PyTorch bindings

The example described below is implemented in the example Python script [`generic_syntax_pytorch.py`](https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/pykeops/examples/generic_syntax_pytorch.py) located in [`./pykeops/examples/`](https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/pykeops/examples/). 
We also provide a PyTorch-free binding, showcased in [`generic_syntax_numpy.py`](https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/pykeops/examples/generic_syntax_numpy.py).

The Python helper functions `generic_sum` and `generic_logsumexp` take as input the formula and aliases `str` objects:

```python
from pykeops.torch import generic_sum, generic_logsumexp

formula = "Square(p-a)*Exp(x+y)"
types   = ["output = Vx(3)", "p = Pm(1)","a = Vy(1)","x = Vx(3)","y = Vy(3)"]
routine = generic_sum(formula, *types)
```

Notice that the variables' indices are inferred from their positions in the list of types:
output first, then the arguments one after the other.
Having defined our symbolic reduction operation, we can then apply it
on arbitrary torch tensors, just like any other (fully-differentiable) PyTorch function:

```python
c = routine(p,a,x,y)
```

In this example, `p`, `a`, `x` and `y` should correspond to tensors with compatible sizes :

- `p` must be a vector of size 1 - with torch.Size([1]), not torch.Size([]) or torch.Size([1,1]).
- `a` must be a "list" of 1-vectors, i.e. a variable of size n*1, where n can be any positive integer,
- `x` must be a "list" of 3-vectors, i.e. a variable of size m*3, where m can be any positive integer,
- `y` must be a "list" of 3-vectors, i.e. a variable of size n*3.
- output `c` will be a "list" of 3-vectors, i.e. a variable of size m*3.

If the required formula has not been used on the current machine since the last KeOps update,
the previous call will first perform an on-the-fly compilation, producing (and storing) the appropriate .dll/.so file.
Otherwise, the CUDA kernel will simply be loaded into RAM memory.

**Autodiff engine.**
KeOps has an internal automatic differentiation engine for symbolic formulas
-- compatible with the PyTorch autograd package -- that allows
us to "bootstrap" all the derivatives required by the user (including gradients of gradients, etc.).
Feel free to use the output of `kernel_product`, `generic_sum` or `generic_logsumexp`
as any other torch tensor!