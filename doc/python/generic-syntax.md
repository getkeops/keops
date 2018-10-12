# The generic operators

We provide a set of **four** low-level generic operations (two for numpy and two for pytorch), that allow users to **define and reduce** custom operations using either a **summation** or a **log-sum-exp** reduction:

```python
from pykeops.numpy import generic_sum_np, generic_logsumexp_np
from pykeops.torch import generic_sum,    generic_logsumexp
```

**N.B.:** If you don't like our naming conventions or the trailing _np's, don't forget that you can use the `as` keyword and imports such as

```python
from pykeops.numpy import generic_sum_np as gensum
```

### Basic usage

These four operators:

- take as input a list of **strings** specifying the computation;
- return a python **function**, callable on torch tensors or numpy arrays.

The convention is that:

- The first string specifies the desired symbolic *formula*, using a [custom syntax](../generic-syntax).
- The second string specifies the *output*'s type using:
  - a (dummy) name;
  - a *category*, either `Vx` or `Vy` which indicates whether the output shall be indexed by i (with reduction over j) or vice-versa;
  - an integer dimension.
- The strings coming thereafter specify the *inputs*' types using:
  - an alphanumerical name, used in the *formula*;
  - a *category*, either `Vx` (indexation by i), `Vy` (indexation by j) or `Pm` (no indexation, the input tensor is a *vector* and not a 2D array);
  - an integer dimension.

### Backends

The callable routines given by `generic_sum` and co. also accept an optional keyword argument: `backend`. Setting its value by hand may be useful while debugging and optimizing your code.  Supported values are:

- `"auto"`, let KeOps decide which backend is best suited to your data, using a simple heuristic based on the tensors' shapes.
- `"CPU"`, run a [for loop](https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keops/core/CpuConv.cpp) on a single CPU core. Very inefficient!
- `"GPU_1D"`, use a [simple multithreading scheme](https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keops/core/GpuConv1D.cu) on the GPU - basically, one thread per value of the output index.
- `"GPU_2D"`, use a more sophisticated [2D parallelization scheme](https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keops/core/GpuConv2D.cu) on the GPU.
- `"GPU"`, let KeOps decide which one of `GPU_1D` or `GPU_2D` will run faster on the given input. Note that if your data is already located on the GPU, KeOps won't have to load it "by hand".

**N.B.:** Going forward, we will put as much effort as we can into the backend choice and the compilation options.  Keeping `backend="auto"` should thus be your go-to choice.

### Example

Using the low-level generic syntax, computing a Gaussian-RBF kernel product can be done with:

```python
import torch
from pykeops.torch import generic_sum

# Notice that the parameter sigma is a dim-1 vector, *not* a scalar:
sigma  = torch.tensor([.5], requires_grad=True)
# Generate the data as pytorch tensors.
# If you intend to compute gradients, don't forget the `requires_grad` flag!
x = torch.randn(1000,3, requires_grad=True) # `x` and `y` are point clouds
y = torch.randn(2000,3, requires_grad=True) #     in a 3-dimensional space.
b = torch.randn(2000,2, requires_grad=True) # `b` is a collection of 2D vectors.

gaussian_conv = generic_sum("Exp(-G*SqDist(X,Y)) * B", # f(g,x,y,b) = exp( -g*|x-y|^2 ) * b
                            "A = Vx(2)",  # The output is indexed by "i", of dim 2 -> summation over "j"
                            "G = Pm(1)",  # First arg  is a parameter,    of dim 1
                            "X = Vx(3)",  # Second arg is indexed by "i", of dim 3
                            "Y = Vy(3)",  # Third arg  is indexed by "j", of dim 3
                            "B = Vy(2)" ) # Fourth arg is indexed by "j", of dim 2

# By explicitely specifying the backend, you can try to optimize your pipeline:
a = gaussian_conv( 1./sigma**2, x,y,b) # "auto" backend
a = gaussian_conv( 1./sigma**2, x,y,b, backend="CPU")
a = gaussian_conv( 1./sigma**2, x,y,b, backend="GPU_2D")
```
so that
```math
 \text{for~} i = 1, ..., 1000, \quad\quad a_i ~=~  \sum_{j=1}^{2000} \exp(-g\,\|x_i-y_j\|^2) \,\cdot\, b_j.
```

The list of supported *generic syntax* operations
can be found in the docfile [generic_syntax.md](../generic-syntax).


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

