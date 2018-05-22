```
                        oooo    oooo             .oooooo.                                88
                        `888   .8P'             d8P'  `Y8b                             .8'`8.
 oo.ooooo.  oooo    ooo  888  d8'     .ooooo.  888      888 oo.ooooo.   .oooo.o       .8'  `8.
  888' `88b  `88.  .8'   88888[      d88' `88b 888      888  888' `88b d88(  "8      .8'    `8.
  888   888   `88..8'    888`88b.    888ooo888 888      888  888   888 `"Y88b.      .8'      `8.
  888   888    `888'     888  `88b.  888    .o `88b    d88'  888   888 o.  )88b    .8'        `8.
  888bod8P'     .8'     o888o  o888o `Y8bod8P'  `Y8bood8P'   888bod8P' 8""888P'    88oooooooooo88
  888       .o..P'                                           888
 o888o      `Y8P'                                           o888o
```

# Installation

## Requirements

- a unix-like system (typically Linux or MacOs X)
- Cmake>=2.9
- a C++ compiler (gcc>=4.8, clang or nvcc)
- **Python 3** with packages : numpy, GPUtil (installed via pip)
- optional : Cuda (>=9.0 is recommended), PyTorch>=0.4

## Using pip (recommended)

A good starting point is to check your python and pip path.
In a terminal carefully verify the **consistency** of the output of the following commands `which python`,
`python --version`, `which pip` and `pip --version`.
Then, check that cmake is working on your system by running in a terminal:
`cmake --version`. If needed, please run

```bash
pip install cmake
```

to get a proper version of cmake.
Finally, simply run

```bash
pip install pykeops
```

## From source

Warning: we assume here that cmake is working properly.

- Clone the git repo.
- Manually add the directory `/path/to/libkeops/` to you python path.

This can be done once and for all, by adding

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/libkeops/
```

to your `~/.bashrc`. Otherwise, you can add the following line to the
beginning of your python scripts:

```python
import os.path
import sys
sys.path.append('/path/to/libkeops/')
```

## Testing your installation

Go in your `/path/to/libkeops/pykeops/test/` folder and run

```bash
python unit_tests_numpy.py
python unit_tests_pytorch.py # if needed
```

Hopefully, we find a cleaner way to do this soon!

# Usage

Having assumed that the reader is already familiar with
**the syntax showcased in the [readme/usage](../readme.md#usage) section**,
we now fully document the public interface of the pykeops module.
A set of minimal [examples](./examples/) and more complex [tutorials](./tutorials/)
are provided in the pykeops folder. Feel free to run them and
inspect their codes!
If you are already familiar with the LDDMM theory and want to get started quickly,
you can also check Jean's shapes toolboxes:
[shapes_toolbox](https://plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox)
and [lddmm_pytorch](https://plmlab.math.cnrs.fr/jeanfeydy/lddmm_pytorch).

**N.B.:** If you run a KeOps script for the first time,
the internal engine may take a **few minutes** to compile all the relevant formulas.
Do not worry: this work is done **once and for all** as KeOps stores the resulting
.dll/.so files in a cache/build directory - located in your home folder.

## The generic operators

We provide a set of **four** low-level generic operations
(two for numpy and two for pytorch), that allow users to
**define and reduce** custom operations using either
a **summation** or a **log-sum-exp** reduction:

```python
from pykeops.numpy import generic_sum_np, generic_logsumexp_np
from pykeops.torch import generic_sum,    generic_logsumexp
```

**N.B.:** If you don't like our naming conventions or the trailing "_np"'s,
don't forget that you can use the `as` keyword and imports such as

```python
from pykeops.numpy import generic_sum_np as gensum
```

### Basic usage

These four operators:

- take as input a list of **strings** specifying the computation;
- return a python **function**, callable on torch tensors or numpy arrays.

The convention is that:

- The first string specifies the desired symbolic *formula*, using a [custom syntax](../generic_syntax.md).
- The second string specifies the *output*'s type using:
  - a (dummy) name;
  - a *category*, either `Vx` or `Vy` which indicates whether the output shall be indexed by i (with reduction over j) or vice-versa;
  - an integer dimension.
- The strings coming thereafter specify the *inputs*' types using:
  - an alphanumerical name, used in the *formula*;
  - a *category*, either `Vx` (indexation by i), `Vy` (indexation by j) or `Pm` (no indexation, the input tensor is a *vector* and not a 2D array);
  - an integer dimension.

### Backends

The callable routines given by `generic_sum` and co. also accept
an optional keyword argument: `backend`. Setting its value by hand
may be useful while debugging and optimizing your code.
Supported values are:

- `"auto"`, let KeOps decide which backend is best suited to your data, using a simple heuristic based on the tensors' shapes.
- `"CPU"`, run a [for loop](../keops/core/CpuConv.cpp) on a single CPU core. Very inefficient!
- `"GPU_1D"`, use a [simple multithreading scheme](../keops/core/GpuConv1D.cu) on the GPU - basically, one thread per value of the output index.
- `"GPU_2D"`, use a more sophisticated [2D parallelization scheme](../keops/core/GpuConv2D.cu) on the GPU.
- `"GPU"`, let KeOps decide which one of `GPU_1D` or `GPU_2D` will run faster on the given input. Note that if your data is already located on the GPU, KeOps won't have to load it "by hand".

**N.B.:** Going forward, we will put as much effort as we can in the backend choice
and the compilation options.
Keeping `backend="auto"` should thus be your go-to choice.

### Example

Using the low-level generic syntax, computing
a Gaussian-RBF kernel product can be done with:

```python
import torch
from pykeops.torch.generic_sum import generic_sum

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

The list of supported *generic syntax* operations
can be found in the docfile [generic_syntax.md](../generic_syntax.md).




## The convenient `kernel_product` helper - pytorch only

On top of the low-level syntax, we also provide
a **kernel name parser** that lets you quickly define and work with
most of the kernel products used in shape analysis.
This high-level interface relies on two operators:

```python
from pykeops.torch import Kernel, kernel_product
```

- `Kernel` is the name parser: it turns a string identifier (say, `"gaussian(x,y) * (1 + linear(u,v)**2 )"`) into a set of KeOps formulas.
- `kernel_product` is the "numerical" torch routine. It takes as input a dict of parameters and a set of input tensors, to return a fully differentiable torch variable.

Before going into details, let's showcase a typical example: the computation of
a **Cauchy-Binet varifold kernel** on the space of points+orientations.
Given:

- a set $`(x_i)`$ of target points in $`\mathbb{R}^3`$;
- a set $`(u_i)`$ of target orientations in $`\mathbb{S}^1`$, encoded as unit-norm vectors in $`\mathbb{R}^3`$;
- a set $`(y_j)`$ of source points in $`\mathbb{R}^3`$;
- a set $`(v_j)`$ of source orientations in $`\mathbb{S}^1`$, encoded as unit-norm vectors in $`\mathbb{R}^3`$;
- a set $`(b_j)`$ of source signal values in $`\mathbb{R}^4`$;

we will compute the "target" signal values

```math
 a_i ~=~  \sum_j K(\,x_i,u_i\,;\,y_j,v_j\,) b_j ~=~ \sum_j k(x_i,y_j)\cdot \langle u_i, v_j\rangle^2 \cdot b_j,
```

where $`k(x_i,y_j) = \exp(-\|x_i - y_j\|^2 / \sigma^2)`$.

```python
import torch
import torch.nn.functional as F
from pykeops.torch import Kernel, kernel_product

N, M = 1000, 2000 # number of "i" and "j" indices
# Generate the data as pytorch tensors.

# First, the "i" variables:
x = torch.randn(N,3) # Positions,    in R^3
u = torch.randn(N,2) # Orientations, in R^2 (for example)

# Then, the "j" ones:
y = torch.randn(M,3) # Positions,    in R^3
v = torch.randn(M,2) # Orientations, in R^2

# The signal b_j, supported by the (y_j,v_j)'s
b = torch.randn(M,4)

# Pre-defined kernel: using custom expressions is also possible!
# Notice that the parameter sigma is a dim-1 vector, *not* a scalar:
sigma  = torch.tensor([.5])
params = {
    # The "id" is defined using a set of special variable names and functions
    "id"      : Kernel("gaussian(x,y) * (linear(u,v)**2) "),
    # gaussian(x,y) requires a standard deviation; linear(u,v) requires no parameter
    "gamma"   : ( 1./sigma**2 , None ) ,
}

# Don't forget to normalize the orientations:
u = F.normalize(u, p=2, dim=1)
v = F.normalize(v, p=2, dim=1)

# We're good to go! Notice how we grouped together the "i" and "j" features:
a = kernel_product(params, (x,u), (y,v), b)
```

### The `Kernel` parser



# Troubleshooting

First of all, make sure that you are using a recent C/C++ compiler (say, gcc/g++-6 or above, or clang);
otherwise, CUDA compilation may fail in unexpected ways.
On Linux, this can be done simply by using [update-alternatives](https://askubuntu.com/questions/26498/choose-gcc-and-g-version).

Note that you can activate a "verbose" compilation mode by adding these lines *after* your KeOps imports:

```python
import pykeops
pykeops.common.compile_routines.verbose = True
```

Then, if you installed from source and recently updated KeOps, make sure that your
`keops/build/` folder (the cache of already-compiled formulas) has been emptied.
