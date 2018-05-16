# KErnel OPerationS, on CPUs and GPUs, with autodiff and without memory overflows

```
          88           oooo    oooo             .oooooo.                                 88
        .8'`8.         `888   .8P'             d8P'  `Y8b                              .8'`8.
       .8'  `8.         888  d8'     .ooooo.  888      888 oo.ooooo.   .oooo.o        .8'  `8.
      .8'    `8.        88888[      d88' `88b 888      888  888' `88b d88(  "8       .8'    `8.
     .8'      `8.       888`88b.    888ooo888 888      888  888   888 `"Y88b.       .8'      `8.
    .8'        `8.      888  `88b.  888    .o `88b    d88'  888   888 o.  )88b     .8'        `8.
    88oooooooooo88     o888o  o888o `Y8bod8P'  `Y8bood8P'   888bod8P' 8""888P'     88oooooooooo88
                                                            888
                                                           o888o
```

## What is KeOps?

KeOps is small plugin to pytorch, numpy, Matlab or R that enables
them to compute efficiently **Kernel dot products**
and other similar operations on the GPU.
It provides good performances, full differentiability
and linear (instead of quadratic) memory footprint through
a minimal interface: the `kernel_product` and `generic_sum` operations.

## Motivations

In recent years, Deep Learning frameworks such as
[Theano](http://deeplearning.net/software/theano/), [TensorFlow](http://www.tensorflow.org) 
or [PyTorch](http://pytorch.org) have evolved
into fully-fledged applied math libraries:
With negligible overhead, these tools now bring **automatic differentiation**
and **seamless GPU support** to research communities that were used
to array-friendly frameworks -- Matlab and numpy.

Unfortunately, *no magic* is involved:
optimised CUDA codes still have to be written
for every single atomic operation provided to end-users, and
supporting all the standard mathematical computations
thus comes at a **huge engineering cost** for the developers
of the main frameworks.
As of 2018, this considerable effort has been mostly restricted to the
operations relevant to the study Convolutional Neural Networks:
linear algebra routines and *grid* convolutions.

As researchers focused on statistical shape analysis,
we thus got confronted with a **memory bottleneck** when trying
to compute Gaussian convolutions on non regular grids in $`\mathbb R^3`$
(aka. **RBF kernel product**). Given:

- a target point cloud $`(x_i)_{i=1}^N \in  \mathbb R^{N \times 3}`$
- a source point cloud $`(y_j)_{j=1}^M \in  \mathbb R^{M \times 3}`$
- a signal or vector field $`(b_j)_{j=1}^M \in  \mathbb R^{M \times D}`$ attached to the $`y_j`$'s

we strive to compute efficiently the array $`(a_i)_{i=1}^N \in  \mathbb R^{N \times D}`$ given by

```math
 a_i =  \sum_j K(x_i,y_j) b_j,  \qquad i=1,\cdots,N
```

where $`K(x_i,y_j) = \exp(-\|x_i - y_j\|^2 / \sigma^2)`$.

A standard way of doing so in array-centric frameworks is to
create and store in memory the full M-by-N kernel matrix $`K_{i,j}=K(x_i,y_j)`$,
before computing $`(a_i)`$ as the dot product between $`(K_ij)`$ and $`(b_j)`$

This is all good for many users...
But as soon as we try to scale this method
to large datasets or point clouds (say, M and N >= 100,000),
it becomes intractable:
**huge 100,000-by-100,000 kernel matrices just don't fit in GPU memories**.

The purpose of KeOps, simply put, is to **let users break through this memory bottleneck**
by computing *online sum reductions*.
As we provide a simple `kernel_product` operator (+ a flexible, low-level `generic_sum` interface),
we hope that more and more researchers will come to enjoy the benefits of automatic differentiation.

![Benchmark](./benchmark.png)


----

{:.no_toc}

- TOC
{:toc}

----


## What we provide

The KeOps library allows you to compute efficiently *any* expression of the form

```math
a_i = \text{Reduction}_j \big[ f(p^1, p^2, ..., x^1_i, x^2_i, ..., y^1_j, y^2_j, ...)  \big]
```

and its derivatives with respects to *all* the (vector) variables,
where $`i`$ goes from $`1`$ to $`N`$ and $`j`$ from $`1`$ to $`M`$.

The core of KeOps relies on a set **C++/CUDA routines**, for which we provide
**pytorch**, **numpy**, **MATLAB** and **R** (soon) bindings.
As of today, we support:

- Summation and (online, numerically stable) LogSumExp reductions.
- User-defined formulas, using a simple string format (`"gaussian(x,y) * (1+linear(u,v)**2)"`) or a custom low-level syntax (`"Exp(-G*SqDist(X,Y)) * ( IntCst(1) + Pow((U,V), 2) )"`).
- Simple syntax for kernels on feature spaces (say, locations+orientations varifold kernels used in shape analysis).
- High-order derivatives with respect to all parameters and variables.
- Non-radial kernels.

The library comes with various examples ranging from LDDMM theory (non rigid deformations) to kernel density estimations (non parametric statistics).
A **reference paper** will soon be put on Arxiv.

## Usage

We provide bindings in python (both numpy and pytorch complient), Matlab and R.

### Computing Kernel dot-products

Using PyTorch, we can compute a fully differentiable Gaussian-RBF kernel product
by typing:

```python
import torch
from pykeops.torch.kernels import Kernel, kernel_product

# Generate the data as pytorch Variables
x = torch.randn(1000,3, requires_grad=True)
y = torch.randn(2000,3, requires_grad=True)
b = torch.randn(2000,2, requires_grad=True)

# Pre-defined kernel: using custom expressions is also possible!
sigma  = torch.tensor([.5], requires_grad=True)
params = {
    "id"      : Kernel("gaussian(x,y)"),
    "gamma"   : 1./sigma**2,
}

# Depending on the inputs' types, 'a' is a CPU or a GPU variable.
# It can be differentiated wrt. x, y, b and sigma.
a = kernel_product(params, x, y, b)
```

### Generic formulas

Using the low-level generic syntax, an equivalent call would be:

```python
from pykeops.torch.generic_sum import generic_sum

gaussian_conv = generic_sum("Exp(-G*SqDist(X,Y)) * B",
                            "A = Vx(2)",  # The output is indexed by "i", of dim 2
                            "G = Pm(1)",  # First arg  is a parameter,    of dim 1
                            "X = Vx(3)",  # Second arg is indexed by "i", of dim 3
                            "Y = Vy(3)",  # Third arg  is indexed by "j", of dim 3
                            "B = Vy(2)" ) # Fourth arg is indexed by "j", of dim 2
a = gaussian_conv( 1./sigma**2, x,y,b)
```

Details about the generic syntax can be found in the docfile [generic_syntax.md](generic_syntax.md).


## Under the hood

In order to scale up on large datasets, we use a **tiled implementation** that allows us to get a $`O(N+M)`$ memory footprint instead of the usual $`O(NM)`$ codes generated by high level libraries - Thrust or cuda version of pyTorch and TensorFlow. CUDA kernels are compiled on-the-fly: one '.so' or '.dll' file is generated per mathematical expression, and can be re-used for other data samples and values of $`M`$ and $`N`$.

As of today, KeOps provides two backends:

- a naive pytorch implementation for the `kernel_product` method, that can be used on small samples and for testing purposes.
- a homemade C++/CUDA engine, located in the [`./keops/core`](./keops/core) folder. Automatic differentiation of formulas is performed using variadic templating.

We're currently investigating the possibility of developing a third backend, that would rely on a genuine CUDA library such as [Tensor Comprehensions](http://facebookresearch.github.io/TensorComprehensions/introduction.html).

## Quick start

### Python users

Requirements:

- a unix-like system (typically Linux or Mac Os X)
- Python 3 with packages  : numpy, gputil (install via pip)
- optional : Cuda (>=9.0 is recommended), PyTorch>=0.4

Two steps:

1) Install pykeops package.

2) Run the out-of-the-box working examples located in [`./pykeops/examples/`](./pykeops/examples/) and [`./pykeops/tutorials/`](./pykeops/tutorials/).

If you are already familiar with the LDDMM theory and want to get started quickly, please check the shapes toolboxes: [plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox](https://plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox) and [plmlab.math.cnrs.fr/jeanfeydy/lddmm_pytorch](https://plmlab.math.cnrs.fr/jeanfeydy/lddmm_pytorch).

### Matlab users

Three steps:

1) Download keops library and unzip it at a location of your choice. Note that temporary files will be written into keopslab/build folder, so that this directory must hhave write permissions.

2) Within Matlab, run the out-of-the-box working examples located in `./matlab/examples/`

3) To use keops in your own Matlab codes, set the Matlab path to include "keopslab" folder and all its subfolders.

N.B. Everytime you need to update or reinstall the library, make sure you replace the full directory keopslab, so that temporary files will be erased.

### R users

To do.

## Known issues

First of all, make sure that you are using a recent C/C++ compiler (say, gcc/g++-7);
otherwise, CUDA compilation may fail in unexpected ways.
On Linux, this can be done simply by using [update-alternatives](https://askubuntu.com/questions/26498/choose-gcc-and-g-version).

Note that you can activate a "verbose" compilation mode by adding these lines *after* your KeOps imports:

```python
import pykeops
pykeops.common.compile_routines.verbose = True
```

Then, if you installed from source and recently updated KeOps, make sure that your
`keops/build/` folder (the cache of already-compiled formulas) has been emptied.

If an error involving libstdc++.so.6 occurs like

```
cmake: /usr/local/MATLAB/R2017b/sys/os/glnxa64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by cmake)
cmake: /usr/local/MATLAB/R2017b/sys/os/glnxa64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by cmake)
cmake: /usr/local/MATLAB/R2017b/sys/os/glnxa64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /usr/lib/x86_64-linux-gnu/libjsoncpp.so.1)
```

try to load matlab with the following linking variable :

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6;matlab
```

......
authors : [Benjamin Charlier](http://www.math.univ-montp2.fr/~charlier/), [Jean Feydy](http://www.math.ens.fr/~feydy/), [Joan Alexis Glaunès](http://www.mi.parisdescartes.fr/~glaunes/)
