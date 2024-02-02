
<!-- README.md is generated from README.Rmd. Please edit that file -->

# RKeOps

<a href="https://www.kernel-operations.io/rkeops/"><img src="man/figures/rkeops_logo.png" align="right" height="120" /></a>

<!-- badges: start -->

[![CRAN
status](https://www.r-pkg.org/badges/version/rkeops)](https://cran.r-project.org/package=rkeops)
[![CRAN
downloads](http://cranlogs.r-pkg.org/badges/grand-total/rkeops?color=yellowgreen)](https://cran.r-project.org/package=rkeops)
<!-- badges: end -->

- Documentation URL: <https://www.kernel-operations.io/rkeops/>
- KeOps project URL: <https://www.kernel-operations.io/>
- Source: <https://github.com/getkeops/keops>
- Licence and Copyright: see
  <https://github.com/getkeops/keops/blob/main/licence.txt>

# Authors

Please contact us for any **bug report**, **question** or **feature
request** by filing a report on our [GitHub issue
tracker](https://github.com/getkeops/keops/issues)!

**Core library - KeOps, PyKeOps, KeOpsLab:**

- [Benjamin Charlier](https://imag.umontpellier.fr/~charlier/), from the
  University of Montpellier.
- [Jean Feydy](https://www.jeanfeydy.com), from Inria.
- [Joan Alexis Glaunès](http://helios.mi.parisdescartes.fr/~glaunes/),
  from the University of Paris.

**R bindings - RKeOps:**

- Amelie Vernay, from the University of Montpellier.
- Chloe Serre-Combe, from the University of Montpellier.
- [Ghislain Durif](https://gdurif.perso.math.cnrs.fr/), from CNRS.

**Contributors:**

- [François-David Collin](https://github.com/fradav), from the
  University of Montpellier: Tensordot operation, CI setup.
- [Tanguy Lefort](https://github.com/tanglef), from the University of
  Montpellier: conjugate gradient solver.
- [Mauricio Diaz](https://github.com/mdiazmel), from Inria of Paris: CI
  setup.
- [Benoît Martin](https://github.com/benoitmartin88), from the Aramis
  Inria team: multi-GPU support.
- [Francis Williams](https://www.fwilliams.info), from New York
  University: maths operations.
- [Kshiteej Kalambarkar](https://github.com/kshitij12345), from
  Quansight: maths operations.
- [D. J. Sutherland](https://djsutherland.ml), from the TTI-Chicago: bug
  fix in the Python package.
- [David
  Völgyes](https://scholar.google.no/citations?user=ngT2GvMAAAAJ&hl=en),
  from the Norwegian Institute of Science and Technology: bug fix in the
  formula parser.

Beyond explicit code contributions, KeOps has grown out of numerous
discussions with applied mathematicians and machine learning experts. We
would especially like to thank [Alain
Trouvé](https://atrouve.perso.math.cnrs.fr/), [Stanley
Durrleman](https://who.rocq.inria.fr/Stanley.Durrleman/), [Gabriel
Peyré](http://www.gpeyre.com/) and [Michael
Bronstein](https://people.lu.usi.ch/bronstem/) for their valuable
suggestions and financial support.

------------------------------------------------------------------------

# Citation

If you use this code in a research paper, **please cite** our [original
publication](https://jmlr.org/papers/v22/20-275.html):

> Charlier, B., Feydy, J., Glaunès, J. A., Collin, F.-D. & Durif, G.
> Kernel Operations on the GPU, with Autodiff, without Memory Overflows.
> Journal of Machine Learning Research 22, 1–6 (2021).

``` tex
@article{JMLR:v22:20-275,
  author  = {Benjamin Charlier and Jean Feydy and Joan Alexis Glaunès and François-David Collin and Ghislain Durif},
  title   = {Kernel Operations on the GPU, with Autodiff, without Memory Overflows},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {74},
  pages   = {1-6},
  url     = {https://jmlr.org/papers/v22/20-275.html}
}
```

For applications to **geometric (deep) learning**, you may also consider
our [NeurIPS 2020
paper](https://www.jeanfeydy.com/Papers/KeOps_NeurIPS_2020.pdf):

``` tex
@article{feydy2020fast,
    title={Fast geometric learning with symbolic matrices},
    author={Feydy, Jean and Glaun{\`e}s, Joan and Charlier, Benjamin and Bronstein, Michael},
    journal={Advances in Neural Information Processing Systems},
    volume={33},
    year={2020}
}
```

------------------------------------------------------------------------

# What is RKeOps?

RKeOps is the R package interfacing the KeOps library.

## KeOps

> Seamless Kernel Operations on GPU (or CPU), with auto-differentiation
> and without memory overflows

The KeOps library (<http://www.kernel-operations.io>) provides routines
to compute generic reductions of large 2d arrays whose entries are given
by a mathematical formula. Using a C++/CUDA-based implementation with
GPU support, it combines a tiled reduction scheme with an automatic
differentiation engine. Relying on online map-reduce schemes, it is
perfectly suited to the scalable computation of kernel dot products and
the associated gradients, even when the full kernel matrix does not fit
into the GPU memory.

KeOps is all about breaking through this memory bottleneck and making
GPU power available for seamless standard mathematical routine
computations. As of 2019, this effort has been mostly restricted to the
operations needed to implement Convolutional Neural Networks: linear
algebra routines and convolutions on grids, images and volumes. KeOps
provides CPU and GPU support without the cost of developing a specific
CUDA implementation of your custom mathematical operators.

To ensure its versatility, KeOps can be used through Matlab, Python
(NumPy or PyTorch) and R back-ends.

## RKeOps

RKeOps is a library that can<br><br>

- Compute **generic reduction** (row-wise or column-wise) of very large
  array/matrices,
  i.e. $$\sum_{i=1}^M a_{ij} \ \ \ \ \text{or}\ \ \ \ \sum_{j=1}^N a_{ij}$$
  for some matrix $A = [a_{ij}]_{M \times N}$ with $M$ rows and $N$
  columns, whose entries $a_{ij}$ can be defined with basic math
  formulae or matrix operators.<br><br>

- Compute **kernel dot products**,
  i.e. $$\sum_{i=1}^M K(\mathbf x_i, \mathbf y_j)\ \ \ \ \text{or}\ \ \ \ \sum_{j=1}^N K(\mathbf x_i, \mathbf y_j)$$
  for a kernel function $K$ and some vectors $\mathbf x_i$,
  $\mathbf y_j\in \mathbb{R}^D$ that are generally rows of some data
  matrices $\mathbf X = [x_{ik}]_{M \times D}$ and
  $\mathbf Y = [y_{jk}]_{N \times D}$ respectively.<br><br>

- Compute the **associated gradients**<br><br>

> ***Applications***: RKeOps can be used to implement a wide range of
> problems encountered in ***machine learning***, ***statistics*** and
> more: such as $k$-nearest neighbor classification, $k$-means
> clustering, Gaussian-kernel-based problems (e.g. linear system with
> Ridge regularization), etc.

## Why using RKeOps?

RKeOps provides<br>

- a **symbolic computation** framework to seamlessly process very large
  arrays writing **R-base-matrix-like operations** based on lazy
  evaluations.<br>

- an API to create **user-defined operators** based on generic
  mathematical formulae, that can be applied to data matrices such as
  $\mathbf X = [x_{ik}]_{M \times D}$ and
  $\mathbf Y = [y_{jk}]_{N \times D}$.<br>

- fast computation on **GPU** without memory overflow, especially to
  process **very large dimensions** $M$ and $N$ (e.g. $\approx 10^4$ or
  $10^6$) over indexes $i$ and $j$.<br>

- automatic differentiation and **gradient computations** for
  user-defined operators.<br>

------------------------------------------------------------------------

# Installing and using RKeOps

``` r
# install rkeops
install.packages("rkeops")
# load rkeops
library(rkeops)
# create a dedicated Python environment with reticulate (to be done only once)
reticulate::virtualenv_create("rkeops")
# activate the dedicated Python environment
reticulate::use_virtualenv(virtualenv = "rkeops", required = TRUE)
# install rkeops requirements (to be done only once)
install_rkeops()
```

For more details, see the specific **“Using RKeOps”
[article](https://www.kernel-operations.io/rkeops/articles/using_rkeops.html)**
or the corresponding vignette:

``` r
vignette("using_rkeops", package = "rkeops")
```

------------------------------------------------------------------------

# Examples in R

## Example in R with LazyTensors

Lazy evaluation allows to write intermediate computations as symbolic
operations that are not directly evaluated. The real evaluation is only
made on final computation.

To do so, you can use `LazyTensors`. These are objects wrapped around R
matrices or vectors used to create symbolic formulas for the `KeOps`
reduction operations. A typical use case is the following:

Let us say that we want to compute (for all $j=1,\dots,N$):

$$
a_j = \sum_{i=1}^{M} \exp\left(-\frac{\Vert \mathbf x_i - \mathbf y_j\Vert^2}{s^2}\right)
$$

with

- parameter: $s \in\mathbb R$<br>

- $i$-indexed variables
  $\mathbf X = [\mathbf x_i]_{i=1,...,M} \in\mathbb R^{M\times 3}$<br>

- $j$-indexed variables
  $\mathbf Y = [\mathbf y_j]_{j=1,...,N} \in\mathbb R^{N\times 3}$<br><br>

The associated code would be:

``` r
# to run computation on CPU (default mode)
rkeops_use_cpu()
# OR
# to run computations on GPU (to be used only if relevant)
rkeops_use_gpu()

# Data
M <- 10000
N <- 15000
x <- matrix(runif(N * 3), nrow = M, ncol = 3) # arbitrary R matrix representing 
                                              # 10000 data points in R^3
y <- matrix(runif(M * 3), nrow = N, ncol = 3) # arbitrary R matrix representing 
                                              # 15000 data points in R^3
s <- 0.1                                      # scale parameter

# Turn our Tensors into KeOps symbolic variables:
x_i <- LazyTensor(x, "i")     # symbolic object representing an arbitrary row of x, 
                              # indexed by the letter "i"
y_j <- LazyTensor(y, "j")     # symbolic object representing an arbitrary row of y, 
                              # indexed by the letter "j"

# Perform large-scale computations, without memory overflows:
D_ij <- sum((x_i - y_j)^2)    # symbolic matrix of pairwise squared distances, 
                              # with 10000 rows and 15000 columns

K_ij <- exp(- D_ij / s^2)     # symbolic matrix, 10000 rows and 15000 columns

# D_ij and K_ij are only symbolic at that point, no computation is done

# Computing the result without storing D_ij and K_ij:
a_j <- sum(K_ij, index = "i") # actual R matrix (in fact a row vector of 
                              # length 15000 here)
                              # containing the column sums of K_ij
                              # (i.e. the sums over the "i" index, for each 
                              # "j" index)
```

**More in the dedicated “RKeOps LazyTensor”
[article](https://www.kernel-operations.io/rkeops/articles/LazyTensor_rkeops.html)**
or in the corresponding vignette:

``` r
vignette("LazyTensor_rkeops", package = "rkeops")
```

## Example in R with generic reductions

We want to implement with RKeOps the following mathematical formula
$$\sum_{j=1}^{N} \exp\Big(-\sigma || \mathbf x_i - \mathbf y_j ||_2^{\,2}\Big)\,\mathbf b_j$$
with

- parameter: $\sigma\in\mathbb R$<br>

- $i$-indexed variables
  $\mathbf X = [\mathbf x_i]_{i=1,...,M} \in\mathbb R^{M\times 3}$<br>

- $j$-indexed variables
  $\mathbf Y = [\mathbf y_j]_{j=1,...,N} \in\mathbb R^{N\times 3}$ and
  $\mathbf B = [\mathbf b_j]_{j=1,...,N} \in\mathbb R^{N\times 6}$<br><br>

In R, we can define the corresponding KeOps formula as a **simple text
string**:

``` r
formula = "Sum_Reduction(Exp(-s * SqNorm2(x - y)) * b, 0)"
```

- `SqNorm2` = squared $\ell_2$ norm
- `Exp` = exponential
- `Sum_reduction(..., 0)` = sum reduction over the dimension 0 i.e. sum
  on the $j$’s (1 to sum over the $i$’s)<br>

and the corresponding arguments of the formula, i.e. parameters or
variables indexed by $i$ or $j$ with their corresponding inner
dimensions:

``` r
args = c("x = Vi(3)",      # vector indexed by i (of dim 3)
         "y = Vj(3)",      # vector indexed by j (of dim 3)
         "b = Vj(6)",      # vector indexed by j (of dim 6)
         "s = Pm(1)")      # parameter (scalar) 
```

Then we just compile the corresponding operator and apply it to some
data

``` r
# compilation
op <- keops_kernel(formula, args)
# data and parameter values
nx <- 100
ny <- 150
X <- matrix(runif(nx*3), nrow=nx)   # matrix 100 x 3
Y <- matrix(runif(ny*3), nrow=ny)   # matrix 150 x 3
B <- matrix(runif(ny*6), nrow=ny)   # matrix 150 x 6
s <- 0.2
# computation (order of the input arguments should be similar to `args`)
res <- op(list(X, Y, B, s))
```

------------------------------------------------------------------------

# CPU and GPU computing

Based on your LazyTensor-based operations or formulae, RKeOps compiles
on the fly operators that can be used to run the corresponding
computations on CPU or GPU, it uses a tiling scheme to decompose the
data and avoid (i) useless and costly memory transfers between host and
GPU (performance gain) and (ii) memory overflow.

> **Note:** You can use the same code (i.e. define the same operators)
> for CPU or GPU computing. The only difference will be the compiler
> used for the compilation of your operators (upon the availability of
> CUDA on your system).

To use CPU computing mode, you can call `rkeops_use_cpu()` (with an
optional argument `ncore` specifying the number of cores used to run
parallel computations).

To use GPU computing mode, you can call `rkeops_use_gpu()` (with an
optional argument `device` to choose a specific GPU id to run
computations).

------------------------------------------------------------------------

# Matrix reduction and kernel operator

The general framework of RKeOps (and KeOps) is to provide fast and
scalable matrix operations on GPU, in particular kernel-based
computations of the form
$$\underset{i=1,...,M}{\text{reduction}}\ G(\boldsymbol{\sigma}, \mathbf x_i, \mathbf y_j) \ \ \ \ \text{or}\ \ \ \ \underset{j=1,...,N}{\text{reduction}}\ G(\boldsymbol{\sigma}, \mathbf x_i, \mathbf y_j)$$
where<br>

- $\boldsymbol{\sigma}\in\mathbb R^L$ is a vector of parameters<br>

- $\mathbf x_i\in \mathbb{R}^D$ and $\mathbf y_j\in \mathbb{R}^{D'}$ are
  two vectors of data (potentially with different dimensions)<br>

- $G: \mathbb R^L \times \mathbb R^D \times \mathbb R^{D'} \to \mathbb R$
  is a function of the data and the parameters, that can be expressed
  through a composition of generic operators<br>

- $\text{reduction}$ is a generic reduction operation over the index $i$
  or $j$ (e.g. sum)<br><br>

RKeOps creates (and compiles on the fly) an operator implementing your
formula. You can apply it to your data, or compute its gradient
regarding some data points.<br><br>

> ***Note:*** You can use a wide range of reduction such as `sum`,
> `min`, `argmin`, `max`, `argmax`, etc.

## What you need to do

To use RKeOps you only need to express your computations as a formula
with the previous form.<br><br>

RKeOps allows to use a wide range of mathematical functions to define
your operators (see
<https://www.kernel-operations.io/keops/api/math-operations.html>).<br><br>

You can use two type of input matrices with RKeOps:<br>

- ones whose rows (or columns) are indexed by $i=1,...,M$ such as
  $\mathbf X = [x_{ik}]_{M \times D}$<br>

- others whose rows (or columns) are indexed by $j=1,...,N$ such as
  $\mathbf Y = [y_{ik'}]_{N \times D'}$<br><br>

More details about input matrices (size, storage order) are given in the
**“Using RKeOps”
[article](https://www.kernel-operations.io/rkeops/articles/using_rkeops.html)**
or the corresponding vignette::

``` r
vignette("using_rkeops", package = "rkeops")
```

## Generic kernel function

With RKeOps, you can define kernel functions
$K: \mathbb R^D \times \mathbb R^D \to \mathbb R$ such as, for some
vectors $\mathbf x_i$, $\mathbf y_j\in \mathbb{R}^D$<br>

- the linear kernel (standard scalar product)
  $K(\mathbf x_i, \mathbf y_j) = \big\langle \mathbf x_i \, , \, \mathbf y_j \big\rangle$
  <br>

- the Gaussian kernel
  $K(\mathbf x_i, \mathbf y_j) = \exp\left(-\frac{1}{2\sigma^2} \Vert \mathbf x_i - \mathbf y_j \Vert_2^{\,2}\right)$
  with $\sigma>0$ <br>

- and more…<br><br>

Then you can compute reductions based on such functions, especially when
the $M \times N$ matrix $\mathbf K = [K(\mathbf x_i, \mathbf y_j)]$ is
too large to fit into memory, such as<br>

- Kernel reduction:
  $$\sum_{i=1}^M K(\mathbf x_i, \mathbf y_j)\ \ \ \ \text{or}\ \ \ \ \sum_{j=1}^N K(\mathbf x_i, \mathbf y_j)$$

- Convolution-like operations:
  $$\sum_{i=1}^M K(\mathbf x_i, \mathbf y_j)\boldsymbol\beta_j\ \ \ \ \text{or}\ \ \ \ \sum_{j=1}^N K(\mathbf x_i, \mathbf y_j)\boldsymbol\beta_j$$
  for some vectors
  $(\boldsymbol\beta_j)_{j=1,...,N} \in \mathbb R^{N\times D}$<br><br>

- More complex operations:
  $$\sum_{i=1}^{M}\, K_1(\mathbf x_i, \mathbf y_j)\, K_2(\mathbf u_i, \mathbf v_j)\,\langle \boldsymbol\alpha_i\, ,\,\boldsymbol\beta_j\rangle \ \ \ \ \text{or}\ \ \ \ \sum_{j=1}^{N}\, K_1(\mathbf x_i, \mathbf y_j)\, K_2(\mathbf u_i, \mathbf v_j)\,\langle \boldsymbol\alpha_i\, ,\,\boldsymbol\beta_j\rangle$$
  for some kernels $K_1$ and $K_2$, and some $D$-vectors
  $(\mathbf x_i)_{i=1,...,M}, (\mathbf u_i)_{i=1,...,M}, (\boldsymbol\alpha_i)_{i=1,...,M} \in \mathbb R^{M\times D}$
  and
  $(\mathbf y_j)_{j=1,...,N}, (\mathbf v_j)_{j=1,...,N}, (\boldsymbol\beta_j)_{j=1,...,N} \in \mathbb R^{N\times D}$.

# Examples/Tutorials/Benchmarks

## Using RKeOps

See this
[article](https://www.kernel-operations.io/rkeops/articles/using_rkeops.html)
or the corresponding vignette:

``` r
vignette("using_rkeops", package = "rkeops")
```

## LazyTensors

See this
[article](https://www.kernel-operations.io/rkeops/articles/LazyTensor_rkeops.html)
or the corresponding vignette:

``` r
vignette("LazyTensor_rkeops", package = "rkeops")
```

## Kernel interpolation

See this
[article](https://www.kernel-operations.io/rkeops/articles/Kernel_Interpolation_rkeops.html)
or the corresponding vignette:

``` r
vignette("kernel_interpolation_rkeops", package = "rkeops")
```

## Benchmarks

See the corresponding directory
[`rkeops/benchmarks`](https://github.com/getkeops/keops/tree/main/rkeops/benchmarks)
in the project source repository.
