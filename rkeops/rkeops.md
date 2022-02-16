![logo rkeops](man/figures/rkeops_logo.png)

RKeOps contains the R bindings for the cpp/cuda library [KeOps](https://www.kernel-operations.io/). It provides
standard R functions that can be used in any R (>=3) codes.

For a full documentation you may read:

* [Installation](https://www.kernel-operations.io/keops/introduction/installation.html)
* [Documentation](https://www.kernel-operations.io/)
* [Learning KeOps syntax with examples](https://www.kernel-operations.io/keops/_auto_examples/index.html)
* [Tutorials gallery](https://www.kernel-operations.io/keops/_auto_tutorials/index.html)

# Authors

Please contact us for any **bug report**, **question** or **feature request** by filing
a report on our [GitHub issue tracker](https://github.com/getkeops/keops/issues)!

**Core library - KeOps, PyKeOps, KeOpsLab:**

- [Benjamin Charlier](https://imag.umontpellier.fr/~charlier/), from the University of Montpellier.
- [Jean Feydy](https://www.jeanfeydy.com), from Imperial College London.
- [Joan Alexis Glaunès](http://helios.mi.parisdescartes.fr/~glaunes/), from the University of Paris.

**R bindings - RKeOps:**

- [Ghislain Durif](https://gdurif.perso.math.cnrs.fr/), from the University of Montpellier.

**Contributors:**

- [François-David Collin](https://github.com/fradav), from the University of Montpellier: Tensordot operation, CI setup.
- [Tanguy Lefort](https://github.com/tanglef), from the University of Montpellier: conjugate gradient solver.
- [Mauricio Diaz](https://github.com/mdiazmel), from Inria of Paris: CI setup.
- [Benoît Martin](https://github.com/benoitmartin88), from the Aramis Inria team: multi-GPU support.
- [Francis Williams](https://www.fwilliams.info), from New York University: maths operations.
- [Kshiteej Kalambarkar](https://github.com/kshitij12345), from Quansight: maths operations.
- [D. J. Sutherland](https://djsutherland.ml), from the TTI-Chicago: bug fix in the Python package.
- [David Völgyes](https://scholar.google.no/citations?user=ngT2GvMAAAAJ&hl=en), from the Norwegian Institute of Science and Technology: bug fix in the formula parser.


Beyond explicit code contributions, KeOps has grown out of numerous discussions with applied mathematicians and machine learning experts. We would especially like to thank 
[Alain Trouvé](https://atrouve.perso.math.cnrs.fr/), 
[Stanley Durrleman](https://who.rocq.inria.fr/Stanley.Durrleman/), 
[Gabriel Peyré](http://www.gpeyre.com/) and 
[Michael Bronstein](https://people.lu.usi.ch/bronstem/)
for their valuable suggestions and financial support.

# Citation

If you use this code in a research paper, **please cite** our 
[original publication](https://jmlr.org/papers/v22/20-275.html):

> Charlier, B., Feydy, J., Glaunès, J. A., Collin, F.-D. & Durif, G. Kernel Operations on the GPU, with Autodiff, without Memory Overflows. Journal of Machine Learning Research 22, 1–6 (2021).

```tex
@article{JMLR:v22:20-275,
  author  = {Benjamin Charlier and Jean Feydy and Joan Alexis Glaunès and François-David Collin and Ghislain Durif},
  title   = {Kernel Operations on the GPU, with Autodiff, without Memory Overflows},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {74},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v22/20-275.html}
}
```

For applications to **geometric (deep) learning**, 
you may also consider our [NeurIPS 2020 paper](https://www.jeanfeydy.com/Papers/KeOps_NeurIPS_2020.pdf):

```tex
@article{feydy2020fast,
    title={Fast geometric learning with symbolic matrices},
    author={Feydy, Jean and Glaun{\`e}s, Joan and Charlier, Benjamin and Bronstein, Michael},
    journal={Advances in Neural Information Processing Systems},
    volume={33},
    year={2020}
}
```

---

# Details
The KeOps library provides seamless kernel operations on GPU, with 
auto-differentiation and without memory overflows.

With RKeOps, you can compute generic reductions of very large arrays whose 
entries are given by a mathematical formula. It combines a tiled reduction 
scheme with an automatic differentiation engine. It is perfectly suited to 
the computation of Kernel dot products and the associated gradients, even 
when the full kernel matrix does not fit into the GPU memory.

For more information (installation, usage), please visit 
<https://www.kernel-operations.io/> (especially the section dedicated to 
RKeOps) and read the vignettes available in R with the command 
`browseVignettes("rkeops")` or on the CRAN.

# Installation

## SystemRequirements

* C++11: you should have a C++ compiler compatible with C++11 (see 
  <https://en.cppreference.com/w/cpp/compiler_support> for details). We 
  recommend GCC or clang.
* cmake (>=3.10): should be available in standard path. You can get cmake 
  at <https://cmake.org/>.
* clang (optional): clang compiler is supported.
* CUDA (optional but recommended for performance): you need CUDA to enable GPU 
  computing in RKeOps. CUDA can be found at 
  <https://developer.nvidia.com/cuda-downloads>.
* At the moment, RKeOps is not available for Windows.

## Install from CRAN

> **Note:** RKeOps is avaible on CRAN but only for UNIX environment (GNU/Linux and MacOS) and not for Windows.

```R
install.packages("rkeops")
```

## Install from sources

> !! In most recent version of devtools, the `args` argument is not available anymore and it is not possible to use `devtools::install_git`. Please check next section to install from sources.

* Install directly from Github (requires `git`)
```R
devtools::install_git("https://github.com/getkeops/keops", 
                      ref = "v1.5",
                      subdir = "rkeops", 
                      args="--recursive")
# not possible to use `devtools::intall_github()` because of the required submodule
```

## Get sources and install from local repository

* Get KeOps sources (bash command)
```bash
git clone --recurse-submodules="keops/lib/sequences" -b v1.5 https://github.com/getkeops/keops
# or
git clone -b v1.5 https://github.com/getkeops/keops
cd keops
git submodule update --init -- keops/lib/sequences
# other submodules are not necessary for RKeOps
```

* Install from local source in R (assuming you are in the `keops` directory)
```R
devtools::install("rkeops")
```

# Quick start

## Defining a new operator

Here is an example how to define and compute a Gaussian convolution with RKeOps.

```R
# implementation of a convolution with a Gaussian kernel
formula = "Sum_Reduction(Exp(-s * SqNorm2(x - y)) * b, 0)"

# input arguments
args = c("x = Vi(3)",      # vector indexed by i (of dim 3)
         "y = Vj(3)",      # vector indexed by j (of dim 3)
         "b = Vj(6)",      # vector indexed by j (of dim 6)
         "s = Pm(1)")      # parameter (scalar)

# compilation of the corresponding operator
op <- keops_kernel(formula, args)

# data and parameter values
nx <- 100
ny <- 150
X <- matrix(runif(nx*3), nrow=nx)   # matrix 100 x 3
Y <- matrix(runif(ny*3), nrow=ny)   # matrix 150 x 3
B <- matrix(runif(ny*6), nrow=ny)   # matrix 150 x 6
s <- 0.2

# to run computation on CPU (default mode)
use_cpu()
# to run computations on GPU (to be used only if relevant)
use_gpu()

# computation (order of the input arguments should be similar to `args`)
res <- op(list(X, Y, B, s))
```

## Computing gradients

Here is an example how to define and compute the gradient of an existing KeOps operators.

```R
# defining an operator (reduction on squared distance)
formula <- "Sum_Reduction(SqNorm2(x-y), 0)"
args <- c("x=Vi(0,3)", "y=Vj(1,3)")
op <- keops_kernel(formula, args)
# defining its gradient regarding x
grad_op <- keops_grad(op, var="x")

# data
nx <- 100
ny <- 150
x <- matrix(runif(nx*3), nrow=nx, ncol=3)     # matrix 100 x 3
y <- matrix(runif(ny*3), nrow=ny, ncol=3)     # matrix 150 x 3
eta <- matrix(runif(nx*1), nrow=nx, ncol=1)   # matrix 100 x 1

# computation
input <- list(x, y, eta)
res <- grad_op(input)
```


## CPU and GPU computing

Based on your formulae, RKeOps compile on the fly operators that can be used to run the corresponding computations on CPU or GPU, it uses a tiling scheme to decompose the data and avoid (i) useless and costly memory transfers between host and GPU (performance gain) and (ii) memory overflow.

> **_Note:_** You can use the same code (i.e. define the same operators) for CPU or GPU computing. The only difference will be the compiler used for the compilation of your operators (upon the availability of CUDA on your system).

To use CPU computing mode, you can call `use_cpu()` (with an optional argument `ncore` specifying the number of cores used to run parallel computations).

To use GPU computing mode, you can call `use_gpu()` (with an optional argument `device` to choose a specific GPU id to run computations).
