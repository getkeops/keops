Using RKeOps
============

.. _section-1:

2023-03-31
^^^^^^^^^^

.. container::
   :name: TOC

   -  `Installing RKeOps <#installing-rkeops>`__

      -  `Requirements <#requirements>`__
      -  `Python requirement <#python-requirement>`__
      -  `Install from CRAN <#install-from-cran>`__
      -  `Install development version <#install-development-version>`__

   -  `How to use RKeOps <#how-to-use-rkeops>`__

      -  `Example <#example>`__
      -  `Formula <#formula>`__
      -  `Arguments <#arguments>`__

         -  `Input matrix <#input-matrix>`__
         -  `Notations <#notations>`__

      -  `Creating a new operator <#creating-a-new-operator>`__
      -  `Run computations <#run-computations>`__
      -  `Computing gradients <#computing-gradients>`__
      -  `RKeOps options <#rkeops-options>`__

         -  `Choosing CPU or GPU computing at
            runtime <#choosing-cpu-or-gpu-computing-at-runtime>`__
         -  `Computation precision <#computation-precision>`__
         -  `Verbosity <#verbosity>`__

      -  `Data storage orientation <#data-storage-orientation>`__

         -  `Compilation files and
            cleaning <#compilation-files-and-cleaning>`__

| RKeOps is a R front-end for the KeOps C++/Cuda library. It provides
  standard functions that can be used in any R code.

Thanks to RKeOps, you can use **GPU computing directly inside R**
without the cost of developing a specific CUDA implementation of your
custom mathematical operators.

.. container:: section level1
   :name: installing-rkeops

   .. rubric:: Installing RKeOps
      :name: installing-rkeops

   .. container:: section level2
      :name: requirements

      .. rubric:: Requirements
         :name: requirements

      -  R (tested with R >= 4.2)
      -  C++ compiler (g++ >=7 or clang) for CPU computing or CUDA
         compiler (nvcc >=10) and CUDA libs for GPU computing
      -  Python (>= 3.10)

      **Important:** Python is a requirement as an intern machinery for
      the package to work but you will not need to create nor manipulate
      Python codes to use the RKeOps package.

      **Disclaimer:** KeOps (including RKeOps) is not functional on
      Windows, it was only tested on Linux and MacOS.

   .. container:: section level2
      :name: python-requirement

      .. rubric:: Python requirement
         :name: python-requirement

      RKeOps now uses PyKeOps Python package under the hood thanks to
      the ```reticulate`` <https://rstudio.github.io/reticulate/>`__ R
      package that provides an “R Interface to Python”.

      We recommend to use a dedicated Python environment (through
      ``reticulate``) to install and use RKeOps Python dependencies.
      **To do so, you can run the following code before installing
      RKeOps:**

      .. container:: sourceCode
         :name: cb1

         .. code:: r

            install.packages("reticulate")
            library(reticulate)
            virtualenv_create("rkeops")
            use_virtualenv(virtualenv = "rkeops", required = TRUE)
            py_config()

      ..

         **Note:** To get more information about **managing** which
         **version of Python** you are using, you can refer to
         ``reticulate``
         `documentation <https://rstudio.github.io/reticulate/articles/versions.html>`__
         about “Python Version Configuration”. In particular, if you are
         a Miniconda/Anaconda Python distribution user, you can either
         user a Python virtual environment (c.f. above) or a **Conda
         environment** with the same results. Please refer to the
         ``reticulate``
         `documentation <https://rstudio.github.io/reticulate/articles/versions.html#providing-hints>`__
         in this case.

   .. container:: section level2
      :name: install-from-cran

      .. rubric:: Install from CRAN
         :name: install-from-cran

      RKeOps >=2 is not available on CRAN yet, please refer to the next
      section.

   .. container:: section level2
      :name: install-development-version

      .. rubric:: Install development version
         :name: install-development-version

      .. container:: sourceCode
         :name: cb2

         .. code:: r

            install.packages("remotes")
            remotes::install_github("getkeops/keops", subdir = "rkeops")

      --------------

.. container:: section level1
   :name: how-to-use-rkeops

   .. rubric:: How to use RKeOps
      :name: how-to-use-rkeops

   If you are using a Python environment (c.f.
   `previously <#python-requirement>`__), you should always run the
   following commands before loading RKeOps:

   .. container:: sourceCode
      :name: cb3

      .. code:: r

         use_virtualenv(virtualenv = "rkeops", required = TRUE)
         py_config()

   Load RKeOps in R:

   .. container:: sourceCode
      :name: cb4

      .. code:: r

         library(rkeops)

   You can verify that every thing is ok:

   .. container:: sourceCode
      :name: cb5

      .. code:: r

         check_rkeops()

   RKeOps allows to define and compile new operators that run
   computations on GPU.

   .. container:: section level2
      :name: example

      .. rubric:: Example
         :name: example

      .. container:: sourceCode
         :name: cb6

         .. code:: r

            # implementation of a convolution with a Gaussian kernel
            formula = "Sum_Reduction(Exp(-s * SqNorm2(x - y)) * b, 0)"
            # input arguments
            args = c("x = Vi(3)",      # vector indexed by i (of dim 3)
                     "y = Vj(3)",      # vector indexed by j (of dim 3)
                     "b = Vj(6)",      # vector indexed by j (of dim 6)
                     "s = Pm(1)")      # parameter (scalar)
            # compilation
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

      The different elements (formula, arguments, compilation,
      computation) in the previous example will be detailed in the next
      sections.

   .. container:: section level2
      :name: formula

      .. rubric:: Formula
         :name: formula

      To use RKeOps and define new operators, you need to write the
      corresponding *formula* which is a text string defining a
      composition of mathematical operations. It should be characterized
      by two elements:

      1. a composition of generic functions applied to some input
         matrices, whose one of their dimensions is either indexed by
         \\(i=1,...,M\\) or \\(j=1,...,N\\)

      2. a reduction over indexes \\(i=1,...,M\\) (row-wise) or
         \\(j=1,...,N\\) (column-wise) of the \\(M \\times N\\) matrix
         whose entries are defined by 1.

      | RKeOps implements a wide range of mathematical operators and
        reduction: please refers to
        https://www.kernel-operations.io/keops/api/math-operations.html
        for more details.

      **Example:** We want to implement the following kernel-based
      reduction (convolution with a Gaussian kernel):
      \\[\\sum\_{j=1}^{N} \\exp\\Big(-\\sigma \|\| \\mathbf x_i -
      \\mathbf y_j \||_2^{\\,2}\\Big)\\,\\mathbf b_j\\] with

      -  | parameter: \\(\\sigma\\in\\mathbb R\\)

      -  | \\(i\\)-indexed variables \\([\\mathbf x_i]\_{i=1,...,M}
           \\in\\mathbb R^{M\\times 3}\\)

      -  | \\(j\\)-indexed variables \\([\\mathbf y_j]\_{j=1,...,N}
           \\in\\mathbb R^{N\\times 3}\\) and \\([\\mathbf
           b_j]\_{j=1,...,N} \\in\\mathbb R^{N\\times 6}\\)

      In R, we can define the corresponding KeOps formula as a simple
      **text string**:

      .. container:: sourceCode
         :name: cb7

         .. code:: r

            formula = "Sum_Reduction(Exp(-s * SqNorm2(x - y)) * b, 0)"

      -  ``SqNorm2`` = squared \\(\\ell_2\\) norm
      -  ``Exp`` = exponential
      -  ``Sum_reduction(..., 0)`` = sum reduction over the dimension 0
         i.e. sum on the \\(j\\)’s (1 to sum over the \\(i\\)’s)

   .. container:: section level2
      :name: arguments

      .. rubric:: Arguments
         :name: arguments

      The formula describing your computation can take several input
      arguments: variables and parameters. The input variables will
      generally corresponds to rows or columns of your data matrices,
      you need to be cautious with their dimensions.

      .. container:: section level3
         :name: input-matrix

         .. rubric:: Input matrix
            :name: input-matrix

         | You can use two type of input matrices with RKeOps:

         -  | ones whose rows (or columns) are indexed by
              \\(i=1,...,M\\) such as \\(\\mathbf X = [x\_{ik}]\_{M
              \\times D}\\)

         -  | others whose rows (or columns) are indexed by
              \\(j=1,...,N\\) such as \\(\\mathbf Y = [y\_{ik'}]\_{N
              \\times D'}\\)

         | The dimensions over indexes \\(i\\) or \\(j\\) are called the
           **outer dimensions** (i.e. \\(M\\) or \\(N\\)). The other
           dimensions (i.e. \\(D\\) or \\(D'\\)) are called the **inner
           dimensions**. These terms refer to the contiguity of the data
           in memory:

         -  | **Outer dimensions** \\(M\\) and \\(N\\) (over indexes
              \\(i\\) and \\(j\\) respectively) can be **very large**,
              even too large for GPU memory.

         -  | **Inner dimensions** \\(D\\) and \\(D'\\) should be
              **small** enough to fit in GPU memory, in particular to
              ensure data colocality and avoid useless memory transfers.
              Corresponding columns (or rows) should be contiguous in
              memory (this point is handled for you in RKeOps, see this
              `section <#data-storage-orientation>`__).

         ..

            **Note 1:** The outer dimension can correspond to the rows
            or the columns of the input matrices (and vice-versa for the
            inner dimension). The optimal orientation of input matrices
            is discussed in this `section <#data-storage-orientation>`__
            .

            | **Note 2:** All matrices indexed by \\(i\\) should have
              the same outer dimension \\(M\\) over \\(i\\), same for
              all matrices indexed by \\(j\\) (outer dimension \\(N\\)).
              **Only the inner dimensions \\(D\\) and \\(D'\\) should be
              known for the compilation of your operators.** The
              respective outer dimensions \\(M\\) and \\(N\\) are
              discovered at runtime (and can change from one run to
              another).

      .. container:: section level3
         :name: notations

         .. rubric:: Notations
            :name: notations

         Input arguments of the formula are defined by using keywords,
         they can be of different types:

         ======= =========================
         keyword meaning
         ======= =========================
         ``Vi``  variable indexed by ``i``
         ``Vj``  variable indexed by ``j``
         ``Pm``  parameter
         ======= =========================

         You should provide a vector of text string specifying the name
         and the type of all arguments in your formula.

         | Each keyword takes as parameter the inner dimension of the
           corresponding object. For instance, to define an input
           variable indexed by \\(i\\) corresponding to a
           \\(D\\)-dimensional vector, you can use ``"Vi(D)"``, same for
           a \\(D\\)-dimensional variable indexed by \\(j\\) being
           ``"Vj(D)"`` or a \\(D\\)-dimensional parameter ``"Pm(D)"``.

         The vector of arguments should be

         .. container:: sourceCode
            :name: cb8

            .. code:: r

               args = c("<name1>=<type1>(dim1)", "<name2>=<type2>(dim2)", "<nameX>=<typeX>(dimX)")

         where

         -  ``<nameX>`` is the name
         -  ``<type1>`` is the type (among ``Vi``, ``Vj`` or ``Pm``)
         -  ``<dimX>`` is the **inner dimension**

         | of the ``X``\ \\(^\\text{th}\\) variable in the formula.

            **Important:** The names should correspond to the ones used
            in the formula. The input parameter order will be the one
            used when calling the compiled operator.

         **Example:** We define the corresponding arguments of the
         previous `formula <#formula>`__, i.e. parameters or variables
         indexed by \\(i\\) or \\(j\\) with their corresponding inner
         dimensions:

         .. container:: sourceCode
            :name: cb9

            .. code:: r

               args = c("x = Vi(3)",      # vector indexed by i (of dim 3)
                        "y = Vj(3)",      # vector indexed by j (of dim 3)
                        "b = Vj(6)",      # vector indexed by j (of dim 6)
                        "s = Pm(1)")      # parameter (scalar)

   .. container:: section level2
      :name: creating-a-new-operator

      .. rubric:: Creating a new operator
         :name: creating-a-new-operator

      By using the function ``keops_kernel``, based on the formula and
      its arguments that we previously defined, we can compile and load
      into R the corresponding operator:

      .. container:: sourceCode
         :name: cb10

         .. code:: r

            # compilation
            op <- keops_kernel(formula, args)

      | Calling ``keops_kernel(formula, args)`` returns a function that
        can be later used to run computations on your data with your
        value of parameters. You should only be cautious with the
        similarity of each argument inner dimension.

      The returned function (here ``op``) expects a list of input values
      in the order specified in the vector ``args``.

      The result of compilation (shared library file) is stored on the
      system and will be reused when calling again the function
      ``keops_kernel`` on the same formula with the same arguments and
      the same conditions (e.g. precision), to avoid useless
      recompilation.

   .. container:: section level2
      :name: run-computations

      .. rubric:: Run computations
         :name: run-computations

      We generate data with inner dimensions (number of columns)
      corresponding to each arguments expected by the operator ``op``.
      The function ``op`` takes in input a list of input arguments. If
      the list if named, ``op`` checks the association between the
      supplied names and the names of the formula arguments. In this
      case only, it can also correct the order of the input list to
      match the expected order of arguments.

      .. container:: sourceCode
         :name: cb11

         .. code:: r

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
            res <- op(list(x, y, beta, s))

   .. container:: section level2
      :name: computing-gradients

      .. rubric:: Computing gradients
         :name: computing-gradients

      You can define gradients directly in the formula, e.g.

      .. container:: sourceCode
         :name: cb12

         .. code:: r

            # defining a formula with a Gradient
            formula <- "Grad(Sum_Reduction(SqNorm2(x-y), 0), x, eta)"
            args <- c("x=Vi(0,3)", "y=Vj(1,3)", "eta=Vi(2,1)")
            # compiling the corresponding operator
            op <- keops_kernel(formula, args)

            # data
            nx <- 100
            ny <- 150
            x <- matrix(runif(nx*3), nrow=nx, ncol=3)     # matrix 100 x 3
            y <- matrix(runif(ny*3), nrow=ny, ncol=3)     # matrix 150 x 3
            eta <- matrix(runif(nx*1), nrow=nx, ncol=1)   # matrix 100 x 1

            # computation
            input <- list(x, y, eta)
            res <- op(input)

      where ``eta`` is the new variable at which the gradient is
      computed, its dimension should correspond to the output dimension
      of the operation inside the gradient (here ``SqNorm2(x-y)`` is of
      dimension 1).

      You can also use the function ``keops_grad`` to compute partial
      derivative for existing KeOps operators.

      .. container:: sourceCode
         :name: cb13

         .. code:: r

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

      **Note:** when defining a gradient, the operator created by
      ``keops_grad``\ requires an additional variable, of the same type
      (``Vi``, ``Vj``, ``Pm``) as the variable with respect to which the
      differentiation is made (here ``x`` hence ``Vi``), and whose inner
      dimension corresponds to the output dimension of the derived
      formula (here ``SqNorm2(x-y)`` is a real-valued function, hence
      dimension 1). In addition, (if ``Vi``, ``Vj`` variable), its outer
      dimension should corresponds to the outer dimension of the
      variable regarding which the gradient is taken (here ``x``).

   .. container:: section level2
      :name: rkeops-options

      .. rubric:: RKeOps options
         :name: rkeops-options

      RKeOps behavior is driven by specific options in ``R`` global
      options scope. Such options are set up when loading RKeOps
      (i.e. by calling ``library(rkeops)``).

      You can get the current values of RKeOps options with

      .. container:: sourceCode
         :name: cb14

         .. code:: r

            get_rkeops_options()

      To (re)set RKeOps options to default values, run:

      .. container:: sourceCode
         :name: cb15

         .. code:: r

            set_rkeops_options()

      You can either use ``set_rkeops_options()`` to modify a specific
      options (c.f. ``?set_rkeops_options``) or you can use helper
      functions, c.f. below.

      .. container:: section level3
         :name: choosing-cpu-or-gpu-computing-at-runtime

         .. rubric:: Choosing CPU or GPU computing at runtime
            :name: choosing-cpu-or-gpu-computing-at-runtime

         By default, RKeOps runs computations on CPU (even for
         GPU-compiled operators). To enable GPU computing, you can run
         (before calling your operator):

         .. container:: sourceCode
            :name: cb16

            .. code:: r

               rkeops_use_gpu()

         You can also specify the GPU id that you want to use,
         e.g. ``rkeops_use_gpu(device=0)`` to use GPU 0 (default) for
         instance.

         To deactivate GPU computations, you can run
         ``rkeops_use_cpu()``.

            In CPU mode, you can control the number of CPU cores used by
            RKeOps for computations, e.g. with
            ``rkeops_use_cpu(ncore = 2)`` to run on 2 cores.

      .. container:: section level3
         :name: computation-precision

         .. rubric:: Computation precision
            :name: computation-precision

         By default, RKeOps uses 32bits float precision for
         computations. Since R only considers 64bits floating point
         numbers, if you want to use float 32bits, input data and output
         results will be casted befors and after computations
         respectively in your RKeOps operator. If your application
         requires to use 64bits float (double) precision, keep in mind
         that you will suffer a performance loss (potentially not an
         issue on high-end GPUs). In any case, compensated summation
         reduction is available in KeOps to correct for 32bits floating
         point arithmetic errors (see ``sum_scheme`` argument for
         ``keops_kernel()`` function, c.f. ``?keops_kernel``).

         You use the following functions to choose between 32bits float
         and 64bits float precision (respectively):

         .. container:: sourceCode
            :name: cb17

            .. code:: r

               rkeops_use_float32()
               rkeops_use_float64()

      .. container:: section level3
         :name: verbosity

         .. rubric:: Verbosity
            :name: verbosity

         You can use the following functions to respectively enable and
         disable compilation verbosity:

         .. container:: sourceCode
            :name: cb18

            .. code:: r

               rkeops_enable_verbosity()
               rkeops_disable_verbosity()

   .. container:: section level2
      :name: data-storage-orientation

      .. rubric:: Data storage orientation
         :name: data-storage-orientation

      | In R, matrices are stored using a column-major order, meaning
        that a \\(M \\times D\\) matrix is stored in memory as a
        succession of \\(D\\) vectors of length \\(M\\) representing
        each of its columns. A consequence is that two successive
        entries of a column are contiguous in memory, but two successive
        entries of a row are separated by \\(M\\) elements. See this
        `page <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`__
        for more details.

      **Important:** In machine learning and statistics, we generally
      use data matrices where each sample/observation/individual is a
      row, i.e. matrices where the outer dimensions correspond to rows,
      e.g. \\(\\mathbf X = [x\_{ik}]\_{M \\times D}\\), \\(\\mathbf Y =
      [y\_{ik'}]\_{N \\times D'}\\). This is the default using case of
      RKeOps. RKeOps will then automatically process the input data to
      enforce colocality along the inner dimension.

      If you want to use data where the inner dimension corresponds to
      rows of your matrices, i.e. \\(\\mathbf X^{t} = [x\_{ki}]\_{D
      \\times M}\\) or \\(\\mathbf Y^{t} = [y\_{k'i}]\_{D' \\times
      N}\\), you just need to specify the input parameter
      ``inner_dim="row"`` when calling your operator.

      Example:

      .. container:: sourceCode
         :name: cb19

         .. code:: r

            # standard column reduction of a matrix product
            op <- keops_kernel(formula = "Sum_Reduction((x|y), 1)",
                               args = c("x=Vi(3)", "y=Vj(3)"))

            # data (inner dimension = columns)
            nx <- 10
            ny <- 15
            # x_i = rows of the matrix X
            X <- matrix(runif(nx*3), nrow=nx, ncol=3)
            # y_j = rows of the matrix Y
            Y <- matrix(runif(ny*3), nrow=ny, ncol=3)
            # computing the result (here, by default `inner_dim="col"` and columns 
            # corresponds to the inner dimension)
            res <- op(list(X,Y))

            # data (inner dimension = rows)
            nx <- 10
            ny <- 15
            # x_i = columns of the matrix X
            X <- matrix(runif(nx*3), nrow=3, ncol=nx)
            # y_j = columns of the matrix Y
            Y <- matrix(runif(ny*3), nrow=3, ncol=ny)
            # computing the result (we specify `inner_dim="row"` to indicate that rows
            # corresponds to the inner dimension)
            res <- op(list(X,Y), inner_dim="row")

      .. container:: section level3
         :name: compilation-files-and-cleaning

         .. rubric:: Compilation files and cleaning
            :name: compilation-files-and-cleaning

         The compilation of new operators produces shared library (or
         share object ``.so``) files stored in a ``build`` sub-directory
         of the package installation directory, to be reused and avoid
         recompilation of already defined operators.

         You can check where your compiled operators are stored by
         running ``get_rkeops_build_dir()``. To clean RKeOps install and
         remove all shared library files, you can run
         ``clean_rkeops()``.
