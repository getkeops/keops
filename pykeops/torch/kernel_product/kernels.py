import re
import inspect
import copy

import torch

from pykeops.torch.kernel_product.formula import Formula
from pykeops.torch.kernel_product.features_kernels import FeaturesKP

# Define the standard kernel building blocks.
# They will be concatenated depending on the "name" argument of Kernel.__init__
# Feel free to add your own "pet formula" at run-time, 
# using for instance :
#  " kernel_formulas["mykernel"] = Formula(... ) "

# In some cases, due to PyTorch's behavior mainly, we have to add a small
# epsilon in front of square roots and logs. As for KeOps code,
# note that [dSqrt(x)/dx](x=0) has been conventionally set to 0.

Epsilon = "IntInv(100000000)"
epsilon = 1e-8

# Formulas in "x_i" and "y_j", with parameters "g" (=1/sigma^2, for instance)
kernel_formulas = dict(
    linear=Formula(  # Linear kernel
        formula_sum="({X}|{Y})",
        routine_sum=lambda xsy=None, **kwargs: xsy,
        formula_log="(IntInv(2) * Log(Square(({X}|{Y})) + " + Epsilon + "))",
        routine_log=lambda xsy=None, **kwargs: .5 * (xsy ** 2 + epsilon).log()
    ),
    distance=Formula(  # -1* Energy distance kernel
        formula_sum="Sqrt(WeightedSqDist({G},{X},{Y}))",
        routine_sum=lambda gxmy2=None, **kwargs: gxmy2.sqrt(),
        formula_log="(IntInv(2) * Log(WeightedSqDist({G},{X},{Y})) + " + Epsilon + "))",
        routine_log=lambda gxmy2=None, **kwargs: .5 * (gxmy2 ** 2 + epsilon).log()
    ),
    gaussian=Formula(  # Standard RBF kernel
        formula_sum="Exp( -(WeightedSqDist({G},{X},{Y})))",
        routine_sum=lambda gxmy2=None, **kwargs: (-gxmy2).exp(),
        formula_log="(-(WeightedSqDist({G},{X},{Y})))",
        routine_log=lambda gxmy2=None, **kwargs: -gxmy2,
    ),
    cauchy=Formula(  # Heavy tail kernel
        formula_sum="Inv( IntCst(1) + WeightedSqDist({G},{X},{Y}))",
        routine_sum=lambda gxmy2=None, **kwargs: 1. / (1 + gxmy2),
        formula_log="(IntInv(-1) * Log(IntCst(1) + WeightedSqDist({G},{X},{Y})))",
        routine_log=lambda gxmy2=None, **kwargs: -(1 + gxmy2).log(),
    ),
    laplacian=Formula(  # Pointy kernel
        formula_sum="Exp(-Sqrt( WeightedSqDist({G},{X},{Y})))",
        routine_sum=lambda gxmy2=None, **kwargs: (-(gxmy2 + epsilon).sqrt()).exp(),
        formula_log="(-Sqrt(WeightedSqDist({G},{X},{Y})))",
        routine_log=lambda gxmy2=None, **kwargs: -(gxmy2 + epsilon).sqrt(),
    ),
    inverse_multiquadric=Formula(  # Heavy tail kernel
        formula_sum="Inv(Sqrt(IntCst(1) + WeightedSqDist({G},{X},{Y})))",
        routine_sum=lambda gxmy2=None, **kwargs: torch.rsqrt(1 + gxmy2),
        formula_log="(IntInv(-2) * Log(IntCst(1) + WeightedSqDist({G},{X},{Y})))",
        routine_log=lambda gxmy2=None, **kwargs: -.5 * (1 + gxmy2).log(),
    ))


def set_indices(formula, f_ind, v_ind) :
    """
    Modify the patterns stored in kernel_formulas, taking into account the fact that
    the current formula is the f_ind-th, working with the v_ind-th pair of variables.
    """

    # KeOps backend -------------------------------------------------------------------------
    n_params = formula.n_params
    n_vars = formula.n_vars

    if n_params == 1:
        G_str = "G_"+str(f_ind)
    else:
        G_str = None

    if n_vars == 2:
        X_str, Y_str = "X_" + str(v_ind), "Y_" + str(v_ind)
    else:
        X_str, Y_str = None, None

    formula.formula_sum = formula.formula_sum.format(G = G_str, X = X_str, Y = Y_str)
    formula.formula_log = formula.formula_log.format(G = G_str, X = X_str, Y = Y_str)

    # Vanilla PyTorch backend -------------------------------------------------------------------
    # Guess which quantities will be needed by the vanilla pytorch binding:
    params_sum = inspect.signature(formula.routine_sum).parameters
    needs_x_y_gxmy2_xsy_sum = (v_ind, 'x' in params_sum, 'y' in params_sum, 'gxmy2' in params_sum, 'xsy' in params_sum)
    formula.subroutine_sum = formula.routine_sum
    formula.routine_sum = lambda x=None, y=None, g=None, gxmy2=None, xsy=None : \
                          formula.subroutine_sum(x=x[v_ind], y=y[v_ind], gxmy2=gxmy2[f_ind], xsy=xsy[f_ind])
    
    params_log = inspect.signature(formula.routine_log).parameters
    needs_x_y_gxmy2_xsy_log = (v_ind, 'x' in params_log, 'y' in params_log, 'gxmy2' in params_log, 'xsy' in params_log)
    formula.subroutine_log = formula.routine_log
    formula.routine_log = lambda x=None, y=None, g=None, gxmy2=None, xsy=None : \
                          formula.subroutine_log(x=x[v_ind], y=y[v_ind], gxmy2=gxmy2[f_ind], xsy=xsy[f_ind])
    return formula, f_ind+1, needs_x_y_gxmy2_xsy_sum, needs_x_y_gxmy2_xsy_log


class Kernel:
    """Defines a new Kernel identifier for :func:`kernel_product`.
    
    """
    def __init__(self, name=None, formula_sum=None, routine_sum=None, formula_log=None, routine_log=None):
        """
        Examples of valid names :
            " gaussian(x,y) * linear(u,v)**2 * gaussian(s,t)"
            " gaussian(x,y) * (1 + linear(u,v)**2 ) "
        """
        if name is not None:
            # in the comments, let's suppose that name="gaussian(x,y) + laplacian(x,y) * linear(u,v)**2"
            # Determine the features type from the formula : ------------------------------------------------
            variables = re.findall(r'(\([a-z],[a-z]\))', name) # ['(x,y)', '(x,y)', '(u,v)']
            used = set()
            variables = [x for x in variables if x not in used and (used.add(x) or True)]
            #         = ordered, "unique" list of pairs "(x,y)", "(u,v)", etc. used
            #         = ['(x,y)', '(u,v)']
            var_to_ind = {k: i for (i, k) in enumerate(variables)}
            #          = {'(x,y)': 0, '(u,v)': 1}

            subformulas_str = re.findall(r'([a-zA-Z_][a-zA-Z_0-9]*)(\([a-z],[a-z]\))', name)
            #               = [('gaussian', '(x,y)'), ('laplacian', '(x,y)'), ('linear', '(u,v)')]

            # f_ind = index of the current formula
            # subformulas = list of formulas used in the kernel_product
            # vars_needed_sum and vars_needed_log keep in mind the symbolic pre-computations
            # |x-y|^2 and <x,y> that may be needed by the Vanilla PyTorch backend.
            f_ind, subformulas, vars_needed_sum, vars_needed_log = 0, [], [], []
            for formula_str, var_str in subformulas_str:
                # Don't forget the copy! This code should have no side effect on kernel_formulas!
                formula = copy.copy(kernel_formulas[formula_str]) # = Formula(...)
                # Modify the symbolic "formula" to let it take into account the formula and variable indices:
                formula, f_ind, need_sum, need_log = set_indices(formula, f_ind, var_to_ind[var_str])
                # Store everyone for later use and substitution:
                subformulas.append(formula)
                vars_needed_sum.append(need_sum)
                vars_needed_log.append(need_log)
                
            # One after another, replace the symbolic "name(x,y)" by references to our list of "index-aware" formulas
            for (i, _) in enumerate(subformulas):
                name = re.sub(r'[a-zA-Z_][a-zA-Z_0-9]*\([a-z],[a-z]\)',  r'subformulas[{}]'.format(i), name, count=1)
            #        = "subformulas[0] + subformulas[1] * subformulas[2]**2"

            # Replace int values "N" with "Formula(intvalue=N)"" (except the indices of subformulas!)
            name = re.sub(r'(?<!subformulas\[)([0-9]+)', r'Formula(intvalue=\1)', name)

            # Final result : ----------------------------------------------------------------------------------
            kernel = eval(name) # It's a bit dirty... Please forgive me !
            
            # Store the required info
            self.formula_sum = kernel.formula_sum
            self.routine_sum = kernel.routine_sum
            self.formula_log = kernel.formula_log
            self.routine_log = kernel.routine_log
            # Two lists, needed by the vanilla torch binding
            self.routine_sum.vars_needed = vars_needed_sum
            self.routine_log.vars_needed = vars_needed_log

        else :
            self.formula_sum = formula_sum
            self.routine_sum = routine_sum
            self.formula_log = formula_log
            self.routine_log = routine_log


def kernel_product(params, x, y, *bs, mode=None, backend=None, cuda_type='float32'):
    """:doc:`Math-friendly wrapper <kernel-product>` around the :func:`Genred` routine. 

    This routine allows you to compute kernel dot products (aka. as discrete convolutions)
    with arbitrary formulas, using a **Sum** or a **LogSumExp** reduction operation.
    It is syntactic sugar, meant to ease the implementation of mixture models
    on point clouds.
    
    Its use is explained in the :doc:`documentation <kernel-product>`
    and showcased in the :doc:`anisotropic kernels <../_auto_examples/plot_anisotropic_kernels>`
    and :doc:`GMM-fitting <../_auto_tutorials/gaussian_mixture/plot_gaussian_mixture>` tutorials.

    Args:

        params (dict): Describes the kernel being used.
            It should have the following attributes :

              - ``"id"`` (:class:`Kernel`): Describes the formulas for 
                :math:`k(x_i,y_j)`, :math:`c(x_i,y_j)` 
                and the number of features `F`
                (locations, location+directions, etc.) being used.
              - ``"gamma"`` (`tuple of Tensors`): Parameterizes
                the kernel. Typically, something along the lines of
                ``(.5/σ**2, .5/τ**2, .5/κ**2)`` for Gaussian or Laplacian kernels...

        x (Tensor, or F-tuple of Tensors): 
            The `F` features associated to points :math:`x_i`, 
            for :math:`i\in [0,M)`.
            All feature Tensors should be 2d, with the
            same number of lines :math:`M` and an arbitrary number of columns.
        y (Tensor, or F-tuple of Tensors): 
            The `F` features associated to points :math:`y_j`, 
            for :math:`j\in [0,N)`.
            All feature Tensors should be 2d, with the
            same number of lines :math:`N` and an arbitrary number of columns,
            compatible with **x**.
        b ((N,E) Tensor):
            The `weights`, or `signal` vectors :math:`b_j` 
            associated to the points :math:`y_j`.
        a_log ((M,1) Tensor, optional): If **mode** is one of
            the ``"log_*"`` reductions, specifies the scalar
            variable :math:`\\text{Alog}_i`.

        b_log ((N,1) Tensor, optional): If **mode** is one of
            the ``"log_*"`` reductions, specifies the scalar
            variable :math:`\\text{Blog}_j`.

    Keyword Args:
        mode (string): Specifies the reduction operation.
            The supported values are:

              - ``"sum"`` (default):

                .. math::
                    v_i =     \sum_j k(x_i, y_j) \cdot b_j
    
              - ``"lse"``:

                .. math::
                    v_i = \log \sum_j \exp(c(x_i, y_j) + b_j )
                
                with :math:`c(x_i,y_j) = \log(k(x_i,y_j) )`
    
              - ``"log_scaled"``:

                .. math::
                    v_i = \sum_j \exp(c(x_i,y_j) + \\text{Alog}_i + \\text{Blog}_j ) \cdot b_j
            
              - ``"log_scaled_log"``:

                .. math::
                    v_i = \log \sum_j \exp(c(x_i,y_j) + \\text{Alog}_i + \\text{Blog}_j + b_j )
            
              -  ``"log_primal"`` (:math:`b_j` is not used):

                .. math::
                    v_i = \sum_j (\\text{Alog}_i+\\text{Blog}_j-1) \cdot \exp(c(x_i,y_j) + \\text{Alog}_i + \\text{Blog}_j )
                
              - ``"log_cost"`` (:math:`b_j` is not used):

                .. math::
                    v_i = \sum_j -c(x_i,y_j) \cdot \exp(c(x_i,y_j) + \\text{Alog}_i + \\text{Blog}_j )
        
        backend (string, default=``"auto"``): Specifies the implementation to run.
            The supported values are:

            - ``"auto"``: to use libkp's CPU or CUDA routines.
            - ``"pytorch"``: to fall back on a reference tensorized implementation.

        cuda_type (string, default = ``"float32"``): Specifies the numerical ``dtype`` 
            of the input and output arrays. 
            The supported values are:

              - ``cuda_type = "float32"`` or ``"float"``.
              - ``cuda_type = "float64"`` or ``"double"``.
        


    Returns:
        (M,E) Tensor:

        The output scalar or vector :math:`v_i` sampled on the
        :math:`x_i`'s.
    
    Example:
        >>> # Generate the data as pytorch tensors
        >>> x = torch.randn(1000,3, requires_grad=True)
        >>> y = torch.randn(2000,3, requires_grad=True)
        >>> b = torch.randn(2000,2, requires_grad=True)
        >>> #
        >>> # Pre-defined kernel: using custom expressions is also possible!
        >>> # Notice that the parameter sigma is a dim-1 vector, *not* a scalar:
        >>> sigma  = torch.tensor([.5], requires_grad=True)
        >>> params = {
        ...    "id"      : Kernel("gaussian(x,y)"),
        ...    "gamma"   : .5/sigma**2,
        ... }
        >>> #
        >>> # Depending on the inputs' types, 'a' is a CPU or a GPU variable.
        >>> # It can be differentiated wrt. x, y, b and sigma.
        >>> a = kernel_product(params, x, y, b)
        >>> print(a)
        tensor([[-0.0898, -0.3760],
                [-0.8888, -1.3352],
                [ 1.0236, -1.3245],
                ...,
                [ 2.5233, -2.6578],
                [ 1.3097,  4.3967],
                [ 0.4095, -0.3039]], grad_fn=<GenredAutogradBackward>)
    """

    kernel  = params["id"]
    if mode is None:    mode    = params.get("mode", "sum")
    if backend is None: backend = params.get("backend", "auto")
    # gamma should have been generated along the lines of "Variable(torch.Tensor([1/(s**2)])).type(dtype)"
    gamma   = params["gamma"]

    if not gamma.__class__ in [tuple, list]: gamma = (gamma,)
    if not     x.__class__ in [tuple, list]:     x = (x,)
    if not     y.__class__ in [tuple, list]:     y = (y,)

    return FeaturesKP(kernel, gamma, x, y, bs, mode=mode, backend=backend, cuda_type=cuda_type)
