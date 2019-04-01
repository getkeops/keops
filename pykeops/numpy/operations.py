import numpy as np

from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import load_keops
from pykeops.common.operations import ConjugateGradientSolver
from pykeops.common.parse_type import get_sizes, complete_aliases
from pykeops.common.utils import axis2cat
from pykeops.numpy import default_dtype


class KernelSolve:
    r"""
    Creates a new conjugate gradient solver.

    Supporting the same :ref:`generic syntax <part.generic_formulas>` as :func:`pykeops.numpy.Genred`,
    this module allows you to solve generic optimization problems of
    the form:

    .. math::
       & & a^{\star} & =\operatorname*{argmin}_a  \| ( \alpha \operatorname{Id}+K_{xx}) a -b \|^2_2, \\\\
        &\text{i.e.}\quad &  a^{\star} & = (\alpha \operatorname{Id} + K_{xx})^{-1}  b,

    where :math:`K_{xx}` is a **symmetric**, **positive** definite **linear** operator
    and :math:`\alpha` is a **nonnegative regularization** parameter.

    
    Warning:
        Even for variables of size 1 (e.g. :math:`a_i\in\mathbb{R}`
        for :math:`i\in[0,M)`), KeOps expects inputs to be formatted
        as 2d arrays of size ``(M,dim)``. In practice,
        ``a.view(-1,1)`` should be used to turn a vector of weights
        into a *list of scalar values*.

    Note:
        :func:`KernelSolve` relies on C++ or CUDA kernels that are compiled on-the-fly 
        and stored in ``pykeops.build_folder`` as ".dll" or ".so" files for later use.

    Args:
        formula (string): The scalar- or vector-valued expression
            that should be computed and reduced.
            The correct syntax is described in the :doc:`documentation <../../Genred>`,
            using appropriate :doc:`mathematical operations <../../../api/math-operations>`.
        aliases (list of strings): A list of identifiers of the form ``"AL = TYPE(DIM)"`` 
            that specify the categories and dimensions of the input variables. Here:

              - ``AL`` is an alphanumerical alias, used in the **formula**.
              - ``TYPE`` is a *category*. One of:

                - ``Vi``: indexation by :math:`i` along axis 0.
                - ``Vj``: indexation by :math:`j` along axis 1.
                - ``Pm``: no indexation, the input tensor is a *vector* and not a 2d array.

              - ``DIM`` is an integer, the dimension of the current variable.
            
            As described below, :meth:`__call__` will expect input arrays whose
            shape are compatible with **aliases**.
        varinvalias (string): The alphanumerical **alias** of the variable with
            respect to which we shall perform our conjugate gradient descent.
            **formula** is supposed to be linear with respect to **varinvalias**,
            but may be more sophisticated than a mere ``"K(x,y) * {varinvalias}"``.

    Keyword Args:
        alpha (float, default = 1e-10): Non-negative 
            **ridge regularization** parameter, added to the diagonal
            of the Kernel matrix :math:`K_{xx}`.

        axis (int, default = 0): Specifies the dimension of the kernel matrix :math:`K_{x_ix_j}` that is reduced by our routine. 
            The supported values are:

              - **axis** = 0: reduction with respect to :math:`i`, outputs a ``Vj`` or ":math:`j`" variable.
              - **axis** = 1: reduction with respect to :math:`j`, outputs a ``Vi`` or ":math:`i`" variable.

        dtype (string, default = ``"float32"``): Specifies the numerical ``dtype`` of the input and output arrays. 
            The supported values are:

              - **dtype** = ``"float32"`` or ``"float"``.
              - **dtype** = ``"float64"`` or ``"double"``.

    **To apply the routine on arbitrary NumPy arrays:**
        
    
    Args:
        *args (2d arrays (variables ``Vi(..)``, ``Vj(..)``) and 1d arrays (parameters ``Pm(..)``)): The input numerical arrays, 
            which should all have the same ``dtype``, be **contiguous** and be stored on 
            the **same device**. KeOps expects one array per alias, 
            with the following compatibility rules:

                - All ``Vi(Dim_k)`` variables are encoded as **2d-arrays** with ``Dim_k`` columns and the same number of lines :math:`M`.
                - All ``Vj(Dim_k)`` variables are encoded as **2d-arrays** with ``Dim_k`` columns and the same number of lines :math:`N`.
                - All ``Pm(Dim_k)`` variables are encoded as **1d-arrays** (vectors) of size ``Dim_k``.

    Keyword Args:
        backend (string): Specifies the map-reduce scheme,
            as detailed in the documentation 
            of the :func:`Genred` module.

        device_id (int, default=-1): Specifies the GPU that should be used 
            to perform   the computation; a negative value lets your system 
            choose the default GPU. This parameter is only useful if your 
            system has access to several GPUs.

        ranges (6-uple of IntTensors, None by default):
            Ranges of integers that specify a 
            :doc:`block-sparse reduction scheme <../../sparsity>`
            with *Mc clusters along axis 0* and *Nc clusters along axis 1*,
            as detailed in the documentation 
            of the :func:`Genred` module.

            If **None** (default), we simply use a **dense Kernel matrix**
            as we loop over all indices
            :math:`i\in[0,M)` and :math:`j\in[0,N)`.

    Returns:
        (M,D) or (N,D) array:

        The solution of the optimization problem, which is always a 
        **2d-array** with :math:`M` or :math:`N` lines (if **axis** = 1 
        or **axis** = 0, respectively) and a number of columns 
        that is inferred from the **formula**.

    

    Example:
        >>> formula = "Exp(-Norm2(x - y)) * a"  # Exponential kernel
        >>> aliases =  ["x = Vi(3)",  # 1st input: target points, one dim-3 vector per line
        ...             "y = Vj(3)",  # 2nd input: source points, one dim-3 vector per column
        ...             "a = Vj(2)"]  # 3rd input: source signal, one dim-2 vector per column
        >>> K = Genred(formula, aliases, axis = 1)  # Reduce formula along the lines of the kernel matrix
        >>> K_inv = KernelSolve(formula, aliases, "a",  # The formula above is linear wrt. 'a'
        ...                     axis = 1, alpha = .1)   # Let's try not to overfit the data...
        >>> # Generate some random data:
        >>> x = np.random.randn(10000, 3)  # Sampling locations
        >>> b = np.random.randn(10000, 2)  # Random observed signal
        >>> a = K_inv(x, x, b)  # Linear solve: a_i = (.1*Id + K(x,x)) \ b
        >>> print(a.shape)
        (10000, 2) 
        >>> # Mean squared error:
        >>> print( ( np.sum( np.sqrt( ( .1 * a + K(x,x,a) - b)**2 ) ) / len(x) ).item() )
        1.5619025770417854e-06
    
    """
    
    def __init__(self, formula, aliases, varinvalias, alpha=1e-10, axis=0, dtype=default_dtype, opt_arg=None):
        reduction_op = 'Sum'
        if opt_arg:
            self.formula = reduction_op + '_Reduction(' + formula + ',' + str(opt_arg) + ',' + str(axis2cat(axis)) + ')'
        else:
            self.formula = reduction_op + '_Reduction(' + formula + ',' + str(axis2cat(axis)) + ')'
        self.aliases = complete_aliases(formula, aliases)
        self.varinvalias = varinvalias
        self.dtype = dtype
        self.myconv = load_keops(self.formula, self.aliases, self.dtype, 'numpy')
        self.alpha = alpha
        tmp = aliases.copy()
        for (i, s) in enumerate(tmp):
            tmp[i] = s[:s.find("=")].strip()
        self.varinvpos = tmp.index(varinvalias)
    
    def __call__(self, *args, backend='auto', device_id=-1, eps=1e-6, ranges=None):
        # Get tags
        tagCpuGpu, tag1D2D, _ = get_tag_backend(backend, args)
        nx, ny = get_sizes(self.aliases, *args)
        varinv = args[self.varinvpos]
        
        if ranges is None: ranges = ()  # ranges should be encoded as a tuple

        def linop(var):
            newargs = args[:self.varinvpos] + (var,) + args[self.varinvpos + 1:]
            res = self.myconv.genred_numpy(nx, ny, tagCpuGpu, tag1D2D, 0, device_id, ranges, *newargs)
            if self.alpha:
                res += self.alpha * var
            return res

        return ConjugateGradientSolver('numpy', linop, varinv, eps=eps)
