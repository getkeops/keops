import torch

from pykeops.torch.generic.generic_red import GenredAutograd
from pykeops.torch import default_dtype
from pykeops.common.utils import axis2cat
from pykeops.common.parse_type import get_type, get_sizes, complete_aliases
from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import load_keops

from pykeops.torch import include_dirs

from pykeops.common.operations import ConjugateGradientSolver


class KernelSolveAutograd(torch.autograd.Function):
    """
    This class is the entry point to pytorch auto grad engine.
    """

    @staticmethod
    def forward(ctx, formula, aliases, varinvpos, alpha, backend, dtype, device_id, eps, ranges, *args):

        myconv = load_keops(formula, aliases, dtype, 'torch', ['-DPYTORCH_INCLUDE_DIR=' + ';'.join(include_dirs)])
        
        # Context variables: save everything to compute the gradient:
        ctx.formula = formula
        ctx.aliases = aliases
        ctx.varinvpos = varinvpos
        ctx.alpha = alpha
        ctx.backend = backend
        ctx.dtype = dtype
        ctx.device_id = device_id
        ctx.eps = eps
        ctx.myconv = myconv
        ctx.ranges = ranges
        if ranges is None: ranges = () # To keep the same type
            
        varinv = args[varinvpos]
        ctx.varinvpos = varinvpos

        nx, ny = get_sizes(aliases, *args)

        tagCPUGPU, tag1D2D, tagHostDevice = get_tag_backend(backend, args)

        if tagCPUGPU==1 & tagHostDevice==1:
            device_id = args[0].device.index
            for i in range(1,len(args)):
                if args[i].device.index != device_id:
                    raise ValueError("[KeOps] Input arrays must be all located on the same device.")

        def linop(var):
            newargs = args[:varinvpos] + (var,) + args[varinvpos+1:]
            res = myconv.genred_pytorch(nx, ny, tagCPUGPU, tag1D2D, tagHostDevice, device_id, ranges, *newargs)
            if alpha:
                res += alpha*var
            return res

        global copy
        result = ConjugateGradientSolver('torch',linop,varinv.data,eps)

        # relying on the 'ctx.saved_variables' attribute is necessary  if you want to be able to differentiate the output
        #  of the backward once again. It helps pytorch to keep track of 'who is who'.
        ctx.save_for_backward(*args,result)

        return result

    @staticmethod
    def backward(ctx, G):
        formula = ctx.formula
        aliases = ctx.aliases
        varinvpos = ctx.varinvpos
        backend = ctx.backend
        alpha = ctx.alpha
        dtype = ctx.dtype
        device_id = ctx.device_id
        eps = ctx.eps
        myconv = ctx.myconv
        varinvpos = ctx.varinvpos
        ranges = ctx.ranges

        args = ctx.saved_tensors[:-1]  # Unwrap the saved variables
        nargs = len(args)
        result = ctx.saved_tensors[-1]

        # If formula takes 5 variables (numbered from 0 to 4), then the gradient
        # wrt. the output, G, should be given as a 6-th variable (numbered 5),
        # with the same dim-cat as the formula's output.
        eta = 'Var(' + str(nargs) + ',' + str(myconv.dimout) + ',' + str(myconv.tagIJ) + ')'
      
        # there is also a new variable for the formula's output
        resvar = 'Var(' + str(nargs+1) + ',' + str(myconv.dimout) + ',' + str(myconv.tagIJ) + ')'
        
        newargs = args[:varinvpos] + (G,) + args[varinvpos+1:]
        KinvG = KernelSolveAutograd.apply(formula, aliases, varinvpos, alpha, backend, dtype, device_id, eps, ranges, *newargs)

        grads = []  # list of gradients wrt. args;

        for (var_ind, sig) in enumerate(aliases):  # Run through the arguments
            # If the current gradient is to be discarded immediatly...
            if not ctx.needs_input_grad[var_ind + 9]:  # because of (formula, aliases, varinvpos, alpha, backend, dtype, device_id, eps, ranges)
                grads.append(None)  # Don't waste time computing it.

            else:  # Otherwise, the current gradient is really needed by the user:

                if var_ind == varinvpos:
                    grads.append(KinvG)
                else:
                    # adding new aliases is way too dangerous if we want to compute
                    # second derivatives, etc. So we make explicit references to Var<ind,dim,cat> instead.
                    # New here (Joan) : we still add the new variables to the list of "aliases" (without giving new aliases for them)
                    # these will not be used in the C++ code, 
                    # but are useful to keep track of the actual variables used in the formula
                    _, cat, dim, pos = get_type(sig, position_in_list=var_ind)
                    var = 'Var(' + str(pos) + ',' + str(dim) + ',' + str(cat) + ')'  # V
                    formula_g = 'Grad_WithSavedForward(' + formula + ', ' + var + ', ' + eta + ', ' + resvar + ')'  # Grad<F,V,G,R>
                    aliases_g = aliases + [eta, resvar]
                    args_g = args[:varinvpos] + (result,) + args[varinvpos+1:] + (-KinvG,) + (result,)  # Don't forget the gradient to backprop !

                    # N.B.: if I understand PyTorch's doc, we should redefine this function every time we use it?
                    genconv = GenredAutograd().apply

                    if cat == 2:  # we're referring to a parameter, so we'll have to sum both wrt 'i' and 'j'
                        # WARNING !! : here we rely on the implementation of DiffT in files in folder keops/core/reductions
                        # if tagI==cat of V is 2, then reduction is done wrt j, so we need to further sum output wrt i
                        grad = genconv(formula_g, aliases_g, backend, dtype, device_id, ranges, *args_g)
                        # Then, sum 'grad' wrt 'i' :
                        # I think that '.sum''s backward introduces non-contiguous arrays,
                        # and is thus non-compatible with GenredAutograd: grad = grad.sum(0)
                        # We replace it with a 'handmade hack' :
                        grad = torch.ones(1, grad.shape[0]).type_as(grad.data) @ grad
                        grad = grad.view(-1)
                    else:
                        grad = genconv(formula_g, aliases_g, backend, dtype, device_id, ranges, *args_g)
                    grads.append(grad)
         
        # Grads wrt. formula, aliases, varinvpos, alpha, backend, dtype, device_id, eps, ranges, *args
        return (None, None, None, None, None, None, None, None, None, *grads)



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
        as 2d Tensors of size ``(M,dim)``. In practice,
        ``a.view(-1,1)`` should be used to turn a vector of weights
        into a *list of scalar values*.

    Note:
        :func:`KernelSolve` relies on CUDA kernels that are compiled on-the-fly 
        and stored in ``pykeops.build_folder`` as ".dll" or ".so" files for later use.

    Note:
        :func:`KernelSolve` is fully compatible with PyTorch's :mod:`autograd` engine:
        you can **backprop** through the KernelSolve :meth:`__call__` just
        as if it was a vanilla PyTorch operation.

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
            
            As described below, :meth:`__call__` will expect input Tensors whose
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

    **To apply the routine on arbitrary torch Tensors:**
        
    
    Args:
        *args (2d Tensors (variables ``Vi(..)``, ``Vj(..)``) and 1d Tensors (parameters ``Pm(..)``)): The input numerical arrays, 
            which should all have the same ``dtype``, be **contiguous** and be stored on 
            the **same device**. KeOps expects one array per alias, 
            with the following compatibility rules:

                - All ``Vi(Dim_k)`` variables are encoded as **2d-tensors** with ``Dim_k`` columns and the same number of lines :math:`M`.
                - All ``Vj(Dim_k)`` variables are encoded as **2d-tensors** with ``Dim_k`` columns and the same number of lines :math:`N`.
                - All ``Pm(Dim_k)`` variables are encoded as **1d-tensors** (vectors) of size ``Dim_k``.

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
        (M,D) or (N,D) Tensor:

        The solution of the optimization problem, stored on the same device
        as the input Tensors. The output of a :func:`KernelSolve` 
        call is always a 
        **2d-tensor** with :math:`M` or :math:`N` lines (if **axis** = 1 
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
        >>> x = torch.randn(10000, 3, requires_grad=True).cuda()  # Sampling locations
        >>> b = torch.randn(10000, 2).cuda()                      # Random observed signal
        >>> a = K_inv(x, x, b)  # Linear solve: a_i = (.1*Id + K(x,x)) \ b
        >>> print(a.shape)
        torch.Size([10000, 2]) 
        >>> # Mean squared error:   
        >>> print( ((( .1 * a + K(x,x,a) - b)**2 ).sqrt().sum() / len(x) ).item() )
        0.0002317614998901263
        >>> [g_x] = torch.autograd.grad((a ** 2).sum(), [x])  # KernelSolve supports autograd!
        >>> print(g_x.shape)
        torch.Size([10000, 3]) 
    """
    def __init__(self, formula, aliases, varinvalias, alpha=1e-10, axis=0, dtype=default_dtype, cuda_type=None):
        if cuda_type:
            # cuda_type is just old keyword for dtype, so this is just a trick to keep backward compatibility
            dtype = cuda_type 
        reduction_op='Sum'
        # get the index of 'varinv' in the argument list
        tmp = aliases.copy()
        for (i,s) in enumerate(tmp):
            tmp[i] = s[:s.find("=")].strip()
        varinvpos = tmp.index(varinvalias)
        self.formula = reduction_op + '_Reduction(' + formula + ',' + str(axis2cat(axis)) + ')'
        self.aliases = complete_aliases(formula, list(aliases)) # just in case the user provided a tuple
        self.varinvpos = varinvpos
        self.dtype = dtype
        self.alpha = alpha

    def __call__(self, *args, backend='auto', device_id=-1, eps=1e-6, ranges=None):
        return KernelSolveAutograd.apply(self.formula, self.aliases, self.varinvpos, self.alpha, backend, self.dtype, device_id, eps, ranges, *args)



