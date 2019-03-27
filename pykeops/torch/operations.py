import torch

from pykeops.torch.generic.generic_red import GenredAutograd
from pykeops.torch import default_cuda_type
from pykeops.common.utils import axis2cat, cat2axis
from pykeops.common.parse_type import get_type, get_sizes, complete_aliases
from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import load_keops

from torch.utils.cpp_extension import include_paths
include_dirs = include_paths()[0:2]





from pykeops.common.operations import Genred_common
def Genred(formula, aliases, reduction_op='Sum', axis=0, cuda_type=default_cuda_type, opt_arg=None, formula2=None):
    r"""
    Creates a new generic operation.

    This is KeOps' main function, whose usage is documented in
    the :doc:`user-guide <generic-syntax>`, 
    the :doc:`gallery of examples <../_auto_examples/index>` 
    and the :doc:`high-level tutorials <../_auto_tutorials/index>`.
    Taking as input a handful of strings and integers that specify
    a custom Map-Reduce operation, it returns a C++ wrapper
    that can be called just like any other PyTorch function.

    Warning:
        Even for variables of size 1 (e.g. :math:`a_i\in\mathbb{R}`
        for :math:`i\in[0,M)`), KeOps expects inputs to be formatted
        as 2d Tensors of size ``(M,dim)``. In practice,
        ``a.view(-1,1)`` should be used to turn a vector of weights
        into a *list of scalar values*.

    Warning:
        As of today, vector-valued formulas are only supported by 
        the ``"Sum"`` reduction. All the 
        :ref:`other reductions <part.reduction>` expect
        **formula** to be scalar-valued.

    Note:
        :func:`Genred` relies on CUDA kernels that are compiled on-the-fly, 
        and stored in ``pykeops.build_folder`` as ".dll" or ".so" files for later use.

    Note:
        :func:`Genred` is fully compatible with PyTorch's :mod:`autograd` engine:
        If **reduction_op** is **Sum** or **LogSumExp**,
        you can **backprop** through the KeOps :meth:`__call__` just
        as if it was a vanilla PyTorch operation.

    Note:
        On top of the **Sum** and **LogSumExp** reductions, KeOps
        supports 
        :ref:`variants of the ArgKMin reduction <part.reduction>` 
        that can be used
        to implement k-nearest neighbor search.
        These routines return indices encoded as **floating point numbers**, and
        produce no gradient. Fortunately though, you can simply
        turn them into ``LongTensors`` and use them to index
        your arrays, as showcased in the documentation
        of :func:`generic_argmin`, :func:`generic_argkmin` and in the
        :doc:`K-means tutorial <../_auto_tutorials/kmeans/plot_kmeans_torch>`.


    Args:
        formula (string): The scalar- or vector-valued expression
            that should be computed and reduced.
            The correct syntax is described in the :doc:`documentation <generic-syntax>`,
            using appropriate :doc:`mathematical operations <../api/math-operations>`.
        aliases (list of strings): A list of identifiers of the form ``"AL = TYPE(DIM)"`` 
            that specify the categories and dimensions of the input variables. Here:

              - ``AL`` is an alphanumerical alias, used in the **formula**.
              - ``TYPE`` is a *category*. One of:

                - ``Vi``: indexation by :math:`i` along axis 0.
                - ``Vj``: indexation by :math:`j` along axis 1.
                - ``Pm``: no indexation, the input tensor is a *vector* and not a 2d array.

              - ``DIM`` is an integer, the dimension of the current variable.
            
            As described below, :meth:`__call__` will expect as input Tensors whose
            shape are compatible with **aliases**.

    Keyword Args:
        reduction_op (string, default = ``"Sum"``): Specifies the reduction
            operation that is applied to reduce the values
            of ``formula(x_i, y_j, ...)`` along axis 0 or axis 1. 
            The supported values are:

              - ``"Sum"``: :math:`\sum(\cdots)`.
              - ``"LogSumExp"``: :math:`\log\left(\sum\exp(\cdots)\\right)`.
              - ``"Min"``: :math:`\min(\cdots)`.
              - ``"ArgMin"``: :math:`\\text{argmin}(\cdots)`.
              - ``"MinArgMin"``: :math:`(\min(...),\\text{argmin}(\cdots))`.
              - ``"Max"``: :math:`\max(\cdots)`.
              - ``"ArgMax"``: :math:`\\text{argmax}(\cdots)`.
              - ``"MaxArgMax"``: :math:`(\max(...),\\text{argmax}(\cdots))`.
              - ``"KMin"``: :math:`(\cdots)_{(1)},\ldots,(\cdots)_{(K)}`, the K first order statistics.
              - ``"ArgKMin"``: :math:`(1),\ldots,(K)`, the indices of order statistics.
              - ``"KMinArgKMin"``: :math:`\left((\cdots)_{(1)},\ldots,(\cdots)_{(K)},(1),\ldots,(K)\\right)`.

        axis (int, default = 0): Specifies the dimension of the "kernel matrix" that is reduced by our routine. 
            The supported values are:

              - **axis** = 0: reduction with respect to :math:`i`, outputs a ``Vj`` or ":math:`j`" variable.
              - **axis** = 1: reduction with respect to :math:`j`, outputs a ``Vi`` or ":math:`i`" variable.

        cuda_type (string, default = ``"float32"``): Specifies the numerical ``dtype`` of the input and output arrays. 
            The supported values are:

              - **cuda_type** = ``"float32"`` or ``"float"``.
              - **cuda_type** = ``"float64"`` or ``"double"``.

        opt_arg (int, default = None): If **reduction_op** is in ``["KMin", "ArgKMin", "KMinArgKMin"]``,
            this argument allows you to specify the number ``K`` of neighbors to consider.



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
        backend (string): Specifies the map-reduce scheme.
            The supported values are:

                - ``"auto"`` (default): let KeOps decide which backend is best suited to your data, based on the tensors' shapes. ``"GPU_1D"`` will be chosen in most cases.
                - ``"CPU"``: use a simple C++ ``for`` loop on a single CPU core.
                - ``"GPU_1D"``: use a `simple multithreading scheme <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keops/core/GpuConv1D.cu>`_ on the GPU - basically, one thread per value of the output index.
                - ``"GPU_2D"``: use a more sophisticated `2D parallelization scheme <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keops/core/GpuConv2D.cu>`_ on the GPU.
                - ``"GPU"``: let KeOps decide which one of the ``"GPU_1D"`` or the ``"GPU_2D"`` scheme will run faster on the given input.

        device_id (int, default=-1): Specifies the GPU that should be used 
            to perform   the computation; a negative value lets your system 
            choose the default GPU. This parameter is only useful if your 
            system has access to several GPUs.

        ranges (6-uple of IntTensors, None by default):
            Ranges of integers that specify a 
            :doc:`block-sparse reduction scheme <sparsity>`
            with *Mc clusters along axis 0* and *Nc clusters along axis 1*.
            If None (default), we simply loop over all indices
            :math:`i\in[0,M)` and :math:`j\in[0,N)`.
            
            **The first three ranges** will be used if **axis** = 1
            (reduction along the axis of ":math:`j` variables"),
            and to compute gradients with respect to ``Vi(..)`` variables:
            
                - ``ranges_i``, (Mc,2) IntTensor - slice indices
                  :math:`[\\text{start}^I_k,\\text{end}^I_k)` in :math:`[0,M]`
                  that specify our Mc blocks along the axis 0
                  of ":math:`i` variables". 
                - ``slices_i``, (Mc,2) IntTensor - slice indices
                  :math:`[\\text{start}^S_k,\\text{end}^S_k)`
                  that specify Mc ranges in ``redranges_j``.
                - ``redranges_j``, (Mcc,2) IntTensor - slice indices
                  :math:`[\\text{start}^J_l,\\text{end}^J_l)` in :math:`[0,N]`
                  that specify reduction ranges along the axis 1
                  of ":math:`j` variables".

            If **axis** = 1, 
            these integer arrays allow us to say
            that ``for k in range(Mc)``, the output values for 
            indices ``i in range( ranges_i[k,0], ranges_i[k,1] )``
            should be computed using a Map-Reduce scheme over
            indices ``j in Union( range( redranges_j[l, 0], redranges_j[l, 1] ))``
            for ``l in range( slices_i[k,0], slices_i[k,1] )``.

            **Likewise, the last three ranges** will be used if **axis** = 0
            (reduction along the axis of ":math:`i` variables"),
            and to compute gradients with respect to ``Vj(..)`` variables:
            
                - ``ranges_j``, (Nc,2) IntTensor - slice indices
                  :math:`[\\text{start}^J_k,\\text{end}^J_k)` in :math:`[0,N]`
                  that specify our Nc blocks along the axis 1
                  of ":math:`j` variables". 
                - ``slices_j``, (Nc,2) IntTensor - slice indices
                  :math:`[\\text{start}^S_k,\\text{end}^S_k)`
                  that specify Nc ranges in ``redranges_i``.
                - ``redranges_i``, (Ncc,2) IntTensor - slice indices
                  :math:`[\\text{start}^I_l,\\text{end}^I_l)` in :math:`[0,M]`
                  that specify reduction ranges along the axis 0
                  of ":math:`i` variables".

            If **axis** = 0, 
            these integer arrays allow us to say
            that ``for k in range(Nc)``, the output values for 
            indices ``j in range( ranges_j[k,0], ranges_j[k,1] )``
            should be computed using a Map-Reduce scheme over
            indices ``i in Union( range( redranges_i[l, 0], redranges_i[l, 1] ))``
            for ``l in range( slices_j[k,0], slices_j[k,1] )``.

    Returns:
        (M,D) or (N,D) Tensor:

        The output of the reduction, stored on the same device
        as the input Tensors. The output of a Genred call is always a 
        **2d-tensor** with :math:`M` or :math:`N` lines (if **axis** = 1 
        or **axis** = 0, respectively) and a number of columns 
        that is inferred from the **formula**.

    

    Example:
        >>> my_conv = Genred('Exp(-SqNorm2(x - y))',  # formula
        ...                  ['x = Vi(3)',            # 1st input: dim-3 vector per line
        ...                   'y = Vj(3)'],           # 2nd input: dim-3 vector per column
        ...                  reduction_op='Sum',      # we also support LogSumExp, Min, etc.
        ...                  axis=1)                  # reduce along the lines of the kernel matrix
        >>> # Apply it to 2d arrays x and y with 3 columns and a (huge) number of lines
        >>> x = torch.randn(1000000, 3, requires_grad=True).cuda()
        >>> y = torch.randn(2000000, 3).cuda()
        >>> a = my_conv(x, y)  # a_i = sum_j exp(-|x_i-y_j|^2)
        >>> print(a.shape)
        torch.Size([1000000, 1])    
        >>> [g_x] = torch.autograd.grad((a ** 2).sum(), [x])  # KeOps supports autograd!
        >>> print(g_x.shape)
        torch.Size([1000000, 3]) 

    """
    return Genred_common('torch', formula, aliases, reduction_op, axis, cuda_type, opt_arg, formula2)


from pykeops.common.operations import ConjugateGradientSolver
class KernelSolveAutograd(torch.autograd.Function):
    """
    This class is the entry point to pytorch auto grad engine.
    """

    @staticmethod
    def forward(ctx, formula, aliases, varinvpos, alpha, backend, cuda_type, device_id, eps, ranges, *args):

        myconv = load_keops(formula, aliases, cuda_type, 'torch', ['-DPYTORCH_INCLUDE_DIR=' + ';'.join(include_dirs)])
        
        # Context variables: save everything to compute the gradient:
        ctx.formula = formula
        ctx.aliases = aliases
        ctx.varinvpos = varinvpos
        ctx.alpha = alpha
        ctx.backend = backend
        ctx.cuda_type = cuda_type
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
        cuda_type = ctx.cuda_type
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
        KinvG = KernelSolveAutograd.apply(formula, aliases, varinvpos, alpha, backend, cuda_type, device_id, eps, ranges, *newargs)

        grads = []  # list of gradients wrt. args;

        for (var_ind, sig) in enumerate(aliases):  # Run through the arguments
            # If the current gradient is to be discarded immediatly...
            if not ctx.needs_input_grad[var_ind + 9]:  # because of (formula, aliases, varinvpos, alpha, backend, cuda_type, device_id, eps, ranges)
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
                        grad = genconv(formula_g, aliases_g, backend, cuda_type, device_id, ranges, *args_g)
                        # Then, sum 'grad' wrt 'i' :
                        # I think that '.sum''s backward introduces non-contiguous arrays,
                        # and is thus non-compatible with GenredAutograd: grad = grad.sum(0)
                        # We replace it with a 'handmade hack' :
                        grad = torch.ones(1, grad.shape[0]).type_as(grad.data) @ grad
                        grad = grad.view(-1)
                    else:
                        grad = genconv(formula_g, aliases_g, backend, cuda_type, device_id, ranges, *args_g)
                    grads.append(grad)
         
        # Grads wrt. formula, aliases, varinvpos, alpha, backend, cuda_type, device_id, eps, ranges, *args
        return (None, None, None, None, None, None, None, None, None, *grads)



class KernelSolve:
    r"""
    Creates a new conjugate gradient solver.

    Supporting the same :ref:`generic syntax <part.generic_formulas>` as :func:`pykeop.torch.Genred`,
    this module allows you to solve generic optimization problems of
    the form:

    .. math::
        a^{\star}~=~\\arg\min_a &\,\|\, (\\alpha \,\\text{Id}~+~K_{xx})\,a ~-~b\, \|^2_2, \\\\
        \\text{i.e.}~~~~ a^{\star}~&=~
        (\\alpha \,\\text{Id}~+~K_{xx})^{-1}  b,

    where :math:`K_{xx}` is a **symmetric**, **positive** definite **linear** operator
    and :math:`\\alpha` is a **nonnegative regularization** parameter.

    
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
            The correct syntax is described in the :doc:`documentation <generic-syntax>`,
            using appropriate :doc:`mathematical operations <../api/math-operations>`.
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

        cuda_type (string, default = ``"float32"``): Specifies the numerical ``dtype`` of the input and output arrays. 
            The supported values are:

              - **cuda_type** = ``"float32"`` or ``"float"``.
              - **cuda_type** = ``"float64"`` or ``"double"``.

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
            :doc:`block-sparse reduction scheme <sparsity>`
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
    def __init__(self, formula, aliases, varinvalias, alpha=1e-10, axis=0, cuda_type=default_cuda_type):
        reduction_op='Sum'
        # get the index of 'varinv' in the argument list
        tmp = aliases.copy()
        for (i,s) in enumerate(tmp):
            tmp[i] = s[:s.find("=")].strip()
        varinvpos = tmp.index(varinvalias)
        self.formula = reduction_op + 'Reduction(' + formula + ',' + str(axis2cat(axis)) + ')'
        self.aliases = complete_aliases(formula, list(aliases)) # just in case the user provided a tuple
        self.varinvpos = varinvpos
        self.cuda_type = cuda_type
        self.alpha = alpha

    def __call__(self, *args, backend='auto', device_id=-1, eps=1e-6, ranges=None):
        return KernelSolveAutograd.apply(self.formula, self.aliases, self.varinvpos, self.alpha, backend, self.cuda_type, device_id, eps, ranges, *args)



