import torch

from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import keops_binder
from pykeops.common.operations import ConjugateGradientSolver
from pykeops.common.parse_type import (
    get_type,
    get_sizes,
    complete_aliases,
    get_optional_flags,
)
from pykeops.common.utils import axis2cat
from pykeops.torch.generic.generic_red import GenredAutograd
from pykeops import default_device_id
from pykeops.common.utils import pyKeOps_Warning


class KernelSolveAutograd(torch.autograd.Function):
    """
    This class is the entry point to pytorch auto grad engine.
    """

    @staticmethod
    def forward(
        ctx,
        formula,
        aliases,
        varinvpos,
        alpha,
        backend,
        dtype,
        device_id_request,
        eps,
        ranges,
        optional_flags,
        rec_multVar_highdim,
        nx,
        ny,
        *args
    ):

        # N.B. when rec_multVar_highdim option is set, it means that formula is of the form "sum(F*b)", where b is a variable
        # with large dimension. In this case we set option multVar_highdim to allow for the use of the special "final chunk" computation
        # mode. However, this may not be also true for the gradients of the same formula. In fact only the gradient
        # with respect to variable b will have the same form. Hence, we save optional_flags current status into ctx,
        # before adding the multVar_highdim option.
        ctx.optional_flags = optional_flags.copy()
        if rec_multVar_highdim:
            optional_flags["multVar_highdim"] = 1
        else:
            optional_flags["multVar_highdim"] = 0

        tagCPUGPU, tag1D2D, tagHostDevice = get_tag_backend(backend, args)

        # number of batch dimensions
        # N.B. we assume here that there is at least a cat=0 or cat=1 variable in the formula...
        nbatchdims = max(len(arg.shape) for arg in args) - 2
        use_ranges = nbatchdims > 0 or ranges

        device_args = args[0].device
        if tagCPUGPU == 1 & tagHostDevice == 1:
            for i in range(1, len(args)):
                if args[i].device.index != device_args.index:
                    raise ValueError(
                        "[KeOps] Input arrays must be all located on the same device."
                    )

        if device_id_request == -1:  # -1 means auto setting
            if device_args.index:  # means args are on Gpu
                device_id_request = device_args.index
            else:
                device_id_request = default_device_id if tagCPUGPU == 1 else -1
        else:
            if device_args.index:
                if device_args.index != device_id_request:
                    raise ValueError(
                        "[KeOps] Gpu device id of arrays is different from device id requested for computation."
                    )

        myconv = keops_binder["nvrtc" if tagCPUGPU else "cpp"](
            tagCPUGPU,
            tag1D2D,
            tagHostDevice,
            use_ranges,
            device_id_request,
            formula,
            aliases,
            len(args),
            dtype,
            "torch",
            optional_flags,
        ).import_module()

        # Context variables: save everything to compute the gradient:
        ctx.formula = formula
        ctx.aliases = aliases
        ctx.varinvpos = varinvpos
        ctx.alpha = alpha
        ctx.backend = backend
        ctx.dtype = dtype
        ctx.device_id_request = device_id_request
        ctx.eps = eps
        ctx.nx = nx
        ctx.ny = ny
        ctx.myconv = myconv
        ctx.ranges = ranges
        ctx.rec_multVar_highdim = rec_multVar_highdim
        ctx.optional_flags = optional_flags

        varinv = args[varinvpos]
        ctx.varinvpos = varinvpos

        def linop(var):
            newargs = args[:varinvpos] + (var,) + args[varinvpos + 1 :]
            res = myconv.genred_pytorch(
                device_args, ranges, nx, ny, nbatchdims, None, *newargs
            )
            if alpha:
                res += alpha * var
            return res

        global copy
        result = ConjugateGradientSolver("torch", linop, varinv.data, eps)

        # relying on the 'ctx.saved_variables' attribute is necessary  if you want to be able to differentiate the output
        #  of the backward once again. It helps pytorch to keep track of 'who is who'.
        ctx.save_for_backward(*args, result)

        return result

    @staticmethod
    def backward(ctx, G):
        formula = ctx.formula
        aliases = ctx.aliases
        varinvpos = ctx.varinvpos
        backend = ctx.backend
        alpha = ctx.alpha
        dtype = ctx.dtype
        device_id_request = ctx.device_id_request
        eps = ctx.eps
        nx = ctx.nx
        ny = ctx.ny
        myconv = ctx.myconv
        ranges = ctx.ranges
        optional_flags = ctx.optional_flags
        rec_multVar_highdim = ctx.rec_multVar_highdim

        args = ctx.saved_tensors[:-1]  # Unwrap the saved variables
        nargs = len(args)
        result = ctx.saved_tensors[-1]

        # If formula takes 5 variables (numbered from 0 to 4), then the gradient
        # wrt. the output, G, should be given as a 6-th variable (numbered 5),
        # with the same dim-cat as the formula's output.
        eta = (
            "Var("
            + str(nargs)
            + ","
            + str(myconv.dimout)
            + ","
            + str(myconv.tagIJ)
            + ")"
        )

        # there is also a new variable for the formula's output
        resvar = (
            "Var("
            + str(nargs + 1)
            + ","
            + str(myconv.dimout)
            + ","
            + str(myconv.tagIJ)
            + ")"
        )

        newargs = args[:varinvpos] + (G,) + args[varinvpos + 1 :]
        KinvG = KernelSolveAutograd.apply(
            formula,
            aliases,
            varinvpos,
            alpha,
            backend,
            dtype,
            device_id_request,
            eps,
            ranges,
            optional_flags,
            rec_multVar_highdim,
            nx,
            ny,
            *newargs
        )

        grads = []  # list of gradients wrt. args;

        for (var_ind, sig) in enumerate(aliases):  # Run through the arguments
            # If the current gradient is to be discarded immediatly...
            if not ctx.needs_input_grad[
                var_ind + 13
            ]:  # because of (formula, aliases, varinvpos, alpha, backend, dtype, device_id, eps, ranges, optional_flags, rec_multVar_highdim, nx, ny)
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
                    var = "Var(" + str(pos) + "," + str(dim) + "," + str(cat) + ")"  # V
                    formula_g = (
                        "Grad_WithSavedForward("
                        + formula
                        + ", "
                        + var
                        + ", "
                        + eta
                        + ", "
                        + resvar
                        + ")"
                    )  # Grad<F,V,G,R>
                    aliases_g = aliases + [eta, resvar]
                    args_g = (
                        args[:varinvpos]
                        + (result,)
                        + args[varinvpos + 1 :]
                        + (-KinvG,)
                        + (result,)
                    )  # Don't forget the gradient to backprop !

                    # N.B.: if I understand PyTorch's doc, we should redefine this function every time we use it?
                    genconv = GenredAutograd.apply

                    if (
                        cat == 2
                    ):  # we're referring to a parameter, so we'll have to sum both wrt 'i' and 'j'
                        # WARNING !! : here we rely on the implementation of DiffT in files in folder keopscore/core/formulas/reductions
                        # if tagI==cat of V is 2, then reduction is done wrt j, so we need to further sum output wrt i
                        grad = genconv(
                            formula_g,
                            aliases_g,
                            backend,
                            dtype,
                            device_id_request,
                            ranges,
                            optional_flags,
                            None,
                            nx,
                            ny,
                            None,
                            *args_g
                        )
                        # Then, sum 'grad' wrt 'i' :
                        # I think that '.sum''s backward introduces non-contiguous arrays,
                        # and is thus non-compatible with GenredAutograd: grad = grad.sum(0)
                        # We replace it with a 'handmade hack' :
                        grad = torch.ones(1, grad.shape[0]).type_as(grad.data) @ grad
                        grad = grad.view(-1)
                    else:
                        grad = genconv(
                            formula_g,
                            aliases_g,
                            backend,
                            dtype,
                            device_id_request,
                            ranges,
                            optional_flags,
                            None,
                            nx,
                            ny,
                            None,
                            *args_g
                        )
                    grads.append(grad)

        # Grads wrt. formula, aliases, varinvpos, alpha, backend, dtype, device_id_request, eps, ranges, optional_flags, rec_multVar_highdim, nx, ny, *args
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *grads,
        )


class KernelSolve:
    r"""
    Creates a new conjugate gradient solver.

    Supporting the same :ref:`generic syntax <part.generic_formulas>` as :class:`torch.Genred <pykeops.torch.Genred>`,
    this module allows you to solve generic optimization problems of
    the form:

    .. math::
       & & a^{\star} & =\operatorname*{argmin}_a  \tfrac 1 2 \langle a,( \alpha \operatorname{Id}+K_{xx}) a\rangle - \langle a,b \rangle, \\\\
        &\text{i.e.}\quad &  a^{\star} & = (\alpha \operatorname{Id} + K_{xx})^{-1}  b,

    where :math:`K_{xx}` is a **symmetric**, **positive** definite **linear** operator
    and :math:`\alpha` is a **nonnegative regularization** parameter.

    
    Note:
        :class:`KernelSolve` is fully compatible with PyTorch's :mod:`autograd` engine:
        you can **backprop** through the KernelSolve :meth:`__call__` just
        as if it was a vanilla PyTorch operation.

    Example:
        >>> formula = "Exp(-Norm2(x - y)) * a"  # Exponential kernel
        >>> aliases =  ["x = Vi(3)",  # 1st input: target points, one dim-3 vector per line
        ...             "y = Vj(3)",  # 2nd input: source points, one dim-3 vector per column
        ...             "a = Vj(2)"]  # 3rd input: source signal, one dim-2 vector per column
        >>> K = Genred(formula, aliases, axis = 1)  # Reduce formula along the lines of the kernel matrix
        >>> K_inv = KernelSolve(formula, aliases, "a",  # The formula above is linear wrt. 'a'
        ...                     axis = 1)
        >>> # Generate some random data:
        >>> x = torch.randn(10000, 3, requires_grad=True).cuda()  # Sampling locations
        >>> b = torch.randn(10000, 2).cuda()                      # Random observed signal
        >>> a = K_inv(x, x, b, alpha = .1)  # Linear solve: a_i = (.1*Id + K(x,x)) \ b
        >>> print(a.shape)
        torch.Size([10000, 2]) 
        >>> # Mean squared error:   
        >>> print( ((( .1 * a + K(x,x,a) - b)**2 ).sqrt().sum() / len(x) ).item() )
        0.0002317614998901263
        >>> [g_x] = torch.autograd.grad((a ** 2).sum(), [x])  # KernelSolve supports autograd!
        >>> print(g_x.shape)
        torch.Size([10000, 3]) 
    """

    def __init__(
        self,
        formula,
        aliases,
        varinvalias,
        axis=0,
        dtype_acc="auto",
        use_double_acc=False,
        sum_scheme="auto",
        enable_chunks=True,
        rec_multVar_highdim=None,
        dtype=None,
        cuda_type=None,
    ):
        r"""
        Instantiate a new KernelSolve operation.

        Note:
            :class:`KernelSolve` relies on CUDA kernels that are compiled on-the-fly
            and stored in a :ref:`cache directory <part.cache>` as shared libraries (".so" files) for later use.


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

            dtype_acc (string, default ``"auto"``): type for accumulator of reduction, before casting to dtype.
                It improves the accuracy of results in case of large sized data, but is slower.
                Default value "auto" will set this option to the value of dtype. The supported values are:

                  - **dtype_acc** = ``"float16"`` : allowed only if dtype is "float16".
                  - **dtype_acc** = ``"float32"`` : allowed only if dtype is "float16" or "float32".
                  - **dtype_acc** = ``"float64"`` : allowed only if dtype is "float32" or "float64".

            use_double_acc (bool, default False): same as setting dtype_acc="float64" (only one of the two options can be set)
                If True, accumulate results of reduction in float64 variables, before casting to float32.
                This can only be set to True when data is in float32 or float64.
                It improves the accuracy of results in case of large sized data, but is slower.

            sum_scheme (string, default ``"auto"``): method used to sum up results for reductions.
                Default value "auto" will set this option to "block_red". Possible values are:
                  - **sum_scheme** =  ``"direct_sum"``: direct summation
                  - **sum_scheme** =  ``"block_sum"``: use an intermediate accumulator in each block before accumulating in the output. This improves accuracy for large sized data.
                  - **sum_scheme** =  ``"kahan_scheme"``: use Kahan summation algorithm to compensate for round-off errors. This improves accuracy for large sized data.

            enable_chunks (bool, default True): enable automatic selection of special "chunked" computation mode for accelerating reductions
                                with formulas involving large dimension variables.
        """

        if dtype:
            pyKeOps_Warning(
                "keyword argument dtype in Genred is deprecated ; argument is ignored."
            )
        if cuda_type:
            pyKeOps_Warning(
                "keyword argument cuda_type in Genred is deprecated ; argument is ignored."
            )

        self.reduction_op = "Sum"

        self.optional_flags = get_optional_flags(
            self.reduction_op,
            dtype_acc,
            use_double_acc,
            sum_scheme,
            enable_chunks,
        )

        self.formula = (
            self.reduction_op
            + "_Reduction("
            + formula
            + ","
            + str(axis2cat(axis))
            + ")"
        )
        self.aliases = complete_aliases(
            formula, list(aliases)
        )  # just in case the user provided a tuple
        if varinvalias[:4] == "Var(":
            # varinv is given directly as Var(*,*,*) so we just have to read the index
            varinvpos = int(varinvalias[4 : varinvalias.find(",")])
        else:
            # we need to recover index from alias
            tmp = self.aliases.copy()
            for (i, s) in enumerate(tmp):
                tmp[i] = s[: s.find("=")].strip()
            varinvpos = tmp.index(varinvalias)
        self.varinvpos = varinvpos
        self.rec_multVar_highdim = rec_multVar_highdim
        self.axis = axis

    def __call__(
        self, *args, backend="auto", device_id=-1, alpha=1e-10, eps=1e-6, ranges=None
    ):
        r"""
        Apply the routine on arbitrary torch Tensors.

        Warning:
            Even for variables of size 1 (e.g. :math:`a_i\in\mathbb{R}`
            for :math:`i\in[0,M)`), KeOps expects inputs to be formatted
            as 2d Tensors of size ``(M,dim)``. In practice,
            ``a.view(-1,1)`` should be used to turn a vector of weights
            into a *list of scalar values*.

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
                of the :class:`torch.Genred <pykeops.torch.Genred>` module.

            device_id (int, default=-1): Specifies the GPU that should be used
                to perform   the computation; a negative value lets your system
                choose the default GPU. This parameter is only useful if your
                system has access to several GPUs.

            ranges (6-uple of IntTensors, None by default): Ranges of integers
                that specify a :doc:`block-sparse reduction scheme <../../sparsity>`
                with *Mc clusters along axis 0* and *Nc clusters along axis 1*,
                as detailed in the documentation
                of the :class:`torch.Genred <pykeops.torch.Genred>` module.

                If **None** (default), we simply use a **dense Kernel matrix**
                as we loop over all indices :math:`i\in[0,M)` and :math:`j\in[0,N)`.

        Returns:
            (M,D) or (N,D) Tensor:

            The solution of the optimization problem, stored on the same device
            as the input Tensors. The output of a :class:`KernelSolve`
            call is always a
            **2d-tensor** with :math:`M` or :math:`N` lines (if **axis** = 1
            or **axis** = 0, respectively) and a number of columns
            that is inferred from the **formula**.

        """

        dtype = args[0].dtype.__str__().split(".")[1]
        nx, ny = get_sizes(self.aliases, *args)

        return KernelSolveAutograd.apply(
            self.formula,
            self.aliases,
            self.varinvpos,
            alpha,
            backend,
            dtype,
            device_id,
            eps,
            ranges,
            self.optional_flags,
            self.rec_multVar_highdim,
            nx,
            ny,
            *args
        )
