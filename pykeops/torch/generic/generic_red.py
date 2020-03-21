import torch

from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import LoadKeOps
from pykeops.common.operations import preprocess, postprocess
from pykeops.torch.half2_convert import preprocess_half2, postprocess_half2
from pykeops.common.parse_type import get_type, get_sizes, complete_aliases
from pykeops.common.parse_type import parse_aliases, get_accuracy_flags
from pykeops.common.utils import axis2cat
from pykeops.torch import default_dtype, include_dirs, torch_cxx11_abi_flag

class GenredAutograd(torch.autograd.Function):
    """
    This class is the entry point to pytorch auto grad engine.
    """

    @staticmethod
    def forward(ctx, formula, aliases, backend, dtype, device_id, ranges, accuracy_flags, *args):
    
        optional_flags = ['-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch_cxx11_abi_flag)), '-DPYTORCH_INCLUDE_DIR=' + ';'.join(include_dirs)] + accuracy_flags

        myconv = LoadKeOps(formula, aliases, dtype, 'torch', optional_flags).import_module()

        # Context variables: save everything to compute the gradient:
        ctx.formula = formula
        ctx.aliases = aliases
        ctx.backend = backend
        ctx.dtype = dtype
        ctx.device_id = device_id
        ctx.ranges = ranges
        ctx.accuracy_flags = accuracy_flags
        ctx.myconv = myconv

        tagCPUGPU, tag1D2D, tagHostDevice = get_tag_backend(backend, args)

        if tagCPUGPU==1 & tagHostDevice==1:
            device_id = args[0].device.index
            for i in range(1,len(args)):
                if args[i].device.index != device_id:
                    raise ValueError("[KeOps] Input arrays must be all located on the same device.")
        
        if ranges is None : ranges = () # To keep the same type

        result = myconv.genred_pytorch(tagCPUGPU, tag1D2D, tagHostDevice, device_id, ranges, *args)

        # relying on the 'ctx.saved_variables' attribute is necessary  if you want to be able to differentiate the output
        #  of the backward once again. It helps pytorch to keep track of 'who is who'.
        ctx.save_for_backward(*args, result)

        return result

    @staticmethod
    def backward(ctx, G):
        formula = ctx.formula
        aliases = ctx.aliases
        backend = ctx.backend
        dtype = ctx.dtype
        ranges = ctx.ranges
        accuracy_flags = ctx.accuracy_flags
        device_id = ctx.device_id
        myconv = ctx.myconv
        args = ctx.saved_tensors[:-1]  # Unwrap the saved variables
        nargs = len(args)
        result = ctx.saved_tensors[-1].detach()

        not_supported = ["Min_ArgMin_Reduction", "Min_Reduction",
                         "Max_ArgMax_Reduction", "Max_Reduction",
                         "KMin_ArgKMin_Reduction", "KMin_Reduction"]
        for red in not_supported:
            if formula.startswith(red):
                raise NotImplementedError("As of today, KeOps does not support "
                                          + "backpropagation through the " + red + " reduction. "
                                          + "Adding this feature to LazyTensors is on the cards "
                                          + "for future releases... But until then, you may want "
                                          + "to consider extracting the relevant integer indices "
                                          + "with a '.argmin()', '.argmax()' or '.argKmin()' reduction "
                                          + "before using PyTorch advanced indexing to create a fully-differentiable "
                                          + "tensor containing the relevant 'minimal' values.")

        # If formula takes 5 variables (numbered from 0 to 4), then the gradient
        # wrt. the output, G, should be given as a 6-th variable (numbered 5),
        # with the same dim-cat as the formula's output.
        eta = 'Var(' + str(nargs) + ',' + str(myconv.dimout) + ',' + str(myconv.tagIJ) + ')'

        # there is also a new variable for the formula's output
        resvar = 'Var(' + str(nargs+1) + ',' + str(myconv.dimout) + ',' + str(myconv.tagIJ) + ')'
        
        grads = []  # list of gradients wrt. args;

        for (var_ind, (sig, arg_ind)) in enumerate(zip(aliases, args)):  # Run through the arguments
            # If the current gradient is to be discarded immediatly...
            if not ctx.needs_input_grad[var_ind + 7]:  # because of (formula, aliases, backend, dtype, device_id, ranges, accuracy_flags)
                grads.append(None)  # Don't waste time computing it.

            else:  # Otherwise, the current gradient is really needed by the user:
                # adding new aliases is way too dangerous if we want to compute
                # second derivatives, etc. So we make explicit references to Var<ind,dim,cat> instead.
                # New here (Joan) : we still add the new variables to the list of "aliases" (without
                # giving new aliases for them) these will not be used in the C++ code,
                # but are useful to keep track of the actual variables used in the formula
                _, cat, dim, pos = get_type(sig, position_in_list=var_ind)
                var = 'Var(' + str(pos) + ',' + str(dim) + ',' + str(cat) + ')'  # V
                formula_g = 'Grad_WithSavedForward(' + formula + ', ' + var + ', ' + eta + ', ' + resvar + ')'  # Grad<F,V,G,R>
                aliases_g = aliases + [eta, resvar]
                args_g = args + (G,) + (result,)  # Don't forget the gradient to backprop !
                
                # N.B.: if I understand PyTorch's doc, we should redefine this function every time we use it?
                genconv = GenredAutograd().apply

                if cat == 2:  # we're referring to a parameter, so we'll have to sum both wrt 'i' and 'j'
                    # WARNING !! : here we rely on the implementation of DiffT in files in folder keops/core/formulas/reductions
                    # if tagI==cat of V is 2, then reduction is done wrt j, so we need to further sum output wrt i
                    grad = genconv(formula_g, aliases_g, backend, dtype, device_id, ranges, accuracy_flags, *args_g)
                    # Then, sum 'grad' wrt 'i' :
                    # I think that '.sum''s backward introduces non-contiguous arrays,
                    # and is thus non-compatible with GenredAutograd: grad = grad.sum(0)
                    # We replace it with a 'handmade hack' :
                    # grad = torch.ones(1, grad.shape[0]).type_as(grad.data) @ grad
                    # grad = grad.view(-1)
                    grad = (1. * grad).sum(-2)
                    dims_to_collapse = tuple(
                        i for (i, (x, y)) in enumerate(zip(arg_ind.shape[:-1], grad.shape[:-1])) if x < y)

                else:
                    grad = genconv(formula_g, aliases_g, backend, dtype, device_id, ranges, accuracy_flags, *args_g)

                    # N.B.: 'grad' is always a full [A, .., B, M, D] or [A, .., B, N, D] or [A, .., B, D] tensor,
                    #       whereas 'arg_ind' may have some broadcasted batched dimensions.
                    #       Before returning our gradient, we must collapse 'grad' with a .sum() operation,
                    #       which is the adjoint of the good old "repmat" that could have been used
                    #       to emulate the batch broadcasting.
                    dims_to_collapse = tuple(
                        i for (i, (x, y)) in enumerate(zip(arg_ind.shape[:-2], grad.shape[:-2])) if x < y)

                if dims_to_collapse != ():
                    grad = (1. * grad).sum(dims_to_collapse, keepdim=True)
                grad = grad.reshape(arg_ind.shape)  # The gradient should have the same shape as the input!
                grads.append(grad)
        
        # Grads wrt. formula, aliases, backend, dtype, device_id, ranges, *args
        return (None, None, None, None, None, None, None, *grads)


class Genred():
    r"""
        Creates a new generic operation.

        This is KeOps' main function, whose usage is documented in
        the :doc:`user-guide <../../Genred>`,
        the :doc:`gallery of examples <../../../_auto_examples/index>`
        and the :doc:`high-level tutorials <../../../_auto_tutorials/index>`.
        Taking as input a handful of strings and integers that specify
        a custom Map-Reduce operation, it returns a C++ wrapper
        that can be called just like any other PyTorch function.

        Note:
            :func:`Genred` is fully compatible with PyTorch's :mod:`autograd` engine:
            You can **backprop** through the KeOps :meth:`__call__` just
            as if it was a vanilla PyTorch operation (except for Min or Max reduction types,
            see :ref:`reductions <part.reduction>`)

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
    
    def __init__(self, formula, aliases, reduction_op='Sum', axis=0, dtype=default_dtype, opt_arg=None,
                 formula2=None, cuda_type=None, dtype_acc="auto", use_double_acc=False, sum_scheme="auto"):
        r"""
        Instantiate a new generic operation.

        Note:
            :class:`Genred` relies on C++ or CUDA kernels that are compiled on-the-fly,
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

                As described below, :meth:`__call__` will expect as input Tensors whose
                shape are compatible with **aliases**.

        Keyword Args:
            reduction_op (string, default = ``"Sum"``): Specifies the reduction
                operation that is applied to reduce the values
                of ``formula(x_i, y_j, ...)`` along axis 0 or axis 1.
                The supported values are one of  :ref:`part.reduction`.

            axis (int, default = 0): Specifies the dimension of the "kernel matrix" that is reduced by our routine.
                The supported values are:

                  - **axis** = 0: reduction with respect to :math:`i`, outputs a ``Vj`` or ":math:`j`" variable.
                  - **axis** = 1: reduction with respect to :math:`j`, outputs a ``Vi`` or ":math:`i`" variable.

            dtype (string, default = ``"float32"``): Specifies the numerical ``dtype`` of the input and output arrays.
                The supported values are:

                  - **dtype** = ``"float16"`` or ``"half"``.
                  - **dtype** = ``"float32"`` or ``"float"``.
                  - **dtype** = ``"float64"`` or ``"double"``.

            opt_arg (int, default = None): If **reduction_op** is in ``["KMin", "ArgKMin", "KMin_ArgKMin"]``,
                this argument allows you to specify the number ``K`` of neighbors to consider.

            dtype_acc (string, default ``"auto"``): type for accumulator of reduction, before casting to dtype. 
                It improves the accuracy of results in case of large sized data, but is slower.
                Default value "auto" will set this option to the value of dtype. The supported values are: 

                  - **dtype_acc** = ``"float16"`` : allowed only if dtype is "float16".
                  - **dtype_acc** = ``"float32"`` : allowed only if dtype is "float16" or "float32".
                  - **dtype_acc** = ``"float64"`` : allowed only if dtype is "float32" or "float64"..

            use_double_acc (bool, default False): same as setting dtype_acc="float64" (only one of the two options can be set)
                If True, accumulate results of reduction in float64 variables, before casting to float32. 
                This can only be set to True when data is in float32 or float64.
                It improves the accuracy of results in case of large sized data, but is slower.
           
            sum_scheme (string, default ``"auto"``): method used to sum up results for reductions.
                Default value "auto" will set this option to "block_red". Possible values are:
                  - **sum_scheme** =  ``"direct_sum"``: direct summation
                  - **sum_scheme** =  ``"block_sum"``: use an intermediate accumulator in each block before accumulating 
                    in the output. This improves accuracy for large sized data. 
                  - **sum_scheme** =  ``"kahan_scheme"``: use Kahan summation algorithm to compensate for round-off errors. This improves
                accuracy for large sized data. 

        """
        if cuda_type:
            # cuda_type is just old keyword for dtype, so this is just a trick to keep backward compatibility
            dtype = cuda_type 
        self.reduction_op = reduction_op
        reduction_op_internal, formula2 = preprocess(reduction_op, formula2)
        
        self.accuracy_flags = get_accuracy_flags(dtype_acc, use_double_acc, sum_scheme, dtype, reduction_op_internal)

        str_opt_arg = ',' + str(opt_arg) if opt_arg else ''
        str_formula2 = ',' + formula2 if formula2 else ''
        
        self.formula = reduction_op_internal + '_Reduction(' + formula + str_opt_arg + ',' + str(
            axis2cat(axis)) + str_formula2 + ')'
        self.aliases = complete_aliases(self.formula, list(aliases)) # just in case the user provided a tuple
        self.dtype = dtype
        self.axis = axis
        self.opt_arg = opt_arg

    def __call__(self, *args, backend='auto', device_id=-1, ranges=None):
        r"""
        To apply the routine on arbitrary torch Tensors.

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
            backend (string): Specifies the map-reduce scheme.
                The supported values are:

                    - ``"auto"`` (default): let KeOps decide which backend is best suited to your data, based on the tensors' shapes. ``"GPU_1D"`` will be chosen in most cases.
                    - ``"CPU"``: use a simple C++ ``for`` loop on a single CPU core.
                    - ``"GPU_1D"``: use a `simple multithreading scheme <https://github.com/getkeops/keops/blob/master/keops/core/GpuConv1D.cu>`_ on the GPU - basically, one thread per value of the output index.
                    - ``"GPU_2D"``: use a more sophisticated `2D parallelization scheme <https://github.com/getkeops/keops/blob/master/keops/core/GpuConv2D.cu>`_ on the GPU.
                    - ``"GPU"``: let KeOps decide which one of the ``"GPU_1D"`` or the ``"GPU_2D"`` scheme will run faster on the given input.

            device_id (int, default=-1): Specifies the GPU that should be used
                to perform   the computation; a negative value lets your system
                choose the default GPU. This parameter is only useful if your
                system has access to several GPUs.

            ranges (6-uple of IntTensors, None by default):
                Ranges of integers that specify a
                :doc:`block-sparse reduction scheme <../../sparsity>`
                with *Mc clusters along axis 0* and *Nc clusters along axis 1*.
                If None (default), we simply loop over all indices
                :math:`i\in[0,M)` and :math:`j\in[0,N)`.

                **The first three ranges** will be used if **axis** = 1
                (reduction along the axis of ":math:`j` variables"),
                and to compute gradients with respect to ``Vi(..)`` variables:

                    - ``ranges_i``, (Mc,2) IntTensor - slice indices
                      :math:`[\operatorname{start}^I_k,\operatorname{end}^I_k)` in :math:`[0,M]`
                      that specify our Mc blocks along the axis 0
                      of ":math:`i` variables".
                    - ``slices_i``, (Mc,) IntTensor - consecutive slice indices
                      :math:`[\operatorname{end}^S_1, ..., \operatorname{end}^S_{M_c}]`
                      that specify Mc ranges :math:`[\operatorname{start}^S_k,\operatorname{end}^S_k)` in ``redranges_j``,
                      with :math:`\operatorname{start}^S_k = \operatorname{end}^S_{k-1}`.
                      **The first 0 is implicit**, meaning that :math:`\operatorname{start}^S_0 = 0`, and we typically expect that
                      ``slices_i[-1] == len(redrange_j)``.
                    - ``redranges_j``, (Mcc,2) IntTensor - slice indices
                      :math:`[\operatorname{start}^J_l,\operatorname{end}^J_l)` in :math:`[0,N]`
                      that specify reduction ranges along the axis 1
                      of ":math:`j` variables".

                If **axis** = 1,
                these integer arrays allow us to say
                that ``for k in range(Mc)``, the output values for
                indices ``i in range( ranges_i[k,0], ranges_i[k,1] )``
                should be computed using a Map-Reduce scheme over
                indices ``j in Union( range( redranges_j[l, 0], redranges_j[l, 1] ))``
                for ``l in range( slices_i[k-1], slices_i[k] )``.

                **Likewise, the last three ranges** will be used if **axis** = 0
                (reduction along the axis of ":math:`i` variables"),
                and to compute gradients with respect to ``Vj(..)`` variables:

                    - ``ranges_j``, (Nc,2) IntTensor - slice indices
                      :math:`[\operatorname{start}^J_k,\operatorname{end}^J_k)` in :math:`[0,N]`
                      that specify our Nc blocks along the axis 1
                      of ":math:`j` variables".
                    - ``slices_j``, (Nc,) IntTensor - consecutive slice indices
                      :math:`[\operatorname{end}^S_1, ..., \operatorname{end}^S_{N_c}]`
                      that specify Nc ranges :math:`[\operatorname{start}^S_k,\operatorname{end}^S_k)` in ``redranges_i``,
                      with :math:`\operatorname{start}^S_k = \operatorname{end}^S_{k-1}`.
                      **The first 0 is implicit**, meaning that :math:`\operatorname{start}^S_0 = 0`, and we typically expect that
                      ``slices_j[-1] == len(redrange_i)``.
                    - ``redranges_i``, (Ncc,2) IntTensor - slice indices
                      :math:`[\operatorname{start}^I_l,\operatorname{end}^I_l)` in :math:`[0,M]`
                      that specify reduction ranges along the axis 0
                      of ":math:`i` variables".

                If **axis** = 0,
                these integer arrays allow us to say
                that ``for k in range(Nc)``, the output values for
                indices ``j in range( ranges_j[k,0], ranges_j[k,1] )``
                should be computed using a Map-Reduce scheme over
                indices ``i in Union( range( redranges_i[l, 0], redranges_i[l, 1] ))``
                for ``l in range( slices_j[k-1], slices_j[k] )``.

        Returns:
            (M,D) or (N,D) Tensor:

            The output of the reduction, stored on the same device
            as the input Tensors. The output of a Genred call is always a
            **2d-tensor** with :math:`M` or :math:`N` lines (if **axis** = 1
            or **axis** = 0, respectively) and a number of columns
            that is inferred from the **formula**.

        """

        nx, ny = get_sizes(self.aliases, *args)
        nout, nred = (nx, ny) if self.axis==1 else (ny, nx)

        if "Arg" in self.reduction_op:
            # when using Arg type reductions,
            # if nred is greater than 16 millions and dtype=float32, the result is not reliable
            # because we encode indices as floats, so we raise an exception ;
            # same with float16 type and nred>2048
            if nred>1.6e7 and self.dtype in ("float32","float"):
                raise ValueError('size of input array is too large for Arg type reduction with single precision. Use double precision.')  
            elif nred>2048 and self.dtype in ("float16","half"):
                raise ValueError('size of input array is too large for Arg type reduction with float16 dtype..')  

        if self.dtype in ('float16','half'):
            args, ranges, tag_dummy, N = preprocess_half2(args, self.aliases, self.axis, ranges, nx, ny)
        
        out = GenredAutograd.apply(self.formula, self.aliases, backend, self.dtype, 
                                   device_id, ranges, self.accuracy_flags, *args)

        if self.dtype in ('float16','half'):
            out = postprocess_half2(out, tag_dummy, self.reduction_op, N)

        return postprocess(out, "torch", self.reduction_op, nout, self.opt_arg, self.dtype)
