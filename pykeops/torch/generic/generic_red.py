import torch

from pykeops import default_cuda_type
from pykeops.common.utils import axis2cat, cat2axis
from pykeops.common.parse_type import get_type, get_sizes, complete_aliases
from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import load_keops

from torch.utils.cpp_extension import include_paths
include_dirs = include_paths()[0:2]

class GenredAutograd(torch.autograd.Function):
    """
    This class is the entry point to pytorch auto grad engine.
    """

    @staticmethod
    def forward(ctx, formula, aliases, backend, cuda_type, device_id, ranges, *args):

        myconv = load_keops(formula, aliases, cuda_type, 'torch', ['-DPYTORCH_INCLUDE_DIR=' + ';'.join(include_dirs)])

        # Context variables: save everything to compute the gradient:
        ctx.formula = formula
        ctx.aliases = aliases
        ctx.backend = backend
        ctx.cuda_type = cuda_type
        ctx.device_id = device_id
        ctx.ranges = ranges
        ctx.myconv = myconv
        
        nx, ny = get_sizes(aliases, *args)

        tagCPUGPU, tag1D2D, tagHostDevice = get_tag_backend(backend, args)

        if tagCPUGPU==1 & tagHostDevice==1:
            device_id = args[0].device.index
            for i in range(1,len(args)):
                if args[i].device.index != device_id:
                    raise ValueError("[KeOps] Input arrays must be all located on the same device.")
        
        if ranges is None : ranges = () # To keep the same type
        result = myconv.genred_pytorch(nx, ny, tagCPUGPU, tag1D2D, tagHostDevice, device_id, ranges, *args)

        # relying on the 'ctx.saved_variables' attribute is necessary  if you want to be able to differentiate the output
        #  of the backward once again. It helps pytorch to keep track of 'who is who'.
        ctx.save_for_backward(*args,result)

        return result

    @staticmethod
    def backward(ctx, G):
        formula = ctx.formula
        aliases = ctx.aliases
        backend = ctx.backend
        cuda_type = ctx.cuda_type
        ranges    = ctx.ranges
        device_id = ctx.device_id
        myconv = ctx.myconv
        args = ctx.saved_tensors[:-1]  # Unwrap the saved variables
        nargs = len(args)
        result = ctx.saved_tensors[-1].detach()

        # If formula takes 5 variables (numbered from 0 to 4), then the gradient
        # wrt. the output, G, should be given as a 6-th variable (numbered 5),
        # with the same dim-cat as the formula's output.
        eta = 'Var(' + str(nargs) + ',' + str(myconv.dimout) + ',' + str(myconv.tagIJ) + ')'

        # there is also a new variable for the formula's output
        resvar = 'Var(' + str(nargs+1) + ',' + str(myconv.dimout) + ',' + str(myconv.tagIJ) + ')'
        
        grads = []  # list of gradients wrt. args;

        for (var_ind, sig) in enumerate(aliases):  # Run through the arguments
            # If the current gradient is to be discarded immediatly...
            if not ctx.needs_input_grad[var_ind + 6]:  # because of (formula, aliases, backend, cuda_type, device_id, ranges)
                grads.append(None)  # Don't waste time computing it.

            else:  # Otherwise, the current gradient is really needed by the user:
                # adding new aliases is way too dangerous if we want to compute
                # second derivatives, etc. So we make explicit references to Var<ind,dim,cat> instead.
                # New here (Joan) : we still add the new variables to the list of "aliases" (without giving new aliases for them)
                # these will not be used in the C++ code, 
                # but are useful to keep track of the actual variables used in the formula
                _, cat, dim, pos = get_type(sig, position_in_list=var_ind)
                var = 'Var(' + str(pos) + ',' + str(dim) + ',' + str(cat) + ')'  # V
                formula_g = 'Grad_WithSavedForward(' + formula + ', ' + var + ', ' + eta + ', ' + resvar + ')'  # Grad<F,V,G,R>
                aliases_g = aliases + [eta, resvar]
                args_g = args + (G,) + (result,)  # Don't forget the gradient to backprop !
                
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
        
        # Grads wrt. formula, aliases, backend, cuda_type, device_id, ranges, *args
        return (None, None, None, None, None, None, *grads)


class Genred:
    """
    Creates a new generic operation.

    This 
    """
    def __init__(self, formula, aliases, reduction_op='Sum', axis=0, cuda_type=default_cuda_type):        
        """Creates a new generic operation."""
        self.reduction_op = reduction_op
        self.formula = reduction_op + 'Reduction(' + formula + ',' + str(axis2cat(axis)) + ')'
        self.aliases = complete_aliases(formula, list(aliases)) # just in case the user provided a tuple
        self.cuda_type = cuda_type

    def __call__(self, *args, backend='auto', device_id=-1, ranges=None):
        """Applies the routine on arbitrary torch Tensors.
        
        Note:
            ``Genred`` is fully compatible with PyTorch's ``autograd`` engine:
            You can **backprop** through a KeOps ``__call__`` just
            as if it was a vanilla PyTorch operation.

        Warning:
            Even for variables of size 1 (e.g. :math:`a_i\in\mathbb{R}`
            for :math:`i\in[0,M)`), KeOps expects inputs to be formatted
            as 2d Tensors of size ``(M,dim)``. For instance,
            ``a.view(-1,1)`` should be used to turn a vector of weights
            into a *list of scalar values*.
        
        Args:
            *args (2d Tensors (variables ``Vx(..)``, ``Vy(..)``) and 1d Tensors (parameters ``Pm(..)``)): The input numerical arrays, 
                which should all have the same ``dtype``, be **contiguous** and be stored on 
                the **same device**. KeOps expects one array per alias, 
                with the following compatibility rules:
  
                  - All ``Vx(Dim_k)`` variables are encoded as **2d-tensors** with the same number :math:`M` of lines and ``Dim_k`` columns.
                  - All ``Vy(Dim_k)`` variables are encoded as **2d-tensors** with the same number :math:`N` of lines and ``Dim_k`` columns.
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
                
                **The first three ranges** will be used if ``axis=1``
                (reduction along the axis of ":math:`j` variables"),
                and to compute gradients with respect to ``Vx(..)`` variables:
                
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

                If ``axis=1``, 
                these integer arrays allow us to say
                that ``for k in range(Mc)``, the output values for 
                indices ``i in range( ranges_i[k,0], ranges_i[k,1] )``
                should be computed using a Map-Reduce scheme over
                indices ``j in Union( range( redranges_j[l, 0], redranges_j[l, 1] ))``
                for ``l in range( slices_i[k,0], slices_i[k,1] )``.

                **Likewise, the last three ranges** will be used if ``axis=0``
                (reduction along the axis of ":math:`i` variables"),
                and to compute gradients with respect to ``Vy(..)`` variables:
                
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

                If ``axis=0``, 
                these integer arrays allow us to say
                that ``for k in range(Nc)``, the output values for 
                indices ``j in range( ranges_j[k,0], ranges_j[k,1] )``
                should be computed using a Map-Reduce scheme over
                indices ``i in Union( range( redranges_i[l, 0], redranges_i[l, 1] ))``
                for ``l in range( slices_j[k,0], slices_j[k,1] )``.

        Returns:
            (M,D) or (N,D) Tensor:

            The output of the KeOps reduction, stored on the same device
            as the input Tensors. The output of a KeOps call is always a 
            **2d-tensor** with :math:`M` or :math:`N` lines (if ``axis=1`` 
            or ``axis=0``, respectively) and a number of columns 
            that is inferred from the ``formula``.

        """
        result = GenredAutograd.apply(self.formula, self.aliases, backend, self.cuda_type, device_id, ranges, *args)

        if self.reduction_op == "LogSumExp" : 
            # KeOps core returns pairs of floats (M,S), such that the result
            # is equal to  M+log(S)...
            # Users shouldn't have to bother with that!
            return (result[:,0] + result[:,1].log()).view(-1,1)
        else :
            return result
