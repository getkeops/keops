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
    return Genred_common('torch', formula, aliases, reduction_op, axis, cuda_type, opt_arg, formula2)


from pykeops.common.operations import ConjugateGradientSolver
class InvKernelOpAutograd(torch.autograd.Function):
    """
    This class is the entry point to pytorch auto grad engine.
    """

    @staticmethod
    def forward(ctx, formula, aliases, varinvpos, lmbda, backend, cuda_type, device_id, eps, *args):

        myconv = load_keops(formula, aliases, cuda_type, 'torch', ['-DPYTORCH_INCLUDE_DIR=' + ';'.join(include_dirs)])
        
        # Context variables: save everything to compute the gradient:
        ctx.formula = formula
        ctx.aliases = aliases
        ctx.varinvpos = varinvpos
        ctx.lmbda = lmbda
        ctx.backend = backend
        ctx.cuda_type = cuda_type
        ctx.device_id = device_id
        ctx.eps = eps
        ctx.myconv = myconv

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
            res = myconv.genred_pytorch(nx, ny, tagCPUGPU, tag1D2D, tagHostDevice, device_id, *newargs)
            if lmbda:
                res += lmbda*var
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
        lmbda = ctx.lmbda
        cuda_type = ctx.cuda_type
        device_id = ctx.device_id
        eps = ctx.eps
        myconv = ctx.myconv
        varinvpos = ctx.varinvpos
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
        KinvG = InvKernelOpAutograd.apply(formula, aliases, varinvpos, lmbda, backend, cuda_type, device_id, eps, *newargs)

        grads = []  # list of gradients wrt. args;

        for (var_ind, sig) in enumerate(aliases):  # Run through the arguments
            # If the current gradient is to be discarded immediatly...
            if not ctx.needs_input_grad[var_ind + 8]:  # because of (formula, aliases, varinvpos, lmbda, backend, cuda_type, device_id, eps)
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
                        grad = genconv(formula_g, aliases_g, backend, cuda_type, device_id, *args_g)
                        # Then, sum 'grad' wrt 'i' :
                        # I think that '.sum''s backward introduces non-contiguous arrays,
                        # and is thus non-compatible with GenredAutograd: grad = grad.sum(0)
                        # We replace it with a 'handmade hack' :
                        grad = torch.ones(1, grad.shape[0]).type_as(grad.data) @ grad
                        grad = grad.view(-1)
                    else:
                        grad = genconv(formula_g, aliases_g, backend, cuda_type, device_id, *args_g)
                    grads.append(grad)
         
        # Grads wrt. formula, aliases, varinvpos, lmbda, backend, cuda_type, device_id, eps, *args
        return (None, None, None, None, None, None, None, None, *grads)



class InvKernelOp:
    """
    Note: Class should not inherit from GenredAutograd due to pickling errors
    """
    def __init__(self, formula, aliases, varinvalias, lmbda=0, axis=0, cuda_type=default_cuda_type):
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
        self.lmbda = lmbda

    def __call__(self, *args, backend='auto', device_id=-1, eps=1e-6):
        return InvKernelOpAutograd.apply(self.formula, self.aliases, self.varinvpos, self.lmbda, backend, self.cuda_type, device_id, eps, *args)



