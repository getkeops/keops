import torch
from torch.autograd import Variable

from pykeops import default_cuda_type
from pykeops.common.utils import axis2cat, cat2axis
from pykeops.common.parse_type import get_type
from pykeops.common.generic_reduction import genred_pytorch, nargs, dimout


class generic_sum:
    def __init__(self, formula, aliases, axis=0, backend = "auto", cuda_type=default_cuda_type) :
        self.formula = formula
        self.aliases = aliases
        self.axis = axis
        self.backend = backend
        self.cuda_type = cuda_type


    def __call__(self, *args):
        return pytorch_genred.apply(self.formula, self.aliases, self.axis, self.backend, self.cuda_type, *args)


class generic_logsumexp:
    def __init__(self, formula, aliases, axis=0, backend = "auto", cuda_type=default_cuda_type) :
        self.formula = "LogSumExp(" + formula + ")"
        self.aliases = aliases
        self.axis = axis
        self.backend = backend
        self.cuda_type = cuda_type

    def __call__(self, *args):
        return pytorch_genred.apply(self.formula, self.aliases, self.axis, self.backend, self.cuda_type, *args)


class pytorch_genred(torch.autograd.Function):
    """
    This class
    """


    @staticmethod
    def forward(ctx, formula, aliases, axis, backend, cuda_type, *args):
        # Context variables: save everything to compute the gradient:
        ctx.formula = formula
        ctx.aliases = aliases
        ctx.axis = axis
        ctx.backend = backend
        ctx.cuda_type = cuda_type
        #relying on the "ctx.saved_variables" attribute is necessary  if you want to be able to differentiate the output
        #  of the backward once again. It helps pytorch to keep track of "who is who".
        ctx.save_for_backward(*args)

        result = genred_pytorch(formula, aliases, *args, axis=axis, backend=backend, cuda_type=cuda_type)
        return result

    @staticmethod
    def backward(ctx, G):
        formula = ctx.formula
        aliases = ctx.aliases
        axis = ctx.axis
        backend = ctx.backend
        cuda_type = ctx.cuda_type
        args = ctx.saved_tensors  # Unwrap the saved variables

        # If formula takes 5 variables (numbered from 0 to 4), then the gradient
        # wrt. the output, G, should be given as a 6-th variable (numbered 5),
        # with the same dim-cat as the formula's output.
        eta = "Var(" + str(nargs(formula,aliases,cuda_type)) + "," \
              + str(dimout(formula,aliases,cuda_type)) + "," \
              + str(axis2cat(axis)) + ")"

        grads = []  # list of gradients wrt. args;

        for (var_ind, sig) in enumerate(ctx.aliases):  # Run through the arguments
            # If the current gradient is to be discarded immediatly...
            if not ctx.needs_input_grad[var_ind + 5]:   # because of (formula, aliases, axis, backend, cuda_type)
                grads.append(None)  # Don't waste time computing it.

            else:  # Otherwise, the current gradient is really needed by the user:
                # adding new aliases is way too dangerous if we want to compute
                # second derivatives, etc. So we make explicit references to Var<ind,dim,cat> instead.
                _, cat, dim, pos = get_type(sig)
                var = "Var(" + str(pos) + "," + str(dim) + "," + str(cat) + ")"  # V
                formula_g = "Grad(" + formula + "," + var + "," + eta + ")"  # Grad<F,V,G>
                args_g = args + (G,)  # Don't forget the gradient to backprop !

                # N.B.: if I understand PyTorch's doc, we should redefine this function every time we use it?
                genconv = pytorch_genred().apply

                if cat == 2:  # we're referring to a parameter, so we'll have to sum both wrt 'i' and 'j'
                    # TODO : Check sum_index !!!!!!
                    grad = genconv(formula_g, aliases, 0, backend, cuda_type, *args_g)
                    # Then, sum 'grad' wrt 'j' :
                    # I think that ".sum"'s backward introduces non-contiguous arrays,
                    # and is thus non-compatible with pytorch_genred: grad = grad.sum(0)
                    # We replace it with a "handmade hack" :
                    grad = Variable(torch.ones(1, grad.shape[0]).type_as(grad.data)) @ grad
                    grad = grad.view(-1)
                else:
                    # axis is the index in the sum
                    grad = genconv(formula_g, aliases, cat2axis(cat), backend, cuda_type, *args_g)
                grads.append(grad)


        # Grads wrt. formula, aliases, axis, backend, *args
        return (None, None, None, None, None, *grads)
