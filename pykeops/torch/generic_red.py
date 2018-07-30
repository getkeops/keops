import torch
from torch.autograd import Variable

from pykeops import default_cuda_type
from pykeops.common.utils import axis2cat, cat2axis
from pykeops.common.parse_type import get_type
from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import load_keops


class pytorch_genred(torch.autograd.Function):
    """
    This class is the entry point to pytorch auto grad engine.
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
        #ctx.save_for_backward(*args)

        result = genred(formula, aliases, *args, axis=axis, backend=backend, cuda_type=cuda_type)

        ctx.save_for_backward(*args)

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


class generic_sum(pytorch_genred):
    def __init__(self, formula, aliases, axis=0, backend = "auto", cuda_type=default_cuda_type) :
        self.formula = formula
        self.aliases = aliases
        self.axis = axis
        self.backend = backend
        self.cuda_type = cuda_type

    def __call__(self, *args):
        return self.apply(self.formula, self.aliases, self.axis, self.backend, self.cuda_type, *args)




class generic_logsumexp(pytorch_genred):
    def __init__(self, formula, aliases, axis=0, backend = "auto", cuda_type=default_cuda_type) :
        self.formula = "LogSumExp(" + formula + ")"
        self.aliases = aliases
        self.axis = axis
        self.backend = backend
        self.cuda_type = cuda_type

    def __call__(self, *args):
        return self.apply(self.formula, self.aliases, self.axis, self.backend, self.cuda_type, *args)

    # @staticmethod
    # def finalize_fw(result):
    #     result = result[:, 0] + result[:, 1].log()
    #     result = result.view(-1, 1)
    #     return result
    #
    # @staticmethod
    # def saved_for_bw(*args):
    #     return
    #
    # def formula_bw(self):
    #     return




def genred(formula, aliases, *args, axis=0, backend="auto", cuda_type=default_cuda_type):
    myconv = load_keops(formula, aliases, cuda_type, 'torch')

    tagIJ = axis2cat(axis)  # tagIJ=0 means sum over j, tagIJ=1 means sum over j
    tagCPUGPU, tag1D2D, tagHostDevice = get_tag_backend(backend, args)

    # Perform computation using KeOps
    # print('\n======   DEBUG   ==========')
    # print('Compiled formula :', myconv.formula)
    # print('Called formula : ', formula)
    # print('Nbr of args in : ', myconv.nargs)
    # print('Dim of Output : ', myconv.dimout)
    # print('tagHostDevice : ', tagHostDevice)
    # print('\n=======   DEBUG   =========')
    result = myconv.genred_pytorch(tagIJ, tag1D2D, tagCPUGPU, tagHostDevice, *args)
    return result


def nargs(formula, aliases, cuda_type):
    myconv = load_keops(formula, aliases, cuda_type, 'torch')
    return myconv.nargs


def dimout(formula, aliases, cuda_type):
    myconv = load_keops(formula, aliases, cuda_type, 'torch')
    return myconv.dimout
