import torch
from torch.autograd import Variable

from pykeops.common.cudaconv import cuda_conv_generic
from pykeops.common.parse_types import parse_types


class generic_sum :
    def __init__(self, formula, *types) :
        self.formula = formula
        self.aliases, self.signature, self.sum_index = parse_types( types )
        
    def __call__(self, *args, backend = "auto") :
        return GenericSum.apply(backend, self.aliases, self.formula, self.signature, self.sum_index, *args)

class GenericSum(torch.autograd.Function):
    """
    """

    @staticmethod
    def forward(ctx, backend, aliases, formula, signature, sum_index, *args):
        """
        Computes a Generic Summation specified by a formula (string) such as :
        ```
        formula = "Scal< Square<Scalprod<U,V>>, Scal< Exp< Scal<C, Minus<SqNorm2<Subtract<X,Y>>> > >,  B> >"
        ```
        i.e.       <U,V>^2 * exp(-C*|X-Y|^2 ) * B

        aliases is a list of strings, which specifies "who is who"; for example :
        ```
        aliases = [ "DIMPOINT = 3", "DIMVECT = 4", "DIMOUT = 5",
                    "C = Param<0,1>"          ,   # 1st parameter
                    "X = Var<1,DIMPOINT,0>" ,   # 1st variable, dim 3, indexed by i
                    "Y = Var<2,DIMPOINT,1>" ,   # 2nd variable, dim 3, indexed by j
                    "U = Var<3,DIMVECT ,0>" ,   # 3rd variable, dim 4, indexed by i
                    "V = Var<4,DIMVECT ,1>" ,   # 4th variable, dim 4, indexed by j
                    "B = Var<5,DIMOUT  ,1>" ]   # 5th variable, dim 5, indexed by j
        ```

        signature is a list of (DIM, CAT) integer pairs allowing the user to specify
        the respective dimensions of the output (head) and variables (tail of the list).
        Remember that CAT=0 for "x_i" indexing  variables,
                      CAT=1 for "y_j" summation variables,
                      CAT=2 for parameters.
        For instance,
        ```
        signature = [ (5,0), (1,2), (3,0), (3,1), (4,0), (4,1), (5,1) ]
        # stands for:  R_i ,   C  ,  X_i  , Y_j  , U_i  , V_j  , B_j   .
        ```

        Theoretically, signature could be inferred from formula+aliases...
        But asking the user to provide it explicitely is a good way to let him double-check
        his formula, and makes debugging easier.


        A POINT ABOUT EFFICIENCY :
            The naive behavior of GenericKernelProduct.backward would be to compute the derivatives
            with respect to all of its variables, even the ones that are not needed...
            This would be woefully inefficient!
            Thankfully, PyTorch provides a nice "ctx.needs_input_grad[index]" which allows
            the "backward" method to automatically "skip" the computation of gradients
            that are not needed to answer the current user's request.
            So no need to worry about this :-)



        With the values defined above,
        ```
        genconv = GenericSum().apply
        R = genconv( aliases, formula, signature, 0, C, X, Y, U, V, B )
        ```
        is a legal call, where :
        - C is a scalar              (torch Variable)
        - X is a nx-by-3 float array (torch Variable)
        - Y is a ny-by-3 float array (torch Variable)
        - U is a nx-by-4 float array (torch Variable)
        - V is a ny-by-4 float array (torch Variable)
        - B is a ny-by-5 float array (torch Variable)
        which outputs:
        - R, an  nx-by-5 float array (torch Variable)

        (nx and ny are automatically inferred from the data;
        an error is thrown if the lengths of the input arrays are not compatible with each other)

        Eventually, in this example, we've computed a "Gaussian-CauchyBinet varifold kernel"
        with a signal B of dimension 4 :

        R_i = \sum_j <U_i,V_j>^2 * exp(-C*|X_i-Y_j|^2 ) * B_j

        Its derivatives wrt. X, Y, U, V, B are automatically computed (symbolically, without backprop),
        and are accessible using standard PyTorch syntax.

        N.B.: The data type (float v. double) is inferred automatically from the PyTorch type of args.
              The CPU/GPU mode is chosen automatically.
        """
        # Save everything to compute the gradient -----------------------------------------------
        # N.B.: relying on the "ctx.saved_variables" attribute is necessary
        #       if you want to be able to differentiate the output of the backward
        #       once again. It helps pytorch to keep track of "who is who".
        ctx.save_for_backward(*args)  # Call at most once in the "forward".
        ctx.backend = backend
        ctx.aliases = aliases
        ctx.formula = formula
        ctx.signature = signature
        ctx.sum_index = sum_index

        # Get the size nx by looping on the signature until we've found an "x_i" ----------------
        n = -1
        for (index, sig) in enumerate(signature[1:]):  # Omit the output
            if sig[1] == sum_index:
                n = len(args[index])  # Lengths compatibility is done by cuda_conv_generic
                break
        if n == -1 and sum_index == 0: raise ValueError(
            "The signature should contain at least one indexing argument x_i.")
        if n == -1 and sum_index == 1: raise ValueError(
            "The signature should contain at least one indexing argument y_j.")

        # Actual computation --------------------------------------------------------------------
        if args[0].is_cuda:
            result = torch.cuda.FloatTensor(n,signature[0][0]).fill_(0)
        else:
            result = torch.zeros(n, signature[0][0])  # Init the output of the convolution
            
        cuda_conv_generic(formula, signature, result, *args,  # Inplace CUDA routine
                          backend=backend,
                          aliases=aliases, sum_index=sum_index
                          )
        result = result.view(n, signature[0][0])
        return result

    @staticmethod
    def backward(ctx, G):
        """
        Backward scheme.
        G has the same shape (and thus, dim-cat signature) as the formula's output.

        Denoting s = i if sum_index == 0,                t = j if sum_index == 0
                   = j if sum_index == 1,                  = i if sum_index == 1
        We have designed the forward pass so that

        R_s = \sum_t F( P^0, ..., X^0_i, X^1_i, ..., Y^0_j, Y^1_j, ... ) .         (*)

        G, the gradient wrt. the output R, has the same shape as the latter and is thus
        indexed by "s".
        If V is a variable (be it a parameter P, an "i" variable X^n or a "j" variable Y^n), we have:

        [\partial_V R].G
          = \sum_s [\partial_V R_s].G_s                                   (by definition of the L^2 scalar product)
          = \sum_s [\partial_V \sum_t F( P^0, X^0_i, Y^0_j, ...) ].G_s    (formula (*)  )
          = \sum_s \sum_t [\partial_V F( P^0, X^0_i, Y^0_j, ...) ].G_s    (linearity of the gradient operator)

          = \sum_i \sum_j [\partial_V F( P^0, X^0_i, Y^0_j, ...) ].G_s    (Fubini theorem : the summation order doesn't matter)
          = \sum_j \sum_i [\partial_V F( P^0, X^0_i, Y^0_j, ...) ].G_s    (Fubini theorem : the summation order doesn't matter)

        Then, there are three cases depending on the CAT(EGORY) of V:

        - if CAT == 0, i.e. V is an "X^n" : -----------------------------------------------------

            \sum_j [\partial_V F( P^0, X^0_i, Y^0_j, ...) ].G_s
              =   \sum_j [\partial_{X^n} F( P^0, X^0_i, Y^0_j, ...) ].G_s

                | 0 ..................................................... 0 |
                | 0 ..................................................... 0 |
              = | \sum_j [\partial_{X^n_i} F( P^0, X^0_i, Y^0_j, ...) ].G_s |  <- (i-th line)
                | 0 ..................................................... 0 |
                | 0 ..................................................... 0 |

            Hence,
            [\partial_V R].G  = \sum_i ( \sum_j ... )

              | \sum_j [\partial_{X^n_1} F( P^0, X^0_1, Y^0_j, ...) ].G_s |
              | \sum_j [\partial_{X^n_2} F( P^0, X^0_2, Y^0_j, ...) ].G_s |
            = |                              .                            |
              |                              .                            |
              | \sum_j [\partial_{X^n_I} F( P^0, X^0_I, Y^0_j, ...) ].G_s |

            = GenericSum(  Grad( F, V, G_s ), sum_index = 0 )

        - if CAT == 1, i.e. V is an "Y^m" : -----------------------------------------------------

            \sum_i [\partial_V F( P^0, X^0_i, Y^0_j, ...) ].G_s
              =   \sum_i [\partial_{Y^m} F( P^0, X^0_i, Y^0_j, ...) ].G_s

                | 0 ..................................................... 0 |
                | 0 ..................................................... 0 |
              = | \sum_i [\partial_{Y^m_j} F( P^0, X^0_i, Y^0_j, ...) ].G_s |  <- (j-th line)
                | 0 ..................................................... 0 |
                | 0 ..................................................... 0 |
                | 0 ..................................................... 0 |

            Hence,
            [\partial_V R].G  = \sum_j ( \sum_i ... )

              | \sum_i [\partial_{Y^m_1} F( P^0, X^0_1, Y^0_j, ...) ].G_s |
              | \sum_i [\partial_{Y^m_2} F( P^0, X^0_2, Y^0_j, ...) ].G_s |
            = |                              .                            |
              |                              .                            |
              |                              .                            |
              | \sum_i [\partial_{Y^m_J} F( P^0, X^0_I, Y^0_j, ...) ].G_s |

            = GenericSum(  Grad( F, V, G_s ), sum_index = 1 )

        - if CAT==2, i.e. V is a parameter P^l: ----------------------------------------------------

            [\partial_V R].G = \sum_{i,j} \partial_{P^l} F( P^0, X^0_I, Y^0_j, ...) ].G_s

            That is, the gradient wrt. P^l is the reduction of a convolution product
                GenericSum(  Grad( F, V, G ), sum_index = whatever )


        Bottom line : ---------------------------------------------------------------------------

            If V.CAT == 0 or 1, the gradient [\partial_V F].G is given by
                  GenericSum(  Grad( F, V, G ), sum_index = V.CAT )

            If V.CAT == 2, the gradient [\partial_V F].G is given by
                  GenericSum(  Grad( F, V, G ), sum_index = 1 ).sum(0)
                = GenericSum(  Grad( F, V, G ), sum_index = 0 ).sum(0)
        """
        backend = ctx.backend
        aliases = ctx.aliases
        formula = ctx.formula
        signature = ctx.signature
        sum_index = ctx.sum_index
        args = ctx.saved_tensors  # Unwrap the saved variables

        # number of arguments (including parameters)
        nvars = 0;
        for sig in signature[1:]:
            nvars += 1

        # If formula takes 5 variables (numbered from 0 to 4), then the gradient
        # wrt. the output, G, should be given as a 6-th variable (numbered 5),
        # with the same dim-cat as the formula's output.
        eta = "Var(" + str(nvars) + "," + str(signature[0][0]) + "," + str(signature[0][1]) + ")"
        grads = []  # list of gradients wrt. args;
        arg_ind = 5  # current arg index (4 since backend, ... are in front of the tensors); 
        var_ind = 0  # current Variable index;

        for sig in signature[1:]:  # Run through the actual parameters, given in *args in the forward.
            if not ctx.needs_input_grad[arg_ind]:  # If the current gradient is to be discarded immediatly...
                grads.append(None)  # Don't waste time computing it.
            else:  # Otherwise, the current gradient is really needed by the user:
                # adding new aliases is waaaaay too dangerous if we want to compute
                # second derivatives, etc. So we make explicit references to Var<ind,dim,cat> instead.
                var = "Var(" + str(var_ind) + "," + str(sig[0]) + "," + str(sig[1]) + ")"  # V
                formula_g = "Grad(" + formula + "," + var + "," + eta + ")"  # Grad<F,V,G>
                args_g = args + (G,)  # Don't forget the gradient to backprop !
                
                # N.B.: if I understand PyTorch's doc, we should redefine this function every time we use it?
                genconv = GenericSum().apply

                if sig[1] == 2:  # we're referring to a parameter, so we'll have to sum both wrt 'i' and 'j'
                    sumindex_g  = 1  # The first sum will be done wrt 'i'
                    signature_g = [ [sig[0],1] ] + signature[1:] + signature[:1]
                    grad = genconv(backend, aliases, formula_g, signature_g, sumindex_g, *args_g)
                    # Then, sum 'grad' wrt 'j' :
                    # I think that ".sum"'s backward introduces non-contiguous arrays,
                    # and is thus non-compatible with GenericSum:
                    # grad = grad.sum(0) 
                    # We replace it with a "handmade hack" :
                    grad = Variable(torch.ones(1, grad.shape[0]).type_as(grad.data)) @ grad
                    grad = grad.view(-1)
                else :
                    # sumindex is "the index that stays in the end", not "the one in the sum"
                    # (It's ambiguous, I know... But it's the convention chosen by Joan, which makes
                    #  sense if we were to expand our model to 3D tensors or whatever.)
                    sumindex_g  = sig[1]  # The sum will be "eventually indexed just like V".
                    signature_g = [sig] + signature[1:] + signature[:1]
                    grad = genconv(backend, aliases, formula_g, signature_g, sumindex_g, *args_g)
                grads.append(grad)

            # increment the Variable counts
            arg_ind += 1 ; var_ind += 1  

        # Grads wrt.  backend, aliases, formula, signature, sum_index, *args
        return (None, None, None, None, None, *grads)
