import numpy as np

from pykeops.common.cudaconv import cuda_conv_generic

def GenericSum_np(backend, aliases, formula, signature, sum_index, *args):
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
        result = np.zeros((n, signature[0][0]), dtype='float32')  # Init the output of the convolution
            
        cuda_conv_generic(formula, signature, result, *args,  # Inplace CUDA routine
                          backend=backend,
                          aliases=aliases, sum_index=sum_index,
                          cuda_type="float", grid_scheme="2D")
        result = result.reshape(n, signature[0][0])
        return result
