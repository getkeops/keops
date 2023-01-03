from pykeops.numpy import Genred


class GenredR(Genred):
    r"""Extension of the Genred class for RKeOps."""
    def __init__(self,
        formula,
        aliases,
        reduction_op="Sum",
        axis=0,
        dtype=None,
        opt_arg=None,
        formula2=None,
        cuda_type=None,
        dtype_acc="auto",
        use_double_acc=False,
        sum_scheme="auto",
        enable_chunks=True,
        rec_multVar_highdim=False,):
        
        super().__init__(formula, aliases, reduction_op, axis,
                         dtype, opt_arg, formula2, cuda_type,
                         dtype_acc, use_double_acc, sum_scheme,
                         enable_chunks, rec_multVar_highdim)
        
    def __call__(self,
            list_args,
            ranges=None,
            backend="auto",
            device_id=-1,
            out=None):

        return super().__call__(*list_args, ranges=ranges, backend=backend,
                         device_id=device_id, out=out)
