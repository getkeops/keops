import numpy as np
import warnings

from pykeops.common.utils import get_tools


# Some advance operations defined at user level use in fact other reductions.
def preprocess(reduction_op, formula2):
    reduction_op = reduction_op
    
    if reduction_op == 'SumSoftMaxWeight' or reduction_op == 'SoftMax': # SoftMax is just old naming for SumSoftMaxWeight
        # SumSoftMaxWeight relies on KeOps Max_SumShiftExpWeight reduction, with a custom finalize
        reduction_op_internal = 'Max_SumShiftExpWeight'
        # we concatenate the 2nd formula (g) with a constant 1, so that we get sum_j exp(m_i-f_ij) g_ij and sum_j exp(m_i-f_ij) together
        formula2 = 'Concat(IntCst(1),' + formula2 + ')'
    elif reduction_op == 'LogSumExp':
        # LogSumExp relies also on Max_SumShiftExp or Max_SumShiftExpWeight reductions, with a custom finalize
        if formula2:
            # here we want to compute a log-sum-exp with weights: log(sum_j(exp(f_ij)g_ij))
            reduction_op_internal = 'Max_SumShiftExpWeight'
        else:
            # here we want to compute a usual log-sum-exp: log(sum_j(exp(f_ij)))
            reduction_op_internal = 'Max_SumShiftExp'
    else:
        reduction_op_internal = reduction_op
    
    return reduction_op_internal, formula2

def postprocess(out, binding, reduction_op, nout, opt_arg, dtype):
    tools = get_tools(binding)
    # Post-processing of the output:
    if reduction_op == 'SumSoftMaxWeight' or reduction_op == 'SoftMax':
        # we compute sum_j exp(f_ij) g_ij / sum_j exp(f_ij) from sum_j exp(m_i-f_ij) [1,g_ij]
        out = out[..., 2:] / out[..., 1][..., None]
    elif reduction_op == 'ArgMin' or reduction_op == 'ArgMax':
        # outputs are encoded as floats but correspond to indices, so we cast to integers
        out = tools.long(out)
    elif reduction_op == 'Min_ArgMin' or reduction_op == 'MinArgMin' or reduction_op == 'Max_ArgMax' or reduction_op == 'MaxArgMax':
        # output is one array of size N x 2D, giving min and argmin value for each dimension. 
        # We convert to one array of floats of size NxD giving mins, and one array of size NxD giving argmins (casted to integers)
        shape_out = out.shape
        tmp = tools.view(out, shape_out[:-1] + (2, -1))
        vals = tmp[..., 0, :]
        indices = tmp[..., 1, :]
        out = (vals, tools.long(indices))
    elif reduction_op == 'KMin':
        # output is of size N x KD giving K minimal values for each dim. We convert to array of size N x K x D
        shape_out = out.shape
        out = tools.view(out, shape_out[:-1] + (opt_arg, -1))
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
    elif reduction_op == 'ArgKMin':
        # output is of size N x KD giving K minimal values for each dim. We convert to array of size N x K x D
        # and cast to integers
        shape_out = out.shape
        out = tools.view(tools.long(out), shape_out[:-1] + (opt_arg, -1))
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
    elif reduction_op == 'KMin_ArgKMin' or reduction_op == 'KMinArgKMin':
        # output is of size N x 2KD giving K min and argmin for each dim. We convert to 2 arrays of size N x K x D
        # and cast to integers the second array
        shape_out = out.shape
        out = tools.view(out, shape_out[:-1] + (opt_arg, 2, -1))
        out = (out[..., 0, :], tools.long(out[..., 1, :]))
        if out[0].shape[-1] == 1:
            out = (out[0].squeeze(-1), out[1].squeeze(-1))
    elif reduction_op == 'LogSumExp':
        # finalize the log-sum-exp computation as m + log(s)
        if out.shape[-1] == 2:  # means (m,s) with m scalar and s scalar
            out = (out[..., 0] + tools.log(out[..., 1]))[..., None]
        else:  # here out.shape[-1]>2, means (m,s) with m scalar and s vectorial
            out = out[..., 0][..., None] + tools.log(out[..., 1:])
    return out


def ConjugateGradientSolver(binding, linop, b, eps=1e-6, callback=None, maxiter=None):
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    if binding not in ("torch", "numpy", "pytorch"):
        raise ValueError(
            "Language not supported, please use numpy, torch or pytorch.")
    tools = get_tools(binding)
    delta = tools.size(b) * eps ** 2
    if maxiter == None:
        maxiter = 10 * tools.size(b)
    a = 0
    r = tools.copy(b)
    nr2 = (r ** 2).sum()
    if nr2 < delta:
        return 0 * r
    p = tools.copy(r)
    k = 1
    while k <= maxiter:
        Mp = linop(p)
        alp = nr2 / (p * Mp).sum()
        a += alp * p
        r -= alp * Mp
        nr2new = (r ** 2).sum()
        if nr2new < delta:
            break
        p = r + (nr2new / nr2) * p
        nr2 = nr2new
        k += 1
        if callback is not None:
            callback(a)
    if k == maxiter:
        warnings.warn("Warning ----------- Conjugate gradient reached maximum iteration !")
    return a


def KernelLinearSolver(binding, K, x, b, alpha=0, eps=1e-6, precond=False, precondKernel=None):
    tools = get_tools(binding)
    dtype = tools.dtype(x)

    def PreconditionedConjugateGradientSolver(linop, b, invprecondop, eps=1e-6):
        # Preconditioned conjugate gradient algorithm to solve linear system of the form
        # Ma=b where linop is a linear operation corresponding
        # to a symmetric and positive definite matrix
        # invprecondop is linear operation corresponding to the inverse of the preconditioner matrix
        a = 0
        r = tools.copy(b)
        z = invprecondop(r)
        p = tools.copy(z)
        rz = (r * z).sum()
        k = 0
        while True:
            alp = rz / (p * linop(p)).sum()
            a += alp * p
            r -= alp * linop(p)
            if (r ** 2).sum() < eps ** 2:
                break
            z = invprecondop(r)
            rznew = (r * z).sum()
            p = z + (rznew / rz) * p
            rz = rznew
            k += 1
        return a

    def NystromInversePreconditioner(K, Kspec, x, alpha):
        N, D = x.shape
        m = int(np.sqrt(N))
        ind = np.random.choice(range(N), m, replace=False)
        u = x[ind, :]
        M = K(u, u) + Kspec(tools.tile(u, (m, 1)), tools.tile(u, (1, m)).reshape(-1, D), x).reshape(m, m)
        
        def invprecondop(r):
            a = tools.solve(M, K(u, x, r))
            return (r - K(x, u, a)) / alpha
        
        return invprecondop

    def KernelLinOp(a):
        return K(x, x, a) + alpha * a

    def GaussKernel(D, Dv, sigma):
        formula = 'Exp(-oos2*SqDist(x,y))*b'
        variables = ['x = Vi(' + str(D) + ')',  # First arg   : i-variable, of size D
                     'y = Vj(' + str(D) + ')',  # Second arg  : j-variable, of size D
                     'b = Vj(' + str(Dv) + ')',  # Third arg  : j-variable, of size Dv
                     'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
        my_routine = tools.Genred(formula, variables, reduction_op='Sum', axis=1, dtype=dtype)
        oos2 = tools.array([1.0 / sigma ** 2],dtype=dtype)
        KernelMatrix = GaussKernelMatrix(sigma)

        def K(x, y, b=None):
            if b is None:
                return KernelMatrix(x, y)
            else:
                return my_routine(x, y, b, oos2)
        
        return K

    def GaussKernelNystromPrecond(D, sigma):
        formula = 'Exp(-oos2*(SqDist(u,x)+SqDist(v,x)))'
        variables = ['u = Vi(' + str(D) + ')',  # First arg   : i-variable, of size D
                     'v = Vi(' + str(D) + ')',  # Second arg  : i-variable, of size D
                     'x = Vj(' + str(D) + ')',  # Third arg  : j-variable, of size D
                     'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
        my_routine = tools.Genred(formula, variables, reduction_op='Sum', axis=1, dtype=dtype)
        oos2 = tools.array([1.0 / sigma ** 2],dtype=dtype)
        KernelMatrix = GaussKernelMatrix(sigma)

        def K(u, v, x):
            return my_routine(u, v, x, oos2)
        
        return K

    def GaussKernelMatrix(sigma):
        oos2 = 1.0 / sigma ** 2
    
        def f(x, y):
            D = x.shape[1]
            sqdist = 0
            for k in range(D):
                sqdist += (x[:, k][:, None] - tools.transpose(y[:, k][:, None])) ** 2
            return tools.exp(-oos2 * sqdist)
        
        return f

    if type(K) == tuple:
        if K[0] == "gaussian":
            D = K[1]
            Dv = K[2]
            sigma = K[3]
            K = GaussKernel(D, Dv, sigma)
            if precond:
                precondKernel = GaussKernelNystromPrecond(D, sigma)
    
    if precond:
        invprecondop = NystromInversePreconditioner(K, precondKernel, x, alpha)
        a = PreconditionedConjugateGradientSolver(KernelLinOp, b, invprecondop, eps)
    else:
        a = ConjugateGradientSolver(binding, KernelLinOp, b, eps=eps)
    
    return a
