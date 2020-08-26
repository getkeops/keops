import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest
import itertools
import numpy as np

import pykeops
from pykeops.numpy.utils import squared_distances, np_kernel, log_np_kernel, grad_np_kernel, differences, log_sum_exp


class PytorchUnitTestCase(unittest.TestCase):
    A = int(5)  # Batchdim 1
    B = int(3)  # Batchdim 2
    M = int(10)
    N = int(6)
    D = int(3)
    E = int(3)
    nbatchdims = int(2)
    
    x = np.random.rand(M, D)
    a = np.random.rand(M, E)
    e = np.random.rand(M, E)
    f = np.random.rand(M, 1)
    y = np.random.rand(N, D)
    b = np.random.rand(N, E)
    g = np.random.rand(N, 1)
    p = np.random.rand(2)
    sigma = np.array([0.4])
    alpha = np.array([0.1])
    
    X = np.random.rand(A, B, M, D)
    L = np.random.rand(A, 1, M, 1)
    Y = np.random.rand(1, B, N, D)
    S = np.random.rand(A, B, 1) + 1
    
    try:
        import torch
        
        use_cuda = torch.cuda.is_available()
        device = 'cuda' if use_cuda else 'cpu'
        
        dtype = torch.float32
        xc = torch.tensor(x, dtype=dtype, device=device, requires_grad=True)
        ac = torch.tensor(a, dtype=dtype, device=device, requires_grad=True)
        ec = torch.tensor(e, dtype=dtype, device=device, requires_grad=True)
        fc = torch.tensor(f, dtype=dtype, device=device, requires_grad=True)
        yc = torch.tensor(y, dtype=dtype, device=device, requires_grad=True)
        bc = torch.tensor(b, dtype=dtype, device=device, requires_grad=True)
        gc = torch.tensor(g, dtype=dtype, device=device, requires_grad=True)
        pc = torch.tensor(p, dtype=dtype, device=device, requires_grad=True)
        sigmac = torch.tensor(sigma, dtype=dtype, device=device, requires_grad=False)
        alphac = torch.tensor(alpha, dtype=dtype, device=device, requires_grad=False)

        Xc = torch.tensor(X, dtype=dtype, device=device, requires_grad=True)
        Lc = torch.tensor(L, dtype=dtype, device=device, requires_grad=True)
        Yc = torch.tensor(Y, dtype=dtype, device=device, requires_grad=True)
        Sc = torch.tensor(S, dtype=dtype, device=device, requires_grad=True)

        
        dtype = torch.float64
        xcd = torch.tensor(x, dtype=dtype, device=device, requires_grad=True)
        acd = torch.tensor(a, dtype=dtype, device=device, requires_grad=True)
        ecd = torch.tensor(e, dtype=dtype, device=device, requires_grad=True)
        fcd = torch.tensor(f, dtype=dtype, device=device, requires_grad=True)
        ycd = torch.tensor(y, dtype=dtype, device=device, requires_grad=True)
        bcd = torch.tensor(b, dtype=dtype, device=device, requires_grad=True)
        gcd = torch.tensor(g, dtype=dtype, device=device, requires_grad=True)
        pcd = torch.tensor(p, dtype=dtype, device=device, requires_grad=True)
        sigmacd = torch.tensor(sigma, dtype=dtype, device=device, requires_grad=False)
        alphacd = torch.tensor(alpha, dtype=dtype, device=device, requires_grad=False)
        Xcd = torch.tensor(X, dtype=dtype, device=device, requires_grad=True)
        Lcd = torch.tensor(L, dtype=dtype, device=device, requires_grad=True)
        Ycd = torch.tensor(Y, dtype=dtype, device=device, requires_grad=True)
        Scd = torch.tensor(S, dtype=dtype, device=device, requires_grad=True)
        
        print('Running Pytorch tests.')
    except:
        print('Pytorch could not be loaded. Skip tests.')
        pass
    
    ############################################################
    def test_conv_kernels_feature(self):
    ############################################################
        from pykeops.torch.kernel_product.kernels import Kernel, kernel_product
        params = {
            'gamma': 1. / self.sigmac ** 2,
            'mode': 'sum',
        }
        if pykeops.gpu_available:
            backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'pytorch']
        else:
            backend_to_test = ['auto', 'pytorch']
        
        for k, b in itertools.product(['gaussian', 'laplacian', 'cauchy', 'inverse_multiquadric'], backend_to_test):
            with self.subTest(k=k, b=b):
                params['id'] = Kernel(k + '(x,y)')
                params['backend'] = b
                # Call cuda kernel
                gamma = kernel_product(params, self.xc, self.yc, self.bc).cpu()
                
                # Numpy version
                gamma_py = np.matmul(np_kernel(self.x, self.y, self.sigma, kernel=k), self.b)
                
                # compare output
                self.assertTrue(np.allclose(gamma.cpu().data.numpy(), gamma_py))
    
    ############################################################
    def test_grad1conv_kernels_feature(self):
    ############################################################
        import torch
        from pykeops.torch import Kernel, kernel_product

        params = {
            'gamma': 1. / self.sigmac ** 2,
            'mode': 'sum',
        }
        if pykeops.gpu_available:
            backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'pytorch']
        else:
            backend_to_test = ['auto', 'pytorch']
        
        for k, b in itertools.product(['gaussian', 'laplacian', 'cauchy', 'inverse_multiquadric'], backend_to_test):
            with self.subTest(k=k, b=b):
                params['id'] = Kernel(k + '(x,y)')
                params['backend'] = b
                
                # Call cuda kernel
                aKxy_b = torch.dot(self.ac.view(-1), kernel_product(params, self.xc, self.yc, self.bc).view(-1))
                gamma_keops = torch.autograd.grad(aKxy_b, self.xc, create_graph=False)[0].cpu()
                
                # Numpy version
                A = differences(self.x, self.y) * grad_np_kernel(self.x, self.y, self.sigma, kernel=k)
                gamma_py = 2 * (np.sum(self.a * (np.matmul(A, self.b)), axis=2)).T
                
                # compare output
                self.assertTrue(np.allclose(gamma_keops.cpu().data.numpy(), gamma_py, atol=1e-6))
    
    ############################################################
    def test_generic_syntax_float(self):
    ############################################################
        from pykeops.torch import Genred
        aliases = ['p=Pm(1)', 'a=Vj(1)', 'x=Vi(3)', 'y=Vj(3)']
        formula = 'Square(p-a)*Exp(x+y)'
        if pykeops.gpu_available:
            backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'GPU']
        else:
            backend_to_test = ['auto']
        
        for b in backend_to_test:
            with self.subTest(b=b):
                # Call cuda kernel
                gamma_keops = Genred(formula,aliases,axis=1,dtype='float32')(self.sigmac, self.gc, self.xc, self.yc, backend=b)
                # Numpy version
                gamma_py = np.sum((self.sigma - self.g) ** 2
                                  * np.exp((self.y.T[:, :, np.newaxis] + self.x.T[:, np.newaxis, :])), axis=1).T
                # compare output
                self.assertTrue(np.allclose(gamma_keops.cpu().data.numpy(), gamma_py, atol=1e-6))
    
    ############################################################
    def test_generic_syntax_double(self):
    ############################################################
        from pykeops.torch import Genred
        aliases = ['p=Pm(1)', 'a=Vj(1)', 'x=Vi(3)', 'y=Vj(3)']
        formula = 'Square(p-a)*Exp(x+y)'
        if pykeops.gpu_available:
            backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'GPU']
        else:
            backend_to_test = ['auto']
        
        for b in backend_to_test:
            with self.subTest(b=b):
                # Call cuda kernel
                gamma_keops = Genred(formula,aliases,axis=1,dtype='float64')(self.sigmacd, self.gcd, self.xcd, self.ycd, backend=b)
                # Numpy version
                gamma_py = np.sum((self.sigma - self.g) ** 2
                                  * np.exp((self.y.T[:, :, np.newaxis] + self.x.T[:, np.newaxis, :])), axis=1).T
                # compare output
                self.assertTrue(np.allclose(gamma_keops.cpu().data.numpy(), gamma_py, atol=1e-6))
    
    ############################################################
    def test_generic_syntax_softmax(self):
    ############################################################
        from pykeops.torch import Genred
        aliases = ['p=Pm(1)', 'a=Vj(1)', 'x=Vi(3)', 'y=Vj(3)']
        formula = 'Square(p-a)*Exp(-SqNorm2(x-y))'
        formula_weights = 'y'
        if pykeops.gpu_available:
            backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'GPU']
        else:
            backend_to_test = ['auto']
        
        for b in backend_to_test:
            with self.subTest(b=b):
                # Call cuda kernel
                myop = Genred(formula,aliases,reduction_op='SumSoftMaxWeight',axis=1,dtype='float64',formula2=formula_weights)
                gamma_keops = myop(self.sigmacd, self.gcd, self.xcd, self.ycd, backend=b)
                # Numpy version
                def np_softmax(x,w):
                    x -= np.max(x,axis=1)[:,None] # subtract the max for robustness
                    return np.exp(x)@w/np.sum(np.exp(x),axis=1)[:,None]
                gamma_py = np_softmax((self.sigma - self.g.T)**2 * np.exp(-squared_distances(self.x, self.y)), self.y)
                
                # compare output
                self.assertTrue(np.allclose(gamma_keops.cpu().data.numpy(), gamma_py, atol=1e-6))
    
    ############################################################
    def test_generic_syntax_simple(self):
    ############################################################
        from pykeops.torch import Genred
        
        aliases = ['P = Pm(2)',  # 1st argument,  a parameter, dim 2.
                   'X = Vi(' + str(self.xc.shape[1]) + ') ',  # 2nd argument, indexed by i, dim D.
                   'Y = Vj(' + str(self.yc.shape[1]) + ') ']  # 3rd argument, indexed by j, dim D.
        
        formula = 'Pow((X|Y),2) * ((Elem(P,0) * X) + (Elem(P,1) * Y))'
        
        if pykeops.gpu_available:
            backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'GPU']
        else:
            backend_to_test = ['auto']
        
        for b in backend_to_test:
            with self.subTest(b=b):
                my_routine = Genred(formula, aliases, reduction_op='Sum', axis=1)
                gamma_keops = my_routine(self.pc, self.xc, self.yc, backend=b)
                
                # Numpy version
                scals = (self.x @ self.y.T) ** 2  # Memory-intensive computation!
                gamma_py = self.p[0] * scals.sum(1).reshape(-1, 1) * self.x + self.p[1] * (scals @ self.y)
                
                # compare output
                self.assertTrue(np.allclose(gamma_keops.cpu().data.numpy(), gamma_py, atol=1e-6))
    
    ############################################################
    def test_logSumExp_kernels_feature(self):
    ############################################################
        from pykeops.torch import Kernel, kernel_product

        params = {'gamma': 1. / self.sigmac ** 2, 'mode': 'lse'}
        if pykeops.gpu_available:
            backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'pytorch']
        else:
            backend_to_test = ['auto', 'pytorch']
        
        for k, b in itertools.product(['gaussian', 'laplacian', 'cauchy', 'inverse_multiquadric'], backend_to_test):
            with self.subTest(k=k, b=b):
                params['id'] = Kernel(k + '(x,y)')
                params['backend'] = b
                # Call cuda kernel
                gamma = kernel_product(params, self.xc, self.yc, self.gc).cpu()
                
                # Numpy version
                log_K = log_np_kernel(self.x, self.y, self.sigma, kernel=k)
                log_KP = log_K + self.g.T
                gamma_py = log_sum_exp(log_KP, axis=1)
                
                # compare output
                self.assertTrue(np.allclose(gamma.data.numpy().ravel(), gamma_py, atol=1e-6))
    
    ############################################################
    def test_logSumExp_gradient_kernels_feature(self):
    ############################################################
        import torch
        from pykeops.torch import Genred
        
        aliases = ['P = Pm(2)',  # 1st argument,  a parameter, dim 2.
                   'X = Vi(' + str(self.gc.shape[1]) + ') ',  # 2nd argument, indexed by i, dim D.
                   'Y = Vj(' + str(self.fc.shape[1]) + ') ']  # 3rd argument, indexed by j, dim D.
        
        formula = '(Elem(P,0) * X) + (Elem(P,1) * Y)'
        
        # Pytorch version
        my_routine = Genred(formula, aliases, reduction_op='LogSumExp', axis=1)
        tmp = my_routine(self.pc, self.fc, self.gc, backend='auto')
        res = torch.dot(torch.ones_like(tmp).view(-1),
                        tmp.view(-1))  # equivalent to tmp.sum() but avoiding contiguity pb
        gamma_keops = torch.autograd.grad(res, [self.fc, self.gc], create_graph=False)
        
        # Numpy version
        tmp = self.p[0] * self.f + self.p[1] * self.g.T
        res_py = (np.exp(tmp)).sum(axis=1)
        tmp2 = np.exp(tmp.T) / res_py.reshape(1, -1)
        gamma_py = [np.ones(self.M) * self.p[0], self.p[1] * tmp2.T.sum(axis=0)]
        
        # compare output
        self.assertTrue(np.allclose(gamma_keops[0].cpu().data.numpy().ravel(), gamma_py[0], atol=1e-6))
        self.assertTrue(np.allclose(gamma_keops[1].cpu().data.numpy().ravel(), gamma_py[1], atol=1e-6))
    
    ############################################################
    def test_non_contiguity(self):
    ############################################################
        from pykeops.torch import Genred
        
        aliases = ['P = Pm(2)',  # 1st argument,  a parameter, dim 2.
                   'X = Vi(' + str(self.xc.shape[1]) + ') ',  # 2nd argument, indexed by i, dim D.
                   'Y = Vj(' + str(self.yc.shape[1]) + ') ']  # 3rd argument, indexed by j, dim D.
        
        formula = 'Pow((X|Y),2) * ((Elem(P,0) * X) + (Elem(P,1) * Y))'
        
        my_routine = Genred(formula, aliases, reduction_op='Sum', axis=1)
        yc_tmp = self.yc.t().contiguous().t()  # create a non contiguous copy
        
        # check output
        self.assertFalse(yc_tmp.is_contiguous())
        with self.assertRaises(RuntimeError):
            my_routine(self.pc, self.xc, yc_tmp, backend='auto')

    ############################################################
    def test_heterogeneous_var_aliases(self):
    ############################################################
        from pykeops.torch import Genred
        from pykeops.numpy.utils import squared_distances

        aliases = ['p=Pm(0,1)', 'x=Vi(1,3)', 'y=Vj(2,3)']
        formula = 'Square(p-Var(3,1,1))*Exp(-SqNorm2(y-x))'
        
        # Call cuda kernel
        myconv = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype='float32' )
        gamma_keops = myconv(self.sigmac, self.xc, self.yc, self.gc, backend='auto')

        # Numpy version
        gamma_py = np.sum((self.sigma - self.g.T)**2 * np.exp(-squared_distances(self.x, self.y)), axis=1)
        
        # compare output
        self.assertTrue(np.allclose(gamma_keops.cpu().data.numpy().ravel(), gamma_py.ravel(), atol=1e-6))


    ############################################################
    def test_invkernel(self):
    ############################################################
        import torch
        from pykeops.torch.operations import KernelSolve
        formula = 'Exp(-oos2*SqDist(x,y))*b'
        aliases = ['x = Vi(' + str(self.D) + ')',  # First arg   : i-variable, of size D
                   'y = Vj(' + str(self.D) + ')',  # Second arg  : j-variable, of size D
                   'b = Vj(' + str(self.E) + ')',  # Third arg  : j-variable, of size Dv
                   'oos2 = Pm(1)']  # Fourth arg  : scalar parameter

        Kinv = KernelSolve(formula, aliases, 'b', axis=1)
        
        c = Kinv(self.xc, self.xc ,self.ac ,self.sigmac, alpha=self.alphac)
        c_ = torch.solve(self.ac, self.alphac * torch.eye(self.M, device=self.device) + torch.exp(-torch.sum((self.xc[:,None,:] - self.xc[None,:,:]) ** 2, dim=2) * self.sigmac))[0]
        
        self.assertTrue(np.allclose (c.cpu().data.numpy().ravel(), c_.cpu().data.numpy().ravel(), atol=1e-4))
        
        u, = torch.autograd.grad(c, self.xc, self.ec)
        u_, = torch.autograd.grad(c_, self.xc, self.ec)
        self.assertTrue(np.allclose (u.cpu().data.numpy().ravel(), u_.cpu().data.numpy().ravel(), atol=1e-4))
    
    ############################################################
    def test_softmax(self):
    ############################################################

        import torch
        from pykeops.torch import Genred

        formula = 'SqDist(x,y)'
        formula_weights = 'b'
        aliases = ['x = Vi(' + str(self.D) + ')',  # First arg   : i-variable, of size D
                   'y = Vj(' + str(self.D) + ')',  # Second arg  : j-variable, of size D
                   'b = Vj(' + str(self.E) + ')'] # third arg : j-variable, of size Dv

        softmax_op = Genred(formula, aliases, reduction_op='SumSoftMaxWeight', axis=1, formula2=formula_weights)

        c = softmax_op(self.xc, self.yc, self.bc)

        # compare with direct implementation
        cc = 0
        for k in range(self.D):
            xk = self.xc[:, k][:, None]
            yk = self.yc[:, k][:, None]
            cc += (xk - yk.t()) ** 2
        cc -= torch.max(cc, dim=1)[0][:,None] # subtract the max for robustness
        cc = torch.exp(cc) @ self.bc / torch.sum(torch.exp(cc), dim=1)[:, None]

        self.assertTrue(np.allclose(c.cpu().data.numpy().ravel(), cc.cpu().data.numpy().ravel(), atol=1e-6))

    ############################################################
    def test_pickle(self):
    ############################################################
        from pykeops.torch import Genred
        import pickle

        formula = 'SqDist(x,y)'
        aliases = ['x = Vi(' + str(self.D) + ')',  # First arg   : i-variable, of size D
                   'y = Vj(' + str(self.D) + ')',  # Second arg  : j-variable, of size D
                   ]
        
        kernel_instance = Genred(formula, aliases, reduction_op='Sum', axis=1)

        # serialize/pickle
        serialized_kernel = pickle.dumps(kernel_instance)
        # deserialize/unpickle
        deserialized_kernel = pickle.loads(serialized_kernel)

        self.assertTrue(type(kernel_instance), type(deserialized_kernel))
    
    ############################################################
    def test_LazyTensor_sum(self):
        ############################################################
        import torch
        from pykeops.torch import LazyTensor
        
        full_results = []
        for use_keops in [True, False]:
            
            results = []
            
            # N.B.: We could loop over float32 and float64, but this would take longer...
            for (x, l, y, s) in [(self.Xc, self.Lc, self.Yc, self.Sc)]:  # Float32
                
                x_i = x.unsqueeze(-2)
                l_i = l.unsqueeze(-2)
                y_j = y.unsqueeze(-3)
                s_p = s.unsqueeze(-2).unsqueeze(-2)
                
                if use_keops:
                    x_i, l_i, y_j, s_p = LazyTensor(x_i), LazyTensor(l_i), LazyTensor(y_j), LazyTensor(s_p)
                
                D_ij = (.5 * (l_i * x_i - y_j) ** 2 / s_p).sum(-1)
                K_ij = (- D_ij).exp()
                a_i = K_ij.sum(self.nbatchdims + 1)
                if use_keops: a_i = a_i.squeeze(-1)
                [g_x, g_y, g_s] = torch.autograd.grad((a_i ** 2).sum(), [x, y, s], create_graph=True)
                [g_xx] = torch.autograd.grad((g_x ** 2).sum(), [x], create_graph=True)
                
                results += [a_i, g_x, g_y, g_s, g_xx]
            
            full_results.append(results)
        
        for (res_keops, res_torch) in zip(full_results[0], full_results[1]):
            self.assertTrue(res_keops.shape == res_torch.shape)
            self.assertTrue(np.allclose(res_keops.cpu().data.numpy().ravel(),
                                        res_torch.cpu().data.numpy().ravel(), atol=1e-3),
                            "KeOps:\n" + str(res_keops) + "\nPyTorch:\n" + str(res_torch)
                            + "\nMax error: {:.2e}".format((res_keops - res_torch).abs().max()))
    
    ############################################################
    def test_LazyTensor_logsumexp(self):
        ############################################################
        import torch
        from pykeops.torch import LazyTensor
        
        full_results = []
        for use_keops in [True, False]:
            
            results = []
            
            # N.B.: We could loop over float32 and float64, but this would take longer...
            for (x, l, y, s) in [(self.Xcd, self.Lcd, self.Ycd, self.Scd)]:  # Float64
                
                x_i = x.unsqueeze(-2)
                l_i = l.unsqueeze(-2)
                y_j = y.unsqueeze(-3)
                s_p = s.unsqueeze(-2).unsqueeze(-2)
                
                if use_keops:
                    x_i, l_i, y_j, s_p = LazyTensor(x_i), LazyTensor(l_i), LazyTensor(y_j), LazyTensor(s_p)
                
                D_ij = ((l_i * x_i + y_j).relu() * s_p / 9).sum(-1)
                K_ij = - 1 / (1 + D_ij)
                a_i = K_ij.logsumexp(self.nbatchdims + 1)
                if use_keops: a_i = a_i.squeeze(-1)
                [g_x, g_y, g_s] = torch.autograd.grad((1. * a_i).sum(), [x, y, s], create_graph=True)
                [g_xs] = torch.autograd.grad((g_x.abs()).sum(), [s], create_graph=True)
                
                results += [a_i, g_x, g_y, g_s, g_xs]
            
            full_results.append(results)
        
        for (res_keops, res_torch) in zip(full_results[0], full_results[1]):
            self.assertTrue(res_keops.shape == res_torch.shape)
            self.assertTrue(np.allclose(res_keops.cpu().data.numpy().ravel(),
                                        res_torch.cpu().data.numpy().ravel(), atol=1e-5))
    
    ############################################################
    def test_LazyTensor_min(self):
    ############################################################
        from pykeops.torch import LazyTensor
        
        full_results = []
        for use_keops in [True, False]:
            
            results = []
            
            # N.B.: We could loop over float32 and float64, but this would take longer...
            for (x, l, y, s) in [(self.Xc, self.Lc, self.Yc, self.Sc)]:  # Float32
                
                x_i = x.unsqueeze(-2)
                l_i = l.unsqueeze(-2)
                y_j = y.unsqueeze(-3)
                s_p = s.unsqueeze(-2).unsqueeze(-2)
                
                if use_keops:
                    x_i, l_i, y_j, s_p = LazyTensor(x_i), LazyTensor(l_i), LazyTensor(y_j), LazyTensor(s_p)
                
                D_ij = ((1 + ((l_i * x_i + y_j).relu() * s_p) ** 2).log()).sum(-1, keepdim=True)
                K_ij = (D_ij ** 1.5 + 1).cos() * (D_ij * (3.2 + s_p)).sin()
                
                if use_keops:
                    m, am = K_ij.min_argmin(dim=self.nbatchdims)
                else:
                    m, am = K_ij.min(dim=self.nbatchdims)
                
                results += [m, am]
            
            full_results.append(results)
        
        for (res_keops, res_torch) in zip(full_results[0], full_results[1]):
            self.assertTrue(res_keops.shape == res_torch.shape)
            self.assertTrue(np.allclose(res_keops.cpu().data.numpy().ravel(),
                                        res_torch.cpu().data.numpy().ravel(), atol=1e-5))

    ############################################################
    def test_TensorDot_with_permute(self):
    ############################################################
        import torch
        from pykeops.torch import LazyTensor

        def my_tensordort_perm(a, b, dims=None, perm=None):
            return torch.tensordot(a, b, dims=dims).sum(3).permute(perm)

        def invert_permutation_numpy(permutation):
            return np.arange(len(permutation))[np.argsort(permutation)]

        x = torch.randn(self.M, 2, 3, 2, 2, 4, requires_grad=True, dtype=torch.float64)
        y = torch.randn(self.N, 2, 4, 2, 3, 2, 3, requires_grad=True, dtype=torch.float64)

        dimfa, dimfb = x.shape[1:], y.shape[1:]
        contfa, contfb = [5, 1, 3], [2, 5, 3]
        perm = [4, 3, 2, 0, 1]
        perm_torch = (0,) + tuple([(i + 1) for i in invert_permutation_numpy(perm)])
        sum_f_torch2 = my_tensordort_perm(x, y, dims=(contfa, contfb), perm=perm_torch)

        f_keops = LazyTensor(x.reshape(self.M, 1, int(np.array((dimfa)).prod()))).keops_tensordot(
            LazyTensor(y.reshape(1, self.N, int(np.array(dimfb).prod()))),
            dimfa,
            dimfb,
            tuple(np.array(contfa) - 1),
            tuple(np.array(contfb) - 1),
            tuple(perm)
        )
        sum_f_keops = f_keops.sum_reduction(dim=1)
        self.assertTrue(torch.allclose(sum_f_keops.flatten(), sum_f_torch2.flatten()))

        e = torch.randn_like(sum_f_torch2)
        # checking gradients
        grad_keops = torch.autograd.grad(sum_f_keops, x, e.reshape(self.M, -1), retain_graph=True)[0]
        grad_torch = torch.autograd.grad(sum_f_torch2, x, e, retain_graph=True)[0]
        self.assertTrue(torch.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4))

        grad_keops = torch.autograd.grad(sum_f_keops, y, e.reshape(self.M, -1))[0]
        grad_torch = torch.autograd.grad(sum_f_torch2, y, e)[0]
        self.assertTrue(torch.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4))

    ############################################################
    def test_cg_dic(self):
    ############################################################
        from pykeops.torch import KernelSolve, Genred
        formula = 'Exp(- g * SqDist(x,y)) * a'
        aliases = ['x = Vi(3)',   # First arg:  i-variable of size D
                   'y = Vj(3)',   # Second arg: j-variable of size D
                   'a = Vj(1)',  # Third arg:  j-variable of size Dv
                   'g = Pm(1)']
        K = Genred(formula, aliases, axis=1, dtype="float32")
        Kinv = KernelSolve(formula, aliases, "a", axis=1,
                           dtype="float32")
        ans = Kinv.cg(self.xc, self.xc, self.fc,
                          self.sigmac, alpha=self.sigmac)[0]
        err = ((self.sigmac * ans + K(self.xc, self.xc,
                                     ans, self.sigmac) - self.fc) ** 2).sum()
        self.assertTrue(np.allclose(err.cpu().data.numpy(), np.zeros(err.shape)))

    #############################################################
    def test_cg(self):
    ############################################################
        from pykeops.torch import KernelSolve, Genred
        formula = 'Exp(- g * SqDist(x,y)) * a'
        aliases = ['x = Vi(3)',   # First arg:  i-variable of size D
                   'y = Vj(3)',   # Second arg: j-variable of size D
                   'a = Vj(1)',  # Third arg:  j-variable of size Dv
                   'g = Pm(1)']
        K = Genred(formula, aliases, axis=1, dtype="float32")
        Kinv = KernelSolve(formula, aliases, "a", axis=1,
                           dtype="float32")
        ans = Kinv(self.xc, self.xc, self.fc,
                          self.sigmac, alpha=self.sigmac)
        err = ((self.sigmac * ans + K(self.xc, self.xc,
                                     ans, self.sigmac) - self.fc) ** 2).sum()
        self.assertTrue(np.allclose(err.cpu().data.numpy(), np.zeros(err.shape)))

if __name__ == '__main__':
    """
    run tests
    """
    unittest.main()
