import sys, os.path

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest
import itertools
import numpy as np

import pykeops
from pykeops.numpy.utils import squared_distances, np_kernel, log_np_kernel, grad_np_kernel, differences, log_sum_exp


class PytorchUnitTestCase(unittest.TestCase):
    M = int(10)
    N = int(6)
    D = int(3)
    E = int(3)
    
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
    
    try:
        import torch
        
        use_cuda = torch.cuda.is_available()
        device = 'cuda' if use_cuda else 'cpu'
        
        type = torch.float32
        xc = torch.tensor(x, dtype=type, device=device, requires_grad=True)
        ac = torch.tensor(a, dtype=type, device=device, requires_grad=True)
        ec = torch.tensor(e, dtype=type, device=device, requires_grad=True)
        fc = torch.tensor(f, dtype=type, device=device, requires_grad=True)
        yc = torch.tensor(y, dtype=type, device=device, requires_grad=True)
        bc = torch.tensor(b, dtype=type, device=device, requires_grad=True)
        gc = torch.tensor(g, dtype=type, device=device, requires_grad=True)
        pc = torch.tensor(p, dtype=type, device=device, requires_grad=True)
        sigmac = torch.tensor(sigma, dtype=type, device=device, requires_grad=False)
        alphac = torch.tensor(alpha, dtype=type, device=device, requires_grad=False)
        
        type = torch.float64
        xcd = torch.tensor(x, dtype=type, device=device, requires_grad=True)
        acd = torch.tensor(a, dtype=type, device=device, requires_grad=True)
        ecd = torch.tensor(e, dtype=type, device=device, requires_grad=True)
        fcd = torch.tensor(f, dtype=type, device=device, requires_grad=True)
        ycd = torch.tensor(y, dtype=type, device=device, requires_grad=True)
        bcd = torch.tensor(b, dtype=type, device=device, requires_grad=True)
        gcd = torch.tensor(g, dtype=type, device=device, requires_grad=True)
        pcd = torch.tensor(p, dtype=type, device=device, requires_grad=True)
        sigmacd = torch.tensor(sigma, dtype=type, device=device, requires_grad=False)
        alphacd = torch.tensor(alpha, dtype=type, device=device, requires_grad=False)
        
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
                gamma_keops = Genred(formula,aliases,axis=1,cuda_type='float32')(self.sigmac, self.gc, self.xc, self.yc, backend=b)
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
                gamma_keops = Genred(formula,aliases,axis=1,cuda_type='float64')(self.sigmacd, self.gcd, self.xcd, self.ycd, backend=b)
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
                myop = Genred(formula,aliases,reduction_op='SoftMax',axis=1,cuda_type='float64',formula2=formula_weights)
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
        
        aliases = ['p=Pm(0,1)', 'x=Vi(2,3)', 'y=Vj(3,3)']
        formula = 'Square(p-Var(1,1,1))*Exp(-SqNorm2(y-x))'
        
        # Call cuda kernel
        myconv = Genred(formula, aliases, reduction_op='Sum', axis=1, cuda_type='float32' )
        gamma_keops= myconv(self.sigmac, self.gc, self.xc, self.yc, backend='auto')

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

        Kinv = KernelSolve(formula, aliases, 'b', alpha=self.alphac, axis=1)
        
        c = Kinv(self.xc, self.xc ,self.ac ,self.sigmac)
        c_ = torch.gesv(self.ac, self.alphac * torch.eye(self.M, device=self.device) + torch.exp(-torch.sum((self.xc[:,None,:] - self.xc[None,:,:]) ** 2, dim=2) * self.sigmac))[0]
        
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

        softmax_op = Genred(formula, aliases, reduction_op='SoftMax', axis=1, formula2=formula_weights)

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


if __name__ == '__main__':
    """
    run tests
    """
    unittest.main()
