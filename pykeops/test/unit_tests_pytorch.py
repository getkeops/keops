import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import unittest
import itertools
import numpy as np

from pykeops import torch_found, gpu_available
from pykeops.numpy.utils import np_kernel, log_np_kernel, grad_np_kernel, differences, log_sum_exp

class PytorchUnitTestCase(unittest.TestCase):

    M = int(10)
    N = int(6)
    D = int(3)
    E = int(3)

    a = np.random.rand(M,E)
    x = np.random.rand(M,D)
    g = np.random.rand(M,1)
    y = np.random.rand(N,D)
    f = np.random.rand(N,1)
    b = np.random.rand(N,E)
    p = np.random.rand(2)
    sigma = np.array([0.4])

    try:
        import torch

        use_cuda = torch.cuda.is_available()
        device   = 'cuda' if use_cuda else 'cpu'

        type = torch.float32
        ac = torch.tensor(a, dtype=type, device=device, requires_grad=True)
        xc = torch.tensor(x, dtype=type, device=device, requires_grad=True)
        gc = torch.tensor(g, dtype=type, device=device, requires_grad=True)
        yc = torch.tensor(y, dtype=type, device=device, requires_grad=True)
        fc = torch.tensor(f, dtype=type, device=device, requires_grad=True)
        bc = torch.tensor(b, dtype=type, device=device, requires_grad=True)
        pc = torch.tensor(p, dtype=type, device=device, requires_grad=True)
        sigmac = torch.tensor(sigma, dtype=type, device=device, requires_grad=False)

        type = torch.float64
        acd = torch.tensor(a, dtype=type, device=device, requires_grad=True)
        xcd = torch.tensor(x, dtype=type, device=device, requires_grad=True)
        gcd = torch.tensor(g, dtype=type, device=device, requires_grad=True)
        ycd = torch.tensor(y, dtype=type, device=device, requires_grad=True)
        fcd = torch.tensor(f, dtype=type, device=device, requires_grad=True)
        bcd = torch.tensor(b, dtype=type, device=device, requires_grad=True)
        pcd = torch.tensor(p, dtype=type, device=device, requires_grad=True)
        sigmacd = torch.tensor(sigma, dtype=type, device=device, requires_grad=False)

        print('Running Pytorch tests.')
    except:
        print('Pytorch could not be loaded. Skip tests.')
        pass

#     ############################################################
#     def test_conv_kernels_feature(self):
#     ############################################################
#         from pykeops.torch.kernels import Kernel, kernel_product
#         params = {
#             'gamma': 1./self.sigmac**2,
#             'mode' : 'sum',
#         }
#         if gpu_available:
#             backend_to_test = ['auto','GPU_1D','GPU_2D','pytorch']
#         else:
#             backend_to_test = ['auto','pytorch']
# 
#         for k,b in itertools.product(["gaussian", "laplacian", "cauchy", "inverse_multiquadric"],backend_to_test):
#             with self.subTest(k=k,b=b):
#                 params["id"] = Kernel(k+"(x,y)")
#                 params["backend"] = b
#                 # Call cuda kernel
#                 gamma = kernel_product( params, self.xc,self.yc,self.bc).cpu()
# 
#                 # Numpy version
#                 gamma_py = np.matmul(np_kernel(self.x, self.y,self.sigma,kernel=k), self.b)
# 
#                 # compare output
#                 self.assertTrue(np.allclose(gamma.data.numpy(), gamma_py))
# 
#     ############################################################
#     def test_grad1conv_kernels_feature(self):
#     ############################################################
#         import torch
#         from pykeops.torch.kernels import Kernel, kernel_product
#         params = {
#             'gamma': 1./self.sigmac**2,
#             'mode' : 'sum',
#         }
#         if gpu_available:
#             backend_to_test = ['auto','GPU_1D','GPU_2D','pytorch']
#         else:
#             backend_to_test = ['auto','pytorch']
# 
#         for k,b in itertools.product(["gaussian", "laplacian", "cauchy", "inverse_multiquadric"],backend_to_test):
#             with self.subTest(k=k,b=b):
#                 params["id"] = Kernel(k+"(x,y)")
#                 params["backend"] = b
# 
#                 # Call cuda kernel
#                 aKxy_b = torch.dot(self.ac.view(-1), kernel_product( params, self.xc,self.yc,self.bc ).view(-1))
#                 gamma_keops   = torch.autograd.grad(aKxy_b, self.xc, create_graph=False)[0].cpu()
# 
#                 # Numpy version
#                 A = differences(self.x, self.y) * grad_np_kernel(self.x,self.y,self.sigma,kernel=k)
#                 gamma_py = 2*(np.sum( self.a * (np.matmul(A,self.b)),axis=2) ).T
# 
#                 # compare output
#                 self.assertTrue( np.allclose(gamma_keops.cpu().data.numpy(), gamma_py , atol=1e-6))
# 
#     ############################################################
#     def test_generic_syntax_float(self):
#     ############################################################
#         from pykeops.torch.generic_red import Genred
#         aliases = ['p=Pm(1)', 'a=Vy(1)', 'x=Vx(3)', 'y=Vy(3)']
#         formula = 'SumReduction(Square(p-a)*Exp(x+y),0)'
#         if gpu_available:
#             backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'GPU']
#         else:
#             backend_to_test = ['auto']
# 
#         for b in backend_to_test:
#             with self.subTest(b=b):
#                 # Call cuda kernel
#                 gamma_keops = Genred.apply(formula, aliases, b, 'float32', self.sigmac, self.fc, self.xc, self.yc)
#                 # Numpy version
#                 gamma_py = np.sum((self.sigma - self.f)**2 * np.exp((self.y.T[:,:,np.newaxis] + self.x.T[:,np.newaxis,:])),axis=1).T
#                 # compare output
#                 self.assertTrue(np.allclose(gamma_keops.cpu().data.numpy(), gamma_py, atol=1e-6))
# 
#     ############################################################
#     def test_generic_syntax_double(self):
#     ############################################################
#         from pykeops.torch.generic_red import Genred
#         aliases = ['p=Pm(1)', 'a=Vy(1)', 'x=Vx(3)', 'y=Vy(3)']
#         formula = 'SumReduction(Square(p-a)*Exp(x+y),0)'
#         if gpu_available:
#             backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'GPU']
#         else:
#             backend_to_test = ['auto']
# 
#         for b in backend_to_test:
#             with self.subTest(b=b):
#                 # Call cuda kernel
#                 gamma_keops = Genred.apply(formula, aliases, b, 'float64', self.sigmacd, self.fcd, self.xcd, self.ycd)
#                 # Numpy version
#                 gamma_py = np.sum((self.sigma - self.f)**2 * np.exp((self.y.T[:,:,np.newaxis] + self.x.T[:,np.newaxis,:])),axis=1).T
#                 # compare output
#                 self.assertTrue(np.allclose(gamma_keops.cpu().data.numpy(), gamma_py, atol=1e-6))
# 
#     ############################################################
#     def test_generic_syntax_simple(self):
#     ############################################################
#         from pykeops.torch.generic_red import Sum
# 
#         aliases = ['P = Pm(2)',                               # 1st argument,  a parameter, dim 2.
#                    'X = Vx(' + str(self.xc.shape[1]) + ') ',  # 2nd argument, indexed by i, dim D.
#                    'Y = Vy(' + str(self.yc.shape[1]) + ') ']  # 3rd argument, indexed by j, dim D.
# 
#         formula = 'Pow((X|Y),2) * ((Elem(P,0) * X) + (Elem(P,1) * Y))'
# 
#         if gpu_available:
#             backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'GPU']
#         else:
#             backend_to_test = ['auto']
# 
#         for b in backend_to_test:
#             with self.subTest(b=b):
# 
#                 my_routine  = Sum(formula, aliases, axis=1, backend=b)
#                 gamma_keops = my_routine(self.pc, self.xc, self.yc)
# 
#                 # Numpy version
#                 scals = (self.x @ self.y.T)**2 # Memory-intensive computation!
#                 gamma_py = self.p[0] * scals.sum(1).reshape(-1,1) * self.x + self.p[1] * (scals @ self.y)
# 
#                 # compare output
#                 self.assertTrue( np.allclose(gamma_keops.cpu().data.numpy(), gamma_py , atol=1e-6))
# 
#     ############################################################
#     def test_logSumExp_kernels_feature(self):
#     ############################################################
#         from pykeops.torch.kernels import Kernel, kernel_product
#         params = {'gamma': 1./self.sigmac**2, 'mode': 'lse'}
#         if gpu_available:
#             backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'pytorch']
#         else:
#             backend_to_test = ['auto', 'pytorch']
# 
#         for k,b in itertools.product(['gaussian', 'laplacian', 'cauchy', 'inverse_multiquadric'], backend_to_test):
#             with self.subTest(k=k,b=b):
#                 params['id'] = Kernel(k + '(x,y)')
#                 params['backend'] = b
#                 # Call cuda kernel
#                 gamma = kernel_product(params, self.xc, self.yc, self.fc).cpu()
# 
#                 # Numpy version
#                 log_K  = log_np_kernel(self.x, self.y, self.sigma, kernel=k)
#                 log_KP = log_K + self.f.T
#                 gamma_py = log_sum_exp(log_KP, axis=1)
# 
#                 # compare output
#                 self.assertTrue(np.allclose(gamma.data.numpy().ravel(), gamma_py, atol=1e-6))

    ############################################################
    def test_logSumExp_gradient_kernels_feature(self):
    ############################################################
        import torch
        from pykeops.torch.generic_red import LogSumExp

        # aliases = ['P = Pm(2)',                               # 1st argument,  a parameter, dim 2.
                   # 'X = Vx(' + str(self.fc.shape[1]) + ') ',  # 2nd argument, indexed by i, dim D.
                   # 'Y = Vy(' + str(self.gc.shape[1]) + ') ']  # 3rd argument, indexed by j, dim D.

        # formula = 'Pow((X|Y),2) * ((Elem(P,0) * X) + (Elem(P,1) * Y))'

        # if gpu_available:
            # backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'GPU']
        # else:
            # backend_to_test = ['auto']

        # for b in backend_to_test:
            # with self.subTest(b=b):

                # my_routine  = LogSumExp(formula, aliases, axis=1, backend=b)
                # gamma_keops = my_routine(self.pc, self.gc, self.fc, axis=0)

                # # Numpy version
                # scals = (self.g @ self.f.T)**2 # Memory-intensive computation!
                # gamma_py = log_sum_exp(scals * (self.p[0] * self.g) + scals * (self.p[1] *  self.f.T), axis=0)

                # # compare output
                # self.assertTrue(np.allclose(gamma_keops.cpu().data.numpy().ravel(), gamma_py , atol=1e-6))
        aliases = ['P = Pm(2)',                               # 1st argument,  a parameter, dim 2.
                   'X = Vx(' + str(self.fc.shape[1]) + ') ',  # 2nd argument, indexed by i, dim D.
                   'Y = Vy(' + str(self.gc.shape[1]) + ') ']  # 3rd argument, indexed by j, dim D.

        formula = '(Elem(P,0) * X) + (Elem(P,1) * Y)'

        if gpu_available:
            # backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'GPU']
            backend_to_test = ['auto']
        else:
            backend_to_test = ['auto']

        for b in backend_to_test:
            with self.subTest(b=b):


                my_routine  = LogSumExp(formula, aliases, axis=1, backend=b)
                # gamma_keops = my_routine(self.pc, self.gc, self.fc)
                res = my_routine(self.pc, self.gc, self.fc).sum() 
                gamma_keops = torch.autograd.grad(res, self.gc, create_graph=False)

                # Numpy version
                # gamma_py = log_sum_exp( (self.p[0] * self.g) + (self.p[1] *  self.f.T), axis=1)

                tmp = self.p[0] * self.g + self.p[1] *  self.f.T
                res_py =  log_sum_exp(tmp, axis=1)
                
                dres_py = self.p[0] * np.exp(self.p[0] * self.g + self.p[1] *  self.f)
                tmp2 = np.exp(tmp* dres_py - res_py.reshape(-1,1))
                gamma_py = np.sum( tmp2 , axis=1 )

                print(gamma_keops)
                print(gamma_py)
                print(tmp2)
                print(res_py)
                print(dres_py)

if __name__ == '__main__':
    """
    run tests
    """
    unittest.main()

