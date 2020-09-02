import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import unittest
import itertools
import numpy as np

import pykeops
from pykeops.numpy.utils import np_kernel, grad_np_kernel, differences, squared_distances, log_sum_exp, np_kernel_sphere

class NumpyUnitTestCase(unittest.TestCase):
    A = int(4)  # Batchdim 1
    B = int(6)  # Batchdim 2
    M = int(10)
    N = int(6)
    D = int(3)
    E = int(3)
    nbatchdims = int(2)

    x = np.random.rand(M, D)
    a = np.random.rand(M, E)
    f = np.random.rand(M, 1)
    y = np.random.rand(N, D)
    b = np.random.rand(N, E)
    g = np.random.rand(N, 1)
    sigma = np.array([0.4])
    
    X = np.random.rand(A, 1, M, D)
    L = np.random.rand(1, B, M, 1)
    Y = np.random.rand(1, B, N, D)
    S = np.random.rand(A, B, 1) + 1

    type_to_test = ['float32', 'float64']

    @unittest.skipIf(not pykeops.gpu_available, 'No GPU detected. Skip tests.')
    ############################################################
    def test_fshape_scp_specific(self):
    ############################################################
        from pykeops.numpy.shape_distance import FshapeScp
        for k, t in itertools.product(['gaussian', 'cauchy'], self.type_to_test):
            # Call cuda kernel
            kgeom = k
            ksig = 'gaussian'
            ksphere = 'gaussian_oriented'

            sigma_geom = 1.0
            sigma_sig = 1.0
            sigma_sphere = np.pi / 2
            # Call cuda kernel
            my_fshape_scp = FshapeScp(kernel_geom=kgeom, kernel_sig=ksig, kernel_sphere=ksphere, dtype=t)
            gamma = my_fshape_scp(self.x.astype(t), self.y.astype(t),
                                  self.f.astype(t), self.g.astype(t),
                                  self.a.astype(t), self.b.astype(t),
                                  sigma_geom=sigma_geom, sigma_sig=sigma_sig, sigma_sphere=sigma_sphere).ravel()

            # Python version
            areaa = np.linalg.norm(self.a, axis=1)
            areab = np.linalg.norm(self.b, axis=1)

            nalpha = self.a / areaa[:, np.newaxis]
            nbeta = self.b / areab[:, np.newaxis]

            gamma_py = np.sum((areaa[:, np.newaxis] * areab[np.newaxis, :])
                              * np_kernel(self.x, self.y, sigma_geom, kgeom)
                              * np_kernel(self.f, self.g, sigma_sig, ksig)
                              * np_kernel_sphere(nalpha, nbeta, sigma_sphere, ksphere), axis=1)

            # compare output
            print("ee")
            self.assertTrue(np.allclose(gamma, gamma_py, atol=1e-6))

    @unittest.skipIf(not pykeops.gpu_available,'No GPU detected. Skip tests.')
    ############################################################
    def test_gaussian_conv_specific(self):
    ############################################################
        from pykeops.numpy.convolutions.radial_kernel import RadialKernelConv
        for k, t in itertools.product(['gaussian', 'laplacian', 'cauchy', 'inverse_multiquadric'], self.type_to_test):
            with self.subTest(k=k):
                # Call cuda kernel
                my_radial_conv = RadialKernelConv(t)
                gamma = my_radial_conv(self.x.astype(t), self.y.astype(t), self.b.astype(t), self.sigma.astype(t), kernel = k)

                # Numpy version
                gamma_py = np.matmul(np_kernel(self.x, self.y, self.sigma, kernel=k), self.b)

                # compare output
                self.assertTrue(np.allclose(gamma, gamma_py, atol=1e-6))

    @unittest.skipIf(not pykeops.gpu_available, 'No GPU detected. Skip tests.')
    ############################################################
    def test_gaussian_grad1conv_specific(self):
    ############################################################
        from pykeops.numpy.convolutions.radial_kernel import RadialKernelGrad1conv
        for k, t in itertools.product(['gaussian', 'laplacian', 'cauchy', 'inverse_multiquadric'], self.type_to_test):
            with self.subTest(k=k, t=t):
                # Call cuda kernel
                my_radial_conv = RadialKernelGrad1conv(t)
                gamma = my_radial_conv(self.a.astype(t), self.x.astype(t), self.y.astype(t), self.b.astype(t), self.sigma.astype(t), kernel=k)

                # Numpy version
                tmp = differences(self.x, self.y) * grad_np_kernel(self.x, self.y, self.sigma, kernel=k)
                gamma_py = 2 * (np.sum(self.a * (np.matmul(tmp, self.b)), axis=2)).T

                # compare output
                self.assertTrue( np.allclose(gamma, gamma_py,atol=1e-6))

    ############################################################
    def test_generic_syntax_sum(self):
    ############################################################
        from pykeops.numpy import Genred
        aliases = ['p=Pm(0,1)', 'a=Vj(1,1)', 'x=Vi(2,3)', 'y=Vj(3,3)']
        formula = 'Square(p-a)*Exp(x+y)'
        axis = 1  # 0 means summation over i, 1 means over j

        if pykeops.gpu_available:
            backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'GPU']
        else:
            backend_to_test = ['auto']

        for b, t in itertools.product(backend_to_test, self.type_to_test):
            with self.subTest(b=b, t=t):

                # Call cuda kernel
                myconv = Genred(formula, aliases, reduction_op='Sum', axis=axis, dtype=t)
                gamma_keops = myconv(self.sigma.astype(t), self.g.astype(t), self.x.astype(t), self.y.astype(t), backend=b)

                # Numpy version
                gamma_py = np.sum((self.sigma - self.g)**2
                                  * np.exp((self.y.T[:,:,np.newaxis] + self.x.T[:,np.newaxis,:])), axis=1).T

                # compare output
                self.assertTrue( np.allclose(gamma_keops, gamma_py , atol=1e-6))

    ############################################################
    def test_generic_syntax_lse(self):
    ############################################################
        from pykeops.numpy import Genred
        aliases = ['p=Pm(0,1)', 'a=Vj(1,1)', 'x=Vi(2,3)', 'y=Vj(3,3)']
        formula = 'Square(p-a)*Exp(-SqNorm2(x-y))'

        if pykeops.gpu_available:
            backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'GPU']
        else:
            backend_to_test = ['auto']

        for b, t in itertools.product(backend_to_test, self.type_to_test):
            with self.subTest(b=b, t=t):

                # Call cuda kernel
                myconv = Genred(formula, aliases, reduction_op='LogSumExp', axis=1, dtype=t)
                gamma_keops= myconv(self.sigma.astype(t), self.g.astype(t), self.x.astype(t), self.y.astype(t), backend=b)

                # Numpy version
                gamma_py = log_sum_exp((self.sigma - self.g.T)**2 * np.exp(-squared_distances(self.x, self.y)), axis=1)
                
                # compare output
                self.assertTrue(np.allclose(gamma_keops.ravel(), gamma_py, atol=1e-6))
            
    ############################################################
    def test_generic_syntax_softmax(self):
    ############################################################
        from pykeops.numpy import Genred
        aliases = ['p=Pm(0,1)', 'a=Vj(1,1)', 'x=Vi(2,3)', 'y=Vj(3,3)']
        formula = 'Square(p-a)*Exp(-SqNorm2(x-y))'
        formula_weights = 'y'
        
        if pykeops.gpu_available:
            backend_to_test = ['auto', 'GPU_1D', 'GPU_2D', 'GPU']
        else:
            backend_to_test = ['auto']

        for b, t in itertools.product(backend_to_test, self.type_to_test):
            with self.subTest(b=b, t=t):

                # Call cuda kernel
                myop = Genred(formula, aliases, reduction_op='SumSoftMaxWeight', axis=1, dtype=t, formula2=formula_weights)
                gamma_keops= myop(self.sigma.astype(t), self.g.astype(t), self.x.astype(t), self.y.astype(t), backend=b)

                # Numpy version
                def np_softmax(x,w):
                    x -= np.max(x,axis=1)[:,None] # subtract the max for robustness
                    return np.exp(x)@w/np.sum(np.exp(x),axis=1)[:,None]
                gamma_py = np_softmax((self.sigma - self.g.T)**2 * np.exp(-squared_distances(self.x, self.y)), self.y)
                
                # compare output
                self.assertTrue(np.allclose(gamma_keops.ravel(), gamma_py.ravel(), atol=1e-6))
                        
    ############################################################
    def test_non_contiguity(self):
    ############################################################
        from pykeops.numpy import Genred
        
        t = self.type_to_test[0]

        aliases = ['p=Pm(0,1)', 'a=Vj(1,1)', 'x=Vi(2,3)', 'y=Vj(3,3)']
        formula = 'Square(p-a)*Exp(-SqNorm2(y-x))'

        my_routine = Genred(formula, aliases, reduction_op='Sum', axis=1)
        gamma_keops1 = my_routine(self.sigma.astype(t), self.g.astype(t), self.x.astype(t), self.y.astype(t), backend='auto')
        
        yc_tmp = np.ascontiguousarray(self.y.T).T  # create a non contiguous copy
        gamma_keops2 = my_routine(self.sigma.astype(t), self.g.astype(t), self.x.astype(t), yc_tmp.astype(t))

        # check output
        self.assertFalse(yc_tmp.flags.c_contiguous)
        self.assertTrue(np.allclose(gamma_keops1, gamma_keops2))
        
    ############################################################
    def test_heterogeneous_var_aliases(self):
    ############################################################
        from pykeops.numpy import Genred
        
        t = self.type_to_test[0]

        aliases = ['p=Pm(0,1)', 'x=Vi(1,3)', 'y=Vj(2,3)']
        formula = 'Square(p-Var(3,1,1))*Exp(-SqNorm2(y-x))'

        # Call cuda kernel
        myconv = Genred(formula, aliases, reduction_op='Sum', axis=1)
        gamma_keops = myconv(self.sigma.astype(t), self.x.astype(t), self.y.astype(t), self.g.astype(t), backend='auto')

        # Numpy version
        gamma_py = np.sum((self.sigma - self.g.T)**2 * np.exp(-squared_distances(self.x, self.y)), axis=1)
        
        # compare output
        self.assertTrue(np.allclose(gamma_keops.ravel(), gamma_py, atol=1e-6))
        
    ############################################################
    def test_formula_simplification(self):
    ############################################################
        from pykeops.numpy import Genred
        
        t = self.type_to_test[0]

        aliases = ['x=Vi(0,3)']
        formula = 'Grad(Grad(x + Var(1,3,1), x, Var(2,3,0)),x, Var(3,3,0))'

        # Call cuda kernel
        myconv = Genred(formula, aliases, reduction_op='Sum', axis=1)
        gamma_keops= myconv(self.x.astype(t), self.y.astype(t), self.x.astype(t), self.x.astype(t), backend='auto')

        # Numpy version
        gamma_py = np.zeros_like(self.x)
        
        # compare output
        self.assertTrue(np.allclose(gamma_keops, gamma_py, atol=1e-6))

    ############################################################
    def test_argkmin(self):
    ############################################################

        from pykeops.numpy import Genred
        formula = 'SqDist(x,y)'
        variables = ['x = Vi('+str(self.D)+')',  # First arg   : i-variable, of size D
                     'y = Vj('+str(self.D)+')']  # Second arg  : j-variable, of size D


        my_routine = Genred(formula, variables, reduction_op='ArgKMin', axis=1, dtype=self.type_to_test[1], opt_arg=3)

        c = my_routine(self.x, self.y, backend="auto").astype(int)
        cnp = np.argsort(np.sum((self.x[:,np.newaxis,:] - self.y[np.newaxis,:,:]) ** 2, axis=2), axis=1)[:,:3]
        self.assertTrue(np.allclose(c.ravel(),cnp.ravel()))
    
    ############################################################
    def test_LazyTensor_sum(self):
    ############################################################
        from pykeops.numpy import LazyTensor
        
        full_results = []
        for use_keops in [True, False]:
            
            results = []
            
            for (x, l, y, s) in [(self.X.astype(t), self.L.astype(t), self.Y.astype(t), self.S.astype(t)) for t in
                                 self.type_to_test]:
                
                x_i = x[:, :, :, None, :]
                l_i = l[:, :, :, None, :]
                y_j = y[:, :, None, :, :]
                s_p = s[:, :, None, None, :]
                
                if use_keops:
                    x_i, l_i, y_j, s_p = LazyTensor(x_i), LazyTensor(l_i), LazyTensor(y_j), LazyTensor(s_p)
                
                D_ij = ((l_i + x_i * y_j) ** 2 + s_p).sum(-1)
                
                if use_keops:
                    K_ij = 1 / (1 + D_ij).exp()
                else:
                    K_ij = 1 / np.exp(1 + D_ij)
                
                a_i = K_ij.sum(self.nbatchdims + 1)
                if use_keops: a_i = a_i.squeeze(-1)
                
                results += [a_i]
            
            full_results.append(results)
        
        for (res_keops, res_numpy) in zip(full_results[0], full_results[1]):
            self.assertTrue(res_keops.shape == res_numpy.shape)
            self.assertTrue(np.allclose(res_keops, res_numpy, atol=1e-3))


if __name__ == '__main__':
    unittest.main()
