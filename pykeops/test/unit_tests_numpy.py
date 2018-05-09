import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import unittest
import numpy as np

from pykeops.numpy.utils import np_kernel, grad_np_kernel, differences, squared_distances

from pykeops import gpu_available

class NumpyUnitTestCase(unittest.TestCase):

    N    = int(6)
    M    = int(10)
    D = int(3)
    E  = int(3)

    a = np.random.rand(N,E).astype('float32')
    x = np.random.rand(N,D).astype('float32')
    y = np.random.rand(M,D).astype('float32')
    f = np.random.rand(M,1).astype('float32')
    b = np.random.rand(M,E).astype('float32')
    sigma = np.array([0.4]).astype('float32')

    @unittest.skipIf(not gpu_available,"No GPU detected. Skip tests.")
#--------------------------------------------------------------------------------------
    def test_gaussian_conv_specific(self):
#--------------------------------------------------------------------------------------
        from pykeops.numpy.convolutions.radial_kernels import radial_kernels_conv
        for k in (["gaussian", "laplacian", "cauchy", "inverse_multiquadric"]):
            with self.subTest(k=k):
                # Call cuda kernel
                gamma = np.zeros(self.E*self.N).astype('float32')
                radial_kernels_conv(self.x, self.y, self.b, gamma, self.sigma,kernel=k)

                # Numpy version    
                gamma_py = np.matmul(np_kernel(self.x, self.y,self.sigma,kernel=k), self.b)

                # compare output
                self.assertTrue( np.allclose(gamma, gamma_py.ravel(),atol=1e-6))

    @unittest.skipIf(not gpu_available,"No GPU detected. Skip tests.")
#--------------------------------------------------------------------------------------
    def test_gaussian_grad1conv_specific(self):
#--------------------------------------------------------------------------------------
        from pykeops.numpy.convolutions.radial_kernels_grad1 import radial_kernels_grad1conv
        for k in (["gaussian", "laplacian", "cauchy", "inverse_multiquadric"]):
            with self.subTest(k=k):
                # Call cuda kernel
                gamma = np.zeros(self.D*self.N).astype('float32')
                radial_kernels_grad1conv(self.a,self.x, self.y, self.b, gamma, self.sigma,kernel=k) # In place, gamma_i = k(x_i,y_j) @ beta_j

                # Numpy version
                A = differences(self.x, self.y) * grad_np_kernel(self.x,self.y,self.sigma,kernel=k)
                gamma_py = 2*(np.sum( self.a * (np.matmul(A,self.b)),axis=2) ).T

                # compare output
                self.assertTrue( np.allclose(gamma, gamma_py.ravel(),atol=1e-6))

#--------------------------------------------------------------------------------------
    def test_generic_syntax(self):
#--------------------------------------------------------------------------------------
        from pykeops.numpy.generic_sum import GenericSum_np
        aliases = ["p=Pm(0,1)","a=Vy(1,1)","x=Vx(2,3)","y=Vy(3,3)"]
        formula = "Square(p-a)*Exp(x+y)"
        signature   =   [ (3, 0), (1, 2), (1, 1), (3, 0), (3, 1) ]
        sum_index = 0       # 0 means summation over j, 1 means over i 

        if gpu_available:
            backend_to_test = ['auto','GPU_1D','GPU_2D','GPU']
        else:
            backend_to_test = ['auto']

        for b in backend_to_test:
            with self.subTest(b=b):

                # Call cuda kernel
                gamma_keops = GenericSum_np(b,aliases,formula,signature,sum_index,self.sigma,self.f,self.x,self.y)

                # Numpy version
                gamma_py = np.sum((self.sigma - self.f)**2 *np.exp( (self.y.T[:,:,np.newaxis] + self.x.T[:,np.newaxis,:])),axis=1).T

                # compare output
                self.assertTrue( np.allclose(gamma_keops, gamma_py , atol=1e-6))


if __name__ == '__main__':
    unittest.main()
