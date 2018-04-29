import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import unittest
import numpy as np

from pykeops.numpy.utils import np_kernel, grad_np_kernel, differences, squared_distances
from pykeops.numpy.convolutions.radial_kernels import radial_kernels_conv
from pykeops.numpy.convolutions.radial_kernels_grad1 import radial_kernels_grad1conv


class NumpyUnitTestCase(unittest.TestCase):

    sizeX    = int(60)
    sizeY    = int(10)
    dimPoint = int(3)
    dimVect  = int(3)
    sigma    = float(2)

    alpha = np.random.rand(sizeX,dimVect ).astype('float32')
    x     = np.random.rand(sizeX,dimPoint).astype('float32')
    y     = np.random.rand(sizeY,dimPoint).astype('float32')
    beta  = np.random.rand(sizeY,dimVect ).astype('float32')

    def test_gaussian_conv_specific(self):
        for k in (["gaussian", "laplacian", "cauchy", "inverse_multiquadric"]):
            with self.subTest(k=k):
                # Call cuda kernel
                gamma = np.zeros(self.dimVect*self.sizeX).astype('float32')
                radial_kernels_conv(self.x, self.y, self.beta, gamma, self.sigma,kernel=k)

                # Numpy version    
                gamma_py = np.matmul(np_kernel(self.x, self.y,self.sigma,kernel=k), self.beta)

                # compare output
                self.assertTrue( np.allclose(gamma, gamma_py.ravel(),atol=1e-6))

    def test_gaussian_grad1conv_specific(self):
        for k in (["gaussian", "laplacian", "cauchy", "inverse_multiquadric"]):
            with self.subTest(k=k):
                # Call cuda kernel
                gamma = np.zeros(self.dimPoint*self.sizeX).astype('float32')
                radial_kernels_grad1conv(self.alpha,self.x, self.y, self.beta, gamma, self.sigma,kernel=k) # In place, gamma_i = k(x_i,y_j) @ beta_j

                # Numpy version
                A = differences(self.x, self.y) * grad_np_kernel(self.x,self.y,self.sigma,kernel=k)
                gamma_py = 2*(np.sum( self.alpha * (np.matmul(A,self.beta)),axis=2) ).T

                # compare output
                self.assertTrue( np.allclose(gamma, gamma_py.ravel(),atol=1e-6))

if __name__ == '__main__':
    unittest.main()
