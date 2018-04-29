import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import unittest
import numpy as np

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

    def squared_distances(self,x, y):
        return np.sum((x[:,np.newaxis,:] - y[np.newaxis,:,:]) ** 2, axis=2)

    def differences(self,x, y):
        return (x.T[:,:,np.newaxis] - y.T[:,np.newaxis,:])

    def test_gaussian_conv_specific(self):
        # Call cuda kernel
        gamma = np.zeros(self.dimVect*self.sizeX).astype('float32')
        radial_kernels_conv(self.x, self.y, self.beta, gamma, self.sigma)
        gamma = gamma.reshape((self.sizeX,self.dimVect))

        # Numpy version    
        gamma_py = np.matmul(np.exp(-self.squared_distances(self.x, self.y) / (self.sigma * self.sigma)), self.beta)

        # compare output
        self.assertTrue(np.allclose(gamma, gamma_py ))

    def test_gaussian_grda1conv_specific(self):
        # Call cuda kernel
        gamma = np.zeros(self.dimPoint*self.sizeX).astype('float32')
        radial_kernels_grad1conv(self.alpha,self.x, self.y, self.beta, gamma, self.sigma) # In place, gamma_i = k(x_i,y_j) @ beta_j
        gamma = gamma.reshape((self.sizeX,self.dimPoint))

        # Numpy version
        A = self.differences(self.x, self.y) * np.exp(-self.squared_distances(self.x, self.y) / (self.sigma * self.sigma))
        gamma_py = -2*(np.sum( self.alpha * (np.matmul(A,self.beta)),axis=2) / (self.sigma * self.sigma)).T

        # compare output
        self.assertTrue(np.allclose(gamma, gamma_py ))

if __name__ == '__main__':
    unittest.main()
