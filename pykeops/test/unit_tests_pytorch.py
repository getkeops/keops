import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import unittest
import itertools
import numpy as np

from pykeops.numpy.utils import np_kernel, log_np_kernel, grad_np_kernel, differences

from pykeops import torch_found, gpu_available

@unittest.skipIf(not torch_found,"Pytorch was not found on your system. Skip tests.")
class PytorchUnitTestCase(unittest.TestCase):

    N    = int(6)
    M    = int(10)
    D = int(3)
    E  = int(3)

    a = np.random.rand(N,E).astype('float32')
    x = np.random.rand(N,D).astype('float32')
    y = np.random.rand(M,D).astype('float32')
    f = np.random.rand(M,1).astype('float32')
    b = np.random.rand(M,E).astype('float32')
    p = np.random.rand(2).astype('float32')
    sigma = np.array([0.4]).astype('float32')

    try:
        import torch
        from torch.autograd import Variable

        use_cuda = torch.cuda.is_available()
        dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        ac = Variable(torch.from_numpy(a.copy()).type(dtype), requires_grad=True).type(dtype)
        xc = Variable(torch.from_numpy(x.copy()).type(dtype), requires_grad=True).type(dtype)
        yc = Variable(torch.from_numpy(y.copy()).type(dtype), requires_grad=True).type(dtype)
        fc = Variable(torch.from_numpy(f.copy()).type(dtype), requires_grad=True).type(dtype)
        bc = Variable(torch.from_numpy(b.copy()).type(dtype), requires_grad=True).type(dtype)
        pc = Variable(torch.from_numpy(p.copy()).type(dtype), requires_grad=True).type(dtype)
        sigmac = torch.autograd.Variable(torch.from_numpy(sigma.copy()).type(dtype), requires_grad=False).type(dtype)
        print("Running Pytorch tests.")
    except:
        print("Pytorch could not be loaded.")
        pass

#--------------------------------------------------------------------------------------
    def test_conv_kernels_feature(self):
#--------------------------------------------------------------------------------------
        from pykeops.torch.kernels import Kernel, kernel_product
        params = {
            "gamma" : 1./self.sigmac**2,
            "mode"  : 'sum',
        }
        if gpu_available:
            backend_to_test = ['auto','GPU_1D','GPU_2D','pytorch']
        else:
            backend_to_test = ['auto','pytorch']
        
        for k,b in itertools.product(["gaussian", "laplacian", "cauchy", "inverse_multiquadric"],backend_to_test):
            with self.subTest(k=k,b=b):
                params["id"] = Kernel(k+"(x,y)")
                params["backend"] = b
                # Call cuda kernel
                gamma = kernel_product( params, self.xc,self.yc,self.bc).cpu()

                # Numpy version    
                gamma_py = np.matmul(np_kernel(self.x, self.y,self.sigma,kernel=k), self.b)

                # compare output
                self.assertTrue( np.allclose(gamma.data.numpy(), gamma_py))

#--------------------------------------------------------------------------------------
    def test_grad1conv_kernels_feature(self):
#--------------------------------------------------------------------------------------
        import torch
        from torch.autograd import grad
        from pykeops.torch.kernels import Kernel, kernel_product
        params = {
            "gamma"   : 1./self.sigmac**2,
            "mode"  : 'sum',
        }
        if gpu_available:
            backend_to_test = ['auto','GPU_1D','GPU_2D','pytorch']
        else:
            backend_to_test = ['auto','pytorch']

        for k,b in itertools.product(["gaussian", "laplacian", "cauchy", "inverse_multiquadric"],backend_to_test):
            with self.subTest(k=k,b=b):
                params["id"] = Kernel(k+"(x,y)")
                params["backend"] = b

                # Call cuda kernel
                aKxy_b = torch.dot(self.ac.view(-1), kernel_product( params, self.xc,self.yc,self.bc ).view(-1))
                gamma_keops   = torch.autograd.grad(aKxy_b, self.xc, create_graph=False)[0].cpu()

                # Numpy version
                A = differences(self.x, self.y) * grad_np_kernel(self.x,self.y,self.sigma,kernel=k)
                gamma_py = 2*(np.sum( self.a * (np.matmul(A,self.b)),axis=2) ).T

                # compare output
                self.assertTrue( np.allclose(gamma_keops.cpu().data.numpy(), gamma_py , atol=1e-6))

#--------------------------------------------------------------------------------------
    def test_generic_syntax(self):
#--------------------------------------------------------------------------------------
        from pykeops.torch.kernels import Kernel
        from pykeops.torch.generic_sum import GenericSum
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
                gamma_keops = GenericSum.apply(b,aliases,formula,signature,sum_index,self.sigmac,self.fc,self.xc,self.yc)

                # Numpy version
                gamma_py = np.sum((self.sigma - self.f)**2 *np.exp( (self.y.T[:,:,np.newaxis] + self.x.T[:,np.newaxis,:])),axis=1).T

                # compare output
                self.assertTrue( np.allclose(gamma_keops.cpu().data.numpy(), gamma_py , atol=1e-6))


#--------------------------------------------------------------------------------------
    def test_generic_syntax_simple(self):
#--------------------------------------------------------------------------------------
        from pykeops.torch.kernels import Kernel
        from pykeops.torch.generic_sum import generic_sum

        types = [ "A = Vx(" + str(self.xc.shape[1]) + ") ",  # output,       indexed by i, dim D.
                    "P = Pm(2)",                               # 1st argument,  a parameter, dim 2. 
                    "X = Vx(" + str(self.xc.shape[1]) + ") ",  # 2nd argument, indexed by i, dim D.
                    "Y = Vy(" + str(self.yc.shape[1]) + ") "]  # 3rd argument, indexed by j, dim D.
        # The actual formula:
        # a_i   =   (<x_i,y_j>**2) * (       p[0]*x_i  +       p[1]*y_j )
        formula = "Pow( (X,Y) , 2) * ( (Elem(P,0) * X) + (Elem(P,1) * Y) )"

        if gpu_available:
            backend_to_test = ['auto','GPU_1D','GPU_2D','GPU']
        else:
            backend_to_test = ['auto']

        for b in backend_to_test:
            with self.subTest(b=b):

                my_routine  = generic_sum(formula, *types)
                gamma_keops = my_routine(self.pc, self.xc, self.yc, backend=b)
        
                # Numpy version
                scals = (self.x @ self.y.T)**2 # Memory-intensive computation!
                gamma_py = self.p[0] * scals.sum(1).reshape(-1,1) * self.x \
                         + self.p[1] * (scals @ self.y)

                # compare output
                self.assertTrue( np.allclose(gamma_keops.cpu().data.numpy(), gamma_py , atol=1e-6))

    #@unittest.expectedFailure
#--------------------------------------------------------------------------------------
    def test_logSumExp_kernels_feature(self):
#--------------------------------------------------------------------------------------
        from pykeops.torch.kernels import Kernel, kernel_product
        params = {
            "gamma" : 1./self.sigmac**2,
            "mode"  : "lse",
        }
        if gpu_available:
            backend_to_test = ['auto','GPU_1D','GPU_2D','pytorch']
        else:
            backend_to_test = ['auto','pytorch']

        for k,b in itertools.product(["gaussian", "laplacian", "cauchy", "inverse_multiquadric"],backend_to_test):
            with self.subTest(k=k,b=b):
                params["id"] = Kernel(k+"(x,y)")
                params["backend"] = b
                # Call cuda kernel
                gamma = kernel_product( params, self.xc,self.yc,self.fc).cpu()

                # Numpy version    
                log_K  = log_np_kernel(self.x, self.y,self.sigma,kernel=k)
                log_KP = log_K + self.f.T
                maxexp = np.amax(log_KP, axis=1)

                gamma_py = maxexp + np.log(np.sum(np.exp( log_KP - maxexp.reshape(-1,1) ), axis=1))
                # compare output
                self.assertTrue( np.allclose(gamma.data.numpy().ravel(), gamma_py))

if __name__ == '__main__':
    """
    run tests
    """
    unittest.main()

