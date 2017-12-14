import numpy as np
import ctypes
from ctypes import *
import os.path

#nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=192"  -Xcompiler -fPIC -shared -o cuda_conv.so cuda_conv.cu

# extract cuda_conv function pointer in the shared object cuda_conv.so
def get_cuda_convs():
	"""
	Loads the convolution routine from the compiled .so file.
	"""
	dll_name = 'cuda_conv.so'
	dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + 'build' + os.path.sep + dll_name
	dll = ctypes.CDLL(dllabspath, mode=ctypes.RTLD_GLOBAL)
	
	func_dict = {}
	for (name, routine) in [("gaussian",  dll.GaussGpuEvalConv), 
	                        ("laplacian", dll.LaplaceGpuEvalConv), 
	                        ("energy",    dll.EnergyGpuEvalConv) ] :
		func = routine
		# Arguments :     1/s^2,         x,                y,              beta,             result,
		func.argtypes = [c_float, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
		#                dim-xy,  dim-beta,   nx,    ny
						 c_int,   c_int,     c_int, c_int]
		func_dict[name] = func
	return func_dict

# create __cuda_conv function with get_cuda_conv()
__cuda_convs = get_cuda_convs()

# convenient python wrapper for __cuda_conv it does all job with types convertation from python ones to C++ ones 
def cuda_conv(x, y, beta, result, sigma, kernel = "gaussian"):
	"""
	Implements the operation :
	
	(x_i, y_j, beta_j)  ->  ( \sum_j k(x_i,y_j) beta_j )_i ,
	
	where k is a kernel function of parameter "sigma".
	Unlike a naive pytorch implementation, this code won't store in memory the matrix
	k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
	without getting a "memory overflow".
	
	N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p". 
	"""
	# From python to C float pointers and int :
	x_p      =      x.ctypes.data_as(POINTER(c_float))
	y_p      =      y.ctypes.data_as(POINTER(c_float))
	beta_p   =   beta.ctypes.data_as(POINTER(c_float))
	result_p = result.ctypes.data_as(POINTER(c_float))
	
	nx = x.shape[0] ; ny = y.shape[0]
	
	dimPoint =    x.shape[1]
	dimVect  = beta.shape[1]
	
	ooSigma2 = float(1/ (sigma*sigma)) # Compute this once and for all
	
	# Let's use our GPU, which works "in place" :
	__cuda_convs[kernel](ooSigma2, x_p, y_p, beta_p, result_p, dimPoint, dimVect, nx, ny )



if __name__ == '__main__':
	"""
	testing, benchmark convolution with two naive python implementations of the Gaussian convolution
	"""
	np.set_printoptions(linewidth=200)
	
	sizeX    = int(600)
	sizeY    = int(100)
	dimPoint = int(3)
	dimVect  = int(3)
	sigma    = float(2)
	
	if True : # Random test
		x    = np.random.rand(sizeX,dimPoint).astype('float32')
		y    = np.random.rand(sizeY,dimPoint).astype('float32')
		beta = np.random.rand(sizeY,dimVect ).astype('float32')
	else :    # Deterministic one
		x    = np.sin([ np.arange(float(sizeX)), np.arange(float(sizeX))* 2,np.arange(float(sizeX))**2] ).astype('float32')
		y    = np.cos([ np.arange(float(sizeY)), np.arange(float(sizeY))* 2,np.arange(float(sizeY))**2] ).astype('float32')
		beta = np.array(np.arange(float(dimVect*sizeY)) +1).reshape(dimVect,sizeY).astype('float32')
		
	# Call cuda kernel
	gamma = np.zeros(dimVect*sizeX).astype('float32')
	cuda_conv(x, y, beta, gamma, sigma) # In place, gamma_i = k(x_i,y_j) @ beta_j
	gamma = gamma.reshape((sizeX,dimVect))
	
	# A first implementation, with (shock horror !) a bunch of "for" loops
	oosigma2 = 1 / (sigma * sigma) 
	gamma_py = np.zeros((sizeX,dimVect)).astype('float32')
	
	for i in range(sizeX):
		for j in range(sizeY):
			rij2 = 0.
			for k in range(dimPoint):
				rij2 += (x[i,k] - y[j,k]) ** 2
			for l in range(dimVect):
				gamma_py[i,l] +=  np.exp(-rij2 * oosigma2) * beta[j,l]

	# A second implementation, a bit more efficient
	r2 = np.zeros((sizeX,sizeY)).astype('float32')
	for i in range(sizeX):
		for j in range(sizeY):
			for k in range(dimPoint):
				r2[i,j] += (x[i,k] - y[j,k]) ** 2
				
	K         = np.exp(-r2 * oosigma2)
	gamma_py2 = np.dot(K,beta)
	
	# compare output
	print("\nCuda convolution :")
	print(gamma)
	
	print("\nPython convolution 1 :")
	print(gamma_py)
	
	print("\nPython convolution 2 :")
	print(gamma_py2)
	
	print("\nIs everything okay ? ")
	print(np.allclose(gamma, gamma_py ))
	print(np.allclose(gamma, gamma_py2))
