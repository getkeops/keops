# To test, first compile the kernel via :
# ./compile "GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<2,3>>"
#
# This will compile the isotropic Gaussian kernel in dimension 3,
# which takes as input :
# - a scalar parameter P<0>, inverse of the variance
# - an array X_0 (x_i) of dimension N-by-3
# - an array Y_1 (y_j) of dimension M-by-3
# - an array Y_2 (b_j) of dimension M-by-3

import numpy as np
import ctypes
from ctypes import *
import os.path

# extract cuda_conv function pointer in the shared object 
def get_cuda_convs():
	"""
	Loads the convolution routine from the compiled .so file.
	"""
	dll_name = 'GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<2,3>>_float.so'
	dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + 'build' + os.path.sep + dll_name
	dll = ctypes.CDLL(dllabspath, mode=ctypes.RTLD_GLOBAL)
	
	func_dict = {}
	func_dict
	for (name, routine) in [("gaussian",  dll.GpuConv) ] :
		func = routine
		# Arguments :     params,          nx,    ny,    result,                args
		func.argtypes = [POINTER(c_float), c_int, c_int, POINTER(c_float), POINTER(POINTER(c_float))]
		func_dict[name] = func
	return func_dict

# create __cuda_conv function with get_cuda_conv()
__cuda_convs = get_cuda_convs()

# convenient python wrapper for __cuda_conv it does all job with types convertation from python ones to C++ ones 
def cuda_conv(x, y, beta, result, ooSigma2, kernel = "gaussian"):
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
	x_p = x.ctypes.data_as(POINTER(c_float))
	y_p = y.ctypes.data_as(POINTER(c_float))
	beta_p = beta.ctypes.data_as(POINTER(c_float))
	args_p      = (POINTER(c_float)*3)(x_p,y_p,beta_p)
	result_p  = result.ctypes.data_as(POINTER(c_float))
	
	nx = x.shape[0] ; ny = y.shape[0]
	
	dimPoint =    x.shape[1]
	dimVect  = beta.shape[1]

	params_p = ooSigma2.ctypes.data_as(POINTER(c_float))
	
	# Let's use our GPU, which works "in place" :
	__cuda_convs[kernel](params_p, nx, ny, result_p, args_p )

# Ideally, this routine could be implemented by Joan :
def cuda_conv_generic(formula, result, *args, cuda_type = "float", grid_scheme = "2D", aliases = []):
	"""
	Executes the "autodiff" kernel associated to "formula". 
	Aliases can be given as a list of strings.
	The arguments are given as :
		params, then variables, sorted in the order specified by the "Var<index,dimension,I-or-J>" syntax.
	For instance,
		```    
		aliases = [ "DIMPOINT = 3", "DIMVECT = 4",
		            "X = Var<0,DIMPOINT,0>" ,
		            "Y = Var<1,DIMPOINT,1>" ,
		            "U = Var<2,DIMVECT ,0>" ,
		            "V = Var<3,DIMVECT ,1>" ,
		            "B = Var<4,DIMPOINT,1>" ,
		            "C = Param<0>"          ]
		formula = "Scal< Square<Scalprod<U,V>>, " \
		        + "Scal< Exp< Scal<Constant<C>, Minus<SqNorm2<Subtract<X,Y>>> > >,  B> >"
		cuda_conv_generic( formula,
		                   R, C, X, Y, U, V, B,
		                   aliases = aliases )
		```
	is a legal call, where :
	- R is a nx-by-4 float array (the output array)
	- C is a scalar
	- X is a nx-by-3 float array
	- Y is a ny-by-3 float array
	- U is a nx-by-4 float array
	- V is a ny-by-4 float array
	- B is a ny-by-4 float array
	
	(nx and ny are automatically inferred from the data; 
	an error is thrown if the lengths of the input arrays are not compatible with each other)
	
	If the CUDA kernel associated to the given formula is not found in the "build/" folder,
	the routine is compiled on-the-fly using the "compile" script.
	"""
	None

if __name__ == '__main__':
	"""
	testing, benchmark convolution with two naive python implementations of the Gaussian convolution
	"""
	np.set_printoptions(linewidth=200)
	
	sizeX    = int(500)
	sizeY    = int(100)
	dimPoint = int(3)
	dimVect  = int(3)
	sigma    = float(2)
	
	if True : # Random test
		x    = np.random.rand(sizeX,dimPoint).astype('float32')
		y    = np.random.rand(sizeY,dimPoint).astype('float32')
		beta = np.random.rand(sizeY,dimVect ).astype('float32')
	else :    # Deterministic one
		x    = np.ones((sizeX,dimPoint)).astype('float32')
		y    = np.ones((sizeY,dimPoint)).astype('float32')
		beta = np.ones((sizeY,dimVect)).astype('float32')
		
	ooSigma2 = np.array([float(1/ (sigma*sigma))]).astype('float32') # Compute this once and for all
	# Call cuda kernel
	gamma = np.zeros(dimVect*sizeX).astype('float32')
	cuda_conv(x, y, beta, gamma, ooSigma2) # In place, gamma_i = k(x_i,y_j) @ beta_j
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
