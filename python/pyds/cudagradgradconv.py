import numpy as np
import ctypes
from ctypes import *
import os.path

#nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=192"  -Xcompiler -fPIC -shared -o cuda_conv.so cuda_conv.cu

# extract cuda_gradconv function pointer in the shared object cuda_gradconv.so
def get_cuda_gradconv_xa_xx_xy_xb():
	"""
	Loads the gradient of the convolution routine from the compiled .so files.
	"""
	cuda_routines = []
	for dll_name in ['cuda_gradconv_xa.so', 'cuda_gradconv_xx.so', 'cuda_gradconv_xy.so', 'cuda_gradconv_xb.so']:
                dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + '..' + os.path.sep+ 'build' + os.path.sep + dll_name
		dll = ctypes.CDLL(dllabspath, mode=ctypes.RTLD_GLOBAL)
		if   dll_name == 'cuda_gradconv_xa.so' :
			func = dll.GaussGpuGradConvXA
		elif dll_name == 'cuda_gradconv_xx.so' :
			func = dll.GaussGpuGradConvXX
		elif dll_name == 'cuda_gradconv_xy.so' :
			func = dll.GaussGpuGradConvXY
		elif dll_name == 'cuda_gradconv_xb.so' :
			func = dll.GaussGpuGradConvXB
		
		# Arguments :          1/s^2,              e,
		func.argtypes = [     c_float,     POINTER(c_float), 
		#                      alpha,              x,                y,              beta,             result,
						 POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), 
		#                 dim-xy,  dim-beta,   nx,    ny
						  c_int,    c_int,   c_int, c_int  ]
		cuda_routines.append(func)
		
	return cuda_routines

# create __cuda_gradconv_** function with get_cuda_gradconv_**()
__cuda_gradconv_xa, __cuda_gradconv_xx, __cuda_gradconv_xy, __cuda_gradconv_xb = get_cuda_gradconv_xa_xx_xy_xb()

# convenient python wrapper for __cuda_gradconv_** it does all job with types convertation from python ones to C++ ones 
def cuda_gradconv_order2(e, alpha, x, y, beta, result, sigma, mode):
	"""
	
	"""
	# From python to C float pointers and int :
	e_p      =      e.ctypes.data_as(POINTER(c_float))
	alpha_p  =  alpha.ctypes.data_as(POINTER(c_float))
	x_p      =      x.ctypes.data_as(POINTER(c_float))
	y_p      =      y.ctypes.data_as(POINTER(c_float))
	beta_p   =   beta.ctypes.data_as(POINTER(c_float))
	result_p = result.ctypes.data_as(POINTER(c_float))
	
	nx = x.shape[0] ; ny = y.shape[0]
	
	dimPoint = x.shape[1]
	dimVect  = beta.shape[1]
	
	ooSigma2 = float(1/ (sigma*sigma))  # Compute this once and for all
	
	# Let's use our GPU, which works "in place" :
	if   mode == "xa" :
		__cuda_gradconv_xa(ooSigma2, e_p, alpha_p, x_p, y_p, beta_p, result_p, dimPoint, dimVect, nx, ny )
	elif mode == "xx" :
		__cuda_gradconv_xx(ooSigma2, e_p, alpha_p, x_p, y_p, beta_p, result_p, dimPoint, dimVect, nx, ny )
	elif mode == "xy" :
		__cuda_gradconv_xy(ooSigma2, e_p, alpha_p, x_p, y_p, beta_p, result_p, dimPoint, dimVect, nx, ny )
	elif mode == "xb" :
		__cuda_gradconv_xb(ooSigma2, e_p, alpha_p, x_p, y_p, beta_p, result_p, dimPoint, dimVect, nx, ny )
	else :
		a = 1/0
	

def cuda_gradconv_xa(e, alpha, x, y, beta, result, sigma) :
	cuda_gradconv_order2(e, alpha, x, y, beta, result, sigma, "xa")

def cuda_gradconv_xx(e, alpha, x, y, beta, result, sigma) :
	cuda_gradconv_order2(e, alpha, x, y, beta, result, sigma, "xx")

def cuda_gradconv_xy(e, alpha, x, y, beta, result, sigma) :
	cuda_gradconv_order2(e, alpha, x, y, beta, result, sigma, "xy")

def cuda_gradconv_xb(e, alpha, x, y, beta, result, sigma) :
	cuda_gradconv_order2(e, alpha, x, y, beta, result, sigma, "xb")


















