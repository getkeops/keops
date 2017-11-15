import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..')

from pykp import cudaconv,cudagradconv,cudagradgradconv
import numpy as np



N = 100000 ; M = 150000; D = 3; E = 3

if False :
	e = .1 * np.linspace(  0,  5, N*D).reshape((N,D)).astype('float32')
	a = .2 * np.linspace(  0,  5, N*E).reshape((N,E)).astype('float32')
	x = .3 * np.linspace(  0,  5, N*D).reshape((N,D)).astype('float32')
	y = .4 * np.linspace(  0,  5, M*D).reshape((M,D)).astype('float32')
	b = .5 * np.linspace( 0, .2, M*E).reshape((M,E)).astype('float32')
else :
	e = np.random.rand(N,D).astype('float32')
	a = np.random.rand(N,E).astype('float32')
	x = np.random.rand(N,D).astype('float32')
	y = np.random.rand(M,D).astype('float32')
	b = np.random.rand(M,E).astype('float32')
	
s = np.array([.2]).astype('float32')

print("x : \n", x)
print("y : \n", y)
print("b : \n", b)

# Order 0
g = np.zeros(a.shape).astype('float32')
cudaconv.cuda_conv(x, y, b, g, s)
print("conv : \n", g)

# Order 1
g_x = np.zeros(x.shape).astype('float32')
cudagradconv.cuda_gradconv(a, x, y, b, g_x, s)
print("g_x : \n", g_x)

# Order 2
g_xa = np.zeros(a.shape).astype('float32')
cudagradgradconv.cuda_gradconv_xa(e, a, x, y, b, g_xa, s)
print("g_xa : \n", g_xa)

g_xx = np.zeros(x.shape).astype('float32')
cudagradgradconv.cuda_gradconv_xx(e, a, x, y, b, g_xx, s)
print("g_xx : \n", g_xx)

g_xy = np.zeros(y.shape).astype('float32')
cudagradgradconv.cuda_gradconv_xy(e, a, x, y, b, g_xy, s)
print("g_xy : \n", g_xy)

g_xb = np.zeros(b.shape).astype('float32')
cudagradgradconv.cuda_gradconv_xb(e, a, x, y, b, g_xb, s)
print("g_xb : \n", g_xb)

