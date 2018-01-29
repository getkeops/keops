# Import the relevant tools
import time                 # to measure performance
import numpy as np          # standard array library
import torch
from   torch          import Tensor
from   torch.autograd import Variable

import os
FOLDER = os.path.dirname(os.path.abspath(__file__))+os.path.sep

from ..toolbox.data_attachment import _data_attachment
from ..toolbox.kernel_product  import Kernel
import matplotlib.pyplot as plt

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
use_cuda = False # torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

Mu = (
    Variable(torch.Tensor([ 1., 1.])).type(dtype).view(-1) , # p_i
    Variable(torch.Tensor([-.5, .5])).type(dtype).view(-1,1) # x_i
)

Nu = (
    Variable(torch.Tensor([ 1., 1.])).type(dtype).view(-1) , # q_j
    Variable(torch.Tensor([-.5, .5])).type(dtype).view(-1,1) # y_j
)


def scal_to_var(x) :
	return Variable(Tensor([x])).type(dtype)


backend = "pytorch"

kernel        = Kernel("energy(x,y)")
params_kernel = {
		"formula"  : "kernel",
		"features" : "none",
		"id"     : kernel ,
		"gamma"  : scal_to_var(1/.1**2) ,
		"backend": backend,
}

epsilon = .05**2
eps     = scal_to_var(epsilon)
params_ot = {
		"formula"            : "wasserstein",

		"cost"               : "dual",
		"features"           : "none",
		"kernel" : {"id"     : Kernel("gaussian(x,y)") ,
					"gamma"  : 1/eps ,
					"backend": backend  },
		"epsilon"            : eps,
		"rho"                : -1,              # < 0 -> no unbalanced transport
		"tau"                : 0.,              # Using acceleration with nits < ~40 leads to *very* unstable transport plans
		"nits"               : 100,
		"tol"                : 1e-7,
		"transport_plan"     : "none",
}


def translate_cost(params, t) :
    Mu = (
        Variable(torch.Tensor([ 1.,   1.  ])).type(dtype).view(-1) , # p_i
        Variable(torch.Tensor([-.5+t, .5+t])).type(dtype).view(-1,1) # x_i
    )
    return _data_attachment(Mu, Nu, params)[0].data.cpu().numpy()[0]


T = np.linspace(-6,6,1201)

costs_kernel = np.array( [translate_cost(params_kernel, t) for t in T] )
costs_ot     = np.array( [translate_cost(params_ot    , t) for t in T] )

if False : # This example is so simple that we converge in one iteration...
	params_ot['nits'] = 1
	costs_ot_1   = np.array( [translate_cost(params_ot    , t) for t in T] )
	params_ot['nits'] = 2
	costs_ot_2   = np.array( [translate_cost(params_ot    , t) for t in T] )

params_ot['nits'] = 100
params_ot['rho']  = .5
costs_ot_rho_05   = np.array( [translate_cost(params_ot    , t) for t in T] )
params_ot['rho']  = 1.
costs_ot_rho_1    = np.array( [translate_cost(params_ot    , t) for t in T] )
params_ot['rho']  = 2.
costs_ot_rho_2    = np.array( [translate_cost(params_ot    , t) for t in T] )


#plt.ion()

plt.figure()
plt.plot(T, costs_kernel, label = 'Kernel distance')
plt.plot(T, costs_ot,     label='OT, $\\rho = +\\infty$')
#plt.plot(T, costs_ot_1,   label='One iteration')
#plt.plot(T, costs_ot_2,   label='Two iterations')
plt.plot(T, costs_ot_rho_2, label='OT, $\\rho=2$')
plt.plot(T, costs_ot_rho_1, label='OT, $\\rho=1$')
plt.plot(T, costs_ot_rho_05, label='OT, $\\rho=.5$')
plt.axis([-6,6,0,10])
plt.gca().set_aspect('equal', adjustable='box')

plt.legend(loc='upper right')
plt.draw()
from matplotlib2tikz import save as tikz_save
tikz_save(FOLDER+'/output/unbalanced/curve_unbalanced.tex', figurewidth='12cm', figureheight='12cm')
plt.show() 














