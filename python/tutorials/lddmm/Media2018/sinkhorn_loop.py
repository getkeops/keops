# LDDMM registration using PyTorch
# Author : Jean Feydy

# Note about function names :
#  - MyFunction  is an input-output code
#  - my_routine  is a numpy routine
#  - _my_formula is a PyTorch symbolic function

# Import the relevant tools
import time                 # to measure performance
import numpy as np          # standard array library
import torch
from   torch          import Tensor
from   torch.autograd import Variable

import os
FOLDER = os.path.dirname(os.path.abspath(__file__))+os.path.sep

from ..toolbox                import shapes
from ..toolbox.shapes         import Curve, Surface
from ..toolbox.matching       import GeodesicMatching
from ..toolbox.model_fitting  import FitModel
from ..toolbox.kernel_product import Kernel
from ..toolbox.data_attachment import _data_attachment

import matplotlib.pyplot as plt

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

# Make sure that everybody's on the same wavelength:
shapes.dtype = dtype ; shapes.dtypeint = dtypeint

Source = Curve.from_file(FOLDER+"data/amoeba_1.png", npoints=100)
Target = Curve.from_file(FOLDER+"data/amoeba_2.png", npoints=100)

def scal_to_var(x) :
	return Variable(Tensor([x])).type(dtype)

s_def = .1
s_att = .01
eps   = scal_to_var(s_att**2)
G     = 1/eps           # "gamma" of the gaussian
backend = "pytorch"




Mu = Source.to_measure()
Nu = Source.to_measure()

#Mu = (Mu[0]/Mu[0].sum(), Mu[1])
#Nu = (Nu[0]/Nu[0].sum(), Nu[1])


params_att = {
	"formula"            : "wasserstein",

	# Parameters for OT:
	"cost"               : "dual",
	"features"           : "none",
	"kernel" : {"id"     : Kernel("gaussian(x,y)") ,
				"gamma"  : G ,
				"backend": backend                 },
	"epsilon"            : eps,
	"rho"                : 10.,            # < 0 -> no unbalanced transport
	"tau"                : 0.,              # Using acceleration with nits < ~40 leads to *very* unstable transport plans
	"nits"               : 100,
	"tol"                : 1e-10,
	"transport_plan"     : "full",
}


# Since this test is only going to be done for the paper,
# I don't wan't to break the code's architecture...
# Let's run the algorithm plenty of times...
def cost(formula, nits, tau=0., s_att= .01) :
	params_att["cost"] = formula
	params_att["nits"] = nits
	params_att["tau"]  = 0.

	eps   = scal_to_var(s_att**2)
	G     = 1/eps
	params_att["kernel"]["gamma"]   = G
	params_att["kernel"]["epsilon"] = eps
	return _data_attachment(Mu, Nu, params_att)[0].data.cpu().numpy()[0]

primal_costs = np.array( [cost("primal", n, tau=0., s_att=.1) for n in range(1,50) ]  )
spring_costs = np.array( [cost("spring", n, tau=0., s_att=.1) for n in range(1,50) ]  )
dual_costs   = np.array( [cost("dual",   n, tau=0., s_att=.1) for n in range(1,50) ]  )

print(primal_costs[-1], spring_costs[-1],  dual_costs[-1])
plt.figure()

plt.plot(primal_costs)
plt.plot(spring_costs)
plt.plot(dual_costs)


plt.figure()

plt.plot(primal_costs - dual_costs)
plt.semilogy()
plt.show()
