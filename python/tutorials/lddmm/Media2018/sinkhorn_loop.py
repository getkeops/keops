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


from matplotlib2tikz import save as tikz_save
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

Mu = (Mu[0]/Mu[0].sum(), Mu[1])
Nu = (Nu[0]/Nu[0].sum(), Nu[1])


params_att = {
	"formula"            : "wasserstein",

	# Parameters for OT:
	"cost"               : "dual",
	"features"           : "none",
	"kernel" : {"id"     : Kernel("gaussian(x,y)") ,
				"gamma"  : G ,
				"backend": backend                 },
	"epsilon"            : eps,
	"rho"                : -1.,            # < 0 -> no unbalanced transport
	"tau"                : 0.,              # Using acceleration with nits < ~40 leads to *very* unstable transport plans
	"nits"               : 100,
	"tol"                : 0.,
	"transport_plan"     : "full",
}


# Since this test is only going to be done for the paper,
# I don't wan't to break the code's architecture...
# Let's run the algorithm plenty of times...
def cost(formula, nits, tau=0., s_att= .01) :
	params_att["cost"] = formula
	params_att["nits"] = nits
	params_att["tau"]  = tau

	eps   = scal_to_var(s_att**2)
	G     = 1/eps
	params_att["kernel"]["gamma"]   = G
	params_att["kernel"]["epsilon"] = eps
	return _data_attachment(Mu, Nu, params_att)[0].data.cpu().numpy()[0]

iters = [ i for i in range(1,51)]

if False :
	# CONVERGENCE OF THE SINKHORN ALGORITHM
	plt.figure()

	primal_costs = np.array( [cost("primal", n, s_att=.1) for n in iters ]  )
	dual_costs   = np.array( [cost("dual",   n, s_att=.1) for n in iters ]  )

	plt.plot(iters, primal_costs, label="Primal cost")
	plt.plot(iters, dual_costs,   label="Dual cost")
	#ax1.set_xlabel('Sinkhorn iterations')
	#ax1.set_ylabel('Primal cost', color='b')
	#ax1.tick_params('y', colors='b')

	#ax2 = ax1.twinx()
	#ax2.plot(iters, dual_costs, 'r')
	#ax2.set_ylabel('Dual cost', color='r')
	#ax2.tick_params('y', colors='r')

	plt.legend(loc='upper right')

	plt.xlabel("Sinkhorn iterations")
	plt.ylabel("Optimal Transport cost")
	plt.xlim(1, 50)
	plt.draw()

	tikz_save(FOLDER+'/output/sinkhorn/convergence_primal_dual.tex', figurewidth='12cm', figureheight='12cm')


if False :
	# INFLUENCE OF EPSILON ON THE CONVERGENCE SPEED
	plt.figure()
	for (scale_att,label) in [(0.5, "0.5"), (0.4, "0.4"), (0.3, "0.3"), (0.2, "0.2"), (0.1, "0.1"), (0.05, "0.05")] :
		primal_costs = np.array( [cost("primal", n, s_att=scale_att) for n in iters ]  )
		dual_costs   = np.array( [cost("dual",   n, s_att=scale_att) for n in iters ]  )
		plt.plot(iters, np.maximum(primal_costs - dual_costs, 1e-10*np.ones( len(iters)) ), 
		         label = "$\\sqrt{\\varepsilon} = "+label+"$")
		print(primal_costs[-1],  dual_costs[-1])

	plt.xlabel("Sinkhorn iterations")
	plt.xlim(1, 50)
	plt.ylabel("Duality gap")
	plt.semilogy()
	plt.legend(loc='upper right')
	plt.draw()

	tikz_save(FOLDER+'/output/sinkhorn/influence_of_epsilon.tex', figurewidth='12cm', figureheight='12cm')

if False :
	# INFLUENCE OF TAU ON THE CONVERGENCE SPEED
	params_att["rho"] = 10 # this figure will come after the unbalanced transport, so we may as well use it...
	plt.figure()
	for (tau,label) in [ (.5, "+0.5"), (0., "\\,\\,0"), (-.3, "-0.3"), (-.5, "-0.5"), (-.8, "-0.8") ] :
		primal_costs = np.array( [cost("primal", n, tau=tau, s_att=.1) for n in iters ]  )
		dual_costs   = np.array( [cost("dual",   n, tau=tau, s_att=.1) for n in iters ]  )
		plt.plot(iters, primal_costs - dual_costs, label = "$\\tau = "+label+"$")
		print(primal_costs[-1],  dual_costs[-1])

	plt.xlabel("Sinkhorn iterations")
	plt.xlim(1, 50)
	plt.ylabel("Duality gap")
	plt.semilogy()
	plt.legend(loc='upper right')
	plt.draw()

	tikz_save(FOLDER+'/output/sinkhorn/influence_of_tau.tex', figurewidth='12cm', figureheight='12cm')


plt.show()
