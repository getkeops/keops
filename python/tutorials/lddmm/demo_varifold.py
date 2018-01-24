# LDDMM registration using PyTorch
# Author : Jean Feydy

# Note about function names :
#  - MyFunction  is an input-output code
#  - my_routine  is a numpy routine
#  - _my_formula is a PyTorch symbolic function

# Import the relevant tools
import torch
from   torch          import Tensor
from   torch.autograd import Variable

import os, sys
FOLDER = os.path.dirname(os.path.abspath(__file__))+os.path.sep

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +(os.path.sep+'..')*4)
from libkp.torch.kernels    import Kernel

from .toolbox               import shapes
from .toolbox.shapes        import Curve, Surface
from .toolbox.matching      import GeodesicMatching
from .toolbox.model_fitting import FitModel


import matplotlib.pyplot as plt
plt.ion()
plt.show()

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

# Make sure that everybody's on the same wavelength:
shapes.dtype = dtype ; shapes.dtypeint = dtypeint

if False :
	Source = Curve.from_file(FOLDER+"data/amoeba_1.png", npoints=200)
	Target = Curve.from_file(FOLDER+"data/amoeba_2.png", npoints=200)
else :
	Source = Surface.from_file(FOLDER+"data/venus_1.vtk")
	Target = Surface.from_file(FOLDER+"data/venus_4.vtk")


s_def = .15
s_att = .05
backend = "auto"
def scal_to_var(x) :
	return Variable(Tensor([x])).type(dtype)

params = {
	"weight_regularization" : .1,
	"weight_data_attachment": 1.,

	"deformation_model" : {
		"id"         : Kernel("exponential(x,y)"),
		"gamma"      : scal_to_var(1/s_def**2),
		"backend"    : backend,
		"normalize"  : False,
	},

	"data_attachment"   : {
		"formula"            : "kernel",
		"features"           : "locations+directions",
		"id"         : Kernel("energy(x,y) * linear(u,v)**2"),
		"gamma"      : (scal_to_var(1/s_att**2),None),
		"backend"    : backend,
	},
	"save" : {
		"output_directory"   : FOLDER+"output/",
	}
}

# Define our (simplistic) matching model
Model = GeodesicMatching(Source)

# Train it
FitModel(params, Model, Target)

# That's it :-)
