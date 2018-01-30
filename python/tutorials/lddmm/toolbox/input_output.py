import numpy as np
import torch
from   torch.autograd import Variable
# Display routines :
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from   matplotlib.collections  import LineCollection

from .       import shapes
from .shapes import Curve, Surface

# Pyplot Output =================================================================================

def new_grid(grid_ticks) :
	"Returns a standard grid, given as a curve."

	# x_l denotes the coordinates at which lines start, whereas x_d is used to "supsample" them.
	x_l = [np.linspace(min_r, max_r, nlines          ) for (min_r,max_r,nlines) in grid_ticks]
	x_d = [np.linspace(min_r, max_r, (nlines-1)*4 + 1) for (min_r,max_r,nlines) in grid_ticks]
	
	v = [] ; c = [] ; i = 0

	if   len(grid_ticks) == 2 :
		for x in x_l[0] :                    # One vertical line per x :
			v += [ [x, y] for y in x_d[1] ]  # Add points to the list of vertices.
			c += [ [i+j,i+j+1] for j in range( len(x_d[1])-1 ) ] # + appropriate connectivity
			i += len(x_d[1])
		for y in x_l[1] :                    # One horizontal line per y :
			v += [ [x, y] for x in x_d[0] ]  # Add points to the list of vertices.
			c += [ [i+j,i+j+1] for j in range( len(x_d[0])-1)] # + appropriate connectivity
			i += len(x_d[0])
	elif len(grid_ticks) == 3 :
		for y in x_l[1] :
			for z in x_l[2] :
				v += [ [x, y, z] for x in x_d[0] ]  # Add points to the list of vertices.
				c += [ [i+j,i+j+1] for j in range( len(x_d[0])-1 ) ] # + appropriate connectivity
				i += len(x_d[0])
		for x in x_l[0] :
			for z in x_l[2] :
				v += [ [x, y, z] for y in x_d[1] ]  # Add points to the list of vertices.
				c += [ [i+j,i+j+1] for j in range( len(x_d[1])-1 ) ] # + appropriate connectivity
				i += len(x_d[1])
		for x in x_l[0] :
			for y in x_l[1] :
				v += [ [x, y, z] for z in x_d[2] ]  # Add points to the list of vertices.
				c += [ [i+j,i+j+1] for j in range( len(x_d[2])-1 ) ] # + appropriate connectivity
				i += len(x_d[2])
	else :
		raise NotImplementedError("For the sake of simplicity, we only implemented the generation of 2D/3D grids.")
	
	points = Variable(torch.Tensor( v ), requires_grad=True).type(shapes.dtype)
	connec = Variable(torch.Tensor( c )                    ).type(shapes.dtypeint)

	return Curve( points, connec )

def save_momentum(filename, q, p, q_mu=None) :
	None # I've forgotten the best way to store this...


def save_info(filename, model, target, info, params_att) :
	if info is not None :
		formula = params_att["formula"]

		if isinstance(info,(Curve, Surface)) :
			info.save(filename)
		elif isinstance(info, tuple) and isinstance(info[0], (Curve,Surface)) :
			info[0].save(filename+"_mu2nu")
			info[1].save(filename+"_nu2mu")

		elif formula == "kernel" :
			"""ax.imshow(info, interpolation='bilinear', origin='lower', 
					vmin = -scale_attach, vmax = scale_attach, cmap=cm.RdBu, 
					extent=(0,1, 0, 1)) """
			None
		elif formula in ["likelihood_symmetric", "likelihood_source_wrt_target", "likelihood_target_wrt_source"] :
			None
		else :
			raise NotImplementedError('I can\'t save the "info" provided by the attachment formula "'+formula+'". '\
			                         +'So far, only shapes output have been implemented.')



