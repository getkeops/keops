# Import the relevant tools
import numpy as np          # standard array library

# Display routines :
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Input/Output routines =========================================================================
# from '.png' to level curves  ------------------------------------------------------------------
from skimage.measure import find_contours
from scipy import misc
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d

def arclength_param(line) :
	"Arclength parametrisation of a piecewise affine curve."
	vel = line[1:, :] - line[:-1, :]
	vel = np.sqrt(np.sum( vel ** 2, 1 ))
	return np.hstack( ( [0], np.cumsum( vel, 0 ) ) )
def arclength(line) :
	"Total length of a piecewise affine curve."
	return arclength_param(line)[-1]
	
def resample(line, npoints) :
	"Resamples a curve by arclength through linear interpolation."
	s = arclength_param(line)
	f = interp1d(s, line, kind = 'linear', axis = 0, assume_sorted = True)
	
	p = f( np.linspace(0, s[-1], npoints) )
	connec = np.vstack( (np.arange(0, len(p) - 1), 
						 np.arange(1, len(p)    )) ).T
	if np.array_equal(p[0], p[-1]) : # i.e. p is a loop
		p = p[:-1]
		connec = np.vstack( (connec[:-1,:],  [len(p)-1, 0]) )
	return (p, connec)

def level_curves(fname, npoints = 200, smoothing = 10, level = 0.5) :
	"Loads regularly sampled curves from a .PNG image."
	# Find the contour lines
	img = misc.imread(fname, flatten = True) # Grayscale
	img = (img.T[:, ::-1])  / 255.
	img = gaussian_filter(img, smoothing, mode='nearest')
	lines = find_contours(img, level)
	
	# Compute the sampling ratio for every contour line
	lengths = np.array( [arclength(line) for line in lines] )
	points_per_line = np.ceil( npoints * lengths / np.sum(lengths) )
	
	# Interpolate accordingly
	points = [] ; connec = [] ; index_offset = 0
	for ppl, line in zip(points_per_line, lines) :
		(p, c) = resample(line, ppl)
		points.append(p)
		connec.append(c + index_offset)
		index_offset += len(p)
	
	size   = np.maximum(img.shape[0], img.shape[1])
	points = np.vstack(points) / size
	connec = np.vstack(connec)
	return (points, connec)
# Pyplot Output =================================================================================

XMIN, XMAX = -2, 2
YMIN, YMAX = -2, 2
NLINES     = 41


def GridData() :
	"Returns the coordinates and connectivity of the grid carried along by a deformation."
	nlines = NLINES ; ranges = [ (XMIN, XMAX), (YMIN, YMAX) ] # one square = (.1,.1)
	np_per_lines = (nlines-1) * 4 + 1       # Supsample lines to get smooth figures
	x_l = [np.linspace(min_r, max_r, nlines      ) for (min_r,max_r) in ranges]
	x_d = [np.linspace(min_r, max_r, np_per_lines) for (min_r,max_r) in ranges]
	
	v = [] ; c = [] ; i = 0
	for x in x_l[0] :                    # One vertical line per x :
		v += [ [x, y] for y in x_d[1] ]  # Add points to the list of vertices.
		c += [ [i+j,i+j+1] for j in range(np_per_lines-1)] # + appropriate connectivity
		i += np_per_lines
	for y in x_l[1] :                    # One horizontal line per y :
		v += [ [x, y] for x in x_d[1] ]  # Add points to the list of vertices.
		c += [ [i+j,i+j+1] for j in range(np_per_lines-1)] # + appropriate connectivity
		i += np_per_lines
	
	return ( np.vstack(v), np.vstack(c) ) # (vertices, connectivity)
	
def ShowTransport( Q, Xt, Gamma, ax ) :
	"Displays a transport plan."
	from curve import Curve
	points = [] ; connectivity = [] ; curr_id = 0
	Q_points,Q_weights = Q.to_measure()  ;  xtpoints = Xt.points # Extract the centers + areas
	for (a, mui, gi) in zip(Q_points, Q_weights, Gamma) :
		gi = gi / mui # gi[j] = fraction of the mass from "a" which goes to xtpoints[j]
		for (seg, gij) in zip(Xt.connectivity, gi) :
			mass_per_line = 0.05
			if gij >= mass_per_line :
				nlines = np.floor(gij / mass_per_line)
				ts     = np.linspace(.35, .65, nlines)
				for t in ts :
					b = (1-t) * xtpoints[seg[0]] + t * xtpoints[seg[1]]
					points += [a, b]; connectivity += [[curr_id, curr_id + 1]]; curr_id += 2
	if len(connectivity) > 0 :
		Plan = Curve(np.vstack(points), np.vstack(connectivity))
		Plan.plot(ax, color = (.6,.8,1.), linewidth = 1)

def DisplayShoot(Q0, G0, p0, Q1, G1, Xt, info, it, scale_momentum, scale_attach, attach_mode) :
	"Displays a pyplot Figure and save it."
	# Figure at "t = 0" : -----------------------------------------------------------------------
	fig = plt.figure(1, figsize = (10,10), dpi=100); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	
	G0.plot(ax, color = (.8,.8,.8), linewidth = 1)
	Xt.plot(ax, color = (.85, .6, 1.))
	Q0.plot(ax)
	
	ax.quiver( Q0.points[:,0], Q0.points[:,1], p0[:,0], p0[:,1], 
	           scale = scale_momentum, color='blue')
	
	ax.axis([XMIN, XMAX, YMIN, YMAX]) ; ax.set_aspect('equal') ; plt.draw() ; plt.pause(0.001)
	fig.savefig( 'output/momentum_' + str(it) + '.png' )
	
	# Figure at "t = 1" : -----------------------------------------------------------------------
	fig = plt.figure(2, figsize = (10,10), dpi=100); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	if attach_mode == "OT" : # Convenient way of saying that we're using a transport plan.
		ShowTransport( Q1, Xt, info, ax)
	elif attach_mode == "kernel" : # Otherwise, it's a kernel matching term.
		ax.imshow(info, interpolation='bilinear', origin='lower', 
				vmin = -scale_attach, vmax = scale_attach, cmap=cm.RdBu, 
				extent=(0,1, 0, 1)) 
	G1.plot(ax, color = (.8,.8,.8), linewidth = 1)
	Xt.plot(ax, color = (.76, .29, 1.))
	Q1.plot(ax)
	
	ax.axis([XMIN, XMAX, YMIN, YMAX]) ; ax.set_aspect('equal') ; plt.draw() ; plt.pause(0.001)
	fig.savefig( 'output/model_' + str(it) + '.png' )

def DisplayTrajectory(Qts, P0 = None, scale_momentum = 1, Xt = None, it = 0) :
	"""
	Displays all steps of a trajectory [Q0, ...] from Q0 to Q1 on the same image, 
	plus target Xt if requested.
	"""
	
	fig = plt.figure(3, figsize = (10,10), dpi=100); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	
	if Xt is not None :
		Xt.plot(ax, color = (.85, .6, 1.))
	
	for (t, Qt) in enumerate(Qts) :
		if t == 0 or t == len(Qts) - 1 :
			Qt.plot(ax, linewidth = 3)
		else :
			Qt.plot(ax, linewidth = 1)
	
	if P0 is not None and len(Qts) > 0 :
		Q0 = Qts[0]
		ax.quiver( Q0.points[:,0], Q0.points[:,1], P0[:,0], P0[:,1], 
			scale = scale_momentum, color='blue')
	
	ax.axis([XMIN, XMAX, YMIN, YMAX]) ; ax.set_aspect('equal') ; plt.draw() ; plt.pause(0.001)
	fig.savefig( 'output/trajectory_' + str(it) + '.png' )
	
	
	
	
	
	
	























