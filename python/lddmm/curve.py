# Import the relevant tools
import numpy as np          # standard array library
import torch

# Display routines :
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections  import LineCollection
from pyvtk import VtkData  # from '.vtk' to Curves objects

from input_output import level_curves

# Curve representations =========================================================================

class Curve :
	"Encodes a 2D curve as an array of float coordinates + a connectivity list."
	def __init__(self, points, connectivity) :
		"points should be a n-by-2 float array, connectivity an nsegments-by-2 int array." 
		self.points       = points
		self.connectivity = connectivity
	
	def segments(self) :
		"Returns the list of segments the curve is made of."
		return np.array( [  [self.points[l[0]], self.points[l[1]]] for l in self.connectivity ] )
		
	def to_measure(self) :
		"""
		Outputs the sum-of-diracs measure associated to the curve.
		Each segment from the connectivity matrix self.c
		is represented as a weighted dirac located at its center,
		with weight equal to the segment length.
		"""
		segments = self.segments()
		centers = [         .5 * (  seg[0] + seg[1]      ) for seg in segments ]
		lengths = [np.sqrt(np.sum( (seg[1] - seg[0])**2 )) for seg in segments ]
		return ( np.array(centers), np.array(lengths) )
	
	@staticmethod
	def _vertices_to_measure( q, connec ) :
		"""
		Transforms a torch array 'q1' into a measure, assuming a connectivity matrix connec.
		It is the Torch equivalent of 'to_measure'.
		"""
		a = q[connec[:,0]] ; b = q[connec[:,1]]
		# A curve is represented as a sum of diracs, one for each segment
		x  = .5 * (a + b)                     # Mean
		mu = torch.sqrt( ((b-a)**2).sum(1) )  # Length
		return (x, mu)
		
	def plot(self, ax, color = 'rainbow', linewidth = 3) :
		"Simple display using a per-id color scheme."
		
		if self.connectivity.ndim > 1 : # Real curve
			segs = self.segments()
			
			if color == 'rainbow' :   # rainbow color scheme to see pointwise displacements
				ncycles    = 5
				cNorm      = colors.Normalize(vmin=0, vmax=(len(segs)-1)/ncycles)
				scalarMap  = cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('hsv') )
				seg_colors = [ scalarMap.to_rgba( i % ((len(segs)-1)/ncycles) ) 
							   for i in range(len(segs)) ]
			else :                    # uniform color
				seg_colors = [ color for i in range(len(segs)) ] 
			
			line_segments = LineCollection(segs, linewidths=(linewidth,), 
										   colors=seg_colors, linestyle='solid')
			ax.add_collection(line_segments)
			
		else : # Landmarks
			if color == 'rainbow' :   # rainbow color scheme to see pointwise displacements
				ncycles    = 5
				cNorm      = colors.Normalize(vmin=0, vmax=(len(self.connectivity)-1)/ncycles)
				scalarMap  = cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('hsv') )
				dot_colors = [ scalarMap.to_rgba( i % ((len(self.connectivity)-1)/ncycles) ) 
							   for i in range(len(self.connectivity)) ]
			else :                    # uniform color
				dot_colors = [ color for i in range(len(self.connectivity)) ] 
			
			ax.scatter(self.points[:,0], self.points[:,1], 
			           s = 9*(linewidth**2)*self.connectivity,
			           c = dot_colors)
			
	@staticmethod
	def from_file(fname) :
		if   fname[-4:] == '.png' :
			(points, connec) = level_curves(fname)
			return Curve(points, connec)
		elif fname[-4:] == '.vtk' :
			data = VtkData(fname)
			points = np.array(data.structure.points)[:,0:2] # Discard "Z"
			connec = np.array(data.structure.polygons)
			return Curve((points + 150)/300, connec) # offset for the skull dataset...
			

