
import os
import numpy as np
import torch
from   torch.autograd import Variable
# Display routines :
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from   matplotlib.collections  import LineCollection


use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor


# Curve representations =========================================================================

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



# from '.vtk' to Curves objects  ----------------------------------------------------------------
from pyvtk import PolyData, VtkData

class Curve :
    "Encodes a 2D curve as an array of float coordinates + a connectivity list."
    def __init__(self, points, connectivity) :
        """
        Creates a curve object from explicit numerical values.

        Args:
            points       ( (N,D) torch Variable) : the vertices of the curve
            connectivity ( (N,2) torch Variable) : connectivity matrix.
                                                   Its type should either be torch.LongTensor
                                                   or torch.cuda.LongTensor, depending on points.type().
        """ 
        self.points       = points
        self.connectivity = connectivity
    
    @staticmethod
    def from_file(fname, *args, dim=2, **kwargs) :
        """
        Creates a curve object from a filename, either a ".png" or a ".vtk".
        """
        if   fname[-4:] == '.png' :
            (points, connec) = level_curves(fname, *args, **kwargs)
        elif fname[-4:] == '.vtk' :
            data = VtkData(fname)
            points = np.array(data.structure.points)[:,0:dim]
            connec = np.array(data.structure.polygons)
        else :
            raise NotImplementedError('Filetype not supported : "'+str(fname)+'". ' \
                                      'Please load either ".vtk" or ".png" files.')
        
        # Convert the convenient numpy arrays to efficient torch tensors, and build the Curve object:
        points = Variable(torch.from_numpy( points ), requires_grad=True).type(dtype)
        connec = Variable(torch.from_numpy( connec )).type(dtypeint)
        return Curve( points, connec) 

    def to_segments(self) :
        return self.points[self.connectivity[:,0],:], self.points[self.connectivity[:,1],:]

    def to_measure(self) :
        """
        Outputs the sum-of-diracs measure associated to the curve.
        Each segment from the connectivity matrix self.c
        is represented as a weighted dirac located at its center,
        with weight equal to the segment length.
        """
        a,b = self.to_segments()
        lengths =   ((a-b)**2).sum(1).sqrt()
        centers = .5*(a+b)
        return lengths, centers
    
    def to_current(self) :
        """
        """
        a,b = self.to_segments()
        lengths    =   ((a-b)**2).sum(1).sqrt()
        centers    = .5*(a+b)
        directions =    (a-b) / (lengths + 1e-5)
        return lengths, (centers, directions)
    
    # Output routines -------------------------------------------------------------------------

    def plot(self, ax, color = 'rainbow', linewidth = 3) :
        "Simple display using a per-id color scheme."

        connectivity = self.connectivity.data.cpu().numpy()
        a,b = self.to_segments()   # Those torch variables...
        a   = a.data.cpu().numpy() # Should be converted back to
        b   = b.data.cpu().numpy() # a pyplot-friendly format.
        segs = [ [a_i,b_i] for (a_i,b_i) in zip(a,b)]

        if color == 'rainbow' :   # rainbow color scheme to see pointwise displacements
            ncycles    = 5
            cNorm      = colors.Normalize(vmin=0, vmax=(len(segs)-1)/ncycles)
            scalarMap  = cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('hsv') )
            seg_colors = [ scalarMap.to_rgba( i % ((len(a)-1)/ncycles) ) 
                           for i in range(len(segs)) ]
        else :                    # uniform color
            seg_colors = [ color for i in range(len(segs)) ] 
        
        line_segments = LineCollection(segs, linewidths=(linewidth,), 
                                       colors=seg_colors, linestyle='solid')
        ax.add_collection(line_segments)
            
        """
        # Landmarks, connectivity = weight
        if color == 'rainbow' :   # rainbow color scheme to see pointwise displacements
            ncycles    = 5
            cNorm      = colors.Normalize(vmin=0, vmax=(len(connectivity)-1)/ncycles)
            scalarMap  = cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('hsv') )
            dot_colors = [ scalarMap.to_rgba( i % ((len(connectivity)-1)/ncycles) ) 
                        for i in range(len(connectivity)) ]
        else :                    # uniform color
            dot_colors = [ color for i in range(len(connectivity)) ] 
        
        ax.scatter(self.points[:,0], self.points[:,1], 
                s = 4*9*(linewidth**2)*connectivity,
                c = dot_colors)
        """
        
    def save(self, filename, ext = ".vtk") :
        structure = PolyData(points  =      self.points.data.cpu().numpy(),
                             polygons=self.connectivity.data.cpu().numpy())
        vtk = VtkData(structure)
        fname = filename + ext ; os.makedirs(os.path.dirname(fname), exist_ok=True)
        vtk.tofile( fname )


