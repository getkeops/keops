
import os
import numpy as np
import torch
from   torch.autograd import Variable
# Display routines :
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from   matplotlib.collections  import LineCollection

from pyvtk import PolyData, PointData, Scalars, VtkData


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



class Curve :
    "Encodes a 2D/3D curve as an array of float coordinates + a connectivity list."
    def __init__(self, points, connectivity, values=None) :
        """
        Creates a curve object from explicit numerical values.

        Args:
            points       ( (N,D) torch Variable) : the vertices of the curve
            connectivity ( (M,2) torch Variable) : connectivity matrix : one line = one segment.
                                                   Its type should either be torch.LongTensor
                                                   or torch.cuda.LongTensor, depending on points.type().
            values       ( (M,E) torch Variable) : a signal of arbitrary dimension, 
                                                   supported by the segments of the curve.
        """ 
        self.points       = points
        self.connectivity = connectivity
        self.values       = values
    
    @staticmethod
    def from_file(fname, *args, dim=2, **kwargs) :
        """
        Creates a curve object from a filename, either a ".png" or a ".vtk".
        N.B.: By default, curves are assumed to be of dimension 2.
              If you're reading a 3D vtk file (say, tractography fibers),
              please set "dim=3" when calling this method !
        """

        values = None 
        if   fname[-4:] == '.png' :
            (points, connec) = level_curves(fname, *args, **kwargs)
        elif fname[-4:] == '.vtk' :
            data = VtkData(fname)
            points = np.array(data.structure.points)[:,0:dim]
            connec = np.array(data.structure.polygons)
            try :
                values = np.array( data.point_data.data[0].scalars )
                values = Variable(torch.from_numpy( values ).view(-1,1) ).type(dtype)
            except :
                values = None 
        else :
            raise NotImplementedError('Filetype not supported : "'+str(fname)+'". ' \
                                      'Please load either ".vtk" or ".png" files.')
        
        # Convert the convenient numpy arrays to efficient torch tensors, and build the Curve object:
        points = Variable(torch.from_numpy( points ), requires_grad=True).type(dtype)
        connec = Variable(torch.from_numpy( connec )).type(dtypeint)
        return Curve( points, connec, values=values) 

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
    
    def to_varifold(self) :
        """
        """
        a,b = self.to_segments()
        u          =    (a-b)
        lengths    =   (  u**2).sum(1).sqrt()
        centers    = .5*(a+b)
        directions =      u / (lengths.view(-1,1) + 1e-5)
        return lengths, (centers, directions)

    def to_fvarifold(self) :
        """
        """
        lengths, (centers, directions) = self.to_varifold()
        return lengths, (centers, directions, self.values)
    
    # Output routines -------------------------------------------------------------------------

    def plot(self, ax, color = 'rainbow', linewidth = 3) :
        "Simple display using a per-id color scheme."

        connectivity = self.connectivity.data.cpu().numpy()
        a,b = self.to_segments()   # Those torch variables...
        a   = a.data.cpu().numpy() # Should be converted back to
        b   = b.data.cpu().numpy() # a pyplot-friendly format.
        segs = [ [a_i,b_i] for (a_i,b_i) in zip(a,b)]

        if isinstance(color, str) and self.values is not None : # Plot the signal
            if color == 'rainbow' : color = 'hsv' # override the non-pyplot default value...
            values     = self.values.data.cpu().numpy()
            maxval     = abs(values).max()
            cNorm      = colors.Normalize(vmin=-maxval, vmax=maxval)
            scalarMap  = cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(color) )
            seg_colors = [ scalarMap.to_rgba( v ) for v in values ]
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
        if self.values is not None :
            values = PointData( Scalars(self.values.data.cpu().numpy()) )
            vtk    = VtkData(structure, values)
        else :
            vtk = VtkData(structure)
        fname = filename + ext ; os.makedirs(os.path.dirname(fname), exist_ok=True)
        vtk.tofile( fname )


# Surfaces ============================================================================================================


class Surface :
    "Encodes a 3D surface as an array of float coordinates + a connectivity list."
    def __init__(self, points, connectivity, values=None) :
        """
        Creates a curve object from explicit numerical values.

        Args:
            points       ( (N,D) torch Variable) : the vertices of the curve
            connectivity ( (M,2) torch Variable) : connectivity matrix : one line = one segment.
                                                   Its type should either be torch.LongTensor
                                                   or torch.cuda.LongTensor, depending on points.type().
            values       ( (M,E) torch Variable) : a signal of arbitrary dimension, 
                                                   supported by the segments of the curve.
        """ 
        self.points       = points
        self.connectivity = connectivity
        self.values       = values
    
    @staticmethod
    def from_file(fname, *args, **kwargs) :
        """
        Creates a curve object from a '.vtk' file.
        """
        values = None 
        if fname[-4:] == '.vtk' :
            data = VtkData(fname)
            points = np.array(data.structure.points)
            connec = np.array(data.structure.polygons)
            try :
                values = np.array( data.point_data.data[0].scalars )
                values = Variable(torch.from_numpy( values ).view(-1,1)).type(dtype)
            except :
                values = None 
        else :
            raise NotImplementedError('Filetype not supported : "'+str(fname)+'". ' \
                                      'Please load ".vtk" files.')
        
        # Convert the convenient numpy arrays to efficient torch tensors, and build the Curve object:
        points = Variable(torch.from_numpy( points ), requires_grad=True).type(dtype)
        connec = Variable(torch.from_numpy( connec )).type(dtypeint)
        return Surface( points, connec, values=values) 

    def to_triangles(self) :
        return self.points[self.connectivity[:,0],:], self.points[self.connectivity[:,1],:], self.points[self.connectivity[:,2],:]

    def to_measure(self) :
        """
        Outputs the sum-of-diracs measure associated to the curve.
        Each triangle from the connectivity matrix self.connectivity
        is represented as a weighted dirac located at its center,
        with weight equal to the triangle area.
        """
        a,b,c   = self.to_triangles()
        u  = b-a ; v = c-a
        ux = u[:,0].view(-1,1) ; uy = u[:,1].view(-1,1) ; uz = u[:,2].view(-1,1)
        vx = v[:,0].view(-1,1) ; vy = v[:,1].view(-1,1) ; vz = v[:,2].view(-1,1)
        normals   = .5 * torch.cat( (uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx), dim=1 ).contiguous()
        areas     = (normals**2).sum(1).sqrt()
        centers =   (a+b+c)/3
        return areas, centers
    
    def to_varifold(self) :
        """
        """
        a,b,c = self.to_triangles()
        u  = b-a ; v = c-a

        ux = u[:,0] ; uy = u[:,1] ; uz = u[:,2]
        vx = v[:,0] ; vy = v[:,1] ; vz = v[:,2]
        normals   = .5 * torch.stack( (uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx) ).t().contiguous()
        print(normals.size())
        areas     =  (normals**2).sum(1).sqrt()
        centers   =  (a+b+c)/3
        normals_u =    normals / (areas.view(-1,1) + 1e-5)
        return areas, (centers, normals_u)

    def to_fvarifold(self) :
        """
        """
        areas, (centers, normals_u) = self.to_varifold()
        return areas, (centers, normals_u, self.values)
    
    # Output routines -------------------------------------------------------------------------

    def plot(self, ax, color = 'rainbow', linewidth = 3) :
        raise Warning("3D plot is not supported. Please use vtk export + paraview !")
        
    def save(self, filename, ext = ".vtk") :
        structure = PolyData(points  =      self.points.data.cpu().numpy(),
                             polygons=self.connectivity.data.cpu().numpy())

        if self.values is not None :
            values = PointData( Scalars(self.values.data.cpu().numpy()) )
            vtk    = VtkData(structure, values)
        else :
            vtk = VtkData(structure)

        fname = filename + ext ; os.makedirs(os.path.dirname(fname), exist_ok=True)
        vtk.tofile( fname )


