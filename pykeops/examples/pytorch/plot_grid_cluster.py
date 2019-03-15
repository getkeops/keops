"""
Grid clustering
================

This script showcases the use of the optional 'ranges' argument
to compute block-sparse kernel products.

"""

#############################
#  Standard imports
#

import time
import numpy as np
import torch
from matplotlib import pyplot as plt

from pykeops.torch import Genred
from pykeops.torch.cluster import grid_cluster, sort_clusters, from_matrix, cluster_ranges_centroids

nump = lambda t : t.cpu().numpy()
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
   
#####################
# Define our dataset
#
M, N = (100000, 100000) if use_cuda else (2000, 2000)

t = torch.linspace(0, 2 * np.pi, N + 1)[:-1]
x = torch.stack((.4 + .4 * (t / 7) * t.cos(), .5 + .3 * t.sin()), 1)
x = x + .02 * torch.randn(x.shape)
x = x.type(dtype)

y = torch.randn(N,2).type(dtype)
y = y/10 + dtype([.6,.6])

####################################################################
# Voxelization 
#

if use_cuda : torch.cuda.synchronize()
    
eps = .05

Start  = time.time()
start  = time.time()
x_labels = grid_cluster(x, eps)  # class labels
y_labels = grid_cluster(y, eps)  # class labels
if use_cuda : torch.cuda.synchronize()
end = time.time()
print("Perform clustering       : {:.4f}s".format(end-start))

# Compute one range and centroid per class
start = time.time()
x_ranges, x_centroids, _  = cluster_ranges_centroids(x, x_labels)
y_ranges, y_centroids, _  = cluster_ranges_centroids(y, y_labels)
if use_cuda : torch.cuda.synchronize()
end = time.time()
print("Compute ranges+centroids : {:.4f}s".format(end-start))

start = time.time()
x, x_labels = sort_clusters(x, x_labels)
y, y_labels = sort_clusters(y, y_labels)
if use_cuda : torch.cuda.synchronize()
end = time.time()
print("Sort the points          : {:.4f}s".format(end-start))

####################################################################
# Interaction threshold
#

sigma = .05

start = time.time()
D = ((x_centroids[:,None,:] - y_centroids[None,:,:])**2).sum(2)
keep = D < (4*sigma)**2
ranges_ij = from_matrix(x_ranges, y_ranges, keep)
if use_cuda : torch.cuda.synchronize()
end = time.time()
print("Process the ranges       : {:.4f}s".format(end-start))

if use_cuda : torch.cuda.synchronize()
End = time.time()
t_cluster = End-Start
print("Total time (synchronized): {:.4f}s".format(End-Start))
print("")

areas = (x_ranges[:,1]-x_ranges[:,0])[:,None] \
      * (y_ranges[:,1]-y_ranges[:,0])[None,:]
total_area  = areas.sum().item() # should be equal to N*M
sparse_area = areas[keep].sum().item()
print("We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
    sparse_area, total_area, int(100*sparse_area/total_area) ))
print("")

####################################################################
# Benchmark Gaussian convolution
#

g = torch.Tensor( [1/(2*sigma**2)] ).type(dtype)
b = torch.randn(N, 1).type(dtype)

my_conv = Genred( "Exp(-G*SqDist(X,Y)) * B",
                 ["G = Pm(1)",
                  "X = Vx(2)",
                  "Y = Vy(2)",
                  "B = Vy(1)"], 
                  axis = 1 )     # Reduction wrt. y

backends = (["CPU", "GPU"] if M*N<4e8 else ["GPU"]) if use_cuda else ["CPU"]
for backend in backends :
    if backend == "CPU" : 
        g_, x_, y_, b_ = g.cpu(), x.cpu(), y.cpu(), b.cpu()
        ranges_ij_ = tuple(r.cpu() for r in ranges_ij)
    else :                
        g_, x_, y_, b_ = g, x, y, b
        ranges_ij_ = ranges_ij
    
    # Warm-up
    a = my_conv(g_, x_, y_, b_, backend=backend)

    start = time.time()
    a_full = my_conv(g_, x_, y_, b_, backend=backend)
    end = time.time()
    t_full = end-start
    print(" Full  convolution, {} backend: {:2.4f}s".format(backend, end-start))

    start = time.time()
    a_sparse = my_conv(g_, x_, y_, b_, backend=backend, ranges=ranges_ij_ )
    end = time.time()
    t_sparse = end-start
    print("Sparse convolution, {} backend: {:2.4f}s".format(backend, end-start) )
    print("Relative time : {:3d}% ({:3d}% including clustering), ".format(
        int(100*t_sparse/t_full),
        int(100*(t_sparse+t_cluster)/t_full)))
    print("Relative error:   {:3.4f}%".format( 100* (a_sparse-a_full).abs().sum() / a_full.abs().sum() ))
    print("")

####################################################################
# Display 
#
if M + N <= 500000 :
    clust_i = 75
    ranges_i, slices_j, redranges_j = ranges_ij[0:3]
    start_i, end_i = ranges_i[clust_i]
    start,end = slices_j[clust_i-1], slices_j[clust_i]

    #print(redranges_j[start:end])
    keep = nump(keep.float())
    keep[clust_i] += 2

    plt.ion()
    plt.matshow(keep)

    plt.figure(figsize=(10,10))

    x, x_labels, x_centroids = nump(x), nump(x_labels), nump(x_centroids)
    y, y_labels, y_centroids = nump(y), nump(y_labels), nump(y_centroids)

    plt.scatter(x[:,0],x[:,1],c=x_labels,cmap=plt.cm.Wistia, s= 25*500 / len(x))
    plt.scatter(y[:,0],y[:,1],c=y_labels,cmap=plt.cm.winter, s= 25*500 / len(y))

    # Target clusters
    for start_j,end_j in redranges_j[start:end] :
        plt.scatter(y[start_j:end_j,0],y[start_j:end_j,1],c="magenta",s= 50*500 / len(y))

    # Source cluster
    plt.scatter(x[start_i:end_i,0],x[start_i:end_i,1],c="cyan",s=10)


    plt.scatter(x_centroids[:,0],x_centroids[:,1],c="black",s=10,alpha=.5)


    print("Close the figure to continue.")
    plt.axis("equal") ; plt.axis([0,1,0,1])
    plt.tight_layout()
    plt.show(block=(__name__=="__main__"))

    print("Done.")
