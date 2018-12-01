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
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
   
#####################
# Define our dataset
#
M, N = 2000, 2000

t = torch.linspace(0, 2 * np.pi, N + 1)[:-1]
x = torch.stack((.5 + .4 * (t / 7) * t.cos(), .5 + .3 * t.sin()), 1)
x = x + .02 * torch.randn(x.shape)
x = x.type(dtype)

y = torch.randn(N,2).type(dtype)
y = y/10 + dtype([.7,.6])

####################################################################
# Voxelization 
#

eps = .05

Start  = time.time()
start  = time.time()
x_labels = grid_cluster(x, eps)  # class labels
y_labels = grid_cluster(y, eps)  # class labels
end = time.time()
print("Time to perform clustering:",round(end-start,5),"s")

# Compute one range and centroid per class
start = time.time()
x_ranges, x_centroids  = cluster_ranges_centroids(x, x_labels)
y_ranges, y_centroids  = cluster_ranges_centroids(y, y_labels)
end = time.time()
print("Time to compute auxiliary info:",round(end-start,5),"s")

start = time.time()
x, x_labels = sort_clusters(x, x_labels)
y, y_labels = sort_clusters(y, y_labels)
end = time.time()
print("Time to sort the points:",round(end-start,5),"s")

####################################################################
# Interaction threshold
#

sigma = .05

start = time.time()
D = ((x_centroids[:,None,:] - y_centroids[None,:,:])**2).sum(2)
keep = D < (4*sigma)**2
ranges_ij = from_matrix(x_ranges, y_ranges, keep)
end = time.time()
print("Time to process the ranges:",round(end-start,5),"s")
End = time.time()
print("Total time:",round(End-Start,5),"s")



####################################################################
# Display 
#
if M + N <= 5000 :
    clust_i = 54
    ranges_i, slices_j, redranges_j = ranges_ij[0:3]
    start_i, end_i = ranges_i[clust_i]
    start,end = slices_j[clust_i-1], slices_j[clust_i]

    print(redranges_j[start:end])
    keep = nump(keep.float())
    keep[clust_i] += 2

    plt.ion()
    plt.matshow(keep)

    plt.figure()

    x, x_labels, x_centroids = nump(x), nump(x_labels), nump(x_centroids)
    y, y_labels, y_centroids = nump(y), nump(y_labels), nump(y_centroids)

    plt.scatter(x[:,0],x[:,1],c=x_labels,cmap=plt.cm.Wistia, s=10)
    plt.scatter(y[:,0],y[:,1],c=y_labels,cmap=plt.cm.winter, s=10)

    # Target clusters
    for start_j,end_j in redranges_j[start:end] :
        plt.scatter(y[start_j:end_j,0],y[start_j:end_j,1],c="magenta",s=20)

    # Source cluster
    plt.scatter(x[start_i:end_i,0],x[start_i:end_i,1],c="cyan",s=40)


    plt.scatter(x_centroids[:,0],x_centroids[:,1],c="black",s=50,alpha=.5)


    print("Close the figure to continue.")
    plt.axis("equal")
    plt.show(block=(__name__=="__main__"))

    print("Done.")
