from pykeops.torch import LazyTensor
from pykeops.torch.cluster import cluster_ranges_centroids
from pykeops.torch.cluster import from_matrix

import torch

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.synchronize()

class ivf():
  def __init__(self,k=5):
    self.__c=None
    self.__k=k
    self.__x=None
    self.__keep=None
    self.__x_ranges=None
    self.__x_perm=None
    self.__y_perm=None
    self.__device=None

  def __KMeans(self,x, K=10, Niter=15):
      N, D = x.shape  
      c = x[:K, :].clone() 
      x_i = LazyTensor(x.view(N, 1, D).to(self.__device))  
      for i in range(Niter):
          c_j = LazyTensor(c.view(1, K, D).to(self.__device))  
          D_ij = ((x_i - c_j) ** 2).sum(-1)
          cl = D_ij.argmin(dim=1).long().view(-1)  
          c.zero_() 
          c.scatter_add_(0, cl[:, None].repeat(1, D), x) 
          Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
          c /= Ncl  
      return cl, c    

  def __k_argmin(self,x,y,k=1):
    if use_cuda:
        torch.cuda.synchronize()    
    x_LT=LazyTensor(x.unsqueeze(1).to(self.__device))
    y_LT=LazyTensor(y.unsqueeze(0).to(self.__device))
    d=((x_LT-y_LT)**2).sum(-1)
    if k==1:
      return d.argmin(dim=1).long().view(-1)
    else:
      return d.argKmin(K=k,dim=1).long()  

  def __sort_clusters(self,x,lab,store_x=True):
    lab, perm = torch.sort(lab.view(-1))
    if store_x:
      self.__x_perm=perm 
    else:
      self.__y_perm=perm
    return x[perm],lab

  def __unsort(self,nn):
    return torch.index_select(self.__x_perm[nn],0,self.__y_perm.argsort())

  def fit(self,x,clusters=50,a=5,n=15):
    '''
    Fits the main dataset
    '''
    if type(x)!=torch.Tensor:
      raise ValueError('Input must be a torch tensor')
    if type(clusters)!=int:
      raise ValueError('Clusters must be an integer')
    if clusters>=len(x):
      raise ValueError('Number of clusters must be less than length of dataset')
    if type(a)!=int:
      raise ValueError('Number of clusters to search over must be an integer')    
    if a>clusters:
      raise ValueError('Number of clusters to search over must be less than total number of clusters') 
    if len(x.shape)!=2:
      raise ValueError('Input must be a 2D array')    
    x=x.contiguous()
    self.__device=x.device
    cl, c = self.__KMeans(x,clusters,Niter=n)
    self.__c=c

    cl=self.__assign(x)
    if use_cuda:
        torch.cuda.synchronize()

    ncl=self.__k_argmin(c,c,k=a)
    self.__x_ranges, _, _ = cluster_ranges_centroids(x, cl)
    
    x, x_labels = self.__sort_clusters(x,cl,store_x=True)
    self.__x=x
    r=torch.arange(clusters).repeat(a,1).T.reshape(-1).long()
    self.__keep= torch.zeros([clusters,clusters], dtype=torch.bool).to(self.__device)  
    self.__keep[r,ncl.flatten()]=True    
    return self


  def __assign(self,x,c=None):
    if c is None:
      c=self.__c
    return self.__k_argmin(x,c)
    
  def kneighbors(self,y):
    '''
    Obtain the k nearest neighbors of the query dataset y
    '''
    if self.__x is None:
      raise ValueError('Input dataset not fitted yet! Call .fit() first!')
    if type(y)!=torch.Tensor:
      raise ValueError("Query dataset must be a torch tensor")
    if y.device!=self.__device:
      raise ValueError('Input dataset and query dataset must be on same device')
    if len(y.shape)!=2:
      raise ValueError('Query dataset must be a 2D tensor')
    if self.__x.shape[-1]!=y.shape[-1]:
      raise ValueError('Query and dataset must have same dimensions')    
    if use_cuda:
        torch.cuda.synchronize()
    y=y.contiguous()
    y_labels=self.__assign(y)
    
    
    y_ranges,_,_ = cluster_ranges_centroids(y,y_labels)
    self.__y_ranges=y_ranges
    y, y_labels = self.__sort_clusters(y, y_labels,store_x=False)   
    x_LT=LazyTensor(self.__x.unsqueeze(0).to(self.__device).contiguous())
    y_LT=LazyTensor(y.unsqueeze(1).to(self.__device).contiguous())
    D_ij=((y_LT-x_LT)**2).sum(-1)
    
    ranges_ij = from_matrix(y_ranges, self.__x_ranges, self.__keep)
    D_ij.ranges=ranges_ij
    nn=D_ij.argKmin(K=self.__k,axis=1)
    return self.__unsort(nn)

  def brute_force(self,x,y,k=5):
    if use_cuda:
        torch.cuda.synchronize()    
    x_LT=LazyTensor(x.unsqueeze(0))
    y_LT=LazyTensor(y.unsqueeze(1))
    D_ij=((y_LT-x_LT)**2).sum(-1) 
    return D_ij.argKmin(K=k,axis=1)