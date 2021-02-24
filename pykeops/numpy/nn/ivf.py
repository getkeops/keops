from pykeops.numpy import LazyTensor
from pykeops.numpy.cluster import cluster_ranges_centroids
from pykeops.numpy.cluster import from_matrix
import pykeops.config
import numpy as np
class ivf():
  def __init__(self,k=5):
    self.__c=None
    self.__k=k
    self.__x=None
    self.__keep=None
    self.__x_ranges=None
    self.__x_perm=None
    self.__y_perm=None
    self.__use_gpu=None
       
  def __KMeans(self,x, K=10, Niter=15):
      N, D = x.shape  
      c = np.copy(x[:K, :])  
      x_i = LazyTensor(x[:, None, :])  
      for i in range(Niter):
          c_j = LazyTensor(c[None, :, :])  
          D_ij = ((x_i - c_j) ** 2).sum(-1)  
          if self.__use_gpu:
            D_ij.backend='GPU'
          else:
            D_ij.backend='CPU'          
          cl = D_ij.argmin(axis=1).astype(int).reshape(N) 

          Ncl = np.bincount(cl).astype(dtype = "float32") 
          for d in range(D): 
              c[:, d] = np.bincount(cl, weights=x[:, d]) / Ncl
      return cl, c      

  def __k_argmin(self,x,y,k=1):
  
    x_LT=LazyTensor(np.expand_dims(x, 1))
    y_LT=LazyTensor(np.expand_dims(y, 0))
    d=((x_LT-y_LT)**2).sum(-1)
    if self.__use_gpu:
      d.backend='GPU'
    else:
      d.backend='CPU'    
    if k==1:
      return d.argmin(dim=1).flatten()
    else:
      return d.argKmin(K=k,dim=1)

  def __sort_clusters(self,x,lab,store_x=True):   
    perm=np.argsort(lab.flatten())
    if store_x:
      self.__x_perm=perm 
    else:
      self.__y_perm=perm
    return x[perm],lab[perm]

  def __unsort(self,nn):
    return np.take(self.__x_perm[nn],self.__y_perm.argsort(),axis=0)

  def fit(self,x,clusters=50,a=5,use_gpu=True,n=15):
    '''
    Fits the main dataset
    '''
    if type(x)!=np.ndarray:
      raise ValueError('Input must be a numpy ndarray')
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
            
    if use_gpu and not pykeops.config.gpu_available:
      raise ValueError('use_gpu = True but GPU not detected')
    self.__use_gpu=use_gpu
    cl, c = self.__KMeans(x,clusters,Niter=n)
    self.__c=c
    cl=self.__assign(x)

    ncl=self.__k_argmin(c,c,k=a)
    self.__x_ranges, _, _ = cluster_ranges_centroids(x, cl)
    x, x_labels = self.__sort_clusters(x,cl,store_x=True) 
    self.__x=x

    r=np.arange(clusters).repeat(a).T.reshape(-1)
    self.__keep= np.zeros([clusters,clusters], dtype=bool)
    self.__keep[r,ncl.flatten()]=True        
    return self


  def __assign(self,x,c=None):
    if c is None:
      c=self.__c
    return self.__k_argmin(x,c)  
    
  def kneighbors(self,y,sparse=True):
    '''
    Obtain the k nearest neighbors of the query dataset y
    '''
    if self.__x is None:
      raise ValueError('Input dataset not fitted yet! Call .fit() first!')
    if type(y)!=np.ndarray:
      raise ValueError("Query dataset must be a numpy ndarray")
    if len(y.shape)!=2:
      raise ValueError('Query dataset must be a 2D array')      
    if self.__x.shape[-1]!=y.shape[-1]:
      raise ValueError('Query and dataset must have same dimensions')

    y_labels=self.__assign(y,self.__c)
    y_ranges,_,_ = cluster_ranges_centroids(y, y_labels)

    y, y_labels = self.__sort_clusters(y, y_labels,store_x=False)   
    
    x_LT=LazyTensor(np.expand_dims(self.__x,0))
    y_LT=LazyTensor(np.expand_dims(y,1))
    D_ij=((y_LT-x_LT)**2).sum(-1)
    ranges_ij = from_matrix(y_ranges,self.__x_ranges,self.__keep)
    D_ij.ranges=ranges_ij
    if self.__use_gpu:
      D_ij.backend='GPU'
    else:
      D_ij.backend='CPU'
    nn=D_ij.argKmin(K=self.__k,axis=1)
    return self.__unsort(nn)

  def brute_force(self,x,y,k=5):
    x_LT=LazyTensor(np.expand_dims(x,0))
    y_LT=LazyTensor(np.expand_dims(y,1))
    D_ij=((y_LT-x_LT)**2).sum(-1) 
    return D_ij.argKmin(K=k,axis=1)
