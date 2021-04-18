class GenericIVF:
    """Abstract class to compute IVF functions

    End-users should use 'pykeops.numpy.ivf' or 'pykeops.torch.ivf'

    """

    def __init__(self, k, metric, normalise, lazytensor):

        self.__k = k
        self.__normalise = normalise
        self.__update_metric(metric)
        self.__LazyTensor = lazytensor
        self.__c = None

    def __update_metric(self, metric):
        """
        Update the metric used in the class
        """
        if isinstance(metric, str):
            self.__distance = self.tools.distance_function(metric)
            self.__metric = metric
        elif callable(metric):
            self.__distance = metric
            self.__metric = "custom"
        else:
            raise ValueError(
                f"The 'metric' argument has type {type(metric)}, but only strings and functions are supported."
            )

    @property
    def metric(self):
        """Returns the metric used in the search"""
        return self.__metric

    @property
    def clusters(self):
        """Returns the clusters obtained through K-Means"""
        if self.__c is not None:
            return self.__c
        else:
            raise NotImplementedError("Run .fit() first!")

    def __get_tools(self):
        pass

    def __k_argmin(self, x, y, k=1):
        """
        Compute the k nearest neighbors between x and y, for various k
        """
        x_i = self.__LazyTensor(
            self.tools.to(self.tools.unsqueeze(x, 1), self.__device)
        )
        y_j = self.__LazyTensor(
            self.tools.to(self.tools.unsqueeze(y, 0), self.__device)
        )

        D_ij = self.__distance(x_i, y_j)
        if not self.tools.is_tensor(x):
            if self.__backend:
                D_ij.backend = self.__backend

        if k == 1:
            return self.tools.view(self.tools.long(D_ij.argmin(dim=1)), -1)
        else:
            return self.tools.long(D_ij.argKmin(K=k, dim=1))

    def __sort_clusters(self, x, lab, store_x=True):
        """
        Takes in a dataset and sorts according to its labels.

        Args:
          x ((N, D) array): Input dataset of N points in dimension D.
          lab ((N) array): Labels for each point in x
          store_x (bool): Store the sort permutations for use later
        """
        lab, perm = self.tools.sort(self.tools.view(lab, -1))
        if store_x:
            self.__x_perm = perm
        else:
            self.__y_perm = perm
        return x[perm], lab

    def __unsort(self, indices):
        """
        Given an input indices, undo and prior sorting operations.
        First, select the true x indices with __x_perm[indices]
        Then, use index_select to choose the indices in true x, for each true y.
        """
        return self.tools.index_select(
            self.__x_perm[indices], 0, self.__y_perm.argsort()
        )

    def _fit(
        self,
        x,
        clusters=50,
        a=5,
        Niter=15,
        device=None,
        backend=None,
        approx=False,
        n=50,
    ):
        """
        Fits the main dataset
        """

        # basic checks that the hyperparameters are as expected
        if type(clusters) != int:
            raise ValueError("Clusters must be an integer")
        if clusters >= len(x):
            raise ValueError("Number of clusters must be less than length of dataset")
        if type(a) != int:
            raise ValueError("Number of clusters to search over must be an integer")
        if a > clusters:
            raise ValueError(
                "Number of clusters to search over must be less than total number of clusters"
            )
        if len(x.shape) != 2:
            raise ValueError("Input must be a 2D array")
        # normalise the input if selected
        if self.__normalise:
            x = x / self.tools.repeat(self.tools.norm(x, 2, -1), x.shape[1]).reshape(
                -1, x.shape[1]
            )

        # if we want to use the approximation in Kmeans, and our metric is angular, switch to full angular metric
        if approx and self.__metric == "angular":
            self.__update_metric("angular_full")

        x = self.tools.contiguous(x)
        self.__device = device
        self.__backend = backend

        # perform K-Means
        cl, c = self.tools.kmeans(
            x,
            self.__distance,
            clusters,
            Niter=Niter,
            device=self.__device,
            approx=approx,
            n=n,
        )

        self.__c = c
        # perform one final cluster assignment, since K-Means ends on cluster update step
        cl = self.__assign(x)

        # obtain the nearest clusters to each cluster
        ncl = self.__k_argmin(c, c, k=a)
        self.__x_ranges, _, _ = self.tools.cluster_ranges_centroids(x, cl)

        x, x_labels = self.__sort_clusters(x, cl, store_x=True)
        self.__x = x
        r = self.tools.repeat(self.tools.arange(clusters, device=self.__device), a)
        # create a [clusters, clusters] sized boolean matrix
        self.__keep = self.tools.to(
            self.tools.zeros([clusters, clusters], dtype=bool), self.__device
        )
        # set the indices of the nearest clusters to each cluster to True
        self.__keep[r, ncl.flatten()] = True

        return self

    def __assign(self, x, c=None):
        """
        Assigns nearest clusters to a dataset.
        If no clusters are given, uses the clusters found through K-Means.

        Args:
          x ((N, D) array): Input dataset of N points in dimension D.
          c ((M, D) array): Cluster locations of M points in dimension D.
        """
        if c is None:
            c = self.__c
        return self.__k_argmin(x, c)

    def _kneighbors(self, y):
        """
        Obtain the k nearest neighbors of the query dataset y
        """
        if self.__x is None:
            raise ValueError("Input dataset not fitted yet! Call .fit() first!")
        if self.__device and self.tools.device(y) != self.__device:
            raise ValueError("Input dataset and query dataset must be on same device")
        if len(y.shape) != 2:
            raise ValueError("Query dataset must be a 2D tensor")
        if self.__x.shape[-1] != y.shape[-1]:
            raise ValueError("Query and dataset must have same dimensions")
        if self.__normalise:
            y = y / self.tools.repeat(self.tools.norm(y, 2, -1), y.shape[1]).reshape(
                -1, y.shape[1]
            )
        y = self.tools.contiguous(y)
        # assign y to the previously found clusters and get labels
        y_labels = self.__assign(y)

        # obtain y_ranges
        y_ranges, _, _ = self.tools.cluster_ranges_centroids(y, y_labels)
        self.__y_ranges = y_ranges

        # sort y contiguous
        y, y_labels = self.__sort_clusters(y, y_labels, store_x=False)

        # perform actual knn computation
        x_i = self.__LazyTensor(self.tools.unsqueeze(self.__x, 0))
        y_j = self.__LazyTensor(self.tools.unsqueeze(y, 1))
        D_ij = self.__distance(y_j, x_i)
        ranges_ij = self.tools.from_matrix(y_ranges, self.__x_ranges, self.__keep)
        D_ij.ranges = ranges_ij
        indices = D_ij.argKmin(K=self.__k, axis=1)
        return self.__unsort(indices)

    def brute_force(self, x, y, k=5):
        """Performs a brute force search with KeOps

        Args:
          x ((N, D) array): Input dataset of N points in dimension D.
          y ((M, D) array): Query dataset of M points in dimension D.
          k (int): Number of nearest neighbors to obtain

        """
        x_LT = self.__LazyTensor(self.tools.unsqueeze(x, 0))
        y_LT = self.__LazyTensor(self.tools.unsqueeze(y, 1))
        D_ij = self.__distance(y_LT, x_LT)
        return D_ij.argKmin(K=k, axis=1)
