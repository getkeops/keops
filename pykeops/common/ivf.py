class GenericIVF:
    def __init__(self, k, metric, normalise, LazyTensor):
        self.__k = k
        self.__normalise = normalise

        self.__update_metric(metric)
        self.__LazyTensor = LazyTensor

        self.__c = None

    def __update_metric(self, metric):
        if isinstance(metric, str):
            self.__distance = self.tools.distance_function(metric)
            self.__metric = metric
        elif callable(metric):
            self.__distance = metric
            self.__metric = "custom"
        else:
            raise ValueError("Unrecognised metric input type")

    @property
    def metric(self):
        return self.__metric

    @property
    def c(self):
        if self.__c is not None:
            return self.__c
        else:
            raise ValueError("Run .fit() first!")

    def __get_tools(self):
        pass

    def __k_argmin(self, x, y, k=1):
        x_LT = self.__LazyTensor(
            self.tools.to(self.tools.unsqueeze(x, 1), self.__device)
        )
        y_LT = self.__LazyTensor(
            self.tools.to(self.tools.unsqueeze(y, 0), self.__device)
        )

        d = self.__distance(x_LT, y_LT)
        if not self.tools.is_tensor(x):
            if self.__backend:
                d.backend = self.__backend

        if k == 1:
            return self.tools.view(self.tools.long(d.argmin(dim=1)), -1)
        else:
            return self.tools.long(d.argKmin(K=k, dim=1))

    def __sort_clusters(self, x, lab, store_x=True):
        lab, perm = self.tools.sort(self.tools.view(lab, -1))
        if store_x:
            self.__x_perm = perm
        else:
            self.__y_perm = perm
        return x[perm], lab

    def __unsort(self, nn):
        return self.tools.index_select(self.__x_perm[nn], 0, self.__y_perm.argsort())

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

        cl, c = self.tools.kmeans(
            x,
            self.__distance,
            clusters,
            Niter=Niter,
            device=self.__device,
            approx=approx,
            normalise=self.__normalise,
        )

        self.__c = c
        cl = self.__assign(x)

        ncl = self.__k_argmin(c, c, k=a)
        self.__x_ranges, _, _ = self.tools.cluster_ranges_centroids(x, cl)

        x, x_labels = self.__sort_clusters(x, cl, store_x=True)
        self.__x = x
        r = self.tools.repeat(self.tools.arange(clusters, device=self.__device), a)
        self.__keep = self.tools.to(
            self.tools.zeros([clusters, clusters], dtype=bool), self.__device
        )
        self.__keep[r, ncl.flatten()] = True

        return self

    def __assign(self, x, c=None):
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
        y_labels = self.__assign(y)

        y_ranges, _, _ = self.tools.cluster_ranges_centroids(y, y_labels)
        self.__y_ranges = y_ranges
        y, y_labels = self.__sort_clusters(y, y_labels, store_x=False)
        x_LT = self.__LazyTensor(self.tools.unsqueeze(self.__x, 0))
        y_LT = self.__LazyTensor(self.tools.unsqueeze(y, 1))
        D_ij = self.__distance(y_LT, x_LT)
        ranges_ij = self.tools.from_matrix(y_ranges, self.__x_ranges, self.__keep)
        D_ij.ranges = ranges_ij
        nn = D_ij.argKmin(K=self.__k, axis=1)
        return self.__unsort(nn)

    def brute_force(self, x, y, k=5):
        x_LT = self.__LazyTensor(self.tools.unsqueeze(x, 0))
        y_LT = self.__LazyTensor(self.tools.unsqueeze(y, 1))
        D_ij = self.__distance(y_LT, x_LT)
        return D_ij.argKmin(K=k, axis=1)
