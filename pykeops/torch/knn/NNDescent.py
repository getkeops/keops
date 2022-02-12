import torch
import time
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import cluster_ranges_centroids, from_matrix, sort_clusters
from pykeops.torch.utils import torchtools


class NNDescent:
    def __init__(
        self,
        data=None,
        k=5,
        metric="euclidean",
        initialization_method="forest",
        num_trees=5,
        leaf_multiplier=128,
        big_leaf_depth=5,
        verbose=False,
        backend="torch",
    ):
        """Initialize the NNDescent class.

        Initializes the NNDescent class given all relevant parameters. If data is
        provided, it fits the NNDescent search graph to the data.
        NNDescent is an approximation strategy for k-Nearest Neighbor search. It
        constructs a k-NN graph on the dataset, which is then navigated with a
        graph-based search algorithm during query time.

        The original paper on the method: https://www.cs.princeton.edu/cass/papers/www11.pdf
        Our code was inspired by the PyNNDescent documentation: https://pynndescent.readthedocs.io/en/latest/how_pynndescent_works.html

        Args:
            data ((N,D) Tensor): Dataset of N datapoints of dimensionality D.
            k (int): The number of nearest neighbors which we want to find for each query point
            metric (string): Name of metric, either "euclidean" and "manhattan"
            initialization_method (string): The type of initialization to be used for
                the search graph. Can be "random", "random_big", "forest" or "cluster".
            num_trees (int): Number of trees used in "random_big" or "forest" initializations.
            big_leaf_depth (int): The depth at which the big leaves are taken to be used at
                the start of search.
            verbose (boolean): Determines whether or not to print information while fitting.
            backend (string): Either "torch" or "keops". Determines if we want to use LazyTensors in cluster initialization.

        Args not used when initialization_method = "cluster":
            leaf_multiplier (int): Parameter for the Tree class for tree-based initializations.
                when initialization_method = "cluster", this parameter is used to adjust the number
                of clusters to be close to the value specified in the fit function.
        """

        # Setting parameters
        self.k = k
        self.metric = metric
        self.init_method = initialization_method
        self.num_trees = num_trees
        self.leaf_multiplier = leaf_multiplier
        self.big_leaf_depth = big_leaf_depth
        self.big_leaves = None
        self.backend = backend

        # Distance function
        self.distance = torchtools.distance_function(metric)

        # If data is provided, we call the fit function.
        if data is not None:
            self.fit(data, verbose=verbose)

    def fit(self, X, iter=20, verbose=False, clusters=32, a=10, queue=5):
        """Fits the NNDescent search graph to the data set X.

        Args:
            X ((N,D) Tensor): Dataset of N datapoints of dimensionality D.
            iter (int): Maximum number of iterations for graph updates
            verbose (boolean): Determines whether or not to print information while fitting.
            queue (int): The number of neighbors to which each node connects in the search graph.

        Used only when initialization_method = "cluster":
            clusters (int): The min no. of clusters that we want the data to be clustered into
            a (int): The number of clusters we want to search over using the cluster method.

        """
        self.data = X
        self.device = X.device
        self.queue = queue

        if queue < self.k and self.init_method is not "cluster":
            self.queue = self.k
            print(
                "Warning: Value of queue must be larger than or equal to k! Set queue = k."
            )
        elif queue > a and self.init_method is "cluster":
            raise ValueError("Value of queue must be smaller than value of a!")
        elif clusters < 2 ** self.big_leaf_depth:
            # This is neccesary to use the more efficient initial points in the graph search.
            raise ValueError("Minimum number of clusters is 2^big_leaf_depth!")
        elif a > clusters:
            raise ValueError("Number of clusters must be larger than or equal to a!")
        elif X.is_cuda and self.init_method is not "cluster":
            raise ValueError("CUDA not supported for non-cluster version of NNDescent.")

        # A 2D tensor representing a directed graph.
        # The value a = graph[i,j] represents an edge from point x_i to x_a.
        N = X.shape[0]
        self.graph = torch.zeros(size=[N, self.queue], dtype=torch.long)

        # Initialize graph
        if self.init_method == "random":
            self._initialize_graph_randomly()
        elif self.init_method == "random_big":
            self._initialize_graph_big_random(self.data, self.num_trees)
        elif self.init_method == "forest":
            self._initialize_graph_forest(
                self.data, self.num_trees, self.leaf_multiplier, self.big_leaf_depth
            )
        elif self.init_method == "cluster":
            # Parameters used only for cluster search
            self.a = a
            self.num_clusters = clusters
            self._initialize_graph_clusters(self.data)

        # A set of tuples (i,j) of indices for which the distance has already been calculated.
        self.explored_edges = set()

        if self.init_method != "cluster":
            # A 2D tensor representing the distance between point x_i and x_graph[i,j]
            self.k_distances = torch.zeros([N, self.queue])

            # Update the graph
            self._calculate_all_distances()
            self._update_graph(iter=iter, verbose=verbose)

    def _update_graph(self, iter, verbose=False):
        """Updates the current estimate for the kNN-graph with the iterative NN-Descent algorithm

        See https://pynndescent.readthedocs.io/en/latest/how_pynndescent_works.html for detailed explanation.

        Args:
            iter (int): Number of iterations to use when updating search graph.
            verbose (boolean): Printing information about iterations while searching.
        """
        # [STEP 1: Start with random graph.] Iterate
        start = time.time()
        for it in range(iter):
            if verbose:
                print(
                    f"Iteration number {it} with average distance of {torch.mean(self.k_distances).item()}. Took {time.time() - start} seconds."
                )
            has_changed = False

            # [STEP 2: For each node:] (TODO: Investigate whether this can be vectorized.)
            for i, neighbors in enumerate(self.graph):
                # Distances of current neighbors
                dist_current_neighbors = self.k_distances[i]

                # [STEP 3: Measure distance from the node to the neighbors of its neighbors]
                # Find neighbors of neighbors
                potential_neighbors = {
                    a.item()
                    for a in self.graph[neighbors].flatten()
                    if a not in neighbors
                    and a != i
                    and (i, int(a)) not in self.explored_edges
                }
                potential_distances = torch.Tensor(
                    [
                        self.distance(self.data[i], self.data[n])
                        for n in potential_neighbors
                    ]
                )
                self.explored_edges.update([(i, int(r)) for r in potential_neighbors])

                # Concatenate potential neighbors to list of neighbors (indices and distances)
                cat_idx = torch.cat(
                    [neighbors, torch.Tensor(list(potential_neighbors))]
                )
                cat_dist = torch.cat([self.k_distances[i], potential_distances])

                # [STEP 4: If any are closer, then update the graph accordingly, and only keep the k closest]
                dist_sorted, idx = torch.sort(cat_dist)
                if torch.max(idx[: self.queue]) >= self.queue:
                    has_changed = True
                    self.graph[i] = cat_idx[idx[: self.queue]]
                    self.k_distances[i] = dist_sorted[: self.queue]

            # [STEP 5: If any changes were made, repeat iteration, otherwise stop]
            if not has_changed:
                if verbose:
                    print(f"Fitting complete! Took {it} iterations.")
                break

    def kneighbors(self, X, max_num_steps=100, tree_init=True, verbose=False):
        """Returns k nearest neighbors of input X using NNDescent.

        Our code is largely based on this algorithm:
          https://pynndescent.readthedocs.io/en/latest/how_pynndescent_works.html#Searching-using-a-nearest-neighbor-graph

        If init_method = 'clusters', we first cluster the data. Each node in the graph then represents a cluster.
        We then use the KeOps engine to perform the final nearest neighbours search over the nearest clusters to each query point

        Args:
            X ((N,D) Tensor): A query set for which to find k neighbors.
            K (int): How many neighbors to search for. Must be <=self.k for non-cluster methods. Default: self.k
            max_num_steps (int): The maximum number of steps to take during search.
            tree_init (boolean): Determine whether or not to use big leaves from projection trees as the starting point of search.
            verbose (boolean): Printing information about iterations while searching.

        Returns:
            The indices of the k nearest neighbors in the fitted data.
        """

        # N datapoints of dimension d
        N, d = X.shape
        k = self.queue

        # Boolean mask to keep track of those points whose search is still ongoing
        is_active = torch.ones(N) == 1

        # If graph was initialized using trees, we can use information from there to initialize in a diversed manner.
        if self.big_leaves is not None and tree_init:
            candidate_idx = self.big_leaves.unsqueeze(0).repeat(
                N, 1
            )  # Shape: (N,2**self.big_leaf_depth)
        else:
            # Random initialization for starting points of search.
            candidate_idx = torch.randint(
                high=len(self.data), size=[N, k + 1], dtype=torch.long
            )

        if self.init_method == "cluster":
            is_active = is_active.to(self.device)
            candidate_idx = candidate_idx.to(self.device)

        # Sort the candidates by distance from X
        distances = self.distance(self.data[candidate_idx], X.unsqueeze(1))
        # distances = ((self.data[candidate_idx] - X.unsqueeze(1))**2).sum(-1)
        sorted, idx = torch.sort(distances, dim=1)
        candidate_idx = torch.gather(candidate_idx, dim=1, index=idx)
        # Truncate to k+1 nearest
        candidate_idx = candidate_idx[:, : (k + 1)]

        # Track the nodes we have explored already, in N x num_explored tensor
        num_explored = k * 2
        explored = torch.full(size=[N, num_explored], fill_value=-1)

        if self.init_method == "cluster":
            explored = explored.to(self.device)

        start = time.time()
        # The initialization of candidates and explored set is done. Now we can search.
        count = 0
        while count < max_num_steps:
            if verbose:
                print(
                    f"Step {count} - Search is completed for {1 - torch.mean(1.0 * is_active).item()} - this step took {time.time() - start} s"
                )
            start = time.time()

            # [2. Look at nodes connected by an edge to the best untried node in graph]
            # diff_bool.shape is (M, k+1, num_explored), where M is the number of active searches
            diff_bool = (
                candidate_idx[is_active].unsqueeze(2) - explored[is_active].unsqueeze(1)
                == 0
            )
            in_explored = torch.any(diff_bool, dim=2)
            # batch_active is true for those who haven't been fully explored in the current batch
            batch_active = ~torch.all(in_explored[:, :-1], dim=1)

            # Update is_active mask. If none are active, break search
            is_active[is_active.clone()] = batch_active
            if not is_active.any():
                break

            # first_unexplored has indices of first unexplored element per row
            first_unexplored = torch.max(~in_explored[batch_active], dim=1)[
                1
            ].unsqueeze(1)
            # Unexplored nodes to be expanded
            unexplored_idx = torch.gather(
                candidate_idx[is_active], dim=1, index=first_unexplored
            ).squeeze(-1)
            explored[is_active, (count % num_explored)] = unexplored_idx

            # [3. Add all these nodes to our potential candidate pool]
            # Add neighbors of the first unexplored point to the list of candidates
            expanded_idx = torch.cat(
                (self.graph[unexplored_idx], candidate_idx[is_active]), dim=1
            )

            # We remove repeated indices from consideration by adding float('inf') to them.
            expanded_idx = torch.sort(expanded_idx)[0]
            temp = torch.full((len(expanded_idx), 1), -1)

            if self.init_method == "cluster":
                expanded_idx = expanded_idx.to(self.device)
                temp = temp.to(self.device)

            shift = torch.cat(
                (
                    temp,
                    torch.sort(expanded_idx, dim=1)[0][:, :-1],
                ),
                dim=1,
            )
            unwanted_indices = expanded_idx == shift

            # [4. Sort by closeness].
            distances = self.distance(
                self.data[expanded_idx], X[is_active].unsqueeze(1)
            )
            # distances = ((self.data[expanded_idx] - X[is_active].unsqueeze(1))**2).sum(-1)
            distances[unwanted_indices] += float("inf")
            sorted, idx = torch.sort(distances, dim=1)
            expanded_idx = torch.gather(expanded_idx, dim=1, index=idx)

            # [5. Truncate to k+1 best]
            candidate_idx[is_active] = expanded_idx[:, : (k + 1)]

            # [6. Return to step 2. If we have already tried all candidates in pool, we stop in the if not unexplored]
            count += 1

        # Return the k candidates
        if verbose:
            print(
                f"Graph search finished after {count} steps. Finished for: {(1 - torch.mean(1.0 * is_active).item()) * 100}%."
            )

        if self.init_method == "cluster":
            return self.final_brute_force(
                candidate_idx[:, : self.k], X, verbose=verbose
            )
        else:
            return candidate_idx[:, : self.k]

    def _calculate_all_distances(self):
        """Updates the distances (self.k_distances) of the edges found in self.graph."""
        # Uses loop for simplicity.
        for i, row in enumerate(self.graph):
            # Indices of current k neighbors in self.graph
            neighbor_indices = [(i, int(r)) for r in row]
            # The distances of those neighbors are saved in k_distances
            self.k_distances[i] = torch.Tensor(
                [self.distance(self.data[a], self.data[b]) for a, b in neighbor_indices]
            )
            # Add pairs to explored_edges set
            self.explored_edges.update(neighbor_indices)

    def _initialize_graph_randomly(self):
        """Initializes self.graph with random values such that each point has 'queue' distinct neighbors"""
        N, k = self.graph.shape
        # Initialize graph randomly, removing self-loops
        self.graph = torch.randint(high=N - 1, size=[N, k], dtype=torch.long)
        row_indices = torch.arange(N).unsqueeze(1).repeat(1, k)
        self.graph[self.graph >= row_indices] += 1

    def _initialize_graph_big_random(self, data, numtrees):
        """Initializes self.graph randomly, but with more neighbours at the start"""
        N, k = self.graph.shape
        temp_graph = torch.tensor([])

        # make 'trees', combine into giant graph with each element (row) having k * num_trees neighbours
        # this is a small for loop - numtrees and k << datapoints
        for j in range(numtrees):
            tree_graph = torch.tensor([])
            for i in range(k):
                tree_graph = torch.cat(
                    (tree_graph, torch.randperm(N)), 0
                )  # generate randomly shuffled list of N indices
            tree_graph = tree_graph.reshape(
                -1, k
            )  # creates a N x k tensor with N indices, each appearing k times. This represents 1 'tree'
            temp_graph = torch.cat(
                (temp_graph, tree_graph), 1
            )  # combine into giant N x (k*num_trees) tensor. This represents the forest

        # find KNN for each row in giant graph
        # TODO - implement the below without a for loop
        for i, row in enumerate(temp_graph):
            temp_row = torch.unique(row).type(torch.LongTensor)  # remove duplicates
            temp_row = temp_row[temp_row != i]  # remove self

            temp_points = data[temp_row, :]  # pick out elements from dataset
            distances = self.distance(temp_points, data[i])  # Euclidean distances
            indices = distances.topk(
                k=self.queue, largest=False
            ).indices  # find indices of KNN
            self.graph[i] = temp_row[indices]  # assign KNN to graph

    def _initialize_graph_forest(self, data, numtrees, leaf_multiplier, big_leaf_depth):
        """Initializes self.graph with a forest of random trees, such that each point has 'queue' distinct neighbors"""
        N, k = self.graph.shape
        dim = data.shape[1]

        temp_graph = torch.tensor(())
        for j in range(numtrees):
            # Create trees, obtain leaves. RandomProjectionTree class is defined below.
            t = RandomProjectionTree(
                data, k=k * leaf_multiplier, big_leaf_depth=big_leaf_depth
            )

            # Create temporary graph, 1 for each tree
            # Leaves are of uneven size; select smallest leaf size as graph size
            cols = min([len(leaf) for leaf in t.leaves])
            rows = len(t.leaves)
            tree_graph = torch.zeros((N, cols))
            leaves = torch.tensor(())
            idx_update = torch.tensor(())

            # Update graph using leaves
            for leaf in t.leaves:
                temp_idx = torch.as_strided(
                    torch.tensor(leaf).repeat(1, 2),
                    size=[len(leaf), cols],
                    stride=[1, 1],
                    storage_offset=1,
                )
                tree_graph[
                    leaf, :
                ] = temp_idx.float()  # update graph. a lot of overwriting
            # Concatenate all graphs from all trees into 1 giant graph
            temp_graph = torch.cat((temp_graph, tree_graph), 1)

            # Add the first tree's big_leaves to the NNDescent's big_leaves
            if j == 0 and t.big_leaves:
                self.big_leaves = torch.LongTensor(t.big_leaves)

        warning_count = 0  # number of indices for which some neighbours are random

        # find KNN for each row in giant graph
        # TODO - implement the below without a for loop
        for i, row in enumerate(temp_graph):
            temp_row = torch.unique(row).type(torch.LongTensor)  # remove duplicates
            temp_row = temp_row[temp_row != i]  # remove self

            temp_points = data[temp_row, :]  # pick out elements from dataset
            d = self.distance(
                data[i].reshape(1, dim).unsqueeze(1), temp_points.unsqueeze(0)
            )
            distances, indices = torch.sort(d, dim=1)
            indices = indices.flatten()[:k]

            indices = temp_row[indices]

            # pad with random indices if there are not enough neighbours
            warning = False  # warning flag
            while len(indices) < k:
                pad = torch.randint(
                    high=N - 1,
                    size=[
                        k - len(indices),
                    ],
                    dtype=torch.long,
                )
                indices = torch.cat((indices, pad))
                indices = torch.unique(indices).type(
                    torch.LongTensor
                )  # remove duplicates
                indices = indices[indices != i]  # remove self
                warning = True

            self.graph[i] = indices  # assign KNN to graph

            if warning:
                warning_count += 1

        if warning_count:
            print(f"WARNING! {warning_count} INDICES ARE RANDOM!")

    def _initialize_graph_clusters(self, data):
        """Initializes self.graph on cluster centroids, such that each cluster has 'a' distinct neighbors"""
        N, dim = data.shape
        k = self.k
        a = self.a
        backend = self.backend
        leaf_multiplier = (
            N / self.num_clusters / k
        )  # to get number of clusters ~ num_clusters
        self.clusters = (
            torch.ones(
                N,
            )
            * -1
        )

        data = data.to(self.device)

        # Create trees, obtain leaves. RandomProjectionTree class is defined below.
        t = RandomProjectionTree(data, k, self.big_leaf_depth, leaf_multiplier, backend)

        self.leaves = len(t.leaves)

        # Assign each point to a cluster, 1 cluster per tree in forest
        for i, leaf in enumerate(t.leaves):
            self.clusters[leaf] = i
        self.data_orig = self.data.clone()
        self.data = t.centroids.clone()

        # Find nearest centroids
        x_LT = LazyTensor(self.data.unsqueeze(1).to(self.device))
        y_LT = LazyTensor(self.data.unsqueeze(0).to(self.device))
        d = self.distance(x_LT, y_LT)
        indices = d.argKmin(K=a + 1, dim=1).long()
        self.centroids_neighbours = indices[:, 1:].long()

        self.num_clusters = self.centroids_neighbours.shape[0]
        self.graph = self.centroids_neighbours

        # Assign big_leaves by searching for the correct cluster
        self.big_leaves = torch.LongTensor(t.big_leaves)
        for i, index in enumerate(self.big_leaves):
            self.big_leaves[i] = self.clusters[index]
        return

    def final_brute_force(self, nearest_clusters, query_pts, verbose=False):
        """ Final brute force search over clusters in cluster method"""
        if verbose:
            print("Starting brute force search over clusters.")
        return self._final_brute_force(nearest_clusters, query_pts)

    def _final_brute_force(self, nearest_clusters, query_pts):
        """ Final brute force search over clusters in cluster method"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        k = self.k

        x = self.data_orig.to(self.device)
        x_labels = self.clusters.long()
        y = query_pts.to(self.device)
        y_labels = nearest_clusters[:, 0]

        x = x.contiguous()
        y = y.contiguous()
        x_labels = x_labels.to(self.device)
        y_labels = y_labels.to(self.device)

        clusters, a = self.graph.shape
        r = torch.arange(clusters).repeat(a, 1).T.reshape(-1).long()
        keep = torch.zeros([clusters, clusters], dtype=torch.bool).to(self.device)
        keep[r, self.graph.flatten()] = True
        keep += torch.eye(clusters).bool().to(self.device)

        x_ranges, x_centroids, _ = cluster_ranges_centroids(x, x_labels)
        y_ranges, y_centroids, _ = cluster_ranges_centroids(y, y_labels)

        x, x_labels = self.__sort_clusters(x, x_labels, store_x=True)
        y, y_labels = self.__sort_clusters(y, y_labels, store_x=False)

        x_LT = LazyTensor(x.unsqueeze(0).to(self.device).contiguous())
        y_LT = LazyTensor(y.unsqueeze(1).to(self.device).contiguous())
        D_ij = self.distance(y_LT, x_LT)

        x_ranges = x_ranges.to(self.device)
        y_ranges = y_ranges.to(self.device)
        ranges_ij = from_matrix(y_ranges, x_ranges, keep)
        D_ij.ranges = ranges_ij
        nn = D_ij.argKmin(K=k, axis=1)
        return self.__unsort(nn)

    def __sort_clusters(self, x, lab, store_x=True):
        lab, perm = torch.sort(lab.view(-1))
        if store_x:
            self.__x_perm = perm
        else:
            self.__y_perm = perm
        return x[perm], lab

    def __unsort(self, nn):
        return torch.index_select(self.__x_perm[nn], 0, self.__y_perm.argsort())


class RandomProjectionTree:
    """
    Random projection tree class that splits the data evenly per split
    Each split is performed by calculating the projection distance of each datapoint to a random unit vector
    The datapoints are then split by the median of of these projection distances
    The indices of the datapoints are stored in tree.leaves, as a nested list
    """

    def __init__(
        self,
        x,
        k=5,
        big_leaf_depth=5,
        leaf_multiplier=128,
        backend="torch",
        device=None,
    ):
        self.min_size = k * leaf_multiplier
        self.leaves = []
        self.sizes = []
        if device is None:
            self.device = x.device
        else:
            self.device = device
        self.centroids = torch.tensor(()).to(self.device)
        self.big_leaf_depth = big_leaf_depth
        self.big_leaves = []  # leaves at depth = 5
        indices = torch.arange(x.shape[0])

        self.dim = x.shape[1]
        self.data = x.to(self.device)
        self.backend = backend  # Boolean to choose LT or torch initialization

        self.tree = self.make_tree(indices, depth=0)
        self.centroids = self.centroids.reshape(-1, x.shape[1])

    def make_tree(self, indices, depth):
        if depth == 5:  # add to big_leaves if depth=5
            self.big_leaves.append(int(indices[0]))
        if indices.shape[0] > self.min_size:
            v = self.choose_rule().to(self.device)

            if self.backend == "keops":
                distances = self.dot_product(
                    self.data[indices], v
                )  # create list of projection distances
            else:
                distances = torch.tensordot(
                    self.data[indices], v, dims=1
                )  # create list of projection distances

            median = torch.median(distances)
            left_bool = (
                distances <= median
            )  # create boolean array where entries are true if distance <= median
            self.make_tree(indices[left_bool], depth + 1)
            self.make_tree(indices[~left_bool], depth + 1)
        elif indices.shape[0] != 0:
            self.leaves.append(indices.tolist())
            self.sizes.append(indices.shape[0])
            centroid = self.data[indices].mean(dim=0)  # get centroid position
            self.centroids = torch.cat((self.centroids, centroid))
        return

    def choose_rule(self):
        v = torch.rand(self.dim)  # create random vector
        v /= torch.norm(v)  # normalize to unit vector
        return v

    def dot_product(self, x, v):
        # Calculate dot product between matrix x and vector v using LazyTensors
        v_LT = LazyTensor(v.view(1, 1, -1))
        x_LT = LazyTensor(x.unsqueeze(0))
        return (v_LT | x_LT).sum_reduction(axis=0).flatten()
