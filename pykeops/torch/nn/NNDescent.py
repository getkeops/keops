class NNDescent:
    def __init__(
        self,
        data=None,
        k=5,
        metric="euclidian",
        initialization_method="forest",
        num_trees=5,
        leaf_multiplier=1,
        big_leaf_depth=5,
        verbose=False,
    ):
        """Initialize the NNDescent class.

        Initializes the NNDescent class given all relevant parameters. If data is
        provided, it fits the NNDescent search graph to the data.

        Args:
          data ((N,d) Tensor): Dataset of N datapoints of dimensionality d.
          k (int): The number of neighbors to which each node connects in the search graph.
          metric (string): Name of metric, either "euclidian" and "manhattan"
          initialization_method (string): The type of initialization to be used for
            the search graph. Can be "random", "random_big" or "forest".
          num_trees (int): Number of trees used in "random_big" or "forest" initializations.
          leaf_multiplier (int): Parameter for the Tree class for tree-based initializations.
          big_leaf_depth (int): The depth at which the big leaves are taken to be used at
            the start of search.
        """

        # Setting parameters
        self.k = k
        self.metric = metric
        self.init_method = initialization_method
        self.num_trees = num_trees
        self.leaf_multiplier = leaf_multiplier
        self.big_leaf_depth = big_leaf_depth
        self.big_leaves = None

        # If data is provided, we call the fit function.
        if data is not None:
            self.fit(data, verbose=verbose)

    def distance(self, x, y):
        # Square of euclidian distance. Skip the root for faster computation.
        if self.metric == "euclidian":
            return ((x - y) ** 2).sum(-1)
        elif self.metric == "manhattan":
            return ((x - y).abs()).sum(-1)

    def fit(self, X, iter=20, verbose=False):
        """Fits the NNDescent search graph to the data set X.

        Args:
          X ((N,d) Tensor): Dataset of N datapoints of dimensionality d.
        """
        self.data = X

        # A 2D tensor representing a directed graph.
        # The value a = graph[i,j] represents an edge from point x_i to x_a.
        N = X.shape[0]
        self.graph = torch.zeros(size=[N, self.k], dtype=torch.long)

        # Initialize graph
        if self.init_method == "random":
            self._initialize_graph_randomly()
        elif self.init_method == "random_big":
            self._initialize_graph_big_random(self.data, self.num_trees)
        elif self.init_method == "forest":
            self._initialize_graph_forest(
                self.data, self.num_trees, self.leaf_multiplier, self.big_leaf_depth
            )

        # A set of tuples (i,j) of indices for which the distance has already been calculated.
        self.explored_edges = set()

        # A 2D tensor representing the distance between point x_i and x_graph[i,j]
        self.k_distances = torch.zeros([N, self.k])

        # Update the graph
        self._calculate_all_distances()
        self._update_graph(iter=iter, verbose=verbose)

    def _update_graph(self, iter=25, verbose=False):
        """Updates the graph using algorithm: https://pynndescent.readthedocs.io/en/latest/how_pynndescent_works.html

        Args:
          iter (int): Number of iterations to use when updating search graph.
        """
        # [STEP 1: Start with random graph.] Iterate
        start = time.time()
        for it in range(iter):
            if verbose:
                print(
                    "Iteration number",
                    it,
                    "with average distance of",
                    torch.mean(self.k_distances).item(),
                    "Took",
                    time.time() - start,
                    "seconds.",
                )
            has_changed = False

            # [STEP 2: For each node:] (TODO: Investigate whether this can be vectorized.)
            for i, neighbors in enumerate(self.graph):
                # Distances of current neighbors
                dist_current_neighbors = self.k_distances[i]

                # [STEP 3: Measure distance from the node to the neighbors of its neighbors]
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
                if torch.max(idx[: self.k]) >= self.k:
                    has_changed = True
                    self.graph[i] = cat_idx[idx[: self.k]]
                    self.k_distances[i] = dist_sorted[: self.k]

            # [STEP 5: If any changes were made, repeat iteration, otherwise stop]
            if not has_changed:
                if verbose:
                    print("Fitting complete! Took", it, "iterations.")
                break

    def kneighbors(self, X, max_num_steps=100, tree_init=True, verbose=False):
        """Returns k nearest neighbors of input X using NNDescent.

        Our code is largely based on this algorithm:
          https://pynndescent.readthedocs.io/en/latest/how_pynndescent_works.html#Searching-using-a-nearest-neighbor-graph

        Args:
          X ((N,d) Tensor): A query set for which to find k neighbors.
          max_num_steps (int): The maximum number of steps to take during search.

        Returns:
          The indices of the k nearest neighbors in the fitted data.
        """

        # N datapoints of dimension d
        N, d = X.shape
        k = self.k

        # Boolean mask to keep track of those points whose search is still ongoing
        is_active = torch.ones(N) == 1

        # If graph was initialized using trees, we can use information from there to initialize in a diversed manner.
        if self.big_leaves is not None and tree_init:
            candidate_idx = self.big_leaves.unsqueeze(0).repeat(N, 1)  # Shape: (N,32)
        else:
            # Random initialization for starting points of search.
            candidate_idx = torch.randint(
                high=len(self.data), size=[N, k + 1], dtype=torch.long
            )

        # Sort the candidates by distance from X
        distances = self.distance(self.data[candidate_idx], X.unsqueeze(1))
        # distances = ((self.data[candidate_idx] - X.unsqueeze(1))**2).sum(-1)
        sorted, idx = torch.sort(distances, dim=1)
        candidate_idx = torch.gather(candidate_idx, dim=1, index=idx)
        # Truncate to k+1 nearest
        candidate_idx = candidate_idx[:, : (k + 1)]

        # Track the nodes we have explored already, in N x num_explored tensor
        num_explored = self.k * 2
        explored = torch.full(size=[N, num_explored], fill_value=-1)

        start = time.time()
        # The initialization of candidates and explored set is done. Now we can search.
        count = 0
        while count < max_num_steps:
            if verbose:
                print(
                    "Step",
                    count,
                    "- Search is completed for",
                    1 - torch.mean(1.0 * is_active).item(),
                    "- this step took",
                    time.time() - start,
                    "s",
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
            shift = torch.cat(
                (
                    torch.full((len(expanded_idx), 1), -1),
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
            candidate_idx[is_active] = expanded_idx[:, : (self.k + 1)]

            # [6. Return to step 2. If we have already tried all candidates in pool, we stop in the if not unexplored]
            count += 1

        # Return the k candidates
        if verbose:
            print(
                "Graph search finished after",
                count,
                "steps. Finished for:",
                1 - torch.mean(1.0 * is_active).item(),
            )
        return candidate_idx[:, :-1]

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
        """Initializes self.graph with random values such that each point has k distinct neighbors"""
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
                k=self.k, largest=False
            ).indices  # find indices of KNN
            self.graph[i] = temp_row[indices]  # assign KNN to graph

    def _initialize_graph_forest(self, data, numtrees, leaf_multiplier, big_leaf_depth):
        """Initializes self.graph with a forest of random trees, such that each point has k distinct neighbors"""
        N, k = self.graph.shape
        dim = data.shape[1]

        temp_graph = torch.tensor(())
        for j in range(numtrees):
            # Create trees, obtain leaves
            t = Tree(data, k=k * leaf_multiplier, big_leaf_depth=big_leaf_depth)

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
            if j == 0:
                self.big_leaves = torch.LongTensor(t.big_leaves)

        warning_count = 0  # number of indices for which some neighbours are random

        # find KNN for each row in giant graph
        # TODO - implement the below without a for loop
        for i, row in enumerate(temp_graph):
            temp_row = torch.unique(row).type(torch.LongTensor)  # remove duplicates
            temp_row = temp_row[temp_row != i]  # remove self

            temp_points = data[temp_row, :]  # pick out elements from dataset
            d = (
                (data[i].reshape(1, dim).unsqueeze(1) - temp_points.unsqueeze(0)) ** 2
            ).sum(-1)
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
            print("WARNING!", warning_count, " INDICES ARE RANDOM!")


class Tree:
    """
    Random projection tree class that splits the data evenly per split
    Each split is performed by calculating the projection distance of each datapoint to a random unit vector
    The datapoints are then split by the median of of these projection distances
    The indices of the datapoints are stored in tree.leaves, as a nested list
    """

    def __init__(self, x, k=5, big_leaf_depth=5):
        self.min_size = 2 * k - 1
        self.leaves = []
        self.sizes = []
        self.big_leaf_depth = big_leaf_depth
        self.big_leaves = []  # leaves at depth = 5
        indices = torch.arange(x.shape[0])
        self.tree = self.make_tree(x, indices, depth=0)

    def make_tree(self, x, indices, depth):
        if depth == self.big_leaf_depth:  # add to big_leaves if depth=5
            self.big_leaves.append(int(indices[0]))
        if x.shape[0] > self.min_size:
            v = self.choose_rule(x)
            distances = torch.tensordot(
                x, v, dims=1
            )  # create list of projection distances
            median = torch.median(distances)
            left_bool = (
                distances <= median
            )  # create boolean array where entries are true if distance <= median
            right_bool = ~left_bool  # inverse of left_bool
            left_indices = indices[left_bool]
            right_indices = indices[right_bool]
            self.make_tree(x[left_bool, :], left_indices, depth + 1)
            self.make_tree(x[right_bool, :], right_indices, depth + 1)
        elif x.shape[0] != 0:
            self.leaves.append(indices.tolist())
            self.sizes.append(x.shape[0])
        return

    def choose_rule(self, x):
        dim = x.shape[1]
        v = torch.rand(dim)  # create random vector
        v /= torch.norm(v)  # normalize to unit vector
        return v
