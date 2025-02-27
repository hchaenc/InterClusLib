import pandas as pd
import numpy as np
from numpy import linalg, subtract, power, exp, meshgrid, zeros, outer
from math import sqrt
from numpy.random import RandomState
from warnings import warn
from sklearn.decomposition import PCA
from interClusLib.metric import MULTI_SIMILARITY_FUNCTIONS, MULTI_DISTANCE_FUNCTIONS
from collections import defaultdict
import os

class IntervalSOM:

    def __init__(self, 
                 x,
                 y,
                 n_dims,
                 learning_rate,
                 total_iterations,
                 sigma_init,
                 sigma_final = None,
                 decay_function = 'linear_decay_to_zero',
                 neighborhood_function = 'gaussian',
                 activation_distance = 'euclidean',
                 topology='rectangular',
                 random_seed=None,
                 sigma_decay_function='linear_interpolate'):
        """
        Initializes an Interval Self-Organizing Map (SOM).

        Parameters:
        -----------
        x : int
            Number of rows in the SOM grid.
        y : int
            Number of columns in the SOM grid.
        n_dims : int
            Number of dimensions (features) in the input data.
        sigma_init : float
            Initial neighborhood radius.
        sigma_final : float
            Final neighborhood radius.
        learning_rate : float
            Initial learning rate.
        total_iterations : int
            Total number of training iterations.
        decay_function : str or callable, optional (default='asymptotic_decay')
            The function that determines how the learning rate decreases.
        neighborhood_function : str, optional (default='gaussian')
            Defines how neighboring neurons are influenced during training.
        activation_distance : str, optional (default='euclidean')
            The distance metric used for Best Matching Unit (BMU) selection.
        topology : str, optional (default='rectangular')
            Determines the grid structure of the SOM ('rectangular' or 'hexagonal').
        random_seed : int, optional (default=None)
            Random seed for reproducibility.
        sigma_decay_function : str, optional (default='asymptotic_decay')
            The function controlling how the neighborhood radius shrinks.
        """

        if sigma_init > sqrt(x*x + y*y):
            warn('Warning: sigma might be too high ' +
                 'for the dimension of the map.')

        self.x = x
        self.y = y
        self.n_dims = n_dims
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.learning_rate = learning_rate
        self.total_iterations = total_iterations
        self.topology = topology
        self.random_seed = random_seed

        # Create a random number generator for initialization and sampling
        self._rng = np.random.RandomState(self.random_seed)

        # 1) Initialize the SOM prototypes (neurons' weight vectors)
        #    Each neuron stores n_dims interval values: [lower, upper]
        #    Shape: (x, y, n_dims, 2)
        self.prototypes = self._rng.rand(x, y, n_dims, 2)
        # Ensure that lower bounds are always ≤ upper bounds
        self.prototypes = np.sort(self.prototypes, axis=-1)

        # 2) Store the (row, col) coordinates of each neuron in the SOM grid
        #    For rectangular topology, (i, j) represents its position
        #    For hexagonal topology, a slight shift is applied
        self._locations = np.zeros((x, y, 2), dtype=float)
        for i in range(x):
            for j in range(y):
                self._locations[i,j] = [i, j]
        
        # create activation_map for each sample. shape => (x, y)
        self._activation_map = np.zeros((x, y))

        if self.topology == 'hexagonal':
            if neighborhood_function in ['triangle']:
                warn('triangle neighborhood function does not ' +
                     'take in account hexagonal topology')
            # Shift even rows horizontally by 0.5 to create a hexagonal layout
            for i in range(x):
                if i % 2 == 1:
                    self._locations[i,:, 0] = i
                    self._locations[i,:, 1] = np.arange(y) + 0.5
                else:
                    self._locations[i,:, 0] = i
                    self._locations[i,:, 1] = np.arange(y)
        elif self.topology == 'rectangular':
            pass
        else:
            raise ValueError("Topology must be either 'rectangular' or 'hexagonal'.")
        
        # 4) Learning rate decay functions
        self.lr_decay_functions = {
            'inverse_decay_to_zero': self._inverse_decay_to_zero,
            'linear_decay_to_zero': self._linear_decay_to_zero,
            'asymptotic_decay': self._asymptotic_decay}
        
        # Validate and assign decay function
        if isinstance(decay_function, str):
            if decay_function not in self.lr_decay_functions:
                msg = '%s not supported. Available functions: %s'
                raise ValueError(msg % (decay_function, ', '.join(self.lr_decay_functions.keys())))

            self._learning_rate_decay_function = self.lr_decay_functions[decay_function]
        elif callable(decay_function):
            self._learning_rate_decay_function = decay_function
        else:
            raise ValueError("decay_function must be a string or a callable function.")
        
        # 5) Sigma (neighborhood radius) decay functions
        sig_decay_functions = {
            'inverse_decay_to_one': self._inverse_decay_to_one,
            'linear_decay_to_one': self._linear_decay_to_one,
            'asymptotic_decay': self._asymptotic_decay,
            'linear_interpolate': self._linear_interpolate
            
        }

        if sigma_decay_function not in sig_decay_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (sigma_decay_function, ', '.join(sig_decay_functions.keys())))

        if sigma_decay_function == 'linear_interpolate' and self.sigma_final == 0:
            msg = "Error: Using 'linear_interpolate' requires sigma_final > 0."
            raise ValueError(msg)
        self._sigma_decay_function = sig_decay_functions[sigma_decay_function]
        
        # 6) Define neighborhood influence functions
        neig_functions = {
            'gaussian': self._gaussian,
            'mexican_hat': self._mexican_hat,
            'bubble': self._bubble,
            'triangle': self._triangle
        }

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function, ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle', 'bubble'] and (divmod(sigma, 1)[1] != 0 or sigma < 1):
            warn('sigma should be an integer >=1 when triangle or bubble are used as neighborhood function')

        self._neighborhood = neig_functions[neighborhood_function]

        # Validate and assign the activation distance function
        if isinstance(activation_distance, str):
            if activation_distance in MULTI_SIMILARITY_FUNCTIONS:
                self._activation_distance = MULTI_SIMILARITY_FUNCTIONS[activation_distance]
                self.isSim = True
            elif activation_distance in MULTI_DISTANCE_FUNCTIONS:
                self._activation_distance = MULTI_DISTANCE_FUNCTIONS[activation_distance]
                self.isSim = False
            else:
                valid_funcs = ", ".join(list(MULTI_SIMILARITY_FUNCTIONS.keys()) + list(MULTI_DISTANCE_FUNCTIONS.keys()))
                raise ValueError(f"Invalid distance function '{activation_distance}'. Available options: {valid_funcs}")
        elif callable(activation_distance):
            self._activation_distance = activation_distance

        

    def random_weights_init(self, intervals):
        """
        Initializes each neuron prototype by randomly selecting an interval 
        from the input 'intervals' dataset of shape (N, n_dims, 2).

        Parameters:
        -----------
        intervals : np.ndarray
            The dataset containing interval-valued data.
            Shape: (N, n_dims, 2), where:
            - N : Number of samples
            - n_dims : Number of dimensions (features)
            - 2 : Lower and upper bounds for each dimension

        Raises:
        -------
        ValueError:
            If the input dataset does not match the expected shape (N, n_dims, 2).
        """
        n_samples, dims, two_ = intervals.shape
        if dims != self.n_dims or two_ != 2:
            raise ValueError(f"Data shape must be (N, {self.n_dims}, 2). Got {intervals.shape}.")
        # 2) Assign each neuron (i, j) a randomly chosen interval from the dataset
        for i in range(self.x):
            for j in range(self.y):
                rand_idx = self._rng.randint(n_samples)
                self.prototypes[i,j] = intervals[rand_idx]  # shape (n_dims,2)
    
    def pca_weights_init(self, intervals):
        """
        Initializes the SOM neuron prototypes using a PCA-based approach:

        Steps:
        1) For each sample's interval [a,b], compute the center: 
        c = (a + b) / 2  → Shape: (N, n_dims)
        2) Perform PCA on these centers to extract the first two principal components (PC1, PC2).
        3) Map each neuron (i,j) to a 2D coordinate in the range [-1,1]^2 → (c1, c2).
        Then, reconstruct the "center" using:
        center = mean + c1 * PC1 + c2 * PC2 in R^n_dims.
        4) For each dimension `d`, assign:
        prototypes[i, j, d] = [center_d, center_d] → Meaning the interval collapses to a single point.

        Parameters:
        -----------
        intervals : np.ndarray
            Input interval-valued data.
            Shape: (N, n_dims, 2), where:
            - N : Number of samples
            - n_dims : Number of dimensions (features)
            - 2 : Lower and upper bounds for each dimension

        Raises:
        -------
        ValueError:
            If the input dataset does not match the expected shape (N, n_dims, 2).
            If `n_dims == 1`, since PCA-based initialization requires at least 2 dimensions.
        """

        # Validate input shape
        n_samples, dims, two_ = intervals.shape
        if dims != self.n_dims or two_ != 2:
            raise ValueError(f"Data shape must be (N, {self.n_dims}, 2). Got {intervals.shape}.")

        # 1) Compute interval centers: Shape (N, n_dims)
        centers = 0.5 * (intervals[..., 0] + intervals[..., 1])  # Averaging lower and upper bounds

        # 2) Perform PCA
        # Step a) Compute mean
        mean_c = centers.mean(axis=0)  # Shape (n_dims,)

        # Step b) Compute covariance matrix
        X = centers - mean_c  # Centering data, Shape (N, n_dims)
        cov_ = np.cov(X, rowvar=False)  # Shape (n_dims, n_dims)

        # Step c) Eigen decomposition
        eigvals, eigvecs = np.linalg.eig(cov_)

        # Step d) Sort eigenvalues and pick the top 2 principal components
        idx_sorted = np.argsort(-eigvals)  # Descending order
        v1 = eigvecs[:, idx_sorted[0]]  # First principal component (Shape: n_dims,)
        v2 = eigvecs[:, idx_sorted[1]]  # Second principal component (Shape: n_dims,)

        # Ensure that PCA-based initialization is meaningful
        if self.n_dims == 1:
            raise ValueError("PCA initialization requires at least 2 dimensions. Try random initialization instead.")

        # 3) Map (i, j) positions in the SOM grid to a coordinate in [-1,1]^2 → (c1, c2)
        #    - `c1_vals` maps `i` values from 0 to (x-1) linearly into [-1,1]
        #    - `c2_vals` maps `j` values from 0 to (y-1) linearly into [-1,1]
        #    - The neuron center in feature space is reconstructed as:
        #      center = mean_c + c1 * v1 + c2 * v2

        c1_vals = np.linspace(-1, 1, self.x)  # Shape (x,)
        c2_vals = np.linspace(-1, 1, self.y)  # Shape (y,)

        for i in range(self.x):
            for j in range(self.y):
                c1 = c1_vals[i]
                c2 = c2_vals[j]

                # Compute reconstructed center in feature space
                center_ij = mean_c + c1 * v1 + c2 * v2  # Shape (n_dims,)

                # 4) Assign the same center value to both lower and upper bounds
                for d in range(self.n_dims):
                    val_d = center_ij[d]
                    self.prototypes[i, j, d, 0] = val_d  # Lower bound
                    self.prototypes[i, j, d, 1] = val_d  # Upper bound (Collapsed interval)

    # -------------- decay functions ---------------
    def _inverse_decay_to_zero(self, learning_rate, t, max_iter):
        """Learning rate decreases asymptotically to zero."""
        C = max_iter / 100.0
        return learning_rate * C / (C + t)

    def _linear_decay_to_zero(self, learning_rate, t, max_iter):
        """Learning rate decreases linearly to zero."""
        return learning_rate * (1 - t / max_iter)

    def _asymptotic_decay(self, learning_rate, t, max_iter):
        """Learning rate asymptotically decreases to 1/3 of its original value."""
        return learning_rate / (1 + t / (max_iter / 2))
    
    def _inverse_decay_to_one(self, sigma, t, max_iter):
        """Sigma decays asymptotically to 1."""
        C = (sigma - 1) / max_iter
        return sigma / (1 + (t * C))
    
    def _linear_decay_to_one(self, sigma, t, max_iter):
        """Sigma decreases linearly to 1."""
        return sigma + (t * (1 - sigma) / max_iter)
    
    def _linear_interpolate(self, sigma, t, max_iter):
        return sigma + (t/float(max_iter))*(self.sigma_final - sigma)

    # -------------- neighborhood functions -------------
    def _gaussian(self, c, sigma):
        """
        Gaussian neighborhood function centered at BMU `c`.

        Parameters:
            c (tuple): Coordinates of the BMU (row, col).
            sigma (float): Current neighborhood radius.

        Returns:
            np.ndarray: Neighborhood influence matrix.
        """
        d = 2 * sigma * sigma
        ax = np.exp(-np.power(self._locations[:, :, 0] - c[0], 2) / d)
        ay = np.exp(-np.power(self._locations[:, :, 1] - c[1], 2) / d)
        return (ax * ay)
    
    def _mexican_hat(self, c, sigma):
        """
        Mexican hat (Ricker wavelet) neighborhood function.

        Parameters:
            c (tuple): Coordinates of the BMU (row, col).
            sigma (float): Current neighborhood radius.

        Returns:
            np.ndarray: Neighborhood influence matrix.
        """
        d = 2 * sigma * sigma
        p = np.power(self._locations[:, :, 0] - c[0], 2) + np.power(self._locations[:, :, 1] - c[1], 2)
        return np.exp(-p / d) * (1 - 2 / d * p)
    
    def _bubble(self, c, sigma):
        """
        Bubble neighborhood function with constant influence within a fixed radius.

        Parameters:
            c (tuple): Coordinates of the BMU (row, col).
            sigma (float): Current neighborhood radius.

        Returns:
            np.ndarray: Neighborhood influence matrix.
        """
        ax = np.logical_and(self._locations[:, :, 0] > c[0] - sigma,
                            self._locations[:, :, 0] < c[0] + sigma)
        ay = np.logical_and(self._locations[:, :, 1] > c[1] - sigma,
                            self._locations[:, :, 1] < c[1] + sigma)
        return np.outer(ax, ay).astype(float)
    
    def _triangle(self, c, sigma):
        """
        Triangular neighborhood function.

        Parameters:
            c (tuple): Coordinates of the BMU (row, col).
            sigma (float): Current neighborhood radius.

        Returns:
            np.ndarray: Neighborhood influence matrix.
        """
        triangle_x = (-np.abs(self._locations[:, :, 0] - c[0])) + sigma
        triangle_y = (-np.abs(self._locations[:, :, 1] - c[1])) + sigma
        triangle_x[triangle_x < 0] = 0.0
        triangle_y[triangle_y < 0] = 0.0
        return np.outer(triangle_x, triangle_y)
    
    def _distance_from_weights(self, intervals):
        """
        For each sample => compute distance to each neuron => shape(N, x*y)
        """
        N = intervals.shape[0]
        dist_mat = np.zeros((N, self.x*self.y))
        idx=0
        for s in range(N):
            sample = intervals[s]
            col=0
            for i in range(self.x):
                for j in range(self.y):
                    dist_mat[s,col] = self._activation_distance(sample, self.prototypes[i,j])
                    if self.isSim:
                        dist_mat[s,col] = 1 - dist_mat[s,col]
                    col+=1
        return dist_mat
    
    def winner(self, interval):
        """
        for incremental => find best (i,j)
        """
        min_dist=1e15
        best=(0,0)
        for i in range(self.x):
            for j in range(self.y):
                dist_ij = self._activation_distance(interval, self.prototypes[i,j])
                if self.isSim:
                    dist_ij = 1 - dist_ij
                if dist_ij<min_dist:
                    min_dist=dist_ij
                    best=(i,j)
        return best
    
    def train_incremental(self, intervals, sample_mode='random', verbose=False):
        """
        Incremental (online) training for interval SOM.

        Parameters
        ----------
        data : np.ndarray
            Training data of shape (N, n_dims, 2).

        sample_mode : str, optional
            How to pick the sample each iteration.
            - 'random': pick a random sample from data (default).
            - 'sequential': pick sample in sequential order (t % N).

        verbose : bool, optional
            If True, print intermediate quantization error every 10 steps.
        """
        N = intervals.shape[0]
        seq_idx = 0  # for 'sequential' mode

        for t in range(self.total_iterations):
            # 1. Pick sample according to sample_mode
            if sample_mode == 'random':
                rand_idx = self._rng.randint(N)
                sample = intervals[rand_idx]
            elif sample_mode == 'sequential':
                sample = intervals[seq_idx]
                seq_idx = (seq_idx + 1) % N
            else:
                raise ValueError(f"Unknown sample_mode='{sample_mode}'. Use 'random' or 'sequential'.")

            # 2. Find BMU
            bmu_ij = self.winner(sample)

            # 3. Compute new learning rate
            lr_t = self._learning_rate_decay_function(self.learning_rate, t, self.total_iterations)

            # 4. Compute new sigma
            sig_t = self._sigma_decay_function(self.sigma_init, t, self.total_iterations)

            # 5. Compute neighbor matrix
            neigh = self._neighborhood(bmu_ij, sig_t)

            # 6. Update prototypes with partial "pull" on intervals
            for i in range(self.x):
                for j in range(self.y):
                    influence = neigh[i,j] * lr_t
                    if influence > 1e-9:
                        old_ij = self.prototypes[i,j]  # shape (n_dims,2)
                        delta = sample - old_ij        # shape (n_dims,2)
                        self.prototypes[i,j] = old_ij + influence * delta
                        # ensure lower <= upper
                        self.prototypes[i,j] = np.sort(self.prototypes[i,j], axis=-1)

            # (Optional) print progress
            if verbose and (t % 10 == 0):
                qe = self.quantization_error(intervals)
                print(f"[Incremental-{sample_mode}] Iter {t}, QE={qe:.4f}")
    
    def train_batch(self, intervals: np.ndarray, verbose=False):
        """
        1) for t in 0..total_iterations-1:
           - dist_mat => BMU
           - for each neuron => do WeightedMedian update
        """
        N = intervals.shape[0]
        # precompute m_i^d = (a_i^d + b_i^d)/2,  l_i^d = (b_i^d - a_i^d)/2
        intervals_m = 0.5*(intervals[...,0]+intervals[...,1]) # shape(N,n_dims)
        intervals_l = 0.5*(intervals[...,1]-intervals[...,0]) # shape(N,n_dims)
        for t in range(self.total_iterations):
            # compute sigma
            sig_t = self._sigma_decay_function(self.sigma_init, t, self.total_iterations)
            # compute dist => bmu
            dist_mat = self._distance_from_weights(intervals)
            bmu_flat = np.argmin(dist_mat, axis=1)    # shape(N,) => best neuron in flatten (i*y + j)
            # compute neighbor weight => shape(x,y,N)
            neighbor_w = np.zeros((self.x, self.y, N))
            for s in range(N):
                bf = bmu_flat[s]
                c_i = bf//self.y
                c_j = bf%self.y
                # high-level => neighbor
                nb_mat = self._neighborhood( (c_i,c_j), sig_t )  # shape(x,y)
                neighbor_w[:,:,s] = nb_mat

            # update prototypes => WeightedMedian
            self._update_prototypes_batch(intervals, intervals_m, intervals_l, neighbor_w)

            if verbose and t%10==0:
                qe = self.quantization_error(intervals)
                print(f"[Batch] Iter {t}, QE={qe:.4f}")
    
    def _update_prototypes_batch(self, data, data_m, data_l, neighbor_w):
        """
        Weighted median update from Chavent & Lechevallier approach:
         for each neuron (i,j), each dimension d:
           m_k^d = WeightedMedian of data_m[:,d], weights= sum( neighbor_w[i,j,s] )
           l_k^d = WeightedMedian of data_l[:,d], same weights
         => [u_k^d, v_k^d] = [m_k^d - l_k^d, m_k^d + l_k^d]
        neighbor_w[i,j,s] => h_{(i,j),c(s)}
        """
        N = data.shape[0]
        for i in range(self.x):
            for j in range(self.y):
                # shape(N,) => for each sample s => w_s
                w_s = neighbor_w[i,j,:]
                w_sum = w_s.sum()
                # skip if w_sum=0 => no update
                if w_sum<1e-15:
                    continue
                for d in range(self.n_dims):
                    # Weighted median of data_m[:,d]
                    median_m = self._weighted_median(data_m[:,d], w_s)
                    median_l = self._weighted_median(data_l[:,d], w_s)
                    self.prototypes[i,j,d,0] = median_m - median_l
                    self.prototypes[i,j,d,1] = median_m + median_l
                    if self.prototypes[i,j,d,0]>self.prototypes[i,j,d,1]:
                        self.prototypes[i,j,d] = np.sort(self.prototypes[i,j,d])
    
    def _weighted_median(self, values, weights):
        """
        Weighted median => sort by values => find point cumsum >= half
        """
        idx_sort = np.argsort(values)
        vs = values[idx_sort]
        ws = weights[idx_sort]
        wsum = ws.sum()
        half = 0.5*wsum
        cumsum=0
        for i in range(len(vs)):
            cumsum += ws[i]
            if cumsum>=half:
                return vs[i]
        return vs[-1]
    
    def quantization(self, data):
        """
        Return the codebook intervals for each sample => shape(N, n_dims,2).
        """
        dist_mat = self._distance_from_weights(data)
        winners_idx = np.argmin(dist_mat, axis=1)
        coords_i = winners_idx//self.y
        coords_j = winners_idx%self.y
        N = data.shape[0]
        result = np.zeros((N, self.n_dims, 2))
        for s in range(N):
            i,j = coords_i[s], coords_j[s]
            result[s] = self.prototypes[i,j]
        return result

    def quantization_error(self, data):
        """
        average distance to BMU
        """
        dist_mat = self._distance_from_weights(data)
        mins = np.min(dist_mat, axis=1)
        return np.mean(mins)
    
    def topographic_error(self, intervals):
        """Returns the topographic error computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.
        
        A sample for which these two nodes are not adjacent counts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.
        
        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples.
        
        Parameters:
        -----------
        intervals : np.ndarray
            Input interval data of shape (N, n_dims, 2)
            
        Returns:
        --------
        float
            The topographic error value (between 0 and 1)
        """
        # Check input shape
        n_samples, dims, two_ = intervals.shape
        if dims != self.n_dims or two_ != 2:
            raise ValueError(f"Data shape must be (N, {self.n_dims}, 2). Got {intervals.shape}.")
        
        # Check if map is too small for meaningful topographic error
        total_neurons = self.x * self.y
        if total_neurons == 1:
            warn('The topographic error is not defined for a 1-by-1 map.')
            return np.nan
        
        # Call the appropriate implementation based on topology
        if self.topology == 'hexagonal':
            return self._topographic_error_hexagonal(intervals)
        else:
            return self._topographic_error_rectangular(intervals)

    def _topographic_error_hexagonal(self, intervals):
        """Return the topographic error for hexagonal grid"""
        # Get the distances from each sample to all neurons
        dist_mat = self._distance_from_weights(intervals)
        
        # Get indices of the best 2 matching units for each sample
        b2mu_inds = np.argsort(dist_mat, axis=1)[:, :2]
        
        # Convert flat indices to euclidean coordinates
        b2mu_coords = []
        for bmu in b2mu_inds:
            # First BMU coordinates
            bmu1_i, bmu1_j = bmu[0] // self.y, bmu[0] % self.y
            # Apply hexagonal offset to even/odd rows
            bmu1_y = bmu1_j + 0.5 if bmu1_i % 2 == 1 else bmu1_j
            bmu1_coords = np.array([bmu1_i, bmu1_y])
            
            # Second BMU coordinates
            bmu2_i, bmu2_j = bmu[1] // self.y, bmu[1] % self.y
            # Apply hexagonal offset to even/odd rows
            bmu2_y = bmu2_j + 0.5 if bmu2_i % 2 == 1 else bmu2_j
            bmu2_coords = np.array([bmu2_i, bmu2_y])
            
            b2mu_coords.append([bmu1_coords, bmu2_coords])
        
        b2mu_coords = np.array(b2mu_coords)
        
        # Check if BMUs are neighbors (distance ≈ 1 in hexagonal grid)
        b2mu_neighbors = [np.isclose(1.0, np.linalg.norm(bmu1 - bmu2))
                        for bmu1, bmu2 in b2mu_coords]
        
        # Topographic error = 1 - proportion of neighboring BMUs
        te = 1.0 - np.mean(b2mu_neighbors)
        return te

    def _topographic_error_rectangular(self, intervals):
        """Return the topographic error for rectangular grid"""
        t = 1.42  # Threshold for considering neurons as non-adjacent
        
        # Get the distances from each sample to all neurons
        dist_mat = self._distance_from_weights(intervals)
        
        # Get indices of the best 2 matching units for each sample
        b2mu_inds = np.argsort(dist_mat, axis=1)[:, :2]
        
        # Convert flat indices to grid coordinates
        b2mu_i = np.zeros((len(intervals), 2), dtype=int)
        b2mu_j = np.zeros((len(intervals), 2), dtype=int)
        
        for s in range(len(intervals)):
            b2mu_i[s, 0] = b2mu_inds[s, 0] // self.y
            b2mu_j[s, 0] = b2mu_inds[s, 0] % self.y
            b2mu_i[s, 1] = b2mu_inds[s, 1] // self.y
            b2mu_j[s, 1] = b2mu_inds[s, 1] % self.y
        
        # Calculate coordinate differences between first and second BMU
        dxdy = np.hstack([
            np.abs(b2mu_i[:, 0] - b2mu_i[:, 1]).reshape(-1, 1),
            np.abs(b2mu_j[:, 0] - b2mu_j[:, 1]).reshape(-1, 1)
        ])
        
        # Calculate Euclidean distance between BMUs
        distance = np.linalg.norm(dxdy, axis=1)
        
        # Return proportion of samples where BMUs are not adjacent
        return np.mean(distance > t)

    def get_prototypes(self):
        """
        return prototypes
        """
        return self.prototypes

    def get_neuron_assignments(self, data, return_indices=False):
        """
        Maps each neuron (i, j) to the samples assigned to it.

        Parameters:
        -----------
        data : array-like, shape (N, n_dims, 2)
            The input data containing N samples, each with n_dims features.
        
        return_indices : bool, optional (default=False)
            - If False, stores the actual sample data in the output dictionary.
            - If True, stores the indices of samples instead.

        Returns:
        --------
        assignment_map : dict
            A dictionary where:
            - If return_indices=False: { (i, j): [sample_1, sample_2, ...] }
            - If return_indices=True:  { (i, j): [index_1, index_2, ...] }
        """
        # Ensure data shape is valid, e.g., (N, n_dims, 2)
        assignment_map = defaultdict(list)

        for i, sample in enumerate(data):
            bmu_pos = self.winner(sample)  # Find the Best Matching Unit (BMU), returns (row, col)
            
            if return_indices:
                assignment_map[bmu_pos].append(i)  # Store sample index
            else:
                assignment_map[bmu_pos].append(sample)  # Store actual sample data

        return assignment_map
    
    def neuron_label_map(self, data, labels, return_mode='dominant'):
        """
        Maps each neuron (i, j) to a label summary based on assigned samples.

        Parameters:
        -----------
        data : array-like, shape (N, n_dims, 2)
            The input data containing N samples.
        
        labels : array-like, shape (N,)
            Labels corresponding to each sample in `data`.

        return_mode : str, optional (default='dominant')
            - 'dominant' : Returns the most frequent label assigned to each neuron.
            - 'counter' : Returns a `Counter` object with the count of all labels assigned.

        Returns:
        --------
        neuron_label_map : dict
            A dictionary where:
            - If return_mode='dominant': { (i, j): most_frequent_label }
            - If return_mode='counter': { (i, j): Counter({label_1: count_1, label_2: count_2, ...}) }
        """

        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length.")

        # Dictionary mapping each neuron (i, j) to the list of assigned labels
        label_map = defaultdict(list)

        # Assign each sample's label to its BMU (Best Matching Unit)
        for sample, lab in zip(data, labels):
            bmu = self.winner(sample)  # Get the neuron (i, j) to which the sample is assigned
            label_map[bmu].append(lab)

        # Aggregate labels for each neuron
        output = {}
        for neuron_pos, label_list in label_map.items():
            c = Counter(label_list)  # Count occurrences of each label

            if return_mode == 'dominant':
                # Return the most frequent label
                most_label, _count = c.most_common(1)[0]
                output[neuron_pos] = most_label
            elif return_mode == 'counter':
                # Return the full label distribution
                output[neuron_pos] = c
            else:
                raise ValueError("return_mode must be 'dominant' or 'counter'.")

        return output