import numpy as np
from scipy.linalg import eigh, solve
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
# from sklearn.neighbors import NearestNeighbors

import sklearn.manifold

"""
The Locally Linear Embedding Algorithm
"""
def locally_linear_embedding(X, k_neighbors, t_dimensions, reg_factor=1e-3):
    """
    Parameters
    ----------
    X : numpy array
        input data, shape [n_samples, n_features], dtype must be numpy.float64.

    k_neighbors : integer
        number of nearest neighbors to consider for each point.

    t_dimensions : integer
        number of dimensions in the output data.

    reg_factor : float
        regularization factor, for the case k_neighbors > n_features.

    Return
    -------
    Y : numpy array
        dimension-reduced data, shape [n_samples, t_dimensions].
    """

    # check X data: must be a 2-D numpy array, must be np.float64
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("Your input data is NOT a 2-D numpy array")
    if X.dtype != np.float64:
        raise TypeError("Your input data is NOT type: numpy.float64")

    n_samples, n_features = X.shape

    # check Parameters
    if t_dimensions > n_features or t_dimensions < 1:
        raise ValueError("Your input does NOT satisfy: 1 <= output dimension <= input dimension")
    if k_neighbors >= n_samples or k_neighbors <= 0:
        raise ValueError("Your input does NOT satisfy: 0 < k_neighbors < n_samples")

    k_take = k_neighbors + 1

    # step 1, compute the k nearest neighbors of each point
    idx = np.argpartition(cdist(X, X), (1, k_take), axis=0)[1:k_take].T

    # step 1, compute the k-nn of each point (using scikit-learn)
    # knn = NearestNeighbors(k_take).fit(X)
    # idx = knn.kneighbors(X, return_distance=False)[:, 1:]

    Z = X[idx].transpose(0, 2, 1) # own implementation

    # step 2, compute co-variance matrix and then the weights
    # the Matrix to contain the Weights:
    Weights = np.empty((n_samples, k_neighbors), dtype=X.dtype)
    # the ALL-ONE vector:
    Ones = np.ones(k_neighbors, dtype=X.dtype)

    for i, P in enumerate(Z):

        # each neighbors - this point
        D = P.T - X[i]

        # Cov is the local covariance matrix
        Cov = np.dot(D, D.T)

        # regularization
        # Cov = Cov + eye(K,K) * factor * (Cov.trace > 0 ? Cov.trace : 1)
        r = reg_factor
        trace = np.trace(Cov)
        if trace > 0:
            r *= trace
        Cov.flat[::k_take] += r # add the reg factor to the main diagonal of Cov

        # find the weights of each neighbors
        w = solve(Cov, Ones, overwrite_a=True, assume_a='pos')

        # make sum(w) = 1
        Weights[i, :] = w / np.sum(w)

    # put the Weights in to a sparse matrix
    W = csr_matrix(
            (Weights.ravel(),
            idx.ravel(),
            np.arange(0, n_samples * k_neighbors + 1, k_neighbors)),
            shape = (n_samples, n_samples) )

    # Step 3 compute M = (I-W)'(I-W)
    M = (W.T * W - W - W.T).toarray()
    M.flat[::n_samples + 1] += 1

    # Step 4 compute the eigen_values and eigen_vectors of M
    eigen_values, eigen_vectors = eigh(M, eigvals=(1, t_dimensions), overwrite_a=True)

    # Step 5 the 2nd to the d+1'th eigen_vectors is the output
    return eigen_vectors[:, np.argsort(np.abs(eigen_values))]

def main():
    X = np.arange(10).reshape(5,2).astype(np.float64)
    print("The input data:")
    print(X) # for test only
    Y = locally_linear_embedding(X, 2, 2)
    print("The output data:")
    print(Y) # for test only

    Y = sklearn.manifold.locally_linear_embedding(X, 2, 2)
    print("The output data:")
    print(Y) # for test only

    # D1 = get_hd_dataset(5000)
    # D2 = get_swiss_roll_dataset(5000)
    # D3 = get_twin_peaks(5000)
    # D4 = get_helix_dataset(5000)
    # D5 = get_broken_swiss_roll_dataset(5000)
    #
    # print(D1[0])
    #
    # D2 = np.array(D2, dtype=np.float64)
    # D1 = np.array(D1, dtype=np.float64)
    #
    # Y = locally_linear_embedding(D1, 5, 5)
    #
    # print(Y[1])

if __name__ == '__main__':
    main()
