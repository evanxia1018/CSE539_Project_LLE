import numpy as np
from scipy.linalg import eigh, solve
from scipy.sparse import csr_matrix
import Dataset_Generator as dg
# from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import cdist


def locally_linear_embedding(
        X, k_neighbors, t_dimensions, reg_factor=1e-3, n_jobs=1):
    """Perform a Locally Linear Embedding analysis on the data.

    Read more in the :ref:`User Guide <locally_linear_embedding>`.

    Parameters
    ----------
    X : array-like
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array.

    k_neighbors : integer
        number of neighbors to consider for each point.

    t_dimensions : integer
        number of coordinates for the manifold.

    reg : float
        regularization constant, multiplies the trace of the local covariance
        matrix of the distances.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Returns
    -------
    Y : array-like, shape [n_samples, t_dimensions]
        Embedding vectors.

    squared_error : float
        Reconstruction error for the embedding vectors. Equivalent to
        ``norm(Y - W Y, 'fro')**2``, where W are the reconstruction weights.

    """

    # check X dtype: must be np.float64

    n_samples, d_in = X.shape

    if t_dimensions > d_in:
        raise ValueError("Your input does NOT satisfy: output dimension <= input dimension")
    if k_neighbors >= n_samples or k_neighbors <= 0:
        raise ValueError("Your input does NOT satisfy: 0 < k_neighbors < n_samples")

    k_take = k_neighbors + 1

    # step 1, compute the k nearest neighbors of each point
    dists = cdist(X, X)
    #print("The distance: \n" + str(dists))
    idx = np.argpartition(dists, (1, k_take), axis=0)[1:k_take].T
    #print("Own knn: \n" + str(idx))


    # step 1, compute the k-nn of each point
    #knn = NearestNeighbors(k_take, n_jobs=n_jobs).fit(X)
    #ind = knn.kneighbors(X, return_distance=False)[:, 1:]

    #print("\nKNN DIST: ")
    #print(knn.kneighbors(X, return_distance=True))
    #print("\n")

    #print("Neighbors Index: " + str(ind))

    Z = X[idx].transpose(0, 2, 1) # own implementation
    # Z = X[ind].transpose(0, 2, 1)

    #print(Z)  # for test only
    #print("Z.shape[2]: " + str(Z.shape[2]))
    # print(Z[0]) # for test only

    # step 2, compute co-variance matrix and then the weights
    Weights = np.empty((n_samples, k_neighbors), dtype=X.dtype) # the Matrix to contain the Weights
    Ones = np.ones(k_neighbors, dtype=X.dtype)  # the ALL-ONE vector

    for i, P in enumerate(Z):

        D = P.T - X[i]  # each neighbors - this point

        Cov = np.dot(D, D.T) # G is the local covariance matrix

        # regularization
        trace = np.trace(Cov)
        if trace > 0:
            R = reg_factor * trace
        else:
            R = reg_factor
        Cov.flat[::k_take] += R # add the reg factor to the main diagonal of G

        # find the weights of each neighbors
        w = solve(Cov, Ones, assume_a='pos')

        Weights[i, :] = w / np.sum(w) #

    # put the Weights in to a sparse matrix
    indptr = np.arange(0, n_samples * k_neighbors + 1, k_neighbors)
    # W = csr_matrix((B.ravel(), ind.ravel(), indptr), shape=(n_samples, n_samples))
    W = csr_matrix((Weights.ravel(), idx.ravel(), indptr), shape=(n_samples, n_samples))

    # Step 3 compute M = (I-W)'(I-W)
    M = (W.T * W - W.T - W).toarray()
    # print("M.shape:" + str(M.shape))
    M.flat[::n_samples + 1] += 1  #

    # Step 4 compte the eigen_values and eigen_vectors of M
    eigen_values, eigen_vectors = eigh(M, eigvals=(1, t_dimensions), overwrite_a=True)

    # Step 5 the first d+1 eigen_vectors is the output
    index = np.argsort(np.abs(eigen_values))
    return eigen_vectors[:, index], np.sum(eigen_values)


def main():
    # X = np.arange(9).reshape(9,1).astype(np.float64)
    X = np.array(dg.get_swiss_roll_dataset(5000), np.float64)
    print("The input data:")
    print(X) # for test only
    Y, error = locally_linear_embedding(X, 5, 2)
    print("The output data:")
    print(Y) # for test only
    print(error) # for test only

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
    # Y, error = locally_linear_embedding(D1, 5, 5)
    #
    # print(Y[1])

if __name__ == '__main__':
    main()
