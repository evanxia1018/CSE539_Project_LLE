import numpy as np
from scipy.linalg import eigh, solve
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# def barycenter_weights(X, Z, reg=1e-3):
#     """Compute barycenter weights of X from Y along the first axis
#
#     We estimate the weights to assign to each point in Y[i] to recover
#     the point X[i]. The barycenter weights sum to 1.
#
#     Parameters
#     ----------
#     X : array-like, shape (n_samples, n_dim)
#
#     Z : array-like, shape (n_samples, n_neighbors, n_dim)
#
#     reg : float, optional
#         amount of regularization to add for the problem to be
#         well-posed in the case of n_neighbors > n_dim
#
#     Returns
#     -------
#     B : array-like, shape (n_samples, n_neighbors)
#
#     Notes
#     -----
#     See developers note for more information.
#     """
#     # X = check_array(X, dtype=FLOAT_DTYPES)
#     # Z = check_array(Z, dtype=FLOAT_DTYPES, allow_nd=True)
#
#     n_samples, n_neighbors = X.shape[0], Z.shape[1]
#     B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
#     v = np.ones(n_neighbors, dtype=X.dtype)
#
#     # this might raise a LinalgError if G is singular and has trace
#     # zero
#     for i, A in enumerate(Z.transpose(0, 2, 1)):
#         C = A.T - X[i]  # broadcasting
#         G = np.dot(C, C.T)
#         trace = np.trace(G)
#         if trace > 0:
#             R = reg * trace
#         else:
#             R = reg
#         G.flat[::Z.shape[1] + 1] += R
#         w = solve(G, v, sym_pos=True)
#         B[i, :] = w / np.sum(w)
#     return B
#
#
# def barycenter_kneighbors_graph(X, n_neighbors, reg=1e-3, n_jobs=1):
#     """Computes the barycenter weighted graph of k-Neighbors for points in X
#
#     Parameters
#     ----------
#     X : array-like
#         Sample data, shape = (n_samples, n_features), in the form of a
#         numpy array.
#
#     n_neighbors : int
#         Number of neighbors for each sample.
#
#     reg : float, optional
#         Amount of regularization when solving the least-squares
#         problem. Only relevant if mode='barycenter'. If None, use the
#         default.
#
#     n_jobs : int, optional (default = 1)
#         The number of parallel jobs to run for neighbors search.
#         If ``-1``, then the number of jobs is set to the number of CPU cores.
#
#     Returns
#     -------
#     A : sparse matrix in CSR format, shape = [n_samples, n_samples]
#         A[i, j] is assigned the weight of edge that connects i to j.
#
#     See also
#     --------
#     sklearn.neighbors.kneighbors_graph
#     sklearn.neighbors.radius_neighbors_graph
#     """
#     knn = NearestNeighbors(n_neighbors + 1, n_jobs=n_jobs).fit(X)
#     n_samples = X.shape[0]
#
#     print(knn.kneighbors(X, return_distance=True)) # for test only
#
#     ind = knn.kneighbors(X, return_distance=False)[:, 1:]
#     print(ind) # for test only
#     print(X[ind]) # for test only
#     data = barycenter_weights(X, X[ind], reg=reg)
#     indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
#     return csr_matrix((data.ravel(), ind.ravel(), indptr),
#                       shape=(n_samples, n_samples))
#
#
# def null_space(M, k, k_skip=1):
#     """
#     Find the null space of a matrix M.
#
#     Parameters
#     ----------
#     M : {array, matrix, sparse matrix, LinearOperator}
#         Input covariance matrix: should be symmetric positive semi-definite
#
#     k : integer
#         Number of eigenvalues/vectors to return
#
#     k_skip : integer, optional
#         Number of low eigenvalues to skip.
#     """
#
#     if hasattr(M, 'toarray'):
#         M = M.toarray()
#     eigen_values, eigen_vectors = eigh(
#         M, eigvals=(k_skip, k + k_skip - 1), overwrite_a=True)
#     index = np.argsort(np.abs(eigen_values))
#     return eigen_vectors[:, index], np.sum(eigen_values)
#

import random
import math


def get_swiss_roll_dataset(numOfSamples):
    sample_list = []
    for x in range(0, numOfSamples):
        noise_test = random.random()
        if noise_test < 0.1:  # Add gaussian noise
            noise_1 = np.random.normal(0, 0.1)
            noise_2 = np.random.normal(0, 0.1)
            noise_3 = np.random.normal(0, 0.1)
            sample_list.append([noise_1, noise_2, noise_3])
            continue
        p_i = random.random()
        q_i = random.random()
        t_i = math.pi * 3 / 2 * (1 + 2 * p_i)
        x_i = [(t_i * math.cos(t_i)), t_i * math.sin(t_i), 30 * q_i]
        sample_list.append(x_i)
    return sample_list


def get_broken_swiss_roll_dataset(numOfSamples):
    sample_list = []
    for x in range(0, numOfSamples):
        noise_test = random.random()
        if noise_test < 0.1:  # Add gaussian noise
            noise_1 = np.random.normal(0, 0.1)
            noise_2 = np.random.normal(0, 0.1)
            noise_3 = np.random.normal(0, 0.1)
            sample_list.append([noise_1, noise_2, noise_3])
            continue
        while True:
            p_i = random.random()
            q_i = random.random()
            t_i = math.pi * 3 / 2 * (1 + 2 * p_i)
            if p_i >= (4 / 5) or p_i <= (2 / 5):
                break
        x_i = [(t_i * math.cos(t_i)), t_i * math.sin(t_i), 30 * q_i]
        sample_list.append(x_i)
    return sample_list


def get_helix_dataset(numOfSamples):
    sample_list = []
    for x in range(0, numOfSamples):
        noise_test = random.random()
        if noise_test < 0.1:  # Add gaussian noise
            noise_1 = np.random.normal(0, 0.1)
            noise_2 = np.random.normal(0, 0.1)
            noise_3 = np.random.normal(0, 0.1)
            sample_list.append([noise_1, noise_2, noise_3])
            continue
        p_i = random.random()
        x_i = [(2 + math.cos(8 * p_i)) * math.cos(p_i), (2 + math.cos(8 * p_i)) * math.sin(p_i), math.sin(8 * p_i)]
        sample_list.append(x_i)
    return sample_list


def get_twin_peaks(numOfSamples):
    sample_list = []
    for x in range(0, numOfSamples):
        noise_test = random.random()
        if noise_test < 0.1:  # Add gaussian noise
            noise_1 = np.random.normal(0, 0.1)
            noise_2 = np.random.normal(0, 0.1)
            noise_3 = np.random.normal(0, 0.1)
            sample_list.append([noise_1, noise_2, noise_3])
            continue
        p_i = random.random()
        q_i = random.random()
        x_i = [1 - 2 * p_i, math.sin(math.pi - 2 * math.pi * p_i), math.tanh(3 - 6 * q_i)]
        sample_list.append(x_i)
    return sample_list


def get_hd_dataset(numOfSamples):
    sample_list = []
    coef = []
    for x in range(0, 5):
        one_set_coef = []
        for y in range(0, 5):
            one_set_coef.append(random.random())
        coef.append(one_set_coef)
    for x in range(0, numOfSamples):
        d_1 = random.random()
        d_2 = random.random()
        d_3 = random.random()
        d_4 = random.random()
        d_5 = random.random()
        powers = []
        for y in range(0, 5):
            one_set_pow = [pow(d_1, random.random()), pow(d_2, random.random()), pow(d_3, random.random()), pow(d_4, random.random()), pow(d_5, random.random())]
            powers.append(one_set_pow)

        x_i = (np.mat(coef + powers) * np.mat([[d_1], [d_2], [d_3], [d_4], [d_5]])).transpose()
        x_i = x_i.tolist()
        sample_list.append(x_i[0])
    return sample_list


def locally_linear_embedding(
        X, n_neighbors, n_components, reg=1e-3, n_jobs=1):
    """Perform a Locally Linear Embedding analysis on the data.

    Read more in the :ref:`User Guide <locally_linear_embedding>`.

    Parameters
    ----------
    X : array-like
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array.

    n_neighbors : integer
        number of neighbors to consider for each point.

    n_components : integer
        number of coordinates for the manifold.

    reg : float
        regularization constant, multiplies the trace of the local covariance
        matrix of the distances.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Returns
    -------
    Y : array-like, shape [n_samples, n_components]
        Embedding vectors.

    squared_error : float
        Reconstruction error for the embedding vectors. Equivalent to
        ``norm(Y - W Y, 'fro')**2``, where W are the reconstruction weights.

    """

    # check X dtype: must be np.float64

    N, d_in = X.shape

    if n_components > d_in:
        raise ValueError("Not satisfy: output dimension <= input dimension")
    if n_neighbors >= N or n_neighbors <= 0:
        raise ValueError("Not satisfy: 0 < n_neighbors <= n_samples")

    # step 1, compute the k-nn of each point
    knn = NearestNeighbors(n_neighbors + 1, n_jobs=n_jobs).fit(X)
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]

    Z = X[ind]

    # step 2, 
    B = np.empty((N, n_neighbors), dtype=X.dtype) # the Matrix for the
    v = np.ones(n_neighbors, dtype=X.dtype)  # the ALL-ONE vector

    # this might raise a LinalgError if G is singular and has trace
    # zero
    for i, A in enumerate(Z.transpose(0, 2, 1)):
        C = A.T - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::Z.shape[1] + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)

    indptr = np.arange(0, N * n_neighbors + 1, n_neighbors)
    W = csr_matrix((B.ravel(), ind.ravel(), indptr), shape=(N, N))

    # Step 3 compute M = (I-W)'(I-W)
    M = (W.T * W - W.T - W).toarray()
    M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I

    # Step 4 compte the eigen_values and eigen_vectors of M
    eigen_values, eigen_vectors = eigh(M, eigvals=(1, n_components), overwrite_a=True)

    # Step 5 the first d+1 eigen_vectors is the output
    index = np.argsort(np.abs(eigen_values))
    return eigen_vectors[:, index], np.sum(eigen_values)


def main():
    X = np.arange(9).reshape(9,1).astype(np.float64)
    print("The input data:")
    print(X) # for test only
    Y, error = locally_linear_embedding(X, 2, 1)
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
