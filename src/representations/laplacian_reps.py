import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from einops import rearrange


def get_spectral_representation(adjacency, trajectory):
    """
    This returns a spectral representation of the trajectory in the Laplacian basis.

    Inputs:
        adjacency (dict): dictionary of node to list of neighbors
        trajectory (np.array): array of shape T x N x D where T is the number of frames, N is the number of nodes, and D is the dimension of the positions
    """
    # Get the Laplacian matrix
    G = nx.from_dict_of_lists(adjacency)
    L = nx.laplacian_matrix(G).toarray()
    # Get the eigenvectors of the Laplacian matrix
    eigvals, eigvecs = np.linalg.eigh(L)
    spectral_representation = np.einsum("tnd,nm->tmd", trajectory, eigvecs)

    return spectral_representation


import networkx as nx
import numpy as np


def get_spectral_representation_2(adjacency, trajectory):
    """
    This returns a spectral representation of the trajectory in the Laplacian basis.

    Inputs:
        adjacency (dict): dictionary of node to list of neighbors
        trajectory (np.array): array of shape T x N x D where T is the number of frames, N is the number of nodes, and D is the dimension of the positions

    Returns:
        spectral_representation (np.array): array of shape T x N x D representing the trajectory in the Laplacian basis
    """
    # Get the Laplacian matrix
    G = nx.from_dict_of_lists(adjacency)
    L = nx.laplacian_matrix(G).toarray()

    # Get the eigenvectors of the Laplacian matrix
    eigvals, eigvecs = np.linalg.eigh(L)

    # Transform trajectory into the Laplacian basis
    # eigvecs.T is N x N (orthogonal matrix)
    # trajectory is T x N x D
    spectral_representation = np.einsum("ij,tjd->tid", eigvecs.T, trajectory)

    return spectral_representation


def compress_spectral_rep(spectral_rep, filter_high_freq=0.1, dim_red_method="MDS", **kwargs):
    """
    This function compresses the spectral representation of a trajectory by removing high frequency components.

    Inputs:
        spectral_rep (np.array): array of shape T x N x D representing the trajectory in the Laplacian basis
        filter_high_freq (float): the fraction of high frequency components to remove
        dim_red_method (str): the dimensionality reduction method to use

    Returns:
        compressed_spectral_rep (np.array): array of shape T x N x D representing the compressed trajectory in the Laplacian basis
    """
    # Get the number of high frequency components to remove
    num_high_freq = int((1 - filter_high_freq) * spectral_rep.shape[1])
    # Remove high frequency components
    low_freq_spectral_rep = spectral_rep[:, :num_high_freq]

    # Filter zero eigenvalues for translation invariance
    import pdb

    pdb.set_trace()
    if kwargs.get("filter_zero_eigenvalues", False):
        print("Filtering zero eigenvalues")
        low_freq_spectral_rep = low_freq_spectral_rep[:, 1:]

    # Perform dimensionality reduction
    if dim_red_method == "MDS":
        from sklearn.manifold import MDS

        mds = MDS(n_components=2)
        # see if kwargs has a parameter to use_distance_matrix
        use_distance_matrix = kwargs.get("use_distance_matrix", False)
        if use_distance_matrix:
            mds = MDS(n_components=2, dissimilarity="precomputed")
            # get the dissimilarity matrix from the low_freq_spectral_rep using euclidean distance
            # recall that low_freq_spectral_rep has shape T x N x D, we want dissimilarity matrix of shape T x T
            data = np.zeros((low_freq_spectral_rep.shape[0], low_freq_spectral_rep.shape[0]))
            for i in range(low_freq_spectral_rep.shape[0]):
                for j in range(low_freq_spectral_rep.shape[0]):
                    data[i, j] = np.linalg.norm(
                        low_freq_spectral_rep[i] - low_freq_spectral_rep[j]
                    )

        else:
            data = rearrange(low_freq_spectral_rep, "t f d -> t (f d)")

        # for now just reshape, don't do anything interesting with the three dimensions
        compressed_spectral_rep = mds.fit_transform(data)
    elif dim_red_method == "PCA":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        low_freq_spectral_rep = rearrange(low_freq_spectral_rep, "t f d -> t (f d)")
        compressed_spectral_rep = pca.fit_transform(low_freq_spectral_rep)
    else:
        raise ValueError("Invalid dimensionality reduction method")

    return compressed_spectral_rep


if __name__ == "__main__":
    # check the functions and see if the spectral representation is the same
    # get a path graph
    num_nodes = 10
    pos = {i: (i, 0, 0) for i in range(num_nodes)}
    G = nx.path_graph(num_nodes)
    adjacency = G.adj

    # get a trajectory
    T = 30
    trajectory = np.random.rand(T, num_nodes, 3)
    # print(trajectory)
    # get the spectral representation
    spectral_representation = get_spectral_representation(adjacency, trajectory)
    # print(spectral_representation)
    # get the spectral representation using the second function
    spectral_representation_2 = get_spectral_representation_2(adjacency, trajectory)
    # print(spectral_representation_2)
    # check if the two spectral representations are the same
    print(np.allclose(spectral_representation, spectral_representation_2))
    # compress the spectral representation using both MDS and PCA
    compressed_spectral_rep_MDS = compress_spectral_rep(
        spectral_representation, dim_red_method="MDS"
    )
    compressed_spectral_rep_PCA = compress_spectral_rep(
        spectral_representation, dim_red_method="PCA"
    )
    print(compressed_spectral_rep_MDS.shape)
    print(compressed_spectral_rep_PCA.shape)
    # test when using distance matrix for MDS
    compressed_spectral_rep_MDS = compress_spectral_rep(
        spectral_representation, dim_red_method="MDS", use_distance_matrix=True
    )
    print(compressed_spectral_rep_MDS.shape)
