import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from einops import rearrange

from src.representations.representation_base import Representation
from src.visualizers.visualize_base import compress_spectral_rep


class LaplacianRepresentation(Representation):
    """
    LaplacianRepresentation is a class that provides a spectral representation of a trajectory in the Laplacian basis.
    Methods
    -------
    generate():
        This method computes the spectral representation using the adjacency matrix and trajectory,
        and optionally normalizes it based on the configuration settings. It saves the spectral
        representation and eigenvalues as attributes of the class and returns them.
    Attributes
    ----------
    spectral_rep : np.ndarray
        The spectral representation of the trajectory.
    eigenvalues : np.ndarray
        The eigenvalues associated with the spectral representation.
    """

    def generate(self):
        """
        Returns the spectral representation of the trajectory in the Laplacian basis.
        """
        spectral_rep_dict = get_spectral_representation(
            self.adjacency, self.trajectory, normalized=self.cfg.representation.use_normalized
        )
        self.spectral_rep = spectral_rep_dict["spectral_representation"]
        self.eigenvalues = spectral_rep_dict["eigenvalues"]
        self.save(self.spectral_rep)
        return self.spectral_rep, self.eigenvalues


def get_spectral_representation(adjacency, trajectory, normalized=False):
    """
    Returns the spectral representation of the trajectory in the Laplacian basis.

    Inputs:
        adjacency (dict): Dictionary of node to list of neighbors
        trajectory (np.array): Array of shape T x N x D (T: time, N: nodes, D: dimensions)
        normalized (bool): If True, use the normalized Laplacian

    Returns:
        dict: {
            'spectral_representation': np.array of shape T x N x D,
            'eigenvalues': np.array of eigenvalues,
            'eigenvectors': np.array of eigenvectors
        }
    """
    # Get the Laplacian matrix
    G = nx.from_dict_of_lists(adjacency)
    L = (
        nx.normalized_laplacian_matrix(G).toarray()
        if normalized
        else nx.laplacian_matrix(G).toarray()
    )

    # Get the eigenvalues and eigenvectors of the Laplacian matrix
    eigvals, eigvecs = np.linalg.eigh(L)

    # Project the trajectory into the spectral domain
    spectral_representation = np.einsum("tnd,nm->tmd", trajectory, eigvecs)

    return {
        "spectral_representation": spectral_representation,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
    }


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
    spectral_representation_dict = get_spectral_representation(adjacency, trajectory)
    # print(spectral_representation)
    spectral_representation = spectral_representation_dict["spectral_representation"]
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
