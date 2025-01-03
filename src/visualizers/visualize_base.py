import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from einops import rearrange
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from src.visualizers.visualize_reps import (
    make_movie_of_reps,
    make_oriented_area_movie,
    make_spectrogram_movie,
)


class RepresentationVisualizer:
    """
    A class to visualize different representations of spectral data.

    Attributes:
        cfg (dict): Configuration dictionary for visualization settings.
        rep_dir (str): Directory where the representation files will be saved.
        exp_save_name (str): Name used for saving the experiment files.

    Methods:
        visualize_spectrogram(spectral_rep, eigenvalues):
            Visualizes a spectrogram representation of the spectral data.
            Args:
                spectral_rep (ndarray): The spectral representation data.
                eigenvalues (ndarray): The eigenvalues associated with the spectral representation.
            Raises:
                ValueError: If eigenvalues are None.

        visualize_dim_reduction(spectral_rep, dim_red_method):
            Visualizes a dimensionally reduced representation of the spectral data.
            Args:
                spectral_rep (ndarray): The spectral representation data.
                dim_red_method (str): The method used for dimensionality reduction.
    """

    def __init__(self, visualization_cfg, rep_dir, exp_save_name):
        self.cfg = visualization_cfg
        self.rep_dir = rep_dir
        self.exp_save_name = exp_save_name

    def visualize_spectrogram(self, spectral_rep, eigenvalues):
        """
        Visualize the spectrogram representation of the given spectral data.

        Parameters:
        spectral_rep (numpy.ndarray): The spectral representation data.
        eigenvalues (numpy.ndarray): The eigenvalues corresponding to the spectral representation.

        Raises:
        ValueError: If eigenvalues are None.

        Notes:
        - If the configuration specifies to filter zero eigenvalues, the first eigenvalue and its corresponding spectral representation will be removed.
        - The spectrogram movie will be saved to a file in the specified directory with the experiment save name.

        """
        if eigenvalues is None:
            raise ValueError("Spectrogram visualization requires eigenvalues.")

        if self.cfg.representation.filter_zero_eigenvalues:
            print("Filtering zero eigenvalues")
            spectral_rep = spectral_rep[:, 1:, :]
            eigenvalues = eigenvalues[1:]

        make_spectrogram_movie(
            spectral_rep,
            eigenvalues,
            fps=self.cfg.fps,
            output_file=os.path.join(self.rep_dir, f"spectrogram_{self.exp_save_name}.mp4"),
            title="Spectrogram Representation",
            use_abs=False,
        )

    def visualize_oriented_area_movie(self, spectral_rep):
        """
        Visualizes the oriented area representation of the spectral data.

        This method creates a movie of the oriented area representation of the spectral data.

        Args:
            spectral_rep (array-like): The spectral representation to visualize.

        Returns:
            None
        """
        make_oriented_area_movie(
            spectral_rep,
            fps=self.cfg.fps,
            output_file=os.path.join(self.rep_dir, f"oriented_area_{self.exp_save_name}.mp4"),
            title="Oriented Area Representation",
        )

    def visualize_dim_reduction(self, spectral_rep, dim_red_method):
        """
        Visualizes the dimensionality reduction of the spectral representation.

        This method compresses the spectral representation using the specified
        dimensionality reduction method and creates a movie of the resulting
        representations.

        Args:
            spectral_rep (array-like): The spectral representation to be compressed.
            dim_red_method (str): The method to use for dimensionality reduction.

        Returns:
            None
        """
        compressed_rep = compress_spectral_rep(
            spectral_rep,
            dim_red_method=dim_red_method,
            **{k: v for k, v in self.cfg.representation.items() if k != "dim_red_method"},
        )
        output_file = os.path.join(self.rep_dir, f"{dim_red_method}_{self.exp_save_name}.mp4")
        make_movie_of_reps(
            compressed_rep,
            fps=self.cfg.fps,
            output_file=output_file,
            title=f"{self.cfg.representation.get('title', 'Representation Trajectory')} ({dim_red_method})",
        )


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

    if kwargs.get("filter_zero_eigenvalues", False):
        print("Filtering zero eigenvalues")
        low_freq_spectral_rep = low_freq_spectral_rep[:, 1:]

    # Perform dimensionality reduction
    if dim_red_method == "MDS":

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

        pca = PCA(n_components=2)
        low_freq_spectral_rep = rearrange(low_freq_spectral_rep, "t f d -> t (f d)")
        compressed_spectral_rep = pca.fit_transform(low_freq_spectral_rep)
    else:
        raise ValueError("Invalid dimensionality reduction method")

    return compressed_spectral_rep
