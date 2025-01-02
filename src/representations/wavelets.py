# TODO: implement this representation method
import networkx as nx
import numpy as np
from einops import rearrange
from pygsp import filters, graphs

from src.representations.representation_base import Representation

# from representation_base import Representation


class SpectralWaveletRepresentation(Representation):
    """Class for generating a spectral wavelet representation of a trajectory."""

    def generate(self):
        """
        Generate spectral wavelet representation.
        """
        # Pass kwargs from cfg
        kwargs = {
            k: v
            for k, v in self.cfg.representation.items()
            if k not in ["rep_type", "dim_red_methods"]
        }

        self.spectral_rep = get_spectral_wavelet_representation(
            self.adjacency, self.trajectory, **kwargs
        )
        self.save(self.spectral_rep)

        # For compatibility, we'll set an empty eigenvalues array
        self.eigenvalues = None

        return self.spectral_rep, self.eigenvalues


def get_spectral_wavelet_representation(adjacency, trajectory, **kwargs):
    """
    Returns a wavelet representation of the trajectory in the specified graph wavelet basis

    Inputs:
        adjacency (dict): dictionary of node to list of neighbors
        trajectory (np.array): array of shape T x N x D where T is the number of frames, N is the number of nodes, and D is the dimension of the positions
        **kwargs: additional arguments for the wavelet representation

    Returns:
        spectral_representation (np.array): array of shape T x K x D representing the trajectory in the wavelet basis

    """

    # todo, maybe use gsp
    # create a graph from adjacency
    # if adjacency is a dictionary, convert it to an adjacency matrix
    if isinstance(adjacency, dict):
        adjacency = nx.to_numpy_array(nx.from_dict_of_lists(adjacency))
    G = graphs.Graph(adjacency)

    # by default use Abspline wavelets
    wavelet_type = kwargs.get("wavelet_type", "Abspline")

    if wavelet_type == "Abspline":
        Nf = kwargs.get("Nf", 6)
        wavelet = filters.Abspline(G, Nf=Nf)
    elif wavelet_type == "MexicanHat":
        Nf = kwargs.get("Nf", 6)
        wavelet = filters.MexicanHat(G, Nf=Nf)

    method = kwargs.get("method", "chebyshev")
    # filter the trajectory
    # filter isn't parallelized, so run it on each signal separately

    # need to loop over T and D

    wavelet_reps = np.zeros((trajectory.shape[0], trajectory.shape[1], Nf, trajectory.shape[2]))
    for d in range(trajectory.shape[2]):
        for t in range(trajectory.shape[0]):
            wavelet_rep = wavelet.filter(trajectory[t, :, d], method=method)
            wavelet_reps[t, :, :, d] = wavelet_rep

    # rearrange
    wavelet_reps = rearrange(wavelet_reps, "t n f d -> t (n f) d")
    return wavelet_reps
