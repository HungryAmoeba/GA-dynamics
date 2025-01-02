import networkx as nx
import numpy as np
import pytest

from src.representations.wavelets import get_spectral_wavelet_representation


def test_get_spectral_wavelet_representation():
    G = nx.fast_gnp_random_graph(20, 0.2)
    adjacency = nx.to_dict_of_lists(G)
    trajectory = np.random.randn(100, 20, 3)
    Nf = 6
    extra_args = {"wavelet_type": "Abspline", "Nf": Nf}
    spectral_rep = get_spectral_wavelet_representation(adjacency, trajectory, **extra_args)
    assert spectral_rep.shape == (100, Nf * 20, 3)
