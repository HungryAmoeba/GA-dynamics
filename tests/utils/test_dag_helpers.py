import networkx as nx
import numpy as np
import pytest

from src.utils.dag_helpers import convert_tree_to_dag


def test_convert_tree_to_dag_path():
    tree = nx.path_graph(5).adj

    directed_adjacency = convert_tree_to_dag(tree)
    assert directed_adjacency == {0: {1: {}}, 1: {2: {}}, 2: {3: {}}, 3: {4: {}}, 4: {}}


def test_convert_tree_to_dag_binary_tree():
    tree = nx.balanced_tree(2, 2).adj
    print(tree)

    directed_adjacency = convert_tree_to_dag(tree)
    print(directed_adjacency)
    assert directed_adjacency == {
        0: {1: {}, 2: {}},
        1: {3: {}, 4: {}},
        3: {},
        4: {},
        2: {5: {}, 6: {}},
        5: {},
        6: {},
    }
