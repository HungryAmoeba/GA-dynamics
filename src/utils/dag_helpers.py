import networkx as nx
import numpy as np


def convert_tree_to_dag(adjacency_dict, root_node=0):
    """
    Converts a tree represented as an adjacency list to a directed acyclic graph (DAG).
    Parameters:
    adjacency (networkx.classes.coreviews.AdjacencyView): An adjacency view representing the graph structure.
    root_node (int, optional): The root node of the tree. Default is 0.
    Returns:
    directed_adjacency (dict): An adjacency dictionary representing the DAG
    """
    tree = nx.DiGraph()
    for node, neighbors in adjacency_dict.items():
        for neighbor in neighbors:
            tree.add_edge(node, neighbor)

    directed_adjacency = nx.dfs_tree(tree, root_node).adj
    return directed_adjacency
