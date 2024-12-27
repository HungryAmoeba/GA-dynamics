import networkx as nx
import numpy as np


def get_trefoil_pos(num_nodes, upper_limit=5.5 * np.pi / 4, scale=1):
    """
    Generate positions for nodes along a trefoil knot.
    Parameters:
    num_nodes (int): The number of nodes to generate positions for.
    upper_limit (float, optional): The upper limit for the parameter t, which controls the length of the trefoil knot. Default is 5.5 * pi / 4.
    scale (float, optional): A scaling factor to adjust the size of the trefoil knot. Default is 1.
    Returns:
    dict: A dictionary where keys are node indices (0 to num_nodes-1) and values are tuples representing the (x, y, z) coordinates of each node.
    """

    t = np.linspace(0, upper_limit, num_nodes)  # Adjust range for truncation
    x = (np.sin(t) + 2 * np.sin(2 * t)) * scale
    y = (np.cos(t) - 2 * np.cos(2 * t)) * scale
    z = (-np.sin(3 * t)) * scale
    pos = {}
    for i in range(num_nodes):
        pos[i] = (x[i], y[i], z[i])
    return pos


def get_overhand_knot(num_nodes, num_nodes_head=0, num_nodes_tail=0, scale=1, **kwargs):
    """
    Generate the positions and adjacency list for an overhand knot graph.
    This function creates an overhand knot (trefoil knot) graph with a specified number of nodes,
    and optionally extends the ends of the knot with additional nodes.
    Parameters:
    num_nodes (int): The number of nodes in the main part of the knot.
    num_nodes_head (int, optional): The number of nodes to extend at the head of the knot. Default is 0.
    num_nodes_tail (int, optional): The number of nodes to extend at the tail of the knot. Default is 0.
    scale (float, optional): The scale factor for the size of the knot. Default is 1.
    **kwargs: Additional keyword arguments.
    Returns:
    tuple: A tuple containing:
        - pos (list of tuples): The positions of the nodes in the knot.
        - adjacency (list of tuples): The adjacency list representing the edges of the knot graph.
    """

    # make the trefoil knot
    pos = get_trefoil_pos(num_nodes, scale=scale)
    # extend the ends of the trefoil knot each by num_nodes_linear_extension

    pos, adjacency = extend_graph(pos, num_nodes_head, num_nodes_tail)
    return pos, adjacency


def extend_graph(pos, num_nodes_head, num_nodes_tail):
    """
    Extends a given path graph by adding nodes to the head (index zero) and tail (last index).
    Parameters:
    pos (dict): A dictionary where keys are node indices and values are node positions (as arrays or lists).
    num_nodes_head (int): Number of nodes to add to the head of the graph.
    num_nodes_tail (int): Number of nodes to add to the tail of the graph.
    Returns:
    tuple: A tuple containing:
        - pos (dict): Updated dictionary of node positions with new nodes added.
        - adjacency (dict): Adjacency dictionary of the extended path graph.
    """
    # Linearly extends a given PATH graph by adding nodes to the head (index zero) and tail (last index)
    # estimate gradient of the last num_nodes_tail nodes
    grad = np.zeros(3)
    num_nodes = len(pos)
    tail_grad_estimator = 2

    for i in range(num_nodes - tail_grad_estimator, num_nodes):
        grad += np.array(pos[i]) - np.array(pos[i - 1])
    grad /= tail_grad_estimator
    for i in range(num_nodes_tail):
        pos[num_nodes + i] = np.array(pos[num_nodes - 1]) + grad * (i + 1)

    head_grad_estimator = 2
    grad = np.zeros(3)
    for i in range(head_grad_estimator):
        grad += np.array(pos[i]) - np.array(pos[i + 1])
    grad /= head_grad_estimator

    for i in range(num_nodes_head):
        pos[-(i + 1)] = np.array(pos[0]) + grad * (i + 1)

    # shift all the node labelings to start at 0
    pos = {k + num_nodes_head: v for k, v in pos.items()}
    # sort pos by node number
    pos = dict(sorted(pos.items()))
    new_num_nodes = len(pos)
    # get the path graph adjacency
    G = nx.path_graph(new_num_nodes)
    adjacency = G.adj
    return pos, adjacency


# make a helix graph
def get_helical_graph(
    num_nodes,
    orientation="CCW",
    radius=1,
    height=1,
    num_rotations=1,
    num_nodes_head=0,
    num_nodes_tail=0,
    **kwargs
):
    """
    Generate a helical graph with specified parameters.

    Parameters:
        num_nodes (int): Number of nodes in the helical graph.
        orientation (str): Orientation of the helix, either "CCW" (counter-clockwise) or "CW" (clockwise). Default is "CCW".
        radius (float): Radius of the helix. Default is 1.
        height (float): Height of the helix. Default is 1.
        num_rotations (int): Number of rotations of the helix. Default is 1.
        num_nodes_head (int): Number of additional nodes to add to the head of the graph. Default is 0.
        num_nodes_tail (int): Number of additional nodes to add to the tail of the graph. Default is 0.
        **kwargs: Additional keyword arguments.
    Returns:
    tuple: A tuple containing:
        - pos (dict): A dictionary where keys are node indices and values are 3D coordinates (x, y, z).
        - adjacency (dict): An adjacency dictionary representing the graph structure.
    """
    pos = {}
    for i in range(num_nodes):
        theta = 2 * np.pi * i / num_nodes * num_rotations
        if orientation == "CCW":
            pos[i] = (radius * np.cos(theta), radius * np.sin(theta), height * theta / (2 * np.pi))
        else:
            pos[i] = (
                radius * np.cos(theta),
                radius * np.sin(theta),
                -height * theta / (2 * np.pi),
            )

    pos, adjacency = extend_graph(pos, num_nodes_head, num_nodes_tail)

    # G = nx.path_graph(num_nodes)
    # adjacency = G.adj
    return pos, adjacency


def get_path_graph(num_nodes, scale=1, **kwargs):
    """
    Generate a path graph with a specified number of nodes and scale.
    Parameters:
    num_nodes (int): The number of nodes in the path graph.
    scale (int, optional): The scale factor for the node positions. Default is 1.
    **kwargs: Additional keyword arguments (currently not used).
    Returns:
    tuple: A tuple containing:
        - pos (dict): A dictionary where keys are node indices and values are
          tuples representing the (x, y, z) positions of the nodes.
        - adjacency (dict): The adjacency dictionary of the path graph.
    """

    pos = {i: (i * scale - num_nodes // 2, 0, 0) for i in range(num_nodes)}
    G = nx.path_graph(num_nodes)
    adjacency = G.adj
    return pos, adjacency


def get_ER_graph(num_nodes, p, scale=1, **kwargs):
    """
    Generate an Erdos-Renyi graph with a specified number of nodes and edge probability.
    Parameters:
    num_nodes (int): The number of nodes in the graph.
    p (float): The probability of an edge between any two nodes.
    scale (int, optional): The scale factor for the node positions. Default is 1.
    **kwargs: Additional keyword arguments (currently not used).
    Returns:
    tuple: A tuple containing:
        - pos (dict): A dictionary where keys are node indices and values are
          tuples representing the (x, y, z) positions of the nodes.
        - adjacency (dict): The adjacency dictionary of the Erdos-Renyi graph.
    """
    if p > 1:
        raise ValueError("Probability p must be less than 1")
    # if scale in kwargs replace it
    if "scale" in kwargs:
        scale = kwargs["scale"]
    G = nx.fast_gnp_random_graph(num_nodes, p)
    pos = nx.spring_layout(G, dim=3, scale=scale)
    adjacency = G.adj
    return pos, adjacency


if __name__ == "__main__":
    # test the functions and see if extending a graph position works
    # get a line graph
    num_nodes = 10
    pos = {i: (i, 0, 0) for i in range(num_nodes)}
    G = nx.path_graph(num_nodes)
    adjacency = G.adj
    print(pos)
    print(adjacency)
    # extend the graph
    num_nodes_head = 5
    num_nodes_tail = 5
    pos, adjacency = extend_graph(pos, num_nodes_head, num_nodes_tail)
    print(pos)
    print(adjacency)
