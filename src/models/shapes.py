import numpy as np
import networkx as nx

def get_trefoil_pos(num_nodes, upper_limit = 5.5 * np.pi / 4, scale = 1):
    t = np.linspace(0, upper_limit, num_nodes)  # Adjust range for truncation
    x = (np.sin(t) + 2 * np.sin(2 * t)) * scale
    y = (np.cos(t) - 2 * np.cos(2 * t)) * scale
    z = (-np.sin(3 * t)) * scale
    pos = {}
    for i in range(num_nodes):
        pos[i] = (x[i], y[i], z[i])
    return pos

def get_overhand_knot(num_nodes, num_nodes_linear_extension = 10, scale = 1):
    # make the trefoil knot
    pos = get_trefoil_pos(num_nodes, scale = scale)
    # extend the ends of the trefoil knot each by num_nodes_linear_extension

    # extend the first num_nodes_linear_extension nodes

    num_nodes_grad_est = 2
    # estimate gradient of the last num_nodes_linear_extension nodes
    tail_grad = np.zeros(3)

    for i in range(num_nodes - num_nodes_grad_est, num_nodes):
        tail_grad += np.array(pos[i]) - np.array(pos[i - 1])
    tail_grad /= num_nodes_grad_est

    for i in range(num_nodes_linear_extension):
        pos[num_nodes + i] = np.array(pos[num_nodes - 1]) + tail_grad * (i + 1)

    # extend the first num_nodes_linear_extension nodes
    head_grad = np.zeros(3)
    for i in range(num_nodes_grad_est):
        head_grad += np.array(pos[i]) - np.array(pos[i + 1])
    head_grad /= num_nodes_grad_est

    for i in range(num_nodes_linear_extension):
        pos[-(i + 1)] = np.array(pos[0]) + head_grad * (i + 1)
    
    # shift all the node labelings to start at 0
    pos = {k+num_nodes_linear_extension:v for k,v in pos.items()} 
    # make sure its a tuple of floats not np.float
    pos = {k:tuple([float(x) for x in v]) for k,v in pos.items()}

    # make pos non-negative in the z direction
    min_z = min([pos[node][2] for node in pos])
    for node in pos:
        pos[node] = (pos[node][0], pos[node][1], pos[node][2] - min_z + .5)

    # sort pos by node number
    pos = dict(sorted(pos.items()))
    
    # make the adjacency dict 
    adjacency = {}
    for i in range(num_nodes - 1):
        adjacency[i] = [i+1]

    G = nx.path_graph(len(pos))
    adjacency = G.adj
    return pos, adjacency

def extend_graph(pos, num_nodes_head, num_nodes_tail):
    # Linearly extends a given PATH graph by adding nodes to the head (index zero) and tail (last index)
    # estimate gradient of the last num_nodes_tail nodes
    grad = np.zeros(3)
    num_nodes = len(pos)
    tail_grad_estimator = 2
    for i in range(num_nodes - tail_grad_estimator, num_nodes):
        grad += np.array(pos[i]) - np.array(pos[i - 1])
    grad /= num_nodes_tail
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
    pos = {k+num_nodes_head:v for k,v in pos.items()}
    # sort pos by node number
    pos = dict(sorted(pos.items()))
    new_num_nodes = len(pos)
    # get the path graph adjacency
    G = nx.path_graph(new_num_nodes)
    adjacency = G.adj
    return pos, adjacency

# make a helix graph 
def get_helical_graph(num_nodes, orientation='CCW', radius = 1, height = 1, num_rotations = 1):
    pos = {}
    for i in range(num_nodes):
        theta = 2 * np.pi * i / num_nodes * num_rotations
        if orientation == 'CCW':
            pos[i] = (radius * np.cos(theta), radius * np.sin(theta), height * theta / (2 * np.pi))
        else:
            pos[i] = (radius * np.cos(theta), radius * np.sin(theta), -height * theta / (2 * np.pi))
    G = nx.path_graph(num_nodes)
    adjacency = G.adj
    return pos, adjacency

