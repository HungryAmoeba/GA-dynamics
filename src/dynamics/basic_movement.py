import numpy as np

def push_pos_towards_tail(pos, num_for_grad = 2):
    # the tail is the last node (node with the highest number)
    pos = pos.copy()
    nodes = list(pos.keys())
    num_nodes = len(nodes)
    for node_num in range(num_nodes - 1):
        pos[node_num] = pos[node_num + 1]
    
    # to update the last node, find the average gradient of the closest num_for_grad nodes
    grad = np.zeros(3)
    for node_num in range(num_nodes - num_for_grad-1, num_nodes-1):
        delta = np.array(pos[node_num]) - np.array(pos[node_num - 1])
        grad += delta
    grad /= num_for_grad
    new_final_pos = np.array(pos[num_nodes - 1]) + grad
    # make new_final_pos a tuple of type float, not np.float
    new_final_pos = tuple([float(x) for x in new_final_pos])
    pos[nodes[num_nodes - 1]] = new_final_pos
    return pos

def push_pos_towards_head(pos, num_for_grad = 2):
    # push the nodes towards the head (node with the lowest number)
    pos = pos.copy()
    nodes = list(pos.keys())
    num_nodes = len(nodes)

    for node_num in range(num_nodes - 1, 0, -1):
        pos[node_num] = pos[node_num - 1]

    # to update the first node, find the average gradient of the closest num_for_grad nodes
    grad = np.zeros(3)
    for node_num in range(num_for_grad):
        delta = np.array(pos[node_num + 1]) - np.array(pos[node_num])
        grad += delta
    grad /= num_for_grad

    new_final_pos = np.array(pos[0]) - grad
    # make new_final_pos a tuple of type float, not np.float
    new_final_pos = tuple([float(x) for x in new_final_pos])
    pos[nodes[0]] = new_final_pos
    return pos

def animate_positions(pos, head_push, tail_push, gradient_estimator = 3):
    '''
    Animate the positions of a graph by pushing the head and tail nodes

    Args:
        pos (dict): Dictionary of node numbers to positions
        head_push (int): Number of nodes to push the head
        tail_push (int): Number of nodes to push the tail
        gradient_estimator (int): Number of nodes to estimate the direction gradient 
    
    Returns:
        list: List of dictionaries of node numbers to positions
    '''
    positions_list = []
    positions_list.append(pos.copy())

    # first add the head push, then the tail push at the end 
    for _ in range(head_push):
        pos = push_pos_towards_head(pos, gradient_estimator)
        positions_list.append(pos.copy())

    # now reverse the list and add the tail push

    positions_list = positions_list[::-1]
    pos = positions_list[-1]

    for _ in range(tail_push):
        pos = push_pos_towards_tail(pos, gradient_estimator)
        positions_list.append(pos.copy())

    return positions_list
    
def constant_center_of_mass(positions_array, center_of_mass):
    '''
    Aligns the center of mass of the graph to a constant position

    Args:
        positions_array (np.array): array of shape T x N x D where T is the number of frames, N is the number of nodes, and D is the dimension of the positions
        center_of_mass (tuple): the center of mass to shift the graph to
    '''
    # calculate the center of mass of the graph
    center_of_mass = np.array(center_of_mass)
    for i in range(len(positions_array)):
        positions_array[i] += center_of_mass - np.mean(positions_array[i], axis = 0)
    return positions_array

def positive_z(pos, z_floor = .5):
    '''
    Shifts the graph so that the lowest node is at min_z

    Args:
        pos (dict): Dictionary of node numbers to positions
        z_floor (float): Minimum z value for the lowest node
    '''
    min_z = min([pos[node][2] for node in pos])
    for node in pos:
        pos[node] = (pos[node][0], pos[node][1], pos[node][2] - min_z + z_floor)
    return pos



