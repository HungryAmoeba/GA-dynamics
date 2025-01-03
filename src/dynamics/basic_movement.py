import numpy as np

# import PCA and rotation functions
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA


def push_pos_towards_tail(pos, num_for_grad=2):
    """
    Moves a model towards the tail (last node)
    Adjusts the positions of nodes in a dictionary by pushing each node's position towards the position of the next node,
    and updates the position of the last node based on the average gradient of the closest nodes.
    Parameters:
    pos (dict): A dictionary where keys are node identifiers and values are their positions (tuples of coordinates).
    num_for_grad (int, optional): The number of closest nodes to consider for calculating the gradient for the last node. Default is 2.
    Returns:
    dict: A dictionary with updated positions of nodes.
    """

    # the tail is the last node (node with the highest number)
    pos = pos.copy()
    nodes = list(pos.keys())
    num_nodes = len(nodes)

    # to update the last node, find the average gradient of the closest num_for_grad nodes
    grad = np.zeros(3)
    for node_num in range(num_nodes - num_for_grad, num_nodes):
        delta = np.array(pos[node_num]) - np.array(pos[node_num - 1])
        grad += delta
    grad /= num_for_grad

    for node_num in range(num_nodes - 1):
        pos[node_num] = pos[node_num + 1]

    new_final_pos = np.array(pos[num_nodes - 1]) + grad
    # make new_final_pos a tuple of type float, not np.float
    new_final_pos = tuple([float(x) for x in new_final_pos])
    pos[nodes[num_nodes - 1]] = new_final_pos
    return pos


def push_pos_towards_head(pos, num_for_grad=1):
    """
    Pushes the positions of nodes towards the head (node with the lowest number).
    This function updates the positions of nodes in a dictionary by shifting each node's position
    towards the position of the node with the next lower number. The first node's position is updated
    based on the average gradient of the closest `num_for_grad` nodes.
    Parameters:
    pos (dict): A dictionary where keys are node numbers and values are their positions (tuples of coordinates).
    num_for_grad (int): The number of closest nodes to consider for calculating the gradient for the first node. Default is 1.
    Returns:
    dict: A dictionary with updated positions of nodes.
    """

    # push the nodes towards the head (node with the lowest number)
    pos = pos.copy()
    nodes = list(pos.keys())
    num_nodes = len(nodes)

    # first compute the gradient for the first node
    grad = np.zeros(3)
    for node_num in range(num_for_grad):
        delta = np.array(pos[node_num + 1]) - np.array(pos[node_num])
        grad += delta
    grad /= num_for_grad

    for node_num in range(num_nodes - 1, 0, -1):
        pos[node_num] = pos[node_num - 1]

    # then update the first node using the gradient information
    new_final_pos = np.array(pos[0]) - grad
    # make new_final_pos a tuple of type float, not np.float
    new_final_pos = tuple([float(x) for x in new_final_pos])
    pos[0] = new_final_pos
    return pos


def animate_positions(pos, **kwargs):
    """
    Animate the positions of nodes based on the specified type.
    Parameters:
        pos (dict): A dictionary where keys are node identifiers and values are their positions.
        **kwargs: Additional keyword arguments to specify the type of animation and other parameters.
        Keyword Args:
        type (str): The type of animation to perform. Supported types are "follower" and "wiggle".
            - "follower": Calls the animate_positions_follower function.
            - "wiggle": Converts pos to a numpy array and calls the animate_positions_wiggle function.
    Returns:
        list: A list of dictionaries where keys are node identifiers and values are their positions.
    """

    if kwargs.get("type") == "follower":
        return animate_positions_follower(pos, **kwargs)
    if kwargs.get("type") == "wiggle":
        # convert pos to a numpy array
        pos_array = np.array([pos[node] for node in pos])
        return animate_positions_wiggle(pos_array, **kwargs)


def animate_positions_follower(pos, head_push, tail_push, gradient_estimator=3, **kwargs):
    """
    Animate the positions of a graph by pushing the head and tail nodes

    Args:
        pos (dict): Dictionary of node numbers to positions
        head_push (int): Number of nodes to push the head
        tail_push (int): Number of nodes to push the tail
        gradient_estimator (int): Number of nodes to estimate the direction gradient

    Returns:
        list: List of dictionaries of node numbers to positions
    """

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
    """
    Aligns the center of mass of the graph to a constant position

    Args:
        positions_array (np.array): array of shape T x N x D where T is the number of frames, N is the number of nodes, and D is the dimension of the positions
        center_of_mass (tuple): the center of mass to shift the graph to
    """
    # calculate the center of mass of the graph
    center_of_mass = np.array(center_of_mass)
    for i in range(len(positions_array)):
        positions_array[i] += center_of_mass - np.mean(positions_array[i], axis=0)
    return positions_array


def positive_z(pos, z_floor=0.5):
    """
    Shifts the graph so that the lowest node is at min_z

    Args:
        pos (dict): Dictionary of node numbers to positions
        z_floor (float): Minimum z value for the lowest node
    """
    min_z = min([pos[node][2] for node in pos])
    for node in pos:
        pos[node] = (pos[node][0], pos[node][1], pos[node][2] - min_z + z_floor)
    return pos


def positive_z_trajectory(trajectory, z_floor=0.01):
    """
    Adjusts a trajectory by shifting it up so that the lowest node is at min_z

    Args:
        trajectory (np.array): array of shape T x N x D where T is the number of frames, N is the number of nodes, and D is the dimension of the positions
        z_floor (float): Minimum z value for the lowest node

    Returns:
        np.array: array of shape T x N x D where T is the number of frames, N is the number of nodes, and D is the dimension of the positions
    """
    global_min_z = np.min(trajectory[:, :, 2])
    trajectory[:, :, 2] -= global_min_z - z_floor
    return trajectory


def make_constant_z_average(positions_array, z_average=0.5):
    """
    Shifts the graph so that the average z value is z_average

    Args:
        positions_array (np.array): array of shape T x N x D where T is the number of frames, N is the number of nodes, and D is the dimension of the positions
        z_average (float): Average z value for the graph
    """
    for i in range(len(positions_array)):
        positions_array[i] += z_average - np.mean(positions_array[i][:, 2])
    return positions_array


def rotate_to_flat(positions_array, reference_frame_ind=None):
    """
    Rotates a 3D graph so that it is as close to parallel to the x-y plane as possible using PCA.

    Parameters:
    positions_array (numpy.ndarray): An array of 3D positions representing the graph.
    reference_frame_ind (int, optional): The index of the reference frame to use for PCA.
                                         If None, the middle frame is used. Default is None.

    Returns:
    numpy.ndarray: The rotated positions array with the graph parallel to the x-y plane.
    """

    if reference_frame_ind is None:
        reference_frame_ind = len(positions_array) // 2
    reference_frame = positions_array[reference_frame_ind]
    pca = PCA(n_components=3)
    pca.fit(reference_frame)
    # Get the normal vector to the plane
    normal_vector = pca.components_[2]
    # Get the rotation matrix to rotate the normal vector to the z-axis
    rotation, _ = R.align_vectors([normal_vector], [[0, 0, 1]])
    # Apply the rotation matrix to the positions
    rotated_positions = [rotation.apply(frame) for frame in positions_array]
    return np.array(rotated_positions)


def rotate_to_flat_based_on_ends(positions_array):
    """
    Rotates a 3D graph based on the positions at the end of the trajectory so that it is as close to parallel to the x-y plane as possible
    It is assumed that the shape at the start and the end are almost linear

    Inputs:
        positions_array (np.array): array of shape T X N x D where T is the number of time points, N is the number of nodes, and D is the dimension of the positions
    """
    start_frame = positions_array[0]
    end_frame = positions_array[-1]

    # get the direction vector of the shape at the start and end of the trajectory
    derivatives_start = np.diff(start_frame, axis=0).mean(axis=0)
    derivatives_start = derivatives_start / np.linalg.norm(derivatives_start)
    derivatives_end = np.diff(end_frame, axis=0).mean(axis=0)
    derivatives_end = derivatives_end / np.linalg.norm(derivatives_end)

    # get the normal vector to the plane using the cross product of the direction vectors
    normal_vector = np.cross(derivatives_start, derivatives_end)
    # normalize the normal vector
    normal_vector /= np.linalg.norm(normal_vector)
    # get the rotation matrix to rotate the normal vector to the z-axis
    rotation, _ = R.align_vectors([normal_vector], [[0, 0, 1]])
    # Apply the rotation matrix to the positions
    rotated_positions = [rotation.apply(frame) for frame in positions_array]

    return np.array(rotated_positions)


def animate_positions_wiggle(positions_array, final_t=10, coeffs="random", **kwargs):
    """
    Wiggle the positions in 3d with random fourier coefficients or specified coefficients
    Works on a 3D array of positions, which is assumed to be a line graph
    Ensure sampled frequencies are less than Nyquist frequency

    Inputs:
        positions_array (np.array): array of shape N x D where N is the number of nodes, and D is the dimension of the positions
        final_t (int): number of frames to generate
        dt (float): time step between frames
        coeffs (str or list): 'random' or a list containing tuples of (freq, amplitude, axis) for each frequency to wiggle

    Returns:
        list: List of dictionaries of node numbers to positions
    """
    max_freq = kwargs.get("max_freq", 60)  # based on resampled FPS of 30
    max_wiggle = kwargs.get("wiggle_max", 1)
    num_nodes, num_dims = positions_array.shape
    node_to_phase = np.linspace(0, 2 * np.pi, num=num_nodes)
    dt = kwargs.get("dt", 0.2)

    print(f"max wiggle: {max_wiggle}")
    print(f"dt: {dt}")

    t = np.linspace(0, final_t, num=int(final_t / dt))
    wiggle = np.zeros((len(t), num_nodes, num_dims))
    if coeffs == "random":
        # generate random coefficients
        coeffs = []
        num_components = kwargs.get("num_components", 10)
        for _ in range(num_components):
            freq = np.random.uniform(0, max_freq)
            amplitude = np.random.uniform(0, max_wiggle)
            axis = np.random.choice([0, 1, 2])
            coeffs.append((freq, amplitude, axis))

    # freq, amplitude, axis define a sinusoidal component to add to the positions
    for freq, amplitude, axis in coeffs:
        for i in range(num_nodes):
            wiggle[:, i, axis] += amplitude * np.sin(freq * t + node_to_phase[i])

    final = positions_array + wiggle
    # Returns:
    #    list: List of dictionaries of node numbers to positions

    # make sure its a list of dictionaries
    final_list = []
    for i in range(len(final)):
        pos = {}
        for j in range(num_nodes):
            pos[j] = tuple(final[i][j])
        final_list.append(pos)
    return final_list
