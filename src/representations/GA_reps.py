import numpy as np

from src.representations.geometric_number import GeometricNumbers
from src.representations.representation_base import Representation

# from src.utils.dag_helpers import convert_dag_to_tree, convert_tree_to_dag


class GA_OrientedAreaReps(Representation):
    """
    Representation of a trajectory as the oriented areas of triangles formed by adjacent nodes.
    """

    def __init__(self, cfg, adjacency, trajectory, rep_dir, exp_save_name):
        """
        Initialize the representation, adds on its own instance of GeometricNumbers.
        """
        super().__init__(cfg, adjacency, trajectory, rep_dir, exp_save_name)
        self.GA = GeometricNumbers()

    def generate(self):
        GA_rep = get_GA_representation(self.trajectory, self.GA)
        areas = get_areas_path(GA_rep, self.GA)

        # for compatibility, we'll set an empty eigenvalues array
        self.eigenvalues = None

        return areas, self.eigenvalues


def get_GA_representation(trajectory, GA: GeometricNumbers):
    """
    Returns the GA representation of the trajectory.

    Args:
        trajectory (np.array): Shape T x N x D.
        GA (GeometricNumbers): Instance of GeometricNumbers.

    Returns:
        np.array: Shape T x N x 4 x 4.
    """
    T, N, D = trajectory.shape
    trajectory_flat = trajectory.reshape(-1, D)  # Flatten to (T*N, D)

    GA_rep_flat = np.array([GA.vector_to_geometric(coords) for coords in trajectory_flat])
    return GA_rep_flat.reshape(T, N, 4, 4)


def GA_to_3D(trajectory, GA: GeometricNumbers):
    """
    Converts GA representation back to 3D trajectory.

    Args:
        trajectory (np.array): Shape T x N x 4 x 4.
        GA (GeometricNumbers): Instance of GeometricNumbers.

    Returns:
        np.array: Shape T x N x 3.
    """
    T, N, _, _ = trajectory.shape
    trajectory_flat = trajectory.reshape(-1, 4, 4)  # Flatten to (T*N, 4, 4)

    rep_3D_flat = np.array([GA.geometric_to_vector(geom) for geom in trajectory_flat])
    return rep_3D_flat.reshape(T, N, 3)


# TODO: vectorize like the following implementation at some point
# def get_areas_path(GA_trajectory: np.ndarray, GA: GeometricNumbers) -> np.ndarray:
#     """
#     Compute the oriented areas of triangles formed by adjacent nodes in a PATH graph trajectory.

#     Args:
#         GA_trajectory (np.ndarray): Trajectory represented as geometric numbers. Shape: (T, N, 4, 4)
#         GA (GeometricNumbers): Instance of GeometricNumbers for geometric algebra operations.

#     Returns:
#         np.ndarray: Oriented areas of triangles for each timestep. Shape: (T, N-2, 3)
#     """
#     assert len(GA_trajectory.shape) == 4, "The trajectory must be in geometric numbers."
#     T, N, _, _ = GA_trajectory.shape

#     # Get differences between adjacent nodes
#     diff = np.diff(GA_trajectory, axis=1)  # Shape: (T, N-1, 4, 4)

#     # Compute pairwise products of differences
#     v1 = diff[:, :-1]  # Shape: (T, N-2, 4, 4)
#     v2 = diff[:, 1:]   # Shape: (T, N-2, 4, 4)
#     cross_products = np.einsum('tijab,tijbc->tijac', v1, v2)  # Batched matrix product

#     # Extract oriented areas using GA
#     sigma1 = np.vectorize(lambda x: GA.get_GA_component(x, 'sigma1'))(cross_products)
#     sigma2 = np.vectorize(lambda x: GA.get_GA_component(x, 'sigma2'))(cross_products)
#     sigma3 = np.vectorize(lambda x: GA.get_GA_component(x, 'sigma3'))(cross_products)

#     # Stack the results
#     oriented_areas = np.stack([sigma1, sigma2, sigma3], axis=-1)  # Shape: (T, N-2, 3)

#     return oriented_areas


# def get_areas_path(GA_trajectory, GA: GeometricNumbers):
#     """
#     Returns the areas of the triangles formed by the nodes in the trajectory.
#     The trajectory MUST be from a PATH graph.
#     TODO: Extend this to any DAG.

#     Args:
#         GA_trajectory (np.array): Shape T x N x 4 x 4.
#         GA (GeometricNumbers): Instance of GeometricNumbers.

#     Returns:
#         np.array: Shape T x N-2 x 3.
#     """
#     assert len(GA_trajectory.shape) == 4, "The trajectory must be in geometric numbers."
#     T, N, _, _ = GA_trajectory.shape

#     if N < 3:
#         raise ValueError("The graph must have at least 3 nodes to form triangles.")

#     # Calculate differences between adjacent nodes
#     diff = np.diff(GA_trajectory, axis=1)  # Shape: T x (N-1) x 4 x 4

#     # Initialize oriented areas
#     oriented_areas = np.zeros((T, N-2, 3))

#     for t in range(T):
#         for i in range(N-2):
#             v1 = diff[t, i]   # Vector from node i to i+1
#             v2 = diff[t, i+1] # Vector from node i+1 to i+2

#             # Compute bivector representation and extract components
#             bivector = v1 @ v2  # Matrix multiplication
#             oriented_areas[t, i, 0] = GA.extract_component(bivector, 'sigma1')
#             oriented_areas[t, i, 1] = GA.extract_component(bivector, 'sigma2')
#             oriented_areas[t, i, 2] = GA.extract_component(bivector, 'sigma3')

#     return oriented_areas


def get_areas_path(GA_trajectory, GA):
    """
    Returns the areas of the triangles formed by the nodes in the trajectory.
    The trajectory MUST be from a PATH graph.
    TODO: Extend this to any DAG.

    Args:
        GA_trajectory (np.array): Shape T x N x 4 x 4.
        GA (GeometricNumbers): Instance of GeometricNumbers.

    Returns:
        np.array: Shape T x N-2 x 3.
    """
    assert len(GA_trajectory.shape) == 4, "The trajectory must be in geometric numbers."
    T, N, _, _ = GA_trajectory.shape

    if N < 3:
        raise ValueError("The graph must have at least 3 nodes to form triangles.")

    # Calculate differences between adjacent nodes
    diff = np.diff(GA_trajectory, axis=1)  # Shape: T x (N-1) x 4 x 4

    # Compute bivectors by performing batched matrix multiplication for all pairs of consecutive differences
    v1 = diff[:, :-1]  # Shape: T x (N-2) x 4 x 4
    v2 = diff[:, 1:]  # Shape: T x (N-2) x 4 x 4

    bivectors = np.einsum("tijk,tikl->tijl", v1, v2)  # Shape: T x (N-2) x 4 x 4

    # Extract components of the bivectors in a vectorized manner
    oriented_areas = np.stack(
        [
            GA.extract_component(bivectors, "sigma1"),
            GA.extract_component(bivectors, "sigma2"),
            GA.extract_component(bivectors, "sigma3"),
        ],
        axis=-1,
    )  # Shape: T x (N-2) x 3

    return oriented_areas


def get_areas_DAG(adjacency, GA_trajectory, GA):
    """
    Returns the areas of the triangles formed by the nodes in the trajectory.
    The trajectory MUST be from a DAG.

    Args:
        adjacency (np.array): Shape N x N.
        GA_trajectory (np.array): Shape T x N x 4 x 4.
        GA (GeometricNumbers): Instance of GeometricNumbers.

    Returns:
        np.array: Shape T x N-2 x 3.
    """
    assert len(GA_trajectory.shape) == 4, "The trajectory must be in geometric numbers."
    T, N, _, _ = GA_trajectory.shape

    if N < 3:
        raise ValueError("The graph must have at least 3 nodes to form triangles.")

    # Calculate differences between adjacent nodes
    diff = np.diff(GA_trajectory, axis=1)  # Shape: T x (N-1) x 4 x 4

    # Compute bivectors by performing batched matrix multiplication for all pairs of consecutive differences
    v1 = diff[:, :-1]  # Shape: T x (N-2) x 4 x 4
    v2 = diff[:, 1:]  # Shape: T x (N-2) x 4 x 4

    bivectors = np.einsum("tijk,tikl->tijl", v1, v2)  # Shape: T x (N-2) x 4 x 4

    # Extract components of the bivectors in a vectorized manner
    oriented_areas = np.stack(
        [
            GA.extract_component(bivectors, "sigma1"),
            GA.extract_component(bivectors, "sigma2"),
            GA.extract_component(bivectors, "sigma3"),
        ],
        axis=-1,
    )  # Shape: T x (N-2) x 3

    return oriented_areas


def get_areas_dag(GA_trajectory, GA: GeometricNumbers, dag_adjacency):
    """
    Returns the areas of the triangles formed by the nodes in the trajectory for a DAG.

    Args:
        GA_trajectory (np.array): Shape T x N x 4 x 4, where T is the number of time points,
                                   N is the number of nodes, and each node has a 4x4 matrix representation.
        GA (GeometricNumbers): Instance of GeometricNumbers.
        dag_adjacency (dict): Directed adjacency list of the DAG.

    Returns:
        np.array: Shape (T, num_triangles, 3), where num_triangles is the total number of path-connected triplets.
    """
    assert len(GA_trajectory.shape) == 4, "The trajectory must be in geometric numbers."
    T, N, _, _ = GA_trajectory.shape

    # Helper function to find all path-connected triplets
    def find_path_connected_triplets(dag_adjacency):
        triplets = []
        for node in dag_adjacency:
            for mid in dag_adjacency[node]:
                for end in dag_adjacency[mid]:
                    triplets.append((node, mid, end))
        return triplets

    # Find all path-connected triplets
    triplets = find_path_connected_triplets(dag_adjacency)

    if len(triplets) == 0:
        raise ValueError("No path-connected triplets found in the DAG.")

    # Initialize oriented areas
    oriented_areas = np.zeros((T, len(triplets), 3))

    for t in range(T):
        for i, (node, mid, end) in enumerate(triplets):
            v1 = GA_trajectory[t, mid] - GA_trajectory[t, node]  # Vector from node to mid
            v2 = GA_trajectory[t, end] - GA_trajectory[t, mid]  # Vector from mid to end

            # Compute bivector representation and extract components
            bivector = v1 @ v2  # Matrix multiplication
            oriented_areas[t, i, 0] = GA.extract_component(bivector, "sigma1")
            oriented_areas[t, i, 1] = GA.extract_component(bivector, "sigma2")
            oriented_areas[t, i, 2] = GA.extract_component(bivector, "sigma3")

    return oriented_areas


if __name__ == "__main__":
    GA = GeometricNumbers()
    trajectory = np.random.rand(10, 5, 3)  # 10 frames, 5 nodes, 3D positions

    GA_rep = get_GA_representation(trajectory, GA)
    trajectory_3D = GA_to_3D(GA_rep, GA)

    print("Original Trajectory Shape:", trajectory.shape)
    print("GA Representation Shape:", GA_rep.shape)
    print("Reconstructed Trajectory Shape:", trajectory_3D.shape)
