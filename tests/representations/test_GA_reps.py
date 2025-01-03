import numpy as np
import pytest

from src.representations.GA_reps import GA_to_3D, get_areas_path, get_GA_representation
from src.representations.geometric_number import GeometricNumbers


@pytest.fixture
def GA():
    """Fixture to create a GeometricNumbers instance."""
    return GeometricNumbers()


def test_basic_conversions():
    GA = GeometricNumbers()
    trajectory = np.random.rand(10, 5, 3)  # 10 frames, 5 nodes, 3D positions

    GA_rep = get_GA_representation(trajectory, GA)
    trajectory_3D = GA_to_3D(GA_rep, GA)

    print("Original Trajectory Shape:", trajectory.shape)
    print("GA Representation Shape:", GA_rep.shape)
    print("Reconstructed Trajectory Shape:", trajectory_3D.shape)

    assert trajectory.shape == trajectory_3D.shape


def test_get_areas_path_valid_input(GA):
    """
    Test get_areas_path with a valid input trajectory.
    """
    T, N = 5, 4  # 5 frames, 4 nodes (enough to form triangles)
    GA_trajectory = np.random.rand(T, N, 4, 4)

    areas = get_areas_path(GA_trajectory, GA)

    assert areas.shape == (T, N - 2, 3), "Output shape mismatch"
    assert np.isfinite(areas).all(), "All area values should be finite"


def test_get_areas_path_minimum_valid_nodes(GA):
    """
    Test get_areas_path with the minimum number of nodes (3) to form a single triangle.
    """
    T, N = 2, 3  # 2 frames, 3 nodes
    GA_trajectory = np.random.rand(T, N, 4, 4)

    areas = get_areas_path(GA_trajectory, GA)

    assert areas.shape == (T, 1, 3), "Should produce one triangle per frame"
    assert np.isfinite(areas).all(), "All area values should be finite"


def test_get_areas_path_invalid_shape(GA):
    """
    Test get_areas_path with an invalid trajectory shape.
    """
    invalid_trajectory = np.random.rand(5, 4, 4)  # Missing one dimension

    with pytest.raises(AssertionError, match="The trajectory must be in geometric numbers."):
        get_areas_path(invalid_trajectory, GA)


def test_get_areas_path_insufficient_nodes(GA):
    """
    Test get_areas_path with fewer than 3 nodes.
    """
    T, N = 3, 2  # Not enough nodes to form a triangle
    GA_trajectory = np.random.rand(T, N, 4, 4)

    with pytest.raises(
        ValueError, match="The graph must have at least 3 nodes to form triangles."
    ):
        get_areas_path(GA_trajectory, GA)


def test_get_areas_path_edge_case_single_frame(GA):
    """
    Test get_areas_path with a single frame.
    """
    T, N = 1, 5  # Single frame, 5 nodes
    GA_trajectory = np.random.rand(T, N, 4, 4)

    areas = get_areas_path(GA_trajectory, GA)

    assert areas.shape == (T, N - 2, 3), "Shape should be consistent for single frame"
    assert np.isfinite(areas).all(), "All area values should be finite"


def test_get_areas_path_consistency(GA):
    """
    Test if the function behaves consistently for the same input.
    """
    T, N = 3, 4
    np.random.seed(42)
    GA_trajectory = np.random.rand(T, N, 4, 4)

    areas_1 = get_areas_path(GA_trajectory, GA)
    areas_2 = get_areas_path(GA_trajectory, GA)

    np.testing.assert_array_equal(
        areas_1, areas_2, "Results should be consistent for identical inputs"
    )


# # compare get_areas_path_vectorized, get_areas_path
# def test_get_areas_path_vectorized_equivalent_to_non_vectorized(GA):
#     """
#     Test if the vectorized version produces the same output as the non-vectorized version.
#     """
#     T, N = 5, 4
#     GA_trajectory = np.random.rand(T, N, 4, 4)

#     areas = get_areas_path(GA_trajectory, GA)
#     areas_vectorized = get_areas_path_vectorized(GA_trajectory, GA)
#     np.testing.assert_array_almost_equal(areas, areas_vectorized, decimal=10, err_msg="Results should be equivalent")

# def test_tree_to_DAG(tree_adj, tree_pos):
#     """
#     See if code to convert a tree to a DAG works.
#     """
#     assert True, "Test not implemented yet"
