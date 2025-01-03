import numpy as np
import pytest

from src.representations.geometric_number import GeometricNumbers


@pytest.fixture
def geom_numbers():
    return GeometricNumbers()


def test_get_inverse(geom_numbers):
    e1_inv = geom_numbers.get_inverse("e1")
    assert np.array_equal(e1_inv, geom_numbers.components["e1"])

    sigma1_inv = geom_numbers.get_inverse("sigma1")
    assert np.array_equal(sigma1_inv, -geom_numbers.components["sigma1"])

    with pytest.raises(ValueError):
        geom_numbers.get_inverse("unknown")


def test_extract_component(geom_numbers):
    geom_number = geom_numbers.components["sigma1"]
    component_value = geom_numbers.extract_component(geom_number, "sigma1")
    assert np.isclose(component_value, 1.0)


def test_extract_component_vectorized(geom_numbers):
    geom_number = geom_numbers.components["sigma1"]
    geom_numbers_batch = np.array([geom_number, geom_number])
    component_values = geom_numbers.extract_component(geom_numbers_batch, "sigma1")
    assert np.allclose(component_values, [1.0, 1.0])


def test_vector_to_geometric(geom_numbers):
    coords_3d = np.array([1, 2, 3])
    geom_number = geom_numbers.vector_to_geometric(coords_3d)
    expected_geom_number = (
        coords_3d[0] * geom_numbers.components["sigma1"]
        + coords_3d[1] * geom_numbers.components["sigma2"]
        + coords_3d[2] * geom_numbers.components["sigma3"]
    )
    assert np.array_equal(geom_number, expected_geom_number)


def test_geometric_to_vector(geom_numbers):
    coords_3d = np.array([1, 2, 3])
    geom_number = geom_numbers.vector_to_geometric(coords_3d)
    extracted_coords = geom_numbers.geometric_to_vector(geom_number)
    assert np.allclose(extracted_coords, coords_3d)


# def test_extract_component_consistency(geom_numbers):
#     geom_number = geom_numbers.components["sigma1"]
#     geom_numbers_batch = np.array([geom_number])
#     component_value = geom_numbers.extract_component(geom_number, "sigma1")
#     component_value_vectorized = geom_numbers.extract_component_vectorized(geom_numbers_batch, "sigma1")
#     assert np.isclose(component_value, component_value_vectorized)

# def test_extract_component_consistency_over_array(geom_numbers):
#     geom_numbers_batch = np.random.rand(5, 4, 4)
#     non_vectorized = []
#     for geom_number in geom_numbers_batch:
#         non_vectorized.append(geom_numbers.extract_component(geom_number, "sigma1"))
#     component_value = np.array(non_vectorized)
#     component_value_vectorized = geom_numbers.extract_component_vectorized(geom_numbers_batch, "sigma1")
#     assert np.allclose(component_value, component_value_vectorized)
