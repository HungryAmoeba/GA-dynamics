import os

import numpy as np
import pytest

from src.visualizers.visualize_reps import make_oriented_area_movie


def test_make_oriented_area_movie_valid_input(tmp_path):
    """Test if the movie is generated successfully with valid input."""
    T, N = 10, 5
    area_rep = np.random.uniform(-1, 1, (T, N, 3))
    output_file = tmp_path / "test_movie.mp4"

    try:
        make_oriented_area_movie(area_rep, fps=10, output_file=str(output_file))
        assert output_file.exists(), "Animation file was not created."
    except Exception as e:
        pytest.fail(f"make_oriented_area_movie raised an exception: {e}")


def test_make_oriented_area_movie_invalid_input():
    """Test that invalid input raises an appropriate error."""
    area_rep = np.random.uniform(-1, 1, (10, 5))  # Invalid shape, missing third dimension

    with pytest.raises(ValueError):
        make_oriented_area_movie(area_rep, fps=10, output_file="invalid_movie.mp4")


def test_make_oriented_area_movie_no_division_by_zero(tmp_path):
    """Test that normalization handles zeros gracefully."""
    T, N = 10, 5
    area_rep = np.zeros((T, N, 3))  # All zeros, edge case
    output_file = tmp_path / "zero_areas_movie.mp4"

    try:
        make_oriented_area_movie(area_rep, fps=10, output_file=str(output_file))
        assert output_file.exists(), "Animation file was not created for zero areas."
    except Exception as e:
        pytest.fail(f"make_oriented_area_movie raised an exception: {e}")
