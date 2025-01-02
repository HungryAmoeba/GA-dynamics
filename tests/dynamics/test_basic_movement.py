import numpy as np
import pytest

from src.dynamics.basic_movement import (
    animate_positions,
    animate_positions_follower,
    constant_center_of_mass,
    positive_z,
    positive_z_trajectory,
    push_pos_towards_head,
    push_pos_towards_tail,
    rotate_to_flat,
    rotate_to_flat_based_on_ends,
)


def test_push_pos_towards_head():
    pos = {0: (0, 0, 0), 1: (1, 0, 0), 2: (2, 0, 0)}
    new_pos = push_pos_towards_head(pos, num_for_grad=2)
    print(new_pos)
    assert new_pos[0] == (-1, 0, 0)
    assert new_pos[1] == (0, 0, 0)
    assert new_pos[2] == (1, 0, 0)


def test_push_pos_towards_tail():
    pos = {0: (0, 0, 0), 1: (1, 0, 0), 2: (2, 0, 0)}
    new_pos = push_pos_towards_tail(pos, num_for_grad=2)
    print(new_pos)
    assert new_pos[0] == (1, 0, 0)
    assert new_pos[1] == (2, 0, 0)
    assert new_pos[2] == (3, 0, 0)


def test_animate_positions_follower():
    pos = {0: (0, 0, 0), 1: (1, 0, 0), 2: (2, 0, 0)}
    new_pos = animate_positions_follower(pos, head_push=2, tail_push=2, gradient_estimator=2)
    print(new_pos)
    assert len(new_pos) == 5
    # check 0 time point (minus two)
    assert new_pos[0][0] == (-2, 0, 0)
    assert new_pos[0][1] == (-1, 0, 0)
    assert new_pos[0][2] == (0, 0, 0)
    # check 3 time point (middle)
    assert new_pos[2][0] == (0, 0, 0)
    assert new_pos[2][1] == (1, 0, 0)
    assert new_pos[2][2] == (2, 0, 0)
    # check 5 time point (plus two)
    assert new_pos[4][0] == (2, 0, 0)
    assert new_pos[4][1] == (3, 0, 0)
    assert new_pos[4][2] == (4, 0, 0)


def test_rotate_to_flat_based_on_ends_1():
    pos_init = {0: (0, 0, 0), 1: (1, 0, 0), 2: (2, 0, 0)}
    pos_final = {0: (0, 0, 0), 1: (0, 0, 1), 2: (0, 0, 2)}
    # make a np array of the positions
    pos_init_np = np.array(list(pos_init.values()))
    pos_final_np = np.array(list(pos_final.values()))
    # concatenate the two arrays
    pos = np.array([pos_init_np, pos_final_np])
    # rotate the positions
    new_pos = rotate_to_flat_based_on_ends(pos)
    print(new_pos)

    # check that the new positions have no z component or very small z component
    assert np.allclose(new_pos[:, :, 2], 0, atol=1e-10)
