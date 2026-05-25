import numpy as np

from interface.position_player_trajectory import (
    DEFAULT_FIXED_HAND_RADIANS,
    build_fixed_chopstick_trajectory,
    validate_chopstick_angles,
)


def test_fixed_chopstick_trajectory_only_moves_hand1_and_hand5():
    trajectory = build_fixed_chopstick_trajectory(6, 30.0, 25.0)

    assert trajectory.shape == (6, 16)
    np.testing.assert_allclose(trajectory[0], DEFAULT_FIXED_HAND_RADIANS)
    np.testing.assert_allclose(trajectory[-1, 1], DEFAULT_FIXED_HAND_RADIANS[1] + np.deg2rad(25.0))
    np.testing.assert_allclose(trajectory[-1, 5], DEFAULT_FIXED_HAND_RADIANS[5] + np.deg2rad(25.0))
    assert np.min(trajectory[:, 1]) < DEFAULT_FIXED_HAND_RADIANS[1]
    assert np.min(trajectory[:, 5]) < DEFAULT_FIXED_HAND_RADIANS[5]

    static_joint_indices = [index for index in range(16) if index not in (1, 5)]
    np.testing.assert_allclose(
        trajectory[:, static_joint_indices],
        np.tile(DEFAULT_FIXED_HAND_RADIANS[static_joint_indices], (6, 1)),
    )


def test_validate_chopstick_angles_rejects_negative_values():
    validate_chopstick_angles(0.0, 0.0)

    try:
        validate_chopstick_angles(-1.0, 0.0)
    except ValueError as exc:
        assert 'chopstick_open_angle_deg' in str(exc)
    else:
        raise AssertionError('Expected negative open angle to raise ValueError')

    try:
        validate_chopstick_angles(0.0, -1.0)
    except ValueError as exc:
        assert 'chopstick_close_angle_deg' in str(exc)
    else:
        raise AssertionError('Expected negative close angle to raise ValueError')


def test_fixed_chopstick_trajectory_accepts_bias_after_generation():
    trajectory = build_fixed_chopstick_trajectory(4, 10.0, 20.0)
    bias = np.zeros(16, dtype=np.float64)
    bias[2] = 0.25

    biased = trajectory.copy()
    biased[:, :16] += bias

    np.testing.assert_allclose(biased[:, 2], DEFAULT_FIXED_HAND_RADIANS[2] + 0.25)
