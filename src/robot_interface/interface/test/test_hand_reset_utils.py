import numpy as np

from interface.hand_reset_utils import (
    HAND_REAL_TO_SIM_INDICES,
    THUMB_REAL_JOINT_INDICES,
    THUMB_SIM_JOINT_INDICES,
    build_delayed_thumb_reset_trajectory,
)


def test_thumb_real_joints_map_to_expected_sim_indices():
    mapped_sim_indices = []
    for real_joint_index in THUMB_REAL_JOINT_INDICES:
        mapped_sim_indices.append(int(np.where(HAND_REAL_TO_SIM_INDICES == real_joint_index)[0][0]))

    assert tuple(mapped_sim_indices) == THUMB_SIM_JOINT_INDICES
    assert THUMB_SIM_JOINT_INDICES == (4, 5, 6, 7)


def test_delayed_thumb_reset_moves_other_fingers_before_thumb():
    current_hand = np.zeros((1, 16), dtype=float)
    target_hand = np.arange(1, 17, dtype=float).reshape(1, 16)

    trajectory = build_delayed_thumb_reset_trajectory(
        current_hand,
        target_hand,
        trajectory_points=6,
        thumb_delay_points=2,
    )

    assert trajectory.shape == (6, 16)
    np.testing.assert_allclose(trajectory[-1], target_hand.reshape(16))

    thumb_indices = list(THUMB_SIM_JOINT_INDICES)
    non_thumb_indices = [index for index in range(16) if index not in thumb_indices]

    np.testing.assert_allclose(trajectory[:2, thumb_indices], 0.0)
    assert np.all(trajectory[0, non_thumb_indices] > 0.0)
    assert np.all(trajectory[2, thumb_indices] > 0.0)
