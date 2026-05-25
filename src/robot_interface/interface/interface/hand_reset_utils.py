import time

import numpy as np

from interface.motion_utils import as_row


HAND_REAL_TO_SIM_INDICES = np.array([1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11], dtype=int)
THUMB_REAL_JOINT_INDICES = (12, 13, 14, 15)
THUMB_SIM_JOINT_INDICES = tuple(
    int(index) for index, real_index in enumerate(HAND_REAL_TO_SIM_INDICES) if real_index in THUMB_REAL_JOINT_INDICES
)
THUMB_RESET_DELAY_SEC = 1.0


def build_delayed_thumb_reset_trajectory(current_hand, target_hand, trajectory_points, thumb_delay_points):
    if trajectory_points <= 0:
        raise ValueError("trajectory_points must be positive")
    if thumb_delay_points < 0:
        raise ValueError("thumb_delay_points must be non-negative")

    current_hand = as_row(current_hand, 16)
    target_hand = as_row(target_hand, 16)
    if current_hand.shape != target_hand.shape:
        raise ValueError(f"Shape mismatch: current_hand {current_hand.shape}, target_hand {target_hand.shape}")

    phases = np.linspace(0.0, 1.0, trajectory_points + 1, dtype=float)[1:].reshape(-1, 1)
    factors = 0.5 - 0.5 * np.cos(np.pi * phases)
    trajectory = current_hand + factors * (target_hand - current_hand)

    thumb_indices = list(THUMB_SIM_JOINT_INDICES)
    trajectory[:thumb_delay_points, thumb_indices] = current_hand[0, thumb_indices]
    if thumb_delay_points < trajectory_points:
        delayed_phases = np.linspace(0.0, 1.0, trajectory_points - thumb_delay_points + 1, dtype=float)[1:].reshape(-1, 1)
        delayed_factors = 0.5 - 0.5 * np.cos(np.pi * delayed_phases)
        trajectory[thumb_delay_points:, thumb_indices] = (
            current_hand[0, thumb_indices]
            + delayed_factors * (target_hand[0, thumb_indices] - current_hand[0, thumb_indices])
        )

    return trajectory


def command_delayed_thumb_reset(leap, current_hand, target_hand, point_duration, trajectory_points, extra_wait=0.0):
    if point_duration <= 0.0:
        raise ValueError("point_duration must be positive")

    thumb_delay_points = min(trajectory_points, int(np.ceil(THUMB_RESET_DELAY_SEC / point_duration)))
    trajectory = build_delayed_thumb_reset_trajectory(
        current_hand,
        target_hand,
        trajectory_points=trajectory_points,
        thumb_delay_points=thumb_delay_points,
    )
    if not leap.command_joint_position(trajectory, point_duration):
        raise RuntimeError("Failed to publish delayed thumb reset trajectory")
    time.sleep(point_duration * len(trajectory) + extra_wait)
    return trajectory
