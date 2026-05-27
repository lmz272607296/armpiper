import numpy as np


HAND_JOINT_NAMES = (
    'hand0', 'hand1', 'hand2', 'hand3',
    'hand4', 'hand5', 'hand6', 'hand7',
    'hand8', 'hand9', 'hand10', 'hand11',
    'hand12', 'hand13', 'hand14', 'hand15',
)
GRASP_HAND_JOINT_INDICES = (1, 5)
THUMB_HAND_JOINT_INDICES = (12, 13)
DEFAULT_FIXED_HAND_RADIANS_BY_NAME = {
    'hand0': -1.0017,
    'hand1': 0.3791,
    'hand2': 1.8985,
    'hand3': 0.5556,
    'hand4': -0.5768,
    'hand5': 0.1842,
    'hand6': 1.8125,
    'hand7': 1.3429,
    'hand8': 0.8456,
    'hand9': 0.9132,
    'hand10': 1.9230,
    'hand11': 0.0967,
    'hand12': 1.1771,
    'hand13': -0.2393,
    'hand14': 1.2017,
    'hand15': 0.8487,
}
DEFAULT_FIXED_HAND_RADIANS = np.array(
    [DEFAULT_FIXED_HAND_RADIANS_BY_NAME[joint_name] for joint_name in HAND_JOINT_NAMES],
    dtype=np.float64,
)
THUMB_INITIAL_DEGREES = np.array([
    247.45,  # Joint12 raw initial angle, for reference only
    166.29,  # Joint13 raw initial angle, for reference only
], dtype=np.float64)
THUMB_TARGET_DEGREES = np.array([
    188.0,  # Joint12 raw target angle
    202.0,  # Joint13 raw target angle
], dtype=np.float64)
# Thumb joints are sent after subtracting 180 deg.
# Sent initial angles: Joint12=67.45 deg, Joint13=-13.71 deg.
# Sent target angles: Joint12=8 deg, Joint13=22 deg.
THUMB_INITIAL_RADIANS = np.array([1.1771, -0.2393], dtype=np.float64)
THUMB_TARGET_RADIANS = np.deg2rad(THUMB_TARGET_DEGREES - 180.0)


def validate_chopstick_angles(open_angle_deg, close_angle_deg):
    if open_angle_deg < 0.0:
        raise ValueError(f'chopstick_open_angle_deg must be non-negative, got {open_angle_deg}')
    if close_angle_deg < 0.0:
        raise ValueError(f'chopstick_close_angle_deg must be non-negative, got {close_angle_deg}')


def _smooth_alpha(count):
    if count <= 1:
        return np.array([1.0], dtype=np.float64)

    alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def build_fixed_chopstick_trajectory(sample_count, open_angle_deg, close_angle_deg):
    validate_chopstick_angles(open_angle_deg, close_angle_deg)

    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError(f'sample_count must be positive, got {sample_count}')

    trajectory = np.tile(DEFAULT_FIXED_HAND_RADIANS, (sample_count, 1))
    for thumb_column, joint_index in enumerate(THUMB_HAND_JOINT_INDICES):
        trajectory[:, joint_index] = THUMB_INITIAL_RADIANS[thumb_column]
    if sample_count == 1:
        return trajectory

    transition_count = sample_count - 1
    open_count = max(1, transition_count // 2)
    close_count = transition_count - open_count
    open_alpha = _smooth_alpha(open_count + 1)[1:]
    close_alpha = _smooth_alpha(close_count + 1)[1:]

    offset_open_values = -np.deg2rad(open_angle_deg) * open_alpha
    close_start = offset_open_values[-1] if offset_open_values.size else 0.0
    close_target = np.deg2rad(close_angle_deg)
    if close_count > 0:
        close_values = close_start + (close_target - close_start) * close_alpha
        offsets = np.concatenate((
            np.array([0.0], dtype=np.float64),
            offset_open_values,
            close_values,
        ), axis=0)
    else:
        offsets = np.concatenate((
            np.array([0.0], dtype=np.float64),
            offset_open_values,
        ), axis=0)

    offsets = offsets[:sample_count]
    for joint_index in GRASP_HAND_JOINT_INDICES:
        trajectory[:, joint_index] = DEFAULT_FIXED_HAND_RADIANS[joint_index] + offsets

    thumb_trajectory = np.tile(THUMB_INITIAL_RADIANS, (sample_count, 1))
    open_end_index = open_count
    if open_end_index > 0:
        thumb_trajectory[1:open_end_index + 1] = (
            THUMB_INITIAL_RADIANS
            + (THUMB_TARGET_RADIANS - THUMB_INITIAL_RADIANS) * open_alpha[:, np.newaxis]
        )

    if close_count > 0:
        close_offsets = close_values
        non_positive_close_count = int(np.count_nonzero(close_offsets <= 0.0))
        if non_positive_close_count > 0:
            return_alpha = _smooth_alpha(non_positive_close_count + 1)[1:]
            thumb_return = (
                THUMB_TARGET_RADIANS
                + (THUMB_INITIAL_RADIANS - THUMB_TARGET_RADIANS) * return_alpha[:, np.newaxis]
            )
            close_start_index = open_end_index + 1
            close_end_index = close_start_index + non_positive_close_count
            thumb_trajectory[close_start_index:close_end_index] = thumb_return
            thumb_trajectory[close_end_index:] = THUMB_INITIAL_RADIANS
        else:
            thumb_trajectory[open_end_index + 1:] = THUMB_INITIAL_RADIANS

    for thumb_column, joint_index in enumerate(THUMB_HAND_JOINT_INDICES):
        trajectory[:, joint_index] = thumb_trajectory[:, thumb_column]
    return trajectory
