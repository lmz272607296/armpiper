import numpy as np

from interface.camera_base_transform_node import apply_base_position_bias as apply_fixed_camera_bias
from interface.eye_in_hand_calibration_node import apply_base_position_bias as apply_eye_in_hand_bias


def test_fixed_camera_bias_reduces_only_x_by_three_centimeters():
    base_point = np.array([0.52, -0.11, 0.24], dtype=float)

    biased = apply_fixed_camera_bias(base_point, -0.03)

    np.testing.assert_allclose(biased, np.array([0.49, -0.11, 0.24], dtype=float))


def test_eye_in_hand_bias_reduces_only_x_by_three_centimeters():
    base_point = np.array([0.61, 0.08, 0.15], dtype=float)

    biased = apply_eye_in_hand_bias(base_point, -0.03)

    np.testing.assert_allclose(biased, np.array([0.58, 0.08, 0.15], dtype=float))
