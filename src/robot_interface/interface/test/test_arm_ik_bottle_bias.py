import numpy as np

from interface.arm_ik import (
    BOTTLE_TARGET_X_BIAS,
    BottleModel,
    FRUIT_GRASP_HEIGHT,
    apply_bottle_target_bias,
    compute_bottle_grasp_position,
)


def test_apply_bottle_target_bias_only_reduces_x():
    target = np.array([0.62, -0.08, 0.15], dtype=float)

    biased = apply_bottle_target_bias(target)

    np.testing.assert_allclose(
        biased,
        np.array([0.62 + BOTTLE_TARGET_X_BIAS, -0.08, 0.15], dtype=float),
    )


def test_compute_bottle_grasp_position_applies_configured_x_bias():
    bottle = BottleModel(center=np.array([0.60, 0.00, 0.0], dtype=float), radius=0.035)

    target = compute_bottle_grasp_position(bottle)
    unbiased_target = compute_bottle_grasp_position(bottle, x_bias=0.0)

    np.testing.assert_allclose(target[0], unbiased_target[0] + BOTTLE_TARGET_X_BIAS)
    np.testing.assert_allclose(target[1:], unbiased_target[1:])


def test_fruit_grasp_height_is_lowered_for_closer_grasp():
    np.testing.assert_allclose(FRUIT_GRASP_HEIGHT, 0.165)
