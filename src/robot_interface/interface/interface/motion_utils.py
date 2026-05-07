import time

import numpy as np
import rclpy


def as_row(data, width=None):
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"Expected {width} columns, got shape {array.shape}")
    return array


def scale_action(origin, target, scale):
    if not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be a positive integer")

    origin = as_row(origin)
    target = as_row(target)
    if origin.shape[1] != target.shape[1]:
        raise ValueError(f"Shape mismatch: origin {origin.shape}, target {target.shape}")

    factors = np.arange(scale + 1, dtype=float).reshape(-1, 1)
    return origin + factors * ((target - origin) / scale)


def refresh_current_joint_states(leap=None, arm=None, need_hand=False, need_arm=False, timeout=5.0):
    if need_hand:
        if leap is None:
            raise ValueError("leap is required when need_hand=True")
        leap.raw_positions = None
        if hasattr(leap, "real_raw_positions"):
            leap.real_raw_positions = None
    if need_arm:
        if arm is None:
            raise ValueError("arm is required when need_arm=True")
        arm.raw_positions = None

    start_time = time.time()
    while rclpy.ok():
        hand_ready = not need_hand or leap.raw_positions is not None
        arm_ready = not need_arm or arm.raw_positions is not None
        if hand_ready and arm_ready:
            current_hand = None if not need_hand else as_row(leap.raw_positions, leap.joints_num)
            current_arm = None if not need_arm else as_row(arm.raw_positions, arm.joints_num)
            return current_hand, current_arm

        if leap is not None:
            rclpy.spin_once(leap, timeout_sec=0.05)
        if arm is not None:
            rclpy.spin_once(arm, timeout_sec=0.05)
        if time.time() - start_time > timeout:
            missing = []
            if need_hand and leap.raw_positions is None:
                missing.append("/hand_joint_states")
            if need_arm and arm.raw_positions is None:
                missing.append("/joint_states_single")
            raise TimeoutError(f"Timed out waiting for fresh {', '.join(missing)}")

    raise RuntimeError("ROS shutdown while waiting for fresh joint states")


def refresh_current_arm(arm, timeout=5.0):
    _, current_arm = refresh_current_joint_states(arm=arm, need_arm=True, timeout=timeout)
    return current_arm


def refresh_current_hand(leap, timeout=5.0):
    current_hand, _ = refresh_current_joint_states(leap=leap, need_hand=True, timeout=timeout)
    return current_hand


def command_scaled_arm_motion(arm, start, target, steps, duration, extra_wait=0.0):
    if steps <= 0:
        raise ValueError("steps must be positive")

    path = scale_action(as_row(start, 6), as_row(target, 6), steps)
    point_duration = float(duration) / steps
    if not arm.command_joint_position(path, point_duration):
        raise RuntimeError("Failed to publish arm trajectory")
    time.sleep(float(duration) + extra_wait)
    return path
