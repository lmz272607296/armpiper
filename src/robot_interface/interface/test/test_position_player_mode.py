import importlib.util
import os
import sys
import tempfile
import types
import unittest

import numpy as np


MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'interface',
    'position_player.py',
)


def install_stub_modules():
    interface_module = types.ModuleType('interface')
    interface_module.__path__ = []
    sys.modules['interface'] = interface_module

    arm_controller_module = types.ModuleType('interface.arm_controller')
    arm_controller_module.LeapArm = type('LeapArm', (), {})
    sys.modules['interface.arm_controller'] = arm_controller_module

    hand_controller_module = types.ModuleType('interface.hand_controller')
    hand_controller_module.LeapHand = type('LeapHand', (), {})
    sys.modules['interface.hand_controller'] = hand_controller_module

    position_record_module = types.ModuleType('interface.position_record')
    position_record_module.INITIAL_POSITION_OFFSET_DEG = 180.0
    position_record_module.DEFAULT_HAND_INITIAL_POSE_DEG = [
        114.0, 199.5, 288.5, 212.0,
        146.3, 185.0, 267.0, 280.0,
        241.0, 221.6, 284.0, 188.3,
        251.1, 169.0, 261.0, 224.0,
    ]
    position_record_module.DEFAULT_HAND_POSITION_BIAS_DEG = [
        0.0, 20.0, 15.0, 10.0,
        0.0, 0.0, 0.0, -25.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 25.0, 0.0,
    ]
    interface_module.position_record = position_record_module
    sys.modules['interface.position_record'] = position_record_module

    trajectory_module = types.ModuleType('interface.position_player_trajectory')
    trajectory_module.build_fixed_chopstick_trajectory = (
        lambda sample_count, open_angle_deg, close_angle_deg: np.zeros((sample_count, 16), dtype=np.float64)
    )
    trajectory_module.validate_chopstick_angles = lambda open_angle_deg, close_angle_deg: None
    sys.modules['interface.position_player_trajectory'] = trajectory_module

    rcl_interfaces_module = types.ModuleType('rcl_interfaces')
    rcl_interfaces_msg_module = types.ModuleType('rcl_interfaces.msg')
    rcl_interfaces_msg_module.ParameterDescriptor = type('ParameterDescriptor', (), {})
    sys.modules['rcl_interfaces'] = rcl_interfaces_module
    sys.modules['rcl_interfaces.msg'] = rcl_interfaces_msg_module

    rclpy_module = types.ModuleType('rclpy')
    rclpy_module.init = lambda *args, **kwargs: None
    rclpy_module.shutdown = lambda *args, **kwargs: None
    rclpy_module.ok = lambda: True
    sys.modules['rclpy'] = rclpy_module

    rclpy_executors_module = types.ModuleType('rclpy.executors')
    rclpy_executors_module.MultiThreadedExecutor = type('MultiThreadedExecutor', (), {})
    sys.modules['rclpy.executors'] = rclpy_executors_module

    rclpy_node_module = types.ModuleType('rclpy.node')
    rclpy_node_module.Node = type('Node', (), {})
    sys.modules['rclpy.node'] = rclpy_node_module

    rclpy_qos_module = types.ModuleType('rclpy.qos')
    rclpy_qos_module.HistoryPolicy = type('HistoryPolicy', (), {'KEEP_LAST': object()})
    rclpy_qos_module.QoSProfile = type('QoSProfile', (), {})
    rclpy_qos_module.ReliabilityPolicy = type('ReliabilityPolicy', (), {'RELIABLE': object()})
    sys.modules['rclpy.qos'] = rclpy_qos_module

    sensor_msgs_module = types.ModuleType('sensor_msgs')
    sensor_msgs_msg_module = types.ModuleType('sensor_msgs.msg')
    sensor_msgs_msg_module.JointState = type('JointState', (), {})
    sys.modules['sensor_msgs'] = sensor_msgs_module
    sys.modules['sensor_msgs.msg'] = sensor_msgs_msg_module


def load_position_player_module():
    install_stub_modules()
    spec = importlib.util.spec_from_file_location('position_player_under_test', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PositionPlayerModeTests(unittest.TestCase):
    def test_prompt_playback_mode_retries_until_valid_input(self):
        module = load_position_player_module()
        inputs = iter(['', 'bad', '2'])

        mode = module.prompt_playback_mode(input_func=lambda prompt: next(inputs))

        self.assertEqual(mode, module.PLAYBACK_MODE_RECORDED)

    def test_select_source_trajectory_uses_fixed_mode_builder(self):
        module = load_position_player_module()
        player = module.PositionPlayer.__new__(module.PositionPlayer)
        recorded = np.arange(32, dtype=np.float64).reshape(2, 16)
        fixed = np.full((2, 16), 7.0, dtype=np.float64)
        player.playback_mode = module.PLAYBACK_MODE_FIXED
        player.load_trajectory = lambda: recorded
        player.build_hand_trajectory = lambda trajectory: fixed

        selected = module.PositionPlayer.select_source_trajectory(player)

        np.testing.assert_allclose(selected, fixed)

    def test_prepare_trajectory_for_playback_applies_bias_before_mapping(self):
        module = load_position_player_module()
        player = module.PositionPlayer.__new__(module.PositionPlayer)
        source = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
        player.playback_mode = module.PLAYBACK_MODE_FIXED
        player.trajectory_includes_hand_position_bias = False
        player.hand_position_bias = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
        player.select_source_trajectory = lambda: source
        player.map_real_hand_to_controller_order = lambda trajectory: trajectory[:, [1, 0, 3, 2]]

        prepared = module.PositionPlayer.prepare_trajectory_for_playback(player)

        expected = np.array([[1.8, 1.1, 3.6, 3.3]], dtype=np.float64)
        np.testing.assert_allclose(prepared, expected)

    def test_resolve_data_file_prefers_latest_supported_npy(self):
        module = load_position_player_module()
        player = module.PositionPlayer.__new__(module.PositionPlayer)

        with tempfile.TemporaryDirectory() as temp_dir:
            player.data_dir = temp_dir
            invalid_path = os.path.join(temp_dir, 'invalid.npy')
            older_path = os.path.join(temp_dir, 'older.npy')
            newer_path = os.path.join(temp_dir, 'newer.npy')
            np.save(invalid_path, np.zeros((2, 8), dtype=np.float64))
            np.save(older_path, np.zeros((2, 16), dtype=np.float64))
            np.save(newer_path, np.zeros((2, 23), dtype=np.float64))
            os.utime(invalid_path, (1, 1))
            os.utime(older_path, (2, 2))
            os.utime(newer_path, (3, 3))

            resolved = module.PositionPlayer.resolve_data_file(player, '')

        self.assertEqual(resolved, newer_path)

    def test_update_current_limited_joints_locks_only_new_over_limit_joint(self):
        module = load_position_player_module()
        player = module.PositionPlayer.__new__(module.PositionPlayer)
        player.HAND_JOINT_NAMES = [f'hand{i}' for i in range(16)]
        player.hand_order_indices = np.array([1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11])
        player.controller_index_by_hand_name = module.PositionPlayer.build_controller_index_by_hand_name(player)
        player.hand_current_limit_ma = 400.0
        player.locked_joint_positions = np.full(16, np.nan, dtype=np.float64)
        player.latest_joint_sample = np.arange(16, dtype=np.float64) + 0.25
        player.last_published_sample = np.arange(16, dtype=np.float64) + 10.0
        player.get_hand_current_snapshot = lambda: {
            'dxl3': 420.0,
            'hand4': 399.0,
        }
        player.get_logger = lambda: DummyLogger()

        locked = module.PositionPlayer.update_current_limited_joints(player)

        self.assertEqual(locked, {'hand3': 420.0})
        controller_index = player.controller_index_by_hand_name['hand3']
        self.assertEqual(player.locked_joint_positions[controller_index], player.latest_joint_sample[controller_index])
        self.assertTrue(np.isnan(player.locked_joint_positions[player.controller_index_by_hand_name['hand4']]))

    def test_apply_current_limit_locks_replaces_only_locked_joint_targets(self):
        module = load_position_player_module()
        player = module.PositionPlayer.__new__(module.PositionPlayer)
        player.locked_joint_positions = np.full(16, np.nan, dtype=np.float64)
        player.locked_joint_positions[2] = 9.5
        player.locked_joint_positions[5] = -3.0
        sample = np.arange(16, dtype=np.float64)

        limited = module.PositionPlayer.apply_current_limit_locks(player, sample)

        expected = sample.copy()
        expected[2] = 9.5
        expected[5] = -3.0
        np.testing.assert_allclose(limited, expected)

    def test_parse_hand_position_bias_converts_degrees_to_radians(self):
        module = load_position_player_module()
        player = module.PositionPlayer.__new__(module.PositionPlayer)

        bias = module.PositionPlayer.parse_hand_position_bias(player, [0.0, 90.0] + [0.0] * 14)

        self.assertEqual(bias.shape, (16,))
        self.assertAlmostEqual(bias[0], 0.0)
        self.assertAlmostEqual(bias[1], np.pi / 2.0)

    def test_recorded_playback_skips_bias_when_trajectory_is_already_biased(self):
        module = load_position_player_module()
        player = module.PositionPlayer.__new__(module.PositionPlayer)
        source = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
        player.playback_mode = module.PLAYBACK_MODE_RECORDED
        player.trajectory_includes_hand_position_bias = True
        player.hand_position_bias = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
        player.select_source_trajectory = lambda: source
        player.map_real_hand_to_controller_order = lambda trajectory: trajectory

        prepared = module.PositionPlayer.prepare_trajectory_for_playback(player)

        np.testing.assert_allclose(prepared, source)

    def test_align_fixed_trajectory_to_initial_pose_matches_position_record_initial_pose(self):
        module = load_position_player_module()
        player = module.PositionPlayer.__new__(module.PositionPlayer)
        trajectory = np.zeros((2, 16), dtype=np.float64)

        aligned = module.PositionPlayer.align_fixed_trajectory_to_initial_pose(player, trajectory)

        expected_start = np.deg2rad(
            np.asarray(module.DEFAULT_HAND_INITIAL_POSE_DEG, dtype=np.float64)
            - module.INITIAL_POSITION_OFFSET_DEG
        )
        np.testing.assert_allclose(aligned[0], expected_start)
        np.testing.assert_allclose(aligned[1], expected_start)

    def test_defaults_match_recorded_preview_behavior(self):
        module = load_position_player_module()

        self.assertTrue(module.DEFAULT_TRAJECTORY_INCLUDES_HAND_POSITION_BIAS)
        self.assertEqual(module.DEFAULT_PLAY_RATE_HZ, 10.0)
        self.assertEqual(module.DEFAULT_INITIAL_MOVE_DURATION_SEC, 2.0)
        self.assertEqual(module.DEFAULT_INITIAL_HOLD_SEC, 0.0)
        self.assertEqual(
            module.DEFAULT_HAND_POSITION_BIAS,
            [0.0, 20.0, 15.0, 10.0, 0.0, 0.0, 0.0, -25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 0.0],
        )


class DummyLogger:
    def info(self, message):
        return message

    def warn(self, message, *args, **kwargs):
        return message

    def debug(self, message):
        return message

    def error(self, message):
        return message


if __name__ == '__main__':
    unittest.main()
