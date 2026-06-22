import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from interface.position_record import (
    DEFAULT_HAND_INITIAL_POSE_DEG,
    DEFAULT_HAND_POSITION_BIAS_DEG,
    INITIAL_POSITION_OFFSET_DEG,
)


MODULE_PATH = Path(__file__).resolve().parents[1] / 'interface' / 'run_inference.py'


def load_run_inference_module():
    spec = importlib.util.spec_from_file_location('run_inference_under_test', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunInferenceTests(unittest.TestCase):
    def test_resolve_artifact_paths_uses_root_level_files(self):
        module = load_run_inference_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir)
            checkpoint = weights_dir / 'seed_expert_best.pth'
            stats = weights_dir / 'normalizer_stats.npy'
            checkpoint.write_bytes(b'checkpoint')
            np.save(
                stats,
                {
                    'obs_mean': np.zeros(16),
                    'obs_std': np.ones(16),
                    'act_min': np.zeros(16),
                    'act_max': np.ones(16),
                },
            )

            resolved_checkpoint, resolved_stats = module.resolve_artifact_paths(None, None, weights_dir=weights_dir)

            self.assertEqual(resolved_checkpoint, checkpoint.resolve())
            self.assertEqual(resolved_stats, stats.resolve())

    def test_build_short_horizon_trajectory_clamps_step_delta(self):
        module = load_run_inference_module()

        class FakeModel:
            def __init__(self):
                self.act_min = torch.zeros(16)
                self.act_max = torch.ones(16)
                self.obs_mean = torch.zeros(16)

            def predict_denormalized(self, raw_obs):
                return torch.from_numpy(np.full((1, 4, 16), 1.0, dtype=np.float32))

        current_pose = np.zeros(16, dtype=np.float32)
        trajectory = module.build_short_horizon_trajectory(
            model=FakeModel(),
            initial_pose=current_pose,
            horizon_steps=4,
            ensemble_decay=0.7,
            lowpass_alpha=1.0,
            max_delta_rad=0.08,
        )

        self.assertEqual(trajectory.shape, (4, 16))
        self.assertTrue(np.all(np.abs(np.diff(np.vstack([current_pose, trajectory]), axis=0)) <= 0.0800001))

    def test_build_startup_pose_matches_position_record_defaults(self):
        module = load_run_inference_module()

        expected = np.deg2rad(
            np.asarray(DEFAULT_HAND_INITIAL_POSE_DEG, dtype=np.float64)
            + np.asarray(DEFAULT_HAND_POSITION_BIAS_DEG, dtype=np.float64)
            - INITIAL_POSITION_OFFSET_DEG
        )

        np.testing.assert_allclose(module.build_startup_pose(), expected)

    def test_interpolate_trajectory_adds_smooth_midpoints(self):
        module = load_run_inference_module()
        start = np.zeros(16, dtype=np.float64)
        end = np.ones(16, dtype=np.float64)

        interpolated = module.interpolate_trajectory(
            np.vstack([start, end]),
            interpolation_factor=2,
        )

        self.assertEqual(interpolated.shape, (3, 16))
        np.testing.assert_allclose(interpolated[0], start)
        np.testing.assert_allclose(interpolated[1], np.full(16, 0.5))
        np.testing.assert_allclose(interpolated[2], end)

    def test_controller_input_pose_prefers_real_hand_order(self):
        module = load_run_inference_module()

        class FakeHandController:
            raw_positions = np.arange(16, dtype=np.float32).reshape(1, -1) + 100.0
            real_raw_positions = np.arange(16, dtype=np.float32).reshape(1, -1)

        np.testing.assert_allclose(
            module.get_model_input_pose(FakeHandController()),
            np.arange(16, dtype=np.float32),
        )

    def test_real_order_trajectory_converts_to_controller_order_before_publish(self):
        module = load_run_inference_module()

        class FakeHandController:
            real_to_sim_indices = [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]

            def real_to_sim(self, values):
                return values[self.real_to_sim_indices]

        trajectory = np.arange(32, dtype=np.float64).reshape(2, 16)
        expected = trajectory[:, FakeHandController.real_to_sim_indices]

        np.testing.assert_allclose(
            module.real_trajectory_to_controller_order(trajectory, FakeHandController()),
            expected,
        )

    def test_format_prediction_log_uses_real_order_deg_plus_180(self):
        module = load_run_inference_module()
        initial_pose = np.deg2rad(np.arange(16, dtype=np.float64))
        trajectory = np.deg2rad(np.arange(32, dtype=np.float64).reshape(2, 16))

        text = module.format_prediction_log(trajectory, initial_pose=initial_pose)

        self.assertIn('predicted joint0-joint15 deg+180:', text)
        self.assertIn('initial: [180.00, 181.00, 182.00', text)
        self.assertIn('step0: [180.00, 181.00, 182.00', text)
        self.assertIn('step1: [196.00, 197.00, 198.00', text)

    def test_controller_waits_for_previous_command_duration_before_republishing(self):
        module = load_run_inference_module()

        class FakeLogger:
            def info(self, message):
                pass

            def error(self, message):
                raise AssertionError(message)

        class FakeNode:
            def get_logger(self):
                return FakeLogger()

        class FakeHandController:
            def __init__(self):
                self.real_raw_positions = np.zeros((1, 16), dtype=np.float32)
                self.commands = []

            def real_to_sim(self, values):
                return np.asarray(values)

            def command_joint_position(self, desired_pose, speed):
                self.commands.append((np.asarray(desired_pose), float(speed)))
                return True

        class FakeModel:
            def __init__(self):
                self.act_min = torch.full((16,), -10.0)
                self.act_max = torch.full((16,), 10.0)
                self.obs_mean = torch.zeros(16)

            def predict_denormalized(self, raw_obs):
                return torch.zeros((1, 2, 16), dtype=torch.float32)

        controller = module.HandInferenceController.__new__(module.HandInferenceController)
        controller.node = FakeNode()
        controller.hand_controller = FakeHandController()
        controller.model = FakeModel()
        controller.control_hz = 10.0
        controller.interpolation_factor = 1
        controller.seconds_per_point = 0.25
        controller.horizon_steps = 2
        controller.ensemble_decay = 0.7
        controller.lowpass_alpha = 1.0
        controller.max_delta_rad = 0.1
        controller.expected_obs_dim = 16
        controller.waiting_for_state_logged = False
        controller.soft_start_complete = True
        controller.command_execution_finish_time = 0.0
        controller.now = lambda: 10.0

        controller.publish_inference_trajectory()
        controller.publish_inference_trajectory()

        self.assertEqual(len(controller.hand_controller.commands), 1)
        self.assertGreater(controller.command_execution_finish_time, 10.0)

    def test_controller_pauses_after_five_inference_blocks_until_space(self):
        module = load_run_inference_module()

        class FakeLogger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(message)

            def error(self, message):
                raise AssertionError(message)

        class FakeNode:
            def __init__(self):
                self.logger = FakeLogger()

            def get_logger(self):
                return self.logger

        class FakeHandController:
            def __init__(self):
                self.real_raw_positions = np.zeros((1, 16), dtype=np.float32)
                self.commands = []

            def real_to_sim(self, values):
                return np.asarray(values)

            def command_joint_position(self, desired_pose, speed):
                self.commands.append((np.asarray(desired_pose), float(speed)))
                return True

        class FakeModel:
            def __init__(self):
                self.act_min = torch.full((16,), -10.0)
                self.act_max = torch.full((16,), 10.0)
                self.obs_mean = torch.zeros(16)

            def predict_denormalized(self, raw_obs):
                return torch.zeros((1, 2, 16), dtype=torch.float32)

        clock = {'now': 0.0}
        controller = module.HandInferenceController.__new__(module.HandInferenceController)
        controller.node = FakeNode()
        controller.hand_controller = FakeHandController()
        controller.model = FakeModel()
        controller.control_hz = 10.0
        controller.interpolation_factor = 1
        controller.seconds_per_point = 0.1
        controller.horizon_steps = 2
        controller.ensemble_decay = 0.7
        controller.lowpass_alpha = 1.0
        controller.max_delta_rad = 0.1
        controller.expected_obs_dim = 16
        controller.waiting_for_state_logged = False
        controller.soft_start_complete = True
        controller.command_execution_finish_time = 0.0
        controller.inference_blocks_remaining = 5
        controller.waiting_for_keyboard_start = False
        controller.now = lambda: clock['now']

        for _ in range(5):
            controller.publish_inference_trajectory()
            clock['now'] = controller.command_execution_finish_time + 0.001

        controller.publish_inference_trajectory()
        self.assertEqual(len(controller.hand_controller.commands), 5)
        self.assertTrue(controller.waiting_for_keyboard_start)
        self.assertTrue(any('Press SPACE' in message for message in controller.node.logger.messages))

        controller.handle_keyboard_key(' ')
        self.assertEqual(controller.inference_blocks_remaining, 5)
        self.assertFalse(controller.waiting_for_keyboard_start)

    def test_controller_prepares_startup_torque_before_first_inference(self):
        module = load_run_inference_module()

        class FakeLogger:
            def info(self, message):
                pass

            def error(self, message):
                raise AssertionError(message)

        class FakeNode:
            def get_logger(self):
                return FakeLogger()

        class FakeHandController:
            real_raw_positions = np.zeros((1, 16), dtype=np.float32)

        calls = []
        controller = module.HandInferenceController.__new__(module.HandInferenceController)
        controller.node = FakeNode()
        controller.hand_controller = FakeHandController()
        controller.expected_obs_dim = 16
        controller.waiting_for_state_logged = False
        controller.soft_start_complete = True
        controller.command_execution_finish_time = 0.0
        controller.inference_blocks_remaining = 5
        controller.waiting_for_keyboard_start = False
        controller.startup_torque_prepared = False
        controller.startup_torque_restored = False
        controller.startup_waiting_for_space = False
        controller.now = lambda: 0.0
        controller.set_joint_torque_state = lambda dxl_ids, enable: calls.append((tuple(dxl_ids), bool(enable))) or True

        controller.run_startup_torque_worker()

        self.assertEqual(calls, [(tuple(range(8)), False)])
        self.assertTrue(controller.startup_waiting_for_space)

    def test_space_restores_all_torque_before_first_inference_batch(self):
        module = load_run_inference_module()

        class FakeLogger:
            def info(self, message):
                pass

            def error(self, message):
                raise AssertionError(message)

        class FakeNode:
            def get_logger(self):
                return FakeLogger()

        calls = []
        controller = module.HandInferenceController.__new__(module.HandInferenceController)
        controller.node = FakeNode()
        controller.command_execution_finish_time = 0.0
        controller.startup_torque_prepared = True
        controller.startup_torque_restored = False
        controller.startup_waiting_for_space = True
        controller.waiting_for_keyboard_start = False
        controller.inference_blocks_remaining = 0
        controller.now = lambda: 0.0
        controller.set_joint_torque_state = lambda dxl_ids, enable: calls.append((tuple(dxl_ids), bool(enable))) or True

        controller.handle_keyboard_key(' ')

        self.assertEqual(calls, [(tuple(range(16)), True)])
        self.assertTrue(controller.startup_torque_restored)
        self.assertFalse(controller.startup_waiting_for_space)
        self.assertEqual(controller.inference_blocks_remaining, 5)

    def test_startup_torque_disable_is_attempted_only_once_even_if_service_fails(self):
        module = load_run_inference_module()

        class FakeLogger:
            def info(self, message):
                pass

            def warn(self, message):
                pass

            def error(self, message):
                pass

        class FakeNode:
            def get_logger(self):
                return FakeLogger()

        calls = []
        controller = module.HandInferenceController.__new__(module.HandInferenceController)
        controller.node = FakeNode()
        controller.startup_torque_prepared = False
        controller.startup_torque_restored = False
        controller.startup_waiting_for_space = False
        controller.set_joint_torque_state = lambda dxl_ids, enable: calls.append((tuple(dxl_ids), bool(enable))) or False

        controller.run_startup_torque_worker()
        controller.run_startup_torque_worker()

        self.assertEqual(calls, [(tuple(range(8)), False)])
        self.assertTrue(controller.startup_torque_prepared)
        self.assertTrue(controller.startup_waiting_for_space)

    def test_set_joint_torque_state_attempts_all_ids_even_if_one_fails(self):
        module = load_run_inference_module()

        class FakeLogger:
            def __init__(self):
                self.errors = []

            def info(self, message):
                pass

            def error(self, message):
                self.errors.append(message)

        class FakeNode:
            def __init__(self):
                self.logger = FakeLogger()

            def get_logger(self):
                return self.logger

        class FakeClient:
            pass

        requested_ids = []
        controller = module.HandInferenceController.__new__(module.HandInferenceController)
        controller.node = FakeNode()
        controller.dxl_data_service_timeout_sec = 5.0
        controller.dxl_data_service_names = ['service']
        controller.resolve_dxl_data_client = lambda: ('service', FakeClient())

        def fake_call(client, request):
            requested_ids.append(request.id)

            class Response:
                result = request.id != 1

            return Response()

        controller.call_dxl_data_service = fake_call

        result = controller.set_joint_torque_state(tuple(range(8)), False)

        self.assertFalse(result)
        self.assertEqual(requested_ids, list(range(8)))
        self.assertTrue(controller.node.logger.errors)
