import os
import time

import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from interface.arm_controller import LeapArm
from interface.hand_controller import LeapHand
from interface.position_record import (
    DEFAULT_HAND_INITIAL_POSE_DEG as RECORD_DEFAULT_HAND_INITIAL_POSE_DEG,
    DEFAULT_HAND_POSITION_BIAS_DEG as RECORD_DEFAULT_HAND_POSITION_BIAS_DEG,
    INITIAL_POSITION_OFFSET_DEG,
)
from interface.position_player_trajectory import build_fixed_chopstick_trajectory, validate_chopstick_angles


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEFAULT_DATA_FILE = ''
DEFAULT_PLAY_RATE_HZ = 10.0
DEFAULT_START_INDEX = 0
DEFAULT_END_INDEX = -1
DEFAULT_LOOP = False
DEFAULT_TRAJECTORY_INCLUDES_HAND_POSITION_BIAS = True
DEFAULT_HAND_INITIAL_POSE_DEG = list(RECORD_DEFAULT_HAND_INITIAL_POSE_DEG)

# 直接复用录制端默认偏置，避免两处配置漂移。
DEFAULT_HAND_POSITION_BIAS = list(RECORD_DEFAULT_HAND_POSITION_BIAS_DEG)
DEFAULT_ENABLE_INITIAL_MOVE = True
DEFAULT_JOINT_STATES_TOPIC = '/hand_joint_states'
DEFAULT_INITIAL_MOVE_DURATION_SEC = 2.0
DEFAULT_INITIAL_MOVE_RATE_HZ = 20.0
DEFAULT_INITIAL_HOLD_SEC = 0.0
DEFAULT_ENABLE_WRIST_ALIGNMENT = True
DEFAULT_WRIST_JOINT6_TARGET = 2.07
DEFAULT_USE_FIXED_CHOPSTICK_TRAJECTORY = True
DEFAULT_CHOPSTICK_OPEN_ANGLE_DEG = 40.0
DEFAULT_CHOPSTICK_CLOSE_ANGLE_DEG = 15.0
DEFAULT_ENABLE_HAND_CURRENT_LIMIT = True
DEFAULT_HAND_CURRENT_LIMIT_MA = 450.0
DEFAULT_CURRENT_LIMIT_CHECK_RATE_HZ = 20.0
DEFAULT_CURRENT_LIMIT_COMMAND_DURATION_SEC = 0.05
PLAYBACK_MODE_FIXED = 'fixed'
PLAYBACK_MODE_RECORDED = 'recorded'
REFERENCE_CHOPSTICK_ROTATION_DEG = 40.0
REFERENCE_KEY_JOINT_END_POSE = {
    'hand1': 0.1274,
    'hand5': 0.0107,
    'hand12': 0.4880,
    'hand13': 0.0107,
    'hand15': 0.5878,
}


class PositionPlayer(Node):
    """Load a recorded hand trajectory and send it to the hand controller."""

    HAND_JOINT_NAMES = [
        'hand0', 'hand1', 'hand2', 'hand3', 'hand4', 'hand5', 'hand6', 'hand7',
        'hand8', 'hand9', 'hand10', 'hand11', 'hand12', 'hand13', 'hand14', 'hand15',
    ]

    def __init__(self, hand_controller, arm_controller):
        super().__init__('position_player')
        self.hand_controller = hand_controller
        self.arm_controller = arm_controller
        self.hand_order_indices = np.asarray(self.hand_controller.real_to_sim_indices, dtype=np.int64)
        self.playback_mode = prompt_playback_mode()

        self.declare_parameter('data_file', DEFAULT_DATA_FILE)
        self.declare_parameter('data_dir', DEFAULT_DATA_DIR)
        self.declare_parameter('play_rate_hz', DEFAULT_PLAY_RATE_HZ)
        self.declare_parameter('start_index', DEFAULT_START_INDEX)
        self.declare_parameter('end_index', DEFAULT_END_INDEX)
        self.declare_parameter('loop', DEFAULT_LOOP)
        self.declare_parameter(
            'trajectory_includes_hand_position_bias',
            DEFAULT_TRAJECTORY_INCLUDES_HAND_POSITION_BIAS,
        )
        self.declare_parameter(
            'hand_position_bias',
            DEFAULT_HAND_POSITION_BIAS,
            descriptor=ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter('enable_initial_move', DEFAULT_ENABLE_INITIAL_MOVE)
        self.declare_parameter('joint_states_topic', DEFAULT_JOINT_STATES_TOPIC)
        self.declare_parameter('initial_move_duration_sec', DEFAULT_INITIAL_MOVE_DURATION_SEC)
        self.declare_parameter('initial_move_rate_hz', DEFAULT_INITIAL_MOVE_RATE_HZ)
        self.declare_parameter('initial_hold_sec', DEFAULT_INITIAL_HOLD_SEC)
        self.declare_parameter('enable_wrist_alignment', DEFAULT_ENABLE_WRIST_ALIGNMENT)
        self.declare_parameter('wrist_joint6_target', DEFAULT_WRIST_JOINT6_TARGET)
        self.declare_parameter('use_fixed_chopstick_trajectory', DEFAULT_USE_FIXED_CHOPSTICK_TRAJECTORY)
        self.declare_parameter('chopstick_open_angle_deg', DEFAULT_CHOPSTICK_OPEN_ANGLE_DEG)
        self.declare_parameter('chopstick_close_angle_deg', DEFAULT_CHOPSTICK_CLOSE_ANGLE_DEG)
        self.declare_parameter('enable_hand_current_limit', DEFAULT_ENABLE_HAND_CURRENT_LIMIT)
        self.declare_parameter('hand_current_limit_ma', DEFAULT_HAND_CURRENT_LIMIT_MA)

        self.data_file = self.get_parameter('data_file').value
        self.data_dir = self.get_parameter('data_dir').value
        self.play_rate_hz = float(self.get_parameter('play_rate_hz').value)
        self.start_index = int(self.get_parameter('start_index').value)
        self.end_index = int(self.get_parameter('end_index').value)
        self.loop = bool(self.get_parameter('loop').value)
        self.trajectory_includes_hand_position_bias = bool(
            self.get_parameter('trajectory_includes_hand_position_bias').value
        )
        self.hand_position_bias = self.parse_hand_position_bias(
            self.get_parameter('hand_position_bias').value
        )
        self.enable_initial_move = bool(self.get_parameter('enable_initial_move').value)
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.initial_move_duration_sec = float(self.get_parameter('initial_move_duration_sec').value)
        self.initial_move_rate_hz = float(self.get_parameter('initial_move_rate_hz').value)
        self.initial_hold_sec = float(self.get_parameter('initial_hold_sec').value)
        self.enable_wrist_alignment = bool(self.get_parameter('enable_wrist_alignment').value)
        self.wrist_joint6_target = float(self.get_parameter('wrist_joint6_target').value)
        self.use_fixed_chopstick_trajectory = bool(
            self.get_parameter('use_fixed_chopstick_trajectory').value
        )
        self.chopstick_open_angle_deg = float(self.get_parameter('chopstick_open_angle_deg').value)
        self.chopstick_close_angle_deg = float(self.get_parameter('chopstick_close_angle_deg').value)
        self.enable_hand_current_limit = bool(self.get_parameter('enable_hand_current_limit').value)
        self.hand_current_limit_ma = float(self.get_parameter('hand_current_limit_ma').value)
        self.use_fixed_chopstick_trajectory = (self.playback_mode == PLAYBACK_MODE_FIXED)

        if self.play_rate_hz <= 0.0:
            raise ValueError(f'play_rate_hz must be positive, got {self.play_rate_hz}')
        if self.initial_move_rate_hz <= 0.0:
            raise ValueError(f'initial_move_rate_hz must be positive, got {self.initial_move_rate_hz}')
        if self.initial_hold_sec < 0.0:
            raise ValueError(f'initial_hold_sec must be non-negative, got {self.initial_hold_sec}')
        if self.enable_hand_current_limit and self.hand_current_limit_ma <= 0.0:
            raise ValueError(
                f'hand_current_limit_ma must be positive, got {self.hand_current_limit_ma}'
            )
        validate_chopstick_angles(self.chopstick_open_angle_deg, self.chopstick_close_angle_deg)

        self.trajectory = self.prepare_trajectory_for_playback()
        self.current_index = 0
        self.latest_joint_sample = None
        self.startup_joint_sample = None
        self.startup_arm_sample = None
        self.missing_logged = set()
        self.playback_state = 'playing'
        self.initial_move_finish_time = None
        self.initial_hold_finish_time = None
        self.controller_index_by_hand_name = self.build_controller_index_by_hand_name()
        self.locked_joint_positions = np.full(len(self.HAND_JOINT_NAMES), np.nan, dtype=np.float64)
        self.last_published_sample = None
        self.current_limit_data_unavailable_logged = False

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.subscription = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            qos_profile,
        )

        if self.requires_startup_motion():
            self.playback_state = 'waiting_for_initial_state'

        self.timer = self.create_timer(1.0 / self.play_rate_hz, self.publish_next_sample)
        self.current_limit_timer = None
        if self.enable_hand_current_limit:
            self.current_limit_timer = self.create_timer(
                1.0 / DEFAULT_CURRENT_LIMIT_CHECK_RATE_HZ,
                self.monitor_hand_currents,
            )

        self.get_logger().info(
            f'Loaded {self.trajectory.shape[0]} samples from {self.data_file}; '
            f'playing at {self.play_rate_hz:.2f} Hz.'
        )
        if self.playback_mode == PLAYBACK_MODE_FIXED:
            self.get_logger().info(
                'Using fixed matrix playback mode: only key chopstick joints move, '
                f'open={self.chopstick_open_angle_deg:.2f} deg, '
                f'close={self.chopstick_close_angle_deg:.2f} deg.'
            )
        else:
            self.get_logger().info('Using recorded trajectory playback mode.')
        if np.any(self.hand_position_bias != 0.0):
            self.get_logger().info(f'Using hand position bias: {self.hand_position_bias.tolist()}')
        if self.playback_mode == PLAYBACK_MODE_RECORDED and self.trajectory_includes_hand_position_bias:
            self.get_logger().info(
                'Recorded trajectory already includes hand_position_bias; skipping extra bias during playback.'
            )
        if self.enable_hand_current_limit:
            self.get_logger().info(
                f'Per-joint hand current limit enabled at {self.hand_current_limit_ma:.1f} mA.'
            )
        if self.playback_state == 'waiting_for_initial_state':
            startup_actions = []
            if self.enable_initial_move:
                startup_actions.append('hand soft-start')
            if self.enable_wrist_alignment:
                startup_actions.append(
                    f'arm wrist alignment to joint6={self.wrist_joint6_target:.2f} rad'
                )
            self.get_logger().info(
                f'Startup motion enabled ({", ".join(startup_actions)}): waiting for state topics, then moving over '
                f'{self.initial_move_duration_sec:.2f}s and holding for {self.initial_hold_sec:.2f}s.'
            )

    def requires_startup_motion(self):
        if self.initial_move_duration_sec <= 0.0:
            return False
        return self.enable_initial_move or self.enable_wrist_alignment

    def load_trajectory(self):
        data_file = self.resolve_data_file(self.data_file)
        trajectory = np.load(data_file)
        trajectory = self.normalize_trajectory_shape(trajectory)

        start = max(self.start_index, 0)
        end = trajectory.shape[0] if self.end_index < 0 else min(self.end_index, trajectory.shape[0])
        if start >= end:
            raise ValueError(f'Invalid playback range: start_index={self.start_index}, end_index={self.end_index}')

        self.data_file = data_file
        return trajectory[start:end]

    def select_source_trajectory(self):
        trajectory = self.load_trajectory()
        if self.playback_mode == PLAYBACK_MODE_FIXED:
            return self.build_hand_trajectory(trajectory)
        return trajectory

    def prepare_trajectory_for_playback(self):
        trajectory = self.select_source_trajectory()
        if self.should_apply_hand_position_bias():
            trajectory = self.apply_hand_position_bias(trajectory)
        return self.map_real_hand_to_controller_order(trajectory)

    def should_apply_hand_position_bias(self):
        playback_mode = getattr(self, 'playback_mode', PLAYBACK_MODE_RECORDED)
        if playback_mode == PLAYBACK_MODE_FIXED:
            return True
        return not getattr(
            self,
            'trajectory_includes_hand_position_bias',
            DEFAULT_TRAJECTORY_INCLUDES_HAND_POSITION_BIAS,
        )

    def resolve_data_file(self, data_file):
        if data_file:
            if os.path.isabs(data_file):
                return data_file
            return os.path.join(self.data_dir, data_file)

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f'Data directory does not exist: {self.data_dir}')

        npy_files = [
            os.path.join(self.data_dir, filename)
            for filename in os.listdir(self.data_dir)
            if filename.endswith('.npy')
        ]
        if not npy_files:
            raise FileNotFoundError(f'No .npy files found in {self.data_dir}')

        for file_path in sorted(npy_files, key=os.path.getmtime, reverse=True):
            if self.is_supported_trajectory_file(file_path):
                return file_path

        raise FileNotFoundError(
            f'No supported hand trajectory .npy file found in {self.data_dir}. '
            'Expected a shape of (N, 16), (N, 23), (16,), or (23,).'
        )

    def is_supported_trajectory_file(self, file_path):
        try:
            trajectory = np.load(file_path)
        except Exception:
            return False

        if trajectory.ndim == 1:
            return trajectory.shape[0] in (16, 23)
        if trajectory.ndim == 2:
            return trajectory.shape[1] in (16, 23)
        return False

    def normalize_trajectory_shape(self, trajectory):
        if trajectory.ndim == 1:
            trajectory = np.reshape(trajectory, (1, trajectory.shape[0]))

        if trajectory.ndim != 2 or trajectory.shape[1] not in (16, 23):
            raise ValueError(f'Expected npy shape (N, 16) or (N, 23), got {trajectory.shape}')

        if trajectory.shape[1] == 23:
            trajectory = trajectory[:, :16]

        if np.isnan(trajectory).any():
            raise ValueError('Trajectory contains NaN values; cannot publish incomplete position data.')

        return trajectory

    def parse_hand_position_bias(self, value):
        if isinstance(value, str):
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                value = value[1:-1]
            value = [float(item.strip()) for item in value.split(',') if item.strip()]

        bias = np.deg2rad(np.array(value, dtype=np.float64))
        if bias.shape != (16,):
            raise ValueError(f'hand_position_bias must contain 16 values, got shape {bias.shape}')
        return bias

    def apply_hand_position_bias(self, trajectory):
        biased_trajectory = trajectory.copy()
        biased_trajectory[:, :16] += self.hand_position_bias
        return biased_trajectory

    def build_hand_trajectory(self, trajectory):
        if not self.use_fixed_chopstick_trajectory:
            return trajectory

        fixed_trajectory = build_fixed_chopstick_trajectory(
            trajectory.shape[0],
            self.chopstick_open_angle_deg,
            self.chopstick_close_angle_deg,
        )
        fixed_trajectory = self.apply_key_joint_rotation_targets(fixed_trajectory)
        return self.align_fixed_trajectory_to_initial_pose(fixed_trajectory)

    def align_fixed_trajectory_to_initial_pose(self, trajectory):
        aligned_trajectory = np.asarray(trajectory, dtype=np.float64).copy()
        if aligned_trajectory.shape[0] == 0:
            return aligned_trajectory

        desired_start_pose = np.deg2rad(
            np.asarray(DEFAULT_HAND_INITIAL_POSE_DEG, dtype=np.float64) - INITIAL_POSITION_OFFSET_DEG
        )
        start_delta = desired_start_pose - aligned_trajectory[0, :16]
        aligned_trajectory[:, :16] += start_delta
        return aligned_trajectory

    def apply_key_joint_rotation_targets(self, trajectory):
        sample_count = trajectory.shape[0]
        if sample_count <= 1:
            return trajectory

        reference_scale = np.float64(REFERENCE_CHOPSTICK_ROTATION_DEG)
        if reference_scale <= 0.0:
            raise ValueError(
                f'REFERENCE_CHOPSTICK_ROTATION_DEG must be positive, got {reference_scale}'
            )

        joint_index_by_name = {name: index for index, name in enumerate(self.HAND_JOINT_NAMES)}
        start_pose = trajectory[0].copy()
        key_joint_deltas = {
            joint_name: REFERENCE_KEY_JOINT_END_POSE[joint_name] - start_pose[joint_index_by_name[joint_name]]
            for joint_name in REFERENCE_KEY_JOINT_END_POSE
        }

        transition_count = sample_count - 1
        open_count = max(1, transition_count // 2)
        close_count = transition_count - open_count
        open_alpha = self.smooth_alpha(open_count + 1)[1:]
        close_alpha = self.smooth_alpha(close_count + 1)[1:]
        open_scale = self.chopstick_open_angle_deg / reference_scale
        close_scale = self.chopstick_close_angle_deg / reference_scale

        adjusted_trajectory = trajectory.copy()
        for joint_name, reference_delta in key_joint_deltas.items():
            joint_index = joint_index_by_name[joint_name]
            joint_start = start_pose[joint_index]
            joint_open_target = joint_start + reference_delta * open_scale
            joint_close_target = joint_start + reference_delta * close_scale

            if open_count > 0:
                adjusted_trajectory[1:open_count + 1, joint_index] = (
                    joint_start + (joint_open_target - joint_start) * open_alpha
                )

            if close_count > 0:
                adjusted_trajectory[open_count + 1:, joint_index] = (
                    joint_open_target + (joint_close_target - joint_open_target) * close_alpha
                )

        return adjusted_trajectory

    def smooth_alpha(self, count):
        if count <= 1:
            return np.array([1.0], dtype=np.float64)

        alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def build_controller_index_by_hand_name(self):
        controller_index_by_real_index = np.argsort(self.hand_order_indices)
        return {
            hand_name: int(controller_index_by_real_index[real_index])
            for real_index, hand_name in enumerate(self.HAND_JOINT_NAMES)
        }

    def normalize_current_joint_name(self, joint_name):
        name = str(joint_name).strip().lower()
        for prefix in ('hand', 'dxl'):
            if not name.startswith(prefix):
                continue

            suffix = name[len(prefix):]
            if not suffix.isdigit():
                return None

            joint_index = int(suffix)
            if 0 <= joint_index < len(self.HAND_JOINT_NAMES):
                return f'hand{joint_index}'
            return None

        return None

    def get_hand_current_snapshot(self):
        get_current_snapshot = getattr(self.hand_controller, 'get_current_snapshot', None)
        if get_current_snapshot is None:
            if not self.current_limit_data_unavailable_logged:
                self.get_logger().warn('Hand current snapshot interface is unavailable; current limiting is disabled.')
                self.current_limit_data_unavailable_logged = True
            return {}

        hand_currents = {}
        for joint_name, current_ma in get_current_snapshot().items():
            hand_name = self.normalize_current_joint_name(joint_name)
            if hand_name is None:
                continue
            hand_currents[hand_name] = float(current_ma)
        return hand_currents

    def resolve_locked_joint_position(self, controller_index, fallback_sample=None):
        candidates = [
            self.latest_joint_sample,
            self.last_published_sample,
            fallback_sample,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            candidate_array = np.asarray(candidate, dtype=np.float64).reshape(-1)
            if controller_index >= candidate_array.shape[0]:
                continue
            if np.isfinite(candidate_array[controller_index]):
                return float(candidate_array[controller_index])
        return None

    def update_current_limited_joints(self, fallback_sample=None):
        if not getattr(self, 'enable_hand_current_limit', True):
            return {}

        newly_locked = {}
        for joint_name, current_ma in self.get_hand_current_snapshot().items():
            hand_name = self.normalize_current_joint_name(joint_name) or str(joint_name)
            if abs(current_ma) < self.hand_current_limit_ma:
                continue

            controller_index = self.controller_index_by_hand_name.get(hand_name)
            if controller_index is None:
                continue
            if np.isfinite(self.locked_joint_positions[controller_index]):
                continue

            locked_position = self.resolve_locked_joint_position(controller_index, fallback_sample=fallback_sample)
            if locked_position is None:
                continue

            self.locked_joint_positions[controller_index] = locked_position
            newly_locked[hand_name] = float(current_ma)
            self.get_logger().warn(
                f'Current limit hit on {hand_name}: {current_ma:.1f} mA >= {self.hand_current_limit_ma:.1f} mA. '
                f'Locking this joint at {locked_position:.4f} rad while other joints continue.'
            )

        return newly_locked

    def apply_current_limit_locks(self, sample):
        limited_sample = np.asarray(sample, dtype=np.float64).copy()
        locked_mask = np.isfinite(self.locked_joint_positions)
        limited_sample[locked_mask] = self.locked_joint_positions[locked_mask]
        return limited_sample

    def publish_current_limit_override(self):
        if self.last_published_sample is None:
            return True

        limited_sample = self.apply_current_limit_locks(self.last_published_sample)
        if np.allclose(limited_sample, self.last_published_sample, atol=1e-6, rtol=0.0):
            return True

        hand_pose = np.reshape(limited_sample, (1, 16))
        hand_ok = self.hand_controller.command_joint_position(
            hand_pose,
            DEFAULT_CURRENT_LIMIT_COMMAND_DURATION_SEC,
        )
        if hand_ok:
            self.last_published_sample = limited_sample.copy()
        return hand_ok

    def monitor_hand_currents(self):
        if self.playback_state != 'playing':
            return

        newly_locked = self.update_current_limited_joints()
        if not newly_locked:
            return

        if not self.publish_current_limit_override():
            self.get_logger().error('Failed to publish per-joint current-limit override command.')
            rclpy.shutdown()

    def map_real_hand_to_controller_order(self, trajectory):
        return trajectory[:, self.hand_order_indices]

    def get_complete_arm_sample(self):
        arm_sample = getattr(self.arm_controller, 'raw_positions', None)
        if arm_sample is None:
            return None

        arm_sample = np.asarray(arm_sample, dtype=np.float64).reshape(self.arm_controller.joints_num)
        if not np.isfinite(arm_sample).all():
            return None
        return arm_sample

    def joint_state_callback(self, msg):
        if len(msg.position) == 0:
            return

        value_by_name = dict(zip(msg.name, msg.position))
        ordered_values = []
        missing_names = []
        for target_name in self.HAND_JOINT_NAMES:
            if target_name in value_by_name:
                ordered_values.append(value_by_name[target_name])
            else:
                ordered_values.append(np.nan)
                missing_names.append(target_name)

        if missing_names:
            missing_key = tuple(missing_names)
            if missing_key not in self.missing_logged:
                self.get_logger().warn(
                    f'Missing joints in {self.joint_states_topic}.position: {missing_names}. Waiting for complete state.'
                )
                self.missing_logged.add(missing_key)

        sample = np.array(ordered_values, dtype=np.float64)[self.hand_order_indices]
        self.latest_joint_sample = sample
        if self.is_complete_joint_sample(sample):
            if self.startup_joint_sample is None:
                self.get_logger().info('Captured startup hand joint state for initial move.')
            self.startup_joint_sample = sample.copy()

    def is_complete_joint_sample(self, sample):
        if sample is None:
            return False
        return np.isfinite(sample).all()

    def handle_initial_move(self):
        if self.playback_state == 'waiting_for_initial_state':
            if self.enable_initial_move and self.latest_joint_sample is None:
                self.get_logger().warn(
                    f'Waiting for {self.joint_states_topic} before initial move...',
                    throttle_duration_sec=5,
                )
                return

            if self.enable_initial_move and self.startup_joint_sample is None:
                self.get_logger().warn(
                    'Waiting for complete startup hand joint states before initial move...',
                    throttle_duration_sec=5,
                )
                return

            if self.enable_wrist_alignment and self.startup_arm_sample is None:
                self.startup_arm_sample = self.get_complete_arm_sample()
                if self.startup_arm_sample is None:
                    self.get_logger().warn(
                        f'Waiting for {self.arm_controller.state_topic} before wrist alignment...',
                        throttle_duration_sec=5,
                    )
                    return
                self.get_logger().info('Captured startup arm joint state for wrist alignment.')

            if not self.publish_initial_move():
                rclpy.shutdown()
                return

            self.playback_state = 'initial_move_running'
            self.initial_move_finish_time = time.monotonic() + self.initial_move_duration_sec
            self.current_index = 0
            return

        if self.playback_state == 'initial_move_running':
            if time.monotonic() < self.initial_move_finish_time:
                return
            if self.initial_hold_sec > 0.0:
                self.playback_state = 'initial_hold'
                self.initial_hold_finish_time = time.monotonic() + self.initial_hold_sec
                self.get_logger().info(
                    f'Initial move complete; holding first sample for {self.initial_hold_sec:.2f}s before playback.'
                )
                return
            self.start_playback_after_initial_move()

        if self.playback_state == 'initial_hold':
            if time.monotonic() < self.initial_hold_finish_time:
                return
            self.start_playback_after_initial_move()

    def start_playback_after_initial_move(self):
        if self.current_index >= self.trajectory.shape[0]:
            if self.loop:
                self.current_index = 0
            else:
                self.get_logger().info('Initial move complete; no remaining trajectory samples to publish.')
                rclpy.shutdown()
                return

        self.playback_state = 'playing'
        self.get_logger().info(
            f'Initial hold complete; starting trajectory playback at sample index {self.current_index}.'
        )

    def publish_initial_move(self):
        steps = max(2, int(np.ceil(self.initial_move_duration_sec * self.initial_move_rate_hz)))
        alpha = np.linspace(0.0, 1.0, steps)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        seconds_per_point = self.initial_move_duration_sec / steps

        hand_ok = True
        arm_ok = True

        if self.enable_initial_move:
            current = self.startup_joint_sample.copy()
            first_sample = self.trajectory[0]
            current = np.where(np.isfinite(current), current, first_sample)
            initial_trajectory = current + (first_sample - current) * alpha[:, np.newaxis]
            hand_ok = self.hand_controller.command_joint_position(initial_trajectory[:, :16], seconds_per_point)

        if self.enable_wrist_alignment:
            current_arm = self.startup_arm_sample.copy().reshape(1, self.arm_controller.joints_num)
            target_arm = current_arm.copy()
            target_arm[:, 5] = self.wrist_joint6_target
            wrist_trajectory = current_arm + (target_arm - current_arm) * alpha[:, np.newaxis]
            arm_ok = self.arm_controller.command_joint_position(wrist_trajectory, seconds_per_point)

        if not hand_ok or not arm_ok:
            self.get_logger().error('Failed to publish initial move trajectory.')
            return False

        published_actions = []
        if self.enable_initial_move:
            published_actions.append('hand trajectory soft-start')
        if self.enable_wrist_alignment:
            published_actions.append(f'arm wrist alignment to {self.wrist_joint6_target:.2f} rad')
        self.get_logger().info(
            f'Published startup motion ({", ".join(published_actions)}) with {steps} interpolated points over '
            f'{self.initial_move_duration_sec:.2f}s.'
        )
        return True

    def handle_finished_trajectory(self):
        if self.loop:
            self.current_index = 0
            return False

        self.get_logger().info('Finished publishing trajectory.')
        rclpy.shutdown()
        return True

    def publish_next_sample(self):
        if self.playback_state != 'playing':
            self.handle_initial_move()
            return

        if self.current_index >= self.trajectory.shape[0]:
            self.handle_finished_trajectory()
            return

        sample = np.asarray(self.trajectory[self.current_index], dtype=np.float64).copy()
        self.update_current_limited_joints(fallback_sample=sample)
        sample = self.apply_current_limit_locks(sample)
        hand_pose = np.reshape(sample[:16], (1, 16))
        speed = 1.0 / self.play_rate_hz

        hand_ok = self.hand_controller.command_joint_position(hand_pose, speed)
        if not hand_ok:
            self.get_logger().error(f'Failed to publish sample index {self.current_index}')
            rclpy.shutdown()
            return

        self.last_published_sample = sample.copy()
        self.current_index += 1
        if self.current_index >= self.trajectory.shape[0]:
            self.handle_finished_trajectory()


def prompt_playback_mode(input_func=input):
    while True:
        user_input = input_func(
            'Select playback mode: 1=fixed matrix, 2=recorded data: '
        ).strip()
        if user_input == '1':
            return PLAYBACK_MODE_FIXED
        if user_input == '2':
            return PLAYBACK_MODE_RECORDED

        print('Invalid input. Please enter 1 for fixed matrix or 2 for recorded data.')


def main(args=None):
    rclpy.init(args=args)
    hand_controller = LeapHand('position_player_hand_controller')
    arm_controller = LeapArm('position_player_arm_controller')
    player = None
    executor = MultiThreadedExecutor()

    try:
        player = PositionPlayer(hand_controller, arm_controller)
        executor.add_node(hand_controller)
        executor.add_node(arm_controller)
        executor.add_node(player)
        executor.spin()
    finally:
        if player is not None:
            player.destroy_node()
        hand_controller.destroy_node()
        arm_controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
