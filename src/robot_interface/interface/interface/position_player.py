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
from interface.position_player_trajectory import build_fixed_chopstick_trajectory, validate_chopstick_angles


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEFAULT_DATA_FILE = ''
DEFAULT_PLAY_RATE_HZ = 5.0
DEFAULT_START_INDEX = 0
DEFAULT_END_INDEX = -1
DEFAULT_LOOP = False
DEFAULT_HAND_POSITION_BIAS = [
    0.0, 0.0, 0.0, 0.1,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.05,
]
DEFAULT_ENABLE_INITIAL_MOVE = True
DEFAULT_JOINT_STATES_TOPIC = '/hand_joint_states'
DEFAULT_INITIAL_MOVE_DURATION_SEC = 4.0
DEFAULT_INITIAL_MOVE_RATE_HZ = 20.0
DEFAULT_INITIAL_HOLD_SEC = 5.0
DEFAULT_ENABLE_WRIST_ALIGNMENT = True
DEFAULT_WRIST_JOINT6_TARGET = 2.07
DEFAULT_USE_FIXED_CHOPSTICK_TRAJECTORY = True
DEFAULT_CHOPSTICK_OPEN_ANGLE_DEG = 25.0
DEFAULT_CHOPSTICK_CLOSE_ANGLE_DEG = 10.0


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

        self.declare_parameter('data_file', DEFAULT_DATA_FILE)
        self.declare_parameter('data_dir', DEFAULT_DATA_DIR)
        self.declare_parameter('play_rate_hz', DEFAULT_PLAY_RATE_HZ)
        self.declare_parameter('start_index', DEFAULT_START_INDEX)
        self.declare_parameter('end_index', DEFAULT_END_INDEX)
        self.declare_parameter('loop', DEFAULT_LOOP)
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

        self.data_file = self.get_parameter('data_file').value
        self.data_dir = self.get_parameter('data_dir').value
        self.play_rate_hz = float(self.get_parameter('play_rate_hz').value)
        self.start_index = int(self.get_parameter('start_index').value)
        self.end_index = int(self.get_parameter('end_index').value)
        self.loop = bool(self.get_parameter('loop').value)
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

        if self.play_rate_hz <= 0.0:
            raise ValueError(f'play_rate_hz must be positive, got {self.play_rate_hz}')
        if self.initial_move_rate_hz <= 0.0:
            raise ValueError(f'initial_move_rate_hz must be positive, got {self.initial_move_rate_hz}')
        if self.initial_hold_sec < 0.0:
            raise ValueError(f'initial_hold_sec must be non-negative, got {self.initial_hold_sec}')
        validate_chopstick_angles(self.chopstick_open_angle_deg, self.chopstick_close_angle_deg)

        self.trajectory = self.map_real_hand_to_controller_order(
            self.apply_hand_position_bias(self.build_hand_trajectory(self.load_trajectory()))
        )
        self.current_index = 0
        self.latest_joint_sample = None
        self.startup_joint_sample = None
        self.startup_arm_sample = None
        self.missing_logged = set()
        self.playback_state = 'playing'
        self.initial_move_finish_time = None
        self.initial_hold_finish_time = None

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

        self.get_logger().info(
            f'Loaded {self.trajectory.shape[0]} samples from {self.data_file}; '
            f'playing at {self.play_rate_hz:.2f} Hz.'
        )
        if self.use_fixed_chopstick_trajectory:
            self.get_logger().info(
                'Using fixed chopstick trajectory: only hand1/hand5 move, '
                f'open={self.chopstick_open_angle_deg:.2f} deg, '
                f'close={self.chopstick_close_angle_deg:.2f} deg.'
            )
        if np.any(self.hand_position_bias != 0.0):
            self.get_logger().info(f'Using hand position bias: {self.hand_position_bias.tolist()}')
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

        bias = np.array(value, dtype=np.float64)
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

        return build_fixed_chopstick_trajectory(
            trajectory.shape[0],
            self.chopstick_open_angle_deg,
            self.chopstick_close_angle_deg,
        )

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

        sample = self.trajectory[self.current_index]
        hand_pose = np.reshape(sample[:16], (1, 16))
        speed = 1.0 / self.play_rate_hz

        hand_ok = self.hand_controller.command_joint_position(hand_pose, speed)
        if not hand_ok:
            self.get_logger().error(f'Failed to publish sample index {self.current_index}')
            rclpy.shutdown()
            return

        self.current_index += 1
        if self.current_index >= self.trajectory.shape[0]:
            self.handle_finished_trajectory()


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
