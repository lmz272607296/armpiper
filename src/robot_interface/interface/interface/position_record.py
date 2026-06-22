import os
import select
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
from datetime import datetime

import numpy as np
import rclpy
try:
    from control_msgs.msg import DynamicJointState
except ImportError:
    DynamicJointState = None
from dynamixel_interfaces.srv import SetDataToDxl
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from interface.hand_controller import extract_currents_from_dynamic_joint_state


DEFAULT_JOINT_STATES_TOPIC = '/hand_joint_states'
DEFAULT_DYNAMIC_JOINT_STATES_TOPIC = '/hand_dynamic_joint_states'
DEFAULT_HAND_COMMAND_TOPIC = '/hand_controller/joint_trajectory'
DEFAULT_SAMPLE_RATE_HZ = 10.0
DEFAULT_LOCK_COMMAND_RATE_HZ = 20.0
DEFAULT_SOFT_START_DURATION_SEC = 2.0
DEFAULT_SOFT_START_RATE_HZ = 20.0
DEFAULT_HOLD_COMMAND_DURATION_SEC = 0.12
DEFAULT_ENABLE_STARTUP_CURRENT_LIMIT = True
DEFAULT_STARTUP_CURRENT_LIMIT_MA = 550.0
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEFAULT_PREVIEW_PLAY_RATE_HZ = 10.0
DEFAULT_PREVIEW_PLAYBACK_SPEED_SCALE = 1.5
DEFAULT_PREVIEW_INTERPOLATION_FACTOR = 2
DEFAULT_PREVIEW_TEMP_DIR = '/tmp/opencode'
DEFAULT_DXL_DATA_SERVICE_NAME = '/hand/dynamixel_hardware_interface/set_dxl_data'
DEFAULT_DXL_DATA_SERVICE_TIMEOUT_SEC = 5.0
DEFAULT_FALLBACK_DXL_DATA_SERVICE_NAMES = (
    '/dynamixel_hardware_interface/set_dxl_data',
    'dynamixel_hardware_interface/set_dxl_data',
)
INITIAL_POSITION_OFFSET_DEG = 180.0

# ============================================================
# 下面这两个矩阵是给你手动调 16 个 hand 关节用的。
# 顺序必须始终保持为：
# hand0, hand1, hand2, hand3, hand4, hand5, hand6, hand7,
# hand8, hand9, hand10, hand11, hand12, hand13, hand14, hand15
#
# 1. DEFAULT_HAND_INITIAL_POSE_DEG:
#    16 个关节的启动初始位姿，单位是“加上初始位姿 180 度之后”的角度制。
#    其中 hand8-hand14 是固定关节，hand0-hand7 和 hand15 是可移动可录制关节。
# 2. DEFAULT_HAND_POSITION_BIAS_DEG:
#    16 个关节共享的额外偏置矩阵，单位同样是角度制。
#    可移动关节：偏置只作用在启动初始位姿上。
#    固定关节：偏置会一直加在固定关节目标位姿上。
#
# 你后续如果要微调，优先直接改下面这两组数字即可。
# 程序内部发送到控制器时，会自动换算成“减去 180 度后的弧度制”。
# ============================================================
DEFAULT_HAND_INITIAL_POSE_DEG = [
    114.0, 199.5, 298.5, 212.0,
    142.1, 183.8, 283.4, 250.0,
    226.0, 213.6, 284.0, 188.3,
    241.1, 164.0, 270.0, 223.0,
]
DEFAULT_HAND_POSITION_BIAS_DEG = [
    0.0, 0.0, 5.0, 20.0,
    0.0, 0.0, -0.0, -10.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 30.0, 0.0,
]


class PositionRecord(Node):
    HAND_JOINT_NAMES = [f'hand{i}' for i in range(16)]
    FIXED_JOINT_NAMES = [f'hand{i}' for i in range(8, 15)]
    MOVABLE_JOINT_NAMES = [f'hand{i}' for i in range(8)] + ['hand15']
    MOVABLE_DXL_IDS = tuple(range(8)) + (15,)
    FIXED_DXL_IDS = tuple(range(8, 15))
    FIXED_JOINT_INDICES = np.arange(8, 15, dtype=np.int64)

    def __init__(self):
        super().__init__('position_record')

        self.declare_parameter('joint_states_topic', DEFAULT_JOINT_STATES_TOPIC)
        self.declare_parameter('dynamic_joint_states_topic', DEFAULT_DYNAMIC_JOINT_STATES_TOPIC)
        self.declare_parameter('hand_command_topic', DEFAULT_HAND_COMMAND_TOPIC)
        self.declare_parameter('sample_rate_hz', DEFAULT_SAMPLE_RATE_HZ)
        self.declare_parameter('lock_command_rate_hz', DEFAULT_LOCK_COMMAND_RATE_HZ)
        self.declare_parameter('soft_start_duration_sec', DEFAULT_SOFT_START_DURATION_SEC)
        self.declare_parameter('soft_start_rate_hz', DEFAULT_SOFT_START_RATE_HZ)
        self.declare_parameter('hold_command_duration_sec', DEFAULT_HOLD_COMMAND_DURATION_SEC)
        self.declare_parameter('enable_startup_current_limit', DEFAULT_ENABLE_STARTUP_CURRENT_LIMIT)
        self.declare_parameter('startup_current_limit_ma', DEFAULT_STARTUP_CURRENT_LIMIT_MA)
        self.declare_parameter('output_dir', DEFAULT_OUTPUT_DIR)
        self.declare_parameter('dxl_data_service_name', DEFAULT_DXL_DATA_SERVICE_NAME)
        self.declare_parameter('dxl_data_service_timeout_sec', DEFAULT_DXL_DATA_SERVICE_TIMEOUT_SEC)
        self.declare_parameter(
            'hand_initial_pose',
            DEFAULT_HAND_INITIAL_POSE_DEG,
            descriptor=ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter(
            'hand_position_bias',
            DEFAULT_HAND_POSITION_BIAS_DEG,
            descriptor=ParameterDescriptor(dynamic_typing=True),
        )

        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value).strip()
        self.dynamic_joint_states_topic = str(self.get_parameter('dynamic_joint_states_topic').value).strip()
        self.hand_command_topic = str(self.get_parameter('hand_command_topic').value).strip()
        self.sample_rate_hz = float(self.get_parameter('sample_rate_hz').value)
        self.lock_command_rate_hz = float(self.get_parameter('lock_command_rate_hz').value)
        self.soft_start_duration_sec = float(self.get_parameter('soft_start_duration_sec').value)
        self.soft_start_rate_hz = float(self.get_parameter('soft_start_rate_hz').value)
        self.hold_command_duration_sec = float(self.get_parameter('hold_command_duration_sec').value)
        self.enable_startup_current_limit = bool(self.get_parameter('enable_startup_current_limit').value)
        self.startup_current_limit_ma = float(self.get_parameter('startup_current_limit_ma').value)
        self.output_dir = str(self.get_parameter('output_dir').value).strip()
        self.dxl_data_service_timeout_sec = float(self.get_parameter('dxl_data_service_timeout_sec').value)
        self.hand_initial_pose_deg = self.parse_hand_joint_vector(
            self.get_parameter('hand_initial_pose').value,
            parameter_name='hand_initial_pose',
        )
        self.hand_position_bias_deg = self.parse_hand_joint_vector(
            self.get_parameter('hand_position_bias').value,
            parameter_name='hand_position_bias',
        )
        self.hand_position_bias = np.deg2rad(self.hand_position_bias_deg)

        preferred_dxl_data_service_name = str(self.get_parameter('dxl_data_service_name').value).strip()
        self.dxl_data_service_names = self.build_dxl_data_service_names(preferred_dxl_data_service_name)

        if self.sample_rate_hz <= 0.0:
            raise ValueError(f'sample_rate_hz must be positive, got {self.sample_rate_hz}')
        if self.lock_command_rate_hz <= 0.0:
            raise ValueError(f'lock_command_rate_hz must be positive, got {self.lock_command_rate_hz}')
        if self.soft_start_duration_sec <= 0.0:
            raise ValueError(
                f'soft_start_duration_sec must be positive, got {self.soft_start_duration_sec}'
            )
        if self.soft_start_rate_hz <= 0.0:
            raise ValueError(f'soft_start_rate_hz must be positive, got {self.soft_start_rate_hz}')
        if self.hold_command_duration_sec <= 0.0:
            raise ValueError(
                f'hold_command_duration_sec must be positive, got {self.hold_command_duration_sec}'
            )
        if self.enable_startup_current_limit and self.startup_current_limit_ma <= 0.0:
            raise ValueError(
                f'startup_current_limit_ma must be positive, got {self.startup_current_limit_ma}'
            )
        if self.dxl_data_service_timeout_sec <= 0.0:
            raise ValueError(
                f'dxl_data_service_timeout_sec must be positive, got {self.dxl_data_service_timeout_sec}'
            )

        os.makedirs(self.output_dir, exist_ok=True)

        self.latest_sample = None
        self.latest_record_sample = None
        self.samples = []
        self.recording = False
        self.finished = False
        self.awaiting_preview_confirmation = False
        self.awaiting_save_confirmation = False
        self.preview_playback_active = False
        self.saved_output_path = None
        self.lock_ready = False
        self.soft_start_active = False
        self.soft_start_finish_time = 0.0
        self.state_lock = threading.Lock()
        self.samples_lock = threading.Lock()
        self.keyboard_stop_event = threading.Event()
        self.release_prepared = False
        self.latest_currents_ma = {}
        self.startup_current_limit_data_unavailable_logged = False
        self.disabled_joint_targets = np.full(len(self.HAND_JOINT_NAMES), np.nan, dtype=np.float64)
        self.current_joint_names = tuple(f'dxl{i}' for i in range(len(self.HAND_JOINT_NAMES)))
        self.initial_joint_targets_deg = self.hand_initial_pose_deg + self.hand_position_bias_deg
        self.initial_joint_targets = self.pose_deg_to_rad(self.initial_joint_targets_deg)
        self.fixed_joint_targets_deg = self.initial_joint_targets_deg[self.FIXED_JOINT_INDICES]
        self.fixed_joint_targets = self.initial_joint_targets[self.FIXED_JOINT_INDICES]
        self.dxl_data_clients = {
            service_name: self.create_client(SetDataToDxl, service_name)
            for service_name in self.dxl_data_service_names
        }

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
        self.dynamic_subscription = None
        if DynamicJointState is not None:
            self.dynamic_subscription = self.create_subscription(
                DynamicJointState,
                self.dynamic_joint_states_topic,
                self.dynamic_joint_state_callback,
                qos_profile,
            )
        elif self.enable_startup_current_limit:
            self.get_logger().warn(
                'control_msgs.msg.DynamicJointState 不可用，初始位姿电流保护将被禁用。'
            )
        self.command_publisher = self.create_publisher(JointTrajectory, self.hand_command_topic, 10)

        self.lock_timer = self.create_timer(1.0 / self.lock_command_rate_hz, self.maintain_locked_pose)
        self.record_timer = self.create_timer(1.0 / self.sample_rate_hz, self.record_sample)

        self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info(
            'Position recorder is ready. All 16 joints will soft-start to the configured initial pose.'
        )
        self.get_logger().info(
            f'Press SPACE once to release {self.MOVABLE_JOINT_NAMES}, then press SPACE again to start recording at '
            f'{self.sample_rate_hz:.2f} Hz.'
        )
        self.get_logger().info(
            'Press SPACE again to stop. Then you can choose whether to replay the trajectory once before deciding to keep it.'
        )
        self.get_logger().info(f'Joint order in saved data: {self.HAND_JOINT_NAMES}')
        self.get_logger().info(
            f'Hand initial pose deg+180 ({self.HAND_JOINT_NAMES}): {self.hand_initial_pose_deg.tolist()}'
        )
        if np.any(self.hand_position_bias_deg != 0.0):
            self.get_logger().info(
                f'Hand position bias deg ({self.HAND_JOINT_NAMES}): {self.hand_position_bias_deg.tolist()}'
            )
        self.get_logger().info(
            f'Startup target deg+180 ({self.HAND_JOINT_NAMES}): {self.initial_joint_targets_deg.tolist()}'
        )
        self.get_logger().info(
            f'Startup target rad ({self.HAND_JOINT_NAMES}): {np.round(self.initial_joint_targets, 4).tolist()}'
        )
        self.get_logger().info(
            'Recorded samples store movable joint states after adding hand_position_bias; '
            'fixed joints stay at the locked startup targets, and preview playback treats saved samples as already biased.'
        )
        if self.enable_startup_current_limit:
            self.get_logger().info(
                f'Startup current protection enabled at {self.startup_current_limit_ma:.1f} mA. '
                'Any overloaded joint will be frozen and stop accepting further commands in this run.'
            )
        self.get_logger().info(
            f'Fixed joints: {self.FIXED_JOINT_NAMES}; movable/recordable joints: {self.MOVABLE_JOINT_NAMES}'
        )

    def build_dxl_data_service_names(self, preferred_name):
        service_names = []
        for service_name in (preferred_name, *DEFAULT_FALLBACK_DXL_DATA_SERVICE_NAMES):
            normalized_name = str(service_name).strip()
            if normalized_name and normalized_name not in service_names:
                service_names.append(normalized_name)
        return service_names

    def parse_hand_joint_vector(self, value, parameter_name):
        if isinstance(value, str):
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                value = value[1:-1]
            value = [float(item.strip()) for item in value.split(',') if item.strip()]

        vector = np.array(value, dtype=np.float64)
        if vector.shape != (16,):
            raise ValueError(f'{parameter_name} must contain 16 values for hand0-hand15, got shape {vector.shape}')
        return vector

    def pose_deg_to_rad(self, value_deg):
        value_deg = np.asarray(value_deg, dtype=np.float64)
        return np.deg2rad(value_deg - INITIAL_POSITION_OFFSET_DEG)

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

    def joint_name_to_index(self, joint_name):
        normalized = self.normalize_current_joint_name(joint_name)
        if normalized is None:
            return None
        return int(normalized[len('hand'):])

    def dynamic_joint_state_callback(self, msg):
        try:
            current_updates = extract_currents_from_dynamic_joint_state(
                msg,
                joint_names=self.current_joint_names,
            )
            if not current_updates:
                return
            self.latest_currents_ma.update(current_updates)
        except Exception as exc:
            self.get_logger().error(f'Current message launching error: {repr(exc)}')

    def get_hand_current_snapshot(self):
        if DynamicJointState is None or not self.latest_currents_ma:
            if self.enable_startup_current_limit and not self.startup_current_limit_data_unavailable_logged:
                self.get_logger().warn('Hand current snapshot interface is unavailable; startup current protection is disabled.')
                self.startup_current_limit_data_unavailable_logged = True
            return {}

        hand_currents = {}
        for joint_name, current_ma in self.latest_currents_ma.items():
            hand_name = self.normalize_current_joint_name(joint_name)
            if hand_name is None:
                continue
            hand_currents[hand_name] = float(current_ma)
        return hand_currents

    def apply_hand_position_bias(self, sample):
        biased_sample = np.asarray(sample, dtype=np.float64).copy()
        biased_sample += self.hand_position_bias
        return biased_sample

    def resolve_current_limited_joint_position(
        self,
        joint_index,
        fallback_sample=None,
        prefer_fallback_position=False,
    ):
        if prefer_fallback_position:
            candidates = [
                fallback_sample,
                self.latest_record_sample,
                self.latest_sample,
                self.initial_joint_targets,
            ]
        else:
            candidates = [
                self.latest_sample,
                fallback_sample,
                self.initial_joint_targets,
            ]
        for candidate in candidates:
            if candidate is None:
                continue
            candidate_array = np.asarray(candidate, dtype=np.float64).reshape(-1)
            if joint_index >= candidate_array.shape[0]:
                continue
            if np.isfinite(candidate_array[joint_index]):
                return float(candidate_array[joint_index])
        return None

    def update_current_limited_joints(self, fallback_sample=None, prefer_fallback_position=False):
        if not self.enable_startup_current_limit:
            return {}

        newly_disabled = {}
        for joint_name, current_ma in self.get_hand_current_snapshot().items():
            hand_name = self.normalize_current_joint_name(joint_name) or str(joint_name)
            if abs(current_ma) < self.startup_current_limit_ma:
                continue

            joint_index = self.joint_name_to_index(hand_name)
            if joint_index is None:
                continue
            if np.isfinite(self.disabled_joint_targets[joint_index]):
                continue

            locked_position = self.resolve_current_limited_joint_position(
                joint_index,
                fallback_sample=fallback_sample,
                prefer_fallback_position=prefer_fallback_position,
            )
            if locked_position is None:
                continue

            self.disabled_joint_targets[joint_index] = locked_position
            newly_disabled[hand_name] = float(current_ma)
            self.get_logger().warn(
                f'Startup current limit hit on {hand_name}: {current_ma:.1f} mA >= {self.startup_current_limit_ma:.1f} mA. '
                f'This joint will not accept further commands in this run.'
            )

        return newly_disabled

    def apply_current_limit_locks(self, sample):
        limited_sample = np.asarray(sample, dtype=np.float64).copy()
        disabled_mask = np.isfinite(self.disabled_joint_targets)
        limited_sample[disabled_mask] = self.disabled_joint_targets[disabled_mask]
        return limited_sample

    def keyboard_loop(self):
        if not sys.stdin.isatty():
            self.get_logger().warn('Standard input is not a TTY; keyboard control is unavailable.')
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok() and not self.keyboard_stop_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue

                key = sys.stdin.read(1)
                if not key:
                    continue

                normalized_key = key.lower()
                with self.state_lock:
                    preview_playback_active = self.preview_playback_active
                    awaiting_preview_confirmation = self.awaiting_preview_confirmation
                    awaiting_confirmation = self.awaiting_save_confirmation

                if preview_playback_active:
                    continue

                if awaiting_preview_confirmation:
                    if normalized_key == 'y':
                        self.preview_pending_recording()
                        continue
                    if normalized_key == 'n':
                        self.skip_preview_and_prompt_save()
                        continue
                    if normalized_key == 'q' or key in ('\x03', '\x04'):
                        self.finish_recording('Exit requested from keyboard; unsaved samples were discarded.')
                        return
                    self.get_logger().info('Press Y to replay this recording once, or N to skip replay.')
                    continue

                if awaiting_confirmation:
                    if normalized_key == 'y':
                        self.confirm_save()
                        continue
                    if normalized_key == 'n':
                        self.discard_pending_samples()
                        continue
                    if normalized_key == 'q' or key in ('\x03', '\x04'):
                        self.finish_recording('Exit requested from keyboard; unsaved samples were discarded.')
                        return
                    self.get_logger().info('Press Y to keep the recording or N to discard it.')
                    continue

                if key == ' ':
                    self.handle_space_key()
                    continue

                if normalized_key == 'q' or key in ('\x03', '\x04'):
                    self.finish_recording('Exit requested from keyboard.')
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def handle_space_key(self):
        with self.state_lock:
            if self.finished:
                return
            if self.awaiting_preview_confirmation:
                self.get_logger().info('Press Y to replay the current recording once or N to skip replay first.')
                return
            if self.awaiting_save_confirmation:
                self.get_logger().info('Press Y to keep the current recording or N to discard it first.')
                return
            is_recording = self.recording
            lock_ready = self.lock_ready
            release_prepared = self.release_prepared

        if not lock_ready:
            self.get_logger().warn('Locked fingers are still moving to the fixed pose. Please wait a moment.')
            return

        if is_recording:
            self.stop_recording('Recording stopped from keyboard.')
            return

        if not release_prepared:
            self.prepare_free_fingers()
            return

        self.start_recording()

    def start_recording(self):
        with self.state_lock:
            if self.finished:
                return
            if self.awaiting_preview_confirmation:
                self.get_logger().info('Press Y/N to decide whether to replay the previous recording before starting a new one.')
                return
            if self.awaiting_save_confirmation:
                self.get_logger().info('Press Y/N to finish the previous recording before starting a new one.')
                return
            if self.recording:
                self.get_logger().info('Recording is already running.')
                return
            if not self.lock_ready:
                self.get_logger().warn('Locked fingers are not ready yet.')
                return
            if not self.release_prepared:
                self.get_logger().info(
                    f'Please press SPACE once to release {self.MOVABLE_JOINT_NAMES} before recording.'
                )
                return

        if self.latest_sample is None:
            self.get_logger().warn(f'Waiting for complete {self.joint_states_topic} data before recording.')
            return

        with self.samples_lock:
            self.samples = []
            self.saved_output_path = None

        with self.state_lock:
            if self.finished:
                return
            self.recording = True

        self.get_logger().info(
            f'Recording started. {self.FIXED_JOINT_NAMES} stay fixed; {self.MOVABLE_JOINT_NAMES} remain movable.'
        )

    def prepare_free_fingers(self):
        with self.state_lock:
            if self.finished:
                return
            if self.awaiting_preview_confirmation:
                self.get_logger().info('Press Y/N to decide whether to replay the previous recording first.')
                return
            if self.awaiting_save_confirmation:
                self.get_logger().info('Press Y/N to finish the previous recording first.')
                return
            if self.recording:
                self.get_logger().info('Recording is already running.')
                return
            if self.release_prepared:
                self.get_logger().info(
                    f'{self.MOVABLE_JOINT_NAMES} are already released. Press SPACE again to start recording.'
                )
                return
            if not self.lock_ready:
                self.get_logger().warn('Locked fingers are not ready yet.')
                return

        if not self.prepare_recording_torque_state():
            self.get_logger().error(f'Failed to release {self.MOVABLE_JOINT_NAMES} for recording.')
            return

        with self.state_lock:
            if self.finished:
                return
            self.release_prepared = True

        self.get_logger().info(
            f'{self.MOVABLE_JOINT_NAMES} torque released. Adjust them as needed, then press SPACE again to start recording.'
        )

    def stop_recording(self, reason):
        with self.state_lock:
            if self.finished:
                return
            if not self.recording:
                self.get_logger().info('Recording is not running.')
                return
            self.recording = False
            self.awaiting_preview_confirmation = True

        with self.samples_lock:
            sample_count = len(self.samples)

        self.get_logger().info(reason)
        if sample_count == 0:
            with self.state_lock:
                self.awaiting_preview_confirmation = False
                self.awaiting_save_confirmation = False
                self.release_prepared = False
            self.restore_default_torque_state()
            self.get_logger().warn('No samples were recorded; nothing was saved.')
            self.get_logger().info(f'Press SPACE once to release {self.MOVABLE_JOINT_NAMES} again, or q to quit.')
            return

        restored = self.restore_default_torque_state()
        if not restored:
            self.get_logger().warn('Failed to restore torque immediately after recording stopped.')

        self.get_logger().info(
            f'Recording stopped with {sample_count} samples. Press Y to replay it once, or N to skip replay.'
        )

    def preview_pending_recording(self):
        with self.state_lock:
            if self.finished:
                return
            if not self.awaiting_preview_confirmation:
                self.get_logger().info('There is no pending recording to replay.')
                return
            self.awaiting_preview_confirmation = False
            self.preview_playback_active = True

        self.get_logger().info('Replaying the current recorded trajectory once before save confirmation.')

        preview_ok = False
        preview_file_path = None
        try:
            preview_file_path = self.write_preview_file()
            preview_ok = self.run_position_player_preview(preview_file_path)
        finally:
            if preview_file_path and os.path.exists(preview_file_path):
                try:
                    os.remove(preview_file_path)
                except OSError:
                    pass
            with self.state_lock:
                self.preview_playback_active = False
                if not self.finished:
                    self.awaiting_save_confirmation = True

        if preview_ok:
            self.get_logger().info('Replay finished. Press Y to keep this recording or N to discard it.')
        else:
            self.get_logger().warn('Replay failed or exited abnormally. Press Y to keep this recording or N to discard it.')

    def skip_preview_and_prompt_save(self):
        with self.state_lock:
            if self.finished:
                return
            if not self.awaiting_preview_confirmation:
                self.get_logger().info('There is no pending recording waiting for replay selection.')
                return
            self.awaiting_preview_confirmation = False
            self.awaiting_save_confirmation = True

        self.get_logger().info('Replay skipped. Press Y to keep this recording or N to discard it.')

    def write_preview_file(self):
        with self.samples_lock:
            if not self.samples:
                raise RuntimeError('No samples available for preview playback.')
            preview_samples = np.vstack(self.samples)

        os.makedirs(DEFAULT_PREVIEW_TEMP_DIR, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            prefix='position_record_preview_',
            suffix='.npy',
            dir=DEFAULT_PREVIEW_TEMP_DIR,
            delete=False,
        )
        temp_file.close()
        np.save(temp_file.name, preview_samples)
        return temp_file.name

    def run_position_player_preview(self, preview_file_path):
        command = [
            'ros2',
            'run',
            'interface',
            'position_player',
            '--ros-args',
            '-p', f'data_file:={preview_file_path}',
            '-p', f'play_rate_hz:={DEFAULT_PREVIEW_PLAY_RATE_HZ:.6f}',
            '-p', f'playback_speed_scale:={DEFAULT_PREVIEW_PLAYBACK_SPEED_SCALE:.6f}',
            '-p', f'interpolation_factor:={DEFAULT_PREVIEW_INTERPOLATION_FACTOR}',
            '-p', 'enable_initial_move:=false',
            '-p', 'enable_wrist_alignment:=false',
            '-p', 'trajectory_includes_hand_position_bias:=true',
        ]

        try:
            completed = subprocess.run(
                command,
                input='2\n',
                text=True,
                check=False,
            )
        except FileNotFoundError:
            self.get_logger().warn('Could not start ros2 command for preview playback.')
            return False
        except Exception as exc:
            self.get_logger().warn(f'Preview playback failed to start: {repr(exc)}')
            return False

        if completed.returncode != 0:
            self.get_logger().warn(
                f'Preview playback exited with code {completed.returncode}. '
                'The recording can still be kept or discarded manually.'
            )
            return False
        return True

    def confirm_save(self):
        with self.state_lock:
            if self.finished:
                return
            if not self.awaiting_save_confirmation:
                self.get_logger().info('There is no pending recording to confirm.')
                return
            self.awaiting_save_confirmation = False

        output_path = self.save_samples()
        if output_path is not None:
            self.get_logger().info(f'Saved trajectory data to {output_path}')
        else:
            self.get_logger().warn('No samples were recorded; nothing was saved.')
        self.get_logger().info(f'Press SPACE once to release {self.MOVABLE_JOINT_NAMES} again, or q to quit.')

    def discard_pending_samples(self):
        with self.state_lock:
            if self.finished:
                return
            if not self.awaiting_save_confirmation:
                self.get_logger().info('There is no pending recording to discard.')
                return
            self.awaiting_save_confirmation = False

        with self.samples_lock:
            discarded_count = len(self.samples)
            self.samples = []
            self.saved_output_path = None

        self.get_logger().info(f'Discarded {discarded_count} recorded samples.')
        self.get_logger().info(f'Press SPACE once to release {self.MOVABLE_JOINT_NAMES} again, or q to quit.')

    def prepare_recording_torque_state(self):
        if not self.set_joint_torque_state(self.FIXED_DXL_IDS, True):
            return False
        return self.set_joint_torque_state(self.MOVABLE_DXL_IDS, False)

    def restore_default_torque_state(self):
        restored = True
        if not self.set_joint_torque_state(self.FIXED_DXL_IDS, True):
            restored = False
        if not self.set_joint_torque_state(self.MOVABLE_DXL_IDS, True):
            restored = False

        with self.state_lock:
            self.release_prepared = False

        if restored:
            self.get_logger().info(
                f'Restored torque for {self.MOVABLE_JOINT_NAMES}; the next recording will again require two SPACE presses.'
            )
        else:
            self.get_logger().warn('Failed to fully restore torque state after recording.')
        return restored

    def set_joint_torque_state(self, dxl_ids, enable):
        resolved = self.resolve_dxl_data_client()
        if resolved is None:
            self.get_logger().error(
                f'No set_dxl_data service became available within {self.dxl_data_service_timeout_sec:.1f}s. '
                f'Tried: {self.dxl_data_service_names}'
            )
            return False

        service_name, client = resolved
        target_value = 1 if enable else 0
        action = 'enable' if enable else 'disable'

        for dxl_id in dxl_ids:
            request = SetDataToDxl.Request()
            request.id = int(dxl_id)
            request.item_name = 'Torque Enable'
            request.item_data = target_value
            response = self.call_dxl_data_service(client, request)
            if response is None or not response.result:
                self.get_logger().error(
                    f'Failed to {action} torque for dxl{dxl_id} through {service_name}.'
                )
                return False

        target_group = [f'dxl{dxl_id}' for dxl_id in dxl_ids]
        self.get_logger().info(f'{action.capitalize()}d torque for {target_group} through {service_name}.')
        return True

    def resolve_dxl_data_client(self):
        deadline = time.monotonic() + self.dxl_data_service_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            for service_name in self.dxl_data_service_names:
                client = self.dxl_data_clients[service_name]
                if client.wait_for_service(timeout_sec=0.2):
                    return service_name, client
        return None

    def call_dxl_data_service(self, client, request):
        future = client.call_async(request)
        deadline = time.monotonic() + self.dxl_data_service_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if future.done():
                return future.result()
            time.sleep(0.05)
        return future.result() if future.done() else None

    def joint_state_callback(self, msg):
        if len(msg.position) == 0:
            return

        value_by_name = dict(zip(msg.name, msg.position))
        ordered_values = []
        missing_names = []
        for joint_name in self.HAND_JOINT_NAMES:
            joint_value = value_by_name.get(joint_name)
            if joint_value is None:
                missing_names.append(joint_name)
                continue
            ordered_values.append(joint_value)

        if missing_names:
            self.latest_sample = None
            self.latest_record_sample = None
            self.get_logger().warn(
                f'Waiting for complete hand state on {self.joint_states_topic}; missing {missing_names}.',
                throttle_duration_sec=5.0,
            )
            return

        sample = np.array(ordered_values, dtype=np.float64)
        if not np.isfinite(sample).all():
            self.latest_sample = None
            self.latest_record_sample = None
            self.get_logger().warn(
                f'Received non-finite values from {self.joint_states_topic}; sample skipped.',
                throttle_duration_sec=5.0,
            )
            return

        self.latest_sample = sample
        self.latest_record_sample = self.apply_hand_position_bias(sample)

    def maintain_locked_pose(self):
        with self.state_lock:
            if self.finished:
                return
            if self.preview_playback_active:
                return
            lock_ready = self.lock_ready
            soft_start_active = self.soft_start_active
            soft_start_finish_time = self.soft_start_finish_time

        if self.latest_sample is None:
            self.get_logger().warn(
                f'Waiting for complete {self.joint_states_topic} data before moving the hand to its startup pose.',
                throttle_duration_sec=5.0,
            )
            return

        if not lock_ready:
            if not soft_start_active:
                self.start_soft_start()
                return

            if time.monotonic() < soft_start_finish_time:
                return

            with self.state_lock:
                self.soft_start_active = False
                self.lock_ready = True
            self.get_logger().info(
                f'Soft-start complete. {self.FIXED_JOINT_NAMES} are now locked; {self.MOVABLE_JOINT_NAMES} can be released for recording.'
            )

        # Keep fixed joints pinned to their startup targets throughout recording.
        target_pose = self.build_target_pose(self.latest_sample)
        self.publish_hand_trajectory(target_pose.reshape(1, -1), self.hold_command_duration_sec)

    def start_soft_start(self):
        current_pose = self.latest_sample.copy()
        target_pose = self.build_startup_pose(current_pose)
        self.update_current_limited_joints(fallback_sample=target_pose)
        target_pose = self.apply_current_limit_locks(target_pose)
        startup_delta = np.abs(target_pose - current_pose)

        if np.all(startup_delta < 1e-4):
            with self.state_lock:
                self.lock_ready = True
                self.soft_start_active = False
            self.get_logger().info('Current hand pose already matches the configured startup pose; skipping soft-start.')
            return

        steps = max(2, int(np.ceil(self.soft_start_duration_sec * self.soft_start_rate_hz)))
        alpha = np.linspace(0.0, 1.0, steps, dtype=np.float64)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        trajectory = current_pose + (target_pose - current_pose) * alpha[:, np.newaxis]

        seconds_per_point = self.soft_start_duration_sec / steps
        self.publish_hand_trajectory(trajectory, seconds_per_point)

        with self.state_lock:
            self.soft_start_active = True
            self.soft_start_finish_time = time.monotonic() + self.soft_start_duration_sec

        self.get_logger().info(
            f'Published soft-start for all 16 joints over {self.soft_start_duration_sec:.2f}s '
            f'with {steps} interpolated points.'
        )

    def build_startup_pose(self, base_pose):
        pose = np.asarray(base_pose, dtype=np.float64).copy()
        pose[:] = self.initial_joint_targets
        return pose

    def build_target_pose(self, base_pose):
        pose = np.asarray(base_pose, dtype=np.float64).copy()
        pose[self.FIXED_JOINT_INDICES] = self.fixed_joint_targets
        return pose

    def publish_hand_trajectory(self, trajectory, seconds_per_point):
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(1, -1)
        if trajectory.ndim != 2 or trajectory.shape[1] != len(self.HAND_JOINT_NAMES):
            self.get_logger().error(f'Invalid trajectory shape for hand command: {trajectory.shape}')
            return False

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.joint_names = list(self.HAND_JOINT_NAMES)

        elapsed = 0.0
        for sample in trajectory:
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in sample]
            elapsed += seconds_per_point
            point.time_from_start = Duration(seconds=elapsed).to_msg()
            msg.points.append(point)

        self.command_publisher.publish(msg)
        return True

    def record_sample(self):
        with self.state_lock:
            if (
                self.finished
                or not self.recording
                or self.awaiting_preview_confirmation
                or self.awaiting_save_confirmation
                or self.preview_playback_active
            ):
                return

        if self.latest_sample is None or self.latest_record_sample is None:
            self.get_logger().warn(
                f'Waiting for complete {self.joint_states_topic} data before recording samples.',
                throttle_duration_sec=5.0,
            )
            return

        # Saved samples always contain the fixed joints at their configured startup targets.
        sample = self.build_target_pose(self.latest_record_sample)
        with self.samples_lock:
            self.samples.append(sample.copy())

    def save_samples(self):
        with self.samples_lock:
            if self.saved_output_path is not None:
                return self.saved_output_path
            if not self.samples:
                return None

            output_path = self.build_output_path()
            np.save(output_path, np.vstack(self.samples))
            self.saved_output_path = output_path
            return output_path

    def build_output_path(self):
        timestamp = datetime.now().strftime('%m%d_%H%M')
        base_path = os.path.join(self.output_dir, f'{timestamp}_position')
        output_path = f'{base_path}.npy'
        suffix = 1
        while os.path.exists(output_path):
            output_path = f'{base_path}_{suffix:02d}.npy'
            suffix += 1
        return output_path

    def finish_recording(self, reason):
        with self.state_lock:
            if self.finished:
                return
            was_recording = self.recording
            awaiting_preview_confirmation = self.awaiting_preview_confirmation
            awaiting_save_confirmation = self.awaiting_save_confirmation
            self.finished = True
            self.recording = False
            self.awaiting_preview_confirmation = False
            self.awaiting_save_confirmation = False
            self.preview_playback_active = False
            self.release_prepared = False
            self.keyboard_stop_event.set()

        self.get_logger().info(reason)
        if was_recording or awaiting_preview_confirmation or awaiting_save_confirmation:
            with self.samples_lock:
                unsaved_count = len(self.samples) if self.saved_output_path is None else 0
            if unsaved_count > 0:
                self.get_logger().info(f'Discarded {unsaved_count} unsaved recorded samples on shutdown.')

        self.set_joint_torque_state(self.FIXED_DXL_IDS, True)
        self.set_joint_torque_state(self.MOVABLE_DXL_IDS, True)

        if rclpy.ok():
            rclpy.shutdown()

    def cleanup(self):
        with self.state_lock:
            self.finished = True
            self.recording = False
            self.awaiting_preview_confirmation = False
            self.awaiting_save_confirmation = False
            self.preview_playback_active = False
            self.release_prepared = False
            self.keyboard_stop_event.set()

        self.set_joint_torque_state(self.FIXED_DXL_IDS, True)
        self.set_joint_torque_state(self.MOVABLE_DXL_IDS, True)


def main(args=None):
    rclpy.init(args=args)
    node = PositionRecord()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
