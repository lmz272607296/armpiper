import os
import signal
import time

import numpy as np
import rclpy
from std_msgs.msg import String

from interface.arm_controller import LeapArm
from interface.hand_controller import LeapHand
from interface.motion_utils import refresh_current_joint_states, scale_action


STOP_TOPIC = '/action_stop'
ROTATION_ACTION = 'action5_rot'


class CubeRotationDemo:
    def __init__(self):
        self.state_timeout = 5.0
        self.state_stale_timeout = 1.0
        self.batch_size = 12
        self.hand_point_duration = 0.3
        self.soft_start_duration = 2.5
        self.soft_start_steps = 25
        self.playback_points = 901
        self.stop_requested = False
        self.arm_target = np.array([0.0, 0.0, 0.0, 1.14, 0.0, 2.0], dtype=float).reshape(1, 6)
        self.hand_trajectory_path = os.path.join(os.path.dirname(__file__), "cache", "hand16.npy")

        self.leap_dof_upper = np.array([
            1.047, 2.23, 1.885, 2.042,
            1.047, 2.23, 1.885, 2.042,
            1.047, 2.23, 1.885, 2.042,
            2.094, 2.443, 1.90, 1.88,
        ])
        self.leap_dof_lower = np.array([
            -1.047, -0.314, -0.506, -0.366,
            -1.047, -0.314, -0.506, -0.366,
            -1.047, -0.314, -0.506, -0.366,
            -0.349, -0.47, -1.20, -1.34,
        ])

        self.joint_offsets = np.array([
            0.0, 0.0, 0.3, 0.0,
            0.0, 0.0, -0.3, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ], dtype=float)

    def request_stop(self, reason=''):
        if self.stop_requested:
            return
        self.stop_requested = True
        if reason:
            print(f'[{ROTATION_ACTION}] stop requested: {reason}')

    def stop_callback(self, msg):
        token = msg.data.strip().lower()
        if token in ('', 'all', ROTATION_ACTION):
            self.request_stop(f'received stop topic {token or "all"}')

    def stop_signal_handler(self, signum, _frame):
        self.request_stop(f'received signal {signum}')

    def spin_nodes(self, leap=None, arm=None, timeout_sec=0.05):
        if leap is not None:
            rclpy.spin_once(leap, timeout_sec=timeout_sec)
        if arm is not None:
            rclpy.spin_once(arm, timeout_sec=timeout_sec)

    def sleep_with_abort(self, leap=None, arm=None, duration=0.0):
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            self.spin_nodes(leap, arm, timeout_sec=0.05)
            self.ensure_state_fresh(leap, arm)
            if self.stop_requested:
                return False
        return not self.stop_requested

    def ensure_running(self, leap=None, arm=None):
        self.spin_nodes(leap, arm, timeout_sec=0.0)
        self.ensure_state_fresh(leap, arm)
        if self.stop_requested:
            raise InterruptedError('旋转动作已被停止。')

    def ensure_state_fresh(self, leap=None, arm=None):
        now = time.time()
        if leap is not None and leap.last_state_time > 0.0 and now - leap.last_state_time > self.state_stale_timeout:
            self.request_stop('/hand_joint_states 超时，停止旋转动作')
        if arm is not None and arm.last_state_time > 0.0 and now - arm.last_state_time > self.state_stale_timeout:
            self.request_stop('/joint_states_single 超时，停止旋转动作')

    def hold_current_pose(self, leap, arm):
        try:
            self.spin_nodes(leap, arm, timeout_sec=0.1)
            hand_ok = leap.hold_current_position()
            arm_ok = arm.hold_current_position()
            if hand_ok or arm_ok:
                leap.get_logger().info('已发送当前位置保持指令，用于覆盖当前旋转轨迹。')
        except Exception as exc:
            leap.get_logger().warn(f'发送停止保持指令失败: {exc}')

    def load_hand_trajectory(self):
        data = np.load(self.hand_trajectory_path).astype(float)
        if data.ndim != 2 or data.shape[1] != 16:
            raise ValueError(f"Expected hand trajectory shape (N, 16), got {data.shape}")
        return data

    def wait_for_joint_state(self, leap, arm):
        leap.get_logger().info("正在等待来自 /hand_joint_states 的手部消息和 /joint_states_single 的机械臂消息...")
        refresh_current_joint_states(
            leap=leap,
            arm=arm,
            need_hand=True,
            need_arm=True,
            timeout=self.state_timeout,
        )
        self.ensure_running(leap, arm)

    def move_to_initial_pose(self, leap, arm, data):
        current_hand, current_arm = refresh_current_joint_states(
            leap=leap,
            arm=arm,
            need_hand=True,
            need_arm=True,
            timeout=self.state_timeout,
        )
        initial_hand_target = data[:1] + self.joint_offsets
        initial_hand = scale_action(current_hand, initial_hand_target, self.soft_start_steps)
        initial_arm = scale_action(current_arm, self.arm_target, self.soft_start_steps)

        arm_point_duration = self.soft_start_duration / self.soft_start_steps
        hand_point_duration = self.soft_start_duration / initial_hand.shape[0]

        arm.get_logger().info(
            f"软启动到机械臂初始姿态，耗时 {self.soft_start_duration:.1f}s: "
            f"{self.arm_target.squeeze().tolist()}"
        )
        leap.get_logger().info(
            f"软启动到手部旋转初始姿态，耗时 {self.soft_start_duration:.1f}s。"
        )
        if not arm.command_joint_position(initial_arm, arm_point_duration):
            raise RuntimeError("机械臂初始姿态指令发布失败。")
        if not leap.command_joint_position(initial_hand, hand_point_duration):
            raise RuntimeError("手部初始姿态指令发布失败。")

        if not self.sleep_with_abort(leap, arm, self.soft_start_duration + 0.5):
            raise InterruptedError('软启动阶段被停止。')

    def play_rotation(self, leap, arm, data):
        playback_count = min(self.playback_points, data.shape[0])
        leap.get_logger().info(f"开始播放旋转方块手部轨迹，共 {playback_count} 个点。")

        for start in range(0, playback_count, self.batch_size):
            self.ensure_running(leap, arm)
            current_batch = data[start:start + self.batch_size, :]
            compensated_batch = current_batch + self.joint_offsets
            if not leap.command_joint_position(compensated_batch, self.hand_point_duration):
                raise RuntimeError(f"手部轨迹指令发布失败，起始点 {start}。")
            if not self.sleep_with_abort(leap, arm, self.hand_point_duration * len(compensated_batch)):
                raise InterruptedError(f'旋转轨迹在批次 {start} 被停止。')

    def run(self):
        rclpy.init()
        leap = LeapHand("action7_rot_hand")
        arm = LeapArm("action7_rot_arm")
        leap.create_subscription(String, STOP_TOPIC, self.stop_callback, 10)
        arm.create_subscription(String, STOP_TOPIC, self.stop_callback, 10)
        signal.signal(signal.SIGTERM, self.stop_signal_handler)
        signal.signal(signal.SIGINT, self.stop_signal_handler)
        try:
            leap.leap_dof_lower = self.leap_dof_lower
            leap.leap_dof_upper = self.leap_dof_upper

            data = self.load_hand_trajectory()
            self.wait_for_joint_state(leap, arm)
            self.move_to_initial_pose(leap, arm, data)
            self.play_rotation(leap, arm, data)
            leap.get_logger().info("旋转方块展示完成。")
        except InterruptedError as exc:
            leap.get_logger().warn(str(exc))
        finally:
            self.hold_current_pose(leap, arm)
            leap.destroy_node()
            arm.destroy_node()
            rclpy.shutdown()


def main():
    CubeRotationDemo().run()


if __name__ == "__main__":
    main()
