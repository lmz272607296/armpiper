import time
import signal

import numpy as np
import rclpy
from std_msgs.msg import String

from interface.arm_controller import LeapArm
from interface.hand_controller import LeapHand
from interface.hand_reset_utils import THUMB_RESET_DELAY_SEC, build_delayed_thumb_reset_trajectory
from interface.motion_utils import refresh_current_joint_states


ARM_INITIAL_JOINTS = np.array([0.0, 0.0, 0.0, 0.0, 0.2121, 0.0], dtype=float)
HAND_INITIAL_JOINTS = np.zeros(16, dtype=float)
HAND_INITIAL_JOINTS[[0, 8, 12]] = np.deg2rad(49.0)  # hand1, hand5, hand9
HAND_INITIAL_JOINTS[[3, 11, 15]] = np.deg2rad(15.0)  # hand3, hand7, hand11
HAND_INITIAL_JOINTS[[2, 10, 14]] = np.deg2rad(30.0)  # hand2, hand6, hand10

ACTION_NAME = "action8_rst"
SIGNAL_TOPIC = "/recognized_signal"
STOP_TOPIC = "/action_stop"
RESET_SIGNAL = "10"
ARM_SPEEDUP_FACTOR = 2.5
HAND_SPEEDUP_FACTOR = 2.0
STAGE_DELAY_SEC = 0.2


class ArmReset:
    def __init__(self):
        self.state_timeout = 5.0
        self.trajectory_points = 24
        self.arm_point_duration = 0.15 / ARM_SPEEDUP_FACTOR
        self.hand_point_duration = 0.15 / HAND_SPEEDUP_FACTOR
        self.stop_requested = False

    def request_stop(self, reason=""):
        if self.stop_requested:
            return
        self.stop_requested = True
        if reason:
            print(f"[{ACTION_NAME}] stop requested: {reason}")

    def recognized_signal_callback(self, msg):
        if msg.data.strip() == RESET_SIGNAL:
            self.request_stop(f"received reset signal {RESET_SIGNAL}")

    def stop_callback(self, msg):
        token = msg.data.strip().lower()
        if token in ("", "all", ACTION_NAME):
            self.request_stop(f"received stop topic {token or 'all'}")

    def stop_signal_handler(self, signum, _frame):
        self.request_stop(f"received signal {signum}")

    def spin_nodes(self, arm=None, hand=None, timeout_sec=0.05):
        if arm is not None:
            rclpy.spin_once(arm, timeout_sec=timeout_sec)
        if hand is not None:
            rclpy.spin_once(hand, timeout_sec=timeout_sec)

    def ensure_running(self, arm=None, hand=None):
        self.spin_nodes(arm, hand, timeout_sec=0.0)
        if self.stop_requested:
            raise InterruptedError("复位动作已被停止。")

    def sleep_with_abort(self, arm=None, hand=None, duration=0.0):
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            self.spin_nodes(arm, hand, timeout_sec=0.05)
            if self.stop_requested:
                return False
        return not self.stop_requested

    def hold_current_pose(self, arm, hand):
        try:
            self.spin_nodes(arm, hand, timeout_sec=0.1)
            arm_ok = arm.hold_current_position()
            hand_ok = hand.hold_current_position()
            if arm_ok or hand_ok:
                arm.get_logger().info("已发送当前位置保持指令，用于覆盖当前复位轨迹。")
        except Exception as exc:
            arm.get_logger().warn(f"发送停止保持指令失败: {exc}")

    def wait_for_current_position(self, arm):
        arm.get_logger().info("正在等待来自 /joint_states_single 的第一条机械臂消息...")
        return refresh_current_joint_states(
            arm=arm,
            need_arm=True,
            timeout=self.state_timeout,
        )[1]

    def wait_for_current_positions(self, arm, hand):
        arm.get_logger().info("正在强制刷新 /joint_states_single 和 /hand_joint_states 的当前关节消息...")
        current_hand, current_arm = refresh_current_joint_states(
            leap=hand,
            arm=arm,
            need_hand=True,
            need_arm=True,
            timeout=self.state_timeout,
        )
        self.ensure_running(arm, hand)
        return current_arm, current_hand

    def build_reset_trajectory(self, current_position, target_position):
        current_position = np.asarray(current_position, dtype=float)
        target_position = np.asarray(target_position, dtype=float).reshape(current_position.shape)
        # Use an ease-in/ease-out profile so reset starts softly and ends softly.
        phases = np.linspace(0.0, 1.0, self.trajectory_points + 1, dtype=float)[1:].reshape(-1, 1)
        factors = 0.5 - 0.5 * np.cos(np.pi * phases)
        return current_position + factors * (target_position - current_position)

    def send_reset_commands(self, arm, hand, arm_trajectory, hand_trajectory):
        if len(arm_trajectory) != len(hand_trajectory):
            raise ValueError("机械臂和手部复位轨迹点数不一致。")

        self.ensure_running(arm, hand)
        if not arm.command_joint_position(arm_trajectory, self.arm_point_duration):
            raise RuntimeError("机械臂复位轨迹指令发布失败。")
        if not hand.command_joint_position(hand_trajectory, self.hand_point_duration):
            raise RuntimeError("手部复位轨迹指令发布失败。")

        total_duration = max(self.arm_point_duration, self.hand_point_duration) * len(arm_trajectory)
        if not self.sleep_with_abort(arm, hand, total_duration):
            raise InterruptedError("复位轨迹执行过程中被停止。")

    def send_delayed_thumb_reset_commands(self, arm, hand, arm_trajectory, current_hand, hand_target):
        thumb_delay_points = min(len(arm_trajectory), int(np.ceil(THUMB_RESET_DELAY_SEC / self.hand_point_duration)))
        hand_trajectory = build_delayed_thumb_reset_trajectory(
            current_hand,
            hand_target,
            trajectory_points=len(arm_trajectory),
            thumb_delay_points=thumb_delay_points,
        )
        self.send_reset_commands(arm, hand, arm_trajectory, hand_trajectory)

    def run(self):
        rclpy.init()
        arm = LeapArm("action10_rst_arm")
        hand = LeapHand("action10_rst_hand")
        arm.create_subscription(String, SIGNAL_TOPIC, self.recognized_signal_callback, 10)
        hand.create_subscription(String, SIGNAL_TOPIC, self.recognized_signal_callback, 10)
        arm.create_subscription(String, STOP_TOPIC, self.stop_callback, 10)
        hand.create_subscription(String, STOP_TOPIC, self.stop_callback, 10)
        signal.signal(signal.SIGTERM, self.stop_signal_handler)
        signal.signal(signal.SIGINT, self.stop_signal_handler)
        try:
            current_arm, current_hand = self.wait_for_current_positions(arm, hand)
            arm_reset_trajectory = self.build_reset_trajectory(current_arm, ARM_INITIAL_JOINTS)

            arm.get_logger().info(
                f"开始软启动复位：{len(arm_reset_trajectory)} 个平滑轨迹点，预计耗时 "
                f"{len(arm_reset_trajectory) * max(self.arm_point_duration, self.hand_point_duration):.2f}s。"
            )
            self.send_delayed_thumb_reset_commands(arm, hand, arm_reset_trajectory, current_hand, HAND_INITIAL_JOINTS)

            if not self.sleep_with_abort(arm, hand, STAGE_DELAY_SEC):
                raise InterruptedError("复位结束等待阶段被停止。")
            arm.get_logger().info("机械臂和手部复位指令已完成发送。")
        except InterruptedError as exc:
            arm.get_logger().warn(str(exc))
        finally:
            self.hold_current_pose(arm, hand)
            hand.destroy_node()
            arm.destroy_node()
            rclpy.shutdown()


def main():
    ArmReset().run()


if __name__ == "__main__":
    main()
