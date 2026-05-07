import time
import threading

import numpy as np
import rclpy

from interface.arm_controller import LeapArm
from interface.hand_controller import LeapHand
from interface.motion_utils import refresh_current_joint_states


class ArmReset:
    def __init__(self):
        self.state_timeout = 5.0
        self.trajectory_points = 12
        self.point_duration = 0.3

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
        return current_arm, current_hand

    def build_reset_trajectory(self, current_position):
        target_position = np.zeros_like(current_position)
        factors = np.linspace(0.0, 1.0, self.trajectory_points).reshape(-1, 1)
        return current_position + factors * (target_position - current_position)

    def send_reset_commands(self, arm, hand, arm_trajectory, hand_trajectory):
        results = {}

        def send_arm():
            results["arm"] = arm.command_joint_position(arm_trajectory, self.point_duration)

        def send_hand():
            results["hand"] = hand.command_joint_position(hand_trajectory, self.point_duration)

        arm_thread = threading.Thread(target=send_arm)
        hand_thread = threading.Thread(target=send_hand)
        arm_thread.start()
        hand_thread.start()
        arm_thread.join()
        hand_thread.join()

        if not results.get("arm"):
            raise RuntimeError("机械臂复位指令发布失败。")
        if not results.get("hand"):
            raise RuntimeError("手部复位指令发布失败。")

    def run(self):
        rclpy.init()
        arm = LeapArm("action10_rst_arm")
        hand = LeapHand("action10_rst_hand")
        try:
            current_arm, current_hand = self.wait_for_current_positions(arm, hand)
            arm_reset_trajectory = self.build_reset_trajectory(current_arm)
            hand_reset_trajectory = self.build_reset_trajectory(current_hand)

            arm.get_logger().info(
                f"开始软启动复位：{len(arm_reset_trajectory)} 个轨迹点，目标为机械臂和手部所有关节零点。"
            )
            self.send_reset_commands(arm, hand, arm_reset_trajectory, hand_reset_trajectory)

            time.sleep(len(arm_reset_trajectory) * self.point_duration + 0.5)
            rclpy.spin_once(arm, timeout_sec=0.1)
            rclpy.spin_once(hand, timeout_sec=0.1)
            arm.get_logger().info("机械臂和手部复位指令已完成发送。")
        finally:
            hand.destroy_node()
            arm.destroy_node()
            rclpy.shutdown()


def main():
    ArmReset().run()


if __name__ == "__main__":
    main()
