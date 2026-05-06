import time

import numpy as np
import rclpy

from interface.arm_controller import LeapArm


class ArmReset:
    def __init__(self):
        self.state_timeout = 5.0
        self.trajectory_points = 12
        self.point_duration = 0.3

    def wait_for_current_position(self, arm):
        arm.get_logger().info("正在等待来自 /joint_states_single 的第一条机械臂消息...")
        start_time = time.time()
        while arm.raw_positions is None and rclpy.ok():
            rclpy.spin_once(arm, timeout_sec=0.1)
            if time.time() - start_time > self.state_timeout:
                raise TimeoutError(
                    "在5秒内未接收到 /joint_states_single 的 joint1-joint6 消息，请检查发布器节点。"
                )

        if arm.raw_positions is None:
            raise RuntimeError("未读取到机械臂当前位置，复位已取消。")

        return np.asarray(arm.raw_positions, dtype=float).reshape(1, arm.joints_num)

    def build_reset_trajectory(self, current_position):
        target_position = np.zeros_like(current_position)
        factors = np.linspace(0.0, 1.0, self.trajectory_points).reshape(-1, 1)
        return current_position + factors * (target_position - current_position)

    def run(self):
        rclpy.init()
        arm = LeapArm("action10_rst_arm")
        try:
            current_position = self.wait_for_current_position(arm)
            reset_trajectory = self.build_reset_trajectory(current_position)

            arm.get_logger().info(
                f"开始软启动复位：{len(reset_trajectory)} 个轨迹点，目标为所有机械臂关节零点。"
            )
            if not arm.command_joint_position(reset_trajectory, self.point_duration):
                raise RuntimeError("机械臂复位指令发布失败。")

            time.sleep(len(reset_trajectory) * self.point_duration + 0.5)
            rclpy.spin_once(arm, timeout_sec=0.1)
            arm.get_logger().info("机械臂复位指令已完成发送。")
        finally:
            arm.destroy_node()
            rclpy.shutdown()


def main():
    ArmReset().run()


if __name__ == "__main__":
    main()
