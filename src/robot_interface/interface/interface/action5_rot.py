import os
import time

import numpy as np
import rclpy

from interface.arm_controller import LeapArm
from interface.hand_controller import LeapHand


class CubeRotationDemo:
    def __init__(self):
        self.state_timeout = 5.0
        self.batch_size = 12
        self.hand_point_duration = 0.3
        self.init_point_duration = 1.5
        self.playback_points = 901
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

    def load_hand_trajectory(self):
        data = np.load(self.hand_trajectory_path).astype(float)
        if data.ndim != 2 or data.shape[1] != 16:
            raise ValueError(f"Expected hand trajectory shape (N, 16), got {data.shape}")
        return data

    def wait_for_joint_state(self, leap, arm):
        leap.get_logger().info("正在等待来自 /hand_joint_states 的手部消息和 /joint_states_single 的机械臂消息...")
        start_time = time.time()
        while (leap.raw_positions is None or arm.raw_positions is None) and rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.05)
            rclpy.spin_once(arm, timeout_sec=0.05)
            if time.time() - start_time > self.state_timeout:
                raise TimeoutError(
                    "在5秒内未接收到 /hand_joint_states 的 hand0-hand15 或 /joint_states_single 的 joint1-joint6 消息，请检查发布器节点。"
                )

        if leap.raw_positions is None or arm.raw_positions is None:
            raise RuntimeError("未读取到手部或机械臂当前位置，旋转展示已取消。")

    def move_to_initial_pose(self, leap, arm, data):
        current_hand = np.asarray(leap.raw_positions, dtype=float).reshape(1, 16)
        initial_hand = np.concatenate((current_hand, data[:1]), axis=0)

        current_arm = np.asarray(arm.raw_positions, dtype=float).reshape(1, 6)
        initial_arm = np.concatenate((current_arm, self.arm_target), axis=0)

        arm.get_logger().info(f"发送新机械臂初始姿态: {self.arm_target.squeeze().tolist()}")
        if not arm.command_joint_position(initial_arm, self.init_point_duration):
            raise RuntimeError("机械臂初始姿态指令发布失败。")
        if not leap.command_joint_position(initial_hand, self.init_point_duration):
            raise RuntimeError("手部初始姿态指令发布失败。")

        time.sleep(initial_arm.shape[0] * self.init_point_duration + 1.0)

    def play_rotation(self, leap, data):
        playback_count = min(self.playback_points, data.shape[0])
        leap.get_logger().info(f"开始播放旋转方块手部轨迹，共 {playback_count} 个点。")

        for start in range(0, playback_count, self.batch_size):
            current_batch = data[start:start + self.batch_size, :]
            compensated_batch = current_batch + self.joint_offsets
            if not leap.command_joint_position(compensated_batch, self.hand_point_duration):
                raise RuntimeError(f"手部轨迹指令发布失败，起始点 {start}。")
            time.sleep(self.hand_point_duration * len(compensated_batch))

    def run(self):
        rclpy.init()
        leap = LeapHand("action7_rot_hand")
        arm = LeapArm("action7_rot_arm")
        try:
            leap.leap_dof_lower = self.leap_dof_lower
            leap.leap_dof_upper = self.leap_dof_upper

            data = self.load_hand_trajectory()
            self.wait_for_joint_state(leap, arm)
            self.move_to_initial_pose(leap, arm, data)
            self.play_rotation(leap, data)
            leap.get_logger().info("旋转方块展示完成。")
        finally:
            leap.destroy_node()
            arm.destroy_node()
            rclpy.shutdown()


def main():
    CubeRotationDemo().run()


if __name__ == "__main__":
    main()
