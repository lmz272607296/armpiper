import math
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class DualTopicTest(Node):
    def __init__(self) -> None:
        super().__init__("vla_test")

        self.declare_parameter("piper_topic", "/piper/joint_trajectory_cmd")
        self.declare_parameter("hand_topic", "/hand_controller/joint_trajectory")
        self.declare_parameter("lift_deg", 30.0)
        self.declare_parameter("cycles", 5)
        self.declare_parameter("step_sec", 1.0)
        self.declare_parameter("arm_hold_sec", 2.0)
        self.declare_parameter("use_piper_gripper", False)
        self.declare_parameter("piper_gripper_position", 0.0)

        piper_topic = self.get_parameter("piper_topic").value
        hand_topic = self.get_parameter("hand_topic").value

        self.lift_deg = float(self.get_parameter("lift_deg").value)
        self.cycles = int(self.get_parameter("cycles").value)
        self.step_sec = float(self.get_parameter("step_sec").value)
        self.arm_hold_sec = float(self.get_parameter("arm_hold_sec").value)
        self.use_piper_gripper = bool(self.get_parameter("use_piper_gripper").value)
        self.piper_gripper_position = float(
            self.get_parameter("piper_gripper_position").value
        )

        self.piper_pub = self.create_publisher(JointTrajectory, piper_topic, 10)
        self.hand_pub = self.create_publisher(JointTrajectory, hand_topic, 10)

        self.piper_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        ]
        self.hand_joint_names = [f"hand{i}" for i in range(16)]

        self.get_logger().info(f"Publishing Piper command to: {piper_topic}")
        self.get_logger().info(f"Publishing hand command to: {hand_topic}")
        if not self.use_piper_gripper:
            self.get_logger().info("Piper gripper command disabled (safe mode)")

    def publish_dual(self, arm_positions, hand_positions, duration_sec: float) -> None:
        arm_msg = JointTrajectory()
        arm_msg.header.stamp = self.get_clock().now().to_msg()
        arm_msg.joint_names = self.piper_joint_names

        arm_point = JointTrajectoryPoint()
        arm_point.positions = arm_positions
        arm_point.time_from_start = Duration(seconds=duration_sec).to_msg()
        arm_msg.points.append(arm_point)
        self.piper_pub.publish(arm_msg)

        hand_msg = JointTrajectory()
        hand_msg.header.stamp = self.get_clock().now().to_msg()
        hand_msg.joint_names = self.hand_joint_names

        point = JointTrajectoryPoint()
        point.positions = hand_positions
        point.time_from_start = Duration(seconds=duration_sec).to_msg()
        hand_msg.points.append(point)

        self.hand_pub.publish(hand_msg)

    def run_sequence(self) -> None:
        arm_neutral = [0.0] * 7
        arm_lift = [0.0] * 7
        arm_lift[1] = math.radians(self.lift_deg)
        arm_lift[6] = self.piper_gripper_position if self.use_piper_gripper else 0.0

        hand_open = [0.0] * 16
        hand_close = [0.45] * 16

        self.get_logger().info("Step 1: move to neutral/open")
        self.publish_dual(arm_neutral, hand_open, self.step_sec)
        time.sleep(self.step_sec)

        self.get_logger().info(f"Step 2: lift arm by {self.lift_deg:.1f} deg")
        self.publish_dual(arm_lift, hand_open, self.arm_hold_sec)
        time.sleep(self.arm_hold_sec)

        self.get_logger().info("Step 3: repeat hand close/open while holding arm")
        for i in range(self.cycles):
            self.get_logger().info(f"Cycle {i + 1}/{self.cycles}: close")
            self.publish_dual(arm_lift, hand_close, self.step_sec)
            time.sleep(self.step_sec)

            self.get_logger().info(f"Cycle {i + 1}/{self.cycles}: open")
            self.publish_dual(arm_lift, hand_open, self.step_sec)
            time.sleep(self.step_sec)

        self.get_logger().info("Sequence done")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DualTopicTest()
    try:
        node.run_sequence()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
