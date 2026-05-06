import time

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState


class LeapArm(Node):
    def __init__(self, name):
        super().__init__(name)

        # piper_single_ctrl publishes arm feedback on /joint_states_single and,
        # in start_single_piper.launch.py, remaps joint_ctrl_single to /joint_states.
        self.declare_parameter("command_topic", "/joint_states")
        self.declare_parameter("state_topic", "/joint_states_single")
        self.declare_parameter("frame_id", "piper_single")
        self.declare_parameter("min_point_duration", 0.02)

        self.command_topic = self.get_parameter("command_topic").value
        self.state_topic = self.get_parameter("state_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.min_point_duration = float(self.get_parameter("min_point_duration").value)

        self.publisher = self.create_publisher(JointState, self.command_topic, 10)
        self.subscription = self.create_subscription(
            JointState,
            self.state_topic,
            self.controller_state_callback,
            10,
        )
        self.joint_names = [f"joint{i}" for i in range(1, 7)]
        self.command_joint_names = self.joint_names
        self.joints_num = 6
        self.raw_positions = None
        self._warned_missing_state = False

    def controller_state_callback(self, msg):
        value_map = dict(zip(msg.name, msg.position))
        missing_names = [name for name in self.joint_names if name not in value_map]
        if missing_names:
            if not self._warned_missing_state:
                self.get_logger().warn(
                    f"JointState缺少机械臂关节 {missing_names}，可用关节: {list(value_map.keys())}"
                )
                self._warned_missing_state = True
            return

        positions = np.array([value_map[name] for name in self.joint_names], dtype=float)
        if not np.all(np.isfinite(positions)):
            self.get_logger().warn("机械臂JointState包含NaN或无限值，已忽略。")
            return

        self.raw_positions = positions.reshape(1, self.joints_num)

    def _normalize_pose(self, desired_pose):
        pose = np.asarray(desired_pose, dtype=float)
        if pose.ndim == 1:
            pose = pose.reshape(1, -1)
        if pose.ndim != 2:
            raise ValueError(f"desired_pose must be a 1D or 2D array, got shape {pose.shape}")

        if pose.shape[1] == 5:
            pose = np.insert(pose, 3, 0.0, axis=1)
        elif pose.shape[1] == 6:
            pose = pose.copy()
        else:
            raise ValueError(
                "desired_pose must contain 5 or 6 joints. "
                f"Got {pose.shape[1]} joints."
            )

        if not np.all(np.isfinite(pose)):
            raise ValueError("desired_pose contains NaN or infinite values")

        return pose

    def _build_message(self, point, point_duration):
        motion_speed = int(np.clip(round(30.0 / max(point_duration, self.min_point_duration)), 1, 100))

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.name = self.command_joint_names
        msg.position = point.tolist()
        msg.velocity = [motion_speed] * self.joints_num
        msg.effort = [0.0] * self.joints_num

        return msg

    def command_joint_position(self, desired_pose, speed):
        try:
            pose = self._normalize_pose(desired_pose)
            point_duration = max(float(speed), self.min_point_duration)

            for index, point in enumerate(pose):
                msg = self._build_message(point, point_duration)
                self.publisher.publish(msg)
                if index < len(pose) - 1:
                    time.sleep(point_duration)

            self.get_logger().debug(
                f"Published {len(pose)} Piper JointState command points to {self.command_topic}"
            )
            return True
        except Exception as exc:
            self.get_logger().error(f"Publishing error: {repr(exc)}")
            return False
