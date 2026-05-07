import threading
import time

import numpy as np
from builtin_interfaces.msg import Time
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class LeapArm(Node):
    def __init__(self, name):
        super().__init__(name)

        self.declare_parameter("command_topic", "/arm_controller/joint_trajectory")
        self.declare_parameter("state_topic", "/joint_states_single")
        self.declare_parameter("frame_id", "piper_single")
        self.declare_parameter("min_point_duration", 0.02)
        self.declare_parameter("command_mode", "joint_state")
        self.declare_parameter("joint_state_command_topic", "/joint_states")
        self.declare_parameter("joint_state_rate_hz", 50.0)

        self.command_topic = self.get_parameter("command_topic").value
        self.state_topic = self.get_parameter("state_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.min_point_duration = float(self.get_parameter("min_point_duration").value)
        self.command_mode = str(self.get_parameter("command_mode").value).strip().lower()
        self.joint_state_command_topic = self.get_parameter("joint_state_command_topic").value
        self.joint_state_rate_hz = float(self.get_parameter("joint_state_rate_hz").value)

        self.publisher = self.create_publisher(JointTrajectory, self.command_topic, 10)
        self.joint_state_publisher = self.create_publisher(JointState, self.joint_state_command_topic, 10)
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
        self.last_state_time = 0.0
        self._warned_missing_state = False
        self._stream_lock = threading.Lock()
        self._stream_generation = 0
        self._stream_thread = None
        self._destroyed = False

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
        self.last_state_time = time.time()

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

    def _starts_at_current_position(self, pose):
        if self.raw_positions is None or len(pose) <= 1:
            return False

        current_position = np.asarray(self.raw_positions, dtype=float).reshape(-1)
        return np.allclose(pose[0], current_position, atol=1e-4)

    def _build_message(self, pose, point_duration):
        msg = JointTrajectory()
        msg.header.stamp = Time(sec=0, nanosec=0)
        msg.header.frame_id = self.frame_id
        msg.joint_names = list(self.command_joint_names)

        start_time = 0.0 if self._starts_at_current_position(pose) else point_duration
        for index, point in enumerate(pose):
            trajectory_point = JointTrajectoryPoint()
            trajectory_point.positions = [float(value) for value in point]
            trajectory_point.time_from_start = Duration(
                seconds=start_time + index * point_duration
            ).to_msg()
            msg.points.append(trajectory_point)

        return msg

    def _build_joint_state_message(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.name = list(self.command_joint_names)
        msg.position = [float(value) for value in positions]
        msg.velocity = [0.0] * self.joints_num
        msg.effort = [0.0] * self.joints_num
        return msg

    def _timed_waypoints(self, pose, point_duration):
        points = np.asarray(pose, dtype=float).copy()
        if points.shape[0] == 1:
            if self.raw_positions is None:
                return points, np.array([0.0], dtype=float)
            current_position = np.asarray(self.raw_positions, dtype=float).reshape(1, self.joints_num)
            points = np.concatenate((current_position, points), axis=0)
        elif self.raw_positions is not None:
            current_position = np.asarray(self.raw_positions, dtype=float).reshape(self.joints_num)
            if not np.allclose(points[0], current_position, atol=1e-4):
                points = np.concatenate((current_position.reshape(1, self.joints_num), points), axis=0)

        times = np.arange(points.shape[0], dtype=float) * point_duration
        return points, times

    def _interpolate_waypoint(self, points, times, elapsed):
        if points.shape[0] == 1 or elapsed >= times[-1]:
            return points[-1]

        index = int(np.searchsorted(times, elapsed, side="right") - 1)
        index = max(0, min(index, points.shape[0] - 2))
        segment_duration = times[index + 1] - times[index]
        if segment_duration <= 0.0:
            return points[index + 1]

        ratio = (elapsed - times[index]) / segment_duration
        return points[index] + ratio * (points[index + 1] - points[index])

    def _publish_joint_state_stream(self, pose, point_duration, generation):
        points, times = self._timed_waypoints(pose, point_duration)
        total_duration = float(times[-1]) if len(times) else 0.0
        rate_hz = max(5.0, float(self.joint_state_rate_hz))
        sleep_duration = 1.0 / rate_hz

        start_time = time.monotonic()
        while True:
            with self._stream_lock:
                if self._destroyed or generation != self._stream_generation:
                    return

            elapsed = min(time.monotonic() - start_time, total_duration)
            positions = self._interpolate_waypoint(points, times, elapsed)
            self.joint_state_publisher.publish(self._build_joint_state_message(positions))

            if elapsed >= total_duration:
                return
            time.sleep(sleep_duration)

    def _command_joint_state_position(self, pose, point_duration):
        with self._stream_lock:
            self._stream_generation += 1
            generation = self._stream_generation

        thread = threading.Thread(
            target=self._publish_joint_state_stream,
            args=(pose, point_duration, generation),
            daemon=True,
        )
        thread.start()
        with self._stream_lock:
            if not self._destroyed and generation == self._stream_generation:
                self._stream_thread = thread

        self.get_logger().debug(
            f"Streaming {len(pose)} Piper JointState waypoints to {self.joint_state_command_topic} "
            f"at {self.joint_state_rate_hz:.1f} Hz"
        )
        return True

    def command_joint_position(self, desired_pose, speed):
        try:
            pose = self._normalize_pose(desired_pose)
            point_duration = float(speed)
            if not np.isfinite(point_duration) or point_duration <= 0.0:
                raise ValueError(f"speed must be a positive finite number, got {speed}")
            point_duration = max(point_duration, self.min_point_duration)
            if self.command_mode in ("joint_state", "joint_states"):
                return self._command_joint_state_position(pose, point_duration)

            if self.command_mode not in ("trajectory", "joint_trajectory"):
                raise ValueError(
                    "command_mode must be 'joint_state' or 'joint_trajectory', "
                    f"got {self.command_mode!r}"
                )

            msg = self._build_message(pose, point_duration)
            self.publisher.publish(msg)

            self.get_logger().debug(
                f"Published {len(pose)} Piper JointTrajectory command points to {self.command_topic}"
            )
            return True
        except Exception as exc:
            self.get_logger().error(f"Publishing error: {repr(exc)}")
            return False

    def destroy_node(self):
        with self._stream_lock:
            self._destroyed = True
            self._stream_generation += 1
            stream_thread = self._stream_thread

        if stream_thread is not None and stream_thread.is_alive() and stream_thread is not threading.current_thread():
            stream_thread.join(timeout=0.2)
        super().destroy_node()

    def hold_current_position(self, duration=0.05):
        if self.raw_positions is None:
            self.get_logger().warn("当前没有可用的机械臂状态，无法发送停止保持指令。")
            return False
        current_pose = np.asarray(self.raw_positions, dtype=float).reshape(1, self.joints_num)
        return self.command_joint_position(current_pose, duration)
