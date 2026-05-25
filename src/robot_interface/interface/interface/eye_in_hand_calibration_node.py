#!/usr/bin/env python3

import json
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


def quaternion_to_matrix(quat_msg):
    x = float(quat_msg.x)
    y = float(quat_msg.y)
    z = float(quat_msg.z)
    w = float(quat_msg.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return np.eye(3)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def rpy_to_matrix(rpy):
    roll, pitch, yaw = [float(value) for value in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def pose_to_matrix(pose_msg):
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_matrix(pose_msg.orientation)
    transform[:3, 3] = np.array([
        pose_msg.position.x,
        pose_msg.position.y,
        pose_msg.position.z,
    ], dtype=float)
    return transform


def xyz_rpy_to_matrix(xyz, rpy):
    transform = np.eye(4)
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = np.array(xyz, dtype=float)
    return transform


def load_matrix(matrix_file):
    if not matrix_file:
        return None
    expanded_path = os.path.expanduser(str(matrix_file))
    if not os.path.exists(expanded_path):
        return None
    matrix = np.loadtxt(expanded_path, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"外参矩阵必须是 4x4，当前 {expanded_path} 形状为 {matrix.shape}")
    return matrix


def apply_base_position_bias(base_point, x_bias_m):
    base_point = np.asarray(base_point, dtype=float).copy()
    base_point[0] += float(x_bias_m)
    return base_point


class EyeInHandCalibrationNode(Node):
    def __init__(self):
        super().__init__('eye_in_hand_calibration_node')

        self.declare_parameter('input_detection_topic', '/yolo_3d_detections_json')
        self.declare_parameter('output_detection_topic', '/yolo_3d_detections_base_json')
        self.declare_parameter('end_pose_topic', '/end_pose_stamped')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('camera_frame_id', 'camera_color_optical_frame')
        self.declare_parameter('flange_frame_id', 'link6')
        self.declare_parameter('hand_eye_matrix_file', 'hand_eye_calibration.txt')
        self.declare_parameter('camera_xyz_in_flange', [0.07720, -0.0165, 0.09])
        self.declare_parameter('camera_rpy_in_flange', [-1.5707963268, 0.0, -1.5707963268])
        self.declare_parameter('max_end_pose_age_sec', 1.0)
        self.declare_parameter('publish_invalid_detections', True)
        self.declare_parameter('log_interval_sec', 2.0)
        self.declare_parameter('base_x_bias_m', -0.03)

        input_topic = self.get_parameter('input_detection_topic').value
        output_topic = self.get_parameter('output_detection_topic').value
        end_pose_topic = self.get_parameter('end_pose_topic').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.camera_frame_id = self.get_parameter('camera_frame_id').value
        self.flange_frame_id = self.get_parameter('flange_frame_id').value
        self.max_end_pose_age_sec = float(self.get_parameter('max_end_pose_age_sec').value)
        self.publish_invalid_detections = bool(self.get_parameter('publish_invalid_detections').value)
        self.log_interval_sec = float(self.get_parameter('log_interval_sec').value)
        self.base_x_bias_m = float(self.get_parameter('base_x_bias_m').value)
        self.last_log_time = 0.0

        matrix_file = self.get_parameter('hand_eye_matrix_file').value
        try:
            self.flange_to_camera = load_matrix(matrix_file)
        except Exception as exc:
            self.get_logger().fatal(f'读取手眼外参矩阵失败: {exc}')
            raise

        if self.flange_to_camera is None:
            camera_xyz = self.get_parameter('camera_xyz_in_flange').value
            camera_rpy = self.get_parameter('camera_rpy_in_flange').value
            self.flange_to_camera = xyz_rpy_to_matrix(camera_xyz, camera_rpy)
            self.get_logger().warn(
                f'未找到 {matrix_file}，使用参数 camera_xyz_in_flange={camera_xyz}, '
                f'camera_rpy_in_flange={camera_rpy} 作为 link6->camera 外参。'
            )
        else:
            self.get_logger().info(f'已加载手眼外参矩阵: {os.path.expanduser(str(matrix_file))}')

        self.latest_base_to_flange = None
        self.latest_end_pose_time = 0.0
        self.detection_sub = self.create_subscription(String, input_topic, self.detection_callback, 10)
        self.end_pose_sub = self.create_subscription(PoseStamped, end_pose_topic, self.end_pose_callback, 10)
        self.detection_pub = self.create_publisher(String, output_topic, 10)

        self.get_logger().info(
            f'眼在手上坐标转换节点已启动: {input_topic} -> {output_topic}，'
            f'末端位姿 {end_pose_topic}，输出坐标系 {self.base_frame_id}，'
            f'base_x_bias_m={self.base_x_bias_m:.3f}。'
        )

    def end_pose_callback(self, msg):
        self.latest_base_to_flange = pose_to_matrix(msg.pose)
        self.latest_end_pose_time = time.time()

        if msg.header.frame_id and msg.header.frame_id not in (self.base_frame_id, self.flange_frame_id):
            self.get_logger().warn(
                f'收到 /end_pose_stamped frame_id={msg.header.frame_id}。'
                f'本节点按 base->{self.flange_frame_id} 位姿解释该消息。'
            )

    def detection_callback(self, msg):
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f'解析检测 JSON 失败: {exc}')
            return

        if not isinstance(detections, list):
            self.get_logger().warn('检测 JSON 不是列表，已忽略。')
            return

        if not self.end_pose_is_ready():
            self.warn_throttled('尚未收到新鲜的末端法兰位姿，无法把相机坐标转换到基座坐标。')
            if not self.publish_invalid_detections:
                return
            self.publish_detections(detections)
            return

        base_to_camera = self.latest_base_to_flange @ self.flange_to_camera
        transformed_count = 0
        for detection in detections:
            center_3d = detection.get('center_3d') if isinstance(detection, dict) else None
            base_point = self.transform_center(center_3d, base_to_camera)
            if base_point is None:
                continue

            detection['camera_3d'] = center_3d
            detection['center_3d'] = base_point
            detection['center_3d_base'] = base_point
            detection['frame_id'] = self.base_frame_id
            detection['source_frame_id'] = self.camera_frame_id
            transformed_count += 1

        self.publish_detections(detections)
        now = time.time()
        if now - self.last_log_time >= self.log_interval_sec:
            self.get_logger().info(f'已转换 {transformed_count}/{len(detections)} 个目标到 {self.base_frame_id} 坐标系。')
            self.last_log_time = now

    def end_pose_is_ready(self):
        if self.latest_base_to_flange is None:
            return False
        return time.time() - self.latest_end_pose_time <= self.max_end_pose_age_sec

    def transform_center(self, center_3d, base_to_camera):
        if not center_3d:
            return None
        try:
            camera_point = np.array([
                float(center_3d['x']),
                float(center_3d['y']),
                float(center_3d['z']),
                1.0,
            ], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None

        if not np.all(np.isfinite(camera_point)):
            return None

        base_point = base_to_camera @ camera_point
        base_point = apply_base_position_bias(base_point[:3], self.base_x_bias_m)
        return {
            'x': float(base_point[0]),
            'y': float(base_point[1]),
            'z': float(base_point[2]),
        }

    def publish_detections(self, detections):
        msg = String()
        msg.data = json.dumps(detections)
        self.detection_pub.publish(msg)

    def warn_throttled(self, text):
        now = time.time()
        if now - self.last_log_time >= self.log_interval_sec:
            self.get_logger().warn(text)
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = EyeInHandCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
