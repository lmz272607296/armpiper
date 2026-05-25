#!/usr/bin/env python3

import json
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def load_matrix(matrix_file):
    expanded_path = os.path.expanduser(str(matrix_file))
    matrix = np.loadtxt(expanded_path, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f'外参矩阵必须是 4x4，当前 {expanded_path} 形状为 {matrix.shape}')
    return matrix


def apply_base_position_bias(base_point, x_bias_m):
    base_point = np.asarray(base_point, dtype=float).copy()
    base_point[0] += float(x_bias_m)
    return base_point


class CameraBaseTransformNode(Node):
    def __init__(self):
        super().__init__('camera_base_transform_node')

        self.declare_parameter('input_detection_topic', '/yolo_3d_detections_json')
        self.declare_parameter('output_detection_topic', '/yolo_3d_detections_base_json')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('camera_frame_id', 'camera_color_optical_frame')
        self.declare_parameter('matrix_file', 'camera_base_calibration.txt')
        self.declare_parameter('publish_invalid_detections', True)
        self.declare_parameter('log_interval_sec', 2.0)
        self.declare_parameter('base_x_bias_m', -0.03)

        input_topic = self.get_parameter('input_detection_topic').value
        output_topic = self.get_parameter('output_detection_topic').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.camera_frame_id = self.get_parameter('camera_frame_id').value
        self.publish_invalid_detections = bool(self.get_parameter('publish_invalid_detections').value)
        self.log_interval_sec = float(self.get_parameter('log_interval_sec').value)
        self.base_x_bias_m = float(self.get_parameter('base_x_bias_m').value)
        self.last_log_time = 0.0

        matrix_file = self.get_parameter('matrix_file').value
        try:
            self.base_from_camera = load_matrix(matrix_file)
        except Exception as exc:
            self.get_logger().fatal(f'读取 camera->base 外参矩阵失败: {exc}')
            raise

        self.detection_sub = self.create_subscription(String, input_topic, self.detection_callback, 10)
        self.detection_pub = self.create_publisher(String, output_topic, 10)

        self.get_logger().info(
            f'固定相机坐标转换节点已启动: {input_topic} -> {output_topic}，'
            f'{self.camera_frame_id} -> {self.base_frame_id}，矩阵={os.path.expanduser(str(matrix_file))}，'
            f'base_x_bias_m={self.base_x_bias_m:.3f}'
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

        transformed_count = 0
        output_detections = detections if self.publish_invalid_detections else []

        for detection in detections:
            if not isinstance(detection, dict):
                continue
            center_3d = detection.get('center_3d')
            base_point = self.transform_center(center_3d)
            if base_point is None:
                if not self.publish_invalid_detections:
                    continue
            else:
                detection['camera_3d'] = center_3d
                detection['center_3d'] = base_point
                detection['center_3d_base'] = base_point
                detection['frame_id'] = self.base_frame_id
                detection['source_frame_id'] = self.camera_frame_id
                transformed_count += 1

            if not self.publish_invalid_detections:
                output_detections.append(detection)

        self.publish_detections(output_detections)
        now = time.time()
        if now - self.last_log_time >= self.log_interval_sec:
            self.get_logger().info(
                f'已转换 {transformed_count}/{len(detections)} 个目标到 {self.base_frame_id} 坐标系。'
            )
            self.last_log_time = now

    def transform_center(self, center_3d):
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

        base_point = self.base_from_camera @ camera_point
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


def main(args=None):
    rclpy.init(args=args)
    node = CameraBaseTransformNode()
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
