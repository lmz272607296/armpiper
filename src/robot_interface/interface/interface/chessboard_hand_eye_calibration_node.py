#!/usr/bin/env python3

import json
import math
import os
import time

if os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland':
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

import cv2
import cv_bridge
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


HAND_EYE_METHODS = {
    'tsai': cv2.CALIB_HAND_EYE_TSAI,
    'park': cv2.CALIB_HAND_EYE_PARK,
    'horaud': cv2.CALIB_HAND_EYE_HORAUD,
    'andreff': cv2.CALIB_HAND_EYE_ANDREFF,
    'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


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


def pose_to_matrix(pose_msg):
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_matrix(pose_msg.orientation)
    transform[:3, 3] = np.array([
        pose_msg.position.x,
        pose_msg.position.y,
        pose_msg.position.z,
    ], dtype=float)
    return transform


def invert_transform(transform):
    inverse = np.eye(4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def rotation_angle_between(rotation_a, rotation_b):
    delta = rotation_a.T @ rotation_b
    cos_angle = (np.trace(delta) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(cos_angle, -1.0, 1.0))))


class ChessboardHandEyeCalibrationNode(Node):
    def __init__(self):
        super().__init__('chessboard_hand_eye_calibration_node')

        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('end_pose_topic', '/end_pose')
        self.declare_parameter('end_pose_stamped_topic', '/end_pose_stamped')
        self.declare_parameter('prefer_stamped_end_pose', True)
        self.declare_parameter('command_topic', '/hand_eye_calibration_command')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('flange_frame_id', 'link6')
        self.declare_parameter('camera_frame_id', 'camera_color_optical_frame')
        self.declare_parameter('chessboard_inner_corners_x', 10)
        self.declare_parameter('chessboard_inner_corners_y', 8)
        self.declare_parameter('try_swapped_chessboard', True)
        self.declare_parameter('square_size_m', 0.02)
        self.declare_parameter('min_samples', 10)
        self.declare_parameter('max_end_pose_age_sec', 1.0)
        self.declare_parameter('calibration_method', 'tsai')
        self.declare_parameter('output_matrix_file', 'hand_eye_calibration.txt')
        self.declare_parameter('samples_file', 'hand_eye_calibration_samples.json')
        self.declare_parameter('display', True)
        self.declare_parameter('display_rate_hz', 10.0)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_topic', '/hand_eye_calibration_debug_image')
        self.declare_parameter('publish_debug_image_rate_hz', 2.0)
        self.declare_parameter('processing_rate_hz', 2.0)
        self.declare_parameter('resize_width_px', 960)
        self.declare_parameter('use_find_chessboard_sb', False)
        self.declare_parameter('log_interval_sec', 2.0)

        color_topic = self.get_parameter('color_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        end_pose_topic = self.get_parameter('end_pose_topic').value
        end_pose_stamped_topic = self.get_parameter('end_pose_stamped_topic').value
        command_topic = self.get_parameter('command_topic').value

        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.flange_frame_id = self.get_parameter('flange_frame_id').value
        self.camera_frame_id = self.get_parameter('camera_frame_id').value
        self.chessboard_size = (
            int(self.get_parameter('chessboard_inner_corners_x').value),
            int(self.get_parameter('chessboard_inner_corners_y').value),
        )
        self.square_size_m = float(self.get_parameter('square_size_m').value)
        self.try_swapped_chessboard = bool(self.get_parameter('try_swapped_chessboard').value)
        self.min_samples = int(self.get_parameter('min_samples').value)
        self.max_end_pose_age_sec = float(self.get_parameter('max_end_pose_age_sec').value)
        self.calibration_method_name = str(self.get_parameter('calibration_method').value).lower()
        if self.calibration_method_name not in HAND_EYE_METHODS:
            raise ValueError(f'calibration_method 必须是 {sorted(HAND_EYE_METHODS)} 之一')
        self.output_matrix_file = os.path.expanduser(str(self.get_parameter('output_matrix_file').value))
        self.samples_file = os.path.expanduser(str(self.get_parameter('samples_file').value))
        self.display = bool(self.get_parameter('display').value)
        self.display_rate_hz = float(self.get_parameter('display_rate_hz').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.publish_debug_image_rate_hz = float(self.get_parameter('publish_debug_image_rate_hz').value)
        self.processing_rate_hz = float(self.get_parameter('processing_rate_hz').value)
        self.resize_width_px = int(self.get_parameter('resize_width_px').value)
        self.use_find_chessboard_sb = bool(self.get_parameter('use_find_chessboard_sb').value)
        self.prefer_stamped_end_pose = bool(self.get_parameter('prefer_stamped_end_pose').value)
        self.log_interval_sec = float(self.get_parameter('log_interval_sec').value)

        if self.processing_rate_hz <= 0.0:
            raise ValueError('processing_rate_hz 必须大于 0')
        if self.display_rate_hz <= 0.0:
            raise ValueError('display_rate_hz 必须大于 0')
        if self.publish_debug_image_rate_hz <= 0.0:
            raise ValueError('publish_debug_image_rate_hz 必须大于 0')

        if self.display and os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland':
            self.get_logger().warn('检测到 Wayland，仍会尝试打开 OpenCV 弹窗；若窗口未显示，请改用 X11 或查看调试图话题。')

        self.bridge = cv_bridge.CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_color_msg = None
        self.latest_color_image = None
        self.latest_color_image_stamp = None
        self.latest_color_stamp = None
        self.latest_debug_image = None
        self.latest_target_to_camera = None
        self.latest_reprojection_error = None
        self.latest_board_found = False
        self.latest_corners = None
        self.latest_corners_image_shape = None
        self.latest_chessboard_size = self.chessboard_size
        self.latest_base_to_flange = None
        self.latest_end_pose_time = 0.0
        self.latest_end_pose_source = None
        self.samples = []
        self.last_log_time = 0.0
        self.last_board_log_time = 0.0
        self.last_stream_stale_log_time = 0.0
        self.processing = False
        self.latest_color_receive_time = 0.0
        self.received_image_count = 0
        self.last_display_time = 0.0
        self.last_debug_publish_time = 0.0
        self.window_name = 'Chessboard Hand-Eye Calibration'
        self.display_window_created = False
        self.object_points_by_size = self.create_object_points_by_size()

        qos_profile = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.color_sub = self.create_subscription(Image, color_topic, self.image_callback, qos_profile)
        self.camera_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, qos_profile)
        self.pose_sub = self.create_subscription(Pose, end_pose_topic, self.pose_callback, 10)
        self.pose_stamped_sub = self.create_subscription(PoseStamped, end_pose_stamped_topic, self.pose_stamped_callback, 10)
        self.command_sub = self.create_subscription(String, command_topic, self.command_callback, 10)
        self.debug_image_pub = None
        if self.publish_debug_image:
            self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        self.processing_timer = self.create_timer(1.0 / self.processing_rate_hz, self.processing_timer_callback)

        self.get_logger().info(
            f'棋盘格手眼标定节点已启动。内部角点={self.chessboard_size}，'
            f'方格={self.square_size_m * 1000.0:.1f}mm，命令话题={command_topic}'
        )
        self.get_logger().info(
            f'读取 Piper 末端位姿: {end_pose_stamped_topic} / {end_pose_topic}；'
            f'输出 {self.flange_frame_id}->{self.camera_frame_id}: {self.output_matrix_file}'
        )
        if self.display:
            self.get_logger().info('将弹出实时彩色图窗口。未识别到棋盘格时也会持续刷新，方便调整标定板位置。')
        if self.publish_debug_image:
            self.get_logger().info(f'调试图发布到: {self.debug_image_topic}')

    def create_object_points_by_size(self):
        sizes = [self.chessboard_size]
        swapped = (self.chessboard_size[1], self.chessboard_size[0])
        if self.try_swapped_chessboard and swapped != self.chessboard_size:
            sizes.append(swapped)

        object_points_by_size = {}
        for size in sizes:
            points = np.zeros((size[0] * size[1], 3), np.float32)
            points[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
            points *= self.square_size_m
            object_points_by_size[size] = points
        return object_points_by_size

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=float).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=float).reshape(-1, 1)
        if self.dist_coeffs.size == 0:
            self.dist_coeffs = np.zeros((5, 1), dtype=float)
        self.get_logger().info(
            f'已收到相机内参 fx={self.camera_matrix[0, 0]:.3f}, '
            f'fy={self.camera_matrix[1, 1]:.3f}, cx={self.camera_matrix[0, 2]:.3f}, '
            f'cy={self.camera_matrix[1, 2]:.3f}, distortion={self.dist_coeffs.reshape(-1).tolist()}'
        )
        self.destroy_subscription(self.camera_info_sub)

    def pose_callback(self, msg):
        if self.prefer_stamped_end_pose and self.latest_end_pose_source == 'stamped':
            if time.time() - self.latest_end_pose_time <= self.max_end_pose_age_sec:
                return
        self.latest_base_to_flange = pose_to_matrix(msg)
        self.latest_end_pose_time = time.time()
        self.latest_end_pose_source = 'pose'

    def pose_stamped_callback(self, msg):
        self.latest_base_to_flange = pose_to_matrix(msg.pose)
        self.latest_end_pose_time = time.time()
        self.latest_end_pose_source = 'stamped'
        if msg.header.frame_id and msg.header.frame_id not in (self.base_frame_id, ''):
            self.warn_throttled(
                f'收到 {self.get_parameter("end_pose_stamped_topic").value} frame_id={msg.header.frame_id}，'
                f'本节点按 {self.base_frame_id}->{self.flange_frame_id} 位姿解释。'
            )

    def image_callback(self, msg):
        self.latest_color_msg = msg
        self.latest_color_image = None
        self.latest_color_image_stamp = None
        self.latest_color_stamp = msg.header.stamp
        self.latest_color_receive_time = time.time()
        self.received_image_count += 1

    def processing_timer_callback(self):
        if self.processing:
            return
        color_image = self.get_latest_color_image()
        if color_image is None:
            return

        self.processing = True
        try:
            processed_image = self.resize_for_processing(color_image)
            self.latest_target_to_camera, self.latest_reprojection_error = self.detect_target_pose(processed_image)
            self.latest_board_found = self.latest_target_to_camera is not None

            if not self.latest_board_found:
                self.log_board_not_found()

            self.latest_debug_image = self.render_debug_image(color_image)
        finally:
            self.processing = False

    def display_once(self):
        if not self.display:
            return

        now = time.time()
        if now - self.last_display_time < 1.0 / self.display_rate_hz:
            return
        self.last_display_time = now

        color_image = self.get_latest_color_image()
        if color_image is None:
            return

        display = self.render_debug_image(color_image)
        self.show_display(display)

        if self.publish_debug_image and self.debug_image_pub is not None:
            self.publish_debug_image_once(display)

    def resize_for_processing(self, color_image):
        if self.resize_width_px <= 0 or color_image.shape[1] <= self.resize_width_px:
            return color_image.copy()
        scale = float(self.resize_width_px) / float(color_image.shape[1])
        target_size = (self.resize_width_px, int(round(color_image.shape[0] * scale)))
        return cv2.resize(color_image, target_size, interpolation=cv2.INTER_AREA)

    def get_latest_color_image(self):
        if self.latest_color_msg is None:
            return None

        stamp_key = self.stamp_to_key(self.latest_color_msg.header.stamp)
        if self.latest_color_image is not None and self.latest_color_image_stamp == stamp_key:
            self.log_if_stream_stale()
            return self.latest_color_image

        try:
            color_image = self.bridge.imgmsg_to_cv2(self.latest_color_msg, 'bgr8')
        except cv_bridge.CvBridgeError as exc:
            self.get_logger().error(f'彩色图转换失败: {exc}')
            return None

        self.latest_color_image = color_image
        self.latest_color_image_stamp = stamp_key
        self.log_if_stream_stale()
        return color_image

    def stamp_to_key(self, stamp):
        return (int(stamp.sec), int(stamp.nanosec))

    def log_if_stream_stale(self):
        if self.latest_color_receive_time <= 0.0:
            return

        now = time.time()
        stale_threshold_sec = max(1.5, 3.0 / self.display_rate_hz)
        if now - self.latest_color_receive_time < stale_threshold_sec:
            return
        if now - self.last_stream_stale_log_time < self.log_interval_sec:
            return

        self.get_logger().warn(
            '彩色图像流长时间未更新，当前窗口显示的是旧帧。'
            '本节点仅依赖 color/camera_info；若使用 RealSense，建议先关闭 enable_sync、align_depth.enable 和 depth，'
            '仅保留彩色流确认画面连续。'
        )
        self.last_stream_stale_log_time = now

    def detect_target_pose(self, color_image):
        if self.camera_matrix is None:
            self.latest_corners = None
            self.latest_corners_image_shape = None
            return None, None

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners = None
        found_size = None
        for chessboard_size in self.object_points_by_size:
            found, candidate_corners = self.find_chessboard_corners(gray, chessboard_size)
            if found:
                corners = candidate_corners
                found_size = chessboard_size
                break

        if found_size is None:
            self.latest_corners = None
            self.latest_corners_image_shape = None
            return None, None

        object_points = self.object_points_by_size[found_size]

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            corners.reshape(-1, 1, 2),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None, None

        rotation, _ = cv2.Rodrigues(rvec)
        target_to_camera = np.eye(4)
        target_to_camera[:3, :3] = rotation
        target_to_camera[:3, 3] = tvec.reshape(3)

        projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        errors = corners.reshape(-1, 2) - projected.reshape(-1, 2)
        reprojection_error = float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))
        self.latest_corners = corners
        self.latest_corners_image_shape = color_image.shape[:2]
        self.latest_chessboard_size = found_size
        return target_to_camera, reprojection_error

    def find_chessboard_corners(self, gray, chessboard_size):
        if self.use_find_chessboard_sb and hasattr(cv2, 'findChessboardCornersSB'):
            found, corners = cv2.findChessboardCornersSB(gray, chessboard_size)
            if found:
                return True, corners

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, chessboard_size, flags)
        if not found:
            return False, None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners

    def render_debug_image(self, color_image):
        display = color_image.copy()
        status_lines = []

        if self.camera_matrix is None:
            status_lines.append(('waiting for camera info', (0, 165, 255)))
            status_lines.append(('camera is connected but CameraInfo has not arrived yet', (0, 165, 255)))
        elif self.latest_board_found and hasattr(self, 'latest_corners'):
            corners = self.latest_corners
            if corners is not None and display.shape[:2] != getattr(self, 'latest_corners_image_shape', display.shape[:2]):
                corners = self.scale_corners_for_display(corners, self.latest_corners_image_shape, display.shape[:2])
            if corners is not None:
                cv2.drawChessboardCorners(display, self.latest_chessboard_size, corners, True)
            status_lines.append((f'board ok | reproj={self.latest_reprojection_error:.3f}px', (0, 255, 0)))
            status_lines.append(('press c to capture this pose', (0, 255, 0)))
        else:
            corners_x, corners_y = self.chessboard_size
            status_lines.append(('board not found', (0, 0, 255)))
            status_lines.append((f'adjust chessboard until {corners_x}x{corners_y} inner corners are highlighted', (0, 0, 255)))

        image_age_sec = None
        if self.latest_color_receive_time > 0.0:
            image_age_sec = max(0.0, time.time() - self.latest_color_receive_time)
        stream_line = f'frames={self.received_image_count}'
        if image_age_sec is not None:
            stream_line += f' | image_age={image_age_sec:.2f}s'
            if image_age_sec > max(1.0, 2.0 / self.processing_rate_hz):
                stream_line += ' | image stream stale'
        status_lines.append((stream_line, (255, 255, 255)))

        pose_state = 'end pose ok'
        pose_color = (255, 255, 255)
        if self.latest_base_to_flange is None:
            pose_state = 'waiting for /end_pose or /end_pose_stamped'
            pose_color = (0, 165, 255)
        elif time.time() - self.latest_end_pose_time > self.max_end_pose_age_sec:
            pose_state = 'end pose stale'
            pose_color = (0, 165, 255)
        status_lines.append((pose_state, pose_color))

        bottom_lines = [
            (f'samples={len(self.samples)}', (255, 255, 255)),
            ('keyboard: c capture | s solve | u undo | q hide window', (255, 255, 255)),
        ]

        top_y = 30
        for text, color in status_lines:
            cv2.putText(display, text, (10, top_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(display, text, (10, top_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            top_y += 28

        bottom_y = display.shape[0] - 38
        for text, color in bottom_lines:
            cv2.putText(display, text, (10, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(display, text, (10, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            bottom_y += 24
        return display

    def scale_corners_for_display(self, corners, source_shape, target_shape):
        if source_shape == target_shape:
            return corners
        source_h, source_w = source_shape
        target_h, target_w = target_shape
        if source_h <= 0 or source_w <= 0:
            return corners
        scale_x = float(target_w) / float(source_w)
        scale_y = float(target_h) / float(source_h)
        scaled = corners.copy()
        scaled[:, 0, 0] *= scale_x
        scaled[:, 0, 1] *= scale_y
        return scaled

    def show_display(self, display):
        try:
            if not self.display_window_created:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.display_window_created = True
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            self.get_logger().error(f'OpenCV 弹窗显示失败: {exc}')
            self.display = False
            self.display_window_created = False
            return

        if key == ord('c'):
            self.capture_sample()
        elif key == ord('s'):
            self.solve_and_save()
        elif key == ord('u'):
            self.delete_last_sample()
        elif key == ord('q'):
            self.display = False
            if self.display_window_created:
                cv2.destroyWindow(self.window_name)
                self.display_window_created = False

    def publish_debug_image_once(self, display):
        now = time.time()
        if now - self.last_debug_publish_time < 1.0 / self.publish_debug_image_rate_hz:
            return
        self.last_debug_publish_time = now

        try:
            image_msg = self.bridge.cv2_to_imgmsg(display, encoding='bgr8')
        except cv_bridge.CvBridgeError as exc:
            self.get_logger().error(f'调试图转换失败: {exc}')
            return
        if self.latest_color_stamp is not None:
            image_msg.header.stamp = self.latest_color_stamp
        image_msg.header.frame_id = self.camera_frame_id
        self.debug_image_pub.publish(image_msg)

    def log_board_not_found(self):
        now = time.time()
        if now - self.last_board_log_time >= self.log_interval_sec:
            self.get_logger().warn(
                f'未识别到棋盘格。当前尝试内部角点 {list(self.chessboard_size)}，'
                f'若长期失败，请确认棋盘格规格或调整参数。'
            )
            self.last_board_log_time = now

    def command_callback(self, msg):
        command = msg.data.strip().lower()
        if command in ('capture', 'c'):
            self.capture_sample()
        elif command in ('solve', 'save', 's'):
            self.solve_and_save()
        elif command in ('delete_last', 'undo', 'u'):
            self.delete_last_sample()
        elif command in ('clear', 'reset'):
            self.samples.clear()
            self.get_logger().warn('已清空当前采样。')
        elif command == 'status':
            self.log_status()
        else:
            self.get_logger().warn("未知命令。支持: capture, solve, delete_last, clear, status")

    def capture_sample(self):
        if self.camera_matrix is None:
            self.get_logger().warn('尚未收到 CameraInfo，不能采样。')
            return
        if self.latest_target_to_camera is None:
            self.get_logger().warn('当前画面未识别到棋盘格，不能采样。')
            return
        if self.latest_base_to_flange is None:
            self.get_logger().warn('尚未收到 Piper 末端位姿，不能采样。')
            return
        if time.time() - self.latest_end_pose_time > self.max_end_pose_age_sec:
            self.get_logger().warn('Piper 末端位姿已超时，请确认 /end_pose 或 /end_pose_stamped 正在更新。')
            return

        sample = {
            'timestamp': time.time(),
            'end_pose_source': self.latest_end_pose_source,
            'base_to_flange': self.latest_base_to_flange.tolist(),
            'target_to_camera': self.latest_target_to_camera.tolist(),
            'reprojection_error_px': float(self.latest_reprojection_error),
            'chessboard_inner_corners': list(self.latest_chessboard_size),
        }
        self.samples.append(sample)
        translation = self.latest_base_to_flange[:3, 3]
        self.get_logger().info(
            f"采样 #{len(self.samples)}: base->flange xyz={np.round(translation, 4).tolist()}, "
            f"target->camera xyz={np.round(self.latest_target_to_camera[:3, 3], 4).tolist()}, "
            f"reproj={self.latest_reprojection_error:.3f}px"
        )

    def delete_last_sample(self):
        if not self.samples:
            self.get_logger().warn('当前没有可删除的采样。')
            return
        self.samples.pop()
        self.get_logger().warn(f'已删除最后一个采样，剩余 {len(self.samples)} 个。')

    def solve_and_save(self):
        if len(self.samples) < self.min_samples:
            self.get_logger().warn(f'采样不足: {len(self.samples)}/{self.min_samples}。建议 12-20 组不同末端姿态。')
            return

        gripper_to_base = [np.array(sample['base_to_flange'], dtype=float) for sample in self.samples]
        target_to_camera = [np.array(sample['target_to_camera'], dtype=float) for sample in self.samples]

        rotations_gripper_to_base = [transform[:3, :3] for transform in gripper_to_base]
        translations_gripper_to_base = [transform[:3, 3].reshape(3, 1) for transform in gripper_to_base]
        rotations_target_to_camera = [transform[:3, :3] for transform in target_to_camera]
        translations_target_to_camera = [transform[:3, 3].reshape(3, 1) for transform in target_to_camera]

        try:
            rotation_camera_to_gripper, translation_camera_to_gripper = cv2.calibrateHandEye(
                rotations_gripper_to_base,
                translations_gripper_to_base,
                rotations_target_to_camera,
                translations_target_to_camera,
                method=HAND_EYE_METHODS[self.calibration_method_name],
            )
        except cv2.error as exc:
            self.get_logger().error(f'cv2.calibrateHandEye 求解失败: {exc}')
            return

        flange_to_camera = np.eye(4)
        flange_to_camera[:3, :3] = rotation_camera_to_gripper
        flange_to_camera[:3, 3] = translation_camera_to_gripper.reshape(3)
        camera_to_flange = invert_transform(flange_to_camera)
        target_to_base_samples = [np.array(sample['base_to_flange'], dtype=float) @ flange_to_camera @ np.array(sample['target_to_camera'], dtype=float) for sample in self.samples]
        diagnostics = self.compute_diagnostics(flange_to_camera, target_to_base_samples)

        os.makedirs(os.path.dirname(self.output_matrix_file) or '.', exist_ok=True)
        np.savetxt(self.output_matrix_file, flange_to_camera, fmt='%.10f')
        self.save_samples_file(flange_to_camera, camera_to_flange, target_to_base_samples, diagnostics)

        self.get_logger().info(f'手眼标定完成，方法={self.calibration_method_name}')
        self.get_logger().info(f'已保存 {self.flange_frame_id}->{self.camera_frame_id}: {self.output_matrix_file}')
        self.get_logger().info('T_flange_camera =\n' + np.array2string(flange_to_camera, precision=6, suppress_small=True))
        self.log_diagnostics(diagnostics)

    def compute_diagnostics(self, flange_to_camera, target_to_base_samples):
        reprojection_errors = np.array([
            float(sample.get('reprojection_error_px', float('nan')))
            for sample in self.samples
        ], dtype=float)
        reprojection_errors = reprojection_errors[np.isfinite(reprojection_errors)]

        translations = np.array([transform[:3, 3] for transform in target_to_base_samples], dtype=float)
        translation_center = translations.mean(axis=0)
        translation_errors = np.linalg.norm(translations - translation_center, axis=1)
        rotations = [transform[:3, :3] for transform in target_to_base_samples]
        reference_rotation = rotations[0]
        rotation_errors = np.array([rotation_angle_between(reference_rotation, rotation) for rotation in rotations])

        worst_translation_index = int(np.argmax(translation_errors)) if translation_errors.size else -1
        worst_rotation_index = int(np.argmax(rotation_errors)) if rotation_errors.size else -1

        gripper_poses = [np.array(sample['base_to_flange'], dtype=float) for sample in self.samples]
        camera_target_poses = [np.array(sample['target_to_camera'], dtype=float) for sample in self.samples]
        motion_diversity = self.compute_motion_diversity(gripper_poses, camera_target_poses)
        hand_eye_residuals = self.compute_hand_eye_residuals(gripper_poses, camera_target_poses, flange_to_camera)

        target_to_base_consistency = {
            'translation_mean_m': translation_center.tolist(),
            'translation_error_per_sample_m': translation_errors.tolist(),
            'translation_rms_m': float(np.sqrt(np.mean(translation_errors * translation_errors))),
            'translation_mean_abs_m': float(np.mean(translation_errors)),
            'translation_max_m': float(np.max(translation_errors)),
            'translation_worst_sample': worst_translation_index + 1,
            'rotation_error_per_sample_deg': rotation_errors.tolist(),
            'rotation_rms_deg': float(np.sqrt(np.mean(rotation_errors * rotation_errors))),
            'rotation_mean_abs_deg': float(np.mean(rotation_errors)),
            'rotation_max_deg': float(np.max(rotation_errors)),
            'rotation_worst_sample': worst_rotation_index + 1,
        }

        return {
            'sample_count': len(self.samples),
            'reprojection_error_px': self.array_stats(reprojection_errors),
            'target_to_base_consistency': target_to_base_consistency,
            'motion_diversity': motion_diversity,
            'hand_eye_equation_residual_ax_xb': hand_eye_residuals,
            # Backward-compatible aliases used by older saved diagnostics readers.
            'translation_mean_m': target_to_base_consistency['translation_mean_m'],
            'translation_rms_m': target_to_base_consistency['translation_rms_m'],
            'translation_max_m': target_to_base_consistency['translation_max_m'],
            'rotation_rms_deg': target_to_base_consistency['rotation_rms_deg'],
            'rotation_max_deg': target_to_base_consistency['rotation_max_deg'],
        }

    def array_stats(self, values):
        values = np.array(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {
                'count': 0,
                'mean': None,
                'rms': None,
                'min': None,
                'max': None,
                'std': None,
            }
        return {
            'count': int(values.size),
            'mean': float(np.mean(values)),
            'rms': float(np.sqrt(np.mean(values * values))),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'std': float(np.std(values)),
        }

    def compute_motion_diversity(self, gripper_poses, camera_target_poses):
        gripper_pair_translations = []
        gripper_pair_rotations = []
        camera_pair_translations = []
        camera_pair_rotations = []

        for i in range(len(gripper_poses)):
            for j in range(i + 1, len(gripper_poses)):
                gripper_delta = invert_transform(gripper_poses[i]) @ gripper_poses[j]
                camera_delta = camera_target_poses[j] @ invert_transform(camera_target_poses[i])
                gripper_pair_translations.append(np.linalg.norm(gripper_delta[:3, 3]))
                gripper_pair_rotations.append(rotation_angle_between(np.eye(3), gripper_delta[:3, :3]))
                camera_pair_translations.append(np.linalg.norm(camera_delta[:3, 3]))
                camera_pair_rotations.append(rotation_angle_between(np.eye(3), camera_delta[:3, :3]))

        gripper_positions = np.array([pose[:3, 3] for pose in gripper_poses], dtype=float)
        target_positions_in_camera = np.array([pose[:3, 3] for pose in camera_target_poses], dtype=float)
        return {
            'pair_count': len(gripper_pair_translations),
            'gripper_pair_translation_m': self.array_stats(gripper_pair_translations),
            'gripper_pair_rotation_deg': self.array_stats(gripper_pair_rotations),
            'target_pair_translation_in_camera_m': self.array_stats(camera_pair_translations),
            'target_pair_rotation_in_camera_deg': self.array_stats(camera_pair_rotations),
            'gripper_xyz_range_m': (np.max(gripper_positions, axis=0) - np.min(gripper_positions, axis=0)).tolist(),
            'target_xyz_range_in_camera_m': (np.max(target_positions_in_camera, axis=0) - np.min(target_positions_in_camera, axis=0)).tolist(),
        }

    def compute_hand_eye_residuals(self, gripper_poses, camera_target_poses, flange_to_camera):
        translation_residuals = []
        rotation_residuals = []
        for i in range(len(gripper_poses)):
            for j in range(i + 1, len(gripper_poses)):
                gripper_motion = invert_transform(gripper_poses[j]) @ gripper_poses[i]
                camera_motion = camera_target_poses[j] @ invert_transform(camera_target_poses[i])
                left = gripper_motion @ flange_to_camera
                right = flange_to_camera @ camera_motion
                residual = invert_transform(left) @ right
                translation_residuals.append(np.linalg.norm(residual[:3, 3]))
                rotation_residuals.append(rotation_angle_between(np.eye(3), residual[:3, :3]))

        return {
            'pair_count': len(translation_residuals),
            'translation_m': self.array_stats(translation_residuals),
            'rotation_deg': self.array_stats(rotation_residuals),
        }

    def log_diagnostics(self, diagnostics):
        reproj = diagnostics['reprojection_error_px']
        consistency = diagnostics['target_to_base_consistency']
        diversity = diagnostics['motion_diversity']
        residual = diagnostics['hand_eye_equation_residual_ax_xb']

        self.get_logger().info('========== 手眼标定效果评估 ==========')
        self.get_logger().info(
            f"样本数={diagnostics['sample_count']}，重投影误差: "
            f"mean={reproj['mean']:.3f}px, rms={reproj['rms']:.3f}px, "
            f"max={reproj['max']:.3f}px, std={reproj['std']:.3f}px"
        )
        self.get_logger().info(
            f"标定板在基座系一致性: 平移 RMS={consistency['translation_rms_m'] * 1000.0:.2f}mm, "
            f"mean={consistency['translation_mean_abs_m'] * 1000.0:.2f}mm, "
            f"max={consistency['translation_max_m'] * 1000.0:.2f}mm(样本#{consistency['translation_worst_sample']}), "
            f"旋转 RMS={consistency['rotation_rms_deg']:.3f}deg, "
            f"max={consistency['rotation_max_deg']:.3f}deg(样本#{consistency['rotation_worst_sample']})"
        )
        self.get_logger().info(
            f"AX=XB 成对残差: 平移 RMS={residual['translation_m']['rms'] * 1000.0:.2f}mm, "
            f"max={residual['translation_m']['max'] * 1000.0:.2f}mm, "
            f"旋转 RMS={residual['rotation_deg']['rms']:.3f}deg, "
            f"max={residual['rotation_deg']['max']:.3f}deg"
        )
        self.get_logger().info(
            f"末端采样运动覆盖: 两两最大平移={diversity['gripper_pair_translation_m']['max'] * 1000.0:.1f}mm, "
            f"两两最大旋转={diversity['gripper_pair_rotation_deg']['max']:.1f}deg, "
            f"XYZ范围(m)={np.round(diversity['gripper_xyz_range_m'], 4).tolist()}"
        )
        self.get_logger().info(
            f"棋盘格相机视角覆盖: 两两最大平移={diversity['target_pair_translation_in_camera_m']['max'] * 1000.0:.1f}mm, "
            f"两两最大旋转={diversity['target_pair_rotation_in_camera_deg']['max']:.1f}deg, "
            f"XYZ范围(m)={np.round(diversity['target_xyz_range_in_camera_m'], 4).tolist()}"
        )
        self.get_logger().info('经验阈值: 重投影 <0.5px 较好；target->base 平移RMS <5mm 较好，>10mm 通常需重采；末端最大旋转建议 >40deg。')
        self.get_logger().info('======================================')

    def save_samples_file(self, flange_to_camera, camera_to_flange, target_to_base_samples, diagnostics):
        os.makedirs(os.path.dirname(self.samples_file) or '.', exist_ok=True)
        data = {
            'description': 'OpenCV eye-in-hand calibration. Matrix file stores T_flange_camera for eye_in_hand_calibration_node.',
            'base_frame_id': self.base_frame_id,
            'flange_frame_id': self.flange_frame_id,
            'camera_frame_id': self.camera_frame_id,
            'chessboard_inner_corners': list(self.chessboard_size),
            'square_size_m': self.square_size_m,
            'method': self.calibration_method_name,
            'matrix_file': self.output_matrix_file,
            'transform_flange_from_camera': flange_to_camera.tolist(),
            'transform_camera_from_flange': camera_to_flange.tolist(),
            'diagnostics': diagnostics,
            'samples': [
                {
                    **sample,
                    'target_to_base': target_to_base_samples[index].tolist(),
                }
                for index, sample in enumerate(self.samples)
            ],
        }
        with open(self.samples_file, 'w', encoding='utf-8') as file_obj:
            json.dump(data, file_obj, indent=2, ensure_ascii=False)

    def log_status(self):
        end_pose_age = None if self.latest_base_to_flange is None else time.time() - self.latest_end_pose_time
        self.get_logger().info(
            f'samples={len(self.samples)}, camera_info={self.camera_matrix is not None}, '
            f'board_found={self.latest_board_found}, reprojection={self.latest_reprojection_error}, '
            f'end_pose_source={self.latest_end_pose_source}, end_pose_age={end_pose_age}'
        )

    def warn_throttled(self, text):
        now = time.time()
        if now - self.last_log_time >= self.log_interval_sec:
            self.get_logger().warn(text)
            self.last_log_time = now

    def destroy_node(self):
        if self.display or self.display_window_created:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ChessboardHandEyeCalibrationNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.01)
            node.display_once()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
