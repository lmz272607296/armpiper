#!/usr/bin/env python3

import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from piper_msgs.srv import Enable
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class ActionMoveNode(Node):
    def __init__(self):
        super().__init__('action1_4_move')

        self.declare_parameter('arm_action_topic', '/arm_action_command')
        self.declare_parameter('joint_feedback_topic', '/joint_states_single')
        self.declare_parameter('end_pose_topic', '/end_pose_stamped')
        self.declare_parameter('joint_command_topic', '/joint_states')
        self.declare_parameter('enable_service', '/enable_srv')
        self.declare_parameter('status_topic', '/task_status')
        self.declare_parameter('default_distance', 0.02)
        self.declare_parameter('max_distance', 0.24)
        self.declare_parameter('max_joint_step_rad', 0.03)
        self.declare_parameter('max_command_joint_delta_rad', 0.2)
        self.declare_parameter('ik_iterations', 80)
        self.declare_parameter('ik_damping', 0.06)
        self.declare_parameter('position_tolerance', 0.0015)
        self.declare_parameter('max_final_error', 0.02)
        self.declare_parameter('fk_feedback_max_error', 0.08)
        self.declare_parameter('orientation_weight', 0.35)
        self.declare_parameter('cartesian_path_points', 18)
        self.declare_parameter('horizontal_z_bias_m', 0.0)
        self.declare_parameter('enable_wrist_level_compensation', True)
        self.declare_parameter('wrist_level_axis_z', 0.0)
        self.declare_parameter('wrist_level_gain', 1.0)
        self.declare_parameter('max_wrist_level_correction_rad', 0.2)
        self.declare_parameter('home_joints', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('home_after_enable', True)
        self.declare_parameter('home_after_enable_delay_sec', 0.8)
        self.declare_parameter('motion_duration_sec', 1.2)
        self.declare_parameter('home_duration_sec', 2.5)
        self.declare_parameter('command_rate_hz', 30.0)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('exit_after_action', False)

        arm_action_topic = self.get_parameter('arm_action_topic').value
        joint_feedback_topic = self.get_parameter('joint_feedback_topic').value
        end_pose_topic = self.get_parameter('end_pose_topic').value
        joint_command_topic = self.get_parameter('joint_command_topic').value
        enable_service = self.get_parameter('enable_service').value
        status_topic = self.get_parameter('status_topic').value

        self.default_distance = float(self.get_parameter('default_distance').value)
        self.max_distance = float(self.get_parameter('max_distance').value)
        self.max_joint_step_rad = float(self.get_parameter('max_joint_step_rad').value)
        self.max_command_joint_delta_rad = float(self.get_parameter('max_command_joint_delta_rad').value)
        self.ik_iterations = int(self.get_parameter('ik_iterations').value)
        self.ik_damping = float(self.get_parameter('ik_damping').value)
        self.position_tolerance = float(self.get_parameter('position_tolerance').value)
        self.max_final_error = float(self.get_parameter('max_final_error').value)
        self.fk_feedback_max_error = float(self.get_parameter('fk_feedback_max_error').value)
        self.orientation_weight = float(self.get_parameter('orientation_weight').value)
        self.cartesian_path_points = max(2, int(self.get_parameter('cartesian_path_points').value))
        self.horizontal_z_bias_m = float(self.get_parameter('horizontal_z_bias_m').value)
        self.enable_wrist_level_compensation = bool(self.get_parameter('enable_wrist_level_compensation').value)
        self.wrist_level_axis_z = float(self.get_parameter('wrist_level_axis_z').value)
        self.wrist_level_gain = float(self.get_parameter('wrist_level_gain').value)
        self.max_wrist_level_correction_rad = float(self.get_parameter('max_wrist_level_correction_rad').value)
        self.home_joints = np.array([float(v) for v in self.get_parameter('home_joints').value], dtype=float)
        self.home_after_enable = bool(self.get_parameter('home_after_enable').value)
        self.home_after_enable_delay_sec = float(self.get_parameter('home_after_enable_delay_sec').value)
        self.motion_duration_sec = float(self.get_parameter('motion_duration_sec').value)
        self.home_duration_sec = float(self.get_parameter('home_duration_sec').value)
        self.command_rate_hz = float(self.get_parameter('command_rate_hz').value)
        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.exit_after_action = bool(self.get_parameter('exit_after_action').value)

        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_lower = np.array([-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944], dtype=float)
        self.joint_upper = np.array([2.618, 3.14, 0.0, 1.745, 1.22, 2.0944], dtype=float)
        self.current_joints = None
        self.current_joint_time = 0.0
        self.current_end_pos = None
        self.current_end_rot = None
        self.current_end_pose_time = 0.0
        self.motion_lock = threading.Lock()
        self.motion_active = False
        self.action_thread_lock = threading.Lock()
        self.action_thread = None
        self.exit_requested_after_action = False

        self.action_sub = self.create_subscription(
            String, arm_action_topic, self.action_callback, 10)
        self.joint_feedback_sub = self.create_subscription(
            JointState, joint_feedback_topic, self.joint_feedback_callback, 10)
        self.end_pose_sub = self.create_subscription(
            PoseStamped, end_pose_topic, self.end_pose_callback, 10)

        self.joint_cmd_pub = self.create_publisher(JointState, joint_command_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.enable_client = self.create_client(Enable, enable_service)

        self.get_logger().info(
            f'Piper IK 平移节点已启动: 订阅 {arm_action_topic}, {joint_feedback_topic}, {end_pose_topic}，'
            f'发布关节命令到 {joint_command_topic}，dry_run={self.dry_run}, '
            f'exit_after_action={self.exit_after_action}'
        )

    def joint_feedback_callback(self, msg):
        positions = {name: pos for name, pos in zip(msg.name, msg.position)}
        if not all(name in positions for name in self.joint_names):
            return

        self.current_joints = np.array([positions[name] for name in self.joint_names], dtype=float)
        self.current_joint_time = time.time()

    def end_pose_callback(self, msg):
        self.current_end_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)
        self.current_end_rot = self.quaternion_to_matrix(msg.pose.orientation)
        self.current_end_pose_time = time.time()

    def action_callback(self, msg):
        handled = False
        try:
            action = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.publish_status(f'忽略非法动作 JSON: {exc}')
            return

        action_type = action.get('type')
        if action_type == 'move':
            handled = self.start_move_thread(action)
        elif action_type == 'enable':
            self.handle_enable(True)
        elif action_type == 'disable':
            self.handle_enable(False)
        elif action_type == 'reset':
            self.publish_home_pose()
            handled = True
        elif action_type == 'release':
            self.publish_status('收到放下指令：当前节点只控制 Piper 大臂。')
            handled = True
        else:
            self.publish_status(f'未知动作类型: {action_type}')

        if handled and self.exit_after_action:
            self.exit_requested_after_action = True
            if action_type != 'move':
                self.publish_status('本次动作执行结束，action1_4_move 即将退出。')
                threading.Thread(target=self.shutdown_soon, daemon=True).start()

    def start_move_thread(self, action):
        with self.action_thread_lock:
            if self.action_thread is not None and self.action_thread.is_alive():
                self.publish_status('已有移动动作正在执行，忽略新的移动命令。')
                return False

            self.action_thread = threading.Thread(
                target=self.run_move_action,
                args=(dict(action),),
                daemon=True,
            )
            self.action_thread.start()
            return True

    def run_move_action(self, action):
        try:
            self.handle_move(action)
        finally:
            if self.exit_after_action or self.exit_requested_after_action:
                self.publish_status('本次动作执行结束，action1_4_move 即将退出。')
                self.shutdown_soon()

    def shutdown_soon(self):
        time.sleep(0.1)
        if rclpy.ok():
            rclpy.shutdown()

    def handle_move(self, action):
        if not self.feedback_is_ready():
            return

        direction = action.get('direction')
        distance = float(action.get('distance', self.default_distance))
        distance = max(-self.max_distance, min(self.max_distance, distance))
        offset = self.direction_to_offset(direction, distance)
        if offset is None:
            self.publish_status(f'无法移动：未知方向 {direction}')
            return

        start_q = np.clip(self.current_joints.copy(), self.joint_lower, self.joint_upper)
        fk_pos, _, _ = self.forward_kinematics(start_q)
        fk_feedback_error = float(np.linalg.norm(fk_pos - self.current_end_pos))
        if fk_feedback_error > self.fk_feedback_max_error:
            self.publish_status(
                f'FK 与 Piper 末端反馈偏差 {fk_feedback_error:.4f}m，超过阈值 '
                f'{self.fk_feedback_max_error:.4f}m，已拒绝移动。'
            )
            return

        start_pos = self.current_end_pos.copy()
        target_rot = self.current_end_rot.copy()
        offset = np.array(offset, dtype=float)
        start_z = float(start_pos[2])
        path_points = max(2, self.cartesian_path_points)
        joint_path = []
        q_seed = start_q.copy()
        max_final_error = 0.0
        max_joint_delta = 0.0
        total_wrist_level_correction = 0.0
        wrist_level_error = 0.0

        for point_index in range(1, path_points + 1):
            ratio = point_index / path_points
            target_pos = start_pos + offset * ratio
            # Optional smooth feed-forward height bias: zero at both ends, max at mid path.
            if self.horizontal_z_bias_m != 0.0:
                target_pos[2] = start_z + math.sin(math.pi * ratio) * self.horizontal_z_bias_m
            else:
                target_pos[2] = start_z

            target_q, final_error = self.solve_ik(q_seed, target_pos, target_rot)
            if final_error > self.max_final_error:
                self.publish_status(
                    f'第 {point_index}/{path_points} 个笛卡尔路径点 IK 误差过大 '
                    f'{final_error:.4f}m，已拒绝发送。'
                )
                return

            wrist_level_correction = 0.0
            wrist_level_error = self.get_wrist_level_error(target_q)
            if self.enable_wrist_level_compensation:
                target_q, wrist_level_correction, wrist_level_error = self.compensate_joint5_wrist_level(target_q)
                compensated_pos, _, _ = self.forward_kinematics(target_q)
                final_error = float(np.linalg.norm(target_pos - compensated_pos))
                if final_error > self.max_final_error:
                    self.publish_status(
                        f'第 {point_index}/{path_points} 个笛卡尔路径点 joint5 腕部水平补偿后 IK 误差 '
                        f'{final_error:.4f}m 超过阈值，已拒绝发送。'
                    )
                    return

            target_q = np.clip(target_q, self.joint_lower, self.joint_upper)
            if not np.all(np.isfinite(target_q)):
                self.publish_status('IK 结果包含非法数值，已拒绝发送。')
                return

            step_delta = target_q - q_seed
            max_joint_delta = max(max_joint_delta, float(np.max(np.abs(step_delta))))
            if np.max(np.abs(step_delta)) > self.max_command_joint_delta_rad:
                self.publish_status(
                    f'第 {point_index}/{path_points} 个笛卡尔路径点关节跳变 '
                    f'{np.max(np.abs(step_delta)):.3f}rad 超过阈值 '
                    f'{self.max_command_joint_delta_rad:.3f}rad，已拒绝发送。'
                )
                return

            max_final_error = max(max_final_error, final_error)
            total_wrist_level_correction += float(wrist_level_correction)
            joint_path.append(target_q.copy())
            q_seed = target_q.copy()

        if not self.publish_joint_path_command(joint_path, self.motion_duration_sec):
            return

        final_target_pos = start_pos + offset
        self.publish_status(
            f"已发送 Piper 连续笛卡尔 IK {self.direction_text(direction)}平移 {abs(distance):.3f} 米："
            f"目标 XYZ=({final_target_pos[0]:.3f}, {final_target_pos[1]:.3f}, {final_target_pos[2]:.3f})，"
            f"路径点={path_points}，最大IK误差={max_final_error:.4f}m，FK/反馈误差={fk_feedback_error:.4f}m，"
            f"目标关节={np.round(joint_path[-1], 3).tolist() if joint_path else []}，"
            f"最大路径关节步长={max_joint_delta:.3f}rad，"
            f"累计joint5水平补偿={total_wrist_level_correction:.3f}rad，腕轴Z误差={wrist_level_error:.3f}，"
            f"Z前馈={self.horizontal_z_bias_m:.4f}m"
        )

    def feedback_is_ready(self):
        if self.current_joints is None:
            self.publish_status('无法移动：尚未收到 /joint_states_single 关节反馈。')
            return False
        if time.time() - self.current_joint_time > 1.0:
            self.publish_status('无法移动：Piper 关节反馈超过 1 秒未更新。')
            return False
        if self.current_end_pos is None or self.current_end_rot is None:
            self.publish_status('无法移动：尚未收到 /end_pose_stamped 末端位姿反馈。')
            return False
        if time.time() - self.current_end_pose_time > 1.0:
            self.publish_status('无法移动：Piper 末端位姿反馈超过 1 秒未更新。')
            return False
        return True

    def handle_enable(self, enable):
        if not self.enable_client.wait_for_service(timeout_sec=1.0):
            self.publish_status('无法切换使能：/enable_srv 服务不可用，请确认 piper_single_ctrl 已启动。')
            return

        request = Enable.Request()
        request.enable_request = bool(enable)
        future = self.enable_client.call_async(request)
        future.add_done_callback(lambda done: self.enable_done_callback(done, enable))
        self.publish_status('正在使能 Piper...' if enable else '正在关闭 Piper 扭矩...')

    def enable_done_callback(self, future, enable):
        try:
            response = future.result()
        except Exception as exc:
            self.publish_status(f'使能服务调用失败: {exc}')
            return

        if not response.enable_response:
            self.publish_status('Piper 使能失败。' if enable else 'Piper 失能失败。')
            return

        self.publish_status('Piper 使能成功。' if enable else 'Piper 失能成功。')
        if enable and self.home_after_enable:
            if self.home_after_enable_delay_sec > 0.0:
                time.sleep(self.home_after_enable_delay_sec)
            self.publish_home_pose()

    def publish_home_pose(self):
        if self.home_joints.shape[0] != 6:
            self.publish_status(f'无法复位：home_joints 需要 6 个关节值，当前为 {self.home_joints.shape[0]} 个。')
            return
        home_q = np.clip(self.home_joints, self.joint_lower, self.joint_upper)
        if self.publish_smoothed_joint_command(
                home_q, self.home_duration_sec, require_fresh_feedback=False):
            self.publish_status('已平滑发送 Piper 初始姿态 home_joints=' + np.round(home_q, 3).tolist().__repr__())

    def publish_joint_command(self, joints):
        if self.dry_run:
            self.publish_status('dry_run=True，未实际发布关节命令：' + np.round(joints, 4).tolist().__repr__())
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'piper_single'
        msg.name = list(self.joint_names)
        msg.position = [float(value) for value in joints]
        msg.velocity = [0.0] * 6
        msg.effort = [0.0] * 6
        self.joint_cmd_pub.publish(msg)

    def publish_smoothed_joint_command(self, target_joints, duration_sec, require_fresh_feedback=True):
        with self.motion_lock:
            if self.motion_active:
                self.publish_status('已有平滑运动正在执行，忽略新的运动命令。')
                return False
            self.motion_active = True

        try:
            if self.current_joints is None:
                self.publish_status('无法发送平滑关节命令：尚未收到当前关节反馈。')
                return False
            if require_fresh_feedback and time.time() - self.current_joint_time > 1.0:
                self.publish_status('无法发送平滑关节命令：Piper 关节反馈超过 1 秒未更新。')
                return False

            start_joints = np.clip(self.current_joints.copy(), self.joint_lower, self.joint_upper)
            target_joints = np.clip(np.asarray(target_joints, dtype=float), self.joint_lower, self.joint_upper)
            duration_sec = max(0.1, float(duration_sec))
            rate_hz = max(5.0, float(self.command_rate_hz))
            steps = max(2, int(duration_sec * rate_hz))
            sleep_sec = 1.0 / rate_hz

            for index in range(1, steps + 1):
                ratio = index / steps
                smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
                command = start_joints + (target_joints - start_joints) * smooth_ratio
                self.publish_joint_command(command)
                time.sleep(sleep_sec)

            return True
        finally:
            with self.motion_lock:
                self.motion_active = False

    def publish_joint_path_command(self, joint_path, duration_sec, require_fresh_feedback=True):
        if len(joint_path) == 0:
            self.publish_status('无法发送关节路径：路径为空。')
            return False

        with self.motion_lock:
            if self.motion_active:
                self.publish_status('已有平滑运动正在执行，忽略新的运动命令。')
                return False
            self.motion_active = True

        try:
            if self.current_joints is None:
                self.publish_status('无法发送关节路径：尚未收到当前关节反馈。')
                return False
            if require_fresh_feedback and time.time() - self.current_joint_time > 1.0:
                self.publish_status('无法发送关节路径：Piper 关节反馈超过 1 秒未更新。')
                return False

            start_joints = np.clip(self.current_joints.copy(), self.joint_lower, self.joint_upper)
            path = np.vstack([start_joints] + [
                np.clip(np.asarray(point, dtype=float), self.joint_lower, self.joint_upper)
                for point in joint_path
            ])
            duration_sec = max(0.1, float(duration_sec))
            rate_hz = max(5.0, float(self.command_rate_hz))
            steps = max(2, int(duration_sec * rate_hz))
            sleep_sec = 1.0 / rate_hz
            segment_count = path.shape[0] - 1

            for index in range(1, steps + 1):
                ratio = index / steps
                smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
                path_position = smooth_ratio * segment_count
                segment_index = min(int(path_position), segment_count - 1)
                local_ratio = path_position - segment_index
                command = path[segment_index] + (path[segment_index + 1] - path[segment_index]) * local_ratio
                self.publish_joint_command(command)
                time.sleep(sleep_sec)

            return True
        finally:
            with self.motion_lock:
                self.motion_active = False

    def solve_ik(self, initial_q, target_pos, target_rot):
        q = initial_q.copy()
        damping_matrix = (self.ik_damping ** 2) * np.eye(6)

        for _ in range(self.ik_iterations):
            pos, rot, joint_frames = self.forward_kinematics(q)
            pos_error = target_pos - pos
            if np.linalg.norm(pos_error) <= self.position_tolerance:
                break

            rot_error = self.rotation_error(rot, target_rot) * self.orientation_weight
            error = np.concatenate([pos_error, rot_error])

            jacobian = self.compute_geometric_jacobian(pos, joint_frames)
            weighted_jacobian = jacobian.copy()
            weighted_jacobian[3:6, :] *= self.orientation_weight
            step = weighted_jacobian.T @ np.linalg.solve(
                weighted_jacobian @ weighted_jacobian.T + damping_matrix,
                error,
            )
            q = np.clip(q + np.clip(step, -self.max_joint_step_rad, self.max_joint_step_rad), self.joint_lower, self.joint_upper)

        final_pos, _, _ = self.forward_kinematics(q)
        return q, float(np.linalg.norm(target_pos - final_pos))

    def forward_kinematics(self, q):
        transform = np.eye(4)
        joint_frames = []
        for index, origin in enumerate(self.joint_origins()):
            transform = transform @ origin
            joint_origin = transform[:3, 3].copy()
            joint_axis = transform[:3, :3] @ np.array([0.0, 0.0, 1.0])
            joint_frames.append((joint_origin, joint_axis))
            transform = transform @ self.axis_angle_transform(np.array([0.0, 0.0, 1.0]), q[index])
        return transform[:3, 3], transform[:3, :3], joint_frames

    def compute_geometric_jacobian(self, ee_pos, joint_frames):
        jacobian = np.zeros((6, 6))
        for index, (joint_origin, joint_axis) in enumerate(joint_frames):
            axis = joint_axis / max(np.linalg.norm(joint_axis), 1e-9)
            jacobian[:3, index] = np.cross(axis, ee_pos - joint_origin)
            jacobian[3:, index] = axis
        return jacobian

    def compensate_joint5_wrist_level(self, q):
        compensated_q = q.copy()
        start_joint5 = compensated_q[4]
        max_correction = max(0.0, self.max_wrist_level_correction_rad)

        for _ in range(8):
            error = self.get_wrist_level_error(compensated_q)
            if abs(error) < 1e-3:
                break

            epsilon = 1e-4
            probe_q = compensated_q.copy()
            probe_q[4] = np.clip(probe_q[4] + epsilon, self.joint_lower[4], self.joint_upper[4])
            derivative = (self.get_wrist_level_error(probe_q) - error) / epsilon
            if abs(derivative) < 1e-6:
                break

            step = -self.wrist_level_gain * error / derivative
            step = float(np.clip(step, -self.max_joint_step_rad, self.max_joint_step_rad))
            next_joint5 = compensated_q[4] + step
            next_joint5 = float(np.clip(next_joint5, start_joint5 - max_correction, start_joint5 + max_correction))
            next_joint5 = float(np.clip(next_joint5, self.joint_lower[4], self.joint_upper[4]))
            if abs(next_joint5 - compensated_q[4]) < 1e-6:
                break
            compensated_q[4] = next_joint5

        final_error = self.get_wrist_level_error(compensated_q)
        return compensated_q, float(compensated_q[4] - start_joint5), final_error

    def get_wrist_level_error(self, q):
        _, _, joint_frames = self.forward_kinematics(q)
        joint6_axis = joint_frames[5][1]
        joint6_axis = joint6_axis / max(np.linalg.norm(joint6_axis), 1e-9)
        return float(joint6_axis[2] - self.wrist_level_axis_z)

    def rotation_error(self, current_rot, target_rot):
        error_rot = target_rot @ current_rot.T
        cos_angle = np.clip((np.trace(error_rot) - 1.0) / 2.0, -1.0, 1.0)
        angle = math.acos(cos_angle)
        if angle < 1e-9:
            return np.zeros(3)
        axis = np.array([
            error_rot[2, 1] - error_rot[1, 2],
            error_rot[0, 2] - error_rot[2, 0],
            error_rot[1, 0] - error_rot[0, 1],
        ]) / (2.0 * math.sin(angle))
        return axis * angle

    def joint_origins(self):
        return [
            self.transform_from_xyz_rpy([0.0, 0.0, 0.123], [0.0, 0.0, 0.0]),
            self.transform_from_xyz_rpy([0.0, 0.0, 0.0], [1.5708, -0.1359, -3.1416]),
            self.transform_from_xyz_rpy([0.28503, 0.0, 0.0], [0.0, 0.0, -1.7939]),
            self.transform_from_xyz_rpy([-0.021984, -0.25075, 0.0], [1.5708, 0.0, 0.0]),
            self.transform_from_xyz_rpy([0.0, 0.0, 0.0], [-1.5708, 0.0, 0.0]),
            self.transform_from_xyz_rpy([0.000088259, -0.091, 0.0], [1.5708, 0.0, 0.0]),
        ]

    def transform_from_xyz_rpy(self, xyz, rpy):
        transform = np.eye(4)
        transform[:3, :3] = self.rpy_to_matrix(rpy)
        transform[:3, 3] = np.array(xyz, dtype=float)
        return transform

    def rpy_to_matrix(self, rpy):
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return rz @ ry @ rx

    def axis_angle_transform(self, axis, angle):
        transform = np.eye(4)
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        x, y, z = axis
        c, s = math.cos(angle), math.sin(angle)
        one_c = 1.0 - c
        transform[:3, :3] = np.array([
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ])
        return transform

    def quaternion_to_matrix(self, quat_msg):
        x = quat_msg.x
        y = quat_msg.y
        z = quat_msg.z
        w = quat_msg.w
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-9:
            return np.eye(3)
        x /= norm
        y /= norm
        z /= norm
        w /= norm
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=float)

    def direction_to_offset(self, direction, distance):
        return {
            'front': (distance, 0.0, 0.0),
            'back': (-distance, 0.0, 0.0),
            'left': (0.0, distance, 0.0),
            'right': (0.0, -distance, 0.0),
        }.get(direction)

    def direction_text(self, direction):
        return {
            'front': '向前',
            'back': '向后',
            'left': '向左',
            'right': '向右',
        }.get(direction, str(direction))

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(status)


def main(args=None):
    rclpy.init(args=args)
    node = ActionMoveNode()
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
