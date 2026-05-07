#!/usr/bin/env python3

import queue
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class PiperKeyboardJointJog(Node):
    def __init__(self):
        super().__init__('piper_keyboard_joint_jog')

        self.declare_parameter('joint_state_topic', '/joint_states_single')
        self.declare_parameter('joint_command_topic', '/arm_controller/joint_trajectory')
        self.declare_parameter('step_rad', 0.03)
        self.declare_parameter('duration_sec', 0.25)
        self.declare_parameter('joint_lower', [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944])
        self.declare_parameter('joint_upper', [2.618, 3.14, 0.0, 1.745, 1.22, 2.0944])

        joint_state_topic = self.get_parameter('joint_state_topic').value
        joint_command_topic = self.get_parameter('joint_command_topic').value
        self.step_rad = float(self.get_parameter('step_rad').value)
        self.duration_sec = float(self.get_parameter('duration_sec').value)
        self.joint_lower = np.array(self.get_parameter('joint_lower').value, dtype=float)
        self.joint_upper = np.array(self.get_parameter('joint_upper').value, dtype=float)
        self.joint_names = [f'joint{i}' for i in range(1, 7)]
        self.current_joints = None
        self.current_joint_time = 0.0
        self.keyboard_queue = queue.Queue()

        self.joint_sub = self.create_subscription(JointState, joint_state_topic, self.joint_callback, 10)
        self.command_pub = self.create_publisher(JointTrajectory, joint_command_topic, 10)

        self.key_bindings = {
            'a': (0, 1.0),
            'd': (0, -1.0),
            'w': (1, 1.0),
            's': (1, -1.0),
            'r': (2, 1.0),
            'f': (2, -1.0),
            't': (4, 1.0),
            'g': (4, -1.0),
            'y': (3, 1.0),
            'h': (3, -1.0),
            'u': (5, 1.0),
            'j': (5, -1.0),
        }

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info(
            f'Piper 键盘关节点动已启动，订阅 {joint_state_topic}，发布 {joint_command_topic}。'
        )
        self.print_help()

    def joint_callback(self, msg):
        value_map = dict(zip(msg.name, msg.position))
        if not all(name in value_map for name in self.joint_names):
            return
        joints = np.array([value_map[name] for name in self.joint_names], dtype=float)
        if not np.all(np.isfinite(joints)):
            return
        self.current_joints = joints
        self.current_joint_time = time.time()

    def keyboard_loop(self):
        if not sys.stdin.isatty():
            self.get_logger().warn('标准输入不是 TTY，无法启用即时键盘读取。')
            return

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue

                char = sys.stdin.read(1)
                if not char:
                    continue

                char = char.lower()
                if char in ('\x03', '\x04'):
                    self.keyboard_queue.put('q')
                    return
                if char in ('?',):
                    self.keyboard_queue.put('help')
                    continue
                if char in ('\r', '\n', ' '):
                    continue
                self.keyboard_queue.put(char)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def timer_callback(self):
        while True:
            try:
                token = self.keyboard_queue.get_nowait()
            except queue.Empty:
                return
            self.handle_token(token)

    def handle_token(self, token):
        if token in ('q', 'quit', 'exit'):
            self.get_logger().info('收到退出命令。')
            rclpy.shutdown()
            return
        if token in ('help', '?'):
            self.print_help()
            return
        if token.startswith('step '):
            self.set_step(token)
            return
        if token not in self.key_bindings:
            self.get_logger().warn(f'未知按键: {token}')
            return
        if self.current_joints is None or time.time() - self.current_joint_time > 1.0:
            self.get_logger().warn('尚未收到新鲜的 /joint_states_single，不能点动。')
            return

        joint_index, direction = self.key_bindings[token]
        target = self.current_joints.copy()
        target[joint_index] += direction * self.step_rad
        target = np.clip(target, self.joint_lower, self.joint_upper)
        self.publish_target(target)
        self.get_logger().info(
            f'发送 joint{joint_index + 1} 点动，目标={np.round(target, 4).tolist()}'
        )

    def set_step(self, token):
        try:
            step_deg = float(token.split(maxsplit=1)[1])
        except (IndexError, ValueError):
            self.get_logger().warn('用法: step <角度deg>，例如 step 2')
            return
        if step_deg <= 0.0 or step_deg > 15.0:
            self.get_logger().warn('步长建议在 (0, 15] deg。')
            return
        self.step_rad = np.deg2rad(step_deg)
        self.get_logger().info(f'点动步长已设置为 {step_deg:.3f} deg ({self.step_rad:.4f} rad)')

    def publish_target(self, target):
        msg = JointTrajectory()
        msg.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in target]
        point.time_from_start = Duration(seconds=self.duration_sec).to_msg()
        msg.points.append(point)
        self.command_pub.publish(msg)

    def print_help(self):
        text = (
            '键位: A/D=joint1, W/S=joint2, R/F=joint3, T/G=joint5, '
            'Y/H=joint4, U/J=joint6；按键立即生效，无需回车，可连续按键；? 显示帮助；q 退出。'
        )
        self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = PiperKeyboardJointJog()
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
