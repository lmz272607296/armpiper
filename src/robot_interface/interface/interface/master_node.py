#!/usr/bin/env python3

import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


BOTTLE = 'bottle'
FRUIT = 'fruit'
OBJECT_TYPE_TOPIC = '/grasp_object_type'
GRASP_SELECTION_TOPIC = '/grasp_selection'
ROTATION_ACTION = 'action5_rot'
MOVE_ACTION = 'action1_4_move'
ACTION_STOP_TOPIC = '/action_stop'
DEFAULT_GRASP_POSITION = 'right'
DEFAULT_INPUT_MODE = 'keyboard'
DEFAULT_TCP_LOCAL_HOST = '0.0.0.0'
DEFAULT_TCP_LOCAL_PORT = 9999
DEFAULT_TCP_REMOTE_HOST = '192.168.3.204'
DEFAULT_TCP_REMOTE_PORT = 5000
INPUT_MODE_SWITCH_TOPIC = '/master_input_mode'
MASTER_RECEIVED_SIGNAL_TOPIC = '/master_received_signal'
MASTER_RECEIVED_SIGNAL_PERIOD_SEC = 0.2
CONTROL_MESSAGE_REPEAT_COUNT = 3
CONTROL_MESSAGE_REPEAT_INTERVAL_SEC = 0.05

BASE_POSITION_LIMITS = {
    BOTTLE: {
        'x': (0.45, 0.70),
        'y': (-0.25, 0.25),
    },
    FRUIT: {
        'x': (0.45, 0.70),
        'y': (-0.25, 0.25),
    },
}


def normalize_object_keyword(value):
    normalized = str(value).strip().lower()
    aliases = {
        '8': BOTTLE,
        BOTTLE: BOTTLE,
        '9': FRUIT,
        FRUIT: FRUIT,
        'furit': FRUIT,
        'cube': FRUIT,
        'block': FRUIT,
        'square': FRUIT,
        '方块': FRUIT,
        'orange': FRUIT,
        'apple': FRUIT,
    }
    return aliases.get(normalized, normalized)


class MasterNode(Node):
    def __init__(self):
        super().__init__('master_node')

        self.declare_parameter('signal_topic', '/recognized_signal')
        self.declare_parameter('detection_topic', '/yolo_3d_detections_json')
        self.declare_parameter('status_topic', '/task_status')
        self.declare_parameter('arm_action_topic', '/arm_action_command')
        self.declare_parameter('grasp_pose_topic', '/grasp_target_pose')
        self.declare_parameter('grasp_selection_topic', GRASP_SELECTION_TOPIC)
        self.declare_parameter('object_type_topic', OBJECT_TYPE_TOPIC)
        self.declare_parameter('package_name', 'interface')
        self.declare_parameter('enable_keyboard_input', True)
        self.declare_parameter('enable_tcp_input', True)
        self.declare_parameter('input_mode', DEFAULT_INPUT_MODE)
        self.declare_parameter('tcp_local_host', DEFAULT_TCP_LOCAL_HOST)
        self.declare_parameter('tcp_local_port', DEFAULT_TCP_LOCAL_PORT)
        self.declare_parameter('tcp_remote_host', DEFAULT_TCP_REMOTE_HOST)
        self.declare_parameter('tcp_remote_port', DEFAULT_TCP_REMOTE_PORT)
        self.declare_parameter('input_mode_switch_topic', INPUT_MODE_SWITCH_TOPIC)
        self.declare_parameter('master_received_signal_topic', MASTER_RECEIVED_SIGNAL_TOPIC)
        self.declare_parameter('master_received_signal_period_sec', MASTER_RECEIVED_SIGNAL_PERIOD_SEC)
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('cache_duration', 3.0)
        self.declare_parameter('detection_timeout_sec', 3.0)
        self.declare_parameter('move_step_m', 0.04)
        self.declare_parameter('debug_grasp_mode', False)
        self.declare_parameter('action_log_to_console', True)

        signal_topic = self.get_parameter('signal_topic').value
        detection_topic = self.get_parameter('detection_topic').value
        status_topic = self.get_parameter('status_topic').value
        arm_action_topic = self.get_parameter('arm_action_topic').value
        grasp_pose_topic = self.get_parameter('grasp_pose_topic').value
        self.grasp_selection_topic = self.get_parameter('grasp_selection_topic').value
        self.object_type_topic = self.get_parameter('object_type_topic').value
        self.package_name = self.get_parameter('package_name').value
        self.enable_keyboard_input = bool(self.get_parameter('enable_keyboard_input').value)
        self.enable_tcp_input = bool(self.get_parameter('enable_tcp_input').value)
        self.input_mode_switch_topic = self.get_parameter('input_mode_switch_topic').value
        self.master_received_signal_topic = self.get_parameter('master_received_signal_topic').value
        self.master_received_signal_period_sec = float(
            self.get_parameter('master_received_signal_period_sec').value
        )
        self.tcp_local_host = str(self.get_parameter('tcp_local_host').value)
        self.tcp_local_port = int(self.get_parameter('tcp_local_port').value)
        self.tcp_remote_host = str(self.get_parameter('tcp_remote_host').value)
        self.tcp_remote_port = int(self.get_parameter('tcp_remote_port').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.cache_duration = float(self.get_parameter('cache_duration').value)
        self.detection_timeout_sec = float(self.get_parameter('detection_timeout_sec').value)
        self.move_step_m = float(self.get_parameter('move_step_m').value)
        self.debug_grasp_mode = bool(self.get_parameter('debug_grasp_mode').value)
        self.action_log_to_console = bool(self.get_parameter('action_log_to_console').value)
        self.keyboard_available = self.enable_keyboard_input and sys.stdin.isatty()
        self.tcp_input_enabled = self.enable_tcp_input
        self.manual_input_mode = DEFAULT_INPUT_MODE
        self.debug_grasp_position = {
            'x': 0.55,
            'y': 0.0,
            'z': 0.0,
        }
        self.debug_grasp_positions = {
            BOTTLE: self.debug_grasp_position,
            FRUIT: {
                'x': 0.55,
                'y': 0.0,
                'z': 0.0,
            },
        }

        self.latest_detections = {}
        self.last_detection_msg_time = 0.0
        self.selected_object = None
        self.pending_action = None
        self.pending_grasp_position = None
        self.active_grasp_request = None
        self.holding_object = False
        self.held_object = None
        self.processes = {}
        self.keyboard_queue = queue.Queue()
        self.tcp_queue = queue.Queue()
        self.tcp_stop_event = threading.Event()
        self.tcp_local_server_socket = None
        self.tcp_socket = None
        self.latest_received_signal = self.build_received_signal_payload(
            source='system',
            raw_signal='',
            token='',
            accepted=False,
            note='master_node_started',
        )

        self.signal_sub = self.create_subscription(String, signal_topic, self.signal_callback, 10)
        self.detection_sub = self.create_subscription(String, detection_topic, self.detection_callback, 10)
        self.input_mode_switch_sub = self.create_subscription(
            String,
            self.input_mode_switch_topic,
            self.input_mode_switch_callback,
            10,
        )
        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.arm_action_pub = self.create_publisher(String, arm_action_topic, 10)
        self.grasp_pose_pub = self.create_publisher(PoseStamped, grasp_pose_topic, 10)
        self.grasp_selection_pub = self.create_publisher(String, self.grasp_selection_topic, 10)
        self.object_type_pub = self.create_publisher(String, self.object_type_topic, 10)
        self.action_stop_pub = self.create_publisher(String, ACTION_STOP_TOPIC, 10)
        self.received_signal_pub = self.create_publisher(String, self.master_received_signal_topic, 10)
        self.grasp_wait_timer = self.create_timer(0.1, self.grasp_wait_timer_callback)
        self.received_signal_timer = self.create_timer(
            self.master_received_signal_period_sec,
            self.received_signal_timer_callback,
        )

        self.signal_map = {
            '1': 'front',
            '2': 'back',
            '3': 'left',
            '4': 'right',
            '5': 'rotate',
            '6': 'grasp',
            '7': 'release',
            '8': BOTTLE,
            '9': FRUIT,
            '10': 'reset',
        }
        self.direction_text = {
            'front': '前',
            'back': '后',
            'left': '左',
            'right': '右',
        }
        self.target_text = {
            BOTTLE: '瓶子',
            FRUIT: '水果',
        }

        self.get_logger().info(
            f'中控节点已启动，订阅 {signal_topic}；1-4=前后左右，5=旋转，6=抓取，'
            f'7=释放，8=bottle，9=fruit，10=复位。'
        )
        self.get_logger().info(
            f'手动输入模式切换话题: {self.input_mode_switch_topic}；'
            '11=键盘模式，12=本地 TCP 监听模式，13=远端 TCP 连接模式。'
        )
        self.get_logger().info(
            f'接收信号转发话题: {self.master_received_signal_topic}，'
            f'周期 {self.master_received_signal_period_sec:.2f} 秒。'
        )
        self.get_logger().info(
            f'本地 TCP 监听地址 {self.tcp_local_host}:{self.tcp_local_port}；'
            f'远端 TCP 连接地址 {self.tcp_remote_host}:{self.tcp_remote_port}。'
        )
        if self.debug_grasp_mode:
            self.get_logger().warn(
                '抓取调试模式已启用：抓取时将跳过 YOLO 检测，固定发送 '
                'X=0.55, Y=0.00, Z=0.00。'
            )
        if self.keyboard_available:
            self.keyboard_timer = self.create_timer(0.05, self.keyboard_timer_callback)
            self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
            self.keyboard_thread.start()
            self.get_logger().info(
                '键盘模拟已启用：请在此终端输入 1-10 后回车，'
                '输入 11/12/13 可切换输入模式。'
            )
        elif self.enable_keyboard_input:
            self.get_logger().warn('当前不是交互式终端，键盘模拟未启用。')
        if self.tcp_input_enabled:
            self.tcp_timer = self.create_timer(0.05, self.tcp_timer_callback)
            self.start_tcp_local_server()
            self.start_tcp_client()
        self.publish_received_signal_state()

    def signal_callback(self, msg):
        raw_signal = msg.data.strip()
        if not raw_signal:
            return
        self.process_signal(raw_signal, source='ros_topic')

    def input_mode_switch_callback(self, msg):
        raw_signal = msg.data.strip()
        if not raw_signal:
            return
        command = self.parse_mode_switch_command(raw_signal)
        if command is None:
            self.publish_status(
                f'输入模式切换指令无效: {raw_signal}。请使用 keyboard/tcp_local/tcp_remote 或 11/12/13。'
            )
            return
        self.switch_manual_input_mode(command, source='mode_topic', raw_signal=raw_signal)

    def process_signal(self, raw_signal, source):
        token = self.signal_map.get(raw_signal, normalize_object_keyword(raw_signal))
        self.record_received_signal(
            source=source,
            raw_signal=raw_signal,
            token=token,
            accepted=True,
        )
        self.get_logger().info(f'收到信号[{source}]: {raw_signal} -> {token}')
        self.handle_token(token)

    def process_manual_signal(self, raw_signal, source):
        command = self.parse_mode_switch_command(raw_signal)
        if command is not None and source not in ['tcp_local', 'tcp_remote']:
            self.switch_manual_input_mode(command, source=source, raw_signal=raw_signal)
            return

        if self.manual_input_mode != source:
            token = self.signal_map.get(raw_signal, normalize_object_keyword(raw_signal))
            self.record_received_signal(
                source=source,
                raw_signal=raw_signal,
                token=token,
                accepted=False,
                note=f'{source}_input_ignored',
            )
            self.get_logger().info(
                f'忽略 {source} 输入 {raw_signal}：当前手动输入模式为 {self.manual_input_mode}。'
            )
            return

        self.process_signal(raw_signal, source=source)

    def parse_mode_switch_command(self, raw_signal):
        normalized = str(raw_signal).strip().lower()
        mapping = {
            '11': 'keyboard',
            'keyboard': 'keyboard',
            'kbd': 'keyboard',
            'mode keyboard': 'keyboard',
            '键盘': 'keyboard',
            '12': 'tcp_local',
            'tcp_local': 'tcp_local',
            'local': 'tcp_local',
            'local tcp': 'tcp_local',
            'mode tcp_local': 'tcp_local',
            '本地': 'tcp_local',
            '本地监听': 'tcp_local',
            '13': 'tcp_remote',
            'tcp_remote': 'tcp_remote',
            'remote': 'tcp_remote',
            'remote tcp': 'tcp_remote',
            'mode tcp_remote': 'tcp_remote',
            '远端': 'tcp_remote',
            '远端监听': 'tcp_remote',
        }
        return mapping.get(normalized)

    def switch_manual_input_mode(self, command, source='system', raw_signal=''):
        target_mode = command

        if target_mode == 'keyboard' and not self.keyboard_available:
            self.record_received_signal(
                source=source,
                raw_signal=raw_signal or command,
                token='switch_to_keyboard',
                accepted=False,
                note='keyboard_input_unavailable',
            )
            self.publish_status('无法切换到键盘输入模式：当前终端不是交互式终端或键盘输入未启用。')
            return False

        if target_mode in ['tcp_local', 'tcp_remote'] and not self.tcp_input_enabled:
            self.record_received_signal(
                source=source,
                raw_signal=raw_signal or command,
                token=f'switch_to_{target_mode}',
                accepted=False,
                note='tcp_input_unavailable',
            )
            self.publish_status('无法切换到 TCP 输入模式：TCP 输入未启用。')
            return False

        self.manual_input_mode = target_mode
        if target_mode == 'keyboard':
            self.close_tcp_socket()
        self.record_received_signal(
            source=source,
            raw_signal=raw_signal or command,
            token=f'switch_to_{target_mode}',
            accepted=True,
        )
        if target_mode == 'tcp_local':
            self.publish_status(
                f'已切换到{self.manual_input_mode_text(target_mode)}，监听 '
                f'{self.tcp_local_host}:{self.tcp_local_port}。'
            )
        elif target_mode == 'tcp_remote':
            self.publish_status(
                f'已切换到{self.manual_input_mode_text(target_mode)}，正在连接 '
                f'{self.tcp_remote_host}:{self.tcp_remote_port}。'
            )
        else:
            self.publish_status(f'已切换到{self.manual_input_mode_text(target_mode)}。')
        return True

    def manual_input_mode_text(self, mode=None):
        resolved_mode = self.manual_input_mode if mode is None else mode
        mapping = {
            'keyboard': '键盘输入模式',
            'tcp_local': '本地 TCP 监听模式',
            'tcp_remote': '远端 TCP 连接模式',
        }
        return mapping.get(resolved_mode, resolved_mode)

    def build_received_signal_payload(self, source, raw_signal, token, accepted, note=''):
        return {
            'source': source,
            'raw_signal': str(raw_signal),
            'token': str(token),
            'accepted': bool(accepted),
            'manual_input_mode': self.manual_input_mode,
            'note': note,
            'timestamp': time.time(),
        }

    def record_received_signal(self, source, raw_signal, token, accepted, note=''):
        self.latest_received_signal = self.build_received_signal_payload(
            source=source,
            raw_signal=raw_signal,
            token=token,
            accepted=accepted,
            note=note,
        )
        self.publish_received_signal_state()

    def publish_received_signal_state(self):
        msg = String()
        msg.data = json.dumps(self.latest_received_signal, ensure_ascii=False)
        self.received_signal_pub.publish(msg)

    def received_signal_timer_callback(self):
        self.publish_received_signal_state()

    def keyboard_loop(self):
        prompt = '请输入信号 1-10（11=键盘，12=本地TCP，13=远端TCP，q 退出键盘输入）：'
        while rclpy.ok():
            try:
                raw_signal = input(prompt).strip()
            except EOFError:
                return
            except KeyboardInterrupt:
                return

            if not raw_signal:
                continue
            if raw_signal.lower() in ['q', 'quit', 'exit']:
                self.keyboard_queue.put(raw_signal)
                return
            self.keyboard_queue.put(raw_signal)

    def keyboard_timer_callback(self):
        while True:
            try:
                raw_signal = self.keyboard_queue.get_nowait()
            except queue.Empty:
                return

            if raw_signal.lower() in ['q', 'quit', 'exit']:
                self.publish_status('键盘输入已退出，ROS 话题订阅仍保持运行。')
                return

            self.process_manual_signal(raw_signal, source='keyboard')

    def start_tcp_local_server(self):
        try:
            self.tcp_local_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_local_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_local_server_socket.bind((self.tcp_local_host, self.tcp_local_port))
            self.tcp_local_server_socket.listen()
            self.tcp_local_server_socket.settimeout(1.0)
        except OSError as exc:
            self.publish_status(
                f'本地 TCP 监听启动失败 {self.tcp_local_host}:{self.tcp_local_port}：{exc}'
            )
            self.tcp_local_server_socket = None
            return

        self.tcp_local_thread = threading.Thread(target=self.tcp_local_server_loop, daemon=True)
        self.tcp_local_thread.start()
        self.get_logger().info(
            f'本地 TCP 监听线程已启动：{self.tcp_local_host}:{self.tcp_local_port}。'
        )

    def tcp_local_server_loop(self):
        while rclpy.ok() and not self.tcp_stop_event.is_set():
            if self.tcp_local_server_socket is None:
                return

            try:
                connection, address = self.tcp_local_server_socket.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if self.tcp_stop_event.is_set():
                    return
                self.get_logger().warn(f'本地 TCP accept 失败：{exc}')
                time.sleep(0.2)
                continue

            self.get_logger().info(f'收到本地 TCP 连接: {address[0]}:{address[1]}')
            with connection:
                connection.settimeout(1.0)
                buffer = ''
                while rclpy.ok() and not self.tcp_stop_event.is_set():
                    if self.manual_input_mode != 'tcp_local':
                        break

                    try:
                        chunk = connection.recv(1024)
                    except socket.timeout:
                        continue
                    except OSError as exc:
                        self.get_logger().warn(f'本地 TCP 接收失败：{exc}')
                        break

                    if not chunk:
                        break

                    buffer += chunk.decode('utf-8', errors='ignore')
                    buffer = self.handle_tcp_buffer(buffer, source='tcp_local')

                tail = buffer.strip()
                if tail:
                    self.tcp_queue.put(('tcp_local', tail))

    def start_tcp_client(self):
        self.tcp_thread = threading.Thread(target=self.tcp_client_loop, daemon=True)
        self.tcp_thread.start()
        self.get_logger().info(
            f'远端 TCP 接收线程已启动：切换到远端模式后将连接 '
            f'{self.tcp_remote_host}:{self.tcp_remote_port}。'
        )

    def tcp_client_loop(self):
        buffer = ''
        while rclpy.ok() and not self.tcp_stop_event.is_set():
            if self.manual_input_mode != 'tcp_remote':
                self.close_tcp_socket()
                buffer = ''
                time.sleep(0.1)
                continue

            if self.tcp_socket is None:
                try:
                    self.tcp_socket = socket.create_connection(
                        (self.tcp_remote_host, self.tcp_remote_port),
                        timeout=3.0,
                    )
                    self.tcp_socket.settimeout(1.0)
                    self.get_logger().info(
                        f'已连接远端 TCP {self.tcp_remote_host}:{self.tcp_remote_port}。'
                    )
                except OSError as exc:
                    self.get_logger().warn(
                        f'连接远端 TCP {self.tcp_remote_host}:{self.tcp_remote_port} 失败：{exc}'
                    )
                    time.sleep(1.0)
                    continue

            try:
                chunk = self.tcp_socket.recv(1024)
            except socket.timeout:
                continue
            except OSError as exc:
                if self.tcp_stop_event.is_set():
                    return
                self.get_logger().warn(f'TCP 接收失败，将尝试重连：{exc}')
                self.close_tcp_socket()
                buffer = ''
                time.sleep(0.5)
                continue

            if not chunk:
                self.get_logger().warn('远端 TCP 连接已关闭，准备重连。')
                self.close_tcp_socket()
                buffer = ''
                time.sleep(0.5)
                continue

            buffer += chunk.decode('utf-8', errors='ignore')
            buffer = self.handle_tcp_buffer(buffer, source='tcp_remote')

        tail = buffer.strip()
        if tail:
            self.tcp_queue.put(('tcp_remote', tail))

    def close_tcp_socket(self):
        if self.tcp_socket is None:
            return
        try:
            self.tcp_socket.close()
        except OSError:
            pass
        self.tcp_socket = None

    def close_tcp_local_server_socket(self):
        if self.tcp_local_server_socket is None:
            return
        try:
            self.tcp_local_server_socket.close()
        except OSError:
            pass
        self.tcp_local_server_socket = None

    def handle_tcp_buffer(self, buffer, source):
        separators = ['\r', '\n', ',', ';', ' ', '\t']
        normalized = buffer
        for separator in separators:
            normalized = normalized.replace(separator, '\n')

        parts = normalized.split('\n')
        tail = ''
        if normalized and normalized[-1] != '\n':
            tail = parts.pop()

        for item in parts:
            raw_signal = item.strip()
            if raw_signal:
                self.tcp_queue.put((source, raw_signal))

        return tail

    def tcp_timer_callback(self):
        while True:
            try:
                source, raw_signal = self.tcp_queue.get_nowait()
            except queue.Empty:
                return

            self.process_manual_signal(raw_signal, source=source)

    def detection_callback(self, msg):
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f'解析 YOLO JSON 失败: {exc}')
            return

        now = time.time()
        self.last_detection_msg_time = now
        grouped = {}
        for det in detections:
            object_type = normalize_object_keyword(det.get('object_type') or det.get('class_name', ''))
            if object_type not in (BOTTLE, FRUIT):
                continue

            confidence = det.get('confidence')
            center_3d = det.get('center_3d')
            if confidence is None or not center_3d:
                continue
            if float(confidence) <= self.conf_threshold:
                continue

            grouped.setdefault(object_type, []).append({
                'object_type': object_type,
                'class_name': object_type,
                'source_class_name': det.get('class_name') or object_type,
                'instance_name': det.get('instance_name') or object_type,
                'position_label': det.get('position_label'),
                'position_rank': det.get('position_rank'),
                'center_pixel': det.get('center_pixel'),
                'confidence': float(confidence),
                'position': {
                    'x': float(center_3d.get('x', 0.0)),
                    'y': float(center_3d.get('y', 0.0)),
                    'z': float(center_3d.get('z', 0.0)),
                },
                'timestamp': now,
            })

        self.latest_detections = grouped

        if self.active_grasp_request is not None:
            self.try_complete_active_grasp(grouped)

    def grasp_wait_timer_callback(self):
        if self.active_grasp_request is None:
            return

        now = time.time()
        if now < self.active_grasp_request['deadline']:
            return

        request = self.active_grasp_request
        self.active_grasp_request = None
        prefix = self.position_text(request.get('position_label'))
        self.publish_status(
            f"拒绝抓取：等待 {self.detection_timeout_sec:.1f} 秒后仍未收到有效的"
            f"{prefix}{self.target_text[request['object_type']]}坐标。"
        )

    def handle_token(self, token):
        if token in ['front', 'back', 'left', 'right']:
            self.handle_direction(token)
            return

        self.stop_move_action_if_running(f'收到非方向信号 {token}')

        if token == 'rotate':
            self.clear_pending()
            self.launch_action(ROTATION_ACTION)
            return
        if token == 'grasp':
            self.handle_grasp_signal()
            return
        if token == 'release':
            self.handle_release_signal()
            return
        if token in [BOTTLE, FRUIT]:
            self.handle_object_selection(token)
            return
        if token == 'reset':
            self.handle_reset_signal()
            return

        self.publish_status(f'未知信号: {token}')

    def handle_direction(self, direction):
        if self.pending_action == 'grasp' and not self.holding_object and direction in ['left', 'right']:
            self.pending_grasp_position = direction
            self.publish_status(
                f"已选择抓取目标方位：{self.position_text(direction)}。请继续发送 8(bottle) 或 9(fruit)。"
            )
            return

        self.launch_action(MOVE_ACTION)
        self.wait_for_subscribers(self.arm_action_pub, '/arm_action_command')
        self.publish_arm_action({
            'type': 'move',
            'direction': direction,
            'distance': self.move_step_m,
        })
        self.publish_status(f"已发送向{self.direction_text[direction]}移动 {self.move_step_m:.2f} 米指令。")
        self.clear_pending()

    def handle_grasp_signal(self):
        if self.holding_object:
            self.publish_status('当前已经处于抓取状态，必须先发送释放指令后才能再次抓取。')
            return

        self.clear_pending()
        self.selected_object = None
        self.pending_action = 'grasp'
        self.pending_grasp_position = None
        self.publish_status(
            '已收到抓取指令，可先发送 3(左) 或 4(右) 选择同类目标方位，'
            '再发送 8(bottle) 或 9(fruit) 选择物体；若未选择方位，默认抓取最右侧目标。'
        )

    def handle_release_signal(self):
        self.launch_action('action7_flat')

        object_type = self.selected_object or self.held_object
        if object_type is None:
            self.pending_action = 'release'
            self.publish_status('已启动释放节点，请继续发送 8(bottle) 或 9(fruit) 选择释放动作。')
            return

        self.execute_release(object_type)

    def handle_object_selection(self, object_type):
        self.selected_object = object_type
        self.publish_object_type(object_type)
        self.publish_status(f"已选择物体类型：{object_type}。")

        if self.pending_action == 'grasp':
            self.execute_grasp(object_type, self.pending_grasp_position)
        elif self.pending_action == 'release':
            self.execute_release(object_type)

    def execute_grasp(self, object_type, position_label=None):
        grasp_position = self.resolve_grasp_position_label(position_label)
        if self.debug_grasp_mode:
            detection = {
                'position': self.debug_grasp_positions[object_type],
                'position_label': grasp_position,
                'confidence': 1.0,
            }
            return self.publish_grasp_target(object_type, detection)
        else:
            self.launch_action('action6_gsp')
            self.active_grasp_request = {
                'object_type': object_type,
                'position_label': grasp_position,
                'deadline': time.time() + self.detection_timeout_sec,
            }
            self.wait_for_subscribers(self.grasp_selection_pub, self.grasp_selection_topic)
            self.publish_grasp_selection(object_type, grasp_position)
            prefix = self.position_text(grasp_position)
            self.publish_status(
                f"等待 {self.detection_timeout_sec:.1f} 秒内的新 YOLO 3D 检测，"
                f"用于抓取{prefix}{self.target_text[object_type]}。"
            )
            return True

    def try_complete_active_grasp(self, grouped):
        if self.active_grasp_request is None:
            return False

        request = self.active_grasp_request
        if time.time() > request['deadline']:
            self.grasp_wait_timer_callback()
            return False

        detection = self.select_detection_from_grouped(
            grouped,
            request['object_type'],
            request.get('position_label'),
        )
        if detection is None:
            return False

        self.active_grasp_request = None
        return self.publish_grasp_target(request['object_type'], detection)

    def publish_grasp_target(self, object_type, detection):
        self.launch_action('action6_gsp')
        self.wait_for_subscribers(self.object_type_pub, self.object_type_topic)
        self.wait_for_subscribers(self.grasp_pose_pub, '/grasp_target_pose')
        self.publish_object_type_repeated(object_type)

        pos = self.clamp_detection_position(object_type, detection['position'])
        grasp_pose = PoseStamped()
        grasp_pose.header.stamp = self.get_clock().now().to_msg()
        grasp_pose.header.frame_id = f"{object_type}:{self.resolve_grasp_position_label(detection.get('position_label'))}"
        grasp_pose.pose.position.x = pos['x']
        grasp_pose.pose.position.y = pos['y']
        grasp_pose.pose.position.z = pos['z']
        grasp_pose.pose.orientation.w = 1.0

        self.publish_repeated(self.grasp_pose_pub, grasp_pose)

        self.holding_object = True
        self.held_object = object_type
        self.clear_pending()
        prefix = self.position_text(detection.get('position_label'))
        mode_text = '调试模式固定' if self.debug_grasp_mode else ''
        self.publish_status(
            f"已向 action6_gsp 发送抓取{prefix}{self.target_text[object_type]}{mode_text}目标："
            f"X={pos['x']:.2f}, Y={pos['y']:.2f}, Z={pos['z']:.2f}，"
            f"置信度={detection['confidence']:.1%}。"
        )
        return True

    def execute_release(self, object_type):
        self.launch_action('action7_flat')
        self.wait_for_subscribers(self.object_type_pub, self.object_type_topic)
        self.publish_object_type(object_type)
        self.holding_object = False
        self.held_object = None
        self.clear_pending()
        self.publish_status(f"已向 action7_flat 发送释放物体类型：{object_type}。")
        return True

    def select_detection(self, object_type, position_label=None):
        now = time.time()
        candidates = [
            detection for detection in self.latest_detections.get(object_type, [])
            if now - detection['timestamp'] <= self.cache_duration
        ]
        return self.select_detection_from_candidates(candidates, position_label)

    def select_detection_from_grouped(self, grouped, object_type, position_label=None):
        candidates = grouped.get(object_type, [])
        return self.select_detection_from_candidates(candidates, position_label)

    def select_detection_from_candidates(self, candidates, position_label=None):
        if not candidates:
            return None

        resolved_position = self.resolve_grasp_position_label(position_label)
        candidates_with_y = [
            detection for detection in candidates
            if self.detection_axis_value(detection, 'y') is not None
        ]
        if not candidates_with_y:
            selected = max(candidates, key=lambda item: item.get('confidence', 0.0))
            selected = dict(selected)
            selected['position_label'] = resolved_position
            return selected

        if resolved_position == 'left':
            selected = max(
                candidates_with_y,
                key=lambda item: (self.detection_axis_value(item, 'y'), item.get('confidence', 0.0)),
            )
        else:
            selected = min(
                candidates_with_y,
                key=lambda item: (self.detection_axis_value(item, 'y'), -item.get('confidence', 0.0)),
            )

        selected = dict(selected)
        selected['position_label'] = resolved_position
        return selected

    def clamp_detection_position(self, object_type, position):
        limits = BASE_POSITION_LIMITS.get(object_type, {})
        clamped = dict(position)
        for axis, (low, high) in limits.items():
            value = clamped.get(axis)
            if value is None:
                continue
            clamped[axis] = min(max(float(value), low), high)

        return clamped

    def detection_is_fresh(self):
        if self.last_detection_msg_time <= 0.0:
            return False
        return time.time() - self.last_detection_msg_time <= self.detection_timeout_sec

    def handle_reset_signal(self):
        self.stop_all_actions('收到复位指令 10')
        self.reset_runtime_state()
        self.launch_action('action8_rst')
        self.publish_status('已取消当前全部逻辑，内部状态已恢复到初始状态，并重新启动复位节点 action8_rst。')

    def stop_move_action_if_running(self, reason=''):
        self.stop_process(MOVE_ACTION, reason, timeout_sec=1.5)

    def reset_runtime_state(self):
        self.latest_detections = {}
        self.last_detection_msg_time = 0.0
        self.selected_object = None
        self.holding_object = False
        self.held_object = None
        self.clear_pending()

    def stop_all_actions(self, reason=''):
        self.cleanup_finished_processes()
        for executable in list(self.processes.keys()):
            self.stop_process(executable, reason, timeout_sec=1.5)

    def is_process_running(self, executable):
        process = self.processes.get(executable)
        if process is None:
            return False
        if process.poll() is not None:
            self.processes.pop(executable, None)
            return False
        return True

    def launch_action(self, executable):
        if executable == ROTATION_ACTION:
            self.broadcast_action_stop(ROTATION_ACTION)
            self.stop_process(ROTATION_ACTION, '重启旋转动作')
        else:
            self.broadcast_action_stop(ROTATION_ACTION)
            self.stop_process(ROTATION_ACTION, f'收到新动作 {executable}')

        self.cleanup_finished_processes()
        process = self.processes.get(executable)
        if process is not None and process.poll() is None:
            self.publish_status(f'{executable} 已在运行。')
            return

        command = ['ros2', 'run', self.package_name, executable]

        try:
            stdout_target = None if self.action_log_to_console else subprocess.DEVNULL
            stderr_target = None if self.action_log_to_console else subprocess.DEVNULL
            self.processes[executable] = subprocess.Popen(
                command,
                stdout=stdout_target,
                stderr=stderr_target,
                start_new_session=True,
            )
            self.publish_status(f'已启动 {executable}。')
        except Exception as exc:
            self.publish_status(f'启动 {executable} 失败: {exc}')

    def cleanup_finished_processes(self):
        for executable, process in list(self.processes.items()):
            if process.poll() is not None:
                self.processes.pop(executable, None)

    def stop_process(self, executable, reason='', timeout_sec=2.0):
        process = self.processes.get(executable)
        if process is None:
            return False
        if process.poll() is not None:
            self.processes.pop(executable, None)
            return False

        suffix = f'：{reason}' if reason else ''
        self.publish_status(f'正在停止 {executable}{suffix}。')
        if executable == ROTATION_ACTION:
            self.broadcast_action_stop(ROTATION_ACTION)
            time.sleep(0.15)
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=timeout_sec)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            self.publish_status(f'{executable} 未及时退出，强制终止该动作节点。')
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=1.0)
            except ProcessLookupError:
                pass
            except Exception as exc:
                self.publish_status(f'强制终止 {executable} 失败: {exc}')
                return False
        except Exception as exc:
            self.publish_status(f'停止 {executable} 失败: {exc}')
            return False
        finally:
            if process.poll() is not None:
                self.processes.pop(executable, None)

        return True

    def shutdown_actions(self):
        self.tcp_stop_event.set()
        self.close_tcp_local_server_socket()
        self.close_tcp_socket()
        self.broadcast_action_stop(ROTATION_ACTION)
        time.sleep(0.15)
        for executable in list(self.processes.keys()):
            self.stop_process(executable, 'master_node 退出', timeout_sec=1.5)

    def wait_for_subscribers(self, publisher, topic_name, timeout_sec=5.0):
        deadline = time.time() + timeout_sec
        while time.time() < deadline and rclpy.ok():
            count = publisher.get_subscription_count()
            if hasattr(publisher, 'get_intra_process_subscription_count'):
                count += publisher.get_intra_process_subscription_count()
            if count > 0:
                time.sleep(0.3)
                return True
            time.sleep(0.05)

        self.get_logger().warn(f'{topic_name} 暂无订阅者，仍继续发布；若节点刚启动，首条消息可能丢失。')
        return False

    def publish_object_type(self, object_type):
        msg = String()
        msg.data = object_type
        self.object_type_pub.publish(msg)

    def publish_object_type_repeated(self, object_type):
        msg = String()
        msg.data = object_type
        self.publish_repeated(self.object_type_pub, msg)

    def publish_grasp_selection(self, object_type, position_label):
        msg = String()
        msg.data = json.dumps({
            'object_type': object_type,
            'position_label': self.resolve_grasp_position_label(position_label),
        })
        self.publish_repeated(self.grasp_selection_pub, msg)

    def publish_repeated(self, publisher, msg):
        for index in range(CONTROL_MESSAGE_REPEAT_COUNT):
            publisher.publish(msg)
            if index + 1 < CONTROL_MESSAGE_REPEAT_COUNT:
                time.sleep(CONTROL_MESSAGE_REPEAT_INTERVAL_SEC)

    def publish_arm_action(self, payload):
        msg = String()
        msg.data = json.dumps(payload)
        self.arm_action_pub.publish(msg)

    def broadcast_action_stop(self, action_name):
        msg = String()
        msg.data = action_name
        self.action_stop_pub.publish(msg)

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(status)

    def clear_pending(self):
        self.pending_action = None
        self.pending_grasp_position = None
        self.active_grasp_request = None

    def resolve_grasp_position_label(self, position_label):
        return 'left' if position_label == 'left' else DEFAULT_GRASP_POSITION

    def detection_axis_value(self, detection, axis):
        position = detection.get('position') or {}
        value = position.get(axis)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def position_text(self, position_label):
        return {'left': '左边的', 'middle': '中间的', 'right': '右边的'}.get(position_label, '')


def main(args=None):
    rclpy.init(args=args)
    node = MasterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_actions()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
