#!/usr/bin/env python3

import json
import queue
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


def normalize_object_keyword(value):
    normalized = str(value).strip().lower()
    aliases = {
        '8': BOTTLE,
        BOTTLE: BOTTLE,
        '9': FRUIT,
        FRUIT: FRUIT,
        'furit': FRUIT,
        'cube': FRUIT,
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
        self.declare_parameter('object_type_topic', OBJECT_TYPE_TOPIC)
        self.declare_parameter('package_name', 'interface')
        self.declare_parameter('enable_keyboard_input', True)
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('cache_duration', 3.0)
        self.declare_parameter('detection_timeout_sec', 3.0)
        self.declare_parameter('move_step_m', 0.02)

        signal_topic = self.get_parameter('signal_topic').value
        detection_topic = self.get_parameter('detection_topic').value
        status_topic = self.get_parameter('status_topic').value
        arm_action_topic = self.get_parameter('arm_action_topic').value
        grasp_pose_topic = self.get_parameter('grasp_pose_topic').value
        self.object_type_topic = self.get_parameter('object_type_topic').value
        self.package_name = self.get_parameter('package_name').value
        self.enable_keyboard_input = bool(self.get_parameter('enable_keyboard_input').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.cache_duration = float(self.get_parameter('cache_duration').value)
        self.detection_timeout_sec = float(self.get_parameter('detection_timeout_sec').value)
        self.move_step_m = float(self.get_parameter('move_step_m').value)

        self.latest_detections = {}
        self.last_detection_msg_time = 0.0
        self.selected_object = None
        self.pending_action = None
        self.pending_grasp_position = None
        self.holding_object = False
        self.held_object = None
        self.processes = {}
        self.keyboard_queue = queue.Queue()

        self.signal_sub = self.create_subscription(String, signal_topic, self.signal_callback, 10)
        self.detection_sub = self.create_subscription(String, detection_topic, self.detection_callback, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.arm_action_pub = self.create_publisher(String, arm_action_topic, 10)
        self.grasp_pose_pub = self.create_publisher(PoseStamped, grasp_pose_topic, 10)
        self.object_type_pub = self.create_publisher(String, self.object_type_topic, 10)

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
        if self.enable_keyboard_input and sys.stdin.isatty():
            self.keyboard_timer = self.create_timer(0.05, self.keyboard_timer_callback)
            self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
            self.keyboard_thread.start()
            self.get_logger().info('键盘模拟已启用：请在此终端输入 1-10 后回车。')
        elif self.enable_keyboard_input:
            self.get_logger().warn('当前不是交互式终端，键盘模拟未启用。')

    def signal_callback(self, msg):
        raw_signal = msg.data.strip()
        token = self.signal_map.get(raw_signal, normalize_object_keyword(raw_signal))
        self.get_logger().info(f'收到信号: {raw_signal} -> {token}')
        self.handle_token(token)

    def keyboard_loop(self):
        prompt = '请输入信号 1-10（q 退出键盘输入）：'
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

            token = self.signal_map.get(raw_signal, normalize_object_keyword(raw_signal))
            self.get_logger().info(f'键盘信号: {raw_signal} -> {token}')
            self.handle_token(token)

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

    def handle_token(self, token):
        if token in ['front', 'back', 'left', 'right']:
            self.handle_direction(token)
            return
        if token == 'rotate':
            self.clear_pending()
            self.launch_action('action5_rot')
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
            self.clear_pending()
            self.launch_action('action1_4_move')
            self.wait_for_subscribers(self.arm_action_pub, '/arm_action_command')
            self.publish_arm_action({'type': 'reset'})
            self.publish_status('已发送复位指令。')
            return

        self.publish_status(f'未知信号: {token}')

    def handle_direction(self, direction):
        if self.pending_action == 'grasp' and not self.holding_object and direction in ['left', 'right']:
            self.pending_grasp_position = direction
            self.publish_status(
                f"已选择抓取目标方位：{self.position_text(direction)}。请继续发送 8(bottle) 或 9(fruit)。"
            )
            return

        self.launch_action('action1_4_move')
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

        self.launch_action('action6_gsp')
        if self.selected_object is None:
            self.pending_action = 'grasp'
            self.pending_grasp_position = None
            self.publish_status('已启动抓取节点，请继续发送 8(bottle) 或 9(fruit) 选择物体。')
            return

        self.execute_grasp(self.selected_object, self.pending_grasp_position)

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
        if not self.detection_is_fresh():
            self.publish_status(
                f"拒绝抓取：{self.detection_timeout_sec:.1f} 秒内未收到 YOLO 3D 检测消息，"
                "请确认相机和 ultimate_yolo_node_cpu 正常运行。"
            )
            return False

        detection = self.select_detection(object_type, position_label)
        if detection is None:
            prefix = self.position_text(position_label)
            self.publish_status(f"未找到可抓取的{prefix}{self.target_text[object_type]}。")
            return False

        self.launch_action('action6_gsp')
        self.wait_for_subscribers(self.object_type_pub, self.object_type_topic)
        self.wait_for_subscribers(self.grasp_pose_pub, '/grasp_target_pose')
        self.publish_object_type(object_type)

        pos = detection['position']
        grasp_pose = PoseStamped()
        grasp_pose.header.stamp = self.get_clock().now().to_msg()
        grasp_pose.header.frame_id = object_type
        grasp_pose.pose.position.x = pos['x']
        grasp_pose.pose.position.y = pos['y']
        grasp_pose.pose.position.z = pos['z']
        grasp_pose.pose.orientation.w = 1.0

        self.grasp_pose_pub.publish(grasp_pose)

        self.holding_object = True
        self.held_object = object_type
        self.clear_pending()
        prefix = self.position_text(detection.get('position_label'))
        self.publish_status(
            f"已向 action6_gsp 发送抓取{prefix}{self.target_text[object_type]}目标："
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
        if not candidates:
            return None

        if position_label:
            positioned = [
                detection for detection in candidates
                if detection.get('position_label') == position_label
            ]
            if positioned:
                return max(positioned, key=lambda item: item['confidence'])
            return None

        if len(candidates) == 1:
            return candidates[0]

        middle_candidates = [
            detection for detection in candidates
            if detection.get('position_label') == 'middle'
        ]
        if middle_candidates:
            return max(middle_candidates, key=lambda item: item['confidence'])

        centered_candidates = [
            detection for detection in candidates
            if detection.get('center_pixel')
        ]
        if centered_candidates:
            return min(centered_candidates, key=lambda item: abs(item['center_pixel'][0] - 320))

        return max(candidates, key=lambda item: item['confidence'])

    def detection_is_fresh(self):
        if self.last_detection_msg_time <= 0.0:
            return False
        return time.time() - self.last_detection_msg_time <= self.detection_timeout_sec

    def launch_action(self, executable):
        process = self.processes.get(executable)
        if process is not None and process.poll() is None:
            self.publish_status(f'{executable} 已在运行。')
            return

        try:
            self.processes[executable] = subprocess.Popen(
                ['ros2', 'run', self.package_name, executable],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.publish_status(f'已启动 {executable}。')
        except Exception as exc:
            self.publish_status(f'启动 {executable} 失败: {exc}')

    def wait_for_subscribers(self, publisher, topic_name, timeout_sec=2.0):
        deadline = time.time() + timeout_sec
        while time.time() < deadline and rclpy.ok():
            count = publisher.get_subscription_count()
            if hasattr(publisher, 'get_intra_process_subscription_count'):
                count += publisher.get_intra_process_subscription_count()
            if count > 0:
                return True
            time.sleep(0.05)

        self.get_logger().warn(f'{topic_name} 暂无订阅者，仍继续发布；若节点刚启动，首条消息可能丢失。')
        return False

    def publish_object_type(self, object_type):
        msg = String()
        msg.data = object_type
        self.object_type_pub.publish(msg)

    def publish_arm_action(self, payload):
        msg = String()
        msg.data = json.dumps(payload)
        self.arm_action_pub.publish(msg)

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(status)

    def clear_pending(self):
        self.pending_action = None
        self.pending_grasp_position = None

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
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
