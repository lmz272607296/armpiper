import importlib.util
import os
import sys
import types
import unittest


MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'interface',
    'master_node.py',
)


class FakeString:
    def __init__(self):
        self.data = ''


class FakePoseStamped:
    def __init__(self):
        self.header = types.SimpleNamespace(stamp=None, frame_id='')
        self.pose = types.SimpleNamespace(
            position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=types.SimpleNamespace(w=0.0),
        )


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)

    def get_subscription_count(self):
        return 1

    def get_intra_process_subscription_count(self):
        return 0


def install_stub_modules():
    rclpy_module = types.ModuleType('rclpy')
    rclpy_module.ok = lambda: True
    rclpy_module.init = lambda *args, **kwargs: None
    rclpy_module.shutdown = lambda *args, **kwargs: None
    sys.modules['rclpy'] = rclpy_module

    rclpy_node_module = types.ModuleType('rclpy.node')
    rclpy_node_module.Node = type('Node', (), {})
    sys.modules['rclpy.node'] = rclpy_node_module

    geometry_msgs_module = types.ModuleType('geometry_msgs')
    geometry_msgs_msg_module = types.ModuleType('geometry_msgs.msg')
    geometry_msgs_msg_module.PoseStamped = FakePoseStamped
    sys.modules['geometry_msgs'] = geometry_msgs_module
    sys.modules['geometry_msgs.msg'] = geometry_msgs_msg_module

    std_msgs_module = types.ModuleType('std_msgs')
    std_msgs_msg_module = types.ModuleType('std_msgs.msg')
    std_msgs_msg_module.String = FakeString
    sys.modules['std_msgs'] = std_msgs_module
    sys.modules['std_msgs.msg'] = std_msgs_msg_module


def load_master_node_module():
    install_stub_modules()
    spec = importlib.util.spec_from_file_location('master_node_under_test', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MasterNodeGraspPublishTests(unittest.TestCase):
    def test_publish_grasp_target_repeats_target_messages_for_new_action_node(self):
        module = load_master_node_module()
        module.time.sleep = lambda seconds: None
        node = module.MasterNode.__new__(module.MasterNode)
        node.object_type_pub = FakePublisher()
        node.grasp_pose_pub = FakePublisher()
        node.object_type_topic = module.OBJECT_TYPE_TOPIC
        node.launch_action = lambda executable: None
        node.wait_for_subscribers = lambda publisher, topic_name: True
        node.get_clock = lambda: types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(to_msg=lambda: 'stamp')
        )
        node.publish_status = lambda status: None
        node.clear_pending = lambda: None
        node.target_text = {module.BOTTLE: '瓶子', module.FRUIT: '水果'}
        node.debug_grasp_mode = False

        node.publish_grasp_target(module.FRUIT, {
            'position': {'x': 0.55, 'y': 0.01, 'z': 0.02},
            'position_label': 'right',
            'confidence': 0.9,
        })

        self.assertGreaterEqual(len(node.object_type_pub.messages), 3)
        self.assertGreaterEqual(len(node.grasp_pose_pub.messages), 3)


if __name__ == '__main__':
    unittest.main()
