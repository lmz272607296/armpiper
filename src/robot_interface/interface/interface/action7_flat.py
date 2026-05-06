import os
import time

import hydra
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from interface.action6_gsp import BOTTLE, FRUIT, OBJECT_TYPE_TOPIC, normalize_object_type, select_grasp_config
from interface.action_utils import as_row, scale_action, wait_for_joint_state
from interface.arm_ik import BOTTLE_GRASP_HEIGHT, DEFAULT_TABLE_Z, FRUIT_GRASP_HEIGHT, PiperArmIK
from interface.arm_controller import LeapArm
from interface.hand_controller import LeapHand


class ObjectReleaseNode(Node):
    def __init__(self, config):
        super().__init__("action7_release_node")
        self.config = config
        self.object_type = None
        self.task_running = False
        self.declare_parameter("table_z", DEFAULT_TABLE_Z)
        self.table_z = float(self.get_parameter("table_z").value)
        self.ik_solver = PiperArmIK()
        self.subscription = self.create_subscription(
            String,
            OBJECT_TYPE_TOPIC,
            self.object_type_callback,
            10,
        )
        self.get_logger().info(
            f"action7_flat is ready. Publish bottle/fruit to {OBJECT_TYPE_TOPIC} to choose the release motion."
        )

    def object_type_callback(self, msg):
        if self.task_running:
            self.get_logger().warn("A release task is running; ignored new object type.")
            return

        try:
            self.object_type = normalize_object_type(msg.data)
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return

        self.get_logger().info(f"Selected release object type: {self.object_type}")

    def execute_release_task(self, object_type):
        object_type = normalize_object_type(object_type)
        object_config = select_grasp_config(object_type)
        leap = LeapHand("action7_hand")
        arm = LeapArm("action7_arm")
        self.task_running = True

        try:
            self.move_down_and_release(leap, arm, object_config)
            self.get_logger().info(f"action7_flat {object_type} release completed.")
        except Exception as exc:
            self.get_logger().error(f"action7_flat {object_type} release failed: {exc}")
            import traceback

            self.get_logger().error(traceback.format_exc())
        finally:
            leap.destroy_node()
            arm.destroy_node()
            self.task_running = False

    def move_down_and_release(self, leap, arm, object_config):
        object_type = object_config.name
        wait_for_joint_state(leap=leap, arm=arm, need_hand=True, need_arm=True)

        current_arm = as_row(arm.raw_positions, 6)
        release_height = self.default_release_height(object_type)
        ik_result = self.ik_solver.solve_release(
            current_arm.reshape(6),
            release_height=release_height,
            table_z=self.table_z,
        )
        self.get_logger().info(
            f"{object_type} release IK target={np.round(ik_result.target_position, 4)}, "
            f"achieved={np.round(ik_result.achieved_position, 4)}, "
            f"position_error={ik_result.position_error:.5f} m, "
            f"release_height={release_height:.3f} m"
        )
        self.get_logger().info(f"{object_type} release arm command: {ik_result.joint_positions}")
        if not ik_result.success:
            raise RuntimeError(
                f"Release IK did not satisfy tolerance: position_error={ik_result.position_error:.5f} m"
            )

        release_path = scale_action(current_arm, ik_result.joint_positions.reshape(1, 6), 12)
        arm.command_joint_position(release_path, 0.3)
        time.sleep(12 * 0.3 + 1.0)

        object_config.release_action(leap, arm)

    def default_release_height(self, object_type):
        if object_type == BOTTLE:
            return BOTTLE_GRASP_HEIGHT
        if object_type == FRUIT:
            return FRUIT_GRASP_HEIGHT
        raise ValueError(f"Unsupported release object type '{object_type}'")


@hydra.main(config_name="config", config_path=os.path.join(os.path.dirname(__file__), "cfg"))
def main(config):
    rclpy.init()
    node = ObjectReleaseNode(config)
    try:
        while rclpy.ok():
            while node.object_type is None and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
            if not rclpy.ok():
                break

            object_type = node.object_type
            node.object_type = None
            node.execute_release_task(object_type)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
