import os
import time
from dataclasses import dataclass

import hydra
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String

from interface.action_utils import (
    GraspInferenceAgent,
    as_row,
    camera_to_robot_transform,
    execute_bottle_grasp_hand,
    execute_bottle_release,
    execute_cube_grasp_hand,
    execute_cube_release,
    load_hand_eye_matrix,
    scale_action,
    wait_for_joint_state,
)
from interface.arm_controller import LeapArm
from interface.hand_controller import LeapHand


CUBE = "CUBE"
BOTTLE = "BOTTLE"
OBJECT_TYPE_TOPIC = "/grasp_object_type"
GRASP_TARGET_TOPIC = "/grasp_target_pose"


@dataclass(frozen=True)
class GraspObjectConfig:
    name: str
    checkpoint_path: str
    dof_limit_type: str
    position_bias: np.ndarray
    control_bias: np.ndarray
    grasp_hand_action: object
    release_action: object


_BASE_DIR = os.path.dirname(__file__)
GRASP_OBJECT_CONFIGS = {
    BOTTLE: GraspObjectConfig(
        name=BOTTLE,
        checkpoint_path=os.path.join(_BASE_DIR, "runs", "LeapHand_xie.pth"),
        dof_limit_type="xie",
        position_bias=np.array([0.1, 0.0, 0.1], dtype=float),
        control_bias=np.array([0.0, 0.0, -0.1, -0.1, 0.0], dtype=float),
        grasp_hand_action=execute_bottle_grasp_hand,
        release_action=execute_bottle_release,
    ),
    CUBE: GraspObjectConfig(
        name=CUBE,
        checkpoint_path=os.path.join(_BASE_DIR, "runs", "LeapHand.pth"),
        dof_limit_type="default",
        position_bias=np.array([0.0, 0.0, 0.1], dtype=float),
        control_bias=np.array([0.03, -0.05, 0.0, 0.12, 0.0], dtype=float),
        grasp_hand_action=execute_cube_grasp_hand,
        release_action=execute_cube_release,
    ),
}


def normalize_object_type(object_type):
    normalized = str(object_type).strip().upper()
    if normalized not in GRASP_OBJECT_CONFIGS:
        raise ValueError(
            f"Unsupported object type '{object_type}'. Expected one of: {', '.join(GRASP_OBJECT_CONFIGS)}"
        )
    return normalized


def select_grasp_config(object_type):
    return GRASP_OBJECT_CONFIGS[normalize_object_type(object_type)]


class ObjectGraspNode(Node):
    def __init__(self, config):
        super().__init__("action1_grasp_node")
        self.config = config
        self.object_type = None
        self.position = None
        self.task_running = False
        self.hand_eye_matrix = load_hand_eye_matrix(self.get_logger())

        self.object_subscription = self.create_subscription(
            String,
            OBJECT_TYPE_TOPIC,
            self.object_type_callback,
            10,
        )
        self.target_subscription = self.create_subscription(
            PoseStamped,
            GRASP_TARGET_TOPIC,
            self.grasp_target_callback,
            10,
        )

        self.get_logger().info(
            f"action1_gsp is ready. Publish CUBE/BOTTLE to {OBJECT_TYPE_TOPIC}, "
            f"then publish target pose to {GRASP_TARGET_TOPIC}."
        )

    def object_type_callback(self, msg):
        try:
            self.object_type = normalize_object_type(msg.data)
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return

        self.get_logger().info(f"Selected grasp object type: {self.object_type}")

    def grasp_target_callback(self, msg):
        if self.task_running:
            self.get_logger().warn("A grasp task is running; ignored new target.")
            return

        frame_id = msg.header.frame_id.strip()
        if frame_id:
            try:
                self.object_type = normalize_object_type(frame_id)
                self.get_logger().info(
                    f"Selected grasp object type from PoseStamped.header.frame_id: {self.object_type}"
                )
            except ValueError:
                self.get_logger().debug(
                    f"Ignoring PoseStamped.header.frame_id '{frame_id}' as object type."
                )

        self.position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=float,
        )
        if self.object_type is None:
            self.get_logger().warn(
                f"Received grasp target but no object type. Publish CUBE/BOTTLE to {OBJECT_TYPE_TOPIC}."
            )

    def execute_grasp_task(self, object_type, position):
        object_config = select_grasp_config(object_type)
        self.task_running = True

        try:
            camera_position = np.asarray(position, dtype=float) + object_config.position_bias
            robot_position = camera_to_robot_transform(camera_position, self.hand_eye_matrix)
            self.get_logger().info(
                f"{object_config.name} target camera position {camera_position}, "
                f"robot position {robot_position}"
            )

            agent = GraspInferenceAgent(
                config=self.config,
                checkpoint_path=object_config.checkpoint_path,
                dof_limit_type=object_config.dof_limit_type,
            )
            stable = agent.infer(robot_position) + object_config.control_bias
            target_arm = agent.reorder(stable.reshape(1, 5))
            self.get_logger().info(f"{object_config.name} arm command: {target_arm}")

            self.move_arm_and_grasp(target_arm, object_config)
            self.get_logger().info(f"action1_gsp {object_config.name} grasp completed.")
        except Exception as exc:
            self.get_logger().error(f"action1_gsp {object_config.name} grasp failed: {exc}")
            import traceback

            self.get_logger().error(traceback.format_exc())
        finally:
            self.task_running = False

    def move_arm_and_grasp(self, target_arm, object_config):
        leap = LeapHand("action1_hand")
        arm = LeapArm("action1_arm")
        try:
            wait_for_joint_state(leap=leap, arm=arm, need_hand=True, need_arm=True)
            target_arm = as_row(target_arm, 6)
            current_arm = as_row(arm.raw_positions, 6)

            if object_config.name == BOTTLE:
                first_target = current_arm.copy()
                first_target[:, 0:1] = target_arm[:, 0:1]
                first_path = scale_action(current_arm, first_target, 5)
                arm.command_joint_position(first_path, 0.3)
                time.sleep(5 * 0.3 + 1.0)

                wait_for_joint_state(arm=arm, need_arm=True)
                current_arm = as_row(arm.raw_positions, 6)
                full_path = scale_action(current_arm, target_arm, 15)
                arm.command_joint_position(full_path, 0.3)
                time.sleep(15 * 0.3 + 2.0)
            else:
                arm_path = scale_action(current_arm, target_arm, 15)
                arm.command_joint_position(arm_path, 0.4)
                time.sleep(15 * 0.4 + 1.0)

            object_config.grasp_hand_action(leap)
        finally:
            leap.destroy_node()
            arm.destroy_node()


@hydra.main(config_name="config", config_path=os.path.join(os.path.dirname(__file__), "cfg"))
def main(config):
    rclpy.init()
    node = ObjectGraspNode(config)
    try:
        while rclpy.ok():
            while (node.position is None or node.object_type is None) and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
            if not rclpy.ok():
                break

            object_type = node.object_type
            position = node.position.copy()
            node.position = None
            node.execute_grasp_task(object_type, position)
            node.object_type = None
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
