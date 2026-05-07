import os
import time

import hydra
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from interface.action6_gsp import BOTTLE, FRUIT, OBJECT_TYPE_TOPIC, normalize_object_type, select_grasp_config
from interface.arm_ik import BOTTLE_GRASP_HEIGHT, DEFAULT_TABLE_Z, FRUIT_RELEASE_HEIGHT, PiperArmIK
from interface.arm_controller import LeapArm
from interface.hand_controller import LeapHand
from interface.motion_utils import command_scaled_arm_motion, refresh_current_joint_states, scale_action


ARM_RELEASE_RESET_JOINTS = np.zeros(6, dtype=float)


class ObjectReleaseNode(Node):
    def __init__(self, config):
        super().__init__("action7_release_node")
        self.config = config
        self.object_type = None
        self.task_running = False
        self.declare_parameter("table_z", DEFAULT_TABLE_Z)
        self.declare_parameter("release_move_steps", 12)
        self.declare_parameter("release_move_duration", 3.6)
        self.declare_parameter("release_move_extra_wait", 1.0)
        self.declare_parameter("release_settle_sec", 2.0)
        self.declare_parameter("hand_open_steps", 8)
        self.declare_parameter("hand_open_duration", 2.4)
        self.declare_parameter("hand_open_extra_wait", 0.6)
        self.declare_parameter("post_hand_open_pause_sec", 0.5)
        self.declare_parameter("arm_reset_steps", 14)
        self.declare_parameter("arm_reset_duration", 4.2)
        self.declare_parameter("arm_reset_extra_wait", 1.0)
        self.table_z = float(self.get_parameter("table_z").value)
        self.release_move_steps = int(self.get_parameter("release_move_steps").value)
        self.release_move_duration = float(self.get_parameter("release_move_duration").value)
        self.release_move_extra_wait = float(self.get_parameter("release_move_extra_wait").value)
        self.release_settle_sec = float(self.get_parameter("release_settle_sec").value)
        self.hand_open_steps = int(self.get_parameter("hand_open_steps").value)
        self.hand_open_duration = float(self.get_parameter("hand_open_duration").value)
        self.hand_open_extra_wait = float(self.get_parameter("hand_open_extra_wait").value)
        self.post_hand_open_pause_sec = float(self.get_parameter("post_hand_open_pause_sec").value)
        self.arm_reset_steps = int(self.get_parameter("arm_reset_steps").value)
        self.arm_reset_duration = float(self.get_parameter("arm_reset_duration").value)
        self.arm_reset_extra_wait = float(self.get_parameter("arm_reset_extra_wait").value)
        self.ik_solver = PiperArmIK()
        self.fruit_ik_solver = PiperArmIK(joint6_locked_value=0.0)
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
        _, current_arm = refresh_current_joint_states(
            leap=leap,
            arm=arm,
            need_hand=True,
            need_arm=True,
        )

        release_height = self.default_release_height(object_type)
        ik_result = self.solve_release_pose(object_type, current_arm.reshape(6), release_height)
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

        self.get_logger().info(f"[{object_type}] Phase 1/4: move arm to release pose.")
        command_scaled_arm_motion(
            arm,
            current_arm,
            ik_result.joint_positions.reshape(1, 6),
            steps=self.release_move_steps,
            duration=self.release_move_duration,
            extra_wait=self.release_move_extra_wait,
        )

        self.get_logger().info(
            f"[{object_type}] Phase 2/4: hold release pose for {self.release_settle_sec:.2f}s to let the object settle on the ground."
        )
        self.wait_between_phases(self.release_settle_sec, leap=leap, arm=arm)

        self.get_logger().info(f"[{object_type}] Phase 3/4: open hand back to origin at release pose.")
        self.open_hand_to_origin(leap)

        self.get_logger().info(f"[{object_type}] Phase 4/4: reset arm after hand is fully open.")
        self.wait_between_phases(self.post_hand_open_pause_sec, leap=leap, arm=arm)
        self.reset_arm_to_origin(arm)

    def solve_release_pose(self, object_type, current_arm, release_height):
        if object_type == BOTTLE:
            return self.ik_solver.solve_release(
                current_arm,
                release_height=release_height,
                table_z=self.table_z,
            )

        if object_type == FRUIT:
            return self.fruit_ik_solver.solve_fruit_release(
                current_arm,
                release_height=release_height,
                table_z=self.table_z,
            )

        raise ValueError(f"Unsupported release object type '{object_type}'")

    def wait_between_phases(self, duration, leap=None, arm=None):
        duration = float(duration)
        if duration <= 0.0:
            return

        deadline = time.monotonic() + duration
        while rclpy.ok():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return

            if leap is not None:
                rclpy.spin_once(leap, timeout_sec=0.0)
            if arm is not None:
                rclpy.spin_once(arm, timeout_sec=0.0)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(min(0.05, remaining))

    def open_hand_to_origin(self, leap):
        if self.hand_open_steps <= 0:
            raise ValueError("hand_open_steps must be positive")

        current_hand, _ = refresh_current_joint_states(
            leap=leap,
            need_hand=True,
        )
        hand_zero = np.zeros((1, leap.joints_num), dtype=float)
        hand_path = scale_action(current_hand, hand_zero, self.hand_open_steps)
        point_duration = self.hand_open_duration / self.hand_open_steps
        if point_duration <= 0.0:
            raise ValueError("hand_open_duration must be positive")

        if not leap.command_joint_position(hand_path, point_duration):
            raise RuntimeError("Failed to publish hand opening trajectory")

        self.wait_between_phases(
            len(hand_path) * point_duration + self.hand_open_extra_wait,
            leap=leap,
        )

    def reset_arm_to_origin(self, arm):
        if self.arm_reset_steps <= 0:
            raise ValueError("arm_reset_steps must be positive")

        _, current_arm = refresh_current_joint_states(
            arm=arm,
            need_arm=True,
        )
        arm_zero = ARM_RELEASE_RESET_JOINTS.reshape(1, arm.joints_num)
        command_scaled_arm_motion(
            arm,
            current_arm,
            arm_zero,
            steps=self.arm_reset_steps,
            duration=self.arm_reset_duration,
            extra_wait=self.arm_reset_extra_wait,
        )

    def default_release_height(self, object_type):
        if object_type == BOTTLE:
            return BOTTLE_GRASP_HEIGHT
        if object_type == FRUIT:
            return FRUIT_RELEASE_HEIGHT
        raise ValueError(f"Unsupported release object type '{object_type}'")


@hydra.main(config_name="config", config_path=os.path.join(os.path.dirname(__file__), "cfg"))
def main(config):
    rclpy.init()
    node = ObjectReleaseNode(config)
    try:
        while node.object_type is None and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
        if rclpy.ok():
            node.execute_release_task(node.object_type)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
