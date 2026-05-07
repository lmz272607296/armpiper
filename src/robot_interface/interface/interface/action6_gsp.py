import os
import time
from dataclasses import dataclass

import hydra
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String

try:
    from ai_msgs.msg import PerceptionTargets
except ImportError:  # ai_msgs is only needed when consuming YOLO detections directly.
    PerceptionTargets = None

from interface.action_utils import (
    GraspInferenceAgent,
    as_row,
    camera_to_robot_transform,
    execute_bottle_grasp_hand,
    execute_bottle_release,
    execute_fruit_grasp_hand,
    execute_fruit_release,
    load_hand_eye_matrix,
)
from interface.arm_ik import DEFAULT_BOTTLE_DIAMETER, DEFAULT_TABLE_Z, FRUIT_GRASP_HEIGHT, PiperArmIK
from interface.arm_controller import LeapArm
from interface.hand_controller import LeapHand
from interface.motion_utils import command_scaled_arm_motion, refresh_current_arm, refresh_current_joint_states


BOTTLE = "bottle"
FRUIT = "fruit"
OBJECT_TYPE_TOPIC = "/grasp_object_type"
GRASP_TARGET_TOPIC = "/grasp_target_pose"
YOLO_DETECTIONS_TOPIC = "/yolo_3d_detections"
BOTTLE_GRASP_HEIGHT = 0.15
GRASP_FOLLOW_JOINTS = np.array([0.0, 1.57, -1.57, 0.0, 0.0, 0.0], dtype=float)
FOLLOW_MOVE_DURATION = 2.5
FOLLOW_MOVE_STEPS = 15


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
    FRUIT: GraspObjectConfig(
        name=FRUIT,
        checkpoint_path=os.path.join(_BASE_DIR, "runs", "LeapHandCatch.pth"),
        dof_limit_type="default",
        position_bias=np.array([0.0, 0.0, 0.1], dtype=float),
        control_bias=np.array([0.03, -0.05, 0.0, 0.12, 0.0], dtype=float),
        grasp_hand_action=execute_fruit_grasp_hand,
        release_action=execute_fruit_release,
    ),
}


def normalize_object_type(object_type):
    normalized = str(object_type).strip().lower()
    normalized = {
        "furit": FRUIT,
        "cube": FRUIT,
        "orange": FRUIT,
        "apple": FRUIT,
    }.get(normalized, normalized)
    if normalized not in GRASP_OBJECT_CONFIGS:
        raise ValueError(
            f"Unsupported object type '{object_type}'. Expected one of: {', '.join(GRASP_OBJECT_CONFIGS)}"
        )
    return normalized


def select_grasp_config(object_type):
    return GRASP_OBJECT_CONFIGS[normalize_object_type(object_type)]


def _ros_point_to_array(point):
    return np.array([point.x, point.y, point.z], dtype=float)


def _target_point_to_array(target_point):
    points = getattr(target_point, "point", None)
    if not points:
        return None
    return _ros_point_to_array(points[0])


def _target_type(target):
    rois = getattr(target, "rois", None)
    if not rois:
        return ""
    try:
        return normalize_object_type(getattr(rois[0], "type", ""))
    except ValueError:
        return ""


def _extract_bottle_center_from_yolo_target(target):
    center = None
    for target_point in getattr(target, "points", []):
        label = str(getattr(target_point, "type", "")).strip().lower()
        point = _target_point_to_array(target_point)
        if point is None:
            continue
        if "center" in label:
            center = point
            break

    return center


class ObjectGraspNode(Node):
    def __init__(self, config):
        super().__init__("action6_grasp_node")
        self.config = config
        self.object_type = None
        self.position = None
        self.task_running = False
        self.declare_parameter("table_z", DEFAULT_TABLE_Z)
        self.declare_parameter("target_frame", "base")
        self.declare_parameter("max_inference_steps", 600)
        self.table_z = float(self.get_parameter("table_z").value)
        self.target_frame = str(self.get_parameter("target_frame").value).strip().lower()
        self.max_inference_steps = int(self.get_parameter("max_inference_steps").value)
        self.hand_eye_matrix = load_hand_eye_matrix(self.get_logger())
        self.ik_solver = PiperArmIK()
        self.fruit_ik_solver = PiperArmIK(joint6_locked_value=0.0)

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
        self.yolo_subscription = None
        if PerceptionTargets is not None:
            self.yolo_subscription = self.create_subscription(
                PerceptionTargets,
                YOLO_DETECTIONS_TOPIC,
                self.yolo_detections_callback,
                10,
            )

        self.get_logger().info(
            f"action6_gsp is ready. Publish bottle/fruit to {OBJECT_TYPE_TOPIC}, "
            f"then publish target pose to {GRASP_TARGET_TOPIC}. Bottle IK can also consume "
            f"center points from {YOLO_DETECTIONS_TOPIC}. target_frame={self.target_frame}, "
            f"max_inference_steps={self.max_inference_steps}."
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
                f"Received grasp target but no object type. Publish bottle/fruit to {OBJECT_TYPE_TOPIC}."
            )

    def yolo_detections_callback(self, msg):
        if self.task_running:
            return

        for target in getattr(msg, "targets", []):
            if _target_type(target) != BOTTLE:
                continue

            bottle_center = _extract_bottle_center_from_yolo_target(target)
            if bottle_center is None:
                continue

            self.object_type = BOTTLE
            self.position = bottle_center.copy()
            self.get_logger().info(
                "Received bottle center from YOLO: "
                f"center={np.round(bottle_center, 4)}, diameter={DEFAULT_BOTTLE_DIAMETER:.4f} m"
            )
            return

    def execute_grasp_task(self, object_type, position):
        object_config = select_grasp_config(object_type)
        self.task_running = True

        try:
            if object_config.name == BOTTLE:
                ik_result = self.ik_solver.solve_bottle_grasp(
                    position,
                    table_z=self.table_z,
                    grasp_height=BOTTLE_GRASP_HEIGHT,
                )
                tilt_deg = np.degrees(ik_result.palm_axis_error) if ik_result.palm_axis_error is not None else 0.0
                self.get_logger().info(
                    f"{object_config.name} IK target={np.round(ik_result.target_position, 4)}, "
                    f"achieved={np.round(ik_result.achieved_position, 4)}, "
                    f"position_error={ik_result.position_error:.5f} m, palm_y_error={tilt_deg:.3f} deg"
                )
                self.get_logger().info(f"{object_config.name} IK arm command: {ik_result.joint_positions}")
                if not ik_result.success:
                    raise RuntimeError(
                        f"Bottle IK did not satisfy tolerance: position_error={ik_result.position_error:.5f} m, "
                        f"palm_y_error={tilt_deg:.3f} deg"
                    )

                self.move_arm_and_grasp(ik_result.joint_positions.reshape(1, 6), object_config)
                self.get_logger().info(f"action6_gsp {object_config.name} grasp completed.")
                return

            if object_config.name == FRUIT:
                input_position = np.asarray(position, dtype=float)
                if self.target_frame == "camera":
                    robot_position = camera_to_robot_transform(input_position, self.hand_eye_matrix)
                    self.get_logger().info(
                        f"{object_config.name} target camera position {input_position}, "
                        f"robot position {robot_position}"
                    )
                else:
                    robot_position = input_position.copy()
                    self.get_logger().info(
                        f"{object_config.name} target base position {robot_position}"
                    )

                ik_result = self.fruit_ik_solver.solve_fruit_grasp(
                    robot_position,
                    table_z=self.table_z,
                    grasp_height=FRUIT_GRASP_HEIGHT,
                )
                tilt_deg = np.degrees(ik_result.palm_axis_error) if ik_result.palm_axis_error is not None else 0.0
                self.get_logger().info(
                    f"{object_config.name} IK target={np.round(ik_result.target_position, 4)}, "
                    f"achieved={np.round(ik_result.achieved_position, 4)}, "
                    f"position_error={ik_result.position_error:.5f} m, palm_z_error={tilt_deg:.3f} deg"
                )
                self.get_logger().info(f"{object_config.name} IK arm command: {ik_result.joint_positions}")
                if not ik_result.success:
                    raise RuntimeError(
                        f"Fruit IK did not satisfy tolerance: position_error={ik_result.position_error:.5f} m, "
                        f"palm_z_error={tilt_deg:.3f} deg"
                    )

                self.move_arm_and_grasp(ik_result.joint_positions.reshape(1, 6), object_config)
                self.get_logger().info(f"action6_gsp {object_config.name} grasp completed.")
                return

            input_position = np.asarray(position, dtype=float)
            if self.target_frame == "camera":
                camera_position = input_position + object_config.position_bias
                robot_position = camera_to_robot_transform(camera_position, self.hand_eye_matrix)
                self.get_logger().info(
                    f"{object_config.name} target camera position {camera_position}, "
                    f"robot position {robot_position}"
                )
            else:
                robot_position = input_position.copy()
                self.get_logger().info(
                    f"{object_config.name} target base position {robot_position}"
                )

            inference_start_time = time.time()
            self.get_logger().info(
                f"Starting {object_config.name} grasp inference with max_inference_steps="
                f"{self.max_inference_steps}"
            )

            agent = GraspInferenceAgent(
                config=self.config,
                checkpoint_path=object_config.checkpoint_path,
                dof_limit_type=object_config.dof_limit_type,
                max_inference_steps=self.max_inference_steps,
            )
            stable = agent.infer(robot_position) + object_config.control_bias
            target_arm = agent.reorder(stable.reshape(1, 5))
            self.get_logger().info(
                f"{object_config.name} inference finished in "
                f"{time.time() - inference_start_time:.3f}s, arm command: {target_arm}"
            )

            self.move_arm_and_grasp(target_arm, object_config)
            self.get_logger().info(f"action6_gsp {object_config.name} grasp completed.")
        except Exception as exc:
            self.get_logger().error(f"action6_gsp {object_config.name} grasp failed: {exc}")
            import traceback

            self.get_logger().error(traceback.format_exc())
        finally:
            self.task_running = False

    def move_arm_and_grasp(self, target_arm, object_config):
        leap = LeapHand("action6_hand")
        arm = LeapArm("action6_arm")
        try:
            _, current_arm = refresh_current_joint_states(
                leap=leap,
                arm=arm,
                need_hand=True,
                need_arm=True,
            )
            target_arm = as_row(target_arm, 6)

            if object_config.name == BOTTLE:
                first_target = current_arm.copy()
                first_target[:, 0:1] = target_arm[:, 0:1]
                command_scaled_arm_motion(arm, current_arm, first_target, 5, 1.5, extra_wait=1.0)

                current_arm = refresh_current_arm(arm)
                command_scaled_arm_motion(arm, current_arm, target_arm, 15, 4.5, extra_wait=2.0)
            else:
                command_scaled_arm_motion(arm, current_arm, target_arm, 15, 6.0, extra_wait=1.0)

            object_config.grasp_hand_action(leap)

            current_arm = refresh_current_arm(arm)
            follow_target = GRASP_FOLLOW_JOINTS.reshape(1, 6).copy()
            follow_target[:, 5:6] = target_arm[:, 5:6]
            self.get_logger().info(
                "Moving from current arm state to grasp follow joints: "
                f"start={np.round(current_arm.reshape(6), 4)}, "
                f"target={np.round(follow_target.reshape(6), 4)}, "
                f"duration={FOLLOW_MOVE_DURATION:.2f}s"
            )
            command_scaled_arm_motion(
                arm,
                current_arm,
                follow_target,
                FOLLOW_MOVE_STEPS,
                FOLLOW_MOVE_DURATION,
                extra_wait=0.5,
            )
        finally:
            leap.destroy_node()
            arm.destroy_node()


@hydra.main(config_name="config", config_path=os.path.join(os.path.dirname(__file__), "cfg"))
def main(config):
    rclpy.init()
    node = ObjectGraspNode(config)
    try:
        while (node.position is None or node.object_type is None) and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
        if rclpy.ok():
            node.execute_grasp_task(node.object_type, node.position.copy())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
