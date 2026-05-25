import json
import os
import threading
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
from interface.arm_ik import (
    DEFAULT_BOTTLE_DIAMETER,
    DEFAULT_TABLE_Z,
    FRUIT_GRASP_HEIGHT,
    PiperArmIK,
    bottle_approach_angle_offset_for_y,
)
from interface.arm_controller import LeapArm
from interface.hand_controller import LeapHand
from interface.motion_utils import command_scaled_arm_motion, refresh_current_arm, refresh_current_joint_states, scale_action


BOTTLE = "bottle"
FRUIT = "fruit"
OBJECT_TYPE_TOPIC = "/grasp_object_type"
GRASP_SELECTION_TOPIC = "/grasp_selection"
GRASP_TARGET_TOPIC = "/grasp_target_pose"
YOLO_DETECTIONS_TOPIC = "/yolo_3d_detections_json"
BOTTLE_GRASP_HEIGHT = 0.14
GRASP_FOLLOW_JOINTS = np.array([0.0, 1.57, -1.57, 0.0, 0.0, 0.0], dtype=float)
FOLLOW_MOVE_DURATION = 2.5
FOLLOW_MOVE_STEPS = 15
HAND_OPEN_TARGET_JOINTS = np.zeros(16, dtype=float)
HAND_OPEN_SPEED_MULTIPLIER = 2.0
DEFAULT_GRASP_POSITION = "right"
HAND_CURRENT_LIMIT_MA = 350.0
HAND_CURRENT_CONFIRM_DELAY_SEC = 2.0
HAND_CURRENT_POLL_INTERVAL_SEC = 0.05
ARM_SPEEDUP_FACTOR = 2.5
BOTTLE_GRASP_HAND_SPEEDUP_FACTOR = 2.0
FRUIT_GRASP_HAND_SPEEDUP_FACTOR = 1.5
STAGE_DELAY_SEC = 0.2


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
        "chair": FRUIT,
        "tv": FRUIT,
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


class MotorOvercurrentError(RuntimeError):
    pass


class HandCurrentSafetyMonitor:
    def __init__(
        self,
        leap,
        arm,
        logger,
        limit_ma=HAND_CURRENT_LIMIT_MA,
        confirm_delay_sec=HAND_CURRENT_CONFIRM_DELAY_SEC,
        poll_interval_sec=HAND_CURRENT_POLL_INTERVAL_SEC,
        joint_names=None,
    ):
        self.leap = leap
        self.arm = arm
        self.logger = logger
        self.limit_ma = float(limit_ma)
        self.confirm_delay_sec = float(confirm_delay_sec)
        self.poll_interval_sec = max(0.01, float(poll_interval_sec))
        self.joint_names = tuple(joint_names) if joint_names is not None else None

    def ros_ok(self):
        try:
            return rclpy.ok()
        except Exception:
            return True

    def spin_nodes(self, timeout_sec=None):
        timeout = self.poll_interval_sec if timeout_sec is None else max(0.0, float(timeout_sec))
        if self.leap is not None and hasattr(self.leap, "handle"):
            rclpy.spin_once(self.leap, timeout_sec=timeout)
        if self.arm is not None and hasattr(self.arm, "handle"):
            rclpy.spin_once(self.arm, timeout_sec=0.0)

    def _over_limit_currents(self):
        if self.leap is None:
            return {}
        return self.leap.get_over_limit_currents(self.limit_ma, joint_names=self.joint_names)

    def hold_position(self):
        if self.leap is not None:
            self.leap.hold_current_position()
        if self.arm is not None:
            self.arm.hold_current_position()

    def _wait_for_confirmation(self):
        deadline = time.time() + self.confirm_delay_sec
        while time.time() < deadline and self.ros_ok():
            self.spin_nodes()
            remaining = deadline - time.time()
            if remaining <= 0.0:
                break
            time.sleep(min(self.poll_interval_sec, remaining))

    def check(self):
        self.spin_nodes(timeout_sec=0.0)
        first_hit = self._over_limit_currents()
        if not first_hit:
            return

        self.logger.warn(
            f"Detected hand motor current over limit {self.limit_ma:.1f} mA: {first_hit}. "
            f"Waiting {self.confirm_delay_sec:.1f}s before recheck."
        )
        self._wait_for_confirmation()
        self.spin_nodes(timeout_sec=0.0)
        second_hit = self._over_limit_currents()
        if not second_hit:
            self.logger.info("Hand motor current recovered below limit after confirmation wait.")
            return

        self.hold_position()
        raise MotorOvercurrentError(
            f"Hand motor current remained over {self.limit_ma:.1f} mA after {self.confirm_delay_sec:.1f}s: {second_hit}"
        )

    def sleep_with_checks(self, duration_sec):
        deadline = time.time() + max(0.0, float(duration_sec))
        while time.time() < deadline and self.ros_ok():
            self.check()
            remaining = deadline - time.time()
            if remaining <= 0.0:
                break
            time.sleep(min(self.poll_interval_sec, remaining))


class ObjectGraspNode(Node):
    def __init__(self, config):
        super().__init__("action6_grasp_node")
        self.config = config
        self.object_type = None
        self.position = None
        self.position_label = DEFAULT_GRASP_POSITION
        self.awaiting_pose_target = True
        self.task_running = False
        self.declare_parameter("table_z", DEFAULT_TABLE_Z)
        self.declare_parameter("target_frame", "base")
        self.declare_parameter("detection_topic", YOLO_DETECTIONS_TOPIC)
        self.declare_parameter("max_inference_steps", 600)
        self.table_z = float(self.get_parameter("table_z").value)
        self.target_frame = str(self.get_parameter("target_frame").value).strip().lower()
        self.detection_topic = str(self.get_parameter("detection_topic").value).strip()
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
        self.grasp_selection_subscription = self.create_subscription(
            String,
            GRASP_SELECTION_TOPIC,
            self.grasp_selection_callback,
            10,
        )
        self.target_subscription = self.create_subscription(
            PoseStamped,
            GRASP_TARGET_TOPIC,
            self.grasp_target_callback,
            10,
        )
        self.yolo_subscription = self.create_subscription(
            String,
            self.detection_topic,
            self.yolo_detections_callback,
            10,
        )

        self.get_logger().info(
            f"action6_gsp is ready. Publish bottle/fruit to {OBJECT_TYPE_TOPIC}, "
            f"publish grasp selection to {GRASP_SELECTION_TOPIC}, then publish target pose to "
            f"{GRASP_TARGET_TOPIC}. Bottle/fruit IK also listens to preview center points from "
            f"{self.detection_topic}. target_frame={self.target_frame}, "
            f"max_inference_steps={self.max_inference_steps}."
        )

    def object_type_callback(self, msg):
        try:
            self.object_type = normalize_object_type(msg.data)
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return

        self.get_logger().info(f"Selected grasp object type: {self.object_type}")

    def grasp_selection_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Invalid grasp selection JSON message: {exc}")
            return

        if not isinstance(payload, dict):
            self.get_logger().warn("Grasp selection message is not a JSON object.")
            return

        object_type = payload.get("object_type")
        if object_type is not None:
            try:
                self.object_type = normalize_object_type(object_type)
            except ValueError as exc:
                self.get_logger().warn(str(exc))
                return

        self.position_label = self.resolve_position_label(payload.get("position_label"))
        self.position = None
        self.awaiting_pose_target = True
        self.get_logger().info(
            f"Selected grasp target: object_type={self.object_type}, position_label={self.position_label}"
        )

    def grasp_target_callback(self, msg):
        if self.task_running:
            self.get_logger().warn("A grasp task is running; ignored new target.")
            return

        frame_id = msg.header.frame_id.strip()
        object_type_from_header = None
        position_label_from_header = None
        if frame_id:
            if ":" in frame_id:
                object_type_from_header, position_label_from_header = frame_id.split(":", 1)
            else:
                object_type_from_header = frame_id

            try:
                self.object_type = normalize_object_type(object_type_from_header)
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
        if position_label_from_header is not None:
            self.position_label = self.resolve_position_label(position_label_from_header)
        self.awaiting_pose_target = False
        if self.object_type is None:
            self.get_logger().warn(
                f"Received grasp target but no object type. Publish bottle/fruit to {OBJECT_TYPE_TOPIC}."
            )

    def yolo_detections_callback(self, msg):
        if self.task_running:
            return

        if self.object_type is None or not self.awaiting_pose_target:
            return

        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Invalid YOLO JSON message: {exc}")
            return

        if not isinstance(detections, list):
            self.get_logger().warn("YOLO JSON message is not a detection list.")
            return

        selected_detection = self.select_detection_from_yolo(detections)
        if selected_detection is None:
            return

        center_3d = selected_detection["center_3d"]
        target_center = np.array(
            [center_3d["x"], center_3d["y"], center_3d["z"]],
            dtype=float,
        )
        self.get_logger().info(
            f"Previewed {self.object_type} center from YOLO JSON: "
            f"position_label={self.position_label}, "
            f"center={np.round(target_center, 4)}, "
            f"frame={selected_detection.get('center_3d_frame', 'unknown')}"
        )

    def select_detection_from_yolo(self, detections):
        candidates = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue

            try:
                object_type = normalize_object_type(
                    detection.get("object_type") or detection.get("class_name") or ""
                )
            except ValueError:
                continue

            if object_type != self.object_type:
                continue

            center_3d = detection.get("center_3d")
            if not isinstance(center_3d, dict):
                continue

            y_value = self.extract_axis_value(center_3d, "y")
            if y_value is None:
                continue

            confidence = detection.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0

            candidates.append({
                "detection": detection,
                "y": y_value,
                "confidence": confidence,
            })

        if not candidates:
            return None

        if self.position_label == "left":
            selected = max(candidates, key=lambda item: (item["y"], item["confidence"]))
        else:
            selected = min(candidates, key=lambda item: (item["y"], -item["confidence"]))
        return selected["detection"]

    def resolve_position_label(self, position_label):
        return "left" if str(position_label).strip().lower() == "left" else DEFAULT_GRASP_POSITION

    def extract_axis_value(self, mapping, axis):
        value = mapping.get(axis)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def execute_grasp_task(self, object_type, position):
        object_config = select_grasp_config(object_type)
        self.task_running = True

        try:
            if object_config.name == BOTTLE:
                input_position = np.asarray(position, dtype=float)
                angle_offset_deg = np.degrees(bottle_approach_angle_offset_for_y(input_position[1]))
                self.get_logger().info(
                    f"{object_config.name} target base position {input_position}, "
                    f"using bottle IK angle_offset={angle_offset_deg:.1f} deg"
                )
                ik_result = self.ik_solver.solve_bottle_grasp(
                    input_position,
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

    def open_hand_during_arm_motion(
        self,
        leap,
        arm,
        arm_start,
        arm_target,
        arm_steps,
        arm_duration,
        hand_speedup_factor,
        arm_extra_wait=0.0,
        safety_monitor=None,
    ):
        current_hand, _ = refresh_current_joint_states(
            leap=leap,
            need_hand=True,
        )
        hand_target = HAND_OPEN_TARGET_JOINTS.reshape(1, leap.joints_num)
        hand_steps = max(1, int(round(arm_steps / HAND_OPEN_SPEED_MULTIPLIER)))
        hand_duration = (
            float(arm_duration)
            * ARM_SPEEDUP_FACTOR
            / (float(hand_speedup_factor) * HAND_OPEN_SPEED_MULTIPLIER)
        )
        hand_path = scale_action(current_hand, hand_target, hand_steps)
        hand_point_duration = hand_duration / hand_steps

        results = {}

        def command_arm():
            try:
                command_scaled_arm_motion(
                    arm,
                    arm_start,
                    arm_target,
                    arm_steps,
                    arm_duration,
                    extra_wait=0.0,
                )
                results["arm"] = True
            except Exception as exc:
                results["arm_error"] = exc

        def command_hand():
            try:
                results["hand"] = leap.command_joint_position(hand_path, hand_point_duration)
            except Exception as exc:
                results["hand_error"] = exc

        arm_thread = threading.Thread(target=command_arm)
        hand_thread = threading.Thread(target=command_hand)
        arm_thread.start()
        hand_thread.start()

        while arm_thread.is_alive() or hand_thread.is_alive():
            if safety_monitor is not None:
                safety_monitor.check()
            arm_thread.join(timeout=HAND_CURRENT_POLL_INTERVAL_SEC)
            hand_thread.join(timeout=HAND_CURRENT_POLL_INTERVAL_SEC)

        if arm_extra_wait > 0.0:
            if safety_monitor is not None:
                safety_monitor.sleep_with_checks(arm_extra_wait)
            else:
                time.sleep(arm_extra_wait)

        if "arm_error" in results:
            raise results["arm_error"]
        if "hand_error" in results:
            raise results["hand_error"]
        if not results.get("hand"):
            raise RuntimeError("Failed to publish hand opening trajectory")

    def command_arm_motion_with_monitor(
        self,
        arm,
        start,
        target,
        steps,
        duration,
        extra_wait=0.0,
        safety_monitor=None,
    ):
        result = {}

        def command_arm():
            try:
                command_scaled_arm_motion(
                    arm,
                    start,
                    target,
                    steps,
                    duration,
                    extra_wait=0.0,
                )
                result["ok"] = True
            except Exception as exc:
                result["error"] = exc

        arm_thread = threading.Thread(target=command_arm)
        arm_thread.start()
        while arm_thread.is_alive():
            if safety_monitor is not None:
                safety_monitor.check()
            arm_thread.join(timeout=HAND_CURRENT_POLL_INTERVAL_SEC)

        if extra_wait > 0.0:
            if safety_monitor is not None:
                safety_monitor.sleep_with_checks(extra_wait)
            else:
                time.sleep(extra_wait)

        if "error" in result:
            raise result["error"]

    def move_arm_and_grasp(self, target_arm, object_config):
        leap = LeapHand("action6_hand")
        arm = LeapArm("action6_arm")
        safety_monitor = HandCurrentSafetyMonitor(
            leap=leap,
            arm=arm,
            logger=self.get_logger(),
        )
        try:
            _, current_arm = refresh_current_joint_states(
                leap=leap,
                arm=arm,
                need_hand=True,
                need_arm=True,
            )
            target_arm = as_row(target_arm, 6)
            safety_monitor.check()

            if object_config.name == BOTTLE:
                first_target = current_arm.copy()
                first_target[:, 0:1] = target_arm[:, 0:1]
                self.get_logger().info("Opening hand to zero while arm starts moving toward grasp target.")
                self.open_hand_during_arm_motion(
                    leap,
                    arm,
                    current_arm,
                    first_target,
                    arm_steps=5,
                    arm_duration=1.5 / ARM_SPEEDUP_FACTOR,
                    hand_speedup_factor=BOTTLE_GRASP_HAND_SPEEDUP_FACTOR,
                    arm_extra_wait=STAGE_DELAY_SEC,
                    safety_monitor=safety_monitor,
                )

                current_arm = refresh_current_arm(arm)
                self.command_arm_motion_with_monitor(
                    arm,
                    current_arm,
                    target_arm,
                    15,
                    4.5 / ARM_SPEEDUP_FACTOR,
                    extra_wait=STAGE_DELAY_SEC,
                    safety_monitor=safety_monitor,
                )
            else:
                self.get_logger().info("Opening hand to zero while arm moves toward grasp target.")
                self.open_hand_during_arm_motion(
                    leap,
                    arm,
                    current_arm,
                    target_arm,
                    arm_steps=15,
                    arm_duration=6.0 / ARM_SPEEDUP_FACTOR,
                    hand_speedup_factor=FRUIT_GRASP_HAND_SPEEDUP_FACTOR,
                    arm_extra_wait=STAGE_DELAY_SEC,
                    safety_monitor=safety_monitor,
                )

            safety_monitor.check()
            object_config.grasp_hand_action(leap, safety_check=safety_monitor.check)
            safety_monitor.sleep_with_checks(STAGE_DELAY_SEC)

            current_arm = refresh_current_arm(arm)
            follow_target = GRASP_FOLLOW_JOINTS.reshape(1, 6).copy()
            follow_target[:, 5:6] = target_arm[:, 5:6]
            self.get_logger().info(
                "Moving from current arm state to grasp follow joints: "
                f"start={np.round(current_arm.reshape(6), 4)}, "
                f"target={np.round(follow_target.reshape(6), 4)}, "
                f"duration={FOLLOW_MOVE_DURATION:.2f}s"
            )
            self.command_arm_motion_with_monitor(
                arm,
                current_arm,
                follow_target,
                FOLLOW_MOVE_STEPS,
                FOLLOW_MOVE_DURATION / ARM_SPEEDUP_FACTOR,
                extra_wait=STAGE_DELAY_SEC,
                safety_monitor=safety_monitor,
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
