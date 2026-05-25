import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


BOTTLE_GRASP_HEIGHT = 0.15
FRUIT_GRASP_HEIGHT = 0.18
FRUIT_RELEASE_HEIGHT = 0.18
DEFAULT_TABLE_Z = 0.0
DEFAULT_BOTTLE_DIAMETER = 0.07
DEFAULT_BOTTLE_RADIUS = DEFAULT_BOTTLE_DIAMETER * 0.5
# Bottle grasp target x bias in meters.
# Tune this single value to move the solved bottle grasp target forward/backward.
# Negative values move the target backward (smaller x); positive values move it forward.
BOTTLE_TARGET_X_BIAS = -0.03
BOTTLE_TANGENT_Y_CLEARANCE = 0.0
BOTTLE_HOVER_Y_OFFSET = 0.0
# Bottle side-grasp angle offsets selected by bottle y position.
# The values are in radians; use math.radians(...) to tune in degrees.
#
# y < -0.1           -> BOTTLE_RIGHT_APPROACH_ANGLE_OFFSET         ( 8 deg)
# -0.1 <= y < -0.05  -> BOTTLE_RIGHT_MIDDLE_APPROACH_ANGLE_OFFSET  ( 9 deg)
# -0.05 <= y <= 0.1  -> BOTTLE_MIDDLE_APPROACH_ANGLE_OFFSET        (11 deg)
# 0.1 < y <= 0.2     -> BOTTLE_LEFT_APPROACH_ANGLE_OFFSET          ( 2 deg)
# y > 0.2            -> BOTTLE_LEFT_FAR_APPROACH_ANGLE_OFFSET      (-1 deg)
#
# BOTTLE_APPROACH_ANGLE_OFFSET is the legacy/default offset and is kept here
# for manual tuning or fallback use.
BOTTLE_APPROACH_ANGLE_OFFSET = math.radians(4.0)
BOTTLE_RIGHT_APPROACH_ANGLE_OFFSET = math.radians(8.0)
BOTTLE_RIGHT_MIDDLE_APPROACH_ANGLE_OFFSET = math.radians(9.0)
BOTTLE_MIDDLE_APPROACH_ANGLE_OFFSET = math.radians(6)
BOTTLE_LEFT_APPROACH_ANGLE_OFFSET = math.radians(2.0)
BOTTLE_LEFT_FAR_APPROACH_ANGLE_OFFSET = math.radians(-1.0)
PALM_LINK_NAME = "palm_lower"
BASE_LINK_NAME = "base_link"
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
POSITION_TOLERANCE = 0.02
PALM_TILT_TOLERANCE = math.radians(5.0)
WORLD_Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)


def default_urdf_path():
    env_path = os.environ.get("VLAARM_URDF_PATH")
    if env_path:
        return os.path.abspath(env_path)

    current_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(os.path.join(
            get_package_share_directory("interface"), "assets", "urdf", "vlaarm.urdf"
        ))
    except Exception:
        pass

    candidates.append(os.path.join(current_dir, "..", "..", "assets", "urdf", "vlaarm.urdf"))
    parent = current_dir
    for _ in range(8):
        candidates.append(os.path.join(parent, "src", "robot_interface", "assets", "urdf", "vlaarm.urdf"))
        parent = os.path.dirname(parent)

    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate

    return os.path.abspath(candidates[0])


@dataclass(frozen=True)
class BottleModel:
    center: np.ndarray
    radius: float = DEFAULT_BOTTLE_RADIUS
    endpoint_a: np.ndarray | None = None
    endpoint_b: np.ndarray | None = None


@dataclass(frozen=True)
class IkResult:
    joint_positions: np.ndarray
    target_position: np.ndarray
    achieved_position: np.ndarray
    position_error: float
    palm_axis_error: float | None
    success: bool
    score: float


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


def _parse_vector(text, default):
    if not text:
        return np.array(default, dtype=float)
    return np.array([float(value) for value in text.split()], dtype=float)


def _rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _axis_angle_matrix(axis, angle):
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def _homogeneous(rotation=None, translation=None):
    matrix = np.eye(4, dtype=float)
    if rotation is not None:
        matrix[:3, :3] = rotation
    if translation is not None:
        matrix[:3, 3] = translation
    return matrix


def _angle_between(unit_a, unit_b):
    dot = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
    return math.acos(dot)


def _as_point(point):
    array = np.asarray(point, dtype=float).reshape(-1)
    if array.size < 2:
        raise ValueError(f"Expected at least x/y coordinates, got {array}")
    if array.size == 2:
        array = np.array([array[0], array[1], DEFAULT_TABLE_Z], dtype=float)
    return array[:3]


def apply_bottle_target_bias(target_position, x_bias=BOTTLE_TARGET_X_BIAS):
    target_position = np.asarray(target_position, dtype=float).reshape(3).copy()
    target_position[0] += float(x_bias)
    return target_position


def make_bottle_model(center, endpoint_a=None, endpoint_b=None, default_radius=DEFAULT_BOTTLE_RADIUS):
    center = _as_point(center)
    endpoint_a = None if endpoint_a is None else _as_point(endpoint_a)
    endpoint_b = None if endpoint_b is None else _as_point(endpoint_b)
    radius = float(default_radius)

    if endpoint_a is not None and endpoint_b is not None:
        radius = 0.5 * float(np.linalg.norm(endpoint_a[:2] - endpoint_b[:2]))
    elif endpoint_a is not None:
        radius = float(np.linalg.norm(endpoint_a[:2] - center[:2]))

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"Bottle radius must be positive, got {radius}")

    return BottleModel(center=center, radius=radius, endpoint_a=endpoint_a, endpoint_b=endpoint_b)


def bottle_approach_angle_offset_for_y(y_coord):
    y_coord = float(y_coord)
    # Y-axis classification for bottle side grasp approach:
    # y > 0.2            -> left far
    # 0.1 < y <= 0.2     -> left
    # -0.05 <= y <= 0.1  -> middle
    # -0.1 <= y < -0.05  -> right middle
    # y < -0.1           -> right
    if y_coord > 0.2:
        return BOTTLE_LEFT_FAR_APPROACH_ANGLE_OFFSET
    if y_coord > 0.1:
        return BOTTLE_LEFT_APPROACH_ANGLE_OFFSET
    if -0.1 <= y_coord < -0.05:
        return BOTTLE_RIGHT_MIDDLE_APPROACH_ANGLE_OFFSET
    if y_coord > -0.1:
        return BOTTLE_MIDDLE_APPROACH_ANGLE_OFFSET
    return BOTTLE_RIGHT_APPROACH_ANGLE_OFFSET


def compute_bottle_grasp_position(
    bottle_model,
    table_z=DEFAULT_TABLE_Z,
    grasp_height=BOTTLE_GRASP_HEIGHT,
    angle_offset=None,
    x_bias=BOTTLE_TARGET_X_BIAS,
):
    center_xy = np.asarray(bottle_model.center[:2], dtype=float)
    radius = float(bottle_model.radius)
    distance = float(np.linalg.norm(center_xy))
    if distance <= radius:
        raise ValueError(
            f"Cannot draw tangents from base origin to bottle circle: distance={distance:.4f}, radius={radius:.4f}"
        )

    if angle_offset is None:
        angle_offset = bottle_approach_angle_offset_for_y(center_xy[1])

    tangent_angle = math.asin(radius / distance)
    approach_angle = math.atan2(center_xy[1], center_xy[0]) - tangent_angle - float(angle_offset)
    approach_x = distance * math.cos(approach_angle)
    approach_y = distance * math.sin(approach_angle)
    target = np.array(
        [approach_x, approach_y, float(table_z) + float(grasp_height)],
        dtype=float,
    )
    return apply_bottle_target_bias(target, x_bias=x_bias)


class PiperArmKinematics:
    def __init__(self, urdf_path=None, base_link=BASE_LINK_NAME, palm_link=PALM_LINK_NAME, joint6_locked_value=None):
        self.urdf_path = os.path.abspath(urdf_path or default_urdf_path())
        self.base_link = base_link
        self.palm_link = palm_link
        self.joint6_locked_value = joint6_locked_value
        self.joints = self._load_chain()
        self.command_joint_names = ARM_JOINT_NAMES
        self.command_index = {name: index for index, name in enumerate(self.command_joint_names)}
        self.lower_limits, self.upper_limits = self._command_limits()
        self.active_indices = self._active_command_indices()

    def _load_chain(self):
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        root = ET.parse(self.urdf_path).getroot()
        joints_by_child = {}
        for elem in root.findall("joint"):
            origin = elem.find("origin")
            axis = elem.find("axis")
            limit = elem.find("limit")
            parent = elem.find("parent")
            child = elem.find("child")
            if parent is None or child is None:
                continue

            lower = float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None
            upper = float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None
            joints_by_child[child.attrib["link"]] = _Joint(
                name=elem.attrib["name"],
                joint_type=elem.attrib.get("type", "fixed"),
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                xyz=_parse_vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                rpy=_parse_vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                axis=_parse_vector(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
                lower=lower,
                upper=upper,
            )

        chain = []
        link = self.palm_link
        while link != self.base_link:
            if link not in joints_by_child:
                raise ValueError(f"No joint connects link '{link}' back to '{self.base_link}'")
            joint = joints_by_child[link]
            chain.append(joint)
            link = joint.parent

        return tuple(reversed(chain))

    def _command_limits(self):
        lower = np.full(6, -math.pi, dtype=float)
        upper = np.full(6, math.pi, dtype=float)
        joint_map = {joint.name: joint for joint in self.joints}

        for index, name in enumerate(self.command_joint_names):
            joint = joint_map.get(name)
            if joint is None or joint.joint_type == "fixed":
                lower[index] = 0.0
                upper[index] = 0.0
                continue
            if joint.lower is not None and joint.upper is not None:
                lower[index] = min(joint.lower, joint.upper)
                upper[index] = max(joint.lower, joint.upper)

        # joint6 is mechanically locked in the current URDF and is treated as fixed for IK.
        joint6_index = self.command_index["joint6"]
        if self.joint6_locked_value is None:
            joint6_value = 0.5 * (lower[joint6_index] + upper[joint6_index])
        else:
            joint6_value = float(self.joint6_locked_value)
        lower[joint6_index] = joint6_value
        upper[joint6_index] = joint6_value
        return lower, upper

    def _active_command_indices(self):
        return np.array(
            [index for index, (low, high) in enumerate(zip(self.lower_limits, self.upper_limits)) if high - low > 1e-6],
            dtype=int,
        )

    def clamp(self, joint_positions):
        joint_positions = np.asarray(joint_positions, dtype=float).reshape(6)
        return np.clip(joint_positions, self.lower_limits, self.upper_limits)

    def neutral(self):
        neutral = 0.5 * (self.lower_limits + self.upper_limits)
        neutral[self.command_index["joint1"]] = 0.0
        neutral[self.command_index["joint4"]] = 0.0
        return self.clamp(neutral)

    def forward(self, joint_positions):
        joint_positions = self.clamp(joint_positions)
        transform = np.eye(4, dtype=float)
        joint_axes = []

        for joint in self.joints:
            origin_transform = _homogeneous(_rpy_matrix(joint.rpy), joint.xyz)
            joint_transform = transform @ origin_transform

            if joint.joint_type in ("revolute", "continuous") and joint.name in self.command_index:
                command_index = self.command_index[joint.name]
                axis = joint.axis / max(np.linalg.norm(joint.axis), 1e-12)
                joint_axes.append((command_index, joint_transform[:3, 3].copy(), joint_transform[:3, :3] @ axis))
                rotation = _homogeneous(_axis_angle_matrix(axis, joint_positions[command_index]), None)
                transform = joint_transform @ rotation
            else:
                transform = joint_transform

        return transform, joint_axes

    def palm_pose(self, joint_positions):
        transform, _ = self.forward(joint_positions)
        return transform[:3, 3].copy(), transform[:3, :3].copy()

    def jacobian(self, joint_positions):
        transform, joint_axes = self.forward(joint_positions)
        palm_position = transform[:3, 3]
        jacobian = np.zeros((6, 6), dtype=float)
        for command_index, joint_origin, axis_world in joint_axes:
            jacobian[:3, command_index] = np.cross(axis_world, palm_position - joint_origin)
            jacobian[3:, command_index] = axis_world
        return jacobian


class PiperArmIK:
    def __init__(self, urdf_path=None, joint6_locked_value=None):
        self.kinematics = PiperArmKinematics(urdf_path=urdf_path, joint6_locked_value=joint6_locked_value)

    def _seed_commands(self, target_position=None, seed=None):
        seeds = []
        if seed is not None:
            seeds.append(np.asarray(seed, dtype=float).reshape(6))

        if target_position is not None:
            target_position = np.asarray(target_position, dtype=float).reshape(3)
            yaw = math.atan2(target_position[1], target_position[0])
            seeds.extend(
                [
                    np.array([yaw, 0.8, -0.8, 0.0, 0.0, 1.57]),
                    np.array([yaw, 1.2, -1.2, 0.0, 0.0, 1.57]),
                    np.array([yaw, 1.6, -1.6, 0.0, 0.0, 1.57]),
                ]
            )

        seeds.append(self.kinematics.neutral())
        seeds.extend(
            [
                np.array([0.0, 0.4, -0.4, 0.0, 0.0, 1.57]),
                np.array([0.0, 0.8, -0.8, 0.0, 0.0, 1.57]),
                np.array([0.0, 1.2, -1.2, 0.0, 0.0, 1.57]),
                np.array([0.0, 1.6, -1.6, 0.0, 0.0, 1.57]),
                np.array([0.0, 2.0, -2.0, 0.0, 0.0, 1.57]),
                np.array([0.0, 2.4, -2.4, 0.0, 0.0, 1.57]),
                np.array([0.8, 1.2, -1.2, 0.0, 0.0, 1.57]),
                np.array([-0.8, 1.2, -1.2, 0.0, 0.0, 1.57]),
            ]
        )
        return [self.kinematics.clamp(seed_value) for seed_value in seeds]

    def solve(
        self,
        target_position,
        target_palm_y=np.array([0.0, 0.0, 1.0], dtype=float),
        target_palm_z=None,
        seed=None,
        max_iters=250,
        damping=0.05,
        step_size=0.2,
        orientation_weight=1.0,
        position_tolerance=POSITION_TOLERANCE,
        palm_tilt_tolerance=PALM_TILT_TOLERANCE,
        locked_joints=None,
    ):
        target_position = np.asarray(target_position, dtype=float).reshape(3)
        if target_palm_y is not None:
            target_palm_y = np.asarray(target_palm_y, dtype=float).reshape(3)
            target_palm_y = target_palm_y / max(np.linalg.norm(target_palm_y), 1e-12)
        if target_palm_z is not None:
            target_palm_z = np.asarray(target_palm_z, dtype=float).reshape(3)
            target_palm_z = target_palm_z / max(np.linalg.norm(target_palm_z), 1e-12)
        locked_indices = []
        locked_values = {}
        for name, value in (locked_joints or {}).items():
            if name not in self.kinematics.command_index:
                raise ValueError(f"Unknown locked joint '{name}'")
            index = self.kinematics.command_index[name]
            locked_indices.append(index)
            locked_values[index] = float(value)

        def apply_locked(joints):
            for index, value in locked_values.items():
                joints[index] = value
            return self.kinematics.clamp(joints)

        active = np.array(
            [index for index in self.kinematics.active_indices if index not in locked_indices],
            dtype=int,
        )

        best = None
        for seed_value in self._seed_commands(target_position=target_position, seed=seed):
            joint_positions = apply_locked(seed_value.copy())
            for _ in range(max_iters):
                palm_position, palm_rotation = self.kinematics.palm_pose(joint_positions)
                position_error = target_position - palm_position
                jacobian = self.kinematics.jacobian(joint_positions)

                orientation_errors = []
                orientation_jacobians = []
                if target_palm_y is not None and orientation_weight > 0.0:
                    current_palm_y = palm_rotation[:, 1]
                    orientation_errors.append(np.cross(current_palm_y, target_palm_y) * orientation_weight)
                    orientation_jacobians.append(jacobian[3:, :] * orientation_weight)
                if target_palm_z is not None and orientation_weight > 0.0:
                    current_palm_z = palm_rotation[:, 2]
                    orientation_errors.append(np.cross(current_palm_z, target_palm_z) * orientation_weight)
                    orientation_jacobians.append(jacobian[3:, :] * orientation_weight)

                if orientation_errors:
                    error = np.concatenate([position_error, *orientation_errors])
                    task_jacobian = np.vstack([jacobian[:3, :], *orientation_jacobians])
                else:
                    error = position_error
                    task_jacobian = jacobian[:3, :]

                active_jacobian = task_jacobian[:, active]
                if active_jacobian.size == 0:
                    break
                lhs = active_jacobian @ active_jacobian.T + (damping * damping) * np.eye(active_jacobian.shape[0])
                try:
                    delta_active = active_jacobian.T @ np.linalg.solve(lhs, error)
                except np.linalg.LinAlgError:
                    delta_active = active_jacobian.T @ (np.linalg.pinv(lhs) @ error)

                joint_positions[active] += step_size * delta_active
                joint_positions = apply_locked(joint_positions)

            result = self._make_result(
                joint_positions,
                target_position,
                target_palm_y,
                position_tolerance,
                palm_tilt_tolerance,
                target_palm_z=target_palm_z,
            )
            if best is None or result.score < best.score:
                best = result

        return best

    def _make_result(
        self,
        joint_positions,
        target_position,
        target_palm_y,
        position_tolerance,
        palm_tilt_tolerance,
        target_palm_z=None,
    ):
        achieved_position, achieved_rotation = self.kinematics.palm_pose(joint_positions)
        position_error = float(np.linalg.norm(achieved_position - target_position))

        axis_errors = []
        if target_palm_y is not None:
            current_palm_y = achieved_rotation[:, 1]
            axis_errors.append(_angle_between(current_palm_y, target_palm_y))
        if target_palm_z is not None:
            current_palm_z = achieved_rotation[:, 2]
            axis_errors.append(_angle_between(current_palm_z, target_palm_z))

        if axis_errors:
            palm_axis_error = max(axis_errors)
            score = max(position_error / position_tolerance, palm_axis_error / palm_tilt_tolerance)
        else:
            palm_axis_error = None
            score = position_error / position_tolerance

        return IkResult(
            joint_positions=self.kinematics.clamp(joint_positions),
            target_position=target_position.copy(),
            achieved_position=achieved_position,
            position_error=position_error,
            palm_axis_error=palm_axis_error,
            success=score <= 1.0,
            score=float(score),
        )

    def solve_fruit_grasp(
        self,
        fruit_center,
        seed=None,
        table_z=DEFAULT_TABLE_Z,
        grasp_height=FRUIT_GRASP_HEIGHT,
    ):
        fruit_center = _as_point(fruit_center)
        target_position = np.array(
            # Fruit is assumed to be resting on the table regardless of detected z.
            [fruit_center[0], fruit_center[1], float(table_z) + float(grasp_height)],
            dtype=float,
        )
        return self.solve(
            target_position,
            target_palm_y=None,
            target_palm_z=WORLD_Z_AXIS,
            seed=seed,
            locked_joints={"joint4": 0.0, "joint6": 0.0},
            orientation_weight=1.0,
        )

    def solve_fruit_release(
        self,
        current_joint_positions,
        release_height=FRUIT_RELEASE_HEIGHT,
        table_z=DEFAULT_TABLE_Z,
    ):
        return self.solve_vertical_release(
            current_joint_positions,
            release_height=release_height,
            table_z=table_z,
        )

    def solve_bottle_grasp(
        self,
        bottle_center,
        seed=None,
        table_z=DEFAULT_TABLE_Z,
        grasp_height=BOTTLE_GRASP_HEIGHT,
        diameter=DEFAULT_BOTTLE_DIAMETER,
        angle_offset=None,
    ):
        if isinstance(bottle_center, BottleModel):
            bottle_model = bottle_center
        else:
            bottle_model = make_bottle_model(bottle_center, default_radius=float(diameter) * 0.5)
        target_position = compute_bottle_grasp_position(
            bottle_model,
            table_z=table_z,
            grasp_height=grasp_height,
            angle_offset=angle_offset,
        )
        return self.solve(target_position, seed=seed)

    def solve_vertical_release(self, current_joint_positions, release_height, table_z=DEFAULT_TABLE_Z):
        current_joint_positions = self.kinematics.clamp(current_joint_positions)
        current_position, current_rotation = self.kinematics.palm_pose(current_joint_positions)
        target_position = np.array(
            [current_position[0], current_position[1], float(table_z) + float(release_height)],
            dtype=float,
        )
        return self.solve(
            target_position,
            target_palm_y=current_rotation[:, 1],
            target_palm_z=current_rotation[:, 2],
            seed=current_joint_positions,
            orientation_weight=1.0,
        )

    def solve_release(self, current_joint_positions, release_height, table_z=DEFAULT_TABLE_Z):
        return self.solve_vertical_release(
            current_joint_positions,
            release_height=release_height,
            table_z=table_z,
        )


def demo_points():
    ik = PiperArmIK()
    examples = [
        make_bottle_model([0.60, 0.10, 0.0], [0.60, 0.065, 0.0], [0.60, 0.135, 0.0]),
        make_bottle_model([0.60, 0.00, 0.0], [0.60, -0.035, 0.0], [0.60, 0.035, 0.0]),
        make_bottle_model([0.55, -0.25, 0.0], [0.55, -0.285, 0.0], [0.55, -0.215, 0.0]),
        make_bottle_model([0.65, 0.25, 0.0], [0.65, 0.215, 0.0], [0.65, 0.285, 0.0]),
        make_bottle_model([0.60, -0.10, 0.0], [0.60, -0.135, 0.0], [0.60, -0.065, 0.0]),
    ]

    for bottle in examples:
        target = compute_bottle_grasp_position(bottle)
        result = ik.solve_bottle_grasp(bottle)
        tilt_deg = None if result.palm_axis_error is None else math.degrees(result.palm_axis_error)
        print(
            "bottle center="
            f"{np.round(bottle.center[:2], 4).tolist()} radius={bottle.radius:.4f} "
            f"target={np.round(target, 4).tolist()} joints(rad)={np.round(result.joint_positions, 6).tolist()} "
            f"pos_err={result.position_error:.5f} tilt_deg={tilt_deg:.3f} success={result.success}"
        )

    release_examples = [
        ("BOTTLE", compute_bottle_grasp_position(examples[0]) + np.array([0.0, 0.0, 0.10]), 0.0),
        ("BOTTLE", compute_bottle_grasp_position(examples[2]) + np.array([0.0, 0.0, 0.08]), 0.0),
        ("fruit", np.array([0.60, 0.00, 0.12], dtype=float), 0.0),
    ]

    for name, carry_target, release_height in release_examples:
        if name == "BOTTLE":
            carry = ik.solve(carry_target)
        else:
            carry = ik.solve(carry_target, target_palm_y=None)
        current = carry.joint_positions
        release = ik.solve_release(current, release_height)
        print(
            f"release {name} from={np.round(carry.achieved_position, 4).tolist()} "
            f"target={np.round(release.target_position, 4).tolist()} "
            f"joints(rad)={np.round(release.joint_positions, 6).tolist()} "
            f"pos_err={release.position_error:.5f} success={release.success}"
        )


if __name__ == "__main__":
    demo_points()
