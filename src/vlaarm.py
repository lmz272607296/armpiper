import multiprocessing
import time
import threading
import asyncio
import websockets
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full
from typing import Optional
import cv2
import numpy as np
import tyro
from loguru import logger

import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig

# ================= 工具函数 =================
def mj2scipy_quat(q): return np.array([q[1], q[2], q[3], q[0]])

OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT  = np.array([[0, 0, -1], [1,  0, 0], [0, -1, 0]])


@dataclass
class ArmSafetyConfig:
    # Easily adjustable safety knobs
    singularity_cond_threshold: float = 80.0
    singularity_min_scale: float = 0.2
    max_joint_step_rad: float = 0.05
    torque_limit_abs: float = 2.5


@dataclass
class CommandFilterConfig:
    arm_alpha: float = 0.25
    hand_alpha: float = 0.35


@dataclass
class RosTopicConfig:
    arm_topic: str = "/piper/joint_trajectory_cmd"
    hand_topic: str = "/hand_controller/joint_trajectory"
    frame_id: str = "base_link"
    default_duration_sec: float = 0.06


class ExponentialSmoother:
    def __init__(self, size: int, alpha: float):
        self.size = size
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.state = np.zeros(size, dtype=float)
        self.initialized = False

    def update(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        if value.shape[0] != self.size:
            raise ValueError(f"Expected size {self.size}, got {value.shape[0]}")
        if not self.initialized:
            self.state = value.copy()
            self.initialized = True
            return self.state.copy()
        self.state = self.alpha * value + (1.0 - self.alpha) * self.state
        return self.state.copy()


class RosCommandPublisher(Node):
    def __init__(self, topic_cfg: RosTopicConfig):
        super().__init__("vlaarm_bridge")
        self.topic_cfg = topic_cfg
        self.arm_pub = self.create_publisher(JointTrajectory, topic_cfg.arm_topic, 10)
        self.hand_pub = self.create_publisher(JointTrajectory, topic_cfg.hand_topic, 10)
        self.arm_joint_names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
        self.hand_joint_names = [f"hand{i}" for i in range(16)]

    def publish(self, arm_cmd6: np.ndarray, hand_cmd16: np.ndarray):
        arm_msg = JointTrajectory()
        arm_msg.header.stamp = self.get_clock().now().to_msg()
        arm_msg.header.frame_id = self.topic_cfg.frame_id
        arm_msg.joint_names = self.arm_joint_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = arm_cmd6.tolist() + [0.0]
        arm_point.time_from_start = Duration(seconds=self.topic_cfg.default_duration_sec).to_msg()
        arm_msg.points.append(arm_point)
        self.arm_pub.publish(arm_msg)

        hand_msg = JointTrajectory()
        hand_msg.header.stamp = self.get_clock().now().to_msg()
        hand_msg.header.frame_id = self.topic_cfg.frame_id
        hand_msg.joint_names = self.hand_joint_names
        hand_point = JointTrajectoryPoint()
        hand_point.positions = hand_cmd16.tolist()
        hand_point.time_from_start = Duration(seconds=self.topic_cfg.default_duration_sec).to_msg()
        hand_msg.points.append(hand_point)
        self.hand_pub.publish(hand_msg)


# ================= 核心类 0：WebXR 数据接收器 =================
class WebXRHandDetector:
    def __init__(self, hand_type="Right", ws_url="ws://127.0.0.1:8765", robot_name=None):
        self.hand_type = hand_type
        self.robot_name = robot_name
        self.ws_url = ws_url
        self.operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT

        self.latest_landmarks = None
        self.latest_wrist_orientation = None  # 四元数 [x, y, z, w]
        self.is_running = True
        self.first_frame_received = False

        self.thread = threading.Thread(target=self._start_ws_loop, daemon=True)
        self.thread.start()

    def _start_ws_loop(self):
        asyncio.run(self._ws_listener())

    async def _ws_listener(self):
        while self.is_running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.success(f"手势检测器成功连接到中转站: {self.ws_url}")
                    async for msg in ws:
                        self._parse_msg(msg)
            except Exception:
                logger.warning(f"等待中转站开启... ({self.ws_url})")
                await asyncio.sleep(2)

    def _parse_msg(self, msg):
        """
        解析 app.js 实际发送的 JSON 格式：
        {"right": {"wrist": {"x":..,"y":..,"z":..}, "thumb-metacarpal": {...}, ...}}
        """
        if isinstance(msg, bytes):
            msg = msg.decode('utf-8', errors='ignore')

        import json

        JOINT_NAMES = [
            "wrist",
            "thumb-metacarpal", "thumb-phalanx-proximal", "thumb-phalanx-distal", "thumb-tip",
            "index-finger-metacarpal", "index-finger-phalanx-proximal", "index-finger-phalanx-intermediate", "index-finger-phalanx-distal", "index-finger-tip",
            "middle-finger-metacarpal", "middle-finger-phalanx-proximal", "middle-finger-phalanx-intermediate", "middle-finger-phalanx-distal", "middle-finger-tip",
            "ring-finger-metacarpal", "ring-finger-phalanx-proximal", "ring-finger-phalanx-intermediate", "ring-finger-phalanx-distal", "ring-finger-tip",
            "pinky-finger-metacarpal", "pinky-finger-phalanx-proximal", "pinky-finger-phalanx-intermediate", "pinky-finger-phalanx-distal", "pinky-finger-tip"
        ]
        KEEP_NAMES = [JOINT_NAMES[i] for i in [0,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24]]

        try:
            data = json.loads(msg)
            hand_key = self.hand_type.lower()

            if hand_key not in data:
                return

            hand_data = data[hand_key]
            landmarks = []

            for joint_name in KEEP_NAMES:
                joint = hand_data.get(joint_name)
                if joint is None:
                    return
                landmarks.extend([joint["x"], joint["y"], joint["z"]])

            if len(landmarks) != 63:
                return

            self.latest_landmarks = np.array(landmarks, dtype=float).reshape(21, 3)

            # 手腕朝向：如果 app.js 在 wrist 里附加了 ox,oy,oz,ow 字段则读取
            wrist_joint = hand_data.get("wrist", {})
            if "ox" in wrist_joint:
                self.latest_wrist_orientation = np.array([
                    wrist_joint["ox"], wrist_joint["oy"],
                    wrist_joint["oz"], wrist_joint["ow"]
                ], dtype=float)

            if not self.first_frame_received:
                logger.success("✅ [系统撒花] 成功解析到VR三维坐标！")
                self.first_frame_received = True

        except Exception:
            pass

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        points = keypoint_3d_array[[0, 5, 9], :]
        x_vector = points[0] - points[2]
        points = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points)
        normal = v[2, :]
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        return np.stack([x, normal, z], axis=1)

    def detect(self):
        if self.latest_landmarks is None:
            return 0, None, None, None

        raw_landmarks = self.latest_landmarks.copy()

        # WebXR 世界坐标，不需要镜像翻转
        global_wrist_pos = raw_landmarks[0].copy()
        local_landmarks = raw_landmarks - global_wrist_pos
        wrist_rot = self.estimate_frame_from_hand_points(local_landmarks)
        joint_pos = local_landmarks @ wrist_rot @ self.operator2mano

        if self.robot_name and "bidexhand" in self.robot_name:
            joint_pos *= 1.5

        return 1, joint_pos, global_wrist_pos, wrist_rot


# ================= 核心类 1：大臂逆运动学 (IK) =================
class ArmIKSolver:
    def __init__(self, model, data, ee_body_name="link6"):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        self.arm_dof = 6

    def get_ee_pose(self):
        return self.data.xpos[self.ee_id].copy(), self.data.xquat[self.ee_id].copy()

    def solve_step(self, target_pos, target_quat_wxyz, current_qpos):
        curr_pos, curr_quat_wxyz = self.get_ee_pose()
        err_pos = target_pos - curr_pos

        r_curr   = R.from_quat(mj2scipy_quat(curr_quat_wxyz))
        r_target = R.from_quat(mj2scipy_quat(target_quat_wxyz))
        err_rot  = (r_target * r_curr.inv()).as_rotvec()

        err_6d = np.hstack([err_pos, err_rot])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)

        J      = np.vstack([jacp[:, :self.arm_dof], jacr[:, :self.arm_dof]])
        jjt = J @ J.T + (0.1 ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(jjt)
        singular_values = np.linalg.svd(J, compute_uv=False)
        cond_number = np.inf if singular_values[-1] < 1e-8 else singular_values[0] / singular_values[-1]

        delta_q = np.clip(J_pinv @ err_6d, -0.1, 0.1)
        return current_qpos[:self.arm_dof] + delta_q, cond_number


# ================= 核心类 2：握拳手势使能控制器 =================
# ================= 核心类 2：握拳与复位手势控制器 =================
class GestureToggleController:
    def __init__(self, hold_time=3.0, fist_threshold=0.7, open_threshold=0.3):
        self.hold_time       = hold_time
        self.fist_threshold  = fist_threshold
        self.open_threshold  = open_threshold
        self.is_enabled      = False
        self.fist_start_time = None
        self.action_locked   = False
        
        # --- 新增复位参数 ---
        self.reset_start_time = None
        self.reset_hold_time  = 2.0   # 捏合 2 秒触发复位
        self.pinch_threshold  = 0.04  # 拇指与食指的距离阈值 (米)，小于4厘米视为捏合

    def update(self, finger_qpos, joint_pos):
        avg_flexion  = np.mean(finger_qpos)
        current_time = time.time()
        is_reset = False

        # --- 1. 优先检测复位手势 (拇指指尖4 与 食指指尖8 捏合) ---
        thumb_tip = joint_pos[4]
        index_tip = joint_pos[8]
        pinch_dist = np.linalg.norm(thumb_tip - index_tip)

        if pinch_dist < self.pinch_threshold:
            if self.reset_start_time is None:
                self.reset_start_time = current_time
                logger.info(">>> ⚠️ 检测到捏合手势，保持 2 秒进行全局复位... <<<")
            else:
                if current_time - self.reset_start_time >= self.reset_hold_time:
                    is_reset = True
                    self.is_enabled = False  # 强制解除使能
                    self.action_locked = True
                    self.reset_start_time = None
                    self.fist_start_time = None
                    return self.is_enabled, is_reset, avg_flexion
        else:
            self.reset_start_time = None

        # --- 2. 正常的握拳使能逻辑 ---
        if avg_flexion < self.open_threshold:
            self.action_locked   = False
            self.fist_start_time = None
            return self.is_enabled, is_reset, avg_flexion

        if avg_flexion > self.fist_threshold and not self.action_locked:
            if self.fist_start_time is None:
                self.fist_start_time = current_time
                logger.info(">>> 检测到用力握拳！保持住！正在倒数 3 秒... <<<")
            else:
                elapsed = current_time - self.fist_start_time
                if elapsed >= self.hold_time:
                    self.is_enabled = not self.is_enabled
                    state_str = "🟢 【已使能】大臂开始跟随" if self.is_enabled else "🔴 【取消使能】大臂锁定"
                    logger.success(f"触发成功！当前状态: {state_str}")
                    self.action_locked   = True
                    self.fist_start_time = None
        else:
            if not self.action_locked:
                self.fist_start_time = None

        return self.is_enabled, is_reset, avg_flexion


# ================== 消费者进程 ==================
def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    hand_type   = "Right" if "right" in config_path.lower() else "Left"

    filepath = Path(RetargetingConfig.load_from_file(config_path).urdf_path)
    detector = WebXRHandDetector(
        hand_type=hand_type,
        ws_url="ws://127.0.0.1:8765",
        robot_name=filepath.stem
    )

    model      = mujoco.MjModel.from_xml_path("piper_mujoco.xml")
    data       = mujoco.MjData(model)
    ik_solver  = ArmIKSolver(model, data, ee_body_name="link6")
    gesture_controller = GestureToggleController(
        hold_time=3.0, fist_threshold=0.7, open_threshold=0.3
    )

    # --------- Safety / Filter / ROS configuration ---------
    arm_safety_cfg = ArmSafetyConfig()
    cmd_filter_cfg = CommandFilterConfig()
    ros_topic_cfg = RosTopicConfig()

    rclpy.init(args=None)
    ros_publisher = RosCommandPublisher(ros_topic_cfg)

    arm_smoother = ExponentialSmoother(size=6, alpha=cmd_filter_cfg.arm_alpha)
    hand_smoother = ExponentialSmoother(size=16, alpha=cmd_filter_cfg.hand_alpha)

    arm_joint_min = model.jnt_range[:6, 0].copy()
    arm_joint_max = model.jnt_range[:6, 1].copy()
    if not np.all(np.isfinite(arm_joint_min)) or not np.all(np.isfinite(arm_joint_max)):
        arm_joint_min = np.array([-3.0] * 6)
        arm_joint_max = np.array([3.0] * 6)

    hand_joint_min = model.jnt_range[6:22, 0].copy()
    hand_joint_max = model.jnt_range[6:22, 1].copy()
    if hand_joint_min.shape[0] != 16 or not np.all(np.isfinite(hand_joint_min)):
        hand_joint_min = np.array([-0.2] * 16)
        hand_joint_max = np.array([1.6] * 16)

    real_robot_joint_names = [str(i) for i in range(16)]
    retargeting_to_mujoco  = np.array(
        [retargeting.joint_names.index(name) for name in real_robot_joint_names]
    ).astype(int)

    # 状态变量
    was_enabled           = False
    vr_palm_pos_initial   = np.zeros(3)
    robot_ee_pos_initial  = np.zeros(3)
    robot_ee_quat_initial = np.array([1.0, 0.0, 0.0, 0.0])
    vr_initial_rot        = np.eye(3)   # 使能瞬间的手腕朝向矩阵

    mujoco.mj_step(model, data)
    locked_arm_qpos = data.qpos[:6].copy()
    prev_arm_cmd = locked_arm_qpos.copy()

    default_home_qpos = data.qpos[:6].copy()

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        try:
            while viewer.is_running():
                try:
                    bgr = queue.get(timeout=0.01)
                except Empty:
                    pass

                _, joint_pos, global_wrist_pos, wrist_rot_matrix = detector.detect()

                if joint_pos is None:
                    continue

                # --- 手指重定向 ---
                indices = retargeting.optimizer.target_link_human_indices
                if retargeting.optimizer.retargeting_type == "POSITION":
                    ref_value = joint_pos[indices, :]
                else:
                    ref_value = joint_pos[indices[1, :], :] - joint_pos[indices[0, :], :]

                raw_hand_qpos    = retargeting.retarget(ref_value)
                hand_qpos_mapped = raw_hand_qpos[retargeting_to_mujoco]
                if hand_qpos_mapped.shape[0] != 16:
                    logger.warning(f"手部关节维度异常: expected 16, got {hand_qpos_mapped.shape[0]}")
                    continue
                hand_qpos_mapped = np.nan_to_num(hand_qpos_mapped, nan=0.0, posinf=0.0, neginf=0.0)

                vr_palm_pos = global_wrist_pos
  # 💥 传入 joint_pos 以便计算指尖距离
                is_enabled, is_reset, current_avg_flexion = gesture_controller.update(hand_qpos_mapped, joint_pos)

                if step_count % 30 == 0:
                    logger.info(
                        f"VR绝对坐标: [{vr_palm_pos[0]:.2f}, {vr_palm_pos[1]:.2f}, {vr_palm_pos[2]:.2f}] | "
                        f"弯曲度: {current_avg_flexion:.2f}"
                    )

                # --- 状态机：处理复位与切换 ---
                if is_reset:
                    # 触发复位：强制覆盖锁定位置为原点
                    logger.warning("🔄 触发复位！机械臂回到初始安全位置")
                    locked_arm_qpos = default_home_qpos.copy()
                    was_enabled = False
                    model.geom_rgba[floor_id] = [0.8, 0.2, 0.2, 1.0] # 变成红色提示复位！

                elif is_enabled and not was_enabled:
                    vr_palm_pos_initial   = vr_palm_pos.copy()
                    robot_ee_pos_initial, robot_ee_quat_initial = ik_solver.get_ee_pose()
                    was_enabled = True

                    ori = detector.latest_wrist_orientation
                    if ori is not None:
                        vr_initial_rot = R.from_quat(ori).as_matrix()
                        logger.info("✅ 已记录手腕朝向，启用局部坐标系增量控制")
                    else:
                        vr_initial_rot = np.eye(3)
                    
                    model.geom_rgba[floor_id] = [0.2, 0.8, 0.2, 1.0] # 绿色使能

                elif not is_enabled and was_enabled:
                    locked_arm_qpos = data.qpos[:6].copy()
                    was_enabled     = False
                    model.geom_rgba[floor_id] = [0.2, 0.2, 0.2, 1.0] # 灰色待机
                # --- 大臂控制 ---
# --- 大臂控制 ---
                if is_enabled:
                    # 1. 算出现实房间里的绝对物理位移增量
                    world_delta = np.array([
                        vr_palm_pos[0] - vr_palm_pos_initial[0], # VR 左右 (X)
                        vr_palm_pos[1] - vr_palm_pos_initial[1], # VR 上下 (Y)
                        vr_palm_pos[2] - vr_palm_pos_initial[2], # VR 前后 (Z)
                    ])

                    motion_scale = 1.5  # 可以根据手感调大调小

                    # 2. 【核心映射】：把 VR 的轴重新接线到机械臂的轴上！
                    delta_pos = np.array([
                        -world_delta[2],   
                        -world_delta[0],   
                        world_delta[1],    
                    ]) * motion_scale

                    target_ee_pos  = robot_ee_pos_initial + delta_pos
                    target_ee_quat = robot_ee_quat_initial  # 姿态保持锁死

                    target_arm_qpos, cond_number = ik_solver.solve_step(target_ee_pos, target_ee_quat, data.qpos)

                    if cond_number > arm_safety_cfg.singularity_cond_threshold:
                        singularity_scale = max(
                            arm_safety_cfg.singularity_min_scale,
                            arm_safety_cfg.singularity_cond_threshold / cond_number,
                        )
                        target_arm_qpos = data.qpos[:6] + singularity_scale * (target_arm_qpos - data.qpos[:6])
                        if step_count % 30 == 0:
                            logger.warning(
                                f"检测到奇异区风险 cond={cond_number:.1f}, 自动降速系数={singularity_scale:.2f}"
                            )
                
                # 💥 必须把 else 加回来！告诉程序不使能时发什么指令
                else:
                    target_arm_qpos = locked_arm_qpos

                # --- 安全限制：关节范围 / 单步变化 / 控制量范围 ---
                target_arm_qpos = np.clip(target_arm_qpos, arm_joint_min, arm_joint_max)

                step_delta = np.clip(
                    target_arm_qpos - prev_arm_cmd,
                    -arm_safety_cfg.max_joint_step_rad,
                    arm_safety_cfg.max_joint_step_rad,
                )
                limited_arm_cmd = prev_arm_cmd + step_delta
                smoothed_arm_cmd = arm_smoother.update(limited_arm_cmd)

                # 按“扭矩/控制量”维度做范围保护，便于后续替换更高级控制器
                arm_ctrl_cmd = np.clip(
                    smoothed_arm_cmd,
                    -arm_safety_cfg.torque_limit_abs,
                    arm_safety_cfg.torque_limit_abs,
                )
                prev_arm_cmd = arm_ctrl_cmd.copy()

                hand_qpos_limited = np.clip(hand_qpos_mapped, hand_joint_min, hand_joint_max)
                hand_cmd = hand_smoother.update(hand_qpos_limited)

                # --- 发送电机指令 ---
                data.ctrl[:6] = arm_ctrl_cmd
                data.ctrl[6:22] = hand_cmd

                # --- ROS 发布：大臂与灵巧手同步输出 ---
                ros_publisher.publish(arm_ctrl_cmd, hand_cmd)
                rclpy.spin_once(ros_publisher, timeout_sec=0.0)

                for _ in range(10):
                    mujoco.mj_step(model, data)

                viewer.sync()
                step_count += 1

        finally:
            ros_publisher.destroy_node()
            rclpy.shutdown()
            logger.info("退出仿真循环。")


# ================== 生产者进程 ==================
def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    cap = cv2.VideoCapture(0) if camera_path is None else cv2.VideoCapture(camera_path)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        try:
            queue.put_nowait(image)
        except Full:
            pass


# ================== 主程序 ==================
def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
):
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir   = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"

    queue            = multiprocessing.Queue(maxsize=3)
    producer_process = multiprocessing.Process(target=produce_frame,      args=(queue, camera_path))
    consumer_process = multiprocessing.Process(target=start_retargeting,  args=(queue, str(robot_dir), str(config_path)))

    producer_process.start()
    consumer_process.start()
    producer_process.join()
    consumer_process.join()


if __name__ == "__main__":
    tyro.cli(main)
