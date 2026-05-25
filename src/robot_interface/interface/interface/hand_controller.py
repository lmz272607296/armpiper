# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


import time

from matplotlib import scale
from matplotlib.pyplot import sca
import rclpy
try:
    from control_msgs.msg import DynamicJointState
except ImportError:
    DynamicJointState = None
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time

import numpy as np
#from builtin_interfaces.msg import Duration
from rclpy.duration import Duration


CURRENT_INTERFACE_NAMES = ("Present Current", "Current")


def extract_currents_from_dynamic_joint_state(msg, joint_names=None):
    selected_joint_names = None if joint_names is None else set(joint_names)
    currents = {}
    for joint_name, interface_values in zip(msg.joint_names, msg.interface_values):
        if selected_joint_names is not None and joint_name not in selected_joint_names:
            continue

        interface_names = list(getattr(interface_values, "interface_names", []))
        values = list(getattr(interface_values, "values", []))
        current_index = None
        for interface_name in CURRENT_INTERFACE_NAMES:
            try:
                current_index = interface_names.index(interface_name)
                break
            except ValueError:
                continue

        if current_index is None or current_index >= len(values):
            continue

        try:
            currents[str(joint_name)] = float(values[current_index])
        except (TypeError, ValueError):
            continue

    return currents


class LeapHand(Node):
    def __init__(self,name):
        super().__init__(name)
        """通信处理——通过定义Publisher和Subscription实现读取和发布"""
        
        #声明两个用于存储电机数据的参数，两个参数是名称和默认值。
        self.declare_parameter('command_topic', '/hand_controller/joint_trajectory')
        self.declare_parameter('state_topic', '/hand_joint_states')
        self.declare_parameter('dynamic_state_topic', '/hand_dynamic_joint_states')
        self.command_topic = self.get_parameter('command_topic').value
        self.state_topic = self.get_parameter('state_topic').value
        self.dynamic_state_topic = self.get_parameter('dynamic_state_topic').value

        #创建发布器，Publisher=create_publisher(msg_type,topic_name,Qosfile)
        #Qos可能会影响性能，后面可能微调
        self.publisher = self.create_publisher(JointTrajectory, self.command_topic, 10)

        qos_profile = QoSProfile(
            depth=10,  # 队列深度
            reliability=ReliabilityPolicy.RELIABLE,  # 设置可靠性为Reliable
            history=HistoryPolicy.KEEP_LAST,  # 设置只保留最近的消息
            )
        
        
        self.subscription = self.create_subscription(
            JointState,
            self.state_topic,
            self.controller_state_callback,
            qos_profile
        )
        self.dynamic_subscription = None
        if DynamicJointState is not None:
            self.dynamic_subscription = self.create_subscription(
                DynamicJointState,
                self.dynamic_state_topic,
                self.dynamic_state_callback,
                qos_profile,
            )
        else:
            self.get_logger().warn('control_msgs.msg.DynamicJointState 不可用，无法读取手部电机电流。')
        
        #读取电机状态的订阅器，订阅话题为/hand_joint_states，避免与Piper机械臂冲突
        self.raw_positions = None
        self.real_raw_positions = None
        self.last_state_time = 0.0
        # 定义灵巧手电机数量，后续电流监控和维度检查都会用到。
        self.joints_num = 16
        self.latest_currents_ma = {}
        self.last_current_time = 0.0
        self.current_joint_names = tuple(f'dxl{i}' for i in range(self.joints_num))



        self.sim_to_real_indices=[1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7]
        self.real_to_sim_indices=[1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]

        self.log_to_real_indices = [8, 12, 11, 0, 2, 1, 13, 3, 4, 15, 6, 7, 9, 5, 14, 10]

    def sim_to_real(self, values):
        return values[self.sim_to_real_indices]
    

    #------------------------------------------
    # 用于对齐话题顺序和实际电机顺序


    def log_to_sim(self,received_names: list,received_values: list) -> np.ndarray:
        hand_names = [
        # 机械手关节
            'hand0', 'hand1', 'hand2', 'hand3', 'hand4',
            'hand5', 'hand6', 'hand7', 'hand8', 'hand9',
            'hand10', 'hand11', 'hand12', 'hand13', 'hand14', 'hand15',
        ]
        # 提高效率：创建一个从名称到值的映射（字典），用于快速查找,避免了低效的嵌套循环
        value_map = dict(zip(received_names, received_values))

        # 构建结果：根据标准顺序从映射中提取值
        ordered_values = []


        for name in hand_names:

        # 使用 .get() 方法安全地查找值
            value = value_map.get(name)
            if value is not None:
                ordered_values.append(value)
            else:
                raise ValueError(
                    f"在收到的消息中找不到期望的关节 '{name}'。 "
                    f"可用的关节有: {list(value_map.keys())}"
                )

        return np.array(ordered_values)


    def controller_state_callback(self,msg):

        if not isinstance(msg, JointState):
            self.get_logger().error("收到非 JointState 消息!")
            return

        try:
            self.raw_positions =(msg.position)
            self.raw_name=(msg.name)

            
            self.real_raw_positions=self.log_to_sim(self.raw_name,self.raw_positions)
            self.raw_positions = self.real_to_sim(self.real_raw_positions)
            
            self.raw_positions = np.reshape(self.raw_positions, (1, len(self.raw_positions)))
            self.real_raw_positions = np.reshape(self.real_raw_positions, (1, len(self.real_raw_positions)))
            self.last_state_time = time.time()
              
        except Exception as e:
            self.get_logger().error(f"Positions message launching error: {repr(e)}")

    def dynamic_state_callback(self, msg):
        try:
            current_updates = extract_currents_from_dynamic_joint_state(
                msg,
                joint_names=self.current_joint_names,
            )
            if not current_updates:
                return
            self.latest_currents_ma.update(current_updates)
            self.last_current_time = time.time()
        except Exception as exc:
            self.get_logger().error(f"Current message launching error: {repr(exc)}")

    def command_joint_position(self, desired_pose,speed):
        desired_pose = np.asarray(desired_pose, dtype=float)
        if desired_pose.ndim == 1:
            desired_pose = desired_pose.reshape(1, -1)
        if desired_pose.ndim != 2 or desired_pose.shape[1] != self.joints_num:
            self.get_logger().warn(f'Invalid hand joint dimension: expected 16, got shape {desired_pose.shape}')
            return False
        scaled_pose=[]

        #------------------------------------- 
        # 微调各个电机的角度
        opration=np.array([0,0.0,0,0,
                           0,0.0,0,0,
                           0,0.0,0,0,
                           0,0,0,0])
            
        #-------------------------------------
            
        try:
            for i in range(desired_pose.shape[0]):
                rawdata=desired_pose[i,:]
                realdata= self.sim_to_real(rawdata)
                realdata=realdata+opration
                realdata = realdata.tolist()
                scaled_pose.append(realdata)
            scaled_pose = np.array(scaled_pose)
        
            
            msg = JointTrajectory()
            # 3. 设置消息头
            msg.header.stamp = Time(sec=0, nanosec=0)
            msg.header.frame_id = 'base_link'  # 根据实际情况设置坐标系
                
            # 4. 设置关节名称（必须与YAML配置完全一致）
            msg.joint_names = [
                'hand0', 'hand1', 'hand2', 'hand3', 'hand4',
                'hand5', 'hand6', 'hand7', 'hand8', 'hand9',
                'hand10', 'hand11', 'hand12', 'hand13', 'hand14', 'hand15'
            ]
                
            # 5. 创建并填充轨迹点
            # point = JointTrajectoryPoint()
            transtime=0.0
            for point in scaled_pose:
                point_ja=JointTrajectoryPoint()
                point_ja.positions=[float(value) for value in point]
                # print("point:",point)
                
                #############################################################################
                ##                        在这里调整运行速度！
                transtime+=speed
                point_ja.time_from_start = Duration(seconds=transtime).to_msg()  

                msg.points.append(point_ja)
                ##                        在这里调整运行速度！
                #############################################################################
            
                
            # 6. 将轨迹点添加到消息中
            # msg.points = [point]  # 只包含一个轨迹点
                
            # 7. 发布消息
            self.publisher.publish(msg)
            print ('Published joint commands')
            self.get_logger().debug('Published joint commands')
            return True
        except Exception as e:
            self.get_logger().error(f"Publishing error: {str(e)}")
            return False

    def hold_current_position(self, duration=0.05):
        if self.raw_positions is None:
            self.get_logger().warn('当前没有可用的手部状态，无法发送停止保持指令。')
            return False
        current_pose = np.asarray(self.raw_positions, dtype=float).reshape(1, self.joints_num)
        return self.command_joint_position(current_pose, duration)

    def has_current_samples(self, joint_names=None):
        monitored_names = tuple(joint_names or self.latest_currents_ma.keys())
        if not monitored_names:
            return False
        return any(name in self.latest_currents_ma for name in monitored_names)

    def get_current_snapshot(self, joint_names=None):
        if joint_names is None:
            selected_names = self.latest_currents_ma.keys()
        else:
            selected_names = joint_names
        return {
            name: self.latest_currents_ma[name]
            for name in selected_names
            if name in self.latest_currents_ma
        }

    def get_over_limit_currents(self, limit_ma, joint_names=None):
        limit_ma = abs(float(limit_ma))
        current_snapshot = self.get_current_snapshot(joint_names=joint_names)
        return {
            joint_name: current_ma
            for joint_name, current_ma in current_snapshot.items()
            if abs(current_ma) >= limit_ma
        }

   
    def real_to_sim(self, values):
        return values[self.real_to_sim_indices]
    def LEAPsim_limits(self):
        sim_min = self.sim_to_real(self.leap_dof_lower)
        sim_max = self.sim_to_real(self.leap_dof_upper)

        return sim_min, sim_max

    def LEAPhand_to_LEAPsim(self, joints):
        joints = np.array(joints)
        ret_joints = joints 
        return ret_joints

    
    # def log_to_sim(self, values,joints_num):
    #     if len(values) == joints_num:
    #         values=self.LEAPhand_to_LEAPsim(values)
    #         values=values[self.log_to_real_indices]
    #         values=self.real_to_sim(values)
    #         return values
    #     else:
    #         while True:
    #             print("Error: log_to_sim: values length is not equal to joints_num")
    #             print("values length:", len(values), "joints_num:", joints_num)
    #         return None
