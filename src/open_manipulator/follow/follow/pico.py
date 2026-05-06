# --- START OF FILE pico.py ---

#视觉依赖
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe import Image as mp_Image
from mediapipe import ImageFormat as mp_ImageFormat

#绘图依赖
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from .leap_hand_utils.dynamixel_client import *
from .leap_hand_utils import leap_hand_utils as lhu
from .leap_hand_utils.fileter import OneEuroFilter,LowPassFilter,LandmarkSmoother
import pyorbbecsdk as obs
import time
import threading



#ROS2依赖
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory 

#辅助功能依赖
import threading
import queue
import datetime
import os
from scipy.spatial.transform import Rotation as R
class Publish(Node):
    def __init__(self,name):

        super().__init__(name)
        """通信处理——通过定义Publisher和Subscription实现读取和发布"""
        
        #声明两个用于存储电机数据的参数，两个参数是名称和默认值。
        self.declare_parameter('arm_topic', '/arm_controller/joint_trajectory')
        arm_topic = self.get_parameter('arm_topic').value
        self.arm_publisher = self.create_publisher(JointTrajectory, arm_topic, 10)


        self.declare_parameter('command_topic', '/hand_controller/joint_trajectory')
        command_topic = self.get_parameter('command_topic').value
        self.hand_publisher = self.create_publisher(JointTrajectory, command_topic, 10)  


        #-----------------------------------
        #订阅者创建

        qos_profile = QoSProfile(
        depth=10,  # 队列深度
        reliability=ReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,  # 设置可靠性为Reliable
        history=HistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,  # 设置只保留最近的消息
            )
        
        self.subscription = self.create_subscription(
        JointState,
        '/joint_states',
        self.controller_state_callback,
        qos_profile
        )
        self.subscription  # prevent unused variable warning

        self.raw_positions=None

        #订阅者创建
        #-----------------------------------


        #-----------------------------------
        #订阅者回调函数
        
    def controller_state_callback(self,msg):

        if not isinstance(msg, JointState):
            self.get_logger().error("收到非 JointState 消息!")
            return

        try:
            self.raw_positions =(msg.position)
            self.raw_name = (msg.name)
            self.raw_positions,self.raw_arm_positions=self.log_to_sim(self.raw_name,self.raw_positions)
            self.raw_positions = np.reshape(self.raw_positions, (1, len(self.raw_positions)))
            self.raw_arm_positions = np.reshape(self.raw_arm_positions, (1, len(self.raw_arm_positions)))


        except Exception as e:
            self.get_logger().error(f"Positions message launching error: {str(e)}")

        #订阅者回调函数
        #-----------------------------------


        #-----------------------------------
        #订阅信息重排
    def log_to_sim(self,received_names: list,received_values: list) -> np.ndarray:
        #由于joint_states的通用性，同时接受手和臂的数据
        standard_names = [
        # 机械手关节
            'hand0', 'hand1', 'hand2', 'hand3', 'hand4',
            'hand5', 'hand6', 'hand7', 'hand8', 'hand9',
            'hand10', 'hand11', 'hand12', 'hand13', 'hand14', 'hand15',
        # 机械臂关节
            'joint1','joint2','joint3','joint4','joint5','joint6','joint7'
        ]

        assert len(received_names) == len(standard_names)

        # 提高效率：创建一个从名称到值的映射（字典），用于快速查找,避免了低效的嵌套循环
        value_map = dict(zip(received_names, received_values))

        # 构建结果：根据标准顺序从映射中提取值
        ordered_values = []


        for name in standard_names:

        # 使用 .get() 方法安全地查找值
            value = value_map.get(name)
            if value is not None:
                ordered_values.append(value)
            else:
                raise ValueError(
                    f"在收到的消息中找不到期望的关节 '{name}'。 "
                    f"可用的关节有: {list(value_map.keys())}"
                )

        # 注意要分别返回手和臂的关节数据    
        ordered_values = np.array(ordered_values)
        hand_values = ordered_values[:16]  # 前面十六个是手的关节
        arm_values = ordered_values[16:]   # 后面七个是臂的关节
        return hand_values, arm_values
    
        #订阅信息重排
        #-----------------------------------



    def transfer_angle(self,sequence):
        #将角度转换为0弧度制
        sequence=np.asarray(sequence)
        sequence=sequence/180*np.pi-np.pi
        return sequence
    

    def command_joint_position(self, angle,indices,speed):
        try:
            # 2. 创建JointTrajectory消息
            msg = JointTrajectory()
            # 3. 设置消息头
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'  # 根据实际情况设置坐标系
            # 4. 设置关节名称（必须与YAML配置完全一致）
            if len(indices) == 16:
                joint_names = [
                    'hand0', 'hand1', 'hand2', 'hand3', 'hand4',
                    'hand5', 'hand6', 'hand7', 'hand8', 'hand9',
                    'hand10', 'hand11', 'hand12', 'hand13', 'hand14', 'hand15',
                ]
            
            #如果不满足条件，弹出报错信息
            else :
                self.get_logger().error(f"Invalid joint indices length: {len(indices)}")
                return False

            msg.joint_names = joint_names
                
            run_time=0.0
            
            for i in range(angle.shape[0]):
                raw=angle[i]

                point=JointTrajectoryPoint()                
                point.positions=raw.tolist()
                run_time+=speed  # 假设每个点的运行时间为speed秒
                point.time_from_start = Duration(seconds=run_time).to_msg()  # 改为1秒完成
                msg.points.append(point)
   
            # 7. 发布消息
            self.hand_publisher.publish(msg)

            self.get_logger().debug('Published joint commands')
            return True
        except Exception as e:
            self.get_logger().error(f"Publishing error: {repr(e)}")
            return False
        


    def arm_command_joint_position(self, angle,indices,speed):
        try:
            # 2. 创建JointTrajectory消息
            msg = JointTrajectory()
            # 3. 设置消息头
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'  # 根据实际情况设置坐标系

            # 4. 设置关节名称（必须与YAML配置完全一致）

            if len(indices) == 7:
                joint_names = [
                    'joint1','joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7' 
                ]
            elif len(indices) == 5:
                joint_names = [
                    'joint2',
                    'joint3',
                    'joint4',
                    'joint5',
                    'joint7'
                ]
            else:
                self.get_logger().error(f"Invalid joint indices length: {len(indices)}")
                return False

            msg.joint_names = joint_names
                

            run_time=0.0

            for i in range(angle.shape[0]):
                raw=angle[i]

                # --------------------------
                # 检查共轭电机角度是否合规
                if len(indices)==7:
                    assert abs(raw[1] + raw[2]) < 0.035
                    assert abs(raw[3] + raw[4]) < 0.035
                #---------------------------
                
                point=JointTrajectoryPoint()                
                point.positions=raw.tolist()

                run_time+=speed  # 假设每个点的运行时间为speed秒
                point.time_from_start = Duration(seconds=run_time).to_msg()  
                msg.points.append(point)
                
            # 7. 发布消息
            self.arm_publisher.publish(msg)

            self.get_logger().debug('Published joint commands')
            return True
        except Exception as e:
            self.get_logger().error(f"Publishing error: {repr(e)}")
            return False
        
from std_msgs.msg import Float32MultiArray
from typing import List, Optional


class Joint:
    """关节点数据类型，包含x, y, z三个属性"""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class PicoJointsSubscriber(Node):
    """
    订阅pico_joints话题的ROS2节点类
    从26个关节点中读取指定的17个关节点数据
    跳过第1,7,12,17,22,23,24,25,26个关节点（1-based索引）
    """
    
    def __init__(self):
        super().__init__('pico_joints_subscriber')
        
        # 创建订阅器
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/pico_joints',
            self.joints_callback,
            10
        )
        
        # 存储最新的关节点数据
        self.latest_joints_data: Optional[List[Joint]] = None
        
        # 定义需要跳过的关节点索引（1-based转换为0-based）
        self.skip_indices = {0, 6, 11, 16, 21, 22, 23, 24, 25}  # 1,7,12,17,22,23,24,25,26 -> 0-based
        
        self.get_logger().info('Pico Joints Subscriber initialized')

        self.sequence=None

        pyplot.ion()  # 开启交互模式
        self.fig = pyplot.figure(1, figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(111, projection='3d')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.set_title('Hand Joint Points 3D Visualization')        
        self.get_logger().info('Pico Joints Subscriber initialized')

        self.sequence1=np.asarray([180]*21)
        self.sequence2=np.asarray([180]*21)
        self.sequence3=np.asarray([180]*21)
        self.sequence4=np.asarray([180]*21)
        self.sequence5=np.asarray([180]*21)
        self.sequence6=np.asarray([180]*21)
        self.sequence7=np.asarray([180]*21)
        self.sequence8=np.asarray([180]*21)
        self.sequence9=np.asarray([180]*21)
        self.sequence10=np.asarray([180]*21)


        
    def draw_plot(self):
        """
        绘制手部关节点的3D可视化
        参照原有的draw_plot函数结构，绘制关节点和连接线
        """
        if self.latest_joints_data is None or len(self.latest_joints_data) != 17:
            return
            
        points = self.latest_joints_data
        
        if True:
            # 清除绘图区
            pyplot.figure(1)
            pyplot.cla()
            
            # 重新设置坐标轴
            self.ax1 = self.fig.add_subplot(111, projection='3d')
            self.ax1.set_xlabel('X')
            self.ax1.set_ylabel('Y')
            self.ax1.set_zlabel('Z')
            self.ax1.set_title('Hand Joint Points 3D Visualization')
            
            # 画手部关节坐标点
            for id, i in enumerate(points):
                # 绘制所有17个关节点
                self.ax1.scatter3D(i.x, i.y, i.z, s=50, alpha=0.8)
            
            # 连接各个关节的线
            # 大拇指 [0, 1, 2, 3, 4]
            xdata = []
            ydata = []
            zdata = []
            for i in [0, 1, 2, 3, 4]:
                if i < len(points):
                    xdata.append(points[i].x)
                    ydata.append(points[i].y)
                    zdata.append(points[i].z)
            if len(xdata) > 1:
                self.ax1.plot(xdata, ydata, zdata, 'b-', linewidth=2, label='Thumb')

            # 食指 [0, 5, 6, 7, 8]
            xdata = []
            ydata = []
            zdata = []
            for i in [0, 5, 6, 7, 8]:
                if i < len(points):
                    xdata.append(points[i].x)
                    ydata.append(points[i].y)
                    zdata.append(points[i].z)
            if len(xdata) > 1:
                self.ax1.plot(xdata, ydata, zdata, 'r-', linewidth=2, label='Index')

            # 中指 [9, 10, 11, 12]
            xdata = []
            ydata = []
            zdata = []
            for i in [9, 10, 11, 12]:
                if i < len(points):
                    xdata.append(points[i].x)
                    ydata.append(points[i].y)
                    zdata.append(points[i].z)
            if len(xdata) > 1:
                self.ax1.plot(xdata, ydata, zdata, 'g-', linewidth=2, label='Middle')

            # 无名指 [0, 13, 14, 15, 16]
            xdata = []
            ydata = []
            zdata = []
            for i in [0, 13, 14, 15, 16]:
                if i < len(points):
                    xdata.append(points[i].x)
                    ydata.append(points[i].y)
                    zdata.append(points[i].z)
            if len(xdata) > 1:
                self.ax1.plot(xdata, ydata, zdata, 'm-', linewidth=2, label='Ring')

            # 连接手指根部的线 [5, 9, 13]
            xdata = []
            ydata = []
            zdata = []
            for i in [5, 9, 13]:
                if i < len(points):
                    xdata.append(points[i].x)
                    ydata.append(points[i].y)
                    zdata.append(points[i].z)
            if len(xdata) > 1:
                self.ax1.plot(xdata, ydata, zdata, 'k-', linewidth=2, alpha=0.7)

            # 手腕连接线 - 使用最后几个点作为手腕连接
            if len(points) >= 3:
                xdata = []
                ydata = []
                zdata = []
                # 使用点0和最后两个点作为手腕连接
                wrist_indices = [0]
                if len(points) > 15:
                    wrist_indices.extend([15, 16])  # 使用最后两个点
                
                for i in wrist_indices:
                    if i < len(points):
                        xdata.append(points[i].x)
                        ydata.append(points[i].y)
                        zdata.append(points[i].z)
                if len(xdata) > 1:
                    self.ax1.plot(xdata, ydata, zdata, 'c-', linewidth=3, alpha=0.8, label='Wrist')
            
            # 设置图例和显示
            self.ax1.legend()
            pyplot.draw()
            pyplot.pause(0.01)

    def joints_callback(self, msg: Float32MultiArray):
        """
        接收pico_joints话题数据的回调函数
        
        Args:
            msg: 包含78个数据的Float32MultiArray消息 (26个关节点 × 3个坐标)
        """
        try:
            data = msg.data
            
            # 验证数据长度
            if len(data) != 82:  # 26个关节点 × 3个坐标
                self.get_logger().error(f'Expected 78 data points, got {len(data)}')
                return
            
            # 解析关节点数据
            joints = []
            
            for i in range(26):  # 遍历26个关节点
                if i not in self.skip_indices:  # 跳过指定的关节点
                    x = data[i * 3]
                    y = data[i * 3 + 1]
                    z = data[i * 3 + 2]
                    joints.append(Joint(x=x, y=y, z=z))
            
            self.rot=[]
            for i in range(4):
                self.rot.append(data[i+78]) 


            self.latest_joints_data = joints
            self.get_logger().debug(f'Received {len(joints)} joint points')
            self.caculate_angle_list(self.latest_joints_data)

            self.sequence=self.get_sequence()


        except Exception as e:
            self.get_logger().error(f'Error processing joints data: {repr(e)}')


    def convert_quaternions_to_euler(self,quaternion_data):
        rotation = R.from_quat(quaternion_data)

        # 使用'zyx'顺序将四元数转换为欧拉角。
        # as_euler的输出顺序与指定的'zyx'对应，即 [yaw, pitch, roll]
        euler_angles = rotation.as_euler('zyx', degrees=True)

        # 为了更直观，我们将结果按 Roll, Pitch, Yaw 的顺序重新排列并命名
        yaw_z = euler_angles[0]
        pitch_y = euler_angles[1]
        roll_x = euler_angles[2]

        return yaw_z



    def caculate_angle_list(self,points):
            if points is not None and len(points) == 17:
                print("here")
                # 算大拇指角度
                self.thumb_angle_1 = 360-self.get_angle((points[0].x, points[0].y, points[0].z), 
                                        (points[1].x, points[1].y, points[1].z),
                                        (points[2].x, points[2].y, points[2].z))
                self.thumb_angle_2 = 360-self.get_angle((points[1].x, points[1].y, points[1].z),
                                        (points[2].x, points[2].y, points[2].z),
                                        (points[3].x, points[3].y, points[3].z))
                self.thumb_angle_3 = 360-self.get_angle((points[2].x, points[2].y, points[2].z),
                                        (points[3].x, points[3].y, points[3].z),
                                        (points[4].x, points[4].y, points[4].z))
                # 算食指角度

                self.index_finger_angle_1 = 360-self.get_angle((points[0].x, points[0].y, points[0].z),
                                        (points[5].x, points[5].y, points[5].z),
                                        (points[6].x, points[6].y, points[6].z))
                self.index_finger_angle_2 = 360-self.get_angle((points[5].x, points[5].y, points[5].z),
                                        (points[6].x, points[6].y, points[6].z),
                                        (points[7].x, points[7].y, points[7].z))
                self.index_finger_angle_3 = 360-self.get_angle((points[6].x, points[6].y, points[6].z),
                                        (points[7].x, points[7].y, points[7].z),
                                        (points[8].x, points[8].y, points[8].z))
                # 算中指角度
                self.middle_finger_angle_1 = 360-self.get_angle((points[0].x, points[0].y, points[0].z),
                                        (points[9].x, points[9].y, points[9].z),
                                        (points[10].x, points[10].y, points[10].z))
                self.middle_finger_angle_2 = 360-self.get_angle((points[9].x, points[9].y, points[9].z),
                                        (points[10].x, points[10].y, points[10].z),
                                        (points[11].x, points[11].y, points[11].z))
                self.middle_finger_angle_3 = 360-self.get_angle((points[10].x, points[10].y, points[10].z),
                                        (points[11].x, points[11].y, points[11].z),
                                        (points[12].x, points[12].y, points[12].z))
                # 算无名指角度
                self.ring_finger_angle_1 = 360-self.get_angle((points[0].x, points[0].y, points[0].z),
                                        (points[9].x, points[9].y, points[9].z),
                                        (points[10].x, points[10].y, points[10].z))
                self.ring_finger_angle_2 = 360-self.get_angle((points[9].x, points[9].y, points[9].z),
                                        (points[10].x, points[10].y, points[10].z),
                                        (points[11].x, points[11].y, points[11].z))
                self.ring_finger_angle_3 = 360-self.get_angle((points[10].x, points[10].y, points[10].z),
                                        (points[11].x, points[11].y, points[11].z),
                                        (points[12].x, points[12].y, points[12].z))
                
                #不是哥们手指间距有点小呀，在这里放缩一下
                scale=1

                
                #食指向拇指偏移，角度减小                
                self.angle_between_finger_1 = 185-self.get_angle_4p((points[6].x, points[6].y, points[6].z),
                                        (points[5].x, points[5].y, points[5].z),
                                        (points[9].x, points[9].y, points[9].z),
                                        (points[10].x, points[10].y, points[10].z))*scale

                #中指保持不变time.sleep
                self.angle_between_finger_2 = 180
                
                #无名指向拇指偏离，角度增大
                self.angle_between_finger_3 = 180+self.get_angle_4p((points[10].x, points[10].y, points[10].z),
                                        (points[9].x, points[9].y, points[9].z),
                                        (points[13].x, points[13].y, points[13].z),
                                        (points[14].x, points[14].y, points[14].z))*scale

                # self.angle_between_finger_3=180+self.angle_between(points[10], points[9], points[13], points[14])

                self.angle_between_finger_4=self.get_angle((points[5].x, points[5].y, points[5].z),
                                        (points[0].x, points[0].y, points[0].z),
                                        (points[1].x, points[1].y, points[1].z))
                # self.angle_between_finger_4=180+self.angle_between(points[3], points[2], points[5], points[6])
                
                #算手臂映射的角度
                #腕部关节的两个自由度
                self.angle_arm_wrist1=  self.convert_quaternions_to_euler(self.rot)+180

                self.angle_arm_wrist2=  180      #self.get_angle((points[9].x, points[9].y, points[9].z),
                                        #(points[0].x, points[0].y, points[0].z),
                                        #(points[22].x, points[22].y, points[22].z))
                

                #肘部关节的两个自由度
                self.angle_arm_elbow1=   180     #259-self.get_angle((points[0].x, points[0].y, points[0].z),
                                        # (points[22].x, points[22].y, points[22].z),
                                        # (points[0].x, points[22].y, points[0].z))
                self.angle_arm_elbow2=     180   #self.get_angle((points[0].x, points[0].y, points[0].z),
                                        # (points[22].x, points[22].y, points[22].z),
                                        # (points[21].x, points[21].y, points[21].z))+self.angle_arm_elbow1-70
                # for hand_landmarks in self.latest_hand_result.hand_landmarks:
                #     self.mp_drawing.draw_landmarks(self.frame1, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    #返回各个电机的角度列表
    def get_sequence(self):

        # <--- 新增/修改: 重构此方法以解决clip问题并提高可读性
        finger_sequence_raw = [180] * 21 # 如果没有检测到手，则使用默认值
        finger_sequence_raw = np.array(finger_sequence_raw)

        if self.latest_joints_data is not None :
            if self.thumb_angle_2 > 220:
                finger_sequence_raw = [
                    self.angle_between_finger_1,     self.index_finger_angle_1,      self.index_finger_angle_2,    self.index_finger_angle_3, 
                    180,                             self.middle_finger_angle_1,     self.middle_finger_angle_2,   self.middle_finger_angle_3,
                    self.angle_between_finger_3,     self.ring_finger_angle_1,       self.ring_finger_angle_2,     self.ring_finger_angle_3,
                    270,                             self.angle_between_finger_4,    self.thumb_angle_2,           self.thumb_angle_3,
                    180,           180,      180,        180,
                    self.angle_arm_wrist1
                ]
            else:
                finger_sequence_raw = [
                    self.angle_between_finger_1,     self.index_finger_angle_1,      self.index_finger_angle_2,    self.index_finger_angle_3,
                    180,                             self.middle_finger_angle_1,     self.middle_finger_angle_2,   self.middle_finger_angle_3,
                    self.angle_between_finger_3,     self.ring_finger_angle_1,       self.ring_finger_angle_2,     self.ring_finger_angle_3,
                    180,                             180,                            self.angle_between_finger_4,  self.thumb_angle_3,
                    180,           180,      180,        180,
                    self.angle_arm_wrist1
                ]
        
        # 步骤1: 仅对前16个手指电机角度应用clip
        finger_0 = np.clip(finger_sequence_raw[0:1], 160, 190)
        finger_4=np.clip(finger_sequence_raw[4:5], 160, 190)
        finger_8=np.clip(finger_sequence_raw[8:9], 160, 190)
        finger_hand_clipped1 = np.clip(finger_sequence_raw[1:4], 160, 270)
        finger_hand_clipped2 = np.clip(finger_sequence_raw[5:8], 160, 270)
        finger_hand_clipped3 = np.clip(finger_sequence_raw[9:16], 160, 270)
        finger_arm16_clipped  = np.clip(finger_sequence_raw[16:17], 0, 360)

        finger_arm171819_clipped = np.clip(finger_sequence_raw[17:20],100,260)
        finger_arm20_clipped     = np.clip(finger_sequence_raw[20:21],0,360)
        assert finger_sequence_raw[17]+finger_sequence_raw[18]==360

        # 拼接所有部分
        finger_clipped_all = np.concatenate([
            finger_0,
            finger_hand_clipped1,
            finger_4,
            finger_hand_clipped2,
            finger_8,
            finger_hand_clipped3,
            finger_arm16_clipped,
            finger_arm171819_clipped,
            finger_arm20_clipped
        ])
        self.suquence=finger_clipped_all
        self.finger_history=finger_clipped_all
        return self.suquence

    #给定三个点，算出角ABC
    def get_angle(self, A, B, C):
        # 将点的坐标转换为numpy数组
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)

        # 计算向量BA和BC
        BA = A - B
        BC = C - B

        # 计算向量的点积
        dot_product = np.dot(BA, BC)

        # 计算向量的模
        norm_BA = np.linalg.norm(BA)
        norm_BC = np.linalg.norm(BC)

        # 计算夹角的余弦值
        cos_theta = dot_product / (norm_BA * norm_BC)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        # 使用反余弦函数求出夹角
        theta = np.arccos(cos_theta)

        # 将弧度转换为度
        theta_degrees = np.degrees(theta)

        return theta_degrees
    
    def get_angle_4p(self, A, B, C, D):
        #这个函数用于计算手指之间的夹角，
        #返回值一定为一个正的锐角。中指不动，食指左偏，无名指右偏。
        # 将点的坐标转换为numpy数组
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)
        D = np.array(D)

        # 计算向量BA和CD
        BA = A - B
        CD = D - C

        # 计算向量的点积
        dot_product = np.dot(BA, CD)

        # 计算向量的模
        norm_BA = np.linalg.norm(BA)
        norm_CD = np.linalg.norm(CD)

        # 计算夹角的余弦值
        cos_theta = dot_product / (norm_BA * norm_CD)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        # 使用反余弦函数求出夹角
        theta = np.arccos(cos_theta)

        # 将弧度转换为度
        theta_degrees = np.degrees(theta)
        #确保角度大于0且为锐角
        if theta_degrees < 0:
            theta_degrees += 360
        if theta_degrees > 180:
            theta_degrees = 360 - theta_degrees
        if theta_degrees>90:
            theta_degrees = 180 - theta_degrees
        return theta_degrees
  
    def get_joints_list(self) -> Optional[List[Joint]]:
        """
        Returns:
            List[Joint]: 包含17个Joint对象的列表，如果没有数据则返回None
        """
        return self.latest_joints_data

    def has_valid_data(self) -> bool:
        """
        检查是否有有效的关节点数据
        
        Returns:
            bool: 如果有有效数据返回True，否则返回False
        """
        return self.latest_joints_data is not None and len(self.latest_joints_data) == 17

    def get_joint_by_index(self, index: int) -> Optional[Joint]:
        """
        根据索引获取特定的关节点
        
        Args:
            index: 关节点索引 (0-16)
            
        Returns:
            Joint: 指定索引的关节点，如果索引无效或无数据则返回None
        """
        if self.latest_joints_data is None:
            return None
        
        if 0 <= index < len(self.latest_joints_data):
            return self.latest_joints_data[index]
        
        return None

    def angle_filter(self,sequence,loop):
        if loop%10==0:
            self.sequence1=sequence
        if loop%10==1:
            self.sequence2=sequence
        if loop%10==2:
            self.sequence3=sequence
        if loop%10==3:
            self.sequence4=sequence
        if loop%10==4:
            self.sequence5=sequence
        if loop%10==5:
            self.sequence6=sequence
        if loop%10==6:
            self.sequence7=sequence
        if loop%10==7:
            self.sequence8=sequence
        if loop%10==8:
            self.sequence9=sequence
        if loop%10==9:
            self.sequence10=sequence



        sequence=(self.sequence1+self.sequence2+self.sequence3+self.sequence4+self.sequence5+self.sequence6+self.sequence7+self.sequence8+self.sequence9+self.sequence10)/10


        if loop<10:
            sequence=np.asarray([180]*21)
            
            return sequence
        else:
            return sequence
        

def main():
    rclpy.init()
    pico=PicoJointsSubscriber()
    publish=Publish("publish_node")
    loop=0
    #调试模式，不发送电机指令
    test_mode=False
    


    #--------------------------------
    #在这里调控电机数量
    single_indices=np.array([16,17,18,19,20,21,22])
    arm_indices = np.array([16,17,18,19,20,21,22])
    hand_indices = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    hand_indices = hand_indices[:16]  # 只使用前16个手指电机

    #在这里调控电机数量
    #--------------------------------

    #--------------------------------
    #在这里调控电机速度

    speed=0.1
    init_speed=3 #初始化时的电机速度
    number=10 #每存储number帧就进行一次播放

    #在这里调控电机速度
    #--------------------------------


    #--------------------------------
    #电机角度归零
    if not test_mode:
        publish.get_logger().info("正在等待来自 /joint_states 的第一条消息...")
        start_time = time.time()
        timeout = 5.0 # 5秒超时

        while publish.raw_positions is None and rclpy.ok():
            rclpy.spin_once(publish, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                publish.get_logger().error("在5秒内未接收到 /joint_states 消息，请检查发布器节点是否正常工作。")
                return

        initial_pose= np.zeros(23)
        initial_pose = initial_pose.reshape(1,23)

        initial_hand_pose = np.concatenate(( publish.raw_positions, initial_pose[:,:16]),0)
        initial_arm_pose = np.concatenate((publish.raw_arm_positions, initial_pose[:,16:]), 0)  # 添加机械臂的初始角度
        initial_arm_pose[:,5:6]=1.58

        publish.command_joint_position(initial_hand_pose, hand_indices,     1.5)
        publish.arm_command_joint_position(initial_arm_pose, arm_indices,   init_speed)

    #电机角度归零
    #--------------------------------

    track=[]
    time.sleep(6)

    history_sequence=[180]*21  # 初始化手指角度历史记录
    history_sequence=np.array(history_sequence)
    while True:
        loop+=1
        start_loop=datetime.datetime.now()


    #-------------------------------------------------
    # 获取手部关节数据
        if rclpy.ok():
            rclpy.spin_once(pico, timeout_sec=0.1)
        else:
            pico.get_logger().warn("PicoJointsSubscriber is not OK")

        if test_mode and pico.sequence is None:
            pico.get_logger().warn("test_mode setted")
        
        if np.any(np.isnan(pico.sequence)):
            pico.sequence = history_sequence  # 如果有NaN，使用上一次的角度



    #-------------------------------------------------
        pico.sequence=pico.angle_filter(pico.sequence,loop)  # 对角度进行滤波处理
        angle_rule=publish.transfer_angle(pico.sequence)
        print("raw_angle:",angle_rule.shape)
        angle_rule = np.insert(angle_rule, len(angle_rule) - 1, [0, 1.58])
        print("angle_rule:",angle_rule)
    
        angle_rule= angle_rule.tolist()
        track.append(angle_rule) 


        #每存储十帧就进行一次播放,播放后清空数组
        if loop%number==0 and test_mode==False:
            print("publish")
            track= np.vstack(track)
            # print(track)

            publish.command_joint_position(track[:,:16],hand_indices,speed)

            print("hand:",track[:,:16])
            publish.arm_command_joint_position(track[:,16:],single_indices,speed)

            
            track=[]
        history_sequence=pico.suquence
        end_loop=datetime.datetime.now()

        # print("main_loop:",(end_loop-start_loop),"\n")

        # pico.draw_plot()

 
    print("while out")

if(__name__ == "__main__"):
    main()