# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


from matplotlib import scale
from matplotlib.pyplot import sca
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32
from control_msgs.msg import DynamicJointState

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import time
#from builtin_interfaces.msg import Duration
from rclpy.duration import Duration
class LeapArm(Node):
    def __init__(self,name):
        super().__init__(name)
        
        #声明两个用于存储电机数据的参数，两个参数是名称和默认值。
        self.declare_parameter('command_topic', '/arm_controller/joint_trajectory')
        command_topic = self.get_parameter('command_topic').value
        self.publisher = self.create_publisher(JointTrajectory, command_topic, 10)
        self.joints_num = 7
        self.raw_positions = None



        qos_profile = QoSProfile(
            depth=10,  # 队列深度
            reliability=ReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,  # 设置可靠性为Reliable
            history=HistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,  # 设置只保留最近的消息
        )

        self.joints_to_monitor = ['dxl1','dxl5','dxl9','dxl8','dxl12','dxl13','dxl14','dxl15',
                                  'dxl17','dxl18','dxl19','dxl20', 'dxl21']
        self.attributes_to_monitor = ['Present Position', 'Present Current']
        
        self.subscription = self.create_subscription(
            DynamicJointState,
            '/dynamic_joint_states',
            self.current_callback,
            qos_profile
        )
        self.subscription


        #--------------------------------------------
        # 电流监测回调函数
    def current_callback(self, msg):
        """
        回调函数，用于处理来自/dynamic_joint_states话题的消息。
        它会查找指定的关节并将其属性存储到 self.log1 和 self.log2 数组中。
        self.log1 存储第一个属性 'Present Position' 的值。
        self.log2 存储第二个属性 'Present Current' 的值。
        """
        # 创建临时列表以收集当前消息的数据
        temp_log1 = []
        temp_log2 = []

        # 遍历所有需要监控的关节
        for joint_name in self.joints_to_monitor:
            try:
                # 1. 查找关节在消息中的索引
                joint_index = msg.joint_names.index(joint_name)
                interface_values = msg.interface_values[joint_index]

                # 2. 提取第一个要监控的属性
                try:
                    attribute_name_1 = self.attributes_to_monitor[0]
                    attribute_index_1 = interface_values.interface_names.index(attribute_name_1)
                    value1 = interface_values.values[attribute_index_1]
                    temp_log1.append(value1)
                except (ValueError, IndexError):
                    temp_log1.append(None) # 如果找不到属性，则添加占位符
                    self.get_logger().warn(f"在关节 '{joint_name}' 中未找到属性 '{self.attributes_to_monitor[0]}'")

                # 3. 提取第二个要监控的属性
                try:
                    attribute_name_2 = self.attributes_to_monitor[1]
                    attribute_index_2 = interface_values.interface_names.index(attribute_name_2)
                    value2 = interface_values.values[attribute_index_2]
                    temp_log2.append(value2)
                except (ValueError, IndexError):
                    temp_log2.append(None) # 如果找不到属性，则添加占位符
                    self.get_logger().warn(f"在关节 '{joint_name}' 中未找到属性 '{self.attributes_to_monitor[1]}'")

            except ValueError:
                # 如果在消息中未找到关节，为两个日志都添加占位符以保持数组对齐
                self.get_logger().warn(f"在接收到的DynamicJointState消息中未找到关节 '{joint_name}'", throttle_duration_sec=5)
                temp_log1.append(None)
                temp_log2.append(None)
            except IndexError:
                # 处理潜在的索引越界错误
                self.get_logger().error(f"处理关节 '{joint_name}' 时发生索引错误。")
                temp_log1.append(None)
                temp_log2.append(None)

        # 4. 将收集到的完整数据更新到实例变量中
        self.log1 = temp_log1
        self.log2 = temp_log2
        self.arm_current = self.log2[-5:]

        # 电流监测回调
        #--------------------------------------------

        #电机和isaac坐标映射，后面可能修改。 [1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7,16,17,18,19,20]
        self.sim_to_real_indices=[16, 17, 18, 19, 20, 21, 22, 1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7]
        self.real_to_sim_indices=[8, 7, 9, 10, 19, 20, 21, 22, 12, 11, 13, 14, 16, 15, 17, 18, 0, 1, 2, 3, 4, 5, 6]


        #定义电机数量。以便灵活调整手臂电机配合。


    def sim_to_real(self, values):
        return values[self.sim_to_real_indices]


    #-------------------------------------------
    # 电机数组必须只包含机械臂
    def command_joint_position(self, desired_pose,speed):
        scaled_pose=[]

        #--------------------------------------------
        # 在这里微调各个电机的角度！
        if desired_pose.shape[1] == self.joints_num:
            opration=np.array([0, -0.209, 0.209, -0.179, 0.179, 0.41, 0])

            assert desired_pose.shape[1] == self.joints_num

        print("Desired Pose:", desired_pose)
        #-------------------------------------------
        # 正式开始发送数据
        try:

            for i in range(desired_pose.shape[0]):
                rawdata=desired_pose[i,:]
                realdata = rawdata

                #-------------------------------------------
                #  电机角度微调部分，激活需要先保证电机数量一致
                if desired_pose.shape[1] == self.joints_num:  
                    realdata=realdata+opration
                    
            
                realdata = realdata.tolist()
                scaled_pose.append(realdata)
            
            scaled_pose = np.array(scaled_pose)
            
            msg = JointTrajectory()
            # 3. 设置消息头
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'  # 根据实际情况设置坐标系
                
            # 4. 设置关节名称（必须与YAML配置完全一致）
            msg.joint_names = [
                'joint1',
                'joint2',
                'joint3',
                'joint4',
                'joint5',
                'joint6',
                'joint7'
            ]
                
            # 5. 创建并填充轨迹点
            transtime=0.0
            for point in scaled_pose:
                
                # --------------------------
                # 检查共轭电机角度是否合规
                assert abs(point[1] + point[2]) < 0.05
                assert abs(point[3] + point[4]) < 0.05
                #---------------------------
                
                point_ja=JointTrajectoryPoint()
                point_ja.positions=point.tolist()                
                transtime+=speed
                point_ja.time_from_start = Duration(seconds=transtime).to_msg()  
                msg.points.append(point_ja)
            
            # 7. 发布消息
            self.publisher.publish(msg)
            print ('Published joint commands')
            self.get_logger().debug('Published joint commands')
            return True
        except Exception as e:
            self.get_logger().error(f"Publishing error: {repr(e)}")
            return False
        
    #-------------------------------------------
    # 电机数组必须只包含机械臂
    def recover(self, desired_pose,present_state,speed):
        scaled_pose=[]
        #--------------------------------------------
        # 在这里微调各个电机的角度！
        if desired_pose.shape[1] == self.joints_num:
            bias=np.array([    0,
                               0.03 , 0.03, #共轭，必须一正一负,注意，前一位大于上扬
                               0.02  ,0.02, #共轭，必须一正一负,注意，前一位增大上扬
                               0.045  ,0  
                               ])
            # downbias=np.array([0  ,
            #                    -0.015  ,-0.015  , #共轭，必须一正一负,注意，前一位大于下压
            #                    0.01  ,0.01, #共轭，必须一正一负,注意，前一位增大下压
            #                    0.05  ,0  ,
            #                    ])
            assert desired_pose.shape[1] == self.joints_num


        goal_pose= desired_pose[-1,:]

        
        if desired_pose.shape[1] == self.joints_num:
            opration=np.array([0, -0.212, 0.212, -0.180, 0.180, 0.21, 0])
            goal_pose = goal_pose + opration
            assert desired_pose.shape[1] == self.joints_num


        # -------------------------------------------
        # 进行电流监测。如果电流过大，根据电机差距情况进行调整
        rclpy.spin_once(self, timeout_sec=0.1)

        # delta = np.abs(goal_pose - present_state)  
        self.arm_current = np.array(self.arm_current)
        if np.any(self.arm_current > 245):
            self.get_logger().warn("电流过大，正在进行恢复操作。")
            indices = np.where(self.arm_current > 245)
            indices = indices[0]
            recover_bias=np.zeros(7)

        #-------------------------------------------
        # 只要共轭电机中有一个电机电流过大，就同时进行恢复操作
            for indice in indices:
                if indice==0 or indice==1: # dxl17, dxl8
                    recover_bias[1] = bias[1]
                    recover_bias[2] = bias[2]
                elif indice==2 or indice==3: # dxl18, dxl9
                    recover_bias[3] = bias[3]
                    recover_bias[4] = bias[4]
                else:
                    recover_bias[indice+1] = bias[indice+1]
                
        #-------------------------------------------
        # 计算恢复姿态

            new_recovery_pose = present_state + np.sign(goal_pose - present_state) * recover_bias
            new_recovery_pose = np.reshape(new_recovery_pose, (1, 7))

            goal_pose = np.reshape(goal_pose, (1, 7))
            desired_pose = np.concatenate((goal_pose, new_recovery_pose), axis=0)
            print("New recovery pose:", desired_pose)
        else:
            self.get_logger().info("目标姿态与当前姿态之间的差异在可接受范围内，无需恢复。")
            return True


        #-------------------------------------------
        # 正式开始发送数据
        try:
            for i in range(desired_pose.shape[0]):
                rawdata=desired_pose[i,:]
                realdata = rawdata

                realdata = realdata.tolist()
                scaled_pose.append(realdata)
            
            scaled_pose = np.array(scaled_pose)
            
            msg = JointTrajectory()
            # 3. 设置消息头
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'  # 根据实际情况设置坐标系
                
            # 4. 设置关节名称（必须与YAML配置完全一致）

            msg.joint_names = [
                'joint1',
                'joint2',
                'joint3',
                'joint4',
                'joint5',
                'joint6',
                'joint7'
            ]
            # 5. 创建并填充轨迹点
            transtime=0.0
            for point in scaled_pose:
                
                # --------------------------
                # 检查共轭电机角度是否合规
                assert abs(point[1] + point[2]) < 0.035
                assert abs(point[3] + point[4]) < 0.035
                #---------------------------
                
                point_ja=JointTrajectoryPoint()
                point_ja.positions=point.tolist()                
                transtime+=speed
                point_ja.time_from_start = Duration(seconds=transtime).to_msg()  
                msg.points.append(point_ja)
            
            # 7. 发布消息
            self.publisher.publish(msg)
            print ('Published joint commands')
            self.get_logger().debug('Published joint commands')
            return True
        except Exception as e:
            self.get_logger().error(f"Publishing error: {repr(e)}")
            return False
    

   
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

