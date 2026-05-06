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
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time

import numpy as np
#from builtin_interfaces.msg import Duration
from rclpy.duration import Duration
class LeapHand(Node):
    def __init__(self,name):
        super().__init__(name)
        """通信处理——通过定义Publisher和Subscription实现读取和发布"""
        
        #声明两个用于存储电机数据的参数，两个参数是名称和默认值。
        self.declare_parameter('command_topic', '/hand_controller/joint_trajectory')
        self.declare_parameter('state_topic', '/hand_joint_states')
        self.command_topic = self.get_parameter('command_topic').value
        self.state_topic = self.get_parameter('state_topic').value

        #创建发布器，Publisher=create_publisher(msg_type,topic_name,Qosfile)
        #Qos可能会影响性能，后面可能微调
        self.publisher = self.create_publisher(JointTrajectory, self.command_topic, 10)

        qos_profile = QoSProfile(
            depth=10,  # 队列深度
            reliability=ReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,  # 设置可靠性为Reliable
            history=HistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,  # 设置只保留最近的消息
            )
        
        
        self.subscription = self.create_subscription(
            JointState,
            self.state_topic,
            self.controller_state_callback,
            qos_profile
        )
        
        #读取电机状态的订阅器，订阅话题为/hand_joint_states，避免与Piper机械臂冲突
        self.raw_positions = None
        self.real_raw_positions = None



        self.sim_to_real_indices=[1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7]
        self.real_to_sim_indices=[1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]

        self.log_to_real_indices = [8, 12, 11, 0, 2, 1, 13, 3, 4, 15, 6, 7, 9, 5, 14, 10]

        #定义灵巧手电机数量。
        self.joints_num = 16

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
             
        except Exception as e:
            self.get_logger().error(f"Positions message launching error: {repr(e)}")

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
        opration=np.array([0,0.1,0,0,
                           0,0.1,0,0,
                           0,0.1,0,0,
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
                point_ja.positions=point.tolist()
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
