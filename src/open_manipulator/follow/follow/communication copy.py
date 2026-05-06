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

#辅助功能依赖
import threading
import queue
import datetime
import os
from ament_index_python.packages import get_package_share_directory 




# class LeapNode:
#     def __init__(self):
#         ####Some parameters
#         # self.ema_amount = float(rospy.get_param('/leaphand_node/ema', '1.0')) #take only current
#         self.kP = 600
#         self.kI = 0
#         self.kD = 200
#         self.curr_lim = 350
#         self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(21))
           
#         #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
#         self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # 0-11 are the fingers, 12-15 are the thumb, 16-20 are the palm motors
#         try:
#             self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 1000000)
#             self.dxl_client.connect()
#         except Exception:
#             try:
#                 self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 1000000)
#                 self.dxl_client.connect()
#             except Exception:
#                 self.dxl_client = DynamixelClient(motors, 'COM13', 1000000)
#                 self.dxl_client.connect()
#         #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
#         self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
#         self.dxl_client.set_torque_enabled(motors, True)
#         self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
#         self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
#         self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
#         self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
#         self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
#         #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
#         self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
#         self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
#     #Receive LEAP pose and directly control the robot
#     def set_leap(self, pose):
#         self.prev_pos = self.curr_pos
#         self.curr_pos = np.array(pose)
#         self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
#     #allegro compatibility
#     def set_allegro(self, pose):
#         pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
#         self.prev_pos = self.curr_pos
#         self.curr_pos = np.array(pose)

#         self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
#     #Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
#     def set_ones(self, pose):
#         pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
#         self.prev_pos = self.curr_pos
#         self.curr_pos = np.array(pose)
#         self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
#     #read position
#     def read_pos(self):
#         return self.dxl_client.read_pos()
#     #read velocity
#     def read_vel(self):
#         return self.dxl_client.read_vel()
#     #read current
#     def read_cur(self):
#         return self.dxl_client.read_cur()

#     def set_degree(self, pose):
#         self.prev_pos = self.curr_pos
#         self.curr_pos = np.array(pose)
#         self.curr_pos = np.deg2rad(self.curr_pos)

#         self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    
class Hand_Process_And_Plot(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True) 
        self.running = True
        self.cameraqueue = queue.Queue(maxsize=60)#存取六十帧照片



        cv2.namedWindow("maincam", 1)
        self.cap1 = cv2.VideoCapture(0)
        pyplot.ion()
        self.ax1 = pyplot.axes(projection='3d')


        # --- 这是你需要修改的部分 ---
        # 1. 获取你的包的 'share' 目录的路径
        package_share_path = get_package_share_directory('follow')

        # 2. 构建模型的绝对路径
        hand_model_path = os.path.join(package_share_path, 'hand_landmarker.task')
        pose_model_path = os.path.join(package_share_path, 'pose_landmarker_full.task')



        # #模型路径
        # hand_model_path = 'hand_landmarker.task' #手部模型： 输入形状192 x 192、224 x 224 量化类型float16
        # pose_model_path = 'pose_landmarker_full.task'#人体姿势模型： 	姿势检测器：224 x 224 x 3  姿势地标：256 x 256 x 3 量化类型： float16



        #检测器回调函数
        def _hand_result_callback(result: vision.HandLandmarkerResult, output_image: mp_Image, timestamp_ms: int):
            with self.result_lock:
                self.latest_hand_result = result

        def _pose_result_callback(result: vision.PoseLandmarkerResult, output_image: mp_Image, timestamp_ms: int):
            with self.result_lock:
                self.latest_pose_result = result

        #初始化检测器
        # 配置手部检测器
        base_options_hand = python.BaseOptions(model_asset_path=hand_model_path)
        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            running_mode=vision.RunningMode.LIVE_STREAM, # 使用图像模式，逐帧处理
            num_hands=1,
            min_hand_detection_confidence=0.1,
            min_hand_presence_confidence=0.1,
            min_tracking_confidence=0.7,
            result_callback=_hand_result_callback
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # 配置姿态（手臂）检测器
        base_options_pose = python.BaseOptions(model_asset_path=pose_model_path)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=vision.RunningMode.LIVE_STREAM, # 使用图像模式，逐帧处理
            min_pose_detection_confidence=0.1,
            min_tracking_confidence=0.6,
            result_callback=_pose_result_callback
        )






        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        self.thumb_angle_1_list = []
        self.thumb_angle_2_list = []
        self.thumb_angle_3_list = []
        self.index_finger_angle_1_list = []
        self.index_finger_angle_2_list = []
        self.index_finger_angle_3_list = []
        self.middle_finger_angle_1_list = []
        self.middle_finger_angle_2_list = []
        self.middle_finger_angle_3_list = []
        self.ring_finger_angle_1_list = []
        self.ring_finger_angle_2_list = []
        self.ring_finger_angle_3_list = []
        self.angle_between_finger_1_list = []
        self.angle_between_finger_2_list = []
        self.angle_between_finger_3_list = []
        self.suquence = [180]*21
        self.finger_history= [180]*21
        self.finger_history=np.array(self.finger_history)
        self.result_lock = threading.Lock()
        self.mp_drawing = mp.solutions.drawing_utils
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



        ########以下为深度相机的初始化
        # 1. 初始化相机和配置
        self.pipeline = obs.Pipeline()
        self.config = obs.Config()
        
        # 获取设备并创建流配置
        self.profile_list = self.pipeline.get_stream_profile_list(obs.OBSensorType.COLOR_SENSOR)
        self.color_profile = self.profile_list.get_video_stream_profile(640, 480, obs.OBFormat.RGB, 30)
        self.config.enable_stream(self.color_profile)
        
        # 同样配置深度流
        self.profile_list = self.pipeline.get_stream_profile_list(obs.OBSensorType.DEPTH_SENSOR)
        self.depth_profile = self.profile_list.get_video_stream_profile(640, 400, obs.OBFormat.Y16, 30)
        self.config.enable_stream(self.depth_profile)

        # 关键步骤：启用D2C（Depth to Color）对齐
        # 这告诉SDK我们希望深度图对齐到彩色图的视角
        self.config.set_align_mode(obs.OBAlignMode.HW_MODE) # Orbbec提供了硬件和软件对齐模式

        #配置相机内参，用于深度坐标的转换
        self.color_intrinsics = self.color_profile.get_intrinsic()
        # 2. 启动管道
        self.pipeline.start(self.config)



        #####滤波器初始化
        self.smoother = LandmarkSmoother(
            num_landmarks=23, 
            num_dims=3, 
            freq=30.0,      # 与你的相机帧率匹配
            min_cutoff=1.0, # 静止时去抖强度，值越小越平滑
            beta=0.6        # 移动时响应速度，值越大延迟越小
        )

       

    def get_frames(self):
        """从深度相机获取并返回对齐的彩色和深度帧及数据。"""
        if not self.pipeline:
            return None, None, None

        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None, None
        
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None

        # 将帧数据转换为Numpy数组
        color_data = np.asanyarray(color_frame.get_data()) # 直接是 (480, 640, 3) RGB
        color_image = color_data.reshape((480, 640, 3))


        # 对深度图进行处理
        data_uint8 = np.asanyarray(depth_frame.get_data())
        # 2. 然后用.view()改变对内存的看法
        data_uint16 = data_uint8.view(dtype=np.uint16)
        # 3. 现在，data_uint16的元素数量就是307,200了，可以安全地reshape
        depth_image = data_uint16.reshape((480, 640))
        # 4. 获取深度图的缩放因子
        depth_scale = depth_frame.get_depth_scale()

        return color_image, depth_image, depth_scale


    # #获取相机图像
    # def get_camera(self):
    #     self.ret, self.frame1 = self.cap1.read()
    #     self.frame1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)

    #处理手部，得出关键点坐标
    def hand_process(self,color_image,depth_image,depth_scale):
        
         # 1. 创建时间戳
        timestamp_ms = int(time.time() * 1000)

        mp_image = mp_Image(image_format=mp_ImageFormat.SRGB, data=color_image)
        self.frame1=color_image

        # 2. 发送前清除旧结果，并下达异步处理命令
        with self.result_lock: self.latest_hand_result = None
        self.hand_landmarker.detect_async(mp_image, timestamp_ms)

        # 3. 等待回调函数送来结果（带超时保护）
        timeout = time.time() + 0.5
        while time.time() < timeout:
            with self.result_lock:
                if self.latest_hand_result is not None: 
                    #print("found hand result")
                    break
            
            time.sleep(0.001)   
        hand_result = self.latest_hand_result

        point_3d_list = []

        cx = self.color_intrinsics.cx
        cy = self.color_intrinsics.cy
        fx = self.color_intrinsics.fx
        fy = self.color_intrinsics.fy
            
        for hand_landmarks in hand_result.hand_landmarks:
            for landmark in hand_landmarks:
                # 将归一化坐标转换为像素坐标
                    u = int(landmark.x * 640)
                    v = int(landmark.y * 480)

                    # 检查坐标是否在图像范围内
                    if 0 <= u < 640 and 0 <= v < 480:
                        depth_value = depth_image[v, u]
                        real_depth_in_meters = depth_value * depth_scale
                        
                        # 如果深度值为0，说明这个点没有有效的深度信息，跳过
                        if real_depth_in_meters > 0:
                            # 逆投影公式
                            Z = real_depth_in_meters
                            X = (u - cx) * Z / fx
                            Y = (v - cy) * Z / fy
                            # Orbbec SDK 有一个 Point3D 类，但用字典或元组更简单
                            # 为了和您原有的 .x .y .z 访问方式兼容，我们模拟一个简单的对象
                            point_3d = type('Point3D', (), {'x': X, 'y': Y, 'z': Z})()
                            point_3d_list.append(point_3d)
                        else:
                            point_3d_list.append(None) # 无有效深度，添加占位符
                    else:
                        point_3d_list.append(None) # 坐标出界，添加占位符
        
        return point_3d_list, hand_result
    
    def arm_process(self, color_image_rgb, depth_image, depth_scale):
        """使用MediaPipe PoseLandmarker检测手臂，并返回3D坐标列表。"""
        # 1. 创建时间戳
        timestamp_ms = int(time.time() * 1000)
        mp_image = mp_Image(image_format=mp_ImageFormat.SRGB, data=color_image_rgb)
        
        # 2. 发送前清除旧结果，并下达异步处理命令
        with self.result_lock: self.latest_pose_result = None
        self.pose_landmarker.detect_async(mp_image, timestamp_ms)

        # 3. 等待回调函数送来结果（带超时保护）
        timeout = time.time() + 0.5
        while time.time() < timeout:
            with self.result_lock:
                if self.latest_pose_result is not None: break
            time.sleep(0.001)

        pose_result = self.latest_pose_result




        point_3d_list = []
        
        # 我们只关心手臂的几个关键点: 肩膀, 肘部, 手腕
        # 在PoseLandmarker中, 右臂对应索引 12 (肩), 14 (肘)
        ARM_INDICES = [12, 14]


        cx = self.color_intrinsics.cx
        cy = self.color_intrinsics.cy
        fx = self.color_intrinsics.fx
        fy = self.color_intrinsics.fy
        # 处理检测结果
        if pose_result.pose_landmarks:
            # 提取指定索引的关键点
            for i in ARM_INDICES:
                landmark = pose_result.pose_landmarks[0][i] # 使用第一个检测到的人
                if landmark.visibility > 0.5: # 只使用可见度高的点
                    u = int(landmark.x * 640)
                    v = int(landmark.y * 480)

                    if 0 <= u < 640 and 0 <= v < 480:
                        depth_value = depth_image[v, u]
                        real_depth_in_meters = depth_value * depth_scale
                        
                        # 如果深度值为0，说明这个点没有有效的深度信息，跳过
                        if real_depth_in_meters > 0:
                            # 逆投影公式
                            Z = real_depth_in_meters
                            X = (u - cx) * Z / fx
                            Y = (v - cy) * Z / fy
                            # Orbbec SDK 有一个 Point3D 类，但用字典或元组更简单
                            # 为了和您原有的 .x .y .z 访问方式兼容，我们模拟一个简单的对象
                            point_3d = type('Point3D', (), {'x': X, 'y': Y, 'z': Z})()
                            point_3d_list.append(point_3d)
                        else:
                            point_3d_list.append(None) # 无有效深度，添加占位符
                    else:
                        point_3d_list.append(None) # 坐标出界，添加占位符
        
        return point_3d_list, pose_result

    
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
    
    #计算画手部3D骨架图
    def draw_plot(self):

        points=self.drawpoints
        if True:
            #清除绘图区
            pyplot.figure(1)
            pyplot.cla()
            pyplot.figure(2)
            pyplot.cla()
            #画手部关节坐标点

            for id,i in enumerate(points):
                if(id!=17 and id!=18 and id!=19 ):
                    self.ax1.scatter3D(i.x, i.y, i.z)
                #连接各个关节的线
                xdata = []
                ydata = []
                zdata = []
                for i in [0, 1, 2, 3, 4]:
                    xdata.append(points[i].x)
                    ydata.append(points[i].y)
                    zdata.append(points[i].z)
                self.ax1.plot(xdata,ydata,zdata)

                xdata = []
                ydata = []
                zdata = []
                for i in [0, 5, 6, 7, 8]:
                    xdata.append(points[i].x)
                    ydata.append(points[i].y)
                    zdata.append(points[i].z)
                self.ax1.plot(xdata,ydata,zdata)

                xdata = []
                ydata = []
                zdata = []
                for i in [9, 10, 11, 12]:
                    xdata.append(points[i].x)
                    ydata.append(points[i].y)
                    zdata.append(points[i].z)
                self.ax1.plot(xdata,ydata,zdata)

                xdata = []
                ydata = []
                zdata = []
                for i in [0, 13, 14, 15, 16]:
                    xdata.append(points[i].x)
                    ydata.append(points[i].y)
                    zdata.append(points[i].z)
                self.ax1.plot(xdata, ydata, zdata)

                xdata = []
                ydata = []
                zdata = []
                for i in [5, 9, 13]:
                    xdata.append(points[i].x)
                    ydata.append(points[i].y)
                    zdata.append(points[i].z)
                self.ax1.plot(xdata, ydata, zdata)

                for i in [0,21,22]:
                    xdata.append(points[i].x)
                    ydata.append(points[i].y)
                    zdata.append(points[i].z)
                self.ax1.plot(xdata, ydata, zdata)

    def angle_between(self,p1,p2,p3,p4):

        # --- 第一步：将输入点转换为 NumPy 数组 ---
        # NumPy 数组让我们能方便地进行向量运算。
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        p3 = np.asarray(p3)
        p4 = np.asarray(p4)

        # --- 第二步：计算两条线的方向向量 ---
        # 方向向量 = 终点坐标 - 起点坐标
        v1 = p2 - p1
        v2 = p4 - p3

        # --- 第三步：计算点积 ---
        # np.dot() 函数可以计算两个向量的点积。
        dot_product = np.dot(v1, v2)

        # --- 第四步：计算向量的模（长度） ---
        # np.linalg.norm() 函数可以计算向量的模。
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # --- 边缘情况处理 ---
        # 如果任何一个向量的长度为零（即两个点重合），则无法定义角度。
        # 在这种情况下，我们返回 0 度或抛出错误。这里我们返回 0。
        if norm_v1 == 0 or norm_v2 == 0:
            print("警告: 其中一条线的两个点重合，无法确定方向。返回0度。")
            return 0.0

        # --- 第五步：计算夹角的余弦值 ---
        # cos(theta) = |v1 · v2| / (|v1| * |v2|)
        # 使用 abs() 来确保我们得到的是锐角（0-90度）。
        cosine_angle = abs(dot_product) / (norm_v1 * norm_v2)
        
        # --- 浮点数精度修正 ---
        # 由于计算中可能存在的浮点数误差，cosine_angle 可能会略大于1.0，
        # 这会导致 arccos 函数出错。我们将其限制在 [-1.0, 1.0] 的范围内。
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        # --- 第六步：计算角度并转换为度数 ---
        # 使用 arccos 计算弧度，然后用 np.degrees() 转换为度。
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg
        
    def caculate_angle_list(self,points):
            if self.latest_hand_result.hand_landmarks:
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
                
                #算手指之间的角度
                # self.angle_between_finger_1 = 180-self.angle_between(points[6], points[5], points[9],points[10])
                

                #不是哥们手指间距有点小呀，在这里放缩一下
                scale=1.5

                
                #食指向拇指偏移，角度减小                
                self.angle_between_finger_1 = 180-self.get_angle_4p((points[6].x, points[6].y, points[6].z),
                                        (points[5].x, points[5].y, points[5].z),
                                        (points[9].x, points[9].y, points[9].z),
                                        (points[10].x, points[10].y, points[10].z))*scale

                #中指保持不变
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
                self.angle_arm_wrist2=  180      #self.get_angle((points[9].x, points[9].y, points[9].z),
                                        #(points[0].x, points[0].y, points[0].z),
                                        #(points[22].x, points[22].y, points[22].z))
                
                self.angle_arm_wrist1=   180    # self.get_angle((points[24].x, points[24].y, points[24].z),
                                        # (points[23].x, points[23].y, points[23].z),
                                        # (points[22].x, points[22].y, points[22].z))
                #肘部关节的两个自由度
                self.angle_arm_elbow1=        259-self.get_angle((points[0].x, points[0].y, points[0].z),
                                        (points[22].x, points[22].y, points[22].z),
                                        (points[0].x, points[22].y, points[0].z))
                self.angle_arm_elbow2=        self.get_angle((points[0].x, points[0].y, points[0].z),
                                        (points[22].x, points[22].y, points[22].z),
                                        (points[21].x, points[21].y, points[21].z))+self.angle_arm_elbow1-70
                # for hand_landmarks in self.latest_hand_result.hand_landmarks:
                #     self.mp_drawing.draw_landmarks(self.frame1, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)


                


    #返回各个电机的角度列表
    def get_sequence(self):

        # <--- 新增/修改: 重构此方法以解决clip问题并提高可读性
        finger_sequence_raw = [180] * 21 # 如果没有检测到手，则使用默认值
        finger_sequence_raw = np.array(finger_sequence_raw)

        if self.latest_hand_result.hand_landmarks:
            if self.thumb_angle_2 > 220:
                finger_sequence_raw = [
                    self.angle_between_finger_1,     self.index_finger_angle_1,      self.index_finger_angle_2,    self.index_finger_angle_3, 
                    180,                             self.middle_finger_angle_1,     self.middle_finger_angle_2,   self.middle_finger_angle_3,
                    self.angle_between_finger_3,     self.ring_finger_angle_1,       self.ring_finger_angle_2,     self.ring_finger_angle_3,
                    270,                             self.angle_between_finger_4,    self.thumb_angle_2,           self.thumb_angle_3,
                    self.angle_arm_elbow2,           360-self.angle_arm_elbow1,      self.angle_arm_elbow1,        self.angle_arm_wrist2,
                    self.angle_arm_wrist1
                ]
            else:
                finger_sequence_raw = [
                    self.angle_between_finger_1,     self.index_finger_angle_1,      self.index_finger_angle_2,    self.index_finger_angle_3,
                    180,                             self.middle_finger_angle_1,     self.middle_finger_angle_2,   self.middle_finger_angle_3,
                    self.angle_between_finger_3,     self.ring_finger_angle_1,       self.ring_finger_angle_2,     self.ring_finger_angle_3,
                    180,                             180,                            self.angle_between_finger_4,  self.thumb_angle_3,
                    self.angle_arm_elbow2,           360-self.angle_arm_elbow1,      self.angle_arm_elbow1,        self.angle_arm_wrist2,
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

            
    def run(self):
        while True:
            start=datetime.datetime.now()
            mark=1
                    
            color_image,depth_image,depth_scale=self.get_frames()

            if color_image is None or depth_image is None:
                print("Failed to get frames from camera.")
                continue
            
            hand_points,hand=self.hand_process(color_image,depth_image,depth_scale)
            # arm_points,arm=self.arm_process(color_image,depth_image,depth_scale)
            constant=hand_points[19:21]
            points=hand_points+constant#arm_points

            

            # if first==0:
                # history_sequence=[180]*21

            #处理识别不到手的情况
                #情况1,这里的23是特征点，根电机数量无关。无需改动。
            if len(points)!=23:
                mark=0
                # sequence=history_sequence

                #情况2

            for point in points:
                if point==None:
                    mark=0
                    # sequence=history_sequence
                    print("no hand and arm")
                    break
            
            #能识别到手，三维坐标滤波+角度滤波
            if mark==1:
                self.plotpoints=points

                #滤波部分
                
                # raw_points_np = np.zeros((23, 3))
                raw_points_np = np.zeros((len(points), 3)) # 动态调整大小以适应实际点数
                for i, p in enumerate(points):
                    if p is not None:
                        raw_points_np[i] = [p.x, p.y, p.z]
                smoothed_points_np = self.smoother(raw_points_np)
                    
                    # 步骤 4.3: 将平滑后的Numpy数组转换回Point3D对象列表，以便与你旧代码兼容
                points_for_processing = []
                Point3D = type('Point3D', (), {}) # 创建一个简单的类
                for i in range(smoothed_points_np.shape[0]):
                    p = Point3D()
                    p.x, p.y, p.z = smoothed_points_np[i]
                    points_for_processing.append(p)
                
                # 更新历史记录
                points = points_for_processing
                self.caculate_angle_list(points)
                self.drawpoints=points
                
                sequence = self.get_sequence()
                print("get")
                # history_sequence= sequence
                try:
                    # 尝试用非阻塞方式放入新信息
                    self.cameraqueue.put_nowait(sequence)

                
                except queue.Full:
                # 如果队列满了，什么也不做，直接进入下一次循环去处理更更新的帧。
                # 这种“丢帧”策略是保证系统实时性的关键。
                # 你可以在这里加一个打印语句用于调试，但生产环境中通常省略。
                # print("Queue is full, dropping a frame to maintain real-time performance.")
                    pass
            
            end=datetime.datetime.now()
            # print("camera_capture:",(end-start),"\n")

    def angle_filter(self,sequence,history_sequence,loop):
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
        
        #上面函数的返回值是numpy数组。


from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

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
            print("here")
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

            else:
                self.get_logger().error(f"Invalid joint indices length: {len(indices)}")
                return False

            msg.joint_names = joint_names
                

            run_time=0.0

            for i in range(angle.shape[0]):
                raw=angle[i]

                # --------------------------
                # 检查共轭电机角度是否合规
                assert abs(raw[1] + raw[2]) < 0.035
                assert abs(raw[3] + raw[4]) < 0.035
                #---------------------------
                
                point=JointTrajectoryPoint()                
                point.positions=raw.tolist()

                run_time+=speed  # 假设每个点的运行时间为speed秒
                point.time_from_start = Duration(seconds=run_time).to_msg()  # 改为1秒完成
                msg.points.append(point)
                
            # 7. 发布消息
            self.arm_publisher.publish(msg)

            self.get_logger().debug('Published joint commands')
            return True
        except Exception as e:
            self.get_logger().error(f"Publishing error: {repr(e)}")
            return False
        
        


        


def main():
    rclpy.init()
    hand_process_and_plot = Hand_Process_And_Plot()
    # leapnode = LeapNode()
    publish=Publish("publish_node")
    hand_process_and_plot.start()
    loop=0
    history_sequence=np.asarray([180]*21)

    #调试模式，不发送电机指令
    test_mode=False

    #--------------------------------
    #在这里调控电机数量

    arm_indices = np.array([16,17,18,19,20,21,22])
    hand_indices = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    hand_indices = hand_indices[:16]  # 只使用前16个手指电机

    #在这里调控电机数量
    #--------------------------------


    
    #--------------------------------
    #在这里调控电机速度

    speed=0.033333
    # arm_speed=0.033333 #机械臂的电机速度
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
    time.sleep(4)

    while True:
        start_loop=datetime.datetime.now()



        #--------------------------------------------------
        # 在显示前将其从 RGB 转换为 BGR
        image_to_show_rgb = hand_process_and_plot.frame1

        if image_to_show_rgb is not None:
             image_to_show_bgr = cv2.cvtColor(image_to_show_rgb, cv2.COLOR_RGB2BGR)
             cv2.imshow('maincam', image_to_show_bgr) # <--- 显示转换后的图像
        # hand_process_and_plot.draw_plot()

        #--------------------------------------------------


        loop+=1
        sequence=hand_process_and_plot.cameraqueue.get()
        sequence= hand_process_and_plot.angle_filter(sequence, history_sequence, loop)

        angle_rule=publish.transfer_angle(sequence)

        #以每秒30帧的速度存取轨迹点 

        angle_rule= angle_rule.tolist()
        track.append(angle_rule[:16])  # 只存储手部的角度数据

        #每存储十帧就进行一次播放,播放后清空数组
        if loop%number==0 :
            print("publish")
            track= np.vstack(track)
            print("track:",track.shape)
            print(track)

            publish.command_joint_position(track,hand_indices,speed)
            # publish.arm_command_joint_position(track[len(hand_indices):],arm_indices,arm_speed)
            time.sleep(1)
            track=[]


        if cv2.waitKey(1) & 0xFF == 27:
            break
        pyplot.pause(0.01)
        end_loop=datetime.datetime.now()
        print("main_loop:",(end_loop-start_loop),"\n")


 
    print("while out")
    hand_process_and_plot.cap1.release()

if(__name__ == "__main__"):
    main()