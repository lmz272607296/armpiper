
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe import Image as mp_Image
from mediapipe import ImageFormat as mp_ImageFormat


from matplotlib import pyplot
import numpy as np
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
from leap_hand_utils.fileter import OneEuroFilter,LowPassFilter,LandmarkSmoother
import pyorbbecsdk as obs
import time
import threading
import queue


class LeapNode:
    def __init__(self):
        ####Some parameters
        # self.ema_amount = float(rospy.get_param('/leaphand_node/ema', '1.0')) #take only current
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(21))
           
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]#,16,17,18,19,20] # 0-11 are the fingers, 12-15 are the thumb, 16-20 are the palm motors
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 1000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 1000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM13', 1000000)
                self.dxl_client.connect()
        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #allegro compatibility
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)

        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #read position
    def read_pos(self):
        return self.dxl_client.read_pos()
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()

    def set_degree(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.curr_pos = np.deg2rad(self.curr_pos)

        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    
class Hand_Process_And_Plot(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True) 
        self.running = True
        self.cameraqueue = queue.Queue(maxsize=10)



        cv2.namedWindow("maincam", 1)
        self.cap1 = cv2.VideoCapture(0)
        pyplot.ion()
        self.ax1 = pyplot.axes(projection='3d')

        #模型路径
        hand_model_path = 'hand_landmarker.task' #手部模型： 输入形状192 x 192、224 x 224 量化类型float16
        pose_model_path = 'pose_landmarker_full.task'#人体姿势模型： 	姿势检测器：224 x 224 x 3  姿势地标：256 x 256 x 3 量化类型： float16



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
            min_tracking_confidence=0.4,
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
        self.sequence1=[180]*21
        self.sequence2=[180]*21
        self.sequence3=[180]*21
        self.sequence4=[180]*21
        self.sequence5=[180]*21
        self.sequence6=[180]*21
        self.sequence7=[180]*21
        self.sequence8=[180]*21
        self.sequence9=[180]*21
        self.sequence10=[180]*21



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
    
    #计算画手部3D骨架图
    def draw_plot(self, points):

        if True:
            #清除绘图区
            pyplot.figure(1)
            pyplot.cla()
            pyplot.figure(2)
            pyplot.cla()
            #画手部关节坐标点

            for id,i in enumerate(points):
                if(id!=17 and id!=18 and id!=19 and id!=21 and id!=22):
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
                self.angle_between_finger_1 = (180-(self.get_angle((points[6].x, points[6].y, points[6].z),
                                        (points[5].x, points[5].y, points[5].z),
                                        (points[9].x, points[9].y, points[9].z))+100))*1.5+180

                self.angle_between_finger_2 = (180-(self.get_angle((points[13].x, points[13].y, points[13].z),
                                        (points[9].x, points[9].y, points[9].z),
                                        (points[10].x, points[10].y, points[10].z))+87))*1.5+180

                self.angle_between_finger_3 = (180-(self.get_angle((points[17].x, points[17].y, points[17].z),
                                        (points[13].x, points[13].y, points[13].z),
                                        (points[14].x, points[14].y, points[14].z))+65))+180

                self.angle_between_finger_4=self.get_angle((points[5].x, points[5].y, points[5].z),
                                        (points[0].x, points[0].y, points[0].z),
                                        (points[1].x, points[1].y, points[1].z))
                

                #算手臂映射的角度
                #腕部关节的两个自由度
                self.angle_arm_wrist2=        self.get_angle((points[9].x, points[9].y, points[9].z),
                                        (points[0].x, points[0].y, points[0].z),
                                        (points[22].x, points[22].y, points[22].z))
                
                self.angle_arm_wrist1=   180    # self.get_angle((points[24].x, points[24].y, points[24].z),
                                        # (points[23].x, points[23].y, points[23].z),
                                        # (points[22].x, points[22].y, points[22].z))
                #肘部关节的两个自由度
                self.angle_arm_elbow1=        90+self.get_angle((points[0].x, points[0].y, points[0].z),
                                        (points[22].x, points[22].y, points[22].z),
                                        (points[0].x, points[22].y, points[0].z))
                self.angle_arm_elbow2=        self.get_angle((points[0].x, points[0].y, points[0].z),
                                        (points[22].x, points[22].y, points[22].z),
                                        (points[21].x, points[21].y, points[21].z))+self.angle_arm_elbow1
                # for hand_landmarks in self.latest_hand_result.hand_landmarks:
                #     self.mp_drawing.draw_landmarks(self.frame1, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)


    #返回各个电机的角度列表
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
                self.angle_between_finger_1 = (180-(self.get_angle((points[6].x, points[6].y, points[6].z),
                                        (points[5].x, points[5].y, points[5].z),
                                        (points[9].x, points[9].y, points[9].z))+100))*1.2+180

                self.angle_between_finger_2 = (180-(self.get_angle((points[13].x, points[13].y, points[13].z),
                                        (points[9].x, points[9].y, points[9].z),
                                        (points[10].x, points[10].y, points[10].z))+87))*1.2+180

                self.angle_between_finger_3 = (180-(self.get_angle((points[17].x, points[17].y, points[17].z),
                                        (points[13].x, points[13].y, points[13].z),
                                        (points[14].x, points[14].y, points[14].z))+65))+180

                self.angle_between_finger_4=self.get_angle((points[5].x, points[5].y, points[5].z),
                                        (points[0].x, points[0].y, points[0].z),
                                        (points[1].x, points[1].y, points[1].z))
                
                
                #算手臂映射的角度
                #腕部关节的两个自由度
                self.angle_arm_wrist2=        self.get_angle((points[9].x, points[9].y, points[9].z),
                                        (points[0].x, points[0].y, points[0].z),
                                        (points[22].x, points[22].y, points[22].z))
                
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
                    self.angle_between_finger_1, self.index_finger_angle_1,  self.index_finger_angle_2,  self.index_finger_angle_3, 
                    self.angle_between_finger_2, self.middle_finger_angle_1, self.middle_finger_angle_2, self.middle_finger_angle_3,
                    self.angle_between_finger_3, self.ring_finger_angle_1,   self.ring_finger_angle_2,   self.ring_finger_angle_3,
                    270,                         self.angle_between_finger_4,self.thumb_angle_2,         self.thumb_angle_3,
                    self.angle_arm_elbow2,       360-self.angle_arm_elbow1,     self.angle_arm_elbow1,   180,#self.angle_arm_wrist2,
                    self.angle_arm_wrist1
                ]
            else:
                finger_sequence_raw = [
                    self.angle_between_finger_1, self.index_finger_angle_1, self.index_finger_angle_2, self.index_finger_angle_3,
                    self.angle_between_finger_2, self.middle_finger_angle_1, self.middle_finger_angle_2, self.middle_finger_angle_3,
                    self.angle_between_finger_3, self.ring_finger_angle_1, self.ring_finger_angle_2, self.ring_finger_angle_3,
                    180,  180, self.thumb_angle_2, self.thumb_angle_3,
                    self.angle_arm_elbow2,       360-self.angle_arm_elbow1,     self.angle_arm_elbow1,      180,#self.angle_arm_wrist2,
                    self.angle_arm_wrist1
                ]
        
        # 步骤1: 仅对前16个手指电机角度应用clip
        finger_hand_clipped = np.clip(finger_sequence_raw[:16], 160, 270)
        finger_arm16_clipped  = np.clip(finger_sequence_raw[16:17], 0, 360)

        finger_arm171819_clipped = np.clip(finger_sequence_raw[17:20],100,260)
        finger_arm20_clipped     = np.clip(finger_sequence_raw[20:21],0,360)
        assert finger_sequence_raw[17]+finger_sequence_raw[18]==360

        # 拼接所有部分
        finger_clipped_all = np.concatenate([
            finger_hand_clipped,
            finger_arm16_clipped,
            finger_arm171819_clipped,
            finger_arm20_clipped
        ])

            
        self.suquence=finger_clipped_all
        self.finger_history=finger_clipped_all
        return self.suquence



    def run(self):


        while True:
            mark=1
                    
            color_image,depth_image,depth_scale=self.get_frames()

            if color_image is None or depth_image is None:
                print("Failed to get frames from camera.")
                continue
            
            hand_points,hand=self.hand_process(color_image,depth_image,depth_scale)
            arm_points,arm=self.arm_process(color_image,depth_image,depth_scale)
            constant=hand_points[19:21]
            points=hand_points+constant#+arm_points
            

            # if first==0:
                # history_sequence=[180]*21

            #处理识别不到手的情况
                #情况1
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
                raw_points_np = np.zeros((23, 3))
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
                
                sequence = self.get_sequence()
                print("get")
                # history_sequence= sequence
                try:
                    # 尝试用非阻塞方式放入新信息
                    self.cameraqueue.put_nowait(sequence)
                    print("in")
                
                except queue.Full:
                # 如果队列满了，什么也不做，直接进入下一次循环去处理更更新的帧。
                # 这种“丢帧”策略是保证系统实时性的关键。
                # 你可以在这里加一个打印语句用于调试，但生产环境中通常省略。
                # print("Queue is full, dropping a frame to maintain real-time performance.")
                    pass





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
        scale=(sequence-history_sequence)/20


        if loop<10:
            sequence=[180]*21
            scale=0
            return sequence,scale
        else:
            return sequence,scale





def main():

    hand_process_and_plot = Hand_Process_And_Plot()
    # leapnode = LeapNode()
    
    hand_process_and_plot.start()
    loop=-1
    history_sequence=[180]*21
    test=np.array([180]*21)
    time.sleep(5)
    while True:
        
        cv2.imshow('maincam', hand_process_and_plot.frame1)

        hand_process_and_plot.draw_plot(hand_process_and_plot.plotpoints)


        loop+=1
        sequence=hand_process_and_plot.cameraqueue.get()

        # print(sequence)
        sequence,scale= hand_process_and_plot.angle_filter(sequence, history_sequence, loop)
        for i in range(20):
            
            history_sequence=np.array(history_sequence)+scale
            if i:
                # print(loop)

                # print(test-history_sequence)

                # print(np.sum(np.abs(test-history_sequence)))
                # print(i)
                # print("--")
                # print(sequence)
                test=history_sequence
            time.sleep(0.001)

            # leapnode.set_degree(history_sequence[:16])

            #  history_sequence=sequence

        # if loop>4 and loop<50:
        #     print(sequence)
        #     print("\n")
        #     print(time.time())
        # leapnode.set_allegro(np.zeros(21))


        if cv2.waitKey(1) & 0xFF == 27:
            break
        pyplot.pause(0.01)
 
    print("while out")
    hand_process_and_plot.cap1.release()

if(__name__ == "__main__"):
    main()