import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe import Image as mp_Image
from mediapipe import ImageFormat as mp_ImageFormat

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


from matplotlib import pyplot
import numpy as np
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time
import threading


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
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # 0-11 are the fingers, 12-15 are the thumb, 16-20 are the palm motors
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
        print(self.curr_pos)
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
        print(self.curr_pos)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    
class Hand_Process_And_Plot(Node):
    def __init__(self):
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
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.75,
            result_callback=_hand_result_callback
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # 配置姿态（手臂）检测器
        base_options_pose = python.BaseOptions(model_asset_path=pose_model_path)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=vision.RunningMode.LIVE_STREAM, # 使用图像模式，逐帧处理
            min_pose_detection_confidence=0.3,
            min_tracking_confidence=0.5,
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



############ROS2相关初始化


         # 创建CvBridge实例用于图像转换
        self.bridge = CvBridge()

        # 订阅彩色图像话题
        self.color_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            10
        )

        # 订阅深度图像话题
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )


    def color_callback(self, msg):
        # 将ROS图像消息转换为OpenCV格式
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # 使用MediaPipe处理彩色图像
        results = self.hands.process(cv_image)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(cv_image, landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # 显示处理后的图像
        cv2.imshow("MediaPipe Hand Tracking", cv_image)
        cv2.waitKey(1)

    def depth_callback(self, msg):
        # 深度图像的处理（例如，获取深度信息等）
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
            return
        
        # 可以在这里处理深度图像，提取深度信息
        # 例如，你可以打印某个像素的深度值
        depth_value = depth_image[100, 100]  # 假设我们查看位置(100, 100)的深度值
        self.get_logger().info(f"Depth value at (100, 100): {depth_value}")


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
        color_image = np.asanyarray(color_frame.get_data()) # 直接是 (480, 640, 3) RGB


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
        self.frame1=mp_image
        # 2. 发送前清除旧结果，并下达异步处理命令
        with self.result_lock: self.latest_hand_result = None
        self.hand_landmarker.detect_async(mp_image, timestamp_ms)

        # 3. 等待回调函数送来结果（带超时保护）
        timeout = time.time() + 0.5
        while time.time() < timeout:
            with self.result_lock:
                if self.latest_hand_result is not None: break
            time.sleep(0.001)   
        hand_result = self.latest_hand_result
        
        point_3d_list = []
        
        # 处理检测结果
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                for landmark in hand_landmarks:
                    u = int(landmark.x * 640)
                    v = int(landmark.y * 480)

                    if 0 <= u < 640 and 0 <= v < 480:
                        depth_value = depth_image[v, u]
                        real_depth_in_meters = depth_value * depth_scale
                        # 反投影得到3D坐标
                        point_3d = self.pipeline.convert_2d_to_3d(u, v, real_depth_in_meters, self.camera_intrinsics)
                        point_3d_list.append(point_3d)
        
        return point_3d_list, hand_result
    
    def arm_process(self, color_image_rgb, depth_image, depth_scale):
        """使用MediaPipe PoseLandmarker检测手臂，并返回3D坐标列表。"""
        # 1. 创建时间戳
        timestamp_ms = int(time.time() * 1000)
        mp_image = mp_Image(image_format=mp_ImageFormat.SRGB, data=color_image_rgb)
        
        # 2. 发送前清除旧结果，并下达异步处理命令
        with self.result_lock: self.latest_hand_result = None
        self.hand_landmarker.detect_async(mp_image, timestamp_ms)

        # 3. 等待回调函数送来结果（带超时保护）
        timeout = time.time() + 0.5
        while time.time() < timeout:
            with self.result_lock:
                if self.latest_hand_result is not None: break
            time.sleep(0.001)

        pose_result = self.latest_pose_result




        point_3d_list = []
        
        # 我们只关心手臂的几个关键点: 肩膀, 肘部, 手腕
        # 在PoseLandmarker中, 右臂对应索引 12 (肩), 14 (肘), 16 (腕),20 (用于定位的手掌点)
        ARM_INDICES = [12, 14, 16, 20]


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
                        # 反投影得到3D坐标
                        point_3d = self.pipeline.convert_2d_to_3d(u, v, real_depth_in_meters, self.camera_intrinsics)
                        point_3d_list.append(point_3d)
                else:
                    point_3d_list.append(None) # 如果点不可见，添加占位符
        
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
        cv2.imshow('maincam', self.frame1)
        if self.results1.multi_hand_landmarks:
            #清除绘图区
            pyplot.figure(1)
            pyplot.cla()
            pyplot.figure(2)
            pyplot.cla()
            #画手部关节坐标点
            # for hand_landmarks in self.results1.multi_hand_landmarks:
            for id,i in enumerate(points):
                if(id!=17 and id!=18 and id!=19 and id!=20):
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
                self.angle_between_finger_1 = self.get_angle((points[6].x, points[6].y, points[6].z),
                                        (points[5].x, points[5].y, points[5].z),
                                        (points[9].x, points[9].y, points[9].z))
                #self.angle_between_finger_1+=100                                     
                self.angle_between_finger_2 = self.get_angle((points[13].x, points[13].y, points[13].z),
                                        (points[9].x, points[9].y, points[9].z),
                                        (points[10].x, points[10].y, points[10].z))
                #self.angle_between_finger_2+=70
                self.angle_between_finger_3 = self.get_angle((points[17].x, points[17].y, points[17].z),
                                        (points[13].x, points[13].y, points[13].z),
                                        (points[14].x, points[14].y, points[14].z))
                #self.angle_between_finger_3+=50
                self.angle_between_finger_4=self.get_angle((points[5].x, points[5].y, points[5].z),
                                        (points[0].x, points[0].y, points[0].z),
                                        (points[1].x, points[1].y, points[1].z))
                

                #算手臂映射的角度
                #腕部关节的两个自由度
                self.angle_arm_wrist2=        self.get_angle((points[24].x, points[24].y, points[24].z),
                                        (points[23].x, points[23].y, points[23].z),
                                        (points[22].x, points[22].y, points[22].z))
                
                self.angle_arm_wrist1=   180    # self.get_angle((points[24].x, points[24].y, points[24].z),
                                        # (points[23].x, points[23].y, points[23].z),
                                        # (points[22].x, points[22].y, points[22].z))
                #肘部关节的两个自由度
                self.angle_arm_elbow1=        90+self.get_angle((points[23].x, points[23].y, points[23].z),
                                        (points[22].x, points[22].y, points[22].z),
                                        (points[23].x, points[22].y, points[23].z))
                self.angle_arm_elbow2=        self.get_angle((points[23].x, points[23].y, points[23].z),
                                        (points[22].x, points[22].y, points[22].z),
                                        (points[21].x, points[21].y, points[21].z))+self.angle_arm_elbow1
                for hand_landmarks in self.results1.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(self.frame1, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    self.mp_drawing.draw_landmarks(self.frame1, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            
            # #初始化第二个图纸
            # fig = pyplot.figure(2)
            # pyplot.axis('off')
            # pyplot.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.5)
            # pyplot.yticks([])
            # pyplot.xticks([])
            # #画大拇指变化曲线图
            # self.thumb_angle_1_list.append(self.thumb_angle_1)
            # self.thumb_angle_2_list.append(self.thumb_angle_2)
            # self.thumb_angle_3_list.append(self.thumb_angle_3)
            # self.ax2 = pyplot.subplot(5, 3, 1)
            # pyplot.ylim(0, 360)
            # pyplot.yticks([0,180,360])
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="", loc="center")
            # self.ax2.plot(self.thumb_angle_1_list)
            # self.ax2 = pyplot.subplot(5, 3, 2)
            # pyplot.ylim(0, 360)
            # pyplot.yticks([])
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="", loc="center")
            # self.ax2.plot(self.thumb_angle_2_list)
            # self.ax2 = pyplot.subplot(5, 3, 3)
            # pyplot.ylim(0, 360)#设置y轴范围
            # pyplot.yticks([])#删除刻度显示
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="", loc="center")
            # self.ax2.plot(self.thumb_angle_3_list)

            # #画食指变化曲线图
            # self.index_finger_angle_1_list.append(self.index_finger_angle_1)
            # self.index_finger_angle_2_list.append(self.index_finger_angle_2)
            # self.index_finger_angle_3_list.append(self.index_finger_angle_3)
            # self.ax2 = pyplot.subplot(5, 3, 4)
            # pyplot.ylim(0, 360)
            # pyplot.yticks([0, 180, 360])
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="index1 motor1", loc="center")
            # self.ax2.plot(self.index_finger_angle_1_list)
            # self.ax2 = pyplot.subplot(5, 3, 5)
            # pyplot.ylim(0, 360)
            # pyplot.yticks([])
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="index2 motor2", loc="center")
            # self.ax2.plot(self.index_finger_angle_2_list)
            # self.ax2 = pyplot.subplot(5, 3, 6)
            # pyplot.ylim(0, 360)#设置y轴范围
            # pyplot.yticks([])#删除刻度显示
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="index3 motor3", loc="center")
            # self.ax2.plot(self.index_finger_angle_3_list)

            # #画中指变化曲线图
            # self.middle_finger_angle_1_list.append(self.middle_finger_angle_1)
            # self.middle_finger_angle_2_list.append(self.middle_finger_angle_2)
            # self.middle_finger_angle_3_list.append(self.middle_finger_angle_3)
            # self.ax2 = pyplot.subplot(5, 3, 7)
            # pyplot.ylim(0, 360)
            # pyplot.yticks([0,180,360])
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="middle1 motor5", loc="center")
            # self.ax2.plot(self.middle_finger_angle_1_list)
            # self.ax2 = pyplot.subplot(5, 3, 8)
            # pyplot.yticks([])
            # pyplot.ylim(0, 360)
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="middle2 motor6", loc="center")
            # self.ax2.plot(self.middle_finger_angle_2_list)
            # self.ax2 = pyplot.subplot(5, 3, 9)
            # pyplot.yticks([])
            # pyplot.ylim(0, 360)
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="middle3 motor7", loc="center")
            # self.ax2.plot(self.middle_finger_angle_3_list)

            # #画无名指变化曲线图
            # self.ring_finger_angle_1_list.append(self.ring_finger_angle_1)
            # self.ring_finger_angle_2_list.append(self.ring_finger_angle_2)
            # self.ring_finger_angle_3_list.append(self.ring_finger_angle_3)
            # self.ax2 = pyplot.subplot(5, 3, 10)
            # pyplot.ylim(0, 360)
            # pyplot.yticks([0,180,360])
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="ring1 motor9", loc="center")
            # self.ax2.plot(self.ring_finger_angle_1_list)
            # self.ax2 = pyplot.subplot(5, 3, 11)
            # pyplot.ylim(0, 360)
            # pyplot.yticks([])
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="ring2 motor10", loc="center")
            # self.ax2.plot(self.ring_finger_angle_2_list)
            # self.ax2 = pyplot.subplot(5, 3, 12)
            # pyplot.ylim(0, 360)
            # pyplot.yticks([])
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="ring3 motor11", loc="center")
            # self.ax2.plot(self.ring_finger_angle_3_list)

            # #画手指张开角度变化曲线图
            # self.angle_between_finger_1_list.append(self.angle_between_finger_1)
            # self.angle_between_finger_2_list.append(self.angle_between_finger_2)
            # self.angle_between_finger_3_list.append(self.angle_between_finger_3)
            # self.ax2 = pyplot.subplot(5, 3, 13)
            # pyplot.ylim(0, 360)
            # pyplot.yticks([0,180,360])
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="between1 motor0", loc="center")
            # self.ax2.plot(self.angle_between_finger_1_list)
            # self.ax2 = pyplot.subplot(5, 3, 14)
            # pyplot.ylim(0, 360)
            # pyplot.yticks([])
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="between2 motor4", loc="center")
            # self.ax2.plot(self.angle_between_finger_2_list)
            # self.ax2 = pyplot.subplot(5, 3, 15)
            # pyplot.ylim(0, 360)
            # pyplot.yticks([])
            # pyplot.xticks([])
            # pyplot.xlabel(xlabel="between3 motor8", loc="center")
            # self.ax2.plot(self.angle_between_finger_3_list)
            # pyplot.axis('on')

    #返回各个电机的角度列表
    def get_sequence(self):

        # <--- 新增/修改: 重构此方法以解决clip问题并提高可读性
        finger_sequence_raw = [180] * 21 # 如果没有检测到手，则使用默认值
        finger_sequence_raw = np.array(finger_sequence_raw)

        if self.results1.multi_hand_landmarks:
            if self.thumb_angle_2 > 220:
                finger_sequence_raw = [
                    self.angle_between_finger_1, self.index_finger_angle_1,  self.index_finger_angle_2,  self.index_finger_angle_3, 
                    self.angle_between_finger_2, self.middle_finger_angle_1, self.middle_finger_angle_2, self.middle_finger_angle_3,
                    self.angle_between_finger_3, self.ring_finger_angle_1,   self.ring_finger_angle_2,   self.ring_finger_angle_3,
                    270,                         self.angle_between_finger_4,self.thumb_angle_2,         self.thumb_angle_3,
                    self.angle_arm_wrist1,       self.angle_arm_wrist2,     self.angle_arm_elbow1,       self.angle_arm_elbow1,
                    self.angle_arm_elbow2
                ]
            else:
                finger_sequence_raw = [
                    self.angle_between_finger_1, self.index_finger_angle_1, self.index_finger_angle_2, self.index_finger_angle_3,
                    self.angle_between_finger_2, self.middle_finger_angle_1, self.middle_finger_angle_2, self.middle_finger_angle_3,
                    self.angle_between_finger_3, self.ring_finger_angle_1, self.ring_finger_angle_2, self.ring_finger_angle_3,
                    180,  180, self.thumb_angle_2, self.thumb_angle_3,
                    self.angle_arm_wrist1,       self.angle_arm_wrist2,     self.angle_arm_elbow1,       self.angle_arm_elbow1,
                    self.angle_arm_elbow2
                ]
        
        # 步骤1: 仅对前16个手指电机角度应用clip
        finger_hand_clipped = np.clip(finger_sequence_raw[:16], 160, 270)
        finger_arm16_clipped  = np.clip(finger_sequence_raw[16:17], 0, 360)
        assert finger_sequence_raw[18]==finger_sequence_raw[19]
        finger_arm171819_clipped = np.clip(finger_sequence_raw[17:20],100,260)
        finger_arm20_clipped     = np.clip(finger_sequence_raw[20],0,360)
        finger_hand_clipped = np.clip(finger_sequence_raw[:16], 160, 270)

        # 拼接所有部分
        finger_clipped_all = np.concatenate([
            finger_hand_clipped,
            finger_arm16_clipped,
            finger_arm171819_clipped,
            finger_arm20_clipped
        ])
        if(self.finger_history-finger_clipped_all>10 or self.finger_history-finger_clipped_all<-10 ):
            finger_clipped_all=self.finger_history
            
        self.suquence=finger_clipped_all
        self.finger_history=finger_clipped_all
        return self.suquence

def main():
    print("here")
    hand_process_and_plot = Hand_Process_And_Plot()
    # leapnode = LeapNode()
    while True:
        color_image,depth_image,depth_scale=hand_process_and_plot.get_frames()
        if color_image is None or depth_image is None:
            print("Failed to get frames from camera.")
            continue

        hand_points,hand=hand_process_and_plot.hand_process(color_image,depth_image,depth_scale)
        arm_points,arm=hand_process_and_plot.arm_process(color_image,depth_image,depth_scale)
        points=hand_points+arm_points
        hand_process_and_plot.draw_plot(points)
        sequence = hand_process_and_plot.get_sequence()
        print(sequence)
        # leapnode.set_allegro(np.zeros(21))
        #leapnode.set_degree(sequence)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        pyplot.pause(0.01)
        print("while end")
    print("while out")
    hand_process_and_plot.cap1.release()

if(__name__ == "__main__"):
    main()