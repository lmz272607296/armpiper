import cv2
import mediapipe as mp
from matplotlib import pyplot
import numpy as np
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu



class LeapNode:
    def __init__(self):
        ####Some parameters
        # self.ema_amount = float(rospy.get_param('/leaphand_node/ema', '1.0')) #take only current
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
           
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] # 0-11 are the fingers, 12-15 are the thumb, 16-20 are the palm motors
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM13', 4000000)
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

    
class Hand_Process_And_Plot():
    def __init__(self):
        cv2.namedWindow("maincam", 1)
        self.cap1 = cv2.VideoCapture(0)
        pyplot.ion()
        self.ax1 = pyplot.axes(projection='3d')
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
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
        self.suquence = [180]*16


    #获取相机图像
    def get_camera(self):
        self.ret, self.frame1 = self.cap1.read()
        self.frame1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)

    #处理手部，得出关键点坐标
    def hand_process(self):
        self.results1 = self.hands.process(self.frame1)
        self.frame1 = cv2.cvtColor(self.frame1, cv2.COLOR_RGB2BGR)
    
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
    def draw_plot(self):
        cv2.imshow('maincam', self.frame1)
        if self.results1.multi_hand_landmarks:
            #清除绘图区
            pyplot.figure(1)
            pyplot.cla()
            pyplot.figure(2)
            pyplot.cla()
            #画手部关节坐标点
            for hand_landmarks in self.results1.multi_hand_landmarks:
                for id,i in enumerate(hand_landmarks.landmark):
                    if(id!=17 and id!=18 and id!=19 and id!=20):
                        self.ax1.scatter3D(i.x, i.y, i.z)
                #连接各个关节的线
                xdata = []
                ydata = []
                zdata = []
                for i in [0, 1, 2, 3, 4]:
                    xdata.append(hand_landmarks.landmark[i].x)
                    ydata.append(hand_landmarks.landmark[i].y)
                    zdata.append(hand_landmarks.landmark[i].z)
                self.ax1.plot(xdata,ydata,zdata)

                xdata = []
                ydata = []
                zdata = []
                for i in [0, 5, 6, 7, 8]:
                    xdata.append(hand_landmarks.landmark[i].x)
                    ydata.append(hand_landmarks.landmark[i].y)
                    zdata.append(hand_landmarks.landmark[i].z)
                self.ax1.plot(xdata,ydata,zdata)

                xdata = []
                ydata = []
                zdata = []
                for i in [9, 10, 11, 12]:
                    xdata.append(hand_landmarks.landmark[i].x)
                    ydata.append(hand_landmarks.landmark[i].y)
                    zdata.append(hand_landmarks.landmark[i].z)
                self.ax1.plot(xdata,ydata,zdata)

                xdata = []
                ydata = []
                zdata = []
                for i in [0, 13, 14, 15, 16]:
                    xdata.append(hand_landmarks.landmark[i].x)
                    ydata.append(hand_landmarks.landmark[i].y)
                    zdata.append(hand_landmarks.landmark[i].z)
                self.ax1.plot(xdata, ydata, zdata)

                xdata = []
                ydata = []
                zdata = []
                for i in [5, 9, 13]:
                    xdata.append(hand_landmarks.landmark[i].x)
                    ydata.append(hand_landmarks.landmark[i].y)
                    zdata.append(hand_landmarks.landmark[i].z)
                self.ax1.plot(xdata, ydata, zdata)
                # 算大拇指角度
                self.thumb_angle_1 = 360-self.get_angle((hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z), 
                                        (hand_landmarks.landmark[1].x, hand_landmarks.landmark[1].y, hand_landmarks.landmark[1].z),
                                        (hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y, hand_landmarks.landmark[2].z))
                self.thumb_angle_2 = 360-self.get_angle((hand_landmarks.landmark[1].x, hand_landmarks.landmark[1].y, hand_landmarks.landmark[1].z),
                                        (hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y, hand_landmarks.landmark[2].z),
                                        (hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y, hand_landmarks.landmark[3].z))
                self.thumb_angle_3 = 360-self.get_angle((hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y, hand_landmarks.landmark[2].z),
                                        (hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y, hand_landmarks.landmark[3].z),
                                        (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z))
                # 算食指角度
                self.index_finger_angle_1 = 360-self.get_angle((hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z),
                                        (hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z),
                                        (hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y, hand_landmarks.landmark[6].z))
                self.index_finger_angle_2 = 360-self.get_angle((hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z),
                                        (hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y, hand_landmarks.landmark[6].z),
                                        (hand_landmarks.landmark[7].x, hand_landmarks.landmark[7].y, hand_landmarks.landmark[7].z))
                self.index_finger_angle_3 = 360-self.get_angle((hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y, hand_landmarks.landmark[6].z),
                                        (hand_landmarks.landmark[7].x, hand_landmarks.landmark[7].y, hand_landmarks.landmark[7].z),
                                        (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z))
                # 算中指角度
                self.middle_finger_angle_1 = 360-self.get_angle((hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z),
                                        (hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z),
                                        (hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y, hand_landmarks.landmark[10].z))
                self.middle_finger_angle_2 = 360-self.get_angle((hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z),
                                        (hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y, hand_landmarks.landmark[10].z),
                                        (hand_landmarks.landmark[11].x, hand_landmarks.landmark[11].y, hand_landmarks.landmark[11].z))
                self.middle_finger_angle_3 = 360-self.get_angle((hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y, hand_landmarks.landmark[10].z),
                                        (hand_landmarks.landmark[11].x, hand_landmarks.landmark[11].y, hand_landmarks.landmark[11].z),
                                        (hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z))
                # 算无名指角度
                self.ring_finger_angle_1 = 360-self.get_angle((hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z),
                                        (hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z),
                                        (hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y, hand_landmarks.landmark[10].z))
                self.ring_finger_angle_2 = 360-self.get_angle((hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z),
                                        (hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y, hand_landmarks.landmark[10].z),
                                        (hand_landmarks.landmark[11].x, hand_landmarks.landmark[11].y, hand_landmarks.landmark[11].z))
                self.ring_finger_angle_3 = 360-self.get_angle((hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y, hand_landmarks.landmark[10].z),
                                        (hand_landmarks.landmark[11].x, hand_landmarks.landmark[11].y, hand_landmarks.landmark[11].z),
                                        (hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z))
                
                #算手指之间的角度
                self.angle_between_finger_1 = self.get_angle((hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y, hand_landmarks.landmark[6].z),
                                        (hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z),
                                        (hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z))
                #self.angle_between_finger_1+=100                                     
                self.angle_between_finger_2 = self.get_angle((hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y, hand_landmarks.landmark[13].z),
                                        (hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z),
                                        (hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y, hand_landmarks.landmark[10].z))
                #self.angle_between_finger_2+=70
                self.angle_between_finger_3 = self.get_angle((hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z),
                                        (hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y, hand_landmarks.landmark[13].z),
                                        (hand_landmarks.landmark[14].x, hand_landmarks.landmark[14].y, hand_landmarks.landmark[14].z))
                #self.angle_between_finger_3+=50
                self.angle_between_finger_4=self.get_angle((hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z),
                                        (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z),
                                        (hand_landmarks.landmark[1].x, hand_landmarks.landmark[1].y, hand_landmarks.landmark[1].z))
                self.mp_drawing.draw_landmarks(self.frame1, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                self.mp_drawing.draw_landmarks(self.frame1, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            
            #初始化第二个图纸
            fig = pyplot.figure(2)
            pyplot.axis('off')
            pyplot.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.5)
            pyplot.yticks([])
            pyplot.xticks([])
            #画大拇指变化曲线图
            self.thumb_angle_1_list.append(self.thumb_angle_1)
            self.thumb_angle_2_list.append(self.thumb_angle_2)
            self.thumb_angle_3_list.append(self.thumb_angle_3)
            self.ax2 = pyplot.subplot(5, 3, 1)
            pyplot.ylim(0, 360)
            pyplot.yticks([0,180,360])
            pyplot.xticks([])
            pyplot.xlabel(xlabel="", loc="center")
            self.ax2.plot(self.thumb_angle_1_list)
            self.ax2 = pyplot.subplot(5, 3, 2)
            pyplot.ylim(0, 360)
            pyplot.yticks([])
            pyplot.xticks([])
            pyplot.xlabel(xlabel="", loc="center")
            self.ax2.plot(self.thumb_angle_2_list)
            self.ax2 = pyplot.subplot(5, 3, 3)
            pyplot.ylim(0, 360)#设置y轴范围
            pyplot.yticks([])#删除刻度显示
            pyplot.xticks([])
            pyplot.xlabel(xlabel="", loc="center")
            self.ax2.plot(self.thumb_angle_3_list)

            #画食指变化曲线图
            self.index_finger_angle_1_list.append(self.index_finger_angle_1)
            self.index_finger_angle_2_list.append(self.index_finger_angle_2)
            self.index_finger_angle_3_list.append(self.index_finger_angle_3)
            self.ax2 = pyplot.subplot(5, 3, 4)
            pyplot.ylim(0, 360)
            pyplot.yticks([0, 180, 360])
            pyplot.xticks([])
            pyplot.xlabel(xlabel="index1 motor1", loc="center")
            self.ax2.plot(self.index_finger_angle_1_list)
            self.ax2 = pyplot.subplot(5, 3, 5)
            pyplot.ylim(0, 360)
            pyplot.yticks([])
            pyplot.xticks([])
            pyplot.xlabel(xlabel="index2 motor2", loc="center")
            self.ax2.plot(self.index_finger_angle_2_list)
            self.ax2 = pyplot.subplot(5, 3, 6)
            pyplot.ylim(0, 360)#设置y轴范围
            pyplot.yticks([])#删除刻度显示
            pyplot.xticks([])
            pyplot.xlabel(xlabel="index3 motor3", loc="center")
            self.ax2.plot(self.index_finger_angle_3_list)

            #画中指变化曲线图
            self.middle_finger_angle_1_list.append(self.middle_finger_angle_1)
            self.middle_finger_angle_2_list.append(self.middle_finger_angle_2)
            self.middle_finger_angle_3_list.append(self.middle_finger_angle_3)
            self.ax2 = pyplot.subplot(5, 3, 7)
            pyplot.ylim(0, 360)
            pyplot.yticks([0,180,360])
            pyplot.xticks([])
            pyplot.xlabel(xlabel="middle1 motor5", loc="center")
            self.ax2.plot(self.middle_finger_angle_1_list)
            self.ax2 = pyplot.subplot(5, 3, 8)
            pyplot.yticks([])
            pyplot.ylim(0, 360)
            pyplot.xticks([])
            pyplot.xlabel(xlabel="middle2 motor6", loc="center")
            self.ax2.plot(self.middle_finger_angle_2_list)
            self.ax2 = pyplot.subplot(5, 3, 9)
            pyplot.yticks([])
            pyplot.ylim(0, 360)
            pyplot.xticks([])
            pyplot.xlabel(xlabel="middle3 motor7", loc="center")
            self.ax2.plot(self.middle_finger_angle_3_list)

            #画无名指变化曲线图
            self.ring_finger_angle_1_list.append(self.ring_finger_angle_1)
            self.ring_finger_angle_2_list.append(self.ring_finger_angle_2)
            self.ring_finger_angle_3_list.append(self.ring_finger_angle_3)
            self.ax2 = pyplot.subplot(5, 3, 10)
            pyplot.ylim(0, 360)
            pyplot.yticks([0,180,360])
            pyplot.xticks([])
            pyplot.xlabel(xlabel="ring1 motor9", loc="center")
            self.ax2.plot(self.ring_finger_angle_1_list)
            self.ax2 = pyplot.subplot(5, 3, 11)
            pyplot.ylim(0, 360)
            pyplot.yticks([])
            pyplot.xticks([])
            pyplot.xlabel(xlabel="ring2 motor10", loc="center")
            self.ax2.plot(self.ring_finger_angle_2_list)
            self.ax2 = pyplot.subplot(5, 3, 12)
            pyplot.ylim(0, 360)
            pyplot.yticks([])
            pyplot.xticks([])
            pyplot.xlabel(xlabel="ring3 motor11", loc="center")
            self.ax2.plot(self.ring_finger_angle_3_list)

            #画手指张开角度变化曲线图
            self.angle_between_finger_1_list.append(self.angle_between_finger_1)
            self.angle_between_finger_2_list.append(self.angle_between_finger_2)
            self.angle_between_finger_3_list.append(self.angle_between_finger_3)
            self.ax2 = pyplot.subplot(5, 3, 13)
            pyplot.ylim(0, 360)
            pyplot.yticks([0,180,360])
            pyplot.xticks([])
            pyplot.xlabel(xlabel="between1 motor0", loc="center")
            self.ax2.plot(self.angle_between_finger_1_list)
            self.ax2 = pyplot.subplot(5, 3, 14)
            pyplot.ylim(0, 360)
            pyplot.yticks([])
            pyplot.xticks([])
            pyplot.xlabel(xlabel="between2 motor4", loc="center")
            self.ax2.plot(self.angle_between_finger_2_list)
            self.ax2 = pyplot.subplot(5, 3, 15)
            pyplot.ylim(0, 360)
            pyplot.yticks([])
            pyplot.xticks([])
            pyplot.xlabel(xlabel="between3 motor8", loc="center")
            self.ax2.plot(self.angle_between_finger_3_list)
            pyplot.axis('on')
    #返回各个电机的角度列表
    def get_sequence(self):

        # <--- 新增/修改: 重构此方法以解决clip问题并提高可读性
        finger_sequence_raw = [180] * 16 # 如果没有检测到手，则使用默认值
        if self.results1.multi_hand_landmarks:
            if self.thumb_angle_2 > 220:
                finger_sequence_raw = [
                    self.angle_between_finger_1, self.index_finger_angle_1, self.index_finger_angle_2, self.index_finger_angle_3, 
                    self.angle_between_finger_2, self.middle_finger_angle_1, self.middle_finger_angle_2, self.middle_finger_angle_3,
                    self.angle_between_finger_3, self.ring_finger_angle_1, self.ring_finger_angle_2, self.ring_finger_angle_3,
                    270,  self.angle_between_finger_4, self.thumb_angle_2, self.thumb_angle_3,
                ]
            else:
                finger_sequence_raw = [
                    self.angle_between_finger_1, self.index_finger_angle_1, self.index_finger_angle_2, self.index_finger_angle_3,
                    self.angle_between_finger_2, self.middle_finger_angle_1, self.middle_finger_angle_2, self.middle_finger_angle_3,
                    self.angle_between_finger_3, self.ring_finger_angle_1, self.ring_finger_angle_2, self.ring_finger_angle_3,
                    180,  180, self.thumb_angle_2, self.thumb_angle_3,
                ]
        
        # 步骤1: 仅对前16个手指电机角度应用clip
        finger_angles_clipped = np.clip(finger_sequence_raw, 160, 270)
        self.suquence=finger_sequence_raw
        return self.suquence

def main():
    hand_preocess_and_plot = Hand_Process_And_Plot()
    # leapnode = LeapNode()
    while True:
        hand_preocess_and_plot.get_camera()
        hand_preocess_and_plot.hand_process()
        hand_preocess_and_plot.draw_plot()
        sequence = hand_preocess_and_plot.get_sequence()
        print(sequence)
        # leapnode.set_allegro(np.zeros(21))
        #leapnode.set_degree(sequence)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        pyplot.pause(0.01)
    hand_preocess_and_plot.cap1.release()

if(__name__ == "__main__"):
    main()