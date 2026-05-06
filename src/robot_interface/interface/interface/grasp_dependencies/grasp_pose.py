from interface.hand_controller import LeapHand
from interface.arm_controller import LeapArm
import rclpy
from rclpy.node import Node
import time
import numpy as np
import os
import hydra

class Grasp(Node):
    def __init__(self,name):
        super().__init__(name)
        self.zero_pose = np.zeros(16)
        self.home_path = os.path.expanduser("~")
        #-------------------------------------------
        #控制电机速度参数
        self.scale=5            # 控制电机插值点
        self.number=5           # 一条轨迹包含的轨迹点数量（与scale相同）
        self.hand_speed=0.3     # 电机运动速度，单位秒
        self.arm_speed=0.3      # 手臂运动速度，单位秒
        self.init_speed=1.5     # 运动到初始位置的速度，单位秒

    def wait_for_hand_state(self, leap, timeout=5.0):
        start_time = time.time()
        while leap.raw_positions is None and rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                raise TimeoutError("在5秒内未接收到 /hand_joint_states 的 hand0-hand15 消息。")
        return np.asarray(leap.raw_positions, dtype=float).reshape(1, 16)

    def wait_for_arm_state(self, arm, timeout=5.0):
        start_time = time.time()
        while arm.raw_positions is None and rclpy.ok():
            rclpy.spin_once(arm, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                raise TimeoutError("在5秒内未接收到 /joint_states_single 的 joint1-joint6 消息。")
        return np.asarray(arm.raw_positions, dtype=float).reshape(1, 6)

    def graspcube(self):
        leap= LeapHand("rot")
        arm = LeapArm("arm")
        #-------------------------------------------
        # 首先初始化手的抓取姿势
        original_pose = np.array([251,  180,  185,  232,
                                  219,    256,  190,  235,
                                  251,    180,  185,  232,
                                  260,    215,  200,  172   ])
        original_action = self.angle_transfer(original_pose)

        zero=np.zeros(16)
        original_action = self.scale_action(zero,original_action, self.scale)
        print("Grasping action started")
        time.sleep(1) 

        leap.command_joint_position(original_action, 0.4)
        time.sleep(self.number * 0.4 + 1)  # 等待手部动作完成

        print("Hand action completed, starting arm action")
        time.sleep(0.5) 
        # 首先初始化手臂的抓取姿势
        #---------------------------------------------

        #--------------------------------------------
        # 下面开始进行夹持

        lower= original_pose 
        
        upper= np.array([260,  180,  175,  222,
                        219,    256,  255,  235,
                        260,    180,  175,  222,
                        260,    142,  200,  172   ])
        
        lower = self.angle_transfer(lower)
        upper = self.angle_transfer(upper)

        grasp_scale = 19 # 控制电机插值点
        grasp_action = self.scale_action(lower, upper, grasp_scale)

        #-------------------------------------------
        # 缓慢夹持，同时检查电流值
        for i in range(grasp_scale):
            leap.command_joint_position(grasp_action[i:i+2,:], self.hand_speed)
            time.sleep(self.hand_speed+0.1)

        # 缓慢加持，同时检查电流
        #-------------------------------------------
        leap.destroy_node()
        arm.destroy_node()

    def rotcube(self):
        arm= LeapArm("arm_rot")
        leap= LeapHand("rot_rot")

        # --------------------------------------------
        # 先加载转方块的数据
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "hand16.npy")
        data = np.load(data_path)

        # --------------------------------------------
        # 将机械手运动到位
        arm_zero = np.zeros((1, 6), dtype=float)
        current_arm = self.wait_for_arm_state(arm)
        arm_loose=self.scale_action(current_arm,arm_zero,  10)
        
        arm.command_joint_position(arm_loose, 0.5)
        time.sleep(6)

        if rclpy.ok():
            rclpy.spin_once(arm, timeout_sec=0.1)
            arm.command_joint_position(arm_loose, 1)
        time.sleep(1)
        print("Arm action completed, starting hand action")
        # --------------------------------------------
        # 开始发送转方块的逻辑
        if rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)

            first_position = np.concatenate((leap.raw_positions, data[0:,:]), axis=0)
            leap.command_joint_position(first_position, 2)
            time.sleep(2)

            for i in range( 0,901, 12):
                leap.command_joint_position(data[i:i+12,:], 0.3)
                time.sleep(0.3*12)

        leap.destroy_node()
        arm.destroy_node()


    def loose_cube(self):
        leap= LeapHand("rot2")
        arm = LeapArm("arm2")
        #-------------------------------------------
        # 首先初始化臂的抓取姿势
        arm_zero = np.zeros((1, 6), dtype=float)
        arm.raw_positions=None
        current_arm = self.wait_for_arm_state(arm)
        arm_loose=self.scale_action(current_arm,arm_zero,  4)
        
        arm.command_joint_position(arm_loose, 0.5)
        time.sleep(5)

        if rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
            arm.command_joint_position(arm_loose, 1)
        time.sleep(2)

        # 首先初始化手的抓取姿势
        #-------------------------------------------

        #-------------------------------------------
        # 下面开始进行松开
       
        hand_zero = np.array([
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        ])
        hand_zero = np.reshape(hand_zero, (1, 16))
        if rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
            print(leap.raw_positions)
            hand_loose = np.concatenate((leap.raw_positions, hand_zero), axis=0)

        leap.command_joint_position(hand_loose, 3)
        time.sleep(3)


        # 已经松开夹持物体
        #-------------------------------------------
        leap.destroy_node()
        arm.destroy_node()

    """
    瓶子夹持函数
    """
    def graspbottle(self):
        leap= LeapHand("rot1")
        arm = LeapArm("arm1")
        #-------------------------------------------
        # 首先初始化手的抓取姿势
        original_pose = np.array([209,    180,  190,  253,
                                  260,    279,  179,  236,
                                  209,    180,  190,  253,
                                  209,    180,  190,  253   ])
        
        original_action = self.angle_transfer(original_pose)

        zero=np.zeros(16)
        original_action = self.scale_action(zero,original_action, self.scale)
        print("Grasping action started")
        time.sleep(1) 
        leap.command_joint_position(original_action, 0.4)
        time.sleep(self.number * 0.4 + 1)  # 等待手部动作完成

        print("Hand action completed, starting arm action")
        time.sleep(0.5) 
        # 首先初始化手臂的抓取姿势
        #---------------------------------------------

        #--------------------------------------------
        # 下面开始进行夹持

        lower= original_pose 
        
        upper= np.array([272,    180,  190,  253,
                         260,    279,  179,  236,
                         272,    180,  190,  253,
                         272,    180,  190,  253   ])
        
        lower = self.angle_transfer(lower)
        upper = self.angle_transfer(upper)

        grasp_scale = 50 # 控制电机插值点
        grasp_action = self.scale_action(lower, upper, grasp_scale)

        #-------------------------------------------
        # 缓慢夹持，同时检查电流值
        for i in range(grasp_scale):
            leap.command_joint_position(grasp_action[i:i+2,:], 0.1)
            time.sleep(0.1)

        leap.destroy_node()
        arm.destroy_node()
        # 缓慢加持，同时检查电流
        #-------------------------------------------
    
    """
    瓶子释放函数
    """
    def loose_bottle(self):
        leap= LeapHand("rot2")
        arm = LeapArm("arm2")
        #-------------------------------------------
        # 首先初始化臂的抓取姿势
        arm_zero = np.zeros((1, 6), dtype=float)
        arm.raw_positions=None
        current_arm = self.wait_for_arm_state(arm)
        arm_loose=self.scale_action(current_arm,arm_zero,  10)
        
        arm.command_joint_position(arm_loose, 0.5)
        time.sleep(6)

        if rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
            arm.command_joint_position(arm_loose, 1)
        time.sleep(1)

        # 首先初始化手的抓取姿势
        #-------------------------------------------

        #-------------------------------------------
        # 下面开始进行松开
       
        hand_zero = np.array([
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        ])
        hand_zero = np.reshape(hand_zero, (1, 16))
        if rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
            print(leap.raw_positions)
            hand_loose = np.concatenate((leap.raw_positions, hand_zero), axis=0)

        leap.command_joint_position(hand_loose, 3)
        time.sleep(4)

        # 已经松开夹持物体
        #-------------------------------------------

        #-------------------------------------------
        # 松开后手腕复位
        arm_zero = np.zeros((1, 6), dtype=float)
        if rclpy.ok():
            current_arm = self.wait_for_arm_state(arm)
            arm_loose=np.concatenate((current_arm,arm_zero), axis=0)
            arm.command_joint_position(arm_loose, 2)
            time.sleep(4)

        
        leap.destroy_node()
        arm.destroy_node()





    def angle_transfer(self, data):
        """
        不论输入的数据是几维的，都将其中的每一个数据的角度值先减去180度，
        再将角度值转换为弧度值，且用小数表示。
        注意，最后的输出要维持原来的数组形状不变。
        """
        data = np.array(data, dtype=float)
        data = data - 180  # 减去180度
        data = np.radians(data)  # 转换为弧度
        return data

    def scale_action(self, origin, action, scale):
        # 确保 scale 是一个正整数

        if not isinstance(scale, int) or scale <= 0:
            raise ValueError("scale 必须是一个正整数。")
        action_np = np.asarray(action, dtype=float)
        origin_np = np.asarray(origin, dtype=float)
        difference = action_np - origin_np
        increment = difference / scale
        scaling_factors = np.arange(scale + 1).reshape(-1, 1)
        offsets = scaling_factors * increment
        scaled_actions = origin_np + offsets
        final_result = scaled_actions
        return final_result

def main():
    # 创建Grasp实例
    rclpy.init()
    grasp = Grasp("grasp_node")
    
    while rclpy.ok():
            # 请求键盘命令
            key = input("请输入命令: (1: 执行 grasptube, 2: 执行抓瓶子, q: 退出): ")

            if key == '1':
                print("现在执行 graspcube 函数")
                # 调用 graspcube 函数，请确保这个函数在您的 Grasp 类中已经定义
                grasp.graspcube() 
                print("graspcube 函数执行完毕")
                # grasp.rotcube()
                grasp.loose_cube()
            elif key == '2':
                # 执行夹持动作
                grasp.graspbottle()
                time.sleep(2)
                print("现在执行抓取动作后摇")
                grasp.loose_bottle()
            elif key == 'q':
                print("程序退出")
                break
            else:
                print("无效的输入，请重新输入。")
    # 关闭rclpy
    rclpy.shutdown()
if __name__ == '__main__':
    main()
