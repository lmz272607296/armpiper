import time
import numpy as np
import rclpy

from leapsim.hand_controller import LeapHand
from leapsim.arm_controller import LeapArm
import os
import hydra

class HardwarePlayer:
    def __init__(self, config):
        self.config = config
        self.action_scale = 1 / 24
        self.init_pose = np.zeros(16)
        self.leap_dof_upper = np.array([1.047, 2.23, 1.885, 2.042, 1.047, 2.23, 1.885, 2.042, 1.047, 2.23, 1.885, 2.042, 2.094, 2.443, 1.90, 1.88])
        self.leap_dof_lower = np.array([-1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366, -0.349, -0.47, -1.20, -1.34])
        self.home_path = os.path.expanduser("~")

        #-------------------------------------------
        #控制电机速度参数
        self.number=12    #一条轨迹包含的轨迹点数量
        self.speed=2   #电机运动速度，单位秒
        self.arm_speed=0.3   #手臂运动速度，单位秒
        self.init_speed=1.5   #运动到初始位置的速度，单位秒


    def ping(self):
        rclpy.init()
        leap = LeapHand("rot")
        arm=LeapArm("arm")
        leap.leap_dof_lower = self.leap_dof_lower
        leap.leap_dof_upper = self.leap_dof_upper

        #-------------------------------------------
        #下面是读取与初始化逻辑
        leap.get_logger().info("正在等待来自 /joint_states 的第一条消息...")
        start_time = time.time()
        timeout = 5.0 # 5秒超时
        while leap.raw_positions is None and leap.raw_arm_positions is None and rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                leap.get_logger().error("在5秒内未接收到 /joint_states 消息，请检查发布器节点是否正常工作。")
                return
            

        #-------------------------------------------
        #下面是发送逻辑

        #初始位置设置
        data=np.zeros((1,16))
        data_arm=[0, 0, 0, 0, 0, 1.57, 0]
        data_arm=np.reshape(data_arm, (1, 7))
        #拼接初始位置
        init_hand=np.concatenate((leap.raw_positions, data), axis=0)
        init_arm=np.concatenate((leap.raw_arm_positions,data_arm), axis=0)
        #发送初始位置
        leap.command_joint_position(init_hand, self.speed)
        arm.command_joint_position(init_arm, self.init_speed)

        #等待电机到达初始位置
        time.sleep(5)

        #误差回调
        rclpy.spin_once(leap, timeout_sec=0.1)
        arm.recover(init_arm, leap.raw_arm_positions, self.init_speed)


        i=0
        while i<10 :
            rclpy.spin(arm)
            i+=1    

@hydra.main(config_name='config', config_path=os.path.join(os.path.dirname(__file__), "cfg"))
def main(config):
    agent = HardwarePlayer(config)
    agent.ping()

if __name__ == '__main__':
    main()