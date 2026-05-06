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
        self.home_path = os.path.expanduser("~")
        #-------------------------------------------
        #控制电机速度参数
        self.number=12     #一条轨迹包含的轨迹点数量
        self.speed=0.3     #电机运动速度，单位秒
        self.arm_speed=2   #手臂运动速度，单位秒
        self.init_speed=2  #运动到初始位置的速度，单位秒

    def hello(self):
        rclpy.init()
        leap = LeapHand("rot")
        arm  = LeapArm("arm")

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
        # 姿势初始化
        init=np.zeros((1,16))
        data_arm=[0, 0, 0, 0, 0, 1.0, 0]
        data_arm=np.reshape(data_arm, (1, 7))

        # 拼接初始位置
        init_hand=np.concatenate((leap.raw_positions, init), axis=0)
        init_arm=np.concatenate((leap.raw_arm_positions, data_arm), axis=0)

        # 发送初始位置
        leap.command_joint_position(init_hand, 2)
        arm.command_joint_position(init_arm, 1.5)

        # 等待一段时间以确保初始位置到达
        time.sleep(5)

        # 初始位置误差回调
        if rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
            arm.recover(init_arm, leap.raw_arm_positions, self.arm_speed)

        else:
            leap.get_logger().error("ROS2节点未正常运行，无法执行恢复操作。")
            return

        #主要内容发送,这里只发送手部的关节
        data= np.load(os.path.join(self.home_path,"colcon_ws/src/open_manipulator/leapsim/leapsim/cache/hello16.npy"))
        data =np.concatenate((init,data), axis=0)
        print("Initial hand position:", data.shape)
        time.sleep(1)
        leap.command_joint_position(data,0.8)

        i=0
        while i<10 :
            rclpy.spin(arm)
            i+=1    
        print("done")


@hydra.main(config_name='config', config_path=os.path.join(os.path.dirname(__file__), "cfg"))
def main(config):
    agent = HardwarePlayer(config)
    agent.hello()

if __name__ == '__main__':
    main()