import time
import numpy as np
import rclpy
from leapsim.hardware_controlleryi import LeapHand
from leapsim.arm_controller import LeapArm
import os
import hydra
    
class HardwarePlayer:
    def __init__(self, config):
        self.config = config
        self.action_scale = 1 / 24
        self.init_pose = np.zeros(16)
        self.home_path = os.path.expanduser("~")

    def game(self):
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
        # 初始位置设置
        data=np.zeros((1,16))
        data_arm=[0, -0.209, 0.209, -0.139, 0.139, 1.22, 0]
        data_arm=np.reshape(data_arm, (1, 7))
        # 拼接初始位置
        init_hand=np.concatenate((leap.raw_positions, data), axis=0)
        init_arm=np.concatenate((leap.raw_arm_positions, data_arm), axis=0)
        # 发送初始位置
        leap.command_joint_position(init_hand, 1.5)
        arm.command_joint_position(init_arm, 1.5)
        # 等待一段时间以确保初始位置到达
        time.sleep(5)
        # 初始位置误差回调
        if rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
            arm.recover(init_arm, leap.raw_arm_positions, 1.5)
        #-------------------------------------------
        # 这里是发送游戏动作的逻辑
        print("Commanding initial position")
        time.sleep(2)
        # 这里添加需要摆出来的动作，注意检查路径！
        data = np.load(os.path.join(self.home_path,"colcon_ws/src/open_manipulator/leapsim/leapsim/cache/game16.npy"))
        np.random.seed(None)

        #这里添加需要摆出来的动作，注意检查路径！
        np.random.seed(None)
        initial_position = data[np.random.choice([0, 1, 2], p=[1/3, 1/3, 1/3])]
        leap.command_joint_position(initial_position)

        while rclpy.ok():
            rclpy.spin_once(arm, timeout_sec=0.1)
        print("done")
   

@hydra.main(config_name='config', config_path=os.path.join(os.path.dirname(__file__), "cfg"))
def main(config):
    agent = HardwarePlayer(config)
    agent.game()

if __name__ == '__main__':
    main()