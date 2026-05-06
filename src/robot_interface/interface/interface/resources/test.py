import time
import numpy as np
import rclpy
from interface.hand_controller import LeapHand
from interface.arm_controller import LeapArm
import os
import hydra
    
class HardwarePlayer:
    def __init__(self, config):
        self.config = config
        self.action_scale = 1 / 24

        self.home_path = os.path.expanduser("~")


    def game(self):
        rclpy.init()
        leap = LeapHand("rot")
        arm  = LeapArm("arm")
        #-------------------------------------------
        #下面是读取与初始化逻辑
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "arm_model_trajectory_2.npy")
        data = np.load(data_path)
        data=self.reorder(data)
        print(data[:2, :6])
        time.sleep(1)  # 等待一段时间以确保节点初始化完成

        time.sleep(1)  # 等待一段时间以确保节点初始化完成
        leap.get_logger().info("正在等待来自 /hand_joint_states 的手部消息和 /joint_states_single 的机械臂消息...")
        start_time = time.time()
        timeout = 5.0 # 5秒超时

        while (leap.raw_positions is None or arm.raw_positions is None) and rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
            rclpy.spin_once(arm, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                leap.get_logger().error("在5秒内未接收到 /hand_joint_states 或 /joint_states_single 消息，请检查发布器节点是否正常工作。")
                return
        #-------------------------------------------
        #下面是发送逻辑
        # 初始位置设置
        data_hand=data[:1, 6:]
        data_arm=data[:1, :6]
        bias=[0, 0, 0, 0, 0, 1.00]
        data_arm=data_arm + bias
        data_arm=np.reshape(data_arm, (1, 6))

        # 拼接初始位置
        # init_hand=np.concatenate((leap.raw_positions, data_hand), axis=0)
        init_arm=np.concatenate((arm.raw_positions, data_arm), axis=0)

        # 发送初始位置

        arm.command_joint_position(init_arm, 3)
        # 等待一段时间以确保初始位置到达
        time.sleep(5)
        # 初始位置误差回调
        if rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
            arm.command_joint_position(init_arm, 1.5)
        #-------------------------------------------
        # 这里是发送游戏动作的逻辑

        print("Commanding initial position")
        time.sleep(2)
        # 这里添加需要摆出来的动作，注意检查路径！
        for i in range(0, 15, 6):

            arm.command_joint_position(data[i:i+6, :6], 0.6)

            time.sleep(3)  # 等待每个动作完成
            rclpy.spin_once(arm, timeout_sec=0.1)
            arm.command_joint_position(data[i:i+6, :6], 1)
            time.sleep(1)  # 等待每个动作完成
            rclpy.spin_once(arm, timeout_sec=0.1)


        print("done")

    def reorder(self,data):
        rows, cols = data.shape

                # 检查数组是否至少有3行，以确保操作可以进行
        if data.shape[1] < 3:
            print("错误：数组的列数少于3，无法执行操作。")
        else:
            # 使用 np.concatenate 沿着列（axis=1）拼接数组切片
            # 这是最高效的方法
            
            # 第1部分：第1列 (索引0)
            part1 = data[:, 0:1]
            
            # 第2部分：第2列 (索引1)
            part2 = data[:, 1:2]
            
            # 第3部分：第2列的相反数
            part3 = -data[:, 1:2]
            
            # 第4部分：第3列 (索引2)
            part4 = data[:, 2:3]
            
            # 第5部分：第3列的相反数
            part5 = -data[:, 2:3]
            
            # 第6部分：从第4列到末尾的所有列 (索引3及之后)
            part6 = -data[:, 3:4]

            part7 =  data[:, 4:5]
            
            # 沿着列轴 (axis=1) 将所有部分拼接起来
            new_array = np.concatenate((part1, part2, part3, part4, part5, part6, part7), axis=1)

        return new_array
   

@hydra.main(config_name='config', config_path=os.path.join(os.path.dirname(__file__), "cfg"))
def main(config):
    agent = HardwarePlayer(config)
    agent.game()

if __name__ == '__main__':
    main()
