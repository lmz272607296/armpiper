import time
import numpy as np
import rclpy
from leapsim.hand_controller import LeapHand
from leapsim.arm_controller import LeapArm
import os
import hydra

def _restore(player, args):
    """
    从检查点文件（checkpoint）中恢复模型和运行时的均值/标准差。
    """
    pass # 保持原样

class Rot:
    def __init__(self, config):
        self.config = config
        self.action_scale = 1 / 24
        self.init_pose = np.zeros(16)
        self.leap_dof_upper = np.array([1.047, 2.23, 1.885, 2.042, 1.047, 2.23, 1.885, 2.042, 1.047, 2.23, 1.885, 2.042, 2.094, 2.443, 1.90, 1.88])
        self.leap_dof_lower = np.array([-1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366, -0.349, -0.47, -1.20, -1.34])
        
        self.home_path = os.path.expanduser("~")
        
        #-------------------------------------------
        # 控制电机速度参数
        self.number = 12     # 一条轨迹包含的轨迹点数量
        self.speed = 0.3     # 电机运动速度，单位秒
        self.init_speed = 1.5 # 运动到初始位置的速度，单位秒

        #-------------------------------------------
        # 【新增】16个电机的补偿偏置 (Offset)
        # 你可以在这里随时手动修改具体的值（注意正负号取决于电机下沉的方向）
        # 数据类型为 numpy 数组，shape: (16,)
        self.joint_offsets = np.array([
            0.0, 0.0, 0.3, 0.0,  # 电机 0 1 2 3
            0.0, 0.0, -0.3, 0.0,  # 电机 4 5 6 7
            0.0, 0.0, 0.0, 0.0,  # 电机 8 9 10 11
            0.0, 0.0, 0.0, 0.0   # 电机 12 13 14 15
        ])
  
    def deploy(self):
        rclpy.init()
        leap = LeapHand("rot")
        arm  = LeapArm("arm")
        leap.leap_dof_lower = self.leap_dof_lower
        leap.leap_dof_upper = self.leap_dof_upper

        #-------------------------------------------
        # # 下面是读取与初始化逻辑
        leap.get_logger().info("正在等待来自 /joint_states 的第一条消息...")
        start_time = time.time()
        timeout = 5.0 # 5秒超时
        while leap.raw_positions is None and rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                leap.get_logger().error("在5秒内未接收到 /joint_states 消息，请检查发布器节点是否正常工作。")
                return

        #-------------------------------------------
        # 下面是发送逻辑
        data_path = os.path.join(self.home_path, "colcon_ws/src/open_manipulator/leapsim/leapsim/cache/hand16.npy")
        data = np.load(data_path)

        #-------------------------------------------
        # 这里是将raw_positions转换为初始位置的逻辑

        # 手部初始位置 (如果你希望初始位置也加上偏置，可以修改这里的数据)
        initial_pose = data[0,:]
        initial_pose = np.reshape(initial_pose, (1, 16))
        initial_pose = np.concatenate((leap.raw_positions, initial_pose), 0)

        # 臂部初始位置
        initial_arm = [0, 0, 0, 0, 0, 0, 3.14]
        initial_arm = np.reshape(initial_arm, (1, 7))

        initial_arm = np.concatenate((leap.raw_arm_positions, initial_arm), 0)
        
        print("Initial hand position:", initial_arm)
        time.sleep(1)

        arm.command_joint_position(initial_arm, 1.5)        
        leap.command_joint_position(initial_pose, 1.5)

        time.sleep(6)
        rclpy.spin_once(leap, timeout_sec=0.1)
        arm.recover(initial_arm, leap.raw_arm_positions, 1.5)

        #-------------------------------------------
        # 这里是发送轨迹的逻辑
        print("Commanding initial position")
        time.sleep(2)
        
        for i in range(0, 901, self.number):
            # 【修改点】取出当前的 (12, 16) 数据切片
            current_batch = data[i:i+self.number, :]
            
            # 【修改点】利用 numpy 广播机制，直接加上 (16,) 的偏置
            # 这样这 12 组数据里的每一个 16 个电机指令都会被加上对应的偏置
            compensated_batch = current_batch + self.joint_offsets
            
            # 发送补偿后的指令
            leap.command_joint_position(compensated_batch, self.speed)
            
            rclpy.spin_once(arm, timeout_sec=0.1)
            time.sleep(self.speed * self.number)

        # print("Starting playback")
        # counter = 0
        # while True:
        #     counter += 1
        #     commands = data[counter, :]
        #     leap.command_joint_position(data)
        #     print("Commanding:", commands)
        #     time.sleep(1)

@hydra.main(config_name='config', config_path=os.path.join(os.path.dirname(__file__), "cfg"))
def main(config):
    agent = Rot(config)
    agent.deploy()

if __name__ == '__main__':
    main()