#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
import threading
import time
import numpy as np
class TargetDetector(Node):
    """
    一个工具类，用于订阅目标检测话题，并获取特定目标的3D坐标。
    """
    def __init__(self):
        # 初始化节点，使用一个独特的名字防止冲突
        super().__init__('target_detector_node_')
        self.tv_position = None
        self.found_type = None
        self.target_types_to_find = []
        # 使用Event来通知主调函数，坐标已经收到
        self.target_found_event = threading.Event()

        # 用于传递权重和关节限制
        self.checkpoint_path_to_use = None
        self.dof_limit_type_to_use = None

        # 用于传递转换后的坐标
        self.found_position = None
        self.object_type = None
 
    def _is_coord_valid(self, position):
        """
        私有方法，用于验证坐标是否在预设的有效范围内。
        坐标单位为米。
        - x 坐标不能超过 50cm (0.5m)
        - y 坐标的绝对值不能超过 40cm (0.4m)
        """
        # 将厘米转换为米
        max_x = 0.5  # 50 cm
        max_y_abs = 0.4  # 40 cm

        if position.x > max_x:
            self.get_logger().warn(
                f"目标被拒绝：X坐标 {position.x:.2f}m > {max_x}m (太远了)。"
            )
            return False
        
        if abs(position.y) > max_y_abs:
            self.get_logger().warn(
                f"目标被拒绝：Y坐标绝对值 |{position.y:.2f}m| > {max_y_abs}m (太偏了)。"
            )
            return False
        
        return True
    def _listener_callback(self, msg):
        """
        内部回调函数，处理收到的消息。
        """
        # 如果已经找到了目标，就不再处理新的消息
        if self.target_found_event.is_set():
            return

        
        for target in msg.targets:
                    object_type = target.rois[0].type if target.rois else ""
                    
                    # 检查1：目标类型是否在我们想要的列表中
                    if object_type.upper() in self.target_types_to_find:
                        for point in target.points:
                            if point.type == "center_3d" and point.point:
                                position_3d = point.point[0]
                                self.get_logger().info(f"检测到候选目标 '{object_type}'，正在验证坐标...")

                                # 检查2：坐标是否在有效范围内
                                if self._is_coord_valid(position_3d):
                                    self.get_logger().info(
                                        f"坐标有效！已锁定目标 '{object_type}'，"
                                        f"坐标: (x={position_3d.x:.3f}, y={position_3d.y:.3f}, z={position_3d.z:.3f})"
                                    )
                                    # 保存找到的目标类型和坐标
                                    self.found_type = object_type
                                    self.found_position = [position_3d.x, position_3d.y, position_3d.z]
                                    self.target_found_event.set()
                                    return # 目标有效，停止处理并返回
                                else:
                                    # 坐标无效，继续检查此消息中的下一个目标
                                    continue

    def get_target_coordinates(self, target_types, timeout_sec=30.0):
        """
        公开的调用函数。此函数会阻塞，直到找到目标坐标或超时。

        :param timeout_sec: 等待的秒数。
        :return: 返回一个包含 [x, y, z] 的列表，如果超时则返回 None。
        """
        self.target_types_to_find = [t.upper() for t in target_types]
        self.get_logger().info(f"开始监听 /yolo_3d_detections 话题，寻找 {self.target_types_to_find}，超时时间: {timeout_sec}秒...")

        # 创建订阅者
        subscription = self.create_subscription(
            PerceptionTargets,
            '/yolo_3d_detections',
            self._listener_callback,
            10)
            
        # 等待事件被设置，或者等待超时
        # 我们在这里不使用 rclpy.spin()，而是使用 spin_once() 来手动处理消息，
        # 这样可以更好地控制阻塞和超时。
        
        # 等待事件被设置（意味着找到有效目标），或者超时
        self.target_found_event.wait(timeout=timeout_sec)
        
        self.destroy_subscription(subscription)
        
        if not self.target_found_event.is_set():
            self.get_logger().warn("在规定时间内未检测到任何有效的目标。")

        return self.found_type, self.found_position

    def transfer(base_coords):
        # 提取旋转矩阵和平移向量
        T_base_to_camera = np.array([
        [0, 0, 9.828e-01, 0.07720],
        [-9.829e-01, 0, 0, -0.0165],
        [0, -9.999e-01, 0, 0.09  ],
                           # 注意，推理时，物体z坐标为0时，对应比基部低0.09米
                           # 因此这里加入-0.09作为偏移
        [0.00, 0.00, 0.00, 1.00]
    ])
        # 备用矩阵

        # T_base_to_camera=np.array([
        #  [-0.1815,  0.0489,  0.9822,  0.0787],
        #  [-0.9833,  0.0012, -0.1818, -0.017 ],
        #  [-0.01  , -0.9988,  0.0479, -0.0012],
        #  [ 0.  ,    0.  ,    0.  ,     1.    ]])

        # T_base_to_camera=np.array([
        #  [-1.841e-01 , 1.390e-02 , 9.828e-01 , 7.720e-02],
        #  [-9.829e-01 , 2.400e-03 , -1.841e-01  ,-1.650e-02],
        #  [ -2.000e-04 , -9.999e-01, -1.410e-02  ,  0],
        #  [ 0.000e+00 , 0.000e+00 , 0.000e+00  ,1.000e+00]])

        rotation_matrix = T_base_to_camera[:3, :3]  # 左上角3x3的旋转矩阵
        translation_vector = T_base_to_camera[:3, 3]  # 最右边一列的平移向量
        
        # 旋转基部坐标
        rotated_coords = np.dot(rotation_matrix, base_coords)
        # 平移处理：减去平移向量
        camera_coords = rotated_coords + translation_vector
        return camera_coords
    

    def get_positions(self):
        target_object_list = ['TV', 'APPLE', 'BOTTLE']
        found_object_type, target_coords_list = self.get_target_coordinates(
            target_types=target_object_list, 
            timeout_sec=30.0
        )       

        if found_object_type is None:
            print(f"错误：未能从列表 {target_object_list} 中找到任何有效的目标，程序将退出。")
            return # 退出 main 函数


        if found_object_type.upper() == 'BOTTLE':
            print("决策：检测到瓶子。使用 'bottle' 配置。")
                # 假设瓶子的权重文件叫这个名字
            self.checkpoint_path_to_use = "./runs/LeapHand_xie.pth" 
            self.dof_limit_type_to_use = 'xie' # 使用倾斜的关节限制
            self.object_type = 'BOTTLE'
        elif found_object_type.upper() == 'APPLE' or found_object_type.upper() == 'TV':
            print(f"决策：检测到 '{found_object_type}'。使用默认配置。")
                # 默认的权重文件
            self.checkpoint_path_to_use = "./runs/LeapHand.pth"
            self.dof_limit_type_to_use = 'default' # 使用普通的关节限制
            # 将坐标列表转换为 numpy 数组
            self.object_type = found_object_type.upper()
        else:
            self.get_logger().warn(f"错误：检测到未知目标 '{found_object_type}'，程序将退出。")
            return
        external_target_position = np.array([target_coords_list])
        goal_position = self.transfer(external_target_position)
        goal_position[2]=0
        self.found_position = goal_position
        return



# # 这部分代码用于独立测试该模块
# if __name__ == '__main__':
#     rclpy.init()
#     detector_node = TargetDetector()
    
#     # 调用函数获取坐标
#     tv_coords = detector_node.get_tv_coordinates(timeout_sec=15)
    
#     if tv_coords:
#         print(f"\n[主程序] 成功获取坐标: {tv_coords}")
#     else:
#         print("\n[主程序] 在规定时间内未检测到'TV'。")
        
#     # 销毁节点并关闭rclpy
#     detector_node.destroy_node()
#     rclpy.shutdown()