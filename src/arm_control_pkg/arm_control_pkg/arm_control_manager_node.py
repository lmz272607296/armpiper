#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
import subprocess
import threading
import os
import time
import signal 

from robot_interfaces.srv import StringString 

class ArmControlManagerNode(Node):
    def __init__(self):
        super().__init__('arm_control_manager_node')

        self.get_logger().info("机械臂控制管理器节点启动")

        self.is_torque_open = False
        self.torque_process = None 
        self.current_action_name = "无" 
        self.current_action_process = None 
        
        self.lock = threading.Lock()

        self.arm_command_service = self.create_service(
            StringString,             
            '/arm_manager/command',   
            self.handle_arm_command   
        )

        self.arm_status_publisher = self.create_publisher(
            String, '/arm_manager/status', 10)

        self.ros_env_setup = "source /opt/ros/humble/setup.bash && source /root/colcon_ws/install/setup.bash && "

        # 启动时发布一次初始状态
        self.publish_arm_status() 

    def publish_arm_status(self):
        msg = String()
        status_text = f"扭矩: {'已打开' if self.is_torque_open else '已关闭'}|当前动作: {self.current_action_name}"
        msg.data = status_text
        self.arm_status_publisher.publish(msg)
        self.get_logger().info(f"发布机械臂状态: {status_text}")

    # --- 修改开始：增加管道读取线程，防止阻塞 ---
    def _read_pipe(self, pipe, name, logger_func):
        """在一个单独的线程中读取管道内容并用日志记录器打印"""
        for line in iter(pipe.readline, b''):
            log_line = line.decode('utf-8', errors='ignore').strip()
            if log_line:
                logger_func(f"[{name}]: {log_line}")
        pipe.close()
        self.get_logger().info(f"'{name}' pipe reader finished.")

    def _start_process(self, command_parts, name):
        """启动一个子进程，并为stdout/stderr启动读取线程"""
        full_command = self.ros_env_setup + " ".join(command_parts)
        self.get_logger().info(f"尝试启动 '{name}' 命令: {full_command}")
        try:
            # 关键：明确使用/bin/bash，并捕获管道
            process = subprocess.Popen(
                full_command, 
                shell=True, 
                executable='/bin/bash', 
                preexec_fn=os.setsid, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.get_logger().info(f"'{name}' 进程 (PID: {process.pid}) 启动成功。")
            
            # 启动线程来实时读取stdout和stderr，防止管道阻塞
            threading.Thread(target=self._read_pipe, args=(process.stdout, f"{name} stdout", self.get_logger().info), daemon=True).start()
            threading.Thread(target=self._read_pipe, args=(process.stderr, f"{name} stderr", self.get_logger().error), daemon=True).start()

            return process
        except Exception as e:
            self.get_logger().error(f"启动 '{name}' 进程失败: {e}")
            return None

    def _stop_process(self, process, name):
        """安全地停止一个子进程及其进程组"""
        if process and process.poll() is None: 
            self.get_logger().info(f"尝试停止 '{name}' 进程组 (PGID: {os.getpgid(process.pid)})...")
            try:
                # 使用 killpg 发送信号到整个进程组
                os.killpg(os.getpgid(process.pid), signal.SIGTERM) 
                time.sleep(1) 
                if process.poll() is None: 
                    self.get_logger().warn(f"'{name}' 进程未响应 SIGTERM，强制使用 SIGKILL。")
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=2) 
                self.get_logger().info(f"'{name}' 进程已终止。")
            except (ProcessLookupError, OSError):
                self.get_logger().info(f"'{name}' 进程在停止操作期间已消失。")
            except Exception as e:
                self.get_logger().error(f"停止 '{name}' 进程时发生错误: {e}")
            finally:
                process = None
        return process
    # --- 修改结束 ---

    def handle_arm_command(self, request, response):
        command = request.data.data.strip() 
        self.get_logger().info(f"收到机械臂命令服务请求: '{command}'")

        response_msg_data = "未知错误。" 

        with self.lock: 
            if command == "打开扭矩":
                if self.is_torque_open:
                    response_msg_data = "扭矩已处于打开状态。"
                else:
                    # 确保旧进程已清理
                    self.torque_process = self._stop_process(self.torque_process, "旧的扭矩进程")
                    self.torque_process = self._start_process(
                        ["ros2", "launch", "open_manipulator_x_bringup", "hardware.launch.py"], 
                        "打开扭矩"
                    )
                    if self.torque_process:
                        # 给予一点时间让launch文件启动并稳定
                        time.sleep(2) 
                        self.is_torque_open = True
                        response_msg_data = "扭矩已成功打开。"
                    else:
                        response_msg_data = "扭矩打开失败。"
                self.publish_arm_status() 

            elif command == "关闭扭矩":
                if not self.is_torque_open:
                    response_msg_data = "扭矩已处于关闭状态。"
                else:
                    if self.current_action_process:
                        self.current_action_process = self._stop_process(self.current_action_process, self.current_action_name)
                        self.current_action_name = "无" 
                    self.torque_process = self._stop_process(self.torque_process, "打开扭矩")
                    self.is_torque_open = False
                    response_msg_data = "扭矩已成功关闭。"
                self.publish_arm_status() 

            # ... 其他命令处理逻辑保持不变 ...
            elif command == "停止机械臂动作":
                if self.current_action_process:
                    self.current_action_process = self._stop_process(self.current_action_process, self.current_action_name)
                    self.current_action_name = "无"
                    response_msg_data = "当前机械臂动作已停止。"
                else:
                    response_msg_data = "没有正在运行的机械臂动作。"
                self.publish_arm_status()

            elif command in ["转方块", "剪刀石头布", "打招呼", "展平手"]:
                if not self.is_torque_open:
                    response_msg_data = "请先打开机械臂扭矩。"
                elif self.current_action_process and self.current_action_process.poll() is None:
                    response_msg_data = f"已有动作'{self.current_action_name}'正在进行中。"
                else:
                    leapsim_action_map = {
                        "转方块": "rot", "剪刀石头布": "game", "打招呼": "hello", "展平手": "zero"
                    }
                    action_arg = leapsim_action_map.get(command)
                    self.current_action_process = self._start_process(
                        ["ros2", "run", "leapsim", action_arg], command
                    )
                    if self.current_action_process:
                        self.current_action_name = command
                        response_msg_data = f"'{command}' 动作已启动。"
                        threading.Thread(target=self._monitor_action_process, 
                                         args=(self.current_action_process, command), daemon=True).start()
                    else:
                        response_msg_data = f"'{command}' 动作启动失败。"
                self.publish_arm_status()

            else:
                response_msg_data = f"未知机械臂命令: {command}"
        
        response.response.data = response_msg_data 
        return response

    def _monitor_action_process(self, process, name):
        """监控动作进程，结束后更新状态"""
        if process:
            process.wait()
        
        self.get_logger().info(f"'{name}' 动作进程已结束。")
        with self.lock: 
            # 确保我们只清理我们监控的那个进程
            if self.current_action_process == process: 
                self.current_action_process = None
                self.current_action_name = "无"
                self.publish_arm_status() 

def main(args=None):
    rclpy.init(args=args)
    # 多线程执行器是必须的，因为我们有服务回调和后台线程
    executor = rclpy.executors.MultiThreadedExecutor() 
    node = ArmControlManagerNode()
    executor.add_node(node)
    
    try:
        executor.spin() 
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("正在关闭机械臂控制管理器...")
        # 确保在退出时清理所有子进程
        with node.lock: 
            node._stop_process(node.current_action_process, node.current_action_name)
            node._stop_process(node.torque_process, "打开扭矩")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
