import time
from time import sleep
from attr import has
# import isaacgym
import xml.etree.ElementTree as ET
import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import to_absolute_path
import torch 
from leapsim.utils.reformat import omegaconf_to_dict, print_dict
# from leapsim.utils.utils import set_np_formatting, set_seed, get_current_commit_hash
from leapsim.utils.no_isaac_rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
# from leapsim.utils.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner, _override_sigma
from rl_games.algos_torch import model_builder
from leapsim.learning import no_isaac_amp_continuous

# from leapsim.learning import amp_continuous

from leapsim.learning import amp_players
from leapsim.learning import amp_models
from leapsim.learning import amp_network_builder
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from collections import deque
import math
import random
# from grasp_dependencies.target_detector import TargetDetector
from leapsim.grasp_dependencies.grasp_pose import Grasp
import threading 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from leapsim.hand_controller import LeapHand
from leapsim.arm_controller import LeapArm

def _restore(player, args):
    """
    从检查点文件（checkpoint）中恢复模型和运行时的均值/标准差。
    """
    checkpoint_path = args.get('checkpoint')
    if checkpoint_path:
        expanded_path = os.path.expanduser(checkpoint_path)
        print(f"=> loading checkpoint '{expanded_path}'")
        # 关键！这里使用了 map_location
        checkpoint = torch.load(expanded_path, map_location=player.device, weights_only=False)
        player.model.load_state_dict(checkpoint['model'])
        
        if player.normalize_input and 'running_mean_std' in checkpoint:
            player.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            
            print("\n=== 运行时统计信息检查 ===")
            
            # 方法1：直接从checkpoint读取原始数据
            print("1. 从checkpoint文件中的原始数据:")
            running_mean_std_data = checkpoint['running_mean_std']
            for key, value in running_mean_std_data.items():
                if torch.is_tensor(value):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    print(f"    min={value.min():.6f}, max={value.max():.6f}, mean={value.mean():.6f}")
                    if 'mean' in key.lower():
                        print(f"    obs_means前5个值: {value.flatten()[:5]}")
                    elif 'var' in key.lower():
                        obs_std = torch.sqrt(value + 1e-8)
                        print(f"    obs_std前5个值: {obs_std.flatten()[:5]}")
                else:
                    print(f"  {key}: {value}")
            
            # 方法2：尝试从loaded model中读取
            print("\n2. 从已加载模型中的数据:")
            try:
                # 检查running_mean_std对象的所有属性
                rms_obj = player.model.running_mean_std
                print(f"  running_mean_std对象类型: {type(rms_obj)}")
                
                # 列出所有属性
                attrs = [attr for attr in dir(rms_obj) if not attr.startswith('_')]
                print(f"  可用属性: {attrs}")
                
                # 尝试不同的属性名称
                mean_attrs = ['running_mean', 'mean', 'running_means']
                var_attrs = ['running_var', 'var', 'variance', 'running_vars']
                
                obs_mean = None
                obs_var = None
                
                # 查找均值
                for attr in mean_attrs:
                    if hasattr(rms_obj, attr):
                        obs_mean = getattr(rms_obj, attr)
                        print(f"  找到均值属性 '{attr}': shape={obs_mean.shape}")
                        print(f"    obs_means前5个值: {obs_mean.flatten()[:5]}")
                        break
                
                # 查找方差
                for attr in var_attrs:
                    if hasattr(rms_obj, attr):
                        obs_var = getattr(rms_obj, attr)
                        obs_std = torch.sqrt(obs_var + 1e-8)
                        print(f"  找到方差属性 '{attr}': shape={obs_var.shape}")
                        print(f"    obs_std前5个值: {obs_std.flatten()[:5]}")
                        break
                
                if obs_mean is None:
                    print("  警告：未找到均值属性")
                if obs_var is None:
                    print("  警告：未找到方差属性")
                    
            except Exception as e:
                print(f"  从模型中读取统计信息时出错: {e}")
            
            # 方法3：尝试调用对象的方法或属性来获取统计信息
            print("\n3. 尝试其他方式获取统计信息:")
            try:
                rms_obj = player.model.running_mean_std
                
                # 检查是否有state_dict方法
                if hasattr(rms_obj, 'state_dict'):
                    state = rms_obj.state_dict()
                    print("  通过state_dict()获取:")
                    for key, value in state.items():
                        print(f"    {key}: shape={value.shape}")
                        if 'mean' in key.lower():
                            print(f"      obs_means前5个值: {value.flatten()[:5]}")
                        elif 'var' in key.lower():
                            obs_std = torch.sqrt(value + 1e-8)
                            print(f"      obs_std前5个值: {obs_std.flatten()[:5]}")
                
                # 检查是否有直接的tensor属性
                if hasattr(rms_obj, 'count'):
                    print(f"  count: {rms_obj.count}")
                    
            except Exception as e:
                print(f"  其他方式获取统计信息时出错: {e}")
            
            print("=== 统计信息检查完成 ===\n")
            
        print(f"=> loaded checkpoint '{expanded_path}' successfully.")

class HardwarePlayer(object):
    def __init__(self, config, checkpoint_path, dof_limit_type):
        self.config = omegaconf_to_dict(config)
        self.set_defaults()
        self.action_scale = 1 / 24
        self.actions_num = 21

        self.device = 'cpu'

        self.debug_viz = self.config["task"]['env']['enableDebugVis']
        self.checkpoint_to_load = checkpoint_path
        self.external_target_position = None
        # hand setting
        self.init_pose = np.zeros(23)
        self.init_pose = self.fetch_grasp_state()
        print(f"正在应用 '{dof_limit_type}' 类型的关节限制...")
        if dof_limit_type == 'xie':
            self.get_dof_limits_xie()
        else: # 默认为普通模式
            self.get_dof_limits()

        if self.debug_viz:
            self.setup_plot()

        self.stable=None#用于存储稳定的推理结果。
        self.object=0
        self.scaled_action=None #用于存储分解的动作。

    def initpose(self):
        leap= LeapHand("hand")
        arm= LeapArm("arm")

        if rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
        
        init_hand = np.zeros((1, 16))   
        init_arm  = np.zeros((1, 7))
        # 拼接初始位置
        init_hand = np.concatenate((leap.raw_positions, init_hand), axis=0)
        init_arm = np.concatenate((arm.raw_positions, init_arm), axis=0)

        # 发送初始位置
        leap.command_joint_position(init_hand, 2)
        arm.command_joint_position(init_arm, 2.5)
        time.sleep(4)
        # 误差回调
        if rclpy.ok():
            rclpy.spin_once(leap, timeout_sec=0.1)
            arm.recover(init_arm, leap.raw_positions, 1)
            time.sleep(2)
            rclpy.spin_once(leap, timeout_sec=0.1) # 获取当前状态作为初始值
            
            history_hand=leap.raw_positions
            history_arm=leap.raw_arm_positions

        leap.destroy_node()
        arm.destroy_node()

        return history_hand, history_arm
    


    def real_to_sim(self, values):
        if not hasattr(self, "real_to_sim_indices"):
            self.construct_sim_to_real_transformation()

        return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        if not hasattr(self, "sim_to_real_indices"):
            self.construct_sim_to_real_transformation()
        
        return values[:, self.sim_to_real_indices]

    def construct_sim_to_real_transformation(self):
        self.sim_to_real_indices = self.config["task"]["env"]["sim_to_real_indices"]
        self.real_to_sim_indices= self.config["task"]["env"]["real_to_sim_indices"]

    def get_dof_limits(self):
        self.leap_dof_upper = np.array([
        0.0, 0.0, 0.0,0.0,0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.14, 1.5, 1.4, 0.5, 0.001])
        self.leap_dof_lower =np.array([
        -0.01, -0.01,-0.01,-0.01,
        -0.01, -0.01,-0.01,-0.01,
        -0.01, -0.01,-0.01,-0.01,
        -0.01, -0.01,-0.01,-0.01,
        -3.14, -1.5, -1.5, -1.97, 0.0
        ])

        self.leap_dof_lower = np.array(self.leap_dof_lower)[None, :] 
        self.leap_dof_upper = np.array(self.leap_dof_upper)[None, :] 

        self.leap_dof_lower=np.reshape(self.leap_dof_lower, (1, 21))
        self.leap_dof_upper=np.reshape(self.leap_dof_upper, (1, 21))

        self.leap_dof_lower = self.real_to_sim(self.leap_dof_lower).squeeze()
        self.leap_dof_upper = self.real_to_sim(self.leap_dof_upper).squeeze()
    def get_dof_limits_xie(self):
        self.leap_dof_upper = np.array([
        0.0, 0.0, 0.0,0.0,0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.14, 1.5, 1.4, 0.5, 1.5701])
        self.leap_dof_lower =np.array([
        -0.01, -0.01,-0.01,-0.01,
        -0.01, -0.01,-0.01,-0.01,
        -0.01, -0.01,-0.01,-0.01,
        -0.01, -0.01,-0.01,-0.01,
        -3.14, -1.5, -1.5, -1.97, 1.5
        ])

        self.leap_dof_lower = np.array(self.leap_dof_lower)[None, :] 
        self.leap_dof_upper = np.array(self.leap_dof_upper)[None, :] 

        self.leap_dof_lower=np.reshape(self.leap_dof_lower, (1, 21))
        self.leap_dof_upper=np.reshape(self.leap_dof_upper, (1, 21))

        self.leap_dof_lower = self.real_to_sim(self.leap_dof_lower).squeeze()
        self.leap_dof_upper = self.real_to_sim(self.leap_dof_upper).squeeze()
    def plot_callback(self):
        self.fig.canvas.restore_region(self.bg)

        # self.ydata.append(self.object_rpy[0, 2].item())
        self.ydata.append(self.cur_obs_joint_angles[0, 9].item())
        self.ydata2.append(self.cur_obs_joint_angles[0, 4].item())

        self.ln.set_ydata(list(self.ydata))
        self.ln.set_xdata(range(len(self.ydata)))

        self.ln2.set_ydata(list(self.ydata2))
        self.ln2.set_xdata(range(len(self.ydata2)))

        self.ax.draw_artist(self.ln)
        self.ax.draw_artist(self.ln2)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
    
    def setup_plot(self):   
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-1, 1)
        self.ydata = deque(maxlen=100) # Plot 5 seconds of data 
        self.ydata2 = deque(maxlen=100)
        (self.ln,) = self.ax.plot(range(len(self.ydata)), list(self.ydata), animated=True)
        (self.ln2,) = self.ax.plot(range(len(self.ydata2)), list(self.ydata2), animated=True)
        plt.show(block=False)
        plt.pause(0.1)

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.ln)
        self.fig.canvas.blit(self.fig.bbox)

    def set_defaults(self):
        if "include_history" not in self.config["task"]["env"]:
            self.config["task"]["env"]["include_history"] = True

        if "include_targets" not in self.config["task"]["env"]:
            self.config["task"]["env"]["include_targets"] = True

    def fetch_grasp_state(self, s=1.0):
        self.grasp_cache_name = self.config['task']['env']['grasp_cache_name']
        grasping_states = np.zeros((1, 21)) #np.load(f'cache/{self.grasp_cache_name}_grasp_50k_s{str(s).replace(".", "")}.npy')

        if "sampled_pose_idx" in self.config["task"]["env"]:
            idx = self.config["task"]["env"]["sampled_pose_idx"]
        else:
            idx = random.randint(0, grasping_states.shape[0] - 1)

        return grasping_states[idx][:21] # first 16 are hand dofs, last 16 is object state

    def deploy(self, found_object_type, external_target_position):
        #     # 根据设备类型确定打印头
        # actor_type = 'CPU' if not hasattr(self, 'use_bpu') or not self.use_bpu else 'BPU'
    
        # print("\n" + "="*20 + f" {actor_type}-侧 DEPLOYMENT INFO " + "="*20)

        #  # 1. 打印关节顺序 (sim_to_real_indices)
        # if hasattr(self, 'sim_to_real_indices'):
        #     print(f"\n[1] 关节顺序 (sim_to_real_indices):")
        #     print(f"    {self.sim_to_real_indices}")
        # else:
        #     # 如果还没初始化，就手动初始化一下
        #     self.construct_sim_to_real_transformation()
        #     print(f"\n[1] 关节顺序 (sim_to_real_indices):")
        #     print(f"    {self.sim_to_real_indices}")

        # # 2. 打印重排后的关节限制
        # print(f"\n[2] 重排后的关节限制 (用于unscale):")
        # print(f"    Lower [前5]: {self.leap_dof_lower[:5]}")
        # print(f"    Upper [前5]: {self.leap_dof_upper[:5]}")
        # # 打印最可疑的那个索引
        # print(f"    检查索引4: Lower={self.leap_dof_lower[4]:.6f}, Upper={self.leap_dof_upper[4]:.6f}")

        # print("="*60 + "\n")
        
        # # 将坐标列表转换为模型需要的numpy数组格式
        self.external_target_position = external_target_position
        leap= LeapHand("hand")
        arm= LeapArm("arm")
        # 根据找到的物体类型，设置用于后续抓取的标志
        if found_object_type.upper() == 'TV': # 假设TV是方块
            self.object = 1
        elif found_object_type.upper() == 'BOTTLE':
            self.object = 2
        elif found_object_type.upper() == 'APPLE':
            self.object = 1 # 假设苹果也用方块抓取方式
        else:
            self.object = 0 # 未知的物体类型

        # 修正这里的打印信息
        print(f"--- 成功锁定目标 '{found_object_type}'，坐标: {self.external_target_position}，开始推理... ---")

        # try to set up rospy
        num_obs = self.config['task']['env']['numObservations'] 
        num_obs_single = num_obs // 3 
        hz=1/20

        #数据说明
        # leap.raw_positions: 形状为 (1, 16)，顺序为real
        # leap.all_positions: 形状为 (1, 23)，顺序为real
        # commands: 形状为 (1, 21)，          顺序为sim
        # history_command: 形状为 (1, 7)，    顺序为real & sim
        # obses: 形状为(1, 21)，              顺序为sim

        obses= np.zeros(21)
        history_command= np.zeros((1, 7))

        # hardware deployment buffer
        obs_buf = np.zeros((1, 0), dtype=np.float32)

        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)

        obses = obses.astype(np.float32)
        prev_target = obses[None].copy()
        cur_obs_buf = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)[None]

        if self.config["task"]["env"]["include_history"]:
            num_append_iters = 3
        else:
            num_append_iters = 1

        for i in range(num_append_iters):   
            obs_buf = np.concatenate([obs_buf, cur_obs_buf.copy()], axis=-1)
            # print(f"添加关节状态后维度: {obs_buf.shape[1]}")
            if self.config["task"]["env"]["include_targets"]:
                obs_buf = np.concatenate([obs_buf, prev_target.copy()], axis=-1)
                # print(f"添加目标位置后维度: {obs_buf.shape[1]}")
            if self.config["task"]["env"]["include_obj_pose"]:
                        # 设置你的目标位置 - 这需要根据实际任务调整
                #external_target_position = np.array([[0.34, 0.15, 0.00]])  # [x, y, z]
                self.external_target_position = np.reshape(self.external_target_position, (1, 3))  # 确保是二维数组
                # print(f"添加外部目标位置: {self.external_target_position.shape}")
                obs_buf = np.concatenate([obs_buf, self.external_target_position], axis=-1)
                # print(f"添加外部目标后维度: {obs_buf.shape[1]}")
        
        if "obs_mask" in self.config["task"]["env"]:
            obs_buf = obs_buf * np.array(self.config["task"]["env"]["obs_mask"])[None, :]

        obs_buf = obs_buf.astype(np.float32)

        counter = 0 

        if "debug" in self.config["task"]["env"]:
            self.obs_list = []
            self.target_list = []

            if "record" in self.config["task"]["env"]["debug"]:
                self.record_duration = int(self.config["task"]["env"]["debug"]["record"]["duration"] / self.control_dt)

            if "actions_file" in self.config["task"]["env"]["debug"]:
                self.actions_list = np.load(self.config["task"]["env"]["debug"]["actions_file"])        
                self.record_duration = self.actions_list.shape[0]

        if self.player.is_rnn:
            self.player.init_rnn()
        start_time = time.time()
        
        time_check=False
        publish=False
        publish_true = False
        while True:
            counter += 1
            # obs = self.running_mean_std(obs_buf.clone()) # ! Need to check if this is implemented 
            if hasattr(self, "actions_list"):
                action = self.actions_list[counter-1][None, :]
            else:
                action = self.forward_network(obs_buf)

            action = np.clip(action, -1.0, 1.0)

            if "actions_mask" in self.config["task"]["env"]:
                action = action * np.array(self.config["task"]["env"]["actions_mask"])[None, :]

            target = prev_target + self.action_scale * action 
            target = np.clip(target, self.leap_dof_lower, self.leap_dof_upper)
            # commands = data[counter,:]
            prev_target = target.copy()

            #---------------------------------------------
            # 推理结果发布
            commands = target[0]
            
            if counter >600:
                self.stable=commands[:5]
                if time.time() - start_time < 12:
                    print("推理耗时: {:.4f} 秒".format(time.time() - start_time))
                    return
                else : 
                    print("推理耗时: {:.4f} 秒".format(time.time() - start_time))
                    return

            if "disable_actions" not in self.config["task"]["env"]:
                if counter == 300 or counter == 600 or counter == 6000 or counter == 200:
                    endtime = time.time()

                    # print(f"第 {counter} 步: 发送动作 {commands}")
                    # print(f"推理耗时: {endtime - start_time:.4f} 秒")

#--------------------------------------------------------------------------------

            
            # if publish and rclpy.ok() and counter>400:
            #     # rclpy.spin_once(arm, timeout_sec=0.1)
            #     if time.time() - publish_time > 1.5:
            #         publish=False

            #         rclpy.spin_once(leap, timeout_sec=0.1)

            #         # 发布动作
            #         print("command",commands)
            #         arm_command= self.reorder(commands[None, :5]) 
            #         print("arm_command",arm_command)


            #         if found_object_type.upper() == 'APPLE':
            #             publish_arm= np.concatenate((leap.raw_arm_positions, arm_command), axis=0)
            #         else:
            #             angle= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            #             angle = np.reshape(angle, (1, 7))

            #             angle[:,0:1]=arm_command[:,0:1]
            #             publish_arm= np.concatenate((leap.raw_arm_positions, angle), axis=0)
            #             arm.command_joint_position(publish_arm, 0.5)
            #             time.sleep(1)

            #         print("中间推理\n")
            #         print("publish",publish_arm)
            #         rclpy.spin_once(leap, timeout_sec=0.1)

            #         publish_arm=self.scale_action(leap.raw_arm_positions,arm_command,15)
            #         arm.command_joint_position(publish_arm, 0.4)
            #         publishnew_time = time.time()
            #         publish_true=True
            
            # if rclpy.ok() and publish_true:  
            #     if time.time() - publishnew_time > 6:
            #         publish_true = False
            #         rclpy.spin_once(arm, timeout_sec=0.1)
            #         rclpy.spin_once(leap, timeout_sec=0.1)  
            #         init_arm = np.zeros((1, 7))         
            #         arm.recover(init_arm, leap.raw_arm_positions, 1)

        

                obses = commands


            obses = obses.astype(np.float32)

            # obs_buf_list.append(obses.cpu().numpy().squeeze())
            cur_obs_buf = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)[None]
            self.cur_obs_joint_angles = cur_obs_buf.copy()

            if self.debug_viz:
                self.plot_callback()

            if hasattr(self, "obs_list"):
                self.obs_list.append(cur_obs_buf[0].copy())
                self.target_list.append(target[0].copy().squeeze())

                if counter == self.record_duration - 1:
                    self.obs_list = np.stack(self.obs_list, axis=0)

                    self.target_list = np.stack(self.target_list, axis=0)

                    if "actions_file" in self.config["task"]["env"]["debug"]:
                        actions_file = os.path.basename(self.config["task"]["env"]["debug"]["actions_file"])
                        folder = os.path.dirname(self.config["task"]["env"]["debug"]["actions_file"])
                        suffix = "_".join(actions_file.split("_")[1:])
                        joints_file = os.path.join(folder, "joints_real_{}".format(suffix)) 
                        target_file = os.path.join(folder, "targets_real_{}".format(suffix))
                    else:
                        suffix = self.config["task"]["env"]["debug"]["record"]["suffix"]
                        joints_file = "debug/joints_real_{}.npy".format(suffix)
                        target_file = "debug/targets_real_{}.npy".format(suffix)

                    np.save(joints_file, self.obs_list)
                    np.save(target_file, self.target_list) 
                    exit()

            if self.config["task"]["env"]["include_history"]:
                obs_buf = obs_buf[:, num_obs_single:].copy()
            else:
                obs_buf = np.zeros((1, 0), dtype=np.float32)

            obs_buf = np.concatenate([obs_buf, cur_obs_buf.copy()], axis=-1)

            if self.config["task"]["env"]["include_targets"]:
                obs_buf = np.concatenate([obs_buf, target.copy()], axis=-1)

                # 【关键修复】添加外部目标位置 (3维)
            if self.config["task"]["env"]["include_obj_pose"]:
                # external_target_position = np.array([[0.34, 0.15, 0.00]])
                obs_buf = np.concatenate([obs_buf, self.external_target_position], axis=-1)


            if "obs_mask" in self.config["task"]["env"]:
                obs_buf = obs_buf * np.array(self.config["task"]["env"]["obs_mask"])[None, :]

            obs_buf = obs_buf.astype(np.float32)

        leap.destroy_node()
        arm.destroy_node()


    def forward_network(self, obs):
        # if not hasattr(self, 'debug_printed'):
        #     print("\n" + "="*20 + " CPU-侧 输入向量详细分解 " + "="*20)
        #     print(f"obs_buf 总形状: {obs.shape}")
        
        #     num_obs_single = obs.shape[1] // 3
        #     print(f"单帧观测维度 (num_obs_single): {num_obs_single}")

        #     # 分解历史数据
        #     for i in range(3):
        #         start_idx = i * num_obs_single
        #         end_idx = (i + 1) * num_obs_single
            
        #        # 分解单帧数据
        #         cur_obs_start = start_idx
        #         cur_obs_end = start_idx + 21
        #         prev_target_start = cur_obs_end
        #         prev_target_end = prev_target_start + 21
        #         obj_pose_start = prev_target_end
        #         obj_pose_end = obj_pose_start + 3

        #         print(f"\n--- 历史帧 T-{2-i} ---")
            
        #         cur_obs_part = obs[0, cur_obs_start:cur_obs_end]
        #         print(f"  cur_obs_buf (归一化关节角) [前5]: {cur_obs_part[:5]}")
            
        #         prev_target_part = obs[0, prev_target_start:prev_target_end]
        #         print(f"  prev_target (原始目标关节角) [前5]: {prev_target_part[:5]}")
            
        #         obj_pose_part = obs[0, obj_pose_start:obj_pose_end]
        #         print(f"  obj_pose (目标物体位置) [3个]: {obj_pose_part}")

        #     # 打印整个向量的前25个元素，用于直接对比
        #     print(f"\n完整 obs_buf [前25]: {obs[0, :25]}")
        #     print("="*60 + "\n")
        
        #     self.debug_printed = True
        obs_tensor = torch.from_numpy(obs).to(self.device)
        action_tensor = self.player.get_action(obs_tensor, is_deterministic=True)


        return action_tensor.cpu().numpy()

    def restore(self):
        rlg_config_dict = self.config['train']
        if 'params' in rlg_config_dict and 'config' in rlg_config_dict['params']:
            rlg_config_dict['params']['config']['device'] = 'cpu'
            rlg_config_dict['params']['config']['device_name'] = 'cpu'
            rlg_config_dict["params"]["config"]["env_info"] = {}
        self.num_obs = self.config["task"]["env"]["numObservations"]
        self.num_actions = 21
        observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        rlg_config_dict["params"]["config"]["env_info"]["observation_space"] = observation_space
        action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
        rlg_config_dict["params"]["config"]["env_info"]["action_space"] = action_space
        rlg_config_dict["params"]["config"]["env_info"]["agents"] = 1

        def build_runner(algo_observer):
            runner = Runner(algo_observer)

            # 如果要在没有Isaac Gym的环境中运行，使用no_isaac_amp_continuous
            # 反之，去掉no_isaac前缀,同时修改引用
            runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : no_isaac_amp_continuous.AMPAgent(**kwargs))
            runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
            model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
            model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

            return runner

        runner = build_runner(RLGPUAlgoObserver())
        runner.load(rlg_config_dict)
        runner.reset()

        args = {
            'train': False,
            'play': True,
            'checkpoint' : self.checkpoint_to_load,
            'sigma' : None
        }

        self.player = runner.create_player()
        _restore(self.player, args)
        _override_sigma(self.player, args)

    def reorder(self,data):
        #从21维数据中提取出23维数据，数据顺序必须为sim
        if data.shape[1] < 3:
            print("错误：数组的列数少于3，无法执行操作。")
        else:
            part1 = data[:, 0:1]
            part2 = data[:, 1:2]
            part3 = -data[:, 1:2]
            part4 = data[:, 2:3]
            part5 = -data[:, 2:3]
            part6 = -data[:, 3:4]
            part7 =  data[:, 4:]
            new_array = np.concatenate((part1, part2, part3, part4, part5, part6, part7), axis=1)
        return new_array
    
    def order(self,data):
        #从23维数据中提取出21维数据, 数据顺序必须为sim
        if data.shape[1] < 23:
            print("错误：数组的列数少于23，无法执行操作。")
        else:
            part1 = data[:, 0:1]
            part2 = data[:, 1:2]
            part3 = data[:, 3:4]
            part4 = -data[:, 5:6]
            part5 = data[:, 6:7]
            part6 = data[:, 7:]
            new_array = np.concatenate((part1, part2, part3, part4, part5, part6), axis=1)
        return new_array
    
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

    def control(self, data):

        import time
        leap = None
        arm = None
        grasp = None
        
        try:
            leap = LeapHand("hand")
            arm = LeapArm("arm")
            grasp = Grasp("grasp")

            # self.initpose()

            #--------------------------------------------
            # 开始推理执行
            if rclpy.ok():
                rclpy.spin_once(leap, timeout_sec=0.1) # 获取当前状态作为初始值
                scale=15                           # 动作分解个数
            
            #----------------------------------------------
            #  避免碰倒瓶子进行分别处理：
                if  self.object == 1: # 方块
                    print("方块在这里")
                    self.scaled_action = self.scale_action(leap.raw_arm_positions, data, scale)
                    speed = 0.4
                    arm.command_joint_position(self.scaled_action, speed)
                    time.sleep(scale*speed+1)
                
                elif self.object == 2: # 瓶子
                    first_action=np.zeros((1, 7)) # 初始化第一个动作
                    opration=np.array([0, -0.209, 0.209, -0.179, 0.179, 0.21, 0])
                    # 先将手部电机旋转到位

                    first_action[:,0:1]= data[:,0:1] # 先将根部电机旋转到位
                    
                    rclpy.spin_once(leap, timeout_sec=0.1)

                    first_action=np.concatenate((leap.raw_arm_positions, first_action), axis=0) # 拼接手部位置
                    arm.command_joint_position(first_action, 1.5)
                    time.sleep(3)

                    #开始移动其他电机
                    rclpy.spin_once(leap, timeout_sec=0.1)
                    updated_arm_pos = np.array(leap.raw_arm_positions)
                    if updated_arm_pos.ndim == 1:
                        updated_arm_pos = updated_arm_pos.reshape(1, -1)
                    self.scaled_action = self.scale_action(updated_arm_pos, data, scale)
                    speed = 0.3
                    arm.command_joint_position(self.scaled_action, speed)
                    time.sleep(scale*speed+2)

            #--------------------------------------------
            # 误差回调
                rclpy.spin_once(leap, timeout_sec=0.1)
                final_arm_pos = np.array(leap.raw_arm_positions)
                if final_arm_pos.ndim == 1:
                    final_arm_pos = final_arm_pos.reshape(1, -1)
                arm.recover(self.scaled_action, final_arm_pos, 1)
                
                print("误差回调完成",self.scaled_action, final_arm_pos)

                time.sleep(2)
                print("done")

            #--------------------------------------------
            # 夹持动作

                leap.destroy_node()
                arm.destroy_node()
                if self.object == 1:
                    grasp.graspcube()
                    grasp.loose_cube()
                elif self.object == 2:
                    grasp.graspbottle()
                    grasp.loose_bottle()
                else:
                    print("未指定夹持物体，请检查配置。")

        except Exception as e:
            print(f"控制过程中发生错误: {e}")


        if rclpy.ok():
            leap= LeapHand("hand")
            arm= LeapArm("arm")

            rclpy.spin_once(arm, timeout_sec=0.1)
            rclpy.spin_once(leap, timeout_sec=0.1)

            init_arm = np.zeros((1, 7))
            arm.recover(init_arm, leap.raw_arm_positions, 1)
            time.sleep(1)
            leap.destroy_node()
            arm.destroy_node()

        grasp.destroy_node()


class GraspControllerNode(Node):
    def __init__(self, config):
        super().__init__('grasp_controller_node')
        self.config = config
        self.task_running = False  # 添加任务状态标志
        self.object_type = None  # 用于存储物体类型
        self.position = None  # 用于存储抓取目标位置
        # 加载手眼标定矩阵
        self.load_hand_eye_calibration()
        
        # 创建订阅器
        self.subscription = self.create_subscription(
            PoseStamped,
            '/grasp_target_pose',
            self.grasp_target_callback,
            10)
        
        self.get_logger().info("抓取控制器节点已启动，等待抓取目标...")
        self.get_logger().info("使用命令测试: ros2 topic pub --once /grasp_target_pose geometry_msgs/msg/PoseStamped '{header: {frame_id: \"bottle\"}, pose: {position: {x: 0.0, y: 0.0, z: 0.4}}}'")

    def load_hand_eye_calibration(self):
        """加载手眼标定矩阵"""
        try:
            # 方法1：尝试从文件加载
            matrix_file = "hand_eye_calibration.txt"  # 替换为你的文件路径
            if os.path.exists(matrix_file):
                self.hand_eye_matrix = np.loadtxt(matrix_file)
                self.get_logger().info(f"已从文件加载手眼标定矩阵: {matrix_file}")
            else:
                # 方法2：使用默认矩阵（你需要替换为实际值）
                self.hand_eye_matrix = np.array([
                    [0, 0, 1, 0.07720],
                    [-1, 0, 0, -0.0165],
                    [0, -1, 0, 0.09  ],
                                   # 注意，推理时，物体z坐标为0时，对应比基部低0.09米
                                  # 因此这里加入-0.09作为偏移
                    [0.00, 0.00, 0.00, 1.00]   
                        ])
                self.get_logger().warn("使用默认手眼标定矩阵，请替换为实际标定结果！")
                
        except Exception as e:
            self.get_logger().error(f"加载手眼标定矩阵失败: {e}")
            # 使用单位矩阵作为后备
            self.hand_eye_matrix = np.eye(4)

    def camera_to_robot_transform(self, camera_coords):
        """
        将相机坐标系下的坐标转换为机器人基座坐标系下的坐标
        
        Args:
            camera_coords: [x, y, z] 相机坐标系下的坐标
            
        Returns:
            robot_coords: [x, y, z] 机器人坐标系下的坐标
        """
        # 转换为齐次坐标
        camera_coords_homo = np.array([camera_coords[0], camera_coords[1], camera_coords[2], 1.0])
        
        # 应用变换矩阵
        robot_coords_homo = self.hand_eye_matrix @ camera_coords_homo
        
        # 返回3D坐标
        robot_coords = robot_coords_homo[:3].tolist()
        robot_coords[2]=0
        
        return robot_coords

    def execute_grasp_task(self, object_type, position):
        """执行抓取任务"""
        agent = None
        try:
            # 根据物体类型选择合适的模型和配置
            if object_type.upper() == 'BOTTLE':
                checkpoint_path = "/home/lmz/colcon_ws/src/open_manipulator/leapsim/leapsim/runs/LeapHand_xie.pth"
                dof_limit_type = 'xie'
            else:
                checkpoint_path = "/home/lmz/colcon_ws/src/open_manipulator/leapsim/leapsim/runs/LeapHand.pth" 
                dof_limit_type = 'default'
            
            self.get_logger().info(f"使用模型: {checkpoint_path}, 配置: {dof_limit_type}")
            
            # --- 手眼标定坐标转换 ---
            # 将相机坐标系转换为机器人基座坐标系



            #---------------------------------------
            # 在这里调整位置偏置
            bottle_bias= np.array([0.1, 0.00, 0.1]) # 瓶子需要偏移0.1米
            apple_bias= np.array([0.0, 0, 0.1]) # 苹果不需要偏移
            if self.object_type.upper() == 'BOTTLE':
                position = position+bottle_bias

            else :
                position = position+apple_bias

            # 在这里调整位置偏置
            #---------------------------------------
                
            robot_position = self.camera_to_robot_transform(position)
            self.get_logger().info(f"相机坐标: {position} -> 机器人坐标: {robot_position}")
            
            # 创建HardwarePlayer实例
            agent = HardwarePlayer(
                config=self.config,
                checkpoint_path=checkpoint_path,
                dof_limit_type=dof_limit_type
            )
            
            # 恢复模型
            agent.restore()
            
            # 执行推理 - 使用转换后的机器人坐标
            print("object_postion",robot_position)
            agent.deploy(object_type, robot_position)
            
            # 如果推理成功，执行控制
            if agent.stable is not None:
                self.get_logger().warn(f"推理完成，推理坐标为{position}开始执行控制...")
                
                # 应用偏差


                bias = np.array([0.03, -0.05, -0.0, 0.12, 0])
                bottle_pbias=np.array([0.00, 0.0, -0.1, -0.1, 0])
                if self.object_type.upper() == 'BOTTLE':
                    agent.stable = agent.stable+bottle_pbias

                else :

                    agent.stable = agent.stable + bias


                agent.stable = agent.reorder(np.reshape(agent.stable, (1, 5)))
                
                self.get_logger().info(f"最终控制指令: {agent.stable}")
                
                # 执行控制
                agent.control(agent.stable)
                agent.stable = None
                
                self.get_logger().info("抓取任务完成！")
            else:
                self.get_logger().error("推理失败，无法执行控制")
                
        except Exception as e:
            self.get_logger().error(f"执行抓取任务时发生错误: {str(e)}")
            import traceback
            self.get_logger().error(f"详细错误信息:\n{traceback.format_exc()}")
        finally:
            # 清理资源
            if agent is not None:
                try:
                    agent.stable = None
                except:
                    pass
            
            # 重置任务状态
            self.task_running = False
            self.get_logger().info("任务执行完成，等待下一个任务...")

    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            if hasattr(self, 'subscription'):
                self.destroy_subscription(self.subscription)
        except:
            pass

    def grasp_target_callback(self, msg):
        """处理接收到的抓取目标"""
        # 检查是否有任务正在运行
        if self.task_running:
            self.get_logger().warn("有任务正在执行，忽略新的抓取请求")
            return
        
        # 从消息中提取信息
        self.object_type = msg.header.frame_id
        self.position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        # 设置任务状态
        self.task_running = True
        
        # 在新线程中执行抓取任务，避免阻塞ROS回调
        # task_thread = threading.Thread(target=self.execute_grasp_task, args=(object_type, position))
        # task_thread.daemon = True
        # task_thread.start()


@hydra.main(config_name='config', config_path=os.path.join(os.path.dirname(__file__), "cfg"))
def main(config: DictConfig):
    # 初始化ROS2
    rclpy.init()
    
    print("--- 启动ROS抓取控制器 ---")
    
    grasp_node = None
    grasp_node = GraspControllerNode(config)

    while rclpy.ok() :
        try:
            
            # 运行ROS节点
            while grasp_node.object_type is None or grasp_node.position is None:
                rclpy.spin_once(grasp_node)
            grasp_node.execute_grasp_task(grasp_node.object_type, grasp_node.position)

            grasp_node.get_logger().info("等待新的抓取目标...")
            grasp_node.object_type = None
            grasp_node.position = None
            
            
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"\n程序运行时发生错误: {e}")
            import traceback
            traceback.print_exc()

    grasp_node.get_logger().info("正在关闭抓取控制器节点...")
    if grasp_node is not None:
        grasp_node.destroy_node()   
    rclpy.shutdown()



if __name__ == '__main__':
    main()