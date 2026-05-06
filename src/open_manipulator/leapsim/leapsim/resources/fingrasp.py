
import time
from time import sleep
from attr import has
#import isaacgym
import xml.etree.ElementTree as ET
import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import to_absolute_path
import torch 
from leapsim.utils.reformat import omegaconf_to_dict, print_dict
# from leapsim.utils.utils import set_np_formatting, set_seed, get_current_commit_hash
from leapsim.utils.no_isaac_rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner, _override_sigma
from rl_games.algos_torch import model_builder
from leapsim.learning import no_isaac_amp_continuous
from leapsim.learning import amp_players
from leapsim.learning import amp_models
from leapsim.learning import amp_network_builder
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from collections import deque
import math
import random
import rclpy
from ament_index_python.packages import get_package_share_directory
import threading 


def _restore(player, args):
    """
    从检查点文件（checkpoint）中恢复模型和运行时的均值/标准差。
    """
    checkpoint_path = args.get('checkpoint')
    if checkpoint_path:
        expanded_path = os.path.expanduser(checkpoint_path)
        print(f"=> loading checkpoint '{expanded_path}'")
        # 关键！这里使用了 map_location
        checkpoint = torch.load(expanded_path, map_location=player.device,weights_only=False)
        player.model.load_state_dict(checkpoint['model'])
        if player.normalize_input and 'running_mean_std' in checkpoint:
            player.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        print(f"=> loaded checkpoint '{expanded_path}' successfully.")


class HardwarePlayer(object):
    def __init__(self, config):
        self.config = omegaconf_to_dict(config)
        self.set_defaults()
        self.action_scale = 1 / 24
        self.actions_num = 21

        self.device = 'cpu'

        self.debug_viz = self.config["task"]['env']['enableDebugVis']

        # hand setting
        self.init_pose = np.zeros(23)
        self.init_pose = self.fetch_grasp_state()
        self.get_dof_limits()


        if self.debug_viz:
            self.setup_plot()

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
        3.14, 1.5, 1.4, 0.5, 0.001, 1.047, 2.23, 1.885, 2.042, 1.047, 
        2.23, 1.885, 2.042, 1.047, 2.23, 1.885, 2.042, 2.094, 2.443, 1.9, 1.88]
        )
        self.leap_dof_lower =np.array([
        -3.14, -1.5, -1.5, -1.97, 0.0, -1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, 
        -0.366, -1.047, -0.314, -0.506, -0.366, -0.349, -0.47, -1.2, -1.34
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

    def deploy(self):
        import rclpy
        from rclpy.node import Node
        from hand_controller import LeapHand
        from arm_controller import LeapArm
        # try to set up rospy
        num_obs = self.config['task']['env']['numObservations'] 
        num_obs_single = num_obs // 3 
        rclpy.init()
        leap = LeapHand("rot")
        arm  = LeapArm("arm")

        leap.leap_dof_lower = self.leap_dof_lower
        leap.leap_dof_upper = self.leap_dof_upper
        leap.sim_to_real_indices = self.sim_to_real_indices
        leap.real_to_sim_indices = self.real_to_sim_indices

        hz=1/20
        test_mode=True
        # self.control_dt = 1 / hz
        # rate=rclpy.create_node('rate_control')
        # rate=rate.create_rate(hz)

        #数据说明
        # leap.raw_positions: 形状为 (1, 16)，顺序为real
        # leap.all_positions: 形状为 (1, 23)，顺序为real
        # commands: 形状为 (1, 21)，          顺序为sim
        # history_command: 形状为 (1, 7)，    顺序为real & sim
        # obses: 形状为(1, 21)，              顺序为sim

        #-------------------------------------------
        # 推理初始化准备
        if rclpy.ok and not test_mode :
            start_time = time.time()
            timeout = 5.0 # 5秒超时
            while leap.raw_positions is None and rclpy.ok():
                rclpy.spin_once(leap, timeout_sec=0.1)
                if time.time() - start_time > timeout:
                    leap.get_logger().error("在5秒内未接收到 /joint_states 消息，请检查发布器节点是否正常工作。")
                    return
                  
            init_hand = np.zeros((1, 16))
            init_arm  = np.zeros((1, 7))

            # 拼接初始位置
            init_hand = np.concatenate((leap.raw_positions, init_hand), axis=0)
            init_arm = np.concatenate ((leap.raw_arm_positions, init_arm), axis=0)
            # 发送初始位置
            leap.command_joint_position(init_hand, 1.5)
            arm.command_joint_position(init_arm, 2)
            time.sleep(5)

        # 误差回调
            rclpy.spin_once(leap, timeout_sec=0.1)
            arm.recover(init_arm, leap.raw_arm_positions, 1)
            time.sleep(1)

        #-------------------------------------------
        # 状态稳定，开始推理
            rclpy.spin_once(leap, timeout_sec=0.1)
            obses= self.order(leap.sim_all_positions)
            print("check\n")
            history_command= leap.raw_arm_positions


        else:
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
            
            if self.config["task"]["env"]["include_targets"]:
                obs_buf = np.concatenate([obs_buf, prev_target.copy()], axis=-1)

            if "phase_period" in self.config["task"]["env"]:
                phase = np.array([[0., 1.]], dtype=np.float32)
                obs_buf = np.concatenate([obs_buf, phase], axis=-1)

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
            if "disable_actions" not in self.config["task"]["env"]:
            
                send_commands = np.reshape(commands, (1, 21))
                send_commands = self.reorder(send_commands)

                send_commands = np.concatenate((history_command, send_commands[:, :7]), axis=0)
                if rclpy.ok() and not test_mode:
                    arm.command_joint_position(send_commands,1)
                    time.sleep(2)

            # 误差回调
                    rclpy.spin_once(leap, timeout_sec=0.1)
                    arm.recover(send_commands, leap.raw_arm_positions, 1)
                    time.sleep(2)
            # command_list.append(commands)
            # get o_{t+1}
            # 历史保存
                    rclpy.spin_once(leap, timeout_sec=0.1)
                    obses = self.order(leap.sim_all_positions)
                    history_command = leap.raw_arm_positions
                else:
                    obses = commands
                    history_command = send_commands[-1, :]
                    history_command = np.reshape(history_command, (1, 7))   

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

            if "phase_period" in self.config["task"]["env"]:
                omega = 2 * math.pi / self.config["task"]["env"]["phase_period"]
                phase_angle = (counter - 1) * omega / hz 
                num_envs = obs_buf.shape[0]
                phase = np.zeros((num_envs, 2), dtype=np.float32)
                phase[:, 0] = math.sin(phase_angle)
                phase[:, 1] = math.cos(phase_angle)
                obs_buf = np.concatenate([obs_buf, phase.copy()], axis=-1)

            if "obs_mask" in self.config["task"]["env"]:
                obs_buf = obs_buf * np.array(self.config["task"]["env"]["obs_mask"])[None, :]

            obs_buf = obs_buf.astype(np.float32)

    def forward_network(self, obs):

        obs_tensor = torch.from_numpy(obs).to(self.device)
        action_tensor = self.player.get_action(obs_tensor, is_deterministic=True)
        return action_tensor.cpu().numpy()
    
    def restore(self):
        rlg_config_dict = self.config['train']
        rlg_config_dict["params"]["config"]["env_info"] = {}
        self.num_obs = self.config["task"]["env"]["numObservations"]
        self.num_actions = 21
        observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        rlg_config_dict["params"]["config"]["env_info"]["observation_space"] = observation_space
        action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
        rlg_config_dict["params"]["config"]["env_info"]["action_space"] = action_space
        rlg_config_dict["params"]["config"]["env_info"]["agents"] = 1

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
            runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
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
            'checkpoint' : self.config['checkpoint'],
            'sigma' : None
        }

        self.player = runner.create_player()
        _restore(self.player, args)
        _override_sigma(self.player, args)

    def reorder(self,data):
        #从21维数据中提取出23维数据，数据顺序必须为sim
        rows, cols = data.shape

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
        rows, cols = data.shape

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

@hydra.main(config_name='config', config_path='cfg')
def main(config: DictConfig):
    agent = HardwarePlayer(config)
    agent.restore()
    agent.deploy()

if __name__ == '__main__':
    main()