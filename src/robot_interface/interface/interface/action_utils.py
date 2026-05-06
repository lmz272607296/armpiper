#NECESSARY UTILS


import os
import random
import time

import numpy as np
import rclpy
import torch
from gym import spaces
from rl_games.algos_torch import model_builder
from rl_games.torch_runner import Runner, _override_sigma

from interface.learning import amp_models, amp_network_builder, amp_players
from interface.learning import no_isaac_amp_continuous
from interface.utils.no_isaac_rlgames_utils import RLGPUAlgoObserver
from interface.utils.reformat import omegaconf_to_dict


def as_row(data, width=None):
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"Expected {width} columns, got shape {array.shape}")
    return array


def scale_action(origin, target, scale):
    if not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be a positive integer")

    origin = as_row(origin)
    target = as_row(target)
    if origin.shape[1] != target.shape[1]:
        raise ValueError(f"Shape mismatch: origin {origin.shape}, target {target.shape}")

    factors = np.arange(scale + 1, dtype=float).reshape(-1, 1)
    return origin + factors * ((target - origin) / scale)


def angle_transfer(data):
    return np.radians(np.asarray(data, dtype=float) - 180.0)


def wait_for_joint_state(leap=None, arm=None, need_hand=False, need_arm=False, timeout=5.0):
    start_time = time.time()
    while rclpy.ok():
        hand_ready = not need_hand or (leap is not None and leap.raw_positions is not None)
        arm_ready = not need_arm or (arm is not None and arm.raw_positions is not None)
        if hand_ready and arm_ready:
            return

        if leap is not None:
            rclpy.spin_once(leap, timeout_sec=0.05)
        if arm is not None:
            rclpy.spin_once(arm, timeout_sec=0.05)
        if time.time() - start_time > timeout:
            missing = []
            if need_hand and (leap is None or leap.raw_positions is None):
                missing.append("/hand_joint_states")
            if need_arm and (arm is None or arm.raw_positions is None):
                missing.append("/joint_states_single")
            raise TimeoutError(f"Timed out waiting for {', '.join(missing)}")


def smooth_arm_to_current_target(arm, target, scale, speed, extra_wait=0.0):
    target = as_row(target, 6)
    current = as_row(arm.raw_positions, 6)
    trajectory = scale_action(current, target, scale)
    arm.command_joint_position(trajectory, speed)
    time.sleep(scale * speed + extra_wait)
    return trajectory


def load_hand_eye_matrix(logger=None):
    matrix_file = "hand_eye_calibration.txt"
    if os.path.exists(matrix_file):
        matrix = np.loadtxt(matrix_file)
        if logger is not None:
            logger.info(f"Loaded hand-eye calibration: {matrix_file}")
        return matrix

    if logger is not None:
        logger.warn("Using default hand-eye calibration matrix.")
    return np.array([
        [0, 0, 1, 0.07720],
        [-1, 0, 0, -0.0165],
        [0, -1, 0, 0.09],
        [0.0, 0.0, 0.0, 1.0],
    ])


def camera_to_robot_transform(camera_coords, hand_eye_matrix):
    camera_coords = np.asarray(camera_coords, dtype=float)
    camera_homo = np.array([camera_coords[0], camera_coords[1], camera_coords[2], 1.0])
    robot_homo = hand_eye_matrix @ camera_homo
    robot_coords = robot_homo[:3].tolist()
    robot_coords[2] = 0
    return robot_coords


def load_checkpoint(player, checkpoint_path):
    expanded_path = os.path.expanduser(checkpoint_path)
    print(f"=> loading checkpoint '{expanded_path}'")
    checkpoint = torch.load(expanded_path, map_location=player.device, weights_only=False)
    player.model.load_state_dict(checkpoint["model"])

    if player.normalize_input and "running_mean_std" in checkpoint:
        player.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    print(f"=> loaded checkpoint '{expanded_path}' successfully.")


class GraspInferenceAgent:
    def __init__(self, config, checkpoint_path, dof_limit_type):
        self.config = omegaconf_to_dict(config)
        self._set_defaults()
        self.action_scale = 1 / 24
        self.actions_num = 21
        self.device = "cpu"
        self.checkpoint_to_load = checkpoint_path
        self.external_target_position = None
        self.init_pose = self._fetch_grasp_state()

        if dof_limit_type == "xie":
            self._get_dof_limits_xie()
        else:
            self._get_dof_limits_default()
        self.player = self._create_player()

    def _set_defaults(self):
        env = self.config["task"]["env"]
        env.setdefault("include_history", True)
        env.setdefault("include_targets", True)

    def _fetch_grasp_state(self, s=1.0):
        grasping_states = np.zeros((1, 21))
        if "sampled_pose_idx" in self.config["task"]["env"]:
            idx = self.config["task"]["env"]["sampled_pose_idx"]
        else:
            idx = random.randint(0, grasping_states.shape[0] - 1)
        return grasping_states[idx][:21]

    def _construct_sim_to_real_transformation(self):
        env = self.config["task"]["env"]
        self.sim_to_real_indices = env["sim_to_real_indices"]
        self.real_to_sim_indices = env["real_to_sim_indices"]

    def _real_to_sim(self, values):
        if not hasattr(self, "real_to_sim_indices"):
            self._construct_sim_to_real_transformation()
        return values[:, self.real_to_sim_indices]

    def _set_limits(self, lower, upper):
        self.leap_dof_lower = self._real_to_sim(lower.reshape(1, 21)).squeeze()
        self.leap_dof_upper = self._real_to_sim(upper.reshape(1, 21)).squeeze()

    def _get_dof_limits_default(self):
        lower = np.array([
            -0.01, -0.01, -0.01, -0.01,
            -0.01, -0.01, -0.01, -0.01,
            -0.01, -0.01, -0.01, -0.01,
            -0.01, -0.01, -0.01, -0.01,
            -3.14, -1.5, -1.5, -1.97, 0.0,
        ])
        upper = np.array([
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            3.14, 1.5, 1.4, 0.5, 0.001,
        ])
        self._set_limits(lower, upper)

    def _get_dof_limits_xie(self):
        lower = np.array([
            -0.01, -0.01, -0.01, -0.01,
            -0.01, -0.01, -0.01, -0.01,
            -0.01, -0.01, -0.01, -0.01,
            -0.01, -0.01, -0.01, -0.01,
            -3.14, -1.5, -1.5, -1.97, 1.5,
        ])
        upper = np.array([
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            3.14, 1.5, 1.4, 0.5, 1.5701,
        ])
        self._set_limits(lower, upper)

    def _create_player(self):
        rlg_config_dict = self.config["train"]
        if "params" in rlg_config_dict and "config" in rlg_config_dict["params"]:
            rlg_config_dict["params"]["config"]["device"] = "cpu"
            rlg_config_dict["params"]["config"]["device_name"] = "cpu"
            rlg_config_dict["params"]["config"]["env_info"] = {}

        num_obs = self.config["task"]["env"]["numObservations"]
        observation_space = spaces.Box(np.ones(num_obs) * -np.inf, np.ones(num_obs) * np.inf)
        action_space = spaces.Box(np.ones(self.actions_num) * -1.0, np.ones(self.actions_num) * 1.0)
        rlg_config_dict["params"]["config"]["env_info"]["observation_space"] = observation_space
        rlg_config_dict["params"]["config"]["env_info"]["action_space"] = action_space
        rlg_config_dict["params"]["config"]["env_info"]["agents"] = 1

        runner = Runner(RLGPUAlgoObserver())
        runner.algo_factory.register_builder(
            "amp_continuous",
            lambda **kwargs: no_isaac_amp_continuous.AMPAgent(**kwargs),
        )
        runner.player_factory.register_builder(
            "amp_continuous",
            lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs),
        )
        model_builder.register_model(
            "continuous_amp",
            lambda network, **kwargs: amp_models.ModelAMPContinuous(network),
        )
        model_builder.register_network("amp", lambda **kwargs: amp_network_builder.AMPBuilder())

        runner.load(rlg_config_dict)
        runner.reset()
        player = runner.create_player()

        args = {"train": False, "play": True, "checkpoint": self.checkpoint_to_load, "sigma": None}
        load_checkpoint(player, args["checkpoint"])
        _override_sigma(player, args)
        return player

    def _forward_network(self, obs):
        obs_tensor = torch.from_numpy(obs).to(self.device)
        action_tensor = self.player.get_action(obs_tensor, is_deterministic=True)
        return action_tensor.cpu().numpy()

    def infer(self, external_target_position):
        self.external_target_position = np.asarray(external_target_position, dtype=np.float32).reshape(1, 3)

        num_obs = self.config["task"]["env"]["numObservations"]
        num_obs_single = num_obs // 3
        obs = np.zeros(21, dtype=np.float32)
        prev_target = obs[None].copy()

        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)

        cur_obs_buf = unscale(obs, self.leap_dof_lower, self.leap_dof_upper)[None]
        obs_buf = np.zeros((1, 0), dtype=np.float32)
        env = self.config["task"]["env"]
        num_append_iters = 3 if env["include_history"] else 1

        for _ in range(num_append_iters):
            obs_buf = np.concatenate([obs_buf, cur_obs_buf.copy()], axis=-1)
            if env["include_targets"]:
                obs_buf = np.concatenate([obs_buf, prev_target.copy()], axis=-1)
            if env["include_obj_pose"]:
                obs_buf = np.concatenate([obs_buf, self.external_target_position], axis=-1)

        if "obs_mask" in env:
            obs_buf = obs_buf * np.array(env["obs_mask"])[None, :]
        obs_buf = obs_buf.astype(np.float32)

        if self.player.is_rnn:
            self.player.init_rnn()

        start_time = time.time()
        counter = 0
        while True:
            counter += 1
            action = self._forward_network(obs_buf)
            action = np.clip(action, -1.0, 1.0)

            if "actions_mask" in env:
                action = action * np.array(env["actions_mask"])[None, :]

            target = prev_target + self.action_scale * action
            target = np.clip(target, self.leap_dof_lower, self.leap_dof_upper)
            prev_target = target.copy()
            commands = target[0]

            if counter > 600:
                print("Inference time: {:.4f} seconds".format(time.time() - start_time))
                return commands[:5].copy()

            obs = commands.astype(np.float32)
            cur_obs_buf = unscale(obs, self.leap_dof_lower, self.leap_dof_upper)[None]

            if env["include_history"]:
                obs_buf = obs_buf[:, num_obs_single:].copy()
            else:
                obs_buf = np.zeros((1, 0), dtype=np.float32)

            obs_buf = np.concatenate([obs_buf, cur_obs_buf.copy()], axis=-1)
            if env["include_targets"]:
                obs_buf = np.concatenate([obs_buf, target.copy()], axis=-1)
            if env["include_obj_pose"]:
                obs_buf = np.concatenate([obs_buf, self.external_target_position], axis=-1)
            if "obs_mask" in env:
                obs_buf = obs_buf * np.array(env["obs_mask"])[None, :]
            obs_buf = obs_buf.astype(np.float32)

    def reorder(self, data):
        data = as_row(data)
        if data.shape[1] < 5:
            raise ValueError(f"reorder needs at least 5 columns, got shape {data.shape}")
        fixed_joint = np.zeros((data.shape[0], 1), dtype=data.dtype)
        return np.concatenate((data[:, :3], fixed_joint, data[:, 3:]), axis=1)


BOTTLE_PREGRASP = np.array([
    209, 180, 190, 253,
    260, 279, 179, 236,
    209, 180, 190, 253,
    209, 180, 190, 253,
])

BOTTLE_GRASP = np.array([
    272, 180, 190, 253,
    260, 279, 179, 236,
    272, 180, 190, 253,
    272, 180, 190, 253,
])

FRUIT_PREGRASP = np.array([
    251, 180, 185, 232,
    219, 256, 190, 235,
    251, 180, 185, 232,
    260, 215, 200, 172,
])

FRUIT_GRASP = np.array([
    260, 180, 175, 222,
    219, 256, 255, 235,
    260, 180, 175, 222,
    260, 142, 200, 172,
])


def execute_bottle_grasp_hand(leap):
    wait_for_joint_state(leap=leap, need_hand=True)
    lower = angle_transfer(BOTTLE_PREGRASP).reshape(1, 16)
    current_hand = as_row(leap.raw_positions, 16)
    pregrasp_action = scale_action(current_hand, lower, 5)

    leap.command_joint_position(pregrasp_action, 0.4)
    time.sleep(5 * 0.4 + 1.0)

    upper = angle_transfer(BOTTLE_GRASP).reshape(1, 16)
    grasp_action = scale_action(lower, upper, 50)
    for i in range(50):
        leap.command_joint_position(grasp_action[i:i + 2, :], 0.1)
        time.sleep(0.1)


def execute_fruit_grasp_hand(leap):
    wait_for_joint_state(leap=leap, need_hand=True)
    lower = angle_transfer(FRUIT_PREGRASP).reshape(1, 16)
    current_hand = as_row(leap.raw_positions, 16)
    pregrasp_action = scale_action(current_hand, lower, 5)

    leap.command_joint_position(pregrasp_action, 0.4)
    time.sleep(5 * 0.4 + 1.0)

    upper = angle_transfer(FRUIT_GRASP).reshape(1, 16)
    grasp_action = scale_action(lower, upper, 19)
    for i in range(19):
        leap.command_joint_position(grasp_action[i:i + 2, :], 0.3)
        time.sleep(0.4)


def execute_bottle_release(leap, arm):
    wait_for_joint_state(leap=leap, arm=arm, need_hand=True, need_arm=True)
    arm_zero = np.zeros((1, 6), dtype=float)
    smooth_arm_to_current_target(arm, arm_zero, scale=10, speed=0.5, extra_wait=1.0)

    wait_for_joint_state(leap=leap, need_hand=True)
    hand_zero = np.zeros((1, 16), dtype=float)
    hand_loose = scale_action(as_row(leap.raw_positions, 16), hand_zero, 1)
    leap.command_joint_position(hand_loose, 3.0)
    time.sleep(4.0)

    wait_for_joint_state(arm=arm, need_arm=True)
    smooth_arm_to_current_target(arm, arm_zero, scale=10, speed=0.2, extra_wait=2.0)


def execute_fruit_release(leap, arm):
    wait_for_joint_state(leap=leap, arm=arm, need_hand=True, need_arm=True)
    arm_zero = np.zeros((1, 6), dtype=float)
    smooth_arm_to_current_target(arm, arm_zero, scale=4, speed=0.5, extra_wait=3.0)

    wait_for_joint_state(leap=leap, need_hand=True)
    hand_zero = np.zeros((1, 16), dtype=float)
    hand_loose = scale_action(as_row(leap.raw_positions, 16), hand_zero, 1)
    leap.command_joint_position(hand_loose, 3.0)
    time.sleep(3.0)


execute_cube_grasp_hand = execute_fruit_grasp_hand
execute_cube_release = execute_fruit_release
