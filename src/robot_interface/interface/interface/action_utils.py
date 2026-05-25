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

from interface.motion_utils import (
    as_row,
    refresh_current_arm,
    refresh_current_hand,
    scale_action,
)
from interface.hand_reset_utils import command_delayed_thumb_reset
from interface.learning import amp_models, amp_network_builder, amp_players
from interface.learning import no_isaac_amp_continuous
from interface.utils.no_isaac_rlgames_utils import RLGPUAlgoObserver
from interface.utils.reformat import omegaconf_to_dict


ARM_SPEEDUP_FACTOR = 2.5
RELEASE_HAND_SPEEDUP_FACTOR = 2.0
BOTTLE_GRASP_HAND_SPEEDUP_FACTOR = 2.0
FRUIT_GRASP_HAND_SPEEDUP_FACTOR = 1.5
STAGE_DELAY_SEC = 0.2


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
    current = refresh_current_arm(arm)
    trajectory = scale_action(current, target, scale)
    if not arm.command_joint_position(trajectory, speed):
        raise RuntimeError("Failed to publish arm trajectory")
    time.sleep(scale * speed + extra_wait)
    return trajectory


def _wait_with_safety(duration_sec, safety_check=None, sleep_step_sec=0.05):
    deadline = time.time() + max(0.0, float(duration_sec))
    step = max(0.01, float(sleep_step_sec))
    while True:
        if safety_check is not None:
            safety_check()
        remaining = deadline - time.time()
        if remaining <= 0.0:
            return
        time.sleep(min(step, remaining))


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
    def __init__(self, config, checkpoint_path, dof_limit_type, max_inference_steps=600):
        self.config = omegaconf_to_dict(config)
        self.action_scale = 1 / 24
        self.actions_num = 21
        self.device = "cpu"
        self.checkpoint_to_load = checkpoint_path
        self.external_target_position = None
        self.max_inference_steps = max(1, int(max_inference_steps))
        self.include_phase_feature = False
        self.include_obj_scale_feature = False
        self.expected_num_obs = None
        self._set_defaults()
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
        env.setdefault("include_obj_pose", False)
        self.expected_num_obs = self._infer_num_observations_from_checkpoint()
        configured_num_obs = int(env.get("numObservations", 0) or 0)
        if self.expected_num_obs is not None and configured_num_obs != self.expected_num_obs:
            print(
                f"Override numObservations from {configured_num_obs} to {self.expected_num_obs} "
                f"based on checkpoint '{os.path.expanduser(self.checkpoint_to_load)}'"
            )
            env["numObservations"] = self.expected_num_obs
        elif configured_num_obs > 0:
            self.expected_num_obs = configured_num_obs

        self._configure_optional_observation_features()

    def _infer_num_observations_from_checkpoint(self):
        expanded_path = os.path.expanduser(self.checkpoint_to_load)
        checkpoint = torch.load(expanded_path, map_location="cpu", weights_only=False)

        running_stats = checkpoint.get("running_mean_std")
        if running_stats and "running_mean" in running_stats:
            running_mean = running_stats["running_mean"]
            if hasattr(running_mean, "shape") and running_mean.ndim == 1:
                return int(running_mean.shape[0])

        model_state = checkpoint.get("model", {})
        rnn_weight = model_state.get("a2c_network.rnn.rnn.weight_ih_l0")
        if hasattr(rnn_weight, "shape") and len(rnn_weight.shape) == 2:
            return int(rnn_weight.shape[1])

        return None

    def _configure_optional_observation_features(self):
        env = self.config["task"]["env"]
        append_iters = 3 if env["include_history"] else 1
        per_frame_dim = 21
        if env["include_targets"]:
            per_frame_dim += 21
        if env["include_obj_pose"]:
            per_frame_dim += 3

        expected_num_obs = int(self.expected_num_obs or env["numObservations"])
        remaining_dim = expected_num_obs - append_iters * per_frame_dim

        if remaining_dim >= append_iters * 2 and "phase_period" in env:
            self.include_phase_feature = True
            remaining_dim -= append_iters * 2

        if remaining_dim >= append_iters:
            self.include_obj_scale_feature = True
            remaining_dim -= append_iters

        if remaining_dim != 0:
            print(
                f"Observation dim mismatch remains after optional features: {remaining_dim}. "
                "Inference inputs will be zero-padded or trimmed as needed."
            )

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

    def _phase_features(self, counter):
        env = self.config["task"]["env"]
        if not self.include_phase_feature:
            return None

        if counter is None:
            return np.array([[0.0, 1.0]], dtype=np.float32)

        phase_period = float(env["phase_period"])
        omega = 2 * np.pi / phase_period
        phase_angle = (counter - 1) * omega / 20.0
        return np.array([[np.sin(phase_angle), np.cos(phase_angle)]], dtype=np.float32)

    def _object_scale_feature(self):
        if not self.include_obj_scale_feature:
            return None
        base_obj_scale = float(self.config["task"]["env"].get("baseObjScale", 1.0))
        return np.array([[base_obj_scale]], dtype=np.float32)

    def _append_observation_frame(self, obs_buf, current_obs, target, counter=None):
        env = self.config["task"]["env"]
        obs_buf = np.concatenate([obs_buf, current_obs.copy()], axis=-1)
        if env["include_targets"]:
            obs_buf = np.concatenate([obs_buf, target.copy()], axis=-1)
        if env["include_obj_pose"]:
            obs_buf = np.concatenate([obs_buf, self.external_target_position], axis=-1)

        phase = self._phase_features(counter)
        if phase is not None:
            obs_buf = np.concatenate([obs_buf, phase], axis=-1)

        obj_scale = self._object_scale_feature()
        if obj_scale is not None:
            obs_buf = np.concatenate([obs_buf, obj_scale], axis=-1)

        return obs_buf

    def _align_observation_size(self, obs_buf):
        expected_num_obs = int(self.expected_num_obs or obs_buf.shape[1])
        current_dim = obs_buf.shape[1]
        if current_dim == expected_num_obs:
            return obs_buf
        if current_dim < expected_num_obs:
            padding = np.zeros((obs_buf.shape[0], expected_num_obs - current_dim), dtype=np.float32)
            return np.concatenate([obs_buf, padding], axis=-1)
        return obs_buf[:, :expected_num_obs]

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
            obs_buf = self._append_observation_frame(obs_buf, cur_obs_buf, prev_target)

        if "obs_mask" in env:
            obs_buf = obs_buf * np.array(env["obs_mask"])[None, :]
        obs_buf = self._align_observation_size(obs_buf).astype(np.float32)

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

            if counter == 1 or counter % 100 == 0:
                print(
                    f"Inference step {counter}: target[:5]={np.round(commands[:5], 6).tolist()} "
                    f"obj={np.round(self.external_target_position.reshape(3), 6).tolist()}"
                )

            if counter >= self.max_inference_steps:
                print("Inference time: {:.4f} seconds".format(time.time() - start_time))
                return commands[:5].copy()

            obs = commands.astype(np.float32)
            cur_obs_buf = unscale(obs, self.leap_dof_lower, self.leap_dof_upper)[None]

            if env["include_history"]:
                obs_buf = obs_buf[:, num_obs_single:].copy()
            else:
                obs_buf = np.zeros((1, 0), dtype=np.float32)

            obs_buf = self._append_observation_frame(obs_buf, cur_obs_buf, target, counter=counter)
            if "obs_mask" in env:
                obs_buf = obs_buf * np.array(env["obs_mask"])[None, :]
            obs_buf = self._align_observation_size(obs_buf).astype(np.float32)

    def reorder(self, data):
        data = as_row(data)
        if data.shape[1] < 5:
            raise ValueError(f"reorder needs at least 5 columns, got shape {data.shape}")
        fixed_joint = np.zeros((data.shape[0], 1), dtype=data.dtype)
        return np.concatenate((data[:, :3], fixed_joint, data[:, 3:]), axis=1)


BOTTLE_PREGRASP = np.array([
    209, 180, 190, 220,
    260, 279, 179, 236,
    209, 180, 190, 220,
    209, 180, 190, 220,
])

BOTTLE_GRASP = np.array([
    272, 180, 190, 253,
    260, 279, 200, 236,
    287, 180, 190, 253,
    287, 180, 190, 253,
])

FRUIT_PREGRASP = np.array([
    240, 180, 175, 235,
    219, 256, 190, 235,
    240, 180, 175, 235,
    250, 215, 210, 180,
])

FRUIT_GRASP = np.array([
    240, 180, 175, 235,
    219, 256, 255, 235,
    240, 180, 175, 235,
    260, 142, 220, 180,
])

def execute_bottle_grasp_hand(leap, safety_check=None):
    lower = angle_transfer(BOTTLE_PREGRASP).reshape(1, 16)
    current_hand = refresh_current_hand(leap)
    pregrasp_action = scale_action(current_hand, lower, 5)
    pregrasp_point_duration = 0.4 / BOTTLE_GRASP_HAND_SPEEDUP_FACTOR

    if not leap.command_joint_position(pregrasp_action, pregrasp_point_duration):
        raise RuntimeError("Failed to publish bottle pregrasp hand trajectory")
    _wait_with_safety(5 * pregrasp_point_duration, safety_check=safety_check)

    upper = angle_transfer(BOTTLE_GRASP).reshape(1, 16)
    grasp_action = scale_action(lower, upper, 50)
    grasp_point_duration = 0.1 / BOTTLE_GRASP_HAND_SPEEDUP_FACTOR
    for i in range(50):
        if safety_check is not None:
            safety_check()
        leap.command_joint_position(grasp_action[i:i + 2, :], grasp_point_duration)
        _wait_with_safety(grasp_point_duration, safety_check=safety_check)


def execute_fruit_grasp_hand(leap, safety_check=None):
    lower = angle_transfer(FRUIT_PREGRASP).reshape(1, 16)
    current_hand = refresh_current_hand(leap)
    pregrasp_action = scale_action(current_hand, lower, 5)
    pregrasp_point_duration = 0.4 / FRUIT_GRASP_HAND_SPEEDUP_FACTOR

    if not leap.command_joint_position(pregrasp_action, pregrasp_point_duration):
        raise RuntimeError("Failed to publish fruit pregrasp hand trajectory")
    _wait_with_safety(5 * pregrasp_point_duration, safety_check=safety_check)

    upper = angle_transfer(FRUIT_GRASP).reshape(1, 16)
    grasp_action = scale_action(lower, upper, 19)
    grasp_point_duration = 0.3 / FRUIT_GRASP_HAND_SPEEDUP_FACTOR
    for i in range(19):
        if safety_check is not None:
            safety_check()
        leap.command_joint_position(grasp_action[i:i + 2, :], grasp_point_duration)
        _wait_with_safety(grasp_point_duration, safety_check=safety_check)


def execute_bottle_release(leap, arm):
    arm_zero = np.zeros((1, 6), dtype=float)
    smooth_arm_to_current_target(
        arm,
        arm_zero,
        scale=10,
        speed=0.5 / ARM_SPEEDUP_FACTOR,
        extra_wait=STAGE_DELAY_SEC,
    )

    hand_zero = np.zeros((1, 16), dtype=float)
    command_delayed_thumb_reset(
        leap,
        refresh_current_hand(leap),
        hand_zero,
        point_duration=3.0 / RELEASE_HAND_SPEEDUP_FACTOR,
        trajectory_points=1,
        extra_wait=STAGE_DELAY_SEC,
    )

    smooth_arm_to_current_target(
        arm,
        arm_zero,
        scale=10,
        speed=0.2 / ARM_SPEEDUP_FACTOR,
        extra_wait=STAGE_DELAY_SEC,
    )


def execute_fruit_release(leap, arm):
    arm_zero = np.zeros((1, 6), dtype=float)
    smooth_arm_to_current_target(
        arm,
        arm_zero,
        scale=4,
        speed=0.5 / ARM_SPEEDUP_FACTOR,
        extra_wait=STAGE_DELAY_SEC,
    )

    hand_zero = np.zeros((1, 16), dtype=float)
    command_delayed_thumb_reset(
        leap,
        refresh_current_hand(leap),
        hand_zero,
        point_duration=3.0 / RELEASE_HAND_SPEEDUP_FACTOR,
        trajectory_points=1,
        extra_wait=STAGE_DELAY_SEC,
    )


execute_cube_grasp_hand = execute_fruit_grasp_hand
execute_cube_release = execute_fruit_release
