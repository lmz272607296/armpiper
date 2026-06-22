# Hand Inference Controller Design

## Goal

Add a ROS2 runtime path for the ACT hand policy so it can read live LEAP hand joint positions and publish inferred joint targets through the existing hand controller topics.

## Architecture

The existing `interface.run_inference` module remains the home of the ACT model and offline CLI. It will gain a ROS2 node entry point that composes `LeapHand` instead of duplicating topic mapping and `JointTrajectory` publishing logic. The node loads `weights/seed_expert_best.pth` and `weights/normalizer_stats.npy`, waits for `LeapHand.raw_positions`, runs inference at 60 Hz, and sends a short horizon trajectory by calling `LeapHand.command_joint_position`.

## Data Flow

1. `LeapHand` subscribes to `/hand_joint_states` and maps incoming joint names to the simulation/model order.
2. The inference node reads the latest 16-D `LeapHand.raw_positions` vector.
3. The ACT policy predicts a future action chunk from the current observation.
4. The node applies temporal ensembling with decay `0.7`, low-pass gain `1.0`, and per-joint delta clamp `0.08` rad.
5. The node publishes a short horizon trajectory through `/hand_controller/joint_trajectory` using `LeapHand.command_joint_position` with `1/60` seconds per point.

## Fixed Parameters

- `control_hz=60.0`
- `rollout_steps=73`
- `ensemble_decay=0.7`
- `lowpass_alpha=1.0`
- `max_delta_rad=0.08`
- `device=cpu`
- `horizon_steps=73`, capped by the model chunk size

## Entry Point

Add `hand_inference_controller=interface.run_inference:main_ros` to the package console scripts. The operator command is `ros2 run interface hand_inference_controller`.

## Error Handling

Startup fails clearly if the checkpoint, normalizer stats, or model dimensions are invalid. Runtime cycles skip publishing until a fresh hand state arrives. If publishing fails, the node logs the failure and keeps running so a later valid state can recover.

## Testing

Add unit coverage for weight artifact resolution with files directly under `weights/` and for horizon generation shape/clamping behavior. Run the package tests plus a Python syntax compile check for the edited files.
