# Smooth Hand Inference Design

## Goal

Make the online hand inference controller slower and safer by moving to the recorded startup pose before inference and publishing smoother interpolated trajectories.

## Architecture

`interface.run_inference` remains the runtime entry point. It will import the initial-pose constants from `position_record.py`, compute the same startup pose, publish a smoothstep soft-start trajectory through `LeapHand.command_joint_position`, then begin inference after the soft-start duration elapses.

## Control Parameters

- `control_hz=10.0`, matching the recorder sample rate.
- `interpolation_factor=2`, matching recorder preview interpolation.
- `lowpass_alpha=0.35`.
- `max_delta_rad=0.025`.
- `soft_start_duration_sec=2.0`.
- `soft_start_rate_hz=20.0`.
- Publish interpolated command points at `1 / (control_hz * interpolation_factor)` seconds per point.

## Startup Behavior

The controller waits for a valid hand state. It computes `DEFAULT_HAND_INITIAL_POSE_DEG + DEFAULT_HAND_POSITION_BIAS_DEG`, converts with `deg2rad(value - 180)`, then publishes a smoothstep interpolation from the current pose to that target. Inference is disabled until the soft-start finish time.

## Inference Behavior

After startup, each timer cycle builds a short horizon trajectory using the lower max delta and low-pass gain. The short horizon trajectory is then densified with smooth interpolation between points before publishing. This lowers per-command jumps and gives the hardware controller smaller time-spaced targets.

## Testing

Add unit tests for startup-pose conversion and smooth interpolation. Existing tests continue covering root-level weight resolution and per-step clamp behavior.
