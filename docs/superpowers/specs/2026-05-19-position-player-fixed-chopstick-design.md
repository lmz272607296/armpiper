## Context

`src/robot_interface/interface/interface/position_player.py` was migrated from `~/colcon_ws/src/open_manipulator/leapsim/leapsim/position_player.py`, but the desired fixed chopstick behavior actually comes from `~/colcon_ws/src/open_manipulator/leapsim/leapsim/demo.py`.

The old fixed-hand behavior is:

- all 16 hand joints are generated from a fixed base pose rather than read from the recorded hand trajectory
- only `hand1` and `hand5` move during grasping
- `hand1` and `hand5` share the same grasp offset
- the grasp sequence opens first, then closes

Relevant old-source references:

- `demo.py:51-56` `DEFAULT_FIXED_HAND_RADIANS`
- `demo.py:60` `DEFAULT_GRASP_HAND_JOINTS = [1, 5]`
- `demo.py:61-63` open/close angle parameters
- `demo.py:653` `sample[joint_index] = self.base_sample[joint_index] + self.grasp_offset_radians`

## Goal

Keep the current `armpiper` `position_player` structure, but add an explicit fixed chopstick trajectory mode that restores the old grasp behavior in a controlled way.

The new mode must:

- cover all 16 hand joints using a generated fixed pose
- move only `hand1` and `hand5`
- open first, then close
- preserve the existing soft-start behavior so startup does not jump
- keep motion smooth both during startup and during the generated chopstick trajectory

## Parameters

Add three ROS parameters to `PositionPlayer`:

1. `use_fixed_chopstick_trajectory: bool`
- `False`: keep current recorded-hand playback behavior
- `True`: ignore recorded hand trajectory values and generate a fixed-hand chopstick trajectory

2. `chopstick_open_angle_deg: float`
- non-negative
- applied as a negative offset to `hand1` and `hand5`

3. `chopstick_close_angle_deg: float`
- non-negative
- applied as a positive offset to `hand1` and `hand5`

## Fixed Trajectory Behavior

When `use_fixed_chopstick_trajectory=True`:

1. Build a 16-joint base hand pose from the old `DEFAULT_FIXED_HAND_RADIANS` values from `demo.py`.
2. Ignore recorded hand joint samples from the `.npy` file for all 16 hand joints.
3. Generate a hand trajectory where:
- joints other than `hand1` and `hand5` stay fixed at the base pose
- `hand1` and `hand5` first move to `base - deg2rad(chopstick_open_angle_deg)`
- then move to `base + deg2rad(chopstick_close_angle_deg)`
4. Apply existing hand bias behavior after the fixed trajectory is generated so user calibration still works consistently.
5. Keep the existing controller-order mapping after trajectory generation.

When `use_fixed_chopstick_trajectory=False`:

- preserve current recorded-hand playback flow

## Smoothness Requirements

### Startup soft-start

Reuse the existing startup flow in `position_player.py`.

- before playback, read the current real hand joint state from the joint-state topic
- interpolate from the current real state to the first sample of the active trajectory
- do not jump directly into the fixed chopstick trajectory

This guarantees the fixed mode still enters smoothly from the robot's current pose.

### Fixed trajectory execution

The generated chopstick motion must also be smooth.

- generate multiple intermediate samples for the open phase
- generate multiple intermediate samples for the close phase
- preserve the existing timer-based playback path so the controller keeps receiving sequential points at `play_rate_hz`

The implementation should avoid introducing a second motion system or a second runtime state machine if precomputing the trajectory is sufficient.

## Data Flow

Recommended preprocessing order inside `PositionPlayer`:

1. load trajectory source data
2. if fixed chopstick mode is enabled, replace the hand trajectory with generated fixed-hand samples
3. apply `hand_position_bias`
4. map to controller joint order

The fixed chopstick mode should not affect the existing arm wrist alignment startup behavior.

## Validation

Parameter validation must enforce:

- `chopstick_open_angle_deg >= 0`
- `chopstick_close_angle_deg >= 0`

Runtime logging should clearly report:

- whether fixed chopstick trajectory mode is enabled
- the configured open/close angles
- that only `hand1` and `hand5` are moving while the rest of the hand stays at the fixed base pose

## Implementation Notes

- remove or bypass the current `hand8..hand15` fixed override, because it does not match the intended old behavior
- prefer minimal changes inside the existing `PositionPlayer` flow
- keep wrist soft-alignment logic untouched
- do not restore the old 23-joint hand+arm playback path

## Verification Plan

1. Fixed mode off:
- recorded hand playback still works as before

2. Fixed mode on:
- all 16 hand joints are generated from the fixed pose
- only `hand1` and `hand5` vary over time
- motion opens first, then closes

3. Startup:
- initial motion still interpolates from the live joint state to the first target sample
- no visible first-frame jump is introduced by the fixed mode

4. Existing wrist alignment:
- still works unchanged when enabled
