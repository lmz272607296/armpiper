# Smooth Hand Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Slow and smooth the hand inference controller by adding recorder-style startup pose movement, lower-rate inference, interpolation, and gentler filtering.

**Architecture:** Keep changes inside `interface.run_inference` and reuse constants from `interface.position_record`. Add pure helper functions for startup-pose conversion and trajectory interpolation so tests can verify behavior without ROS.

**Tech Stack:** Python, NumPy, PyTorch, ROS2 `rclpy`, existing `LeapHand`, pytest/unittest.

---

### Task 1: Startup Pose Helper

**Files:**
- Modify: `src/robot_interface/interface/test/test_run_inference.py`
- Modify: `src/robot_interface/interface/interface/run_inference.py`

- [ ] **Step 1: Write failing test**

Add a test that imports `DEFAULT_HAND_INITIAL_POSE_DEG`, `DEFAULT_HAND_POSITION_BIAS_DEG`, and `INITIAL_POSITION_OFFSET_DEG` from `interface.position_record`, then asserts `run_inference.build_startup_pose()` returns `np.deg2rad(initial + bias - 180)`.

- [ ] **Step 2: Verify red**

Run: `python3 -m pytest src/robot_interface/interface/test/test_run_inference.py -q`
Expected: FAIL because `build_startup_pose` is missing.

- [ ] **Step 3: Implement helper**

Add imports from `interface.position_record` and implement `build_startup_pose()`.

- [ ] **Step 4: Verify green**

Run: `python3 -m pytest src/robot_interface/interface/test/test_run_inference.py -q`
Expected: PASS.

### Task 2: Smooth Interpolation Helper

**Files:**
- Modify: `src/robot_interface/interface/test/test_run_inference.py`
- Modify: `src/robot_interface/interface/interface/run_inference.py`

- [ ] **Step 1: Write failing test**

Add a test that passes two 16-D points and `interpolation_factor=2`, then asserts the output has 3 points: start, smoothstep midpoint, end.

- [ ] **Step 2: Verify red**

Run: `python3 -m pytest src/robot_interface/interface/test/test_run_inference.py -q`
Expected: FAIL because `interpolate_trajectory` is missing.

- [ ] **Step 3: Implement helper**

Add `interpolate_trajectory(trajectory, interpolation_factor)` using the same smoothstep alpha formula used in `position_record.py` and `position_player.py`.

- [ ] **Step 4: Verify green**

Run: `python3 -m pytest src/robot_interface/interface/test/test_run_inference.py -q`
Expected: PASS.

### Task 3: Runtime Smoothing and Soft Start

**Files:**
- Modify: `src/robot_interface/interface/interface/run_inference.py`

- [ ] **Step 1: Update constants**

Set `DEFAULT_CONTROL_HZ=10.0`, `DEFAULT_MAX_DELTA_RAD=0.025`, `DEFAULT_LOWPASS_ALPHA=0.35`, and add soft-start/interpolation constants matching `position_record.py`.

- [ ] **Step 2: Update node state machine**

Change `HandInferenceController.publish_inference_trajectory` to wait for hand state, publish soft-start once, wait until finish time, then publish inference trajectories.

- [ ] **Step 3: Publish interpolated inference trajectories**

Interpolate the short horizon before calling `command_joint_position`, using `seconds_per_point = 1 / (control_hz * interpolation_factor)`.

- [ ] **Step 4: Verify**

Run: `python3 -m pytest src/robot_interface/interface/test/test_run_inference.py -q`
Expected: PASS.

Run: `python3 -m py_compile src/robot_interface/interface/interface/run_inference.py src/robot_interface/interface/setup.py`
Expected: no output.

### Self-Review

- Spec coverage: startup pose, soft-start, lower frequency, interpolation, lower max delta, and low-pass smoothing are covered.
- Placeholder scan: no placeholders remain.
- Type consistency: helpers accept and return NumPy arrays, while ROS publishing remains delegated to `LeapHand.command_joint_position`.
