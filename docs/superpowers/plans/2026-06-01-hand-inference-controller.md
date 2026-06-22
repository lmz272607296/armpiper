# Hand Inference Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ROS2 entry point that runs the ACT hand policy online and publishes short horizon hand joint trajectories.

**Architecture:** Extend `interface.run_inference` with ROS-specific helper functions and a `HandInferenceController` node that composes `LeapHand`. Keep model loading and existing offline CLI intact, and add `main_ros` as a separate console-script target.

**Tech Stack:** ROS2 `rclpy`, existing `LeapHand`, PyTorch ACT model, NumPy, setuptools console scripts, Python `unittest`/`pytest`.

---

### Task 1: Artifact Resolution Tests

**Files:**
- Modify: `src/robot_interface/interface/test/test_run_inference.py`
- Modify: `src/robot_interface/interface/interface/run_inference.py`

- [ ] **Step 1: Write the failing test**

Create `src/robot_interface/interface/test/test_run_inference.py` with tests that create a temporary `weights` directory containing `seed_expert_best.pth` and `normalizer_stats.npy`, then assert `resolve_artifact_paths(None, None, weights_dir)` returns those root files.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/robot_interface/interface/test/test_run_inference.py -q`
Expected: FAIL because root-level weight files are not detected.

- [ ] **Step 3: Implement minimal artifact resolution**

Update `resolve_latest_weight_dir` and `resolve_artifact_paths` so root-level files in `weights/` are valid defaults before falling back to run subdirectories.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest src/robot_interface/interface/test/test_run_inference.py -q`
Expected: PASS.

### Task 2: Short Horizon ROS Planner

**Files:**
- Modify: `src/robot_interface/interface/test/test_run_inference.py`
- Modify: `src/robot_interface/interface/interface/run_inference.py`

- [ ] **Step 1: Write the failing test**

Add a fake model with `act_min`, `act_max`, and `predict_denormalized`, then test a helper builds a `(horizon, 16)` trajectory from a 16-D current pose and clamps each per-step joint delta to `0.08` radians.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/robot_interface/interface/test/test_run_inference.py -q`
Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement helper**

Add `build_short_horizon_trajectory(model, current_pose, horizon_steps, ensemble_decay, lowpass_alpha, max_delta_rad)` that repeatedly predicts chunks, ensembles targets with `temporal_ensemble`, clamps deltas, and returns only publishable future points.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest src/robot_interface/interface/test/test_run_inference.py -q`
Expected: PASS.

### Task 3: ROS Node and Entry Point

**Files:**
- Modify: `src/robot_interface/interface/interface/run_inference.py`
- Modify: `src/robot_interface/interface/setup.py`
- Modify: `src/robot_interface/interface/package.xml`

- [ ] **Step 1: Implement ROS node**

Add fixed constants, `HandInferenceController`, and `main_ros`. The node loads CPU model artifacts, instantiates `LeapHand`, creates a 60 Hz timer, waits for a 16-D state, builds a 73-step short horizon, and calls `command_joint_position(trajectory, 1.0 / 60.0)`.

- [ ] **Step 2: Wire console script**

Add `hand_inference_controller=interface.run_inference:main_ros` to `setup.py` and add `python3-torch` as an exec dependency if available in the ROS environment packaging expectations.

- [ ] **Step 3: Verify syntax and tests**

Run: `python3 -m py_compile src/robot_interface/interface/interface/run_inference.py src/robot_interface/interface/setup.py`
Expected: no output.

Run: `python3 -m pytest src/robot_interface/interface/test/test_run_inference.py -q`
Expected: PASS.

### Self-Review

- Spec coverage: artifact loading, fixed parameters, short horizon publishing, entry point, and basic error handling are covered.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: helper and node use NumPy arrays in model order and call existing `LeapHand.command_joint_position` with a 2-D trajectory.
