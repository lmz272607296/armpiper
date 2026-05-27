# Position Player Fixed Chopstick Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed chopstick trajectory mode to `position_player` that restores the old fixed-hand grasp behavior with smooth startup and smooth open-then-close motion.

**Architecture:** Keep `PositionPlayer` as the runtime entrypoint, but move fixed-trajectory generation into a small pure helper module so it can be tested without ROS node setup. `PositionPlayer` will choose between recorded-hand playback and generated fixed-hand playback before applying bias and controller-order mapping.

**Tech Stack:** Python, numpy, ROS 2 `rclpy`, pytest

---

### Task 1: Add failing tests for fixed chopstick trajectory generation

**Files:**
- Create: `src/robot_interface/interface/test/test_position_player_trajectory.py`
- Test: `src/robot_interface/interface/test/test_position_player_trajectory.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from interface.position_player_trajectory import (
    DEFAULT_FIXED_HAND_RADIANS,
    build_fixed_chopstick_trajectory,
    validate_chopstick_angles,
)


def test_fixed_chopstick_trajectory_only_moves_hand1_and_hand5():
    trajectory = build_fixed_chopstick_trajectory(6, 30.0, 25.0)

    assert trajectory.shape == (6, 16)
    np.testing.assert_allclose(trajectory[0], DEFAULT_FIXED_HAND_RADIANS)
    np.testing.assert_allclose(trajectory[-1, 1], DEFAULT_FIXED_HAND_RADIANS[1] + np.deg2rad(25.0))
    np.testing.assert_allclose(trajectory[-1, 5], DEFAULT_FIXED_HAND_RADIANS[5] + np.deg2rad(25.0))
    assert np.min(trajectory[:, 1]) < DEFAULT_FIXED_HAND_RADIANS[1]
    assert np.min(trajectory[:, 5]) < DEFAULT_FIXED_HAND_RADIANS[5]

    static_joint_indices = [index for index in range(16) if index not in (1, 5)]
    np.testing.assert_allclose(
        trajectory[:, static_joint_indices],
        np.tile(DEFAULT_FIXED_HAND_RADIANS[static_joint_indices], (6, 1)),
    )


def test_validate_chopstick_angles_rejects_negative_values():
    validate_chopstick_angles(0.0, 0.0)

    try:
        validate_chopstick_angles(-1.0, 0.0)
    except ValueError as exc:
        assert 'chopstick_open_angle_deg' in str(exc)
    else:
        raise AssertionError('Expected negative open angle to raise ValueError')

    try:
        validate_chopstick_angles(0.0, -1.0)
    except ValueError as exc:
        assert 'chopstick_close_angle_deg' in str(exc)
    else:
        raise AssertionError('Expected negative close angle to raise ValueError')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/robot_interface/interface/test/test_position_player_trajectory.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for `interface.position_player_trajectory`

- [ ] **Step 3: Write minimal implementation**

```python
DEFAULT_FIXED_HAND_RADIANS = np.array([
    -1.036, 0.422, 2.356, 0.490,
    -0.055, 0.595, 1.161, 1.493,
    0.23635, -0.34515, 1.43804, 1.87391,
    0.744, -0.262, 1.3727, 0.842,
], dtype=np.float64)


def validate_chopstick_angles(open_angle_deg, close_angle_deg):
    ...


def build_fixed_chopstick_trajectory(sample_count, open_angle_deg, close_angle_deg):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/robot_interface/interface/test/test_position_player_trajectory.py -q`
Expected: PASS

### Task 2: Integrate fixed chopstick mode into `PositionPlayer`

**Files:**
- Modify: `src/robot_interface/interface/interface/position_player.py`
- Modify: `src/robot_interface/interface/interface/setup.py` (only if package data or module discovery needs adjustment)
- Test: `src/robot_interface/interface/test/test_position_player_trajectory.py`

- [ ] **Step 1: Write the failing integration test expectation**

Use the existing unit test file to add one more pure behavior test for bias application order:

```python
def test_fixed_chopstick_trajectory_accepts_bias_after_generation():
    trajectory = build_fixed_chopstick_trajectory(4, 10.0, 20.0)
    bias = np.zeros(16, dtype=np.float64)
    bias[2] = 0.25

    biased = trajectory.copy()
    biased[:, :16] += bias

    np.testing.assert_allclose(biased[:, 2], DEFAULT_FIXED_HAND_RADIANS[2] + 0.25)
```

- [ ] **Step 2: Run test to verify it fails if helper behavior is missing**

Run: `pytest src/robot_interface/interface/test/test_position_player_trajectory.py -q`
Expected: FAIL only if helper output shape or mutability assumptions are wrong

- [ ] **Step 3: Write minimal implementation in `position_player.py`**

Update `PositionPlayer` to:

```python
self.declare_parameter('use_fixed_chopstick_trajectory', False)
self.declare_parameter('chopstick_open_angle_deg', 30.0)
self.declare_parameter('chopstick_close_angle_deg', 25.0)

self.use_fixed_chopstick_trajectory = bool(...)
self.chopstick_open_angle_deg = float(...)
self.chopstick_close_angle_deg = float(...)
validate_chopstick_angles(self.chopstick_open_angle_deg, self.chopstick_close_angle_deg)
```

Replace trajectory preprocessing with:

```python
raw_trajectory = self.load_trajectory()
if self.use_fixed_chopstick_trajectory:
    raw_trajectory = build_fixed_chopstick_trajectory(
        raw_trajectory.shape[0],
        self.chopstick_open_angle_deg,
        self.chopstick_close_angle_deg,
    )

self.trajectory = self.map_real_hand_to_controller_order(
    self.apply_hand_position_bias(raw_trajectory)
)
```

Add startup logging for fixed mode and remove the old `hand8..hand15` override path.

- [ ] **Step 4: Run focused tests to verify they pass**

Run: `pytest src/robot_interface/interface/test/test_position_player_trajectory.py -q`
Expected: PASS

### Task 3: Verify package-level behavior and lint surface

**Files:**
- Modify: `src/robot_interface/interface/interface/position_player.py`
- Create: `src/robot_interface/interface/interface/position_player_trajectory.py`
- Test: `src/robot_interface/interface/test/test_position_player_trajectory.py`

- [ ] **Step 1: Run package-focused verification**

Run: `pytest src/robot_interface/interface/test/test_position_player_trajectory.py -q`
Expected: PASS

- [ ] **Step 2: Run style checks for touched Python files if dependencies are available**

Run: `pytest src/robot_interface/interface/test/test_flake8.py -q`
Expected: PASS, or skip/fail only if local ROS lint dependencies are unavailable

- [ ] **Step 3: Inspect git diff before completion**

Run: `git diff -- docs/superpowers/plans/2026-05-19-position-player-fixed-chopstick.md docs/superpowers/specs/2026-05-19-position-player-fixed-chopstick-design.md src/robot_interface/interface/interface/position_player.py src/robot_interface/interface/interface/position_player_trajectory.py src/robot_interface/interface/test/test_position_player_trajectory.py`
Expected: Diff only includes the planned fixed-chopstick design, helper, integration, and tests
