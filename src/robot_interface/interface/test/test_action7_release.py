from types import SimpleNamespace

import numpy as np

from interface import action7_flat


class FakeLogger:
    def info(self, message, *args, **kwargs):
        pass

    def error(self, message, *args, **kwargs):
        pass


class FakeArm:
    joints_num = 6


class FakeLeap:
    pass


def test_bottle_release_resets_middle_arm_joints_before_base_and_wrist(monkeypatch):
    node = action7_flat.ObjectReleaseNode.__new__(action7_flat.ObjectReleaseNode)
    node.release_move_steps = 12
    node.release_move_duration = 1.0
    node.release_move_extra_wait = 0.0
    node.release_settle_sec = 0.0
    node.post_hand_open_pause_sec = 0.0
    node.arm_reset_steps = 14
    node.arm_reset_duration = 1.0
    node.arm_reset_extra_wait = 0.0
    node.get_logger = lambda: FakeLogger()
    node.wait_between_phases = lambda *args, **kwargs: None
    node.open_hand_to_full_open = lambda *args, **kwargs: None
    node.hand_to_reset_pose = lambda *args, **kwargs: None
    node.default_release_height = lambda object_type: 0.14
    node.solve_release_pose = lambda object_type, current_arm, release_height: SimpleNamespace(
        success=True,
        joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.2], dtype=float),
        target_position=np.zeros(3),
        achieved_position=np.zeros(3),
        position_error=0.0,
    )
    node.ik_solver = SimpleNamespace(
        kinematics=SimpleNamespace(
            palm_pose=lambda joints: (np.zeros(3), np.eye(3))
        )
    )

    current_arm = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 1.2]], dtype=float)
    monkeypatch.setattr(action7_flat, 'refresh_current_joint_states', lambda **kwargs: (None, current_arm.copy()))

    commands = []

    def record_arm_motion(arm, start, target, steps, duration, extra_wait):
        commands.append(np.asarray(target, dtype=float).copy())

    monkeypatch.setattr(action7_flat, 'command_scaled_arm_motion', record_arm_motion)

    node.move_down_and_release(
        FakeLeap(),
        FakeArm(),
        SimpleNamespace(name=action7_flat.BOTTLE),
    )

    assert len(commands) == 3
    np.testing.assert_allclose(commands[1][0, 0], current_arm[0, 0])
    np.testing.assert_allclose(commands[1][0, 1:5], action7_flat.ARM_RELEASE_RESET_JOINTS[1:5])
    np.testing.assert_allclose(commands[1][0, 5], current_arm[0, 5])
    np.testing.assert_allclose(commands[2][0], action7_flat.ARM_RELEASE_RESET_JOINTS)
