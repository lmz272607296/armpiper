from types import SimpleNamespace

import numpy as np
import pytest

from interface import action_utils
from interface.action6_gsp import HandCurrentSafetyMonitor, MotorOvercurrentError
from interface.hand_controller import extract_currents_from_dynamic_joint_state


def make_dynamic_joint_state(entries):
    return SimpleNamespace(
        joint_names=[entry[0] for entry in entries],
        interface_values=[
            SimpleNamespace(interface_names=names, values=values)
            for _, names, values in entries
        ],
    )


def test_extract_currents_from_dynamic_joint_state_reads_supported_interfaces():
    msg = make_dynamic_joint_state([
        ("hand0", ["Velocity", "Present Current"], [0.0, 301.0]),
        ("hand1", ["Current", "Temperature"], [-299.0, 42.0]),
        ("joint1", ["Present Current"], [888.0]),
    ])

    currents = extract_currents_from_dynamic_joint_state(msg, joint_names={"hand0", "hand1"})

    assert currents == {"hand0": 301.0, "hand1": -299.0}


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warn(self, message, *args, **kwargs):
        self.warnings.append(str(message))

    def info(self, message, *args, **kwargs):
        self.infos.append(str(message))


class FakeLeap:
    def __init__(self, over_limit_sequence):
        self.over_limit_sequence = list(over_limit_sequence)
        self.hold_calls = 0

    def get_over_limit_currents(self, limit_ma, joint_names=None):
        if self.over_limit_sequence:
            return self.over_limit_sequence.pop(0)
        return {}

    def hold_current_position(self, duration=0.05):
        self.hold_calls += 1
        return True


class FakeArm:
    def __init__(self):
        self.hold_calls = 0

    def hold_current_position(self, duration=0.05):
        self.hold_calls += 1
        return True


def test_hand_current_safety_monitor_requires_two_consecutive_over_limit_reads():
    logger = FakeLogger()
    leap = FakeLeap([{"hand0": 320.0}, {}])
    arm = FakeArm()
    monitor = HandCurrentSafetyMonitor(
        leap=leap,
        arm=arm,
        logger=logger,
        limit_ma=350.0,
        confirm_delay_sec=0.0,
        poll_interval_sec=0.0,
    )

    monitor.check()

    assert leap.hold_calls == 0
    assert arm.hold_calls == 0


def test_hand_current_safety_monitor_stops_after_confirmed_over_limit():
    logger = FakeLogger()
    leap = FakeLeap([{"hand0": 320.0}, {"hand0": 318.0}])
    arm = FakeArm()
    monitor = HandCurrentSafetyMonitor(
        leap=leap,
        arm=arm,
        logger=logger,
        limit_ma=350.0,
        confirm_delay_sec=0.0,
        poll_interval_sec=0.0,
    )

    with pytest.raises(MotorOvercurrentError):
        monitor.check()

    assert leap.hold_calls == 1
    assert arm.hold_calls == 1


def test_execute_bottle_grasp_hand_uses_safety_check_during_waits(monkeypatch):
    class FakeHand:
        joints_num = 16

        def command_joint_position(self, desired_pose, speed):
            return True

    monkeypatch.setattr(action_utils, "refresh_current_hand", lambda leap: np.zeros((1, 16), dtype=float))

    def stop_now():
        raise RuntimeError("stop")

    def fake_wait(duration_sec, safety_check=None, sleep_step_sec=0.05):
        assert safety_check is stop_now
        safety_check()

    monkeypatch.setattr(action_utils, "_wait_with_safety", fake_wait)

    with pytest.raises(RuntimeError, match="stop"):
        action_utils.execute_bottle_grasp_hand(FakeHand(), safety_check=stop_now)
