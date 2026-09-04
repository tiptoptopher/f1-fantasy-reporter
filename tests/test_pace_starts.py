"""Launch-performance detection, against synthetic telemetry with a known answer.

Matches energy.py's test-construction style: rows built directly from
(time, speed, throttle) triples so the exact answer is known in advance.
"""

from __future__ import annotations

import pandas as pd
import pytest

from f1_fantasy.pace import starts as starts_module
from f1_fantasy.pace.starts import detect_anti_stall, launch_time_to_speed


def _telemetry(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta([r[0] for r in rows], unit="s"),
            "Speed": [r[1] for r in rows],
            "Throttle": [r[2] for r in rows],
        }
    )


def test_launch_time_to_speed_finds_the_first_sample_reaching_the_target():
    tel = _telemetry([(0.0, 0.0, 100.0), (1.0, 60.0, 100.0), (2.0, 100.0, 100.0), (3.0, 150.0, 100.0)])

    assert launch_time_to_speed(tel, target_speed=100.0) == pytest.approx(2.0)


def test_launch_time_to_speed_is_none_if_target_never_reached():
    tel = _telemetry([(0.0, 0.0, 100.0), (1.0, 40.0, 100.0), (2.0, 80.0, 100.0)])

    assert launch_time_to_speed(tel, target_speed=100.0) is None


def test_detect_anti_stall_flags_a_clean_car_as_false():
    # A normal launch: speed climbs immediately under full throttle.
    tel = _telemetry([(0.0, 0.0, 100.0), (0.3, 20.0, 100.0), (0.6, 45.0, 100.0), (1.0, 70.0, 100.0)])

    assert detect_anti_stall(tel) is False


def test_detect_anti_stall_flags_a_dip_well_beyond_any_real_2026_launch():
    # Driver is on the throttle but the car barely moves until well after
    # the window closes -- the escape sample (the first non-stalled point)
    # lands at t=3.0, so the stalled run's measured span is 3.0s, clearing
    # STALL_DURATION_THRESHOLD (2.5s, set just above the real observed max
    # of 2.09s across 257 2026 driver-races).
    tel = _telemetry([(0.0, 0.0, 80.0), (0.5, 1.0, 80.0), (1.0, 2.0, 80.0), (1.9, 4.0, 80.0), (3.0, 40.0, 80.0)])

    assert detect_anti_stall(tel) is True


def test_detect_anti_stall_ignores_the_pre_launch_moment_with_no_throttle():
    # Stationary with throttle near zero -- this is just "hasn't launched
    # yet", not a stall (the driver isn't trying to accelerate).
    tel = _telemetry([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 60.0, 100.0)])

    assert detect_anti_stall(tel) is False


def test_detect_anti_stall_ignores_a_brief_sub_threshold_dip():
    # Near-zero speed under throttle, but for far less than the 0.5s minimum.
    tel = _telemetry([(0.0, 0.0, 80.0), (0.1, 1.0, 80.0), (0.2, 50.0, 80.0), (1.0, 100.0, 80.0)])

    assert detect_anti_stall(tel) is False


def _fake_session_laps_raise():
    from fastf1.exceptions import DataNotLoadedError

    class FakeSession:
        def load(self, **kwargs):
            return None  # "succeeds" without actually populating _laps

        @property
        def laps(self):
            raise DataNotLoadedError("laps data has not been loaded yet")

    return FakeSession()


def test_round_launch_performance_returns_empty_when_laps_fail_to_load_after_a_successful_load_call(monkeypatch):
    """Regression test for the same real failure hit live in GitHub Actions
    (see tests/test_pace_sessions.py's identical case): FastF1's ``.load()``
    can return without raising even when a session's data genuinely isn't
    available yet, so the real failure surfaces later on the ``.laps``
    property access (DataNotLoadedError). This used to propagate uncaught
    out of _load_race_lap1_telemetry; it must now degrade to the same empty
    result this function already returns for "no lap data", not crash."""
    import fastf1

    from f1_fantasy.pace import sessions as sessions_module

    monkeypatch.setattr(sessions_module, "_ensure_cache", lambda: None)
    monkeypatch.setattr(fastf1, "get_session", lambda season, rnd, session: _fake_session_laps_raise())

    assert starts_module.round_launch_performance(2026, 13) == {}


def test_positions_gained_lap1_returns_empty_when_laps_fail_to_load_after_a_successful_load_call(monkeypatch):
    """Same failure mode as above, for positions_gained_lap1's own inline
    session loader."""
    import fastf1

    from f1_fantasy.pace import sessions as sessions_module

    monkeypatch.setattr(sessions_module, "_ensure_cache", lambda: None)
    monkeypatch.setattr(fastf1, "get_session", lambda season, rnd, session: _fake_session_laps_raise())

    assert starts_module.positions_gained_lap1(2026, 13) == {}


def test_detect_anti_stall_only_looks_within_the_launch_window():
    # A clean, fast launch (dense samples, like real telemetry), then a
    # near-zero-speed/high-throttle moment well outside the 2s window (e.g.
    # stuck behind a backmarker mid-lap) that must not count as a stall.
    tel = _telemetry(
        [(0.0, 0.0, 100.0), (0.2, 40.0, 100.0), (0.4, 80.0, 100.0), (5.0, 2.0, 90.0), (5.8, 3.0, 90.0)]
    )

    assert detect_anti_stall(tel) is False
