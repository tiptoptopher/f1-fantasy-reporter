"""Constructor direction-sensitivity correlation, against synthetic rounds
with a known relationship built in."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from f1_fantasy.pace import tyre_asymmetry as tyre_asymmetry_module
from f1_fantasy.pace.tyre_asymmetry import constructor_direction_sensitivity, team_by_driver


def test_team_by_driver_reads_the_laps_team_column():
    laps = pd.DataFrame({"Driver": ["VER", "PER"], "Team": ["Red Bull", "Red Bull"]})

    assert team_by_driver(laps) == {"VER": "Red Bull", "PER": "Red Bull"}


def test_a_constructor_whose_degradation_tracks_direction_balance_shows_positive_correlation():
    # Degradation rises in lockstep with corner-direction balance -- a
    # textbook right-side-limited signature.
    rounds = [
        {"balance": -0.5, "degradation": {"VER": ("Red Bull", 0.1)}},
        {"balance": 0.0, "degradation": {"VER": ("Red Bull", 0.3)}},
        {"balance": 0.5, "degradation": {"VER": ("Red Bull", 0.5)}},
        {"balance": 1.0, "degradation": {"VER": ("Red Bull", 0.7)}},
    ]

    result = constructor_direction_sensitivity(rounds)

    assert result["Red Bull"]["correlation"] > 0.99
    assert result["Red Bull"]["n"] == 4


def test_an_inverse_relationship_shows_negative_correlation():
    rounds = [
        {"balance": -0.5, "degradation": {"HAM": ("Ferrari", 0.7)}},
        {"balance": 0.0, "degradation": {"HAM": ("Ferrari", 0.5)}},
        {"balance": 0.5, "degradation": {"HAM": ("Ferrari", 0.3)}},
        {"balance": 1.0, "degradation": {"HAM": ("Ferrari", 0.1)}},
    ]

    result = constructor_direction_sensitivity(rounds)

    assert result["Ferrari"]["correlation"] < -0.99


def test_too_few_rounds_reports_no_correlation_rather_than_a_noisy_one():
    rounds = [
        {"balance": -0.5, "degradation": {"NOR": ("McLaren", 0.2)}},
        {"balance": 0.5, "degradation": {"NOR": ("McLaren", 0.4)}},
    ]

    result = constructor_direction_sensitivity(rounds)

    assert result["McLaren"]["correlation"] is None
    assert result["McLaren"]["n"] == 2


def test_a_blank_team_name_is_excluded_rather_than_bucketed_as_a_fake_team():
    """Confirmed live: FP1's Team column comes back blank for some
    driver/round pairs. Bucketing those under "" would mix unrelated
    drivers' degradation into one meaningless "team"."""
    rounds = [
        {"balance": -0.5, "degradation": {"A": ("", 0.1), "VER": ("Red Bull", 0.2)}},
        {"balance": 0.5, "degradation": {"B": ("", 0.9), "VER": ("Red Bull", 0.4)}},
    ]

    result = constructor_direction_sensitivity(rounds)

    assert "" not in result
    assert "Red Bull" in result


def test_run_backfill_skips_a_round_whose_fp1_laps_fail_to_load_after_a_successful_load_call(monkeypatch):
    """Regression test for the same real failure hit live in GitHub Actions
    (see tests/test_pace_sessions.py's identical case): FastF1's ``.load()``
    can return without raising even when a session's data genuinely isn't
    available yet, so the real failure surfaces later on the ``.laps``
    property access (DataNotLoadedError). This is a loop over multiple
    rounds, so the fix must skip just this round -- like every other
    session-unavailable case here -- rather than letting one bad round
    abort the whole multi-round backfill."""
    import fastf1
    from fastf1.exceptions import DataNotLoadedError

    from f1_fantasy import calendar as calendar_module
    from f1_fantasy.pace import dataset as dataset_module
    from f1_fantasy.pace import sessions as sessions_module
    from f1_fantasy.pace import track_backtest as track_backtest_module

    event = SimpleNamespace(round=6, name="Monaco Grand Prix", circuit="Monaco", is_sprint_weekend=False)
    monkeypatch.setattr(calendar_module, "fetch_calendar", lambda season: [event])

    profile = SimpleNamespace(corner_direction_balance=0.2, band_distance_pct=lambda band: 0.0, corners=[])
    monkeypatch.setattr(
        track_backtest_module, "round_track_and_segments", lambda season, round_number: (profile, [])
    )
    monkeypatch.setattr(dataset_module, "round_pace", lambda season, round_number, sprint_weekend=False: ([], []))

    class FakeFP1Session:
        def load(self, **kwargs):
            return None  # "succeeds" without actually populating _laps

        @property
        def laps(self):
            raise DataNotLoadedError("laps data has not been loaded yet")

    monkeypatch.setattr(sessions_module, "_ensure_cache", lambda: None)
    monkeypatch.setattr(fastf1, "get_session", lambda season, rnd, name: FakeFP1Session())

    result = tyre_asymmetry_module.run_backfill(2026, [6])

    assert result["skipped"] == [{"round": 6, "reason": "FP1 R6 2026: laps data has not been loaded yet"}]
    assert result["constructor_direction_sensitivity"] == {}


def test_missing_degradation_values_are_skipped_not_treated_as_zero():
    rounds = [
        {"balance": -0.5, "degradation": {"NOR": ("McLaren", 0.2)}},
        {"balance": 0.0, "degradation": {"NOR": ("McLaren", None)}},
        {"balance": 0.5, "degradation": {"NOR": ("McLaren", 0.4)}},
        {"balance": 1.0, "degradation": {"NOR": ("McLaren", 0.6)}},
    ]

    result = constructor_direction_sensitivity(rounds)

    assert result["McLaren"]["n"] == 3
