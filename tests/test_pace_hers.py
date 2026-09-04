"""H-ERS aggregation logic, against synthetic circuit/team data.

The real telemetry functions (round_clipping_by_team,
round_fastest_lap_seconds_by_team) need FastF1 sessions and are exercised by
the CLI's hers-backtest command against real 2025/2026 data, not here --
these tests check matched_circuits, year_on_year_deltas and the
falsification test's aggregation (ranking, correlation, Monaco-rank) against
known-answer synthetic inputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from f1_fantasy.pace import hers


def test_matched_circuits_keeps_only_circuits_run_in_both_seasons():
    def fake_calendar(season):
        if season == 2025:
            return [
                SimpleNamespace(circuit="Monaco", round=8),
                SimpleNamespace(circuit="Spa", round=13),
                SimpleNamespace(circuit="Only2025", round=20),
            ]
        return [
            SimpleNamespace(circuit="Monaco", round=6),
            SimpleNamespace(circuit="Spa", round=10),
            SimpleNamespace(circuit="Only2026", round=14),
        ]

    import f1_fantasy.pace.hers as hers_module

    orig = hers_module.fetch_calendar
    hers_module.fetch_calendar = fake_calendar
    try:
        result = hers.matched_circuits(2025, 2026)
    finally:
        hers_module.fetch_calendar = orig

    assert result == {"Monaco": (8, 6), "Spa": (13, 10)}


def test_year_on_year_deltas_only_includes_teams_present_in_both_seasons(monkeypatch):
    def fake_lap_seconds(season, round_number):
        if season == 2025:
            return {"Mercedes": 70.0, "Ferrari": 71.0, "OldTeamOnly": 72.0}
        return {"Mercedes": 70.5, "Ferrari": 73.0, "NewTeamOnly": 69.0}

    monkeypatch.setattr(hers, "round_fastest_lap_seconds_by_team", fake_lap_seconds)

    result = hers.year_on_year_deltas(2025, 2026, {"Monaco": (8, 6)})

    assert result == {"Monaco": {"Mercedes": pytest.approx(0.5), "Ferrari": pytest.approx(2.0)}}


def test_falsification_test_ranks_spa_above_monaco_and_finds_mercedes_smallest_loss(monkeypatch):
    # A third circuit, mid-severity, is included purely so the correlation
    # has the minimum 3 points this module requires to report one at all --
    # 2 points would trivially correlate at +-1 regardless of signal.
    monkeypatch.setattr(
        hers, "matched_circuits", lambda old, new: {"Monaco": (8, 6), "Spa": (13, 10), "Silverstone": (12, 9)}
    )
    monkeypatch.setattr(
        hers,
        "circuit_clipping_severity",
        lambda season, round_number: {6: 0.01, 10: 0.25, 9: 0.12}[round_number],
    )
    monkeypatch.setattr(
        hers,
        "year_on_year_deltas",
        lambda old, new, circuit_rounds: {
            "Monaco": {"Mercedes": 0.5, "Ferrari": 2.2, "McLaren": 2.6},
            "Spa": {"Mercedes": 3.0, "Ferrari": 3.4, "McLaren": 3.5},
            "Silverstone": {"Mercedes": 1.6, "Ferrari": 1.9, "McLaren": 2.0},
        },
    )

    result = hers.hers_falsification_test(2026, 2025, [6, 10, 9])

    assert result["spa_rank"] == 1
    assert result["monaco_rank"] == 3  # lowest clipping of the three
    assert result["mercedes_monaco_rank"] == 1
    assert result["correlation_severity_vs_delta"] > 0.9  # more clipping, more time lost


def test_load_quali_laps_converts_a_post_load_failure_to_session_unavailable(monkeypatch):
    """Regression test for the same real failure hit live in GitHub Actions
    (see tests/test_pace_sessions.py's identical case for pace/sessions.py's
    load_laps, which this function follows): FastF1's ``.load()`` can return
    without raising even when a session's data genuinely isn't available
    yet -- it swallows its own per-category SessionNotAvailableError
    internally and just logs a warning, leaving ``_laps`` unset -- so the
    real failure surfaces later, on the ``.laps`` property access
    (DataNotLoadedError), which used to propagate uncaught past this
    function's try/except instead of degrading to SessionUnavailable."""
    import fastf1
    from fastf1.exceptions import DataNotLoadedError

    from f1_fantasy.pace import hers as hers_module
    from f1_fantasy.pace.sessions import SessionUnavailable

    class FakeSession:
        def load(self, **kwargs):
            return None  # "succeeds" without actually populating _laps

        @property
        def laps(self):
            raise DataNotLoadedError("laps data has not been loaded yet")

    monkeypatch.setattr(hers_module, "_ensure_cache", lambda: None)
    monkeypatch.setattr(fastf1, "get_session", lambda season, rnd, session: FakeSession())

    with pytest.raises(SessionUnavailable):
        hers_module._load_quali_laps(2026, 13)


def test_falsification_test_handles_a_session_that_cannot_be_loaded(monkeypatch):
    from f1_fantasy.pace.sessions import SessionUnavailable

    monkeypatch.setattr(hers, "matched_circuits", lambda old, new: {"Monaco": (8, 6)})

    def raises(season, round_number):
        raise SessionUnavailable("no data")

    monkeypatch.setattr(hers, "circuit_clipping_severity", raises)
    monkeypatch.setattr(hers, "year_on_year_deltas", lambda old, new, circuit_rounds: {})

    result = hers.hers_falsification_test(2026, 2025, [6])

    assert result["circuit_severities"] == {}
    assert result["correlation_severity_vs_delta"] is None
