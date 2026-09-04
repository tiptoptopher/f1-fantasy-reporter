"""A directional proxy for asymmetric tyre load -- not a wear measurement.

FastF1 carries no tyre sensor data, so which specific corner (front-right,
rear-left, ...) is limiting a car is not something public data can answer.
What it *can* answer: whether a constructor's long-run degradation (already
computed per round in pace/features.py) tracks a track's own corner-direction
balance (pace/track_profile.py's ``corner_direction_balance`` -- more
left-hand corner distance loads the right-side tyres more, and vice versa).
A constructor whose degradation correlates with that balance across the
season is a real, if indirect, signal that one side of the car is the
limiting one -- just not which specific corner.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def team_by_driver(laps) -> dict[str, str]:
    """Driver -> team, from any session's laps (the 'Team' column FastF1
    already carries -- no separate fetch needed)."""
    return dict(zip(laps["Driver"], laps["Team"]))


def constructor_direction_sensitivity(
    rounds: list[dict],
) -> dict[str, dict]:
    """*rounds* is a list of ``{"balance": float, "degradation": {driver:
    (team, degradation_value)}}`` -- one entry per round. Returns, per
    constructor, the Pearson correlation between track corner-direction
    balance and that constructor's mean degradation across the rounds where
    both are known, plus how many rounds fed it.

    A positive correlation: the constructor degrades faster (higher slope)
    at left-corner-dominant tracks -- consistent with a right-side-limited
    car. A negative correlation: the reverse, consistent with a
    left-side-limited car. Small |correlation| or a low round count means
    no real read either way -- reported as such, not forced to a verdict.
    """
    by_team: dict[str, list[tuple[float, float]]] = {}
    for round_data in rounds:
        balance = round_data["balance"]
        for driver, (team, degradation) in round_data["degradation"].items():
            # An unlabelled team (seen live: FP1's Team column comes back
            # blank for some driver/round pairs, likely reserve-driver
            # sessions) would otherwise bucket unrelated cars together under
            # one fake "team" and produce a meaningless correlation.
            if degradation is None or not team:
                continue
            by_team.setdefault(team, []).append((balance, degradation))

    results = {}
    for team, pairs in by_team.items():
        n = len(pairs)
        if n < 4:
            results[team] = {"correlation": None, "n": n, "mean_degradation": None}
            continue
        balances = np.array([p[0] for p in pairs])
        degradations = np.array([p[1] for p in pairs])
        if np.ptp(balances) == 0 or np.ptp(degradations) == 0:
            correlation = None
        else:
            correlation = float(np.corrcoef(balances, degradations)[0, 1])
        results[team] = {
            "correlation": correlation,
            "n": n,
            "mean_degradation": float(np.mean(degradations)),
        }
    return results


def run_backfill(season: int, rounds: list[int]) -> dict:
    """Build the corner-direction/degradation dataset across *rounds* and
    compute each constructor's direction sensitivity.

    Two independent pulls per round: qualifying telemetry (for the track's
    own corner-direction balance, via track_backtest.round_track_and_segments)
    and FP1 lap data (for degradation, via pace.dataset.round_pace, and for
    the Team column this needs but the quali session doesn't cheaply expose).
    """
    import fastf1

    from f1_fantasy.calendar import fetch_calendar
    from f1_fantasy.pace.dataset import round_pace
    from f1_fantasy.pace.sessions import SessionUnavailable, _ensure_cache
    from f1_fantasy.pace.track_backtest import round_track_and_segments

    _ensure_cache()
    events = {e.round: e for e in fetch_calendar(season)}

    track_summaries = []
    rounds_data = []
    skipped = []

    for round_number in rounds:
        event = events.get(round_number)
        if event is None:
            skipped.append({"round": round_number, "reason": "not in calendar"})
            continue

        try:
            result = round_track_and_segments(season, round_number)
        except SessionUnavailable as exc:
            skipped.append({"round": round_number, "reason": str(exc)})
            continue
        if result is None:
            skipped.append({"round": round_number, "reason": "no usable session data"})
            continue
        profile, _segment_profiles = result
        balance = profile.corner_direction_balance
        track_summaries.append(
            {
                "round": round_number,
                "name": event.name,
                "corner_direction_balance": balance,
                "pct_slow": profile.band_distance_pct("slow"),
                "pct_medium": profile.band_distance_pct("medium"),
                "pct_fast": profile.band_distance_pct("fast"),
                "pct_straight": profile.band_distance_pct("straight"),
                "n_corners": len(profile.corners),
            }
        )

        pace, _sessions_used = round_pace(season, round_number, sprint_weekend=event.is_sprint_weekend)

        try:
            session = fastf1.get_session(season, round_number, "FP1")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            # ``.load()`` can return without raising even when the session's
            # data genuinely isn't available yet -- FastF1 swallows its own
            # per-category SessionNotAvailableError internally and just logs
            # a warning, leaving ``_laps`` unset. Accessing ``.laps`` is what
            # actually raises in that case
            # (fastf1.exceptions.DataNotLoadedError) -- kept inside this same
            # try so one bad round is skipped like every other
            # session-unavailable case in this backfill loop, instead of
            # aborting the whole multi-round run.
            teams = team_by_driver(session.laps)
        except Exception as exc:  # noqa: BLE001 -- FastF1 raises several distinct types for "no such session"
            skipped.append({"round": round_number, "reason": f"FP1 R{round_number} {season}: {exc}"})
            continue

        degradation = {}
        for driver_pace in pace:
            team = teams.get(driver_pace.driver)
            if team is None:
                continue
            degradation[driver_pace.driver] = (team, driver_pace.degradation)
        rounds_data.append({"balance": balance, "degradation": degradation})

    sensitivity = constructor_direction_sensitivity(rounds_data)

    return {
        "season": season,
        "rounds_requested": rounds,
        "rounds_used": [t["round"] for t in track_summaries],
        "skipped": skipped,
        "track_summaries": track_summaries,
        "constructor_direction_sensitivity": sensitivity,
    }
