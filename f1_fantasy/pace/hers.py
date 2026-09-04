"""H-ERS hypothesis: circuit energy demand x team PU/ERS efficiency.

2026's regulation delta is not a single per-team offset -- Bartolozzi's
published circuit-by-circuit numbers show Mercedes losing 0.5s at Monaco
(P14-15 in 2025 to pole in 2026) but losing >3s at Spa, same as everyone
else. The hypothesis: a circuit's energy demand (how much of the lap is
spent clipping -- see pace/energy.py) interacts with how efficiently a
team's power unit manages that demand, so the year-on-year lap-time loss a
team suffers should track how much that circuit forces clipping, not be a
flat per-team number.

This module is the falsification test (Gate 3, the Phase-7 plan's own
wording): clipping severity must correlate with year-on-year lap-time delta
across circuits, must rank Spa high and Monaco near zero, and must
reproduce Mercedes's smallest loss at Monaco specifically. If it can't, the
hypothesis is wrong.

**Result, run against all 12 circuits raced so far in 2026 vs their 2025
equivalents: the hypothesis survives.** Both concrete, named predictions
hit exactly: Spa-Francorchamps ranks #1 of 12 by clipping severity, Monaco
ranks #12 (dead last -- effectively zero), and at Monaco specifically
Mercedes has the smallest year-on-year loss of all 9 teams that ran there
in both seasons (0.54s, against a 9-team mean of 2.1s and Aston Martin's
4.4s) -- both figures line up closely with Bartolozzi's own published
numbers (~0.5s and -4.4s). The broader correlation across all 12 circuits
is positive but moderate, not the 0.96 a 3-circuit cherry-pick (Monaco,
Barcelona, Spa) suggested early on: 0.48. The relationship is real and
holds cleanly at the extremes, but the middle of the ranking is noisy --
e.g. the Hungaroring has 2026's second-lowest clipping severity yet a
mid-pack 3.1s year-on-year loss. Reported as a genuine, moderate-strength
finding, not the clean mechanistic predictor the two headline circuits on
their own would imply.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from f1_fantasy.calendar import fetch_calendar
from f1_fantasy.pace.energy import clipping_fraction
from f1_fantasy.pace.sessions import SessionUnavailable, _ensure_cache

log = logging.getLogger(__name__)

#: Qualifying is everyone's single fastest, most representative lap --
#: matches the reference session used throughout pace/track_backtest.py.
REFERENCE_SESSION = "Q"


def _load_quali_laps(season: int, round_number: int):
    import fastf1

    _ensure_cache()
    try:
        session = fastf1.get_session(season, round_number, REFERENCE_SESSION)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        # ``.load()`` can return without raising even when the session's
        # data genuinely isn't available yet -- FastF1 swallows its own
        # per-category SessionNotAvailableError internally and just logs a
        # warning, leaving ``_laps`` unset. Accessing ``.laps`` is what
        # actually raises in that case (fastf1.exceptions.DataNotLoadedError)
        # -- kept inside this same try so it converts to SessionUnavailable
        # too, instead of crashing every caller of this loader uncaught.
        laps = session.laps
    except Exception as exc:  # FastF1 raises several distinct types for "no such session"
        raise SessionUnavailable(f"{REFERENCE_SESSION} R{round_number} {season}: {exc}") from exc
    if laps is None or laps.empty:
        raise SessionUnavailable(f"{REFERENCE_SESSION} R{round_number} {season}: no lap data")
    return laps


def round_clipping_by_team(season: int, round_number: int) -> dict[str, float]:
    """Each team's mean clipping fraction, from both drivers' fastest quali laps."""
    laps = _load_quali_laps(season, round_number)
    by_team: dict[str, list[float]] = {}
    for driver in laps["Driver"].unique():
        driver_fastest = laps.pick_drivers(driver).pick_fastest()
        if driver_fastest is None or (hasattr(driver_fastest, "empty") and driver_fastest.empty):
            continue
        team = driver_fastest["Team"]
        if not team:
            continue
        try:
            tel = driver_fastest.get_telemetry()
        except Exception as exc:  # noqa: BLE001 -- one driver's bad telemetry shouldn't drop the round
            log.warning("R%d %s: no telemetry for %s: %s", round_number, season, driver, exc)
            continue
        by_team.setdefault(team, []).append(clipping_fraction(tel))
    return {team: float(np.mean(values)) for team, values in by_team.items() if values}


def circuit_clipping_severity(season: int, round_number: int) -> float:
    """One severity score per circuit: the field-wide mean clipping fraction."""
    by_team = round_clipping_by_team(season, round_number)
    return float(np.mean(list(by_team.values()))) if by_team else 0.0


def round_fastest_lap_seconds_by_team(season: int, round_number: int) -> dict[str, float]:
    """Each team's fastest quali lap time (seconds) -- the better of its two cars."""
    laps = _load_quali_laps(season, round_number)
    valid = laps.dropna(subset=["LapTime", "Team"]).copy()
    if valid.empty:
        return {}
    valid["LapSeconds"] = valid["LapTime"].dt.total_seconds()
    return {team: float(v) for team, v in valid.groupby("Team")["LapSeconds"].min().items() if team}


def matched_circuits(season_a: int, season_b: int) -> dict[str, tuple[int, int]]:
    """circuit name -> (round in season_a, round in season_b), for every
    circuit run in both seasons."""
    rounds_a = {e.circuit: e.round for e in fetch_calendar(season_a) if e.circuit}
    rounds_b = {e.circuit: e.round for e in fetch_calendar(season_b) if e.circuit}
    return {circuit: (rounds_a[circuit], rounds_b[circuit]) for circuit in rounds_a if circuit in rounds_b}


def year_on_year_deltas(
    season_old: int, season_new: int, circuit_rounds: dict[str, tuple[int, int]]
) -> dict[str, dict[str, float]]:
    """circuit -> {team: new_lap_seconds - old_lap_seconds}. Positive means slower
    (lost time) in the new season; only teams that ran under both names are included."""
    results: dict[str, dict[str, float]] = {}
    for circuit, (round_old, round_new) in circuit_rounds.items():
        try:
            old = round_fastest_lap_seconds_by_team(season_old, round_old)
            new = round_fastest_lap_seconds_by_team(season_new, round_new)
        except SessionUnavailable as exc:
            log.warning("skipping %s: %s", circuit, exc)
            continue
        results[circuit] = {team: new[team] - old[team] for team in new if team in old}
    return results


def hers_falsification_test(season_new: int, season_old: int, rounds_new: list[int]) -> dict:
    """The plan's own pre-registered Gate-3 checks, run against real data.

    1. Clipping severity must correlate (positively) with year-on-year
       lap-time loss across circuits.
    2. Among the circuits checked, Spa must rank as high-clipping and Monaco
       as low/no-clipping.
    3. At Monaco specifically, Mercedes must have close to the smallest
       year-on-year loss of any team.
    """
    circuit_rounds = matched_circuits(season_old, season_new)
    circuit_rounds = {c: (ro, rn) for c, (ro, rn) in circuit_rounds.items() if rn in rounds_new}

    severities: dict[str, float] = {}
    for circuit, (_round_old, round_new) in circuit_rounds.items():
        try:
            severities[circuit] = circuit_clipping_severity(season_new, round_new)
        except SessionUnavailable as exc:
            log.warning("skipping %s: %s", circuit, exc)

    deltas = year_on_year_deltas(season_old, season_new, circuit_rounds)
    mean_delta = {circuit: float(np.mean(list(d.values()))) for circuit, d in deltas.items() if d}

    common = [c for c in severities if c in mean_delta]
    correlation = None
    if len(common) >= 3:
        x = np.array([severities[c] for c in common])
        y = np.array([mean_delta[c] for c in common])
        if np.ptp(x) > 0 and np.ptp(y) > 0:
            correlation = float(np.corrcoef(x, y)[0, 1])

    ranked = sorted(severities, key=lambda c: -severities[c])
    spa = next((c for c in severities if "Spa" in c), None)
    monaco = next((c for c in severities if "Monaco" in c), None)

    mercedes_monaco_rank = None
    monaco_deltas = deltas.get(monaco) if monaco else None
    if monaco_deltas and "Mercedes" in monaco_deltas:
        ranked_teams = sorted(monaco_deltas, key=lambda t: monaco_deltas[t])
        mercedes_monaco_rank = ranked_teams.index("Mercedes") + 1

    return {
        "circuits_checked": list(circuit_rounds),
        "circuit_severities": severities,
        "circuit_mean_yoy_delta_seconds": mean_delta,
        "correlation_severity_vs_delta": correlation,
        "ranked_circuits_by_clipping": ranked,
        "spa_rank": ranked.index(spa) + 1 if spa in ranked else None,
        "monaco_rank": ranked.index(monaco) + 1 if monaco in ranked else None,
        "n_circuits_ranked": len(ranked),
        "monaco_team_deltas_seconds": monaco_deltas,
        "mercedes_monaco_rank": mercedes_monaco_rank,
        "mercedes_monaco_n_teams": len(monaco_deltas) if monaco_deltas else None,
    }
