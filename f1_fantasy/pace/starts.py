"""Launch performance: lap-1 race telemetry, not qualifying.

Own loader, following ``pace/hers.py``'s precedent of not touching
``pace/sessions.py``'s ``load_laps`` (different session, different need --
the race session's first lap, not qualifying's fastest lap). Confirmed live
against 2026 round 10: a single-lap ``Laps`` slice's own ``get_telemetry()``
returns ``Time`` already relative to that lap's own start, so no separate
"zero the clock at the green light" step is needed -- ``Time`` at the first
telemetry sample is already ~0.

``detect_anti_stall`` is deliberately a different signal from
``pace/energy.py``'s clipping: clipping is full throttle *with* a negative
speed derivative (an energy-limited power unit that cannot hold speed);
anti-stall/wheelspin is partial, modulated throttle *with* speed pinned near
zero for longer than a clean launch should take (the driver is trying to
get away but the car isn't responding) -- the opposite throttle signature,
so it needs its own detector rather than reusing energy.py's.

**Gate A (same-race, explanatory), run against real 2026 rounds 1-12**:
launch time-to-100km/h correlates with lap-1 positions gained at r = -0.11
(n = 254 driver-races) -- directionally right (a slower launch should cost
places) but weak, and not distinguishable from zero at conventional
significance for this sample size (|r| would need to exceed ~0.12). No
clear anti-stall outlier occurred in this 12-round sample (see
STALL_DURATION_THRESHOLD's own docstring for the diagnostic that found
this). Reported as a weak, inconclusive Gate A result, not a pass --
wiring this into points.py's overtake/positions-gained rate is explicitly
deferred per the approved plan until a real signal exists to justify it.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

#: Launch speed FastF1's own telemetry channel reports in km/h. Bartolozzi's
#: published launch analyses use a 0-100 km/h figure.
DEFAULT_TARGET_SPEED = 100.0

#: Below this speed (km/h), in the window right after the green light, with
#: the driver still on the throttle, counts as "the car isn't responding".
STALL_SPEED_THRESHOLD = 5.0

#: Throttle reading above which the driver is genuinely trying to accelerate
#: (not simply off the pedal at the very first sample).
STALL_THROTTLE_THRESHOLD = 10.0

#: How long a stalled spell has to last, within the window, to count as a
#: real anti-stall event rather than ordinary launch dynamics.
#:
#: Not a guess: an initial 0.5s threshold was checked against all 257 real
#: driver-races in 2026 rounds 1-12 and flagged 219 of them (86%) -- clearly
#: wrong, since every standing start involves *some* sub-5-km/h, throttle-on
#: moment while the clutch engages, and a real anti-stall failure is meant
#: to be rare. The real distribution has no outliers at all: median 0.73s,
#: p99 1.68s, max 2.09s (Leclerc, round 1) -- a single smooth curve with no
#: separate "stalled" cluster. This threshold is set just above that
#: observed maximum, so the detector fires on genuine outliers beyond
#: anything any real 2026 launch produced, not on normal variation. It has
#: not been validated against a confirmed real anti-stall incident, because
#: none occurred in this 12-round sample -- reported as a calibration
#: choice, not a proven detector, same honesty standard as the rest of this
#: module's sibling docstrings.
STALL_DURATION_THRESHOLD = 2.5

#: How far into the lap to look for a stall at all -- a genuine launch
#: problem shows up in the first couple of seconds, not mid-straight.
STALL_WINDOW_SECONDS = 2.0


def _load_race_lap1_telemetry(season: int, round_number: int) -> dict[str, pd.DataFrame]:
    import fastf1

    from f1_fantasy.pace.sessions import _ensure_cache

    _ensure_cache()
    try:
        session = fastf1.get_session(season, round_number, "R")
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        # ``.load()`` can return without raising even when the session's
        # data genuinely isn't available yet -- FastF1 swallows its own
        # per-category SessionNotAvailableError internally and just logs a
        # warning, leaving ``_laps`` unset. Accessing ``.laps`` is what
        # actually raises in that case (fastf1.exceptions.DataNotLoadedError)
        # -- kept inside this same try so a load failure degrades to the
        # same "no lap data" empty result below, instead of crashing the
        # caller uncaught.
        laps = session.laps
    except Exception as exc:  # noqa: BLE001 -- FastF1 raises several distinct types for "no such session"
        log.warning("R%d %s: could not load lap-1 telemetry: %s", round_number, season, exc)
        return {}
    if laps is None or laps.empty:
        return {}
    lap1 = laps.pick_laps(1)

    telemetry: dict[str, pd.DataFrame] = {}
    for driver in lap1["Driver"].unique():
        driver_lap1 = lap1.pick_drivers(driver)
        if driver_lap1.empty:
            continue
        try:
            tel = driver_lap1.get_telemetry()
        except Exception:  # noqa: BLE001 -- a retirement before the flag can leave no telemetry at all
            continue
        if tel is None or tel.empty:
            continue
        telemetry[driver] = tel
    return telemetry


def launch_time_to_speed(telemetry: pd.DataFrame, *, target_speed: float = DEFAULT_TARGET_SPEED) -> float | None:
    """Seconds from the green light to first reaching *target_speed* km/h.
    None if the car never reaches it within this lap's telemetry (e.g. a
    retirement before getting there, or a very slow opening lap)."""
    tel = telemetry.dropna(subset=["Speed", "Time"]).sort_values("Time")
    if tel.empty:
        return None
    reached = tel[tel["Speed"] >= target_speed]
    if reached.empty:
        return None
    return float(reached["Time"].iloc[0].total_seconds())


def detect_anti_stall(
    telemetry: pd.DataFrame,
    *,
    stall_speed_threshold: float = STALL_SPEED_THRESHOLD,
    stall_throttle_threshold: float = STALL_THROTTLE_THRESHOLD,
    stall_duration_threshold: float = STALL_DURATION_THRESHOLD,
    window_seconds: float = STALL_WINDOW_SECONDS,
) -> bool:
    """A pronounced near-zero-speed dip in the first couple of seconds after
    the green light, with the driver on the throttle throughout -- distinct
    from simply not having moved yet (speed near zero with throttle near
    zero too, which is just the normal pre-launch state)."""
    tel = telemetry.dropna(subset=["Speed", "Throttle", "Time"]).sort_values("Time")
    if len(tel) < 2:
        return False
    time_s = tel["Time"].dt.total_seconds().to_numpy(dtype=float)
    speed = tel["Speed"].to_numpy(dtype=float)
    throttle = tel["Throttle"].to_numpy(dtype=float)

    stalled = (time_s <= window_seconds) & (speed < stall_speed_threshold) & (throttle > stall_throttle_threshold)

    total = 0.0
    run_start: int | None = None
    for i, is_stalled in enumerate(stalled):
        if is_stalled and run_start is None:
            run_start = i
        elif not is_stalled and run_start is not None:
            total += time_s[i] - time_s[run_start]
            run_start = None
    if run_start is not None:
        total += time_s[len(stalled) - 1] - time_s[run_start]
    return bool(total >= stall_duration_threshold)


def round_launch_performance(season: int, round_number: int) -> dict[str, dict]:
    """Per driver: time_to_100kmh and an anti_stall flag, from this round's
    real lap-1 race telemetry."""
    telemetry = _load_race_lap1_telemetry(season, round_number)
    return {
        driver: {
            "time_to_100kmh": launch_time_to_speed(tel),
            "anti_stall": detect_anti_stall(tel),
        }
        for driver, tel in telemetry.items()
    }


def positions_gained_lap1(season: int, round_number: int) -> dict[str, int]:
    """Grid position minus position at the end of lap 1 -- ground truth for
    whether launch performance predicts early position changes, the one
    scoring.py source (POINTS_PER_PLACE_GAINED) this module could feed."""
    import fastf1

    from f1_fantasy.pace.sessions import _ensure_cache
    from f1_fantasy.results import fetch_race_results

    _ensure_cache()
    try:
        session = fastf1.get_session(season, round_number, "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        # ``.load()`` can return without raising even when the session's
        # data genuinely isn't available yet -- FastF1 swallows its own
        # per-category SessionNotAvailableError internally and just logs a
        # warning, leaving ``_laps`` unset. Accessing ``.laps`` is what
        # actually raises in that case (fastf1.exceptions.DataNotLoadedError)
        # -- kept inside this same try so a load failure degrades to the
        # same "no lap data" empty result below, instead of crashing the
        # caller uncaught.
        laps = session.laps
    except Exception as exc:  # noqa: BLE001 -- FastF1 raises several distinct types for "no such session"
        log.warning("R%d %s: could not load lap-1 positions: %s", round_number, season, exc)
        return {}
    if laps is None or laps.empty:
        return {}
    lap1 = laps.pick_laps(1).dropna(subset=["Position"])
    position_after_lap1 = dict(zip(lap1["Driver"], lap1["Position"]))

    gains = {}
    for result in fetch_race_results(season, round_number):
        position = position_after_lap1.get(result.driver_code)
        if position is None or result.grid <= 0:
            continue
        gains[result.driver_code] = int(result.grid - position)
    return gains
