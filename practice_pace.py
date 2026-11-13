"""Cache OpenF1 practice data and derive strictly pre-race long-run features.

No network requests occur during prediction: run this module with --update first.
"""

import argparse
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from race_data import DATA_PATH, load_data

DEFAULT_CACHE = Path(__file__).resolve().parent / "data" / "openf1"
API = "https://api.openf1.org/v1"
ENDPOINTS = ("laps", "stints", "pit", "race_control")
MIN_LAPS = 5
MIN_PEERS = 2
MAX_LAP_RATIO = 1.07  # Remove cooldown/recharge laps before identifying a long run.
DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


def timestamp(value):
    if not value:
        raise ValueError("Missing UTC timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def meeting_key(race):
    match = re.search(r"/results/2026/races/(\d+)/", race.get("source", ""))
    return int(match[1]) if match else None


def read_cache(path):
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload.get("data"), list):
        raise ValueError(f"Invalid OpenF1 cache: {path}")
    return payload["data"]


def download(cache, endpoint, params, refresh=False):
    name = "sessions-2026.json" if endpoint == "sessions" else f"{params['session_key']}/{endpoint}.json"
    path = cache / name
    if path.exists() and not refresh:
        return read_cache(path)
    url = f"{API}/{endpoint}?{urlencode(params)}"
    http_status = 200
    for attempt in range(4):
        # Keep requests below OpenF1's public rate limit.
        time.sleep(2.1)
        try:
            with urlopen(Request(url, headers={"User-Agent": "F1Predictor/1.0"}), timeout=40) as response:
                rows = json.load(response)
            break
        except HTTPError as error:
            if error.code == 404:
                # OpenF1 reports absent endpoint data as 404, including practice pit feeds.
                rows, http_status = [], 404
                break
            if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            time.sleep(min(8, 2 ** (attempt + 1)))
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected response from {url}")
    payload = {"url": url, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
               "http_status": http_status, "data": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return rows


def race_sessions(race, sessions):
    key = meeting_key(race)
    if key is None:
        return None, []
    candidates = [s for s in sessions if s.get("meeting_key") == key and s.get("year") == 2026]
    grands_prix = [s for s in candidates if s.get("session_name") == "Race"
                  and not s.get("is_cancelled")]
    if len(grands_prix) != 1:
        return None, []
    cutoff = timestamp(grands_prix[0]["date_start"])
    if race.get("forecast_cutoff_utc"):
        cutoff = min(cutoff, timestamp(race["forecast_cutoff_utc"]))
    practice = [s for s in candidates if s.get("session_type") == "Practice"
                and s.get("session_name") in {"Practice 1", "Practice 2", "Practice 3"}
                and not s.get("is_cancelled") and timestamp(s["date_end"]) < cutoff]
    return cutoff, sorted(practice, key=lambda s: s["date_start"])


def update_cache(races, cache=DEFAULT_CACHE, refresh=False):
    sessions = download(cache, "sessions", {"year": 2026}, refresh)
    failures = []
    for race in races:
        cutoff, practice = race_sessions(race, sessions)
        if cutoff is None:
            failures.append({"race": race["race"], "error": "Race session not matched"})
        for session in practice:
            if timestamp(session["date_end"]) > datetime.now(timezone.utc):
                continue
            key = session["session_key"]
            for endpoint in ENDPOINTS:
                try:
                    rows = download(cache, endpoint, {"session_key": key}, refresh)
                    print(f"{race['race']} {session['session_name']} {endpoint}: {len(rows)}", flush=True)
                except (OSError, ValueError) as error:
                    failures.append({"race": race["race"], "session_key": key,
                                     "endpoint": endpoint, "error": str(error)})
                    print(f"Missing {race['race']} {key}/{endpoint}: {error}", flush=True)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "download-errors.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    return failures


def flag_intervals(events, session_end):
    """Conservatively exclude any lap overlapping a flag, SC/VSC or suspension."""
    active, intervals = {}, []
    for event in sorted(events, key=lambda e: timestamp(e["date"])):
        date = timestamp(event["date"])
        message = event.get("message", "").upper()
        flag = (event.get("flag") or "").upper()
        scope = event.get("scope") or "Track"
        driver = str(event.get("driver_number")) if scope == "Driver" else None
        key = (scope, event.get("sector"), driver)
        clear = flag in {"GREEN", "CLEAR"} or "CLEAR IN TRACK SECTOR" in message
        if clear:
            keys = list(active) if scope == "Track" else [key]
            for cleared in keys:
                if cleared in active:
                    start, target = active.pop(cleared)
                    intervals.append((start, date, target))
        elif (flag and flag not in {"CHEQUERED"}) or event.get("category") == "SafetyCar" or (
                event.get("category") == "SessionStatus" and any(
                    word in message for word in ("SUSPENDED", "ABORTED", "STOPPED"))):
            active.setdefault(key, (date, driver))
    intervals.extend((start, session_end, target) for start, target in active.values())
    return intervals


def deleted_laps(events, laps):
    deleted = set()
    for event in events:
        message = event.get("message", "").upper()
        if "DELETED" not in message:
            continue
        driver = event.get("driver_number")
        car = re.search(r"CAR\s+(\d+)", message)
        driver = str(driver if driver is not None else car[1] if car else "")
        if not driver:
            # Unresolvable deletion: do not certify this session's laps as valid.
            deleted.update((str(l["driver_number"]), l["lap_number"]) for l in laps)
            continue
        lap_match = re.search(r"\bLAP\s+(\d+)\b", message)
        lap_number = event.get("lap_number") or (int(lap_match[1]) if lap_match else None)
        duration = re.search(r"\b(\d+):(\d{2}\.\d+)\b", message)
        if lap_number:
            deleted.add((driver, lap_number))
        elif duration:
            seconds = 60 * int(duration[1]) + float(duration[2])
            deleted.update((driver, l["lap_number"]) for l in laps
                           if str(l["driver_number"]) == driver and l.get("lap_duration") is not None
                           and abs(l["lap_duration"] - seconds) < 0.002)
        else:
            deleted.update((driver, l["lap_number"]) for l in laps if str(l["driver_number"]) == driver)
    return deleted


def session_pace(session, raw, cutoff):
    """Return clean consecutive long-run laps grouped by driver and dry compound."""
    key = session["session_key"]
    start, end = timestamp(session["date_start"]), timestamp(session["date_end"])
    if end >= cutoff or session.get("session_type") != "Practice":
        return {}, "session_after_cutoff"
    for endpoint in ENDPOINTS:
        if any(row.get("session_key") != key for row in raw[endpoint]):
            return {}, "session_key_mismatch"
    if not raw["race_control"] or not raw["laps"] or not raw["stints"]:
        return {}, "missing_session_data"
    # Later practice corrections are allowed only if they were available before the cutoff.
    events = [event for event in raw["race_control"] if timestamp(event["date"]) < cutoff]
    if not events:
        return {}, "missing_pre_cutoff_race_control"
    intervals = flag_intervals(events, end)
    deleted = deleted_laps(events, raw["laps"])
    pit_laps = {(str(p["driver_number"]), p["lap_number"] + offset)
                for p in raw["pit"] if timestamp(p["date"]) < cutoff for offset in (0, 1)}
    pit_laps.update((str(lap["driver_number"]), lap["lap_number"] - 1)
                    for lap in raw["laps"] if lap.get("is_pit_out_lap") is True
                    and lap.get("date_start") and timestamp(lap["date_start"]) < cutoff)
    runs = defaultdict(dict)
    for lap in raw["laps"]:
        driver, number = str(lap["driver_number"]), lap["lap_number"]
        duration = lap.get("lap_duration")
        if (not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
                or lap.get("is_pit_out_lap") is not False or (driver, number) in pit_laps
                or (driver, number) in deleted or lap.get("is_valid") is False
                or lap.get("is_accurate") is False or lap.get("deleted") is True):
            continue
        sectors = [lap.get(f"duration_sector_{i}") for i in (1, 2, 3)]
        segments = [lap.get(f"segments_sector_{i}") for i in (1, 2, 3)]
        if (not all(isinstance(s, (int, float)) and math.isfinite(s) and s > 0 for s in sectors)
                or abs(sum(sectors) - duration) > 0.1
                # Timing mini-sector colours are not race-control flag states.
                # Partial mini-sector gaps are common; require some known timing
                # in each sector and reject any pitlane/unknown status code.
                or not all(segment and any(value in {2048, 2049, 2051} for value in segment)
                           and all(value in {None, 0, 2048, 2049, 2051} for value in segment)
                           for segment in segments)):
            continue
        if not lap.get("date_start"):
            continue
        lap_start = timestamp(lap["date_start"])
        lap_end = lap_start + timedelta(seconds=duration)
        if lap_start < start or lap_end > end or lap_end >= cutoff:
            continue
        if any(a < lap_end and b > lap_start and target in {None, driver} for a, b, target in intervals):
            continue
        stints = [s for s in raw["stints"] if str(s["driver_number"]) == driver
                  and s.get("compound") in DRY_COMPOUNDS and s.get("lap_start") is not None
                  and s.get("lap_end") is not None and s["lap_start"] < number < s["lap_end"]]
        if len(stints) != 1:
            continue
        stint = stints[0]
        runs[(driver, stint["stint_number"], stint["compound"])][number] = duration
    groups = defaultdict(list)
    for (driver, _, compound), lap_times in runs.items():
        limit = min(lap_times.values()) * MAX_LAP_RATIO
        run = []
        previous = None
        for number, duration in sorted(lap_times.items()):
            if duration > limit or (previous is not None and number != previous + 1):
                if len(run) >= MIN_LAPS:
                    groups[(driver, compound)].extend(run)
                run = []
            if duration > limit:
                previous = None
                continue
            run.append(duration)
            previous = number
        if len(run) >= MIN_LAPS:
            groups[(driver, compound)].extend(run)
    return dict(groups), "ok"


def empty_feature(status, cutoff=None):
    return {"practice_pace_gap": None, "practice_sample_count": 0, "practice_missing": 1,
            "practice_status": status, "practice_cutoff_utc": cutoff,
            "practice_groups": ""}


def source_coverage(races, cache=DEFAULT_CACHE):
    """Audit endpoint availability separately from usable driver sample coverage."""
    sessions_path = cache / "sessions-2026.json"
    sessions = read_cache(sessions_path) if sessions_path.exists() else []
    rows = []
    for race in races:
        cutoff, practice = race_sessions(race, sessions)
        for session in practice:
            for endpoint in ENDPOINTS:
                path = cache / str(session["session_key"]) / f"{endpoint}.json"
                payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                data = payload.get("data", [])
                rows.append({"race": race["race"], "session_key": session["session_key"],
                             "session_name": session["session_name"], "endpoint": endpoint,
                             "status": "available" if data else "empty_feed" if payload else "cache_missing",
                             "records": len(data), "http_status": payload.get("http_status", ""),
                             "source_url": payload.get("url", f"{API}/{endpoint}?session_key={session['session_key']}"),
                             "fetched_at_utc": payload.get("fetched_at_utc", ""),
                             "forecast_cutoff_utc": cutoff.isoformat()})
    return rows


def practice_features(race, cache=DEFAULT_CACHE):
    """One median gap in seconds per driver; peers are equally weighted drivers."""
    drivers = [str(e["number"]) for e in race["entries"]]
    path = cache / "sessions-2026.json"
    if not path.exists():
        return {d: empty_feature("cache_missing") for d in drivers}
    cutoff, sessions = race_sessions(race, read_cache(path))
    if cutoff is None:
        return {d: empty_feature("race_session_missing") for d in drivers}
    output = {d: empty_feature("insufficient_samples", cutoff.isoformat()) for d in drivers}
    values, counts, provenance = defaultdict(list), defaultdict(int), defaultdict(list)
    errors = []
    for session in sessions:
        try:
            raw = {endpoint: read_cache(cache / str(session["session_key"]) / f"{endpoint}.json")
                   for endpoint in ENDPOINTS}
        except FileNotFoundError:
            errors.append("session_cache_missing")
            continue
        groups, status = session_pace(session, raw, cutoff)
        if status != "ok":
            errors.append(status)
            continue
        medians = {group: median(laps) for group, laps in groups.items()}
        for (driver, compound), pace in medians.items():
            peers = [v for (other, tyre), v in medians.items() if other != driver and tyre == compound]
            if len(peers) < MIN_PEERS:
                continue
            values[driver].append(pace - median(peers))
            counts[driver] += len(groups[(driver, compound)])
            provenance[driver].append(f"{session['session_key']}:{compound}:{len(groups[(driver, compound)])}")
    for driver in drivers:
        if values[driver]:
            output[driver].update(practice_pace_gap=median(values[driver]),
                                  practice_sample_count=counts[driver], practice_missing=0,
                                  practice_status="available", practice_groups=";".join(provenance[driver]))
        elif not sessions:
            output[driver]["practice_status"] = "no_prerace_practice"
        elif errors:
            output[driver]["practice_status"] = ";".join(sorted(set(errors))) + ";insufficient_samples"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--race", help="Only cache this race (default: all bundled races)")
    parser.add_argument("--update", action="store_true", required=True)
    parser.add_argument("--refresh", action="store_true", help="Replace existing cached responses")
    args = parser.parse_args()
    races = load_data(args.data)
    if args.race:
        races = [race for race in races if race["race"] == args.race]
        if not races:
            parser.error("Unknown race")
    try:
        failures = update_cache(races, args.cache, args.refresh)
    except (OSError, ValueError) as error:
        parser.exit(1, f"OpenF1 error: {error}\n")
    print(f"Cache: {args.cache.resolve()} | download failures: {len(failures)}")
    if failures:
        parser.exit(1, "Some sources are missing; see download-errors.json. Cached replays remain available.\n")


if __name__ == "__main__":
    main()
