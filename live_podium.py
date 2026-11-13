"""Capture a fixed pre-race podium experiment, then score its original picks."""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from lxml import html

from practice_pace import (API, DEFAULT_CACHE, ENDPOINTS, MAX_LAP_RATIO, MIN_LAPS,
                           MIN_PEERS, download, meeting_key, race_sessions, timestamp)
from race_data import (DATA_PATH, INDEX, SITE, load_data, parse_race, race_links,
                       table_rows)

DEFAULT_OUTPUT = Path("predictions/2026/live")
SETS = ("practice_on", "practice_off", "grid")


def utc_now():
    return datetime.now(timezone.utc)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def official_page(path):
    """Fetch the requested official page, refusing redirects to another event."""
    if not path.startswith(INDEX):
        raise ValueError("Unexpected official results path")
    url = SITE + path
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 F1 podium test"}), timeout=30) as response:
        final = urlsplit(response.geturl())
        if final.hostname not in {"www.formula1.com", "formula1.com"} or final.path.rstrip("/") != path.rstrip("/"):
            raise ValueError(f"Official source redirected away from {path}; publication cannot be verified")
        body = response.read()
    return html.fromstring(body), {"url": url, "fetched_at_utc": utc_now().isoformat(),
                                    "sha256": sha256(body)}


def api_read(endpoint, params):
    url = f"{API}/{endpoint}?{urlencode(params)}"
    status = 200
    try:
        with urlopen(Request(url, headers={"User-Agent": "F1Predictor/1.0"}), timeout=30) as response:
            body = response.read()
        rows = json.loads(body)
    except HTTPError as error:
        if error.code != 404 or endpoint != "race_control":
            raise
        rows, body, status = [], b"[]", 404
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected OpenF1 {endpoint} response")
    return rows, {"url": url, "fetched_at_utc": utc_now().isoformat(),
                  "http_status": status, "sha256": sha256(body)}


def grand_prix(race, sessions):
    matches = [s for s in sessions if s.get("meeting_key") == meeting_key(race)
               and s.get("year") == 2026 and s.get("session_name") == "Race"]
    if len(matches) != 1 or matches[0].get("is_cancelled"):
        raise ValueError("Cannot verify a unique, non-cancelled Grand Prix session")
    return matches[0]


def require_before_start(session, now, events=()):
    start = timestamp(session["date_start"])
    if now >= start:
        raise ValueError(f"Race has reached its scheduled start ({start.isoformat()}); live capture is closed")
    for event in events:
        if timestamp(event["date"]) > now:
            continue
        message = event.get("message", "").upper()
        if (event.get("category") == "SessionStatus" and any(
                word in message for word in ("STARTED", "RESUMED", "FINISHED", "ENDED", "ABORTED"))) or (
                event.get("lap_number") or 0) > 0 or "LIGHTS OUT" in message:
            raise ValueError("Race-control activity indicates that the Grand Prix has already started")


def verify_live(race):
    """Use fresh schedule and race-control responses; no stale-cache fallback."""
    sessions, schedule_source = api_read("sessions", {"year": 2026})
    session = grand_prix(race, sessions)
    require_before_start(session, utc_now())
    events, control_source = api_read("race_control", {"session_key": session["session_key"]})
    if any(event.get("session_key") != session["session_key"] for event in events):
        raise ValueError("Race-control source does not match the selected Grand Prix")
    checked = utc_now()
    require_before_start(session, checked, events)
    return session, sessions, {"checked_at_utc": checked.isoformat(), "session": session,
                               "race_control": events, "sources": [schedule_source, control_source]}


def published_grid(page):
    rows = table_rows(page, "Time")
    if len(rows) < 3:
        raise ValueError("Published starting grid is unavailable; run capture after it is published")
    numbers = [row["No."] for row in rows]
    if len(numbers) != len(set(numbers)) or any(not row["Driver"] for row in rows):
        raise ValueError("Published grid has duplicate or missing driver identities")
    positions = [int(row["Pos."]) for row in rows if row["Pos."].isdigit()]
    if not {1, 2, 3}.issubset(positions) or len(positions) != len(set(positions)):
        raise ValueError("Published grid does not contain three distinct first starters")
    return rows


def fetch_grid(path):
    try:
        page, source = official_page(path)
    except HTTPError as error:
        if error.code in {404, 410}:
            raise ValueError("Published starting grid is unavailable; run capture after it is published") from error
        raise
    return page, source, published_grid(page)


def training_history(data_path, target, sessions, completed, cutoff):
    history = []
    for race in load_data(data_path):
        if race["race"] == target["race"] or not race["results"]:
            continue
        session = grand_prix(race, sessions)
        if timestamp(session["date_end"]) < cutoff:
            history.append((timestamp(session["date_start"]), race))
    history.sort(key=lambda pair: pair[0])
    history = [race for _, race in history]
    missing = set(completed) - {race["race"] for race in history}
    if missing:
        raise ValueError("Local training data is missing completed races: " + ", ".join(sorted(missing))
                         + ". Refresh it with predict.py --list --update before capturing")
    return history


def cache_practice(history, target, cache):
    """Fill historical gaps and refresh the selected weekend before freezing inputs."""
    sessions = download(cache, "sessions", {"year": 2026}, refresh=True)
    for race in [*history, target]:
        _, practice = race_sessions(race, sessions)
        for session in practice:
            for endpoint in ENDPOINTS:
                download(cache, endpoint, {"session_key": session["session_key"]},
                         refresh=race["race"] == target["race"])


def save_new(path, payload):
    """Exclusive creation: existing captures and scores are never overwritten."""
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        file.write(encoded)


def pick_records(rows, entries, model=True):
    identities = {entry["driver"]: entry for entry in entries}
    if len(identities) != len(entries):
        raise ValueError("Driver abbreviations are not unique in the published grid")
    picks = []
    for rank, row in enumerate(rows[:3], 1):
        entry = identities[row["driver"]]
        record = {"pick_rank": rank, "number": str(entry["number"]),
                  "driver": row["driver"], "team": row["team"], "grid_position": row["grid_position"]}
        if model:
            record["podium_score"] = row["podium_score"]
        picks.append(record)
    return picks


def capture(race_slug, data_path=DATA_PATH, cache=DEFAULT_CACHE, output=DEFAULT_OUTPUT):
    if not re.fullmatch(r"[a-z0-9-]+", race_slug):
        raise ValueError("Use a race slug such as las-vegas")
    index, index_source = official_page(INDEX)
    completed, available = race_links(index)
    if race_slug not in available:
        raise ValueError(f"No official 2026 Grand Prix found for {race_slug}")
    if race_slug in completed:
        raise ValueError("Official race results are already published; live capture is closed")
    path = available[race_slug]
    target = {"race": race_slug, "source": SITE + path}
    session, sessions, initial_check = verify_live(target)
    base = path.rsplit("/", 1)[0]
    grid, grid_source, original_grid = fetch_grid(base + "/starting-grid")
    qualifying, qualifying_source = official_page(base + "/qualifying")
    target = parse_race(race_slug, path, {"starting-grid": grid, "qualifying": qualifying})
    # Only actual published numbered starters can form the grid baseline.
    starters = sorted((row for row in original_grid if row["Pos."].isdigit()),
                      key=lambda row: int(row["Pos."]))[:3]
    now = utc_now()
    history = training_history(data_path, target, sessions, completed, now)
    target["forecast_cutoff_utc"] = now.isoformat()
    cache_practice(history, target, cache)
    cutoff = utc_now()
    require_before_start(session, cutoff)
    target["forecast_cutoff_utc"] = cutoff.isoformat()

    # Model and feature settings deliberately have no CLI tuning switches here.
    from sklearn import __version__ as sklearn_version
    from sklearn.linear_model import LogisticRegression
    from predict import FORM_WINDOW, MIN_TRAINING_RACES, predict_races

    replays = {}
    for enabled, name in ((True, "practice_on"), (False, "practice_off")):
        replay = list(predict_races([*history, target], "podium", recent_movement=False,
                                   grid_weight=0.0, practice_pace=enabled, practice_cache=cache))
        _, method, count, rows = replay[-1]
        if method != "podium":
            raise ValueError("Insufficient earlier training results for the fixed logistic podium model")
        replays[name] = {"training_races": count, "rows": rows}

    on = replays["practice_on"]["rows"]
    entries = target["entries"]
    by_number = {str(entry["number"]): entry for entry in entries}
    grid_rows = [{"driver": by_number[row["No."]]["driver"], "team": by_number[row["No."]]["team"],
                  "grid_position": int(row["Pos."])} for row in starters]
    picks = {name: pick_records(replays[name]["rows"], entries) for name in ("practice_on", "practice_off")}
    picks["grid"] = pick_records(grid_rows, entries, model=False)
    settings = {"model": "podium", "estimator": "LogisticRegression",
                "estimator_parameters": LogisticRegression(C=1.0, max_iter=1000).get_params(),
                "recent_movement": False, "grid_weight": 0.0, "form_window": FORM_WINDOW,
                "min_training_races": MIN_TRAINING_RACES,
                "preprocessing": {"imputer_strategy": "median", "add_indicator": True,
                                  "keep_empty_features": True, "scaler": "StandardScaler"},
                "practice": {"min_consecutive_clean_laps": MIN_LAPS, "min_peers": MIN_PEERS,
                             "max_lap_ratio": MAX_LAP_RATIO, "gap_unit": "seconds"},
                "practice_on": True, "practice_off": False, "scores_calibrated": False,
                "fia_disclosures": "display-only", "sklearn_version": sklearn_version}
    coverage = {"drivers_with_pace": sum(not row["practice_missing"] for row in on),
                "driver_entries": len(on), "usable_samples": sum(row["practice_sample_count"] for row in on),
                "drivers": [{"number": str(next(e["number"] for e in entries if e["driver"] == row["driver"])),
                             "driver": row["driver"],
                             **{k: v for k, v in row.items() if k.startswith("practice_")}} for row in on]}
    provenance = {"training_data_path": str(data_path.resolve()),
                  "training_data_sha256": sha256(data_path.read_bytes()),
                  "python_version": sys.version, "source_sha256": {}, "practice_cache_sha256": {}}
    for name in ("predict.py", "practice_pace.py", "live_podium.py"):
        provenance["source_sha256"][name] = sha256(Path(__file__).with_name(name).read_bytes())
    for file in sorted(cache.rglob("*.json")):
        if file.name != "download-errors.json":
            provenance["practice_cache_sha256"][str(file.relative_to(cache))] = sha256(file.read_bytes())

    # Recheck the exact published grid and live status after potentially slow downloads/fits.
    _, final_grid_source, final_grid_rows = fetch_grid(base + "/starting-grid")
    if final_grid_rows != original_grid:
        raise ValueError("Published grid changed during capture; retry to forecast from the updated grid")
    final_session, _, final_check = verify_live(target)
    captured = utc_now()
    require_before_start(final_session, captured, final_check["race_control"])
    if cutoff >= timestamp(final_session["date_start"]):
        raise ValueError("Forecast cutoff is not before the verified race start")
    payload = {"schema_version": 1, "season": 2026, "race": race_slug,
               "captured_at_utc": captured.isoformat(), "forecast_cutoff_utc": cutoff.isoformat(),
               "race_start_utc": final_session["date_start"], "official_result_url": SITE + path,
               "model_settings": settings, "picks": picks, "practice_coverage": coverage,
               "forecasts": replays, "training_races": history, "target_inputs": target,
               "sources": [index_source, grid_source, qualifying_source, final_grid_source],
               "verification": {"initial": initial_check, "final": final_check}, "provenance": provenance}
    filename = f"{race_slug}-{captured.strftime('%Y%m%dT%H%M%S%fZ')}.snapshot.json"
    destination = output / filename
    # This is the first snapshot write. All rejection paths above leave it unwritten.
    save_new(destination, payload)
    return destination, payload


def validate_snapshot(snapshot):
    if snapshot.get("schema_version") != 1 or snapshot.get("season") != 2026:
        raise ValueError("Unsupported snapshot version or season")
    race = snapshot["race"]
    path = urlsplit(snapshot["official_result_url"])
    if path.hostname != "www.formula1.com" or path.scheme != "https" or not re.fullmatch(
            rf"{INDEX}/\d+/{re.escape(race)}/race-result", path.path):
        raise ValueError("Snapshot does not identify an official result for its selected Grand Prix")
    if not (timestamp(snapshot["forecast_cutoff_utc"]) <= timestamp(snapshot["captured_at_utc"])
            < timestamp(snapshot["race_start_utc"])):
        raise ValueError("Snapshot was not captured before its race start")
    if set(snapshot["picks"]) != set(SETS):
        raise ValueError("Snapshot must contain practice_on, practice_off and grid picks")
    for name in SETS:
        picks = snapshot["picks"][name]
        if (len(picks) != 3 or len({str(p["number"]) for p in picks}) != 3
                or any(not str(p["number"]).isdigit() for p in picks)):
            raise ValueError(f"Snapshot {name} must contain three distinct driver numbers")
    return path.path


def score_snapshot(snapshot_path, output=None):
    original = snapshot_path.read_bytes()
    snapshot = json.loads(original)
    path = validate_snapshot(snapshot)
    index, index_source = official_page(INDEX)
    completed, _ = race_links(index)
    if completed.get(snapshot["race"]) != path:
        raise ValueError("Official result is not yet published for this Grand Prix; saved picks remain unscored")
    page, result_source = official_page(path)
    rows = table_rows(page, "Time / Retired")
    podium = [row for row in rows if row["Pos."] in {"1", "2", "3"}]
    if len(podium) != 3 or len({row["No."] for row in podium}) != 3 or {row["Pos."] for row in podium} != {"1", "2", "3"}:
        raise ValueError("Official classification does not yet contain a complete distinct P1-P3 podium")
    podium.sort(key=lambda row: int(row["Pos."]))
    actual = {row["No."] for row in podium}
    results = []
    for name in SETS:
        picks = snapshot["picks"][name]
        matches = [pick for pick in picks if str(pick["number"]) in actual]
        results.append({"set": name, "podium_hits": len(matches), "total_picks": 3,
                        "original_picks": picks, "matched_picks": matches})
    scored = utc_now()
    payload = {"schema_version": 1, "race": snapshot["race"], "season": 2026,
               "scored_at_utc": scored.isoformat(), "snapshot_path": str(snapshot_path.resolve()),
               "snapshot_sha256": sha256(original), "captured_at_utc": snapshot["captured_at_utc"],
               "official_podium": [{"position": int(row["Pos."]), "number": row["No."],
                                    "driver": row["Driver"]} for row in podium],
               "sources": [index_source, result_source], "scores": results}
    destination = (output if output is not None else snapshot_path.parent / "scores") / (
        f"{snapshot_path.stem}-{scored.strftime('%Y%m%dT%H%M%S%fZ')}.score.json")
    if snapshot_path.read_bytes() != original:
        raise ValueError("Snapshot changed while scoring; score was not saved")
    save_new(destination, payload)
    return destination, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture", help="Save all three fixed pick sets before the race")
    capture_parser.add_argument("--race", required=True)
    capture_parser.add_argument("--data", type=Path, default=DATA_PATH)
    capture_parser.add_argument("--practice-cache", type=Path, default=DEFAULT_CACHE)
    capture_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    score_parser = commands.add_parser("score", help="Score original picks against the published official result")
    score_parser.add_argument("--snapshot", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, help="Folder for a separate timestamped score report")
    args = parser.parse_args()
    try:
        if args.command == "capture":
            path, snapshot = capture(args.race, args.data, args.practice_cache, args.output)
            for name in SETS:
                print(f"{name}: " + " / ".join(p["driver"] for p in snapshot["picks"][name]))
            coverage = snapshot["practice_coverage"]
            print(f"Practice coverage: {coverage['drivers_with_pace']}/{coverage['driver_entries']} drivers")
            print(f"Forecast cutoff UTC: {snapshot['forecast_cutoff_utc']}")
            print(f"Saved snapshot: {path.resolve()}")
        else:
            path, report = score_snapshot(args.snapshot, args.output)
            for score in report["scores"]:
                print(f"{score['set']}: {score['podium_hits']}/3 podium matches (any order)")
            print(f"Saved score report: {path.resolve()}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        action = "snapshot" if args.command == "capture" else "score report"
        parser.exit(1, f"Live podium error: {error}\nNo {action} was written.\n")


if __name__ == "__main__":
    main()
