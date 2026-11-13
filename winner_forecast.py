"""Pre-qualifying race-winner forecasts using earlier-round information only."""

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path
from statistics import mean

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from live_podium import (api_read, grand_prix, official_page, save_new, sha256,
                         utc_now)
from practice_pace import download, meeting_key, timestamp
from race_data import DATA_PATH, INDEX, race_links

DEFAULT_CACHE = Path(__file__).resolve().parent / "data" / "winner_sources"
DEFAULT_OUTPUT = Path("predictions/2026/prequalifying")
FEATURES = ("driver_finish_last3", "team_finish_last3", "driver_q1_gap_last3", "team_q1_gap_last3")
MIN_TRAINING_RACES = 3
WINDOW = 3


def qualifying(race, sessions):
    matches = [s for s in sessions if s.get("meeting_key") == meeting_key(race)
               and s.get("year") == 2026 and s.get("session_name") == "Qualifying"
               and not s.get("is_cancelled")]
    if len(matches) != 1:
        raise ValueError(f"Cannot verify the Grand Prix qualifying session for {race['race']}")
    return matches[0]


def entry_roster(rows, gp):
    if len(rows) < 3:
        raise ValueError("Published race entry roster is unavailable; no forecast will be saved")
    entries = []
    for row in rows:
        if row.get("session_key") != gp["session_key"] or row.get("meeting_key") != gp["meeting_key"]:
            raise ValueError("Entry roster belongs to a different Grand Prix or session")
        if not row.get("name_acronym") or not row.get("team_name"):
            raise ValueError("Entry roster is missing driver or team identity")
        entries.append({"number": str(row["driver_number"]), "driver": row["name_acronym"],
                        "full_name": row.get("full_name", row["name_acronym"]), "team": row["team_name"]})
    if len({e["number"] for e in entries}) != len(entries):
        raise ValueError("Entry roster contains duplicate driver numbers")
    return sorted(entries, key=lambda e: int(e["number"]))


def schedule(races, sessions):
    output = {}
    for race in races:
        gp, q = grand_prix(race, sessions), qualifying(race, sessions)
        output[race["race"]] = {"race": gp, "qualifying": q}
    return output


def feature_rows(target, cutoff, races, times):
    """Compute four last-three-round statistics, with per-value source evidence.

    This function never reads the target's grid, qualifying gaps or results.
    Session completion times bound event availability; archive publication times
    remain explicitly unverified for historical reconstructions.
    """
    prior = sorted((r for r in races if r["race"] != target["race"]
                    and timestamp(times[r["race"]]["race"]["date_start"]) <
                    timestamp(times[target["race"]]["race"]["date_start"])),
                   key=lambda r: timestamp(times[r["race"]]["race"]["date_start"]))
    output = []
    for entry in target["entries"]:
        values = {key: [] for key in FEATURES}
        for race in prior:
            race_time = times[race["race"]]
            result_end = timestamp(race_time["race"]["date_end"])
            q_end = timestamp(race_time["qualifying"]["date_end"])
            team_entries = [e for e in race["entries"] if e["team"] == entry["team"]]
            source = race["source"].rsplit("/", 1)[0]

            def add(key, value, available, kind):
                if value is not None and math.isfinite(value):
                    values[key].append({"value": value, "source_race": race["race"],
                                        "available_after_event_utc": available.isoformat(),
                                        "source_url": source + "/" + kind})

            if result_end < cutoff:
                add("driver_finish_last3", race["results"].get(entry["number"]), result_end, "race-result")
                finishes = [race["results"][e["number"]] for e in team_entries if e["number"] in race["results"]]
                if finishes:
                    add("team_finish_last3", mean(finishes), result_end, "race-result")
            if q_end < cutoff:
                driver = next((e for e in race["entries"] if e["number"] == entry["number"]), None)
                add("driver_q1_gap_last3", driver.get("q1_gap_pct") if driver else None, q_end, "qualifying")
                gaps = [e["q1_gap_pct"] for e in team_entries if e.get("q1_gap_pct") is not None]
                if gaps:
                    add("team_q1_gap_last3", mean(gaps), q_end, "qualifying")
        selected = {key: samples[-WINDOW:] for key, samples in values.items()}
        statistics = {key: mean(s["value"] for s in samples) if samples else None
                      for key, samples in selected.items()}
        for samples in selected.values():
            if any(timestamp(s["available_after_event_utc"]) >= cutoff for s in samples):
                raise ValueError("Feature timestamp verification failed")
        output.append({**entry, **statistics, "missing": {key: int(value is None) for key, value in statistics.items()},
                       "counts": {key: len(samples) for key, samples in selected.items()},
                       "evidence": selected, "forecast_cutoff_utc": cutoff.isoformat()})
    return output


def matrix(rows):
    return [[row[key] if row[key] is not None else float("nan") for key in FEATURES]
            + [row["missing"][key] for key in FEATURES] for row in rows]


def baseline_pick(rows):
    # A missing finish history ranks behind any observed finish history. Ties use
    # driver number, never grid order or current-weekend qualifying information.
    return min(rows, key=lambda r: (r["driver_finish_last3"] if r["driver_finish_last3"] is not None else float("inf"),
                                   int(r["number"])))


def examples(races, rosters, times, lead_seconds):
    output = []
    for race in sorted(races, key=lambda r: timestamp(times[r["race"]]["race"]["date_start"])):
        cutoff = timestamp(times[race["race"]]["qualifying"]["date_start"]) - timedelta(seconds=lead_seconds)
        target = {**race, "entries": rosters[race["race"]]}
        rows = feature_rows(target, cutoff, races, times)
        actual = [number for number, position in race["results"].items() if position == 1]
        if len(actual) != 1 or actual[0] not in {r["number"] for r in rows}:
            raise ValueError(f"Cannot match a unique official winner to the entry roster for {race['race']}")
        output.append({"race": race["race"], "cutoff": cutoff, "rows": rows,
                       "result_available": timestamp(times[race["race"]]["race"]["date_end"]),
                       "labels": [int(row["number"] == actual[0]) for row in rows], "actual_winner": actual[0]})
    return output


def forecast(rows, cutoff, training):
    eligible = [r for r in training if r["result_available"] < cutoff and r["cutoff"] < cutoff]
    if len(eligible) < MIN_TRAINING_RACES:
        return None, eligible
    features = [row for race in eligible for row in race["rows"]]
    labels = [label for race in eligible for label in race["labels"]]
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=1000))
    model.fit(matrix(features), labels)
    scores = model.predict_proba(matrix(rows))[:, 1]
    ranked = sorted(({**row, "winner_score": float(score)} for row, score in zip(rows, scores)),
                    key=lambda row: (-row["winner_score"], int(row["number"])))
    return ranked, eligible


def backtest(training):
    details = []
    for race in training:
        ranked, eligible = forecast(race["rows"], race["cutoff"], [r for r in training if r is not race])
        if ranked is None:
            continue
        baseline = baseline_pick(race["rows"])
        actual = next(r for r in race["rows"] if r["number"] == race["actual_winner"])
        details.append({"race": race["race"], "forecast_cutoff_utc": race["cutoff"].isoformat(),
                        "training_races": len(eligible), "training_driver_rows": sum(len(r["rows"]) for r in eligible),
                        "predicted_winner": ranked[0]["driver"], "winner_score": ranked[0]["winner_score"],
                        "baseline_pick": baseline["driver"], "baseline_avg_finish": baseline["driver_finish_last3"],
                        "actual_winner": actual["driver"], "model_correct": int(ranked[0]["number"] == actual["number"]),
                        "baseline_correct": int(baseline["number"] == actual["number"]),
                        "feature_event_times_verified": True, "historical_publication_times_verified": False})
    summary = [{"model": name, "races": len(details), "correct_winners": sum(row[key] for row in details),
                "winner_accuracy": mean(row[key] for row in details) if details else None}
               for name, key in (("logistic-winner", "model_correct"), ("last3-average-finish", "baseline_correct"))]
    return details, summary


def require_prequal(target, sessions, now):
    q = qualifying(target, sessions)
    gp = grand_prix(target, sessions)
    if now >= timestamp(q["date_start"]) or now >= timestamp(gp["date_start"]):
        raise ValueError("Pre-qualifying forecast is closed: qualifying or the Grand Prix has already reached its start")
    return q


def run(race_slug=None, do_backtest=False, data_path=DATA_PATH, output=DEFAULT_OUTPUT,
        dashboard=False, cache=DEFAULT_CACHE, lead_hours=192.0):
    if not race_slug and not do_backtest:
        raise ValueError("Choose --race, --backtest, or both")
    if not math.isfinite(lead_hours) or lead_hours <= 0:
        raise ValueError("--lead-hours must be finite and positive")
    data_bytes = data_path.read_bytes()
    races = [r for r in json.loads(data_bytes) if r["results"]]
    data_observed = utc_now()
    sessions = download(cache, "sessions", {"year": 2026}, refresh=True)
    target, target_source, target_entries, entry_observed = None, None, None, None
    if race_slug:
        index, _ = official_page(INDEX)
        completed, available = race_links(index)
        if race_slug not in available or race_slug in completed:
            raise ValueError("Selected Grand Prix is unknown or already has a published result")
        target = {"race": race_slug, "source": "https://www.formula1.com" + available[race_slug]}
        require_prequal(target, sessions, utc_now())
        gp = grand_prix(target, sessions)
        raw, target_source = api_read("drivers", {"session_key": gp["session_key"]})
        target_entries = entry_roster(raw, gp)
        entry_observed = utc_now()
        target["entries"] = target_entries
        # Retain the published roster itself, not a grid or previous-race field.
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"target-{race_slug}-entries.json").write_text(
            json.dumps({"source": target_source, "observed_at_utc": entry_observed.isoformat(), "data": raw}, indent=2) + "\n",
            encoding="utf-8")
    times = schedule([*races, *([target] if target else [])], sessions)
    rosters = {}
    for race in races:
        gp = times[race["race"]]["race"]
        raw = download(cache, "drivers", {"session_key": gp["session_key"]})
        rosters[race["race"]] = entry_roster(raw, gp)
    cutoff = utc_now()
    if target:
        q = require_prequal(target, sessions, cutoff)
        if entry_observed > cutoff or timestamp(target_source["fetched_at_utc"]) > cutoff:
            raise ValueError("Entry list was not observed before the forecast cutoff")
        if data_observed > cutoff:
            raise ValueError("Historical feature inputs were not observed before the forecast cutoff")
        lead_seconds = (timestamp(q["date_start"]) - cutoff).total_seconds()
    else:
        lead_seconds = lead_hours * 3600
    training = examples(races, rosters, times, lead_seconds)
    forecast_payload = None
    if target:
        rows = feature_rows(target, cutoff, races, times)
        ranked, eligible = forecast(rows, cutoff, training)
        if ranked is None:
            raise ValueError("Not enough completed earlier races before the forecast cutoff")
        # Revalidate a fresh schedule after downloads and fitting; never publish late.
        fresh, _ = api_read("sessions", {"year": 2026})
        fresh_q = require_prequal(target, fresh, utc_now())
        if fresh_q["date_start"] != qualifying(target, sessions)["date_start"]:
            raise ValueError("Qualifying schedule changed during the run; retry")
        forecast_payload = {"mode": "Pre-qualifying forecast", "race": race_slug,
                            "forecast_cutoff_utc": cutoff.isoformat(), "lead_seconds_before_qualifying": lead_seconds,
                            "qualifying_start_utc": q["date_start"], "entry_list_source": target_source,
                            "entry_list_observed_at_utc": entry_observed.isoformat(), "published_entries": target_entries,
                            "historical_inputs_observed_at_utc": data_observed.isoformat(),
                            "training_races": [r["race"] for r in eligible],
                            "training_driver_rows": sum(len(r["rows"]) for r in eligible),
                            "rankings": ranked, "baseline_pick": baseline_pick(rows),
                            "feature_event_times_verified": True, "live_entry_list_available_at_cutoff": True,
                            "live_feature_inputs_observed_before_cutoff": True,
                            "model_settings": {"estimator": "LogisticRegression", "C": 1.0, "max_iter": 1000,
                                               "features": FEATURES, "missing_indicators": True,
                                               "history_window": WINDOW, "scores_calibrated": False},
                            "data_sha256": sha256(data_bytes),
                            "historical_archive_note": "Session completion timestamps verify event ordering. Historical roster/publication vintages are not available; this is a reconstructed backtest, not fully verified point-in-time evidence."}
        path = output / f"{race_slug}-{cutoff.strftime('%Y%m%dT%H%M%S%fZ')}.winner.json"
        save_new(path, forecast_payload)
        if dashboard:
            from terminal_dashboard import display_winner_forecast
            display_winner_forecast(forecast_payload)
        else:
            print("Pre-qualifying forecast: " + " / ".join(r["full_name"] for r in ranked[:3]))
            print(f"Forecast cutoff UTC: {cutoff.isoformat()}")
        print(f"Saved forecast: {path.resolve()}")
    if do_backtest:
        from predict import write_csv
        details, summary = backtest(training)
        if not details:
            raise ValueError("No races have sufficient earlier training results at the selected cutoffs")
        write_csv(output / "winner-backtest-per-race.csv", details)
        write_csv(output / "winner-backtest-summary.csv", summary)
        metadata = {"lead_seconds_before_qualifying": lead_seconds,
                    "cutoff_rule": "Each Grand Prix qualifying start minus this fixed lead time",
                    "training_warmup": MIN_TRAINING_RACES,
                    "historical_publication_times_verified": False,
                    "note": "Current archived entry rosters and classifications; event times are checked, but historical publication vintages are unavailable.",
                    "features": FEATURES, "data_sha256": sha256(data_bytes),
                    "examples": [{"race": r["race"], "cutoff": r["cutoff"].isoformat(), "features": r["rows"],
                                  "labels": r["labels"], "result_available": r["result_available"].isoformat()}
                                 for r in training]}
        (output / "winner-backtest-audit.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for row in summary:
            print(f"{row['model']}: {row['correct_winners']}/{row['races']} winners ({row['winner_accuracy']:.1%})")
        print(f"Backtest cutoff lead: {lead_seconds / 3600:.6f} hours before qualifying")
        print("Historical publication vintages are unverified; see winner-backtest-audit.json.")
    return forecast_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race", help="Forecast an upcoming Grand Prix using its published race roster")
    parser.add_argument("--backtest", action="store_true", help="Also compare model and last-three-finish baseline")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lead-hours", type=float, default=192.0,
                        help="Backtest-only lead before qualifying; with --race the actual live lead is used")
    args = parser.parse_args()
    try:
        run(args.race, args.backtest, args.data, args.output, args.dashboard, args.cache, args.lead_hours)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Pre-qualifying error: {error}\n")


if __name__ == "__main__":
    main()
