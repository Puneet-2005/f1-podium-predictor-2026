"""Predict the 2026 F1 finishing order using only information available before each race."""

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from race_data import DATA_PATH, load_data, update_data

MIN_TRAINING_RACES = 3
FORM_WINDOW = 3


def build_features(race, driver_form, team_form, driver_movement=None):
    field_size = len(race["entries"])
    return [
        [
            (entry["grid"] - 1) / max(field_size - 1, 1),
            entry["q1_gap_pct"] if entry["q1_gap_pct"] is not None else float("nan"),
            mean(driver_form[entry["number"]]) if driver_form[entry["number"]] else 0.5,
            mean(team_form[entry["team"]]) if team_form[entry["team"]] else 0.5,
        ] + ([mean(driver_movement[entry["number"]])
              if driver_movement[entry["number"]] else 0.0]
             if driver_movement is not None else [])
        for entry in race["entries"]
    ]


def predict_races(races, model_name="ridge", recent_movement=False, grid_weight=0.0,
                  practice_pace=False, practice_cache=None):
    if practice_pace and model_name != "podium":
        raise ValueError("Practice pace is supported only for podium")
    if not 0 <= grid_weight <= 1:
        raise ValueError("Grid weight must be between 0 and 1")
    if grid_weight != 0 and model_name != "ridge":
        raise ValueError("Grid weight is supported only for Ridge")
    if model_name not in {"ridge", "grid-delta", "podium"}:
        raise ValueError(f"Unknown model: {model_name}")
    if recent_movement and model_name not in {"ridge", "podium"}:
        raise ValueError("Recent movement is supported only for Ridge and podium")
    driver_form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    team_form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    driver_movement = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    training_rows = []
    training_labels = []
    completed_count = 0

    for race in races:
        entries = race["entries"]
        features = build_features(race, driver_form, team_form,
                                  driver_movement if recent_movement else None)
        if practice_pace:
            from practice_pace import DEFAULT_CACHE, practice_features

            pace = practice_features(race, practice_cache if practice_cache is not None else DEFAULT_CACHE)
            for entry, feature in zip(entries, features):
                item = pace[str(entry["number"])]
                feature.extend([item["practice_pace_gap"] if not item["practice_missing"] else float("nan"),
                                item["practice_sample_count"], item["practice_missing"]])
        if completed_count >= MIN_TRAINING_RACES and (
                model_name != "podium" or len(set(training_labels)) == 2):
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=1000) if model_name == "podium" else Ridge(alpha=10.0),
            )
            model.fit(training_rows, training_labels)
            scores = (-model.predict_proba(features)[:, 1] if model_name == "podium"
                      else model.predict(features))
            if model_name == "grid-delta":
                scores = [entry["grid"] + delta for entry, delta in zip(entries, scores)]
            elif grid_weight:
                scores = [(1 - grid_weight) * ridge_score
                          + grid_weight * (entry["grid"] - 1) / max(len(entries) - 1, 1)
                          for entry, ridge_score in zip(entries, scores)]
            method = model_name
        else:
            scores = [entry["grid"] for entry in entries]
            method = "grid"

        ordered = sorted(zip(entries, scores), key=lambda pair: (pair[1], pair[0]["grid"]))
        actual = race["results"]
        rows = [
            {
                "predicted_position": position,
                "driver": entry["driver"],
                "team": entry["team"],
                "grid_position": entry["grid"],
                "actual_position": actual.get(entry["number"]),
            }
            for position, (entry, _) in enumerate(ordered, 1)
        ]
        if model_name == "podium":
            for row, (_, score) in zip(rows, ordered):
                row["podium_score"] = float(-score) if method == "podium" else None
        if practice_pace:
            for row, (entry, _) in zip(rows, ordered):
                row.update(pace[str(entry["number"])])
        yield race["race"], method, completed_count, rows

        if actual:
            # Add the current race only after producing its prediction.
            team_finishes = defaultdict(list)
            for entry, feature_row in zip(entries, features):
                finish = actual.get(entry["number"])
                if finish is None:
                    continue
                label = (finish - 1) / max(len(actual) - 1, 1)
                training_rows.append(feature_row)
                training_labels.append(int(finish <= 3) if model_name == "podium" else
                                       finish - entry["grid"] if model_name == "grid-delta" else label)
                driver_form[entry["number"]].append(label)
                driver_movement[entry["number"]].append(entry["grid"] - finish)
                team_finishes[entry["team"]].append(label)
            for team, finishes in team_finishes.items():
                team_form[team].append(mean(finishes))
            completed_count += 1


def metrics(rows):
    matched = [row for row in rows if row["actual_position"] is not None]
    if not matched:
        return None
    return (
        mean(abs(row["predicted_position"] - row["actual_position"]) for row in matched),
        mean(abs(row["grid_position"] - row["actual_position"]) for row in matched),
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--race", help="Race slug, e.g. monaco, great-britain or las-vegas")
    mode.add_argument("--backtest", action="store_true", help="Replay all completed races")
    mode.add_argument("--list", action="store_true", help="List races in the current data file")
    parser.add_argument("--update", action="store_true", help="Fetch newly published F1 data")
    parser.add_argument("--dashboard", action="store_true",
                        help="Show the terminal dashboard for a single --race run")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the 2026 data file")
    parser.add_argument("--output", type=Path, default=Path("predictions/2026"))
    parser.add_argument("--model", choices=["ridge", "grid-delta", "podium", "winner"], default="ridge",
                        help="Prediction model (winner uses pre-qualifying inputs; grid-delta is experimental)")
    parser.add_argument("--recent-movement", action="store_true",
                        help="Add a driver's mean grid minus finish over their last 3 results (Ridge or podium)")
    parser.add_argument("--grid-weight", type=float, default=0.0,
                        help="Blend Ridge scores with normalized grid positions (0 to 1; default 0)")
    parser.add_argument("--practice-pace", action="store_true",
                        help="Add cached pre-race long-run pace, sample count and missing indicator (podium only)")
    parser.add_argument("--practice-cache", type=Path,
                        help="OpenF1 cache folder (default: data/openf1 next to predict.py)")
    args = parser.parse_args()
    if args.model == "winner":
        if args.list or args.update or args.practice_pace or args.recent_movement or args.grid_weight != 0 or args.practice_cache:
            parser.error("--model winner supports --race or --backtest, --data, --output and --dashboard only")
        if args.dashboard and not args.race:
            parser.error("--dashboard requires --race")
        from winner_forecast import run

        try:
            run(args.race, args.backtest, args.data, args.output, args.dashboard)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.exit(1, f"Pre-qualifying error: {error}\n")
        return
    if args.dashboard and not args.race:
        parser.error("--dashboard requires --race")
    if args.practice_pace and args.model != "podium":
        parser.error("--practice-pace requires --model podium")
    if not 0 <= args.grid_weight <= 1:
        parser.error("--grid-weight must be between 0 and 1")
    if args.grid_weight != 0 and args.model != "ridge":
        parser.error("--grid-weight requires --model ridge")
    if args.recent_movement and args.model not in {"ridge", "podium"}:
        parser.error("--recent-movement requires --model ridge or podium")
    try:
        races = update_data(args.data, args.race) if args.update else load_data(args.data)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Data error: {error}\n")

    available = {race["race"] for race in races}
    if args.list:
        for race in races:
            print(race["race"], "(completed)" if race["results"] else "(pre-race)")
        return
    if args.race and args.race not in available:
        parser.error(f"No data for {args.race}. Try --list or use --update when its grid is published.")

    summary = []
    for name, method, count, rows in predict_races(races, args.model, args.recent_movement,
                                                args.grid_weight, args.practice_pace, args.practice_cache):
        if args.race and name != args.race:
            continue
        errors = metrics(rows)
        if args.backtest and errors is None:
            continue
        output = args.output / f"{name}.csv"
        write_csv(output, rows)
        podium = " / ".join(row["driver"] for row in rows[:3])
        model_label = {"ridge": "Ridge", "grid-delta": "Grid-delta",
                       "podium": "Logistic regression podium (uncalibrated model scores)"}[args.model]
        if args.recent_movement:
            model_label += " + recent movement"
        if args.practice_pace:
            model_label += " + practice pace"
        if args.model == "ridge":
            model_label += f" (grid weight {args.grid_weight:g})"
        label = f"{model_label} trained on {count} earlier races" if method != "grid" else "grid warm-up"
        print(f"{name}: {podium} ({label})")
        if errors:
            print(f"  MAE in positions: prediction {errors[0]:.2f}, grid {errors[1]:.2f}")
            summary.append({
                "race": name, "method": method, "training_races": count,
                "prediction_mae": round(errors[0], 4), "grid_mae": round(errors[1], 4),
            })
        else:
            print("  Race result pending.")
        if args.race:
            print(f"  Full order: {output}")
            if args.dashboard:
                try:
                    from terminal_dashboard import render_dashboard

                    render_dashboard(name, rows, label)
                except (ImportError, OSError, ValueError, KeyError, TypeError) as error:
                    parser.exit(1, f"Dashboard error: {error}\n")
            break

    if args.backtest and summary:
        write_csv(args.output / "backtest.csv", summary)
        modeled = [row for row in summary if row["method"] != "grid"]
        if modeled:
            prediction_mae = mean(row["prediction_mae"] for row in modeled)
            grid_mae = mean(row["grid_mae"] for row in modeled)
            print(f"\nMean MAE across {len(modeled)} modeled races: "
                  f"{model_label} {prediction_mae:.2f}, grid {grid_mae:.2f} positions.")


if __name__ == "__main__":
    main()
