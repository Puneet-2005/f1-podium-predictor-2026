"""Compare chronological model replays with the starting-grid baseline."""

import argparse
import hashlib
import json
from math import isclose
from pathlib import Path
from statistics import mean

from predict import predict_races, write_csv
from race_data import DATA_PATH, load_data


def score(rows, position_key):
    matched = [row for row in rows if row["actual_position"] is not None]
    ordered = sorted(rows, key=lambda row: row[position_key])
    return {
        "mae": mean(abs(row[position_key] - row["actual_position"]) for row in matched),
        "winner_correct": int(ordered[0]["actual_position"] == 1),
        "podium_hits": sum(row["actual_position"] in (1, 2, 3) for row in ordered[:3]),
    }


def compare(races, recent_movement=False):
    experimental = "ridge-recent-movement" if recent_movement else "grid-delta"
    replays = {
        "ridge": list(predict_races(races)),
        experimental: list(predict_races(races, "ridge", True) if recent_movement
                           else predict_races(races, "grid-delta")),
    }
    details = []
    for ridge, delta in zip(replays["ridge"], replays[experimental]):
        name, method, count, rows = ridge
        assert (name, count) == (delta[0], delta[2])
        if method == "grid" or not any(row["actual_position"] is not None for row in rows):
            continue
        scores = {
            "ridge": score(rows, "predicted_position"),
            experimental: score(delta[3], "predicted_position"),
            "grid": score(rows, "grid_position"),
        }
        best = min(result["mae"] for result in scores.values())
        for model, result in scores.items():
            details.append({"race": name, "model": model, "training_races": count,
                            **result, "mae_win": int(isclose(result["mae"], best, abs_tol=1e-12))})
    summaries = []
    for scope in ("with-sao-paulo", "without-sao-paulo"):
        for model in ("ridge", experimental, "grid"):
            selected = [row for row in details if row["model"] == model
                        and (scope == "with-sao-paulo" or row["race"] not in {"brazil", "sao-paulo"})]
            if not selected:
                continue
            summaries.append({
                "scope": scope, "model": model, "races": len(selected),
                "mean_mae": mean(row["mae"] for row in selected),
                "mae_wins_including_ties": sum(row["mae_win"] for row in selected),
                "winner_correct": sum(row["winner_correct"] for row in selected),
                "winner_accuracy": mean(row["winner_correct"] for row in selected),
                "mean_podium_hits_out_of_3": mean(row["podium_hits"] for row in selected),
            })
    return details, summaries


def compare_podium(races):
    """Score the same completed post-warm-up rounds for every ranking."""
    replays = {
        "podium": list(predict_races(races, "podium")),
        "ridge-recent-movement-grid-0.5": list(predict_races(
            races, "ridge", recent_movement=True, grid_weight=0.5)),
    }
    details = []
    for podium, blend in zip(replays["podium"], replays["ridge-recent-movement-grid-0.5"]):
        name, method, count, rows = podium
        assert (name, count) == (blend[0], blend[2])
        if method == "grid" or not any(row["actual_position"] is not None for row in rows):
            continue
        for model, predictions, key in (
                ("podium", rows, "predicted_position"),
                ("grid", rows, "grid_position"),
                ("ridge-recent-movement-grid-0.5", blend[3], "predicted_position")):
            result = score(predictions, key)
            picks = sorted(predictions, key=lambda row: row[key])[:3]
            details.append({"race": name, "model": model, "training_races": count,
                            "podium_picks": " / ".join(row["driver"] for row in picks),
                            **result})
    summaries = []
    for model in (*replays, "grid"):
        selected = [row for row in details if row["model"] == model]
        if selected:
            hits = sum(row["podium_hits"] for row in selected)
            summaries.append({"model": model, "races": len(selected),
                              "correct_podium_picks": hits, "total_picks": 3 * len(selected),
                              "podium_pick_accuracy": hits / (3 * len(selected)),
                              "winner_correct": sum(row["winner_correct"] for row in selected),
                              "mean_rank_mae": mean(row["mae"] for row in selected)})
    return details, summaries


def compare_practice_pace(races, practice_cache=None):
    """Paired chronological comparison; missing coverage never changes the race set."""
    off = list(predict_races(races, "podium"))
    on = list(predict_races(races, "podium", practice_pace=True, practice_cache=practice_cache))
    details, coverage = [], []
    for baseline, experiment in zip(off, on):
        name, method, count, off_rows = baseline
        assert (name, count) == (experiment[0], experiment[2])
        on_rows = experiment[3]
        scored = method != "grid" and any(r["actual_position"] is not None for r in off_rows)
        for row in on_rows:
            coverage.append({"race": name, "scored": int(scored), "driver": row["driver"],
                             **{key: value for key, value in row.items() if key.startswith("practice_")}})
        if not scored:
            continue
        off_picks = [row["driver"] for row in off_rows[:3]]
        on_picks = [row["driver"] for row in on_rows[:3]]
        grid = sorted(off_rows, key=lambda row: row["grid_position"])[:3]
        details.append({
            "race": name, "training_races": count,
            "podium_off_hits": score(off_rows, "predicted_position")["podium_hits"],
            "podium_on_hits": score(on_rows, "predicted_position")["podium_hits"],
            "grid_hits": sum(row["actual_position"] in (1, 2, 3) for row in grid),
            "podium_off_picks": " / ".join(off_picks), "podium_on_picks": " / ".join(on_picks),
            "grid_picks": " / ".join(row["driver"] for row in grid),
            "added_picks": " / ".join(d for d in on_picks if d not in off_picks),
            "removed_picks": " / ".join(d for d in off_picks if d not in on_picks),
            "pick_set_changed": int(set(off_picks) != set(on_picks)),
            "pick_order_changed": int(off_picks != on_picks),
            "drivers_with_pace": sum(not row["practice_missing"] for row in on_rows),
            "drivers": len(on_rows),
            "practice_samples": sum(row["practice_sample_count"] for row in on_rows),
            "forecast_cutoff_utc": on_rows[0]["practice_cutoff_utc"],
        })
    summaries = []
    for model in ("podium_off", "podium_on", "grid"):
        for scope in ("all-post-warmup-races", "races-with-practice-coverage"):
            selected = details if scope == "all-post-warmup-races" else [r for r in details if r["drivers_with_pace"]]
            drivers = sum(r["drivers"] for r in selected)
            covered = sum(r["drivers_with_pace"] for r in selected)
            summaries.append({"scope": scope, "model": model, "races": len(selected),
                              "correct_podium_picks": sum(r[f"{model}_hits"] for r in selected),
                              "total_picks": 3 * len(selected), "drivers_with_pace": covered,
                              "driver_entries": drivers, "coverage_fraction": covered / drivers if drivers else 0,
                              "races_with_pace": sum(bool(r["drivers_with_pace"]) for r in selected),
                              "races_with_changed_picks": sum(r["pick_set_changed"] for r in selected)})
    return details, summaries, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=Path("predictions/2026/comparison"))
    parser.add_argument("--recent-movement", action="store_true",
                        help="Compare Ridge with and without recent movement against the grid")
    parser.add_argument("--podium", action="store_true",
                        help="Compare logistic podium scores, starting grid, and the 50/50 recent-movement Ridge blend")
    parser.add_argument("--practice-pace", action="store_true",
                        help="With --podium, compare practice pace off/on and starting grid on identical races")
    parser.add_argument("--practice-cache", type=Path, help="OpenF1 cache folder")
    args = parser.parse_args()
    if args.practice_pace and not args.podium:
        parser.error("--practice-pace requires --podium")
    if args.practice_pace:
        from practice_pace import DEFAULT_CACHE, MAX_LAP_RATIO, MIN_LAPS, MIN_PEERS, source_coverage

        races = load_data(args.data)
        details, summaries, coverage = compare_practice_pace(races, args.practice_cache)
        write_csv(args.output / "coverage.csv", coverage)
        cache = args.practice_cache if args.practice_cache is not None else DEFAULT_CACHE
        sources = source_coverage(races, cache)
        if sources:
            write_csv(args.output / "sources.csv", sources)
        manifest = {
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "min_consecutive_clean_laps": MIN_LAPS, "min_other_drivers_per_group": MIN_PEERS,
            "max_lap_ratio_to_stint_fastest": MAX_LAP_RATIO,
            "cutoff": "Grand Prix session start, or earlier forecast_cutoff_utc if provided",
            "pace_gap": "Median of driver median minus median of other driver medians within session/compound, seconds",
            "added_model_inputs": ["practice_pace_gap", "practice_sample_count", "practice_missing"],
            "source_note": "Historical reconstruction from current OpenF1 archives, not contemporaneous snapshots; later source corrections cannot be ruled out.",
            "cache_sha256": {str(p.relative_to(cache)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(cache.rglob("*.json")) if p.name != "download-errors.json"},
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for row in details:
            print(f"{row['race']}: off {row['podium_off_hits']}/3, on {row['podium_on_hits']}/3, "
                  f"grid {row['grid_hits']}/3 | pace {row['drivers_with_pace']}/{row['drivers']} | "
                  f"added: {row['added_picks'] or '-'}; removed: {row['removed_picks'] or '-'}")
    elif args.podium:
        details, summaries = compare_podium(load_data(args.data))
    else:
        details, summaries = compare(load_data(args.data), args.recent_movement)
    if not summaries:
        parser.error("No completed races after the training warm-up")
    write_csv(args.output / "per-race.csv", details)
    write_csv(args.output / "summary.csv", summaries)
    for row in summaries:
        if args.practice_pace:
            print(f"{row['scope']} {row['model']}: {row['correct_podium_picks']}/{row['total_picks']} | "
                  f"coverage {row['drivers_with_pace']}/{row['driver_entries']} driver entries")
            continue
        if args.podium:
            print(f"{row['model']}: {row['correct_podium_picks']}/{row['total_picks']} correct podium picks across {row['races']} races")
            continue
        print(f"{row['scope']:18} {row['model']:10} n={row['races']} "
              f"MAE={row['mean_mae']:.4f} wins={row['mae_wins_including_ties']} "
              f"winner={row['winner_correct']}/{row['races']} "
              f"podium={row['mean_podium_hits_out_of_3']:.4f}/3")
    print(f"Reports: {args.output.resolve()}")


if __name__ == "__main__":
    main()
