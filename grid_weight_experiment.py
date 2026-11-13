"""Replay recent-movement Ridge at grid weights 0 and 0.5 against the grid."""

import argparse
from pathlib import Path
from statistics import mean

from compare_models import score
from predict import predict_races, write_csv
from race_data import DATA_PATH, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path,
                        default=Path("predictions/2026/grid-weight-experiment"))
    args = parser.parse_args()
    # A fresh folder protects both previous experiments and existing predictions.
    args.output.mkdir(parents=True, exist_ok=False)
    races = load_data(args.data)
    details = []
    for weight in (0, 0.5):
        model = f"ridge-recent-grid-{weight:g}"
        for name, method, count, rows in predict_races(
                races, recent_movement=True, grid_weight=weight):
            if method == "grid" or not any(r["actual_position"] is not None for r in rows):
                continue
            write_csv(args.output / model / f"{name}.csv", rows)
            details.append({"race": name, "model": model, "training_races": count,
                            **score(rows, "predicted_position")})
            if weight == 0:
                grid_rows = sorted(rows, key=lambda r: r["grid_position"])
                grid_rows = [dict(r, predicted_position=i)
                             for i, r in enumerate(grid_rows, 1)]
                write_csv(args.output / "grid" / f"{name}.csv", grid_rows)
                details.append({"race": name, "model": "grid", "training_races": count,
                                **score(rows, "grid_position")})
    summaries = []
    for model in ("ridge-recent-grid-0", "ridge-recent-grid-0.5", "grid"):
        selected = [r for r in details if r["model"] == model]
        summaries.append({"model": model, "races": len(selected),
                          "podium_hits": sum(r["podium_hits"] for r in selected),
                          "podium_slots": 3 * len(selected),
                          "winner_correct": sum(r["winner_correct"] for r in selected),
                          "mean_mae": mean(r["mae"] for r in selected)})
    write_csv(args.output / "per-race.csv", details)
    write_csv(args.output / "summary.csv", summaries)
    for row in summaries:
        print(row)
    for row in details:
        if row["race"] in {"brazil", "sao-paulo", "mexico"}:
            print(row)
    print(f"Reports: {args.output.resolve()}")


if __name__ == "__main__":
    main()
