"""Show an existing F1 predictor CSV alongside FIA car disclosures."""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DEFAULT_FIA_DATA = Path(__file__).resolve().parent / "data" / "fia_upgrades_2026.json"
REQUIRED_COLUMNS = {
    "predicted_position", "driver", "team", "grid_position", "actual_position"
}


def load_predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("Select a single-race prediction CSV, such as brazil.csv.")
        rows = list(reader)
    if not rows:
        raise ValueError("The prediction CSV has no driver rows.")
    for row in rows:
        row["predicted_position"] = int(row["predicted_position"])
        row["grid_position"] = int(row["grid_position"])
        row["actual_position"] = int(row["actual_position"]) if row["actual_position"] else None
        if "podium_score" in row:
            row["podium_score"] = float(row["podium_score"]) if row["podium_score"] else None
        if "practice_pace_gap" in row:
            row["practice_pace_gap"] = float(row["practice_pace_gap"]) if row["practice_pace_gap"] else None
            row["practice_sample_count"] = int(row["practice_sample_count"])
            row["practice_missing"] = int(row["practice_missing"])
    return sorted(rows, key=lambda row: row["predicted_position"])


def display_race(console, race, rows, model_label):
    is_podium = "podium_score" in rows[0]
    title = f"F1 2026 | {race.replace('-', ' ').upper()}"
    pick_label = "Pick " if is_podium else "P"
    podium = "  |  ".join(f"{pick_label}{row['predicted_position']} {row['driver']}" for row in rows[:3])
    details = [Text(model_label, style="dim"), Text(podium, style="bold yellow")]
    matched = [row for row in rows if row["actual_position"] is not None]
    if matched:
        podium_matches = sum(
            row["actual_position"] <= 3
            for row in rows[:3]
            if row["actual_position"] is not None
        )
        winner = "Correct" if rows[0]["actual_position"] == 1 else "Missed"
        model_mae = mean(abs(r["predicted_position"] - r["actual_position"]) for r in matched)
        grid_mae = mean(abs(r["grid_position"] - r["actual_position"]) for r in matched)
        badge = f"Podium picks {podium_matches}/3"
        if not is_podium:
            badge += f"  |  Winner {winner}"
        details.append(Text(badge))
        details.append(Text(f"MAE {model_mae:.2f} positions  |  Starting grid {grid_mae:.2f}"))
    else:
        details.append(Text("Race result pending", style="dim"))
    console.print(Panel(Group(*details), title=Text(title), border_style="red"))

    if is_podium:
        console.print("Podium scores are uncalibrated model scores; the top three are podium picks.", style="dim")
    table = Table(box=box.SIMPLE, header_style="bold cyan")
    columns = ["Pick rank" if is_podium else "Pred.", "Driver", "Team", "Grid", "Actual", "Error"]
    if is_podium:
        columns.append("Podium score")
    for column in columns:
        table.add_column(column, justify="right" if column in {"Pred.", "Pick rank", "Grid", "Actual", "Error"} else "left")
    for row in rows:
        pred, actual = row["predicted_position"], row["actual_position"]
        table.add_row(
            str(pred), Text(row["driver"]), Text(row["team"]), str(row["grid_position"]),
            str(actual) if actual is not None else "-",
            str(abs(pred - actual)) if actual is not None else "-",
            *([f"{row['podium_score']:.4f}" if row["podium_score"] is not None else "-"]
              if is_podium else []),
            style="bold yellow" if pred <= 3 else None,
        )
    console.print(table)
    if "practice_pace_gap" in rows[0]:
        covered = sum(not row["practice_missing"] for row in rows)
        console.print(Text(f"Practice pace: {covered}/{len(rows)} drivers | "
                           f"cutoff UTC: {rows[0]['practice_cutoff_utc'] or 'unavailable'}"))
        console.print("Gap in seconds to other drivers on the same compound/session; negative is faster.", style="dim")
        practice = Table(box=box.SIMPLE, header_style="bold cyan")
        for column in ("Driver", "Practice gap (s)", "Samples", "Practice status"):
            practice.add_column(column)
        for row in rows:
            gap = row["practice_pace_gap"]
            practice.add_row(Text(row["driver"]), f"{gap:+.3f}" if gap is not None else "missing",
                             str(row["practice_sample_count"]), Text(row["practice_status"]))
        console.print(practice)


def display_upgrades(console, race, data):
    document = next((item for item in data["documents"] if item["race"] == race), None)
    if document is None:
        console.print("FIA car disclosures: no document indexed for this race.", style="dim")
        return

    header = Text(f"FIA car disclosures | Doc {document['document']} | {document['document_date']}")
    header.stylize("bold cyan")
    console.print(header)
    console.print(Text(f"PDF time as printed: {document['document_time_printed']}"))
    published = document["published_at_utc"] or "not recorded"
    console.print(Text(f"FIA publication (UTC): {published}"))
    console.print(Text("Open source PDF", style=f"link {document['source_url']}"))
    upgrades = [item for item in data["upgrades"] if item["race"] == race]
    if race not in data["fully_transcribed_races"]:
        console.print("This round is indexed. Component transcription is pending.", style="dim")
        if not upgrades:
            return

    counts = Counter(item["team"] for item in upgrades)
    console.print(Text("Reported components by team: " + ", ".join(
        f"{team} {count}" for team, count in counts.items()
    )))
    table = Table(box=box.SIMPLE, header_style="bold cyan")
    for column in ("Team", "Reported component", "FIA reason", "PDF page"):
        table.add_column(column)
    for item in upgrades:
        table.add_row(item["team"], item["component"], item["category"], str(item["pdf_page"]))
    console.print(table)
    console.print("Team declarations describe intended changes. Race fitment is unverified.", style="dim")


def render_dashboard(race, rows, model_label, fia_data=DEFAULT_FIA_DATA):
    """Render predictions from the current run without reading a saved CSV."""
    with fia_data.open(encoding="utf-8") as file:
        upgrades = json.load(file)
    console = Console()
    display_race(console, race, rows, model_label)
    display_upgrades(console, race, upgrades)


def display_winner_forecast(forecast):
    """Display a winner forecast without post-grid or podium-membership labels."""
    console = Console()
    rows = forecast["rankings"]
    labels = ("Predicted winner", "Contender 2", "Contender 3")
    picks = [Text(f"{label}: {row['full_name']} ({row['driver']}) | score {row['winner_score']:.4f}",
                  style="bold yellow" if index == 0 else "bold")
             for index, (label, row) in enumerate(zip(labels, rows[:3]))]
    console.print(Panel(Group(
        Text(forecast["race"].replace("-", " ").upper(), style="bold"),
        Text(f"Cutoff UTC: {forecast['forecast_cutoff_utc']}"),
        Text(f"Qualifying starts UTC: {forecast['qualifying_start_utc']}"),
        *picks,
        Text(f"Trained on {len(forecast['training_races'])} earlier races / {forecast['training_driver_rows']} driver rows"),
        Text("Model scores are uncalibrated and do not sum to a race-wide probability.", style="dim")),
        title="Pre-qualifying forecast", border_style="red"))
    table = Table(box=box.SIMPLE, header_style="bold cyan")
    for column in ("Rank", "Driver", "Team", "Model score", "Last 3 finish", "Prior Q1 gap %"):
        table.add_column(column)
    for rank, row in enumerate(rows, 1):
        finish, gap = row["driver_finish_last3"], row["driver_q1_gap_last3"]
        table.add_row(str(rank), Text(row["driver"]), Text(row["team"]), f"{row['winner_score']:.4f}",
                      f"{finish:.2f}" if finish is not None else "missing",
                      f"{gap:.3f}" if gap is not None else "missing",
                      style="bold yellow" if rank == 1 else None)
    console.print(table)
    console.print(Text(f"Baseline (best last-three average finish): {forecast['baseline_pick']['full_name']}"))
    console.print("Inputs: earlier-round results and qualifying only. Entry roster observed before cutoff.", style="dim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_csv", type=Path)
    parser.add_argument("--label", default="Historical replay / race prediction")
    parser.add_argument("--fia-data", type=Path, default=DEFAULT_FIA_DATA)
    args = parser.parse_args()
    race = args.prediction_csv.stem
    if not re.fullmatch(r"[a-z0-9-]+", race):
        parser.error("Name the prediction file after its race slug, such as brazil.csv.")
    try:
        rows = load_predictions(args.prediction_csv)
        with args.fia_data.open(encoding="utf-8") as file:
            upgrades = json.load(file)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Dashboard error: {exc}\n")

    console = Console()
    display_race(console, race, rows, args.label)
    display_upgrades(console, race, upgrades)


if __name__ == "__main__":
    main()
