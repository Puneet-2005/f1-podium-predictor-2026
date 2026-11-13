import copy
import io
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from compare_models import compare_practice_pace
from practice_pace import ENDPOINTS, practice_features, session_pace, timestamp
from predict import predict_races
from race_data import load_data
from terminal_dashboard import display_race, load_predictions
from predict import write_csv


class PracticePaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name)
        self.session = {"session_key": 1, "meeting_key": 999, "year": 2026,
                        "session_type": "Practice", "session_name": "Practice 1",
                        "date_start": "2026-03-06T10:00:00Z", "date_end": "2026-03-06T11:00:00Z"}
        gp = {**self.session, "session_key": 2, "session_name": "Race", "session_type": "Race",
              "date_start": "2026-03-08T10:00:00Z", "date_end": "2026-03-08T12:00:00Z"}
        self.race = {"race": "fixture", "source": "https://www.formula1.com/en/results/2026/races/999/fixture/race-result",
                     "entries": [{"number": str(d), "driver": str(d)} for d in (1, 2, 3, 4)]}
        self.raw = {endpoint: [] for endpoint in ENDPOINTS}
        self.raw["race_control"] = [{"session_key": 1, "date": self.session["date_start"],
                                    "flag": "GREEN", "scope": "Track", "message": "GREEN LIGHT"}]
        for driver in (1, 2, 3, 4):
            duration = 79.0 + driver
            self.raw["stints"].append({"session_key": 1, "driver_number": driver, "stint_number": 1,
                                       "lap_start": 1, "lap_end": 10,
                                       "compound": "SOFT" if driver == 4 else "MEDIUM"})
            for number in range(1, 11):
                self.raw["laps"].append({
                    "session_key": 1, "driver_number": driver, "lap_number": number,
                    "date_start": (timestamp(self.session["date_start"]) + timedelta(seconds=100 * number)).isoformat(),
                    "lap_duration": duration, "is_pit_out_lap": number == 1,
                    **{f"duration_sector_{i}": duration / 3 for i in (1, 2, 3)},
                    **{f"segments_sector_{i}": [2048, 2049, 2051] for i in (1, 2, 3)}})
        self.store(self.cache / "sessions-2026.json", [self.session, gp])
        self.save_raw()

    def store(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"data": data}), encoding="utf-8")

    def save_raw(self):
        for endpoint, data in self.raw.items():
            self.store(self.cache / "1" / f"{endpoint}.json", data)

    def test_same_session_compound_other_driver_median_and_missing(self):
        features = practice_features(self.race, self.cache)
        for driver, gap in (("1", -1.5), ("2", 0.0), ("3", 1.5)):
            self.assertEqual(features[driver]["practice_pace_gap"], gap)
            self.assertEqual(features[driver]["practice_sample_count"], 8)
            self.assertEqual(features[driver]["practice_missing"], 0)
        self.assertIsNone(features["4"]["practice_pace_gap"])
        self.assertEqual(features["4"]["practice_sample_count"], 0)
        self.assertEqual(features["4"]["practice_status"], "insufficient_samples")

    def test_pit_invalid_deleted_flag_and_cooldown_laps_break_runs(self):
        original = copy.deepcopy(self.raw)
        for case in ("pit_out", "pit_segment", "invalid", "sector_missing", "deleted", "yellow", "vsc", "cooldown", "pit_feed"):
            with self.subTest(case=case):
                self.raw = copy.deepcopy(original)
                lap = next(l for l in self.raw["laps"] if l["driver_number"] == 1 and l["lap_number"] == 4)
                if case == "pit_out":
                    lap["is_pit_out_lap"] = True
                elif case == "pit_segment":
                    lap["segments_sector_3"] = [2064]
                elif case == "invalid":
                    lap["is_valid"] = False
                elif case == "sector_missing":
                    lap["duration_sector_2"] = None
                elif case == "cooldown":
                    lap["lap_duration"] = 120
                    for i in (1, 2, 3):
                        lap[f"duration_sector_{i}"] = 40
                elif case == "pit_feed":
                    self.raw["pit"].append({"session_key": 1, "driver_number": 1, "lap_number": 3,
                                            "date": lap["date_start"]})
                elif case == "deleted":
                    self.raw["race_control"].append({"session_key": 1, "date": "2026-03-06T11:01:00Z",
                                                    "message": "CAR 1 TIME 1:20.000 DELETED LAP 4"})
                else:
                    self.raw["race_control"].extend([
                        {"session_key": 1, "date": lap["date_start"], "category": "SafetyCar" if case == "vsc" else "Flag",
                         "flag": "YELLOW" if case == "yellow" else None,
                         "scope": "Track", "message": "YELLOW" if case == "yellow" else "VSC DEPLOYED"},
                        {"session_key": 1, "date": "2026-03-06T10:08:10Z", "scope": "Track",
                         "flag": "GREEN", "message": "GREEN"}])
                self.save_raw()
                result = practice_features(self.race, self.cache)["1"]
                self.assertEqual(result["practice_sample_count"], 5)

    def test_cutoff_and_future_laps_or_corrections(self):
        baseline = practice_features(self.race, self.cache)
        self.raw["race_control"].append({"session_key": 1, "date": "2026-03-08T11:00:00Z",
                                        "message": "CAR 1 TIME 1:20.000 DELETED LAP 4"})
        future = copy.deepcopy(self.raw["laps"][1])
        future.update(date_start="2026-03-08T12:00:00Z", lap_duration=1)
        self.raw["laps"].append(future)
        self.save_raw()
        self.assertEqual(baseline, practice_features(self.race, self.cache))
        self.race["forecast_cutoff_utc"] = "2026-03-06T10:30:00Z"
        result = practice_features(self.race, self.cache)
        self.assertTrue(all(f["practice_status"] == "no_prerace_practice" for f in result.values()))
        groups, status = session_pace(self.session, self.raw, timestamp("2026-03-06T10:30:00Z"))
        self.assertEqual((groups, status), ({}, "session_after_cutoff"))

    def test_missing_cache_and_flag_source_are_explicit(self):
        self.raw["race_control"] = []
        self.save_raw()
        self.assertIn("missing_session_data", practice_features(self.race, self.cache)["1"]["practice_status"])
        with tempfile.TemporaryDirectory() as empty:
            result = practice_features(self.race, Path(empty))
        self.assertEqual(result["1"]["practice_status"], "cache_missing")

    def test_training_cutoff_and_identical_comparison_races(self):
        races = load_data()[:6]

        def fake_features(race, cache):
            return {str(e["number"]): {"practice_pace_gap": e["grid"] / 10,
                    "practice_sample_count": 5, "practice_missing": 0, "practice_status": "available",
                    "practice_cutoff_utc": "fixture", "practice_groups": "fixture"} for e in race["entries"]}

        with patch("practice_pace.practice_features", side_effect=fake_features):
            original = list(predict_races(races, "podium", practice_pace=True))
            altered = copy.deepcopy(races)
            altered[5]["results"] = {k: 23 - v for k, v in altered[5]["results"].items()}
            changed = list(predict_races(altered, "podium", practice_pace=True))
        self.assertEqual([(r["driver"], r["podium_score"]) for r in original[5][3]],
                         [(r["driver"], r["podium_score"]) for r in changed[5][3]])
        details, summaries, coverage = compare_practice_pace(races, self.cache)
        self.assertEqual(len(details), 3)
        self.assertTrue(all(r["total_picks"] == 9 for r in summaries if r["scope"] == "all-post-warmup-races"))
        self.assertTrue(all(r["pick_set_changed"] == 0 for r in details))
        self.assertTrue(all(r["practice_missing"] == 1 for r in coverage))
        with self.assertRaisesRegex(ValueError, "only for podium"):
            list(predict_races(races, practice_pace=True))

    def test_dashboard_and_csv_practice_fields(self):
        row = {"predicted_position": 1, "driver": "ONE", "team": "A", "grid_position": 1,
               "actual_position": None, "podium_score": 0.5, **practice_features(self.race, self.cache)["1"]}
        path = self.cache / "fixture.csv"
        write_csv(path, [row])
        self.assertEqual(load_predictions(path), [row])
        stream = io.StringIO()
        display_race(Console(file=stream, width=120), "fixture", [row], "Podium + practice")
        text = stream.getvalue()
        for expected in ("Podium score", "Practice gap (s)", "Samples", "-1.500", "8", "available"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
