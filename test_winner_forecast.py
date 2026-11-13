import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
from lxml import html
from rich.console import Console

import winner_forecast as winner
from practice_pace import timestamp
from terminal_dashboard import display_winner_forecast


class WinnerForecastTests(unittest.TestCase):
    def setUp(self):
        self.races, self.sessions, self.rosters = [], [], {}
        for index in range(5):
            start = timestamp("2026-03-08T12:00:00Z") + timedelta(days=14 * index)
            name = f"race-{index}"
            entries = [{"number": str(d), "driver": f"D{d}", "full_name": f"Driver {d}",
                        "team": "A" if d <= 2 else "B", "grid": 5 - d, "q1_gap_pct": d / 10 + index / 100}
                       for d in range(1, 5)]
            result = {str(d): (d - index) % 4 + 1 for d in range(1, 5)}
            self.races.append({"race": name, "source": f"https://www.formula1.com/en/results/2026/races/{100+index}/{name}/race-result",
                               "entries": entries, "results": result})
            self.rosters[name] = [{k: e[k] for k in ("number", "driver", "full_name", "team")} for e in entries]
            for session_type, offset in (("Race", 0), ("Qualifying", -1)):
                session_start = start + timedelta(days=offset)
                self.sessions.append({"meeting_key": 100 + index, "session_key": 1000 + index * 2 + (offset == -1),
                                      "year": 2026, "session_name": session_type, "session_type": session_type,
                                      "date_start": session_start.isoformat(),
                                      "date_end": (session_start + timedelta(hours=2)).isoformat()})
        self.times = winner.schedule(self.races, self.sessions)
        self.lead = 7 * 24 * 3600

    def test_same_relative_cutoff_and_one_binary_label_per_entry(self):
        examples = winner.examples(self.races, self.rosters, self.times, self.lead)
        for race in examples:
            self.assertEqual(len(race["labels"]), 4)
            self.assertEqual(sum(race["labels"]), 1)
            q = timestamp(self.times[race["race"]]["qualifying"]["date_start"])
            self.assertEqual((q - race["cutoff"]).total_seconds(), self.lead)
            for row in race["rows"]:
                for feature in winner.FEATURES:
                    self.assertLessEqual(len(row["evidence"][feature]), 3)
                    for sample in row["evidence"][feature]:
                        self.assertLess(timestamp(sample["available_after_event_utc"]), race["cutoff"])

    def test_current_and_future_grid_qualifying_and_results_are_excluded(self):
        target = {**self.races[3], "entries": self.rosters["race-3"]}
        cutoff = timestamp(self.times["race-3"]["qualifying"]["date_start"]) - timedelta(seconds=self.lead)
        expected = winner.feature_rows(target, cutoff, self.races, self.times)
        changed = copy.deepcopy(self.races)
        for race in changed[3:]:
            race["results"] = {str(d): 5 - d for d in range(1, 5)}
            for entry in race["entries"]:
                entry["grid"] = 999
                entry["q1_gap_pct"] = 999
        self.assertEqual(expected, winner.feature_rows(target, cutoff, changed, self.times))
        self.assertTrue(all("grid" not in row and "q1_gap_pct" not in row for row in expected))

    def test_earlier_round_result_not_yet_finished_is_excluded(self):
        target = {**self.races[3], "entries": self.rosters["race-3"]}
        cutoff = timestamp(self.times["race-2"]["race"]["date_start"]) + timedelta(hours=1)
        rows = winner.feature_rows(target, cutoff, self.races, self.times)
        for row in rows:
            self.assertNotIn("race-2", [s["source_race"] for s in row["evidence"]["driver_finish_last3"]])
            self.assertIn("race-2", [s["source_race"] for s in row["evidence"]["driver_q1_gap_last3"]])

    def test_fit_uses_only_available_prior_labels_and_train_rows(self):
        training = winner.examples(self.races, self.rosters, self.times, self.lead)
        captured = {}

        class FakeModel:
            def fit(self, features, labels):
                captured["features"], captured["labels"] = features, labels

            def predict_proba(self, features):
                return np.array([[0.9 - i / 10, 0.1 + i / 10] for i in range(len(features))])

        with patch("winner_forecast.make_pipeline", return_value=FakeModel()):
            ranked, eligible = winner.forecast(training[3]["rows"], training[3]["cutoff"], training)
        self.assertEqual(len(eligible), 3)
        self.assertEqual(len(captured["labels"]), 12)
        self.assertEqual(sum(captured["labels"]), 3)
        self.assertEqual(ranked[0]["number"], "4")
        self.assertTrue(all(r["result_available"] < training[3]["cutoff"] for r in eligible))

    def test_baseline_is_last_three_average_with_number_tie_break(self):
        rows = [{"number": "9", "driver_finish_last3": 2.0},
                {"number": "2", "driver_finish_last3": 2.0},
                {"number": "1", "driver_finish_last3": None}]
        self.assertEqual(winner.baseline_pick(rows)["number"], "2")
        training = winner.examples(self.races, self.rosters, self.times, self.lead)
        details, summary = winner.backtest(training)
        self.assertEqual(len(details), 2)
        self.assertEqual([row["races"] for row in summary], [2, 2])
        self.assertTrue(all(row["feature_event_times_verified"] for row in details))
        self.assertTrue(all(not row["historical_publication_times_verified"] for row in details))

    def test_published_roster_identity_and_qualifying_deadline(self):
        gp = self.times["race-4"]["race"]
        raw = [{"session_key": gp["session_key"], "meeting_key": gp["meeting_key"], "driver_number": d,
                "name_acronym": f"D{d}", "team_name": "A"} for d in range(1, 5)]
        self.assertEqual(len(winner.entry_roster(raw, gp)), 4)
        raw[0]["session_key"] = -1
        with self.assertRaisesRegex(ValueError, "different Grand Prix"):
            winner.entry_roster(raw, gp)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            winner.entry_roster([], gp)
        deadline = timestamp(self.times["race-4"]["qualifying"]["date_start"])
        with self.assertRaisesRegex(ValueError, "already reached"):
            winner.require_prequal(self.races[4], self.sessions, deadline)

    def test_existing_predict_cli_routes_winner_without_postgrid_loading(self):
        from predict import main

        with patch("sys.argv", ["predict.py", "--race", "las-vegas", "--model", "winner", "--dashboard"]), patch(
                "winner_forecast.run") as run, patch("predict.load_data", side_effect=AssertionError("Post-grid loader used")):
            main()
        self.assertEqual(run.call_args.args[:2], ("las-vegas", False))
        self.assertTrue(run.call_args.args[4])

    def test_live_run_without_grid_uses_observed_roster_and_shared_backtest_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.json"
            data.write_text(json.dumps(self.races[:4]), encoding="utf-8")
            target = self.races[4]
            gp = self.times[target["race"]]["race"]
            now = timestamp(self.times[target["race"]]["qualifying"]["date_start"]) - timedelta(seconds=self.lead)

            def roster(session):
                i = (session - 1000) // 2
                return [{"meeting_key": 100 + i, "session_key": session, "driver_number": int(e["number"]),
                         "name_acronym": e["driver"], "full_name": e["full_name"], "team_name": e["team"]}
                        for e in self.rosters[f"race-{i}"]]

            def cached(cache, endpoint, params, refresh=False):
                return self.sessions if endpoint == "sessions" else roster(params["session_key"])

            def api(endpoint, params):
                rows = self.sessions if endpoint == "sessions" else roster(params["session_key"])
                return rows, {"fetched_at_utc": now.isoformat(), "url": "fixture"}

            links = {r["race"]: r["source"].replace("https://www.formula1.com", "") for r in self.races}
            with patch("winner_forecast.utc_now", return_value=now), patch("winner_forecast.download", side_effect=cached), patch(
                    "winner_forecast.api_read", side_effect=api), patch("winner_forecast.official_page", return_value=(None, {})), patch(
                    "winner_forecast.race_links", return_value=({k: v for k, v in links.items() if k != "race-4"}, links)):
                with redirect_stdout(io.StringIO()):
                    forecast = winner.run("race-4", True, data, root / "output", cache=root / "cache")
            self.assertEqual(forecast["lead_seconds_before_qualifying"], self.lead)
            self.assertEqual(forecast["forecast_cutoff_utc"], now.isoformat())
            self.assertTrue(forecast["live_entry_list_available_at_cutoff"])
            self.assertEqual(len(forecast["rankings"]), 4)
            self.assertEqual(forecast["training_driver_rows"], 16)
            self.assertTrue((root / "output/winner-backtest-audit.json").exists())
            stream = io.StringIO()
            with patch("terminal_dashboard.Console", return_value=Console(file=stream, width=120)):
                display_winner_forecast(forecast)
            for label in ("Pre-qualifying forecast", "Predicted winner", "Contender 2", "Contender 3", "Cutoff UTC", "Model score"):
                self.assertIn(label, stream.getvalue())


if __name__ == "__main__":
    unittest.main()
