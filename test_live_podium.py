import copy
import io
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from lxml import html
from rich.console import Console

import live_podium as live
from practice_pace import meeting_key, timestamp
from race_data import DATA_PATH, load_data
from terminal_dashboard import display_race


def table(columns, rows):
    return html.fromstring("<html><table><thead><tr>" + "".join(f"<th>{c}</th>" for c in columns)
                           + "</tr></thead><tbody>" + "".join(
                               "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
                           + "</tbody></table></html>")


class LivePodiumTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "snapshots"
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.history = load_data()
        self.sessions = []
        for i, race in enumerate(self.history):
            start = timestamp("2026-03-01T12:00:00Z") + timedelta(days=i * 7)
            self.sessions.append({"session_key": 100 + i, "meeting_key": meeting_key(race),
                                  "year": 2026, "session_name": "Race", "session_type": "Race",
                                  "date_start": start.isoformat(), "date_end": (start + timedelta(hours=2)).isoformat()})
        self.sessions.append({"session_key": 11420, "meeting_key": 1300, "year": 2026,
                              "session_name": "Race", "session_type": "Race",
                              "date_start": "2026-11-22T04:00:00Z", "date_end": "2026-11-22T06:00:00Z"})
        (self.cache / "sessions-2026.json").write_text(json.dumps({"data": self.sessions}), encoding="utf-8")
        self.now = timestamp("2026-11-21T12:00:00Z")
        self.path = "/en/results/2026/races/1300/las-vegas/race-result"
        self.base = self.path.rsplit("/", 1)[0]
        index = table(["Grand Prix", "Date", "Winner"], [
            [f'<a href="{race["source"].removeprefix(live.SITE)}">{race["race"]}</a>', "date", "winner"]
            for race in self.history])
        index.append(html.fromstring(f'<a href="{self.path}">Las Vegas</a>'))
        entries = sorted(self.history[-1]["entries"], key=lambda e: e["grid"])
        self.grid = table(["Pos.", "No.", "Driver", "Team", "Time"], [
            [e["grid"], e["number"], e["driver"], e["team"], "1:20.000"] for e in entries])
        qualifying = table(["Pos.", "No.", "Driver", "Team", "Q1"], [
            [e["grid"], e["number"], e["driver"], e["team"], f'1:{20+e["grid"] / 10:.3f}'] for e in entries])
        self.pages = {live.INDEX: index, self.base + "/starting-grid": self.grid,
                      self.base + "/qualifying": qualifying}
        self.events = []
        self.start_patch("live_podium.utc_now", return_value=self.now)
        self.official = self.start_patch("live_podium.official_page", side_effect=self.official_page)
        self.api = self.start_patch("live_podium.api_read", side_effect=self.api_read)
        self.cache_update = self.start_patch("live_podium.cache_practice")

    def start_patch(self, name, **kwargs):
        patcher = patch(name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def official_page(self, path):
        return self.pages[path], {"url": live.SITE + path, "fetched_at_utc": self.now.isoformat(), "sha256": "fixture"}

    def api_read(self, endpoint, params):
        return (self.sessions if endpoint == "sessions" else self.events), {"url": "fixture"}

    def do_capture(self):
        return live.capture("las-vegas", DATA_PATH, self.cache, self.output)

    def test_capture_freezes_three_sets_inputs_settings_and_cutoffs(self):
        before = DATA_PATH.read_bytes()
        path, snapshot = self.do_capture()
        self.assertTrue(path.exists())
        self.assertIn("20261121T120000000000Z", path.name)
        self.assertEqual(json.loads(path.read_text()), snapshot)
        self.assertEqual(set(snapshot["picks"]), set(live.SETS))
        self.assertTrue(all(len(picks) == 3 for picks in snapshot["picks"].values()))
        self.assertEqual(snapshot["model_settings"]["estimator_parameters"]["C"], 1.0)
        self.assertEqual(snapshot["model_settings"]["estimator_parameters"]["max_iter"], 1000)
        self.assertFalse(snapshot["model_settings"]["recent_movement"])
        self.assertEqual(snapshot["forecast_cutoff_utc"], self.now.isoformat())
        self.assertEqual(snapshot["practice_coverage"]["driver_entries"], 22)
        self.assertTrue(all(r["actual_position"] is None for r in snapshot["forecasts"]["practice_on"]["rows"]))
        self.assertEqual([p["grid_position"] for p in snapshot["picks"]["grid"]], [1, 2, 3])
        self.assertEqual(snapshot["forecasts"]["practice_off"]["training_races"], 20)
        self.assertEqual(DATA_PATH.read_bytes(), before)
        self.assertEqual(self.api.call_count, 4)  # Initial and final fresh schedule/control checks.

    def test_no_grid_writes_no_snapshot_or_model_run(self):
        self.pages[self.base + "/starting-grid"] = html.fromstring("<html>Not published</html>")
        with patch("predict.predict_races") as predict:
            with self.assertRaisesRegex(ValueError, "grid is unavailable"):
                self.do_capture()
            predict.assert_not_called()
        self.cache_update.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unpublished_http_grid_and_completed_event_are_rejected(self):
        with patch("live_podium.official_page", side_effect=HTTPError("grid", 404, "Not Found", {}, None)):
            with self.assertRaisesRegex(ValueError, "grid is unavailable"):
                live.fetch_grid(self.base + "/starting-grid")
        with self.assertRaisesRegex(ValueError, "already published"):
            live.capture("mexico", DATA_PATH, self.cache, self.output)
        self.assertFalse(self.output.exists())

    def test_started_race_and_unverifiable_sources_write_nothing(self):
        for case in ("clock", "race_control", "network"):
            with self.subTest(case=case):
                self.events = []
                if case == "race_control":
                    self.events = [{"session_key": 11420, "date": self.now.isoformat(),
                                    "category": "SessionStatus", "message": "SESSION STARTED"}]
                if case == "network":
                    self.api.side_effect = OSError("source unavailable")
                with patch("live_podium.utc_now", return_value=timestamp("2026-11-23T12:00:00Z") if case == "clock" else self.now):
                    with self.assertRaises((ValueError, OSError)):
                        self.do_capture()
                self.assertFalse(self.output.exists())

    def test_deadline_crossed_during_capture_writes_nothing(self):
        # Four clock reads precede computation; the final live check crosses race start.
        with patch("live_podium.utc_now", side_effect=[self.now] * 4 + [timestamp("2026-11-23T12:00:00Z")]):
            with self.assertRaisesRegex(ValueError, "scheduled start"):
                self.do_capture()
        self.assertFalse(self.output.exists())

    def test_grid_changed_during_capture_writes_nothing(self):
        calls = 0

        def changing(path):
            nonlocal calls
            page, source = self.official_page(path)
            if path.endswith("/starting-grid"):
                calls += 1
                if calls == 2:
                    page = copy.deepcopy(page)
                    page.xpath("//tbody/tr/td[3]")[0].text = "CHANGED"
            return page, source

        self.official.side_effect = changing
        with self.assertRaisesRegex(ValueError, "grid changed"):
            self.do_capture()
        self.assertFalse(self.output.exists())

    def minimal_snapshot(self):
        def picks(numbers):
            return [{"pick_rank": i, "number": str(n), "driver": f"D{n}"} for i, n in enumerate(numbers, 1)]

        return {"schema_version": 1, "season": 2026, "race": "las-vegas",
                "official_result_url": live.SITE + self.path,
                "captured_at_utc": self.now.isoformat(), "forecast_cutoff_utc": self.now.isoformat(),
                "race_start_utc": "2026-11-22T04:00:00Z",
                "picks": {"practice_on": picks([1, 2, 3]), "practice_off": picks([1, 3, 4]), "grid": picks([4, 2, 3])}}

    def save_fixture(self):
        path = self.root / "original.snapshot.json"
        live.save_new(path, self.minimal_snapshot())
        return path

    def publish_result(self):
        index = self.pages[live.INDEX]
        index.xpath("//tbody")[0].append(html.fromstring(f'<tr><td><a href="{self.path}">Las Vegas</a></td></tr>'))
        self.pages[self.path] = table(["Pos.", "No.", "Driver", "Team", "Time / Retired"],
                                      [[i, number, f"D{number}", "A", "time"] for i, number in enumerate((3, 1, 4), 1)])

    def test_scores_original_picks_any_order_without_model_or_snapshot_write(self):
        path = self.save_fixture()
        original = path.read_bytes()
        self.publish_result()
        with patch("predict.predict_races", side_effect=AssertionError("Must never rerun")), patch(
                "live_podium.utc_now", return_value=timestamp("2026-11-23T12:00:00Z")):
            report_path, report = live.score_snapshot(path)
        self.assertEqual([r["podium_hits"] for r in report["scores"]], [2, 3, 2])
        self.assertEqual(report["snapshot_sha256"], live.sha256(original))
        self.assertEqual(path.read_bytes(), original)
        self.assertNotEqual(report_path, path)
        self.assertTrue(report_path.exists())
        self.api.assert_not_called()
        self.cache_update.assert_not_called()

    def test_unpublished_or_incomplete_result_keeps_snapshot_unchanged(self):
        path = self.save_fixture()
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "not yet published"):
            live.score_snapshot(path)
        self.publish_result()
        page = self.pages[self.path]
        page.xpath("//tbody")[0].remove(page.xpath("//tbody/tr")[2])
        with self.assertRaisesRegex(ValueError, "complete distinct"):
            live.score_snapshot(path)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((path.parent / "scores").exists())

    def test_exclusive_writes_and_snapshot_validation(self):
        path = self.save_fixture()
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            live.save_new(path, {"replacement": True})
        self.assertEqual(path.read_bytes(), original)
        snapshot = self.minimal_snapshot()
        snapshot["captured_at_utc"] = "2026-11-23T12:00:00Z"
        with self.assertRaisesRegex(ValueError, "before its race start"):
            live.validate_snapshot(snapshot)
        snapshot = self.minimal_snapshot()
        snapshot["picks"]["grid"][0]["number"] = snapshot["picks"]["grid"][1]["number"]
        with self.assertRaisesRegex(ValueError, "distinct driver"):
            live.validate_snapshot(snapshot)


class DashboardPickLabelTests(unittest.TestCase):
    def test_podium_labels_and_ridge_labels_are_separate(self):
        rows = [{"predicted_position": i, "driver": f"D{i}", "team": "A", "grid_position": i,
                 "actual_position": i, "podium_score": 0.5} for i in range(1, 4)]
        original = copy.deepcopy(rows)
        stream = io.StringIO()
        display_race(Console(file=stream, width=120), "mexico", rows, "Podium")
        text = stream.getvalue()
        for label in ("Pick 1", "Pick 2", "Pick 3", "Pick rank", "Podium picks 3/3", "Podium score"):
            self.assertIn(label, text)
        self.assertNotIn("Winner", text)
        self.assertEqual(rows, original)  # Rendering must not alter CSV fields or values.
        for row in rows:
            row.pop("podium_score")
        stream = io.StringIO()
        display_race(Console(file=stream, width=120), "mexico", rows, "Ridge")
        text = stream.getvalue()
        for label in ("P1", "P2", "P3", "Pred.", "Winner Correct"):
            self.assertIn(label, text)
        self.assertNotIn("Pick rank", text)


if __name__ == "__main__":
    unittest.main()
