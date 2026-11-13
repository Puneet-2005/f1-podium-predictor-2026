import copy
import unittest
from unittest.mock import patch

from compare_models import compare, score
from predict import build_features, predict_races
from race_data import DATA_PATH, load_data


class PredictionTests(unittest.TestCase):
    def test_podium_scores_and_chronology(self):
        races = load_data(DATA_PATH)
        original = list(predict_races(races, "podium"))
        changed = copy.deepcopy(races)
        changed[5]["results"] = {key: 23 - value for key, value in changed[5]["results"].items()}
        altered = list(predict_races(changed[:6], "podium"))
        self.assertEqual(
            [(row["driver"], row["podium_score"]) for row in original[5][3]],
            [(row["driver"], row["podium_score"]) for row in altered[5][3]])
        for _, method, _, rows in original:
            scores = [row["podium_score"] for row in rows]
            if method == "podium":
                self.assertTrue(all(0 <= score <= 1 for score in scores))
                self.assertEqual(scores, sorted(scores, reverse=True))
            else:
                self.assertEqual(scores, [None] * len(rows))

    def test_podium_binary_targets(self):
        races = load_data(DATA_PATH)[:4]
        captured = {}

        class FakeModel:
            def fit(self, features, labels):
                captured["labels"] = labels[:]

            def predict_proba(self, features):
                import numpy as np
                return np.array([[0.75, 0.25] for _ in features])

        with patch("predict.make_pipeline", return_value=FakeModel()):
            replay = list(predict_races(races, "podium"))
        self.assertEqual(captured["labels"], [
            int(race["results"][entry["number"]] <= 3)
            for race in races[:3] for entry in race["entries"]
            if entry["number"] in race["results"]])
        self.assertTrue(all(row["podium_score"] == 0.25 for row in replay[-1][3]))

    def test_grid_weight_endpoints_and_validation(self):
        races = load_data(DATA_PATH)
        for movement in (False, True):
            self.assertEqual(list(predict_races(races, recent_movement=movement)),
                             list(predict_races(races, recent_movement=movement, grid_weight=0)))
            for _, _, _, rows in predict_races(races, recent_movement=movement, grid_weight=1):
                self.assertEqual([r["grid_position"] for r in rows],
                                 sorted(r["grid_position"] for r in rows))
            original = list(predict_races(races, recent_movement=movement, grid_weight=0.5))
            changed = copy.deepcopy(races)
            changed[5]["results"] = {k: 23 - v for k, v in changed[5]["results"].items()}
            altered = list(predict_races(changed, recent_movement=movement, grid_weight=0.5))
            self.assertEqual([r["driver"] for r in original[5][3]],
                             [r["driver"] for r in altered[5][3]])
        for weight in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                list(predict_races(races, grid_weight=weight))
        self.assertEqual(list(predict_races(races, "grid-delta")),
                         list(predict_races(races, "grid-delta", grid_weight=0)))
        with self.assertRaises(ValueError):
            list(predict_races(races, "grid-delta", grid_weight=0.5))

    def test_recent_movement_history_and_inputs(self):
        races = []
        for index, (grid, finish) in enumerate(zip((10, 1, 4, 9, 1, 4), (5, 2, None, 7, 3, 9))):
            races.append({"race": str(index), "entries": [
                {"number": "1", "driver": "ONE", "team": "A", "grid": grid, "q1_gap_pct": 0.0}],
                "results": {} if finish is None else {"1": finish}})
        races[-1]["entries"].append(
            {"number": "2", "driver": "NEW", "team": "A", "grid": 2, "q1_gap_pct": None})
        captured = []

        def inspect_features(race, driver_form, team_form, movement=None):
            rows = build_features(race, driver_form, team_form, movement)
            base = build_features(race, driver_form, team_form)
            for row, original in zip(rows, base):
                # Compare missing values by representation as NaN is not equal to itself.
                self.assertEqual(repr(row[:4]), repr(original))
            captured.append(rows)
            return rows

        with patch("predict.build_features", side_effect=inspect_features):
            list(predict_races(races, recent_movement=True))
        self.assertEqual([rows[0][-1] for rows in captured], [0, 5, 2, 2, 2, -1 / 3])
        self.assertEqual(captured[-1][1][-1], 0)

    def test_default_and_chronology(self):
        races = load_data(DATA_PATH)
        self.assertEqual(list(predict_races(races)), list(predict_races(races, "ridge")))
        changed = copy.deepcopy(races)
        changed[5]["results"] = {key: 23 - value for key, value in changed[5]["results"].items()}
        for model, movement in (("ridge", False), ("grid-delta", False), ("ridge", True)):
            original = list(predict_races(races[:6], model, movement))
            altered = list(predict_races(changed[:6], model, movement))
            for replay in (original, altered):
                for _, _, _, rows in replay:
                    for row in rows:
                        row.pop("actual_position")
            self.assertEqual(original, altered)

    def test_delta_targets_and_grid_addition(self):
        races = load_data(DATA_PATH)[:4]
        captured = {}

        class FakeModel:
            def fit(self, features, labels):
                captured["labels"] = labels[:]

            def predict(self, features):
                # Adding these deltas preserves grid order; sorting deltas reverses it.
                return [-0.5 * entry["grid"] for entry in races[3]["entries"]]

        with patch("predict.make_pipeline", return_value=FakeModel()):
            replay = list(predict_races(races, "grid-delta"))
        expected = [race["results"][entry["number"]] - entry["grid"]
                    for race in races[:3] for entry in race["entries"]
                    if entry["number"] in race["results"]]
        self.assertEqual(captured["labels"], expected)
        self.assertEqual([row["grid_position"] for row in replay[-1][3]],
                         sorted(entry["grid"] for entry in races[3]["entries"]))

    def test_comparison_scopes_and_metrics(self):
        details, summaries = compare(load_data(DATA_PATH))
        self.assertTrue(all(row["training_races"] >= 3 for row in details))
        for model in ("ridge", "grid-delta", "grid"):
            scopes = [row for row in summaries if row["model"] == model]
            self.assertEqual(scopes[0]["races"] - scopes[1]["races"], 1)
        rows = [{"predicted_position": i, "actual_position": actual}
                for i, actual in enumerate((2, 1, 4, 3), 1)]
        self.assertEqual(score(rows, "predicted_position"),
                         {"mae": 1, "winner_correct": 0, "podium_hits": 2})


if __name__ == "__main__":
    unittest.main()
