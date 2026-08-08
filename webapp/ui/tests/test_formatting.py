"""Tests for the UI's pure display formatting helpers."""

import unittest

from webapp.ui.formatting import history_rows, percentage, position_delta_label, prediction_rows


class FormattingTests(unittest.TestCase):
    def test_percentage_uses_one_decimal_place(self) -> None:
        self.assertEqual(percentage(0.634), "63.4%")

    def test_position_delta_labels_are_friendly_italian_text(self) -> None:
        self.assertEqual(position_delta_label(-2), "↑ 2 meglio")
        self.assertEqual(position_delta_label(3), "↓ 3 peggio")
        self.assertEqual(position_delta_label(0), "= come previsto")
        self.assertEqual(position_delta_label(None), "Non classificato")

    def test_prediction_rows_sort_and_hide_raw_model_scores(self) -> None:
        document = {
            "drivers": [
                {
                    "display_name": "Max Verstappen",
                    "team_name": "Red Bull Racing",
                    "predicted_position": 2,
                    "expected_position": 2.62,
                    "win_probability": 0.291,
                    "podium_probability": 0.681,
                    "points_probability": 0.931,
                    "dnf_probability": 0.071,
                    "raw_xgboost_score": 9.5,
                },
                {
                    "display_name": "Lando Norris",
                    "team_name": "McLaren",
                    "predicted_position": 1,
                    "expected_position": 2.34,
                    "win_probability": 0.354,
                    "podium_probability": 0.704,
                    "points_probability": 0.944,
                    "dnf_probability": 0.064,
                    "raw_xgboost_score": 10.5,
                },
            ]
        }

        self.assertEqual(
            prediction_rows(document),
            [
                {
                    "Posizione": 1,
                    "Pilota": "Lando Norris",
                    "Team": "McLaren",
                    "Vittoria": "35.4%",
                    "Podio": "70.4%",
                    "Punti": "94.4%",
                    "DNF": "6.4%",
                    "Posizione media": "2.3",
                },
                {
                    "Posizione": 2,
                    "Pilota": "Max Verstappen",
                    "Team": "Red Bull Racing",
                    "Vittoria": "29.1%",
                    "Podio": "68.1%",
                    "Punti": "93.1%",
                    "DNF": "7.1%",
                    "Posizione media": "2.6",
                },
            ],
        )

    def test_history_rows_keep_predicted_and_actual_positions_with_label(self) -> None:
        document = {
            "comparisons": [
                {
                    "display_name": "Max Verstappen",
                    "predicted_position": 2,
                    "actual_position": 1,
                    "position_difference": -1,
                },
                {
                    "display_name": "Lewis Hamilton",
                    "predicted_position": 6,
                    "actual_position": None,
                    "position_difference": None,
                },
            ]
        }

        self.assertEqual(
            history_rows(document),
            [
                {"Prevista": 2, "Pilota": "Max Verstappen", "Reale": 1, "Differenza": "↑ 1 meglio"},
                {
                    "Prevista": 6,
                    "Pilota": "Lewis Hamilton",
                    "Reale": None,
                    "Differenza": "Non classificato",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
