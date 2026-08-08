"""Tests for the historical comparison page view-model helpers."""

import unittest

from webapp.ui.history import _display_history_rows, history_options, summary_messages


class HistoryViewModelTests(unittest.TestCase):
    def test_history_options_keep_race_name_round_and_publication_type(self) -> None:
        index_document = {
            "season": 2026,
            "races": [
                {
                    "season": 2026,
                    "round": 3,
                    "name": "Japanese Grand Prix",
                    "publication_type": "live",
                },
                {
                    "season": 2026,
                    "round": 1,
                    "name": "Australian Grand Prix",
                    "publication_type": "backtest",
                },
            ],
        }

        self.assertEqual(
            history_options(index_document),
            {
                "Round 3 · Japanese Grand Prix · PREVISIONE LIVE": (2026, 3),
                "Round 1 · Australian Grand Prix · BACKTEST STORICO": (2026, 1),
            },
        )

    def test_summary_messages_explain_historical_accuracy_in_plain_language(self) -> None:
        document = {
            "summary": {
                "mean_absolute_position_error": 2.1,
                "podium_hits": 2,
                "podium_total": 3,
                "top_five_hits": 4,
                "top_five_total": 5,
            }
        }

        self.assertEqual(summary_messages(document), [
            "Errore medio: 2,1 posizioni",
            "Podio previsto: 2 piloti corretti su 3",
            "Top 5 prevista: 4 piloti corretti su 5",
        ])

    def test_history_options_make_publication_badges_explicit(self) -> None:
        backtest = {
            "season": 2026,
            "races": [
                {
                    "season": 2026,
                    "round": 1,
                    "name": "Australian Grand Prix",
                    "publication_type": "backtest",
                }
            ],
        }
        live = {
            "season": 2026,
            "races": [
                {
                    "season": 2026,
                    "round": 3,
                    "name": "Japanese Grand Prix",
                    "publication_type": "live",
                }
            ],
        }

        self.assertIn("BACKTEST STORICO", next(iter(history_options(backtest))))
        self.assertIn("PREVISIONE LIVE", next(iter(history_options(live))))

    def test_history_rows_show_non_classified_status_instead_of_a_fake_position(self) -> None:
        document = {
            "comparisons": [
                {
                    "display_name": "Lewis Hamilton",
                    "predicted_position": 6,
                    "actual_position": None,
                    "position_difference": None,
                    "status": "DNF",
                }
            ]
        }

        self.assertEqual(
            _display_history_rows(document),
            [
                {
                    "Prevista": 6,
                    "Pilota": "Lewis Hamilton",
                    "Reale": "DNF",
                    "Differenza": "Non classificato",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
