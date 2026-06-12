import unittest
import json
import os
import tempfile
from contextlib import redirect_stdout
from io import StringIO

from lotery_neural import lotery_cli


class CliTests(unittest.TestCase):
    def test_main_returns_error_for_invalid_count_without_traceback(self):
        output = StringIO()

        with redirect_stdout(output):
            result = lotery_cli.main(["generate", "model", "--count", "0"])

        self.assertEqual(result, 1)
        self.assertIn("Erro:", output.getvalue())

    def test_save_report_writes_json_file(self):
        with tempfile.NamedTemporaryFile("r", delete=False, suffix=".json") as report_file:
            report_path = report_file.name

        try:
            output = StringIO()

            with redirect_stdout(output):
                lotery_cli.save_report({"hello": "world"}, report_path)

            with open(report_path, "r", encoding="utf-8") as file:
                self.assertEqual(json.load(file), {"hello": "world"})
        finally:
            os.unlink(report_path)

    def test_compare_backtest_reports_identifies_best_baseline(self):
        baseline_report = {
            "results": {
                "random": {"average_hits": 9.0, "samples": 100},
                "most_frequent": {"average_hits": 9.2, "samples": 100},
            }
        }
        neural_report = {
            "results": {
                "neural": {"average_hits": 9.5, "samples": 20}
            }
        }

        comparison = lotery_cli.compare_backtest_reports(baseline_report, neural_report)

        self.assertEqual(comparison["best_baseline"], "most_frequent")
        self.assertAlmostEqual(comparison["difference"], 0.3)

    def test_summarize_against_baseline_handles_hybrid_report(self):
        baseline_report = {
            "results": {
                "random": {"average_hits": 9.0, "samples": 100},
                "most_frequent": {"average_hits": 9.2, "samples": 100},
            }
        }
        hybrid_report = {
            "type": "hybrid_backtest",
            "results": {
                "neural": {"average_hits": 9.1, "samples": 20},
                "hybrid": {"average_hits": 9.6, "samples": 20},
            },
        }

        summary = lotery_cli.summarize_against_baseline(baseline_report, hybrid_report)

        self.assertEqual(summary["best_baseline"], "most_frequent")
        self.assertEqual(summary["best_evaluation"], "híbrido")
        self.assertEqual(summary["best_overall"], "híbrido")
        self.assertTrue(summary["hybrid_outperformed_baseline"])
        self.assertTrue(summary["hybrid_outperformed_neural"])
        self.assertAlmostEqual(summary["hybrid_difference_from_neural"], 0.5)

    def test_summarize_against_baseline_handles_neural_report(self):
        baseline_report = {
            "results": {
                "random": {"average_hits": 9.0, "samples": 100},
            }
        }
        neural_report = {
            "type": "neural_backtest",
            "results": {
                "neural": {"average_hits": 8.9, "samples": 20},
            },
        }

        summary = lotery_cli.summarize_against_baseline(baseline_report, neural_report)

        self.assertEqual(summary["best_overall"], "baseline:random")
        self.assertIsNone(summary["hybrid_outperformed_baseline"])
        self.assertIsNone(summary["hybrid_outperformed_neural"])

    def test_summarize_multiple_against_baseline_ranks_reports_together(self):
        baseline_report = {
            "results": {
                "random": {"average_hits": 9.0, "samples": 100},
            }
        }
        neural_report = {
            "type": "neural_backtest",
            "results": {
                "neural": {"average_hits": 9.1, "samples": 20},
            },
        }
        hybrid_report = {
            "type": "hybrid_backtest",
            "results": {
                "neural": {"average_hits": 8.8, "samples": 20},
                "hybrid": {"average_hits": 9.4, "samples": 20},
            },
        }

        summary = lotery_cli.summarize_multiple_against_baseline(
            baseline_report,
            [
                ("reports-neural.json", neural_report),
                ("reports-hybrid.json", hybrid_report),
            ],
        )

        self.assertEqual(summary["best_overall"], "híbrido")
        self.assertEqual(summary["ranking"][0]["average_hits"], 9.4)
        self.assertEqual(summary["ranking"][1]["label"], "neural")
        self.assertEqual(summary["ranking"][2]["label"], "baseline:random")


if __name__ == "__main__":
    unittest.main()
