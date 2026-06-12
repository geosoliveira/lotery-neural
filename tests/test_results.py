import unittest
import os
import tempfile
from contextlib import redirect_stdout
from io import StringIO

from lotery_neural import lotery_results


class ResultsTests(unittest.TestCase):
    def test_parse_contest_numbers_returns_sorted_integers(self):
        contest = {
            "listaDezenas": ["07", "02", "25", "01", "11", "08", "09", "10", "03", "04", "05", "06", "12", "13", "14"],
        }

        numbers = lotery_results.parse_contest_numbers(contest)

        self.assertEqual(numbers, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 25])

    def test_parse_contest_numbers_validates_amount(self):
        with self.assertRaisesRegex(ValueError, "esperado 15"):
            lotery_results.parse_contest_numbers({"listaDezenas": ["01", "02"]})

    def test_parse_contest_number_requires_numero(self):
        with self.assertRaisesRegex(ValueError, "número do concurso"):
            lotery_results.parse_contest_number({})

    def test_format_game_uses_semicolon_separator(self):
        self.assertEqual(lotery_results.format_game([1, 2, 15, 25]), "1;2;15;25")

    def test_sync_missing_games_appends_only_missing_contests(self):
        contests = {
            None: {"numero": 3, "listaDezenas": [str(number).zfill(2) for number in range(1, 16)]},
            3: {"numero": 3, "listaDezenas": [str(number).zfill(2) for number in range(11, 26)]},
        }
        original_fetch = lotery_results.fetch_caixa_contest

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as csv_file:
            csv_file.write("1;2;3;4;5;6;7;8;9;10;11;12;13;14;15\n")
            csv_file.write("2;3;4;5;6;7;8;9;10;11;12;13;14;15;16\n")
            csv_path = csv_file.name

        try:
            lotery_results.fetch_caixa_contest = lambda contest_number=None: contests[contest_number]
            output = StringIO()

            with redirect_stdout(output):
                result = lotery_results.sync_missing_games(csv_path)

            self.assertTrue(result["updated"])
            self.assertEqual(result["from_contest"], 3)
            self.assertEqual(result["added_games"], 1)
            self.assertEqual(result["total_games"], 3)
            self.assertIn("Buscando a partir do concurso 3", output.getvalue())
        finally:
            lotery_results.fetch_caixa_contest = original_fetch
            os.unlink(csv_path)


if __name__ == "__main__":
    unittest.main()
