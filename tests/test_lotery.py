import os
import tempfile
import unittest

try:
    import numpy as np
    from lotery_neural import lotery_backtest
    from lotery_neural import lotery_data
    from lotery_neural import lotery_evaluation
    from lotery_neural import lotery_metadata
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Dependência do projeto não instalada: {exc.name}")


def make_game(start):
    numbers = [((start + offset - 1) % 25) + 1 for offset in range(15)]
    return numbers


def make_matrix(numbers):
    return lotery_data.numbers_to_matrix(numbers)


class DataConversionTests(unittest.TestCase):
    def test_csv_to_matrices_validates_game_size(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as csv_file:
            csv_file.write("1;2;3\n")
            csv_path = csv_file.name

        try:
            with self.assertRaisesRegex(ValueError, "esperado 15"):
                lotery_data.csv_to_matrices(csv_path)
        finally:
            os.unlink(csv_path)

    def test_csv_to_matrices_validates_duplicate_numbers(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as csv_file:
            csv_file.write("1;1;3;4;5;6;7;8;9;10;11;12;13;14;15\n")
            csv_path = csv_file.name

        try:
            with self.assertRaisesRegex(ValueError, "duplicados"):
                lotery_data.csv_to_matrices(csv_path)
        finally:
            os.unlink(csv_path)

    def test_csv_to_matrices_validates_number_range(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as csv_file:
            csv_file.write("1;2;3;4;5;6;7;8;9;10;11;12;13;14;26\n")
            csv_path = csv_file.name

        try:
            with self.assertRaisesRegex(ValueError, "fora do intervalo"):
                lotery_data.csv_to_matrices(csv_path)
        finally:
            os.unlink(csv_path)

    def test_matrix_to_numbers_round_trip(self):
        numbers = [1, 2, 3, 7, 8, 9, 11, 12, 15, 18, 19, 20, 21, 24, 25]
        matrix = lotery_data.numbers_to_matrix(numbers)

        self.assertEqual(lotery_data.matrix_to_numbers(matrix), numbers)


class TemporalDatasetTests(unittest.TestCase):
    def test_describe_temporal_window_counts_pairs(self):
        diagnostics = lotery_data.describe_temporal_window(total_games=100, window_size=10)

        self.assertEqual(diagnostics["total_games"], 100)
        self.assertEqual(diagnostics["window_size"], 10)
        self.assertEqual(diagnostics["temporal_pairs"], 90)
        self.assertEqual(diagnostics["warnings"], [])

    def test_describe_temporal_window_warns_when_window_is_too_large(self):
        diagnostics = lotery_data.describe_temporal_window(total_games=20, window_size=18)

        self.assertEqual(diagnostics["temporal_pairs"], 2)
        self.assertGreaterEqual(len(diagnostics["warnings"]), 1)

    def test_build_temporal_dataset_uses_previous_window_as_input(self):
        matrices = [make_matrix(make_game(index)) for index in range(5)]

        games, targets = lotery_data.build_temporal_dataset(matrices, window_size=2)

        self.assertEqual(games.shape, (3, 2, 5, 5, 1))
        self.assertEqual(targets.shape, (3, 25))
        np.testing.assert_array_equal(games[0, 0, :, :, 0], matrices[0])
        np.testing.assert_array_equal(games[0, 1, :, :, 0], matrices[1])
        np.testing.assert_array_equal(targets[0], matrices[2].reshape(25))

    def test_build_temporal_dataset_requires_enough_games(self):
        matrices = [make_matrix(make_game(index)) for index in range(2)]

        with self.assertRaisesRegex(ValueError, "Quantidade insuficiente"):
            lotery_data.build_temporal_dataset(matrices, window_size=2)

    def test_split_train_validation_reserves_last_pairs(self):
        games = np.arange(5)
        targets = np.arange(10, 15)

        train_games, train_targets, validation_games, validation_targets = lotery_data.split_train_validation(
            games,
            targets,
            validation_size=2,
        )

        np.testing.assert_array_equal(train_games, np.array([0, 1, 2]))
        np.testing.assert_array_equal(train_targets, np.array([10, 11, 12]))
        np.testing.assert_array_equal(validation_games, np.array([3, 4]))
        np.testing.assert_array_equal(validation_targets, np.array([13, 14]))


class EvaluationTests(unittest.TestCase):
    def test_probabilities_to_suggestion_picks_top_15_numbers(self):
        probabilities = np.arange(25)

        self.assertEqual(lotery_evaluation.probabilities_to_suggestion(probabilities), list(range(11, 26)))

    def test_evaluate_predictions_counts_hits_in_top_15(self):
        predictions = np.array([
            np.arange(25),
            np.arange(24, -1, -1),
        ])
        targets = np.array([
            make_matrix(list(range(11, 26))).reshape(25),
            make_matrix(list(range(1, 16))).reshape(25),
        ])

        metrics = lotery_evaluation.evaluate_predictions(predictions, targets)

        self.assertEqual(metrics["validation_samples"], 2)
        self.assertEqual(metrics["average_hits"], 15.0)
        self.assertEqual(metrics["min_hits"], 15)
        self.assertEqual(metrics["max_hits"], 15)

    def test_most_frequent_baseline_uses_frequency_then_number_order(self):
        history = [
            list(range(1, 16)),
            list(range(1, 16)),
            list(range(11, 26)),
        ]

        suggestion = lotery_evaluation.most_frequent_baseline(history)

        self.assertEqual(len(suggestion), 15)
        self.assertEqual(suggestion[:10], list(range(1, 11)))

    def test_random_baseline_is_reproducible_with_seed(self):
        first = lotery_evaluation.random_baseline([], seed=123)
        second = lotery_evaluation.random_baseline([], seed=123)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 15)
        self.assertEqual(len(set(first)), 15)

    def test_run_baseline_backtest_returns_metrics_for_each_baseline(self):
        games = [make_game(index) for index in range(8)]

        results = lotery_backtest.run_baseline_backtest(games, window_size=3, seed=123)

        self.assertEqual(set(results), {"random", "most_frequent", "recent_most_frequent", "recency_weighted"})
        self.assertEqual(results["random"]["samples"], 5)
        self.assertIn("average_hits", results["most_frequent"])

    def test_recency_weighted_baseline_prioritizes_recent_games(self):
        history = [
            list(range(1, 16)),
            list(range(11, 26)),
        ]

        suggestion = lotery_evaluation.recency_weighted_baseline(history, decay=0.5)

        self.assertEqual(suggestion, list(range(11, 26)))

    def test_recent_frequency_scores_prioritizes_recent_window(self):
        history = [
            list(range(1, 16)),
            list(range(11, 26)),
        ]

        scores = lotery_evaluation.recent_frequency_scores(history, window_size=1)

        self.assertEqual(scores[10], 1.0)
        self.assertEqual(scores[24], 1.0)
        self.assertEqual(scores[0], 0.0)

    def test_delay_scores_prioritizes_numbers_absent_for_longer(self):
        history = [
            list(range(1, 16)),
            list(range(11, 26)),
        ]

        scores = lotery_evaluation.delay_scores(history)

        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores[10], 0.0)

    def test_hybrid_scores_combines_model_and_history_scores(self):
        history = [
            list(range(1, 16)),
            list(range(11, 26)),
        ]
        model_probabilities = np.zeros(25)
        model_probabilities[0] = 1.0

        scores = lotery_evaluation.hybrid_scores(
            model_probabilities,
            history,
            model_weight=1.0,
            recent_weight=0.0,
            recency_weight=0.0,
            delay_weight=0.0,
        )

        self.assertEqual(scores.shape, (25,))
        self.assertEqual(scores[0], 1.0)

    def test_neural_backtest_indices_uses_recent_steps_when_limited(self):
        indices = lotery_backtest.neural_backtest_indices(
            total_games=12,
            window_size=3,
            min_training_games=5,
            max_steps=2,
        )

        self.assertEqual(indices, [10, 11])

    def test_run_neural_backtest_supports_fake_model_factory(self):
        class FakeModel:
            def fit(self, *_args, **_kwargs):
                return None

            def predict(self, *_args, **_kwargs):
                return np.array([np.arange(25)])

        games = [make_game(index) for index in range(8)]

        result = lotery_backtest.run_neural_backtest(
            games,
            window_size=3,
            epochs=1,
            min_training_games=5,
            max_steps=2,
            seed=123,
            model_factory=lambda _window_size: FakeModel(),
        )

        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["neural"]["samples"], 2)

    def test_run_hybrid_backtest_supports_fake_model_factory(self):
        class FakeModel:
            def fit(self, *_args, **_kwargs):
                return None

            def predict(self, *_args, **_kwargs):
                return np.array([np.arange(25)])

        games = [make_game(index) for index in range(8)]

        result = lotery_backtest.run_hybrid_backtest(
            games,
            window_size=3,
            epochs=1,
            min_training_games=5,
            max_steps=2,
            seed=123,
            model_factory=lambda _window_size: FakeModel(),
            model_weight=1.0,
            recent_weight=0.0,
            recency_weight=0.0,
            delay_weight=0.0,
        )

        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["neural"]["samples"], 2)
        self.assertEqual(result["hybrid"]["samples"], 2)
        self.assertEqual(result["neural"]["average_hits"], result["hybrid"]["average_hits"])


class ReproducibilityAndMetadataTests(unittest.TestCase):
    def test_set_random_seed_makes_numpy_random_reproducible(self):
        lotery_metadata.set_random_seed(123)
        first = np.random.random(3)

        lotery_metadata.set_random_seed(123)
        second = np.random.random(3)

        np.testing.assert_array_equal(first, second)

    def test_get_final_loss_returns_last_loss(self):
        class FakeHistory:
            history = {"loss": [0.9, 0.4, 0.1]}

        self.assertEqual(lotery_metadata.get_final_loss(FakeHistory()), 0.1)

    def test_build_training_metadata_includes_reproducibility_context(self):
        metadata = lotery_metadata.build_training_metadata(
            window_size=10,
            epochs=500,
            seed=123,
            history_size=100,
            temporal_pairs=90,
            training_pairs=70,
            validation_pairs=20,
            validation_metrics={"average_hits": 9.5},
            final_loss=0.123,
        )

        self.assertIn("trained_at", metadata)
        self.assertIn("python_version", metadata)
        self.assertIn("package_versions", metadata)
        self.assertEqual(metadata["parameters"]["seed"], 123)
        self.assertEqual(metadata["parameters"]["window_size"], 10)
        self.assertEqual(metadata["final_loss"], 0.123)
        self.assertEqual(metadata["validation_metrics"]["average_hits"], 9.5)


if __name__ == "__main__":
    unittest.main()
