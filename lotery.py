from lotery_neural.lotery_cli import main


if __name__ == "__main__":
    raise SystemExit(main())

from lotery_neural.lotery_backtest import (
    neural_backtest_indices,
    prepare_neural_prediction_input,
    prepare_neural_training_data,
    run_baseline_backtest,
    run_baseline_backtest_from_csv,
    run_neural_backtest,
    run_neural_backtest_from_csv,
)
from lotery_neural.lotery_data import (
    build_temporal_dataset,
    csv_to_matrices,
    csv_to_numbers,
    matrix_to_numbers,
    numbers_to_matrix,
    split_train_validation,
)
from lotery_neural.lotery_evaluation import (
    count_hits,
    evaluate_model,
    evaluate_predictions,
    most_frequent_baseline,
    most_frequent_recent_baseline,
    probabilities_to_suggestion,
    random_baseline,
    recency_weighted_baseline,
    summarize_hit_counts,
)
from lotery_neural.lotery_metadata import (
    build_training_metadata,
    get_final_loss,
    get_package_versions,
    load_json,
    model_history_file,
    model_metadata_file,
    save_json,
    set_random_seed,
)
from lotery_neural.lotery_model import (
    DEFAULT_EPOCHS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_VALIDATION_SIZE,
    DEFAULT_WINDOW_SIZE,
    build_model,
    generate_unique_suggestions,
    load_history,
    load_metadata,
    train_model,
)
from lotery_neural.lotery_results import (
    append_latest_game,
    fetch_caixa_contest,
    format_game,
    parse_contest_number,
    parse_contest_numbers,
    sync_games,
)
