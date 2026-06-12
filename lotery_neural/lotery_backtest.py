from .lotery_data import csv_to_numbers, print_temporal_window_diagnostics
from .lotery_evaluation import (
    count_hits,
    hybrid_scores,
    most_frequent_baseline,
    most_frequent_recent_baseline,
    probabilities_to_suggestion,
    random_baseline,
    recency_weighted_baseline,
    summarize_hit_counts,
)
from .lotery_metadata import configure_tensorflow_logging, set_random_seed

PROGRESS_INTERVAL = 250


def print_backtest_progress(label, current_step, total_steps):
    if current_step == 1 or current_step == total_steps or current_step % PROGRESS_INTERVAL == 0:
        print(f"{label}: amostra {current_step}/{total_steps}...")


def run_baseline_backtest(games, window_size, seed=None, decay=0.95):
    if window_size <= 0:
        raise ValueError("O tamanho da janela deve ser um inteiro positivo.")
    if len(games) <= window_size:
        raise ValueError(
            f"Quantidade insuficiente de jogos: {len(games)}. "
            f"Para window_size={window_size}, informe pelo menos {window_size + 1} jogos."
        )
    print_temporal_window_diagnostics(len(games), window_size)

    hit_counts = {
        "random": [],
        "most_frequent": [],
        "recent_most_frequent": [],
        "recency_weighted": [],
    }

    total_steps = len(games) - window_size
    for step, index in enumerate(range(window_size, len(games)), 1):
        print_backtest_progress("Backtest de baselines", step, total_steps)
        history = games[:index]
        target = games[index]

        hit_counts["random"].append(count_hits(random_baseline(history, seed=seed + index if seed is not None else None), target))
        hit_counts["most_frequent"].append(count_hits(most_frequent_baseline(history), target))
        hit_counts["recent_most_frequent"].append(
            count_hits(most_frequent_recent_baseline(history, window_size), target)
        )
        hit_counts["recency_weighted"].append(count_hits(recency_weighted_baseline(history, decay), target))

    return {
        name: summarize_hit_counts(counts)
        for name, counts in hit_counts.items()
    }


def run_baseline_backtest_from_csv(csv_file, window_size, seed=None, decay=0.95):
    return run_baseline_backtest(csv_to_numbers(csv_file), window_size, seed, decay)


def default_model_factory(window_size):
    from .lotery_model import build_model

    return build_model(window_size)


def prepare_neural_training_data(games, window_size, end_index):
    import numpy as np

    from .lotery_data import numbers_to_matrix

    x = []
    y = []
    for index in range(window_size, end_index):
        x.append([numbers_to_matrix(game) for game in games[index - window_size:index]])
        y.append(numbers_to_matrix(games[index]).reshape(25))

    return np.array(x).reshape(-1, window_size, 5, 5, 1), np.array(y)


def prepare_neural_prediction_input(games, window_size, target_index):
    import numpy as np

    from .lotery_data import numbers_to_matrix

    recent_games = games[target_index - window_size:target_index]
    return np.array([numbers_to_matrix(game) for game in recent_games]).reshape(1, window_size, 5, 5, 1)


def neural_backtest_indices(total_games, window_size, min_training_games, max_steps=None):
    first_target_index = max(window_size + 1, min_training_games)
    indices = list(range(first_target_index, total_games))
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("--max-steps deve ser um inteiro positivo.")
        indices = indices[-max_steps:]
    return indices


def run_neural_backtest(
    games,
    window_size,
    epochs,
    min_training_games,
    max_steps=None,
    seed=None,
    model_factory=None,
    tf_verbose=False,
):
    if window_size <= 0:
        raise ValueError("O tamanho da janela deve ser um inteiro positivo.")
    if epochs <= 0:
        raise ValueError("--epochs deve ser um inteiro positivo.")
    if min_training_games <= window_size:
        raise ValueError("--min-training-games deve ser maior que --window-size.")
    if len(games) <= min_training_games:
        raise ValueError(
            f"Quantidade insuficiente de jogos: {len(games)}. "
            f"Informe mais de {min_training_games} jogos para executar o backtest neural."
        )
    print_temporal_window_diagnostics(len(games), window_size)
    configure_tensorflow_logging(tf_verbose)

    model_factory = model_factory or default_model_factory
    target_indices = neural_backtest_indices(len(games), window_size, min_training_games, max_steps)
    hit_counts = []

    for step, target_index in enumerate(target_indices):
        set_random_seed(seed + step if seed is not None else None, tf_verbose)
        print(
            f"Backtest neural {step + 1}/{len(target_indices)}: "
            f"preparando treino para concurso índice {target_index}..."
        )

        train_games, train_targets = prepare_neural_training_data(games, window_size, target_index)
        prediction_input = prepare_neural_prediction_input(games, window_size, target_index)

        model = model_factory(window_size)
        model.fit(train_games, train_targets, epochs=epochs, batch_size=16, verbose=0)

        prediction = model.predict(prediction_input, verbose=0)[0]

        suggestion = probabilities_to_suggestion(prediction)
        hit_counts.append(count_hits(suggestion, games[target_index]))
        print(
            f"Backtest neural {step + 1}/{len(target_indices)}: "
            f"concurso índice {target_index}, {hit_counts[-1]} acertos"
        )

    return {
        "neural": summarize_hit_counts(hit_counts),
        "steps": len(target_indices),
        "window_size": window_size,
        "epochs": epochs,
        "min_training_games": min_training_games,
    }


def run_neural_backtest_from_csv(
    csv_file,
    window_size,
    epochs,
    min_training_games,
    max_steps=None,
    seed=None,
    tf_verbose=False,
):
    return run_neural_backtest(
        csv_to_numbers(csv_file),
        window_size=window_size,
        epochs=epochs,
        min_training_games=min_training_games,
        max_steps=max_steps,
        seed=seed,
        tf_verbose=tf_verbose,
    )


def run_hybrid_backtest(
    games,
    window_size,
    epochs,
    min_training_games,
    max_steps=None,
    seed=None,
    model_factory=None,
    tf_verbose=False,
    model_weight=0.5,
    recent_weight=0.2,
    recency_weight=0.2,
    delay_weight=0.1,
    recent_window=50,
    decay=0.95,
):
    if window_size <= 0:
        raise ValueError("O tamanho da janela deve ser um inteiro positivo.")
    if epochs <= 0:
        raise ValueError("--epochs deve ser um inteiro positivo.")
    if min_training_games <= window_size:
        raise ValueError("--min-training-games deve ser maior que --window-size.")
    if len(games) <= min_training_games:
        raise ValueError(
            f"Quantidade insuficiente de jogos: {len(games)}. "
            f"Informe mais de {min_training_games} jogos para executar o backtest híbrido."
        )
    print_temporal_window_diagnostics(len(games), window_size)
    configure_tensorflow_logging(tf_verbose)

    model_factory = model_factory or default_model_factory
    target_indices = neural_backtest_indices(len(games), window_size, min_training_games, max_steps)
    neural_hits = []
    hybrid_hits = []

    for step, target_index in enumerate(target_indices):
        set_random_seed(seed + step if seed is not None else None, tf_verbose)
        print(
            f"Backtest híbrido {step + 1}/{len(target_indices)}: "
            f"preparando treino para concurso índice {target_index}..."
        )

        train_games, train_targets = prepare_neural_training_data(games, window_size, target_index)
        prediction_input = prepare_neural_prediction_input(games, window_size, target_index)
        history = games[:target_index]

        model = model_factory(window_size)
        model.fit(train_games, train_targets, epochs=epochs, batch_size=16, verbose=0)

        prediction = model.predict(prediction_input, verbose=0)[0]
        neural_suggestion = probabilities_to_suggestion(prediction)
        hybrid_suggestion = probabilities_to_suggestion(
            hybrid_scores(
                prediction,
                history,
                model_weight=model_weight,
                recent_weight=recent_weight,
                recency_weight=recency_weight,
                delay_weight=delay_weight,
                recent_window=recent_window,
                decay=decay,
            )
        )

        neural_hits.append(count_hits(neural_suggestion, games[target_index]))
        hybrid_hits.append(count_hits(hybrid_suggestion, games[target_index]))
        print(
            f"Backtest híbrido {step + 1}/{len(target_indices)}: "
            f"concurso índice {target_index}, neural {neural_hits[-1]} acertos, "
            f"híbrido {hybrid_hits[-1]} acertos"
        )

    return {
        "neural": summarize_hit_counts(neural_hits),
        "hybrid": summarize_hit_counts(hybrid_hits),
        "steps": len(target_indices),
        "window_size": window_size,
        "epochs": epochs,
        "min_training_games": min_training_games,
        "hybrid_parameters": {
            "model_weight": model_weight,
            "recent_weight": recent_weight,
            "recency_weight": recency_weight,
            "delay_weight": delay_weight,
            "recent_window": recent_window,
            "decay": decay,
        },
    }


def run_hybrid_backtest_from_csv(
    csv_file,
    window_size,
    epochs,
    min_training_games,
    max_steps=None,
    seed=None,
    tf_verbose=False,
    model_weight=0.5,
    recent_weight=0.2,
    recency_weight=0.2,
    delay_weight=0.1,
    recent_window=50,
    decay=0.95,
):
    return run_hybrid_backtest(
        csv_to_numbers(csv_file),
        window_size=window_size,
        epochs=epochs,
        min_training_games=min_training_games,
        max_steps=max_steps,
        seed=seed,
        tf_verbose=tf_verbose,
        model_weight=model_weight,
        recent_weight=recent_weight,
        recency_weight=recency_weight,
        delay_weight=delay_weight,
        recent_window=recent_window,
        decay=decay,
    )
