import numpy as np

from .lotery_data import matrix_to_numbers


def probabilities_to_suggestion(probabilities):
    predicted_matrix = probabilities.reshape(5, 5)
    suggested_positions = np.unravel_index(np.argsort(predicted_matrix, axis=None)[-15:], (5, 5))

    suggested_matrix = np.zeros((5, 5), dtype=int)
    for pos in zip(suggested_positions[0], suggested_positions[1]):
        suggested_matrix[pos] = 1

    return matrix_to_numbers(suggested_matrix)


def normalize_scores(values):
    scores = np.asarray(values, dtype=float).reshape(25)
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score == min_score:
        return np.zeros(25, dtype=float)
    return (scores - min_score) / (max_score - min_score)


def recent_frequency_scores(history, window_size):
    if not history:
        raise ValueError("Histórico vazio; não é possível calcular scores recentes.")

    source = history[-window_size:] if window_size else history
    scores = np.zeros(25, dtype=float)
    for game in source:
        for number in game:
            scores[int(number) - 1] += 1

    return normalize_scores(scores)


def recency_weighted_scores(history, decay=0.95):
    if not history:
        raise ValueError("Histórico vazio; não é possível calcular scores por recência.")
    if not 0 < decay <= 1:
        raise ValueError("--decay deve estar no intervalo 0 < decay <= 1.")

    scores = np.zeros(25, dtype=float)
    total_games = len(history)
    for index, game in enumerate(history):
        weight = decay ** (total_games - index - 1)
        for number in game:
            scores[int(number) - 1] += weight

    return normalize_scores(scores)


def delay_scores(history):
    if not history:
        raise ValueError("Histórico vazio; não é possível calcular scores de atraso.")

    scores = np.zeros(25, dtype=float)
    for number in range(1, 26):
        games_since_last_seen = len(history)
        for offset, game in enumerate(reversed(history)):
            if number in game:
                games_since_last_seen = offset
                break
        scores[number - 1] = games_since_last_seen

    return normalize_scores(scores)


def hybrid_scores(
    model_probabilities,
    history,
    model_weight=0.5,
    recent_weight=0.2,
    recency_weight=0.2,
    delay_weight=0.1,
    recent_window=50,
    decay=0.95,
):
    weights = {
        "model_weight": model_weight,
        "recent_weight": recent_weight,
        "recency_weight": recency_weight,
        "delay_weight": delay_weight,
    }
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Pesos da geração híbrida não podem ser negativos.")

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Informe pelo menos um peso positivo para a geração híbrida.")

    combined = (
        normalize_scores(model_probabilities) * model_weight
        + recent_frequency_scores(history, recent_window) * recent_weight
        + recency_weighted_scores(history, decay) * recency_weight
        + delay_scores(history) * delay_weight
    )
    return normalize_scores(combined)


def evaluate_predictions(predictions, targets):
    hit_counts = []
    for prediction, target in zip(predictions, targets):
        suggested_numbers = probabilities_to_suggestion(prediction)
        target_numbers = matrix_to_numbers(target.reshape(5, 5))
        hit_counts.append(len(set(suggested_numbers) & set(target_numbers)))

    if not hit_counts:
        return None

    return {
        "validation_samples": len(hit_counts),
        "average_hits": float(np.mean(hit_counts)),
        "min_hits": int(np.min(hit_counts)),
        "max_hits": int(np.max(hit_counts)),
    }


def summarize_hit_counts(hit_counts):
    if not hit_counts:
        return None

    return {
        "samples": len(hit_counts),
        "average_hits": float(np.mean(hit_counts)),
        "min_hits": int(np.min(hit_counts)),
        "max_hits": int(np.max(hit_counts)),
    }


def count_hits(suggestion, target):
    return len(set(suggestion) & set(target))


def random_baseline(history, seed=None):
    rng = np.random.default_rng(seed)
    return sorted(int(number) for number in rng.choice(np.arange(1, 26), size=15, replace=False))


def most_frequent_baseline(history):
    return most_frequent_recent_baseline(history, window_size=None)


def most_frequent_recent_baseline(history, window_size):
    if not history:
        raise ValueError("Histórico vazio; não é possível calcular baseline.")

    source = history[-window_size:] if window_size else history
    counts = {number: 0 for number in range(1, 26)}
    for game in source:
        for number in game:
            counts[int(number)] += 1

    ranked_numbers = sorted(counts, key=lambda number: (-counts[number], number))
    return sorted(ranked_numbers[:15])


def recency_weighted_baseline(history, decay=0.95):
    if not history:
        raise ValueError("Histórico vazio; não é possível calcular baseline.")
    if not 0 < decay <= 1:
        raise ValueError("--decay deve estar no intervalo 0 < decay <= 1.")

    scores = {number: 0.0 for number in range(1, 26)}
    total_games = len(history)
    for index, game in enumerate(history):
        weight = decay ** (total_games - index - 1)
        for number in game:
            scores[int(number)] += weight

    ranked_numbers = sorted(scores, key=lambda number: (-scores[number], number))
    return sorted(ranked_numbers[:15])


def evaluate_model(model, validation_games, validation_targets):
    if validation_games is None or validation_targets is None:
        return None

    predictions = model.predict(validation_games, verbose=0)
    return evaluate_predictions(predictions, validation_targets)
