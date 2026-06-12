import os

import numpy as np

from .lotery_data import (
    build_temporal_dataset,
    csv_to_matrices,
    csv_to_numbers,
    numbers_to_matrix,
    print_temporal_window_diagnostics,
    split_train_validation,
)
from .lotery_evaluation import evaluate_model, hybrid_scores, probabilities_to_suggestion
from .lotery_metadata import (
    build_training_metadata,
    get_final_loss,
    load_json,
    model_history_file,
    model_metadata_file,
    save_json,
    configure_tensorflow_logging,
    set_random_seed,
)
from .lotery_config import (
    DEFAULT_EPOCHS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_VALIDATION_SIZE,
    DEFAULT_WINDOW_SIZE,
)


def build_model(window_size):
    from keras.layers import Input, Conv3D, Flatten, Dense
    from keras.models import Sequential

    model = Sequential()
    model.add(Input(shape=(window_size, 5, 5, 1)))
    model.add(Conv3D(32, kernel_size=(min(3, window_size), 3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(Dense(25, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def load_history(model_file_w_format):
    history_file = model_history_file(model_file_w_format)
    if not os.path.exists(history_file):
        raise FileNotFoundError(f"Arquivo {history_file} não encontrado. Treine o modelo novamente.")
    return load_json(history_file)


def load_metadata(model_file_w_format):
    metadata_file = model_metadata_file(model_file_w_format)
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Arquivo {metadata_file} não encontrado. Treine o modelo novamente.")
    return load_json(metadata_file)


def metadata_window_size(metadata):
    parameters = metadata.get("parameters", {})
    return int(parameters.get("window_size", metadata.get("window_size")))


def train_model(
    csv_file,
    model_file=None,
    window_size=DEFAULT_WINDOW_SIZE,
    epochs=DEFAULT_EPOCHS,
    validation_size=DEFAULT_VALIDATION_SIZE,
    seed=None,
    tf_verbose=False,
):
    configure_tensorflow_logging(tf_verbose)
    set_random_seed(seed, tf_verbose)

    from keras.models import load_model

    if not model_file:
        model_file = ""

    matrices = csv_to_matrices(csv_file)
    print_temporal_window_diagnostics(len(matrices), window_size)
    games, targets = build_temporal_dataset(matrices, window_size)
    train_games, train_targets, validation_games, validation_targets = split_train_validation(
        games,
        targets,
        validation_size,
    )

    model_file_w_format = model_file + ".keras"

    metadata_file = model_metadata_file(model_file_w_format)
    can_reuse_model = False
    if model_file_w_format and os.path.exists(model_file_w_format) and os.path.exists(metadata_file):
        metadata = load_json(metadata_file)
        can_reuse_model = metadata_window_size(metadata) == window_size

    if can_reuse_model:
        model = load_model(model_file_w_format)
        print(f"Carregando modelo temporal existente de {model_file_w_format}")
    else:
        model = build_model(window_size)
        if os.path.exists(model_file_w_format):
            print("Modelo existente incompatível com o treinamento temporal atual. Construindo um novo modelo.")
        else:
            print("Construindo um novo modelo")

    print(f"Treinando o modelo com {len(train_games)} pares temporais")
    training_history = model.fit(train_games, train_targets, epochs=epochs, batch_size=16, verbose=0)
    final_loss = get_final_loss(training_history)
    validation_metrics = evaluate_model(model, validation_games, validation_targets)
    if validation_metrics:
        print(
            "Validação: média de "
            f"{validation_metrics['average_hits']:.2f} acertos em 15 números "
            f"({validation_metrics['validation_samples']} amostras)"
        )

    if model_file_w_format:
        model.save(model_file_w_format)
        print(f"Modelo salvo em {model_file_w_format}")

        history = csv_to_numbers(csv_file)
        save_json(history, model_history_file(model_file_w_format))
        save_json(
            build_training_metadata(
                window_size=window_size,
                epochs=epochs,
                seed=seed,
                history_size=len(history),
                temporal_pairs=len(games),
                training_pairs=len(train_games),
                validation_pairs=0 if validation_games is None else len(validation_games),
                validation_metrics=validation_metrics,
                final_loss=final_loss,
            ),
            model_metadata_file(model_file_w_format),
        )
        print(f"Histórico de jogos salvo em {model_history_file(model_file_w_format)}")
    else:
        print("Treinamento concluído, mas o modelo não foi salvo.")


def generate_unique_suggestions(
    model_file,
    count,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    tf_verbose=False,
    hybrid=False,
    model_weight=0.5,
    recent_weight=0.2,
    recency_weight=0.2,
    delay_weight=0.1,
    recent_window=50,
    decay=0.95,
):
    configure_tensorflow_logging(tf_verbose)

    from keras.models import load_model

    if not model_file:
        raise ValueError("Forneça o nome do modelo de teste sem extensão.")

    model_file_w_format = model_file + ".keras"
    if not os.path.exists(model_file_w_format):
        raise FileNotFoundError(f"O arquivo {model_file_w_format} não existe.")

    model = load_model(model_file_w_format)
    print(f"Modelo carregado de '{model_file_w_format}'")

    previous_games = load_history(model_file_w_format)
    metadata = load_metadata(model_file_w_format)
    window_size = metadata_window_size(metadata)
    if hybrid:
        print(
            "Geração híbrida ativa: "
            f"modelo={model_weight}, recente={recent_weight}, "
            f"recência={recency_weight}, atraso={delay_weight}."
        )

    unique_suggestions = []
    rolling_history = list(previous_games)
    attempts = 0

    while len(unique_suggestions) < count and attempts < max_attempts:
        attempts += 1
        recent_games = rolling_history[-window_size:]
        model_input = np.array([numbers_to_matrix(game) for game in recent_games]).reshape(1, window_size, 5, 5, 1)

        predicted = model.predict(model_input, verbose=0)
        scores = predicted[0]
        if hybrid:
            scores = hybrid_scores(
                predicted[0],
                rolling_history,
                model_weight=model_weight,
                recent_weight=recent_weight,
                recency_weight=recency_weight,
                delay_weight=delay_weight,
                recent_window=recent_window,
                decay=decay,
            )
        suggested_numbers = probabilities_to_suggestion(scores)

        if suggested_numbers not in previous_games and suggested_numbers not in unique_suggestions:
            unique_suggestions.append(suggested_numbers)
            rolling_history.append(suggested_numbers)
        else:
            print("Sugestão repetida. Gerando uma nova.")

    if len(unique_suggestions) < count:
        raise RuntimeError(
            f"Não foi possível gerar {count} sugestões únicas em {max_attempts} tentativas. "
            "Tente treinar novamente o modelo ou aumentar --max-attempts."
        )

    return unique_suggestions
