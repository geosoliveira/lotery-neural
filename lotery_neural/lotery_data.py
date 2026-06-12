import numpy as np
import pandas as pd

MIN_RECOMMENDED_TEMPORAL_PAIRS = 50
MAX_RECOMMENDED_WINDOW_RATIO = 0.8


def matrix_to_numbers(matrix):
    numbers = []
    for i in range(5):
        for j in range(5):
            if matrix[i, j] == 1:
                numbers.append(i * 5 + j + 1)
    return numbers


def numbers_to_matrix(numbers):
    matrix = np.zeros((5, 5), dtype=int)
    for num in numbers:
        row_idx = (num - 1) // 5
        col_idx = (num - 1) % 5
        matrix[row_idx, col_idx] = 1
    return matrix


def csv_to_matrices(csv_file):
    data = pd.read_csv(csv_file, sep=';', header=None)
    matrices = []
    for index, row in data.iterrows():
        numbers = [int(num) for num in row.dropna()]
        if len(numbers) != 15:
            raise ValueError(f"Linha {index + 1}: esperado 15 números, encontrado {len(numbers)}.")
        if len(set(numbers)) != 15:
            raise ValueError(f"Linha {index + 1}: existem números duplicados.")

        matrix = np.zeros((5, 5), dtype=int)
        for num in numbers:
            if num < 1 or num > 25:
                raise ValueError(f"Linha {index + 1}: número fora do intervalo 1..25: {num}.")
            row_idx = (num - 1) // 5
            col_idx = (num - 1) % 5
            matrix[row_idx, col_idx] = 1
        matrices.append(matrix)
    return matrices


def csv_to_numbers(csv_file):
    return [matrix_to_numbers(matrix) for matrix in csv_to_matrices(csv_file)]


def describe_temporal_window(total_games, window_size):
    if window_size <= 0:
        raise ValueError("O tamanho da janela deve ser um inteiro positivo.")
    if total_games <= window_size:
        raise ValueError(
            f"Quantidade insuficiente de jogos: {total_games}. "
            f"Para window_size={window_size}, informe pelo menos {window_size + 1} jogos."
        )

    temporal_pairs = total_games - window_size
    window_ratio = window_size / total_games
    warnings = []

    if temporal_pairs < MIN_RECOMMENDED_TEMPORAL_PAIRS:
        warnings.append(
            "a janela escolhida gera poucos pares temporais; "
            "isso pode deixar o treino e as métricas pouco confiáveis"
        )
    if window_ratio > MAX_RECOMMENDED_WINDOW_RATIO:
        warnings.append(
            "a janela consome uma parte muito grande do histórico; "
            "janelas menores geralmente geram mais exemplos de treino"
        )

    return {
        "total_games": total_games,
        "window_size": window_size,
        "temporal_pairs": temporal_pairs,
        "window_ratio": window_ratio,
        "warnings": warnings,
    }


def print_temporal_window_diagnostics(total_games, window_size):
    diagnostics = describe_temporal_window(total_games, window_size)
    print(
        "Diagnóstico da janela temporal: "
        f"{diagnostics['total_games']} concursos, "
        f"window-size={diagnostics['window_size']}, "
        f"{diagnostics['temporal_pairs']} pares temporais."
    )
    for warning in diagnostics["warnings"]:
        print(f"Aviso: {warning}.")
    return diagnostics


def build_temporal_dataset(matrices, window_size):
    describe_temporal_window(len(matrices), window_size)

    x = []
    y = []
    for index in range(window_size, len(matrices)):
        x.append(matrices[index - window_size:index])
        y.append(matrices[index].reshape(25))

    return np.array(x).reshape(-1, window_size, 5, 5, 1), np.array(y)


def split_train_validation(games, targets, validation_size):
    if validation_size <= 0 or len(games) <= validation_size:
        return games, targets, None, None

    split_index = len(games) - validation_size
    return games[:split_index], targets[:split_index], games[split_index:], targets[split_index:]
