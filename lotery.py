import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Flatten, Dense
import argparse
import os

def matrix_to_numbers(matrix):
    numbers = []
    for i in range(5):
        for j in range(5):
            if matrix[i, j] == 1:
                numbers.append(i * 5 + j + 1)
    return numbers

def csv_to_matrices(csv_file):
    data = pd.read_csv(csv_file, sep=';', header=None)
    matrices = []
    for index, row in data.iterrows():
        matrix = np.zeros((5, 5), dtype=int)
        for num in row:
            row_idx = (num - 1) // 5
            col_idx = (num - 1) % 5
            matrix[row_idx, col_idx] = 1
        matrices.append(matrix)
    return matrices

def build_model(input_shape=(5, 5, 1)):
    model = Sequential()
    #model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu', input_shape=input_shape))
    model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(25, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def train_model(csv_file, model_file=None):
    matrices = csv_to_matrices(csv_file)
    games = np.array(matrices).reshape(-1, 5, 5, 1)
    y = games.reshape(-1, 25)
    
    model_file_w_format = model_file + ".keras"
    
    if model_file_w_format and os.path.exists(model_file_w_format):
        model = load_model(model_file_w_format)
        print(f"Carregando modelo existente de {model_file_w_format}")
    else:
        model = build_model()
        print("Construindo um novo modelo")
    
    print("Treinando o modelo")
    model.fit(games, y, epochs=1000, batch_size=16, verbose=0)
    
    if model_file_w_format:
        # Certifique-se de usar uma extensão válida para o arquivo do modelo
        model.save(model_file_w_format)
        print(f"Modelo salvo em {model_file_w_format}")
    else:
        print("Treinamento concluído, mas o modelo não foi salvo.")

def generate_suggestions(model_file, count):
    model_file_w_format = model_file + ".keras"
    if not os.path.exists(model_file_w_format):
        print(f"O arquivo {model_file_w_format} não existe.")
        return

    model = load_model(model_file_w_format)
    print(f"Modelo carregado de '{model_file_w_format}'")
    
    for i in range(count):
        dummy_input = np.zeros((1, 5, 5, 1)) if i == 0 else np.random.random((1, 5, 5, 1))  # Gerando uma matriz 5x5 com valores aleatórios a partir do segundo jogo
        
        predicted = model.predict(dummy_input)
        predicted_matrix = predicted.reshape(5, 5)
        
        suggested_positions = np.unravel_index(np.argsort(predicted_matrix, axis=None)[-15:], (5, 5))

        suggested_matrix = np.zeros((5, 5))
        for pos in zip(suggested_positions[0], suggested_positions[1]):
            suggested_matrix[pos] = 1

        suggested_numbers = matrix_to_numbers(suggested_matrix)
        print(f"Números sugeridos {i + 1}:")
        print(suggested_numbers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar um modelo de rede neural convolucional para prever números de loteria e gerar sugestões de números.")
    parser.add_argument('--train', help="Arquivo CSV contendo os jogos anteriores para treinamento.")
    parser.add_argument('--model', help="Arquivo (sem formato) para salvar ou carregar o modelo treinado.")
    parser.add_argument('--generate', help="Arquivo (sem formato) do modelo para gerar números sugeridos.")
    parser.add_argument('--count', type=int, default=1, help="Número de conjuntos de números sugeridos a serem gerados (requerido com --generate).")
    
    args = parser.parse_args()
    
    if args.train:
        train_model(args.train, args.model)
    elif args.generate:
        if args.count <= 0:
            print("O número de conjuntos a serem gerados deve ser um inteiro positivo.")
        else:
            generate_suggestions(args.generate, args.count)
    else:
        print("Por favor, forneça um argumento válido: --train para treinar o modelo ou --generate para gerar sugestões.")
