from cryptography.fernet import Fernet
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Flatten, Dense
import json
import argparse
import os
import sys

# Geração da chave de criptografia (faça isso apenas uma vez e mantenha a chave segura)
def generate_key():
    return Fernet.generate_key()

# Salvar a chave em um arquivo seguro (faça isso uma vez e armazene de forma segura)
def save_key(key, key_file):
    with open(key_file, 'wb') as f:
        f.write(key)

# Carregar a chave de criptografia
def load_key(key_file):
    with open(key_file, 'rb') as f:
        return f.read()
        
# Criptografar e salvar previous_games
def encrypt_and_save(data, file_name, key):
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(json.dumps(data).encode())
    with open(file_name, 'wb') as f:
        f.write(encrypted_data)

# Descriptografar e carregar previous_games
def decrypt_and_load(file_name, key):
    fernet = Fernet(key)
    with open(file_name, 'rb') as f:
        encrypted_data = f.read()
    decrypted_data = fernet.decrypt(encrypted_data)
    return json.loads(decrypted_data.decode())

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
    
def csv_to_numbers(csv_file):
    # Converte o CSV em uma lista de matrizes 5x5
    matrices = csv_to_matrices(csv_file)
    
    # Converte cada matriz em uma lista de números e armazena todas as listas
    numbers_list = []
    for matrix in matrices:
        numbers = matrix_to_numbers(matrix)
        numbers_list.append(numbers)
    
    return numbers_list

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

def train_model(csv_file, key_file, model_file=None):
    if not key_file:
        raise ValueError("O arquivo de chave não pode ser uma string vazia. Use o argumento '--key <nome-chave>.'")
        
    if not model_file:
        model_file = ""
        
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
        
        # Geração e salvamento da chave de criptografia
        key = generate_key()
        save_key(key, key_file)

        # Salvar previous_games criptografado
        previous_games = csv_to_numbers(csv_file)
        encrypt_and_save(previous_games, model_file_w_format.replace('.keras', '_previous_games.enc'), key)
        print(f"Jogos anteriores salvos e criptografados em {model_file_w_format.replace('.keras', '_previous_games.enc')}")
    else:
        print("Treinamento concluído, mas o modelo não foi salvo.")
        
def load_previous_games(model_file_w_format, key_file):
    if not key_file:
        raise ValueError("O arquivo de chave não pode ser uma string vazia. Use o argumento '--key <nome-chave>.'")
        
    if not os.path.exists(key_file):
        print(f"Chave de criptografia {key_file} não encontrada. Treine novamente o modelo para ser novamente criada.")
        sys.exit(1)

    key = load_key(key_file)
    previous_games_file = model_file_w_format.replace('.keras', '_previous_games.enc')
    if os.path.exists(previous_games_file):
        return decrypt_and_load(previous_games_file, key)
    else:
        print(f"Arquivo {previous_games_file} não encontrado.")
        sys.exit(1)

def generate_unique_suggestions(model_file, count, key_file):
    if not model_file:
        raise ValueError("Forneça o nome do modelo de teste (sem formato) por meio do qual os números sugeridos deverão ser criados.'")
        
    model_file_w_format = model_file + ".keras"
    if not os.path.exists(model_file_w_format):
        print(f"O arquivo {model_file_w_format} não existe.")
        return

    model = load_model(model_file_w_format)
    print(f"Modelo carregado de '{model_file_w_format}'")
    
    # Carregar previous_games criptografado
    previous_games = load_previous_games(model_file_w_format, key_file)

    unique_suggestions = []
    
    while len(unique_suggestions) < count:
        dummy_input = np.zeros((1, 5, 5, 1)) if len(unique_suggestions) == 0 else np.random.random((1, 5, 5, 1))  # Gerando uma matriz 5x5 com valores aleatórios a partir do segundo jogo
        
        predicted = model.predict(dummy_input)
        predicted_matrix = predicted.reshape(5, 5)
        
        suggested_positions = np.unravel_index(np.argsort(predicted_matrix, axis=None)[-15:], (5, 5))

        suggested_matrix = np.zeros((5, 5))
        for pos in zip(suggested_positions[0], suggested_positions[1]):
            suggested_matrix[pos] = 1

        suggested_numbers = matrix_to_numbers(suggested_matrix)
        if suggested_numbers not in previous_games and suggested_numbers not in unique_suggestions:
            unique_suggestions.append(suggested_numbers)
            #print(f"Números sugeridos {len(unique_suggestions)}:")
            #print(suggested_numbers)
        else:
            print("Sugestão repetida. Gerando uma nova.")
            
    return unique_suggestions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar um modelo de rede neural convolucional para prever números da LOTOFACIL.")
    parser.add_argument('--train', help="Arquivo CSV contendo os jogos anteriores para treinamento.")
    parser.add_argument('--model', help="Arquivo (sem formato) para salvar ou carregar o modelo treinado.")
    parser.add_argument('--generate', help="Arquivo (sem formato) do modelo para gerar números sugeridos.")
    parser.add_argument('--count', type=int, default=1, help="Número de conjuntos de números sugeridos a serem gerados (requerido com --generate).")
    parser.add_argument("--key", type=str, help="Arquivo de chave para criptografia/descriptografia do arquivo auxiliar")
    
    args = parser.parse_args()
    
    if args.train:
        train_model(args.train, args.key, args.model)
    elif args.generate:
        if args.count <= 0:
            print("O número de conjuntos a serem gerados deve ser um inteiro positivo.")
        else:
            suggestions = generate_unique_suggestions(args.generate, args.count, args.key)
            print("Sugestões geradas:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"Sugestão {i}: {suggestion}")
    else:
        parser.print_help()