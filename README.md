# Loteria Previsões com Rede Neural Convolucional

Este projeto utiliza uma rede neural convolucional para prever posições em uma matriz 5x5, baseada em jogos anteriores. As previsões geram sugestões de números para jogar na loteria.

## Estrutura do Projeto

- `lotery.py`: Script principal que treina o modelo e gera números sugeridos.
- `games.csv`: Arquivo de exemplo com dados de jogos anteriores (cada linha representa um conjunto de números jogados).
- `model.keras`: Arquivo do modelo treinado a partir de 'games.csv'. Pode ser excluído e recriado com a opção '--train'
- `requirements.txt`: Arquivo com as dependências necessárias para executar o projeto.

## Configuração do Ambiente

Para configurar o ambiente e garantir que todas as dependências necessárias estejam instaladas, siga os passos abaixo:

### 1. Clonar o Repositório

```bash
git clone https://github.com/geosoliveira/lotery-neural.git
cd nome-do-repositorio
```
Opcionalmente, você pode baixar o código em formato ZIP, manualmente.

### 2. Criar um Ambiente Virtual (Opcional)

Recomenda-se criar um ambiente virtual para isolar as dependências do projeto:

```bash
python -m venv venv
source venv/bin/activate  # No Windows, use: venv\Scripts\activate
```

### 3. Instalar as Dependências

Instale as dependências necessárias com o comando abaixo:

```bash
pip install -r requirements.txt
```

## Execução do Script

O script pode ser executado com duas principais opções:

### Treinar o Modelo

Para treinar o modelo com um arquivo CSV de jogos anteriores e salvar o modelo treinado:

```bash
python lotery.py --train games.csv --model model
```

Onde **'--model'** especifica o arquivo onde o modelo treinado será salvo.

**OBS:** Não inclua o formato do arquivo do modelo na chamada do script.

### Gerar Sugestões de jogos

Para gerar sugestões de números usando um modelo treinado:

```bash
python lotery.py --generate model --count 5
```

Onde **'--model'** especifica o arquivo do modelo treinado anteriormente e **'--count'** especifica quantos jogos sugeridos você deseja gerar (padrão é 1).

**OBS:** Também neste caso, não inclua o formato do arquivo do modelo na chamada do script.

## Contribuições

Sinta-se à vontade para contribuir com melhorias para este projeto. Para começar, faça um fork do repositório e crie um pull request com suas alterações.

## Licença

Este projeto está licenciado sob a MIT License.
