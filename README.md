# Loteria Previsões com Rede Neural Convolucional

Este projeto usa histórico da Lotofácil para treinar uma rede neural temporal, comparar baselines estatísticos e gerar sugestões de jogos. Ele é um experimento de modelagem: loterias são eventos aleatórios e não há garantia de previsão de resultados futuros.

## Documentação

- [Uso da CLI](docs/usage.md)
- [Modelagem e avaliação](docs/modeling.md)
- [Dados e atualização do `games.csv`](docs/data.md)
- [Desenvolvimento](docs/development.md)

## Requisitos

Use Python 3.12 ou 3.13. O TensorFlow ainda não disponibiliza pacote compatível com Python 3.14 no PyPI.

## Instalação

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ou instale o projeto em modo editável:

```powershell
python -m pip install -e .
```

## Flags Mais Usadas

Os comandos do projeto seguem o formato:

```powershell
python lotery.py <comando> <arquivo-ou-modelo> [opções]
```

As opções mais comuns são:

- `--window-size`: quantos concursos anteriores entram como contexto para prever o próximo. Exemplo: `10` usa os últimos 10 concursos.
- `--seed`: fixa a aleatoriedade para facilitar comparação entre execuções.
- `--output`: salva relatórios de avaliação em JSON.
- `--epochs`: quantidade de ciclos de treino da rede neural. Valores maiores podem demorar mais.
- `--validation-size`: quantidade de pares temporais finais reservados para avaliação durante o treino.
- `--count`: quantidade de sugestões a gerar.
- `--max-steps`: limita quantos cortes temporais serão avaliados no backtest neural.
- `--decay`: controla quanto peso o baseline ponderado por recência dá aos concursos antigos.
- `--hybrid`: na geração, combina a rede neural com sinais simples do histórico.
- `--model-weight`: no modo híbrido, aumenta ou reduz a influência da rede neural.
- `--recent-weight`: no modo híbrido, aumenta ou reduz a influência dos números que mais apareceram recentemente.
- `--recency-weight`: no modo híbrido, aumenta ou reduz a influência dos números frequentes, dando mais peso aos concursos novos.
- `--delay-weight`: no modo híbrido, aumenta ou reduz a influência dos números que estão há mais concursos sem aparecer.
- `--recent-window`: no modo híbrido, define quantos concursos recentes entram no cálculo de frequência recente.
- `--tf-verbose`: mostra logs técnicos do TensorFlow, que ficam ocultos por padrão.

Veja a referência completa em [Uso da CLI](docs/usage.md).

Durante treino e backtests, o projeto mostra um diagnóstico da janela temporal. Se `--window-size` for grande demais para o tamanho do histórico, ele emite um aviso porque sobram poucos pares de treino e as métricas ficam menos confiáveis.

## Fluxo Recomendado

### 1. Sincronize a base histórica

O modelo precisa de dados anteriores para aprender padrões. Este comando baixa os resultados oficiais disponíveis da Lotofácil e recria o `games.csv`, que é a base usada pelo restante do projeto.

```powershell
python lotery.py sync-games games.csv
```

Use este passo quando estiver começando do zero ou quando quiser reconstruir a base inteira.

### 2. Compare baselines simples

Um baseline é uma estratégia simples usada como referência. Por exemplo: escolher números aleatórios, escolher os números mais frequentes ou dar mais peso aos concursos recentes.

Esse passo responde uma pergunta importante: a rede neural está fazendo algo melhor do que uma regra simples?

```powershell
python lotery.py backtest-baselines games.csv --window-size 10 --seed 123 --decay 0.95 --output reports-baselines.json --explain
```

O resultado mostra a média de acertos dessas estratégias simples no histórico. O arquivo `reports-baselines.json` guarda o relatório para comparação futura.

### 3. Avalie a rede neural no passado

Este passo é opcional, mas recomendado para experimentos mais cuidadosos. Ele simula o uso real do modelo: treina com concursos antigos, tenta prever o concurso seguinte e repete isso em vários pontos do histórico.

```powershell
python lotery.py backtest-neural games.csv --window-size 10 --epochs 10 --min-training-games 100 --max-steps 20 --seed 123 --output reports-neural.json --compare-baselines reports-baselines.json --explain
```

Isso ajuda a comparar a rede neural com os baselines. Como ele treina vários modelos, pode demorar; `--max-steps 20` limita a avaliação aos 20 cortes mais recentes.

### 4. Treine o modelo final

Depois de comparar baselines e, se desejar, fazer o backtest neural ou híbrido, treine o modelo que será usado para gerar sugestões.

```powershell
python lotery.py train games.csv --model model --window-size 10 --epochs 500 --validation-size 20 --seed 123
```

Esse comando cria os arquivos `model.keras`, `model_history.json` e `model_metadata.json`.

### 5. Gere sugestões

Com o modelo treinado, gere a quantidade desejada de jogos sugeridos.

```powershell
python lotery.py generate model --count 5
```

As sugestões evitam repetir jogos que já existem no histórico salvo durante o treinamento.

Se o backtest neural ficar empatado com os baselines, uma alternativa prática é gerar sugestões no modo híbrido. Ele mistura as probabilidades do modelo com frequência recente, recência ponderada e atraso dos números:

```powershell
python lotery.py generate model --count 5 --hybrid --model-weight 0.5 --recent-weight 0.2 --recency-weight 0.2 --delay-weight 0.1
```

Na prática, os pesos dizem quanto cada critério participa da escolha final. Com o exemplo acima, metade da decisão vem do modelo neural e a outra metade vem de sinais estatísticos simples do histórico.

Isso não transforma o histórico em previsão garantida; apenas deixa explícito que a sugestão final usa tanto o modelo treinado quanto sinais estatísticos simples.

Para avaliar essa estratégia antes de usá-la, rode o backtest híbrido:

```powershell
python lotery.py backtest-hybrid games.csv --window-size 10 --epochs 10 --min-training-games 100 --max-steps 20 --seed 123 --output reports-hybrid.json --compare-baselines reports-baselines.json --explain
```

Esse comando compara, nos mesmos cortes temporais, a rede neural pura com a sugestão híbrida.

Depois de salvar os relatórios, compare tudo junto:

```powershell
python lotery.py compare-reports reports-baselines.json reports-neural.json reports-hybrid.json
```

### 6. Atualize após novos sorteios

Quando houver novos concursos, não é necessário baixar tudo de novo. Use a sincronização incremental para buscar apenas os concursos faltantes, treine novamente e gere novas sugestões.

```powershell
python lotery.py sync-games games.csv --incremental
python lotery.py train games.csv --model model --window-size 10 --epochs 500 --validation-size 20 --seed 123
python lotery.py generate model --count 5
```

## Testes

```powershell
python -m unittest discover -s tests
```

## Licença

Este projeto está licenciado sob a MIT License.
