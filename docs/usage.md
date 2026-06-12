# Uso da CLI

Todos os comandos podem ser executados com `python lotery.py ...`. Depois de instalar o projeto com `python -m pip install -e .`, também é possível usar o entrypoint `lotery-neural`.

## Como ler os comandos

Os comandos seguem este formato:

```powershell
python lotery.py <comando> <entrada> [opções]
```

Exemplo:

```powershell
python lotery.py train games.csv --model model --window-size 10
```

Nesse caso:

- `train` é o comando;
- `games.csv` é o arquivo de entrada;
- `--model model` informa o nome do modelo a salvar;
- `--window-size 10` configura quantos concursos anteriores serão usados como contexto.

## Sincronizar resultados

Recria o `games.csv` com todos os concursos oficiais disponíveis:

```powershell
python lotery.py sync-games games.csv
```

Limita o intervalo de concursos:

```powershell
python lotery.py sync-games games.csv --from-contest 1 --to-contest 100
```

Busca apenas concursos faltantes:

```powershell
python lotery.py sync-games games.csv --incremental
```

Adiciona apenas o último concurso, se ainda não existir:

```powershell
python lotery.py update-games games.csv
```

Flags úteis:

- `--from-contest`: concurso inicial para uma sincronização completa.
- `--to-contest`: concurso final para uma sincronização completa.
- `--incremental`: busca apenas concursos faltantes, assumindo que a linha 1 do CSV é o concurso 1.

## Avaliar baselines

Baselines são estratégias simples usadas como referência para saber se a rede neural realmente está agregando algo. Este comando testa essas estratégias no histórico.

```powershell
python lotery.py backtest-baselines games.csv --window-size 10 --seed 123 --decay 0.95 --output reports-baselines.json --explain
```

Flags úteis:

- `--window-size`: janela usada pelos baselines que olham concursos recentes.
- `--seed`: fixa o baseline aleatório para comparação reprodutível.
- `--decay`: peso dos concursos antigos no baseline ponderado por recência.
- `--output`: salva o relatório em JSON.
- `--explain`: imprime uma explicação curta sobre como interpretar médias, mínimos e máximos.

Durante a execução, o comando mostra progresso por amostras avaliadas. Com milhares de concursos, isso ajuda a diferenciar processamento normal de travamento.

## Avaliar rede neural

Este comando simula o passado: treina com concursos antigos e tenta prever o concurso seguinte. É mais lento porque treina um modelo para cada corte avaliado.

```powershell
python lotery.py backtest-neural games.csv --window-size 10 --epochs 10 --min-training-games 100 --max-steps 20 --seed 123 --output reports-neural.json --compare-baselines reports-baselines.json --explain
```

Flags úteis:

- `--window-size`: quantos concursos anteriores entram como contexto.
- `--epochs`: épocas de treino em cada corte temporal.
- `--min-training-games`: quantidade mínima de concursos antes do primeiro teste.
- `--max-steps`: limita a avaliação aos cortes mais recentes.
- `--seed`: fixa a aleatoriedade do treino quando possível.
- `--output`: salva o relatório em JSON.
- `--compare-baselines`: compara o resultado neural com um relatório salvo por `backtest-baselines`.
- `--explain`: imprime uma explicação curta sobre como interpretar a saída neural.
- `--tf-verbose`: mostra logs técnicos do TensorFlow. Por padrão, esses logs são ocultados para reduzir poluição visual.

## Comparar relatórios

Se você já tem os relatórios JSON, compare sem rodar os backtests novamente. O primeiro arquivo deve ser o relatório de baselines; depois informe um ou mais relatórios de avaliação.

```powershell
python lotery.py compare-reports reports-baselines.json reports-neural.json reports-hybrid.json
```

O comando mostra:

- melhor baseline;
- média da rede neural, quando houver;
- média da geração híbrida, quando houver;
- diferença de cada avaliação contra o melhor baseline;
- leitura final indicando empate prático ou vantagem visível.

## Avaliar geração híbrida

Este comando avalia o modo híbrido no passado. Em cada corte temporal, ele treina a rede neural, gera uma sugestão com a rede pura e outra com a pontuação híbrida, depois compara as duas contra o concurso real.

```powershell
python lotery.py backtest-hybrid games.csv --window-size 10 --epochs 10 --min-training-games 100 --max-steps 20 --seed 123 --output reports-hybrid.json --compare-baselines reports-baselines.json --explain
```

Flags úteis:

- `--window-size`: quantos concursos anteriores entram como contexto da rede.
- `--epochs`: épocas de treino em cada corte temporal.
- `--min-training-games`: quantidade mínima de concursos antes do primeiro teste.
- `--max-steps`: limita a avaliação aos cortes mais recentes.
- `--seed`: fixa a aleatoriedade do treino quando possível.
- `--output`: salva o relatório em JSON.
- `--compare-baselines`: compara o resultado híbrido com um relatório salvo por `backtest-baselines`.
- `--explain`: imprime uma explicação curta sobre como interpretar a saída.
- `--tf-verbose`: mostra logs técnicos do TensorFlow.
- `--model-weight`: peso das probabilidades do modelo. Padrão: `0.5`.
- `--recent-weight`: peso da frequência recente. Padrão: `0.2`.
- `--recency-weight`: peso da frequência ponderada por recência. Padrão: `0.2`.
- `--delay-weight`: peso do atraso dos números. Padrão: `0.1`.
- `--recent-window`: concursos usados na frequência recente. Padrão: `50`.
- `--decay`: decaimento do sinal ponderado por recência. Padrão: `0.95`.

A saída mostra duas linhas: `neural`, que usa só o modelo, e `híbrido`, que mistura modelo e sinais estatísticos. O híbrido só parece interessante se superar a rede pura e os baselines por uma margem consistente.

Quando `--compare-baselines` é usado junto com `--output`, o JSON salvo recebe também um bloco `summary`, com campos como `best_overall`, `hybrid_outperformed_baseline`, `hybrid_outperformed_neural` e uma recomendação textual.

## Treinar modelo

Treina o modelo final que será usado para gerar sugestões.

```powershell
python lotery.py train games.csv --model model --window-size 10 --epochs 500 --validation-size 20 --seed 123
```

Flags úteis:

- `--model`: nome do modelo salvo, sem `.keras`.
- `--window-size`: quantos concursos anteriores entram como contexto.
- `--epochs`: quantidade de épocas de treinamento.
- `--validation-size`: quantos pares finais são reservados para avaliação.
- `--seed`: fixa a aleatoriedade para facilitar comparação.
- `--tf-verbose`: mostra logs técnicos do TensorFlow. Por padrão, esses logs são ocultados.

Durante o treino, o projeto mostra um diagnóstico com total de concursos, `window-size` e quantidade de pares temporais gerados. Se a janela for grande demais para o histórico disponível, o comando emite um aviso.

## Gerar sugestões

Gera jogos sugeridos a partir de um modelo treinado.

```powershell
python lotery.py generate model --count 5
```

Se houver muitas repetições:

```powershell
python lotery.py generate model --count 5 --max-attempts 1000
```

Flags úteis:

- `--count`: quantidade de sugestões.
- `--max-attempts`: limite de tentativas para encontrar sugestões não repetidas.
- `--hybrid`: ativa a geração híbrida. Sem essa flag, o comando usa apenas a saída da rede neural.
- `--model-weight`: peso das probabilidades do modelo na geração híbrida. Padrão: `0.5`. Aumente se quiser confiar mais no modelo treinado.
- `--recent-weight`: peso da frequência nos concursos recentes. Padrão: `0.2`. Aumente se quiser favorecer números que apareceram bastante nos concursos mais próximos.
- `--recency-weight`: peso da frequência ponderada por recência. Padrão: `0.2`. Aumente se quiser considerar todo o histórico, mas com mais força para concursos novos.
- `--delay-weight`: peso do atraso desde a última ocorrência de cada número. Padrão: `0.1`. Aumente se quiser favorecer números que estão há mais tempo sem aparecer.
- `--recent-window`: quantidade de concursos usados para calcular frequência recente. Padrão: `50`. Valores menores olham um período mais curto; valores maiores deixam o cálculo mais estável.
- `--decay`: fator de decaimento usado no sinal ponderado por recência. Padrão: `0.95`. Quanto mais perto de `1`, mais os concursos antigos continuam influenciando; quanto menor, mais a geração favorece concursos recentes.
- `--tf-verbose`: mostra logs técnicos do TensorFlow. Por padrão, esses logs são ocultados.

Exemplo com geração híbrida:

```powershell
python lotery.py generate model --count 5 --hybrid --model-weight 0.5 --recent-weight 0.2 --recency-weight 0.2 --delay-weight 0.1
```

Use `--hybrid` quando quiser que a sugestão final não dependa somente das probabilidades do modelo. Isso é útil especialmente quando o backtest neural fica muito próximo dos baselines.

Como ler os pesos:

- Os pesos não precisam somar `1`; o projeto normaliza a pontuação final.
- Um peso maior aumenta a influência daquele critério.
- Um peso `0` desliga aquele critério.
- Pesos negativos não são aceitos.

Exemplos de ajustes:

```powershell
python lotery.py generate model --count 5 --hybrid --model-weight 0.8 --recent-weight 0.1 --recency-weight 0.1 --delay-weight 0
```

Esse exemplo confia mais na rede neural e quase não usa atraso.

```powershell
python lotery.py generate model --count 5 --hybrid --model-weight 0.3 --recent-weight 0.3 --recency-weight 0.3 --delay-weight 0.1
```

Esse exemplo divide melhor a decisão entre modelo e histórico.
