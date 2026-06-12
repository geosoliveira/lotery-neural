# Modelagem e Avaliação

O treinamento usa janelas temporais: os últimos `N` sorteios são usados como entrada para treinar o modelo a prever o sorteio seguinte. O valor padrão de `N` é 10 e pode ser alterado com `--window-size`.

Modelos gerados por versões antigas devem ser treinados novamente, pois a estrutura de entrada mudou e agora o script também precisa dos arquivos `*_history.json` e `*_metadata.json`.

## Ideia central

O projeto transforma cada concurso em uma matriz 5x5, representando os números de 1 a 25. Cada posição marcada indica que aquele número saiu no concurso.

Em vez de treinar o modelo para copiar um concurso, o projeto cria pares temporais:

```text
últimos N concursos -> próximo concurso
```

Por exemplo, com `--window-size 10`, o modelo recebe 10 concursos anteriores como entrada e aprende a aproximar o concurso seguinte.

Isso não torna a loteria previsível. A ideia é apenas testar se existe algum padrão histórico que supere estratégias simples.

## Tamanho da janela

`--window-size` não deve ser igual ao total de concursos do CSV. O projeto precisa deixar concursos seguintes disponíveis para formar pares de treino.

Exemplo:

```text
100 concursos com window-size 10  -> 90 pares temporais
100 concursos com window-size 90  -> 10 pares temporais
100 concursos com window-size 100 -> nenhum par possível
```

Durante treino e backtests, o projeto exibe um diagnóstico com total de concursos, janela escolhida e quantidade de pares temporais. Ele também alerta quando a janela gera poucos pares ou consome uma parte muito grande do histórico.

## Métrica

A métrica principal é a quantidade de acertos entre os 15 números mais prováveis sugeridos e os 15 números do concurso real.

Ela serve para comparar experimentos dentro deste projeto, mas não deve ser interpretada como promessa de desempenho futuro.

Exemplo: se o modelo sugeriu 15 números e 9 deles aparecem no concurso real, aquele corte teve 9 acertos.

Ao final, o projeto resume:

- média de acertos;
- menor número de acertos;
- maior número de acertos;
- quantidade de amostras avaliadas.

## Baselines

Um baseline é uma regra simples usada como ponto de comparação. Ele responde: "se eu não usasse rede neural, quão bem uma estratégia simples iria?"

O comando `backtest-baselines` avalia:

- `random`: jogo aleatório com 15 números;
- `most_frequent`: 15 números mais frequentes em todo o histórico disponível até aquele ponto;
- `recent_most_frequent`: 15 números mais frequentes nos últimos `N` concursos;
- `recency_weighted`: números mais frequentes com peso maior para concursos recentes.

O parâmetro `--decay` controla o peso do histórico no baseline `recency_weighted`. Valores próximos de `1`, como `0.95`, preservam mais influência dos concursos antigos; valores menores dão mais força aos concursos recentes.

Se a rede neural não superar esses baselines com alguma consistência, ela provavelmente não está adicionando valor prático.

## Backtest neural

Backtest é uma simulação do passado. O projeto finge estar em um ponto anterior do histórico, treina usando apenas dados disponíveis até ali e tenta prever o concurso seguinte.

O comando `backtest-neural`:

- usa os primeiros `--min-training-games` concursos como histórico inicial;
- treina um modelo usando apenas dados anteriores ao concurso testado;
- prevê o próximo concurso;
- avança para o corte seguinte;
- resume média, mínimo e máximo de acertos.

Como cada corte treina um novo modelo, esse comando é mais caro que `backtest-baselines`. Para uma execução rápida, use poucas épocas e limite `--max-steps`.

Um bom fluxo é:

```text
1. rodar backtest-baselines;
2. rodar backtest-neural;
3. rodar backtest-hybrid se quiser avaliar a geração híbrida;
4. comparar os relatórios JSON;
5. só então treinar o modelo final para gerar sugestões.
```

## Comparação de relatórios

Use `--output` nos backtests para salvar relatórios JSON. Depois compare:

```powershell
python lotery.py compare-reports reports-baselines.json reports-neural.json reports-hybrid.json
```

O comparador procura o melhor baseline pela média de acertos e compara com a média da rede neural e/ou da geração híbrida. Se a diferença for pequena, trate como empate prático. Uma diferença positiva consistente é mais interessante do que um ganho isolado em uma execução curta.

Quando um backtest é executado com `--compare-baselines` e `--output`, o relatório JSON também recebe um bloco `summary`. Esse bloco guarda uma leitura consolidada, incluindo:

- melhor baseline;
- melhor avaliação entre rede neural e híbrido;
- melhor estratégia geral;
- se o híbrido superou o melhor baseline;
- se o híbrido superou a rede neural pura;
- recomendação textual para leitura rápida.

## Geração híbrida

O comando `generate` pode usar apenas as probabilidades do modelo neural ou, com `--hybrid`, combinar essas probabilidades com sinais estatísticos simples do histórico:

- probabilidade estimada pela rede neural: é o palpite do modelo treinado a partir das janelas temporais;
- frequência nos concursos recentes: mede quais números apareceram mais dentro de uma janela curta do histórico;
- frequência ponderada por recência: mede frequência no histórico, mas dá mais importância aos concursos novos;
- atraso desde a última ocorrência: mede há quantos concursos cada número não aparece.

Exemplo:

```powershell
python lotery.py generate model --count 5 --hybrid --model-weight 0.5 --recent-weight 0.2 --recency-weight 0.2 --delay-weight 0.1
```

Essa abordagem é útil quando a rede neural não demonstrou ganho claro no backtest, mas você ainda quer aproveitar o modelo treinado como parte de uma estratégia de geração. Ela não deve ser interpretada como melhoria comprovada de previsão; é uma forma transparente de misturar modelo e heurísticas.

Os pesos controlam a influência de cada sinal. Por exemplo, `--model-weight 0.5` e os demais pesos somando `0.5` indicam uma geração aproximadamente dividida entre a rede neural e estatísticas simples. Se um peso for `0`, aquele sinal não participa. Se todos os pesos forem próximos, a geração fica mais equilibrada.

Uma boa forma de experimentar é mudar uma coisa por vez:

```text
mais confiança na rede      -> aumente --model-weight
mais força para histórico   -> aumente --recent-weight ou --recency-weight
mais força para atraso      -> aumente --delay-weight
histórico recente mais curto -> diminua --recent-window
histórico antigo mais fraco -> diminua --decay
```

Esses ajustes mudam o critério de geração, mas não substituem avaliação. Depois de mudar a estratégia, continue usando backtests e baselines como referência.

Para avaliar a geração híbrida, use:

```powershell
python lotery.py backtest-hybrid games.csv --window-size 10 --epochs 10 --min-training-games 100 --max-steps 20 --output reports-hybrid.json --compare-baselines reports-baselines.json --explain
```

Esse comando treina a rede em cortes temporais e mede duas saídas no mesmo ponto do passado:

- `neural`: sugestão feita apenas com as probabilidades do modelo;
- `hybrid`: sugestão feita com a mistura de modelo, frequência recente, recência ponderada e atraso.

Isso ajuda a responder uma pergunta específica: a mistura híbrida melhora a sugestão em relação à rede pura e aos baselines simples?

## Limitações

Loterias são eventos aleatórios. O projeto gera sugestões com base em padrões históricos observados no arquivo de entrada, mas não oferece garantia de previsão dos próximos resultados.

Um resultado bom na validação pode refletir padrões do histórico, acaso estatístico ou ajuste excessivo aos dados disponíveis. Sempre compare a rede neural com os baselines.
