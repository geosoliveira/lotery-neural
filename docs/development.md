# Desenvolvimento

## Estrutura

- `lotery.py`: wrapper de compatibilidade para executar `python lotery.py ...`;
- `lotery_neural/lotery_cli.py`: CLI e tratamento de erros esperados;
- `lotery_neural/lotery_config.py`: valores padrão compartilhados;
- `lotery_neural/lotery_data.py`: leitura, validação e transformação dos jogos;
- `lotery_neural/lotery_evaluation.py`: sugestões e métricas;
- `lotery_neural/lotery_metadata.py`: JSON, seed e metadados;
- `lotery_neural/lotery_model.py`: treino, carregamento e geração;
- `lotery_neural/lotery_results.py`: atualização via API da Caixa;
- `lotery_neural/lotery_backtest.py`: backtesting de baselines e rede neural.

## Instalação editável

```powershell
python -m pip install -e .
```

## Testes

```powershell
python -m unittest discover -s tests
```

Alguns testes unitários são ignorados automaticamente quando as dependências científicas não estão instaladas no Python ativo.

## Python

Use Python 3.12 ou 3.13. O TensorFlow ainda não disponibiliza pacote compatível com Python 3.14 no PyPI.

## Artefatos locais

Arquivos como `.venv/`, caches, modelos gerados, `*_history.json`, `*_metadata.json`, `*.key` e `*_previous_games.enc` são ignorados pelo `.gitignore`.
