# Dados

O arquivo `games.csv` contém um concurso por linha. Cada linha deve ter 15 números separados por ponto-e-vírgula:

```text
1;2;3;4;5;6;7;8;9;10;11;12;13;14;15
```

## API da Caixa

O projeto usa a API pública da Caixa para buscar resultados oficiais da Lotofácil. Essa é uma dependência externa e pode ficar indisponível temporariamente.

## Sincronização completa

```powershell
python lotery.py sync-games games.csv
```

Esse comando sobrescreve o arquivo informado. Use quando quiser reconstruir a base histórica.

## Sincronização incremental

```powershell
python lotery.py sync-games games.csv --incremental
```

Esse modo assume que a linha 1 corresponde ao concurso 1, a linha 2 ao concurso 2, e assim por diante. Com isso, calcula o próximo concurso faltante a partir da quantidade de linhas existentes e busca apenas o intervalo necessário.

Se o arquivo não existir ou estiver vazio, a sincronização incremental começa no concurso 1. Nesse caso, ela equivale a baixar todo o histórico e pode demorar bastante, pois a API é consultada concurso por concurso. O comando mostra progresso periodicamente e grava cada concurso conforme ele é baixado, então você pode interromper e continuar depois com `--incremental`.

## Atualização manual

Se a API da Caixa estiver indisponível, edite o `games.csv` manualmente e adicione uma nova linha com os 15 números do concurso.

## Criptografia

Versões anteriores exigiam uma chave para criptografar o arquivo auxiliar de jogos anteriores. Essa necessidade foi removida: o histórico agora é salvo em JSON simples.

O arquivo `chave.key`, se existir em instalações antigas, é legado e pode ser removido.
