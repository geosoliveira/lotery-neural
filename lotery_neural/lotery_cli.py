import argparse
import json

from .lotery_config import (
    DEFAULT_EPOCHS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_VALIDATION_SIZE,
    DEFAULT_WINDOW_SIZE,
)


def add_training_options(parser):
    parser.add_argument('--window-size', type=int, default=DEFAULT_WINDOW_SIZE, help="Quantidade de sorteios anteriores usados para prever o próximo.")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help="Quantidade de épocas de treinamento.")
    parser.add_argument('--validation-size', type=int, default=DEFAULT_VALIDATION_SIZE, help="Quantidade de pares temporais finais reservados para avaliação.")
    parser.add_argument('--seed', type=int, help="Semente para tornar o treinamento mais reprodutível.")


def build_parser():
    parser = argparse.ArgumentParser(description="Treinar, avaliar e gerar sugestões para a Lotofácil.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Treina o modelo temporal.")
    train_parser.add_argument("csv_file", help="Arquivo CSV contendo os jogos anteriores.")
    train_parser.add_argument("--model", required=True, help="Arquivo do modelo sem extensão.")
    add_training_options(train_parser)

    generate_parser = subparsers.add_parser("generate", help="Gera sugestões com um modelo treinado.")
    generate_parser.add_argument("model", help="Arquivo do modelo sem extensão.")
    generate_parser.add_argument("--count", type=int, default=1, help="Quantidade de sugestões.")
    generate_parser.add_argument('--max-attempts', type=int, default=DEFAULT_MAX_ATTEMPTS, help="Quantidade máxima de tentativas para gerar sugestões únicas.")
    generate_parser.add_argument('--tf-verbose', action='store_true', help="Mostra logs técnicos do TensorFlow.")
    add_hybrid_generation_options(generate_parser)

    update_parser = subparsers.add_parser("update-games", help="Adiciona o último resultado oficial ao CSV.")
    update_parser.add_argument("csv_file", help="Arquivo CSV a ser atualizado.")

    sync_parser = subparsers.add_parser("sync-games", help="Sincroniza resultados oficiais no CSV.")
    sync_parser.add_argument("csv_file", help="Arquivo CSV a ser sincronizado.")
    sync_parser.add_argument('--from-contest', type=int, default=1, help="Concurso inicial para sincronização completa.")
    sync_parser.add_argument('--to-contest', type=int, help="Concurso final. Se omitido, usa o último concurso.")
    sync_parser.add_argument('--incremental', action='store_true', help="Busca apenas concursos faltantes, assumindo que a linha 1 corresponde ao concurso 1.")

    backtest_parser = subparsers.add_parser("backtest-baselines", help="Avalia baselines simples no histórico.")
    backtest_parser.add_argument("csv_file", help="Arquivo CSV usado no backtest.")
    backtest_parser.add_argument('--window-size', type=int, default=DEFAULT_WINDOW_SIZE, help="Tamanho da janela temporal.")
    backtest_parser.add_argument('--seed', type=int, help="Semente usada no baseline aleatório.")
    backtest_parser.add_argument('--decay', type=float, default=0.95, help="Fator de decaimento do baseline ponderado por recência.")
    backtest_parser.add_argument('--output', help="Arquivo JSON para salvar o relatório do backtest.")
    backtest_parser.add_argument('--explain', action='store_true', help="Explica como interpretar a saída.")

    neural_backtest_parser = subparsers.add_parser("backtest-neural", help="Avalia a rede neural em cortes temporais sucessivos.")
    neural_backtest_parser.add_argument("csv_file", help="Arquivo CSV usado no backtest neural.")
    neural_backtest_parser.add_argument('--window-size', type=int, default=DEFAULT_WINDOW_SIZE, help="Tamanho da janela temporal.")
    neural_backtest_parser.add_argument('--epochs', type=int, default=10, help="Épocas por corte temporal.")
    neural_backtest_parser.add_argument('--min-training-games', type=int, default=100, help="Mínimo de jogos históricos antes do primeiro teste.")
    neural_backtest_parser.add_argument('--max-steps', type=int, help="Limita a quantidade de cortes avaliados, usando os mais recentes.")
    neural_backtest_parser.add_argument('--seed', type=int, help="Semente usada no treino neural.")
    neural_backtest_parser.add_argument('--output', help="Arquivo JSON para salvar o relatório do backtest.")
    neural_backtest_parser.add_argument('--compare-baselines', help="Relatório JSON gerado por backtest-baselines para comparação.")
    neural_backtest_parser.add_argument('--explain', action='store_true', help="Explica como interpretar a saída.")
    neural_backtest_parser.add_argument('--tf-verbose', action='store_true', help="Mostra logs técnicos do TensorFlow.")

    hybrid_backtest_parser = subparsers.add_parser("backtest-hybrid", help="Avalia a geração híbrida em cortes temporais sucessivos.")
    hybrid_backtest_parser.add_argument("csv_file", help="Arquivo CSV usado no backtest híbrido.")
    hybrid_backtest_parser.add_argument('--window-size', type=int, default=DEFAULT_WINDOW_SIZE, help="Tamanho da janela temporal.")
    hybrid_backtest_parser.add_argument('--epochs', type=int, default=10, help="Épocas por corte temporal.")
    hybrid_backtest_parser.add_argument('--min-training-games', type=int, default=100, help="Mínimo de jogos históricos antes do primeiro teste.")
    hybrid_backtest_parser.add_argument('--max-steps', type=int, help="Limita a quantidade de cortes avaliados, usando os mais recentes.")
    hybrid_backtest_parser.add_argument('--seed', type=int, help="Semente usada no treino neural.")
    hybrid_backtest_parser.add_argument('--output', help="Arquivo JSON para salvar o relatório do backtest.")
    hybrid_backtest_parser.add_argument('--compare-baselines', help="Relatório JSON gerado por backtest-baselines para comparação.")
    hybrid_backtest_parser.add_argument('--explain', action='store_true', help="Explica como interpretar a saída híbrida.")
    hybrid_backtest_parser.add_argument('--tf-verbose', action='store_true', help="Mostra logs técnicos do TensorFlow.")
    add_hybrid_weight_options(hybrid_backtest_parser)

    compare_parser = subparsers.add_parser("compare-reports", help="Compara baselines com relatórios neural e/ou híbrido.")
    compare_parser.add_argument("baseline_report", help="JSON gerado por backtest-baselines.")
    compare_parser.add_argument("evaluation_reports", nargs="+", help="JSON gerado por backtest-neural ou backtest-hybrid.")

    return parser


def add_hybrid_generation_options(parser):
    parser.add_argument('--hybrid', action='store_true', help="Combina o modelo neural com sinais estatísticos do histórico.")
    add_hybrid_weight_options(parser)


def add_hybrid_weight_options(parser):
    parser.add_argument('--model-weight', type=float, default=0.5, help="Peso das probabilidades do modelo na geração híbrida.")
    parser.add_argument('--recent-weight', type=float, default=0.2, help="Peso da frequência nos concursos recentes.")
    parser.add_argument('--recency-weight', type=float, default=0.2, help="Peso da frequência ponderada por recência.")
    parser.add_argument('--delay-weight', type=float, default=0.1, help="Peso do atraso desde a última ocorrência de cada número.")
    parser.add_argument('--recent-window', type=int, default=50, help="Quantidade de concursos usados no sinal de frequência recente.")
    parser.add_argument('--decay', type=float, default=0.95, help="Fator de decaimento usado no sinal ponderado por recência.")


def print_baseline_results(results):
    print("Backtest de baselines:")
    for name, metrics in results.items():
        print(
            f"- {name}: média {metrics['average_hits']:.2f}, "
            f"min {metrics['min_hits']}, max {metrics['max_hits']} "
            f"({metrics['samples']} amostras)"
        )


def explain_baseline_results():
    print()
    print("Como interpretar:")
    print("- A média mostra quantos números, em média, aquela estratégia acertou em cada concurso simulado.")
    print("- Na Lotofácil, dois jogos aleatórios de 15 números dentro de 25 tendem a compartilhar perto de 9 números.")
    print("- Portanto, médias perto de 9 são esperadas mesmo para uma estratégia simples.")
    print("- A rede neural só começa a parecer útil se superar esses baselines de forma consistente.")


def explain_neural_results():
    print()
    print("Como interpretar:")
    print("- O backtest neural simula o passado: treina com concursos anteriores e tenta prever o próximo.")
    print("- Compare a média da rede neural com os baselines salvos por backtest-baselines.")
    print("- Se a média neural ficar próxima de 9 ou próxima dos baselines, o ganho prático é fraco ou inexistente.")


def explain_hybrid_results():
    print()
    print("Como interpretar:")
    print("- O backtest híbrido treina a rede em cortes temporais e compara duas saídas no mesmo corte.")
    print("- 'neural' usa apenas as probabilidades do modelo; 'híbrido' mistura modelo e sinais estatísticos.")
    print("- O híbrido só parece útil se superar a rede pura e os baselines por margem consistente.")


def save_report(report, output_file):
    if not output_file:
        return
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"Relatório salvo em {output_file}")


def load_report(report_file):
    with open(report_file, "r", encoding="utf-8") as file:
        return json.load(file)


def compare_backtest_reports(baseline_report, neural_report):
    baseline_results = baseline_report.get("results", {})
    neural_metrics = neural_report.get("results", {}).get("neural")
    if not baseline_results or not neural_metrics:
        raise ValueError("Relatórios inválidos para comparação.")

    best_baseline_name, best_baseline_metrics = max(
        baseline_results.items(),
        key=lambda item: item[1]["average_hits"],
    )
    neural_average = neural_metrics["average_hits"]
    baseline_average = best_baseline_metrics["average_hits"]
    difference = neural_average - baseline_average

    return {
        "best_baseline": best_baseline_name,
        "best_baseline_average": baseline_average,
        "neural_average": neural_average,
        "difference": difference,
        "neural_samples": neural_metrics["samples"],
        "baseline_samples": best_baseline_metrics["samples"],
    }


def compare_baseline_with_metrics(baseline_report, metrics, label):
    baseline_results = baseline_report.get("results", {})
    if not baseline_results or not metrics:
        raise ValueError("Relatórios inválidos para comparação.")

    best_baseline_name, best_baseline_metrics = max(
        baseline_results.items(),
        key=lambda item: item[1]["average_hits"],
    )
    target_average = metrics["average_hits"]
    baseline_average = best_baseline_metrics["average_hits"]
    difference = target_average - baseline_average

    return {
        "label": label,
        "best_baseline": best_baseline_name,
        "best_baseline_average": baseline_average,
        "target_average": target_average,
        "difference": difference,
        "target_samples": metrics["samples"],
        "baseline_samples": best_baseline_metrics["samples"],
    }


def extract_evaluation_metrics(report):
    results = report.get("results", {})
    report_type = report.get("type")

    if report_type == "neural_backtest" and results.get("neural"):
        return [{"label": "neural", "metrics": results["neural"]}]
    if report_type == "hybrid_backtest":
        entries = []
        if results.get("neural"):
            entries.append({"label": "neural", "metrics": results["neural"]})
        if results.get("hybrid"):
            entries.append({"label": "híbrido", "metrics": results["hybrid"]})
        return entries
    if results.get("neural"):
        return [{"label": "neural", "metrics": results["neural"]}]
    if results.get("hybrid"):
        return [{"label": "híbrido", "metrics": results["hybrid"]}]

    raise ValueError("Relatório de avaliação inválido. Use JSON de backtest-neural ou backtest-hybrid.")


def summarize_against_baseline(baseline_report, evaluation_report):
    baseline_results = baseline_report.get("results", {})
    if not baseline_results:
        raise ValueError("Relatório de baseline inválido para comparação.")

    best_baseline_name, best_baseline_metrics = max(
        baseline_results.items(),
        key=lambda item: item[1]["average_hits"],
    )
    best_baseline_average = best_baseline_metrics["average_hits"]
    evaluations = []

    for entry in extract_evaluation_metrics(evaluation_report):
        metrics = entry["metrics"]
        average_hits = metrics["average_hits"]
        evaluations.append({
            "label": entry["label"],
            "average_hits": average_hits,
            "samples": metrics["samples"],
            "difference_from_best_baseline": average_hits - best_baseline_average,
            "outperformed_best_baseline": average_hits > best_baseline_average,
            "practically_outperformed_best_baseline": average_hits - best_baseline_average > 0.25,
        })

    by_label = {entry["label"]: entry for entry in evaluations}
    hybrid_entry = by_label.get("híbrido")
    neural_entry = by_label.get("neural")
    hybrid_outperformed_neural = None
    hybrid_difference_from_neural = None
    if hybrid_entry and neural_entry:
        hybrid_difference_from_neural = hybrid_entry["average_hits"] - neural_entry["average_hits"]
        hybrid_outperformed_neural = hybrid_difference_from_neural > 0

    ranking = sorted(
        [
            {
                "label": f"baseline:{best_baseline_name}",
                "average_hits": best_baseline_average,
                "samples": best_baseline_metrics["samples"],
            },
            *[
                {
                    "label": entry["label"],
                    "average_hits": entry["average_hits"],
                    "samples": entry["samples"],
                }
                for entry in evaluations
            ],
        ],
        key=lambda entry: entry["average_hits"],
        reverse=True,
    )

    best_evaluation = max(evaluations, key=lambda entry: entry["average_hits"])
    return {
        "best_baseline": best_baseline_name,
        "best_baseline_average": best_baseline_average,
        "evaluations": evaluations,
        "best_evaluation": best_evaluation["label"],
        "best_overall": ranking[0]["label"],
        "ranking": ranking,
        "hybrid_outperformed_baseline": None if not hybrid_entry else hybrid_entry["outperformed_best_baseline"],
        "hybrid_practically_outperformed_baseline": None if not hybrid_entry else hybrid_entry["practically_outperformed_best_baseline"],
        "hybrid_outperformed_neural": hybrid_outperformed_neural,
        "hybrid_difference_from_neural": hybrid_difference_from_neural,
        "recommendation": build_recommendation_text(ranking[0]["label"], best_evaluation, hybrid_entry, neural_entry),
    }


def summarize_multiple_against_baseline(baseline_report, named_evaluation_reports):
    baseline_results = baseline_report.get("results", {})
    if not baseline_results:
        raise ValueError("Relatório de baseline inválido para comparação.")

    best_baseline_name, best_baseline_metrics = max(
        baseline_results.items(),
        key=lambda item: item[1]["average_hits"],
    )
    best_baseline_average = best_baseline_metrics["average_hits"]
    evaluations = []
    label_counts = {}

    for source, report in named_evaluation_reports:
        for entry in extract_evaluation_metrics(report):
            base_label = entry["label"]
            label_counts[base_label] = label_counts.get(base_label, 0) + 1
            label = base_label if label_counts[base_label] == 1 else f"{base_label} ({source})"
            metrics = entry["metrics"]
            average_hits = metrics["average_hits"]
            evaluations.append({
                "label": label,
                "source": source,
                "average_hits": average_hits,
                "samples": metrics["samples"],
                "difference_from_best_baseline": average_hits - best_baseline_average,
            })

    if not evaluations:
        raise ValueError("Nenhuma avaliação encontrada nos relatórios informados.")

    ranking = sorted(
        [
            {
                "label": f"baseline:{best_baseline_name}",
                "source": "baseline",
                "average_hits": best_baseline_average,
                "samples": best_baseline_metrics["samples"],
                "difference_from_best_baseline": 0.0,
            },
            *evaluations,
        ],
        key=lambda entry: entry["average_hits"],
        reverse=True,
    )
    return {
        "best_baseline": best_baseline_name,
        "best_baseline_average": best_baseline_average,
        "evaluations": evaluations,
        "ranking": ranking,
        "best_overall": ranking[0]["label"],
    }


def build_recommendation_text(best_overall_label, best_evaluation, hybrid_entry=None, neural_entry=None):
    if best_overall_label.startswith("baseline:"):
        return "O melhor baseline ficou acima das avaliações testadas; não há ganho prático claro."
    if best_evaluation["difference_from_best_baseline"] <= 0.25:
        return "A melhor avaliação ficou muito próxima do melhor baseline; trate como empate prático."
    if hybrid_entry and best_evaluation["label"] == "híbrido":
        if neural_entry and hybrid_entry["average_hits"] <= neural_entry["average_hits"]:
            return "O híbrido superou o baseline, mas não superou a rede pura neste relatório."
        return "O híbrido superou o melhor baseline por margem visível neste relatório."
    return "A rede neural superou o melhor baseline por margem visível neste relatório."


def print_comparison(comparison):
    print()
    print("Comparação neural vs melhor baseline:")
    print(
        f"- melhor baseline: {comparison['best_baseline']} "
        f"({comparison['best_baseline_average']:.2f} acertos médios)"
    )
    print(f"- rede neural: {comparison['neural_average']:.2f} acertos médios")
    print(f"- diferença: {comparison['difference']:+.2f} acertos médios")
    if comparison["difference"] > 0.25:
        print("Leitura: a rede neural superou o melhor baseline por uma margem visível neste backtest.")
    elif comparison["difference"] >= -0.25:
        print("Leitura: a rede neural ficou praticamente empatada com os baselines.")
    else:
        print("Leitura: a rede neural ficou abaixo do melhor baseline neste backtest.")


def print_metric_comparison(comparison):
    print()
    print(f"Comparação {comparison['label']} vs melhor baseline:")
    print(
        f"- melhor baseline: {comparison['best_baseline']} "
        f"({comparison['best_baseline_average']:.2f} acertos médios)"
    )
    print(f"- {comparison['label']}: {comparison['target_average']:.2f} acertos médios")
    print(f"- diferença: {comparison['difference']:+.2f} acertos médios")
    if comparison["difference"] > 0.25:
        print(f"Leitura: {comparison['label']} superou o melhor baseline por uma margem visível neste backtest.")
    elif comparison["difference"] >= -0.25:
        print(f"Leitura: {comparison['label']} ficou praticamente empatado com os baselines.")
    else:
        print(f"Leitura: {comparison['label']} ficou abaixo do melhor baseline neste backtest.")


def print_report_summary(summary):
    print()
    print("Resumo comparativo:")
    print(f"- melhor baseline: {summary['best_baseline']} ({summary['best_baseline_average']:.2f} acertos médios)")
    for entry in summary["evaluations"]:
        print(
            f"- {entry['label']}: {entry['average_hits']:.2f} acertos médios "
            f"({entry['difference_from_best_baseline']:+.2f} vs melhor baseline)"
        )
    print(f"- melhor geral: {summary['best_overall']}")
    print(f"Leitura: {summary['recommendation']}")


def print_multi_report_summary(summary):
    print()
    print("Resumo comparativo consolidado:")
    print(f"- melhor baseline: {summary['best_baseline']} ({summary['best_baseline_average']:.2f} acertos médios)")
    for index, entry in enumerate(summary["ranking"], 1):
        source = f", {entry['source']}" if entry.get("source") and entry["source"] != "baseline" else ""
        print(
            f"{index}. {entry['label']}: {entry['average_hits']:.2f} acertos médios "
            f"({entry['difference_from_best_baseline']:+.2f} vs melhor baseline{source})"
        )
    if summary["best_overall"].startswith("baseline:"):
        print("Leitura: o melhor baseline ficou acima das avaliações informadas.")
    else:
        print(f"Leitura: {summary['best_overall']} teve a maior média entre os relatórios informados.")


def run_train(csv_file, model, window_size, epochs, validation_size, seed, tf_verbose=False):
    from .lotery_model import train_model

    train_model(csv_file, model, window_size, epochs, validation_size, seed, tf_verbose)
    return 0


def run_generate(
    model,
    count,
    max_attempts,
    tf_verbose=False,
    hybrid=False,
    model_weight=0.5,
    recent_weight=0.2,
    recency_weight=0.2,
    delay_weight=0.1,
    recent_window=50,
    decay=0.95,
):
    if count <= 0:
        raise ValueError("O número de conjuntos a serem gerados deve ser um inteiro positivo.")

    from .lotery_model import generate_unique_suggestions

    suggestions = generate_unique_suggestions(
        model,
        count,
        max_attempts=max_attempts,
        tf_verbose=tf_verbose,
        hybrid=hybrid,
        model_weight=model_weight,
        recent_weight=recent_weight,
        recency_weight=recency_weight,
        delay_weight=delay_weight,
        recent_window=recent_window,
        decay=decay,
    )
    print("Sugestões geradas:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"Sugestão {i}: {suggestion}")
    return 0


def run_update_games(csv_file):
    from .lotery_results import append_latest_game

    result = append_latest_game(csv_file)
    if result["updated"]:
        print(f"games.csv atualizado com o concurso {result['contest']}. Total de jogos: {result['total_games']}.")
    else:
        print(f"O concurso {result['contest']} já existe no arquivo. Total de jogos: {result['total_games']}.")
    return 0


def run_sync_games(csv_file, from_contest, to_contest, incremental):
    if incremental:
        from .lotery_results import sync_missing_games

        result = sync_missing_games(csv_file, to_contest)
        if result["updated"]:
            print(
                f"Sincronização incremental adicionou {result['added_games']} jogos "
                f"do concurso {result['from_contest']} ao {result['to_contest']}. "
                f"Total de jogos: {result['total_games']}."
            )
        else:
            print(f"Nenhum concurso novo encontrado. Total de jogos: {result['total_games']}.")
        return 0

    from .lotery_results import sync_games

    result = sync_games(csv_file, from_contest, to_contest)
    print(
        f"Arquivo sincronizado do concurso {result['from_contest']} ao {result['to_contest']}. "
        f"Total de jogos: {result['total_games']}."
    )
    return 0


def run_backtest_baselines(csv_file, window_size, seed, decay, output, explain):
    from .lotery_backtest import run_baseline_backtest_from_csv

    results = run_baseline_backtest_from_csv(csv_file, window_size, seed, decay)
    report = {
        "type": "baseline_backtest",
        "parameters": {
            "csv_file": csv_file,
            "window_size": window_size,
            "seed": seed,
            "decay": decay,
        },
        "results": results,
    }
    print_baseline_results(results)
    if explain:
        explain_baseline_results()
    save_report(report, output)
    return 0


def run_backtest_neural(csv_file, window_size, epochs, min_training_games, max_steps, seed, output, compare_baselines, explain, tf_verbose=False):
    from .lotery_backtest import run_neural_backtest_from_csv

    result = run_neural_backtest_from_csv(
        csv_file,
        window_size=window_size,
        epochs=epochs,
        min_training_games=min_training_games,
        max_steps=max_steps,
        seed=seed,
        tf_verbose=tf_verbose,
    )
    metrics = result["neural"]
    print(
        "Backtest neural: "
        f"média {metrics['average_hits']:.2f}, "
        f"min {metrics['min_hits']}, max {metrics['max_hits']} "
        f"({metrics['samples']} amostras)"
    )
    report = {
        "type": "neural_backtest",
        "parameters": {
            "csv_file": csv_file,
            "window_size": window_size,
            "epochs": epochs,
            "min_training_games": min_training_games,
            "max_steps": max_steps,
            "seed": seed,
        },
        "results": result,
    }
    if explain:
        explain_neural_results()
    if compare_baselines:
        baseline_report = load_report(compare_baselines)
        report["summary"] = summarize_against_baseline(baseline_report, report)
        print_report_summary(report["summary"])
    save_report(report, output)
    return 0


def run_backtest_hybrid(
    csv_file,
    window_size,
    epochs,
    min_training_games,
    max_steps,
    seed,
    output,
    compare_baselines,
    explain,
    tf_verbose=False,
    model_weight=0.5,
    recent_weight=0.2,
    recency_weight=0.2,
    delay_weight=0.1,
    recent_window=50,
    decay=0.95,
):
    from .lotery_backtest import run_hybrid_backtest_from_csv

    result = run_hybrid_backtest_from_csv(
        csv_file,
        window_size=window_size,
        epochs=epochs,
        min_training_games=min_training_games,
        max_steps=max_steps,
        seed=seed,
        tf_verbose=tf_verbose,
        model_weight=model_weight,
        recent_weight=recent_weight,
        recency_weight=recency_weight,
        delay_weight=delay_weight,
        recent_window=recent_window,
        decay=decay,
    )
    neural_metrics = result["neural"]
    hybrid_metrics = result["hybrid"]
    print(
        "Backtest híbrido:"
        f"\n- neural: média {neural_metrics['average_hits']:.2f}, "
        f"min {neural_metrics['min_hits']}, max {neural_metrics['max_hits']} "
        f"({neural_metrics['samples']} amostras)"
        f"\n- híbrido: média {hybrid_metrics['average_hits']:.2f}, "
        f"min {hybrid_metrics['min_hits']}, max {hybrid_metrics['max_hits']} "
        f"({hybrid_metrics['samples']} amostras)"
    )
    report = {
        "type": "hybrid_backtest",
        "parameters": {
            "csv_file": csv_file,
            "window_size": window_size,
            "epochs": epochs,
            "min_training_games": min_training_games,
            "max_steps": max_steps,
            "seed": seed,
            "model_weight": model_weight,
            "recent_weight": recent_weight,
            "recency_weight": recency_weight,
            "delay_weight": delay_weight,
            "recent_window": recent_window,
            "decay": decay,
        },
        "results": result,
    }
    if explain:
        explain_hybrid_results()
    if compare_baselines:
        baseline_report = load_report(compare_baselines)
        report["summary"] = summarize_against_baseline(baseline_report, report)
        print_report_summary(report["summary"])
    save_report(report, output)
    return 0


def run_compare_reports(baseline_report_file, evaluation_report_files):
    baseline_report = load_report(baseline_report_file)
    reports = [(report_file, load_report(report_file)) for report_file in evaluation_report_files]
    print_multi_report_summary(summarize_multiple_against_baseline(baseline_report, reports))
    return 0


def run(args):
    if args.command == "train":
        return run_train(args.csv_file, args.model, args.window_size, args.epochs, args.validation_size, args.seed, getattr(args, "tf_verbose", False))
    if args.command == "generate":
        return run_generate(
            args.model,
            args.count,
            args.max_attempts,
            args.tf_verbose,
            args.hybrid,
            args.model_weight,
            args.recent_weight,
            args.recency_weight,
            args.delay_weight,
            args.recent_window,
            args.decay,
        )
    if args.command == "update-games":
        return run_update_games(args.csv_file)
    if args.command == "sync-games":
        return run_sync_games(args.csv_file, args.from_contest, args.to_contest, args.incremental)
    if args.command == "backtest-baselines":
        return run_backtest_baselines(args.csv_file, args.window_size, args.seed, args.decay, args.output, args.explain)
    if args.command == "backtest-neural":
        return run_backtest_neural(
            args.csv_file,
            args.window_size,
            args.epochs,
            args.min_training_games,
            args.max_steps,
            args.seed,
            args.output,
            args.compare_baselines,
            args.explain,
            args.tf_verbose,
        )
    if args.command == "backtest-hybrid":
        return run_backtest_hybrid(
            args.csv_file,
            args.window_size,
            args.epochs,
            args.min_training_games,
            args.max_steps,
            args.seed,
            args.output,
            args.compare_baselines,
            args.explain,
            args.tf_verbose,
            args.model_weight,
            args.recent_weight,
            args.recency_weight,
            args.delay_weight,
            args.recent_window,
            args.decay,
        )
    if args.command == "compare-reports":
        return run_compare_reports(args.baseline_report, args.evaluation_reports)

    return None


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run(args)
        if result is None:
            parser.print_help()
            return 0
        return result
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Erro: {exc}")
        return 1
    except ModuleNotFoundError as exc:
        print(f"Erro: dependência não instalada: {exc.name}. Execute `python -m pip install -r requirements.txt`.")
        return 1
    except KeyboardInterrupt:
        print("Operação interrompida pelo usuário.")
        return 130
