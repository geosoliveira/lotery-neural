import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CAIXA_LOTOFACIL_API = "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil"
PROGRESS_INTERVAL = 50


def fetch_caixa_contest(contest_number=None, timeout=30):
    url = CAIXA_LOTOFACIL_API
    if contest_number is not None:
        url = f"{url}/{contest_number}"

    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "lotery-neural/1.0",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Falha ao consultar concurso {contest_number or 'mais recente'}: HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"Falha de conexão com a API da Caixa: {exc.reason}.") from exc


def parse_contest_numbers(contest):
    dezenas = contest.get("listaDezenas") or contest.get("dezenasSorteadasOrdemSorteio")
    if not dezenas:
        raise ValueError("Resposta da Caixa não contém dezenas sorteadas.")

    numbers = [int(number) for number in dezenas]
    if len(numbers) != 15:
        raise ValueError(f"Resposta da Caixa contém {len(numbers)} dezenas; esperado 15.")
    if len(set(numbers)) != 15:
        raise ValueError("Resposta da Caixa contém dezenas duplicadas.")
    if any(number < 1 or number > 25 for number in numbers):
        raise ValueError("Resposta da Caixa contém dezenas fora do intervalo 1..25.")

    return sorted(numbers)


def parse_contest_number(contest):
    contest_number = contest.get("numero")
    if contest_number is None:
        raise ValueError("Resposta da Caixa não contém o número do concurso.")
    return int(contest_number)


def format_game(numbers):
    return ";".join(str(number) for number in numbers)


def read_existing_games(csv_file):
    try:
        with open(csv_file, "r", encoding="utf-8") as file:
            games = []
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                numbers = [int(value) for value in line.split(";")]
                if len(numbers) != 15:
                    raise ValueError(f"Linha {line_number}: esperado 15 números, encontrado {len(numbers)}.")
                games.append(sorted(numbers))
            return games
    except FileNotFoundError:
        return []


def write_games(csv_file, games):
    with open(csv_file, "w", encoding="utf-8", newline="\n") as file:
        for game in games:
            file.write(f"{format_game(game)}\n")


def append_game(csv_file, game):
    with open(csv_file, "a", encoding="utf-8", newline="\n") as file:
        file.write(f"{format_game(game)}\n")


def print_sync_progress(current_contest, from_contest, to_contest):
    total = to_contest - from_contest + 1
    done = current_contest - from_contest + 1
    if done == 1 or done == total or done % PROGRESS_INTERVAL == 0:
        print(f"Sincronizando concurso {current_contest}/{to_contest} ({done}/{total})...")


def append_latest_game(csv_file):
    latest_contest = fetch_caixa_contest()
    latest_number = parse_contest_number(latest_contest)
    latest_game = parse_contest_numbers(latest_contest)

    games = read_existing_games(csv_file)
    if latest_game in games:
        return {
            "updated": False,
            "contest": latest_number,
            "total_games": len(games),
        }

    games.append(latest_game)
    write_games(csv_file, games)
    return {
        "updated": True,
        "contest": latest_number,
        "total_games": len(games),
    }


def sync_games(csv_file, from_contest=1, to_contest=None):
    if from_contest <= 0:
        raise ValueError("--from-contest deve ser um inteiro positivo.")

    if to_contest is None:
        latest_contest = fetch_caixa_contest()
        to_contest = parse_contest_number(latest_contest)

    if to_contest < from_contest:
        raise ValueError("--to-contest deve ser maior ou igual a --from-contest.")

    games = []
    for contest_number in range(from_contest, to_contest + 1):
        print_sync_progress(contest_number, from_contest, to_contest)
        contest = fetch_caixa_contest(contest_number)
        games.append(parse_contest_numbers(contest))

    write_games(csv_file, games)
    return {
        "from_contest": from_contest,
        "to_contest": to_contest,
        "total_games": len(games),
    }


def sync_missing_games(csv_file, to_contest=None):
    games = read_existing_games(csv_file)
    from_contest = len(games) + 1

    if to_contest is None:
        latest_contest = fetch_caixa_contest()
        to_contest = parse_contest_number(latest_contest)

    if to_contest < from_contest:
        return {
            "updated": False,
            "from_contest": from_contest,
            "to_contest": to_contest,
            "added_games": 0,
            "total_games": len(games),
        }

    if not games:
        print(
            "Aviso: o arquivo está vazio ou não existe. "
            "A sincronização incremental começará no concurso 1 e pode demorar."
        )
    else:
        print(f"Arquivo atual possui {len(games)} concursos. Buscando a partir do concurso {from_contest}.")

    for contest_number in range(from_contest, to_contest + 1):
        print_sync_progress(contest_number, from_contest, to_contest)
        contest = fetch_caixa_contest(contest_number)
        game = parse_contest_numbers(contest)
        games.append(game)
        append_game(csv_file, game)

    return {
        "updated": True,
        "from_contest": from_contest,
        "to_contest": to_contest,
        "added_games": to_contest - from_contest + 1,
        "total_games": len(games),
    }
